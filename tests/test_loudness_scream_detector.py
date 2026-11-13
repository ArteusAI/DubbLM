from __future__ import annotations

import math
import struct
import unittest
import wave
from pathlib import Path
import tempfile

from src.dubbing.audio.loudness_scream_detector import LoudnessScreamDetector


def _write_wav(path: Path, samples, sample_rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(
            b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32000)) for s in samples)
        )
    return path


def _tone(freq: float, seconds: float, sample_rate: int = 16000, amplitude: float = 0.1):
    return [
        amplitude * math.sin(2 * math.pi * freq * index / sample_rate)
        for index in range(int(seconds * sample_rate))
    ]


class LoudnessScreamDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sample.wav"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_loud_burst_is_detected_among_quiet_speech(self) -> None:
        quiet = _tone(180.0, 2.0, amplitude=0.05)
        loud = _tone(180.0, 0.4, amplitude=0.75)
        quiet_again = _tone(180.0, 2.0, amplitude=0.05)
        _write_wav(self.path, quiet + loud + quiet_again)

        events = LoudnessScreamDetector(margin_db=3.0, min_seconds=0.1).detect(str(self.path))

        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].start, 2.0, delta=0.15)
        self.assertAlmostEqual(events[0].end, 2.4, delta=0.15)
        self.assertEqual(events[0].kind, "scream")

    def test_uniform_speech_produces_nothing(self) -> None:
        _write_wav(self.path, _tone(180.0, 4.0, amplitude=0.05))

        events = LoudnessScreamDetector(margin_db=3.0, min_seconds=0.1).detect(str(self.path))

        self.assertEqual(events, [])

    def test_excluded_ranges_are_skipped(self) -> None:
        quiet = _tone(180.0, 2.0, amplitude=0.05)
        loud = _tone(180.0, 0.4, amplitude=0.75)
        _write_wav(self.path, quiet + loud)

        detector = LoudnessScreamDetector(margin_db=3.0, min_seconds=0.1)

        detected = detector.detect(str(self.path))
        skipped = detector.detect(str(self.path), exclude_ranges=[(1.5, 3.0)])

        self.assertEqual(len(detected), 1)
        self.assertEqual(skipped, [])

    def test_missing_file_is_handled(self) -> None:
        events = LoudnessScreamDetector().detect("/nonexistent/file.wav")

        self.assertEqual(events, [])


class LoudnessDetectionWiringTests(unittest.TestCase):
    def _smart_dubbing(self):
        try:
            from src.dubbing.core.smart_dubbing import SmartDubbing
        except Exception as exc:  # pragma: no cover - optional heavy deps
            self.skipTest(f"smart_dubbing unavailable: {exc}")
        return SmartDubbing

    def test_detector_is_disabled_by_default(self) -> None:
        SmartDubbing = self._smart_dubbing()

        dubber = SmartDubbing.__new__(SmartDubbing)
        dubber.config = {}
        dubber._vocal_events = []

        self.assertEqual(dubber._detect_loud_vocal_events("audio.wav", []), [])

    def test_detector_runs_when_enabled(self) -> None:
        SmartDubbing = self._smart_dubbing()

        dubber = SmartDubbing.__new__(SmartDubbing)
        dubber.config = {
            "enable_vocal_event_loudness_detection": True,
            "vocal_event_loudness_margin_db": 3.0,
        }
        dubber._vocal_events = []

        with patch.object(SmartDubbing, "_resolve_vocals_stem", return_value=None):
            with patch(
                "src.dubbing.audio.loudness_scream_detector.LoudnessScreamDetector.detect",
                return_value=[],
            ) as detect_mock:
                dubber._detect_loud_vocal_events("audio.wav", [])

        detect_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
