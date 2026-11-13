from __future__ import annotations

import math
import struct
import unittest
import wave
from pathlib import Path
import tempfile

try:
    from src.dubbing.core.smart_dubbing import SmartDubbing
except Exception as _exc:  # pragma: no cover - optional heavy dependencies
    SmartDubbing = None  # type: ignore[assignment]


def _write_wav(path: Path, samples, sample_rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(
            b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32000)) for s in samples)
        )
    return path


def _tone(freq: float, seconds: float, sample_rate: int = 16000, amplitude: float = 0.5):
    return [
        amplitude * math.sin(2 * math.pi * freq * index / sample_rate)
        for index in range(int(seconds * sample_rate))
    ]


def _peaks(audio, sample_rate: int = 16000, threshold: float = 0.05):
    """Return the (start, end) of every loud region of a pydub segment."""
    samples = audio.get_array_of_samples()
    step = max(1, int(0.01 * sample_rate))
    loud = []
    for index in range(0, len(samples), step):
        chunk = samples[index:index + step]
        if len(chunk) == 0:
            loud.append(False)
            continue
        rms = (sum(int(value) ** 2 for value in chunk) / len(chunk)) ** 0.5 / 32768.0
        loud.append(rms > threshold)
    regions = []
    start = None
    for index, is_loud in enumerate(loud):
        if is_loud and start is None:
            start = index
        elif not is_loud and start is not None:
            regions.append((start * 0.01, index * 0.01))
            start = None
    if start is not None:
        regions.append((start * 0.01, len(loud) * 0.01))
    return regions


@unittest.skipIf(SmartDubbing is None, "smart_dubbing is not available")
class BackgroundVoiceTrackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # Three beeps in the source timeline: 1-2s, 4-5s, 7-8s.
        samples = (
            [0.0] * 16000
            + _tone(440.0, 1.0)
            + [0.0] * (2 * 16000)
            + _tone(440.0, 1.0)
            + [0.0] * (2 * 16000)
            + _tone(440.0, 1.0)
            + [0.0] * 16000
        )
        self.source = _write_wav(self.dir / "source.wav", samples)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _dubber(self, positions):
        dubber = SmartDubbing.__new__(SmartDubbing)
        dubber.real_segment_positions = positions
        dubber._resolve_vocals_stem = lambda: str(self.source)
        import logging

        dubber.logger = logging.getLogger("test")
        return dubber

    def test_voice_is_moved_to_the_dubbed_timeline(self) -> None:
        from pydub import AudioSegment

        dubber = self._dubber(
            [
                {
                    "start": 0.5,
                    "end": 1.5,
                    "original_start": 1.0,
                    "original_end": 2.0,
                },
                {
                    "start": 2.0,
                    "end": 3.0,
                    "original_start": 4.0,
                    "original_end": 5.0,
                },
            ]
        )

        out_path = dubber.build_background_voice_track(str(self.source))
        self.assertTrue(out_path and Path(out_path).exists())

        track = AudioSegment.from_file(out_path)
        regions = _peaks(track)
        self.assertEqual(len(regions), 2, f"unexpected regions: {regions}")
        # The beeps moved with the segments.
        self.assertAlmostEqual(regions[0][0], 0.5, delta=0.15)
        self.assertAlmostEqual(regions[1][0], 2.0, delta=0.15)

    def test_without_mapping_nothing_is_built(self) -> None:
        dubber = self._dubber([])
        self.assertIsNone(dubber.build_background_voice_track(str(self.source)))


if __name__ == "__main__":
    unittest.main()
