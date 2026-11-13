from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from src.tts.voice_clone_reference import (
    VoiceCloneReferenceBuilder,
    VoiceCloneReferenceConfig,
)


FRAME_RATE = 24000


def _tone(duration_ms: int, frequency: int, volume: float = -12.0) -> AudioSegment:
    return Sine(frequency).to_audio_segment(duration=duration_ms, volume=volume).set_frame_rate(FRAME_RATE)


def _silence(duration_ms: int) -> AudioSegment:
    return AudioSegment.silent(duration=duration_ms, frame_rate=FRAME_RATE)


def _two_speaker_audio() -> AudioSegment:
    """A clean interleaved conversation with silence around every turn."""
    audio = _silence(1000)
    audio += _tone(4000, 220)  # A 1-5
    audio += _silence(1000)
    audio += _tone(3000, 330)  # B 6-9
    audio += _silence(1000)
    audio += _tone(4000, 240)  # A 10-14
    audio += _silence(1000)
    audio += _tone(3000, 350)  # B 15-18
    audio += _silence(2000)
    audio += _tone(4000, 260)  # A 20-24
    audio += _silence(1000)
    audio += _tone(3000, 370)  # B 25-28
    audio += _silence(2000)
    return audio.set_channels(1).set_sample_width(2)


def _clean_rolls() -> dict:
    return {
        (1.0, 5.0): "SPEAKER_A",
        (6.0, 9.0): "SPEAKER_B",
        (10.0, 14.0): "SPEAKER_A",
        (15.0, 18.0): "SPEAKER_B",
        (20.0, 24.0): "SPEAKER_A",
        (25.0, 28.0): "SPEAKER_B",
    }


class VoiceCloneReferenceBuilderTests(unittest.TestCase):
    def test_builds_reference_from_clean_clips_across_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            _two_speaker_audio().export(str(source), format="wav")
            builder = VoiceCloneReferenceBuilder(
                VoiceCloneReferenceConfig(
                    min_clip_seconds=2.0,
                    max_clip_seconds=10.0,
                    target_seconds=18.0,
                    min_total_seconds=8.0,
                    max_clips=4,
                    min_clips=2,
                    min_silence_ms=200,
                )
            )

            references = builder.build_all(
                str(source),
                _clean_rolls(),
                out_dir=str(root / "clones"),
            )

            self.assertIn("SPEAKER_A", references)
            reference = references["SPEAKER_A"]
            self.assertTrue(Path(reference.reference_path).exists())
            self.assertGreaterEqual(reference.duration_seconds, 8.0)
            self.assertGreaterEqual(reference.clip_count, 2)
            self.assertEqual(reference.clip_count, 3)

            # Every selected clip is cut inside the silence around the turn,
            # not exactly at the diarization boundary.
            expected = [(0.5, 5.5), (9.5, 14.5), (19.4, 24.5)]
            for (start, end), (want_start, want_end) in zip(reference.clip_ranges, expected):
                self.assertAlmostEqual(start, want_start, places=1)
                self.assertAlmostEqual(end, want_end, places=1)

            for start, end in reference.clip_ranges:
                for (other_start, other_end), speaker in _clean_rolls().items():
                    if speaker == "SPEAKER_A":
                        continue
                    self.assertFalse(
                        start < other_end and end > other_start,
                        f"clip {start}-{end} overlaps {speaker} {other_start}-{other_end}",
                    )

    def test_rejects_turn_that_overlaps_another_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "overlap.wav"
            audio = _silence(20000)
            audio = audio.overlay(_tone(4000, 220), position=1000)  # A 1-5
            audio = audio.overlay(_tone(3000, 330), position=3000)  # B 3-6 overlaps A
            audio = audio.overlay(_tone(4000, 240), position=10000)  # A 10-14
            audio = audio.overlay(_tone(3000, 350), position=15000)  # B 15-18
            audio = audio.set_channels(1).set_sample_width(2)
            audio.export(str(source), format="wav")

            rolls = {
                (1.0, 5.0): "SPEAKER_A",
                (3.0, 6.0): "SPEAKER_B",
                (10.0, 14.0): "SPEAKER_A",
                (15.0, 18.0): "SPEAKER_B",
            }
            builder = VoiceCloneReferenceBuilder(
                VoiceCloneReferenceConfig(
                    min_clip_seconds=2.0,
                    max_clip_seconds=10.0,
                    target_seconds=10.0,
                    min_total_seconds=3.0,
                    max_clips=3,
                    min_clips=1,
                    min_silence_ms=200,
                )
            )

            references = builder.build_all(str(source), rolls, out_dir=str(root / "clones"))

            self.assertIn("SPEAKER_A", references)
            reference = references["SPEAKER_A"]
            self.assertEqual(reference.clip_count, 1)
            start, end = reference.clip_ranges[0]
            self.assertGreaterEqual(start, 9.0)
            self.assertLessEqual(end, 15.0)

    def test_returns_none_when_there_is_no_quiet_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "continuous.wav"
            _tone(10000, 220).set_channels(1).set_sample_width(2).export(
                str(source), format="wav"
            )
            builder = VoiceCloneReferenceBuilder(VoiceCloneReferenceConfig(min_clips=1))
            references = builder.build_all(
                str(source),
                {(0.0, 10.0): "SPEAKER_A"},
                out_dir=str(root / "clones"),
            )
            self.assertEqual(references, {})

    def test_reuses_cached_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            _two_speaker_audio().export(str(source), format="wav")
            builder = VoiceCloneReferenceBuilder(
                VoiceCloneReferenceConfig(min_total_seconds=3.0, max_clips=2, min_clips=1)
            )
            first = builder.build_all(str(source), _clean_rolls(), out_dir=str(root / "clones"))
            second = builder.build_all(str(source), _clean_rolls(), out_dir=str(root / "clones"))
            self.assertEqual(first["SPEAKER_A"].cache_token, second["SPEAKER_A"].cache_token)
            self.assertEqual(first["SPEAKER_A"].clip_ranges, second["SPEAKER_A"].clip_ranges)


if __name__ == "__main__":
    unittest.main()
