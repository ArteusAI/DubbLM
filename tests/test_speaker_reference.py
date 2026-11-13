from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pydub import AudioSegment
from pydub.generators import Sine

from src.tts.speaker_reference import (
    SpeakerReferenceBuilder,
    SpeakerReferenceProfile,
    best_voice_for_profile,
    cosine_similarity,
)


class _FakeEmbedder:
    """Deterministic embedder: vector depends on clip duration + tone pitch."""

    def extract_embedding(self, audio_segment: AudioSegment):
        duration = max(0.1, len(audio_segment) / 1000.0)
        rms = max(1.0, float(audio_segment.rms))
        return np.array([duration, rms / 1000.0, 1.0], dtype=np.float64)


class _FakeMatcher:
    def __init__(self, embeddings):
        self.sample_embeddings = embeddings

    def find_best_matching_voice_multi_segment(self, embeddings, exclude_voices=None):
        excluded = {v.lower() for v in (exclude_voices or [])}
        best_voice = None
        best_distance = None
        for voice, sample in self.sample_embeddings.items():
            if voice.lower() in excluded:
                continue
            distance = 1.0 - cosine_similarity(np.mean(np.stack(embeddings), axis=0), sample)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_voice = voice
        return best_voice


def _make_reference_file(path: Path) -> None:
    audio = AudioSegment.silent(duration=400, frame_rate=24000)
    for index in range(5):
        tone = Sine(220 + index * 40).to_audio_segment(duration=1200, volume=-10).set_frame_rate(24000)
        audio += tone
        audio += AudioSegment.silent(duration=350, frame_rate=24000)
    audio.set_channels(1).set_sample_width(2).export(str(path), format="wav")


class SpeakerReferenceBuilderTests(unittest.TestCase):
    def test_builds_profile_from_non_silent_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ref_path = tmp_path / "speaker.wav"
            _make_reference_file(ref_path)

            builder = SpeakerReferenceBuilder(_FakeEmbedder(), min_clip_seconds=0.8, min_clips=2, max_clips=4, target_seconds=10.0)
            profile = builder.build_profile("SPEAKER_A", str(ref_path), tmp_path / "profiles")

            self.assertIsNotNone(profile)
            self.assertGreaterEqual(profile.clip_count, 2)
            self.assertEqual(len(profile.clip_embeddings), profile.clip_count)
            self.assertIsNotNone(profile.centroid)
            self.assertTrue(Path(profile.reference_path).exists())
            self.assertGreater(profile.total_seconds, 0.0)
            self.assertTrue((tmp_path / "profiles" / "SPEAKER_A.json").exists())

    def test_prefers_original_audio_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ref_path = tmp_path / "speaker.wav"
            _make_reference_file(ref_path)

            builder = SpeakerReferenceBuilder(_FakeEmbedder(), min_clip_seconds=0.8, min_clips=1, max_clips=2)
            profile = builder.build_profile(
                "SPEAKER_B",
                str(ref_path),
                tmp_path / "profiles",
                time_ranges=[(1.0, 2.5), (3.0, 4.5)],
                original_audio_path=str(ref_path),
            )
            self.assertIsNotNone(profile)
            self.assertEqual(profile.source, "original")
            self.assertEqual(profile.clip_count, 2)
            self.assertAlmostEqual(profile.clip_ranges[0][0], 1.0, places=2)

    def test_returns_none_when_no_usable_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            silent_path = tmp_path / "silent.wav"
            AudioSegment.silent(duration=3000, frame_rate=24000).export(str(silent_path), format="wav")
            builder = SpeakerReferenceBuilder(_FakeEmbedder(), min_clip_seconds=0.8)
            self.assertIsNone(builder.build_profile("SPEAKER_C", str(silent_path), tmp_path / "profiles"))

    def test_best_voice_for_profile_uses_multi_segment_matcher(self) -> None:
        profile = SpeakerReferenceProfile(
            speaker="SPEAKER_A",
            centroid=np.array([1.0, 0.0]),
            clip_embeddings=[np.array([1.0, 0.0]), np.array([0.9, 0.1])],
        )
        matcher = _FakeMatcher(
            {
                "Kore": np.array([0.0, 1.0]),
                "Puck": np.array([1.0, 0.0]),
            }
        )
        self.assertEqual(best_voice_for_profile(profile, matcher), "Puck")
        self.assertEqual(best_voice_for_profile(profile, matcher, exclude_voices=["Puck"]), "Kore")

    def test_cosine_similarity_handles_degenerate_vectors(self) -> None:
        self.assertEqual(cosine_similarity(np.zeros(3), np.ones(3)), 0.0)


if __name__ == "__main__":
    unittest.main()
