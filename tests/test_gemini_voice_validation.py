"""Tests for Gemini TTS voice consistency validation and forced voice fallback."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
import numpy as np

from src.tts.gemini_tts_wrapper import GeminiTTSConfig, GeminiTTSWrapper
from src.tts.models import TTSSegmentData, SegmentSynthesisReport


class GeminiVoiceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wrapper = GeminiTTSWrapper.__new__(GeminiTTSWrapper)
        self.wrapper.config = GeminiTTSConfig(
            enable_voice_consistency_validation=True,
            enable_voice_matching=True,
            voice_similarity_threshold=0.70,
            voice_similarity_relaxed_threshold=0.65,
            min_voice_validation_duration_seconds=0.5,
            voice_reference_mode="max",
        )
        self.wrapper.voice_matcher = mock.Mock()
        self.wrapper.voice_matcher.audio_embedder = mock.Mock()
        self.wrapper.voice_matcher.sample_embeddings = {}
        self.wrapper.voice_sample_manager = None
        self.wrapper._voice_embeddings_lock = threading.Lock()
        self.wrapper._speaker_ref_embeddings = {}
        self.wrapper._speaker_to_ref_path = {}
        self.wrapper._audio_cache = {}
        self.wrapper._cache_dir = None
        self.wrapper.speaker_genders = {}
        self.wrapper.voice_mapping = {}
        self.wrapper.voice_prompt_mapping = {}
        self.wrapper._record_segment_report = mock.Mock()

    def test_validate_voice_consistency_passes_high_similarity(self) -> None:
        dummy_emb = np.array([1.0, 0.0], dtype=np.float32)
        self.wrapper.voice_matcher.sample_embeddings["Puck"] = dummy_emb
        self.wrapper.voice_matcher.extract_embedding_for_audio_file.return_value = dummy_emb
        self.wrapper.voice_matcher.get_voice_similarity.return_value = 0.78

        with mock.patch("src.tts.gemini_tts_wrapper.AudioFileUtils.get_audio_duration_seconds", return_value=1.2):
            is_valid, reason, score = self.wrapper._validate_voice_consistency(
                "/tmp/test_clip.wav", "Puck"
            )

        self.assertTrue(is_valid)
        self.assertEqual(score, 0.78)
        self.assertIn("Voice match ok", reason)

    def test_validate_voice_consistency_rejects_mismatch(self) -> None:
        dummy_emb = np.array([1.0, 0.0], dtype=np.float32)
        self.wrapper.voice_matcher.sample_embeddings["Puck"] = dummy_emb
        self.wrapper.voice_matcher.extract_embedding_for_audio_file.return_value = dummy_emb
        self.wrapper.voice_matcher.get_voice_similarity.return_value = 0.52

        with mock.patch("src.tts.gemini_tts_wrapper.AudioFileUtils.get_audio_duration_seconds", return_value=1.0):
            is_valid, reason, score = self.wrapper._validate_voice_consistency(
                "/tmp/test_clip.wav", "Puck"
            )

        self.assertFalse(is_valid)
        self.assertEqual(score, 0.52)
        self.assertIn("Voice mismatch", reason)

    def test_validate_voice_consistency_skips_short_segment(self) -> None:
        with mock.patch("src.tts.gemini_tts_wrapper.AudioFileUtils.get_audio_duration_seconds", return_value=0.35):
            is_valid, reason, score = self.wrapper._validate_voice_consistency(
                "/tmp/test_clip.wav", "Puck"
            )

        self.assertTrue(is_valid)
        self.assertIsNone(score)
        self.assertIn("short segment", reason)

    def test_validate_voice_consistency_dual_reference_max_mode(self) -> None:
        # Unit vectors: angle 0 deg -> cosine dist 0 -> similarity 1.0
        ref_vec = np.array([1.0, 0.0], dtype=np.float32)
        clip_vec = np.array([1.0, 0.0], dtype=np.float32)

        self.wrapper.voice_matcher.sample_embeddings["Puck"] = np.array([0.0, 1.0], dtype=np.float32)
        self.wrapper.voice_matcher.extract_embedding_for_audio_file.return_value = clip_vec
        # Voice sample similarity is low (0.55), but speaker reference similarity is 1.0
        self.wrapper.voice_matcher.get_voice_similarity.return_value = 0.55
        self.wrapper._speaker_ref_embeddings["SPEAKER_00"] = ref_vec

        with mock.patch("src.tts.gemini_tts_wrapper.AudioFileUtils.get_audio_duration_seconds", return_value=2.0):
            is_valid, reason, score = self.wrapper._validate_voice_consistency(
                "/tmp/test_clip.wav", "Puck", speaker_id="SPEAKER_00"
            )

        self.assertTrue(is_valid)
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, 1.0, places=3)
        self.assertIn("Voice match ok", reason)

    def test_validate_voice_consistency_both_references_low_triggers_mismatch(self) -> None:
        # Two orthogonal vectors: cosine distance 1.0 -> similarity 0.0
        ref_vec = np.array([0.0, 1.0], dtype=np.float32)
        clip_vec = np.array([1.0, 0.0], dtype=np.float32)

        self.wrapper.voice_matcher.sample_embeddings["Puck"] = np.array([0.0, 1.0], dtype=np.float32)
        self.wrapper.voice_matcher.extract_embedding_for_audio_file.return_value = clip_vec
        self.wrapper.voice_matcher.get_voice_similarity.return_value = 0.48
        self.wrapper._speaker_ref_embeddings["SPEAKER_00"] = ref_vec

        with mock.patch("src.tts.gemini_tts_wrapper.AudioFileUtils.get_audio_duration_seconds", return_value=2.0):
            is_valid, reason, score = self.wrapper._validate_voice_consistency(
                "/tmp/test_clip.wav", "Puck", speaker_id="SPEAKER_00"
            )

        self.assertFalse(is_valid)
        self.assertIsNotNone(score)
        # Max of 0.48 and 0.0 is 0.48, which is < 0.70
        self.assertAlmostEqual(score, 0.48, places=2)
        self.assertIn("Voice mismatch", reason)

    def test_validate_voice_consistency_on_demand_sample_loading(self) -> None:
        sample_emb = np.array([1.0, 0.0], dtype=np.float32)
        clip_emb = np.array([1.0, 0.0], dtype=np.float32)

        manager = mock.Mock()
        manager.samples_dir = Path("/tmp/samples")
        manager._get_audio_extension.return_value = "wav"
        self.wrapper.voice_sample_manager = manager

        self.wrapper.voice_matcher.extract_embedding_for_audio_file.side_effect = [sample_emb, clip_emb]
        self.wrapper.voice_matcher.get_voice_similarity.return_value = 0.85

        with mock.patch("pathlib.Path.exists", return_value=True), \
             mock.patch("src.tts.gemini_tts_wrapper.AudioFileUtils.get_audio_duration_seconds", return_value=1.5):
            is_valid, reason, score = self.wrapper._validate_voice_consistency(
                "/tmp/test_clip.wav", "Puck"
            )

        self.assertTrue(is_valid)
        self.assertEqual(score, 0.85)
        # Verify Puck embedding was loaded into sample_embeddings
        self.assertIn("Puck", self.wrapper.voice_matcher.sample_embeddings)

    def test_forced_voice_switch_on_persistent_voice_mismatch(self) -> None:
        self.wrapper.api_client = mock.Mock()
        self.wrapper.api_client.client = mock.Mock()
        self.wrapper.api_client.current_model = "gemini-2.5-pro-preview-tts"
        self.wrapper.config.fallback_model = None  # disable fallback model for clean test
        self.wrapper.speaker_genders = {"SPEAKER_00": "male"}
        self.wrapper.voice_mapping = {"SPEAKER_00": "Zephyr"}  # female voice for male speaker

        segment = TTSSegmentData(speaker="SPEAKER_00", text="Hello world, testing voice validation.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_output = os.path.join(tmp_dir, "output.wav")
            attempt_file = os.path.join(tmp_dir, "attempt.wav")
            forced_file = os.path.join(tmp_dir, "forced.wav")
            with open(attempt_file, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 100)
            with open(forced_file, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 100)

            def fake_attempt(seg, out_path, lang, retries, **kwargs):
                if seg.voice == "Zephyr":
                    # Mismatch with Zephyr
                    return False, 0.01, 0.50, attempt_file, seg.text, "gemini-2.5-pro-preview-tts", True
                elif seg.voice == "Puck":
                    # Forced gender-compatible Puck succeeds!
                    return True, 0.01, 0.85, forced_file, seg.text, "gemini-2.5-pro-preview-tts", True
                return False, 0.01, 0.50, attempt_file, seg.text, "gemini-2.5-pro-preview-tts", True

            self.wrapper._attempt_segment_synthesis = fake_attempt
            self.wrapper._synthesize_by_sentence_split = mock.Mock(return_value=None)

            text_result, model_result, success = self.wrapper._synthesize_single_segment(
                segment, temp_output, "en"
            )

            self.assertTrue(success)
            self.assertEqual(text_result, segment.text)

            # Check that _record_segment_report was called with voice_forced=True
            self.wrapper._record_segment_report.assert_called_once()
            report: SegmentSynthesisReport = self.wrapper._record_segment_report.call_args[0][0]
            self.assertTrue(report.voice_forced)
            self.assertTrue(report.success)
            self.assertEqual(report.voice_similarity, 0.85)

    def test_forced_voice_recorded_when_best_attempt_used(self) -> None:
        self.wrapper.api_client = mock.Mock()
        self.wrapper.api_client.client = mock.Mock()
        self.wrapper.api_client.current_model = "gemini-2.5-pro-preview-tts"
        self.wrapper.config.fallback_model = None
        self.wrapper.speaker_genders = {"SPEAKER_00": "male"}
        self.wrapper.voice_mapping = {"SPEAKER_00": "Zephyr"}

        segment = TTSSegmentData(speaker="SPEAKER_00", text="Hello world, testing voice validation.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_output = os.path.join(tmp_dir, "output.wav")
            attempt_file = os.path.join(tmp_dir, "attempt.wav")
            forced_file = os.path.join(tmp_dir, "forced.wav")
            with open(attempt_file, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 100)
            with open(forced_file, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 100)

            def fake_attempt(seg, out_path, lang, retries, **kwargs):
                # Both fail, but forced voice has better similarity
                if seg.voice == "Puck":
                    return False, 0.01, 0.62, forced_file, seg.text, "gemini-2.5-pro-preview-tts", True
                return False, 0.01, 0.45, attempt_file, seg.text, "gemini-2.5-pro-preview-tts", True

            self.wrapper._attempt_segment_synthesis = fake_attempt
            self.wrapper._synthesize_by_sentence_split = mock.Mock(return_value=None)

            text_result, model_result, success = self.wrapper._synthesize_single_segment(
                segment, temp_output, "en"
            )

            # Synthesis marked False because validation didn't pass, but forced attempt was ranked best
            self.assertFalse(success)
            report: SegmentSynthesisReport = self.wrapper._record_segment_report.call_args[0][0]
            self.assertTrue(report.voice_forced)
            self.assertEqual(report.voice_similarity, 0.62)

    def test_relative_gender_check_rejects_opposite_gender_drift(self) -> None:
        # Clip embedding aligned with the female sample; the (mocked) absolute score
        # passes the threshold but the relative gate must reject the clip.
        puck_vec = np.array([1.0, 0.0], dtype=np.float32)   # male
        kore_vec = np.array([0.0, 1.0], dtype=np.float32)   # female
        clip_vec = kore_vec

        self.wrapper.voice_matcher.sample_embeddings["Puck"] = puck_vec
        self.wrapper.voice_matcher.sample_embeddings["Kore"] = kore_vec
        self.wrapper.voice_matcher.extract_embedding_for_audio_file.return_value = clip_vec
        self.wrapper.voice_matcher.get_voice_similarity.return_value = 0.75  # fake absolute pass

        with mock.patch("src.tts.gemini_tts_wrapper.AudioFileUtils.get_audio_duration_seconds", return_value=2.0):
            is_valid, reason, score = self.wrapper._validate_voice_consistency(
                "/tmp/test_clip.wav", "Puck"
            )

        self.assertFalse(is_valid)
        self.assertIn("Voice gender mismatch", reason)
        self.assertIn("Kore", reason)
        self.assertEqual(score, 0.75)

    def test_relative_gender_check_passes_correct_voice(self) -> None:
        puck_vec = np.array([1.0, 0.0], dtype=np.float32)
        kore_vec = np.array([0.0, 1.0], dtype=np.float32)
        clip_vec = puck_vec

        self.wrapper.voice_matcher.sample_embeddings["Puck"] = puck_vec
        self.wrapper.voice_matcher.sample_embeddings["Kore"] = kore_vec
        self.wrapper.voice_matcher.extract_embedding_for_audio_file.return_value = clip_vec
        self.wrapper.voice_matcher.get_voice_similarity.return_value = 0.90

        with mock.patch("src.tts.gemini_tts_wrapper.AudioFileUtils.get_audio_duration_seconds", return_value=2.0):
            is_valid, reason, score = self.wrapper._validate_voice_consistency(
                "/tmp/test_clip.wav", "Puck"
            )

        self.assertTrue(is_valid)
        self.assertEqual(score, 0.90)

    def test_relative_gender_check_respects_margin(self) -> None:
        # opp_sim = 0.95, expected (mocked) = 0.92 → rejected with margin 0.0,
        # accepted with margin 0.05.
        puck_vec = np.array([1.0, 0.0], dtype=np.float32)
        kore_vec = np.array([0.0, 1.0], dtype=np.float32)
        comp = float(np.sqrt(1.0 - 0.95 ** 2))
        clip_vec = np.array([comp, 0.95], dtype=np.float32)

        self.wrapper.voice_matcher.sample_embeddings["Puck"] = puck_vec
        self.wrapper.voice_matcher.sample_embeddings["Kore"] = kore_vec
        self.wrapper.voice_matcher.extract_embedding_for_audio_file.return_value = clip_vec
        self.wrapper.voice_matcher.get_voice_similarity.return_value = 0.92

        with mock.patch("src.tts.gemini_tts_wrapper.AudioFileUtils.get_audio_duration_seconds", return_value=2.0):
            is_valid, _, _ = self.wrapper._validate_voice_consistency("/tmp/test_clip.wav", "Puck")
            self.assertFalse(is_valid)

            self.wrapper.config.voice_gender_relative_margin = 0.05
            is_valid, reason, score = self.wrapper._validate_voice_consistency("/tmp/test_clip.wav", "Puck")
            self.assertTrue(is_valid)
            self.assertEqual(score, 0.92)

    def test_process_single_segment_cache_hit_validated_ok(self) -> None:
        self.wrapper.voice_sample_manager = mock.Mock()
        self.wrapper.api_client = mock.Mock()
        self.wrapper.api_client.current_model = "gemini-2.5-pro-preview-tts"
        self.wrapper.voice_mapping = {"SPEAKER_00": "Puck"}
        self.wrapper._validate_voice_consistency = mock.Mock(return_value=(True, "Voice match ok (0.900)", 0.900))

        segment = TTSSegmentData(speaker="SPEAKER_00", text="Привет, мир.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            cached_file = os.path.join(tmp_dir, "cached.wav")
            with open(cached_file, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 100)
            cache_key = self.wrapper._get_cache_key(segment, "ru")
            self.wrapper._audio_cache[cache_key] = (cached_file, 1.5)

            alignment = self.wrapper._process_single_segment(
                segment=segment,
                segment_index=0,
                total_segments=1,
                temp_dir=tmp_dir,
                language="ru",
                context_segments=[],
                usage_tracker={},
            )

            self.assertIsNotNone(alignment)
            self.wrapper._validate_voice_consistency.assert_called_once()
            report: SegmentSynthesisReport = self.wrapper._record_segment_report.call_args[0][0]
            self.assertEqual(report.attempts, 0)
            self.assertEqual(report.voice_similarity, 0.900)
            self.assertTrue(report.voice_validation.startswith("cached;"))
            # Cache entry must survive a successful re-validation
            self.assertIn(cache_key, self.wrapper._audio_cache)

    def test_process_single_segment_cache_hit_invalid_dropped_and_resynthesized(self) -> None:
        self.wrapper.voice_sample_manager = mock.Mock()
        self.wrapper.api_client = mock.Mock()
        self.wrapper.api_client.client = mock.Mock()
        self.wrapper.api_client.current_model = "gemini-2.5-pro-preview-tts"
        self.wrapper.voice_mapping = {"SPEAKER_00": "Puck"}
        self.wrapper._validate_voice_consistency = mock.Mock(
            return_value=(False, "Voice gender mismatch: closer to 'Kore' (0.950) than to expected 'Puck' (0.800)", 0.800)
        )

        segment = TTSSegmentData(speaker="SPEAKER_00", text="Привет, мир.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            cached_file = os.path.join(tmp_dir, "cached.wav")
            with open(cached_file, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 100)
            cache_key = self.wrapper._get_cache_key(segment, "ru")
            self.wrapper._audio_cache[cache_key] = (cached_file, 1.5)

            resynth_called = []

            def fake_resynth(seg, out_path, lang, **kwargs):
                resynth_called.append(out_path)
                with open(out_path, "wb") as f:
                    f.write(b"RIFF" + b"\x00" * 200)
                return seg.text, "gemini-2.5-pro-preview-tts", True

            self.wrapper._synthesize_single_segment = fake_resynth

            alignment = self.wrapper._process_single_segment(
                segment=segment,
                segment_index=0,
                total_segments=1,
                temp_dir=tmp_dir,
                language="ru",
                context_segments=[],
                usage_tracker={},
            )

            self.assertIsNotNone(alignment)
            self.assertEqual(len(resynth_called), 1)
            self.assertNotIn(cache_key, self.wrapper._audio_cache)
            self.assertFalse(os.path.exists(cached_file))

    def test_synthesize_single_segment_cache_hit_invalid_dropped(self) -> None:
        self.wrapper.api_client = mock.Mock()
        self.wrapper.api_client.client = mock.Mock()
        self.wrapper.api_client.current_model = "gemini-2.5-pro-preview-tts"
        self.wrapper.config.fallback_model = None
        self.wrapper.voice_mapping = {"SPEAKER_00": "Puck"}
        self.wrapper._validate_voice_consistency = mock.Mock(
            return_value=(False, "Voice gender mismatch: closer to 'Kore' (0.950) than to expected 'Puck' (0.800)", 0.800)
        )

        segment = TTSSegmentData(speaker="SPEAKER_00", text="Hello world.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_output = os.path.join(tmp_dir, "output.wav")
            attempt_file = os.path.join(tmp_dir, "attempt.wav")
            cached_file = os.path.join(tmp_dir, "cached.wav")
            with open(attempt_file, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 100)
            with open(cached_file, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 100)

            cache_key = self.wrapper._get_cache_key(segment, "en")
            self.wrapper._audio_cache[cache_key] = (cached_file, 1.0)

            def fake_attempt(seg, out_path, lang, retries, **kwargs):
                return True, 0.01, 0.88, attempt_file, seg.text, "gemini-2.5-pro-preview-tts", True

            self.wrapper._attempt_segment_synthesis = fake_attempt
            self.wrapper._synthesize_by_sentence_split = mock.Mock(return_value=None)

            text_result, model_result, success = self.wrapper._synthesize_single_segment(
                segment, temp_output, "en"
            )

            self.assertTrue(success)
            self.assertNotIn(cache_key, self.wrapper._audio_cache)
            self.assertFalse(os.path.exists(cached_file))
            self.wrapper._record_segment_report.assert_called_once()
            report: SegmentSynthesisReport = self.wrapper._record_segment_report.call_args[0][0]
            self.assertEqual(report.voice_similarity, 0.88)

    def test_voice_sample_audit_regenerates_contaminated_sample(self) -> None:
        female_vec = np.array([1.0, 0.0], dtype=np.float64)
        male_vec = np.array([0.0, 1.0], dtype=np.float64)

        embeddings = self.wrapper.voice_matcher.sample_embeddings
        # Three female voices and three male voices cluster cleanly...
        for name in ("Kore", "Aoede", "Achernar"):
            embeddings[name] = female_vec
        for name in ("Puck", "Charon", "Orus"):
            embeddings[name] = male_vec
        # ...but Fenrir (male per catalog) is contaminated with a female render.
        embeddings["Fenrir"] = female_vec

        def fake_regenerate(voice_name: str) -> bool:
            if voice_name == "Fenrir":
                embeddings["Fenrir"] = male_vec
            return True

        self.wrapper.voice_sample_manager = mock.Mock()
        self.wrapper.voice_sample_manager.regenerate_voice_sample.side_effect = fake_regenerate

        self.wrapper._audit_voice_sample_embeddings()

        self.wrapper.voice_sample_manager.regenerate_voice_sample.assert_called_once_with("Fenrir")

    def test_voice_sample_audit_passes_clean_samples(self) -> None:
        female_vec = np.array([1.0, 0.0], dtype=np.float64)
        male_vec = np.array([0.0, 1.0], dtype=np.float64)

        embeddings = self.wrapper.voice_matcher.sample_embeddings
        for name in ("Kore", "Aoede", "Achernar"):
            embeddings[name] = female_vec
        for name in ("Puck", "Charon", "Orus"):
            embeddings[name] = male_vec

        self.wrapper.voice_sample_manager = mock.Mock()

        self.wrapper._audit_voice_sample_embeddings()

        self.wrapper.voice_sample_manager.regenerate_voice_sample.assert_not_called()


class GeminiMultiSpeakerReportTests(unittest.TestCase):
    """Per-slice voice validation results must not leak across batch slices."""

    def _make_fake(self, cls):
        class _Fake:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        _Fake.__name__ = cls
        return _Fake

    def test_ms_batch_reports_record_per_slice_similarity(self) -> None:
        import src.tts.gemini_tts_wrapper as g

        wrapper = GeminiTTSWrapper.__new__(GeminiTTSWrapper)
        wrapper.config = GeminiTTSConfig(
            enable_voice_consistency_validation=True,
            voice_similarity_threshold=0.70,
            min_voice_validation_duration_seconds=0.5,
        )
        wrapper.voice_matcher = mock.Mock()
        wrapper.voice_matcher.audio_embedder = mock.Mock()
        wrapper.speaker_genders = {"SPEAKER_A": "male", "SPEAKER_B": "female"}
        wrapper.voice_mapping = {"SPEAKER_A": "Puck", "SPEAKER_B": "Kore"}
        wrapper.voice_prompt_mapping = {}
        wrapper.api_client = mock.Mock()
        wrapper.api_client.current_model = "gemini-2.5-pro-preview-tts"
        wrapper.api_client.synthesize_chunk.return_value = b"\x00" * 16000
        wrapper._cache_dir = None
        wrapper._audio_cache = {}
        wrapper._speaker_ref_embeddings = {}
        wrapper._speaker_to_ref_path = {}
        wrapper._voice_embeddings_lock = threading.Lock()
        wrapper._record_segment_report = mock.Mock()
        wrapper._record_duration_stats = mock.Mock()
        wrapper._register_usage = mock.Mock()
        wrapper._count_tokens = mock.Mock(return_value=10)
        wrapper.cost_tracker = None
        wrapper._find_temporal_conflicts = mock.Mock(return_value={})
        wrapper.diarizer = mock.Mock()

        fake_diar_result = mock.Mock()
        fake_diar_result.segments = [
            self._make_fake("SpeakerSegment")(
                index=0, speaker="SPEAKER_A", start_ms=0, end_ms=1000,
                asr_text="line one", matched_line_idx=0, matched_text="line one",
                fuzzy_score=90.0, asr_speaker_label="A", confidence=0.9,
            ),
            self._make_fake("SpeakerSegment")(
                index=1, speaker="SPEAKER_B", start_ms=1100, end_ms=2000,
                asr_text="line two", matched_line_idx=1, matched_text="line two",
                fuzzy_score=85.0, asr_speaker_label="B", confidence=0.9,
            ),
        ]
        wrapper.diarizer.align_by_text.return_value = fake_diar_result
        wrapper.diarizer.slice_audio.return_value = [b"\x00" * 8000]

        def fake_validate(audio_path, voice_name, speaker_id=None, reference_audio_path=None):
            if voice_name == "Puck":
                return True, "Voice match ok (0.910)", 0.910
            return True, "Voice match ok (0.800)", 0.800

        wrapper._validate_voice_consistency = mock.Mock(side_effect=fake_validate)

        batch = [
            TTSSegmentData(speaker="SPEAKER_A", text="line one"),
            TTSSegmentData(speaker="SPEAKER_B", text="line two"),
        ]

        fake_line_cls = self._make_fake("DialogueLine")
        fake_ss_cls = self._make_fake("SpeakerSegment")

        with mock.patch.object(g, "DIARIZER_AVAILABLE", True), \
             mock.patch.object(g, "DialogueLine", fake_line_cls), \
             mock.patch.object(g, "SpeakerSegment", fake_ss_cls), \
             mock.patch.object(g.AudioValidator, "validate_audio_sample", return_value=(True, "ok", 0.0)):
            alignments, invalid, attempts = wrapper._synthesize_multi_speaker_batch(
                batch=batch,
                batch_indices=[0, 1],
                language="ru",
                usage_tracker={"models": {}},
            )

        self.assertEqual(attempts, 1)
        self.assertEqual(invalid, [])
        self.assertEqual(len(alignments), 2)

        reports = [c[0][0] for c in wrapper._record_segment_report.call_args_list]
        sims = sorted(r.voice_similarity for r in reports)
        self.assertEqual(sims, [0.800, 0.910])
        reasons = {r.speaker: r.voice_validation for r in reports}
        self.assertEqual(reasons["SPEAKER_A"], "Voice match ok (0.910)")
        self.assertEqual(reasons["SPEAKER_B"], "Voice match ok (0.800)")


class SmartDubbingCacheKeyTests(unittest.TestCase):
    def test_cache_key_includes_voice_and_model(self) -> None:
        try:
            from src.dubbing.core.smart_dubbing import SmartDubbing
        except Exception:
            self.skipTest("smart_dubbing dependencies (torch) not available")

        sd = SmartDubbing.__new__(SmartDubbing)
        base = dict(
            base_cache_prefix="p",
            tts_system="gemini",
            segment_index=0,
            speaker="SPEAKER_A",
            text_value="текст",
            voice_prompt_hash="abcd",
        )

        key_auto = sd._build_segment_cache_key(**base)
        key_puck = sd._build_segment_cache_key(**base, voice_name="Puck")
        key_kore = sd._build_segment_cache_key(**base, voice_name="Kore")
        key_model = sd._build_segment_cache_key(**base, voice_name="Puck", tts_model="gemini-2.5-flash-preview-tts")

        self.assertNotEqual(key_auto, key_puck)
        self.assertNotEqual(key_puck, key_kore)
        self.assertNotEqual(key_puck, key_model)
        self.assertIn("auto", key_auto)
        self.assertEqual(
            key_puck,
            sd._build_segment_cache_key(**base, voice_name="Puck"),
        )


if __name__ == "__main__":
    unittest.main()
