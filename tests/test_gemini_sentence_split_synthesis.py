"""Tests for Gemini TTS sentence-split fallback synthesis."""

from __future__ import annotations

import unittest
from unittest import mock

from src.tts.gemini_tts_wrapper import GeminiTTSWrapper
from src.tts.models import TTSSegmentData


class GeminiSentenceSplitSynthesisTests(unittest.TestCase):
    def test_sentence_split_synthesis_concatenates_valid_parts(self) -> None:
        wrapper = GeminiTTSWrapper.__new__(GeminiTTSWrapper)
        wrapper.api_client = mock.Mock()
        wrapper.config = mock.Mock(model="gemini-2.5-pro-preview-tts")
        wrapper._record_segment_report = mock.Mock()

        long_text = (
            "Bonjour à tous. "
            "Je vais vous présenter le modèle de prédiction que nous avons développé. "
            "L'objectif était de prédire la probabilité de clic. "
            "Et en plus, il est important d'identifier les caractéristiques."
        )
        segment = TTSSegmentData(speaker="SPEAKER_A", text=long_text)
        calls: list[str] = []

        def fake_synthesize(
            part_segment: TTSSegmentData,
            part_path: str,
            language: str,
            **kwargs: object,
        ) -> tuple[None, str, bool]:
            calls.append(part_segment.text)
            with open(part_path, "wb") as handle:
                handle.write(b"\x00\x00" * 100)
            return None, "gemini-2.5-pro-preview-tts", True

        wrapper._synthesize_single_segment = fake_synthesize  # type: ignore[method-assign]

        with mock.patch(
            "src.tts.gemini_tts_wrapper.AudioFileUtils.concatenate_audio_files",
            return_value=b"\x00\x00" * 200,
        ) as concat_mock, mock.patch(
            "src.tts.gemini_tts_wrapper.AudioFileUtils.save_wave_file",
        ) as save_mock:
            result = wrapper._synthesize_by_sentence_split(
                segment_data=segment,
                temp_output_path="/tmp/test_segment.wav",
                language="fr",
                max_retries_per_model=4,
                previous_segments=None,
                usage_tracker=None,
                attempts_counter=[0],
                synth_start_ts=0.0,
                report_segment_index=0,
                report_group_id=None,
                primary_model_name="gemini-2.5-pro-preview-tts",
                split_depth=0,
            )

        self.assertIsNotNone(result)
        _, _, success = result
        self.assertTrue(success)
        self.assertEqual(len(calls), 2)
        self.assertEqual(" ".join(calls), long_text)
        concat_mock.assert_called_once()
        save_mock.assert_called_once()

    def test_sentence_split_synthesis_keeps_best_attempt_part(self) -> None:
        """When one part fails validation but still produced audio, the split
        should keep it as a best-effort result rather than discarding the whole
        segment (which previously produced a silent segment downstream)."""
        wrapper = GeminiTTSWrapper.__new__(GeminiTTSWrapper)
        wrapper.api_client = mock.Mock()
        wrapper.config = mock.Mock(model="gemini-2.5-pro-preview-tts")
        wrapper._record_segment_report = mock.Mock()

        long_text = (
            "Bonjour à tous. "
            "Je vais vous présenter le modèle de prédiction que nous avons développé. "
            "L'objectif était de prédire la probabilité de clic. "
            "Et en plus, il est important d'identifier les caractéristiques."
        )
        segment = TTSSegmentData(speaker="SPEAKER_A", text=long_text)
        calls: list[str] = []

        def fake_synthesize(
            part_segment: TTSSegmentData,
            part_path: str,
            language: str,
            **kwargs: object,
        ) -> tuple[None, str, bool]:
            calls.append(part_segment.text)
            with open(part_path, "wb") as handle:
                handle.write(b"\x00\x00" * 100)
            # First part passes, second part fails validation (best-effort).
            part_ok = (len(calls) == 1)
            return None, "gemini-2.5-pro-preview-tts", part_ok

        wrapper._synthesize_single_segment = fake_synthesize  # type: ignore[method-assign]

        with mock.patch(
            "src.tts.gemini_tts_wrapper.AudioFileUtils.concatenate_audio_files",
            return_value=b"\x00\x00" * 200,
        ) as concat_mock, mock.patch(
            "src.tts.gemini_tts_wrapper.AudioFileUtils.save_wave_file",
        ) as save_mock:
            result = wrapper._synthesize_by_sentence_split(
                segment_data=segment,
                temp_output_path="/tmp/test_segment_best.wav",
                language="fr",
                max_retries_per_model=4,
                previous_segments=None,
                usage_tracker=None,
                attempts_counter=[0],
                synth_start_ts=0.0,
                report_segment_index=0,
                report_group_id=None,
                primary_model_name="gemini-2.5-pro-preview-tts",
                split_depth=0,
            )

        # Split should return the best-effort combined audio (not None).
        self.assertIsNotNone(result)
        _, _, success = result
        self.assertFalse(success)
        self.assertEqual(len(calls), 2)
        self.assertEqual(" ".join(calls), long_text)
        concat_mock.assert_called_once()
        save_mock.assert_called_once()

    def test_sentence_split_synthesis_discards_part_without_audio(self) -> None:
        """If a child part produced no audio file at all, the split is unusable
        and should still return None (we cannot concatenate a missing part)."""
        wrapper = GeminiTTSWrapper.__new__(GeminiTTSWrapper)
        wrapper.api_client = mock.Mock()
        wrapper.config = mock.Mock(model="gemini-2.5-pro-preview-tts")
        wrapper._record_segment_report = mock.Mock()

        long_text = (
            "Bonjour à tous. "
            "Je vais vous présenter le modèle de prédiction que nous avons développé. "
            "L'objectif était de prédire la probabilité de clic. "
            "Et en plus, il est important d'identifier les caractéristiques."
        )
        segment = TTSSegmentData(speaker="SPEAKER_A", text=long_text)

        def fake_synthesize(
            part_segment: TTSSegmentData,
            part_path: str,
            language: str,
            **kwargs: object,
        ) -> tuple[None, str, bool]:
            # Never write the part file — simulate total synthesis failure.
            return None, "gemini-2.5-pro-preview-tts", False

        wrapper._synthesize_single_segment = fake_synthesize  # type: ignore[method-assign]

        result = wrapper._synthesize_by_sentence_split(
            segment_data=segment,
            temp_output_path="/tmp/test_segment_noaudio.wav",
            language="fr",
            max_retries_per_model=4,
            previous_segments=None,
            usage_tracker=None,
            attempts_counter=[0],
            synth_start_ts=0.0,
            report_segment_index=0,
            report_group_id=None,
            primary_model_name="gemini-2.5-pro-preview-tts",
            split_depth=0,
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
