from __future__ import annotations

import os
import types
import unittest

from pydub import AudioSegment

from src.dubbing.core.smart_dubbing import SmartDubbing
from src.tts.content_validation_utils import find_unexpected_repetition
from src.tts.gemini38_tts_wrapper import Gemini38TTSWrapper
from src.tts.models import DiarizationSegment, SegmentAlignment


class _StubValidator:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe_text(self, audio_path: str) -> str:
        return self.text


class FindUnexpectedRepetitionTests(unittest.TestCase):
    def test_flags_stuttered_sentence_missing_from_expected(self) -> None:
        expected = (
            "я вовсе не против того чтобы они говорили нам нужно регулирование "
            "первая проблема изоляция и локализация оказались недостаточно надежными "
            "если бы они сработали как надо эта технология по-прежнему находилась в лаборатории"
        )
        asr = (
            "я вовсе не против того чтобы они говорили нам нужно регулирование "
            "первая проблема изоляция и локализация казались недостаточно надежными "
            "если бы они сработали первая проблема изоляция и локализация оказались "
            "недостаточно надежными если бы они сработали как надо эта технология "
            "по-прежнему находилась в лаборатории"
        )
        repetition = find_unexpected_repetition(asr, expected)
        self.assertIsNotNone(repetition)
        phrase, asr_count, expected_count = repetition
        self.assertEqual(asr_count, 2)
        self.assertEqual(expected_count, 1)
        self.assertTrue(phrase)

    def test_ignores_repetition_present_in_expected_text(self) -> None:
        text = "нет нет нет нет я не согласен с этим решением"
        self.assertIsNone(find_unexpected_repetition(text, text))

    def test_ignores_short_phrases(self) -> None:
        self.assertIsNone(find_unexpected_repetition("да да", "нет"))

    def test_ignores_clean_transcript(self) -> None:
        expected = "сегодня мы обсудим новые модели и их возможности"
        self.assertIsNone(find_unexpected_repetition(expected, expected))


class ContentRepetitionValidationTests(unittest.TestCase):
    def _wrapper(self, asr_text: str) -> Gemini38TTSWrapper:
        wrapper = Gemini38TTSWrapper(
            enable_voice_matching=False,
            enable_content_validation=True,
        )
        wrapper._content_validator = _StubValidator(asr_text)
        return wrapper

    def test_validate_segment_content_flags_repetition(self) -> None:
        expected = "Они знают, как всё сделать правильно. Первая проблема — изоляция и локализация оказались недостаточно надёжными."
        asr = (
            "Они знают, как всё сделать правильно. Первая проблема. Изоляция и локализация "
            "оказались недостаточно надёжными. Первая проблема. Изоляция и локализация "
            "оказались недостаточно надёжными."
        )
        wrapper = self._wrapper(asr)
        valid, reason, _score = wrapper._validate_segment_content("dummy.wav", expected)
        self.assertFalse(valid)
        self.assertIn("repetition", reason.lower())

    def test_validate_segment_content_accepts_clean_speech(self) -> None:
        expected = "Они знают, как всё сделать правильно. Первая проблема — изоляция и локализация оказались недостаточно надёжными."
        wrapper = self._wrapper(expected)
        valid, reason, _score = wrapper._validate_segment_content("dummy.wav", expected)
        self.assertTrue(valid, reason)

    def test_repetition_check_can_be_disabled(self) -> None:
        expected = "Они знают, как всё сделать правильно. Первая проблема — изоляция и локализация оказались недостаточно надёжными."
        asr = (
            "Они знают, как всё сделать правильно. Первая проблема. Изоляция и локализация "
            "оказались недостаточно надёжными. Первая проблема. Изоляция и локализация "
            "оказались недостаточно надёжными."
        )
        wrapper = self._wrapper(asr)
        wrapper.config.enable_content_repetition_check = False
        valid, _reason, _score = wrapper._validate_segment_content("dummy.wav", expected)
        self.assertTrue(valid)


class ResynthesisRejectsInvalidCandidatesTests(unittest.TestCase):
    def test_invalid_candidates_do_not_replace_original_audio(self) -> None:
        smart = SmartDubbing.__new__(SmartDubbing)
        smart.config = {"pause_removal": "disabled", "enable_llm_text_adjustment": False}
        smart.translator = None

        class _InvalidCandidateTTS:
            def synthesize(self, segments_data, language=None, previous_context=None):
                segment = segments_data[0]
                if getattr(segment, "output_path", None):
                    AudioSegment.silent(duration=2000).export(segment.output_path, format="wav")
                return [
                    SegmentAlignment(
                        original_segment=segment,
                        diarized_segment=DiarizationSegment(
                            start_time=0.0,
                            end_time=2.0,
                            speaker=segment.speaker,
                            text=segment.text,
                            confidence=1.0,
                        ),
                        alignment_confidence=1.0,
                        content_valid=False,
                        content_validation_reason="Content repetition: 'изоляция и локализация' spoken 2x",
                    )
                ]

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "90.wav")
            AudioSegment.silent(duration=1000).export(output_path, format="wav")

            segment_dict = {
                "start": 0.0,
                "end": 2.0,
                "speaker": "SPEAKER_A",
                "translation": "исходный текст",
                "chosen_text": "исходный текст",
                "very_short_translation": "короткий",
                "short_translation": "короткий текст",
                "long_translation": "исходный текст подлиннее",
                "synthesized_speech_len": 1.0,
            }
            metadata = {
                "index": 90,
                "segment_dict": segment_dict,
                "output_path": output_path,
                "chosen_text": "исходный текст",
                "segment_data_args": {
                    "speaker": "SPEAKER_A",
                    "text": "исходный текст",
                    "output_path": output_path,
                },
                "previous_context": [],
            }

            smart._resynthesize_segment(metadata, _InvalidCandidateTTS(), 0.85, 1.15, current_ratio=0.5)

            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(segment_dict["translation"], "исходный текст")
            leftovers = [n for n in os.listdir(tmp_dir) if ".temp_" in n]
            self.assertEqual(leftovers, [])

            audio = AudioSegment.from_file(output_path)
            self.assertAlmostEqual(len(audio) / 1000.0, 1.0, places=1)

    def test_content_rejected_segment_accepts_any_valid_candidate(self) -> None:
        smart = SmartDubbing.__new__(SmartDubbing)
        smart.config = {"pause_removal": "disabled", "enable_llm_text_adjustment": False}
        smart.translator = None
        smart.cache_manager = types.SimpleNamespace(use_cache=False)

        class _ValidCandidateTTS:
            def synthesize(self, segments_data, language=None, previous_context=None):
                segment = segments_data[0]
                if getattr(segment, "output_path", None):
                    AudioSegment.silent(duration=1500).export(segment.output_path, format="wav")
                return [
                    SegmentAlignment(
                        original_segment=segment,
                        diarized_segment=DiarizationSegment(
                            start_time=0.0,
                            end_time=1.5,
                            speaker=segment.speaker,
                            text=segment.text,
                            confidence=1.0,
                        ),
                        alignment_confidence=1.0,
                        content_valid=True,
                        content_validation_reason="Content match",
                    )
                ]

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "91.wav")
            AudioSegment.silent(duration=2000).export(output_path, format="wav")

            segment_dict = {
                "start": 0.0,
                "end": 2.0,
                "speaker": "SPEAKER_A",
                "translation": "повторяющийся текст",
                "chosen_text": "повторяющийся текст",
                "very_short_translation": "короткий",
                "short_translation": "короткий текст",
                "long_translation": "повторяющийся текст подлиннее",
                "synthesized_speech_len": 2.0,
            }
            metadata = {
                "index": 91,
                "segment_dict": segment_dict,
                "output_path": output_path,
                "chosen_text": "повторяющийся текст",
                "cache_path": os.path.join(tmp_dir, "cache.wav"),
                "segment_data_args": {
                    "speaker": "SPEAKER_A",
                    "text": "повторяющийся текст",
                    "output_path": output_path,
                },
                "previous_context": [],
            }

            smart._resynthesize_segment(
                metadata,
                _ValidCandidateTTS(),
                0.85,
                1.15,
                current_ratio=1.0,
                content_rejected=True,
            )

            audio = AudioSegment.from_file(output_path)
            self.assertAlmostEqual(len(audio) / 1000.0, 1.5, places=1)
            self.assertEqual(segment_dict["chosen_text"], "короткий")


if __name__ == "__main__":
    unittest.main()
