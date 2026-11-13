from __future__ import annotations

import unittest

from src.tts.gemini38_tts_wrapper import (
    LAUGHTER_FALLBACK_PREFIX,
    Gemini38TTSConfig,
    Gemini38TTSWrapper,
    _text_has_laughter_tag,
)
from src.tts.models import TTSSegmentData


def _make_wrapper(verify_results, valid_sequence=None):
    wrapper = object.__new__(Gemini38TTSWrapper)
    wrapper.config = Gemini38TTSConfig(
        enable_laughter_verification=True,
        enable_content_validation=False,
        enable_voice_consistency_validation=False,
        enable_context_style=False,
        enable_emotion_enrichment=False,
    )
    wrapper.emotion_enricher = None
    wrapper.translator = None
    wrapper._laughter_verifier = False

    calls = []
    sequence = list(valid_sequence or [True] * 10)

    def fake_once(segment, working_text, output_path, language, voice_name, models, usage_tracker, context_style):
        calls.append(working_text)
        valid = sequence.pop(0) if sequence else True
        return working_text, "gemini-3.8-flash-lite-tts", valid, "stub reason"

    verify_iter = iter(verify_results)

    def fake_verify(_output_path):
        try:
            value = next(verify_iter)
        except StopIteration:
            value = (True, 0.9)
        return value

    wrapper._synthesize_text_once = fake_once
    wrapper._verify_laughter = fake_verify
    wrapper._resolve_voice_for_segment = lambda segment: "Charon"
    return wrapper, calls


def _segment(text: str) -> TTSSegmentData:
    return TTSSegmentData(speaker="SPEAKER_A", text=text)


class LaughterTagDetectionTests(unittest.TestCase):
    def test_detects_all_laughter_dialects(self) -> None:
        self.assertTrue(_text_has_laughter_tag("Это было смешно. [laughs]"))
        self.assertTrue(_text_has_laughter_tag("Это было смешно. <laugh>"))
        self.assertTrue(_text_has_laughter_tag("Ну да <giggle>"))
        self.assertFalse(_text_has_laughter_tag("Это было смешно."))


class LaughterRetryTests(unittest.TestCase):
    def test_no_tag_skips_verification(self) -> None:
        wrapper, calls = _make_wrapper([(False, 0.0)])
        text, model, valid, reason, info = wrapper._synthesize_segment(_segment("Привет"), "/tmp/x.wav", "ru")
        self.assertTrue(valid)
        self.assertFalse(info["required"])
        self.assertEqual(info["attempts"], 1)
        self.assertEqual(len(calls), 1)

    def test_laughter_verified_first_attempt(self) -> None:
        wrapper, calls = _make_wrapper([(True, 0.9)])
        text, model, valid, reason, info = wrapper._synthesize_segment(
            _segment("Это было смешно. [laughs]"), "/tmp/x.wav", "ru"
        )
        self.assertTrue(valid)
        self.assertTrue(info["required"])
        self.assertTrue(info["detected"])
        self.assertAlmostEqual(info["score"], 0.9)
        self.assertFalse(info["fallback"])
        self.assertEqual(info["attempts"], 1)
        self.assertEqual(len(calls), 1)

    def test_retry_same_text_succeeds(self) -> None:
        wrapper, calls = _make_wrapper([(False, 0.1), (True, 0.8)])
        text, model, valid, reason, info = wrapper._synthesize_segment(
            _segment("Это было смешно. [laughs]"), "/tmp/x.wav", "ru"
        )
        self.assertTrue(valid)
        self.assertTrue(info["detected"])
        self.assertFalse(info["fallback"])
        self.assertEqual(info["attempts"], 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])

    def test_verbal_fallback_is_used_and_stripped_from_result(self) -> None:
        wrapper, calls = _make_wrapper([(False, 0.05), (False, 0.1), (True, 0.95)])
        original = "Это было смешно. [laughs]"
        text, model, valid, reason, info = wrapper._synthesize_segment(_segment(original), "/tmp/x.wav", "ru")
        self.assertTrue(valid)
        self.assertTrue(info["detected"])
        self.assertTrue(info["fallback"])
        self.assertEqual(info["attempts"], 3)
        self.assertTrue(calls[2].startswith(LAUGHTER_FALLBACK_PREFIX))
        self.assertEqual(text, original)

    def test_laughter_failure_keeps_audio_and_reports(self) -> None:
        wrapper, calls = _make_wrapper([(False, 0.01), (False, 0.02), (False, 0.03)])
        text, model, valid, reason, info = wrapper._synthesize_segment(
            _segment("Это было смешно. [laughs]"), "/tmp/x.wav", "ru"
        )
        self.assertTrue(valid)
        self.assertFalse(info["detected"])
        self.assertTrue(info["fallback"])
        self.assertEqual(info["attempts"], 3)
        self.assertEqual(len(calls), 3)

    def test_invalid_synthesis_returns_not_valid(self) -> None:
        wrapper, calls = _make_wrapper([(True, 0.9)], valid_sequence=[False])
        text, model, valid, reason, info = wrapper._synthesize_segment(
            _segment("Это было смешно. [laughs]"), "/tmp/x.wav", "ru"
        )
        self.assertFalse(valid)
        self.assertEqual(info["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
