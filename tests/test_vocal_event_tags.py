from __future__ import annotations

import unittest

from src.tts.gemini38_tags import convert_legacy_tags, sanitize_angle_tags
from src.tts.gemini38_tts_wrapper import (
    Gemini38TTSConfig,
    Gemini38TTSWrapper,
    _text_vocal_kinds,
)
from src.tts.models import TTSSegmentData


class LegacyTagMappingTests(unittest.TestCase):
    def test_screaming_maps_to_inline_scream(self) -> None:
        text, style_hints = convert_legacy_tags("Он [screaming] закричал")
        self.assertIn("<scream>", text)
        self.assertNotIn("[screaming]", text)
        self.assertEqual(style_hints, [])

    def test_gasping_and_panting_map_inline(self) -> None:
        text, _ = convert_legacy_tags("[gasping] и [panting]")
        self.assertIn("<gasp>", text)
        self.assertIn("<pant>", text)

    def test_angle_alias_is_normalized(self) -> None:
        self.assertEqual(sanitize_angle_tags("Да <screaming> нет"), "Да <scream> нет")
        self.assertEqual(sanitize_angle_tags("<whimpers>"), "<whimper>")


class VocalKindDetectionTests(unittest.TestCase):
    def test_detects_non_laugh_kinds(self) -> None:
        kinds = _text_vocal_kinds("Она <sob> и <scream>, потом [gasp]")
        self.assertIn("cry", kinds)
        self.assertIn("scream", kinds)
        self.assertIn("gasp", kinds)

    def test_laugh_not_reported_as_vocal_kind(self) -> None:
        kinds = _text_vocal_kinds("Смешно! [laughs]")
        self.assertEqual(kinds, [])

    def test_no_tags_no_kinds(self) -> None:
        self.assertEqual(_text_vocal_kinds("Обычная реплика"), [])


def _make_wrapper(missing_sequence):
    wrapper = object.__new__(Gemini38TTSWrapper)
    wrapper.config = Gemini38TTSConfig(
        enable_laughter_verification=False,
        enable_vocal_event_verification=True,
        enable_content_validation=False,
        enable_voice_consistency_validation=False,
        enable_context_style=False,
        enable_emotion_enrichment=False,
    )
    wrapper.emotion_enricher = None
    wrapper.translator = None
    wrapper._laughter_verifier = False

    calls = []
    missing_calls = []
    sequence = list(missing_sequence)

    def fake_once(segment, working_text, output_path, language, voice_name, models, usage_tracker, context_style):
        calls.append(working_text)
        return working_text, "gemini-3.8-flash-lite-tts", True, "stub reason"

    def fake_missing(_output_path, kinds):
        if not kinds:
            return []
        missing_calls.append(list(kinds))
        return sequence.pop(0) if sequence else []

    wrapper._synthesize_text_once = fake_once
    wrapper._missing_vocal_kinds = fake_missing
    wrapper._resolve_voice_for_segment = lambda segment: "Charon"
    return wrapper, calls, missing_calls


def _segment(text: str) -> TTSSegmentData:
    return TTSSegmentData(speaker="SPEAKER_A", text=text)


class VocalVerificationTests(unittest.TestCase):
    def test_no_tags_skips_verification(self) -> None:
        wrapper, calls, missing_calls = _make_wrapper([["scream"]])
        text, model, valid, reason, info = wrapper._synthesize_segment(
            _segment("Привет"), "/tmp/x.wav", "ru"
        )
        self.assertTrue(valid)
        self.assertEqual(missing_calls, [])
        self.assertEqual(info["vocal_required"], [])
        self.assertEqual(len(calls), 1)

    def test_missing_then_detected_retries_once(self) -> None:
        wrapper, calls, missing_calls = _make_wrapper([["scream"], []])
        text, model, valid, reason, info = wrapper._synthesize_segment(
            _segment("Он <scream> закричал"), "/tmp/x.wav", "ru"
        )
        self.assertTrue(valid)
        self.assertEqual(info["attempts"], 2)
        self.assertEqual(info["vocal_required"], ["scream"])
        self.assertEqual(len(calls), 2)

    def test_failure_keeps_audio_and_records_missing(self) -> None:
        wrapper, calls, missing_calls = _make_wrapper([["cry"], ["cry"]])
        text, model, valid, reason, info = wrapper._synthesize_segment(
            _segment("Она <sob> плакала"), "/tmp/x.wav", "ru"
        )
        self.assertTrue(valid)
        self.assertEqual(info["attempts"], 2)
        self.assertEqual(info["vocal_missing"], ["cry"])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
