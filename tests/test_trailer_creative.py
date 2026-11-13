from __future__ import annotations

import json
import string
import unittest
from unittest import mock

from src.dubbing.core.config import DubbingConfig
from src.translation.llm_translator import LLMTranslator
from src.translation.prompts import REFINEMENT_PROMPTS


class TrailerCreativePersonaTests(unittest.TestCase):
    EXPECTED_PLACEHOLDERS = {
        "dialogue_summary",
        "glossary_section",
        "domain",
        "tone",
        "themes",
        "terminology",
        "previous_chunk_context",
        "original_conversation_text",
        "next_chunk_context",
        "translated_conversation_text",
        "source_language",
        "target_language",
        "alternative_versions_section",
        "json_output_format",
    }

    def test_persona_is_loaded(self):
        self.assertIn("trailer_creative", REFINEMENT_PROMPTS)

    def test_persona_uses_only_supported_placeholders(self):
        prompt = REFINEMENT_PROMPTS["trailer_creative"]
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(prompt)
            if field_name
        }
        self.assertTrue(fields.issubset(self.EXPECTED_PLACEHOLDERS), fields - self.EXPECTED_PLACEHOLDERS)

    def test_persona_formats_with_supported_values(self):
        prompt = REFINEMENT_PROMPTS["trailer_creative"]
        rendered = prompt.format(
            dialogue_summary="s",
            glossary_section="",
            domain="film",
            tone="dramatic",
            themes="memory",
            terminology="",
            previous_chunk_context="",
            original_conversation_text="SPEAKER_A: hi",
            next_chunk_context="",
            translated_conversation_text="SPEAKER_A: привет",
            source_language="en",
            target_language="ru",
            alternative_versions_section="# variants",
            json_output_format="{}",
        )
        self.assertIn("FACT GUARDRAIL", rendered)
        self.assertIn("same number of lines", rendered)


class Gemini38GuidanceTests(unittest.TestCase):
    def _translator(self, **attrs):
        translator = object.__new__(LLMTranslator)
        translator.tts_system = attrs.get("tts_system", "gemini38")
        translator.tts_system_mapping = attrs.get("tts_system_mapping", {})
        translator.tts_model = attrs.get("tts_model")
        translator.segment_stretch = attrs.get("segment_stretch", "audio_and_video")
        translator.creative_recovery_variants = attrs.get("creative_recovery_variants", False)
        return translator

    def test_gemini38_gets_delivery_guidance(self):
        translator = self._translator(tts_system="gemini38")
        guidance = translator._build_editor_tts_guidance([{"speaker": "SPEAKER_A"}])
        self.assertIn("# Gemini TTS delivery guidance", guidance)
        self.assertIn("`gemini38`", guidance)

    def test_variant_keys_creative_recovery(self):
        default = self._translator(segment_stretch="audio_and_video")
        self.assertEqual(default._required_variant_keys(), ["long"])

        creative = self._translator(
            segment_stretch="audio_and_video", creative_recovery_variants=True
        )
        self.assertEqual(creative._required_variant_keys(), ["very_short", "short", "long"])


class TrailerPresetTests(unittest.TestCase):
    def test_removed_trailers_preset_falls_back_to_ultra(self):
        from src.api.services.preset_service import get_preset_config

        config = get_preset_config("trailers")
        self.assertEqual(config.get("persona_id"), "normal")
        self.assertFalse(config.get("enable_gemini_speaker_labeling"))
        self.assertFalse(config.get("enable_video_emotion_analysis"))

    def test_config_defaults_disable_new_flags(self):
        config = DubbingConfig()
        self.assertFalse(config.get("refiner_emotion_tags", True))
        self.assertFalse(config.get("creative_recovery_variants", True))


class CreativeRefinementIntegrationTests(unittest.TestCase):
    def test_refinement_prompt_gets_persona_and_gemini38_guidance(self):
        translator = LLMTranslator(
            enable_cache=False,
            enable_llm_editor=False,
            refinement_persona="trailer_creative",
            tts_system="gemini38",
            segment_stretch="audio_and_video",
            creative_recovery_variants=True,
        )
        captured_prompts: list[str] = []

        class _StubResponse:
            text = json.dumps(
                {
                    "translations": [
                        {
                            "speaker": "SPEAKER_A",
                            "text": "[excited] Готово [short pause]",
                            "very_short": "Готово",
                            "short": "[excited] Готово",
                            "long": "[excited] Ну что ж, готово [short pause]",
                            "cohesion_with_prev": "normal",
                        }
                    ]
                }
            )

        class _StubLlm:
            def complete(self, prompt: str):
                captured_prompts.append(prompt)
                return _StubResponse()

        translator.refinement_llm = _StubLlm()
        chunk = {
            "segments": [{"speaker": "SPEAKER_A", "text": "Ready", "start": 0.0, "end": 1.0}],
            "translated_pairs": [{"speaker": "SPEAKER_A", "text": "[excited] Готово"}],
        }

        with mock.patch("src.translation.llm_translator.json_repair") as mock_json_repair:
            mock_json_repair.loads.side_effect = json.loads
            translator._refine_translation(
                translated_chunks=[chunk],
                context_info={"domain": "film", "tone": "dramatic", "themes": []},
                source_language="en",
                target_language="ru",
                dialogue_summary="Test summary",
                debug=False,
            )

        self.assertEqual(len(captured_prompts), 1)
        prompt = captured_prompts[0]
        self.assertIn("FACT GUARDRAIL", prompt)
        self.assertIn("# Gemini TTS delivery guidance", prompt)
        self.assertIn("very_short", prompt)


if __name__ == "__main__":
    unittest.main()
