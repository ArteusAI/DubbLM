from __future__ import annotations

import unittest
from unittest.mock import patch

from src.api.services import preset_service


class PresetServiceTests(unittest.TestCase):
    def test_fast_preset_uses_openrouter_defaults(self) -> None:
        with patch.object(preset_service, "_load_preset_overrides", return_value={}):
            fast = preset_service.get_preset_config("fast")
            frontend = preset_service.get_frontend_preset_config("fast")

        self.assertEqual(fast["persona_id"], "none")
        self.assertEqual(fast["llm_provider"], "openrouter")
        self.assertEqual(fast["llm_model_name"], "~google/gemini-flash-latest")
        self.assertFalse(fast["enable_llm_editor"])
        self.assertFalse(fast["enable_emotion_enrichment"])
        self.assertFalse(fast["enable_llm_text_adjustment"])
        self.assertEqual(fast["tts_system"], "gemini38")
        self.assertEqual(fast["tts_transport"], "openrouter")
        self.assertEqual(fast["tts_model"], "google/gemini-3.8-flash-lite-tts")
        self.assertEqual(fast["transcription_system"], "openrouter")
        self.assertEqual(fast["openrouter_stt_model"], "microsoft/mai-transcribe-2")
        self.assertFalse(fast["enable_content_validation"])
        self.assertEqual(frontend["personaId"], "none")
        self.assertEqual(frontend["llmModelName"], "~google/gemini-flash-latest")
        self.assertEqual(frontend["ttsTransport"], "openrouter")
        self.assertFalse(frontend["enableContentValidation"])

    def test_hq_preset_uses_openrouter_tts_and_transcription(self) -> None:
        with patch.object(preset_service, "_load_preset_overrides", return_value={}):
            hq = preset_service.get_preset_config("hq")
            frontend = preset_service.get_frontend_preset_config("hq")

        self.assertEqual(hq["persona_id"], "none")
        self.assertEqual(hq["llm_provider"], "openrouter")
        self.assertEqual(hq["llm_model_name"], "~google/gemini-flash-latest")
        self.assertEqual(hq["refinement_llm_provider"], "openrouter")
        self.assertEqual(hq["refinement_model_name"], "openai/gpt-6-sol")
        self.assertEqual(hq["refinement_temperature"], 1.0)
        self.assertFalse(hq["enable_llm_editor"])
        self.assertFalse(hq["enable_emotion_enrichment"])
        self.assertFalse(hq["enable_llm_text_adjustment"])
        self.assertEqual(hq["tts_system"], "gemini38")
        self.assertEqual(hq["tts_transport"], "openrouter")
        self.assertEqual(hq["tts_model"], "google/gemini-3.8-flash-lite-tts")
        self.assertEqual(hq["tts_fallback_model"], "google/gemini-3.8-flash-lite-tts")
        self.assertEqual(hq["transcription_system"], "openrouter")
        self.assertEqual(frontend["personaId"], "none")
        self.assertEqual(frontend["transcriptionSystem"], "openrouter")
        self.assertFalse(frontend["enableLlmTextAdjustment"])

    def test_ultra_preset_matches_reference_project(self) -> None:
        with patch.object(preset_service, "_load_preset_overrides", return_value={}):
            ultra = preset_service.get_preset_config("ultra")
            frontend = preset_service.get_frontend_preset_config("ultra")

        self.assertFalse(ultra["enable_llm_editor"])
        self.assertEqual(ultra["keep_background"], False)
        self.assertEqual(ultra["llm_provider"], "openrouter")
        self.assertEqual(ultra["llm_model_name"], "~google/gemini-flash-latest")
        self.assertEqual(ultra["refinement_llm_provider"], "openrouter")
        self.assertEqual(ultra["refinement_model_name"], "google/gemini-4-carbon")
        self.assertEqual(ultra["refinement_temperature"], 0.8)
        self.assertEqual(ultra["editor_llm_provider"], "openrouter")
        self.assertEqual(ultra["editor_model_name"], "openai/gpt-5.6-sol")
        self.assertEqual(ultra["editor_temperature"], 1.0)
        self.assertEqual(ultra["editor_reasoning_effort"], "xhigh")
        self.assertEqual(ultra["tts_system"], "gemini38")
        self.assertEqual(ultra["tts_transport"], "openrouter")
        self.assertEqual(ultra["tts_model"], "google/gemini-3.8-flash-tts")
        self.assertEqual(ultra["tts_fallback_model"], "google/gemini-3.8-flash-lite-tts")
        self.assertEqual(ultra["transcription_system"], "openrouter")
        self.assertFalse(ultra["enable_laughter_detection"])
        self.assertEqual(frontend["refinementModelName"], "google/gemini-4-carbon")
        self.assertEqual(frontend["editorModelName"], "openai/gpt-5.6-sol")
        self.assertEqual(frontend["editorTemperature"], 1.0)
        self.assertEqual(frontend["ttsTransport"], "openrouter")

    def test_all_presets_use_openrouter_and_disable_editor(self) -> None:
        with patch.object(preset_service, "_load_preset_overrides", return_value={}):
            for preset_name in ("fast", "hq", "ultra"):
                preset = preset_service.get_preset_config(preset_name)
                self.assertFalse(preset["enable_llm_editor"], preset_name)
                self.assertFalse(preset["normalize_audio"], preset_name)
                self.assertEqual(preset["llm_provider"], "openrouter", preset_name)
                self.assertEqual(preset["transcription_system"], "openrouter", preset_name)
                self.assertEqual(preset["tts_transport"], "openrouter", preset_name)

    def test_removed_trailers_preset_falls_back_to_ultra(self) -> None:
        with patch.object(preset_service, "_load_preset_overrides", return_value={}):
            fallback = preset_service.get_preset_config("trailers")
            ultra = preset_service.get_preset_config("ultra")

        self.assertEqual(fallback["llm_provider"], "openrouter")
        self.assertEqual(fallback["refinement_model_name"], ultra["refinement_model_name"])


if __name__ == "__main__":
    unittest.main()
