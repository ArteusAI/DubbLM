from __future__ import annotations

import unittest

from src.tts.gemini_tts_wrapper import GeminiTTSConfig
from src.tts.gemini_voice_catalog import (
    build_gemini_voice_entries,
    get_gemini_voice_gender,
    is_gemini_voice_gender_compatible,
    resolve_default_gemini_voice,
)


class GeminiVoiceCatalogTests(unittest.TestCase):
    def test_official_gemini_voice_genders_are_exposed(self) -> None:
        self.assertEqual(get_gemini_voice_gender("Enceladus"), "male")
        self.assertEqual(get_gemini_voice_gender("Achernar"), "female")
        self.assertEqual(get_gemini_voice_gender("Schedar"), "male")
        self.assertEqual(get_gemini_voice_gender("Zephyr"), "female")

    def test_gender_compatibility_treats_unknown_as_unrestricted(self) -> None:
        self.assertTrue(is_gemini_voice_gender_compatible("Kore", "unknown"))
        self.assertTrue(is_gemini_voice_gender_compatible("Puck", None))
        self.assertFalse(is_gemini_voice_gender_compatible("Kore", "male"))
        self.assertFalse(is_gemini_voice_gender_compatible("Puck", "female"))

    def test_gender_aware_default_replaces_incompatible_preferred_voice(self) -> None:
        resolved_voice = resolve_default_gemini_voice(
            "male",
            preferred_voice="Kore",
        )

        self.assertNotEqual(resolved_voice, "Kore")
        self.assertEqual(get_gemini_voice_gender(resolved_voice), "male")

    def test_gender_aware_default_respects_blocked_voices(self) -> None:
        resolved_voice = resolve_default_gemini_voice(
            "male",
            preferred_voice="Puck",
            blocked_voices=["Puck", "Enceladus"],
        )

        self.assertNotIn(resolved_voice, {"Puck", "Enceladus"})
        self.assertEqual(get_gemini_voice_gender(resolved_voice), "male")

    def test_resource_entries_use_official_genders(self) -> None:
        voice_entries = {entry["id"]: entry for entry in build_gemini_voice_entries()}

        self.assertEqual(voice_entries["Achernar"]["gender"], "female")
        self.assertEqual(voice_entries["Gacrux"]["gender"], "female")
        self.assertEqual(voice_entries["Schedar"]["gender"], "male")
        self.assertEqual(voice_entries["Umbriel"]["gender"], "male")

    def test_voice_similarity_defaults_are_tightened(self) -> None:
        config = GeminiTTSConfig()

        self.assertAlmostEqual(config.voice_similarity_threshold, 0.70)
        self.assertAlmostEqual(config.voice_similarity_relaxed_threshold, 0.65)


if __name__ == "__main__":
    unittest.main()
