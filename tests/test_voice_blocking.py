from __future__ import annotations

import unittest
from unittest.mock import patch

from src.tts.openrouter_tts_wrapper import OpenRouterTTSWrapper
from src.tts.openrouter_voice_catalog import (
    FISH_BLOCKED_VOICES,
    FISH_DRAMA_TTS_MODEL,
    filter_fish_voices,
)
from src.tts.tts_factory import TTSConfig, TTSFactory


BLOCKED_MELLSTROY = "0a690dbeb3984a9f88cd39353880775f"  # Меллстрой (мем "скуф")
BLOCKED_ZELENSKY = "5f9e15229e1c4f01a0133ab9f2b69566"  # Зеленский (имитация человека)


class VoiceBlocklistTests(unittest.TestCase):
    def test_meme_voices_are_globally_blocked(self) -> None:
        self.assertIn(BLOCKED_MELLSTROY, FISH_BLOCKED_VOICES)

    def test_real_person_impersonations_are_globally_blocked(self) -> None:
        self.assertIn(BLOCKED_ZELENSKY, FISH_BLOCKED_VOICES)

    def test_filter_drops_blocked_voices(self) -> None:
        voices = [BLOCKED_MELLSTROY, BLOCKED_ZELENSKY, "a" * 32, "b" * 32]
        profiles = {
            voice_id: {
                "name": voice_id[:4],
                "gender": "male",
                "age": "adult",
                "character": False,
                "accent": "ru",
                "languages": ["ru"],
                "native": True,
                "models": [FISH_DRAMA_TTS_MODEL],
            }
            for voice_id in voices
        }

        with patch(
            "src.tts.openrouter_voice_catalog.fish_voice_meta",
            return_value=profiles,
        ):
            kept = filter_fish_voices(voices, language="ru", gender="male")
            allowed = filter_fish_voices(
                voices, language="ru", gender="male", allow_blocked=True
            )
            extra = filter_fish_voices(
                voices,
                language="ru",
                gender="male",
                extra_blocked=["a" * 32],
            )

        self.assertNotIn(BLOCKED_MELLSTROY, kept)
        self.assertNotIn(BLOCKED_ZELENSKY, kept)
        self.assertIn(BLOCKED_MELLSTROY, allowed)
        self.assertNotIn("a" * 32, extra)
        self.assertIn("b" * 32, extra)

    def test_wrapper_excludes_blocked_voices(self) -> None:
        wrapper = OpenRouterTTSWrapper(
            model=FISH_DRAMA_TTS_MODEL,
            target_language="ru",
            enable_voice_matching=False,
            enable_content_validation=False,
            blocked_voices=["b" * 32],
        )
        profiles = {
            voice_id: {
                "name": voice_id[:4],
                "gender": "male",
                "age": "adult",
                "character": False,
                "accent": "ru",
                "languages": ["ru"],
                "native": True,
                "models": [FISH_DRAMA_TTS_MODEL],
            }
            for voice_id in (BLOCKED_MELLSTROY, BLOCKED_ZELENSKY, "a" * 32, "b" * 32)
        }

        with patch(
            "src.tts.openrouter_voice_catalog.fish_voice_meta",
            return_value=profiles,
        ):
            with patch.object(
                wrapper, "_voices_for_model", return_value=list(profiles.keys())
            ):
                candidates = wrapper._candidate_voices("SPEAKER_00")

        self.assertNotIn(BLOCKED_MELLSTROY, candidates)
        self.assertNotIn(BLOCKED_ZELENSKY, candidates)
        self.assertNotIn("b" * 32, candidates)
        self.assertIn("a" * 32, candidates)

    def test_blocked_voice_falls_back_when_assigned_manually(self) -> None:
        wrapper = OpenRouterTTSWrapper(
            model=FISH_DRAMA_TTS_MODEL,
            target_language="ru",
            enable_voice_matching=False,
            enable_content_validation=False,
            blocked_voices=["b" * 32],
        )
        wrapper.default_voice = "a" * 32

        with patch.object(
            wrapper, "_voices_for_model", return_value=["a" * 32, "b" * 32]
        ):
            resolved = wrapper._validate_voice_name("b" * 32)

        self.assertEqual(resolved, "a" * 32)

    def test_default_voice_differs_per_speaker(self) -> None:
        """Two speakers of the same gender must not get the same voice."""
        wrapper = OpenRouterTTSWrapper(
            model=FISH_DRAMA_TTS_MODEL,
            target_language="ru",
            enable_voice_matching=False,
            enable_content_validation=False,
            speaker_metadata={
                "SPEAKER_00": {"inferredGender": "male", "inferredConfidence": 1.0},
                "SPEAKER_01": {"inferredGender": "male", "inferredConfidence": 1.0},
            },
        )

        with patch(
            "src.tts.openrouter_tts_wrapper.filter_fish_voices",
            return_value=["a" * 32, "b" * 32],
        ):
            first = wrapper._default_voice_for_speaker("SPEAKER_00")
            wrapper.voice_mapping["SPEAKER_00"] = first
            second = wrapper._default_voice_for_speaker("SPEAKER_01")

        self.assertEqual(first, "a" * 32)
        self.assertEqual(second, "b" * 32)

    def test_globally_blocked_voice_can_still_be_assigned_manually(self) -> None:
        """The global list only prevents automatic selection."""
        wrapper = OpenRouterTTSWrapper(
            model=FISH_DRAMA_TTS_MODEL,
            target_language="ru",
            enable_voice_matching=False,
            enable_content_validation=False,
        )
        wrapper.default_voice = "a" * 32

        with patch.object(
            wrapper,
            "_voices_for_model",
            return_value=["a" * 32, BLOCKED_ZELENSKY],
        ):
            resolved = wrapper._validate_voice_name(BLOCKED_ZELENSKY)

        self.assertEqual(resolved, BLOCKED_ZELENSKY)

    def test_fallback_default_voice_is_pinned_for_the_speaker(self) -> None:
        """Even the fallback voice is pinned, so the next speaker gets another one."""
        wrapper = OpenRouterTTSWrapper(
            model=FISH_DRAMA_TTS_MODEL,
            target_language="ru",
            enable_voice_matching=True,
            enable_content_validation=False,
        )
        wrapper.default_voice = "a" * 32
        wrapper.client = object()
        wrapper.voice_matcher = object()

        with patch.object(
            wrapper, "_default_voice_for_speaker", return_value="a" * 32
        ):
            # No reference audio -> the fallback default is used and pinned.
            pinned = wrapper.find_and_pin_voice_for_speaker(
                "SPEAKER_00", "/nonexistent.wav"
            )

        self.assertEqual(pinned, "a" * 32)
        self.assertEqual(wrapper.voice_mapping.get("SPEAKER_00"), "a" * 32)

    def test_factory_passes_blocked_voices_to_openrouter(self) -> None:
        config = TTSConfig(
            provider="openrouter",
            model=FISH_DRAMA_TTS_MODEL,
            blocked_voices=["b" * 32],
        )

        captured: dict = {}

        class _FakeWrapper:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def initialize(self) -> None:
                return None

            def set_voice_mapping(self, mapping) -> None:
                return None

            def set_voice_prompt_mapping(self, mapping) -> None:
                return None

        with patch.dict(
            "src.tts.tts_factory.TTS_PROVIDERS", {"openrouter": _FakeWrapper}
        ):
            TTSFactory.create_tts_client(config)

        self.assertEqual(captured.get("blocked_voices"), ["b" * 32])


if __name__ == "__main__":
    unittest.main()
