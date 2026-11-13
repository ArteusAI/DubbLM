from __future__ import annotations

import unittest
from unittest.mock import patch

from src.tts.openrouter_tts_wrapper import OpenRouterTTSWrapper
from src.tts.openrouter_voice_catalog import (
    FISH_BLOCKED_VOICES,
    FISH_CURATED_VOICES,
    FISH_DRAMA_TTS_MODEL,
    build_openrouter_voice_entries,
    curated_fish_voice_ids,
    default_voice_for,
    is_fish_voice_id,
    is_voiceless_tts_model,
    voices_for_model,
)


class OpenRouterDramaVoiceTests(unittest.TestCase):
    def test_drama_3_is_not_voiceless(self) -> None:
        # Drama 3 accepts Fish voice ids; only unknown ids are rejected (400).
        self.assertFalse(is_voiceless_tts_model(FISH_DRAMA_TTS_MODEL))
        self.assertFalse(is_voiceless_tts_model("Fish-Audio/Drama-3"))
        self.assertFalse(is_voiceless_tts_model("qwen/qwen-audio-3.0-tts-flash"))
        self.assertFalse(is_voiceless_tts_model(None))

    def test_drama_3_exposes_fish_voices(self) -> None:
        voices = voices_for_model(FISH_DRAMA_TTS_MODEL, allow_live=False)

        self.assertTrue(voices)
        self.assertTrue(all(is_fish_voice_id(voice) for voice in voices))

    def test_drama_3_default_voice_is_a_fish_voice(self) -> None:
        default = default_voice_for(model=FISH_DRAMA_TTS_MODEL, language="ru")

        self.assertTrue(default)
        self.assertTrue(is_fish_voice_id(default))

    def test_drama_3_entries_declare_their_model(self) -> None:
        entries = build_openrouter_voice_entries(model=FISH_DRAMA_TTS_MODEL)

        self.assertTrue(entries)
        for entry in entries:
            self.assertEqual(entry["provider"], "openrouter")
            self.assertIn(FISH_DRAMA_TTS_MODEL, entry.get("models") or [])

    def test_other_models_keep_their_voices(self) -> None:
        self.assertIn("loongjohn", voices_for_model("qwen/qwen-audio-3.0-tts-flash", allow_live=False))
        self.assertIn("eve", voices_for_model("x-ai/grok-voice-tts-1.0", allow_live=False))
        self.assertTrue(default_voice_for(model="qwen/qwen-audio-3.0-tts-flash"))


class CuratedDrama3CastTests(unittest.TestCase):
    def test_curated_cast_has_the_expected_shape(self) -> None:
        self.assertEqual(len(FISH_CURATED_VOICES), 15)
        genders = [meta["gender"] for meta in FISH_CURATED_VOICES.values()]
        self.assertEqual(genders.count("male"), 10)
        self.assertEqual(genders.count("female"), 5)
        for voice_id, meta in FISH_CURATED_VOICES.items():
            self.assertTrue(is_fish_voice_id(voice_id))
            self.assertNotIn(voice_id, FISH_BLOCKED_VOICES)
            self.assertEqual(meta["models"], [FISH_DRAMA_TTS_MODEL])
            self.assertEqual(meta["languages"], ["ru"])
            self.assertNotEqual(meta["age"], "child")
            self.assertNotEqual(meta["age"], "teen")
            self.assertFalse(meta["character"])
            self.assertTrue(meta["curated"])
            self.assertTrue(meta["preview_url"].startswith("https://platform.r2.fish.audio/"))
            self.assertNotIn("?", meta["preview_url"])

    def test_russian_drama3_uses_only_the_curated_cast(self) -> None:
        voices = voices_for_model(FISH_DRAMA_TTS_MODEL, language="ru", allow_live=False)
        self.assertEqual(voices, list(FISH_CURATED_VOICES.keys()))
        self.assertEqual(voices, curated_fish_voice_ids("ru"))

    def test_other_languages_do_not_get_the_russian_cast(self) -> None:
        voices = voices_for_model(FISH_DRAMA_TTS_MODEL, language="en", allow_live=False)
        self.assertTrue(voices)
        for voice_id in FISH_CURATED_VOICES:
            self.assertNotIn(voice_id, voices)

    def test_drama3_entries_are_marked_as_curated(self) -> None:
        entries = build_openrouter_voice_entries(
            model=FISH_DRAMA_TTS_MODEL,
            language="ru",
        )
        self.assertEqual(len(entries), len(FISH_CURATED_VOICES))
        for entry in entries:
            self.assertTrue(entry["curated"])
            self.assertEqual(entry["language"], "ru")
            self.assertIn(FISH_DRAMA_TTS_MODEL, entry["models"])

    def test_defaults_are_the_first_curated_male_and_female(self) -> None:
        male = default_voice_for(
            model=FISH_DRAMA_TTS_MODEL, language="ru", speaker_gender="male"
        )
        female = default_voice_for(
            model=FISH_DRAMA_TTS_MODEL, language="ru", speaker_gender="female"
        )
        curated = list(FISH_CURATED_VOICES.items())
        first_male = next(voice_id for voice_id, meta in curated if meta["gender"] == "male")
        first_female = next(voice_id for voice_id, meta in curated if meta["gender"] == "female")
        self.assertEqual(male, first_male)
        self.assertEqual(female, first_female)


class OpenRouterWrapperVoiceTests(unittest.TestCase):
    def _wrapper(self, model: str) -> OpenRouterTTSWrapper:
        return OpenRouterTTSWrapper(
            model=model,
            target_language="ru",
            enable_voice_matching=False,
            enable_content_validation=False,
        )

    def test_drama_3_wrapper_keeps_a_fish_voice(self) -> None:
        wrapper = self._wrapper(FISH_DRAMA_TTS_MODEL)

        self.assertTrue(is_fish_voice_id(wrapper.default_voice))
        self.assertEqual(wrapper._validate_voice_name(wrapper.default_voice), wrapper.default_voice)

    def test_drama_3_wrapper_falls_back_for_foreign_voices(self) -> None:
        wrapper = self._wrapper(FISH_DRAMA_TTS_MODEL)

        # A Qwen voice is not a valid Fish voice id.
        self.assertEqual(wrapper._validate_voice_name("loongjohn"), wrapper.default_voice)
        # Unknown ids fall back as well.
        self.assertEqual(wrapper._validate_voice_name("not-a-voice"), wrapper.default_voice)

    def test_drama_3_wrapper_accepts_custom_fish_voice_ids(self) -> None:
        wrapper = self._wrapper(FISH_DRAMA_TTS_MODEL)
        custom = "0123456789abcdef0123456789abcdef"

        self.assertEqual(wrapper._validate_voice_name(custom), custom)

    def test_wrapper_keeps_the_voice_for_voiced_models(self) -> None:
        wrapper = self._wrapper("qwen/qwen-audio-3.0-tts-flash")

        self.assertTrue(wrapper.default_voice)
        self.assertEqual(wrapper._validate_voice_name("loongjohn"), "loongjohn")

    def test_voiced_requests_send_the_voice_field(self) -> None:
        wrapper = self._wrapper(FISH_DRAMA_TTS_MODEL)
        voice = wrapper.default_voice

        with patch("src.tts.openrouter_tts_wrapper.requests.post") as post_mock:
            post_mock.return_value.status_code = 200
            post_mock.return_value.content = b"audio-bytes"

            audio = wrapper._request_speech_bytes_direct("Hello there.", 1.0)
            self.assertEqual(audio, b"audio-bytes")
            payload = post_mock.call_args.kwargs["json"]
            self.assertNotIn("voice", payload)

        with patch.object(wrapper, "client") as client_mock:
            client_mock.audio.speech.create.return_value.content = b"audio-bytes"

            audio = wrapper._request_speech_bytes("Hello there.", voice, 1.0)

        self.assertEqual(audio, b"audio-bytes")
        create_kwargs = client_mock.audio.speech.create.call_args.kwargs
        self.assertEqual(create_kwargs["voice"], voice)
        self.assertEqual(create_kwargs["model"], FISH_DRAMA_TTS_MODEL)
        self.assertEqual(create_kwargs["input"], "Hello there.")


if __name__ == "__main__":
    unittest.main()
