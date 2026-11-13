from __future__ import annotations

import unittest
from unittest.mock import patch

from src.tts.models import TTSSegmentData
from src.tts.openrouter_tts_wrapper import OpenRouterTTSWrapper
from src.tts.openrouter_voice_catalog import FISH_DRAMA_TTS_MODEL
from src.translation.llm_translator import LLMTranslator


def _wrapper() -> OpenRouterTTSWrapper:
    return OpenRouterTTSWrapper(
        model=FISH_DRAMA_TTS_MODEL,
        target_language="en",
        enable_voice_matching=False,
        enable_content_validation=False,
    )


class Drama3SpeechTextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wrapper = _wrapper()

    def test_direction_is_prepended_from_emotion_and_style(self) -> None:
        segment = TTSSegmentData(
            speaker="SPEAKER_00",
            text="I'm fine. Really.",
            emotion="nervous",
            style_prompt="Speak in a calm, measured narrator voice:",
        )

        prepared = self.wrapper._prepare_speech_text(segment)

        self.assertTrue(prepared.startswith("["))
        self.assertIn("nervous", prepared)
        self.assertIn("calm, measured narrator voice", prepared)
        self.assertTrue(prepared.endswith("I'm fine. Really."))

    def test_editor_direction_is_kept_and_merged(self) -> None:
        segment = TTSSegmentData(
            speaker="SPEAKER_00",
            text="[fast, nervous] I'm fine. Really.",
            emotion="happy",
        )

        prepared = self.wrapper._prepare_speech_text(segment)

        self.assertIn("fast, nervous", prepared)
        self.assertIn("happy", prepared)
        self.assertEqual(prepared.count("["), 1)

    def test_filler_tags_are_dropped(self) -> None:
        # [uhm] would be spoken aloud as "Um" by Drama 3.
        segment = TTSSegmentData(
            speaker="SPEAKER_00",
            text="I'm fine. [uhm] Really.",
            emotion="neutral",
        )

        prepared = self.wrapper._prepare_speech_text(segment)

        self.assertNotIn("uhm", prepared)
        self.assertEqual(prepared, "I'm fine. Really.")

    def test_plain_text_stays_plain(self) -> None:
        segment = TTSSegmentData(speaker="SPEAKER_00", text="Go on without me.")

        self.assertEqual(self.wrapper._prepare_speech_text(segment), "Go on without me.")

    def test_style_prompts_become_directions(self) -> None:
        cases = {
            "Speak with natural conversational energy, clear articulation:":
                "natural conversational energy, clear articulation",
            "Read aloud in a calm, authoritative and articulate manner with measured pacing:":
                "calm, authoritative and articulate manner with measured pacing",
        }
        for style_prompt, expected in cases.items():
            segment = TTSSegmentData(
                speaker="SPEAKER_00",
                text="The report is ready.",
                style_prompt=style_prompt,
            )

            prepared = self.wrapper._prepare_speech_text(segment)

            self.assertEqual(prepared, f"[{expected}] The report is ready.")

    def test_long_style_prompt_keeps_the_delivery_part(self) -> None:
        style_prompt = (
            "Synthesize only the transcript below in the language it is written; "
            "do not read these directions. Deliver it like a modern news explainer: "
            "confident, crisp broadcast narration with forward momentum:"
        )
        segment = TTSSegmentData(
            speaker="SPEAKER_00",
            text="The report is ready.",
            style_prompt=style_prompt,
        )

        prepared = self.wrapper._prepare_speech_text(segment)

        self.assertIn("confident, crisp broadcast narration", prepared)
        self.assertNotIn("Synthesize only the transcript", prepared)

    def test_qwen_and_grok_paths_are_untouched(self) -> None:
        qwen = OpenRouterTTSWrapper(
            model="qwen/qwen-audio-3.0-tts-flash",
            target_language="en",
            enable_voice_matching=False,
            enable_content_validation=False,
        )
        grok = OpenRouterTTSWrapper(
            model="x-ai/grok-voice-tts-1.0",
            target_language="en",
            enable_voice_matching=False,
            enable_content_validation=False,
        )
        segment = TTSSegmentData(speaker="SPEAKER_00", text="[excited] Hi there.")

        self.assertTrue(qwen._prepare_speech_text(segment).startswith("[excited]"))
        self.assertEqual(grok._prepare_speech_text(segment), "Hi there.")


class Drama3EditorGuidanceTests(unittest.TestCase):
    def _translator(self) -> LLMTranslator:
        return LLMTranslator(
            llm_provider="openrouter",
            tts_system="openrouter",
            tts_model=FISH_DRAMA_TTS_MODEL,
            enable_llm_editor=True,
        )

    def test_editor_guidance_mentions_drama_directions(self) -> None:
        translator = self._translator()

        guidance = translator._build_editor_tts_guidance([{"speaker": "SPEAKER_00"}])

        self.assertIn("Drama 3", guidance)
        self.assertIn("<whisper>", guidance)
        self.assertNotIn("Qwen-Audio-3.0 demo tags", guidance)

    def test_translation_guidance_mentions_drama_directions(self) -> None:
        translator = LLMTranslator(
            llm_provider="openrouter",
            tts_system="openrouter",
            tts_model=FISH_DRAMA_TTS_MODEL,
        )

        guidance = translator._build_translation_tts_guidance(["SPEAKER_00"])

        self.assertIn("Drama 3", guidance)
        self.assertIn("stage direction", guidance)


class Drama3CostTests(unittest.TestCase):
    def test_pricing_is_registered(self) -> None:
        from src.dubbing.debug.cost_tracker import DEFAULT_PRICING_CATALOG

        models = DEFAULT_PRICING_CATALOG["tts"]["openrouter"]["models"]

        self.assertIn(FISH_DRAMA_TTS_MODEL, models)
        self.assertEqual(models[FISH_DRAMA_TTS_MODEL]["per_1m_chars"], 2.00)


class Drama3VoiceLibraryTests(unittest.TestCase):
    def test_fish_library_parsing_keeps_language_and_gender(self) -> None:
        from src.tts import openrouter_voice_catalog as catalog

        payload = {
            "items": [
                {
                    "_id": "a" * 32,
                    "title": "Test Voice",
                    "description": "A test voice.",
                    "tags": ["male", "narration"],
                    "languages": ["ru", "en"],
                    "task_count": 42,
                    "samples": [{"audio": "https://example.invalid/sample.mp3"}],
                }
            ]
        }

        with patch.object(catalog.requests, "get") as get_mock:
            get_mock.return_value.json.return_value = payload
            get_mock.return_value.raise_for_status.return_value = None
            catalog._fish_voice_cache.clear()

            voices = catalog.fish_voice_meta(language="ru", allow_live=True)

        meta = voices["a" * 32]
        self.assertEqual(meta["name"], "Test Voice")
        self.assertEqual(meta["gender"], "male")
        self.assertIn("ru", meta["languages"])
        self.assertEqual(meta["preview_url"], "https://example.invalid/sample.mp3")
        self.assertEqual(meta["models"], [FISH_DRAMA_TTS_MODEL])

    def test_fish_library_failure_falls_back_to_verified_voices(self) -> None:
        from src.tts import openrouter_voice_catalog as catalog

        with patch.object(catalog.requests, "get") as get_mock:
            get_mock.side_effect = RuntimeError("network down")
            catalog._fish_voice_cache.clear()

            voices = catalog.fish_voice_meta(language="ru", allow_live=True)

        self.assertTrue(voices)
        for voice_id in voices:
            self.assertTrue(catalog.is_fish_voice_id(voice_id))

    def test_library_is_queried_with_the_target_language(self) -> None:
        from src.tts import openrouter_voice_catalog as catalog

        with patch.object(catalog.requests, "get") as get_mock:
            get_mock.return_value.json.return_value = {"items": []}
            get_mock.return_value.raise_for_status.return_value = None
            catalog._fish_voice_cache.clear()

            catalog.fish_voice_meta(language="ru", allow_live=True)

        self.assertTrue(get_mock.called)
        self.assertEqual(get_mock.call_args.kwargs["params"]["language"], "ru")

    def test_profile_derives_gender_age_and_character(self) -> None:
        from src.tts import openrouter_voice_catalog as catalog

        cases = [
            (["male", "young"], "male", "young", False),
            (["female", "child"], "female", "child", False),
            (["female", "middle-aged", "anime"], "female", "adult", True),
            (["male", "old"], "male", "senior", False),
            ([], "unknown", "unknown", False),
        ]
        for tags, gender, age, character in cases:
            item = {
                "_id": "b" * 32,
                "title": "Голос",
                "description": "Описание",
                "tags": tags,
                "languages": ["ru"],
                "task_count": 1,
            }
            built = catalog._fish_profile_from_item(item, "ru")
            self.assertIsNotNone(built)
            _, profile = built
            self.assertEqual(profile["gender"], gender)
            self.assertEqual(profile["age"], age)
            self.assertEqual(profile["character"], character)
            self.assertEqual(profile["accent"], "ru")

    def test_native_requires_the_target_language(self) -> None:
        from src.tts import openrouter_voice_catalog as catalog

        with patch.object(catalog.requests, "get") as get_mock:
            get_mock.return_value.json.return_value = {
                "items": [
                    {
                        "_id": "c" * 32,
                        "title": "Русский голос",
                        "description": "Описывает звучание.",
                        "tags": ["male"],
                        "languages": ["ru"],
                        "task_count": 10,
                    },
                    {
                        "_id": "d" * 32,
                        "title": "Adam",
                        "description": "A warm english voice.",
                        "tags": ["male"],
                        "languages": ["ru"],
                        "task_count": 20,
                    },
                ]
            }
            get_mock.return_value.raise_for_status.return_value = None
            catalog._fish_voice_cache.clear()

            voices = catalog.fish_voice_meta(language="ru", allow_live=True)

        self.assertTrue(voices["c" * 32]["native"])
        # Cloned from an English sample: claimed ru but no cyrillic text.
        self.assertFalse(voices["d" * 32]["native"])

    def test_filter_keeps_gender_age_and_accent(self) -> None:
        from src.tts import openrouter_voice_catalog as catalog

        payload = {
            "items": [
                {
                    "_id": "1" * 32,
                    "title": "Мужчина",
                    "description": "Мужской голос для чтения.",
                    "tags": ["male", "middle-aged"],
                    "languages": ["ru"],
                    "task_count": 30,
                },
                {
                    "_id": "2" * 32,
                    "title": "Женщина",
                    "description": "Женский голос для чтения.",
                    "tags": ["female", "middle-aged"],
                    "languages": ["ru"],
                    "task_count": 20,
                },
                {
                    "_id": "3" * 32,
                    "title": "Ребёнок",
                    "description": "Детский голос для сказок.",
                    "tags": ["female", "child"],
                    "languages": ["ru"],
                    "task_count": 10,
                },
                {
                    "_id": "4" * 32,
                    "title": "Иностранец",
                    "description": "An english voice.",
                    "tags": ["male", "middle-aged"],
                    "languages": ["en"],
                    "task_count": 40,
                },
            ]
        }
        with patch.object(catalog.requests, "get") as get_mock:
            get_mock.return_value.json.return_value = payload
            get_mock.return_value.raise_for_status.return_value = None
            catalog._fish_voice_cache.clear()

            voices = catalog.fish_voice_meta(language="ru", allow_live=True)
            male = catalog.filter_fish_voices(
                list(voices.keys()), language="ru", gender="male"
            )
            female = catalog.filter_fish_voices(
                list(voices.keys()), language="ru", gender="female"
            )
            relaxed = catalog.filter_fish_voices(
                list(voices.keys()), language="ru", gender="male", native_only=False
            )

        self.assertIn("1" * 32, male)
        self.assertNotIn("2" * 32, male)
        self.assertNotIn("3" * 32, male)
        self.assertNotIn("4" * 32, male)
        self.assertIn("2" * 32, female)
        self.assertNotIn("3" * 32, female)
        # The foreign accent is only allowed when the strict filter is relaxed.
        self.assertIn("4" * 32, relaxed)
        self.assertNotIn("4" * 32, male)

    def test_default_voice_follows_gender_and_language(self) -> None:
        from src.tts import openrouter_voice_catalog as catalog

        payload = {
            "items": [
                {
                    "_id": "5" * 32,
                    "title": "Мужской рассказчик",
                    "description": "Спокойный мужской голос.",
                    "tags": ["male", "middle-aged"],
                    "languages": ["ru"],
                    "task_count": 50,
                },
                {
                    "_id": "6" * 32,
                    "title": "Женская рассказчица",
                    "description": "Спокойный женский голос.",
                    "tags": ["female", "middle-aged"],
                    "languages": ["ru"],
                    "task_count": 40,
                },
            ]
        }
        with patch.object(catalog.requests, "get") as get_mock:
            get_mock.return_value.json.return_value = payload
            get_mock.return_value.raise_for_status.return_value = None
            catalog._fish_voice_cache.clear()

            male_default = catalog.default_voice_for(
                model=FISH_DRAMA_TTS_MODEL, language="ru", speaker_gender="male"
            )
            female_default = catalog.default_voice_for(
                model=FISH_DRAMA_TTS_MODEL, language="ru", speaker_gender="female"
            )
            meta = catalog.fish_voice_meta(language="ru", allow_live=True)

        # The default voice must match the speaker gender and speak the
        # dubbing language natively.
        self.assertEqual(meta[male_default]["gender"], "male")
        self.assertTrue(meta[male_default]["native"])
        self.assertEqual(meta[female_default]["gender"], "female")
        self.assertTrue(meta[female_default]["native"])
        self.assertNotEqual(male_default, female_default)

class Drama3GenderAwareSelectionTests(unittest.TestCase):
    """Voice matching must respect speaker gender, age and accent."""

    PROFILES = {
        "1" * 32: {
            "name": "Мужчина", "gender": "male", "age": "adult", "character": False,
            "accent": "ru", "languages": ["ru"], "native": True, "models": [FISH_DRAMA_TTS_MODEL],
        },
        "2" * 32: {
            "name": "Женщина", "gender": "female", "age": "adult", "character": False,
            "accent": "ru", "languages": ["ru"], "native": True, "models": [FISH_DRAMA_TTS_MODEL],
        },
        "3" * 32: {
            "name": "Ребёнок", "gender": "female", "age": "child", "character": False,
            "accent": "ru", "languages": ["ru"], "native": True, "models": [FISH_DRAMA_TTS_MODEL],
        },
        "4" * 32: {
            "name": "Иностранец", "gender": "male", "age": "adult", "character": False,
            "accent": "en", "languages": ["en"], "native": False, "models": [FISH_DRAMA_TTS_MODEL],
        },
    }

    def _wrapper(self, metadata):
        wrapper = OpenRouterTTSWrapper(
            model=FISH_DRAMA_TTS_MODEL,
            target_language="ru",
            enable_voice_matching=False,
            enable_content_validation=False,
            speaker_metadata=metadata,
        )
        meta_patcher = patch(
            "src.tts.openrouter_voice_catalog.fish_voice_meta",
            return_value={key: dict(value) for key, value in self.PROFILES.items()},
        )
        meta_patcher.start()
        self.addCleanup(meta_patcher.stop)
        voices_patcher = patch(
            "src.tts.openrouter_tts_wrapper.voices_for_model",
            return_value=list(self.PROFILES.keys()),
        )
        voices_patcher.start()
        self.addCleanup(voices_patcher.stop)
        return wrapper

    def test_male_speaker_only_gets_male_voices(self) -> None:
        wrapper = self._wrapper({
            "SPEAKER_00": {"inferredGender": "male", "inferredConfidence": 0.9},
        })

        candidates = wrapper._candidate_voices("SPEAKER_00")

        self.assertEqual(candidates, ["1" * 32])

    def test_female_speaker_never_gets_a_child_voice(self) -> None:
        wrapper = self._wrapper({
            "SPEAKER_00": {"inferredGender": "female", "inferredConfidence": 0.9},
        })

        candidates = wrapper._candidate_voices("SPEAKER_00")

        self.assertEqual(candidates, ["2" * 32])
        self.assertNotIn("3" * 32, candidates)

    def test_unknown_gender_keeps_both_genders_but_native_only(self) -> None:
        wrapper = self._wrapper({})

        candidates = wrapper._candidate_voices("SPEAKER_00")

        self.assertIn("1" * 32, candidates)
        self.assertIn("2" * 32, candidates)
        self.assertNotIn("3" * 32, candidates)
        self.assertNotIn("4" * 32, candidates)

    def test_default_voice_follows_the_speaker_gender(self) -> None:
        wrapper = self._wrapper({
            "SPEAKER_00": {"inferredGender": "male", "inferredConfidence": 0.9},
            "SPEAKER_01": {"inferredGender": "female", "inferredConfidence": 0.9},
        })

        self.assertEqual(wrapper._default_voice_for_speaker("SPEAKER_00"), "1" * 32)
        self.assertEqual(wrapper._default_voice_for_speaker("SPEAKER_01"), "2" * 32)

    def test_matcher_respects_the_candidate_allow_list(self) -> None:
        import numpy as np

        from src.utils.voice_matcher import VoiceMatcher

        matcher = VoiceMatcher.__new__(VoiceMatcher)
        matcher.sample_embeddings = {
            "1" * 32: np.array([1.0, 0.0]),
            "2" * 32: np.array([0.0, 1.0]),
        }
        reference = [np.array([0.99, 0.01])]

        # Without the allow list the male voice wins.
        self.assertEqual(
            matcher.find_best_matching_voice_multi_segment(reference),
            "1" * 32,
        )
        # With the allow list only the listed voice may be picked.
        self.assertEqual(
            matcher.find_best_matching_voice_multi_segment(
                reference, candidate_voices=["2" * 32]
            ),
            "2" * 32,
        )


if __name__ == "__main__":
    unittest.main()
