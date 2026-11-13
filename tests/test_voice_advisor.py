"""Tests for the voice advice (timbre per speaker) of the delivery analysis."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from src.dubbing.video.gemini_emotion_analyzer import (
    DeliveryAnalysis,
    parse_delivery_response,
)
from src.tts.gemini_voice_catalog import (
    ALL_GEMINI_VOICES,
    GEMINI_VOICE_GENDERS,
    GEMINI_VOICE_TRAITS,
    get_gemini_voice_traits,
)
from src.tts.models import TTSSegmentData
from src.tts.openrouter_tts_wrapper import OpenRouterTTSWrapper
from src.tts.voice_advisor import (
    VoiceProfile,
    build_voice_profiles,
    normalize_age,
    profile_for_speaker,
    rank_voices_by_profile,
    sanitize_timbre,
    score_voice_against_profile,
)


class NormalizeAgeTests(unittest.TestCase):
    def test_known_buckets(self) -> None:
        self.assertEqual(normalize_age("Adult"), "adult")
        self.assertEqual(normalize_age("young adult"), "young adult")
        self.assertEqual(normalize_age("older"), "older")

    def test_aliases(self) -> None:
        self.assertEqual(normalize_age("senior"), "older")
        self.assertEqual(normalize_age("teenage"), "teen")
        self.assertEqual(normalize_age("Молодой"), "young adult")
        self.assertEqual(normalize_age("пожилой"), "older")

    def test_unknown_is_dropped(self) -> None:
        self.assertEqual(normalize_age(""), "")
        self.assertEqual(normalize_age("wizard"), "")


class SanitizeTimbreTests(unittest.TestCase):
    def test_markup_is_removed(self) -> None:
        self.assertEqual(sanitize_timbre('[low, "raspy"] voice'), "low, raspy voice")

    def test_length_is_capped(self) -> None:
        note = sanitize_timbre("word " * 40, max_chars=40)
        self.assertLessEqual(len(note), 40)

    def test_empty(self) -> None:
        self.assertEqual(sanitize_timbre(""), "")
        self.assertEqual(sanitize_timbre(None), "")


class VoiceProfileTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        profile = VoiceProfile.from_dict(
            {
                "speaker": "SPEAKER_A",
                "timbre": "low, slightly raspy",
                "age": "adult",
                "energy": 0.4,
                "accent": "neutral",
                "gender": "male",
            }
        )
        self.assertIsNotNone(profile)
        payload = profile.to_dict()
        self.assertEqual(payload["speaker"], "SPEAKER_A")
        self.assertEqual(payload["timbre"], "low, slightly raspy")
        self.assertEqual(payload["age"], "adult")
        self.assertAlmostEqual(payload["energy"], 0.4)

    def test_invalid_payload_is_dropped(self) -> None:
        self.assertIsNone(VoiceProfile.from_dict({"timbre": "low"}))
        self.assertIsNone(VoiceProfile.from_dict("nope"))

    def test_energy_is_clamped(self) -> None:
        profile = VoiceProfile.from_dict(
            {"speaker": "A", "energy": 7}
        )
        self.assertEqual(profile.energy, 1.0)

    def test_prompt_fragment_uses_the_timbre(self) -> None:
        profile = VoiceProfile(speaker="A", timbre="low, slightly raspy")
        self.assertEqual(profile.prompt_fragment(), "low, slightly raspy")

    def test_prompt_fragment_marks_the_energy(self) -> None:
        self.assertIn("intense", VoiceProfile(speaker="A", timbre="bright", energy=0.9).prompt_fragment())
        self.assertIn("restrained", VoiceProfile(speaker="A", timbre="soft", energy=0.1).prompt_fragment())

    def test_prompt_fragment_falls_back_to_the_age(self) -> None:
        self.assertIn("adult", VoiceProfile(speaker="A", age="adult").prompt_fragment())


class BuildProfilesTests(unittest.TestCase):
    def test_the_known_gender_wins(self) -> None:
        profiles = build_voice_profiles(
            [{"speaker": "SPEAKER_A", "timbre": "low", "gender": "unknown"}],
            {"SPEAKER_A": "male"},
        )
        self.assertEqual(profiles["SPEAKER_A"].gender, "male")

    def test_broken_entries_are_skipped(self) -> None:
        profiles = build_voice_profiles([{"timbre": "low"}, "junk"], {})
        self.assertEqual(profiles, {})

    def test_profile_for_speaker_accepts_dicts_and_objects(self) -> None:
        profiles = {
            "A": {"timbre": "deep", "speaker": "A"},
            "B": VoiceProfile(speaker="B", timbre="bright"),
        }
        self.assertEqual(profile_for_speaker(profiles, "A").timbre, "deep")
        self.assertEqual(profile_for_speaker(profiles, "B").timbre, "bright")
        self.assertIsNone(profile_for_speaker(profiles, "C"))
        self.assertIsNone(profile_for_speaker(None, "A"))


class ScoringTests(unittest.TestCase):
    def _profile(self, timbre: str, age: str = "") -> VoiceProfile:
        return VoiceProfile(speaker="A", timbre=timbre, age=age)

    def test_a_matching_voice_scores_higher(self) -> None:
        profile = self._profile("low, gravelly")
        deep = {"name": "Ivan", "description": "Deep, resonant male voice", "age": "adult"}
        bright = {"name": "Lena", "description": "Bright, cheerful female voice", "age": "adult"}
        self.assertGreater(
            score_voice_against_profile(profile, deep),
            score_voice_against_profile(profile, bright),
        )

    def test_russian_descriptions_are_understood(self) -> None:
        profile = self._profile("хриплый, низкий")
        rough = {"name": "Пётр", "description": "Скрипучий, грубый мужской голос", "age": "adult"}
        soft = {"name": "Аня", "description": "Мягкий, нежный женский голос", "age": "adult"}
        self.assertGreater(
            score_voice_against_profile(profile, rough),
            score_voice_against_profile(profile, soft),
        )

    def test_an_age_conflict_is_penalised(self) -> None:
        profile = self._profile("bright", age="older")
        senior = {"name": "Old", "description": "Mature, calm voice", "age": "senior"}
        child = {"name": "Kid", "description": "Bright, playful voice", "age": "child"}
        self.assertGreater(
            score_voice_against_profile(profile, senior),
            score_voice_against_profile(profile, child),
        )

    def test_without_traits_the_score_is_zero(self) -> None:
        self.assertEqual(score_voice_against_profile(self._profile(""), {"name": "X"}), 0.0)
        self.assertEqual(score_voice_against_profile(None, {"name": "X"}), 0.0)


class RankingTests(unittest.TestCase):
    META = {
        "deep": {"name": "Deep", "description": "Deep, resonant voice", "age": "adult"},
        "bright": {"name": "Bright", "description": "Bright, cheerful voice", "age": "adult"},
        "calm": {"name": "Calm", "description": "Calm, measured voice", "age": "adult"},
    }

    def test_the_best_match_comes_first(self) -> None:
        profile = VoiceProfile(speaker="A", timbre="low, gravelly")
        ranked = rank_voices_by_profile(
            profile, ["bright", "calm", "deep"], self.META.get
        )
        self.assertEqual(ranked[0], "deep")

    def test_without_a_profile_the_order_is_kept(self) -> None:
        ranked = rank_voices_by_profile(None, ["a", "b", "c"], self.META.get)
        self.assertEqual(ranked, ["a", "b", "c"])

    def test_the_limit_keeps_the_safety_net(self) -> None:
        profile = VoiceProfile(speaker="A", timbre="low, gravelly")
        ranked = rank_voices_by_profile(
            profile,
            ["bright", "calm", "deep"],
            self.META.get,
            limit=1,
            keep_voices=["calm"],
        )
        self.assertEqual(len(ranked), 2)
        self.assertIn("calm", ranked)


class GeminiCatalogTests(unittest.TestCase):
    def test_every_voice_has_traits(self) -> None:
        missing = [
            voice for voice in ALL_GEMINI_VOICES if voice not in GEMINI_VOICE_TRAITS
        ]
        self.assertEqual(missing, [])

    def test_the_traits_do_not_contradict_the_gender(self) -> None:
        for voice, traits in GEMINI_VOICE_TRAITS.items():
            self.assertIn(voice, GEMINI_VOICE_GENDERS, f"{voice} has no gender")
            self.assertTrue(traits.get("timbre"), f"{voice} has no timbre")
            self.assertTrue(traits.get("age"), f"{voice} has no age")

    def test_traits_lookup(self) -> None:
        self.assertEqual(get_gemini_voice_traits("Puck")["style"], "upbeat")
        self.assertEqual(get_gemini_voice_traits("nope"), {})
        self.assertEqual(get_gemini_voice_traits(None), {})


class DeliveryAnalysisParsingTests(unittest.TestCase):
    def test_the_speakers_section_is_parsed(self) -> None:
        payload = json.dumps(
            {
                "lines": [{"index": 0, "emotion": "tense", "intensity": 0.5}],
                "speakers": [
                    {
                        "speaker": "SPEAKER_A",
                        "timbre": "low, slightly raspy",
                        "age": "adult",
                        "energy": 0.4,
                        "accent": "neutral",
                    }
                ],
            }
        )
        analysis = parse_delivery_response(payload)
        self.assertEqual(len(analysis.lines), 1)
        self.assertEqual(len(analysis.speakers), 1)
        self.assertEqual(analysis.speakers[0]["speaker"], "SPEAKER_A")
        self.assertEqual(analysis.speakers[0]["timbre"], "low, slightly raspy")

    def test_broken_speaker_entries_are_dropped(self) -> None:
        payload = json.dumps(
            {"lines": [], "speakers": [{"timbre": "low"}, {"speaker": "A", "energy": "x"}]}
        )
        analysis = parse_delivery_response(payload)
        self.assertEqual(len(analysis.speakers), 1)
        self.assertEqual(analysis.speakers[0]["energy"], 0.0)

    def test_a_response_without_speakers_is_still_valid(self) -> None:
        analysis = parse_delivery_response(json.dumps({"lines": []}))
        self.assertEqual(analysis.speakers, [])

    def test_empty_payload(self) -> None:
        analysis = parse_delivery_response("")
        self.assertEqual(analysis.lines, [])
        self.assertEqual(analysis.speakers, [])

    def test_serialisation(self) -> None:
        analysis = DeliveryAnalysis(
            lines=[{"index": 0}], speakers=[{"speaker": "A"}]
        )
        restored = DeliveryAnalysis.from_dict(analysis.to_dict())
        self.assertEqual(len(restored.lines), 1)
        self.assertEqual(len(restored.speakers), 1)


class CandidateNarrowingTests(unittest.TestCase):
    FISH_META = {
        "deep_voice": {"name": "Ivan", "description": "Deep, resonant male voice", "age": "adult"},
        "bright_voice": {"name": "Lena", "description": "Bright, cheerful female voice", "age": "adult"},
        "calm_voice": {"name": "Pavel", "description": "Calm, measured male voice", "age": "adult"},
    }

    def test_drama3_candidates_are_ranked_by_the_advice(self) -> None:
        from src.tts.openrouter_tts_wrapper import OpenRouterTTSWrapper

        fake = mock.Mock()
        fake.speaker_voice_profiles = {
            "SPEAKER_A": {"speaker": "SPEAKER_A", "timbre": "low, gravelly"}
        }
        fake.voice_advice_max_candidates = 2
        fake.target_language = "en"
        fake._default_voice_for_speaker.return_value = "calm_voice"

        with mock.patch(
            "src.tts.openrouter_tts_wrapper.fish_voice_meta",
            return_value=self.FISH_META,
        ):
            ranked = OpenRouterTTSWrapper._apply_voice_advice(
                fake, ["bright_voice", "calm_voice", "deep_voice"], "SPEAKER_A"
            )

        self.assertEqual(ranked[0], "deep_voice")
        self.assertLessEqual(len(ranked), 3)
        # The gender aware default stays in the pool as a safety net.
        self.assertIn("calm_voice", ranked)

    def test_without_advice_the_candidates_are_untouched(self) -> None:
        from src.tts.openrouter_tts_wrapper import OpenRouterTTSWrapper

        fake = mock.Mock()
        fake.speaker_voice_profiles = {}
        ranked = OpenRouterTTSWrapper._apply_voice_advice(
            fake, ["bright_voice", "calm_voice"], "SPEAKER_A"
        )
        self.assertEqual(ranked, ["bright_voice", "calm_voice"])

    def test_gemini_candidates_follow_the_advice(self) -> None:
        from src.tts.gemini38_tts_wrapper import Gemini38TTSWrapper

        fake = mock.Mock()
        fake.speaker_voice_profiles = {
            "SPEAKER_A": {"speaker": "SPEAKER_A", "timbre": "deep, resonant"}
        }
        fake.voice_advice_max_candidates = 3
        fake._get_effective_speaker_gender.return_value = "male"
        fake._get_gender_aware_default_voice.return_value = "Puck"
        fake.voice_matcher.sample_embeddings = {
            "Puck": [0.0],
            "Charon": [0.0],
            "Kore": [0.0],
        }

        candidates = Gemini38TTSWrapper._voice_advice_candidates(fake, "SPEAKER_A")
        self.assertIsNotNone(candidates)
        # Only male voices with an embedding are candidates.
        self.assertNotIn("Kore", candidates)
        self.assertEqual(candidates[0], "Charon")

    def test_gemini_candidates_are_none_without_advice(self) -> None:
        from src.tts.gemini38_tts_wrapper import Gemini38TTSWrapper

        fake = mock.Mock()
        fake.speaker_voice_profiles = {}
        self.assertIsNone(Gemini38TTSWrapper._voice_advice_candidates(fake, "SPEAKER_A"))


class FactoryWiringTests(unittest.TestCase):
    def test_the_profiles_reach_the_wrapper(self) -> None:
        from src.tts import tts_factory

        captured = {}

        class FakeTTS:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def initialize(self):
                pass

            def set_voice_mapping(self, mapping):
                captured["mapping"] = mapping

            def set_voice_prompt_mapping(self, mapping):
                captured["prompts"] = mapping

            def set_speaker_voice_profiles(self, profiles):
                captured["profiles"] = profiles

        original = tts_factory.TTS_PROVIDERS.get("openrouter")
        tts_factory.TTS_PROVIDERS["openrouter"] = FakeTTS
        try:
            client = tts_factory.TTSFactory.create_tts(
                "openrouter",
                speaker_voice_profiles={"SPEAKER_A": {"timbre": "low, resonant"}},
                voice_advice_max_candidates=5,
            )
        finally:
            if original is not None:
                tts_factory.TTS_PROVIDERS["openrouter"] = original
            else:
                tts_factory.TTS_PROVIDERS.pop("openrouter", None)

        self.assertIsInstance(client, FakeTTS)
        self.assertEqual(
            captured["profiles"], {"SPEAKER_A": {"timbre": "low, resonant"}}
        )
        self.assertEqual(captured["kwargs"].get("voice_advice_max_candidates"), 5)


class VoiceNoteTests(unittest.TestCase):
    def test_the_voice_note_comes_from_the_profile(self) -> None:
        from src.tts.voice_advisor import voice_note_for_speaker

        profiles = {
            "SPEAKER_A": {"speaker": "SPEAKER_A", "timbre": "deep, resonant baritone"}
        }
        self.assertEqual(
            voice_note_for_speaker(profiles, "SPEAKER_A"), "deep, resonant baritone"
        )
        self.assertIsNone(voice_note_for_speaker(profiles, "SPEAKER_B"))
        self.assertIsNone(voice_note_for_speaker({}, "SPEAKER_A"))


class TTSInstructionTests(unittest.TestCase):
    def test_gemini38_style_carries_the_voice_advice(self) -> None:
        from src.tts.gemini38_tags import build_gemini38_prompt

        prompt = build_gemini38_prompt(
            "Привет",
            emotion="tense",
            emotion_note="low, controlled",
            voice_note="slightly raspy",
        )
        self.assertIn("slightly raspy", prompt.style)
        self.assertNotIn("raspy", prompt.text)

    def test_drama3_direction_carries_the_voice_advice(self) -> None:
        segment = TTSSegmentData(
            speaker="A",
            text="I'm fine.",
            emotion="tense",
            emotion_note="restrained",
            voice_note="slightly raspy",
        )
        fake_wrapper = mock.Mock()
        fake_wrapper.voice_prompt_mapping = {}
        prepared = OpenRouterTTSWrapper._prepare_drama3_speech_text(
            fake_wrapper, segment
        )
        self.assertIn("slightly raspy", prepared)
        self.assertTrue(prepared.endswith("I'm fine."))

    def test_the_segment_data_carries_the_voice_advice(self) -> None:
        segment = TTSSegmentData(speaker="A", text="hi", voice_note="deep")
        self.assertEqual(segment.voice_note, "deep")


if __name__ == "__main__":
    unittest.main()
