from __future__ import annotations

import unittest

from src.dubbing.audio.vocal_event_detector import VocalEvent
from src.dubbing.core.smart_dubbing import (
    SmartDubbing,
    apply_vocal_tag,
    vocal_event_signature,
)


class _StubDubbing:
    def __init__(self, systems=None, **config):
        self.systems = systems or {}
        self.config = {"vocal_event_max_per_segment": 3}
        self.config.update(config)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def _get_tts_system_for_speaker(self, speaker: str) -> str:
        return self.systems.get(speaker, "gemini38")


def _segment(speaker: str, start: float, end: float, text: str) -> dict:
    return {
        "speaker": speaker,
        "start": start,
        "end": end,
        "translation": text,
        "long_translation": text + " (длинно)",
        "short_translation": "",
        "very_short_translation": "",
    }


class ApplyVocalTagTests(unittest.TestCase):
    def test_tag_is_inserted_into_variants(self) -> None:
        segment = _segment("SPEAKER_A", 0.0, 4.0, "раз два три четыре")
        changed = apply_vocal_tag(segment, 0.5, "scream")
        self.assertTrue(changed)
        self.assertIn("[screaming]", segment["translation"])
        self.assertIn("[screaming]", segment["long_translation"])

    def test_angle_tag_kind(self) -> None:
        segment = _segment("SPEAKER_A", 0.0, 4.0, "раз два три четыре")
        apply_vocal_tag(segment, 0.5, "whisper")
        self.assertIn("<whispers>", segment["translation"])

    def test_idempotent_for_same_kind(self) -> None:
        segment = _segment("SPEAKER_A", 0.0, 4.0, "раз два три четыре")
        apply_vocal_tag(segment, 0.5, "cry")
        first = segment["translation"]
        changed = apply_vocal_tag(segment, 0.5, "cry")
        self.assertFalse(changed)
        self.assertEqual(segment["translation"], first)


class InjectVocalTagsTests(unittest.TestCase):
    def test_distinct_kinds_on_one_segment(self) -> None:
        stub = _StubDubbing()
        segments = [_segment("SPEAKER_A", 0.0, 6.0, "привет как дела друг")]
        events = [VocalEvent("scream", 1.0, 1.5, 0.9), VocalEvent("cry", 3.0, 3.5, 0.8)]
        stats = SmartDubbing.inject_vocal_tags(stub, segments, events)
        self.assertEqual(stats["tagged_segments"], 2)
        self.assertIn("[screaming]", segments[0]["translation"])
        self.assertIn("[crying]", segments[0]["translation"])

    def test_duplicate_kind_is_skipped(self) -> None:
        stub = _StubDubbing()
        segments = [_segment("SPEAKER_A", 0.0, 6.0, "привет как дела друг")]
        events = [VocalEvent("scream", 1.0, 1.5, 0.9), VocalEvent("scream", 3.0, 3.5, 0.8)]
        stats = SmartDubbing.inject_vocal_tags(stub, segments, events)
        self.assertEqual(stats["tagged_segments"], 1)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(segments[0]["translation"].count("[screaming]"), 1)

    def test_max_per_segment_caps_distinct_kinds(self) -> None:
        stub = _StubDubbing(vocal_event_max_per_segment=2)
        segments = [_segment("SPEAKER_A", 0.0, 8.0, "раз два три четыре пять шесть")]
        events = [
            VocalEvent("scream", 1.0, 1.4, 0.9),
            VocalEvent("cry", 2.5, 2.9, 0.9),
            VocalEvent("gasp", 4.0, 4.4, 0.9),
            VocalEvent("sigh", 6.0, 6.4, 0.9),
        ]
        stats = SmartDubbing.inject_vocal_tags(stub, segments, events)
        self.assertEqual(stats["tagged_segments"], 2)
        self.assertEqual(stats["skipped"], 2)

    def test_only_gemini38_speakers_are_tagged(self) -> None:
        stub = _StubDubbing({"SPEAKER_A": "openai"})
        segments = [_segment("SPEAKER_A", 0.0, 4.0, "привет как дела")]
        stats = SmartDubbing.inject_vocal_tags(stub, segments, [VocalEvent("scream", 1.0, 1.5, 0.9)])
        self.assertEqual(stats["tagged_segments"], 0)
        self.assertNotIn("[screaming]", segments[0]["translation"])

    def test_far_gap_event_is_skipped(self) -> None:
        stub = _StubDubbing()
        segments = [_segment("SPEAKER_A", 0.0, 4.0, "привет как дела")]
        stats = SmartDubbing.inject_vocal_tags(stub, segments, [VocalEvent("scream", 30.0, 31.0, 0.9)])
        self.assertEqual(stats["tagged_segments"], 0)
        self.assertEqual(stats["skipped"], 1)

    def test_laugh_events_are_ignored_by_vocal_injection(self) -> None:
        stub = _StubDubbing()
        segments = [_segment("SPEAKER_A", 0.0, 4.0, "привет как дела")]
        stats = SmartDubbing.inject_vocal_tags(stub, segments, [VocalEvent("laugh", 1.0, 1.5, 0.9)])
        self.assertEqual(stats["events"], 0)
        self.assertEqual(stats["tagged_segments"], 0)

    def test_second_injection_is_idempotent(self) -> None:
        stub = _StubDubbing()
        segments = [_segment("SPEAKER_A", 0.0, 4.0, "привет как дела")]
        events = [VocalEvent("scream", 1.0, 1.5, 0.9)]
        SmartDubbing.inject_vocal_tags(stub, segments, events)
        tagged = segments[0]["translation"]
        stats = SmartDubbing.inject_vocal_tags(stub, segments, events)
        self.assertEqual(stats["tagged_segments"], 0)
        self.assertEqual(stats["already_tagged"], 1)
        self.assertEqual(segments[0]["translation"], tagged)


class VocalSignatureTests(unittest.TestCase):
    def test_signature_empty_without_tags(self) -> None:
        segments = [_segment("SPEAKER_A", 0.0, 4.0, "привет как дела")]
        self.assertEqual(vocal_event_signature(segments), "")

    def test_signature_changes_with_tag(self) -> None:
        segments = [_segment("SPEAKER_A", 0.0, 4.0, "привет как дела")]
        before = vocal_event_signature(segments)
        apply_vocal_tag(segments[0], 0.5, "scream")
        after = vocal_event_signature(segments)
        self.assertNotEqual(before, after)
        self.assertTrue(after)


if __name__ == "__main__":
    unittest.main()
