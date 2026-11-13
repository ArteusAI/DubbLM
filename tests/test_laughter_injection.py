from __future__ import annotations

import unittest

from src.dubbing.audio.laughter_detector import LaughterEvent
from src.dubbing.core.smart_dubbing import SmartDubbing, insert_laugh_tag, laughter_signature


class _StubDubbing:
    def __init__(self, systems=None):
        self.systems = systems or {}

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


class InsertTagTests(unittest.TestCase):
    def test_tag_is_inserted_at_word_boundary(self) -> None:
        text, changed = insert_laugh_tag("раз два три четыре", 0.5, "[laughs]")
        self.assertTrue(changed)
        self.assertIn("[laughs]", text)

    def test_tag_at_start_and_end(self) -> None:
        start_text, _ = insert_laugh_tag("раз два", 0.0, "[laughs]")
        self.assertTrue(start_text.startswith("[laughs]"))
        end_text, _ = insert_laugh_tag("раз два", 1.0, "[laughs]")
        self.assertTrue(end_text.endswith("[laughs]"))

    def test_existing_tag_is_not_duplicated(self) -> None:
        text, changed = insert_laugh_tag("раз два [laughs] три", 0.5, "[laughs]")
        self.assertFalse(changed)
        self.assertEqual(text, "раз два [laughs] три")

    def test_empty_text_unchanged(self) -> None:
        text, changed = insert_laugh_tag("", 0.5, "[laughs]")
        self.assertFalse(changed)


class InjectLaughterTagsTests(unittest.TestCase):
    def test_tag_added_to_overlapping_segment_variants(self) -> None:
        stub = _StubDubbing()
        segments = [
            _segment("SPEAKER_A", 0.0, 4.0, "привет как дела"),
            _segment("SPEAKER_B", 10.0, 12.0, "второй говорит"),
        ]
        events = [LaughterEvent(1.0, 2.0, 0.9)]
        stats = SmartDubbing.inject_laughter_tags(stub, segments, events)
        self.assertEqual(stats["tagged_segments"], 1)
        self.assertIn("[laughs]", segments[0]["translation"])
        self.assertIn("[laughs]", segments[0]["long_translation"])
        self.assertNotIn("[laughs]", segments[1]["translation"])

    def test_gap_event_attaches_to_nearest_segment(self) -> None:
        stub = _StubDubbing()
        segments = [_segment("SPEAKER_A", 0.0, 4.0, "привет как дела")]
        events = [LaughterEvent(5.0, 5.6, 0.7)]
        stats = SmartDubbing.inject_laughter_tags(stub, segments, events)
        self.assertEqual(stats["tagged_segments"], 1)
        self.assertTrue(segments[0]["translation"].endswith("[laughs]"))

    def test_far_gap_event_is_skipped(self) -> None:
        stub = _StubDubbing()
        segments = [_segment("SPEAKER_A", 0.0, 4.0, "привет как дела")]
        events = [LaughterEvent(30.0, 31.0, 0.7)]
        stats = SmartDubbing.inject_laughter_tags(stub, segments, events)
        self.assertEqual(stats["tagged_segments"], 0)
        self.assertEqual(stats["skipped"], 1)

    def test_only_gemini38_speakers_are_tagged(self) -> None:
        stub = _StubDubbing({"SPEAKER_A": "openai"})
        segments = [_segment("SPEAKER_A", 0.0, 4.0, "привет как дела")]
        events = [LaughterEvent(1.0, 2.0, 0.9)]
        stats = SmartDubbing.inject_laughter_tags(stub, segments, events)
        self.assertEqual(stats["tagged_segments"], 0)
        self.assertNotIn("[laughs]", segments[0]["translation"])

    def test_one_tag_per_segment(self) -> None:
        stub = _StubDubbing()
        segments = [_segment("SPEAKER_A", 0.0, 6.0, "привет как дела")]
        events = [LaughterEvent(1.0, 1.5, 0.9), LaughterEvent(3.0, 3.5, 0.8)]
        stats = SmartDubbing.inject_laughter_tags(stub, segments, events)
        self.assertEqual(stats["tagged_segments"], 1)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(segments[0]["translation"].count("[laughs]"), 1)

    def test_second_injection_is_idempotent(self) -> None:
        stub = _StubDubbing()
        segments = [_segment("SPEAKER_A", 0.0, 4.0, "привет как дела")]
        events = [LaughterEvent(1.0, 2.0, 0.9)]
        SmartDubbing.inject_laughter_tags(stub, segments, events)
        tagged_text = segments[0]["translation"]
        stats = SmartDubbing.inject_laughter_tags(stub, segments, events)
        self.assertEqual(stats["tagged_segments"], 0)
        self.assertEqual(stats["already_tagged"], 1)
        self.assertEqual(segments[0]["translation"], tagged_text)


class LaughterSignatureTests(unittest.TestCase):
    def test_signature_empty_without_tags(self) -> None:
        stub = _StubDubbing()
        segments = [_segment("SPEAKER_A", 0.0, 4.0, "привет как дела")]
        self.assertEqual(laughter_signature(segments), "")

    def test_signature_changes_with_tag(self) -> None:
        stub = _StubDubbing()
        segments = [_segment("SPEAKER_A", 0.0, 4.0, "привет как дела")]
        before = laughter_signature(segments)
        SmartDubbing.inject_laughter_tags(stub, segments, [LaughterEvent(1.0, 2.0, 0.9)])
        after = laughter_signature(segments)
        self.assertNotEqual(before, after)
        self.assertTrue(after)


class SubtitleMarkupTests(unittest.TestCase):
    def test_tts_markup_is_stripped_from_subtitles(self) -> None:
        from src.dubbing.utils.subtitle_utils import strip_tts_markup

        self.assertEqual(strip_tts_markup("Привет [laughs] мир <laugh>!"), "Привет мир !")
        self.assertEqual(strip_tts_markup("Да |mhm| ну [short pause]"), "Да mhm ну")


if __name__ == "__main__":
    unittest.main()
