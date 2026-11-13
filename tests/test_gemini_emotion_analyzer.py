"""Tests for the video+audio emotion analysis (Gemini VLM)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from src.dubbing.video.gemini_emotion_analyzer import (
    EmotionAnalysisConfig,
    _apply_annotations,
    _build_windows,
    build_transcript_payload,
    choose_silence_boundary,
    build_video_proxy,
    normalize_emotion_label,
    parse_emotion_response,
    sanitize_direction,
)
from src.tts.gemini38_tags import build_gemini38_prompt, sanitize_style_note
from src.tts.models import TTSSegmentData
from src.tts.openrouter_tts_wrapper import (
    OpenRouterTTSWrapper,
    _drama_note_direction,
)


def _drama_prepare(segment: TTSSegmentData) -> str:
    """Call the Drama 3 text preparation without building a full wrapper."""
    fake_wrapper = mock.Mock()
    fake_wrapper.voice_prompt_mapping = {}
    return OpenRouterTTSWrapper._prepare_drama3_speech_text(fake_wrapper, segment)


class NormalizeEmotionLabelTests(unittest.TestCase):
    def test_canonical_labels_pass_through(self) -> None:
        self.assertEqual(normalize_emotion_label("tense"), "tense")
        self.assertEqual(normalize_emotion_label("Triumphant"), "triumphant")
        self.assertEqual(normalize_emotion_label("  excited  "), "excited")

    def test_aliases_map_onto_the_canonical_vocabulary(self) -> None:
        self.assertEqual(normalize_emotion_label("rage"), "furious")
        self.assertEqual(normalize_emotion_label("nervous"), "tense")
        self.assertEqual(normalize_emotion_label("whisper"), "whispering")
        self.assertEqual(normalize_emotion_label("joy"), "joyful")

    def test_empty_label_is_neutral(self) -> None:
        self.assertEqual(normalize_emotion_label(""), "neutral")
        self.assertEqual(normalize_emotion_label(None), "neutral")

    def test_unknown_label_is_kept_lowercase(self) -> None:
        # An unknown word still reaches the TTS instruction instead of being lost.
        self.assertEqual(normalize_emotion_label("Brooding"), "brooding")


class SanitizeDirectionTests(unittest.TestCase):
    def test_markup_and_quotes_are_removed(self) -> None:
        self.assertEqual(
            sanitize_direction('[quiet, "controlled"] whisper'),
            "quiet, controlled whisper",
        )

    def test_length_is_capped(self) -> None:
        note = sanitize_direction("word " * 60, max_chars=40)
        self.assertLessEqual(len(note), 40)
        self.assertNotIn("  ", note)

    def test_empty_note(self) -> None:
        self.assertEqual(sanitize_direction(""), "")
        self.assertEqual(sanitize_direction("[]"), "")


class ParseEmotionResponseTests(unittest.TestCase):
    def test_parses_the_expected_shape(self) -> None:
        payload = json.dumps(
            {
                "lines": [
                    {"index": 0, "emotion": "tense", "intensity": 0.7, "direction": "restrained"},
                    {"index": 1, "emotion": "neutral", "intensity": 0.0, "direction": ""},
                ]
            }
        )
        parsed = parse_emotion_response(payload)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["emotion"], "tense")
        self.assertAlmostEqual(parsed[0]["intensity"], 0.7)
        self.assertEqual(parsed[1]["emotion"], "neutral")

    def test_parses_json_wrapped_in_markdown(self) -> None:
        payload = '```json\n{"lines": [{"index": 3, "emotion": "furious", "intensity": 0.9}]}\n```'
        parsed = parse_emotion_response(payload)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["index"], 3)
        self.assertEqual(parsed[0]["emotion"], "furious")

    def test_parses_a_flat_mapping(self) -> None:
        parsed = parse_emotion_response(json.dumps({"0": "sad", "1": "happy"}))
        self.assertEqual([item["emotion"] for item in parsed], ["sad", "happy"])

    def test_broken_payload_is_ignored(self) -> None:
        self.assertEqual(parse_emotion_response("not json at all"), [])
        self.assertEqual(parse_emotion_response(""), [])

    def test_intensity_is_clamped(self) -> None:
        parsed = parse_emotion_response(
            json.dumps({"lines": [{"index": 0, "emotion": "tense", "intensity": 4}]})
        )
        self.assertEqual(parsed[0]["intensity"], 1.0)


class ApplyAnnotationsTests(unittest.TestCase):
    def test_annotations_are_written_on_the_segments(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
        _apply_annotations(
            segments,
            [
                {"index": 0, "emotion": "tense", "intensity": 0.8, "direction": "low, controlled"},
                {"index": 1, "emotion": "joyful", "intensity": 0.9, "direction": "bright"},
            ],
            min_intensity=0.15,
            direction_max_chars=90,
        )
        self.assertEqual(segments[0]["emotion"], "tense")
        self.assertEqual(segments[0]["emotion_direction"], "low, controlled")
        self.assertEqual(segments[1]["emotion"], "joyful")

    def test_flat_delivery_drops_the_note_but_keeps_the_label(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        _apply_annotations(
            segments,
            [{"index": 0, "emotion": "sad", "intensity": 0.05, "direction": "quiet"}],
            min_intensity=0.15,
            direction_max_chars=90,
        )
        # The label drives the TTS style and must survive a low intensity.
        self.assertEqual(segments[0]["emotion"], "sad")
        self.assertEqual(segments[0]["emotion_intensity"], 0.05)
        self.assertEqual(segments[0]["emotion_direction"], "")

    def test_neutral_delivery_has_no_style_and_no_note(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        _apply_annotations(
            segments,
            [{"index": 0, "emotion": "neutral", "intensity": 0.0, "direction": "flat"}],
            min_intensity=0.15,
            direction_max_chars=90,
        )
        self.assertEqual(segments[0]["emotion"], "neutral")
        self.assertEqual(segments[0]["emotion_intensity"], 0.0)
        self.assertEqual(segments[0]["emotion_direction"], "")

    def test_missing_annotation_becomes_neutral(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        _apply_annotations(segments, [], min_intensity=0.15, direction_max_chars=90)
        self.assertEqual(segments[0]["emotion"], "neutral")


class TranscriptPayloadTests(unittest.TestCase):
    def test_empty_lines_are_skipped_but_indices_are_kept(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "A"},
            {"start": 1.0, "end": 2.0, "text": "", "speaker": "B"},
            {"start": 2.0, "end": 3.0, "translation": "привет", "speaker": "C"},
        ]
        payload = build_transcript_payload(segments)
        self.assertEqual([item["index"] for item in payload], [0, 2])
        self.assertEqual(payload[1]["text"], "привет")


class VideoProxyTests(unittest.TestCase):
    def test_proxy_command_keeps_the_audio_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "in.mp4")
            target = os.path.join(tmp, "proxy.mp4")
            with open(source, "wb") as handle:
                handle.write(b"0")

            captured = {}

            def fake_run(command, **kwargs):  # noqa: ARG001
                captured["command"] = command
                with open(target, "wb") as handle:
                    handle.write(b"0")
                return mock.Mock(returncode=0, stderr="")

            with mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"), mock.patch(
                "subprocess.run", side_effect=fake_run
            ):
                ok = build_video_proxy(source, target, fps=0.5, width=480)

            self.assertTrue(ok)
            command = captured["command"]
            joined = " ".join(command)
            self.assertIn("fps=0.5", joined)
            self.assertIn("scale=480:-2", joined)
            # The voice must survive: without audio there is nothing to judge.
            self.assertIn("aac", command)
            self.assertIn("-ac", command)

    def test_proxy_failure_is_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "in.mp4")
            with open(source, "wb") as handle:
                handle.write(b"0")
            with mock.patch("shutil.which", return_value=None):
                self.assertFalse(build_video_proxy(source, os.path.join(tmp, "proxy.mp4")))


class Gemini38StyleTests(unittest.TestCase):
    def test_the_delivery_note_reaches_the_style(self) -> None:
        prompt = build_gemini38_prompt(
            "Привет",
            emotion="tense",
            emotion_note="low, controlled, rising urgency",
        )
        self.assertIn("tense, restrained", prompt.style)
        self.assertIn("low, controlled, rising urgency", prompt.style)

    def test_the_note_is_never_spoken(self) -> None:
        prompt = build_gemini38_prompt("Привет", emotion_note="whispering")
        self.assertNotIn("whispering", prompt.text)
        self.assertIn("whispering", prompt.style)

    def test_note_sanitizer_strips_markup(self) -> None:
        self.assertEqual(sanitize_style_note("[loud] <shout>"), "loud shout")
        self.assertEqual(sanitize_style_note(""), "")


class Drama3DirectionTests(unittest.TestCase):
    def _segment(self, **kwargs) -> TTSSegmentData:
        data = {
            "speaker": "A",
            "text": "I'm fine. Really.",
            "emotion": "tense",
            "emotion_note": "restrained, almost a whisper",
        }
        data.update(kwargs)
        return TTSSegmentData(**data)

    def test_direction_contains_emotion_and_note(self) -> None:
        prepared = _drama_prepare(self._segment())
        self.assertTrue(prepared.startswith("["))
        self.assertIn("tense", prepared)
        self.assertIn("restrained, almost a whisper", prepared)
        self.assertTrue(prepared.endswith("I'm fine. Really."))

    def test_the_note_never_breaks_the_direction(self) -> None:
        note = _drama_note_direction("loud [broken] {note}")
        self.assertNotIn("[", note)
        self.assertNotIn("]", note)
        self.assertIn("loud", note)

    def test_no_note_keeps_the_previous_direction(self) -> None:
        prepared = _drama_prepare(self._segment(emotion_note=None))
        self.assertIn("[tense]", prepared)


class SegmentDataTests(unittest.TestCase):
    def test_emotion_note_is_part_of_the_segment_data(self) -> None:
        segment = TTSSegmentData(speaker="A", text="hi", emotion_note="whispering")
        self.assertEqual(segment.emotion_note, "whispering")

    def test_emotion_note_defaults_to_none(self) -> None:
        segment = TTSSegmentData(speaker="A", text="hi")
        self.assertIsNone(segment.emotion_note)


class WindowTests(unittest.TestCase):
    def test_short_material_stays_in_one_window(self) -> None:
        segments = [
            {"start": 0.0, "end": 5.0},
            {"start": 5.0, "end": 12.0},
        ]
        windows = _build_windows(segments, 600.0)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][2], [0, 1])

    def test_long_material_is_split(self) -> None:
        segments = [
            {"start": 0.0, "end": 100.0},
            {"start": 100.0, "end": 200.0},
            {"start": 200.0, "end": 300.0},
            {"start": 300.0, "end": 400.0},
        ]
        windows = _build_windows(segments, 250.0)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0][2], [0, 1])
        self.assertEqual(windows[1][2], [2, 3])
        self.assertAlmostEqual(windows[0][0], 0.0)
        self.assertAlmostEqual(windows[1][0], 200.0)
        self.assertAlmostEqual(windows[1][1], 400.0)

    def test_a_segment_longer_than_the_window_stands_alone(self) -> None:
        segments = [
            {"start": 0.0, "end": 100.0},
            {"start": 100.0, "end": 900.0},
            {"start": 900.0, "end": 950.0},
        ]
        windows = _build_windows(segments, 300.0)
        self.assertEqual([window[2] for window in windows], [[0], [1], [2]])

    def test_window_start_snaps_to_preceding_silence(self) -> None:
        boundary = choose_silence_boundary(
            10.0,
            "start",
            [(8.4, 8.8), (9.6, 9.9)],
            pad_seconds=0.5,
        )
        self.assertAlmostEqual(boundary, 9.75, places=2)

    def test_window_end_snaps_to_following_silence(self) -> None:
        boundary = choose_silence_boundary(
            20.0,
            "end",
            [(20.2, 20.6), (21.0, 21.3)],
            pad_seconds=0.5,
        )
        self.assertAlmostEqual(boundary, 20.4, places=2)

    def test_window_boundary_falls_back_to_padding(self) -> None:
        self.assertAlmostEqual(
            choose_silence_boundary(30.0, "start", [], pad_seconds=0.5),
            29.5,
            places=2,
        )
        self.assertAlmostEqual(
            choose_silence_boundary(30.0, "end", [], pad_seconds=0.5),
            30.5,
            places=2,
        )


class ConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        config = EmotionAnalysisConfig()
        self.assertEqual(config.model, "gemini-flash-latest")
        self.assertEqual(config.fps, 0.5)
        self.assertEqual(config.max_window_seconds, 600.0)


if __name__ == "__main__":
    unittest.main()
