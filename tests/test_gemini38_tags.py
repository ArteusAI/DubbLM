from __future__ import annotations

import unittest

from src.tts.gemini38_tags import (
    build_gemini38_prompt,
    build_style,
    convert_legacy_tags,
    extract_quoted_fragments,
    quoted_fragment_leak,
    sanitize_angle_tags,
    sanitize_style,
    strip_tts_markup,
)


class Gemini38TagConversionTests(unittest.TestCase):
    def test_inline_tags_are_converted_to_angle_brackets(self) -> None:
        text, hints = convert_legacy_tags("[sigh] Wait... [short pause] did you hear that? [laughs]")
        self.assertEqual(text, "<sigh> Wait... <short pause> did you hear that? <laugh>")
        self.assertEqual(hints, [])

    def test_style_tags_move_out_of_text(self) -> None:
        text, hints = convert_legacy_tags("[shouting] Fire! [whispering] hide here")
        self.assertEqual(text, "Fire! hide here")
        self.assertEqual(hints, ["shouting", "whispering"])

    def test_medium_pause_maps_to_short_pause(self) -> None:
        text, _ = convert_legacy_tags("one [medium pause] two [long pause] three")
        self.assertEqual(text, "one <short pause> two <long pause> three")

    def test_fillers_become_plain_words(self) -> None:
        text, _ = convert_legacy_tags("Well [uhm] I think [hm] so")
        self.assertEqual(text, "Well uhm I think hm so")

    def test_unknown_tags_are_dropped(self) -> None:
        text, hints = convert_legacy_tags("Hello [totally-unknown] world")
        self.assertEqual(text, "Hello world")
        self.assertEqual(hints, [])

    def test_pipes_are_unwrapped_for_single_speaker_mode(self) -> None:
        text, _ = convert_legacy_tags("So the launch is Thursday |oh hmm| are we ready?")
        self.assertEqual(text, "So the launch is Thursday oh hmm are we ready?")

    def test_strip_tts_markup_removes_all_dialects(self) -> None:
        cleaned = strip_tts_markup("Hello <sigh> world [laughs] |mhm| end")
        self.assertEqual(cleaned, "Hello world mhm end")


class Gemini38AngleTagSanitizationTests(unittest.TestCase):
    def test_medium_pause_angle_is_normalized_to_short(self) -> None:
        self.assertEqual(
            sanitize_angle_tags("one <medium pause> two"),
            "one <short pause> two",
        )

    def test_unknown_angle_tags_are_dropped(self) -> None:
        self.assertEqual(
            sanitize_angle_tags("hello <robot noise> world"),
            "hello world",
        )

    def test_known_angle_tags_are_normalized_and_preserved(self) -> None:
        self.assertEqual(
            sanitize_angle_tags("<sighs> wait <long pause> now"),
            "<sigh> wait <long pause> now",
        )

    def test_build_prompt_sanitizes_enricher_output(self) -> None:
        prompt = build_gemini38_prompt("Hello <medium pause> there <made-up>")
        self.assertEqual(prompt.text, "Hello <short pause> there")


class Gemini38StyleTests(unittest.TestCase):
    def test_legacy_preset_is_compacted(self) -> None:
        compact = sanitize_style("Speak with natural conversational energy, clear articulation:")
        self.assertEqual(compact, "natural conversational energy, clear articulation")

    def test_meta_instructions_are_removed_from_custom_prompts(self) -> None:
        style = sanitize_style(
            "Synthesize only the transcript below in the language it is written; do not read these directions. "
            "Confident broadcast narration:"
        )
        self.assertEqual(style, "Confident broadcast narration")

    def test_build_style_deduplicates_and_caps(self) -> None:
        style = build_style("whispering", "whispering", "calm", max_chars=20)
        self.assertEqual(style, "whispering, calm")

    def test_build_prompt_combines_transcript_and_style(self) -> None:
        prompt = build_gemini38_prompt(
            "[crying] I did NOT expect that",
            prompt_prefix="Speak calmly and clearly, with measured pacing and an authoritative tone:",
            speaker_prompt="warm and reassuring",
            emotion="Sad",
        )
        self.assertEqual(prompt.text, "<sob> I did NOT expect that")
        self.assertIn("calm, measured pacing, authoritative tone", prompt.style)
        self.assertIn("warm and reassuring", prompt.style)
        self.assertIn("sad", prompt.style)

    def test_neutral_emotion_is_not_added_to_style(self) -> None:
        prompt = build_gemini38_prompt("hello", emotion="Neutral")
        self.assertEqual(prompt.style, "")

    def test_context_quote_is_added_last_and_never_in_text(self) -> None:
        context = 'Replying to SPEAKER_B: "The server room went dark." — match the conversational energy'
        prompt = build_gemini38_prompt(
            "Then we heard a fan spin up.",
            speaker_prompt="calm and focused",
            context_style=context,
        )
        self.assertEqual(prompt.text, "Then we heard a fan spin up.")
        self.assertNotIn("server room", prompt.text)
        self.assertTrue(prompt.style.endswith(context))
        self.assertIn("calm and focused", prompt.style)

    def test_context_quote_is_dropped_first_when_style_is_capped(self) -> None:
        context = 'Replying to SPEAKER_B: "A very long previous line that should be dropped first."'
        prompt = build_gemini38_prompt(
            "short reply",
            speaker_prompt="warm and reassuring",
            context_style=context,
            max_style_chars=len("warm and reassuring"),
        )
        self.assertEqual(prompt.style, "warm and reassuring")

    def test_extract_quoted_fragments(self) -> None:
        self.assertEqual(
            extract_quoted_fragments('Replying to A: "hello there" — go on'),
            ["hello there"],
        )
        self.assertEqual(extract_quoted_fragments("no quotes here"), [])

    def test_quoted_fragment_leak_detects_previous_line_spoken(self) -> None:
        context = 'Replying to SPEAKER_B: "the server room went dark at midnight" — match energy'
        asr = "The server room went dark at midnight. Then we heard a fan spin up."
        leak = quoted_fragment_leak(asr, "Then we heard a fan spin up.", context)
        self.assertIsNotNone(leak)

    def test_quoted_fragment_leak_ignores_expected_overlap(self) -> None:
        context = 'Replying to SPEAKER_B: "the server room went dark" — match energy'
        asr = "Then we heard a fan spin up behind the rack."
        self.assertIsNone(quoted_fragment_leak(asr, asr, context))

    def test_quoted_fragment_leak_ignores_short_quotes(self) -> None:
        context = 'Replying to A: "ok" — go on'
        self.assertIsNone(quoted_fragment_leak("ok then", "ok then", context))


if __name__ == "__main__":
    unittest.main()
