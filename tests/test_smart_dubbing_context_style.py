from __future__ import annotations

import unittest

from src.dubbing.core.segment_context import (
    build_context_style,
    context_cache_token,
    normalize_quote_text,
)


class SegmentContextStyleTests(unittest.TestCase):
    def test_empty_previous_text_has_no_context(self) -> None:
        self.assertIsNone(build_context_style("   ", "SPEAKER_A", "SPEAKER_B", 140))

    def test_same_speaker_continues_previous_line(self) -> None:
        hint = build_context_style("The server room went dark.", "SPEAKER_A", "SPEAKER_A", 140)
        self.assertIsNotNone(hint)
        self.assertTrue(hint.startswith('Continuing the previous line: "'))
        self.assertIn("server room went dark", hint)

    def test_other_speaker_gets_reply_frame(self) -> None:
        hint = build_context_style("Did you check the logs?", "SPEAKER_B", "SPEAKER_A", 140)
        self.assertIsNotNone(hint)
        self.assertIn("Replying to SPEAKER_B:", hint)
        self.assertIn("Did you check the logs?", hint)

    def test_markup_is_stripped_from_quote(self) -> None:
        hint = build_context_style("[sigh] I checked the logs twice.", "SPEAKER_A", "SPEAKER_A", 140)
        self.assertIsNotNone(hint)
        self.assertNotIn("[sigh]", hint)
        self.assertIn("I checked the logs twice.", hint)

    def test_long_previous_line_is_truncated_on_word_boundary(self) -> None:
        long_line = "one two three four five six seven eight nine ten eleven twelve"
        hint = build_context_style(long_line, "SPEAKER_A", "SPEAKER_A", 40)
        self.assertIsNotNone(hint)
        quote = hint.split('"')[1]
        self.assertLessEqual(len(quote), 42)
        self.assertTrue(quote.endswith("…"))

    def test_disabled_by_zero_budget(self) -> None:
        self.assertIsNone(build_context_style("previous", "SPEAKER_A", "SPEAKER_A", 0))

    def test_normalize_quote_text_handles_newlines(self) -> None:
        self.assertEqual(normalize_quote_text("line one\nline two", 140), "line one line two")

    def test_cache_token_depends_on_context(self) -> None:
        without = context_cache_token(None)
        with_ctx = context_cache_token('Replying to B: "prev"')
        self.assertNotEqual(without, with_ctx)
        self.assertEqual(without, context_cache_token(""))
        self.assertEqual(with_ctx, context_cache_token('Replying to B: "prev"'))


if __name__ == "__main__":
    unittest.main()
