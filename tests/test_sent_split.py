"""Tests for sentence splitting helpers."""

from __future__ import annotations

import unittest

from src.utils.sent_split import split_at_sentence_midpoint


class SplitAtSentenceMidpointTests(unittest.TestCase):
    def test_splits_near_middle_on_sentence_boundary(self) -> None:
        text = "First sentence here. Second sentence follows. Third sentence ends."
        parts = split_at_sentence_midpoint(text, min_part_chars=10)

        assert parts is not None
        first, second = parts
        self.assertTrue(first.endswith("."))
        self.assertTrue(second.endswith("."))
        self.assertEqual(f"{first} {second}", text)

    def test_returns_none_for_single_sentence(self) -> None:
        self.assertIsNone(split_at_sentence_midpoint("Only one sentence here.", min_part_chars=5))

    def test_returns_none_for_empty_text(self) -> None:
        self.assertIsNone(split_at_sentence_midpoint("   ", min_part_chars=5))


if __name__ == "__main__":
    unittest.main()
