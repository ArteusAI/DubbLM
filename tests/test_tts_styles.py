from __future__ import annotations

import unittest

from src.dubbing.tts_styles import NEWS_PROMPT, resolve_tts_prompt_prefix


class TtsStyleTests(unittest.TestCase):
    def test_news_style_resolves_to_news_prompt(self) -> None:
        prompt = resolve_tts_prompt_prefix(style="news")

        self.assertEqual(prompt, NEWS_PROMPT)
        self.assertIn("modern news explainer", prompt)
        self.assertIn("energetic narrative cadence", prompt)

    def test_auto_can_resolve_to_news_style(self) -> None:
        prompt = resolve_tts_prompt_prefix(style="auto", resolved_style="news")

        self.assertEqual(prompt, NEWS_PROMPT)

    def test_unknown_style_falls_back_to_podcast(self) -> None:
        prompt = resolve_tts_prompt_prefix(style="unknown")

        self.assertIn("natural conversational energy", prompt)


if __name__ == "__main__":
    unittest.main()
