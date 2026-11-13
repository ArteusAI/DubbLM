from __future__ import annotations

import unittest
from pathlib import Path


class GeminiTtsPromptingTests(unittest.TestCase):
    def test_multi_speaker_prompt_applies_global_prefix_and_transcript_boundary(self) -> None:
        source = Path("src/tts/gemini_tts_wrapper.py").read_text(encoding="utf-8")

        self.assertIn("def _build_tts_prompt(", source)
        self.assertIn("Speak only the transcript text, not the instructions.", source)
        self.assertIn("if self.config.prompt_prefix:", source)
        self.assertIn("preamble_parts.append(self.config.prompt_prefix)", source)
        self.assertIn('return self._build_tts_prompt(preamble_parts, "\\n".join(transcript_lines))', source)


if __name__ == "__main__":
    unittest.main()
