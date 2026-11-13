from __future__ import annotations

import json
import unittest
from types import MethodType
from unittest import mock

from src.translation.llm_translator import LLMTranslator


class _StubResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _StubLlm:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, prompt: str) -> _StubResponse:
        return _StubResponse(json.dumps(self.payload))


def _make_translator(payload: dict) -> LLMTranslator:
    translator = LLMTranslator(enable_cache=False)
    translator.llm = _StubLlm(payload)
    translator.refinement_llm = object()
    translator.is_available = MethodType(lambda self: True, translator)
    return translator


class ContextTtsStyleTests(unittest.TestCase):
    def test_context_analysis_preserves_news_tts_style(self) -> None:
        translator = _make_translator(
            {
                "domain": "news",
                "terminology": ["market", "election"],
                "tone": "urgent but factual",
                "themes": ["current affairs"],
                "chapters": [],
                "overall_summary": "A news update.",
                "tts_style": "news",
            }
        )

        with mock.patch("src.translation.llm_translator.json_repair", mock.Mock(loads=json.loads)):
            context = translator._analyze_context(
                [{"start": 0.0, "speaker": "SPEAKER_00", "text": "Tonight's top story."}],
                "en",
                "ru",
            )

        self.assertEqual(context["tts_style"], "news")

    def test_context_analysis_rejects_unknown_tts_style(self) -> None:
        translator = _make_translator(
            {
                "domain": "general",
                "terminology": [],
                "tone": "neutral",
                "themes": [],
                "chapters": [],
                "overall_summary": "A general update.",
                "tts_style": "cinematic",
            }
        )

        with mock.patch("src.translation.llm_translator.json_repair", mock.Mock(loads=json.loads)):
            context = translator._analyze_context(
                [{"start": 0.0, "speaker": "SPEAKER_00", "text": "Hello."}],
                "en",
                "ru",
            )

        self.assertEqual(context["tts_style"], "podcast")


if __name__ == "__main__":
    unittest.main()
