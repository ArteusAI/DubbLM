from __future__ import annotations

import json
import types
import unittest

from src.translation.llm_translator import LLMTranslator
from src.tts.gemini38_tags import build_gemini38_prompt, style_for_emotion


class StyleForEmotionTests(unittest.TestCase):
    def test_known_labels(self) -> None:
        self.assertEqual(style_for_emotion("sad"), "sad")
        self.assertEqual(style_for_emotion("Fearful"), "fearful, trembling")
        self.assertEqual(style_for_emotion("panicked"), "panicked")
        self.assertEqual(style_for_emotion("neutral"), "")
        self.assertEqual(style_for_emotion(None), "")

    def test_unknown_label_passthrough(self) -> None:
        self.assertEqual(style_for_emotion("brooding"), "brooding")

    def test_prompt_uses_mapped_style(self) -> None:
        prompt = build_gemini38_prompt("Привет", emotion="fearful")
        self.assertIn("fearful, trembling", prompt.style)
        prompt = build_gemini38_prompt("Привет", emotion="neutral")
        self.assertNotIn("neutral", prompt.style)


class _FakeLLM:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def complete(self, prompt: str):  # noqa: ARG002
        return types.SimpleNamespace(text=self._response_text)


def _translator(response: str) -> LLMTranslator:
    translator = object.__new__(LLMTranslator)
    translator.enable_emotion_enrichment = True
    translator.enable_llm_editor = False
    translator.debug = False
    translator.cost_tracker = None
    translator.refinement_llm_provider = "fake"
    translator.refinement_model_name = "fake-model"
    translator.refinement_llm = _FakeLLM(response)
    return translator


class EmotionLabelingTests(unittest.TestCase):
    def test_labels_assigned(self) -> None:
        segments = [
            {"speaker": "A", "translation": "прощай"},
            {"speaker": "B", "translation": "привет!"},
        ]
        translator = _translator(json.dumps({"0": "sad", "1": "happy"}))
        out = LLMTranslator._label_segment_emotions(translator, segments, "ru")
        self.assertEqual(out[0]["emotion"], "sad")
        self.assertEqual(out[1]["emotion"], "happy")

    def test_invalid_label_is_ignored(self) -> None:
        segments = [{"translation": "текст"}]
        translator = _translator(json.dumps({"0": "banana"}))
        LLMTranslator._label_segment_emotions(translator, segments, "ru")
        self.assertNotIn("emotion", segments[0])

    def test_disabled_enrichment_skips_labeling(self) -> None:
        segments = [{"translation": "текст"}]
        translator = _translator(json.dumps({"0": "sad"}))
        translator.enable_emotion_enrichment = False
        LLMTranslator._label_segment_emotions(translator, segments, "ru")
        self.assertNotIn("emotion", segments[0])

    def test_editor_pass_skips_labeling(self) -> None:
        segments = [{"translation": "текст"}]
        translator = _translator(json.dumps({"0": "sad"}))
        translator.enable_llm_editor = True
        LLMTranslator._label_segment_emotions(translator, segments, "ru")
        self.assertNotIn("emotion", segments[0])

    def test_malformed_response_is_ignored(self) -> None:
        segments = [{"translation": "текст"}]
        translator = _translator("not json at all")
        LLMTranslator._label_segment_emotions(translator, segments, "ru")
        self.assertNotIn("emotion", segments[0])


if __name__ == "__main__":
    unittest.main()
