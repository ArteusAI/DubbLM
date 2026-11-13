from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.dubbing.debug.cost_tracker import CostTracker
from src.translation import llm_translator as llm_translator_module
from src.translation.llm_translator import LLMTranslator


class _Config:
    def get(self, key, default=None):
        return default


def _make_translator() -> tuple[LLMTranslator, CostTracker]:
    translator = LLMTranslator(
        llm_provider="gemini",
        model_name="gemini-flash-latest",
        refinement_llm_provider="openrouter",
        refinement_model_name="openai/gpt-6-sol",
        cost_tracker=CostTracker(_Config()),
    )
    translator.refinement_llm = SimpleNamespace(complete=lambda prompt: "openrouter")

    def fake_create_llm(provider, model_name, temperature, max_tokens=None, purpose="default"):
        return SimpleNamespace(complete=lambda prompt: "fallback")

    translator._create_llm = fake_create_llm  # type: ignore[method-assign]
    return translator, translator.cost_tracker


class RefinementProviderFallbackTests(unittest.TestCase):
    def test_quota_error_switches_refinement_to_primary_provider(self) -> None:
        translator, tracker = _make_translator()
        attempts = []

        def fake_robust(fn, *args, **kwargs):
            attempts.append(fn)
            if len(attempts) == 1:
                raise RuntimeError(
                    "Error code: 403 - {'error': {'message': 'Key limit exceeded (total limit)'}}"
                )
            return "refined"

        events = []
        with patch.object(llm_translator_module, "robust_llm_call", fake_robust):
            result = translator._call_refinement_llm(
                "prompt", progress_callback=lambda *args: events.append(args)
            )

        self.assertEqual(result, "refined")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(translator.refinement_llm_provider, "gemini")
        self.assertEqual(translator.refinement_model_name, "gemini-flash-latest")
        self.assertEqual(tracker.refinement_downgrades, 1)
        self.assertEqual(events, [("refinement_downgraded", 0, 0, "gemini/gemini-flash-latest")])

    def test_non_quota_error_propagates_without_downgrade(self) -> None:
        translator, tracker = _make_translator()

        with patch.object(
            llm_translator_module,
            "robust_llm_call",
            side_effect=ValueError("Invalid pairs found in refinement"),
        ):
            with self.assertRaises(ValueError):
                translator._call_refinement_llm("prompt")

        self.assertEqual(translator.refinement_llm_provider, "openrouter")
        self.assertEqual(tracker.refinement_downgrades, 0)

    def test_downgrade_happens_once(self) -> None:
        translator, tracker = _make_translator()

        with patch.object(
            llm_translator_module,
            "robust_llm_call",
            side_effect=RuntimeError("Error code: 402 - insufficient credits"),
        ):
            with self.assertRaises(RuntimeError):
                translator._call_refinement_llm("prompt")

        self.assertEqual(translator.refinement_llm_provider, "gemini")
        self.assertEqual(tracker.refinement_downgrades, 1)

        with patch.object(
            llm_translator_module,
            "robust_llm_call",
            side_effect=RuntimeError("Error code: 403 - key limit exceeded"),
        ):
            with self.assertRaises(RuntimeError):
                translator._call_refinement_llm("prompt")

        self.assertEqual(tracker.refinement_downgrades, 1)

    def test_quota_error_markers(self) -> None:
        translator, _ = _make_translator()

        self.assertTrue(translator._is_refinement_quota_error(RuntimeError("Error code: 403")))
        self.assertTrue(translator._is_refinement_quota_error(RuntimeError("Key limit exceeded")))
        self.assertFalse(translator._is_refinement_quota_error(RuntimeError("Empty refinement response")))

    def test_failure_counter_increments(self) -> None:
        _, tracker = _make_translator()
        tracker.record_refinement_failure()
        tracker.record_refinement_failure()
        self.assertEqual(tracker.refinement_failures, 2)


if __name__ == "__main__":
    unittest.main()
