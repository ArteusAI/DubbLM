import json
import unittest
from unittest.mock import patch

from src.dubbing.core.cost_estimator import CostEstimator
from src.dubbing.debug.cost_tracker import CostTracker


class _Config:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, default=None):
        return self.values.get(key, default)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _FakeEstimateTracker:
    def __init__(self):
        self.translation_calls = []

    def estimate_translation_cost(self, *args, **kwargs):
        self.translation_calls.append((args, kwargs))


class CostTrackerPricingTests(unittest.TestCase):
    def test_assemblyai_best_transcription_includes_diarization_addon(self) -> None:
        tracker = CostTracker(_Config())

        cost = tracker.add_transcription_usage(
            "assemblyai",
            3600,
            model="best",
            category="primary_transcription",
            speaker_diarization=True,
        )

        self.assertAlmostEqual(cost, 0.23, places=6)
        rows = tracker.get_api_cost_rows()
        self.assertEqual(rows[0]["stage_key"], "transcription")
        self.assertAlmostEqual(rows[0]["rates"]["per_min"], 0.21 / 60.0, places=9)
        self.assertAlmostEqual(rows[0]["rates"]["speaker_diarization_per_min"], 0.02 / 60.0, places=9)

    def test_gemini_tts_derives_audio_output_tokens_from_duration(self) -> None:
        tracker = CostTracker(_Config())

        cost = tracker.add_tts_actual(
            "gemini",
            model="gemini-2.5-pro-preview-tts",
            input_tokens=1000,
            audio_seconds=60,
        )

        self.assertAlmostEqual(cost, 0.031, places=6)
        row = tracker.get_api_cost_rows()[0]
        self.assertAlmostEqual(row["usage"]["output_tokens"], 1500.0, places=6)
        self.assertEqual(row["pricing_source"], "builtin")

    def test_gemini_31_flash_tts_uses_builtin_preview_pricing(self) -> None:
        tracker = CostTracker(_Config())

        cost = tracker.add_tts_actual(
            "gemini",
            model="gemini-3.1-flash-tts-preview",
            input_tokens=1000,
            audio_seconds=60,
        )

        self.assertAlmostEqual(cost, 0.031, places=6)
        row = tracker.get_api_cost_rows()[0]
        self.assertEqual(row["pricing_source"], "builtin")
        self.assertEqual(row["rates"]["input_per_1m_tokens"], 1.0)
        self.assertEqual(row["rates"]["output_per_1m_tokens"], 20.0)
        self.assertAlmostEqual(row["usage"]["output_tokens"], 1500.0, places=6)

    def test_gemini_38_tts_falls_back_to_builtin_pricing_when_config_lacks_model(self) -> None:
        pricing = {
            "tts": {
                "gemini": {
                    "per_audio_min": None,
                    "models": {
                        "gemini-2.5-pro-preview-tts": {
                            "input_per_1m_tokens": 1.0,
                            "output_per_1m_tokens": 20.0,
                        }
                    },
                }
            }
        }
        tracker = CostTracker(_Config({"pricing": pricing}))

        cost = tracker.add_tts_actual(
            "gemini",
            model="gemini-3.8-flash-tts",
            input_tokens=1000,
            audio_seconds=60,
        )

        # input: 1000/1e6 * $1 = $0.001, output: 1500/1e6 * $18 = $0.027
        self.assertAlmostEqual(cost, 0.028, places=6)
        row = tracker.get_api_cost_rows()[0]
        self.assertEqual(row["pricing_source"], "builtin")
        self.assertEqual(row["rates"]["output_per_1m_tokens"], 18.0)

    def test_openai_tts_uses_audio_minute_fallback_when_output_tokens_absent(self) -> None:
        tracker = CostTracker(_Config())

        cost = tracker.add_tts_actual(
            "openai",
            model="gpt-4o-mini-tts",
            input_tokens=1000,
            audio_seconds=60,
        )

        self.assertAlmostEqual(cost, 0.0156, places=6)

    def test_minimax_tts_uses_character_pricing(self) -> None:
        tracker = CostTracker(_Config())

        cost = tracker.add_tts_actual(
            "minimax",
            model="speech-02-hd",
            input_characters=1000,
            audio_seconds=10,
        )

        self.assertAlmostEqual(cost, 0.10, places=6)
        self.assertEqual(tracker.get_api_cost_rows()[0]["usage"]["input_characters"], 1000.0)

    def test_config_pricing_overrides_builtin_rates(self) -> None:
        tracker = CostTracker(_Config({
            "pricing": {
                "translation": {
                    "gemini": {
                        "models": {
                            "gemini-2.5-pro": {
                                "input_per_1m_tokens": 2.0,
                                "output_per_1m_tokens": 3.0,
                            }
                        }
                    }
                }
            }
        }))

        cost = tracker.add_translation_actual(
            "gemini",
            "gemini-2.5-pro",
            input_tokens=1000,
            output_tokens=2000,
        )

        self.assertAlmostEqual(cost, 0.008, places=6)
        self.assertEqual(tracker.get_api_cost_rows()[0]["pricing_source"], "config")

    def test_cost_estimator_skips_refinement_when_persona_none(self) -> None:
        tracker = _FakeEstimateTracker()
        estimator = CostEstimator(
            _Config({
                "llm_provider": "gemini",
                "llm_model_name": "gemini-flash-latest",
                "refinement_persona": "none",
            }),
            cost_tracker=tracker,
        )

        estimator._estimate_translation(estimated_tokens=600)

        self.assertEqual(len(tracker.translation_calls), 1)
        self.assertEqual(tracker.translation_calls[0][0][1], "gemini-flash-latest")

    def test_openrouter_live_pricing_is_loaded_from_models_api(self) -> None:
        payload = {
            "data": [
                {
                    "id": "openai/gpt-5.4",
                    "canonical_slug": "openai/gpt-5.4-20260401",
                    "pricing": {
                        "prompt": "0.0000025",
                        "completion": "0.000015",
                    },
                }
            ]
        }
        tracker = CostTracker(_Config())

        with patch("src.dubbing.debug.cost_tracker.urllib.request.urlopen", return_value=_FakeResponse(payload)):
            cost = tracker.add_translation_actual(
                "openrouter",
                "openai/gpt-5.4",
                input_tokens=1000,
                output_tokens=2000,
            )

        self.assertAlmostEqual(cost, 0.0325, places=6)
        row = tracker.get_api_cost_rows()[0]
        self.assertEqual(row["pricing_source"], "openrouter-live")
        self.assertAlmostEqual(row["rates"]["input_per_1m_tokens"], 2.5, places=6)
        self.assertAlmostEqual(row["rates"]["output_per_1m_tokens"], 15.0, places=6)

    def test_import_api_cost_rows_rehydrates_totals_and_usage(self) -> None:
        source = CostTracker(_Config())
        source.add_transcription_usage(
            "assemblyai",
            60,
            model="best",
            category="primary_transcription",
            speaker_diarization=True,
        )
        source.add_translation_actual(
            "gemini",
            "gemini-2.5-pro",
            input_tokens=1000,
            output_tokens=1000,
            category="context_analysis",
        )
        source.add_tts_actual(
            "gemini",
            model="gemini-2.5-pro-preview-tts",
            input_tokens=1000,
            audio_seconds=60,
        )

        imported = CostTracker(_Config())
        self.assertEqual(imported.import_api_cost_rows(source.get_api_cost_rows()), 3)

        costs = imported.get_costs_by_step()
        self.assertGreater(costs["transcription"], 0)
        self.assertGreater(costs["context_analysis"], 0)
        self.assertGreater(costs["speech_synthesis"], 0)
        self.assertAlmostEqual(costs["total"], source.actual["total"], places=6)
        self.assertAlmostEqual(imported.actual["total"], source.actual["total"], places=6)
        self.assertEqual(
            imported.usage["transcription"]["details"]["assemblyai:best"]["categories"]["primary_transcription"],
            60.0,
        )

    def test_estimate_then_actual_tts_keeps_input_characters_detail(self) -> None:
        tracker = CostTracker(_Config())
        tracker.estimate_tts_cost(
            "openrouter",
            "google/gemini-3.8-flash-lite-tts",
            input_tokens=10,
            expected_audio_seconds=5.0,
        )
        # The estimate creates the detail entry; the actual call must not fail
        # because the entry was created without input_characters.
        tracker.add_tts_actual(
            "openrouter",
            model="google/gemini-3.8-flash-lite-tts",
            input_tokens=10,
            audio_seconds=5.0,
        )
        detail = tracker.usage["tts"]["details"]["openrouter:google/gemini-3.8-flash-lite-tts"]
        self.assertIn("input_characters", detail)
        self.assertGreater(detail["cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()
