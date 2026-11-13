import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.dubbing.debug.cost_ledger import (
    filter_api_cost_rows,
    import_cost_snapshots,
    load_cost_snapshot_rows,
    write_cost_snapshot_from_tracker,
)
from src.dubbing.debug.cost_tracker import CostTracker


class _Config:
    def get(self, key, default=None):
        return default


class CostLedgerTests(unittest.TestCase):
    def test_snapshots_preserve_cross_stage_costs_without_dub_duplicates(self) -> None:
        with TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)
            first_stage = CostTracker(_Config())
            first_stage.add_transcription_usage(
                "assemblyai",
                60,
                model="best",
                category="primary_transcription",
                speaker_diarization=True,
            )
            first_stage.add_translation_actual(
                "gemini",
                "gemini-2.5-pro",
                input_tokens=1000,
                output_tokens=1000,
                category="context_analysis",
            )
            first_stage.add_translation_actual(
                "gemini",
                "gemini-2.5-pro",
                input_tokens=2000,
                output_tokens=3000,
                category="translation",
            )

            self.assertEqual(write_cost_snapshot_from_tracker(debug_dir, "transcription", first_stage), 1)
            self.assertEqual(write_cost_snapshot_from_tracker(debug_dir, "llm", first_stage), 2)

            final_stage = CostTracker(_Config())
            self.assertEqual(import_cost_snapshots(debug_dir, final_stage), 3)
            final_stage.add_tts_actual(
                "gemini",
                model="gemini-2.5-pro-preview-tts",
                input_tokens=1000,
                audio_seconds=60,
            )
            self.assertEqual(write_cost_snapshot_from_tracker(debug_dir, "dub", final_stage), 1)

            rows = load_cost_snapshot_rows(debug_dir)
            categories = [row["category"] for row in rows]
            self.assertEqual(
                categories,
                ["primary_transcription", "context_analysis", "translation", "tts_synthesis"],
            )

            costs = final_stage.get_costs_by_step()
            self.assertGreater(costs["transcription"], 0)
            self.assertGreater(costs["context_analysis"], 0)
            self.assertGreater(costs["translation"], 0)
            self.assertGreater(costs["speech_synthesis"], 0)
            self.assertAlmostEqual(
                costs["total"],
                costs["transcription"]
                + costs["context_analysis"]
                + costs["translation"]
                + costs["speech_synthesis"],
                places=6,
            )

            dub_payload = json.loads((debug_dir / "cost_snapshots" / "dub.json").read_text())
            self.assertEqual([row["category"] for row in dub_payload["api_costs"]], ["tts_synthesis"])

    def test_filter_api_cost_rows_uses_stage_fallback_for_known_groups(self) -> None:
        rows = [
            {"stage_key": "transcription", "category": "unknown", "cost_usd": 0.1},
            {"stage_key": "speech_synthesis", "category": "unknown", "cost_usd": 0.2},
            {"stage_key": "video_creation", "category": "unknown", "cost_usd": 0.3},
        ]

        self.assertEqual(len(filter_api_cost_rows(rows, "transcription")), 1)
        self.assertEqual(len(filter_api_cost_rows(rows, "dub")), 1)

    def test_refinement_resilience_counters_survive_split_stages(self) -> None:
        with TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)
            translation_stage = CostTracker(_Config())
            translation_stage.record_refinement_failure()
            translation_stage.record_refinement_failure()
            translation_stage.record_refinement_downgrade()

            write_cost_snapshot_from_tracker(debug_dir, "llm", translation_stage)

            dub_stage = CostTracker(_Config())
            import_cost_snapshots(debug_dir, dub_stage)

            self.assertEqual(dub_stage.refinement_failures, 2)
            self.assertEqual(dub_stage.refinement_downgrades, 1)

            payload = json.loads((debug_dir / "cost_snapshots" / "llm.json").read_text())
            self.assertEqual(payload["refinement_failures"], 2)
            self.assertEqual(payload["refinement_downgrades"], 1)


if __name__ == "__main__":
    unittest.main()
