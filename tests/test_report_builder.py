import unittest

from src.dubbing.debug.report_builder import ReportBuilder


class _DummyPerformanceTracker:
    metrics = {}
    costs = {}


class _DummyCostTracker:
    usage = {}


class ReportBuilderMarkdownTests(unittest.TestCase):
    def test_tts_batches_result_cell_uses_plain_text_reason(self) -> None:
        builder = ReportBuilder(
            performance_tracker=_DummyPerformanceTracker(),
            cost_tracker=_DummyCostTracker(),
            tts_systems={},
            segments_metadata=[],
            config={},
            project_id="proj_test",
        )

        markdown = builder._render_markdown(
            {
                "generated_at": "2026-04-24T00:00:00+00:00",
                "project": {},
                "totals": {},
                "api_costs": [],
                "stages": [],
                "transcription_usage": [],
                "artifacts": [],
                "segments": [],
                "batches": [
                    {
                        "batch_index": 0,
                        "mode": "single_segment",
                        "segments_label": "#1",
                        "segment_count": 1,
                        "speakers": ["SPEAKER_A"],
                        "attempts": 1,
                        "duration_seconds": 1.2,
                        "success": True,
                        "fallback_segment_count": 0,
                        "reason": "single-segment synthesis",
                    }
                ],
            }
        )

        self.assertIn("single-segment synthesis", markdown)
        self.assertNotIn("<br/>", markdown)
        self.assertIn("| 1 | #1 (1) | solo | SPEAKER_A | 1 | 1.20s | single-segment synthesis |", markdown)

    def test_transcription_usage_is_split_by_pipeline_stage(self) -> None:
        builder = ReportBuilder(
            performance_tracker=_DummyPerformanceTracker(),
            cost_tracker=_DummyCostTracker(),
            tts_systems={},
            segments_metadata=[],
            config={},
            project_id="proj_test",
        )

        markdown = builder._render_markdown(
            {
                "generated_at": "2026-04-24T00:00:00+00:00",
                "project": {},
                "totals": {"cost_usd": 0.12},
                "api_costs": [],
                "stages": [],
                "artifacts": [],
                "segments": [],
                "batches": [],
                "transcription_usage": [
                    {
                        "provider": "assemblyai",
                        "model": "large-v3",
                        "audio_hours": 0.50,
                        "cost_usd": 0.12,
                        "categories": {
                            "primary_transcription": {"audio_hours": 0.30},
                            "tts_batch_alignment": {"audio_hours": 0.10},
                            "tts_content_validation": {"audio_hours": 0.10},
                        },
                    }
                ],
            }
        )

        self.assertIn("**By stage:** primary 0.30 h, batch alignment 0.10 h, tts validation 0.10 h", markdown)
        self.assertIn("| Provider | Model | Primary | Batch Alignment | TTS Validation | Total | Cost |", markdown)
        self.assertIn("| assemblyai | large-v3 | 0.30 h | 0.10 h | 0.10 h | 0.50 h | $0.1200 |", markdown)
        self.assertNotIn("Breakdown", markdown)

    def test_api_costs_table_renders_total_and_rates(self) -> None:
        builder = ReportBuilder(
            performance_tracker=_DummyPerformanceTracker(),
            cost_tracker=_DummyCostTracker(),
            tts_systems={},
            segments_metadata=[],
            config={},
            project_id="proj_test",
        )

        markdown = builder._render_markdown(
            {
                "generated_at": "2026-04-24T00:00:00+00:00",
                "project": {},
                "totals": {"cost_usd": 0.23},
                "api_costs": [
                    {
                        "stage_label": "Transcription",
                        "category_label": "Primary transcription",
                        "provider": "assemblyai",
                        "model": "best",
                        "usage": {"audio_seconds": 3600.0},
                        "rates": {
                            "per_min": 0.21 / 60.0,
                            "speaker_diarization_per_min": 0.02 / 60.0,
                            "pricing_source": "builtin",
                        },
                        "pricing_source": "builtin",
                        "cost_usd": 0.23,
                    }
                ],
                "stages": [],
                "artifacts": [],
                "segments": [],
                "batches": [],
                "transcription_usage": [],
            }
        )

        self.assertIn("## API Costs", markdown)
        self.assertIn("**Total cost:** $0.2300", markdown)
        self.assertIn("| Stage | Category | Provider | Model | Usage | Rate | Cost |", markdown)
        self.assertIn("| Transcription | Primary transcription | assemblyai | best | 1.00 h audio | $0.0035/min, +$0.0003333/min diarization (builtin) | $0.2300 |", markdown)
        self.assertIn("| **Total** |  |  |  |  |  | **$0.2300** |", markdown)

    def test_refinement_fallbacks_render_warning(self) -> None:
        builder = ReportBuilder(
            performance_tracker=_DummyPerformanceTracker(),
            cost_tracker=_DummyCostTracker(),
            tts_systems={},
            segments_metadata=[],
            config={},
            project_id="proj_test",
        )

        markdown = builder._render_markdown(
            {
                "generated_at": "2026-04-24T00:00:00+00:00",
                "project": {},
                "totals": {"cost_usd": 0.12, "refinement_failures": 2, "refinement_downgrades": 1},
                "api_costs": [],
                "stages": [],
                "artifacts": [],
                "segments": [],
                "batches": [],
                "transcription_usage": [],
            }
        )

        self.assertIn("refinement fell back to unrefined translations for 2 batch(es)", markdown)
        self.assertIn("provider downgrades: 1", markdown)

    def test_build_payload_collects_refinement_counters(self) -> None:
        cost_tracker = _DummyCostTracker()
        cost_tracker.refinement_failures = 3
        cost_tracker.refinement_downgrades = 1
        builder = ReportBuilder(
            performance_tracker=_DummyPerformanceTracker(),
            cost_tracker=cost_tracker,
            tts_systems={},
            segments_metadata=[],
            config={},
            project_id="proj_test",
        )

        payload = builder._build_payload()

        self.assertEqual(payload["totals"]["refinement_failures"], 3)
        self.assertEqual(payload["totals"]["refinement_downgrades"], 1)


if __name__ == "__main__":
    unittest.main()
