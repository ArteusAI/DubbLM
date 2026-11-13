"""Tests for external segment optimization config mapping."""

from src.api.services.segment_optimization_config import apply_segment_optimization_config


def test_apply_segment_optimization_config_minimal_diarization_merge() -> None:
    """Minimal diarization merge settings map to segments_optimization keys."""
    segments_opt = {"post_diarization_merge_gap": 0.3}
    config_data = {
        "postDiarizationMergeGap": 0,
        "repairSpeakerFragmentation": False,
    }

    result = apply_segment_optimization_config(segments_opt, config_data)

    assert result["post_diarization_merge_gap"] == 0
    assert result["repair_speaker_fragmentation"] == "false"


def test_apply_segment_optimization_config_enable_repair() -> None:
    """Explicit repairSpeakerFragmentation=true maps to string true."""
    result = apply_segment_optimization_config(
        {},
        {"repairSpeakerFragmentation": True},
    )

    assert result["repair_speaker_fragmentation"] == "true"


def test_apply_segment_optimization_config_ignores_missing_keys() -> None:
    """Unspecified optimization keys are left unchanged."""
    segments_opt = {"post_diarization_merge_gap": 0.3, "repair_speaker_fragmentation": "auto"}

    result = apply_segment_optimization_config(segments_opt, {})

    assert result == segments_opt
