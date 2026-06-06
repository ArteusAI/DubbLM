"""Map project config segment optimization fields to dubbing config."""

from typing import Any, Dict


def apply_segment_optimization_config(
    segments_opt: Dict[str, Any],
    config_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge project segment optimization settings into dubbing config."""
    if config_data.get("postDiarizationMergeGap") is not None:
        segments_opt["post_diarization_merge_gap"] = config_data["postDiarizationMergeGap"]
    if config_data.get("postTranslationMergeGap") is not None:
        segments_opt["post_translation_merge_gap"] = config_data["postTranslationMergeGap"]
    if config_data.get("repairSpeakerFragmentation") is not None:
        segments_opt["repair_speaker_fragmentation"] = (
            "true" if config_data["repairSpeakerFragmentation"] else "false"
        )
    if config_data.get("maxSegmentDuration") is not None:
        segments_opt["max_segment_duration"] = config_data["maxSegmentDuration"]
    if config_data.get("minSegmentDuration") is not None:
        segments_opt["min_segment_duration"] = config_data["minSegmentDuration"]
    if config_data.get("minPauseDuration") is not None:
        segments_opt["min_pause_duration"] = config_data["minPauseDuration"]
    if config_data.get("preservePauseDuration") is not None:
        segments_opt["preserve_pause_duration"] = config_data["preservePauseDuration"]
    if config_data.get("comfortMinAdjustmentRatio") is not None:
        segments_opt["comfort_min_adjustment_ratio"] = config_data["comfortMinAdjustmentRatio"]
    if config_data.get("comfortMaxAdjustmentRatio") is not None:
        segments_opt["comfort_max_adjustment_ratio"] = config_data["comfortMaxAdjustmentRatio"]
    return segments_opt
