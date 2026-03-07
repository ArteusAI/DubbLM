from __future__ import annotations

import types

import pytest
from pydub import AudioSegment

from src.dubbing.core.smart_dubbing import SmartDubbing


def _make_segment(start: float, end: float, speaker: str, text: str) -> dict:
    return {
        "start": start,
        "end": end,
        "speaker": speaker,
        "text": text,
        "translation": text,
    }


def test_adjust_and_combine_sequential_splits_interleaved_same_speaker() -> None:
    smart = SmartDubbing.__new__(SmartDubbing)
    smart.config = {
        "segment_stretch": "audio_and_video",
        "segments_optimization": {
            "post_translation_merge_gap": 1.5,
            "max_segment_duration": 60,
        },
    }
    smart.debug_data = {}

    captured: dict = {}

    def _fake_process_sequential_mode(self, segments, all_groups, *_args, **_kwargs):
        captured["all_groups"] = all_groups
        return AudioSegment.empty(), [], []

    smart._process_sequential_mode = types.MethodType(_fake_process_sequential_mode, smart)

    segments = [
        _make_segment(0.00, 0.70, "SPEAKER_A", "A1"),
        _make_segment(0.80, 1.30, "SPEAKER_B", "B1"),
        _make_segment(1.40, 2.00, "SPEAKER_A", "A2"),
    ]

    smart._adjust_and_combine_audio_grouped(segments)

    all_groups = captured["all_groups"]
    grouped_indices = [[orig_idx for orig_idx, _ in g["group"]] for g in all_groups]

    assert grouped_indices == [[0], [1], [2]]


def test_process_sequential_mode_does_not_inject_gap_for_overlaps() -> None:
    smart = SmartDubbing.__new__(SmartDubbing)
    smart.config = {}
    smart.debug_data = {}

    seg1 = _make_segment(0.00, 1.00, "SPEAKER_A", "A1")
    seg2 = _make_segment(0.80, 1.80, "SPEAKER_B", "B1")

    all_groups = [
        {
            "speaker": "SPEAKER_A",
            "group_idx": 0,
            "group": [(0, seg1)],
            "group_audio": AudioSegment.silent(duration=1000),
            "group_segment_positions": [
                {
                    "segment": seg1,
                    "start_in_group_ms": 0,
                    "end_in_group_ms": 1000,
                    "original_index": 0,
                }
            ],
            "original_start": 0.00,
            "original_end": 1.00,
            "target_duration_ms": 1000,
            "actual_duration_ms": 1000,
        },
        {
            "speaker": "SPEAKER_B",
            "group_idx": 0,
            "group": [(1, seg2)],
            "group_audio": AudioSegment.silent(duration=1000),
            "group_segment_positions": [
                {
                    "segment": seg2,
                    "start_in_group_ms": 0,
                    "end_in_group_ms": 1000,
                    "original_index": 1,
                }
            ],
            "original_start": 0.80,
            "original_end": 1.80,
            "target_duration_ms": 1000,
            "actual_duration_ms": 1000,
        },
    ]

    combined_audio, real_positions, video_speed_segments = smart._process_sequential_mode(
        segments=[seg1, seg2],
        all_groups=all_groups,
        progress_callback=None,
        mode="audio_and_video",
        comfort_min=0.85,
        comfort_max=1.15,
        minterpolate_threshold=0.75,
    )

    assert len(combined_audio) == 2000
    assert len(video_speed_segments) == 2
    assert real_positions[1]["start"] == pytest.approx(1.0)
