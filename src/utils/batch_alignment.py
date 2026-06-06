"""Helpers for safer multi-speaker batch text alignment."""
from __future__ import annotations

from typing import Any, Dict


def find_temporal_conflicts(
    best_per_line: Dict[int, Any],
    overlap_tolerance_ms: int = 120,
) -> Dict[int, str]:
    """Flag matched ASR slices that would duplicate or scramble audio."""
    conflicts: Dict[int, str] = {}
    ordered = sorted(best_per_line.items(), key=lambda item: item[0])
    if len(ordered) < 2:
        return conflicts

    prev_local_i, prev_seg = ordered[0]
    prev_end = int(getattr(prev_seg, "end_ms", 0))
    if prev_end <= int(getattr(prev_seg, "start_ms", 0)):
        conflicts[prev_local_i] = "ASR slice has non-positive duration"

    for local_i, seg in ordered[1:]:
        start_ms = int(getattr(seg, "start_ms", 0))
        end_ms = int(getattr(seg, "end_ms", 0))
        if end_ms <= start_ms:
            conflicts.setdefault(local_i, "ASR slice has non-positive duration")

        overlap_ms = prev_end - start_ms
        if overlap_ms > overlap_tolerance_ms:
            conflicts.setdefault(
                prev_local_i,
                f"ASR slice overlaps following line by {overlap_ms}ms",
            )
            conflicts.setdefault(
                local_i,
                f"ASR slice overlaps previous line by {overlap_ms}ms",
            )

        prev_local_i = local_i
        prev_end = end_ms

    return conflicts
