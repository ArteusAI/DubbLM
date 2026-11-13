from __future__ import annotations

import unittest
from dataclasses import dataclass

from src.utils.batch_alignment import find_temporal_conflicts


@dataclass(frozen=True)
class FakeSegment:
    start_ms: int
    end_ms: int


def _seg(start_ms: int, end_ms: int) -> FakeSegment:
    return FakeSegment(
        start_ms=start_ms,
        end_ms=end_ms,
    )


class GeminiBatchAlignmentTests(unittest.TestCase):
    def test_temporal_conflicts_flag_overlapping_slices(self) -> None:
        conflicts = find_temporal_conflicts({
            0: _seg(0, 1200),
            1: _seg(950, 1800),
            2: _seg(1820, 2400),
        })

        self.assertIn(0, conflicts)
        self.assertIn(1, conflicts)
        self.assertNotIn(2, conflicts)

    def test_temporal_conflicts_ignore_small_boundary_jitter(self) -> None:
        conflicts = find_temporal_conflicts(
            {
                0: _seg(0, 1000),
                1: _seg(930, 1600),
            },
            overlap_tolerance_ms=120,
        )

        self.assertEqual(conflicts, {})


if __name__ == "__main__":
    unittest.main()
