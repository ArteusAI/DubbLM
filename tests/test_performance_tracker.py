from __future__ import annotations

import unittest

from src.dubbing.debug.performance_tracker import PerformanceTracker


class PerformanceTrackerTests(unittest.TestCase):
    def test_write_performance_summary_handles_zero_total_elapsed(self) -> None:
        tracker = PerformanceTracker()

        # No total metric recorded yet, but a valid duration is present.
        tracker.write_performance_summary(total_duration=42.0)

    def test_write_performance_summary_for_report_handles_zero_total_elapsed(self) -> None:
        tracker = PerformanceTracker()
        tracker.record_metric("video_duration", 42.0)

        tracker.write_performance_summary_for_report()


if __name__ == "__main__":
    unittest.main()
