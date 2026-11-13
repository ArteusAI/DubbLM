from __future__ import annotations

import unittest

from src.dubbing.core.smart_dubbing import SmartDubbing
from src.dubbing.video.video_processor import VideoProcessor


class _Event:
    def __init__(self, kind: str, start: float, end: float, score: float = 0.9) -> None:
        self.kind = kind
        self.start = start
        self.end = end
        self.score = score


class _AudioProcessor:
    def get_total_duration(self) -> float:
        return 60.0


class LaughterPreserveVolumeTests(unittest.TestCase):
    def _dubber(self) -> SmartDubbing:
        obj = SmartDubbing.__new__(SmartDubbing)
        obj.config = {
            "laughter_mode": "preserve",
            "vocal_events_mode": "preserve",
            "laughter_preserve_pad_seconds": 0.15,
            "laughter_preserve_merge_gap": 0.25,
            "laughter_preserve_min_duration": 0.25,
        }
        obj._vocal_events = [
            _Event("laugh", 10.0, 12.0),
            _Event("scream", 20.0, 21.0),
        ]
        obj._laughter_events = []
        obj.audio_processor = _AudioProcessor()
        return obj

    def test_laughter_is_not_a_preservation_segment_split(self) -> None:
        dubber = self._dubber()
        events = dubber._events_for_preservation()
        self.assertEqual([event.kind for event in events], ["scream"])

    def test_laughter_ranges_come_from_vocal_events(self) -> None:
        dubber = self._dubber()
        ranges = dubber.build_laughter_preserve_ranges()
        self.assertEqual(len(ranges), 1)
        self.assertAlmostEqual(ranges[0][0], 9.85, places=2)
        self.assertAlmostEqual(ranges[0][1], 12.15, places=2)

    def test_gain_expression_has_ramps_and_target(self) -> None:
        processor = VideoProcessor.__new__(VideoProcessor)
        expr = processor._build_laughter_gain_expr(
            0.0,
            0.70,
            [(10.0, 12.0), (20.0, 21.0)],
            0.25,
        )
        self.assertIn("0.7", expr)
        self.assertIn("10.0", expr)
        self.assertIn("12.0", expr)
        self.assertIn("20.0", expr)
        self.assertIn("21.0", expr)
        self.assertIn("max(", expr)
        self.assertIn("min(", expr)


if __name__ == "__main__":
    unittest.main()
