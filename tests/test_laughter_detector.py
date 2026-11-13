from __future__ import annotations

import unittest

import numpy as np

from src.dubbing.audio.laughter_detector import (
    LaughterDetector,
    LaughterEvent,
    concat_close,
    frames_to_events,
    remove_short,
)

SR = 16000
FRAME_COUNT = 349
FRAME_SECONDS = 7.0 / FRAME_COUNT


class FrameEventTests(unittest.TestCase):
    def test_single_run_becomes_event(self) -> None:
        probs = [0.0, 0.9, 0.8, 0.0]
        events = frames_to_events(probs, frame_seconds=1.0, min_length=0.0)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].start, 1.0)
        self.assertAlmostEqual(events[0].end, 3.0)
        self.assertAlmostEqual(events[0].score, 0.9)

    def test_short_runs_are_removed(self) -> None:
        probs = [0.0, 0.9, 0.0]
        events = frames_to_events(probs, frame_seconds=0.1, min_length=0.2)
        self.assertEqual(events, [])

    def test_close_events_are_concatenated(self) -> None:
        probs = [0.9, 0.9, 0.0, 0.9, 0.9]
        events = frames_to_events(probs, frame_seconds=1.0, min_length=0.0, concat_gap=1.5)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].start, 0.0)
        self.assertAlmostEqual(events[0].end, 5.0)
        self.assertAlmostEqual(events[0].score, 0.9)

    def test_distant_events_are_kept_separate(self) -> None:
        probs = [0.9, 0.0, 0.0, 0.0, 0.9]
        events = frames_to_events(probs, frame_seconds=1.0, min_length=0.0, concat_gap=0.2)
        self.assertEqual(len(events), 2)

    def test_trailing_run_is_closed(self) -> None:
        probs = [0.0, 0.9, 0.9]
        events = frames_to_events(probs, frame_seconds=0.5, min_length=0.0)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].end, 1.5)

    def test_concat_close_merges_overlaps(self) -> None:
        events = [LaughterEvent(0.0, 2.0, 0.6), LaughterEvent(1.5, 3.0, 0.9)]
        merged = concat_close(events, gap=0.2)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].end, 3.0)
        self.assertAlmostEqual(merged[0].score, 0.9)

    def test_remove_short_filters_by_duration(self) -> None:
        events = [LaughterEvent(0.0, 0.1), LaughterEvent(1.0, 2.0)]
        kept = remove_short(events, min_length=0.2)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0].start, 1.0)


def _marker_scorer(chunk: np.ndarray) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    probs = np.zeros(FRAME_COUNT, dtype=np.float32)
    for index in range(FRAME_COUNT):
        center = int((index + 0.5) * FRAME_SECONDS * SR)
        lo = max(0, center - 80)
        hi = min(len(chunk), center + 80)
        if hi > lo and float(np.max(np.abs(chunk[lo:hi]))) > 0.5:
            probs[index] = 0.9
    return probs


class DetectorWithStubScorerTests(unittest.TestCase):
    def _waveform_with_laugh(self, start_sec: float = 6.0, end_sec: float = 6.9) -> np.ndarray:
        waveform = np.zeros(SR * 10, dtype=np.float32)
        waveform[int(start_sec * SR):int(end_sec * SR)] = 0.9
        return waveform

    def test_detect_from_waveform_finds_marked_region(self) -> None:
        detector = LaughterDetector(scorer=_marker_scorer)
        events = detector.detect_from_waveform(self._waveform_with_laugh())
        self.assertTrue(events, "expected a laughter event")
        self.assertLess(abs(events[0].start - 6.0), 0.3)
        self.assertLess(abs(events[0].end - 6.9), 0.3)

    def test_measure_reports_probability_and_duration(self) -> None:
        detector = LaughterDetector(scorer=_marker_scorer)
        measurement = detector.measure_from_waveform(self._waveform_with_laugh())
        self.assertGreater(measurement.max_prob, 0.8)
        self.assertGreater(measurement.seconds_over_half, 0.5)

    def test_has_laughter_thresholds(self) -> None:
        detector = LaughterDetector(scorer=_marker_scorer)
        ok, measurement = detector.has_laughter_from_waveform(
            self._waveform_with_laugh(), threshold=0.5
        )
        self.assertTrue(ok)
        self.assertGreater(measurement.max_prob, 0.8)

        quiet = np.zeros(SR * 10, dtype=np.float32)
        ok, measurement = detector.has_laughter_from_waveform(quiet, threshold=0.5)
        self.assertFalse(ok)
        self.assertLess(measurement.max_prob, 0.1)

    def test_invalid_window_configuration_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LaughterDetector(input_seconds=2.0, overlap_seconds=2.0)


if __name__ == "__main__":
    unittest.main()
