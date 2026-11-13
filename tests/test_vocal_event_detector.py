from __future__ import annotations

import unittest

import numpy as np

from src.dubbing.audio.vocal_event_detector import (
    AstVocalEventDetector,
    VocalEvent,
    concat_close,
    frames_to_events,
    remove_short,
)

SR = 16000


class FrameEventTests(unittest.TestCase):
    def test_single_run_becomes_event(self) -> None:
        probs = [0.0, 0.9, 0.8, 0.0]
        events = frames_to_events(probs, "scream", frame_seconds=1.0, min_length=0.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "scream")
        self.assertAlmostEqual(events[0].start, 1.0)
        self.assertAlmostEqual(events[0].end, 3.0)
        self.assertAlmostEqual(events[0].score, 0.9)

    def test_short_runs_are_removed(self) -> None:
        events = frames_to_events([0.0, 0.9, 0.0], "cry", frame_seconds=0.1, min_length=0.2)
        self.assertEqual(events, [])

    def test_concat_close_merges_same_kind_only(self) -> None:
        events = [
            VocalEvent("sob", 0.0, 2.0, 0.6),
            VocalEvent("sob", 1.5, 3.0, 0.9),
            VocalEvent("scream", 2.1, 2.6, 0.8),
        ]
        merged = concat_close(events, gap=0.2)
        kinds = [event.kind for event in merged]
        self.assertEqual(len(merged), 2)
        self.assertEqual(kinds.count("sob"), 1)
        self.assertAlmostEqual(next(e for e in merged if e.kind == "sob").end, 3.0)

    def test_remove_short_filters_by_duration(self) -> None:
        events = [VocalEvent("gasp", 0.0, 0.1), VocalEvent("gasp", 1.0, 2.0)]
        kept = remove_short(events, min_length=0.2)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0].start, 1.0)


def _marker_scorer_factory(kind_column: int, kinds: int, active: tuple, frame_seconds: float = 0.1, frames: int = 101):
    def scorer(chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)
        matrix = np.zeros((frames, kinds), dtype=np.float32)
        for index in range(frames):
            center = int((index + 0.5) * frame_seconds * SR)
            lo = max(0, center - 80)
            hi = min(len(chunk), center + 80)
            if hi > lo and float(np.max(np.abs(chunk[lo:hi]))) > 0.5:
                matrix[index, kind_column] = 0.9
        return matrix

    return scorer


def _window_scorer_factory(kind_column: int, kinds: int, frame_seconds: float = 0.1, frames: int = 101):
    def scorer(chunk: np.ndarray):
        chunk = np.asarray(chunk, dtype=np.float32)
        patch = np.zeros((frames, kinds), dtype=np.float32)
        for index in range(frames):
            center = int((index + 0.5) * frame_seconds * SR)
            lo = max(0, center - 80)
            hi = min(len(chunk), center + 80)
            if hi > lo and float(np.max(np.abs(chunk[lo:hi]))) > 0.5:
                patch[index, kind_column] = 0.9
        pooled = np.full(kinds, 0.01, dtype=np.float32)
        if patch[:, kind_column].max() > 0.5:
            pooled[kind_column] = 0.8
        return patch, pooled

    return scorer


class WindowModeStubTests(unittest.TestCase):
    def _waveform(self, start_sec: float = 3.0, end_sec: float = 4.0) -> np.ndarray:
        waveform = np.zeros(SR * 10, dtype=np.float32)
        waveform[int(start_sec * SR):int(end_sec * SR)] = 0.9
        return waveform

    def test_window_detection_localizes_event(self) -> None:
        detector = AstVocalEventDetector(
            kinds=("scream", "cry"),
            window_scorer=_window_scorer_factory(0, 2),
            stub_frame_seconds=0.1,
            hop_seconds=1.0,
            min_event_seconds=0.3,
            concat_gap_seconds=0.5,
        )
        events = detector.detect_from_waveform(self._waveform())
        self.assertTrue(events, "expected a scream event")
        self.assertTrue(all(event.kind == "scream" for event in events))
        event = events[0]
        self.assertAlmostEqual(event.start, 3.0, delta=0.4)
        self.assertAlmostEqual(event.end, 4.0, delta=0.4)

    def test_window_measure_uses_pooled_prob(self) -> None:
        detector = AstVocalEventDetector(
            kinds=("scream", "cry"),
            window_scorer=_window_scorer_factory(1, 2),
            stub_frame_seconds=0.1,
            hop_seconds=1.0,
            min_event_seconds=0.3,
        )
        measurement = detector.measure_from_waveform(self._waveform(), "cry")
        self.assertAlmostEqual(measurement.max_prob, 0.8, delta=0.05)
        quiet = np.zeros(SR * 10, dtype=np.float32)
        measurement = detector.measure_from_waveform(quiet, "cry")
        self.assertLess(measurement.max_prob, 0.1)

    def test_window_mode_requires_threshold(self) -> None:
        detector = AstVocalEventDetector(
            kinds=("scream",),
            window_scorer=_window_scorer_factory(0, 1),
            stub_frame_seconds=0.1,
            hop_seconds=1.0,
            frame_threshold=0.95,
            min_event_seconds=0.3,
        )
        events = detector.detect_from_waveform(self._waveform())
        self.assertEqual(events, [])


class DetectorWithStubScorerTests(unittest.TestCase):
    def _waveform(self, start_sec: float = 3.0, end_sec: float = 4.0) -> np.ndarray:
        waveform = np.zeros(SR * 10, dtype=np.float32)
        waveform[int(start_sec * SR):int(end_sec * SR)] = 0.9
        return waveform

    def test_detect_from_waveform_finds_marked_region(self) -> None:
        detector = AstVocalEventDetector(
            kinds=("scream", "cry"),
            frame_scorer=_marker_scorer_factory(0, 2, (3.0, 4.0)),
            stub_frame_seconds=0.1,
            hop_seconds=1.5,
            min_event_seconds=0.3,
            concat_gap_seconds=0.4,
        )
        events = detector.detect_from_waveform(self._waveform())
        self.assertTrue(events, "expected a scream event")
        starts = [event.start for event in events]
        self.assertLess(min(starts), 3.4)
        self.assertTrue(all(event.kind == "scream" for event in events))

    def test_measure_reports_probability_and_duration(self) -> None:
        detector = AstVocalEventDetector(
            kinds=("scream", "cry"),
            frame_scorer=_marker_scorer_factory(1, 2, (3.0, 4.0)),
            stub_frame_seconds=0.1,
            hop_seconds=1.5,
            min_event_seconds=0.3,
        )
        measurement = detector.measure_from_waveform(self._waveform(), "cry")
        self.assertGreater(measurement.max_prob, 0.8)
        self.assertGreater(measurement.seconds_over_threshold, 0.5)

    def test_quiet_waveform_has_no_event(self) -> None:
        detector = AstVocalEventDetector(
            kinds=("scream",),
            frame_scorer=_marker_scorer_factory(0, 1, (3.0, 4.0)),
            stub_frame_seconds=0.1,
            hop_seconds=1.5,
            min_event_seconds=0.3,
        )
        quiet = np.zeros(SR * 10, dtype=np.float32)
        events = detector.detect_from_waveform(quiet)
        self.assertEqual(events, [])
        measurement = detector.measure_from_waveform(quiet, "scream")
        self.assertLess(measurement.max_prob, 0.1)

    def test_invalid_hop_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AstVocalEventDetector(hop_seconds=0)

    def test_default_kinds_exclude_unknown(self) -> None:
        detector = AstVocalEventDetector(kinds=("scream", "bogus"))
        self.assertEqual(detector.kinds, ("scream",))

    def test_kind_indices_built_from_id2label(self) -> None:
        detector = AstVocalEventDetector(kinds=("scream", "shout", "cry"))
        indices = detector._build_kind_indices(
            {0: "Speech", 8: "Shout", 14: "Screaming", 22: "Crying, sobbing", 44: "Gasp"}
        )
        self.assertEqual(indices.get("shout"), [8])
        self.assertEqual(indices.get("scream"), [14])
        self.assertEqual(indices.get("cry"), [22])
        self.assertNotIn("gasp", indices)


if __name__ == "__main__":
    unittest.main()
