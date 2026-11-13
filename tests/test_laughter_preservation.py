from __future__ import annotations

import os
import tempfile
import unittest

from pydub import AudioSegment

from src.dubbing.audio.laughter_detector import LaughterEvent
from src.dubbing.core.config import DubbingConfig
from src.dubbing.core.smart_dubbing import (
    SmartDubbing,
    build_laughter_preserve_ranges,
    split_translation_at_fraction,
)


class _AudioProcessorStub:
    def __init__(self, duration: float = 100.0):
        self._duration = duration

    def get_total_duration(self) -> float:
        return self._duration


class _StubDubbing:
    def __init__(self, events=None, duration: float = 100.0, **config):
        self.config = config
        self._laughter_events = list(events or [])
        self.audio_processor = _AudioProcessorStub(duration)
        self._laughter_preserve_ranges = []
        self._preserved_laughter_stats = {}
        self._preserved_laughter_output = []

    def get(self, key, default=None):
        return self.config.get(key, default)

    def _laughter_preserve_config(self):
        return SmartDubbing._laughter_preserve_config(self)

    def build_laughter_preserve_ranges(self, audio_file=None):
        return SmartDubbing.build_laughter_preserve_ranges(self, audio_file)

    def map_original_to_output(self, positions, time_value):
        return SmartDubbing.map_original_to_output(positions, time_value)


def _segment(start: float, end: float, text: str, speaker: str = "SPEAKER_A") -> dict:
    return {
        "speaker": speaker,
        "start": start,
        "end": end,
        "text": text,
        "translation": text,
        "long_translation": text,
    }


def _stub(events=None, **config):
    base = {
        "laughter_mode": "preserve",
        "laughter_preserve_pad_seconds": 0.0,
        "laughter_preserve_merge_gap": 0.25,
        "laughter_preserve_min_duration": 0.25,
        "laughter_preserve_boundary_snap_seconds": 1.5,
        "laughter_preserve_target_dbfs": -20.0,
        "segments_optimization": {"min_segment_duration": 0.5},
    }
    base.update(config)
    return _StubDubbing(events=events, **base)


class BuildRangesTests(unittest.TestCase):
    def test_pad_and_clamp(self) -> None:
        ranges = build_laughter_preserve_ranges(
            [LaughterEvent(0.05, 1.0, 0.9)], pad_seconds=0.2, merge_gap=0.1
        )
        self.assertEqual(len(ranges), 1)
        self.assertAlmostEqual(ranges[0][0], 0.0)
        self.assertAlmostEqual(ranges[0][1], 1.2)

    def test_merge_close_events(self) -> None:
        ranges = build_laughter_preserve_ranges(
            [LaughterEvent(1.0, 1.4, 0.9), LaughterEvent(1.5, 1.9, 0.8)],
            pad_seconds=0.0,
            merge_gap=0.3,
        )
        self.assertEqual(ranges, [(1.0, 1.9)])

    def test_drop_short_ranges(self) -> None:
        ranges = build_laughter_preserve_ranges(
            [LaughterEvent(1.0, 1.1, 0.9)], pad_seconds=0.0, min_duration=0.25
        )
        self.assertEqual(ranges, [])

    def test_max_end_clamps(self) -> None:
        ranges = build_laughter_preserve_ranges(
            [LaughterEvent(9.0, 10.5, 0.9)], pad_seconds=0.0, max_end=10.0
        )
        self.assertEqual(ranges, [(9.0, 10.0)])


class SplitTranslationTests(unittest.TestCase):
    def test_split_at_fraction(self) -> None:
        pre, post = split_translation_at_fraction("раз два три четыре", 0.5)
        self.assertEqual(pre, "раз два")
        self.assertEqual(post, "три четыре")

    def test_too_short_returns_none(self) -> None:
        self.assertIsNone(split_translation_at_fraction("раз два", 0.5))
        self.assertIsNone(split_translation_at_fraction("", 0.5))


class PreparePreservationTests(unittest.TestCase):
    def test_laugh_inside_segment_splits_into_two_parts(self) -> None:
        stub = _stub(events=[LaughterEvent(3.0, 4.0, 0.9)])
        segments = [_segment(0.0, 10.0, "раз два три четыре")]
        parts, stats = SmartDubbing.prepare_laughter_preservation(stub, segments, None)

        self.assertEqual(len(parts), 2)
        self.assertEqual(stats["ranges"], 1)
        self.assertEqual(stats["segments_split"], 1)
        self.assertEqual(parts[0]["end"], 3.0)
        self.assertEqual(parts[1]["start"], 4.0)
        self.assertTrue(all(p.get("laughter_preserve") for p in parts))
        self.assertNotEqual(parts[0]["translation"], parts[1]["translation"])
        # original list must stay untouched for subtitles
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["end"], 10.0)

    def test_whole_segment_laugh_is_dropped(self) -> None:
        stub = _stub(events=[LaughterEvent(0.0, 10.0, 0.9)])
        segments = [_segment(0.0, 10.0, "раз два три четыре")]
        parts, _ = SmartDubbing.prepare_laughter_preservation(stub, segments, None)
        self.assertEqual(parts, [])

    def test_segment_without_laugh_untouched(self) -> None:
        stub = _stub(events=[LaughterEvent(30.0, 31.0, 0.9)])
        segments = [_segment(0.0, 10.0, "раз два три четыре")]
        parts, stats = SmartDubbing.prepare_laughter_preservation(stub, segments, None)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], segments[0])
        self.assertEqual(stats["segments_split"], 0)

    def test_too_short_text_is_skipped(self) -> None:
        stub = _stub(events=[LaughterEvent(3.0, 4.0, 0.9)])
        segments = [_segment(0.0, 10.0, "два слова")]
        parts, stats = SmartDubbing.prepare_laughter_preservation(stub, segments, None)
        self.assertEqual(len(parts), 1)
        self.assertEqual(stats["skipped"], 1)

    def test_is_laughter_gap(self) -> None:
        stub = _stub(events=[LaughterEvent(3.0, 4.0, 0.9)])
        SmartDubbing.prepare_laughter_preservation(
            stub, [_segment(0.0, 10.0, "раз два три четыре")], None
        )
        prev = {"end": 3.0}
        cur = {"start": 4.0}
        self.assertTrue(SmartDubbing._is_laughter_gap(stub, prev, cur))
        self.assertFalse(SmartDubbing._is_laughter_gap(stub, {"end": 12.0}, {"start": 13.0}))

    def test_laugh_near_start_snaps_boundary(self) -> None:
        stub = _stub(events=[LaughterEvent(0.8, 1.8, 0.9)])
        segments = [_segment(0.0, 10.0, "раз два три четыре")]
        parts, stats = SmartDubbing.prepare_laughter_preservation(stub, segments, None)
        self.assertEqual(len(parts), 1)
        self.assertAlmostEqual(parts[0]["start"], 1.8)
        self.assertEqual(parts[0]["end"], 10.0)
        self.assertEqual(parts[0]["text"], "раз два три четыре")
        self.assertEqual(parts[0]["laughter_snapped"], "start")
        self.assertEqual(stats["segments_snapped"], 1)
        self.assertEqual(stats["segments_split"], 0)

    def test_laugh_near_end_snaps_boundary(self) -> None:
        stub = _stub(events=[LaughterEvent(8.5, 9.5, 0.9)])
        segments = [_segment(0.0, 10.0, "раз два три четыре")]
        parts, stats = SmartDubbing.prepare_laughter_preservation(stub, segments, None)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["start"], 0.0)
        self.assertAlmostEqual(parts[0]["end"], 8.5)
        self.assertEqual(parts[0]["laughter_snapped"], "end")
        self.assertEqual(stats["segments_snapped"], 1)

    def test_laugh_at_segment_start_is_trimmed(self) -> None:
        stub = _stub(events=[LaughterEvent(0.0, 1.0, 0.9)])
        segments = [_segment(0.0, 10.0, "раз два три четыре")]
        parts, stats = SmartDubbing.prepare_laughter_preservation(stub, segments, None)
        self.assertEqual(len(parts), 1)
        self.assertAlmostEqual(parts[0]["start"], 1.0)
        self.assertEqual(stats["segments_snapped"], 1)

    def test_middle_laugh_still_splits(self) -> None:
        stub = _stub(events=[LaughterEvent(4.0, 5.0, 0.9)])
        segments = [_segment(0.0, 10.0, "раз два три четыре")]
        parts, stats = SmartDubbing.prepare_laughter_preservation(stub, segments, None)
        self.assertEqual(len(parts), 2)
        self.assertEqual(stats["segments_split"], 1)
        self.assertEqual(stats["segments_snapped"], 0)

    def test_boundary_and_middle_laughs_combine(self) -> None:
        stub = _stub(events=[LaughterEvent(0.5, 1.2, 0.9), LaughterEvent(5.0, 5.8, 0.9)])
        segments = [_segment(0.0, 10.0, "раз два три четыре")]
        parts, stats = SmartDubbing.prepare_laughter_preservation(stub, segments, None)
        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(parts[0]["start"], 1.2)
        self.assertEqual(parts[0]["end"], 5.0)
        self.assertEqual(parts[1]["start"], 5.8)
        self.assertEqual(stats["segments_snapped"], 1)
        self.assertEqual(stats["segments_split"], 1)

    def test_laugh_on_both_edges_of_short_segment_is_removed(self) -> None:
        stub = _stub(events=[LaughterEvent(0.0, 3.0, 0.9)])
        segments = [_segment(0.0, 3.0, "раз два три четыре")]
        parts, stats = SmartDubbing.prepare_laughter_preservation(stub, segments, None)
        self.assertEqual(parts, [])
        self.assertEqual(stats["segments_removed"], 1)


class MapTimelineTests(unittest.TestCase):
    def test_map_within_segment_and_gap(self) -> None:
        positions = [
            {"start": 0.0, "end": 4.0, "original_start": 0.0, "original_end": 5.0},
            {"start": 6.0, "end": 10.0, "original_start": 8.0, "original_end": 12.0},
        ]
        # inside first segment (2/5 of the way)
        self.assertAlmostEqual(SmartDubbing.map_original_to_output(positions, 2.5), 2.0)
        # inside the gap (5 -> 8 maps to 4 -> 6)
        self.assertAlmostEqual(SmartDubbing.map_original_to_output(positions, 6.5), 5.0)

    def test_empty_positions(self) -> None:
        self.assertIsNone(SmartDubbing.map_original_to_output([], 1.0))


class SpliceTests(unittest.TestCase):
    def test_splice_replaces_dubbed_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = os.path.join(tmp, "source.wav")
            from pydub.generators import Sine

            tone = Sine(440).to_audio_segment(duration=1000)
            source = AudioSegment.silent(duration=2000) + tone + AudioSegment.silent(duration=7000)
            source.export(source_path, format="wav")

            stub = _stub(events=[LaughterEvent(2.0, 3.0, 0.9)])
            stub._laughter_preserve_ranges = [(2.0, 3.0)]
            combined = AudioSegment.silent(duration=10000)
            positions = [
                {"start": 0.0, "end": 10.0, "original_start": 0.0, "original_end": 10.0}
            ]

            out = SmartDubbing.splice_preserved_laughter(stub, combined, source_path, positions)

            self.assertEqual(len(out), 10000)
            self.assertEqual(len(stub._preserved_laughter_output), 1)
            self.assertAlmostEqual(stub._preserved_laughter_output[0]["output_start"], 2.0)
            window = out[2000:3000]
            self.assertGreater(window.dBFS, -60)


class ConfigTests(unittest.TestCase):
    def test_laughter_mode_default_and_normalization(self) -> None:
        cfg = DubbingConfig()
        self.assertEqual(cfg.get("laughter_mode"), "dub")
        cfg.config["laughter_mode"] = "PRESERVE"
        cfg.process_special_parameters()
        self.assertEqual(cfg.get("laughter_mode"), "preserve")
        cfg.config["laughter_mode"] = "bogus"
        cfg.process_special_parameters()
        self.assertEqual(cfg.get("laughter_mode"), "dub")


if __name__ == "__main__":
    unittest.main()
