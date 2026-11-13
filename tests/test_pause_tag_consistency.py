"""Consistency tests for inline pause tags against the Gemini 3.8 spec.

The prompting guide defines only ``<short pause>`` and ``<long pause>`` and
does not specify durations, so these tests lock the project decisions:
3.8 never emits ``medium`` pauses, out-of-spec angle tags are sanitized, and
duration estimation treats tags explicitly instead of counting them as words.
"""

from __future__ import annotations

from pathlib import Path

from src.dubbing.core.segment_optimizer import SegmentOptimizer
from src.tts.voice_sample_manager import VoiceSampleManager


def _separator(tts_system: str, gap: float) -> str:
    optimizer = SegmentOptimizer.__new__(SegmentOptimizer)
    optimizer.tts_system = tts_system
    return optimizer._get_translation_separator(gap)


def test_gemini38_uses_only_short_and_long_pauses() -> None:
    assert _separator("gemini38", 0.1) == " [short pause] "
    assert _separator("gemini38", 0.59) == " [short pause] "
    assert _separator("gemini38", 0.6) == " [long pause] "
    assert _separator("gemini38", 3.0) == " [long pause] "
    for gap in (0.1, 0.25, 0.5, 0.9, 2.0):
        assert "medium" not in _separator("gemini38", gap)


def test_legacy_gemini_keeps_medium_pauses() -> None:
    assert _separator("gemini", 0.4) == " [medium pause] "


def _manager(tmp_path: Path) -> VoiceSampleManager:
    return VoiceSampleManager(
        tts_provider="gemini38",
        voice_list=["Kore"],
        samples_dir=tmp_path / "samples",
        stats_file=tmp_path / "stats.json",
        adjustments_file=tmp_path / "adjustments.json",
        sample_text="sample text",
        audio_embedder=None,
        voice_matcher=None,
        enable_voice_matching=False,
    )


def test_pause_tags_are_not_counted_as_words(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    base = manager.estimate_duration("This is a short test line for estimation.", "Kore")
    tagged = manager.estimate_duration("This is <short pause> a short test line for estimation.", "Kore")

    assert base is not None and tagged is not None
    # The tag contributes its own pause seconds (~0.4s), not two extra words.
    assert 0.3 < (tagged - base) < 0.5


def test_long_pause_estimated_longer_than_short(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    base = manager.estimate_duration("Wait for the result now.", "Kore")
    short = manager.estimate_duration("Wait for the <short pause> result now.", "Kore")
    long = manager.estimate_duration("Wait for the <long pause> result now.", "Kore")

    assert base is not None and short is not None and long is not None
    assert short > base
    assert long > short


def test_legacy_medium_pause_is_estimated_as_short(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    short = manager.estimate_duration("Wait for the <short pause> result now.", "Kore")
    medium = manager.estimate_duration("Wait for the [medium pause] result now.", "Kore")

    assert short is not None and medium is not None
    assert abs(medium - short) < 1e-6
