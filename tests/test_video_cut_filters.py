"""Regression tests for pause-removal cut filter graphs.

These tests ensure we avoid per-segment fast seeking (`-ss` before `-i`) when
building cut/concat pipelines, because that can snap to keyframes/packet
boundaries and accumulate timing drift.
"""

from __future__ import annotations

from src.dubbing.debug.performance_tracker import PerformanceTracker
from src.dubbing.video.video_processor import VideoProcessor


def test_build_reencoding_cuts_command_uses_trim_filters(monkeypatch, tmp_path):
    vp = VideoProcessor(PerformanceTracker())

    captured = {}

    def fake_run_ffmpeg_with_progress(self, cmd, total_duration, progress_callback=None, operation_name="FFmpeg"):
        captured["cmd"] = cmd
        captured["total_duration"] = total_duration
        return None

    monkeypatch.setattr(VideoProcessor, "_run_ffmpeg_with_progress", fake_run_ffmpeg_with_progress)

    video_path = str(tmp_path / "input.mp4")
    (tmp_path / "input.mp4").write_bytes(b"")  # Dummy file; FFmpeg is not executed in this test.

    original_command = ["ffmpeg", "-y", "-i", video_path, "-c:v", "copy", "out.mp4"]
    cuts_to_keep = [(0.0, 1.234), (2.0, 3.0)]

    vp._build_reencoding_cuts_command(
        original_command=original_command,
        video_path=video_path,
        cuts_to_keep=cuts_to_keep,
        output_path="out.mp4",
        use_two_pass_encoding=False,
        video_info={},
        progress_callback=None,
    )

    cmd = captured["cmd"]

    assert cmd.count("-i") == 1
    assert "-ss" not in cmd
    assert "-t" not in cmd

    assert "-filter_complex" in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "[0:v]trim" in fc
    assert "setpts=PTS-STARTPTS" in fc
    assert "concat=n=2" in fc

