from __future__ import annotations

import os
import subprocess

from src.dubbing.debug.performance_tracker import PerformanceTracker
from src.dubbing.video.video_processor import VideoProcessor


def test_apply_per_segment_video_speed_uses_cpu_minus_one_workers(monkeypatch, tmp_path):
    vp = VideoProcessor(PerformanceTracker())

    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    input_video.write_bytes(b"dummy")

    monkeypatch.setattr(vp, "_get_video_info", lambda _path: {"duration": 20.0})
    monkeypatch.setattr("src.dubbing.video.video_processor.os.cpu_count", lambda: 8)

    captured = {
        "max_workers": None,
        "segment_cmds": [],
        "concat_cmds": [],
    }

    class DummyFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class DummyExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return DummyFuture(fn(*args, **kwargs))

    monkeypatch.setattr("src.dubbing.video.video_processor.ThreadPoolExecutor", DummyExecutor)
    monkeypatch.setattr("src.dubbing.video.video_processor.as_completed", lambda futures: futures)

    def fake_run(cmd, capture_output=True, text=True, check=True):
        if "-vf" in cmd:
            captured["segment_cmds"].append(cmd)
            os.makedirs(os.path.dirname(cmd[-1]), exist_ok=True)
            with open(cmd[-1], "wb") as f:
                f.write(b"segment")
        elif "-f" in cmd and "concat" in cmd:
            captured["concat_cmds"].append(cmd)
            os.makedirs(os.path.dirname(cmd[-1]), exist_ok=True)
            with open(cmd[-1], "wb") as f:
                f.write(b"final")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("src.dubbing.video.video_processor.subprocess.run", fake_run)

    speed_segments = [
        {"original_start": 0.0, "original_end": 2.0, "video_speed": 1.0, "use_minterpolate": False},
        {"original_start": 2.0, "original_end": 5.0, "video_speed": 0.8, "use_minterpolate": True},
    ]

    result_path, timing_adjustments = vp._apply_per_segment_video_speed(
        str(input_video),
        speed_segments,
        str(output_video),
    )

    assert captured["max_workers"] == 7
    assert len(captured["segment_cmds"]) == 2
    for cmd in captured["segment_cmds"]:
        assert "-threads" in cmd
        assert cmd[cmd.index("-threads") + 1] == "1"
        assert "-vsync" in cmd
        assert cmd[cmd.index("-vsync") + 1] == "cfr"
        assert "-r" in cmd
        assert "-tag:v" in cmd
        assert cmd[cmd.index("-tag:v") + 1] == "avc1"
    assert len(captured["concat_cmds"]) == 1
    concat_cmd = captured["concat_cmds"][0]
    assert "-c" not in concat_cmd
    assert "-c:v" in concat_cmd
    assert concat_cmd[concat_cmd.index("-c:v") + 1] == "libx264"
    assert "-fflags" in concat_cmd
    assert concat_cmd[concat_cmd.index("-fflags") + 1] == "+genpts"
    assert "-vsync" in concat_cmd
    assert concat_cmd[concat_cmd.index("-vsync") + 1] == "cfr"
    assert "-tag:v" in concat_cmd
    assert concat_cmd[concat_cmd.index("-tag:v") + 1] == "avc1"
    assert result_path == str(output_video)
    assert len(timing_adjustments) == 2
