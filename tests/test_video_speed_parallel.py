from __future__ import annotations

import io
import os
import subprocess

from src.dubbing.debug.performance_tracker import PerformanceTracker
from src.dubbing.video.video_processor import VideoProcessor


def _build_video_processor(monkeypatch):
    vp = VideoProcessor(PerformanceTracker())
    monkeypatch.setattr(vp, "_get_video_info", lambda _path: {"duration": 20.0})
    monkeypatch.setattr("src.dubbing.video.video_processor.os.cpu_count", lambda: 8)
    return vp


def _install_fakes(monkeypatch, captured):
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

    def fake_wait(futures, return_when=None):
        # All submit() calls complete synchronously in the dummy executor, so
        # every future is already done by the time wait() is invoked.
        return set(futures), set()

    def fake_run(cmd, *args, **kwargs):
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

    monkeypatch.setattr("src.dubbing.video.video_processor.ThreadPoolExecutor", DummyExecutor)
    monkeypatch.setattr("src.dubbing.video.video_processor.wait", fake_wait)
    monkeypatch.setattr("src.dubbing.video.video_processor.subprocess.run", fake_run)


def _extract_threads(cmd):
    return cmd[cmd.index("-threads") + 1]


def test_run_ffmpeg_with_progress_emits_final_100(monkeypatch):
    vp = _build_video_processor(monkeypatch)
    progress_events = []
    log_events = []

    class DummyProc:
        def __init__(self):
            self.stdout = io.StringIO(
                "frame=1\n"
                "out_time_us=95000000\n"
                "progress=continue\n"
            )
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "src.dubbing.video.video_processor.subprocess.Popen",
        lambda *args, **kwargs: DummyProc(),
    )

    result = vp._run_ffmpeg_with_progress(
        ["ffmpeg", "-i", "input.mp4", "output.mp4"],
        total_duration=100.0,
        progress_callback=lambda current, total, label: progress_events.append((current, total, label)),
        log_callback=lambda message: log_events.append(message),
        label="Encoding final stretched video",
    )

    assert result.returncode == 0
    assert progress_events[-1] == (100, 100, "Encoding final stretched video")
    assert "Encoding final stretched video: 100%" in log_events


def test_minterpolate_segments_run_sequentially_with_full_cpu(monkeypatch, tmp_path):
    vp = _build_video_processor(monkeypatch)

    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    input_video.write_bytes(b"dummy")

    captured = {"max_workers": None, "segment_cmds": [], "concat_cmds": []}
    _install_fakes(monkeypatch, captured)

    speed_segments = [
        # Plain: speed >= threshold (0.75) → runs in parallel pool.
        {"original_start": 0.0, "original_end": 2.0, "video_speed": 1.0, "use_minterpolate": False},
        {"original_start": 2.0, "original_end": 5.0, "video_speed": 0.8, "use_minterpolate": True},
        # Minterpolate: speed < threshold → sequential phase with -threads=cpu-1.
        {"original_start": 5.0, "original_end": 8.0, "video_speed": 0.4, "use_minterpolate": True},
        {"original_start": 8.0, "original_end": 11.0, "video_speed": 0.1, "use_minterpolate": True},
    ]

    result_path, timing_adjustments = vp._apply_per_segment_video_speed(
        str(input_video),
        speed_segments,
        str(output_video),
    )

    # 4 segment ffmpegs (two minterpolate sequential + two plain pooled)
    assert len(captured["segment_cmds"]) == 4

    minterpolate_cmds = [cmd for cmd in captured["segment_cmds"] if "minterpolate=" in " ".join(cmd)]
    plain_cmds = [cmd for cmd in captured["segment_cmds"] if "minterpolate=" not in " ".join(cmd)]

    assert len(minterpolate_cmds) == 2
    assert len(plain_cmds) == 2

    # Minterpolate → threads == cpu_count - 1 == 7.
    for cmd in minterpolate_cmds:
        assert _extract_threads(cmd) == "7"

    # Plain → threads == 1 (queue is deeper than pool at submit time).
    for cmd in plain_cmds:
        assert _extract_threads(cmd) == "1"

    # Plain pool is bounded by the number of pending plain segments (2 here),
    # so max_workers is capped at min(segment_workers, 2).
    assert captured["max_workers"] == 2

    # Common ffmpeg invariants.
    for cmd in captured["segment_cmds"]:
        assert "-vsync" in cmd
        assert cmd[cmd.index("-vsync") + 1] == "cfr"
        assert "-r" in cmd
        assert "-tag:v" in cmd
        assert cmd[cmd.index("-tag:v") + 1] == "avc1"

    # Final concat produced exactly once.
    assert len(captured["concat_cmds"]) == 1
    concat_cmd = captured["concat_cmds"][0]
    assert "-c:v" in concat_cmd
    assert concat_cmd[concat_cmd.index("-c:v") + 1] == "libx264"
    assert "-fflags" in concat_cmd
    assert concat_cmd[concat_cmd.index("-fflags") + 1] == "+genpts"
    assert "-vsync" in concat_cmd
    assert concat_cmd[concat_cmd.index("-vsync") + 1] == "cfr"
    assert "-tag:v" in concat_cmd

    assert result_path == str(output_video)
    assert len(timing_adjustments) == 4


def test_chronological_order_is_preserved_across_phases(monkeypatch, tmp_path):
    """Minterpolate segments run out of chronological order (sequential, heaviest
    first), so the final concat list and timing_adjustments must still be rebuilt
    in the original input order regardless of processing order.
    """
    vp = _build_video_processor(monkeypatch)

    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    input_video.write_bytes(b"dummy")

    captured = {"max_workers": None, "segment_cmds": [], "concat_cmds": []}
    _install_fakes(monkeypatch, captured)

    # Index 0: plain. 1: heavy minterpolate (0.1x). 2: plain. 3: light minterpolate
    # (0.5x). 4: plain. Heavy runs before light in phase 1, and both run before
    # plain segments in phase 2.
    speed_segments = [
        {"original_start": 0.0, "original_end": 2.0, "video_speed": 1.0, "use_minterpolate": False},
        {"original_start": 2.0, "original_end": 5.0, "video_speed": 0.1, "use_minterpolate": True},
        {"original_start": 5.0, "original_end": 7.0, "video_speed": 0.9, "use_minterpolate": False},
        {"original_start": 7.0, "original_end": 10.0, "video_speed": 0.5, "use_minterpolate": True},
        {"original_start": 10.0, "original_end": 12.0, "video_speed": 1.0, "use_minterpolate": False},
    ]

    _, timing_adjustments = vp._apply_per_segment_video_speed(
        str(input_video),
        speed_segments,
        str(output_video),
    )

    # Processing order: index 1 (heaviest minterpolate) → 3 (lighter minterpolate)
    # → 0/2/4 via pool.
    processed_order = []
    for cmd in captured["segment_cmds"]:
        out_path = cmd[-1]
        # Extract the numeric id from segment_000001.mp4.
        name = os.path.basename(out_path)
        processed_order.append(int(name.replace("segment_", "").replace(".mp4", "")))
    assert processed_order[0] == 1, f"expected heaviest minterpolate first, got {processed_order}"
    assert processed_order[1] == 3, f"expected lighter minterpolate second, got {processed_order}"
    assert set(processed_order[2:]) == {0, 2, 4}

    # timing_adjustments must be rebuilt in chronological (original) order, i.e.
    # monotonically increasing original_start matching the input segments.
    assert [adj["original_start"] for adj in timing_adjustments] == [0.0, 2.0, 5.0, 7.0, 10.0]
    # And new_start must be strictly monotonically increasing (linear timeline).
    new_starts = [adj["new_start"] for adj in timing_adjustments]
    assert new_starts == sorted(new_starts)
    assert new_starts[0] == 0.0


def test_source_start_offset_is_used_only_for_extraction(monkeypatch, tmp_path):
    vp = _build_video_processor(monkeypatch)
    vp.video_segment_seek_padding = 0.0

    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    input_video.write_bytes(b"dummy")

    captured = {"max_workers": None, "segment_cmds": [], "concat_cmds": []}
    _install_fakes(monkeypatch, captured)

    result_path, timing_adjustments = vp._apply_per_segment_video_speed(
        str(input_video),
        [
            {
                "original_start": 5.0,
                "original_end": 7.0,
                "video_speed": 1.0,
                "use_minterpolate": False,
            }
        ],
        str(output_video),
        source_start_offset=3.0,
    )

    assert result_path == str(output_video)
    assert len(captured["segment_cmds"]) == 1
    cmd = captured["segment_cmds"][0]

    # Extraction must seek into the full source timeline (5s clip time + 3s clip offset).
    assert "-ss" in cmd
    assert cmd[cmd.index("-ss") + 1] == "8.000000"
    assert "-vf" in cmd
    vf = cmd[cmd.index("-vf") + 1]
    assert "trim=start=0.000000:end=2.000000" in vf

    # Subtitle/timing adjustment metadata must remain clip-relative.
    assert len(timing_adjustments) == 1
    assert timing_adjustments[0]["original_start"] == 5.0
    assert timing_adjustments[0]["original_end"] == 7.0


def test_all_plain_segments_skip_sequential_phase(monkeypatch, tmp_path):
    vp = _build_video_processor(monkeypatch)

    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    input_video.write_bytes(b"dummy")

    captured = {"max_workers": None, "segment_cmds": [], "concat_cmds": []}
    _install_fakes(monkeypatch, captured)

    speed_segments = [
        {"original_start": 0.0, "original_end": 2.0, "video_speed": 1.0, "use_minterpolate": False},
        {"original_start": 2.0, "original_end": 4.0, "video_speed": 0.9, "use_minterpolate": True},
    ]

    result_path, timing_adjustments = vp._apply_per_segment_video_speed(
        str(input_video),
        speed_segments,
        str(output_video),
    )

    # All segments run through the parallel pool.
    assert len(captured["segment_cmds"]) == 2
    for cmd in captured["segment_cmds"]:
        assert "minterpolate=" not in " ".join(cmd)
        assert _extract_threads(cmd) == "1"

    assert captured["max_workers"] == 2
    assert result_path == str(output_video)
    assert len(timing_adjustments) == 2
