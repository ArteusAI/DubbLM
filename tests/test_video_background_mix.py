from __future__ import annotations

import subprocess

from src.dubbing.debug.performance_tracker import PerformanceTracker
from src.dubbing.video.video_processor import VideoProcessor


def test_background_audio_is_still_mixed_when_video_speed_changes(monkeypatch, tmp_path):
    vp = VideoProcessor(PerformanceTracker())

    input_video = tmp_path / "input.mp4"
    dubbed_audio = tmp_path / "dubbed.wav"
    background_audio = tmp_path / "background.wav"
    output_video = tmp_path / "output.mp4"

    input_video.write_bytes(b"video")
    dubbed_audio.write_bytes(b"dubbed")
    background_audio.write_bytes(b"background")

    captured = {}

    monkeypatch.setattr(
        vp,
        "_get_video_info",
        lambda _path: {
            "duration": 20.0,
            "video_stream": {"height": 1080},
            "video_bitrate": "1000000",
            "video_codec": "h264",
        },
    )
    monkeypatch.setattr(
        vp,
        "_apply_per_segment_video_speed",
        lambda *args, **kwargs: (str(input_video), []),
    )

    def fake_run_ffmpeg_with_progress(cmd, total_duration, log_callback=None, progress_callback=None, label="FFmpeg"):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(vp, "_run_ffmpeg_with_progress", fake_run_ffmpeg_with_progress)

    vp.combine_audio_with_video(
        video_path=str(input_video),
        translated_audio_path=str(dubbed_audio),
        background_audio_path=str(background_audio),
        output_file=str(output_video),
        video_speed_segments=[
            {
                "original_start": 0.0,
                "original_end": 5.0,
                "video_speed": 1.08,
                "use_minterpolate": False,
            }
        ],
    )

    cmd = captured["cmd"]
    cmd_str = " ".join(cmd)

    assert str(background_audio) in cmd
    assert "-filter_complex" in cmd
    assert "amix=inputs=2:duration=longest" in cmd_str
    assert "[bg_audio_reduced]" in cmd_str


def _capture_combine_cmd(monkeypatch, tmp_path, **combine_kwargs) -> list[str]:
    vp = VideoProcessor(PerformanceTracker())

    input_video = tmp_path / "input.mp4"
    dubbed_audio = tmp_path / "dubbed.wav"
    input_video.write_bytes(b"video")
    dubbed_audio.write_bytes(b"dubbed")

    captured = {}

    monkeypatch.setattr(
        vp,
        "_get_video_info",
        lambda _path: {
            "duration": 20.0,
            "video_stream": {"height": 1080},
            "video_bitrate": "1000000",
            "video_codec": "h264",
        },
    )

    def fake_run_ffmpeg_with_progress(
        cmd, total_duration, log_callback=None, progress_callback=None, label="FFmpeg"
    ):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(vp, "_run_ffmpeg_with_progress", fake_run_ffmpeg_with_progress)

    vp.combine_audio_with_video(
        video_path=str(input_video),
        translated_audio_path=str(dubbed_audio),
        output_file=str(tmp_path / "output.mp4"),
        **combine_kwargs,
    )
    return captured["cmd"]


def test_final_normalization_is_disabled_by_default(monkeypatch, tmp_path):
    cmd_str = " ".join(_capture_combine_cmd(monkeypatch, tmp_path))

    assert "loudnorm" not in cmd_str
    assert "alimiter" not in cmd_str


def test_final_normalization_can_be_enabled(monkeypatch, tmp_path):
    cmd_str = " ".join(_capture_combine_cmd(monkeypatch, tmp_path, normalize_audio=True))

    assert "loudnorm=I=-14" in cmd_str
    assert "alimiter=limit=0.95" in cmd_str


def test_background_voice_layer_is_disabled_by_default(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"vocals")

    cmd_str = " ".join(
        _capture_combine_cmd(
            monkeypatch, tmp_path, background_voice_path=str(vocals)
        )
    )

    assert str(vocals) not in cmd_str
    assert "bg_voice_layer" not in cmd_str


def test_background_voice_layer_is_mixed_under_the_dub(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"vocals")

    cmd_str = " ".join(
        _capture_combine_cmd(
            monkeypatch,
            tmp_path,
            background_voice_path=str(vocals),
            background_voice_volume=0.2,
        )
    )

    assert str(vocals) in cmd_str
    assert "volume=0.2[bg_voice_layer]" in cmd_str
    # The existing streams keep their level, the voice is added on top.
    assert "amix=inputs=2:duration=longest:weights='1 1':normalize=0[dub_mixed_with_voice]" in cmd_str
    assert "[dub_mixed_with_voice]" in cmd_str


def test_background_voice_layer_ducks_with_keep_original_ranges(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"vocals")

    cmd_str = " ".join(
        _capture_combine_cmd(
            monkeypatch,
            tmp_path,
            background_voice_path=str(vocals),
            background_voice_volume=0.2,
            keep_original_audio_ranges=[(5.0, 8.0)],
        )
    )

    assert "bg_voice_conditional" in cmd_str
    assert "amix=inputs=3:duration=longest:weights='0.5 0.5 1':normalize=0[final_mixed_audio]" in cmd_str
