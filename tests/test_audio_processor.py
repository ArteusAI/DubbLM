import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from dubbing.core.cache_manager import CacheManager
from dubbing.debug.performance_tracker import PerformanceTracker


def test_process_background_audio_skips_when_audio_separator_missing(tmp_path, caplog):
    audio_file = tmp_path / "source.wav"
    audio_file.write_bytes(b"fake wav bytes")

    from dubbing.audio.audio_processor import AudioProcessor

    processor = AudioProcessor(
        CacheManager(use_cache=False, input_file=str(audio_file)),
        PerformanceTracker(),
    )

    with caplog.at_level(logging.WARNING):
        background_audio = processor.process_background_audio(str(audio_file))

    assert background_audio is None
    assert "audio_separator" in caplog.text


def test_separate_background_and_vocals_returns_both_tracks(tmp_path):
    audio_file = tmp_path / "source.wav"
    audio_file.write_bytes(b"fake wav bytes")
    separated_background = tmp_path / "source_(Instrumental)_2_HP-UVR.wav"
    separated_vocals = tmp_path / "source_(Vocals)_2_HP-UVR.wav"
    separated_background.write_bytes(b"background")
    separated_vocals.write_bytes(b"vocals")

    class SeparatorStub:
        def load_model(self, model_filename):
            assert model_filename == "2_HP-UVR.pth"

        def separate(self, source_path):
            assert source_path == str(audio_file)
            return [str(separated_background), str(separated_vocals)]

    separator_module = types.ModuleType("audio_separator.separator")
    separator_module.Separator = SeparatorStub
    package_module = types.ModuleType("audio_separator")
    package_module.separator = separator_module
    sys.modules["audio_separator"] = package_module
    sys.modules["audio_separator.separator"] = separator_module

    from dubbing.audio.audio_processor import AudioProcessor

    processor = AudioProcessor(
        CacheManager(use_cache=False, input_file=str(audio_file)),
        PerformanceTracker(),
        artifacts_root=str(tmp_path / "artifacts"),
    )

    background_audio, vocals_audio = processor.separate_background_and_vocals(str(audio_file))

    assert Path(background_audio).is_file()
    assert Path(vocals_audio).is_file()
    assert Path(background_audio).name == "background.wav"
    assert Path(vocals_audio).name == "vocals.wav"


def test_audio_processor_creates_directories_inside_custom_artifacts_root(tmp_path):
    artifacts_root = tmp_path / "Are" / "artifacts"

    from dubbing.audio.audio_processor import AudioProcessor

    AudioProcessor(
        CacheManager(use_cache=False, input_file=str(tmp_path / "clip.mp4")),
        PerformanceTracker(),
        artifacts_root=str(artifacts_root),
    )

    expected_dirs = [
        artifacts_root / "audio",
        artifacts_root / "speakers_audio",
        artifacts_root / "audio_chunks",
        artifacts_root / "su_audio_chunks",
    ]

    assert all(path.is_dir() for path in expected_dirs)


def test_run_ffmpeg_command_uses_safe_text_decoding(tmp_path, monkeypatch):
    from dubbing.audio.audio_processor import AudioProcessor

    audio_file = tmp_path / "clip.mp4"
    audio_file.write_bytes(b"fake")

    processor = AudioProcessor(
        CacheManager(use_cache=False, input_file=str(audio_file)),
        PerformanceTracker(),
    )

    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(stderr="")

    monkeypatch.setattr("dubbing.audio.audio_processor.subprocess.run", fake_run)

    processor._run_ffmpeg_command("ffmpeg -version")

    assert captured["args"] == ("ffmpeg -version",)
    assert captured["kwargs"]["text"] is True
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "replace"
