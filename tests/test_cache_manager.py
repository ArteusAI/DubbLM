"""Regression tests for filesystem-safe cache keys."""

from __future__ import annotations

from pathlib import Path

from src.dubbing.core.cache_manager import CacheManager, _safe_cache_component


def test_safe_cache_component_replaces_provider_model_slash() -> None:
    key = "10ed7952_auto_ru_or1_microsoft/mai-transcribe-2_600_15_spk0"
    assert _safe_cache_component(key) == (
        "10ed7952_auto_ru_or1_microsoft_mai-transcribe-2_600_15_spk0"
    )


def test_pickle_cache_accepts_model_id_with_slash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "input.webm"
    input_file.write_bytes(b"x")

    manager = CacheManager(use_cache=True, input_file=str(input_file))
    key = "abc_or1_microsoft/mai-transcribe-2_600_15_spk0"

    manager.save_to_cache("openrouter_diarization_transcription", key, {"ok": True})

    assert manager.cache_exists("openrouter_diarization_transcription", key)
    assert manager.load_from_cache("openrouter_diarization_transcription", key) == {"ok": True}

    files = list((tmp_path / "cache").rglob("*.pkl"))
    assert len(files) == 1
    assert "microsoft" not in files[0].parts
    assert files[0].name == "abc_or1_microsoft_mai-transcribe-2_600_15_spk0.pkl"


def test_file_and_json_cache_accept_model_id_with_slash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "input.webm"
    input_file.write_bytes(b"x")
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")

    manager = CacheManager(use_cache=True, input_file=str(input_file))
    key = "abc_or1_microsoft/mai-transcribe-2_600_15_spk0"

    manager.save_file_to_cache("vocals_stem", key, str(source), f"{key}.wav")
    assert manager.load_file_from_cache(
        "vocals_stem", key, f"{key}.wav", str(tmp_path / "restored.wav")
    )
    assert (tmp_path / "restored.wav").read_bytes() == b"audio"

    path = manager.save_segments_json("segments", key, [{"text": "hello"}], filename_suffix="_editable")
    assert Path(path).exists()
    assert manager.load_segments_json("segments", key, "_editable")["segments"] == [
        {"text": "hello"}
    ]
    assert manager.segments_json_exists("segments", key, "_editable")
