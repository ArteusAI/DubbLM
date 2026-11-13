from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from src.api.workers import tasks

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


class StopAfterExtractAudio(Exception):
    """Stop the task once extract_audio arguments are captured."""


class FakeQuery:
    """Minimal query stub for the task's DB reads."""

    def __init__(self, model: object, project: object, segments: list[object]) -> None:
        self.model = model
        self.project = project
        self.segments = segments

    def filter(self, *_args, **_kwargs) -> "FakeQuery":
        """Ignore SQLAlchemy filters in the test stub."""
        return self

    def order_by(self, *_args, **_kwargs) -> "FakeQuery":
        """Ignore SQLAlchemy ordering in the test stub."""
        return self

    def first(self) -> object | None:
        """Return the fake project for project lookups."""
        if self.model is tasks.Project:
            return self.project
        return None

    def all(self) -> list[object]:
        """Return fake segments for segment lookups."""
        if self.model is tasks.Segment:
            return self.segments
        return []


class FakeSession:
    """Minimal DB session stub for the task's read-only setup."""

    def __init__(self, project: object, segments: list[object]) -> None:
        self.project = project
        self.segments = segments

    def query(self, model: object) -> FakeQuery:
        """Return a model-specific fake query object."""
        return FakeQuery(model, self.project, self.segments)

    def close(self) -> None:
        """Match the SQLAlchemy session API."""


def test_worker_resolves_refinement_persona_and_llm_adjustment_from_preset() -> None:
    preset_config = {
        "persona_id": "none",
        "enable_llm_text_adjustment": False,
    }

    assert tasks._resolve_refinement_persona({}, preset_config) == "none"
    assert tasks._resolve_refinement_persona({"personaId": "normal"}, preset_config) == "normal"
    assert tasks._resolve_bool_config(
        {},
        "enableLlmTextAdjustment",
        preset_config,
        "enable_llm_text_adjustment",
        True,
    ) is False
    assert tasks._resolve_bool_config(
        {"enableLlmTextAdjustment": True},
        "enableLlmTextAdjustment",
        preset_config,
        "enable_llm_text_adjustment",
        True,
    ) is True


def test_dub_project_passes_clip_duration_to_extract_audio(
    monkeypatch: "MonkeyPatch",
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    source_path = tmp_path / "uploads" / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"video")

    fake_project = SimpleNamespace(
        config={
            "sourceLang": "en",
            "targetLang": "ru",
            "preset": "hq",
            "duration": 300.0,
            "keepBackground": True,
        }
    )
    fake_segment = SimpleNamespace(
        speaker="SPEAKER_A",
        start_time=0.0,
        end_time=5.0,
        original_text="Hello",
        translated_text="Privet",
        voice_id=None,
        provider=None,
        tts_prompt=None,
        is_muted=False,
    )

    class FakeProjectManager:
        """Project manager stub that keeps all paths inside tmp_path."""

        def __init__(self, _project_id: str) -> None:
            self.base_dir = tmp_path
            self.debug_dir = tmp_path / "artifacts" / "debug"

        def ensure_directories(self) -> None:
            """Create directories expected by the task."""
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            (tmp_path / "results").mkdir(parents=True, exist_ok=True)

        def get_source_video_path(self) -> Path:
            """Return the fake uploaded video."""
            return source_path

        def get_result_video_path(self, target_lang: str) -> Path:
            """Return a deterministic output path."""
            return tmp_path / "results" / f"result_{target_lang}.mp4"

    class FakeAudioProcessor:
        """Capture extract_audio arguments and stop the task."""

        def extract_audio(
            self,
            video_path: str,
            start_time: float | None = None,
            duration: float | None = None,
        ) -> str:
            captured["extract_audio"] = (video_path, start_time, duration)
            raise StopAfterExtractAudio()

    class FakeSmartDubbing:
        """SmartDubbing stub that exposes only the audio processor."""

        def __init__(self, config: object) -> None:
            captured["config_start_time"] = config.get("start_time")
            captured["config_duration"] = config.get("duration")
            self.cost_tracker = object()
            self.performance_tracker = SimpleNamespace(start_timing=lambda *_args, **_kwargs: None)
            self.audio_processor = FakeAudioProcessor()

    monkeypatch.setattr(tasks, "update_job_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks, "update_project_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks, "update_job_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks, "_apply_api_keys", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks, "ProjectManager", FakeProjectManager)
    monkeypatch.setattr(tasks, "get_preset_config", lambda _preset: {
        "tts_system": "gemini",
        "llm_provider": "gemini",
        "llm_model_name": "gemini-2.5-pro",
        "llm_temperature": 0.5,
    })
    monkeypatch.setattr(tasks, "resolve_tts_prompt_prefix", lambda **kwargs: None)
    monkeypatch.setattr(tasks, "get_blocked_voices_for_style", lambda **kwargs: [])
    monkeypatch.setattr(tasks, "get_video_quality_preset", lambda *_args, **_kwargs: "original")
    monkeypatch.setattr(tasks, "import_cost_snapshots", lambda *args, **kwargs: 0)
    monkeypatch.setattr(tasks, "_resolve_editor_runtime_config", lambda *_args, **_kwargs: {
        "enable_llm_editor": False,
        "editor_llm_provider": None,
        "editor_model_name": None,
        "editor_temperature": 1.0,
        "editor_reasoning_effort": "none",
    })
    monkeypatch.setattr(
        tasks,
        "get_db_session",
        lambda fresh=False: FakeSession(fake_project, [fake_segment]),
    )
    monkeypatch.setattr("src.dubbing.core.smart_dubbing.SmartDubbing", FakeSmartDubbing)
    monkeypatch.setattr("src.dubbing.core.log_config.setup_logging", lambda **kwargs: None)

    with pytest.raises(StopAfterExtractAudio):
        tasks.dub_project.run("proj_3070822cf168", "job_1")

    assert captured["config_start_time"] is None
    assert captured["config_duration"] == 300.0
    assert captured["extract_audio"] == (str(source_path), None, 300.0)
