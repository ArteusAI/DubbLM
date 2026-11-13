"""Tests for external one-shot translate API."""

from __future__ import annotations

import asyncio
from io import BytesIO
from types import SimpleNamespace

from starlette.datastructures import UploadFile

from src.api.services.external_translate_service import (
    build_external_translate_config,
    start_external_translate,
)


class _FakeQuery:
    def __init__(self, session: "_FakeSession", model: object) -> None:
        self.session = session
        self.model = model

    def filter(self, *_args: object, **_kwargs: object) -> "_FakeQuery":
        return self

    def first(self) -> object | None:
        return None


class _FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.commits = 0
        self.refreshed: list[object] = []
        self._next_id = 1

    def add(self, obj: object) -> None:
        if getattr(obj, "id", None) is None:
            obj.id = f"proj_test_{self._next_id:03d}"
            self._next_id += 1
        self.added.append(obj)

    def commit(self) -> None:
        self.commits += 1

    def refresh(self, obj: object) -> None:
        self.refreshed.append(obj)


def test_build_external_translate_config_defaults() -> None:
    """Default external config uses auto source language and autoProcess."""
    config = build_external_translate_config("ru", "hq")

    assert config == {
        "targetLang": "ru",
        "sourceLang": "auto",
        "preset": "hq",
        "autoProcess": True,
        "keepBackground": False,
        "enableLlmEditor": False,
        "speakerCount": 1,
    }


def test_build_external_translate_config_minimal_merge() -> None:
    """Minimal diarization merge adds gap and repair overrides."""
    config = build_external_translate_config(
        "en",
        "fast",
        minimal_diarization_merge=True,
    )

    assert config["postDiarizationMergeGap"] == 0
    assert config["repairSpeakerFragmentation"] is False


def test_build_external_translate_config_speaker_count() -> None:
    """Speaker count is stored when provided."""
    config = build_external_translate_config("ru", "hq", speaker_count=2)

    assert config["speakerCount"] == 2


def test_build_external_translate_config_no_speaker_count_by_default() -> None:
    """Speaker count defaults to 1 for external API."""
    config = build_external_translate_config("ru", "hq")

    assert config["speakerCount"] == 1


def test_build_external_translate_config_enable_llm_editor() -> None:
    """LLM editor pass can be enabled explicitly."""
    config = build_external_translate_config("ru", "ultra", enable_llm_editor=True)

    assert config["enableLlmEditor"] is True


def test_build_external_translate_config_keep_background() -> None:
    """Background audio can be enabled explicitly."""
    config = build_external_translate_config("ru", "hq", keep_background=True)

    assert config["keepBackground"] is True


def test_build_external_translate_config_persona_id() -> None:
    """Persona id is stored when provided."""
    config = build_external_translate_config("ru", "hq", persona_id="tractorman")

    assert config["personaId"] == "tractorman"


def test_build_external_translate_config_no_persona_by_default() -> None:
    """Persona is omitted by default so preset defaults apply."""
    config = build_external_translate_config("ru", "hq")

    assert "personaId" not in config


def test_start_external_translate_response_urls(monkeypatch: object, tmp_path: object) -> None:
    """External translate returns poll and download URLs for the created project."""
    session = _FakeSession()
    upload = UploadFile(filename="demo.mp4", file=BytesIO(b"fake-video"))

    class FakeProjectManager:
        def __init__(self, project_id: str) -> None:
            self.project_id = project_id

        def ensure_directories(self) -> None:
            return None

    async def fake_save_project_video(project_id: str, file: UploadFile, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            upload_path=f"/tmp/{project_id}/demo.mp4",
            safe_filename="demo.mp4",
            file_size=10,
        )

    def fake_enqueue(db: object, project: object, *, allow_existing: bool) -> SimpleNamespace:
        assert allow_existing is False
        return SimpleNamespace(
            jobId="job_transcribe_test123",
            status="processing",
            projectId=project.id,
            type="transcribe",
            progress=0,
            currentStep="pending",
        )

    monkeypatch.setattr(
        "src.api.services.external_translate_service.ProjectManager",
        FakeProjectManager,
    )
    monkeypatch.setattr(
        "src.api.services.external_translate_service.save_project_video",
        fake_save_project_video,
    )
    monkeypatch.setattr(
        "src.api.services.external_translate_service.enqueue_transcription_job",
        fake_enqueue,
    )

    response = asyncio.run(
        start_external_translate(
            session,
            upload,
            "ru",
            preset="hq",
            minimal_diarization_merge=True,
            name="Screen Demo",
        )
    )

    project = session.added[0]
    assert project.name == "Screen Demo"
    assert project.config["targetLang"] == "ru"
    assert project.config["sourceLang"] == "auto"
    assert project.config["autoProcess"] is True
    assert project.config["keepBackground"] is False
    assert project.config["postDiarizationMergeGap"] == 0
    assert project.config["repairSpeakerFragmentation"] is False
    assert project.source_filename == "demo.mp4"
    assert response.projectId == project.id
    assert response.jobId == "job_transcribe_test123"
    assert response.status == "processing"
    assert response.pollUrl == f"/api/v1/projects/{project.id}/jobs/job_transcribe_test123/status"
    assert response.downloadUrl == f"/api/v1/projects/{project.id}/download/video"
