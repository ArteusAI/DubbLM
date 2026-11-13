from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.api.database.models import ProjectStatus
from src.api.routes import process


class _FakeQuery:
    def __init__(self, session: "_FakeSession", model: object) -> None:
        self.session = session
        self.model = model

    def filter(self, *_args: object, **_kwargs: object) -> "_FakeQuery":
        return self

    def first(self) -> object | None:
        if self.model is process.Project:
            return self.session.project
        return None

    def all(self) -> list[object]:
        if self.model is process.Job:
            return list(self.session.active_jobs)
        return []

    def delete(self) -> int:
        if self.model is process.Segment:
            self.session.segment_deletes += 1
            return 1
        return 0


class _FakeSession:
    def __init__(self, project: object) -> None:
        self.project = project
        self.active_jobs: list[object] = []
        self.added: list[object] = []
        self.commits = 0
        self.expired = False
        self.segment_deletes = 0

    def expire_all(self) -> None:
        self.expired = True

    def query(self, model: object) -> _FakeQuery:
        return _FakeQuery(self, model)

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def commit(self) -> None:
        self.commits += 1


def test_restart_processing_full_reset_clears_tts_cache(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    project_id = "proj_restart"
    source_video = tmp_path / "uploads" / "source.mp4"
    source_video.parent.mkdir(parents=True)
    source_video.write_bytes(b"video")

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    old_result = results_dir / "old.mp4"
    old_result.write_bytes(b"old")

    project = process.Project(
        id=project_id,
        name="Restart",
        status=ProjectStatus.DUBBED,
        config={
            "sourceLang": "en",
            "targetLang": "ru",
            "resultStats": {"stale": True},
        },
    )
    session = _FakeSession(project)
    cleanup_preserve_args: list[bool] = []

    class FakeProjectManager:
        def __init__(self, received_project_id: str) -> None:
            assert received_project_id == project_id
            self.results_dir = results_dir

        def get_source_video_path(self) -> Path:
            return source_video

        def cleanup_artifacts(self, preserve_tts_cache: bool = False) -> None:
            cleanup_preserve_args.append(preserve_tts_cache)

        def ensure_directories(self) -> None:
            results_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(process, "ProjectManager", FakeProjectManager)
    monkeypatch.setattr(process, "generate_job_id", lambda job_type: f"job_{job_type}_test")
    monkeypatch.setattr(
        process,
        "transcribe_project",
        SimpleNamespace(delay=lambda _project_id, _job_id: SimpleNamespace(id="task_restart")),
    )

    response = asyncio.run(process.restart_processing(project_id, db=session))

    assert response.jobId == "job_transcribe_test"
    assert cleanup_preserve_args == [False]
    assert session.segment_deletes == 1
    assert not old_result.exists()
    assert results_dir.exists()
    assert project.config["autoProcess"] is True
    assert "resultStats" not in project.config
