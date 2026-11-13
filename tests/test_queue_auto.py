"""Tests for queue-auto processing endpoint."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Stub the routes package to avoid importing heavy worker/task dependencies
if "src.api.routes" not in sys.modules:
    routes_pkg = types.ModuleType("src.api.routes")
    routes_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "api" / "routes")]
    sys.modules["src.api.routes"] = routes_pkg

# Stub heavy worker/task modules before importing process routes
celery_mod = types.ModuleType("src.api.workers.celery_app")
celery_mod.celery_app = SimpleNamespace(control=SimpleNamespace(revoke=lambda *args, **kwargs: None))
sys.modules["src.api.workers.celery_app"] = celery_mod

tasks_mod = types.ModuleType("src.api.workers.tasks")
tasks_mod.dub_project = lambda *args, **kwargs: None
tasks_mod.retranslate_project = lambda *args, **kwargs: None
sys.modules["src.api.workers.tasks"] = tasks_mod

from src.api.database.models import Project, ProjectStatus
from src.api.routes.process import QueueAutoResponse, queue_auto_process


class _FakeQuery:
    def __init__(self, session: "_FakeSession", model: object) -> None:
        self.session = session
        self.model = model

    def filter(self, *_args: object, **_kwargs: object) -> "_FakeQuery":
        return self

    def first(self) -> object | None:
        if self.model is Project:
            return self.session.project
        return None


class _FakeSession:
    def __init__(self, project: Project | None) -> None:
        self.project = project
        self.commits = 0

    def query(self, model: object) -> _FakeQuery:
        return _FakeQuery(self, model)

    def commit(self) -> None:
        self.commits += 1

    def expire_all(self) -> None:
        pass


def _run(coro):
    return asyncio.get_event_loop_policy().get_event_loop().run_until_complete(coro)


def test_queue_auto_for_downloading_project() -> None:
    project = Project(
        id="proj_test",
        name="Test",
        status=ProjectStatus.DOWNLOADING,
        config={},
    )
    session = _FakeSession(project)

    response = _run(queue_auto_process(project.id, db=session))

    assert isinstance(response, QueueAutoResponse)
    assert response.queued is True
    assert response.started is False
    assert project.config["autoProcess"] is True
    assert project.status == ProjectStatus.DOWNLOADING
    assert session.commits == 1


def test_queue_auto_starts_when_video_available(monkeypatch: object, tmp_path: Path) -> None:
    project = Project(
        id="proj_test",
        name="Test",
        status=ProjectStatus.DRAFT,
        config={},
    )
    session = _FakeSession(project)

    # Create a fake source video file
    pm_base = tmp_path / "proj_test"
    uploads = pm_base / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "video.mp4").write_text("fake video")

    fake_job = SimpleNamespace(jobId="job_transcribe_test")
    calls = []

    def fake_enqueue_transcription_job(db, proj, allow_existing=False):
        calls.append((proj.id, allow_existing))
        return fake_job

    import src.api.routes.process as process_module
    monkeypatch.setattr(process_module, "enqueue_transcription_job", fake_enqueue_transcription_job)
    monkeypatch.setattr(process_module, "ProjectManager", lambda pid: SimpleNamespace(get_source_video_path=lambda: uploads / "video.mp4"))

    response = _run(queue_auto_process(project.id, db=session))

    assert response.queued is False
    assert response.started is True
    assert response.jobId == "job_transcribe_test"
    assert project.config["autoProcess"] is True
    assert calls == [(project.id, False)]


def test_queue_auto_not_found() -> None:
    session = _FakeSession(None)

    with pytest.raises(Exception) as exc_info:
        _run(queue_auto_process("nonexistent", db=session))
    assert exc_info.value.status_code == 404


def test_stop_processing_clears_auto_process(monkeypatch: object) -> None:
    from src.api.routes.process import stop_processing
    from src.api.database.models import Job, JobStatus

    project = Project(
        id="proj_test",
        name="Test",
        status=ProjectStatus.DRAFT,
        config={"autoProcess": True},
    )
    session = _FakeSession(project)
    # No active jobs
    session.jobs = []

    class _FakeJobQuery:
        def filter(self, *_args, **_kwargs):
            return self
        def all(self):
            return session.jobs

    def fake_query(model: object) -> object:
        if model is Project:
            return _FakeQuery(session, model)
        if model is Job:
            return _FakeJobQuery()
        return _FakeQuery(session, model)

    session.query = fake_query

    response = _run(stop_processing(project.id, db=session))

    assert response["stopped"] is False
    assert project.config["autoProcess"] is False
    assert session.commits >= 1
