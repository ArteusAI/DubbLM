"""Tests for the video download URL route."""

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

from src.api.database.models import Project, ProjectStatus
from src.api.routes.download import (
    VideoDownloadRequest,
    download_video_from_url,
    fetch_video_info,
)


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
    def __init__(self, project: Project) -> None:
        self.project = project
        self.commits = 0

    def query(self, model: object) -> _FakeQuery:
        return _FakeQuery(self, model)

    def commit(self) -> None:
        self.commits += 1


def _run(coro):
    return asyncio.get_event_loop_policy().get_event_loop().run_until_complete(coro)


def test_download_url_invalid_project() -> None:
    session = _FakeSession(None)

    with pytest.raises(Exception) as exc_info:
        _run(
            download_video_from_url(
                "nonexistent",
                VideoDownloadRequest(url="https://example.com/video", quality="best"),
                db=session,
            )
        )
    assert exc_info.value.status_code == 404


def test_download_url_invalid_url_format() -> None:
    project = Project(id="proj_test", name="URL Test", status=ProjectStatus.DRAFT)
    session = _FakeSession(project)

    with pytest.raises(Exception) as exc_info:
        _run(
            download_video_from_url(
                project.id,
                VideoDownloadRequest(url="not-a-url", quality="best"),
                db=session,
            )
        )
    assert exc_info.value.status_code == 400
    assert "Invalid URL format" in exc_info.value.detail


def test_download_url_rejects_non_draft_status() -> None:
    project = Project(id="proj_test", name="URL Test", status=ProjectStatus.TRANSCRIBED)
    session = _FakeSession(project)

    with pytest.raises(Exception) as exc_info:
        _run(
            download_video_from_url(
                project.id,
                VideoDownloadRequest(url="https://example.com/video", quality="best"),
                db=session,
            )
        )
    assert exc_info.value.status_code == 400
    assert "Cannot start download" in exc_info.value.detail


def test_download_url_starts_job(monkeypatch: object) -> None:
    project = Project(id="proj_test", name="URL Test", status=ProjectStatus.DRAFT)
    session = _FakeSession(project)

    called = {}

    def fake_enqueue_download_job(db, project, url, quality):
        called["project_id"] = project.id
        called["url"] = url
        called["quality"] = quality
        from src.api.models.schemas import JobResponse
        return JobResponse(
            jobId="job_download_test",
            status="processing",
            projectId=project.id,
            type="download",
            progress=0,
            currentStep="pending",
        )

    import src.api.routes.download as download_module
    monkeypatch.setattr(download_module, "enqueue_download_job", fake_enqueue_download_job)

    response = _run(
        download_video_from_url(
            project.id,
            VideoDownloadRequest(url="https://example.com/video", quality="720p"),
            db=session,
        )
    )

    assert response.jobId == "job_download_test"
    assert response.projectId == project.id
    assert response.status == "processing"
    assert called["url"] == "https://example.com/video"
    assert called["quality"] == "720p"
    assert called["project_id"] == project.id
    assert session.commits == 1


def test_fetch_video_info_returns_metadata(monkeypatch: object) -> None:
    def fake_get_video_info(url: str):
        assert url == "https://example.com/video"
        return {
            "url": url,
            "title": "Test Video Title",
            "duration": 123,
            "uploader": "Test Uploader",
        }

    import src.api.routes.download as download_module
    monkeypatch.setattr(download_module, "get_video_info", fake_get_video_info)

    response = _run(
        fetch_video_info(VideoDownloadRequest(url="https://example.com/video", quality="best"))
    )

    assert response.title == "Test Video Title"
    assert response.duration == 123
    assert response.uploader == "Test Uploader"
    assert response.url == "https://example.com/video"


def test_fetch_video_info_propagates_validation_error(monkeypatch: object) -> None:
    from src.api.services.video_download import VideoDownloadError

    def fake_get_video_info(_url: str):
        raise VideoDownloadError("Invalid URL format", status_code=400)

    import src.api.routes.download as download_module
    monkeypatch.setattr(download_module, "get_video_info", fake_get_video_info)

    with pytest.raises(Exception) as exc_info:
        _run(
            fetch_video_info(VideoDownloadRequest(url="not-a-url", quality="best"))
        )
    assert exc_info.value.status_code == 400
    assert "Invalid URL format" in exc_info.value.detail
