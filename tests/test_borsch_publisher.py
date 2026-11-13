from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.api.services import borsch_publisher


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = ""

    def json(self) -> dict:
        return self._payload


def test_parse_srt_builds_timed_segments(tmp_path: Path) -> None:
    path = tmp_path / "sample_ru.srt"
    path.write_text(
        "1\n"
        "00:00:01,500 --> 00:00:03,000\n"
        "Привет, [short pause] мир.\n\n"
        "2\n"
        "00:00:04,250 --> 00:00:06,000\n"
        "Второй сегмент.\n",
        encoding="utf-8",
    )

    segments = borsch_publisher._parse_srt(path)

    assert segments == [
        {"start_s": 1.5, "end_s": 3.0, "text": "Привет, мир."},
        {"start_s": 4.25, "end_s": 6.0, "text": "Второй сегмент."},
    ]


def test_build_publish_transcript_falls_back_to_db() -> None:
    project = SimpleNamespace(config={"targetLang": "ru"})
    rows = [
        {"start_s": 0.0, "end_s": 2.0, "text": "Привет", "speaker": "SPEAKER_A"},
        {"start_s": 2.0, "end_s": 4.0, "text": "Мир", "speaker": "SPEAKER_B"},
    ]

    with (
        patch.object(borsch_publisher, "get_object_storage", side_effect=RuntimeError("no s3")),
        patch.object(borsch_publisher, "_segments_from_db", return_value=rows),
    ):
        payload = borsch_publisher._build_publish_transcript("proj_test", project)

    assert payload["transcript_language"] == "ru"
    assert payload["transcript"] == "SPEAKER_A: Привет\nSPEAKER_B: Мир"
    assert payload["transcript_segments"] == [
        {"start_s": 0.0, "end_s": 2.0, "text": "Привет"},
        {"start_s": 2.0, "end_s": 4.0, "text": "Мир"},
    ]


def test_publish_payload_sends_only_video_metadata_and_transcript() -> None:
    project = SimpleNamespace(
        name="Andrew Ng",
        config={"targetLang": "ru", "downloadUrl": "https://example.com/video"},
    )
    transcript = {
        "transcript": "Привет",
        "transcript_language": "ru",
        "transcript_segments": [{"start_s": 0.0, "end_s": 1.0, "text": "Привет"}],
    }

    payload = borsch_publisher._publish_payload(project, transcript)

    assert payload["auto_metadata"] is True
    assert payload["metadata_language"] == "ru"
    assert payload["transcript"] == "Привет"
    assert payload["transcript_segments"]
    assert payload["source_url"] == "https://example.com/video"
    assert "summary" not in payload
    assert "why_hn" not in payload
    assert "tags" not in payload
    assert "category_ids" not in payload


def test_register_in_borsch_payload_contains_transcript() -> None:
    project = SimpleNamespace(name="Test", config={"targetLang": "ru"})
    transcript = {
        "transcript": "Привет",
        "transcript_language": "ru",
        "transcript_segments": [{"start_s": 0.0, "end_s": 1.0, "text": "Привет"}],
    }
    captured: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(
            {
                "vid": "vid12345",
                "status": "processing",
                "auto_metadata": True,
                "metadata_status": "pending",
            }
        )

    with patch.object(borsch_publisher.requests, "post", side_effect=fake_post):
        result = borsch_publisher._register_in_borsch(
            "vid12345",
            "videos/vid12345/source.mp4",
            project,
            transcript,
            "token",
        )

    assert result["auto_metadata"] is True
    assert captured["json"]["source_key"] == "videos/vid12345/source.mp4"
    assert captured["json"]["transcript"] == "Привет"
    assert captured["json"]["transcript_segments"][0]["start_s"] == 0.0
    assert "summary" not in captured["json"]


def test_register_rejects_old_borsch_without_remote_metadata() -> None:
    project = SimpleNamespace(name="Test", config={"targetLang": "ru"})
    transcript = {
        "transcript": "Привет",
        "transcript_language": "ru",
        "transcript_segments": [],
    }

    with patch.object(
        borsch_publisher.requests,
        "post",
        return_value=_FakeResponse({"vid": "vid12345", "status": "processing"}),
    ):
        with pytest.raises(borsch_publisher.BorschPublishError) as context:
            borsch_publisher._register_in_borsch(
                "vid12345",
                "videos/vid12345/source.mp4",
                project,
                transcript,
                "token",
            )

    assert context.value.code == "borsch_metadata_not_supported"
