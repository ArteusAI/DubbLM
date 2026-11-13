"""Tests for RFC 5987 / RFC 6266 Content-Disposition header formatting."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from urllib.parse import quote
import pytest

# Stub the routes package to avoid importing heavy worker/task dependencies
if "src.api.routes" not in sys.modules:
    routes_pkg = types.ModuleType("src.api.routes")
    routes_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "api" / "routes")]
    sys.modules["src.api.routes"] = routes_pkg

from src.api.services.object_storage import format_content_disposition
from src.api.routes.download import _stream_s3_response


def test_format_content_disposition_ascii() -> None:
    cd = format_content_disposition("inline", "sample_video.mp4")
    assert cd == 'inline; filename="sample_video.mp4"'
    # Must be valid Latin-1
    assert cd.encode("latin-1")


def test_format_content_disposition_unicode_special_slash() -> None:
    # Character U+29F8 (Big Solidus) used by yt-dlp
    filename = "The OpenAI⧸Hugging Face attack, clearly explained-(720p24)_ru.mp4"
    cd = format_content_disposition("inline", filename)

    # Must be valid Latin-1 (pure ASCII)
    encoded = cd.encode("latin-1")
    assert isinstance(encoded, bytes)

    # Must contain filename* with UTF-8 percent-encoding
    assert "filename*=utf-8''" in cd
    assert quote(filename, safe="") in cd

    # Fallback filename must be ASCII
    assert 'filename="The OpenAIHugging Face attack, clearly explained-(720p24)_ru.mp4"' in cd


def test_format_content_disposition_cyrillic() -> None:
    filename = "видео_ролик.mp4"
    cd = format_content_disposition("attachment", filename)

    # Must be valid Latin-1
    cd.encode("latin-1")
    assert "attachment;" in cd
    assert "filename*=utf-8''" in cd
    assert quote(filename, safe="") in cd


def test_format_content_disposition_quotes_sanitized() -> None:
    filename = 'video"name".mp4'
    cd = format_content_disposition("attachment", filename)
    cd.encode("latin-1")
    assert '"name"' not in cd
    assert "videoname.mp4" in cd


class _FakeObjectStorage:
    def get_object_size(self, key: str) -> int:
        return 1000

    def stream_range(self, key: str, start: int, end: int):
        yield b"x" * (end - start + 1)


def test_stream_s3_response_unicode_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _stream_s3_response initializes headers without UnicodeEncodeError."""
    from types import SimpleNamespace
    from src.api.services import object_storage as obj_module

    fake_store = _FakeObjectStorage()
    monkeypatch.setattr(obj_module, "get_object_storage", lambda: fake_store)

    unicode_filename = "The OpenAI⧸Hugging Face attack, clearly explained-(720p24)_ru.mp4"

    # Test full response (200 OK)
    resp = _stream_s3_response("some/s3/key.mp4", "video/mp4", unicode_filename, request=None, as_attachment=False)
    assert resp.status_code == 200
    cd = resp.headers.get("content-disposition")
    assert cd is not None
    assert cd.encode("latin-1")
    assert "filename*=utf-8''" in cd

    # Test range response (206 Partial Content)
    fake_request = SimpleNamespace(headers={"range": "bytes=0-100"})
    resp_range = _stream_s3_response("some/s3/key.mp4", "video/mp4", unicode_filename, request=fake_request, as_attachment=False)
    assert resp_range.status_code == 206
    cd_range = resp_range.headers.get("content-disposition")
    assert cd_range is not None
    assert cd_range.encode("latin-1")
    assert "filename*=utf-8''" in cd_range
