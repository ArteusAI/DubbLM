"""Tests for Chrome extension download route."""

from __future__ import annotations

import io
import json
import zipfile
import asyncio
from unittest.mock import MagicMock
from fastapi import Request
from src.api.routes import settings as settings_routes
from src.api.routes.settings import _external_base_url


def test_download_extension_endpoint(tmp_path, monkeypatch):
    # Setup mock extension directory
    mock_ext_dir = tmp_path / "extension"
    mock_ext_dir.mkdir()

    config_js = mock_ext_dir / "config.js"
    config_js.write_text(
        'const DEFAULT_CONFIG = {\n'
        '  serverUrl: "http://localhost:8000",\n'
        '  apiToken: ""\n'
        '};',
        encoding="utf-8"
    )

    other_file = mock_ext_dir / "manifest.json"
    other_file.write_text('{"name": "test"}', encoding="utf-8")

    # Point EXTENSION_DIR to our mock directory and inject fake environment variable
    original_dir = settings_routes.EXTENSION_DIR
    settings_routes.EXTENSION_DIR = mock_ext_dir
    monkeypatch.setenv("DUBBLM_API_TOKEN", "super-secure-token-123")

    try:
        # Mock Request
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.base_url = "http://my-external-api.com/"

        # Call the endpoint synchronously using asyncio.run
        async def run_endpoint():
            return await settings_routes.download_extension(mock_request)

        response = asyncio.run(run_endpoint())

        # Verify response headers
        assert response.media_type == "application/zip"
        assert "Content-Disposition" in response.headers
        assert "dubblm-extension.zip" in response.headers["Content-Disposition"]

        # Extract response content
        chunks = []
        async def extract_chunks():
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        asyncio.run(extract_chunks())
        content_bytes = b"".join(chunks)

        # Verify zip content
        zip_buffer = io.BytesIO(content_bytes)
        with zipfile.ZipFile(zip_buffer, "r") as zip_file:
            namelist = zip_file.namelist()
            assert "config.js" in namelist
            assert "manifest.json" in namelist

            # Verify config.js content has the injected API URL and Bearer token
            js_content = zip_file.read("config.js").decode("utf-8")
            assert 'serverUrl: "http://my-external-api.com"' in js_content
            assert 'apiToken: "super-secure-token-123"' in js_content
            assert 'serverUrl: "http://localhost:8000"' not in js_content

            # Verify manifest.json content is unchanged
            manifest_content = zip_file.read("manifest.json").decode("utf-8")
            assert '{"name": "test"}' in manifest_content

    finally:
        # Restore original directory
        settings_routes.EXTENSION_DIR = original_dir


def test_external_base_url_prefers_forwarded_host_and_port():
    request = MagicMock(spec=Request)
    request.headers = {
        "x-forwarded-host": "37.27.141.35:3007",
        "x-forwarded-proto": "http",
    }
    request.base_url = "http://api:8000/"

    assert _external_base_url(request) == "http://37.27.141.35:3007"


def test_external_base_url_falls_back_to_referer_port():
    request = MagicMock(spec=Request)
    request.headers = {"referer": "http://37.27.141.35:3007/project/proj_test/upload"}
    request.base_url = "http://api:8000/"

    assert _external_base_url(request) == "http://37.27.141.35:3007"


def test_extension_manifest_supports_remote_http_servers():
    extension_dir = settings_routes.EXTENSION_DIR
    manifest = json.loads((extension_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["version"] == "1.1"
    assert "http://*/*" in manifest["content_scripts"][0]["matches"]


def test_extension_uses_configured_server_for_cookie_sync():
    extension_dir = settings_routes.EXTENSION_DIR
    background = (extension_dir / "background.js").read_text(encoding="utf-8")
    content = (extension_dir / "content.js").read_text(encoding="utf-8")

    # The page event must not be able to choose an arbitrary upload server.
    assert "result.serverUrl" in background
    assert "request.serverUrl" not in background
    assert "event.detail" not in content
