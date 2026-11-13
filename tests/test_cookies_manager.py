"""Tests for cookies manager service."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.api.services.cookies_manager import (
    COOKIES_FILENAME,
    delete_cookies_file,
    get_cookies_dir,
    get_effective_cookies_path,
    get_global_cookies_path,
    has_cookies_file,
    save_cookies_file,
)


def test_get_cookies_dir_creates_directory(monkeypatch, tmp_path):
    settings_mock = SimpleNamespace(projects_dir=tmp_path)
    monkeypatch.setattr("src.api.services.cookies_manager.get_settings", lambda: settings_mock)

    cookies_dir = get_cookies_dir()
    assert cookies_dir == tmp_path / "cookies"
    assert cookies_dir.exists()


def test_save_and_retrieve_cookies(monkeypatch, tmp_path):
    settings_mock = SimpleNamespace(projects_dir=tmp_path, video_download_cookies_path=None)
    monkeypatch.setattr("src.api.services.cookies_manager.get_settings", lambda: settings_mock)

    content = b"# Netscape HTTP Cookie File\nyoutube.com\tTRUE\t/\tFALSE\t...\n"
    path = save_cookies_file(content)

    assert path.exists()
    assert path.read_bytes() == content
    assert has_cookies_file()

    effective = get_effective_cookies_path()
    assert effective == get_global_cookies_path()


def test_env_path_fallback(monkeypatch, tmp_path):
    env_cookies = tmp_path / "env_cookies.txt"
    env_cookies.write_bytes(b"env cookies")
    settings_mock = SimpleNamespace(projects_dir=tmp_path, video_download_cookies_path=env_cookies)
    monkeypatch.setattr("src.api.services.cookies_manager.get_settings", lambda: settings_mock)

    # Global cookies take priority
    global_path = get_global_cookies_path()
    global_path.write_bytes(b"global cookies")

    effective = get_effective_cookies_path()
    assert effective == global_path

    # Delete global, env path should be used
    delete_cookies_file()
    effective = get_effective_cookies_path()
    assert effective == env_cookies


def test_delete_cookies_file(monkeypatch, tmp_path):
    settings_mock = SimpleNamespace(projects_dir=tmp_path, video_download_cookies_path=None)
    monkeypatch.setattr("src.api.services.cookies_manager.get_settings", lambda: settings_mock)

    assert delete_cookies_file() is False

    save_cookies_file(b"cookies")
    assert has_cookies_file()
    assert delete_cookies_file() is True
    assert not has_cookies_file()
