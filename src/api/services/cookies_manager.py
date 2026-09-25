"""Cookies file management for video download authentication."""

from pathlib import Path
from typing import Optional

from ..config import get_settings

COOKIES_FILENAME = "cookies.txt"
COOKIES_SUBDIR = "cookies"


def get_cookies_dir() -> Path:
    """Get the directory where cookies files are stored."""
    settings = get_settings()
    cookies_dir = settings.projects_dir / COOKIES_SUBDIR
    cookies_dir.mkdir(parents=True, exist_ok=True)
    return cookies_dir


def get_global_cookies_path() -> Path:
    """Get the path for the global cookies.txt file."""
    return get_cookies_dir() / COOKIES_FILENAME


def get_effective_cookies_path() -> Optional[Path]:
    """Return the effective cookies path to use for yt-dlp.

    Priority:
    1. Globally uploaded cookies file (via UI/API).
    2. Path from VIDEO_DOWNLOAD_COOKIES_PATH env variable.
    """
    global_path = get_global_cookies_path()
    if global_path.exists():
        return global_path

    settings = get_settings()
    env_path = settings.video_download_cookies_path
    if env_path and env_path.exists():
        return env_path

    return None


def save_cookies_file(content: bytes) -> Path:
    """Save uploaded cookies file to global cookies path.

    Args:
        content: Raw bytes of the cookies.txt file.

    Returns:
        Path where cookies were saved.
    """
    cookies_path = get_global_cookies_path()
    cookies_path.write_bytes(content)
    return cookies_path


def delete_cookies_file() -> bool:
    """Delete the globally uploaded cookies file if it exists."""
    cookies_path = get_global_cookies_path()
    if cookies_path.exists():
        cookies_path.unlink()
        return True
    return False


def has_cookies_file() -> bool:
    """Check whether a cookies file is available for downloads."""
    return get_effective_cookies_path() is not None
