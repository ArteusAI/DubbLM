"""Tests for the video download service."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.api.services.video_download import (
    ALTERNATE_YOUTUBE_EXTRACTOR_ARGS,
    COMPATIBILITY_FORMAT,
    COMPATIBILITY_YOUTUBE_EXTRACTOR_ARGS,
    QUALITY_FORMATS,
    VideoDownloadError,
    _format_eta,
    _format_speed,
    _is_forbidden_error,
    _is_youtube_url,
    build_ytdlp_options,
    download_video_from_url,
    get_video_info,
    validate_video_url,
)


def test_validate_video_url_accepts_https() -> None:
    url = validate_video_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_validate_video_url_rejects_empty() -> None:
    with pytest.raises(VideoDownloadError) as exc_info:
        validate_video_url("   ")
    assert exc_info.value.status_code == 400
    assert "URL is required" in exc_info.value.detail


def test_validate_video_url_rejects_invalid_scheme() -> None:
    with pytest.raises(VideoDownloadError) as exc_info:
        validate_video_url("ftp://example.com/video")
    assert "Invalid URL format" in exc_info.value.detail


def test_validate_video_url_rejects_youtube_playlist_without_video() -> None:
    with pytest.raises(VideoDownloadError) as exc_info:
        validate_video_url(
            "https://www.youtube.com/playlist?list=PL9f5U-dvcNH7cwl4xK3xqm_DzgiXnyenf"
        )
    assert exc_info.value.status_code == 400
    assert "Playlists are not supported" in exc_info.value.detail


def test_validate_video_url_accepts_youtube_watch_with_playlist() -> None:
    url = (
        "https://www.youtube.com/watch?v=0kC3xOZChdA"
        "&list=PL9f5U-dvcNH7cwl4xK3xqm_DzgiXnyenf&index=81"
    )
    assert validate_video_url(url) == url


def test_validate_video_url_accepts_youtube_shorts_and_youtu_be() -> None:
    assert validate_video_url("https://www.youtube.com/shorts/abc123")
    assert validate_video_url("https://youtu.be/abc123")


def test_quality_format_presets() -> None:
    assert "1080" in QUALITY_FORMATS["1080p"]
    assert "720" in QUALITY_FORMATS["720p"]
    assert "480" in QUALITY_FORMATS["480p"]
    assert "acodec!=none" in QUALITY_FORMATS["best"]


def test_quality_formats_prefer_merged_streams() -> None:
    # Muxed (video+audio) streams top out at 360p on YouTube, so the separate
    # video+audio merge must be the first alternative for every preset.
    for quality, selector in QUALITY_FORMATS.items():
        first = selector.split("/")[0]
        assert first.startswith("bestvideo"), f"{quality} must try video-only first"
        assert "+" in first, f"{quality} must merge video+audio first"
    assert QUALITY_FORMATS["best"].startswith("bestvideo+bestaudio")


def test_build_ytdlp_options_default() -> None:
    options = build_ytdlp_options(Path("/tmp/downloads"), "best")
    assert options["format"] == QUALITY_FORMATS["best"]
    assert options["quiet"] is True
    # Never expand playlists; only download the referenced video.
    assert options["noplaylist"] is True
    assert options["socket_timeout"] == 30
    # Native container is preserved; unsupported containers are remuxed to mp4
    # by the service after download.
    assert "merge_output_format" not in options
    assert options["postprocessors"] == []
    assert "/tmp/downloads" in options["outtmpl"]


def test_build_ytdlp_options_force_mp4() -> None:
    options = build_ytdlp_options(
        Path("/tmp/downloads"), "best", preserve_native_format=False
    )
    assert options["merge_output_format"] == "mp4"
    assert options["postprocessors"] == [
        {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}
    ]


def test_build_ytdlp_options_with_cookies_and_max_size(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# cookies")
    options = build_ytdlp_options(Path("/tmp/downloads"), "720p", cookies_path=cookies, max_filesize=1024)
    assert options["format"] == QUALITY_FORMATS["720p"]
    assert options["cookiefile"] == str(cookies)
    assert options["max_filesize"] == 1024


def test_build_ytdlp_options_accepts_fallback_overrides() -> None:
    options = build_ytdlp_options(
        Path("/tmp/downloads"),
        "best",
        format_selector=COMPATIBILITY_FORMAT,
        extractor_args=COMPATIBILITY_YOUTUBE_EXTRACTOR_ARGS,
    )
    assert options["format"] == COMPATIBILITY_FORMAT
    assert options["extractor_args"] == COMPATIBILITY_YOUTUBE_EXTRACTOR_ARGS


def test_403_error_detection_and_youtube_url_detection() -> None:
    assert _is_forbidden_error(
        "ERROR: unable to download video data: HTTP Error 403: Forbidden"
    )
    assert not _is_forbidden_error("ERROR: Requested format is not available")
    assert _is_youtube_url("https://www.youtube.com/watch?v=abc123")
    assert _is_youtube_url("https://youtu.be/abc123")
    assert _is_youtube_url("https://www.youtube-nocookie.com/embed/abc123")
    assert not _is_youtube_url("https://vimeo.com/123456")


def test_build_ytdlp_options_progress_hook() -> None:
    calls: list[tuple[int, str]] = []

    def callback(percent: int, step: str) -> None:
        calls.append((percent, step))

    options = build_ytdlp_options(Path("/tmp/downloads"), "best", progress_callback=callback)
    hooks = options.get("progress_hooks", [])
    assert len(hooks) == 1

    hooks[0]({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
    assert calls == [(50, "Downloading...")]

    hooks[0]({
        "status": "downloading",
        "downloaded_bytes": 50,
        "total_bytes": 100,
        "speed": 2.5 * 1024 * 1024,
        "eta": 45,
    })
    assert calls[-1] == (50, "Downloading 2.5 MiB/s · ETA 0:45")

    hooks[0]({"status": "finished"})
    assert calls[-1] == (100, "Processing download...")


def test_format_speed_and_eta() -> None:
    assert _format_speed(2.5 * 1024 * 1024) == "2.5 MiB/s"
    assert _format_speed(512) == "512.0 B/s"
    assert _format_speed(2 * 1024) == "2.0 KiB/s"
    assert _format_speed(None) == ""

    assert _format_eta(45) == "0:45"
    assert _format_eta(125) == "2:05"
    assert _format_eta(3665) == "1:01:05"
    assert _format_eta(None) == ""


def test_get_video_info_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _mock_settings(monkeypatch, tmp_path)

    class FakeInfo:
        def get(self, key: str, default=None):
            data = {"title": "Test Video", "duration": 120, "uploader": "Tester", "webpage_url": "https://example.com/video"}
            return data.get(key, default)

    class FakeYDL:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            assert url == "https://example.com/video"
            return FakeInfo()

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)

    info = get_video_info("https://example.com/video")
    assert info["title"] == "Test Video"
    assert info["duration"] == 120
    assert info["uploader"] == "Tester"
    assert info["url"] == "https://example.com/video"


def test_get_video_info_uses_single_video_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mock_settings(monkeypatch, tmp_path)
    captured: dict = {}

    class FakeInfo:
        def get(self, key: str, default=None):
            data = {"title": "Test Video", "duration": 120, "webpage_url": "https://example.com/video"}
            return data.get(key, default)

    class FakeYDL:
        def __init__(self, options: dict):
            captured.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            return FakeInfo()

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)

    get_video_info("https://example.com/video")

    assert captured["noplaylist"] is True
    assert captured["socket_timeout"] == 30


def test_get_video_info_rejects_playlist_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mock_settings(monkeypatch, tmp_path)

    class FakeInfo:
        def get(self, key: str, default=None):
            return {"_type": "playlist"}.get(key, default)

    class FakeYDL:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            return FakeInfo()

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)

    with pytest.raises(VideoDownloadError) as exc_info:
        get_video_info("https://example.com/playlist")
    assert "Playlists are not supported" in exc_info.value.detail


def test_get_video_info_download_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _mock_settings(monkeypatch, tmp_path)

    class FakeYDL:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            raise Exception("Video unavailable")

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)

    with pytest.raises(VideoDownloadError) as exc_info:
        get_video_info("https://example.com/video")
    assert "Failed to retrieve video info" in exc_info.value.detail


def _mock_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings_mock = SimpleNamespace(
        projects_dir=tmp_path,
        max_download_size=8 * 1024 * 1024 * 1024,
        max_download_duration=None,
        video_download_cookies_path=None,
    )
    monkeypatch.setattr("src.api.services.video_download.get_settings", lambda: settings_mock)
    monkeypatch.setattr("src.api.services.project_manager.get_settings", lambda: settings_mock)
    monkeypatch.setattr("src.api.services.cookies_manager.get_settings", lambda: settings_mock)


def test_download_video_from_url_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_id = "proj_test_001"
    uploads_dir = tmp_path / project_id / "uploads"

    # Fake video bytes won't contain a real audio stream.
    monkeypatch.setattr(
        "src.api.services.video_download._has_audio_stream", lambda path: True
    )

    class FakeInfo:
        def get(self, key: str, default=None):
            data = {"title": "Test Video", "duration": 60}
            return data.get(key, default)

    class FakeYDL:
        def __init__(self, options: dict):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            assert url == "https://example.com/video"
            # Simulate yt-dlp writing the file
            uploads_dir.mkdir(parents=True, exist_ok=True)
            fake_video = uploads_dir / "Test Video.mp4"
            fake_video.write_bytes(b"fake video content")
            return FakeInfo()

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)

    _mock_settings(monkeypatch, tmp_path)

    progress_calls: list[tuple[int, str]] = []

    result = download_video_from_url(
        "https://example.com/video",
        project_id,
        quality="720p",
        progress_callback=lambda p, s: progress_calls.append((p, s)),
    )

    assert result.safe_filename == "Test Video.mp4"
    assert result.file_size == len(b"fake video content")
    assert result.title == "Test Video"
    assert result.upload_path.exists()


def test_download_video_from_url_logs_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_id = "proj_test_resolution"
    uploads_dir = tmp_path / project_id / "uploads"

    monkeypatch.setattr(
        "src.api.services.video_download._has_audio_stream", lambda path: True
    )

    class FakeInfo:
        def get(self, key: str, default=None):
            data = {
                "title": "Low Res Video",
                "duration": 60,
                "height": 360,
                "format_id": "18",
            }
            return data.get(key, default)

    class FakeYDL:
        def __init__(self, options: dict):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            uploads_dir.mkdir(parents=True, exist_ok=True)
            (uploads_dir / "Low Res Video.mp4").write_bytes(b"fake video content")
            return FakeInfo()

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)
    _mock_settings(monkeypatch, tmp_path)

    logs: list[tuple[str, str]] = []
    download_video_from_url(
        "https://example.com/video",
        project_id,
        quality="1080p",
        log_callback=lambda msg, level: logs.append((msg, level)),
    )

    assert ("Downloaded format 18 (360p)", "info") in logs
    assert (
        "Requested 1080p but only 360p was available",
        "warning",
    ) in logs


def test_download_video_from_url_duration_exceeded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_id = "proj_test_002"
    uploads_dir = tmp_path / project_id / "uploads"
    uploads_dir.mkdir(parents=True)

    fake_video = uploads_dir / "Long Video.mp4"
    fake_video.write_bytes(b"fake")

    class FakeInfo:
        def get(self, key: str, default=None):
            data = {"title": "Long Video", "duration": 600}
            return data.get(key, default)

    class FakeYDL:
        def __init__(self, options: dict):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            return FakeInfo()

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)

    _mock_settings(monkeypatch, tmp_path)
    # Override max_download_duration for this test
    from src.api.services import video_download as vd_mod
    from src.api.services import project_manager as pm_mod
    duration_settings = SimpleNamespace(
        projects_dir=tmp_path,
        max_download_size=8 * 1024 * 1024 * 1024,
        max_download_duration=300,
        video_download_cookies_path=None,
    )
    monkeypatch.setattr(vd_mod, "get_settings", lambda: duration_settings)
    monkeypatch.setattr(pm_mod, "get_settings", lambda: duration_settings)

    with pytest.raises(VideoDownloadError) as exc_info:
        download_video_from_url("https://example.com/video", project_id)

    assert "duration" in exc_info.value.detail
    assert "exceeds maximum" in exc_info.value.detail
    assert not fake_video.exists()


def test_download_video_from_url_private_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_id = "proj_test_003"
    uploads_dir = tmp_path / project_id / "uploads"
    uploads_dir.mkdir(parents=True)

    class FakeYDL:
        def __init__(self, options: dict):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            raise Exception("Private video")

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)

    _mock_settings(monkeypatch, tmp_path)

    with pytest.raises(VideoDownloadError) as exc_info:
        download_video_from_url("https://example.com/private", project_id)

    assert "private" in exc_info.value.detail.lower()


def test_download_video_from_url_retries_with_best_quality_on_format_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If selected quality is unavailable, service should retry with 'best'."""
    import yt_dlp

    project_id = "proj_test_fallback"
    _mock_settings(monkeypatch, tmp_path)

    # Fake video bytes won't contain a real audio stream.
    monkeypatch.setattr(
        "src.api.services.video_download._has_audio_stream", lambda path: True
    )

    attempts: list[str] = []

    class FakeInfo:
        def get(self, key: str, default=None):
            return {"title": "Fallback Video", "duration": 60}.get(key, default)

    class FakeYDL:
        def __init__(self, options: dict):
            self.options = options
            # The format selector for 1080p now starts with a merged
            # bestvideo[height<=1080]+bestaudio alternative.
            self.quality = options.get("format", "best").split("/")[0]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            attempts.append(self.quality)
            if "1080" in self.quality:
                raise yt_dlp.utils.DownloadError("Requested format is not available")
            # Simulate writing the file on successful fallback
            uploads_dir = tmp_path / project_id / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            (uploads_dir / "Fallback Video.mp4").write_bytes(b"video")
            return FakeInfo()

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)

    download_video_from_url("https://example.com/video", project_id, quality="1080p")

    assert attempts[0].startswith("bestvideo[height<=1080]")
    assert attempts[-1].startswith("bestvideo+bestaudio")


def test_download_video_from_url_retries_403_with_compatibility_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rejected media URL should retry with alternate clients, then 360p."""
    import yt_dlp

    project_id = "proj_test_403_retry"
    uploads_dir = tmp_path / project_id / "uploads"
    _mock_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "src.api.services.video_download._has_audio_stream", lambda path: True
    )

    attempts: list[dict] = []

    class FakeInfo:
        def get(self, key: str, default=None):
            data = {
                "title": "Retry Video",
                "duration": 60,
                "height": 360,
                "format_id": "18",
            }
            return data.get(key, default)

    class FakeYDL:
        def __init__(self, options: dict):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            attempts.append({
                "format": self.options.get("format"),
                "extractor_args": self.options.get("extractor_args"),
            })
            if len(attempts) < 3:
                raise yt_dlp.utils.DownloadError(
                    "ERROR: unable to download video data: HTTP Error 403: Forbidden"
                )
            uploads_dir.mkdir(parents=True, exist_ok=True)
            (uploads_dir / "Retry Video.mp4").write_bytes(b"video")
            return FakeInfo()

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)

    logs: list[tuple[str, str]] = []
    result = download_video_from_url(
        "https://www.youtube.com/watch?v=abc123",
        project_id,
        quality="best",
        log_callback=lambda message, level: logs.append((message, level)),
    )

    assert len(attempts) == 3
    assert attempts[0]["format"] == QUALITY_FORMATS["best"]
    assert attempts[0]["extractor_args"] is None
    assert attempts[1]["extractor_args"] == ALTERNATE_YOUTUBE_EXTRACTOR_ARGS
    assert attempts[2]["format"] == COMPATIBILITY_FORMAT
    assert attempts[2]["extractor_args"] == COMPATIBILITY_YOUTUBE_EXTRACTOR_ARGS
    assert result.safe_filename == "Retry Video.mp4"
    assert any("403" in message for message, level in logs if level == "warning")


def test_download_video_from_url_403_returns_friendly_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import yt_dlp

    project_id = "proj_test_403_error"
    _mock_settings(monkeypatch, tmp_path)

    class FakeYDL:
        def __init__(self, options: dict):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            raise yt_dlp.utils.DownloadError(
                "ERROR: unable to download video data: HTTP Error 403: Forbidden"
            )

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)

    with pytest.raises(VideoDownloadError) as exc_info:
        download_video_from_url(
            "https://www.youtube.com/watch?v=abc123", project_id, quality="best"
        )

    assert "HTTP 403" in exc_info.value.detail
    assert "denied" in exc_info.value.detail.lower()
    assert "unable to download video data" not in exc_info.value.detail


def test_download_video_from_url_non_youtube_403_retries_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import yt_dlp

    project_id = "proj_test_403_non_youtube"
    uploads_dir = tmp_path / project_id / "uploads"
    _mock_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "src.api.services.video_download._has_audio_stream", lambda path: True
    )

    attempts: list[dict] = []

    class FakeInfo:
        def get(self, key: str, default=None):
            return {"title": "Other Site Video", "duration": 60}.get(key, default)

    class FakeYDL:
        def __init__(self, options: dict):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            attempts.append({"extractor_args": self.options.get("extractor_args")})
            if len(attempts) == 1:
                raise yt_dlp.utils.DownloadError(
                    "ERROR: unable to download video data: HTTP Error 403: Forbidden"
                )
            uploads_dir.mkdir(parents=True, exist_ok=True)
            (uploads_dir / "Other Site Video.mp4").write_bytes(b"video")
            return FakeInfo()

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)

    result = download_video_from_url(
        "https://example.com/video", project_id, quality="best"
    )

    assert result.safe_filename == "Other Site Video.mp4"
    assert len(attempts) == 2
    assert attempts[0]["extractor_args"] is None
    assert attempts[1]["extractor_args"] is None


def test_download_video_from_url_no_audio_stream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If the downloaded file has no audio stream, raise a clear error."""
    project_id = "proj_test_no_audio"
    _mock_settings(monkeypatch, tmp_path)

    monkeypatch.setattr(
        "src.api.services.video_download._has_audio_stream", lambda path: False
    )

    class FakeInfo:
        def get(self, key: str, default=None):
            return {"title": "Silent Video", "duration": 60}.get(key, default)

    class FakeYDL:
        def __init__(self, options: dict):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url: str, *, download: bool):
            uploads_dir = tmp_path / project_id / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            (uploads_dir / "Silent Video.mp4").write_bytes(b"no audio")
            return FakeInfo()

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)

    with pytest.raises(VideoDownloadError) as exc_info:
        download_video_from_url("https://example.com/video", project_id)

    assert "no audio stream" in exc_info.value.detail.lower()
    # The invalid file should be cleaned up.
    assert not any((tmp_path / project_id / "uploads").iterdir())


def test_download_video_from_url_yt_dlp_not_installed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_id = "proj_test_004"

    _mock_settings(monkeypatch, tmp_path)

    # Simulate yt_dlp not being importable by removing it from the test module's namespace
    import sys
    real_yt_dlp = sys.modules.get("yt_dlp")
    sys.modules["yt_dlp"] = None  # type: ignore[assignment]

    try:
        with pytest.raises(VideoDownloadError) as exc_info:
            download_video_from_url("https://example.com/video", project_id)
        assert "yt-dlp is not installed" in exc_info.value.detail
    finally:
        if real_yt_dlp is not None:
            sys.modules["yt_dlp"] = real_yt_dlp
        else:
            sys.modules.pop("yt_dlp", None)
