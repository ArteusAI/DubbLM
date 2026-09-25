"""Video download service using yt-dlp for URL-based sources."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import ParseResult, parse_qs, urlparse

from fastapi import HTTPException

from ..config import get_settings
from ..services.cookies_manager import get_effective_cookies_path
from ..services.project_manager import ProjectManager, sanitize_filename


# Quality presets map to yt-dlp format selectors.
# Chain: separate video-only + audio (needs ffmpeg; highest quality) ->
# merged container with audio -> plain best -> worst available.
# Muxed (video+audio) streams top out at 360p on YouTube, so a merged
# bestvideo+bestaudio must be tried before any progressive fallback —
# otherwise the selector matches 360p and never reaches higher resolutions.
QUALITY_FORMATS = {
    "best": "bestvideo+bestaudio/best[acodec!=none]/best/worst",
    "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080][acodec!=none]/best[height<=1080]/worst",
    "720p": "bestvideo[height<=720]+bestaudio/best[height<=720][acodec!=none]/best[height<=720]/worst",
    "480p": "bestvideo[height<=480]+bestaudio/best[height<=480][acodec!=none]/best[height<=480]/worst",
}

# Requested quality -> target height, used to warn when the delivered
# resolution is lower than asked for.
_TARGET_HEIGHTS = {"1080p": 1080, "720p": 720, "480p": 480}

# Containers that are kept as-is after download. Anything else is remuxed to mp4.
PRESERVED_CONTAINERS = {".mp4", ".webm", ".mkv", ".mov", ".m4v", ".avi"}

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}

YOUTUBE_PLAYLIST_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com"}

YOUTUBE_VIDEO_PATH_PREFIXES = ("shorts/", "embed/", "live/", "v/")


class VideoDownloadError(HTTPException):
    """Raised when a video download fails."""

    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail=detail)


@dataclass(frozen=True)
class DownloadedVideo:
    """Result of downloading a video from a URL."""

    upload_path: Path
    safe_filename: str
    file_size: int
    title: Optional[str]


ProgressCallback = Callable[[int, str], None]
LogCallback = Callable[[str, str], None]


def _is_playlist_only_url(parsed: ParseResult) -> bool:
    """Detect YouTube URLs that point to a playlist without a specific video."""
    host = (parsed.netloc or "").lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    if host not in YOUTUBE_PLAYLIST_HOSTS:
        return False

    if parse_qs(parsed.query).get("v"):
        return False

    path = parsed.path.strip("/")
    return not path.startswith(YOUTUBE_VIDEO_PATH_PREFIXES)


def validate_video_url(url: str) -> str:
    """Validate and normalize a video URL.

    Raises VideoDownloadError for invalid URLs.
    """
    url = url.strip()
    if not url:
        raise VideoDownloadError("URL is required")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise VideoDownloadError("Invalid URL format")

    if _is_playlist_only_url(parsed):
        raise VideoDownloadError(
            "Playlists are not supported. Please provide a link to a single video."
        )

    return url


def _sanitize_yt_title(title: str) -> str:
    """Sanitize a video title for use as filename, keeping basic readability."""
    title = sanitize_filename(title)
    # Remove characters that are problematic across filesystems
    title = re.sub(r'[<>:"/\\|?*]', "_", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title or "download"


def build_ytdlp_options(
    output_dir: Path,
    quality: str,
    cookies_path: Optional[Path] = None,
    max_filesize: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
    log_callback: Optional[LogCallback] = None,
    preserve_native_format: bool = True,
) -> dict:
    """Build yt-dlp options dictionary."""
    format_selector = QUALITY_FORMATS.get(quality, QUALITY_FORMATS["best"])

    options: dict = {
        "format": format_selector,
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "quiet": True,
        "no_warnings": False,
        # Never expand playlists: when the URL contains both a video and a
        # playlist, download only the referenced video.
        "noplaylist": True,
        "socket_timeout": 30,
    }

    if not preserve_native_format:
        # Force mp4 output for downstream compatibility.
        options["merge_output_format"] = "mp4"
        options["postprocessors"] = [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
        ]
    else:
        # Let yt-dlp preserve the native container. It will automatically invoke
        # FFmpegMerger when separate video+audio formats are selected.
        options["postprocessors"] = []

    if cookies_path and cookies_path.exists():
        options["cookiefile"] = str(cookies_path)

    if max_filesize:
        options["max_filesize"] = max_filesize

    if progress_callback is not None:

        def _progress_hook(info: dict) -> None:
            if info.get("status") == "downloading":
                total = info.get("total_bytes") or info.get("total_bytes_estimate") or 0
                downloaded = info.get("downloaded_bytes", 0)
                if total > 0:
                    percent = int((downloaded / total) * 100)
                else:
                    percent = 0
                speed = info.get("speed")
                eta = info.get("eta")
                speed_str = _format_speed(speed) if speed else None
                eta_str = _format_eta(eta) if eta is not None else None
                if speed_str and eta_str:
                    step = f"Downloading {speed_str} · ETA {eta_str}"
                elif speed_str:
                    step = f"Downloading {speed_str}"
                else:
                    step = "Downloading..."
                progress_callback(percent, step)
            elif info.get("status") == "finished":
                progress_callback(100, "Processing download...")

        options["progress_hooks"] = [_progress_hook]


    if log_callback is not None:

        class _YtDlpLogger:
            def debug(self, msg: str) -> None:
                # yt-dlp uses debug for verbose info; only log interesting lines
                if msg.startswith("[download]") or "format" in msg.lower():
                    log_callback(msg, "info")

            def info(self, msg: str) -> None:
                log_callback(msg, "info")

            def warning(self, msg: str) -> None:
                log_callback(msg, "warning")

            def error(self, msg: str) -> None:
                log_callback(msg, "error")

        options["logger"] = _YtDlpLogger()

    return options


def _format_speed(speed: Optional[float]) -> str:
    """Format download speed in human readable form."""
    if speed is None:
        return ""
    if speed >= 1024 * 1024:
        return f"{speed / (1024 * 1024):.1f} MiB/s"
    if speed >= 1024:
        return f"{speed / 1024:.1f} KiB/s"
    return f"{speed:.1f} B/s"


def _format_eta(eta: Optional[float]) -> str:
    """Format ETA in mm:ss or hh:mm:ss."""
    if eta is None or eta < 0:
        return ""
    total_seconds = int(eta)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _find_downloaded_file(output_dir: Path) -> Path:
    """Find the downloaded video file in the output directory."""
    candidates = [
        f for f in output_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS
    ]
    if not candidates:
        raise VideoDownloadError(
            "Download completed but no video file was found",
            status_code=500,
        )
    # Return the largest file in case of sidecar files
    return max(candidates, key=lambda f: f.stat().st_size)


def _validate_downloaded_file(path: Path, max_size: Optional[int]) -> None:
    """Validate downloaded file size and extension."""
    if max_size and path.stat().st_size > max_size:
        path.unlink(missing_ok=True)
        raise VideoDownloadError(
            f"Downloaded file exceeds maximum size of {max_size / (1024 ** 3):.1f}GB",
            status_code=413,
        )

    ext = "." + path.suffix.lstrip(".").lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        path.unlink(missing_ok=True)
        raise VideoDownloadError(
            f"Downloaded file type not supported. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}",
            status_code=400,
        )


def _has_audio_stream(path: Path) -> bool:
    """Check whether the file contains at least one audio stream."""
    import subprocess

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return "audio" in result.stdout.lower()
    except Exception:
        return False


def _convert_to_mp4(path: Path) -> Path:
    """Remux the given video to mp4 preserving codecs where possible."""
    import subprocess

    target = path.with_suffix(".mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(path),
        "-c", "copy",
        "-movflags", "+faststart",
        str(target),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as exc:
        raise VideoDownloadError(f"Failed to convert downloaded video to mp4: {exc}") from exc

    if result.returncode != 0 or not target.exists() or target.stat().st_size == 0:
        stderr_tail = (result.stderr or b"")[-1000:]
        target.unlink(missing_ok=True)
        raise VideoDownloadError(
            f"Failed to convert downloaded video to mp4: {stderr_tail.decode(errors='replace')}"
        )

    path.unlink(missing_ok=True)
    return target


def _perform_download(
    url: str,
    project_id: str,
    *,
    quality: str,
    cookies_path: Optional[Path],
    max_filesize: int,
    max_download_duration: Optional[int],
    progress_callback: Optional[ProgressCallback],
    log_callback: Optional[LogCallback],
) -> tuple[Path, Optional[str], int]:
    """Perform a single download attempt with the given quality.

    Returns (uploaded_file_path, title, file_size). Raises on failure.
    """
    pm = ProjectManager(project_id)
    pm.ensure_directories()

    options = build_ytdlp_options(
        output_dir=pm.uploads_dir,
        quality=quality,
        cookies_path=cookies_path,
        max_filesize=max_filesize,
        progress_callback=progress_callback,
        log_callback=log_callback,
        preserve_native_format=True,
    )

    import yt_dlp

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise VideoDownloadError("Could not retrieve video information")

        if info.get("_type") == "playlist":
            for f in pm.uploads_dir.iterdir():
                if f.is_file():
                    f.unlink()
            raise VideoDownloadError(
                "Playlists are not supported. Please provide a link to a single video."
            )

        title = info.get("title")
        duration = info.get("duration")
        height = info.get("height")
        format_id = info.get("format_id")

        if log_callback and height:
            log_callback(f"Downloaded format {format_id} ({height}p)", "info")
            target_height = _TARGET_HEIGHTS.get(quality)
            if target_height and height < target_height:
                log_callback(
                    f"Requested {quality} but only {height}p was available",
                    "warning",
                )

        if (
            duration
            and max_download_duration
            and duration > max_download_duration
        ):
            for f in pm.uploads_dir.iterdir():
                if f.is_file():
                    f.unlink()
            raise VideoDownloadError(
                f"Video duration ({duration}s) exceeds maximum allowed "
                f"({max_download_duration}s)",
                status_code=413,
            )

    uploaded_file = _find_downloaded_file(pm.uploads_dir)
    _validate_downloaded_file(uploaded_file, max_filesize)

    if not _has_audio_stream(uploaded_file):
        uploaded_file.unlink(missing_ok=True)
        raise VideoDownloadError(
            "Downloaded video has no audio stream",
            status_code=400,
        )

    # If yt-dlp produced an unsupported container, convert it to mp4 while
    # preserving common native formats (mp4/webm/mkv/etc.) untouched.
    ext = "." + uploaded_file.suffix.lstrip(".").lower()
    if ext not in PRESERVED_CONTAINERS:
        uploaded_file = _convert_to_mp4(uploaded_file)

    safe_name = _sanitize_yt_title(title) if title else "download"
    target_name = f"{safe_name}{uploaded_file.suffix}"
    target_path = pm.uploads_dir / target_name
    if uploaded_file.name != target_name:
        if target_path.exists() and target_path.resolve() != uploaded_file.resolve():
            target_name = f"{safe_name}_{uploaded_file.stem[-8:]}{uploaded_file.suffix}"
            target_path = pm.uploads_dir / target_name
        uploaded_file.rename(target_path)
        uploaded_file = target_path

    return uploaded_file, title, uploaded_file.stat().st_size


def download_video_from_url(
    url: str,
    project_id: str,
    *,
    quality: str = "best",
    progress_callback: Optional[ProgressCallback] = None,
    log_callback: Optional[LogCallback] = None,
) -> DownloadedVideo:
    """Download a video from a URL into the project's uploads directory.

    Args:
        url: Video URL supported by yt-dlp.
        project_id: Project ID to save the video under.
        quality: One of best, 1080p, 720p, 480p.
        progress_callback: Optional callback(percent, step_message).
        log_callback: Optional callback(message, log_type) for yt-dlp logs.

    Returns:
        DownloadedVideo with path, filename, size, and title.

    Raises:
        VideoDownloadError: For invalid URLs, geo/age restrictions, or failures.
    """
    url = validate_video_url(url)

    settings = get_settings()
    pm = ProjectManager(project_id)
    pm.ensure_directories()

    # Clear existing uploads to keep a single source video per project
    for existing_file in pm.uploads_dir.iterdir():
        if existing_file.is_file():
            existing_file.unlink()

    try:
        import yt_dlp
    except ImportError as exc:
        raise VideoDownloadError(
            "yt-dlp is not installed. Install it to use URL downloads.",
            status_code=500,
        ) from exc

    cookies_path = get_effective_cookies_path()

    last_error: Optional[str] = None
    qualities_to_try = [quality]
    if quality != "best":
        qualities_to_try.append("best")

    for attempt_quality in qualities_to_try:
        try:
            uploaded_file, title, file_size = _perform_download(
                url,
                project_id,
                quality=attempt_quality,
                cookies_path=cookies_path,
                max_filesize=settings.max_download_size,
                max_download_duration=settings.max_download_duration,
                progress_callback=progress_callback,
                log_callback=log_callback,
            )
            return DownloadedVideo(
                upload_path=uploaded_file,
                safe_filename=sanitize_filename(uploaded_file.name),
                file_size=file_size,
                title=title,
            )
        except yt_dlp.utils.DownloadError as exc:
            error_msg = str(exc)
            last_error = error_msg

            # Clean up partial downloads before retry
            for f in pm.uploads_dir.iterdir():
                if f.is_file():
                    f.unlink()

            is_format_error = "Requested format is not available" in error_msg
            if is_format_error and attempt_quality != "best" and len(qualities_to_try) > 1:
                if log_callback:
                    log_callback(f"Quality '{attempt_quality}' unavailable, trying best...", "warning")
                if progress_callback:
                    progress_callback(0, f"Quality '{attempt_quality}' unavailable, trying best...")
                continue

            detail = _map_download_error(error_msg, attempt_quality)
            raise VideoDownloadError(detail) from exc
        except VideoDownloadError:
            raise
        except Exception as exc:
            for f in pm.uploads_dir.iterdir():
                if f.is_file():
                    f.unlink()
            raise VideoDownloadError(f"Failed to download video: {exc}") from exc

    # Should not reach here, but guard just in case
    detail = _map_download_error(last_error or "Unknown error", quality)
    raise VideoDownloadError(detail)


def _map_download_error(error_msg: str, quality: str) -> str:
    """Map a yt-dlp DownloadError message to a user-friendly detail."""
    if "Private video" in error_msg:
        return "This video is private"
    if "age-restricted" in error_msg.lower() or "Age restriction" in error_msg:
        return "This video is age-restricted. Provide cookies to download it."
    if "geo" in error_msg.lower() or "not available in your country" in error_msg.lower():
        return "This video is not available in your region"
    if "Video unavailable" in error_msg:
        return "This video is unavailable"
    if "Requested format is not available" in error_msg:
        return f"Selected quality '{quality}' is not available for this video"
    if "Sign in to confirm" in error_msg:
        return "YouTube requires sign-in. Provide valid cookies to download."
    return f"Failed to download video: {error_msg}"


def get_video_info(url: str) -> dict:
    """Get video metadata without downloading.

    Useful for validating URLs and retrieving titles before starting a job.
    """
    url = validate_video_url(url)

    try:
        import yt_dlp
    except ImportError as exc:
        raise VideoDownloadError(
            "yt-dlp is not installed. Install it to use URL downloads.",
            status_code=500,
        ) from exc

    cookies_path = get_effective_cookies_path()
    options: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
    }
    if cookies_path and cookies_path.exists():
        options["cookiefile"] = str(cookies_path)

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise VideoDownloadError("Could not retrieve video information")
            if info.get("_type") == "playlist":
                raise VideoDownloadError(
                    "Playlists are not supported. Please provide a link to a single video."
                )
            return {
                "url": info.get("webpage_url") or url,
                "title": info.get("title"),
                "duration": info.get("duration"),
                "uploader": info.get("uploader"),
            }
    except VideoDownloadError:
        raise
    except yt_dlp.utils.DownloadError as exc:
        raise VideoDownloadError(f"Failed to retrieve video info: {exc}") from exc
    except Exception as exc:
        raise VideoDownloadError(f"Failed to retrieve video info: {exc}") from exc
