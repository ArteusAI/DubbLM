"""Video probing and project size utilities.

Wraps ffprobe to read video resolution/duration and provides helpers to
compute the on-disk size of a project's directory tree.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from .project_manager import ProjectManager

logger = logging.getLogger(__name__)


def probe_video(video_path: Path | str) -> dict:
    """Probe a video file with ffprobe.

    Returns a dict with keys ``width``, ``height``, ``duration``. Values are
    ``None`` when ffprobe fails or the stream lacks the information.
    """
    path = str(video_path)
    info = {"width": None, "height": None, "duration": None}
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json as _json

        data = _json.loads(result.stdout or "{}")
        streams = data.get("streams") or []
        if streams:
            stream = streams[0]
            try:
                info["width"] = int(stream.get("width")) if stream.get("width") else None
            except (TypeError, ValueError):
                info["width"] = None
            try:
                info["height"] = int(stream.get("height")) if stream.get("height") else None
            except (TypeError, ValueError):
                info["height"] = None
        fmt = data.get("format") or {}
        duration = fmt.get("duration")
        if duration is not None:
            try:
                info["duration"] = float(duration)
            except (TypeError, ValueError):
                info["duration"] = None
    except Exception as exc:
        logger.debug("ffprobe failed for %s: %s", path, exc)
    return info


def file_size(path: Path | str) -> int:
    """Return file size in bytes, or 0 if unavailable."""
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _dir_tree_size(directory: Path) -> int:
    """Sum sizes of all files under ``directory`` recursively."""
    if not directory.exists():
        return 0
    total = 0
    for p in directory.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def compute_project_size(project_manager: ProjectManager) -> int:
    """Compute total size (bytes) occupied by a project on disk.

    Sums uploads + artifacts + cache + results.
    """
    return sum(
        _dir_tree_size(d)
        for d in (
            project_manager.uploads_dir,
            project_manager.artifacts_dir,
            project_manager.cache_dir,
            project_manager.results_dir,
        )
    )


def refresh_project_size(project_manager: ProjectManager) -> int:
    """Recompute and return total project size (alias for clarity)."""
    return compute_project_size(project_manager)


def update_source_video_fields(project_manager: ProjectManager) -> dict:
    """Probe the source video and return fields to assign on the Project row."""
    source_path = project_manager.get_source_video_path()
    fields = {
        "source_width": None,
        "source_height": None,
        "source_duration": None,
        "source_size": None,
    }
    if source_path and Path(source_path).exists():
        info = probe_video(source_path)
        fields["source_width"] = info["width"]
        fields["source_height"] = info["height"]
        fields["source_duration"] = info["duration"]
        fields["source_size"] = file_size(source_path)
    return fields


def update_result_video_fields(
    project_manager: ProjectManager, target_lang: str
) -> dict:
    """Probe the result video for ``target_lang`` and return fields to assign."""
    result_path = project_manager.get_result_video_path(target_lang)
    fields = {
        "result_width": None,
        "result_height": None,
        "result_duration": None,
        "result_size": None,
    }
    if result_path and Path(result_path).exists():
        info = probe_video(result_path)
        fields["result_width"] = info["width"]
        fields["result_height"] = info["height"]
        fields["result_duration"] = info["duration"]
        fields["result_size"] = file_size(result_path)
    return fields
