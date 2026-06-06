"""Shared video upload helpers for project routes and external API."""

from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile

from ..config import get_settings
from .project_manager import ProjectManager, sanitize_filename

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


@dataclass(frozen=True)
class SavedVideo:
    """Result of saving an uploaded video to a project."""

    upload_path: Path
    safe_filename: str
    file_size: int


def validate_video_extension(filename: str) -> str:
    """Return normalized extension or raise HTTPException for unsupported types."""
    file_ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}",
        )
    return file_ext


async def save_project_video(
    project_id: str,
    file: UploadFile,
    *,
    max_upload_size: int | None = None,
    clear_existing: bool = True,
) -> SavedVideo:
    """Stream an uploaded video file into the project's uploads directory."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")

    validate_video_extension(file.filename)
    upload_limit = max_upload_size if max_upload_size is not None else get_settings().max_upload_size

    pm = ProjectManager(project_id)
    pm.ensure_directories()

    if clear_existing:
        for existing_file in pm.uploads_dir.iterdir():
            if existing_file.is_file():
                existing_file.unlink()

    upload_path = pm.get_upload_path(file.filename)
    file_size = 0

    try:
        with open(upload_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > upload_limit:
                    buffer.close()
                    upload_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large. Maximum size: "
                            f"{upload_limit / (1024 * 1024 * 1024):.1f}GB"
                        ),
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    return SavedVideo(
        upload_path=upload_path,
        safe_filename=sanitize_filename(file.filename),
        file_size=file_size,
    )
