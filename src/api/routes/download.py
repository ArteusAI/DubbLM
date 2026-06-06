"""Download routes for results."""

import json
import mimetypes
from pathlib import Path
import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..database.models import Project, ProjectStatus
from ..services.project_manager import ProjectManager

router = APIRouter(prefix="/projects", tags=["download"])


def _resolve_result_video_path(project_id: str, project: Project) -> Path:
    """Resolve dubbed video path for a project."""
    pm = ProjectManager(project_id)
    config = project.config or {}
    target_lang = config.get("targetLang", "ru")

    video_path = pm.get_result_video_path(target_lang)
    if video_path.exists():
        return video_path

    videos = list(pm.results_dir.glob("*.mp4"))
    if videos:
        return videos[0]

    raise HTTPException(status_code=404, detail="Dubbed video not found")


def _iter_file_range(file_path: Path, start: int, end: int, chunk_size: int = 1024 * 1024):
    """Yield file bytes from start to end (inclusive)."""
    with file_path.open("rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            read_size = min(chunk_size, remaining)
            data = f.read(read_size)
            if not data:
                break
            remaining -= len(data)
            yield data


def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int] | None:
    """Parse a single HTTP Range header in bytes units."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match:
        return None

    start_str, end_str = match.groups()
    if not start_str and not end_str:
        return None

    if start_str:
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1
    else:
        suffix_length = int(end_str)
        if suffix_length <= 0:
            return None
        start = max(0, file_size - suffix_length)
        end = file_size - 1

    if start < 0 or end < start or start >= file_size:
        return None

    end = min(end, file_size - 1)
    return start, end


@router.get("/{project_id}/download/video")
async def download_video(project_id: str, db: Session = Depends(get_db)):
    """Download the dubbed video."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.status != ProjectStatus.DUBBED:
        raise HTTPException(status_code=400, detail="Video not ready. Project must be dubbed first.")
    
    video_path = _resolve_result_video_path(project_id, project)
    
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name
    )


@router.get("/{project_id}/stream/video")
async def stream_dubbed_video(project_id: str, request: Request, db: Session = Depends(get_db)):
    """Stream dubbed video with HTTP Range support for immediate playback."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status != ProjectStatus.DUBBED:
        raise HTTPException(status_code=400, detail="Video not ready. Project must be dubbed first.")

    video_path = _resolve_result_video_path(project_id, project)
    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    base_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
    }

    if not range_header:
        headers = {
            **base_headers,
            "Content-Length": str(file_size),
        }
        return StreamingResponse(
            _iter_file_range(video_path, 0, file_size - 1),
            media_type="video/mp4",
            headers=headers,
            status_code=status.HTTP_200_OK,
        )

    byte_range = _parse_range_header(range_header, file_size)
    if byte_range is None:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    start, end = byte_range
    content_length = end - start + 1
    headers = {
        **base_headers,
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Content-Length": str(content_length),
    }
    return StreamingResponse(
        _iter_file_range(video_path, start, end),
        media_type="video/mp4",
        headers=headers,
        status_code=status.HTTP_206_PARTIAL_CONTENT,
    )


@router.get("/{project_id}/download/subtitles")
async def download_subtitles(
    project_id: str,
    format: str = "srt",
    lang: str = "target",
    db: Session = Depends(get_db)
):
    """Download subtitles for source or target language."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.status not in [ProjectStatus.TRANSCRIBED, ProjectStatus.DUBBED]:
        raise HTTPException(status_code=400, detail="Subtitles not ready. Project must be transcribed first.")
    
    pm = ProjectManager(project_id)
    config = project.config or {}
    
    if lang not in ["source", "target"]:
        raise HTTPException(status_code=400, detail="Invalid lang. Use 'source' or 'target'")
    
    if format not in ["srt", "vtt"]:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'srt' or 'vtt'")
    
    language = config.get("sourceLang", "en") if lang == "source" else config.get("targetLang", "ru")
    subtitle_path = pm.get_result_subtitles_path(language, format)
    
    if not subtitle_path.exists():
        # Try to find any subtitle file in results matching the language
        subtitles = list(pm.results_dir.glob(f"*_{language}.{format}"))
        if subtitles:
            subtitle_path = subtitles[0]
        else:
            raise HTTPException(status_code=404, detail=f"Subtitle file not found for {lang} language")
    
    media_type = "text/plain" if format == "srt" else "text/vtt"
    
    return FileResponse(
        path=str(subtitle_path),
        media_type=media_type,
        filename=subtitle_path.name
    )


@router.get("/{project_id}/video")
async def get_source_video(project_id: str, db: Session = Depends(get_db)):
    """Stream the source video."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    pm = ProjectManager(project_id)
    video_path = pm.get_source_video_path()
    
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Source video not found")
    
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name
    )


@router.get("/{project_id}/segments/{segment_id}/preview/audio")
async def get_preview_audio(
    project_id: str,
    segment_id: str,
    voice_id: str | None = None,
    db: Session = Depends(get_db)
):
    """Get preview audio for a segment."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # If no voice_id provided, get it from the segment
    if not voice_id:
        from ..database.models import Segment
        segment = db.query(Segment).filter(Segment.id == segment_id).first()
        voice_id = segment.voice_id if segment else None
    
    pm = ProjectManager(project_id)
    audio_path = pm.get_preview_audio_path(segment_id, voice_id)
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Preview audio not found. Generate preview first.")
    
    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=audio_path.name
    )


@router.get("/{project_id}/stats")
async def get_project_stats(project_id: str, db: Session = Depends(get_db)):
    """Get processing statistics for a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    config = project.config or {}
    result_stats = config.get("resultStats", {})
    
    return {
        "processingTimeSec": result_stats.get("processingTimeSec", 0),
        "totalCost": result_stats.get("totalCost", 0),
    }


def _resolve_report_paths(project_id: str) -> tuple[Path, Path]:
    """Return (markdown_path, json_path) for the summary report."""
    pm = ProjectManager(project_id)
    md_path = pm.artifacts_dir / "report.md"
    json_path = pm.artifacts_dir / "report.json"
    return md_path, json_path


@router.get("/{project_id}/report")
async def get_project_report(project_id: str, db: Session = Depends(get_db)):
    """Return the summary report markdown + structured JSON for a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    md_path, json_path = _resolve_report_paths(project_id)
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="Report not available yet")

    markdown = md_path.read_text(encoding="utf-8")
    report_json = None
    if json_path.exists():
        try:
            report_json = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report_json = None

    return JSONResponse({"markdown": markdown, "json": report_json})


@router.get("/{project_id}/download/report")
async def download_project_report(project_id: str, db: Session = Depends(get_db)):
    """Download the summary report as Markdown."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    md_path, _ = _resolve_report_paths(project_id)
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="Report not available yet")

    pm = ProjectManager(project_id)
    source = pm.get_source_video_path()
    filename = f"{source.stem}_report.md" if source else f"{project_id}_report.md"
    return FileResponse(
        path=str(md_path),
        media_type="text/markdown",
        filename=filename,
    )


@router.get("/{project_id}/artifacts/{path:path}")
async def get_project_artifact(
    project_id: str,
    path: str,
    db: Session = Depends(get_db),
):
    """Serve any file stored under the project's base directory.

    Used by links inside the markdown summary report. Path traversal is
    strictly guarded — requests resolving outside the project root get 403.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    pm = ProjectManager(project_id)
    project_root = pm.base_dir.resolve()
    try:
        requested = (project_root / path).resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")

    try:
        requested.relative_to(project_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
    return FileResponse(
        path=str(requested),
        media_type=media_type,
        filename=requested.name,
    )
