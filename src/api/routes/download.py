"""Download routes for results."""

import asyncio
import json
import logging
import mimetypes
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
import re

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..database.session import get_db
from ..database.models import Project, ProjectStatus
from ..services.project_manager import ProjectManager
from ..services.job_dispatch import enqueue_download_job
from ..services.video_download import VideoDownloadError, validate_video_url, get_video_info
from ..models.schemas import VideoDownloadRequest, VideoDownloadResponse, VideoInfoResponse

router = APIRouter(prefix="/projects", tags=["download"])

logger = logging.getLogger(__name__)


CLIP_QUALITY_PRESETS = {
    "720p": {"max_height": 720, "crf": 20},
    "1080p": {"max_height": 1080, "crf": 19},
    "original": {"max_height": None, "crf": 18},
    "messenger": {"max_height": 480, "crf": 28, "audio_bitrate": "96k"},
}


# In-memory store for clip rendering jobs (process-local; cleared on restart).
CLIP_JOBS: dict[str, dict] = {}
CLIP_JOB_TTL = 3600  # seconds to keep completed/failed jobs around


def _gc_clip_jobs() -> None:
    now = time.time()
    expired = [jid for jid, j in CLIP_JOBS.items() if now - j.get("updated_at", 0) > CLIP_JOB_TTL]
    for jid in expired:
        job = CLIP_JOBS.pop(jid, None)
        if job and job.get("output_path"):
            _cleanup_clip_file(Path(job["output_path"]))


@router.post("/{project_id}/download-url", response_model=VideoDownloadResponse)
async def download_video_from_url(
    project_id: str,
    request: VideoDownloadRequest,
    db: Session = Depends(get_db),
):
    """Start downloading a video from a URL into an existing project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status not in (ProjectStatus.DRAFT, ProjectStatus.ERROR):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start download for project in '{project.status.value}' status",
        )

    try:
        url = validate_video_url(request.url)
        # Store download url and quality in project.config
        config = dict(project.config or {})
        config["downloadUrl"] = url
        config["downloadQuality"] = request.quality
        project.config = config
        db.commit()

        job_response = enqueue_download_job(db, project, url, request.quality)
    except VideoDownloadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return VideoDownloadResponse(
        url=f"/api/v1/projects/{project_id}/video",
        filename=project.source_filename or "",
        size=project.source_size or 0,
        title=None,
        jobId=job_response.jobId,
        status=job_response.status,
        projectId=project_id,
    )


@router.post("/video-info", response_model=VideoInfoResponse)
async def fetch_video_info(request: VideoDownloadRequest):
    """Retrieve video metadata (title, duration, uploader) from a URL without downloading."""
    try:
        url = validate_video_url(request.url)
        # yt-dlp is blocking; keep it off the event loop so one slow request
        # cannot freeze the whole API.
        info = await run_in_threadpool(get_video_info, url)
        return VideoInfoResponse(**info)
    except VideoDownloadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _resolve_result_video_path(project_id: str, project: Project) -> Path:
    """Resolve dubbed video path for a project (local or restore from S3)."""
    pm = ProjectManager(project_id)
    config = project.config or {}
    target_lang = config.get("targetLang", "ru")

    video_path = pm.get_result_video_path(target_lang)
    if video_path.exists():
        return video_path

    videos = list(pm.results_dir.glob("*.mp4")) if pm.results_dir.exists() else []
    if videos:
        return videos[0]

    try:
        from ..services.object_storage import get_object_storage
        store = get_object_storage()
        if store.is_enabled():
            restored = store.ensure_local_result_video(project_id, project)
            if restored and restored.exists():
                return restored
    except Exception as exc:
        logger.warning("Failed to restore result video from S3 for %s: %s", project_id, exc)

    raise HTTPException(status_code=404, detail="Dubbed video not found")


def _stream_s3_response(
    key: str,
    media_type: str,
    filename: str,
    request: Request | None = None,
    *,
    as_attachment: bool = True,
):
    """Stream an S3 object, with HTTP Range support when request is provided."""
    from ..services.object_storage import format_content_disposition, get_object_storage

    store = get_object_storage()
    try:
        file_size = store.get_object_size(key)
    except Exception as exc:
        logger.warning("S3 head failed for %s: %s", key, exc)
        raise HTTPException(status_code=404, detail="File not found in storage") from exc

    range_header = request.headers.get("range") if request else None
    disposition = "attachment" if as_attachment else "inline"
    base_headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": format_content_disposition(disposition, filename),
        "Cache-Control": "no-cache",
    }

    if not range_header:
        if file_size <= 0:
            headers = {**base_headers, "Content-Length": "0"}
            return Response(content=b"", media_type=media_type, headers=headers)
        headers = {
            **base_headers,
            "Content-Length": str(file_size),
        }
        return StreamingResponse(
            store.stream_range(key, 0, file_size - 1),
            media_type=media_type,
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
        store.stream_range(key, start, end),
        media_type=media_type,
        headers=headers,
        status_code=status.HTTP_206_PARTIAL_CONTENT,
    )


def _storage_meta(project: Project) -> dict:
    config = project.config or {}
    storage = config.get("storage") or {}
    return storage if isinstance(storage, dict) else {}


def _resolve_s3_result_key(project_id: str, project: Project, db: Session | None = None) -> str | None:
    """Resolve result video S3 key from config or by listing the bucket; cache into config."""
    storage = _storage_meta(project)
    key = storage.get("resultVideoKey")
    if key:
        return key
    try:
        from ..services.object_storage import get_object_storage
        store = get_object_storage()
        if not store.is_enabled():
            return None
        discovered = store.discover_storage_meta(project_id)
        if not discovered or not discovered.get("resultVideoKey"):
            return None
        # Persist so subsequent ranges skip list_objects
        if db is not None:
            try:
                config = dict(project.config or {})
                config["storage"] = {**(config.get("storage") or {}), **discovered}
                project.config = config
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(project, "config")
                db.commit()
            except Exception as exc:
                logger.warning("Failed to cache discovered storage for %s: %s", project_id, exc)
                db.rollback()
        return discovered.get("resultVideoKey")
    except Exception as exp:
        logger.warning("S3 result key resolve failed for %s: %s", project_id, exp)
        return None


def _resolve_s3_source_key(project_id: str, project: Project, db: Session | None = None) -> str | None:
    storage = _storage_meta(project)
    key = storage.get("sourceKey")
    if key:
        return key
    try:
        from ..services.object_storage import get_object_storage
        store = get_object_storage()
        if not store.is_enabled():
            return None
        discovered = store.discover_storage_meta(project_id)
        if not discovered or not discovered.get("sourceKey"):
            return None
        if db is not None:
            try:
                config = dict(project.config or {})
                config["storage"] = {**(config.get("storage") or {}), **discovered}
                project.config = config
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(project, "config")
                db.commit()
            except Exception as exc:
                logger.warning("Failed to cache discovered storage for %s: %s", project_id, exc)
                db.rollback()
        return discovered.get("sourceKey")
    except Exception as exp:
        logger.warning("S3 source key resolve failed for %s: %s", project_id, exp)
        return None


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
async def download_video(project_id: str, request: Request, db: Session = Depends(get_db)):
    """Download the dubbed video (local file or stream from S3)."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.status != ProjectStatus.DUBBED:
        raise HTTPException(status_code=400, detail="Video not ready. Project must be dubbed first.")

    pm = ProjectManager(project_id)
    config = project.config or {}
    target_lang = config.get("targetLang", "ru")
    local = pm.get_result_video_path(target_lang)
    if local.exists():
        return FileResponse(path=str(local), media_type="video/mp4", filename=local.name)
    if pm.results_dir.exists():
        videos = list(pm.results_dir.glob("*.mp4"))
        if videos:
            return FileResponse(path=str(videos[0]), media_type="video/mp4", filename=videos[0].name)

    storage = _storage_meta(project)
    key = storage.get("resultVideoKey") or _resolve_s3_result_key(project_id, project, db)
    if key:
        filename = Path(key).name
        return _stream_s3_response(key, "video/mp4", filename, request=request)

    raise HTTPException(status_code=404, detail="Dubbed video not found")


@router.get("/{project_id}/share/video")
async def share_result_video(
    project_id: str,
    expires_in: int | None = Query(None, ge=60, le=7 * 24 * 3600),
    db: Session = Depends(get_db),
):
    """Create a time-limited public S3 URL for the dubbed video (for sharing)."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.status != ProjectStatus.DUBBED:
        raise HTTPException(status_code=400, detail="Video not ready. Project must be dubbed first.")

    from ..services.object_storage import get_object_storage
    from ..config import get_settings

    store = get_object_storage()
    if not store.is_enabled():
        raise HTTPException(
            status_code=503,
            detail="S3 storage is not configured. Set S3_* env vars to enable sharing.",
        )

    settings = get_settings()
    ttl = expires_in if expires_in is not None else int(settings.s3_presign_expires)
    ttl = max(60, min(ttl, 7 * 24 * 3600))

    storage = _storage_meta(project)
    key = storage.get("resultVideoKey") or _resolve_s3_result_key(project_id, project, db)

    # If still only local — upload result to S3 so we can share it
    if not key:
        pm = ProjectManager(project_id)
        config = project.config or {}
        target_lang = config.get("targetLang", "ru")
        local = pm.get_result_video_path(target_lang)
        if not local.exists() and pm.results_dir.exists():
            videos = list(pm.results_dir.glob("*.mp4"))
            local = videos[0] if videos else local
        if not local.exists():
            raise HTTPException(status_code=404, detail="Dubbed video not found on disk or S3")
        key = store.object_key(project_id, "results", local.name)
        try:
            store.upload_file(local, key)
        except Exception as exc:
            logger.exception("Failed to upload result for share %s", project_id)
            raise HTTPException(status_code=502, detail=f"Failed to upload to S3: {exc}") from exc
        from sqlalchemy.orm.attributes import flag_modified

        config = dict(project.config or {})
        cfg_storage = dict(config.get("storage") or {})
        cfg_storage.update(
            {
                "provider": "s3",
                "bucket": store.bucket,
                "resultVideoKey": key,
            }
        )
        config["storage"] = cfg_storage
        project.config = config
        flag_modified(project, "config")
        db.commit()

    filename = Path(key).name
    try:
        url = store.presign_get_url(
            key,
            expires_in=ttl,
            filename=filename,
            content_type="video/mp4",
            as_attachment=False,
        )
    except Exception as exc:
        logger.exception("Failed to presign share URL for %s", project_id)
        raise HTTPException(status_code=502, detail=f"Failed to create share link: {exc}") from exc

    from datetime import datetime, timezone, timedelta

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
    return {
        "url": url,
        "expiresIn": ttl,
        "expiresAt": expires_at.isoformat(),
        "filename": filename,
        "key": key,
    }


def _cleanup_clip_file(file_path: Path) -> None:
    try:
        if file_path.exists():
            file_path.unlink()
    except OSError as exc:
        logger.warning("Failed to remove clip file %s: %s", file_path, exc)


async def _run_clip_render(job_id: str, cmd: list[str], out_path: Path, total_duration: float) -> None:
    """Background task: run ffmpeg, parse -progress output, update CLIP_JOBS."""
    job = CLIP_JOBS.get(job_id)
    if not job:
        return

    job.update(
        status="processing",
        progress=0.0,
        updated_at=time.time(),
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        job.update(status="failed", error=f"Failed to start ffmpeg: {exc}", updated_at=time.time())
        _cleanup_clip_file(out_path)
        return

    job["process"] = proc

    assert proc.stdout is not None
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if text.startswith("out_time_ms=") or text.startswith("out_time_us="):
                try:
                    us = int(text.split("=", 1)[1])
                except ValueError:
                    continue
                seconds = us / 1_000_000
                if total_duration > 0:
                    pct = max(0.0, min(99.0, (seconds / total_duration) * 100.0))
                    job.update(progress=pct, updated_at=time.time())
    except Exception as exc:
        logger.warning("clip %s progress read error: %s", job_id, exc)

    try:
        rc = await proc.wait()
    except Exception as exc:
        rc = -1
        logger.warning("clip %s wait error: %s", job_id, exc)

    stderr_tail = ""
    if proc.stderr is not None:
        try:
            err_bytes = await proc.stderr.read()
            stderr_tail = err_bytes.decode("utf-8", errors="replace")[-2000:]
        except Exception:
            pass

    if rc == 0 and out_path.exists() and out_path.stat().st_size > 0:
        job.update(status="completed", progress=100.0, updated_at=time.time())
        logger.info("clip %s rendered successfully (%s bytes)", job_id, out_path.stat().st_size)
    else:
        _cleanup_clip_file(out_path)
        job.update(
            status="failed",
            error=f"ffmpeg exited with rc={rc}. {stderr_tail[-500:]}" or "Render failed",
            updated_at=time.time(),
        )
        logger.error("clip %s render failed rc=%s: %s", job_id, rc, stderr_tail[-500:])


@router.post("/{project_id}/download/video/clip")
async def start_video_clip(
    project_id: str,
    start: float = Query(..., ge=0),
    end: float = Query(..., gt=0),
    quality: str = Query("1080p", pattern="^(720p|1080p|original|messenger)$"),
    db: Session = Depends(get_db),
):
    """Start rendering a re-encoded clip of the dubbed video. Returns a jobId for polling."""
    _gc_clip_jobs()

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status != ProjectStatus.DUBBED:
        raise HTTPException(status_code=400, detail="Video not ready. Project must be dubbed first.")

    video_path = _resolve_result_video_path(project_id, project)

    duration = project.result_duration or 0.0
    tolerance = 1.0
    if duration > 0 and end > duration + tolerance:
        raise HTTPException(status_code=400, detail=f"End time exceeds video duration ({duration:.2f}s)")
    if end <= start:
        raise HTTPException(status_code=400, detail="End time must be greater than start time")

    pm = ProjectManager(project_id)
    clips_dir = pm.cache_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex
    out_path = clips_dir / f"{job_id}.mp4"
    if out_path.exists():
        out_path.unlink()

    preset = CLIP_QUALITY_PRESETS[quality]
    vf_args = []
    if preset["max_height"] is not None:
        vf_args = ["-vf", f"scale=-2:{preset['max_height']}:flags=lanczos"]

    total_duration = max(0.1, end - start)

    audio_bitrate = preset.get("audio_bitrate", "192k")
    preset_name = "fast" if quality == "messenger" else "medium"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-progress", "pipe:1",
        "-nostats",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(video_path),
        *vf_args,
        "-c:v", "libx264",
        "-crf", str(preset["crf"]),
        "-preset", preset_name,
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(out_path),
    ]

    config = project.config or {}
    target_lang = config.get("targetLang", "ru")
    source = pm.get_source_video_path()
    base_name = source.stem if source else project_id
    download_name = f"{base_name}_{target_lang}_clip_{start:.1f}-{end:.1f}_{quality}.mp4"

    CLIP_JOBS[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "error": None,
        "output_path": str(out_path),
        "download_name": download_name,
        "project_id": project_id,
        "start": start,
        "end": end,
        "quality": quality,
        "created_at": time.time(),
        "updated_at": time.time(),
        "process": None,
    }

    logger.info("Starting clip render job %s for project %s [%ss-%ss] quality=%s",
                job_id, project_id, start, end, quality)
    asyncio.create_task(_run_clip_render(job_id, cmd, out_path, total_duration))

    return {"jobId": job_id, "status": "queued", "progress": 0.0}


@router.get("/{project_id}/download/video/clip/{job_id}/status")
async def get_video_clip_status(project_id: str, job_id: str, db: Session = Depends(get_db)):
    """Poll clip render job status."""
    _gc_clip_jobs()
    job = CLIP_JOBS.get(job_id)
    if not job or job.get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="Clip job not found")
    return {
        "jobId": job_id,
        "status": job["status"],
        "progress": round(float(job.get("progress", 0.0)), 1),
        "error": job.get("error"),
        "downloadName": job.get("download_name"),
    }


@router.get("/{project_id}/download/video/clip/{job_id}/result")
async def get_video_clip_result(
    project_id: str,
    job_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Download the rendered clip file (one-shot; file is removed after)."""
    job = CLIP_JOBS.get(job_id)
    if not job or job.get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="Clip job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Clip not ready (status={job['status']})")

    out_path = Path(job["output_path"])
    if not out_path.exists():
        job.update(status="failed", error="Result file no longer available", updated_at=time.time())
        raise HTTPException(status_code=404, detail="Result file no longer available")

    background_tasks.add_task(_cleanup_clip_file, out_path)
    background_tasks.add_task(lambda: CLIP_JOBS.pop(job_id, None))
    return FileResponse(
        path=str(out_path),
        media_type="video/mp4",
        filename=job.get("download_name") or "clip.mp4",
    )


@router.delete("/{project_id}/download/video/clip/{job_id}")
async def cancel_video_clip(project_id: str, job_id: str, db: Session = Depends(get_db)):
    """Cancel a running clip render job (or forget a completed/failed one)."""
    job = CLIP_JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=404, detail="Clip job not found")
    proc = job.get("process")
    if proc and proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    out_path = job.get("output_path")
    if out_path:
        _cleanup_clip_file(Path(out_path))
    return {"ok": True}


@router.get("/{project_id}/stream/video")
async def stream_dubbed_video(project_id: str, request: Request, db: Session = Depends(get_db)):
    """Stream dubbed video with HTTP Range support for immediate playback."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status != ProjectStatus.DUBBED:
        raise HTTPException(status_code=400, detail="Video not ready. Project must be dubbed first.")

    pm = ProjectManager(project_id)
    config = project.config or {}
    target_lang = config.get("targetLang", "ru")
    video_path = pm.get_result_video_path(target_lang)
    if not video_path.exists() and pm.results_dir.exists():
        videos = list(pm.results_dir.glob("*.mp4"))
        if videos:
            video_path = videos[0]

    if video_path.exists():
        file_size = video_path.stat().st_size
        range_header = request.headers.get("range")
        base_headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        }
        if not range_header:
            headers = {**base_headers, "Content-Length": str(file_size)}
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
        headers = {
            **base_headers,
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(end - start + 1),
        }
        return StreamingResponse(
            _iter_file_range(video_path, start, end),
            media_type="video/mp4",
            headers=headers,
            status_code=status.HTTP_206_PARTIAL_CONTENT,
        )

    storage = _storage_meta(project)
    key = storage.get("resultVideoKey") or _resolve_s3_result_key(project_id, project, db)
    if key:
        return _stream_s3_response(
            key, "video/mp4", Path(key).name, request=request, as_attachment=False
        )

    raise HTTPException(status_code=404, detail="Dubbed video not found")


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
        if pm.results_dir.exists():
            subtitles = list(pm.results_dir.glob(f"*_{language}.{format}"))
            if subtitles:
                subtitle_path = subtitles[0]

    if not subtitle_path.exists():
        try:
            from ..services.object_storage import get_object_storage
            store = get_object_storage()
            if store.is_enabled():
                restored = store.ensure_local_subtitle(project_id, project, language, format)
                if restored and restored.exists():
                    subtitle_path = restored
        except Exception as exc:
            logger.warning("Failed to restore subtitles from S3 for %s: %s", project_id, exc)

    if not subtitle_path.exists():
        raise HTTPException(status_code=404, detail=f"Subtitle file not found for {lang} language")
    
    media_type = "text/plain" if format == "srt" else "text/vtt"
    
    return FileResponse(
        path=str(subtitle_path),
        media_type=media_type,
        filename=subtitle_path.name
    )


@router.get("/{project_id}/video")
async def get_source_video(project_id: str, request: Request, db: Session = Depends(get_db)):
    """Stream the source video (local or from S3)."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    pm = ProjectManager(project_id)
    video_path = pm.get_source_video_path()
    if video_path and video_path.exists():
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=video_path.name
        )

    storage = _storage_meta(project)
    key = storage.get("sourceKey") or _resolve_s3_source_key(project_id, project, db)
    if key:
        return _stream_s3_response(
            key, "video/mp4", Path(key).name, request=request, as_attachment=False
        )

    raise HTTPException(status_code=404, detail="Source video not found")


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
        "resultDuration": project.result_duration or 0,
        "resultWidth": project.result_width or 0,
        "resultHeight": project.result_height or 0,
    }


def _resolve_report_paths(project_id: str, project: Project | None = None) -> tuple[Path, Path]:
    """Return (markdown_path, json_path) for the summary report, restoring from S3 if needed."""
    pm = ProjectManager(project_id)
    md_path = pm.artifacts_dir / "report.md"
    json_path = pm.artifacts_dir / "report.json"
    if md_path.exists():
        return md_path, json_path

    try:
        from ..services.object_storage import get_object_storage
        store = get_object_storage()
        if store.is_enabled() and project is not None:
            restored_md = store.ensure_local_report(project_id, project, kind="md")
            restored_json = store.ensure_local_report(project_id, project, kind="json")
            if restored_md and restored_md.exists():
                md_path = restored_md
            if restored_json and restored_json.exists():
                json_path = restored_json
    except Exception as exc:
        logger.warning("Failed to restore report from S3 for %s: %s", project_id, exc)

    return md_path, json_path


@router.get("/{project_id}/report")
async def get_project_report(project_id: str, db: Session = Depends(get_db)):
    """Return the summary report markdown + structured JSON for a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    md_path, json_path = _resolve_report_paths(project_id, project)
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

    md_path, _ = _resolve_report_paths(project_id, project)
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="Report not available yet")

    pm = ProjectManager(project_id)
    source = pm.get_source_video_path()
    storage = _storage_meta(project)
    if source:
        filename = f"{source.stem}_report.md"
    elif storage.get("sourceFilename"):
        filename = f"{Path(storage['sourceFilename']).stem}_report.md"
    else:
        filename = f"{project_id}_report.md"
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
