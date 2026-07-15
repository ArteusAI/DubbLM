"""File upload routes."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..database.models import Project
from ..models.schemas import UploadResponse
from ..services.video_upload import save_project_video
from ..services.video_info import probe_video, compute_project_size
from ..services.project_manager import ProjectManager
from ..services.job_dispatch import enqueue_transcription_job
from ..config import get_settings

router = APIRouter(prefix="/projects", tags=["upload"])


@router.post("/{project_id}/upload", response_model=UploadResponse)
async def upload_video(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload source video for a project."""
    settings = get_settings()

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    saved_video = await save_project_video(
        project_id,
        file,
        max_upload_size=settings.max_upload_size,
    )

    project.source_file = str(saved_video.upload_path)
    project.source_filename = saved_video.safe_filename
    project.source_size = saved_video.file_size

    # Probe resolution/duration of the source video
    probe = probe_video(saved_video.upload_path)
    project.source_width = probe["width"]
    project.source_height = probe["height"]
    project.source_duration = probe["duration"]

    # Refresh total project size on disk
    project.total_size = compute_project_size(ProjectManager(project_id))

    project.updated_at = datetime.now(timezone.utc)
    db.commit()

    # If the user queued auto-processing while uploading, start transcription now
    config = project.config or {}
    if config.get("autoProcess"):
        enqueue_transcription_job(db, project, allow_existing=False)

    return UploadResponse(
        url=f"/api/v1/projects/{project_id}/video",
        filename=saved_video.safe_filename,
        size=saved_video.file_size,
    )
