"""File upload routes."""

import shutil
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..database.models import Project
from ..models.schemas import UploadResponse
from ..services.project_manager import ProjectManager, sanitize_filename
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
    
    # Get project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
    file_ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Initialize project manager
    pm = ProjectManager(project_id)
    pm.ensure_directories()
    
    # Clear existing uploads
    for existing_file in pm.uploads_dir.iterdir():
        if existing_file.is_file():
            existing_file.unlink()
    
    # Save uploaded file
    upload_path = pm.get_upload_path(file.filename)
    
    try:
        # Stream file to disk
        with open(upload_path, "wb") as buffer:
            file_size = 0
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                file_size += len(chunk)
                if file_size > settings.max_upload_size:
                    buffer.close()
                    upload_path.unlink()
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {settings.max_upload_size / (1024*1024*1024):.1f}GB"
                    )
                buffer.write(chunk)
    except Exception as e:
        if upload_path.exists():
            upload_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    # Update project with file info (use sanitized filename)
    safe_filename = sanitize_filename(file.filename)
    project.source_file = str(upload_path)
    project.source_filename = safe_filename
    project.source_size = file_size
    project.updated_at = datetime.now(timezone.utc)
    db.commit()
    
    return UploadResponse(
        url=f"/api/v1/projects/{project_id}/video",
        filename=safe_filename,
        size=file_size
    )

