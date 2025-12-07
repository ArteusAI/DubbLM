"""Video frame extraction routes for dynamic thumbnails."""

import logging
import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..database.models import Project
from ..services.project_manager import ProjectManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["frames"])


def _get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def _extract_frame(video_path: str, timestamp: float, output_path: str, quality: int = 5) -> bool:
    """Extract a single frame from video at given timestamp using ffmpeg."""
    try:
        cmd = [
            'ffmpeg', '-y', '-ss', str(timestamp),
            '-i', video_path,
            '-vframes', '1',
            '-vf', 'scale=480:-1',  # Scale to 480px width for better quality
            '-q:v', str(quality),  # Quality 2-5 is good for thumbnails
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        exists = Path(output_path).exists()
        if exists:
            logger.info(f"Extracted frame at {timestamp}s -> {output_path}")
        return exists
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract frame: {e.stderr}")
        return False


@router.get("/{project_id}/frame")
async def get_video_frame(
    project_id: str,
    progress: float = Query(0, ge=0, le=100, description="Progress percentage (0-100)"),
    db: Session = Depends(get_db)
):
    """Get a video frame at a specific progress percentage.
    
    Frames are cached for reuse. The frame is extracted from the source video
    at the timestamp corresponding to the given progress percentage.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        logger.warning(f"Frame request for non-existent project: {project_id}")
        raise HTTPException(status_code=404, detail="Project not found")
    
    pm = ProjectManager(project_id)
    video_path = pm.get_source_video_path()
    
    if not video_path or not video_path.exists():
        logger.warning(f"Video not found for project {project_id}: {video_path}")
        raise HTTPException(status_code=404, detail="Source video not found")
    
    # Create frames cache directory
    frames_dir = pm.cache_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Round progress to nearest 5% for caching (20 frames max per video)
    progress_bucket = int(round(progress / 5) * 5)
    progress_bucket = max(0, min(100, progress_bucket))
    
    frame_path = frames_dir / f"frame_{progress_bucket:03d}.jpg"
    
    # Check if frame already exists in cache
    if not frame_path.exists():
        # Get video duration
        duration = _get_video_duration(str(video_path))
        if duration <= 0:
            raise HTTPException(status_code=500, detail="Could not determine video duration")
        
        # Calculate timestamp
        timestamp = (progress_bucket / 100) * duration
        # Avoid very start of video (often black)
        timestamp = max(0.5, timestamp)
        
        # Extract frame
        success = _extract_frame(str(video_path), timestamp, str(frame_path))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to extract frame")
    
    return FileResponse(
        path=str(frame_path),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
        }
    )


@router.get("/{project_id}/thumbnail")
async def get_video_thumbnail(
    project_id: str,
    db: Session = Depends(get_db)
):
    """Get a static thumbnail for the video (frame at ~1 second).
    
    This is useful for displaying a preview before processing starts.
    The thumbnail is cached after first generation.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    pm = ProjectManager(project_id)
    video_path = pm.get_source_video_path()
    
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Source video not found")
    
    # Create cache directory
    pm.cache_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_path = pm.cache_dir / "thumbnail.jpg"
    
    # Generate thumbnail if not exists
    if not thumbnail_path.exists():
        success = _extract_frame(str(video_path), 1.0, str(thumbnail_path))
        if not success:
            # Try at 0.1 seconds if 1 second fails
            success = _extract_frame(str(video_path), 0.1, str(thumbnail_path))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
    
    return FileResponse(
        path=str(thumbnail_path),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
        }
    )


@router.get("/{project_id}/video-info")
async def get_video_info(
    project_id: str,
    db: Session = Depends(get_db)
):
    """Get basic video information including duration."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    pm = ProjectManager(project_id)
    video_path = pm.get_source_video_path()
    
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Source video not found")
    
    duration = _get_video_duration(str(video_path))
    
    return {
        "duration": duration,
        "filename": video_path.name,
    }

