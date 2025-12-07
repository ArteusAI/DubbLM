"""Download routes for results."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..database.models import Project, ProjectStatus
from ..services.project_manager import ProjectManager

router = APIRouter(prefix="/projects", tags=["download"])


@router.get("/{project_id}/download/video")
async def download_video(project_id: str, db: Session = Depends(get_db)):
    """Download the dubbed video."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.status != ProjectStatus.DUBBED:
        raise HTTPException(status_code=400, detail="Video not ready. Project must be dubbed first.")
    
    pm = ProjectManager(project_id)
    config = project.config or {}
    target_lang = config.get("targetLang", "ru")
    
    video_path = pm.get_result_video_path(target_lang)
    
    if not video_path.exists():
        # Try to find any video in results
        videos = list(pm.results_dir.glob("*.mp4"))
        if videos:
            video_path = videos[0]
        else:
            raise HTTPException(status_code=404, detail="Dubbed video not found")
    
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name
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

