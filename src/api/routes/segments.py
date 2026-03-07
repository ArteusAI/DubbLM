"""Segment editing routes."""

import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from ..database.session import get_db
from ..database.models import Project, Segment, TTSProvider, Job, JobType, JobStatus
from ..models.schemas import (
    SegmentResponse,
    SegmentUpdate,
    SpeakerRename,
    SpeakerVoiceUpdate,
    RephraseRequest,
    RephraseResponse,
    PreviewRequest,
    PreviewResponse,
    PreviewJobResponse,
    PreviewStatusResponse,
)
from ..workers.tasks import generate_preview, rephrase_segment
from ..services.project_manager import ProjectManager

router = APIRouter(prefix="/projects", tags=["segments"])


@router.get("/{project_id}/segments", response_model=list)
async def list_segments(project_id: str, db: Session = Depends(get_db)):
    """Get all segments for a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    segments = db.query(Segment).filter(
        Segment.project_id == project_id
    ).order_by(Segment.sequence).all()
    
    return [SegmentResponse(**seg.to_dict()) for seg in segments]


@router.patch("/{project_id}/segments/{segment_id}", response_model=SegmentResponse)
async def update_segment(
    project_id: str,
    segment_id: str,
    update_data: SegmentUpdate,
    db: Session = Depends(get_db)
):
    """Update a segment."""
    segment = db.query(Segment).filter(
        Segment.id == segment_id,
        Segment.project_id == project_id
    ).first()
    
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    
    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    
    if "startTime" in update_dict:
        segment.start_time = update_dict["startTime"]
    if "endTime" in update_dict:
        segment.end_time = update_dict["endTime"]
    if "translatedText" in update_dict:
        segment.translated_text = update_dict["translatedText"]
    if "isMuted" in update_dict:
        segment.is_muted = update_dict["isMuted"]
    if "voiceId" in update_dict:
        segment.voice_id = update_dict["voiceId"]
    if "provider" in update_dict:
        try:
            segment.provider = TTSProvider(update_dict["provider"]) if update_dict["provider"] else None
        except ValueError:
            pass
    if "ttsPrompt" in update_dict:
        segment.tts_prompt = update_dict["ttsPrompt"]
    
    # Clear cached audio when content changes
    if any(k in update_dict for k in ["translatedText", "voiceId", "provider", "ttsPrompt"]):
        segment.audio_url = None
    
    db.commit()
    db.refresh(segment)
    
    return SegmentResponse(**segment.to_dict())


@router.post("/{project_id}/speakers/rename")
async def rename_speaker(
    project_id: str,
    rename_data: SpeakerRename,
    db: Session = Depends(get_db)
):
    """Rename a speaker globally across all segments."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Update all segments with the old speaker name
    updated = db.query(Segment).filter(
        Segment.project_id == project_id,
        Segment.speaker == rename_data.oldName
    ).update({"speaker": rename_data.newName})
    
    db.commit()
    
    return {"message": f"Renamed {updated} segments from {rename_data.oldName} to {rename_data.newName}"}


@router.post("/{project_id}/speakers/voice")
async def update_speaker_voice(
    project_id: str,
    voice_data: SpeakerVoiceUpdate,
    db: Session = Depends(get_db)
):
    """Update voice for all segments of a speaker."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        provider = TTSProvider(voice_data.provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {voice_data.provider}")
    
    # Update all segments for the speaker
    updated = db.query(Segment).filter(
        Segment.project_id == project_id,
        Segment.speaker == voice_data.speakerName
    ).update({
        "voice_id": voice_data.voiceId,
        "provider": provider,
        "audio_url": None,  # Clear cached audio
    })

    # Persist mapping at project-config level so full dubbing can reuse it.
    config = dict(project.config or {})
    mappings = dict(config.get("speakerVoiceMappings") or {})
    mappings[voice_data.speakerName] = voice_data.voiceId
    config["speakerVoiceMappings"] = mappings
    project.config = config
    flag_modified(project, "config")
    project.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    
    return {"message": f"Updated voice for {updated} segments of speaker {voice_data.speakerName}"}


@router.post("/{project_id}/segments/{segment_id}/rephrase", response_model=RephraseResponse)
async def rephrase_segment_text(
    project_id: str,
    segment_id: str,
    rephrase_data: RephraseRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Rephrase a segment using LLM."""
    segment = db.query(Segment).filter(
        Segment.id == segment_id,
        Segment.project_id == project_id
    ).first()
    
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    
    # For synchronous response, we'll do inline processing
    # For async, we could use background_tasks.add_task
    try:
        result = rephrase_segment(project_id, segment_id, rephrase_data.prompt)
        
        # Refresh segment to get updated text
        db.refresh(segment)
        
        return RephraseResponse(translatedText=segment.translated_text or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rephrase failed: {str(e)}")


@router.post("/{project_id}/segments/{segment_id}/preview", response_model=PreviewJobResponse)
async def preview_segment(
    project_id: str,
    segment_id: str,
    preview_data: PreviewRequest,
    db: Session = Depends(get_db)
):
    """Start TTS preview generation for a segment via worker."""
    segment = db.query(Segment).filter(
        Segment.id == segment_id,
        Segment.project_id == project_id
    ).first()
    
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    
    voice_id = segment.voice_id
    pm = ProjectManager(project_id)
    preview_path = pm.get_preview_audio_path(segment_id, voice_id)
    
    # Check cache first (unless force regenerate)
    if preview_path.exists() and not preview_data.forceRegenerate:
        return PreviewJobResponse(
            status="completed",
            audioUrl=f"/api/v1/projects/{project_id}/segments/{segment_id}/preview/audio?voice_id={voice_id or ''}",
        )
    
    # Create a job to track the preview task
    job = Job(
        id=secrets.token_hex(16),
        project_id=project_id,
        job_type=JobType.PREVIEW,
        status=JobStatus.PENDING,
        current_step=f"segment:{segment_id}",
    )
    db.add(job)
    db.commit()
    
    # Start async task via celery
    task = generate_preview.delay(project_id, segment_id, preview_data.forceRegenerate, job.id)
    
    # Update job with celery task ID
    job.celery_task_id = task.id
    db.commit()
    
    return PreviewJobResponse(
        status="processing",
        jobId=job.id,
    )


@router.get("/{project_id}/segments/{segment_id}/preview/status", response_model=PreviewStatusResponse)
async def get_preview_status(
    project_id: str,
    segment_id: str,
    job_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get status of preview generation for a segment."""
    # Get segment to know current voice
    segment = db.query(Segment).filter(
        Segment.id == segment_id,
        Segment.project_id == project_id
    ).first()
    
    voice_id = segment.voice_id if segment else None
    
    # If job_id provided, check that specific job
    if job_id:
        job = db.query(Job).filter(
            Job.id == job_id,
            Job.project_id == project_id,
            Job.job_type == JobType.PREVIEW
        ).first()
        
        if job:
            if job.status == JobStatus.COMPLETED:
                return PreviewStatusResponse(
                    status="completed",
                    audioUrl=f"/api/v1/projects/{project_id}/segments/{segment_id}/preview/audio?voice_id={voice_id or ''}",
                )
            elif job.status == JobStatus.FAILED:
                return PreviewStatusResponse(
                    status="failed",
                    error=job.error_message,
                )
            else:
                return PreviewStatusResponse(status="processing")
    
    # Otherwise check if audio already exists for current voice
    pm = ProjectManager(project_id)
    preview_path = pm.get_preview_audio_path(segment_id, voice_id)
    
    if preview_path.exists():
        return PreviewStatusResponse(
            status="completed",
            audioUrl=f"/api/v1/projects/{project_id}/segments/{segment_id}/preview/audio?voice_id={voice_id or ''}",
        )
    
    # Check for any active preview job for this segment
    active_job = db.query(Job).filter(
        Job.project_id == project_id,
        Job.job_type == JobType.PREVIEW,
        Job.current_step == f"segment:{segment_id}",
        Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
    ).first()
    
    if active_job:
        return PreviewStatusResponse(status="processing", jobId=active_job.id)
    
    return PreviewStatusResponse(status="none")
