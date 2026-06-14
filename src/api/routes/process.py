"""Processing routes for transcription and dubbing."""

import logging
import shutil
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

logger = logging.getLogger(__name__)

from ..database.session import get_db
from ..database.models import Project, Segment, Job, JobType, JobStatus, ProjectStatus, generate_job_id
from ..models.schemas import JobResponse, QueueAutoResponse
from ..services.project_manager import ProjectManager
from ..services.job_dispatch import enqueue_transcription_job
from ..workers.tasks import dub_project, retranslate_project
from ..workers.celery_app import celery_app
from src.utils.speaker_gender import is_speaker_gender_translation_stale

router = APIRouter(prefix="/projects", tags=["process"])


@router.post("/{project_id}/process/transcribe", response_model=JobResponse, status_code=202)
async def start_transcription(project_id: str, db: Session = Depends(get_db)):
    """Start transcription pipeline."""
    # Force session to expire cached data and re-read from DB
    db.expire_all()
    
    # Get project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.status == ProjectStatus.DOWNLOADING:
        raise HTTPException(status_code=409, detail="Video download is still in progress")

    # Check if project has a video
    pm = ProjectManager(project_id)
    source_video = pm.get_source_video_path()
    if not source_video or not source_video.exists():
        raise HTTPException(status_code=400, detail="No video uploaded for this project")
    
    # Check for required config
    config = project.config or {}
    logger.debug(f"start_transcription config: sourceLang={config.get('sourceLang')}, speakerCount={config.get('speakerCount')}")
    
    if not config.get("sourceLang") or not config.get("targetLang"):
        raise HTTPException(
            status_code=400,
            detail="Source and target languages must be configured before transcription"
        )
    
    return enqueue_transcription_job(db, project, allow_existing=True)


@router.post("/{project_id}/process/queue-auto", response_model=QueueAutoResponse, status_code=202)
async def queue_auto_process(project_id: str, db: Session = Depends(get_db)):
    """Queue a project for automatic processing once media is available.

    Sets the autoProcess flag. If the project already has a source video and
    is in DRAFT status, transcription is started immediately. Otherwise the
    project will auto-start when the upload/download finishes.
    """
    db.expire_all()

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Set autoProcess flag
    config = dict(project.config) if project.config else {}
    config["autoProcess"] = True
    project.config = config
    flag_modified(project, "config")
    project.updated_at = datetime.now(timezone.utc)
    db.commit()

    # If media is already available, start immediately
    pm = ProjectManager(project_id)
    source_video = pm.get_source_video_path()
    if project.status == ProjectStatus.DRAFT and source_video and source_video.exists():
        job = enqueue_transcription_job(db, project, allow_existing=False)
        return QueueAutoResponse(
            queued=False,
            started=True,
            jobId=job.jobId,
            projectId=project_id,
            status=project.status.value,
        )

    return QueueAutoResponse(
        queued=True,
        started=False,
        projectId=project_id,
        status=project.status.value,
    )


@router.post("/{project_id}/process/retranslate", response_model=JobResponse, status_code=202)
async def start_retranslation(project_id: str, db: Session = Depends(get_db)):
    """Re-run translation for existing segments using current speaker metadata."""
    db.expire_all()

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status not in [ProjectStatus.TRANSCRIBED, ProjectStatus.DUBBED, ProjectStatus.ERROR]:
        raise HTTPException(
            status_code=400,
            detail="Project must already have translated segments before re-translation",
        )

    source_video = ProjectManager(project_id).get_source_video_path()
    if not source_video or not source_video.exists():
        raise HTTPException(status_code=400, detail="No video uploaded for this project")

    if not project.segments:
        raise HTTPException(status_code=400, detail="No segments found for re-translation")

    config = project.config or {}
    if not config.get("sourceLang") or not config.get("targetLang"):
        raise HTTPException(
            status_code=400,
            detail="Source and target languages must be configured before re-translation",
        )

    active_job = db.query(Job).filter(
        Job.project_id == project.id,
        Job.job_type == JobType.TRANSCRIBE,
        Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]),
    ).first()
    if active_job:
        return JobResponse(
            jobId=active_job.id,
            status=active_job.status.value,
            projectId=project.id,
            type="transcribe",
            progress=active_job.progress,
            currentStep=active_job.current_step,
        )

    job = Job(
        id=generate_job_id("retranslate"),
        project_id=project.id,
        job_type=JobType.TRANSCRIBE,
        status=JobStatus.PENDING,
        current_step="retranslation_pending",
    )
    db.add(job)
    project.status = ProjectStatus.TRANSCRIBING
    project.updated_at = datetime.now(timezone.utc)
    db.commit()

    task = retranslate_project.delay(project.id, job.id)
    job.celery_task_id = task.id
    db.commit()

    return JobResponse(
        jobId=job.id,
        status="processing",
        projectId=project.id,
        type="transcribe",
        progress=0,
        currentStep="retranslation_pending",
    )


@router.post("/{project_id}/process/restart", response_model=JobResponse, status_code=202)
async def restart_processing(project_id: str, db: Session = Depends(get_db)):
    """Reset project cache/artifacts and restart end-to-end processing from scratch."""
    db.expire_all()

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    pm = ProjectManager(project_id)
    source_video = pm.get_source_video_path()
    if not source_video or not source_video.exists():
        raise HTTPException(status_code=400, detail="No video uploaded for this project")

    config = dict(project.config or {})
    if not config.get("sourceLang") or not config.get("targetLang"):
        raise HTTPException(
            status_code=400,
            detail="Source and target languages must be configured before restart"
        )

    active_jobs = db.query(Job).filter(
        Job.project_id == project_id,
        Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]),
    ).all()
    for job in active_jobs:
        if job.celery_task_id:
            celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGTERM")
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        job.add_log("Processing cancelled for restart", "info")

    # Remove segment data and all generated artifacts/cache. A full restart
    # must re-run TTS too; use reset-tts-cache for narrower TTS-only resets.
    db.query(Segment).filter(Segment.project_id == project_id).delete()
    pm.cleanup_artifacts(preserve_tts_cache=False)
    if pm.results_dir.exists():
        shutil.rmtree(pm.results_dir)
    pm.ensure_directories()

    # Ensure restart runs full pipeline automatically.
    config["autoProcess"] = True
    config.pop("resultStats", None)
    project.config = config
    flag_modified(project, "config")
    project.status = ProjectStatus.DRAFT
    project.updated_at = datetime.now(timezone.utc)
    db.commit()

    return enqueue_transcription_job(db, project, allow_existing=False)


@router.post("/{project_id}/process/reset-tts-cache", status_code=200)
async def reset_tts_cache(project_id: str, db: Session = Depends(get_db)):
    """Wipe only TTS-related caches/artifacts.

    Clears per-segment synthesized audio and combined TTS audio so the next
    dub run re-synthesizes everything, while preserving transcription,
    translation, diarization, speaker samples, and uploaded source files.
    """
    db.expire_all()

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Refuse while a job is running to avoid racing with the worker.
    active_job = db.query(Job).filter(
        Job.project_id == project_id,
        Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]),
    ).first()
    if active_job:
        raise HTTPException(
            status_code=409,
            detail="Cannot reset TTS cache while a job is running. Stop it first.",
        )

    pm = ProjectManager(project_id)

    # Cache subdirs that store TTS outputs (per-input-hash subfolders).
    tts_cache_subdirs = ("segment_synthesis", "synthesized_speech")
    cache_files_removed = 0
    if pm.cache_dir.exists():
        for input_dir in pm.cache_dir.iterdir():
            if not input_dir.is_dir():
                continue
            for name in tts_cache_subdirs:
                target = input_dir / name
                if target.exists():
                    cache_files_removed += sum(1 for _ in target.rglob("*") if _.is_file())
                    shutil.rmtree(target, ignore_errors=True)

    # Artifact directories with rendered TTS audio.
    artifact_subdirs = ("audio", "audio_chunks", "su_audio_chunks", "previews")
    artifact_files_removed = 0
    for name in artifact_subdirs:
        target = pm.artifacts_dir / name
        if target.exists():
            artifact_files_removed += sum(1 for _ in target.rglob("*") if _.is_file())
            shutil.rmtree(target, ignore_errors=True)

    # Drop cached audio URLs on segments so the UI stops serving stale audio.
    segments_cleared = (
        db.query(Segment)
        .filter(Segment.project_id == project_id, Segment.audio_url.isnot(None))
        .update({Segment.audio_url: None}, synchronize_session=False)
    )

    # If the project was marked as dubbed, roll it back to transcribed.
    if project.status == ProjectStatus.DUBBED:
        project.status = ProjectStatus.TRANSCRIBED

    config = dict(project.config or {})
    config.pop("resultStats", None)
    project.config = config
    flag_modified(project, "config")
    project.updated_at = datetime.now(timezone.utc)
    db.commit()

    pm.ensure_directories()

    logger.info(
        "Reset TTS cache for %s: removed %s cache file(s), %s artifact file(s), "
        "cleared audioUrl on %s segment(s)",
        project_id, cache_files_removed, artifact_files_removed, segments_cleared,
    )

    return {
        "project_id": project_id,
        "cache_files_removed": cache_files_removed,
        "artifact_files_removed": artifact_files_removed,
        "segments_cleared": segments_cleared,
    }


@router.post("/{project_id}/process/dub", response_model=JobResponse, status_code=202)
async def start_dubbing(project_id: str, db: Session = Depends(get_db)):
    """Start final dubbing pipeline."""
    # Get project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check if project is transcribed
    if project.status not in [ProjectStatus.TRANSCRIBED, ProjectStatus.DUBBED, ProjectStatus.ERROR]:
        raise HTTPException(
            status_code=400,
            detail="Project must be transcribed before dubbing"
        )
    
    # Check if there are segments
    if not project.segments:
        raise HTTPException(status_code=400, detail="No segments found for dubbing")

    if is_speaker_gender_translation_stale(project.config or {}, project.segments):
        raise HTTPException(
            status_code=400,
            detail="Speaker gender overrides changed after translation. Re-run translation before dubbing.",
        )
    
    # Check for active dubbing job
    active_job = db.query(Job).filter(
        Job.project_id == project_id,
        Job.job_type == JobType.DUB,
        Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
    ).first()
    
    if active_job:
        return JobResponse(
            jobId=active_job.id,
            status=active_job.status.value,
            projectId=project_id,
            type="dub",
            progress=active_job.progress,
            currentStep=active_job.current_step,
        )
    
    # Create new job
    job = Job(
        id=generate_job_id("dub"),
        project_id=project_id,
        job_type=JobType.DUB,
        status=JobStatus.PENDING,
    )
    db.add(job)
    
    # Update project status
    project.status = ProjectStatus.DUBBING
    project.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    
    # Dispatch Celery task
    task = dub_project.delay(project_id, job.id)
    
    # Update job with celery task ID
    job.celery_task_id = task.id
    db.commit()
    
    return JobResponse(
        jobId=job.id,
        status="processing",
        projectId=project_id,
        type="dub",
        progress=0,
        currentStep="pending",
    )


@router.get("/{project_id}/jobs", response_model=list)
async def list_jobs(project_id: str, db: Session = Depends(get_db)):
    """List all jobs for a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    jobs = db.query(Job).filter(Job.project_id == project_id).order_by(Job.created_at.desc()).all()
    
    return [
        JobResponse(
            jobId=job.id,
            status=job.status.value,
            projectId=project_id,
            type=job.job_type.value,
            progress=job.progress,
            currentStep=job.current_step,
        )
        for job in jobs
    ]


@router.post("/{project_id}/process/stop")
async def stop_processing(project_id: str, db: Session = Depends(get_db)):
    """Stop any active processing for a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Find active jobs
    active_jobs = db.query(Job).filter(
        Job.project_id == project_id,
        Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
    ).all()
    
    if not active_jobs:
        # Still clear any queued auto-processing intent even if nothing is running
        config = dict(project.config) if project.config else {}
        if config.get("autoProcess"):
            config["autoProcess"] = False
            project.config = config
            flag_modified(project, "config")
            project.updated_at = datetime.now(timezone.utc)
            db.commit()
        return {"message": "No active processing to stop", "stopped": False}
    
    stopped_count = 0
    for job in active_jobs:
        # Revoke Celery task if it has one
        if job.celery_task_id:
            celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGTERM")
        
        # Update job status
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        job.add_log("Processing stopped by user", "info")
        stopped_count += 1
    
    # Reset project status to draft (or transcribed if it was dubbing)
    if project.status == ProjectStatus.DUBBING:
        project.status = ProjectStatus.TRANSCRIBED
    elif project.status == ProjectStatus.TRANSCRIBING:
        project.status = ProjectStatus.DRAFT
    elif project.status == ProjectStatus.DOWNLOADING:
        project.status = ProjectStatus.DRAFT

    # Clear queued auto-processing intent
    config = dict(project.config) if project.config else {}
    if config.get("autoProcess"):
        config["autoProcess"] = False
        project.config = config
        flag_modified(project, "config")

    project.updated_at = datetime.now(timezone.utc)
    db.commit()

    
    return {"message": f"Stopped {stopped_count} job(s)", "stopped": True}
