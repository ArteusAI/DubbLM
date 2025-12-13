"""Processing routes for transcription and dubbing."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from ..database.session import get_db
from ..database.models import Project, Job, JobType, JobStatus, ProjectStatus, generate_job_id
from ..models.schemas import JobResponse
from ..services.project_manager import ProjectManager
from ..workers.tasks import transcribe_project, dub_project
from ..workers.celery_app import celery_app

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
    
    # Check if there's already an active transcription job
    active_job = db.query(Job).filter(
        Job.project_id == project_id,
        Job.job_type == JobType.TRANSCRIBE,
        Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
    ).first()
    
    if active_job:
        return JobResponse(
            jobId=active_job.id,
            status=active_job.status.value,
            projectId=project_id,
            type="transcribe",
            progress=active_job.progress,
            currentStep=active_job.current_step,
        )
    
    # Create new job
    job = Job(
        id=generate_job_id("transcribe"),
        project_id=project_id,
        job_type=JobType.TRANSCRIBE,
        status=JobStatus.PENDING,
    )
    db.add(job)
    
    # Update project status
    project.status = ProjectStatus.TRANSCRIBING
    project.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    
    # Dispatch Celery task
    task = transcribe_project.delay(project_id, job.id)
    
    # Update job with celery task ID
    job.celery_task_id = task.id
    db.commit()
    
    return JobResponse(
        jobId=job.id,
        status="processing",
        projectId=project_id,
        type="transcribe",
        progress=0,
        currentStep="pending",
    )


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
    
    project.updated_at = datetime.now(timezone.utc)
    db.commit()
    
    return {"message": f"Stopped {stopped_count} job(s)", "stopped": True}

