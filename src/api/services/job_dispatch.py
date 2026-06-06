"""Shared job dispatch helpers for processing routes and external API."""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..database.models import Job, JobType, JobStatus, Project, ProjectStatus, generate_job_id
from ..models.schemas import JobResponse


def enqueue_transcription_job(
    db: Session,
    project: Project,
    *,
    allow_existing: bool = True,
) -> JobResponse:
    """Create and dispatch a transcription job for a project."""
    if allow_existing:
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
        id=generate_job_id("transcribe"),
        project_id=project.id,
        job_type=JobType.TRANSCRIBE,
        status=JobStatus.PENDING,
    )
    db.add(job)

    project.status = ProjectStatus.TRANSCRIBING
    project.updated_at = datetime.now(timezone.utc)
    db.commit()

    from ..workers.tasks import transcribe_project

    task = transcribe_project.delay(project.id, job.id)
    job.celery_task_id = task.id
    db.commit()

    return JobResponse(
        jobId=job.id,
        status="processing",
        projectId=project.id,
        type="transcribe",
        progress=0,
        currentStep="pending",
    )
