"""SSE status streaming routes."""

import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database.session import get_db, get_db_session
from ..database.models import Project, Job, JobStatus

router = APIRouter(prefix="/projects", tags=["status"])


async def generate_status_events(project_id: str) -> AsyncGenerator[str, None]:
    """Generate SSE events for project status updates."""
    last_progress = -1
    last_log_count = 0
    last_status = None
    poll_interval = 1.0  # seconds
    max_idle_time = 600  # 10 minutes (increased for long operations)
    idle_time = 0
    heartbeat_interval = 15  # Send heartbeat every 15 seconds
    polls_since_heartbeat = 0
    
    while True:
        db = get_db_session()
        try:
            # Get latest active job for project
            job = db.query(Job).filter(
                Job.project_id == project_id,
                Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
            ).order_by(Job.created_at.desc()).first()
            
            if not job:
                # Check if there's a recently completed job
                job = db.query(Job).filter(
                    Job.project_id == project_id,
                    Job.status.in_([JobStatus.COMPLETED, JobStatus.FAILED])
                ).order_by(Job.completed_at.desc()).first()
                
                if job:
                    # Send completion event
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project:
                        event_data = {
                            "status": project.status.value,
                            "jobStatus": job.status.value,
                        }
                        if job.status == JobStatus.FAILED:
                            event_data["error"] = job.error_message
                        
                        yield f"event: complete\ndata: {json.dumps(event_data)}\n\n"
                        return
                
                # No active or recent job, wait a bit
                idle_time += poll_interval
                if idle_time >= max_idle_time:
                    yield f"event: timeout\ndata: {json.dumps({'message': 'No activity'})}\n\n"
                    return
                
                await asyncio.sleep(poll_interval)
                continue
            
            # Reset idle time when we have an active job
            idle_time = 0
            polls_since_heartbeat += 1
            
            # Send heartbeat to keep connection alive during long operations
            if polls_since_heartbeat >= heartbeat_interval:
                yield f": heartbeat\n\n"
                polls_since_heartbeat = 0
            
            # Send progress update if changed
            if job.progress != last_progress:
                progress_event = {
                    "percent": job.progress,
                    "step": job.current_step or "processing",
                }
                yield f"event: progress\ndata: {json.dumps(progress_event)}\n\n"
                last_progress = job.progress
                polls_since_heartbeat = 0  # Reset heartbeat counter on activity
            
            # Send new logs
            logs = job.logs or []
            if len(logs) > last_log_count:
                new_logs = logs[last_log_count:]
                for log in new_logs:
                    yield f"event: log\ndata: {json.dumps(log)}\n\n"
                last_log_count = len(logs)
                polls_since_heartbeat = 0  # Reset heartbeat counter on activity
            
            # Check for status change
            if job.status.value != last_status:
                if job.status == JobStatus.COMPLETED:
                    project = db.query(Project).filter(Project.id == project_id).first()
                    yield f"event: complete\ndata: {json.dumps({'status': project.status.value if project else 'completed'})}\n\n"
                    return
                elif job.status == JobStatus.FAILED:
                    yield f"event: complete\ndata: {json.dumps({'status': 'error', 'error': job.error_message})}\n\n"
                    return
                last_status = job.status.value
            
        finally:
            db.close()
        
        await asyncio.sleep(poll_interval)


@router.get("/{project_id}/status")
async def get_project_status(project_id: str, db: Session = Depends(get_db)):
    """Stream project status updates via SSE."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return StreamingResponse(
        generate_status_events(project_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/{project_id}/jobs/{job_id}/status")
async def get_job_status(project_id: str, job_id: str, db: Session = Depends(get_db)):
    """Get status of a specific job (non-streaming)."""
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.project_id == project_id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "jobId": job.id,
        "projectId": job.project_id,
        "type": job.job_type.value,
        "status": job.status.value,
        "progress": job.progress,
        "currentStep": job.current_step,
        "logs": job.logs or [],
        "errorMessage": job.error_message,
        "createdAt": job.created_at.isoformat() if job.created_at else None,
        "startedAt": job.started_at.isoformat() if job.started_at else None,
        "completedAt": job.completed_at.isoformat() if job.completed_at else None,
    }

