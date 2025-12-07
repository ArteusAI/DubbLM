"""Celery application configuration."""

import os
from celery import Celery

from ..config import get_settings


def create_celery_app() -> Celery:
    """Create and configure Celery application."""
    settings = get_settings()
    
    app = Celery(
        "dubblm_workers",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["src.api.workers.tasks"],
    )
    
    app.conf.update(
        # Task settings
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        
        # Task execution settings
        task_track_started=True,
        task_time_limit=3600 * 4,  # 4 hour hard limit
        task_soft_time_limit=3600 * 3,  # 3 hour soft limit
        
        # Result settings
        result_expires=86400,  # 24 hours
        
        # Worker settings
        worker_prefetch_multiplier=1,  # One task at a time for resource-heavy tasks
        worker_concurrency=2,  # Number of worker processes
        
        # Task routes for different queues
        task_routes={
            "src.api.workers.tasks.transcribe_project": {"queue": "heavy"},
            "src.api.workers.tasks.dub_project": {"queue": "heavy"},
            "src.api.workers.tasks.generate_preview": {"queue": "light"},
            "src.api.workers.tasks.rephrase_segment": {"queue": "light"},
        },
        
        # Default queue
        task_default_queue="default",
    )
    
    return app


celery_app = create_celery_app()

