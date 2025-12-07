#!/usr/bin/env python3
"""Run Celery worker for DubbLM background tasks."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from src.api.workers.celery_app import celery_app
    
    # Default arguments for celery worker
    worker_args = [
        "worker",
        "--loglevel=INFO",
        "--concurrency=2",
        "-Q", "default,heavy,light",  # Process all queues
    ]
    
    # Add any command line arguments
    if len(sys.argv) > 1:
        worker_args.extend(sys.argv[1:])
    
    celery_app.worker_main(argv=worker_args)

