#!/usr/bin/env python3
"""Run the DubbLM API server."""

import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Check if running in Docker (no reload) or development (with reload)
    is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER", "0") == "1"
    
    uvicorn.run(
        "src.api.main:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        reload=not is_docker,
        log_level=os.environ.get("LOG_LEVEL", "info"),
        workers=int(os.environ.get("API_WORKERS", "1")) if is_docker else 1,
    )

