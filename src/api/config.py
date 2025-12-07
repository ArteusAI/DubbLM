"""API configuration settings."""

import os
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # API settings
    api_title: str = "DubbLM API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False
    
    # Database settings
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./dubblm.db"))
    
    # Redis settings for Celery
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    celery_broker_url: str = field(default_factory=lambda: os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"))
    celery_result_backend: str = field(default_factory=lambda: os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"))
    
    # Project storage settings
    projects_dir: Path = field(default_factory=lambda: Path(os.getenv("PROJECTS_DIR", "./projects")).resolve())
    max_upload_size: int = 5 * 1024 * 1024 * 1024  # 5GB
    
    # Authentication (placeholder for future implementation)
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"))


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

