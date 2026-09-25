"""API configuration settings."""

import os
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field


def _parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer from a string value."""
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_optional_path(value: str | None) -> Path | None:
    """Parse an optional file path from a string value."""
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    return None


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
    keep_debug_artifacts: bool = field(
        default_factory=lambda: os.getenv("KEEP_DEBUG_ARTIFACTS", "false").lower() in ("1", "true", "yes")
    )

    # S3 / Hetzner Object Storage (optional)
    s3_endpoint_url: str | None = field(default_factory=lambda: os.getenv("S3_ENDPOINT_URL") or None)
    s3_access_key: str | None = field(default_factory=lambda: os.getenv("S3_ACCESS_KEY") or None)
    s3_secret_key: str | None = field(default_factory=lambda: os.getenv("S3_SECRET_KEY") or None)
    s3_bucket: str | None = field(default_factory=lambda: os.getenv("S3_BUCKET") or None)
    s3_region: str = field(default_factory=lambda: os.getenv("S3_REGION", "fsn1"))
    s3_prefix: str = field(default_factory=lambda: (os.getenv("S3_PREFIX") or "dubblm").strip().strip("/"))
    s3_enabled: bool = field(
        default_factory=lambda: os.getenv("S3_ENABLED", "true").lower() in ("1", "true", "yes")
    )
    s3_presign_expires: int = field(
        default_factory=lambda: int(os.getenv("S3_PRESIGN_EXPIRES", str(7 * 24 * 3600)))  # 7 days
    )
    
    # Video download settings (yt-dlp)
    max_download_size: int = field(default_factory=lambda: int(os.getenv("MAX_DOWNLOAD_SIZE", str(8 * 1024 * 1024 * 1024))))  # 8GB
    max_download_duration: int | None = field(default_factory=lambda: _parse_optional_int(os.getenv("MAX_DOWNLOAD_DURATION")))
    video_download_cookies_path: Path | None = field(default_factory=lambda: _parse_optional_path(os.getenv("VIDEO_DOWNLOAD_COOKIES_PATH")))
    
    # Authentication (placeholder for future implementation)
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"))


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

