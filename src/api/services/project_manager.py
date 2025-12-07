"""Project directory management service for artifact isolation."""

import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Optional

from ..config import get_settings


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by replacing problematic characters.
    
    Handles non-breaking spaces, unicode dashes, and other problematic chars.
    """
    # Replace non-breaking spaces and other whitespace with regular spaces
    filename = filename.replace('\xa0', ' ')  # Non-breaking space
    filename = filename.replace('\u00a0', ' ')  # Unicode non-breaking space
    
    # Replace unicode dashes with regular hyphen
    filename = filename.replace('\u2013', '-')  # En-dash –
    filename = filename.replace('\u2014', '-')  # Em-dash —
    filename = filename.replace('\u2212', '-')  # Minus sign −
    
    # Replace other problematic unicode characters
    filename = filename.replace('\u2018', "'")  # Left single quote '
    filename = filename.replace('\u2019', "'")  # Right single quote '
    filename = filename.replace('\u201c', '"')  # Left double quote "
    filename = filename.replace('\u201d', '"')  # Right double quote "
    
    # Normalize unicode characters
    filename = unicodedata.normalize('NFKC', filename)
    
    # Replace multiple spaces with single space
    filename = re.sub(r'\s+', ' ', filename)
    
    # Strip leading/trailing spaces
    filename = filename.strip()
    
    return filename


class ProjectManager:
    """Manages project directories and artifact isolation."""
    
    def __init__(self, project_id: str):
        """Initialize project manager for a specific project."""
        self.project_id = project_id
        self.settings = get_settings()
        self.base_dir = self.settings.projects_dir / project_id
    
    @property
    def uploads_dir(self) -> Path:
        """Directory for uploaded source files."""
        return self.base_dir / "uploads"
    
    @property
    def artifacts_dir(self) -> Path:
        """Directory for pipeline artifacts (audio/, debug/, speakers_audio/)."""
        return self.base_dir / "artifacts"
    
    @property
    def cache_dir(self) -> Path:
        """Directory for cache files."""
        return self.base_dir / "cache"
    
    @property
    def results_dir(self) -> Path:
        """Directory for final results (dubbed video, subtitles)."""
        return self.base_dir / "results"
    
    @property
    def audio_dir(self) -> Path:
        """Directory for audio artifacts."""
        return self.artifacts_dir / "audio"
    
    @property
    def speakers_audio_dir(self) -> Path:
        """Directory for speaker audio samples."""
        return self.artifacts_dir / "speakers_audio"
    
    @property
    def debug_dir(self) -> Path:
        """Directory for debug artifacts."""
        return self.artifacts_dir / "debug"
    
    def ensure_directories(self) -> None:
        """Create all required project directories."""
        directories = [
            self.base_dir,
            self.uploads_dir,
            self.artifacts_dir,
            self.cache_dir,
            self.results_dir,
            self.audio_dir,
            self.speakers_audio_dir,
            self.debug_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_upload_path(self, filename: str) -> Path:
        """Get path for an uploaded file."""
        self.ensure_directories()
        safe_filename = sanitize_filename(filename)
        return self.uploads_dir / safe_filename
    
    def get_source_video_path(self) -> Optional[Path]:
        """Get path to the uploaded source video."""
        if not self.uploads_dir.exists():
            return None
        videos = list(self.uploads_dir.glob("*.*"))
        return videos[0] if videos else None
    
    def get_result_video_path(self, target_lang: str) -> Path:
        """Get path for the result video."""
        self.ensure_directories()
        source = self.get_source_video_path()
        if source:
            return self.results_dir / f"{source.stem}_{target_lang}{source.suffix}"
        return self.results_dir / f"output_{target_lang}.mp4"
    
    def get_result_subtitles_path(self, target_lang: str, sub_type: str = "srt") -> Path:
        """Get path for result subtitles."""
        self.ensure_directories()
        source = self.get_source_video_path()
        if source:
            return self.results_dir / f"{source.stem}_{target_lang}.{sub_type}"
        return self.results_dir / f"output_{target_lang}.{sub_type}"
    
    def get_segment_audio_path(self, segment_id: str) -> Path:
        """Get path for a segment's cached audio."""
        return self.audio_dir / f"{segment_id}.wav"
    
    def get_preview_audio_path(self, segment_id: str, voice_id: str | None = None) -> Path:
        """Get path for segment preview audio, optionally per voice."""
        preview_dir = self.artifacts_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        if voice_id:
            # Sanitize voice_id for filename
            safe_voice_id = voice_id.replace("/", "_").replace(":", "_")
            return preview_dir / f"{segment_id}_{safe_voice_id}_preview.mp3"
        return preview_dir / f"{segment_id}_preview.mp3"
    
    def cleanup(self) -> None:
        """Remove all project files and directories."""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
    
    def cleanup_artifacts(self) -> None:
        """Remove only artifact files (keep uploads and results)."""
        if self.artifacts_dir.exists():
            shutil.rmtree(self.artifacts_dir)
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
    
    def get_dubbing_config_overrides(self) -> dict:
        """Get configuration overrides for SmartDubbing to use project directories."""
        self.ensure_directories()
        return {
            # Override artifact paths
            "_project_base_dir": str(self.base_dir),
            "_artifacts_dir": str(self.artifacts_dir),
            "_cache_dir": str(self.cache_dir),
            "_results_dir": str(self.results_dir),
        }
    
    def list_artifacts(self) -> dict:
        """List all artifacts in the project directory."""
        result = {
            "uploads": [],
            "artifacts": [],
            "cache": [],
            "results": [],
        }
        
        if self.uploads_dir.exists():
            result["uploads"] = [f.name for f in self.uploads_dir.iterdir() if f.is_file()]
        
        if self.artifacts_dir.exists():
            result["artifacts"] = [f.name for f in self.artifacts_dir.rglob("*") if f.is_file()]
        
        if self.cache_dir.exists():
            result["cache"] = [f.name for f in self.cache_dir.rglob("*") if f.is_file()]
        
        if self.results_dir.exists():
            result["results"] = [f.name for f in self.results_dir.iterdir() if f.is_file()]
        
        return result
    
    @classmethod
    def create_for_project(cls, project_id: str) -> "ProjectManager":
        """Factory method to create and initialize a ProjectManager."""
        manager = cls(project_id)
        manager.ensure_directories()
        return manager
    
    @classmethod
    def cleanup_project(cls, project_id: str) -> None:
        """Static method to cleanup a project by ID."""
        manager = cls(project_id)
        manager.cleanup()

