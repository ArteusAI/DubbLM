"""SQLAlchemy database models."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.orm import relationship, DeclarativeBase
import enum


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class ProjectStatus(str, enum.Enum):
    """Project status enumeration."""
    DRAFT = "draft"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    DUBBING = "dubbing"
    DUBBED = "dubbed"
    ERROR = "error"


class JobType(str, enum.Enum):
    """Job type enumeration."""
    TRANSCRIBE = "transcribe"
    DUB = "dub"
    PREVIEW = "preview"
    REPHRASE = "rephrase"


class JobStatus(str, enum.Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TTSProvider(str, enum.Enum):
    """TTS provider enumeration."""
    OPENAI = "openai"
    GEMINI = "gemini"
    COQUI = "coqui"
    MINIMAX = "minimax"


def generate_uuid() -> str:
    """Generate a prefixed UUID for project IDs."""
    return f"proj_{uuid.uuid4().hex[:12]}"


def generate_segment_id() -> str:
    """Generate a prefixed UUID for segment IDs."""
    return f"seg_{uuid.uuid4().hex[:12]}"


def generate_job_id(job_type: str) -> str:
    """Generate a prefixed UUID for job IDs."""
    return f"job_{job_type}_{uuid.uuid4().hex[:8]}"


class Project(Base):
    """Project model representing a dubbing project."""
    
    __tablename__ = "projects"
    
    id = Column(String(32), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    status = Column(SQLEnum(ProjectStatus), default=ProjectStatus.DRAFT, nullable=False)
    
    # File info
    source_file = Column(String(512), nullable=True)
    source_filename = Column(String(255), nullable=True)
    source_size = Column(Integer, nullable=True)
    
    # Configuration stored as JSON
    config = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                       onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    segments = relationship("Segment", back_populates="project", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="project", cascade="all, delete-orphan")
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
            "config": self.config or {},
        }


class Segment(Base):
    """Segment model representing a translated segment."""
    
    __tablename__ = "segments"
    
    id = Column(String(32), primary_key=True, default=generate_segment_id)
    project_id = Column(String(32), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    
    # Speaker info
    speaker = Column(String(64), nullable=False)
    speaker_color = Column(String(32), default="#3B82F6")
    
    # Timing
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    
    # Text content
    original_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=True)
    
    # TTS settings
    voice_id = Column(String(64), nullable=True)
    provider = Column(SQLEnum(TTSProvider), nullable=True)
    tts_prompt = Column(Text, nullable=True)
    
    # State
    is_muted = Column(Boolean, default=False)
    audio_url = Column(String(512), nullable=True)
    
    # Order for maintaining sequence
    sequence = Column(Integer, nullable=False, default=0)
    
    # Relationship
    project = relationship("Project", back_populates="segments")
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "projectId": self.project_id,
            "speaker": self.speaker,
            "speakerColor": self.speaker_color,
            "startTime": self.start_time,
            "endTime": self.end_time,
            "originalText": self.original_text,
            "translatedText": self.translated_text,
            "voiceId": self.voice_id,
            "provider": self.provider.value if self.provider else None,
            "ttsPrompt": self.tts_prompt,
            "isMuted": self.is_muted,
            "audioUrl": self.audio_url,
        }


class Job(Base):
    """Job model for tracking async processing tasks."""
    
    __tablename__ = "jobs"
    
    id = Column(String(32), primary_key=True)
    project_id = Column(String(32), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    
    # Job info
    job_type = Column(SQLEnum(JobType), nullable=False)
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False)
    
    # Progress tracking
    progress = Column(Integer, default=0)
    current_step = Column(String(64), nullable=True)
    
    # Logs stored as JSON array
    logs = Column(JSON, default=list)
    
    # Error info
    error_message = Column(Text, nullable=True)
    
    # Celery task ID for tracking
    celery_task_id = Column(String(64), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationship
    project = relationship("Project", back_populates="jobs")
    
    def add_log(self, message: str, log_type: str = "info", text: str = None) -> None:
        """Add a log entry to the job."""
        if self.logs is None:
            self.logs = []
        log_entry = {
            "id": f"log_{len(self.logs)}",
            "message": message,
            "type": log_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if text:
            log_entry["text"] = text
        self.logs = self.logs + [log_entry]
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            "jobId": self.id,
            "projectId": self.project_id,
            "type": self.job_type.value,
            "status": self.status.value,
            "progress": self.progress,
            "currentStep": self.current_step,
            "logs": self.logs or [],
            "errorMessage": self.error_message,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
        }

