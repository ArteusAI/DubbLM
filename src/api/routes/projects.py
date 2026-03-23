"""Project management routes."""

from typing import List
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from ..database.session import get_db
from ..database.models import Project, Segment, ProjectStatus
from ..models.schemas import (
    ProjectCreate,
    ProjectResponse,
    ProjectListResponse,
    ProjectConfig,
    ProjectConfigUpdate,
    SegmentResponse,
)
from ..services.project_manager import ProjectManager

router = APIRouter(prefix="/projects", tags=["projects"])


def _project_to_response(project: Project, include_segments: bool = False) -> dict:
    """Convert project model to response dict."""
    config = project.config or {}
    response = {
        "id": project.id,
        "name": project.name,
        "status": project.status.value,
        "createdAt": project.created_at,
        "updatedAt": project.updated_at,
        "config": ProjectConfig(
            sourceLang=config.get("sourceLang"),
            targetLang=config.get("targetLang"),
            personaId=config.get("personaId"),
            speakerCount=config.get("speakerCount"),
            keepBackground=config.get("keepBackground", False),
            pauseRemoval=config.get("pauseRemoval", "disabled"),
            preset=config.get("preset", "hq"),
            apiKeys=None,  # Never expose API keys
            autoProcess=config.get("autoProcess"),
            # LLM settings
            llmProvider=config.get("llmProvider"),
            llmModelName=config.get("llmModelName"),
            llmTemperature=config.get("llmTemperature"),
            speakerTtsPrompts=config.get("speakerTtsPrompts"),
            speakerVoiceMappings=config.get("speakerVoiceMappings"),
            refinementLlmProvider=config.get("refinementLlmProvider"),
            refinementModelName=config.get("refinementModelName"),
            refinementTemperature=config.get("refinementTemperature"),
            translationPromptPrefix=config.get("translationPromptPrefix"),
            # TTS settings
            ttsSystem=config.get("ttsSystem"),
            ttsModel=config.get("ttsModel"),
            ttsPromptPrefix=config.get("ttsPromptPrefix"),
            voiceAutoSelection=config.get("voiceAutoSelection"),
            enableEmotionEnrichment=config.get("enableEmotionEnrichment"),
            # Audio settings
            dubbedVolume=config.get("dubbedVolume"),
            backgroundVolume=config.get("backgroundVolume"),
            keepOriginalAudioRanges=config.get("keepOriginalAudioRanges"),
            useTwoPassEncoding=config.get("useTwoPassEncoding"),
            videoQualityPreset=config.get("videoQualityPreset"),
            maxWorkers=config.get("maxWorkers"),
            # Processing settings
            startTime=config.get("startTime"),
            duration=config.get("duration"),
            transcriptionSystem=config.get("transcriptionSystem"),
            whisperModel=config.get("whisperModel"),
            # Segment optimization
            postDiarizationMergeGap=config.get("postDiarizationMergeGap"),
            postTranslationMergeGap=config.get("postTranslationMergeGap"),
            maxSegmentDuration=config.get("maxSegmentDuration"),
            minSegmentDuration=config.get("minSegmentDuration"),
            comfortMinAdjustmentRatio=config.get("comfortMinAdjustmentRatio"),
            comfortMaxAdjustmentRatio=config.get("comfortMaxAdjustmentRatio"),
            minPauseDuration=config.get("minPauseDuration"),
            preservePauseDuration=config.get("preservePauseDuration"),
            segmentStretch=config.get("segmentStretch"),
            geminiMultiSpeakerEnabled=config.get("geminiMultiSpeakerEnabled"),
            geminiMultiSpeakerMaxBatchTokens=config.get("geminiMultiSpeakerMaxBatchTokens"),
            geminiMultiSpeakerMaxTurns=config.get("geminiMultiSpeakerMaxTurns"),
            geminiMultiSpeakerPauseRepeats=config.get("geminiMultiSpeakerPauseRepeats"),
            geminiMultiSpeakerMinPauseMs=config.get("geminiMultiSpeakerMinPauseMs"),
        ),
        "sourceFile": project.source_file,
        "sourceFilename": project.source_filename,
        "sourceSize": project.source_size,
    }
    
    if include_segments:
        response["segments"] = [
            SegmentResponse(**seg.to_dict()) 
            for seg in project.segments
        ]
    
    return response


@router.get("", response_model=List[ProjectListResponse])
async def list_projects(db: Session = Depends(get_db)):
    """Get list of all projects."""
    projects = db.query(Project).order_by(Project.updated_at.desc()).all()
    return [_project_to_response(p) for p in projects]


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(project_data: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project."""
    project = Project(name=project_data.name)
    db.add(project)
    db.commit()
    db.refresh(project)
    
    # Initialize project directories
    pm = ProjectManager(project.id)
    pm.ensure_directories()
    
    return _project_to_response(project)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db: Session = Depends(get_db)):
    """Get project details including segments."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return _project_to_response(project, include_segments=True)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, db: Session = Depends(get_db)):
    """Delete a project and all associated data."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Delete project files
    ProjectManager.cleanup_project(project_id)
    
    # Delete from database
    db.delete(project)
    db.commit()
    
    return None


@router.patch("/{project_id}/config", response_model=ProjectResponse)
async def update_project_config(
    project_id: str,
    config_update: ProjectConfigUpdate,
    db: Session = Depends(get_db)
):
    """Update project configuration."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get current config
    current_config = project.config or {}
    
    # Update only provided fields
    update_data = config_update.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        if value is not None:
            current_config[key] = value
    
    # Explicitly mark config as modified (required for JSON columns in SQLAlchemy)
    project.config = current_config
    flag_modified(project, "config")
    project.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(project)
    
    
    return _project_to_response(project)
