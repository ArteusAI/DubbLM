"""One-shot external translate workflow for simplified API clients."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import UploadFile
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database.models import Project
from ..models.schemas import ExternalTranslateResponse
from .job_dispatch import enqueue_transcription_job
from .project_manager import ProjectManager
from .video_upload import save_project_video

PresetType = Literal["fast", "hq", "ultra"]


def build_external_translate_config(
    target_lang: str,
    preset: PresetType,
    *,
    source_lang: str = "auto",
    minimal_diarization_merge: bool = False,
    keep_background: bool = False,
    enable_llm_editor: bool = False,
    speaker_count: int = 1,
    persona_id: Optional[str] = None,
) -> dict:
    """Build minimal project config for external translate requests."""
    config = {
        "targetLang": target_lang,
        "sourceLang": source_lang,
        "preset": preset,
        "autoProcess": True,
        "keepBackground": keep_background,
        "enableLlmEditor": enable_llm_editor,
        "speakerCount": speaker_count,
    }
    if persona_id is not None:
        config["personaId"] = persona_id
    if minimal_diarization_merge:
        config["postDiarizationMergeGap"] = 0
        config["repairSpeakerFragmentation"] = False
    return config


def _default_project_name(filename: str) -> str:
    """Generate a project name from the uploaded filename."""
    stem = Path(filename).stem or "video"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{timestamp}"


async def start_external_translate(
    db: Session,
    file: UploadFile,
    target_lang: str,
    *,
    preset: PresetType = "hq",
    source_lang: str = "auto",
    minimal_diarization_merge: bool = False,
    keep_background: bool = False,
    enable_llm_editor: bool = False,
    speaker_count: int = 1,
    persona_id: Optional[str] = None,
    name: Optional[str] = None,
) -> ExternalTranslateResponse:
    """Create a project, upload video, configure, and start full translate pipeline."""
    if not file.filename:
        raise ValueError("Uploaded file must have a filename")

    project = Project(name=name or _default_project_name(file.filename))
    project.config = build_external_translate_config(
        target_lang=target_lang,
        preset=preset,
        source_lang=source_lang,
        minimal_diarization_merge=minimal_diarization_merge,
        keep_background=keep_background,
        enable_llm_editor=enable_llm_editor,
        speaker_count=speaker_count,
        persona_id=persona_id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    pm = ProjectManager(project.id)
    pm.ensure_directories()

    saved_video = await save_project_video(project.id, file)
    project.source_file = str(saved_video.upload_path)
    project.source_filename = saved_video.safe_filename
    project.source_size = saved_video.file_size
    project.updated_at = datetime.now(timezone.utc)
    db.commit()

    job_response = enqueue_transcription_job(db, project, allow_existing=False)

    settings = get_settings()
    api_prefix = settings.api_prefix.rstrip("/")
    project_path = f"{api_prefix}/projects/{project.id}"

    return ExternalTranslateResponse(
        projectId=project.id,
        jobId=job_response.jobId,
        status=job_response.status,
        pollUrl=f"{project_path}/jobs/{job_response.jobId}/status",
        downloadUrl=f"{project_path}/download/video",
    )
