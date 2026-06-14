"""Celery tasks for background processing."""

import os
import sys
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Literal, cast

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from sqlalchemy.orm.attributes import flag_modified

from .celery_app import celery_app
from ..database.session import get_db_session
from ..database.models import Project, Segment, Job, JobStatus, ProjectStatus, JobType
from ..services.project_manager import ProjectManager
from ..services.settings_service import get_api_key, apply_api_keys_to_env, API_KEY_PROVIDERS
from ..services.preset_service import get_preset_config
from ..services.segment_optimization_config import apply_segment_optimization_config
from src.dubbing.tts_styles import get_blocked_voices_for_style, resolve_tts_prompt_prefix
from src.dubbing.debug.cost_ledger import import_cost_snapshots, write_cost_snapshot_from_tracker
from src.dubbing.audio.speaker_gender_inferencer import SpeakerGenderInferencer
from src.utils.speaker_gender import (
    SPEAKER_GENDER_SIGNATURE_CONFIG_KEY,
    SPEAKER_METADATA_CONFIG_KEY,
    build_effective_gender_prompt_section,
    build_translation_gender_signature,
    normalize_bool,
    normalize_speaker_metadata_map,
)

logger = logging.getLogger(__name__)


VideoQualityPreset = Literal["720p", "1080p", "original"]

VIDEO_QUALITY_MAX_HEIGHT: Dict[VideoQualityPreset, Optional[int]] = {
    "720p": 720,
    "1080p": 1080,
    "original": None,
}

def get_video_quality_preset(config_data: Dict[str, Any], preset_config: Dict[str, Any]) -> VideoQualityPreset:
    """Resolve effective video quality preset from project config and preset defaults."""
    requested = config_data.get("videoQualityPreset") or preset_config.get("video_quality_preset") or "original"
    if requested in VIDEO_QUALITY_MAX_HEIGHT:
        return cast(VideoQualityPreset, requested)

    logger.warning(f"Unknown videoQualityPreset={requested!r}, falling back to 'original'")
    return "original"


def _resolve_editor_runtime_config(config_data: Dict[str, Any], preset_config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve editor settings with OpenRouter fallback when the API key is unavailable."""
    resolved = {
        "enable_llm_editor": config_data.get("enableLlmEditor"),
        "editor_llm_provider": config_data.get("editorLlmProvider") or preset_config.get("editor_llm_provider"),
        "editor_model_name": config_data.get("editorModelName") or preset_config.get("editor_model_name"),
        "editor_temperature": config_data.get("editorTemperature") if config_data.get("editorTemperature") is not None else preset_config.get("editor_temperature", 1.0),
        "editor_reasoning_effort": config_data.get("editorReasoningEffort") or preset_config.get("editor_reasoning_effort"),
    }
    if resolved["enable_llm_editor"] is None:
        resolved["enable_llm_editor"] = preset_config.get("enable_llm_editor", False)

    if resolved["editor_llm_provider"] == "openrouter" and not get_api_key("openrouter"):
        fallback_provider = config_data.get("refinementLlmProvider") or preset_config.get("refinement_llm_provider") or config_data.get("llmProvider") or preset_config.get("llm_provider")
        fallback_model = config_data.get("refinementModelName") or preset_config.get("refinement_model_name") or config_data.get("llmModelName") or preset_config.get("llm_model_name")
        resolved["editor_llm_provider"] = fallback_provider
        resolved["editor_model_name"] = fallback_model
        resolved["editor_reasoning_effort"] = "none"

    return resolved


def _resolve_refinement_persona(config_data: Dict[str, Any], preset_config: Dict[str, Any]) -> str:
    """Resolve refinement persona, allowing presets to disable refinement."""
    value = config_data.get("personaId")
    if value is None or str(value).strip() == "":
        value = preset_config.get("persona_id", "normal")
    return str(value or "normal")


def _resolve_bool_config(
    config_data: Dict[str, Any],
    config_key: str,
    preset_config: Dict[str, Any],
    preset_key: str,
    default: bool,
) -> bool:
    value = config_data.get(config_key)
    if value is None:
        value = preset_config.get(preset_key, default)
    return normalize_bool(value)


def _apply_api_keys(project_api_keys: Optional[Dict[str, str]] = None) -> None:
    """Apply API keys from settings and optionally project-specific overrides."""
    # First apply system-wide settings
    apply_api_keys_to_env()
    
    # Then apply project-specific overrides
    if project_api_keys:
        for provider, key in project_api_keys.items():
            if key and provider in API_KEY_PROVIDERS:
                os.environ[API_KEY_PROVIDERS[provider]] = key


def _normalize_speaker_voice_mappings(value: Any) -> Dict[str, str]:
    """Normalize speaker->voice mapping payload from project config."""
    if not isinstance(value, dict):
        return {}

    normalized: Dict[str, str] = {}
    for speaker_name, voice_id in value.items():
        speaker = str(speaker_name).strip()
        voice = str(voice_id).strip() if voice_id is not None else ""
        if speaker and voice:
            normalized[speaker] = voice
    return normalized


def _build_unknown_speaker_metadata(
    speakers_rolls: Dict[Any, str],
    existing_metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build a safe fallback metadata map with unknown genders for every discovered speaker."""
    fallback = dict(existing_metadata)
    for speaker in sorted({str(value).strip() for value in speakers_rolls.values() if str(value).strip()}):
        entry = dict(fallback.get(speaker) or {})
        if not entry:
            entry = {
                "inferredGender": "unknown",
                "inferredConfidence": 0.0,
                "rawLabel": "fallback_unknown",
                "modelId": None,
                "overrideGender": None,
            }
        entry.setdefault("inferredGender", "unknown")
        entry.setdefault("inferredConfidence", 0.0)
        entry.setdefault("rawLabel", "fallback_unknown")
        entry.setdefault("modelId", None)
        entry.setdefault("overrideGender", None)
        fallback[speaker] = entry
    return normalize_speaker_metadata_map(fallback)


def _format_speaker_gender_log_message(speaker_metadata: Dict[str, Dict[str, Any]]) -> str:
    """Format a short UI-friendly summary of inferred speaker genders."""
    if not speaker_metadata:
        return "Speaker genders: unavailable"

    parts = []
    for speaker, entry in sorted(speaker_metadata.items()):
        gender = entry.get("overrideGender") or entry.get("inferredGender") or "unknown"
        parts.append(f"{speaker}={gender}")
    return "Speaker genders: " + ", ".join(parts)


def _load_project_config(project_id: str, *, fresh: bool = False) -> Dict[str, Any]:
    """Load current project config from the database."""
    db = get_db_session(fresh=fresh)
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found")
        return dict(project.config or {})
    finally:
        db.close()


def _save_speaker_metadata(project_id: str, speaker_metadata: Dict[str, Dict[str, Any]]) -> None:
    """Persist speaker metadata in project config."""
    db = get_db_session()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return

        config = dict(project.config or {})
        config.setdefault("enableSpeakerGenderInference", True)
        config[SPEAKER_METADATA_CONFIG_KEY] = normalize_speaker_metadata_map(speaker_metadata)
        project.config = config
        flag_modified(project, "config")
        project.updated_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


def _save_translation_gender_signature(
    project_id: str,
    segments: list[dict],
    speaker_metadata: Dict[str, Dict[str, Any]],
) -> None:
    """Store the effective-gender translation signature for stale detection."""
    db = get_db_session()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return

        config = dict(project.config or {})
        config[SPEAKER_GENDER_SIGNATURE_CONFIG_KEY] = build_translation_gender_signature(
            segments,
            speaker_metadata,
        )
        project.config = config
        flag_modified(project, "config")
        project.updated_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


def _infer_speaker_metadata(
    project_id: str,
    job_id: str,
    dubber: Any,
    speakers_rolls: Dict[Any, str],
    audio_file: str,
    config_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Extract speaker audio and optionally infer gender-like metadata."""
    existing_metadata = normalize_speaker_metadata_map(config_data.get(SPEAKER_METADATA_CONFIG_KEY))

    if not normalize_bool(config_data.get("enableSpeakerGenderInference", True)):
        return existing_metadata

    try:
        speaker_audio_paths = dubber.speaker_processor.extract_speaker_audio(audio_file, speakers_rolls)
        inferencer = SpeakerGenderInferencer(
            cache_manager=dubber.cache_manager,
            device=dubber.torch_device,
        )
        inferred_metadata = inferencer.infer_speakers(speaker_audio_paths)

        merged_metadata = dict(existing_metadata)
        for speaker, inferred_entry in inferred_metadata.items():
            existing_entry = dict(merged_metadata.get(speaker) or {})
            if existing_entry.get("overrideGender") is not None:
                inferred_entry["overrideGender"] = existing_entry["overrideGender"]
            merged_metadata[speaker] = inferred_entry
    except Exception:
        logger.exception(
            "Speaker gender inference stage failed for project %s. Continuing translation with unknown genders.",
            project_id,
        )
        merged_metadata = _build_unknown_speaker_metadata(speakers_rolls, existing_metadata)
        add_job_log(
            job_id,
            "Speaker gender inference failed, continuing with unknown genders",
            "info",
        )

    logger.info(
        "Speaker gender metadata ready for translation: %s",
        {
            speaker: {
                "inferredGender": entry.get("inferredGender"),
                "inferredConfidence": round(float(entry.get("inferredConfidence") or 0.0), 4),
                "rawLabel": entry.get("rawLabel"),
                "overrideGender": entry.get("overrideGender"),
            }
            for speaker, entry in sorted(merged_metadata.items())
        },
    )
    add_job_log(job_id, _format_speaker_gender_log_message(merged_metadata), "info")
    _save_speaker_metadata(project_id, merged_metadata)
    return merged_metadata


def _build_segment_translation_input(segments: list[Segment]) -> list[dict]:
    """Convert DB segments to the minimal structure expected by the translator."""
    return [
        {
            "speaker": segment.speaker,
            "start": segment.start_time,
            "end": segment.end_time,
            "text": segment.original_text,
        }
        for segment in segments
    ]


def update_job_progress(job_id: str, progress: int, step: str, message: Optional[str] = None, text: Optional[str] = None) -> None:
    """Update job progress in database."""
    db = get_db_session()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.progress = progress
            job.current_step = step
            if message:
                job.add_log(message, "info", text=text)
            db.commit()
    finally:
        db.close()


def add_job_log(job_id: str, message: str, log_type: str = "info", text: Optional[str] = None) -> None:
    """Add a log entry to a job without updating progress."""
    db = get_db_session()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.add_log(message, log_type, text=text)
            db.commit()
    finally:
        db.close()


def update_job_status(job_id: str, status: JobStatus, error_message: Optional[str] = None) -> None:
    """Update job status in database."""
    db = get_db_session()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = status
            if status == JobStatus.PROCESSING:
                if job.started_at is None:
                    job.started_at = datetime.now(timezone.utc)
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = datetime.now(timezone.utc)
            if error_message:
                job.error_message = error_message
                job.add_log(error_message, "error")
            db.commit()
    finally:
        db.close()


def update_project_status(project_id: str, status: ProjectStatus) -> None:
    """Update project status in database."""
    db = get_db_session()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = status
            project.updated_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


@celery_app.task(bind=True, name="src.api.workers.tasks.download_project_video")
def download_project_video(self, project_id: str, job_id: str, url: str, quality: str) -> Dict[str, Any]:
    """Download source video from a URL using yt-dlp."""
    from ..services.video_download import download_video_from_url, VideoDownloadError
    from ..services.job_dispatch import enqueue_transcription_job

    update_job_status(job_id, JobStatus.PROCESSING)
    update_project_status(project_id, ProjectStatus.DOWNLOADING)
    update_job_progress(job_id, 0, "initialization", "Starting video download")

    try:
        from ..services.cookies_manager import get_effective_cookies_path
        cookies_path = get_effective_cookies_path()
        add_job_log(job_id, f"Downloading from URL: {url}", "info")
        add_job_log(job_id, f"Requested quality: {quality}", "info")
        add_job_log(job_id, f"Cookies file: {cookies_path or 'not configured'}", "info")

        def progress_callback(percent: int, step: str) -> None:
            update_job_progress(job_id, percent, "downloading", step)

        def log_callback(message: str, log_type: str = "info") -> None:
            add_job_log(job_id, message, log_type)

        downloaded = download_video_from_url(
            url,
            project_id,
            quality=quality,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )

        db = get_db_session(fresh=True)
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"Project {project_id} not found")

            project.source_file = str(downloaded.upload_path)
            project.source_filename = downloaded.safe_filename
            project.source_size = downloaded.file_size
            project.updated_at = datetime.now(timezone.utc)

            config_data = project.config or {}
            auto_process = bool(config_data.get("autoProcess"))

            db.commit()
        finally:
            db.close()

        update_job_progress(job_id, 100, "complete", f"Downloaded {downloaded.safe_filename}")
        update_job_status(job_id, JobStatus.COMPLETED)

        if auto_process:
            update_project_status(project_id, ProjectStatus.TRANSCRIBING)
            db = get_db_session(fresh=True)
            try:
                project = db.query(Project).filter(Project.id == project_id).first()
                if project:
                    enqueue_transcription_job(db, project, allow_existing=False)
            finally:
                db.close()
            return {
                "status": "success",
                "filename": downloaded.safe_filename,
                "size": downloaded.file_size,
                "title": downloaded.title,
                "auto_process": True,
            }

        update_project_status(project_id, ProjectStatus.DRAFT)
        return {
            "status": "success",
            "filename": downloaded.safe_filename,
            "size": downloaded.file_size,
            "title": downloaded.title,
            "auto_process": False,
        }

    except SoftTimeLimitExceeded:
        update_job_status(job_id, JobStatus.FAILED, "Download timed out")
        update_project_status(project_id, ProjectStatus.ERROR)
        raise
    except VideoDownloadError as exc:
        update_job_status(job_id, JobStatus.FAILED, exc.detail)
        update_project_status(project_id, ProjectStatus.ERROR)
        raise
    except Exception as exc:
        update_job_status(job_id, JobStatus.FAILED, str(exc))
        update_project_status(project_id, ProjectStatus.ERROR)
        raise


def _save_processing_stats(project_id: str, job_id: str, total_cost: float) -> None:
    """Save processing statistics to project config."""
    db = get_db_session()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        project = db.query(Project).filter(Project.id == project_id).first()
        
        if project and job:
            processing_time_sec = 0.0
            if job.started_at and job.completed_at:
                processing_time_sec = (job.completed_at - job.started_at).total_seconds()
            
            config = dict(project.config or {})
            config["resultStats"] = {
                "processingTimeSec": processing_time_sec,
                "totalCost": total_cost,
            }
            project.config = config
            flag_modified(project, "config")
            db.commit()
    finally:
        db.close()


@celery_app.task(bind=True, name="src.api.workers.tasks.transcribe_project")
def transcribe_project(self, project_id: str, job_id: str) -> Dict[str, Any]:
    """Run transcription pipeline for a project."""
    update_job_status(job_id, JobStatus.PROCESSING)
    update_project_status(project_id, ProjectStatus.TRANSCRIBING)
    
    try:
        # Initialize project manager
        pm = ProjectManager(project_id)
        pm.ensure_directories()
        
        # Get project config from database with fresh connection
        # (SQLite requires fresh connection to see changes from other processes)
        db = get_db_session(fresh=True)
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            config_data = project.config or {}
            auto_process_enabled = bool(config_data.get("autoProcess"))
            source_file = pm.get_source_video_path()
            
            # Debug: log config loaded from database
            logger.info(f"[WORKER TRANSCRIBE] Loaded config from DB: sourceLang={config_data.get('sourceLang')}, speakerCount={config_data.get('speakerCount')}, targetLang={config_data.get('targetLang')}")
            
            if not source_file or not source_file.exists():
                raise ValueError("No source video uploaded")
        finally:
            db.close()
        
        update_job_progress(job_id, 2, "initialization", "Starting transcription pipeline")
        
        # Change to project base directory for the pipeline
        # (SmartDubbing uses paths like "artifacts/audio", so we need to be in the project root)
        original_cwd = os.getcwd()
        os.chdir(str(pm.base_dir))
        
        try:
            # Import and configure SmartDubbing
            from src.dubbing.core.config import DubbingConfig
            from src.dubbing.core.smart_dubbing import SmartDubbing
            from src.dubbing.core.log_config import setup_logging
            
            # Setup logging to project's debug directory
            setup_logging(output_dir=str(pm.debug_dir), include_console=False)
            
            # Get preset configuration
            preset = config_data.get("preset", "hq")
            preset_config = get_preset_config(preset)
            
            # Build configuration with preset values (config_data overrides preset)
            # Log the values we're extracting from config_data
            source_lang = config_data.get("sourceLang", "en")
            target_lang = config_data.get("targetLang", "ru")
            speaker_count = config_data.get("speakerCount")
            logger.info(f"[WORKER] Building DubbingConfig: source_language={source_lang}, target_language={target_lang}, speakers_expected={speaker_count}")
            logger.info(f"[DEBUG TRANSCRIBE] preset from config_data: {config_data.get('preset')!r} -> using preset_config for: {preset!r}")
            logger.info(f"[DEBUG TRANSCRIBE] llmModelName from config_data: {config_data.get('llmModelName')!r} -> final: {config_data.get('llmModelName') or preset_config['llm_model_name']!r}")
            logger.info(f"[DEBUG TRANSCRIBE] ttsModel from config_data: {config_data.get('ttsModel')!r} -> final: {config_data.get('ttsModel') or preset_config.get('tts_model')!r}")
            logger.info(f"[DEBUG TRANSCRIBE] refinementModelName from config_data: {config_data.get('refinementModelName')!r}")
            logger.info(f"[DEBUG TRANSCRIBE] enableEmotionEnrichment from config_data: {config_data.get('enableEmotionEnrichment')!r}")
            speaker_voice_mappings = _normalize_speaker_voice_mappings(config_data.get("speakerVoiceMappings"))
            video_quality_preset = get_video_quality_preset(config_data, preset_config)
            editor_runtime = _resolve_editor_runtime_config(config_data, preset_config)
            refinement_persona = _resolve_refinement_persona(config_data, preset_config)
            enable_llm_text_adjustment = _resolve_bool_config(
                config_data,
                "enableLlmTextAdjustment",
                preset_config,
                "enable_llm_text_adjustment",
                True,
            )
            
            dubbing_config = DubbingConfig()
            dubbing_config.config.update({
                "project_id": project_id,
                "input": str(source_file),
                "source_language": source_lang,
                "target_language": target_lang,
                "keep_background": config_data.get(
                    "keepBackground",
                    preset_config.get("keep_background", False)
                ),
                "pause_removal": config_data.get("pauseRemoval", "disabled"),
                "speakers_expected": speaker_count,
                "exit_before_synthesis": True,  # Stop after translation
                "no_cache": False,
                # TTS settings (config_data overrides preset)
                "tts_system": config_data.get("ttsSystem") or preset_config["tts_system"],
                "tts_model": config_data.get("ttsModel") or preset_config.get("tts_model"),
                "tts_fallback_model": config_data.get("ttsFallbackModel") or preset_config.get("tts_fallback_model"),
                "tts_prompt_prefix": resolve_tts_prompt_prefix(
                    style=config_data.get("ttsStyle") or preset_config.get("tts_style"),
                    custom_prompt=config_data.get("ttsPromptPrefix") or preset_config.get("tts_prompt_prefix"),
                    resolved_style=config_data.get("resolvedTtsStyle"),
                ),
                "blocked_voices": get_blocked_voices_for_style(
                    style=config_data.get("ttsStyle") or preset_config.get("tts_style"),
                    resolved_style=config_data.get("resolvedTtsStyle"),
                ),
                "voice_name": speaker_voice_mappings,
                "voice_prompt": config_data.get("speakerTtsPrompts", {}),
                "voice_auto_selection": config_data.get("voiceAutoSelection", True),
                "enable_emotion_analysis": config_data.get("enableEmotionAnalysis", False),
                "enable_emotion_enrichment": config_data.get("enableEmotionEnrichment", False),
                "enable_content_validation": config_data.get("enableContentValidation", True),
                "content_validator_provider": config_data.get("contentValidatorProvider", "whisper"),
                "content_validator_whisper_model": config_data.get("contentValidatorWhisperModel", "base"),
                "content_validator_whisper_compute_type": config_data.get("contentValidatorWhisperComputeType", "int8"),
                "content_validator_whisper_cpu_threads": config_data.get("contentValidatorWhisperCpuThreads", 2),
                "content_validator_speech_model": config_data.get("contentValidatorSpeechModel", "nano"),
                # Transcription settings
                "transcription_system": config_data.get("transcriptionSystem", "assemblyai"),
                "whisper_model": config_data.get("whisperModel", "large-v3"),
                # Translation settings (config_data overrides preset)
                "translator_type": "llm",
                "llm_provider": config_data.get("llmProvider") or preset_config["llm_provider"],
                "llm_model_name": config_data.get("llmModelName") or preset_config["llm_model_name"],
                "llm_temperature": config_data.get("llmTemperature") if config_data.get("llmTemperature") is not None else preset_config["llm_temperature"],
                "enable_llm_editor": editor_runtime["enable_llm_editor"],
                "editor_llm_provider": editor_runtime["editor_llm_provider"],
                "editor_model_name": editor_runtime["editor_model_name"],
                "editor_temperature": editor_runtime["editor_temperature"],
                "editor_reasoning_effort": editor_runtime["editor_reasoning_effort"],
                "enable_llm_text_adjustment": enable_llm_text_adjustment,
                "refinement_llm_provider": config_data.get("refinementLlmProvider") or preset_config.get("refinement_llm_provider"),
                "refinement_model_name": config_data.get("refinementModelName") or preset_config.get("refinement_model_name"),
                "refinement_temperature": config_data.get("refinementTemperature") if config_data.get("refinementTemperature") is not None else preset_config.get("refinement_temperature", 1.0),
                "refinement_persona": refinement_persona,
                "translation_prompt_prefix": config_data.get("translationPromptPrefix"),
                "enable_speaker_gender_inference": config_data.get("enableSpeakerGenderInference", True),
                "speaker_metadata": config_data.get(SPEAKER_METADATA_CONFIG_KEY),
                # Audio settings
                "dubbed_volume": config_data.get("dubbedVolume", 1.0),
                "background_volume": config_data.get("backgroundVolume", 0.562341),
                "keep_original_audio_ranges": config_data.get("keepOriginalAudioRanges"),
                "use_two_pass_encoding": config_data.get("useTwoPassEncoding", True),
                "video_quality_preset": video_quality_preset,
                # Processing settings
                "max_workers": config_data.get("maxWorkers", 4),
                # Video processing settings
                "video_minterpolate_threshold": config_data.get("videoMinterpolateThreshold") or preset_config.get("video_minterpolate_threshold"),
                "start_time": config_data.get("startTime"),
                "duration": config_data.get("duration"),
            })
            
            segments_opt = dubbing_config.config.get("segments_optimization", {})
            segments_opt = apply_segment_optimization_config(segments_opt, config_data)
            dubbing_config.config["segments_optimization"] = segments_opt
            
            # Apply segment_stretch mode
            if config_data.get("segmentStretch"):
                dubbing_config.config["segment_stretch"] = config_data["segmentStretch"]
            
            # Apply API keys from settings and project config
            _apply_api_keys(config_data.get("apiKeys"))
            
            dubbing_config.validate = lambda: None  # Skip file validation, we've done it
            dubbing_config.process_special_parameters()
            
            update_job_progress(job_id, 5, "audio_extraction", "Extracting audio from video")
            
            # Initialize SmartDubbing
            dubber = SmartDubbing(dubbing_config)
            
            # Extract audio
            audio_file = dubber.audio_processor.extract_audio(
                str(source_file),
                dubbing_config.get("start_time"),
                dubbing_config.get("duration")
            )
            
            update_job_progress(job_id, 10, "diarization", "Performing speaker diarization")
            
            # Diarize and transcribe
            speakers_rolls, transcription = dubber.diarize_and_transcribe(audio_file)
            # Filter out segments in keep-original-audio ranges
            transcription = dubber._filter_keep_original_segments(transcription)

            if not speakers_rolls or len(speakers_rolls) == 0:
                raise ValueError("No speakers found in the video")

            update_job_progress(job_id, 13, "speaker_analysis", "Analyzing speakers and inferring genders")
            speaker_metadata = _infer_speaker_metadata(
                project_id,
                job_id,
                dubber,
                speakers_rolls,
                audio_file,
                config_data,
            )

            update_job_progress(job_id, 15, "translation", "Translating segments")
            
            # Progress callback for translation, refinement, and optional editor
            # Before segments created: max 35%
            # Translation: 15-22%, Refinement: 22-29%, Editor: 29-35%
            def translation_progress(phase: str, current: int, total: int, text: str = None):
                if phase == "translation":
                    progress = 15 + int((current / total) * 7)
                    step_name = "translation"
                    message = f"Translating chunk {current}/{total}"
                elif phase == "editor":
                    progress = 29 + int((current / total) * 6)
                    step_name = "editor"
                    message = f"LLM editor pass: {current}/{total}" if total > 0 else "LLM editor pass"
                else:  # refinement
                    progress = 22 + int((current / total) * 7)
                    step_name = "refinement"
                    message = f"Refining chunk {current}/{total}"
                update_job_progress(job_id, progress, step_name, message, text=text)
            
            # Translate segments
            translated_segments = dubber.translate_segments(
                transcription,
                audio_file,
                progress_callback=translation_progress,
                speaker_metadata=speaker_metadata,
            )

            # Persist auto-resolved TTS style so the dub stage can pick it up.
            if (config_data.get("ttsStyle") or "podcast") == "auto":
                try:
                    resolved_style = (
                        (getattr(dubber.translator, "last_context_info", None) or {}).get("tts_style")
                        or "podcast"
                    )
                    db_resolve = get_db_session(fresh=True)
                    try:
                        project_row = db_resolve.query(Project).filter(Project.id == project_id).first()
                        if project_row is not None:
                            project_row.config = {
                                **(project_row.config or {}),
                                "resolvedTtsStyle": resolved_style,
                            }
                            flag_modified(project_row, "config")
                            db_resolve.commit()
                            config_data = project_row.config
                            logger.info(f"[WORKER TRANSCRIBE] Auto-resolved ttsStyle -> {resolved_style}")
                    finally:
                        db_resolve.close()
                except Exception as resolve_err:
                    logger.warning(f"Failed to persist resolvedTtsStyle: {resolve_err}")

            update_job_progress(job_id, 35, "saving", "Saving segments to database")
            
            # Save segments to database
            _save_segments_to_db(project_id, translated_segments)
            _save_translation_gender_signature(project_id, translated_segments, speaker_metadata)
            write_cost_snapshot_from_tracker(pm.debug_dir, "transcription", dubber.cost_tracker)
            write_cost_snapshot_from_tracker(pm.debug_dir, "llm", dubber.cost_tracker)

            # If requested, continue straight into final dubbing in the same job.
            # This keeps the SSE stream alive and avoids a "pause" between stages.
            if auto_process_enabled:
                update_job_progress(job_id, 36, "handoff", "Transcription complete, starting dubbing")
                result = dub_project(project_id, job_id)
                if isinstance(result, dict):
                    result.setdefault("segments_count", len(translated_segments))
                return result
            
            update_job_progress(job_id, 100, "complete", f"Transcription complete: {len(translated_segments)} segments")
            
        finally:
            os.chdir(original_cwd)
        
        # Update statuses
        update_job_status(job_id, JobStatus.COMPLETED)
        update_project_status(project_id, ProjectStatus.TRANSCRIBED)
        
        return {"status": "success", "segments_count": len(translated_segments)}
        
    except SoftTimeLimitExceeded:
        update_job_status(job_id, JobStatus.FAILED, "Task timed out")
        update_project_status(project_id, ProjectStatus.ERROR)
        raise
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, str(e))
        update_project_status(project_id, ProjectStatus.ERROR)
        raise


@celery_app.task(bind=True, name="src.api.workers.tasks.retranslate_project")
def retranslate_project(self, project_id: str, job_id: str) -> Dict[str, Any]:
    """Re-run translation for existing segments using the current project config."""
    update_job_status(job_id, JobStatus.PROCESSING)
    update_project_status(project_id, ProjectStatus.TRANSCRIBING)

    try:
        pm = ProjectManager(project_id)
        pm.ensure_directories()

        db = get_db_session(fresh=True)
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"Project {project_id} not found")

            config_data = project.config or {}
            source_file = pm.get_source_video_path()
            segments = db.query(Segment).filter(
                Segment.project_id == project_id
            ).order_by(Segment.sequence).all()

            if not source_file or not source_file.exists():
                raise ValueError("No source video uploaded")
            if not segments:
                raise ValueError("No existing segments found for re-translation")
        finally:
            db.close()

        update_job_progress(job_id, 2, "initialization", "Starting translation refresh")

        original_cwd = os.getcwd()
        os.chdir(str(pm.base_dir))

        try:
            from src.dubbing.core.config import DubbingConfig
            from src.dubbing.core.smart_dubbing import SmartDubbing
            from src.dubbing.core.log_config import setup_logging

            setup_logging(output_dir=str(pm.debug_dir), include_console=False)

            preset = config_data.get("preset", "hq")
            preset_config = get_preset_config(preset)
            speaker_voice_mappings = _normalize_speaker_voice_mappings(config_data.get("speakerVoiceMappings"))
            video_quality_preset = get_video_quality_preset(config_data, preset_config)
            editor_runtime = _resolve_editor_runtime_config(config_data, preset_config)
            refinement_persona = _resolve_refinement_persona(config_data, preset_config)
            enable_llm_text_adjustment = _resolve_bool_config(
                config_data,
                "enableLlmTextAdjustment",
                preset_config,
                "enable_llm_text_adjustment",
                True,
            )

            dubbing_config = DubbingConfig()
            dubbing_config.config.update({
                "project_id": project_id,
                "input": str(source_file),
                "source_language": config_data.get("sourceLang", "en"),
                "target_language": config_data.get("targetLang", "ru"),
                "keep_background": config_data.get("keepBackground", preset_config.get("keep_background", False)),
                "pause_removal": config_data.get("pauseRemoval", "disabled"),
                "exit_before_synthesis": True,
                "no_cache": False,
                "tts_system": config_data.get("ttsSystem") or preset_config["tts_system"],
                "tts_model": config_data.get("ttsModel") or preset_config.get("tts_model"),
                "tts_fallback_model": config_data.get("ttsFallbackModel") or preset_config.get("tts_fallback_model"),
                "tts_prompt_prefix": resolve_tts_prompt_prefix(
                    style=config_data.get("ttsStyle") or preset_config.get("tts_style"),
                    custom_prompt=config_data.get("ttsPromptPrefix") or preset_config.get("tts_prompt_prefix"),
                    resolved_style=config_data.get("resolvedTtsStyle"),
                ),
                "blocked_voices": get_blocked_voices_for_style(
                    style=config_data.get("ttsStyle") or preset_config.get("tts_style"),
                    resolved_style=config_data.get("resolvedTtsStyle"),
                ),
                "voice_name": speaker_voice_mappings,
                "voice_prompt": config_data.get("speakerTtsPrompts", {}),
                "voice_auto_selection": config_data.get("voiceAutoSelection", True),
                "enable_emotion_analysis": config_data.get("enableEmotionAnalysis", False),
                "enable_emotion_enrichment": config_data.get("enableEmotionEnrichment", False),
                "enable_content_validation": config_data.get("enableContentValidation", True),
                "content_validator_provider": config_data.get("contentValidatorProvider", "whisper"),
                "content_validator_whisper_model": config_data.get("contentValidatorWhisperModel", "base"),
                "content_validator_whisper_compute_type": config_data.get("contentValidatorWhisperComputeType", "int8"),
                "content_validator_whisper_cpu_threads": config_data.get("contentValidatorWhisperCpuThreads", 2),
                "content_validator_speech_model": config_data.get("contentValidatorSpeechModel", "nano"),
                "transcription_system": config_data.get("transcriptionSystem", "assemblyai"),
                "whisper_model": config_data.get("whisperModel", "large-v3"),
                "translator_type": "llm",
                "llm_provider": config_data.get("llmProvider") or preset_config["llm_provider"],
                "llm_model_name": config_data.get("llmModelName") or preset_config["llm_model_name"],
                "llm_temperature": config_data.get("llmTemperature") if config_data.get("llmTemperature") is not None else preset_config["llm_temperature"],
                "enable_llm_editor": editor_runtime["enable_llm_editor"],
                "editor_llm_provider": editor_runtime["editor_llm_provider"],
                "editor_model_name": editor_runtime["editor_model_name"],
                "editor_temperature": editor_runtime["editor_temperature"],
                "editor_reasoning_effort": editor_runtime["editor_reasoning_effort"],
                "enable_llm_text_adjustment": enable_llm_text_adjustment,
                "refinement_llm_provider": config_data.get("refinementLlmProvider") or preset_config.get("refinement_llm_provider"),
                "refinement_model_name": config_data.get("refinementModelName") or preset_config.get("refinement_model_name"),
                "refinement_temperature": config_data.get("refinementTemperature") if config_data.get("refinementTemperature") is not None else preset_config.get("refinement_temperature", 1.0),
                "refinement_persona": refinement_persona,
                "translation_prompt_prefix": config_data.get("translationPromptPrefix"),
                "enable_speaker_gender_inference": config_data.get("enableSpeakerGenderInference", True),
                "speaker_metadata": config_data.get(SPEAKER_METADATA_CONFIG_KEY),
                "dubbed_volume": config_data.get("dubbedVolume", 1.0),
                "background_volume": config_data.get("backgroundVolume", 0.562341),
                "keep_original_audio_ranges": config_data.get("keepOriginalAudioRanges"),
                "use_two_pass_encoding": config_data.get("useTwoPassEncoding", True),
                "video_quality_preset": video_quality_preset,
                "max_workers": config_data.get("maxWorkers", 4),
                "video_minterpolate_threshold": config_data.get("videoMinterpolateThreshold") or preset_config.get("video_minterpolate_threshold"),
                "start_time": config_data.get("startTime"),
                "duration": config_data.get("duration"),
            })

            segments_opt = dubbing_config.config.get("segments_optimization", {})
            segments_opt = apply_segment_optimization_config(segments_opt, config_data)
            dubbing_config.config["segments_optimization"] = segments_opt

            if config_data.get("segmentStretch"):
                dubbing_config.config["segment_stretch"] = config_data["segmentStretch"]

            _apply_api_keys(config_data.get("apiKeys"))

            dubbing_config.validate = lambda: None
            dubbing_config.process_special_parameters()

            update_job_progress(job_id, 5, "audio_extraction", "Extracting audio from video")
            dubber = SmartDubbing(dubbing_config)
            audio_file = dubber.audio_processor.extract_audio(
                str(source_file),
                dubbing_config.get("start_time"),
                dubbing_config.get("duration")
            )

            update_job_progress(job_id, 15, "translation", "Re-translating segments")

            def translation_progress(phase: str, current: int, total: int, text: str = None):
                if phase == "translation":
                    progress = 15 + int((current / total) * 7)
                    step_name = "translation"
                    message = f"Re-translating chunk {current}/{total}"
                elif phase == "editor":
                    progress = 29 + int((current / total) * 6)
                    step_name = "editor"
                    message = f"LLM editor pass: {current}/{total}" if total > 0 else "LLM editor pass"
                else:
                    progress = 22 + int((current / total) * 7)
                    step_name = "refinement"
                    message = f"Refining chunk {current}/{total}"
                update_job_progress(job_id, progress, step_name, message, text=text)

            translation_input = _build_segment_translation_input(segments)
            speaker_metadata = normalize_speaker_metadata_map(config_data.get(SPEAKER_METADATA_CONFIG_KEY))
            translated_segments = dubber.translate_segments(
                translation_input,
                audio_file,
                progress_callback=translation_progress,
                speaker_metadata=speaker_metadata,
                preserve_segment_boundaries=True,
            )

            update_job_progress(job_id, 35, "saving", "Saving refreshed translations")
            _update_translated_segments_in_db(project_id, translated_segments)
            _save_translation_gender_signature(project_id, translated_segments, speaker_metadata)
            write_cost_snapshot_from_tracker(pm.debug_dir, "llm", dubber.cost_tracker)

            update_job_progress(job_id, 100, "complete", f"Translation refreshed: {len(translated_segments)} segments")

        finally:
            os.chdir(original_cwd)

        update_job_status(job_id, JobStatus.COMPLETED)
        update_project_status(project_id, ProjectStatus.TRANSCRIBED)
        return {"status": "success", "segments_count": len(segments)}

    except SoftTimeLimitExceeded:
        update_job_status(job_id, JobStatus.FAILED, "Task timed out")
        update_project_status(project_id, ProjectStatus.ERROR)
        raise
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, str(e))
        update_project_status(project_id, ProjectStatus.ERROR)
        raise


@celery_app.task(bind=True, name="src.api.workers.tasks.dub_project")
def dub_project(self, project_id: str, job_id: str) -> Dict[str, Any]:
    """Run final dubbing pipeline for a project."""
    update_job_status(job_id, JobStatus.PROCESSING)
    update_project_status(project_id, ProjectStatus.DUBBING)
    
    try:
        # Initialize project manager
        pm = ProjectManager(project_id)
        pm.ensure_directories()
        
        # Get project and segments from database with fresh connection
        # (SQLite requires fresh connection to see changes from other processes)
        db = get_db_session(fresh=True)
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            segments = db.query(Segment).filter(
                Segment.project_id == project_id
            ).order_by(Segment.sequence).all()
            
            if not segments:
                raise ValueError("No segments found for dubbing")
            
            config_data = project.config or {}
            source_file = pm.get_source_video_path()
            
            # Debug: log config loaded from database
            logger.info(f"[WORKER DUB] Loaded config from DB: sourceLang={config_data.get('sourceLang')}, speakerCount={config_data.get('speakerCount')}, targetLang={config_data.get('targetLang')}")
            
            if not source_file or not source_file.exists():
                raise ValueError("No source video uploaded")
            
            # Convert segments to dict format for SmartDubbing
            segments_data = [_segment_to_dubbing_format(seg) for seg in segments]
        finally:
            db.close()
        
        # Dubbing starts at 35% (continuing from transcription phase)
        update_job_progress(job_id, 37, "initialization", "Starting dubbing pipeline")
        
        # Change to project base directory
        # (SmartDubbing uses paths like "artifacts/audio", so we need to be in the project root)
        original_cwd = os.getcwd()
        os.chdir(str(pm.base_dir))
        
        try:
            from src.dubbing.core.config import DubbingConfig
            from src.dubbing.core.smart_dubbing import SmartDubbing
            from src.dubbing.core.log_config import setup_logging
            
            # Setup logging to project's debug directory
            setup_logging(output_dir=str(pm.debug_dir), include_console=False)
            
            # Get preset configuration
            preset = config_data.get("preset", "hq")
            preset_config = get_preset_config(preset)
            
            # Build TTS prompt prefix based on target language, preset, and selected style
            target_lang = config_data.get("targetLang", "ru")
            tts_prompt_prefix = resolve_tts_prompt_prefix(
                style=config_data.get("ttsStyle") or preset_config.get("tts_style"),
                custom_prompt=config_data.get("ttsPromptPrefix") or preset_config.get("tts_prompt_prefix"),
                resolved_style=config_data.get("resolvedTtsStyle"),
            )
            if tts_prompt_prefix and "{lang}" in tts_prompt_prefix:
                tts_prompt_prefix = tts_prompt_prefix.replace("{lang}", target_lang)
            blocked_voices = get_blocked_voices_for_style(
                style=config_data.get("ttsStyle") or preset_config.get("tts_style"),
                resolved_style=config_data.get("resolvedTtsStyle"),
            )
            
            # Build configuration with preset values (config_data overrides preset)
            pause_removal_value = config_data.get("pauseRemoval", "disabled")
            logger.info(f"[DEBUG DUB] preset from config_data: {config_data.get('preset')!r} -> using preset_config for: {preset!r}")
            logger.info(f"[DEBUG DUB] pause_removal from config_data: {pause_removal_value}")
            logger.info(f"[DEBUG DUB] llmModelName from config_data: {config_data.get('llmModelName')!r} -> final: {config_data.get('llmModelName') or preset_config['llm_model_name']!r}")
            logger.info(f"[DEBUG DUB] ttsModel from config_data: {config_data.get('ttsModel')!r} -> final: {config_data.get('ttsModel') or preset_config.get('tts_model')!r}")
            logger.info(f"[DEBUG DUB] refinementModelName from config_data: {config_data.get('refinementModelName')!r}")
            logger.info(f"[DEBUG DUB] enableEmotionEnrichment from config_data: {config_data.get('enableEmotionEnrichment')!r}")
            logger.info(
                "[DEBUG DUB] clip window from config_data: startTime=%r, duration=%r",
                config_data.get("startTime"),
                config_data.get("duration"),
            )
            logger.info(f"[DEBUG DUB] config_data keys: {list(config_data.keys())}")
            speaker_voice_mappings = _normalize_speaker_voice_mappings(config_data.get("speakerVoiceMappings"))
            video_quality_preset = get_video_quality_preset(config_data, preset_config)
            editor_runtime = _resolve_editor_runtime_config(config_data, preset_config)
            refinement_persona = _resolve_refinement_persona(config_data, preset_config)
            enable_llm_text_adjustment = _resolve_bool_config(
                config_data,
                "enableLlmTextAdjustment",
                preset_config,
                "enable_llm_text_adjustment",
                True,
            )
            max_output_height = VIDEO_QUALITY_MAX_HEIGHT[video_quality_preset]
            logger.info(f"[DEBUG DUB] videoQualityPreset: {video_quality_preset} -> max_output_height={max_output_height}")
            
            dubbing_config = DubbingConfig()
            dubbing_config.config.update({
                "project_id": project_id,
                "input": str(source_file),
                "source_language": config_data.get("sourceLang", "en"),
                "target_language": target_lang,
                "keep_background": config_data.get(
                    "keepBackground",
                    preset_config.get("keep_background", False)
                ),
                "pause_removal": pause_removal_value,
                "output": str(pm.get_result_video_path(target_lang)),
                "save_translated_subtitles": True,
                # TTS settings (config_data overrides preset)
                "tts_system": config_data.get("ttsSystem") or preset_config["tts_system"],
                "tts_model": config_data.get("ttsModel") or preset_config.get("tts_model"),
                "tts_fallback_model": config_data.get("ttsFallbackModel") or preset_config.get("tts_fallback_model"),
                "tts_prompt_prefix": tts_prompt_prefix,
                "blocked_voices": blocked_voices,
                "voice_name": speaker_voice_mappings,
                "voice_prompt": config_data.get("speakerTtsPrompts", {}),
                "voice_auto_selection": config_data.get("voiceAutoSelection", True),
                "enable_emotion_analysis": config_data.get("enableEmotionAnalysis", False),
                "enable_emotion_enrichment": config_data.get("enableEmotionEnrichment", False),
                "enable_content_validation": config_data.get("enableContentValidation", True),
                "content_validator_provider": config_data.get("contentValidatorProvider", "whisper"),
                "content_validator_whisper_model": config_data.get("contentValidatorWhisperModel", "base"),
                "content_validator_whisper_compute_type": config_data.get("contentValidatorWhisperComputeType", "int8"),
                "content_validator_whisper_cpu_threads": config_data.get("contentValidatorWhisperCpuThreads", 2),
                "content_validator_speech_model": config_data.get("contentValidatorSpeechModel", "nano"),
                "max_workers": config_data.get("maxWorkers", 4),
                # Audio settings
                "dubbed_volume": config_data.get("dubbedVolume", 1.0),
                "background_volume": config_data.get("backgroundVolume", 0.562341),
                "keep_original_audio_ranges": config_data.get("keepOriginalAudioRanges"),
                "use_two_pass_encoding": config_data.get("useTwoPassEncoding", True),
                "video_quality_preset": video_quality_preset,
                # Video processing settings
                "video_minterpolate_threshold": config_data.get("videoMinterpolateThreshold") or preset_config.get("video_minterpolate_threshold"),
                "start_time": config_data.get("startTime"),
                "duration": config_data.get("duration"),
                # Translation settings (for any re-translation)
                "translator_type": "llm",
                "llm_provider": config_data.get("llmProvider") or preset_config["llm_provider"],
                "llm_model_name": config_data.get("llmModelName") or preset_config["llm_model_name"],
                "llm_temperature": config_data.get("llmTemperature") if config_data.get("llmTemperature") is not None else preset_config["llm_temperature"],
                "enable_llm_editor": editor_runtime["enable_llm_editor"],
                "editor_llm_provider": editor_runtime["editor_llm_provider"],
                "editor_model_name": editor_runtime["editor_model_name"],
                "editor_temperature": editor_runtime["editor_temperature"],
                "editor_reasoning_effort": editor_runtime["editor_reasoning_effort"],
                "enable_llm_text_adjustment": enable_llm_text_adjustment,
                "refinement_llm_provider": config_data.get("refinementLlmProvider") or preset_config.get("refinement_llm_provider"),
                "refinement_model_name": config_data.get("refinementModelName") or preset_config.get("refinement_model_name"),
                "refinement_temperature": config_data.get("refinementTemperature") if config_data.get("refinementTemperature") is not None else preset_config.get("refinement_temperature", 1.0),
                "refinement_persona": refinement_persona,
                "speaker_metadata": config_data.get(SPEAKER_METADATA_CONFIG_KEY),
            })
            
            segments_opt = dubbing_config.config.get("segments_optimization", {})
            segments_opt = apply_segment_optimization_config(segments_opt, config_data)
            dubbing_config.config["segments_optimization"] = segments_opt
            
            # Apply segment_stretch mode
            if config_data.get("segmentStretch"):
                dubbing_config.config["segment_stretch"] = config_data["segmentStretch"]
            
            # Apply API keys from settings and project config
            _apply_api_keys(config_data.get("apiKeys"))

            dubbing_config.validate = lambda: None
            dubbing_config.process_special_parameters()
            
            update_job_progress(job_id, 40, "initialization", "Initializing TTS systems")
            
            # Initialize SmartDubbing
            dubber = SmartDubbing(dubbing_config)
            imported_cost_rows = import_cost_snapshots(pm.debug_dir, dubber.cost_tracker)
            if imported_cost_rows:
                logger.info("Imported %d API cost row(s) from previous pipeline stages", imported_cost_rows)
            dubber.performance_tracker.start_timing("total")
            
            # Extract audio
            audio_file = dubber.audio_processor.extract_audio(
                str(source_file),
                dubbing_config.get("start_time"),
                dubbing_config.get("duration")
            )
            
            # Restore speakers_rolls from segments
            speakers_rolls = {}
            for seg in segments_data:
                key = (seg["start"], seg["end"])
                speakers_rolls[key] = seg["speaker"]
            
            update_job_progress(job_id, 48, "speech_synthesis", "Synthesizing speech")
            
            # Progress callback for TTS cache prewarming (48% to 52%)
            total_segments = len(segments_data)
            def prewarm_progress(current: int, total: int, text: str = None):
                progress = 48 + int((current / total) * 4) if total > 0 else 48
                message = (
                    f"Prewarming TTS cache: {current}/{total}"
                    if total > 0
                    else "Prewarming TTS cache"
                )
                update_job_progress(job_id, progress, "speech_synthesis", message)

            # Progress callback for main speech synthesis (52% to 70%)
            def synthesis_progress(current: int, total: int, text: str = None):
                progress = 52 + int((current / total) * 18) if total > 0 else 52
                message = f"Synthesizing segment {current}/{total}"
                update_job_progress(job_id, progress, "speech_synthesis", message, text=text)
            
            # Progress callback for grouping and overlay (70% to 74%)
            def grouping_progress(current: int, total: int, message: str = ""):
                progress = 70 + int((current / total) * 4) if total > 0 else 70
                # Use the same "Label: current/total" format as background audio so the frontend renders a progress bar.
                update_job_progress(
                    job_id,
                    progress,
                    "audio_grouping",
                    f"Adjusting segment timing: {current}/{total}" if total > 0 else "Adjusting segment timing",
                )
            
            # Synthesize speech
            translated_audio_path = dubber.synthesize_speech(
                segments_data, speakers_rolls, audio_file, 
                progress_callback=synthesis_progress,
                grouping_progress_callback=grouping_progress,
                prepass_progress_callback=prewarm_progress,
            )
            
            # Process background with progress callback
            background_audio_path = None
            if config_data.get("keepBackground", preset_config.get("keep_background", False)):
                update_job_progress(job_id, 74, "background_audio", "Processing background audio")
                def background_progress(current: int, total: int, message: str = ""):
                    # Progress from 74% to 84% during background audio extraction
                    progress = 74 + int((current / total) * 10) if total > 0 else 74
                    update_job_progress(job_id, progress, "background_audio", f"Processing background audio: {current}/{total}")
                                
                background_audio_path = dubber.audio_processor.process_background_audio(
                    audio_file, progress_callback=background_progress
                )
            
            update_job_progress(job_id, 84, "video_combine", "Combining audio with video")
            
            # Progress callback for video combining (84% to 99%)
            # Segment processing: 84% → 93%, Encoding: 93% → 99%
            def video_combine_progress(current: int, total: int, message: str = ""):
                if message == "Video speed segment processing" and total > 0:
                    # Segments: 84% → 93%
                    progress = 84 + int((current / total) * 9)
                    ui_message = f"Video speed segment processing: {current}/{total}"
                elif message.startswith("Encoding video") or message.startswith("Combining audio"):
                    # Encoding: 93% → 99%
                    progress = 93 + int((current / total) * 6) if total > 0 else 93
                    ui_message = f"{message}: {current}%"
                else:
                    progress = 84 + int((current / total) * 15) if total > 0 else 84
                    ui_message = message or f"Processing video: {current}/{total}s"
                update_job_progress(job_id, progress, "video_combine", ui_message)
            
            def video_combine_log(message: str):
                # Segment-level speed logs are useful in debug files but too noisy for SSE UI logs.
                if message.startswith("Segment ") and "speed=" in message:
                    return
                add_job_log(job_id, message)
            
            # Combine with video
            segments_opt = dubbing_config.config.get("segments_optimization", {})
            
            # Determine effective pause removal
            pause_removal = dubbing_config.get("pause_removal", "disabled")
            
            # Get video speed segments for video/audio_and_video modes
            video_speed_segments = getattr(dubber, 'video_speed_segments', None)
            
            output_video_path, _ = dubber.video_processor.combine_audio_with_video(
                video_path=str(source_file),
                translated_audio_path=translated_audio_path,
                background_audio_path=background_audio_path,
                output_file=str(pm.get_result_video_path(config_data.get("targetLang", "ru"))),
                start_time=config_data.get("startTime"),
                duration=config_data.get("duration"),
                source_language=config_data.get("sourceLang", "en"),
                target_language=config_data.get("targetLang", "ru"),
                keep_original_audio_ranges=dubbing_config.get("keep_original_audio_ranges"),
                pause_removal=pause_removal,
                min_pause_duration=segments_opt.get("min_pause_duration", 3),
                preserve_pause_duration=segments_opt.get("preserve_pause_duration", 1.5),
                max_output_height=max_output_height,
                progress_callback=video_combine_progress,
                log_callback=video_combine_log,
                video_speed_segments=video_speed_segments,
            )
            
            # Save subtitles to results directory
            source_lang = config_data.get("sourceLang", "en")
            target_lang = config_data.get("targetLang", "ru")
            
            source_srt_path = str(pm.get_result_subtitles_path(source_lang, "srt"))
            target_srt_path = str(pm.get_result_subtitles_path(target_lang, "srt"))
            
            dubber.subtitle_manager.save_subtitles(segments_data, "original", source_srt_path)
            dubber.subtitle_manager.save_subtitles(segments_data, "translation", target_srt_path)
            
            update_job_progress(job_id, 100, "complete", "Dubbing complete")
            
            # This task drives the low-level pipeline directly instead of going
            # through SmartDubbing.dub_video(), so it must finalize the summary
            # artifacts explicitly before the UI requests them.
            total_started_at = dubber.performance_tracker._start_times.get("total")
            total_elapsed = (
                time.perf_counter() - total_started_at
                if total_started_at is not None
                else dubber.performance_tracker.get_metric("total")
            )
            dubber.performance_tracker.record_metric("total", total_elapsed)
            write_cost_snapshot_from_tracker(pm.debug_dir, "dub", dubber.cost_tracker)
            dubber.performance_tracker.set_costs(dubber.cost_tracker.get_costs_by_step())
            dubber.performance_tracker.write_performance_summary(
                dubber.audio_processor.get_total_duration()
            )
            dubber.cost_tracker.write_cost_summary()
            dubber._build_summary_report()

            # Collect stats from dubber
            total_cost = dubber.cost_tracker.actual.get("total", 0.0)
            
        finally:
            os.chdir(original_cwd)
        
        # Update statuses and save processing stats
        update_job_status(job_id, JobStatus.COMPLETED)
        update_project_status(project_id, ProjectStatus.DUBBED)
        
        # Save processing stats and clear autoProcess flag
        _save_processing_stats(project_id, job_id, total_cost)
        
        # Clear autoProcess flag
        db = get_db_session()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project and project.config:
                config = dict(project.config)
                config["autoProcess"] = False
                project.config = config
                flag_modified(project, "config")
                db.commit()
        finally:
            db.close()
        
        return {"status": "success", "output": str(output_video_path)}
        
    except SoftTimeLimitExceeded:
        update_job_status(job_id, JobStatus.FAILED, "Task timed out")
        update_project_status(project_id, ProjectStatus.ERROR)
        raise
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, str(e))
        update_project_status(project_id, ProjectStatus.ERROR)
        raise


@celery_app.task(bind=True, name="src.api.workers.tasks.generate_preview")
def generate_preview(
    self,
    project_id: str,
    segment_id: str,
    force_regenerate: bool = False,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generate TTS preview for a single segment."""
    if job_id:
        update_job_status(job_id, JobStatus.PROCESSING)
    
    try:
        # Get segment from database first to get voice_id
        db = get_db_session()
        try:
            segment = db.query(Segment).filter(Segment.id == segment_id).first()
            if not segment:
                raise ValueError(f"Segment {segment_id} not found")
            
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            config_data = project.config or {}
            speaker_voice_mappings = _normalize_speaker_voice_mappings(config_data.get("speakerVoiceMappings"))
            voice_id = segment.voice_id or speaker_voice_mappings.get(segment.speaker)
        finally:
            db.close()
        
        pm = ProjectManager(project_id)
        preview_path = pm.get_preview_audio_path(segment_id, voice_id)
        
        # Check cache first
        if preview_path.exists() and not force_regenerate:
            if job_id:
                update_job_status(job_id, JobStatus.COMPLETED)
            return {
                "audioUrl": f"/api/v1/projects/{project_id}/segments/{segment_id}/preview/audio?voice_id={voice_id or ''}",
                "isCached": True,
            }
        
        # Apply API keys
        _apply_api_keys(config_data.get("apiKeys"))
        
        # Generate preview using TTS
        original_cwd = os.getcwd()
        os.chdir(str(pm.artifacts_dir))
        
        try:
            from src.tts.tts_factory import TTSFactory
            from src.tts.models import TTSSegmentData
            
            # Create TTS instance
            tts_system = segment.provider or config_data.get("ttsSystem", "gemini")
            
            # Build TTS prompt prefix from the selected style
            target_lang = config_data.get("targetLang", "ru")
            tts_prompt_prefix = resolve_tts_prompt_prefix(
                style=config_data.get("ttsStyle"),
                custom_prompt=config_data.get("ttsPromptPrefix"),
                resolved_style=config_data.get("resolvedTtsStyle"),
            )
            if tts_prompt_prefix and "{lang}" in tts_prompt_prefix:
                tts_prompt_prefix = tts_prompt_prefix.replace("{lang}", target_lang)
            preview_blocked_voices = get_blocked_voices_for_style(
                style=config_data.get("ttsStyle"),
                resolved_style=config_data.get("resolvedTtsStyle"),
            )

            tts = TTSFactory.create_tts(
                tts_system=tts_system,
                device="cpu",
                voice_config=speaker_voice_mappings,
                voice_prompt=config_data.get("speakerTtsPrompts", {}),
                prompt_prefix=tts_prompt_prefix,
                blocked_voices=preview_blocked_voices,
                model=config_data.get("ttsModel"),
            )
            
            # Create segment data
            tts_segment = TTSSegmentData(
                speaker=segment.speaker,
                text=segment.translated_text or segment.original_text,
                voice=segment.voice_id,
                style_prompt=segment.tts_prompt,
                output_path=str(preview_path),
            )
            
            # Synthesize - pass as list, audio is saved to output_path
            alignments = tts.synthesize([tts_segment])
            
            if not alignments or not preview_path.exists():
                raise RuntimeError("TTS synthesis failed - no audio generated")
            
        finally:
            os.chdir(original_cwd)
        
        if job_id:
            update_job_status(job_id, JobStatus.COMPLETED)
        
        return {
            "audioUrl": f"/api/v1/projects/{project_id}/segments/{segment_id}/preview/audio?voice_id={voice_id or ''}",
            "isCached": False,
        }
        
    except Exception as e:
        if job_id:
            update_job_status(job_id, JobStatus.FAILED, str(e))
        raise


@celery_app.task(bind=True, name="src.api.workers.tasks.rephrase_segment")
def rephrase_segment(self, project_id: str, segment_id: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    """Rephrase a segment using LLM."""
    try:
        import json
        import google.generativeai as genai
        
        # Get segment and project from database
        db = get_db_session()
        try:
            segment = db.query(Segment).filter(Segment.id == segment_id).first()
            if not segment:
                raise ValueError(f"Segment {segment_id} not found")
            
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            config_data = project.config or {}
            original_text = segment.translated_text or segment.original_text
            target_lang = config_data.get("targetLang", "ru")
            speaker_gender_section = build_effective_gender_prompt_section(
                config_data.get(SPEAKER_METADATA_CONFIG_KEY),
                speakers=[segment.speaker],
            )
        finally:
            db.close()
        
        # Apply API keys from settings and use Gemini for rephrasing
        _apply_api_keys(config_data.get("apiKeys"))
        api_key = get_api_key("gemini") or get_api_key("google")
        if not api_key:
            raise ValueError("Gemini API key not configured. Set it via /settings/api-keys endpoint or environment variable.")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        rephrase_prompt = f"""Rephrase the following text in {target_lang}.
{f"Additional instructions: {prompt}" if prompt else "Make it more natural and conversational."}

{speaker_gender_section}

Use speaker grammar metadata only for grammatical agreement and self-reference. Do not invent biography or facts.

Text to rephrase: {original_text}

Respond with ONLY the rephrased text, nothing else."""
        
        response = model.generate_content(rephrase_prompt)
        rephrased_text = response.text.strip() if response.text else original_text
        
        # Update segment in database
        db = get_db_session()
        try:
            segment = db.query(Segment).filter(Segment.id == segment_id).first()
            if segment:
                segment.translated_text = rephrased_text
                db.commit()
        finally:
            db.close()
        
        return {"translatedText": rephrased_text}
        
    except Exception as e:
        raise


def _save_segments_to_db(project_id: str, segments: list) -> None:
    """Save translated segments to database."""
    db = get_db_session()
    try:
        # Delete existing segments
        db.query(Segment).filter(Segment.project_id == project_id).delete()
        
        # Speaker colors palette
        colors = [
            "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
            "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
        ]
        speaker_colors = {}
        
        # Create new segments
        for i, seg in enumerate(segments):
            speaker = seg.get("speaker", "UNKNOWN")
            if speaker not in speaker_colors:
                speaker_colors[speaker] = colors[len(speaker_colors) % len(colors)]
            
            segment = Segment(
                project_id=project_id,
                speaker=speaker,
                speaker_color=speaker_colors[speaker],
                start_time=seg.get("start", 0),
                end_time=seg.get("end", 0),
                original_text=seg.get("text", ""),
                translated_text=seg.get("translation", ""),
                sequence=i,
            )
            db.add(segment)
        
        db.commit()
    finally:
        db.close()


def _update_translated_segments_in_db(project_id: str, translated_segments: list) -> None:
    """Update only translated text for existing segments, preserving timing and IDs."""
    db = get_db_session()
    try:
        existing_segments = db.query(Segment).filter(
            Segment.project_id == project_id
        ).order_by(Segment.sequence).all()

        if len(existing_segments) != len(translated_segments):
            raise ValueError(
                f"Segment count mismatch during re-translation: {len(existing_segments)} existing vs {len(translated_segments)} translated"
            )

        for existing_segment, translated_segment in zip(existing_segments, translated_segments):
            existing_segment.translated_text = translated_segment.get("translation", "")
            existing_segment.audio_url = None

        db.commit()
    finally:
        db.close()


def _segment_to_dubbing_format(segment: Segment) -> dict:
    """Convert database segment to SmartDubbing format."""
    return {
        "speaker": segment.speaker,
        "start": segment.start_time,
        "end": segment.end_time,
        "text": segment.original_text,
        "translation": segment.translated_text,
        "voice_id": segment.voice_id,
        "provider": segment.provider.value if segment.provider else None,
        "tts_prompt": segment.tts_prompt,
        "is_muted": segment.is_muted,
    }
