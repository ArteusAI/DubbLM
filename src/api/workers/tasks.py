"""Celery tasks for background processing."""

import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Literal

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from sqlalchemy.orm.attributes import flag_modified

from .celery_app import celery_app
from ..database.session import get_db_session
from ..database.models import Project, Segment, Job, JobStatus, ProjectStatus, JobType
from ..services.project_manager import ProjectManager
from ..services.settings_service import get_api_key, apply_api_keys_to_env, API_KEY_PROVIDERS

logger = logging.getLogger(__name__)


PresetType = Literal["fast", "hq", "ultra"]

PRESET_CONFIGS: Dict[PresetType, Dict[str, Any]] = {
    "fast": {
        "llm_provider": "gemini",
        "llm_model_name": "gemini-flash-lite-latest",
        "llm_temperature": 0.5,
        "tts_system": "openai",
        "tts_model": "gpt-4o-mini-tts",
        "tts_fallback_model": "gpt-4o-mini-tts",
        "tts_prompt_prefix": None,
        "pause_removal": "disabled",
    },
    "hq": {
        "llm_provider": "gemini",
        "llm_model_name": "gemini-flash-latest",
        "llm_temperature": 0.5,
        "refinement_llm_provider": "gemini",
        "refinement_model_name": "gemini-2.5-pro",
        "refinement_temperature": 1.0,
        "tts_system": "gemini",
        "tts_model": "gemini-2.5-flash-preview-tts",
        "tts_fallback_model": "gemini-2.5-flash-preview-tts",
        "tts_prompt_prefix": "Speak with natural conversational energy, clear articulation:",
        "pause_removal": "cut",
    },
    "ultra": {
        "llm_provider": "gemini",
        "llm_model_name": "gemini-2.5-pro",
        "llm_temperature": 0.5,
        "refinement_llm_provider": "gemini",
        "refinement_model_name": "gemini-2.5-pro",
        "refinement_temperature": 1.0,
        "tts_system": "gemini",
        "tts_model": "gemini-2.5-pro-preview-tts",
        "tts_fallback_model": "gemini-2.5-pro-preview-tts",
        "tts_prompt_prefix": "Speak with natural conversational energy, clear articulation:",
        "pause_removal": "cut",
        "video_minterpolate_threshold": 1.0,  # Enable smooth slowdown for all slowdowns
    },
}


def get_preset_config(preset: Optional[str]) -> Dict[str, Any]:
    """Get configuration values for a given preset."""
    if preset and preset in PRESET_CONFIGS:
        return PRESET_CONFIGS[preset]
    return PRESET_CONFIGS["hq"]


def _apply_api_keys(project_api_keys: Optional[Dict[str, str]] = None) -> None:
    """Apply API keys from settings and optionally project-specific overrides."""
    # First apply system-wide settings
    apply_api_keys_to_env()
    
    # Then apply project-specific overrides
    if project_api_keys:
        for provider, key in project_api_keys.items():
            if key and provider in API_KEY_PROVIDERS:
                os.environ[API_KEY_PROVIDERS[provider]] = key


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
            
            dubbing_config = DubbingConfig()
            dubbing_config.config.update({
                "input": str(source_file),
                "source_language": source_lang,
                "target_language": target_lang,
                "keep_background": config_data.get("keepBackground", True),
                "pause_removal": config_data.get("pauseRemoval", "disabled"),
                "speakers_expected": speaker_count,
                "exit_before_synthesis": True,  # Stop after translation
                "no_cache": False,
                # TTS settings (config_data overrides preset)
                "tts_system": config_data.get("ttsSystem") or preset_config["tts_system"],
                "tts_model": config_data.get("ttsModel") or preset_config.get("tts_model") or "gemini-2.5-flash-preview-tts",
                "tts_fallback_model": preset_config.get("tts_fallback_model") or "gemini-2.5-flash-preview-tts",
                "tts_prompt_prefix": config_data.get("ttsPromptPrefix") or preset_config.get("tts_prompt_prefix"),
                "voice_prompt": config_data.get("speakerTtsPrompts", {}),
                "voice_auto_selection": config_data.get("voiceAutoSelection", True),
                "enable_emotion_analysis": config_data.get("enableEmotionAnalysis", False),
                "enable_emotion_enrichment": config_data.get("enableEmotionEnrichment", False),
                # Transcription settings
                "transcription_system": config_data.get("transcriptionSystem", "assemblyai"),
                "whisper_model": config_data.get("whisperModel", "large-v3"),
                # Translation settings (config_data overrides preset)
                "translator_type": "llm",
                "llm_provider": config_data.get("llmProvider") or preset_config["llm_provider"],
                "llm_model_name": config_data.get("llmModelName") or preset_config["llm_model_name"],
                "llm_temperature": config_data.get("llmTemperature") if config_data.get("llmTemperature") is not None else preset_config["llm_temperature"],
                "refinement_llm_provider": config_data.get("refinementLlmProvider") or preset_config.get("refinement_llm_provider"),
                "refinement_model_name": config_data.get("refinementModelName") or preset_config.get("refinement_model_name"),
                "refinement_temperature": config_data.get("refinementTemperature") if config_data.get("refinementTemperature") is not None else preset_config.get("refinement_temperature", 1.0),
                "refinement_persona": config_data.get("personaId", "normal"),
                "translation_prompt_prefix": config_data.get("translationPromptPrefix"),
                # Audio settings
                "dubbed_volume": config_data.get("dubbedVolume", 1.0),
                "background_volume": config_data.get("backgroundVolume", 0.562341),
                "use_two_pass_encoding": config_data.get("useTwoPassEncoding", True),
                # Processing settings
                "max_workers": config_data.get("maxWorkers", 4),
                # Video processing settings
                "video_minterpolate_threshold": config_data.get("videoMinterpolateThreshold") or preset_config.get("video_minterpolate_threshold"),
                "start_time": config_data.get("startTime"),
                "duration": config_data.get("duration"),
            })
            
            # Apply segment optimization settings if provided
            segments_opt = dubbing_config.config.get("segments_optimization", {})
            if config_data.get("postDiarizationMergeGap") is not None:
                segments_opt["post_diarization_merge_gap"] = config_data["postDiarizationMergeGap"]
            if config_data.get("postTranslationMergeGap") is not None:
                segments_opt["post_translation_merge_gap"] = config_data["postTranslationMergeGap"]
            if config_data.get("maxSegmentDuration") is not None:
                segments_opt["max_segment_duration"] = config_data["maxSegmentDuration"]
            if config_data.get("minSegmentDuration") is not None:
                segments_opt["min_segment_duration"] = config_data["minSegmentDuration"]
            if config_data.get("minPauseDuration") is not None:
                segments_opt["min_pause_duration"] = config_data["minPauseDuration"]
            if config_data.get("preservePauseDuration") is not None:
                segments_opt["preserve_pause_duration"] = config_data["preservePauseDuration"]
            if config_data.get("comfortMinAdjustmentRatio") is not None:
                segments_opt["comfort_min_adjustment_ratio"] = config_data["comfortMinAdjustmentRatio"]
            if config_data.get("comfortMaxAdjustmentRatio") is not None:
                segments_opt["comfort_max_adjustment_ratio"] = config_data["comfortMaxAdjustmentRatio"]
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
            
            if not speakers_rolls or len(speakers_rolls) == 0:
                raise ValueError("No speakers found in the video")
            
            update_job_progress(job_id, 15, "translation", "Translating segments")
            
            # Extract speaker audio
            dubber.speaker_processor.extract_speaker_audio(audio_file, speakers_rolls)
            
            # Progress callback for translation and refinement
            # Before segments created: max 35%
            # Translation: 15-25%, Refinement: 25-35%
            def translation_progress(phase: str, current: int, total: int, text: str = None):
                if phase == "translation":
                    progress = 15 + int((current / total) * 10)
                    step_name = "translation"
                    message = f"Translating chunk {current}/{total}"
                else:  # refinement
                    progress = 25 + int((current / total) * 10)
                    step_name = "refinement"
                    message = f"Refining chunk {current}/{total}"
                update_job_progress(job_id, progress, step_name, message, text=text)
            
            # Translate segments
            translated_segments = dubber.translate_segments(
                transcription, audio_file, progress_callback=translation_progress
            )
            
            update_job_progress(job_id, 35, "saving", "Saving segments to database")
            
            # Save segments to database
            _save_segments_to_db(project_id, translated_segments)

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
            
            # Build TTS prompt prefix based on target language and preset
            target_lang = config_data.get("targetLang", "ru")
            tts_prompt_prefix = config_data.get("ttsPromptPrefix") or preset_config.get("tts_prompt_prefix")
            if tts_prompt_prefix and "{lang}" in tts_prompt_prefix:
                tts_prompt_prefix = tts_prompt_prefix.replace("{lang}", target_lang)
            
            # Build configuration with preset values (config_data overrides preset)
            pause_removal_value = config_data.get("pauseRemoval", "disabled")
            logger.info(f"[DEBUG DUB] preset from config_data: {config_data.get('preset')!r} -> using preset_config for: {preset!r}")
            logger.info(f"[DEBUG DUB] pause_removal from config_data: {pause_removal_value}")
            logger.info(f"[DEBUG DUB] llmModelName from config_data: {config_data.get('llmModelName')!r} -> final: {config_data.get('llmModelName') or preset_config['llm_model_name']!r}")
            logger.info(f"[DEBUG DUB] ttsModel from config_data: {config_data.get('ttsModel')!r} -> final: {config_data.get('ttsModel') or preset_config.get('tts_model')!r}")
            logger.info(f"[DEBUG DUB] refinementModelName from config_data: {config_data.get('refinementModelName')!r}")
            logger.info(f"[DEBUG DUB] enableEmotionEnrichment from config_data: {config_data.get('enableEmotionEnrichment')!r}")
            logger.info(f"[DEBUG DUB] config_data keys: {list(config_data.keys())}")
            
            dubbing_config = DubbingConfig()
            dubbing_config.config.update({
                "input": str(source_file),
                "source_language": config_data.get("sourceLang", "en"),
                "target_language": target_lang,
                "keep_background": config_data.get("keepBackground", True),
                "pause_removal": pause_removal_value,
                "output": str(pm.get_result_video_path(target_lang)),
                "save_translated_subtitles": True,
                # TTS settings (config_data overrides preset)
                "tts_system": config_data.get("ttsSystem") or preset_config["tts_system"],
                "tts_model": config_data.get("ttsModel") or preset_config.get("tts_model") or "gemini-2.5-flash-preview-tts",
                "tts_fallback_model": preset_config.get("tts_fallback_model") or "gemini-2.5-flash-preview-tts",
                "tts_prompt_prefix": tts_prompt_prefix,
                "voice_prompt": config_data.get("speakerTtsPrompts", {}),
                "voice_auto_selection": config_data.get("voiceAutoSelection", True),
                "enable_emotion_analysis": config_data.get("enableEmotionAnalysis", False),
                "enable_emotion_enrichment": config_data.get("enableEmotionEnrichment", False),
                "max_workers": config_data.get("maxWorkers", 4),
                # Audio settings
                "dubbed_volume": config_data.get("dubbedVolume", 1.0),
                "background_volume": config_data.get("backgroundVolume", 0.562341),
                "use_two_pass_encoding": config_data.get("useTwoPassEncoding", True),
                # Video processing settings
                "video_minterpolate_threshold": config_data.get("videoMinterpolateThreshold") or preset_config.get("video_minterpolate_threshold"),
                # Translation settings (for any re-translation)
                "translator_type": "llm",
                "llm_provider": config_data.get("llmProvider") or preset_config["llm_provider"],
                "llm_model_name": config_data.get("llmModelName") or preset_config["llm_model_name"],
                "llm_temperature": config_data.get("llmTemperature") if config_data.get("llmTemperature") is not None else preset_config["llm_temperature"],
                "refinement_llm_provider": config_data.get("refinementLlmProvider") or preset_config.get("refinement_llm_provider"),
                "refinement_model_name": config_data.get("refinementModelName") or preset_config.get("refinement_model_name"),
                "refinement_temperature": config_data.get("refinementTemperature") if config_data.get("refinementTemperature") is not None else preset_config.get("refinement_temperature", 1.0),
                "refinement_persona": config_data.get("personaId", "normal"),
            })
            
            # Apply segment optimization settings if provided
            segments_opt = dubbing_config.config.get("segments_optimization", {})
            if config_data.get("postDiarizationMergeGap") is not None:
                segments_opt["post_diarization_merge_gap"] = config_data["postDiarizationMergeGap"]
            if config_data.get("postTranslationMergeGap") is not None:
                segments_opt["post_translation_merge_gap"] = config_data["postTranslationMergeGap"]
            if config_data.get("maxSegmentDuration") is not None:
                segments_opt["max_segment_duration"] = config_data["maxSegmentDuration"]
            if config_data.get("minSegmentDuration") is not None:
                segments_opt["min_segment_duration"] = config_data["minSegmentDuration"]
            if config_data.get("minPauseDuration") is not None:
                segments_opt["min_pause_duration"] = config_data["minPauseDuration"]
            if config_data.get("preservePauseDuration") is not None:
                segments_opt["preserve_pause_duration"] = config_data["preservePauseDuration"]
            if config_data.get("comfortMinAdjustmentRatio") is not None:
                segments_opt["comfort_min_adjustment_ratio"] = config_data["comfortMinAdjustmentRatio"]
            if config_data.get("comfortMaxAdjustmentRatio") is not None:
                segments_opt["comfort_max_adjustment_ratio"] = config_data["comfortMaxAdjustmentRatio"]
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
            
            # Progress callback for speech synthesis (48% to 70%)
            total_segments = len(segments_data)
            def synthesis_progress(current: int, total: int, text: str = None):
                progress = 48 + int((current / total) * 22) if total > 0 else 48
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
                grouping_progress_callback=grouping_progress
            )
            
            # Process background with progress callback
            background_audio_path = None
            if config_data.get("keepBackground", True):
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
            def video_combine_progress(current: int, total: int, message: str = ""):
                progress = 84 + int((current / total) * 15) if total > 0 else 84
                update_job_progress(job_id, progress, "video_combine", message or f"Processing video: {current}/{total}s")
            
            def video_combine_log(message: str):
                add_job_log(job_id, message)
            
            # Combine with video
            segments_opt = dubbing_config.config.get("segments_optimization", {})
            
            # Determine effective pause removal
            pause_removal = dubbing_config.get("pause_removal", "disabled")
            
            output_video_path, _ = dubber.video_processor.combine_audio_with_video(
                video_path=str(source_file),
                translated_audio_path=translated_audio_path,
                background_audio_path=background_audio_path,
                output_file=str(pm.get_result_video_path(config_data.get("targetLang", "ru"))),
                source_language=config_data.get("sourceLang", "en"),
                target_language=config_data.get("targetLang", "ru"),
                pause_removal=pause_removal,
                min_pause_duration=segments_opt.get("min_pause_duration", 3),
                preserve_pause_duration=segments_opt.get("preserve_pause_duration", 1.5),
                progress_callback=video_combine_progress,
                log_callback=video_combine_log,
            )
            
            # Save subtitles to results directory
            source_lang = config_data.get("sourceLang", "en")
            target_lang = config_data.get("targetLang", "ru")
            
            source_srt_path = str(pm.get_result_subtitles_path(source_lang, "srt"))
            target_srt_path = str(pm.get_result_subtitles_path(target_lang, "srt"))
            
            dubber.subtitle_manager.save_subtitles(segments_data, "original", source_srt_path)
            dubber.subtitle_manager.save_subtitles(segments_data, "translation", target_srt_path)
            
            update_job_progress(job_id, 100, "complete", "Dubbing complete")
            
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
            voice_id = segment.voice_id
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
            
            # Build TTS prompt prefix
            target_lang = config_data.get("targetLang", "ru")
            tts_prompt_prefix = config_data.get("ttsPromptPrefix")
            if tts_prompt_prefix and "{lang}" in tts_prompt_prefix:
                tts_prompt_prefix = tts_prompt_prefix.replace("{lang}", target_lang)
                
            tts = TTSFactory.create_tts(
                tts_system=tts_system,
                device="cpu",
                voice_prompt=config_data.get("speakerTtsPrompts", {}),
                prompt_prefix=tts_prompt_prefix,
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
