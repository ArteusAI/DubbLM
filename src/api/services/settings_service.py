"""System-wide settings service using YAML file storage."""

import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

SETTINGS_FILE = Path("settings.yml")
DUBBING_CONFIG_FILE = Path("dubbing_config.yml")
_lock = threading.Lock()

# Single source of truth for API key providers and their environment variable names
API_KEY_PROVIDERS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "huggingface": "HF_TOKEN",
    "openrouter": "OPENROUTER_API_KEY",
    "assemblyai": "ASSEMBLYAI_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}

DEFAULT_SETTINGS = {
    "apiKeys": {provider: "" for provider in API_KEY_PROVIDERS},
    "defaults": {},
}


def _deep_merge(base: dict, update: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_dubbing_defaults() -> Dict[str, Any]:
    """Build frontend defaults from dubbing_config.yml."""
    base: Dict[str, Any] = {
        "sourceLang": "auto",
        "targetLang": "ru",
        "personaId": "normal",
        "preset": "ultra",
        "keepBackground": False,
        "pauseRemoval": "disabled",
        "translationPromptPrefix": "",
        "transcriptionSystem": "openrouter",
        "openrouterSttModel": "microsoft/mai-transcribe-2",
        "ttsSystem": "gemini38",
        "ttsTransport": "openrouter",
        "ttsModel": "google/gemini-3.8-flash-tts",
        "enableContextStyle": True,
        "llmProvider": "openrouter",
        "llmModelName": "~google/gemini-flash-latest",
        "llmTemperature": 0.5,
        "enableLlmEditor": False,
        "enableLlmTextAdjustment": True,
        "refinementLlmProvider": "openrouter",
        "refinementModelName": "google/gemini-4-carbon",
        "refinementTemperature": 0.8,
        "editorLlmProvider": "openrouter",
        "editorModelName": "openai/gpt-5.6-sol",
        "editorTemperature": 1.0,
        "editorReasoningEffort": "xhigh",
        "voiceAutoSelection": True,
        "enableVoiceCloning": None,
        "enableSpeakerGenderInference": True,
        "enableEmotionEnrichment": False,
        "enableVocalEventDetection": False,
        "enableVocalEventLoudnessDetection": False,
        "vocalEventsMode": "preserve",
        "enableContentValidation": True,
        "contentValidatorProvider": "whisper",
        "contentValidatorWhisperModel": "base",
        "contentValidatorWhisperComputeType": "int8",
        "contentValidatorWhisperCpuThreads": 2,
        "contentValidatorSpeechModel": "nano",
        "dubbedVolume": 1.0,
        "backgroundVolume": 0.562341,
        "backgroundVoiceVolume": 0.0,
        "normalizeAudio": False,
        "useTwoPassEncoding": True,
        "maxWorkers": 4,
        "segmentStretch": "audio_and_video",
    }

    if not DUBBING_CONFIG_FILE.exists():
        return base

    try:
        with DUBBING_CONFIG_FILE.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return base

    mapping = {
        "source_language": "sourceLang",
        "target_language": "targetLang",
        "persona_id": "personaId",
        "refinement_persona": "personaId",
        "keep_background": "keepBackground",
        "pause_removal": "pauseRemoval",
        "transcription_system": "transcriptionSystem",
        "openrouter_stt_model": "openrouterSttModel",
        "whisper_model": "whisperModel",
        "llm_provider": "llmProvider",
        "llm_model_name": "llmModelName",
        "llm_temperature": "llmTemperature",
        "refinement_llm_provider": "refinementLlmProvider",
        "refinement_model_name": "refinementModelName",
        "refinement_temperature": "refinementTemperature",
        "enable_llm_editor": "enableLlmEditor",
        "enable_llm_text_adjustment": "enableLlmTextAdjustment",
        "editor_llm_provider": "editorLlmProvider",
        "editor_model_name": "editorModelName",
        "editor_temperature": "editorTemperature",
        "editor_reasoning_effort": "editorReasoningEffort",
        "translation_prompt_prefix": "translationPromptPrefix",
        "tts_system": "ttsSystem",
        "tts_transport": "ttsTransport",
        "tts_model": "ttsModel",
        "tts_prompt_prefix": "ttsPromptPrefix",
        "voice_auto_selection": "voiceAutoSelection",
        "enable_voice_cloning": "enableVoiceCloning",
        "enable_emotion_enrichment": "enableEmotionEnrichment",
        "enable_video_emotion_analysis": "enableVideoEmotionAnalysis",
        "video_emotion_analysis_model": "videoEmotionAnalysisModel",
        "video_emotion_analysis_fps": "videoEmotionAnalysisFps",
        "video_emotion_analysis_direction_max_chars": "videoEmotionAnalysisDirectionMaxChars",
        "video_emotion_analysis_language": "videoEmotionAnalysisLanguage",
        "enable_voice_advisor": "enableVoiceAdvisor",
        "voice_advice_max_candidates": "voiceAdviceMaxCandidates",
        "enable_laughter_detection": "enableLaughterDetection",
        "laughter_mode": "laughterMode",
        "enable_vocal_event_detection": "enableVocalEventDetection",
        "enable_vocal_event_loudness_detection": "enableVocalEventLoudnessDetection",
        "vocal_events_mode": "vocalEventsMode",
        "enable_content_validation": "enableContentValidation",
        "enable_context_style": "enableContextStyle",
        "context_style_max_chars": "contextStyleMaxChars",
        "content_validator_provider": "contentValidatorProvider",
        "content_validator_whisper_model": "contentValidatorWhisperModel",
        "content_validator_whisper_compute_type": "contentValidatorWhisperComputeType",
        "content_validator_whisper_cpu_threads": "contentValidatorWhisperCpuThreads",
        "content_validator_speech_model": "contentValidatorSpeechModel",
        "dubbed_volume": "dubbedVolume",
        "background_volume": "backgroundVolume",
        "background_voice_volume": "backgroundVoiceVolume",
        "normalize_audio": "normalizeAudio",
        "use_two_pass_encoding": "useTwoPassEncoding",
        "max_workers": "maxWorkers",
        "segment_stretch": "segmentStretch",
    }
    for source_key, target_key in mapping.items():
        value = cfg.get(source_key)
        if value is not None:
            base[target_key] = value

    default_preset = cfg.get("default_preset")
    if default_preset in {"fast", "hq", "ultra"}:
        base["preset"] = default_preset
        preset_values = cfg.get("presets", {}).get(default_preset)
        if isinstance(preset_values, dict):
            for source_key, target_key in mapping.items():
                value = preset_values.get(source_key)
                if value is not None:
                    base[target_key] = value

    segments_optimization = cfg.get("segments_optimization")
    if isinstance(segments_optimization, dict):
        segment_mapping = {
            "post_diarization_merge_gap": "postDiarizationMergeGap",
            "post_translation_merge_gap": "postTranslationMergeGap",
            "max_segment_duration": "maxSegmentDuration",
            "min_segment_duration": "minSegmentDuration",
            "comfort_min_adjustment_ratio": "comfortMinAdjustmentRatio",
            "comfort_max_adjustment_ratio": "comfortMaxAdjustmentRatio",
            "min_pause_duration": "minPauseDuration",
            "preserve_pause_duration": "preservePauseDuration",
            "video_minterpolate_threshold": "videoMinterpolateThreshold",
        }
        for source_key, target_key in segment_mapping.items():
            value = segments_optimization.get(source_key)
            if value is not None:
                base[target_key] = value

    return base


def get_settings() -> Dict[str, Any]:
    """Load settings from YAML file."""
    with _lock:
        if not SETTINGS_FILE.exists():
            return DEFAULT_SETTINGS.copy()
        
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return _deep_merge(DEFAULT_SETTINGS, data)
        except (yaml.YAMLError, IOError):
            return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> None:
    """Save settings to YAML file."""
    with _lock:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            yaml.dump(settings, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def update_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge updates into current settings and save."""
    current = get_settings()
    merged = _deep_merge(current, updates)
    save_settings(merged)
    return merged


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for provider with environment variable fallback."""
    # First check settings file
    settings = get_settings()
    key = settings.get("apiKeys", {}).get(provider, "")
    
    if key:
        return key
    
    # Fallback to environment variable
    env_var = API_KEY_PROVIDERS.get(provider)
    if env_var:
        return os.environ.get(env_var)
    
    return None


def set_api_key(provider: str, key: str) -> None:
    """Set API key for a provider."""
    settings = get_settings()
    if "apiKeys" not in settings:
        settings["apiKeys"] = {}
    settings["apiKeys"][provider] = key
    save_settings(settings)


def _env_file_path() -> Path:
    """Locate the project .env file (the container workdir is /app)."""
    return Path(os.getenv("DUBBLM_ENV_FILE", ".env")).expanduser()


def _read_env_value(name: str, path: Optional[Path] = None) -> Optional[str]:
    env_path = path or _env_file_path()
    if not env_path.exists() or not env_path.is_file():
        return None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip().removeprefix("export ").strip() == name:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                return value or None
    except OSError:
        return None
    return None


def has_openrouter_key() -> bool:
    """True when an OpenRouter key is available from env, settings or .env."""
    if get_api_key("openrouter"):
        return True
    return bool(_read_env_value("OPENROUTER_API_KEY"))


def _write_env_value(name: str, value: str) -> Path:
    """Atomically update one key in .env while preserving comments/order."""
    env_path = _env_file_path()
    lines: list[str] = []
    if env_path.exists() and env_path.is_file():
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []

    new_line = f"{name}={value}"
    replaced = False
    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            updated.append(line)
            continue
        key = stripped.split("=", 1)[0].strip().removeprefix("export ").strip()
        if key == name:
            if not replaced:
                updated.append(new_line)
                replaced = True
            # Drop duplicate assignments.
            continue
        updated.append(line)
    if not replaced:
        if updated and updated[-1].strip():
            updated.append("")
        updated.append(new_line)
    elif not updated:
        updated.append(new_line)

    env_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = env_path.with_name(env_path.name + ".tmp")
    temp_path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")
    try:
        os.chmod(temp_path, 0o600)
    except OSError:
        pass
    try:
        os.replace(temp_path, env_path)
    except OSError as exc:
        # A single-file bind mount (Docker) is a mount point and cannot be
        # replaced with rename(2). Fall back to truncating the mounted file.
        import errno

        if exc.errno not in (errno.EBUSY, errno.EXDEV, errno.EPERM, errno.EACCES, errno.EROFS):
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        env_path.write_text(temp_path.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            os.chmod(env_path, 0o600)
        except OSError:
            pass
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return env_path


def save_openrouter_key(key: str) -> Dict[str, Any]:
    """Persist the OpenRouter key to .env, settings.yml and the environment."""
    cleaned = str(key or "").strip()
    if not cleaned:
        raise ValueError("OpenRouter key is empty")
    env_path: Optional[Path] = None
    env_error: Optional[str] = None
    try:
        env_path = _write_env_value("OPENROUTER_API_KEY", cleaned)
    except OSError as exc:
        # settings.yml is still enough for the worker; report the env issue.
        env_error = str(exc)
    set_api_key("openrouter", cleaned)
    os.environ["OPENROUTER_API_KEY"] = cleaned
    return {
        "configured": True,
        "source": "env+settings" if env_path else "settings",
        "env_file": str(env_path) if env_path else None,
        "env_error": env_error,
        "masked": mask_api_key(cleaned),
    }


def get_defaults() -> Dict[str, Any]:
    """Get default project settings."""
    settings_defaults = get_settings().get("defaults", {})
    if not isinstance(settings_defaults, dict):
        settings_defaults = {}

    # Source-of-truth defaults come from dubbing_config.yml.
    # settings.yml defaults are merged only as fallback extras.
    return _deep_merge(settings_defaults, _load_dubbing_defaults())


def mask_api_key(key: Optional[str]) -> str:
    """Mask API key for display (show first 4 and last 4 chars)."""
    if not key or len(key) < 12:
        return "***" if key else ""
    return f"{key[:4]}...{key[-4:]}"


def get_settings_masked() -> Dict[str, Any]:
    """Get settings with API keys masked for safe display."""
    settings = get_settings()
    masked = settings.copy()
    
    if "apiKeys" in masked:
        masked["apiKeys"] = {
            provider: mask_api_key(key) 
            for provider, key in masked["apiKeys"].items()
        }
    masked["defaults"] = get_defaults()
    
    return masked


def apply_api_keys_to_env() -> None:
    """Apply API keys from settings to environment variables."""
    settings = get_settings()
    api_keys = settings.get("apiKeys", {})
    
    for provider, env_var in API_KEY_PROVIDERS.items():
        key = api_keys.get(provider, "")
        if key and not os.environ.get(env_var):
            os.environ[env_var] = key
