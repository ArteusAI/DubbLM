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
        "sourceLang": "en",
        "targetLang": "ru",
        "personaId": "normal",
        "preset": "hq",
        "keepBackground": False,
        "pauseRemoval": "disabled",
        "translationPromptPrefix": "",
        "ttsSystem": "gemini",
        "llmProvider": "gemini",
        "llmModelName": "gemini-2.5-pro",
        "llmTemperature": 0.5,
        "voiceAutoSelection": True,
        "enableEmotionEnrichment": False,
        "dubbedVolume": 1.0,
        "backgroundVolume": 0.562341,
        "useTwoPassEncoding": True,
        "maxWorkers": 4,
        "segmentStretch": "audio_and_video",
        "geminiMultiSpeakerEnabled": False,
        "geminiMultiSpeakerMaxBatchTokens": 1200,
        "geminiMultiSpeakerMaxTurns": 8,
        "geminiMultiSpeakerPauseRepeats": 3,
        "geminiMultiSpeakerMinPauseMs": 1200,
        "geminiMultiSpeakerBoundaryRetryAttempts": 2,
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
        "refinement_persona": "personaId",
        "keep_background": "keepBackground",
        "pause_removal": "pauseRemoval",
        "transcription_system": "transcriptionSystem",
        "whisper_model": "whisperModel",
        "llm_provider": "llmProvider",
        "llm_model_name": "llmModelName",
        "llm_temperature": "llmTemperature",
        "refinement_llm_provider": "refinementLlmProvider",
        "refinement_model_name": "refinementModelName",
        "refinement_temperature": "refinementTemperature",
        "translation_prompt_prefix": "translationPromptPrefix",
        "tts_system": "ttsSystem",
        "tts_model": "ttsModel",
        "tts_prompt_prefix": "ttsPromptPrefix",
        "voice_auto_selection": "voiceAutoSelection",
        "enable_emotion_enrichment": "enableEmotionEnrichment",
        "dubbed_volume": "dubbedVolume",
        "background_volume": "backgroundVolume",
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
            "gemini_multi_speaker_enabled": "geminiMultiSpeakerEnabled",
            "gemini_multi_speaker_max_batch_tokens": "geminiMultiSpeakerMaxBatchTokens",
            "gemini_multi_speaker_max_turns": "geminiMultiSpeakerMaxTurns",
            "gemini_multi_speaker_pause_repeats": "geminiMultiSpeakerPauseRepeats",
            "gemini_multi_speaker_min_pause_ms": "geminiMultiSpeakerMinPauseMs",
            "gemini_multi_speaker_boundary_retry_attempts": "geminiMultiSpeakerBoundaryRetryAttempts",
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
