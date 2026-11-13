"""Preset configuration service backed by dubbing_config.yml."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Literal, cast

import yaml

logger = logging.getLogger(__name__)

PresetType = Literal["fast", "hq", "ultra"]

CONFIG_PATH = Path("dubbing_config.yml")

DEFAULT_PRESET_CONFIGS: Dict[PresetType, Dict[str, Any]] = {
    "fast": {
        "persona_id": "none",
        "llm_provider": "openrouter",
        "llm_model_name": "~google/gemini-flash-latest",
        "llm_temperature": 0.5,
        "refinement_llm_provider": "openrouter",
        "refinement_model_name": "openai/gpt-6-sol",
        "refinement_temperature": 1.0,
        "enable_llm_editor": False,
        "editor_llm_provider": "openrouter",
        "editor_model_name": "openai/gpt-5.6-terra",
        "editor_temperature": 1.0,
        "editor_reasoning_effort": "xhigh",
        "tts_system": "gemini38",
        "tts_transport": "openrouter",
        "tts_model": "google/gemini-3.8-flash-lite-tts",
        "tts_fallback_model": "google/gemini-3.8-flash-lite-tts",
        "tts_style": "podcast",
        "tts_prompt_prefix": None,
        "transcription_system": "openrouter",
        "openrouter_stt_model": "microsoft/mai-transcribe-2",
        "keep_background": False,
        "pause_removal": "disabled",
        "voice_auto_selection": True,
        "enable_emotion_enrichment": False,
        "enable_laughter_detection": False,
        "laughter_mode": "dub",
        "enable_content_validation": False,
        "enable_context_style": False,
        "enable_llm_text_adjustment": False,
        "dubbed_volume": 1.0,
        "background_volume": 0.56,
        "normalize_audio": False,
        "use_two_pass_encoding": True,
        "video_quality_preset": "720p",
        "max_workers": 4,
    },
    "hq": {
        "persona_id": "none",
        "llm_provider": "openrouter",
        "llm_model_name": "~google/gemini-flash-latest",
        "llm_temperature": 0.5,
        "refinement_llm_provider": "openrouter",
        "refinement_model_name": "openai/gpt-6-sol",
        "refinement_temperature": 1.0,
        "enable_llm_editor": False,
        "editor_llm_provider": "openrouter",
        "editor_model_name": "openai/gpt-5.6-terra",
        "editor_temperature": 1.0,
        "editor_reasoning_effort": "xhigh",
        "tts_system": "gemini38",
        "tts_transport": "openrouter",
        "tts_model": "google/gemini-3.8-flash-lite-tts",
        "tts_fallback_model": "google/gemini-3.8-flash-lite-tts",
        "tts_style": "podcast",
        "tts_prompt_prefix": None,
        "transcription_system": "openrouter",
        "openrouter_stt_model": "microsoft/mai-transcribe-2",
        "keep_background": False,
        "pause_removal": "disabled",
        "voice_auto_selection": True,
        "enable_emotion_enrichment": False,
        "enable_laughter_detection": False,
        "laughter_mode": "dub",
        "enable_content_validation": True,
        "enable_context_style": True,
        "context_style_max_chars": 140,
        "enable_llm_text_adjustment": False,
        "dubbed_volume": 1.0,
        "background_volume": 0.56,
        "normalize_audio": False,
        "use_two_pass_encoding": True,
        "video_quality_preset": "1080p",
        "max_workers": 4,
    },
    "ultra": {
        "persona_id": "normal",
        "llm_provider": "openrouter",
        "llm_model_name": "~google/gemini-flash-latest",
        "llm_temperature": 0.5,
        "refinement_llm_provider": "openrouter",
        "refinement_model_name": "google/gemini-4-carbon",
        "refinement_temperature": 0.8,
        "enable_llm_editor": False,
        "editor_llm_provider": "openrouter",
        "editor_model_name": "openai/gpt-5.6-sol",
        "editor_temperature": 1.0,
        "editor_reasoning_effort": "xhigh",
        "tts_system": "gemini38",
        "tts_transport": "openrouter",
        "tts_model": "google/gemini-3.8-flash-tts",
        "tts_fallback_model": "google/gemini-3.8-flash-lite-tts",
        "tts_style": "podcast",
        "tts_prompt_prefix": None,
        "transcription_system": "openrouter",
        "openrouter_stt_model": "microsoft/mai-transcribe-2",
        "keep_background": False,
        "pause_removal": "disabled",
        "voice_auto_selection": True,
        "enable_emotion_enrichment": True,
        "enable_laughter_detection": False,
        "laughter_mode": "dub",
        "enable_vocal_event_detection": False,
        "enable_video_emotion_analysis": False,
        "enable_voice_advisor": False,
        "enable_gemini_speaker_labeling": False,
        "enable_content_validation": True,
        "enable_context_style": True,
        "context_style_max_chars": 140,
        "enable_llm_text_adjustment": True,
        "dubbed_volume": 1.0,
        "background_volume": 0.56,
        "background_voice_volume": 0.0,
        "normalize_audio": False,
        "use_two_pass_encoding": True,
        "video_minterpolate_threshold": 1.0,
        "video_quality_preset": "original",
        "max_workers": 4,
    },
}

CAMEL_TO_SNAKE_KEYS: Dict[str, str] = {
    "personaId": "persona_id",
    "keepBackground": "keep_background",
    "llmProvider": "llm_provider",
    "llmModelName": "llm_model_name",
    "llmTemperature": "llm_temperature",
    "refinementLlmProvider": "refinement_llm_provider",
    "refinementModelName": "refinement_model_name",
    "refinementTemperature": "refinement_temperature",
    "enableLlmEditor": "enable_llm_editor",
    "editorLlmProvider": "editor_llm_provider",
    "editorModelName": "editor_model_name",
    "editorTemperature": "editor_temperature",
    "editorReasoningEffort": "editor_reasoning_effort",
    "ttsSystem": "tts_system",
    "ttsTransport": "tts_transport",
    "ttsModel": "tts_model",
    "ttsFallbackModel": "tts_fallback_model",
    "ttsStyle": "tts_style",
    "ttsPromptPrefix": "tts_prompt_prefix",
    "resolvedTtsStyle": "resolved_tts_style",
    "transcriptionSystem": "transcription_system",
    "openrouterSttModel": "openrouter_stt_model",
    "voiceAutoSelection": "voice_auto_selection",
    "enableVoiceCloning": "enable_voice_cloning",
    "enableEmotionEnrichment": "enable_emotion_enrichment",
    "refinerEmotionTags": "refiner_emotion_tags",
    "creativeRecoveryVariants": "creative_recovery_variants",
    "enableVideoEmotionAnalysis": "enable_video_emotion_analysis",
    "videoEmotionAnalysisModel": "video_emotion_analysis_model",
    "videoEmotionAnalysisFps": "video_emotion_analysis_fps",
    "videoEmotionAnalysisDirectionMaxChars": "video_emotion_analysis_direction_max_chars",
    "videoEmotionAnalysisLanguage": "video_emotion_analysis_language",
    "enableVoiceAdvisor": "enable_voice_advisor",
    "voiceAdviceMaxCandidates": "voice_advice_max_candidates",
    "enableGeminiSpeakerLabeling": "enable_gemini_speaker_labeling",
    "geminiSpeakerLabelingModel": "gemini_speaker_labeling_model",
    "geminiSpeakerLabelingMaxSpeakers": "gemini_speaker_labeling_max_speakers",
    "enableLaughterDetection": "enable_laughter_detection",
    "laughterMode": "laughter_mode",
    "enableVocalEventDetection": "enable_vocal_event_detection",
    "enableVocalEventLoudnessDetection": "enable_vocal_event_loudness_detection",
    "vocalEventsMode": "vocal_events_mode",
    "vocalEventKinds": "vocal_event_kinds",
    "enableContentValidation": "enable_content_validation",
    "enableContextStyle": "enable_context_style",
    "contextStyleMaxChars": "context_style_max_chars",
    "enableLlmTextAdjustment": "enable_llm_text_adjustment",
    "dubbedVolume": "dubbed_volume",
    "backgroundVolume": "background_volume",
    "backgroundVoiceVolume": "background_voice_volume",
    "normalizeAudio": "normalize_audio",
    "useTwoPassEncoding": "use_two_pass_encoding",
    "videoQualityPreset": "video_quality_preset",
    "maxWorkers": "max_workers",
    "pauseRemoval": "pause_removal",
    "videoMinterpolateThreshold": "video_minterpolate_threshold",
}

SNAKE_TO_CAMEL_KEYS: Dict[str, str] = {v: k for k, v in CAMEL_TO_SNAKE_KEYS.items()}


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(cast(Dict[str, Any], merged[key]), value)
        else:
            merged[key] = value
    return merged


def _normalize_override_keys(overrides: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in overrides.items():
        normalized[CAMEL_TO_SNAKE_KEYS.get(key, key)] = value
    return normalized


def _load_preset_overrides(config_path: Path = CONFIG_PATH) -> Dict[PresetType, Dict[str, Any]]:
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to read %s: %s", config_path, exc)
        return {}

    raw_presets = loaded.get("presets")
    if not isinstance(raw_presets, dict):
        return {}

    result: Dict[PresetType, Dict[str, Any]] = {}
    for preset_name in DEFAULT_PRESET_CONFIGS:
        data = raw_presets.get(preset_name)
        if isinstance(data, dict):
            result[preset_name] = _normalize_override_keys(data)
    return result


def get_all_preset_configs() -> Dict[PresetType, Dict[str, Any]]:
    """Get all preset configs merged with optional YAML overrides."""
    merged = deepcopy(DEFAULT_PRESET_CONFIGS)
    overrides = _load_preset_overrides()
    for preset_name, override in overrides.items():
        merged[preset_name] = _deep_merge(merged[preset_name], override)


    return merged


def get_preset_config(preset: Optional[str]) -> Dict[str, Any]:
    all_presets = get_all_preset_configs()
    if preset in all_presets:
        return all_presets[cast(PresetType, preset)]
    # Removed legacy presets (e.g. "trailers") fall back to the release
    # flagship profile without rewriting the project's explicit config.
    return all_presets["ultra"]


def _to_frontend_config(data: Dict[str, Any]) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    for key, value in data.items():
        converted[SNAKE_TO_CAMEL_KEYS.get(key, key)] = value
    return converted


def get_frontend_preset_configs() -> Dict[PresetType, Dict[str, Any]]:
    all_presets = get_all_preset_configs()
    return {
        preset_name: _to_frontend_config(cfg)
        for preset_name, cfg in all_presets.items()
    }


def get_frontend_preset_config(preset: Optional[str]) -> Dict[str, Any]:
    return _to_frontend_config(get_preset_config(preset))
