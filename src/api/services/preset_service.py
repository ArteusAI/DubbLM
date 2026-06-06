"""Preset configuration service backed by dubbing_config.yml."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Literal, cast

import yaml

from src.tts.gemini_tts_wrapper import (
    DEFAULT_GEMINI_TTS_MODEL,
    DEFAULT_GEMINI_TTS_FALLBACK_MODEL,
)

from .settings_service import get_api_key

logger = logging.getLogger(__name__)

PresetType = Literal["fast", "hq", "ultra"]

CONFIG_PATH = Path("dubbing_config.yml")

DEFAULT_PRESET_CONFIGS: Dict[PresetType, Dict[str, Any]] = {
    "fast": {
        "persona_id": "none",
        "llm_provider": "gemini",
        "llm_model_name": "gemini-flash-latest",
        "llm_temperature": 0.5,
        "enable_llm_editor": False,
        "editor_llm_provider": "openrouter",
        "editor_model_name": "openai/gpt-5.4",
        "editor_temperature": 1.0,
        "editor_reasoning_effort": "xhigh",
        "tts_system": "openai",
        "tts_model": "gpt-4o-mini-tts",
        "tts_fallback_model": "gpt-4o-mini-tts",
        "tts_style": "podcast",
        "tts_prompt_prefix": None,
        "keep_background": False,
        "pause_removal": "disabled",
        "voice_auto_selection": True,
        "enable_emotion_enrichment": False,
        "enable_content_validation": False,
        "enable_llm_text_adjustment": False,
        "dubbed_volume": 1.0,
        "background_volume": 0.56,
        "use_two_pass_encoding": True,
        "video_quality_preset": "720p",
        "max_workers": 4,
    },
    "hq": {
        "persona_id": "none",
        "llm_provider": "gemini",
        "llm_model_name": "gemini-flash-latest",
        "llm_temperature": 0.5,
        "refinement_llm_provider": "gemini",
        "refinement_model_name": "gemini-flash-latest",
        "refinement_temperature": 1.0,
        "enable_llm_editor": False,
        "editor_llm_provider": "openrouter",
        "editor_model_name": "openai/gpt-5.4",
        "editor_temperature": 1.0,
        "editor_reasoning_effort": "xhigh",
        "tts_system": "gemini",
        "tts_model": DEFAULT_GEMINI_TTS_FALLBACK_MODEL,
        "tts_fallback_model": DEFAULT_GEMINI_TTS_FALLBACK_MODEL,
        "tts_style": "podcast",
        "tts_prompt_prefix": None,
        "keep_background": False,
        "pause_removal": "disabled",
        "voice_auto_selection": True,
        "enable_emotion_enrichment": False,
        "enable_content_validation": True,
        "enable_llm_text_adjustment": False,
        "dubbed_volume": 1.0,
        "background_volume": 0.56,
        "use_two_pass_encoding": True,
        "video_quality_preset": "1080p",
        "max_workers": 4,
    },
    "ultra": {
        "persona_id": "normal",
        "llm_provider": "gemini",
        "llm_model_name": "gemini-flash-latest",
        "llm_temperature": 0.5,
        "refinement_llm_provider": "openrouter",
        "refinement_model_name": "openai/gpt-5.4",
        "refinement_temperature": 0.8,
        "enable_llm_editor": True,
        "editor_llm_provider": "openrouter",
        "editor_model_name": "openai/gpt-5.4",
        "editor_temperature": 1.0,
        "editor_reasoning_effort": "xhigh",
        "tts_system": "gemini",
        "tts_model": DEFAULT_GEMINI_TTS_MODEL,
        "tts_fallback_model": DEFAULT_GEMINI_TTS_FALLBACK_MODEL,
        "tts_style": "podcast",
        "tts_prompt_prefix": None,
        "keep_background": True,
        "pause_removal": "disabled",
        "voice_auto_selection": True,
        "enable_emotion_enrichment": True,
        "enable_content_validation": True,
        "enable_llm_text_adjustment": True,
        "dubbed_volume": 1.0,
        "background_volume": 0.56,
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
    "ttsModel": "tts_model",
    "ttsFallbackModel": "tts_fallback_model",
    "ttsStyle": "tts_style",
    "ttsPromptPrefix": "tts_prompt_prefix",
    "resolvedTtsStyle": "resolved_tts_style",
    "voiceAutoSelection": "voice_auto_selection",
    "enableEmotionEnrichment": "enable_emotion_enrichment",
    "enableContentValidation": "enable_content_validation",
    "enableLlmTextAdjustment": "enable_llm_text_adjustment",
    "dubbedVolume": "dubbed_volume",
    "backgroundVolume": "background_volume",
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


def _apply_ultra_openrouter_gate(preset: PresetType, config: Dict[str, Any]) -> Dict[str, Any]:
    """Use OpenRouter for Ultra only when OPENROUTER_API_KEY is available."""
    if preset != "ultra":
        return config

    if config.get("llm_provider") != "openrouter":
        return config

    if get_api_key("openrouter"):
        return config

    fallback = DEFAULT_PRESET_CONFIGS["ultra"]
    gated = deepcopy(config)
    gated["llm_provider"] = fallback["llm_provider"]
    gated["llm_model_name"] = fallback["llm_model_name"]
    gated["llm_temperature"] = fallback["llm_temperature"]

    if gated.get("refinement_llm_provider") == "openrouter":
        gated["refinement_llm_provider"] = fallback.get("refinement_llm_provider")
        gated["refinement_model_name"] = fallback.get("refinement_model_name")
        gated["refinement_temperature"] = fallback.get("refinement_temperature")

    return gated


def _apply_editor_openrouter_gate(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback editor OpenRouter settings when OPENROUTER_API_KEY is unavailable."""
    if config.get("editor_llm_provider") != "openrouter":
        return config

    if get_api_key("openrouter"):
        return config

    gated = deepcopy(config)
    gated["editor_llm_provider"] = gated.get("refinement_llm_provider") or gated.get("llm_provider")
    gated["editor_model_name"] = gated.get("refinement_model_name") or gated.get("llm_model_name")
    gated["editor_reasoning_effort"] = "none"
    return gated


def get_all_preset_configs() -> Dict[PresetType, Dict[str, Any]]:
    """Get all preset configs merged with optional YAML overrides."""
    merged = deepcopy(DEFAULT_PRESET_CONFIGS)
    overrides = _load_preset_overrides()
    for preset_name, override in overrides.items():
        merged[preset_name] = _deep_merge(merged[preset_name], override)

    for preset_name in list(merged.keys()):
        merged[preset_name] = _apply_ultra_openrouter_gate(preset_name, merged[preset_name])
        merged[preset_name] = _apply_editor_openrouter_gate(merged[preset_name])

    return merged


def get_preset_config(preset: Optional[str]) -> Dict[str, Any]:
    all_presets = get_all_preset_configs()
    if preset in all_presets:
        return all_presets[cast(PresetType, preset)]
    return all_presets["hq"]


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
