"""Preset configuration service backed by dubbing_config.yml."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Literal, cast

import yaml

from .settings_service import get_api_key

logger = logging.getLogger(__name__)

PresetType = Literal["fast", "hq", "ultra"]

CONFIG_PATH = Path("dubbing_config.yml")

DEFAULT_PRESET_CONFIGS: Dict[PresetType, Dict[str, Any]] = {
    "fast": {
        "llm_provider": "gemini",
        "llm_model_name": "gemini-flash-lite-latest",
        "llm_temperature": 0.5,
        "tts_system": "openai",
        "tts_model": "gpt-4o-mini-tts",
        "tts_fallback_model": "gpt-4o-mini-tts",
        "tts_prompt_prefix": None,
        "keep_background": False,
        "pause_removal": "disabled",
        "voice_auto_selection": True,
        "enable_emotion_enrichment": False,
        "dubbed_volume": 1.0,
        "background_volume": 0.56,
        "use_two_pass_encoding": True,
        "video_quality_preset": "720p",
        "max_workers": 4,
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
        "keep_background": False,
        "pause_removal": "disabled",
        "voice_auto_selection": True,
        "enable_emotion_enrichment": False,
        "dubbed_volume": 1.0,
        "background_volume": 0.56,
        "use_two_pass_encoding": True,
        "video_quality_preset": "1080p",
        "max_workers": 4,
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
        "keep_background": False,
        "pause_removal": "disabled",
        "voice_auto_selection": True,
        "enable_emotion_enrichment": True,
        "dubbed_volume": 1.0,
        "background_volume": 0.56,
        "use_two_pass_encoding": True,
        "video_minterpolate_threshold": 1.0,
        "video_quality_preset": "original",
        "max_workers": 4,
    },
}

CAMEL_TO_SNAKE_KEYS: Dict[str, str] = {
    "keepBackground": "keep_background",
    "llmProvider": "llm_provider",
    "llmModelName": "llm_model_name",
    "llmTemperature": "llm_temperature",
    "refinementLlmProvider": "refinement_llm_provider",
    "refinementModelName": "refinement_model_name",
    "refinementTemperature": "refinement_temperature",
    "ttsSystem": "tts_system",
    "ttsModel": "tts_model",
    "ttsFallbackModel": "tts_fallback_model",
    "ttsPromptPrefix": "tts_prompt_prefix",
    "voiceAutoSelection": "voice_auto_selection",
    "enableEmotionEnrichment": "enable_emotion_enrichment",
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


def get_all_preset_configs() -> Dict[PresetType, Dict[str, Any]]:
    """Get all preset configs merged with optional YAML overrides."""
    merged = deepcopy(DEFAULT_PRESET_CONFIGS)
    overrides = _load_preset_overrides()
    for preset_name, override in overrides.items():
        merged[preset_name] = _deep_merge(merged[preset_name], override)

    for preset_name in list(merged.keys()):
        merged[preset_name] = _apply_ultra_openrouter_gate(preset_name, merged[preset_name])

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
