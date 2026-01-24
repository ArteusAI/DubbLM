"""System-wide settings service using YAML file storage."""

import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

SETTINGS_FILE = Path("settings.yml")
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
    "defaults": {
        "ttsSystem": "gemini",
        "sourceLang": "en",
        "targetLang": "ru",
        "personaId": "normal",
        "keepBackground": True,
        "translationPromptPrefix": "",
    },
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
    return get_settings().get("defaults", DEFAULT_SETTINGS["defaults"])


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
    
    return masked


def apply_api_keys_to_env() -> None:
    """Apply API keys from settings to environment variables."""
    settings = get_settings()
    api_keys = settings.get("apiKeys", {})
    
    for provider, env_var in API_KEY_PROVIDERS.items():
        key = api_keys.get(provider, "")
        if key and not os.environ.get(env_var):
            os.environ[env_var] = key

