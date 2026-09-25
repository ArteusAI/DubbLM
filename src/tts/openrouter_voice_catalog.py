"""Catalog metadata and helpers for OpenRouter TTS voices (Qwen, Grok, …)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import time

import requests

from src.utils.speaker_gender import DEFAULT_SPEAKER_GENDER, normalize_gender_value

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_SPEECH_MODELS_URL = f"{OPENROUTER_MODELS_URL}?output_modalities=speech"

GROK_VOICE_TTS_MODEL = "x-ai/grok-voice-tts-1.0"
GROK_MULTILINGUAL = [
    "en", "es", "fr", "de", "it", "pt", "nl", "ru", "pl", "tr",
    "ja", "ko", "zh", "ar", "hi", "id", "th", "vi", "sv", "da", "no", "fi",
]

# Verified working on OpenRouter. DashScope names (Cherry/Alek/…) do NOT work for Qwen here.
OPENROUTER_VOICE_META: Dict[str, Dict[str, Any]] = {
    "loongjohn": {
        "name": "Loongjohn",
        "gender": "male",
        "description": "Male multilingual voice (flash). Good general-purpose RU/EN.",
        "models": ["qwen/qwen-audio-3.0-tts-flash"],
        "languages": ["ru", "en", "zh", "ja", "ko", "de", "es", "fr", "it", "pt", "ar", "id", "th", "vi", "ms", "tl"],
    },
    "longanhuan_v3.6": {
        "name": "Longanhuan",
        "gender": "female",
        "description": "Female multilingual voice (flash). Good general-purpose RU/EN.",
        "models": ["qwen/qwen-audio-3.0-tts-flash"],
        "languages": ["ru", "en", "zh", "ja", "ko", "de", "es", "fr", "it", "pt", "ar", "id", "th", "vi", "ms", "tl"],
    },
    "longanlingxin": {
        "name": "Longanlingxin",
        "gender": "female",
        "description": "Female multilingual voice (plus). Higher quality lattice.",
        "models": ["qwen/qwen-audio-3.0-tts-plus"],
        "languages": ["ru", "en", "zh", "ja", "ko", "de", "es", "fr", "it", "pt", "ar", "id", "th", "vi", "ms", "tl"],
    },
    "longanlufeng": {
        "name": "Longanlufeng",
        "gender": "male",
        "description": "Male multilingual voice (plus). Higher quality lattice.",
        "models": ["qwen/qwen-audio-3.0-tts-plus"],
        "languages": ["ru", "en", "zh", "ja", "ko", "de", "es", "fr", "it", "pt", "ar", "id", "th", "vi", "ms", "tl"],
    },
    "eve": {
        "name": "Eve",
        "gender": "female",
        "description": "Grok Voice TTS. Warm, clear female voice; 20+ languages with auto detection.",
        "models": [GROK_VOICE_TTS_MODEL],
        "languages": list(GROK_MULTILINGUAL),
    },
    "ara": {
        "name": "Ara",
        "gender": "female",
        "description": "Grok Voice TTS. Expressive female voice; 20+ languages with auto detection.",
        "models": [GROK_VOICE_TTS_MODEL],
        "languages": list(GROK_MULTILINGUAL),
    },
    "rex": {
        "name": "Rex",
        "gender": "male",
        "description": "Grok Voice TTS. Confident male voice; 20+ languages with auto detection.",
        "models": [GROK_VOICE_TTS_MODEL],
        "languages": list(GROK_MULTILINGUAL),
    },
    "sal": {
        "name": "Sal",
        "gender": "male",
        "description": "Grok Voice TTS. Calm male voice; 20+ languages with auto detection.",
        "models": [GROK_VOICE_TTS_MODEL],
        "languages": list(GROK_MULTILINGUAL),
    },
    "leo": {
        "name": "Leo",
        "gender": "male",
        "description": "Grok Voice TTS. Energetic male voice; 20+ languages with auto detection.",
        "models": [GROK_VOICE_TTS_MODEL],
        "languages": list(GROK_MULTILINGUAL),
    },
}

# Fallback static model -> voices if Models API is unreachable.
STATIC_MODEL_VOICES: Dict[str, List[str]] = {
    "qwen/qwen-audio-3.0-tts-flash": ["loongjohn", "longanhuan_v3.6"],
    "qwen/qwen-audio-3.0-tts-plus": ["longanlingxin", "longanlufeng"],
    GROK_VOICE_TTS_MODEL: ["eve", "ara", "rex", "sal", "leo"],
}

ALL_OPENROUTER_VOICES: List[str] = list(OPENROUTER_VOICE_META.keys())

DEFAULT_VOICE_BY_GENDER: Dict[str, str] = {
    "male": "loongjohn",
    "female": "longanhuan_v3.6",
    "unknown": "loongjohn",
}

# Preferred defaults when target language is Russian.
DEFAULT_RU_VOICE_BY_GENDER: Dict[str, str] = {
    "male": "loongjohn",
    "female": "longanhuan_v3.6",
    "unknown": "loongjohn",
}

DEFAULT_VOICE_BY_MODEL: Dict[str, str] = {
    "qwen/qwen-audio-3.0-tts-flash": "loongjohn",
    "qwen/qwen-audio-3.0-tts-plus": "longanlufeng",
    GROK_VOICE_TTS_MODEL: "eve",
}


def is_qwen_tts_model(model: Optional[str]) -> bool:
    return str(model or "").strip().lower().startswith("qwen/")


def is_grok_tts_model(model: Optional[str]) -> bool:
    key = str(model or "").strip().lower()
    return key.startswith("x-ai/") or "grok-voice-tts" in key


def max_char_limit_for_model(model: Optional[str]) -> int:
    """Per-request character limit for OpenRouter speech models."""
    if is_grok_tts_model(model):
        return 15000
    return 2000

# Reference only: DashScope system voice names that support Russian natively.
# Not accepted by OpenRouter Qwen endpoints (verified 400). Kept for docs / future.
DASHSCOPE_RU_VOICES: List[Dict[str, str]] = [
    {"id": "Alek", "name": "Alek", "gender": "male",
     "description": "Cold like the Russian spirit — dedicated RU-character male voice (DashScope only)"},
    {"id": "Cherry", "name": "Cherry", "gender": "female", "description": "Sunny young woman"},
    {"id": "Serena", "name": "Serena", "gender": "female", "description": "Gentle young woman"},
    {"id": "Ethan", "name": "Ethan", "gender": "male", "description": "Warm energetic man"},
    {"id": "Neil", "name": "Neil", "gender": "male", "description": "News-anchor male"},
    {"id": "Katerina", "name": "Katerina", "gender": "female", "description": "Mature woman, rich rhythm"},
    {"id": "Jennifer", "name": "Jennifer", "gender": "female", "description": "Cinematic American English female"},
    {"id": "Ryan", "name": "Ryan", "gender": "male", "description": "Dramatic male"},
    {"id": "Moon", "name": "Moon", "gender": "male", "description": "Bold handsome man"},
    {"id": "Kai", "name": "Kai", "gender": "male", "description": "Soothing male"},
    {"id": "Mia", "name": "Mia", "gender": "female", "description": "Gentle female"},
    {"id": "Vivian", "name": "Vivian", "gender": "female", "description": "Confident cute female"},
    {"id": "Maia", "name": "Maia", "gender": "female", "description": "Intellectual gentle female"},
    {"id": "Chelsie", "name": "Chelsie", "gender": "female", "description": "Anime-style girlfriend"},
    {"id": "Momo", "name": "Momo", "gender": "female", "description": "Playful mischievous female"},
    {"id": "Bella", "name": "Bella", "gender": "female", "description": "Bubbly playful female"},
    {"id": "Nofish", "name": "Nofish", "gender": "male", "description": "Designer male"},
    {"id": "Aiden", "name": "Aiden", "gender": "male", "description": "American English young man"},
    {"id": "Eldric Sage", "name": "Eldric Sage", "gender": "male", "description": "Calm wise elder"},
    {"id": "Mochi", "name": "Mochi", "gender": "male", "description": "Clever childlike adult male"},
    {"id": "Bellona", "name": "Bellona", "gender": "female", "description": "Powerful clear hero voice"},
    {"id": "Vincent", "name": "Vincent", "gender": "male", "description": "Raspy smoky male"},
    {"id": "Bunny", "name": "Bunny", "gender": "female", "description": "Cute little girl"},
    {"id": "Elias", "name": "Elias", "gender": "female", "description": "Academic storytelling female"},
    {"id": "Arthur", "name": "Arthur", "gender": "male", "description": "Earthy storytelling male"},
    {"id": "Nini", "name": "Nini", "gender": "female", "description": "Soft clingy female"},
    {"id": "Seren", "name": "Seren", "gender": "female", "description": "Soothing sleep female"},
    {"id": "Pip", "name": "Pip", "gender": "male", "description": "Playful boy"},
    {"id": "Stella", "name": "Stella", "gender": "female", "description": "Sweet teenage girl"},
    {"id": "Andre", "name": "Andre", "gender": "male", "description": "Magnetic steady male"},
]

_live_model_voices_cache: Dict[str, Tuple[float, List[str]]] = {}
_LIVE_CACHE_TTL_SECONDS = 3600.0


def get_openrouter_voice_gender(voice_name: Optional[str]) -> str:
    if not voice_name:
        return DEFAULT_SPEAKER_GENDER
    meta = OPENROUTER_VOICE_META.get(str(voice_name).strip())
    if not meta:
        return DEFAULT_SPEAKER_GENDER
    return normalize_gender_value(meta.get("gender"))


def default_voice_for(
    model: Optional[str] = None,
    language: Optional[str] = None,
    speaker_gender: Optional[str] = None,
) -> str:
    """Pick a sensible default OpenRouter voice for model/language/gender."""
    lang = (language or "").strip().lower()
    gender = normalize_gender_value(speaker_gender)
    model_key = (model or "").strip()
    allowed = voices_for_model(model_key) if model_key else list(ALL_OPENROUTER_VOICES)

    # Model default first when it matches gender (or gender unknown).
    model_default = DEFAULT_VOICE_BY_MODEL.get(model_key)
    if model_default and model_default in allowed:
        if gender == DEFAULT_SPEAKER_GENDER or get_openrouter_voice_gender(model_default) == gender:
            return model_default

    # Global language/gender prefs only if they belong to the active model.
    if lang.startswith("ru"):
        preferred = DEFAULT_RU_VOICE_BY_GENDER.get(gender, DEFAULT_RU_VOICE_BY_GENDER["unknown"])
    else:
        preferred = DEFAULT_VOICE_BY_GENDER.get(gender, DEFAULT_VOICE_BY_GENDER["unknown"])
    if preferred in allowed:
        return preferred

    # Prefer same gender among allowed, else model default, else first allowed.
    for voice_id in allowed:
        if get_openrouter_voice_gender(voice_id) == gender:
            return voice_id
    if model_default and model_default in allowed:
        return model_default
    return allowed[0] if allowed else ALL_OPENROUTER_VOICES[0]


def voices_for_model(model: str, *, allow_live: bool = True) -> List[str]:
    """Return voice IDs supported for a model (live API + static fallback)."""
    model_key = (model or "").strip()
    if allow_live:
        live = _fetch_live_model_voices(model_key)
        if live:
            return live
    if model_key in STATIC_MODEL_VOICES:
        return list(STATIC_MODEL_VOICES[model_key])
    return list(ALL_OPENROUTER_VOICES)


def _fetch_live_model_voices(model: str) -> Optional[List[str]]:
    model_key = (model or "").strip()
    if not model_key:
        return None

    now = time.time()
    cached = _live_model_voices_cache.get(model_key)
    if cached and (now - cached[0]) < _LIVE_CACHE_TTL_SECONDS:
        return list(cached[1])

    try:
        response = requests.get(OPENROUTER_SPEECH_MODELS_URL, timeout=8)
        response.raise_for_status()
        payload = response.json() or {}
        rows = payload.get("data") or []
        found: Optional[List[str]] = None
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_id = str(row.get("id") or "").strip()
            if row_id != model_key and not row_id.startswith(model_key):
                continue
            voices = row.get("supported_voices") or []
            cleaned = [str(v).strip() for v in voices if str(v).strip()]
            if cleaned:
                found = cleaned
                break
        if found:
            _live_model_voices_cache[model_key] = (now, found)
            return list(found)
    except Exception:
        return None
    return None


def build_openrouter_voice_entries(
    *,
    model: Optional[str] = None,
    include_all_known: bool = True,
) -> List[Dict[str, Any]]:
    """Build UI voice entries for OpenRouter provider."""
    entries: List[Dict[str, Any]] = []
    seen = set()

    preferred_ids: Sequence[str]
    if model:
        preferred_ids = voices_for_model(model)
    elif include_all_known:
        preferred_ids = ALL_OPENROUTER_VOICES
    else:
        preferred_ids = ALL_OPENROUTER_VOICES

    for voice_id in preferred_ids:
        if voice_id in seen:
            continue
        seen.add(voice_id)
        meta = OPENROUTER_VOICE_META.get(voice_id, {})
        models = meta.get("models") or []
        model_hint = ""
        if models:
            short = []
            for m in models:
                slug = m.split("/")[-1]
                slug = slug.replace("qwen-audio-3.0-tts-", "").replace("grok-voice-tts-", "grok-")
                short.append(slug)
            model_hint = f" [{'/'.join(short)}]"
        label = meta.get("name") or voice_id
        description = meta.get("description") or ""
        name = f"{label}{model_hint}"
        if description:
            name = f"{label}{model_hint}"
        entries.append({
            "id": voice_id,
            "name": name,
            "provider": "openrouter",
            "gender": meta.get("gender") or "unknown",
            "description": description,
        })
    return entries


def list_openrouter_voices_for_gender(
    speaker_gender: Optional[str],
    *,
    model: Optional[str] = None,
) -> List[str]:
    allowed = voices_for_model(model) if model else list(ALL_OPENROUTER_VOICES)
    gender = normalize_gender_value(speaker_gender)
    if gender == DEFAULT_SPEAKER_GENDER:
        return list(allowed)
    return [v for v in allowed if get_openrouter_voice_gender(v) == gender]
