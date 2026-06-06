"""Catalog metadata and helpers for Gemini prebuilt voices."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from src.utils.speaker_gender import DEFAULT_SPEAKER_GENDER, normalize_gender_value

# Based on the official Gemini TTS voice table in Google Cloud documentation.
GEMINI_VOICE_GENDERS: Dict[str, str] = {
    "Achernar": "female",
    "Achird": "male",
    "Algenib": "male",
    "Algieba": "male",
    "Alnilam": "male",
    "Aoede": "female",
    "Autonoe": "female",
    "Callirrhoe": "female",
    "Charon": "male",
    "Despina": "female",
    "Enceladus": "male",
    "Erinome": "female",
    "Fenrir": "male",
    "Gacrux": "female",
    "Iapetus": "male",
    "Kore": "female",
    "Laomedeia": "female",
    "Leda": "female",
    "Orus": "male",
    "Puck": "male",
    "Pulcherrima": "female",
    "Rasalgethi": "male",
    "Sadachbia": "male",
    "Sadaltager": "male",
    "Schedar": "male",
    "Sulafat": "female",
    "Umbriel": "male",
    "Vindemiatrix": "female",
    "Zephyr": "female",
    "Zubenelgenubi": "male",
}

ALL_GEMINI_VOICES: List[str] = list(GEMINI_VOICE_GENDERS.keys())

# Use stable non-breathy defaults when gender is known and the configured
# default voice is either missing or incompatible.
DEFAULT_GEMINI_VOICE_BY_GENDER: Dict[str, str] = {
    "female": "Kore",
    "male": "Puck",
    "unknown": "Kore",
}


def get_gemini_voice_gender(voice_name: Optional[str]) -> str:
    """Return the documented Gemini voice gender or ``unknown``."""
    if not voice_name:
        return DEFAULT_SPEAKER_GENDER
    return GEMINI_VOICE_GENDERS.get(str(voice_name).strip(), DEFAULT_SPEAKER_GENDER)


def is_gemini_voice_gender_compatible(
    voice_name: Optional[str],
    speaker_gender: Optional[str],
) -> bool:
    """Return whether a Gemini voice is compatible with the speaker gender."""
    normalized_gender = normalize_gender_value(speaker_gender)
    if normalized_gender == DEFAULT_SPEAKER_GENDER:
        return True
    return get_gemini_voice_gender(voice_name) == normalized_gender


def list_gemini_voices_for_gender(speaker_gender: Optional[str]) -> List[str]:
    """List Gemini voices compatible with the given speaker gender."""
    normalized_gender = normalize_gender_value(speaker_gender)
    if normalized_gender == DEFAULT_SPEAKER_GENDER:
        return list(ALL_GEMINI_VOICES)
    return [
        voice_name
        for voice_name, voice_gender in GEMINI_VOICE_GENDERS.items()
        if voice_gender == normalized_gender
    ]


def build_gemini_voice_entries() -> List[Dict[str, str]]:
    """Build API-ready Gemini voice metadata entries."""
    return [
        {
            "id": voice_name,
            "name": voice_name,
            "provider": "gemini",
            "gender": voice_gender,
        }
        for voice_name, voice_gender in GEMINI_VOICE_GENDERS.items()
    ]


def resolve_default_gemini_voice(
    speaker_gender: Optional[str],
    *,
    preferred_voice: Optional[str] = None,
    blocked_voices: Optional[Iterable[str]] = None,
) -> str:
    """Resolve a stable Gemini fallback voice for a speaker gender."""
    normalized_gender = normalize_gender_value(speaker_gender)
    blocked_lookup = {
        str(voice_name).strip().lower()
        for voice_name in (blocked_voices or [])
        if str(voice_name).strip()
    }
    preferred = str(preferred_voice or "").strip()

    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)

    if normalized_gender in DEFAULT_GEMINI_VOICE_BY_GENDER:
        candidates.append(DEFAULT_GEMINI_VOICE_BY_GENDER[normalized_gender])

    if normalized_gender == DEFAULT_SPEAKER_GENDER:
        candidates.extend(ALL_GEMINI_VOICES)
    else:
        candidates.extend(list_gemini_voices_for_gender(normalized_gender))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate.lower() in blocked_lookup:
            continue
        if candidate not in GEMINI_VOICE_GENDERS:
            continue
        if is_gemini_voice_gender_compatible(candidate, normalized_gender):
            return candidate

    fallback_gender = normalized_gender if normalized_gender in {"male", "female"} else DEFAULT_SPEAKER_GENDER
    return DEFAULT_GEMINI_VOICE_BY_GENDER[fallback_gender]
