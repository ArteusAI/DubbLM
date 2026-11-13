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

# Voice character of the prebuilt voices, used to match a speaker against a
# voice. The style column follows the official Gemini voice table; the age
# and energy columns are curated approximations (Google does not publish a
# pitch/energy table), so they are only used to *rank* candidates, never to
# exclude one.
GEMINI_VOICE_TRAITS: Dict[str, Dict[str, str]] = {
    "Achernar": {"style": "soft", "timbre": "soft, gentle, airy", "age": "adult", "energy": "medium"},
    "Achird": {"style": "bright", "timbre": "bright, youthful, clear", "age": "young adult", "energy": "high"},
    "Algenib": {"style": "gravelly", "timbre": "rough, gravelly, low", "age": "older", "energy": "medium"},
    "Algieba": {"style": "smooth", "timbre": "smooth, warm, rounded", "age": "adult", "energy": "medium"},
    "Alnilam": {"style": "firm", "timbre": "firm, steady, clear", "age": "adult", "energy": "medium"},
    "Aoede": {"style": "breezy", "timbre": "light, breezy, bright", "age": "young adult", "energy": "medium"},
    "Autonoe": {"style": "clear", "timbre": "clear, crisp, articulate", "age": "adult", "energy": "medium"},
    "Callirrhoe": {"style": "easy-going", "timbre": "relaxed, warm, casual", "age": "adult", "energy": "low"},
    "Charon": {"style": "deep", "timbre": "deep, resonant, authoritative", "age": "older", "energy": "low"},
    "Despina": {"style": "smooth", "timbre": "smooth, soft, even", "age": "adult", "energy": "low"},
    "Enceladus": {"style": "crisp", "timbre": "crisp, precise, cool", "age": "adult", "energy": "medium"},
    "Erinome": {"style": "clear", "timbre": "clear, precise, cool", "age": "adult", "energy": "medium"},
    "Fenrir": {"style": "excitable", "timbre": "excitable, sharp, energetic", "age": "young adult", "energy": "high"},
    "Gacrux": {"style": "mature", "timbre": "mature, low, measured", "age": "older", "energy": "low"},
    "Iapetus": {"style": "even", "timbre": "even, flat, controlled", "age": "adult", "energy": "low"},
    "Kore": {"style": "firm", "timbre": "firm, confident, clear", "age": "adult", "energy": "medium"},
    "Laomedeia": {"style": "upbeat", "timbre": "upbeat, bright, lively", "age": "young adult", "energy": "high"},
    "Leda": {"style": "soft", "timbre": "soft, warm, gentle", "age": "adult", "energy": "low"},
    "Orus": {"style": "firm", "timbre": "firm, authoritative, deep", "age": "adult", "energy": "medium"},
    "Puck": {"style": "upbeat", "timbre": "upbeat, bright, playful", "age": "young adult", "energy": "high"},
    "Pulcherrima": {"style": "forward", "timbre": "forward, energetic, sharp", "age": "adult", "energy": "high"},
    "Rasalgethi": {"style": "informative", "timbre": "informative, clear, neutral", "age": "adult", "energy": "medium"},
    "Sadachbia": {"style": "lively", "timbre": "lively, bright, quick", "age": "young adult", "energy": "high"},
    "Sadaltager": {"style": "knowledgeable", "timbre": "knowledgeable, calm, warm", "age": "older", "energy": "medium"},
    "Schedar": {"style": "even", "timbre": "even, steady, low", "age": "older", "energy": "low"},
    "Sulafat": {"style": "warm", "timbre": "warm, soft, rich", "age": "adult", "energy": "medium"},
    "Umbriel": {"style": "easy-going", "timbre": "easy-going, relaxed, warm", "age": "adult", "energy": "low"},
    "Vindemiatrix": {"style": "gentle", "timbre": "gentle, soft, calm", "age": "adult", "energy": "low"},
    "Zephyr": {"style": "lively", "timbre": "lively, bright, warm", "age": "young adult", "energy": "high"},
    "Zubenelgenubi": {"style": "casual", "timbre": "casual, relaxed, light", "age": "young adult", "energy": "medium"},
}

# Use stable non-breathy defaults when gender is known and the configured
# default voice is either missing or incompatible.
DEFAULT_GEMINI_VOICE_BY_GENDER: Dict[str, str] = {
    "female": "Kore",
    "male": "Puck",
    "unknown": "Kore",
}


def get_gemini_voice_traits(voice_name: Optional[str]) -> Dict[str, str]:
    """Return the character of a Gemini voice (empty when unknown)."""
    if not voice_name:
        return {}
    return dict(GEMINI_VOICE_TRAITS.get(str(voice_name).strip(), {}))


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
