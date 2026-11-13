"""Voice advice for the dubbing cast (from the Gemini VLM delivery analysis).

The delivery analysis already listens to the original performance, so it can
describe the *voice* of every character: timbre, age impression, energy and
accent.  This module turns that description into three things:

* :class:`VoiceProfile` - a comparable profile of one character;
* :meth:`VoiceProfile.prompt_fragment` - a short phrase for the TTS
  instruction ("low, slightly raspy"), so the dub keeps the character;
* a score of how well a catalog voice matches the profile - used to narrow the
  candidate list before the embedding matcher picks the final voice.

The advice is a *recommendation*: the filters for gender, age and accent stay
authoritative, the matching only re-ranks what is left.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.dubbing.core.log_config import get_logger

logger = get_logger(__name__)

# Canonical age buckets used by the advisor.
AGE_VOCABULARY: Tuple[str, ...] = ("child", "teen", "young adult", "adult", "older")

# Age buckets of the voice catalogs -> advisor buckets.
_AGE_ALIASES: Dict[str, str] = {
    "child": "child",
    "children": "child",
    "kid": "child",
    "kids": "child",
    "детский": "child",
    "детский голос": "child",
    "teen": "teen",
    "teenage": "teen",
    "teenager": "teen",
    "подросток": "teen",
    "подростковый": "teen",
    "young": "young adult",
    "young adult": "young adult",
    "youth": "young adult",
    "youthful": "young adult",
    "молодой": "young adult",
    "молодая": "young adult",
    "молодое": "young adult",
    "adult": "adult",
    "взрослый": "adult",
    "взрослая": "adult",
    "senior": "older",
    "older": "older",
    "old": "older",
    "elderly": "older",
    "старший": "older",
    "старшая": "older",
    "пожилой": "older",
    "пожилая": "older",
}

# Timbre dimensions with their keyword probes (English and Russian, because
# the Fish voice descriptions are written in both languages).
TIMBRE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "low": (
        "low", "deep", "bass", "baritone", "rich tone", "resonant",
        "низк", "глубок", "бас", "низкоголос",
    ),
    "high": (
        "high", "high-pitched", "bright", "shrill", "piercing", "thin",
        "высок", "звонк", "пронзительн", "тонк",
    ),
    "soft": (
        "soft", "gentle", "mellow", "tender", "velvet", "silky",
        "мягк", "нежн", "бархатн", "ласков",
    ),
    "rough": (
        "rough", "raspy", "gravelly", "hoarse", "scratchy", "gritty",
        "хрипл", "скрипуч", "сипл", "груб", "надрыв",
    ),
    "breathy": (
        "breathy", "airy", "whispery", "hushed", "aspirated",
        "шепот", "дым", "приглуш", "шелест",
    ),
    "clear": (
        "clear", "crisp", "articulate", "clean", "precise", "distinct",
        "четк", "ясно", "чист", "разборчив", "чётк",
    ),
    "warm": (
        "warm", "velvet", "round", "full", "sonorous",
        "тепл", "теплый", "округл", "бархатн", "насыщен",
    ),
    "cold": (
        "cold", "icy", "clinical", "flat", "detached",
        "холодн", "ледя", "равнодушн", "отстран",
    ),
    "energetic": (
        "energetic", "lively", "dynamic", "spirited", "vivid",
        "энергичн", "жив", "активн", "задорн",
    ),
    "calm": (
        "calm", "measured", "steady", "relaxed", "unhurried", "smooth",
        "спокойн", "уверен", "размерен", "невозмут", "плавн",
    ),
    "authoritative": (
        "authoritative", "commanding", "dominant", "firm", "decisive",
        "властн", "авторитетн", "решит", "твёрд", "тверд",
    ),
    "young": (
        "young", "youthful", "teenage", "boyish", "girlish",
        "юн", "молод", "подрост",
    ),
    "mature": (
        "mature", "older", "aged", "weathered", "gravelled",
        "зрел", "старш", "возрастн", "устал",
    ),
}

# Pairs that must not be mixed.
_AGE_CONFLICTS: Tuple[Tuple[str, str], ...] = (
    ("child", "adult"),
    ("child", "older"),
    ("child", "young adult"),
    ("teen", "older"),
    ("teen", "adult"),
    ("older", "child"),
    ("older", "young adult"),
)

_WHITESPACE_RE = re.compile(r"\s+")
_MAX_TIMBRE_CHARS = 60
_MAX_WORDS = 10


def normalize_age(value: Any) -> str:
    """Map a free-form age description onto the advisor vocabulary."""
    text = _WHITESPACE_RE.sub(" ", str(value or "").strip().lower())
    if not text:
        return ""
    if text in AGE_VOCABULARY:
        return text
    for alias, canonical in _AGE_ALIASES.items():
        if alias in text:
            return canonical
    return ""


def sanitize_timbre(value: Any, max_chars: int = _MAX_TIMBRE_CHARS) -> str:
    """Clean the timbre description so it can be used in a prompt."""
    text = _WHITESPACE_RE.sub(" ", str(value or "").strip())
    if not text:
        return ""
    text = re.sub(r"[<>\"'`\[\]{}()]", " ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip(" ,.;:-")
    if not text:
        return ""
    words = text.split()
    if len(words) > _MAX_WORDS:
        words = words[:_MAX_WORDS]
    text = " ".join(words)
    if len(text) > max_chars:
        text = text[:max_chars]
        cut = text.rfind(" ")
        if cut > 20:
            text = text[:cut]
    return text.strip(" ,.;:-")


@dataclass
class VoiceProfile:
    """The voice of one character, as described by the delivery analysis."""

    speaker: str
    timbre: str = ""
    age: str = ""
    energy: float = 0.0
    accent: str = ""
    gender: str = "unknown"

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "timbre": self.timbre,
            "age": self.age,
            "energy": round(float(self.energy or 0.0), 3),
            "accent": self.accent,
            "gender": self.gender,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["VoiceProfile"]:
        if not isinstance(data, dict):
            return None
        speaker = str(data.get("speaker") or "").strip()
        if not speaker:
            return None
        try:
            energy = float(data.get("energy") or 0.0)
        except (TypeError, ValueError):
            energy = 0.0
        return cls(
            speaker=speaker,
            timbre=sanitize_timbre(data.get("timbre")),
            age=normalize_age(data.get("age")),
            energy=max(0.0, min(1.0, energy)),
            accent=_WHITESPACE_RE.sub(" ", str(data.get("accent") or "").strip()),
            gender=str(data.get("gender") or "unknown").strip().lower() or "unknown",
        )

    # -- derived values ------------------------------------------------
    def timbre_keys(self) -> set:
        """The timbre dimensions this profile asks for."""
        blob = f"{self.timbre} {self.accent}".lower()
        keys = set()
        for key, keywords in TIMBRE_KEYWORDS.items():
            if any(word in blob for word in keywords):
                keys.add(key)
        return keys

    def prompt_fragment(self) -> str:
        """A short phrase for the TTS instruction (never spoken)."""
        parts = [self.timbre] if self.timbre else []
        if self.energy >= 0.75:
            parts.append("intense")
        elif self.energy and self.energy <= 0.2:
            parts.append("restrained")
        if not parts and self.age:
            parts.append(f"{self.age} voice")
        return ", ".join(part for part in parts if part)[:_MAX_TIMBRE_CHARS]


def build_voice_profiles(
    speaker_payloads: Iterable[Any],
    gender_map: Optional[Dict[str, str]] = None,
) -> Dict[str, VoiceProfile]:
    """Build the profiles of every speaker from the analysis payload.

    ``gender_map`` carries the gender that the pipeline already knows (the
    speaker metadata); it wins over anything the model might have guessed.
    """
    profiles: Dict[str, VoiceProfile] = {}
    for payload in speaker_payloads or []:
        profile = VoiceProfile.from_dict(payload)
        if profile is None:
            continue
        known_gender = (gender_map or {}).get(profile.speaker)
        if known_gender:
            profile.gender = str(known_gender).strip().lower() or profile.gender
        profiles[profile.speaker] = profile
    return profiles


def profile_for_speaker(
    profiles: Optional[Dict[str, Any]],
    speaker_id: Optional[str],
) -> Optional[VoiceProfile]:
    """Return the profile of one speaker from a profile mapping.

    The mapping may hold plain dicts (as persisted in the project config) or
    already built :class:`VoiceProfile` objects.
    """
    if not profiles or not speaker_id:
        return None
    raw = profiles.get(speaker_id)
    if raw is None:
        return None
    if isinstance(raw, VoiceProfile):
        return raw
    if isinstance(raw, dict):
        payload = dict(raw)
        payload.setdefault("speaker", speaker_id)
        return VoiceProfile.from_dict(payload)
    return None


def voice_note_for_speaker(
    profiles: Optional[Dict[str, Any]],
    speaker_id: Optional[str],
) -> Optional[str]:
    """The short voice advice of a speaker, for the TTS instruction."""
    profile = profile_for_speaker(profiles, speaker_id)
    if profile is None:
        return None
    return profile.prompt_fragment() or None


def voice_text_blob(voice_meta: Any) -> str:
    """Collect everything that describes a catalog voice into one text."""
    if isinstance(voice_meta, str):
        return voice_meta.lower()
    if not isinstance(voice_meta, dict):
        return ""
    parts: List[str] = []
    for key in ("name", "description", "timbre", "style", "tags", "title"):
        value = voice_meta.get(key)
        if value:
            parts.append(str(value))
    age = voice_meta.get("age")
    if age:
        parts.append(str(age))
    return " ".join(parts).lower()


def score_voice_against_profile(
    profile: Optional[VoiceProfile],
    voice_meta: Any,
) -> float:
    """How well a catalog voice matches the advised profile (0..1+)."""
    if profile is None:
        return 0.0
    wanted = profile.timbre_keys()
    if not wanted:
        return 0.0
    blob = voice_text_blob(voice_meta)
    if not blob:
        return 0.0

    hits = 0
    for key in wanted:
        if any(word in blob for word in TIMBRE_KEYWORDS[key]):
            hits += 1
    score = hits / float(len(wanted))

    voice_age = ""
    if isinstance(voice_meta, dict):
        voice_age = normalize_age(voice_meta.get("age"))
    if profile.age and voice_age:
        if profile.age == voice_age:
            # The age agrees: a small bonus and no penalty.
            score = score + 0.25
        elif (profile.age, voice_age) in _AGE_CONFLICTS:
            # A child voice for an old character (or the other way round) is
            # wrong even when the timbre matches.
            score = score * 0.2
    return max(0.0, min(1.5, score))


def rank_voices_by_profile(
    profile: Optional[VoiceProfile],
    voice_ids: Sequence[str],
    meta_for_voice,
    limit: Optional[int] = None,
    keep_voices: Optional[Sequence[str]] = None,
) -> List[str]:
    """Re-rank the candidate voices by how well they match the profile.

    The order of the input is preserved for voices with the same score, so a
    missing profile leaves the list untouched.  ``keep_voices`` are always
    kept (the gender aware default is the safety net).
    """
    voices = [voice for voice in (voice_ids or []) if voice]
    if profile is None or not voices:
        return voices
    if not profile.timbre_keys() and not profile.age:
        return voices

    scored = []
    for position, voice_id in enumerate(voices):
        try:
            meta = meta_for_voice(voice_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Voice metadata lookup failed for '%s': %s", voice_id, exc)
            meta = None
        scored.append((score_voice_against_profile(profile, meta), position, voice_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    ordered = [voice_id for _score, _position, voice_id in scored]

    if not limit or limit <= 0 or len(ordered) <= limit:
        return ordered

    kept = ordered[:limit]
    for voice_id in keep_voices or []:
        if voice_id and voice_id in voices and voice_id not in kept:
            kept.append(voice_id)
    return kept
