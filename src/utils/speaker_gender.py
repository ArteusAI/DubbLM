"""Utilities for speaker gender metadata and translation signatures."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Optional


VALID_SPEAKER_GENDERS = {"male", "female", "unknown"}
DEFAULT_SPEAKER_GENDER = "unknown"
DEFAULT_GENDER_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_SPEAKER_GENDER_MODEL_ID = "audeering/wav2vec2-large-robust-24-ft-age-gender"
SPEAKER_METADATA_CONFIG_KEY = "speakerMetadata"
SPEAKER_GENDER_SIGNATURE_CONFIG_KEY = "speakerGenderTranslationSignature"


def normalize_gender_value(value: Any, default: str = DEFAULT_SPEAKER_GENDER) -> str:
    """Normalize a gender-like value to one of the supported string labels."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_SPEAKER_GENDERS:
            return normalized
    return default


def normalize_speaker_metadata_map(value: Any) -> Dict[str, Dict[str, Any]]:
    """Return a normalized speaker metadata map from arbitrary config data."""
    if not isinstance(value, Mapping):
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for raw_speaker, raw_metadata in value.items():
        speaker = str(raw_speaker).strip()
        if not speaker:
            continue

        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
        inferred_gender = normalize_gender_value(metadata.get("inferredGender"))
        override_value = metadata.get("overrideGender")
        override_gender = (
            normalize_gender_value(override_value)
            if override_value is not None
            else None
        )

        normalized[speaker] = {
            "inferredGender": inferred_gender,
            "inferredConfidence": _normalize_confidence(metadata.get("inferredConfidence")),
            "rawLabel": str(metadata.get("rawLabel") or "").strip(),
            "modelId": str(metadata.get("modelId") or DEFAULT_SPEAKER_GENDER_MODEL_ID).strip(),
            "overrideGender": override_gender,
        }

    return normalized


def build_effective_speaker_gender_map(
    metadata_map: Optional[Mapping[str, Mapping[str, Any]]],
    speakers: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    """Resolve the effective gender for each speaker."""
    normalized = normalize_speaker_metadata_map(metadata_map)
    if speakers is None:
        keys = normalized.keys()
    else:
        keys = [str(s).strip() for s in speakers if str(s).strip()]

    effective: Dict[str, str] = {}
    for speaker in keys:
        metadata = normalized.get(speaker) or {}
        override_gender = metadata.get("overrideGender")
        inferred_gender = metadata.get("inferredGender")
        effective[speaker] = normalize_gender_value(
            override_gender or inferred_gender or DEFAULT_SPEAKER_GENDER
        )
    return effective


def build_translation_gender_signature(
    segments: Iterable[Any],
    metadata_map: Optional[Mapping[str, Mapping[str, Any]]],
) -> str:
    """Build a stable signature for gender-dependent translation inputs."""
    items = []
    normalized = normalize_speaker_metadata_map(metadata_map)

    for segment in segments:
        speaker = _extract_speaker(segment)
        if not speaker:
            continue
        effective_gender = build_effective_speaker_gender_map(normalized, [speaker]).get(
            speaker,
            DEFAULT_SPEAKER_GENDER,
        )
        items.append({"speaker": speaker, "gender": effective_gender})

    payload = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def build_effective_gender_prompt_section(
    metadata_map: Optional[Mapping[str, Mapping[str, Any]]],
    speakers: Optional[Iterable[str]] = None,
) -> str:
    """Format effective speaker genders for prompt injection."""
    effective = build_effective_speaker_gender_map(metadata_map, speakers=speakers)
    if not effective:
        return ""

    lines = [
        "# Speaker grammar metadata:",
        "Use this ONLY for grammatical agreement and self-referential forms.",
        "Do not invent biography, social identity, or third-person facts from it.",
    ]
    for speaker, gender in sorted(effective.items()):
        lines.append(f"- {speaker} -> {gender}")
    return "\n".join(lines)


def is_speaker_gender_translation_stale(
    config: Mapping[str, Any],
    segments: Optional[Iterable[Any]] = None,
) -> bool:
    """Return True when the stored translation signature no longer matches config/segments."""
    if not normalize_bool(config.get("enableSpeakerGenderInference")):
        return False

    stored_signature = str(config.get(SPEAKER_GENDER_SIGNATURE_CONFIG_KEY) or "").strip()
    if not stored_signature:
        return False

    if not segments:
        return False

    current_signature = build_translation_gender_signature(
        segments,
        config.get(SPEAKER_METADATA_CONFIG_KEY),
    )
    return bool(current_signature and current_signature != stored_signature)


def rename_speaker_keyed_map(value: Any, old_name: str, new_name: str) -> Any:
    """Rename a key in a speaker->value mapping, preserving other entries."""
    if not isinstance(value, Mapping):
        return value

    old_key = str(old_name).strip()
    new_key = str(new_name).strip()
    if not old_key or not new_key or old_key == new_key:
        return dict(value)

    updated = dict(value)
    if old_key in updated:
        updated[new_key] = updated.pop(old_key)
    return updated


def apply_speaker_gender_override(
    metadata_map: Optional[Mapping[str, Mapping[str, Any]]],
    speaker_name: str,
    override_gender: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """Set or clear a speaker override in a metadata map."""
    normalized = normalize_speaker_metadata_map(metadata_map)
    speaker = str(speaker_name).strip()
    if not speaker:
        return normalized

    entry = dict(normalized.get(speaker) or {
        "inferredGender": DEFAULT_SPEAKER_GENDER,
        "inferredConfidence": 0.0,
        "rawLabel": "",
        "modelId": DEFAULT_SPEAKER_GENDER_MODEL_ID,
        "overrideGender": None,
    })

    if override_gender is None:
        entry["overrideGender"] = None
    else:
        entry["overrideGender"] = normalize_gender_value(override_gender)

    normalized[speaker] = entry
    return normalized


def normalize_bool(value: Any) -> bool:
    """Normalize a truthy config value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _extract_speaker(segment: Any) -> str:
    """Extract speaker name from either dict-like or ORM-like segments."""
    if isinstance(segment, Mapping):
        return str(segment.get("speaker") or "").strip()
    return str(getattr(segment, "speaker", "") or "").strip()


def _normalize_confidence(value: Any) -> float:
    """Clamp arbitrary confidence to [0, 1]."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))
