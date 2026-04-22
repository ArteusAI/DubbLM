"""Resource routes for voices and personas."""

from typing import List
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..models.schemas import VoiceResponse, PersonaResponse
from src.tts.gemini_voice_catalog import build_gemini_voice_entries
from src.translation.prompts import get_available_personas

router = APIRouter(prefix="/resources", tags=["resources"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VOICE_SAMPLE_ROOTS = [
    PROJECT_ROOT / "src" / "tts" / "samples",
    PROJECT_ROOT / "tts" / "samples",
]
AUDIO_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}

# Predefined voices
OPENAI_VOICES = [
    # OpenAI voices
    {"id": "alloy", "name": "Alloy", "provider": "openai", "gender": "neutral"},
    {"id": "echo", "name": "Echo", "provider": "openai", "gender": "male"},
    {"id": "fable", "name": "Fable", "provider": "openai", "gender": "neutral"},
    {"id": "onyx", "name": "Onyx", "provider": "openai", "gender": "male"},
    {"id": "nova", "name": "Nova", "provider": "openai", "gender": "female"},
    {"id": "shimmer", "name": "Shimmer", "provider": "openai", "gender": "female"},
    {"id": "ash", "name": "Ash", "provider": "openai", "gender": "male"},
    {"id": "coral", "name": "Coral", "provider": "openai", "gender": "female"},
    {"id": "sage", "name": "Sage", "provider": "openai", "gender": "neutral"},
]

MINIMAX_VOICES = [
    # MiniMax voices
    {"id": "male-qn-qingse", "name": "Qingse (Male)", "provider": "minimax", "gender": "male"},
    {"id": "female-shaonv", "name": "Shaonv (Female)", "provider": "minimax", "gender": "female"},
    {"id": "male-qn-jingying", "name": "Jingying (Male)", "provider": "minimax", "gender": "male"},
    {"id": "female-yujie", "name": "Yujie (Female)", "provider": "minimax", "gender": "female"},
]

VOICES = OPENAI_VOICES + build_gemini_voice_entries() + MINIMAX_VOICES


def _find_voice_sample_path(provider: str, voice_id: str) -> Path | None:
    provider_name = (provider or "").strip().lower()
    lookup_voice = (voice_id or "").strip()
    if not provider_name or not lookup_voice:
        return None

    for root in VOICE_SAMPLE_ROOTS:
        provider_dir = root / provider_name
        if not provider_dir.exists():
            continue

        # Fast path: direct voice_id.ext in provider root.
        for extension in AUDIO_MEDIA_TYPES:
            direct_path = provider_dir / f"{lookup_voice}{extension}"
            if direct_path.exists():
                return direct_path

        # Fallback: nested directories (e.g. minimax/en, minimax/ru).
        for sample_path in provider_dir.rglob("*"):
            if not sample_path.is_file():
                continue
            if sample_path.suffix.lower() not in AUDIO_MEDIA_TYPES:
                continue
            if sample_path.stem == lookup_voice:
                return sample_path

    return None


def _build_preview_url(provider: str, voice_id: str) -> str:
    return f"/api/v1/resources/voices/{quote(provider, safe='')}/{quote(voice_id, safe='')}/sample"


@router.get("/voices", response_model=List[VoiceResponse])
async def list_voices(provider: str = None):
    """Get list of available voices."""
    voices = VOICES
    if provider:
        voices = [v for v in voices if v["provider"] == provider]

    response_payload: List[VoiceResponse] = []
    for voice in voices:
        data = dict(voice)
        sample_path = _find_voice_sample_path(data["provider"], data["id"])
        if sample_path:
            data["preview_url"] = _build_preview_url(data["provider"], data["id"])
        response_payload.append(VoiceResponse(**data))

    return response_payload


@router.get("/voices/{provider}/{voice_id:path}/sample")
async def get_voice_sample(provider: str, voice_id: str):
    """Stream generated voice sample for a provider voice."""
    sample_path = _find_voice_sample_path(provider, voice_id)
    if not sample_path:
        raise HTTPException(status_code=404, detail="Voice sample not found")

    media_type = AUDIO_MEDIA_TYPES.get(sample_path.suffix.lower(), "application/octet-stream")
    return FileResponse(path=sample_path, media_type=media_type, filename=sample_path.name)


@router.get("/personas", response_model=List[PersonaResponse])
async def list_personas():
    """Get list of available personas loaded from persona files."""
    personas = get_available_personas()
    # Sort so 'none' appears first, then alphabetically by name
    personas_sorted = sorted(personas, key=lambda p: (p.id != 'none', p.name))
    return [
        PersonaResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            promptTemplate=None,
        )
        for p in personas_sorted
    ]


@router.get("/languages")
async def list_languages():
    """Get list of supported languages."""
    return [
        {"code": "en", "name": "English"},
        {"code": "ru", "name": "Russian"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "it", "name": "Italian"},
        {"code": "pt", "name": "Portuguese"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ja", "name": "Japanese"},
        {"code": "ko", "name": "Korean"},
        {"code": "ar", "name": "Arabic"},
        {"code": "hi", "name": "Hindi"},
        {"code": "tr", "name": "Turkish"},
        {"code": "pl", "name": "Polish"},
        {"code": "nl", "name": "Dutch"},
        {"code": "sv", "name": "Swedish"},
        {"code": "da", "name": "Danish"},
        {"code": "no", "name": "Norwegian"},
        {"code": "fi", "name": "Finnish"},
        {"code": "cs", "name": "Czech"},
        {"code": "uk", "name": "Ukrainian"},
    ]
