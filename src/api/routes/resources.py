"""Resource routes for voices and personas."""

from typing import List
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..models.schemas import VoiceResponse, PersonaResponse
from src.translation.prompts import get_available_personas

router = APIRouter(prefix="/resources", tags=["resources"])

# Predefined voices
VOICES = [
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
    
    # Gemini voices
    {"id": "Zephyr", "name": "Zephyr", "provider": "gemini", "gender": "neutral"},
    {"id": "Puck", "name": "Puck", "provider": "gemini", "gender": "male"},
    {"id": "Charon", "name": "Charon", "provider": "gemini", "gender": "male"},
    {"id": "Kore", "name": "Kore", "provider": "gemini", "gender": "female"},
    {"id": "Fenrir", "name": "Fenrir", "provider": "gemini", "gender": "male"},
    {"id": "Leda", "name": "Leda", "provider": "gemini", "gender": "female"},
    {"id": "Orus", "name": "Orus", "provider": "gemini", "gender": "male"},
    {"id": "Aoede", "name": "Aoede", "provider": "gemini", "gender": "female"},
    {"id": "Callirrhoe", "name": "Callirrhoe", "provider": "gemini", "gender": "female"},
    {"id": "Autonoe", "name": "Autonoe", "provider": "gemini", "gender": "female"},
    {"id": "Enceladus", "name": "Enceladus", "provider": "gemini", "gender": "male"},
    {"id": "Iapetus", "name": "Iapetus", "provider": "gemini", "gender": "male"},
    {"id": "Umbriel", "name": "Umbriel", "provider": "gemini", "gender": "neutral"},
    {"id": "Algieba", "name": "Algieba", "provider": "gemini", "gender": "male"},
    {"id": "Despina", "name": "Despina", "provider": "gemini", "gender": "female"},
    {"id": "Erinome", "name": "Erinome", "provider": "gemini", "gender": "female"},
    {"id": "Gacrux", "name": "Gacrux", "provider": "gemini", "gender": "male"},
    {"id": "Pulcherrima", "name": "Pulcherrima", "provider": "gemini", "gender": "female"},
    {"id": "Achird", "name": "Achird", "provider": "gemini", "gender": "male"},
    {"id": "Zubenelgenubi", "name": "Zubenelgenubi", "provider": "gemini", "gender": "male"},
    {"id": "Vindemiatrix", "name": "Vindemiatrix", "provider": "gemini", "gender": "female"},
    {"id": "Sadachbia", "name": "Sadachbia", "provider": "gemini", "gender": "male"},
    {"id": "Sadaltager", "name": "Sadaltager", "provider": "gemini", "gender": "male"},
    {"id": "Sulafat", "name": "Sulafat", "provider": "gemini", "gender": "female"},
    {"id": "Laomedeia", "name": "Laomedeia", "provider": "gemini", "gender": "female"},
    {"id": "Achernar", "name": "Achernar", "provider": "gemini", "gender": "male"},
    {"id": "Alnilam", "name": "Alnilam", "provider": "gemini", "gender": "male"},
    {"id": "Schedar", "name": "Schedar", "provider": "gemini", "gender": "female"},
    {"id": "Rasalgethi", "name": "Rasalgethi", "provider": "gemini", "gender": "male"},
    {"id": "Algenib", "name": "Algenib", "provider": "gemini", "gender": "male"},
    
    # MiniMax voices
    {"id": "male-qn-qingse", "name": "Qingse (Male)", "provider": "minimax", "gender": "male"},
    {"id": "female-shaonv", "name": "Shaonv (Female)", "provider": "minimax", "gender": "female"},
    {"id": "male-qn-jingying", "name": "Jingying (Male)", "provider": "minimax", "gender": "male"},
    {"id": "female-yujie", "name": "Yujie (Female)", "provider": "minimax", "gender": "female"},
]

@router.get("/voices", response_model=List[VoiceResponse])
async def list_voices(provider: str = None):
    """Get list of available voices."""
    voices = VOICES
    if provider:
        voices = [v for v in voices if v["provider"] == provider]
    return [VoiceResponse(**v) for v in voices]


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

