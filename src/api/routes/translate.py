"""External one-shot translate API."""

from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.translation.prompts import get_available_personas

from ..database.session import get_db
from ..models.schemas import ExternalTranslateResponse
from ..services.external_translate_service import start_external_translate

router = APIRouter(tags=["external"])

PresetType = Literal["fast", "hq", "ultra"]


@router.post("/translate", response_model=ExternalTranslateResponse, status_code=202)
async def external_translate(
    file: UploadFile = File(...),
    targetLang: str = Form(...),
    preset: PresetType = Form("hq"),
    sourceLang: str = Form("auto"),
    minimalDiarizationMerge: bool = Form(False),
    keepBackground: bool = Form(False),
    enableLlmEditor: bool = Form(False),
    speakerCount: int = Form(1),
    personaId: Optional[str] = Form(None),
    name: str | None = Form(None),
    db: Session = Depends(get_db),
) -> ExternalTranslateResponse:
    """Upload a video and start full translation + dubbing in one request."""
    if speakerCount < 1:
        raise HTTPException(status_code=400, detail="speakerCount must be at least 1")

    if personaId is not None:
        persona_id = personaId.strip()
        if not persona_id:
            raise HTTPException(status_code=400, detail="personaId must not be empty")
        valid_persona_ids = {persona.id for persona in get_available_personas()}
        if persona_id not in valid_persona_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown personaId '{persona_id}'. Use GET /resources/personas for available values.",
            )
    else:
        persona_id = None

    return await start_external_translate(
        db,
        file,
        targetLang,
        preset=preset,
        source_lang=sourceLang,
        minimal_diarization_merge=minimalDiarizationMerge,
        keep_background=keepBackground,
        enable_llm_editor=enableLlmEditor,
        speaker_count=speakerCount,
        persona_id=persona_id,
        name=name,
    )
