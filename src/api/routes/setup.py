"""First-run setup endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.settings_service import (
    has_openrouter_key,
    mask_api_key,
    save_openrouter_key,
)
from src.utils.openrouter_client import OpenRouterError, validate_openrouter_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/setup", tags=["setup"])


class OpenRouterKeyPayload(BaseModel):
    key: str


@router.get("/status")
async def setup_status() -> Dict[str, Any]:
    """Return whether the first-run OpenRouter key is configured.

    The key itself is never returned.  A configured key disables the setup
    wizard; the regular settings UI remains available for rotation.
    """
    configured = has_openrouter_key()
    return {
        "configured": configured,
        "required": not configured,
        "provider": "openrouter",
    }


@router.post("/openrouter-key")
async def configure_openrouter_key(payload: OpenRouterKeyPayload) -> Dict[str, Any]:
    """Validate and persist the OpenRouter key during first-run setup."""
    if has_openrouter_key():
        raise HTTPException(
            status_code=403,
            detail="OpenRouter is already configured. Use Settings → API Keys to rotate it.",
        )

    key = str(payload.key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="OpenRouter key is empty")
    if not key.startswith("sk-or-"):
        raise HTTPException(
            status_code=400,
            detail="That does not look like an OpenRouter key (expected sk-or-...).",
        )

    try:
        validate_openrouter_key(key)
    except OpenRouterError as exc:
        status = exc.status_code or 0
        if status in (401, 403):
            raise HTTPException(status_code=400, detail="OpenRouter rejected this key.") from exc
        logger.warning("OpenRouter key validation failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Could not validate the key with OpenRouter. Check the network and try again.",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("OpenRouter key validation crashed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not validate the key with OpenRouter.") from exc

    try:
        result = save_openrouter_key(key)
    except Exception as exc:
        logger.exception("Could not persist the OpenRouter key")
        raise HTTPException(status_code=500, detail="Could not save the OpenRouter key.") from exc

    logger.info("OpenRouter key configured through first-run setup (%s)", mask_api_key(key))
    return {
        "configured": True,
        "provider": "openrouter",
        "masked": result.get("masked") or mask_api_key(key),
        "source": result.get("source"),
    }
