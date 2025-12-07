"""Settings management routes."""

from typing import Dict, Any, Optional
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException

from ..services.settings_service import (
    get_settings_masked,
    update_settings,
    get_api_key,
    set_api_key,
    get_defaults,
    mask_api_key,
    API_KEY_PROVIDERS,
)

router = APIRouter(prefix="/settings", tags=["settings"])


class SettingsUpdate(BaseModel):
    """Schema for settings update."""
    apiKeys: Optional[Dict[str, str]] = None
    defaults: Optional[Dict[str, Any]] = None


class ApiKeyUpdate(BaseModel):
    """Schema for API key update."""
    key: str


@router.get("")
async def get_system_settings():
    """Get current system settings (API keys are masked)."""
    return get_settings_masked()


@router.patch("")
async def update_system_settings(data: SettingsUpdate):
    """Update system settings."""
    updates = data.model_dump(exclude_unset=True)
    updated = update_settings(updates)
    
    # Return masked version
    if "apiKeys" in updated:
        updated["apiKeys"] = {
            provider: mask_api_key(key) 
            for provider, key in updated["apiKeys"].items()
        }
    
    return updated


@router.get("/api-keys")
async def list_api_keys():
    """List configured API key providers and their status."""
    return {
        provider: {
            "configured": bool(get_api_key(provider)),
            "masked": mask_api_key(get_api_key(provider)),
        }
        for provider in API_KEY_PROVIDERS
    }


@router.put("/api-keys/{provider}")
async def update_api_key(provider: str, data: ApiKeyUpdate):
    """Set API key for a specific provider."""
    if provider not in API_KEY_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider. Must be one of: {', '.join(API_KEY_PROVIDERS.keys())}"
        )
    
    set_api_key(provider, data.key)
    
    return {
        "provider": provider,
        "configured": True,
        "masked": mask_api_key(data.key),
    }


@router.delete("/api-keys/{provider}")
async def delete_api_key(provider: str):
    """Remove API key for a provider."""
    set_api_key(provider, "")
    return {"provider": provider, "configured": False}


@router.get("/defaults")
async def get_default_settings():
    """Get default project settings."""
    return get_defaults()


@router.patch("/defaults")
async def update_default_settings(data: Dict[str, Any]):
    """Update default project settings."""
    updated = update_settings({"defaults": data})
    return updated.get("defaults", {})

