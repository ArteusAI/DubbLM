"""Settings management routes."""

import io
import os
from pathlib import Path
from typing import Dict, Any, Optional
import zipfile

from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..services.settings_service import (
    get_settings_masked,
    update_settings,
    get_api_key,
    set_api_key,
    get_defaults,
    mask_api_key,
    API_KEY_PROVIDERS,
)
from ..services.cookies_manager import (
    save_cookies_file,
    delete_cookies_file,
    has_cookies_file,
    get_effective_cookies_path,
)
from ..services.preset_service import get_frontend_preset_config, get_frontend_preset_configs

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


@router.get("/presets")
async def get_presets():
    """Get preset defaults resolved from dubbing_config.yml."""
    return get_frontend_preset_configs()


@router.get("/presets/{preset}")
async def get_preset(preset: str):
    """Get one preset defaults resolved from dubbing_config.yml."""
    return get_frontend_preset_config(preset)


@router.post("/cookies")
async def upload_cookies(file: UploadFile = File(...)):
    """Upload a cookies.txt file for video download authentication.

    The file is stored globally and used automatically by yt-dlp for URL downloads.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")

    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Cookies file must be a .txt file")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Cookies file is empty")

    save_cookies_file(content)

    return {
        "message": "Cookies file uploaded successfully",
        "filename": file.filename,
        "size": len(content),
    }


@router.get("/cookies")
async def get_cookies_status():
    """Check whether a cookies file is configured for video downloads."""
    cookies_path = get_effective_cookies_path()
    return {
        "configured": has_cookies_file(),
        "path": str(cookies_path) if cookies_path else None,
        "source": "uploaded" if cookies_path and cookies_path.name == "cookies.txt" and "cookies" in str(cookies_path) else "env",
    }


@router.delete("/cookies")
async def remove_cookies():
    """Delete the globally uploaded cookies file."""
    deleted = delete_cookies_file()
    return {
        "message": "Cookies file deleted" if deleted else "No uploaded cookies file to delete",
        "deleted": deleted,
    }


# Define the default extension directory relative to this file
EXTENSION_DIR = Path(__file__).resolve().parents[3] / "extension"


@router.get("/extension/download")
async def download_extension(request: Request):
    """Download the Chrome extension packaged as a ZIP, with the correct API URL and Bearer Token pre-configured."""
    if not EXTENSION_DIR.exists() or not EXTENSION_DIR.is_dir():
        raise HTTPException(status_code=404, detail="Extension source folder not found")

    # Determine the external API URL. If referer is present, parse it to preserve the original port
    referer = request.headers.get("referer")
    if referer:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            if parsed.scheme and parsed.netloc:
                external_api_url = f"{parsed.scheme}://{parsed.netloc}"
            else:
                external_api_url = str(request.base_url).rstrip("/")
        except Exception:
            external_api_url = str(request.base_url).rstrip("/")
    else:
        external_api_url = str(request.base_url).rstrip("/")

    api_token = os.getenv("DUBBLM_API_TOKEN") or os.getenv("API_TOKEN") or ""

    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in EXTENSION_DIR.glob("**/*"):
            if file_path.is_file():
                content = file_path.read_bytes()
                relative_path = file_path.relative_to(EXTENSION_DIR)
                
                # Perform dynamic injection for config.js
                if file_path.name == "config.js":
                    try:
                        content_str = content.decode("utf-8")
                        # Inject the actual base URL
                        content_str = content_str.replace(
                            'serverUrl: "http://localhost:8000"', 
                            f'serverUrl: "{external_api_url}"'
                        )
                        # Inject the actual Bearer token
                        content_str = content_str.replace(
                            'apiToken: ""',
                            f'apiToken: "{api_token}"'
                        )
                        content = content_str.encode("utf-8")
                    except Exception:
                        pass
                
                zip_file.writestr(str(relative_path), content)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=dubblm-extension.zip"}
    )
