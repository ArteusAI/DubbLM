"""Publishing routes (ИИ Борщ integration)."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from ..config import get_settings
from ..services.borsch_publisher import (
    BorschAuthError,
    BorschPublishError,
    get_publish_state,
    publish_project_to_borsch,
)

router = APIRouter(prefix="/projects", tags=["publish"])


def _token_header(
    x_borsch_admin_token: str | None = Header(default=None, alias="X-Borsch-Admin-Token"),
) -> str | None:
    token = (x_borsch_admin_token or "").strip()
    return token or None


@router.get("/{project_id}/publish/borsch")
async def get_borsch_publish_state(
    project_id: str,
    x_borsch_admin_token: str | None = Header(default=None, alias="X-Borsch-Admin-Token"),
):
    """Return whether the project was published and refresh processing status."""
    settings = get_settings()
    if not settings.borsch_publish_enabled:
        return {"enabled": False, "published": False}
    try:
        return await run_in_threadpool(
            get_publish_state,
            project_id,
            _token_header(x_borsch_admin_token),
        )
    except BorschPublishError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/{project_id}/publish/borsch")
async def publish_borsch(
    project_id: str,
    force: bool = Query(False),
    x_borsch_admin_token: str | None = Header(default=None, alias="X-Borsch-Admin-Token"),
):
    """Generate metadata with the LLM and publish the dubbed video to ИИ Борщ."""
    settings = get_settings()
    if not settings.borsch_publish_enabled:
        raise HTTPException(status_code=404, detail="ИИ Борщ publishing is disabled")

    token = _token_header(x_borsch_admin_token)
    if not token:
        raise HTTPException(status_code=401, detail="borsch_admin_token_required")

    if not force:
        try:
            existing = await run_in_threadpool(get_publish_state, project_id, token)
            if existing.get("published"):
                return existing
        except BorschPublishError as exc:
            if exc.status_code != 404:
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    try:
        return await run_in_threadpool(publish_project_to_borsch, project_id, token)
    except BorschAuthError as exc:
        raise HTTPException(status_code=401, detail="borsch_invalid_token") from exc
    except BorschPublishError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
