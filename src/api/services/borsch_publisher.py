"""Publish a dubbed project to the ИИ Борщ video library.

The feature is intentionally opt-in: ``BORSCH_PUBLISH_ENABLED`` must be true.
The admin token is supplied by the browser on the first publish and is never
persisted on the server.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from sqlalchemy.orm.attributes import flag_modified

from ..config import get_settings
from ..database.models import Project, ProjectStatus, Segment
from ..database.session import get_db_session
from ..services.object_storage import get_object_storage
from ..services.project_manager import ProjectManager
from ..services.settings_service import get_api_key

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "talk",
    "lecture",
    "explainer",
    "tutorial",
    "documentary",
    "interview",
    "demo",
    "commentary",
    "comedy",
    "music",
    "course",
}
ALLOWED_LEVELS = {"intro", "intermediate", "advanced"}
ALLOWED_SHELF_LIFE = {"evergreen", "news", "historical"}
_TAXONOMY_TTL_SECONDS = 300
_taxonomy_cache: dict[str, Any] = {"at": 0.0, "areas": []}


class BorschPublishError(Exception):
    """Base error for the publish flow."""

    def __init__(self, message: str, *, status_code: int = 502, code: str = "borsch_publish_failed"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class BorschAuthError(BorschPublishError):
    def __init__(self, message: str = "Invalid ИИ Борщ admin token"):
        super().__init__(message, status_code=401, code="borsch_invalid_token")


class BorschMetadataError(BorschPublishError):
    def __init__(self, message: str):
        super().__init__(message, status_code=502, code="borsch_metadata_failed")


def _api_url(path: str) -> str:
    settings = get_settings()
    return f"{settings.borsch_api_url}{path}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request_error(response: requests.Response, fallback: str) -> str:
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if detail:
            return str(detail)
    except Exception:
        pass
    return f"{fallback} (HTTP {response.status_code})"


def validate_borsch_token(admin_token: str) -> None:
    if not admin_token:
        raise BorschAuthError("ИИ Борщ admin token is required")
    try:
        response = requests.post(
            _api_url("/api/admin/login"),
            json={"token": admin_token},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise BorschPublishError(f"ИИ Борщ API is unreachable: {exc}") from exc
    if response.status_code == 401:
        raise BorschAuthError()
    if not response.ok:
        raise BorschPublishError(_request_error(response, "ИИ Борщ token validation failed"))


def _taxonomy() -> List[dict[str, Any]]:
    now = time.time()
    if now - float(_taxonomy_cache.get("at") or 0) < _TAXONOMY_TTL_SECONDS:
        return _taxonomy_cache.get("areas") or []
    try:
        response = requests.get(_api_url("/api/browse"), timeout=15)
        response.raise_for_status()
        areas = response.json()
        if not isinstance(areas, list):
            areas = []
    except Exception as exc:
        logger.warning("Could not load ИИ Борщ taxonomy: %s", exc)
        areas = []
    _taxonomy_cache["at"] = now
    _taxonomy_cache["areas"] = areas
    return areas


def _flatten_categories(areas: List[dict[str, Any]]) -> List[dict[str, Any]]:
    out: List[dict[str, Any]] = []
    for area in areas:
        for category in area.get("categories") or []:
            cid = category.get("id")
            if cid:
                out.append(
                    {
                        "id": str(cid),
                        "name": str(category.get("name") or cid),
                        "area": str(area.get("name") or area.get("id") or ""),
                    }
                )
    return out


def _sample_transcript(segments: Iterable[Segment], target_lang: str, max_chars: int = 32000) -> str:
    rows: list[tuple[float, str, str]] = []
    for segment in segments:
        text = (segment.translated_text or segment.original_text or "").strip()
        if not text:
            continue
        rows.append((float(segment.start_time or 0), str(segment.speaker or "speaker"), text))
    if not rows:
        return ""
    rows.sort(key=lambda row: row[0])

    # Keep the first/last part and an even sample in the middle for long videos.
    if len(rows) > 120:
        head = rows[:12]
        tail = rows[-12:]
        middle = rows[12:-12]
        step = max(1, len(middle) // 36)
        sampled = head + middle[::step][:36] + tail
    else:
        sampled = rows

    lines: list[str] = []
    total = 0
    for start, speaker, text in sampled:
        minutes = int(start // 60)
        seconds = int(start % 60)
        line = f"[{minutes:02d}:{seconds:02d}] {speaker}: {text}"
        if total + len(line) > max_chars and lines:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _report_summary(project_id: str, project: Project) -> Dict[str, Any]:
    pm = ProjectManager(project_id)
    json_path = pm.artifacts_dir / "report.json"
    if not json_path.exists():
        try:
            store = get_object_storage()
            restored = store.ensure_local_report(project_id, project, kind="json")
            if restored and restored.exists():
                json_path = restored
        except Exception as exc:
            logger.warning("Could not restore report for %s: %s", project_id, exc)
    if not json_path.exists():
        return {}
    try:
        report = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not parse report for %s: %s", project_id, exc)
        return {}
    return {
        "project": report.get("project") or {},
        "totals": report.get("totals") or {},
        "stages": report.get("stages") or [],
        "artifacts": [
            {"path": item.get("path"), "size_bytes": item.get("size_bytes")}
            for item in (report.get("artifacts") or [])[:40]
            if isinstance(item, dict)
        ],
    }


def _metadata_language(project: Project) -> str:
    settings = get_settings()
    configured = (settings.borsch_metadata_language or "auto").strip().lower()
    if configured and configured != "auto":
        return configured
    target = (project.config or {}).get("targetLang") or "ru"
    return str(target).strip().lower() or "ru"


def _language_label(code: str) -> str:
    labels = {
        "ru": "русский",
        "en": "английский",
        "de": "немецкий",
        "fr": "французский",
        "es": "испанский",
        "it": "итальянский",
        "pt": "португальский",
        "uk": "украинский",
        "tr": "турецкий",
        "zh": "китайский",
        "ja": "японский",
        "ko": "корейский",
    }
    return labels.get(code, code)


def _source_url(project: Project) -> Optional[str]:
    config = project.config or {}
    for key in ("downloadUrl", "sourceUrl", "url"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _llm_chat(messages: List[dict[str, str]]) -> Dict[str, Any]:
    settings = get_settings()
    provider = (settings.borsch_llm_provider or "openrouter").strip().lower()
    if provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        key = get_api_key("openai")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
    elif provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        key = get_api_key("openrouter")
        headers = {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": settings.borsch_public_url,
            "X-Title": "DubbLM",
        } if key else {}
    else:
        raise BorschMetadataError(f"Unsupported BORSCH_LLM_PROVIDER: {provider}")

    if not key:
        raise BorschMetadataError(f"No API key configured for LLM provider '{provider}'")

    payload: dict[str, Any] = {
        "model": settings.borsch_llm_model,
        "messages": messages,
        "temperature": settings.borsch_llm_temperature,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=settings.borsch_llm_timeout)
        if response.status_code == 400 and "response_format" in response.text.lower():
            payload.pop("response_format", None)
            response = requests.post(url, json=payload, headers=headers, timeout=settings.borsch_llm_timeout)
    except requests.RequestException as exc:
        raise BorschMetadataError(f"LLM request failed: {exc}") from exc

    if response.status_code in (401, 403):
        raise BorschMetadataError("LLM authentication failed")
    if not response.ok:
        raise BorschMetadataError(
            f"LLM returned HTTP {response.status_code}: {response.text[:500]}"
        )

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
    except Exception as exc:
        raise BorschMetadataError(f"Unexpected LLM response: {exc}") from exc

    if not isinstance(content, str) or not content.strip():
        raise BorschMetadataError("LLM returned an empty response")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        try:
            import json_repair  # type: ignore

            parsed = json_repair.loads(content)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise BorschMetadataError("LLM did not return a JSON object")
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise BorschMetadataError(f"LLM JSON is invalid: {exc}") from exc

    if not isinstance(parsed, dict):
        raise BorschMetadataError("LLM JSON is not an object")
    return parsed


def _as_str(value: Any, fallback: str = "", limit: int = 4000) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    return text[:limit]


def _as_tags(value: Any) -> List[str]:
    if isinstance(value, str):
        values = re.split(r"[,;]+", value)
    elif isinstance(value, list):
        values = value
    else:
        values = []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        tag = re.sub(r"\s+", " ", str(item)).strip().strip("#")
        tag = tag[:48]
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            out.append(tag)
    return out[:12]


def _normalize_metadata(
    raw: Dict[str, Any],
    project: Project,
    categories: List[dict[str, Any]],
    language: str,
    transcript: str,
) -> Dict[str, Any]:
    allowed_ids = {c["id"] for c in categories}
    name_to_id = {c["name"].strip().lower(): c["id"] for c in categories}
    raw_categories = raw.get("category_ids") or raw.get("categories") or []
    if isinstance(raw_categories, str):
        raw_categories = [raw_categories]
    category_ids: list[str] = []
    for item in raw_categories if isinstance(raw_categories, list) else []:
        value = str(item).strip()
        cid = value if value in allowed_ids else name_to_id.get(value.lower())
        if cid and cid not in category_ids:
            category_ids.append(cid)
    category_ids = category_ids[:4]

    title = _as_str(raw.get("title"), project.name or "Без названия", 300)
    summary = _as_str(raw.get("summary"), "", 3000)
    why_hn = _as_str(raw.get("why_hn"), "", 2000)
    if not summary and transcript:
        summary = transcript[:600].strip()
    if not why_hn:
        why_hn = "Видео переведено и озвучено с помощью DubbLM."

    content_type = _as_str(raw.get("content_type"), "talk", 32).lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        content_type = "talk"
    level = _as_str(raw.get("level"), "intermediate", 32).lower()
    if level not in ALLOWED_LEVELS:
        level = "intermediate"
    shelf_life = _as_str(raw.get("shelf_life"), "evergreen", 32).lower()
    if shelf_life not in ALLOWED_SHELF_LIFE:
        shelf_life = "evergreen"

    tags = _as_tags(raw.get("tags"))
    primary_topic = _as_str(raw.get("primary_topic"), tags[0] if tags else "", 160)
    reason = _as_str(raw.get("reason"), "Опубликовано из DubbLM", 300)

    return {
        "title": title,
        "channel": get_settings().borsch_channel,
        "summary": summary,
        "why_hn": why_hn,
        "content_type": content_type,
        "level": level,
        "shelf_life": shelf_life,
        "primary_topic": primary_topic,
        "tags": tags,
        "category_ids": category_ids,
        "reason": reason,
        "license": "Права на исходное видео принадлежат автору; дубляж выполнен DubbLM.",
    }


def generate_metadata(project_id: str, admin_token: str) -> tuple[Project, Dict[str, Any], Optional[str]]:
    """Build LLM metadata for a dubbed project. Returns (project, metadata, source_url)."""
    settings = get_settings()
    db = get_db_session(fresh=True)
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise BorschPublishError("Project not found", status_code=404, code="project_not_found")
        if project.status != ProjectStatus.DUBBED:
            raise BorschPublishError(
                "Video is not dubbed yet",
                status_code=400,
                code="project_not_dubbed",
            )
        segments = (
            db.query(Segment)
            .filter(Segment.project_id == project_id)
            .order_by(Segment.sequence.asc())
            .all()
        )
        config = dict(project.config or {})
        target_lang = str(config.get("targetLang") or "ru")
        language = _metadata_language(project)
        source_url = _source_url(project)
        transcript = _sample_transcript(segments, target_lang)
        report = _report_summary(project_id, project)
        areas = _taxonomy()
        categories = _flatten_categories(areas)
        category_catalog = [
            {"id": c["id"], "name": c["name"], "area": c["area"]}
            for c in categories
        ]

        prompt_payload = {
            "project": {
                "name": project.name,
                "source_filename": project.source_filename,
                "source_language": config.get("sourceLang"),
                "target_language": target_lang,
                "duration_seconds": project.result_duration or project.source_duration,
                "source_url": source_url,
            },
            "metadata_language": language,
            "metadata_language_label": _language_label(language),
            "transcript_digest": transcript,
            "report": report,
            "taxonomy": category_catalog,
            "allowed_content_types": sorted(ALLOWED_CONTENT_TYPES),
            "allowed_levels": sorted(ALLOWED_LEVELS),
            "allowed_shelf_life": sorted(ALLOWED_SHELF_LIFE),
        }

        system_prompt = (
            "Ты редактор видеотеки «ИИ Борщ» — библиотеки лучших видео для инженеров, "
            "разработчиков и людей из Hacker News. Ты получаешь артефакты переведённого "
            "и озвученного видео: название проекта, язык, транскрипт и отчёт пайплайна. "
            "Придумай честную, конкретную карточку видео. Не выдумывай факты, которых "
            "нет в транскрипте. Верни ТОЛЬКО валидный JSON без markdown."
        )
        user_prompt = f"""
Сгенерируй метаданные для публикации видео в ИИ Борщ.

Язык всех текстовых полей: {_language_label(language)} ({language}).
Пиши title, summary, why_hn, primary_topic, tags и reason на этом языке.
Категории выбирай только из списка taxonomy по id.

Верни JSON строго такой структуры:
{{
  "title": "краткий цепляющий заголовок, до 120 символов",
  "summary": "2–4 предложения о содержании видео",
  "why_hn": "1–3 предложения, почему это интересно инженерам / HN-аудитории",
  "content_type": "одно из allowed_content_types",
  "level": "одно из allowed_levels",
  "shelf_life": "одно из allowed_shelf_life",
  "primary_topic": "главная тема, коротко",
  "tags": ["5–10 конкретных тегов"],
  "category_ids": ["1–4 id из taxonomy"],
  "reason": "короткая причина для ленты, до 160 символов"
}}

Данные проекта:
{json.dumps(prompt_payload, ensure_ascii=False)}
"""
        raw = _llm_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        metadata = _normalize_metadata(raw, project, categories, language, transcript)
        return project, metadata, source_url
    finally:
        db.close()


def _result_local_path(project_id: str, project: Project) -> Optional[Path]:
    pm = ProjectManager(project_id)
    target_lang = (project.config or {}).get("targetLang", "ru")
    local = pm.get_result_video_path(target_lang)
    if local.exists():
        return local
    if pm.results_dir.exists():
        videos = list(pm.results_dir.glob("*.mp4"))
        if videos:
            return videos[0]
    return None


def _copy_result_to_borsch(
    project_id: str,
    project: Project,
    vid: str,
    *,
    admin_token: str,
) -> str:
    """Copy the result video into the ИИ Борщ bucket. Returns the destination key."""
    settings = get_settings()
    store = get_object_storage()
    if not store.is_enabled():
        raise BorschPublishError("S3 storage is not configured in DubbLM")

    dest_key = f"{settings.borsch_s3_prefix}/{vid}/source.mp4"
    local = _result_local_path(project_id, project)
    storage = (project.config or {}).get("storage") or {}
    source_key = storage.get("resultVideoKey") if isinstance(storage, dict) else None

    try:
        if local and local.exists():
            store.client.upload_file(
                str(local),
                settings.borsch_s3_bucket,
                dest_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
        elif source_key:
            # Hetzner Object Storage does not currently support cross-bucket
            # CopyObject reliably. Fall back to a server-side download + upload
            # through DubbLM, which still avoids the public 2 GB upload limit.
            try:
                store.client.copy_object(
                    Bucket=settings.borsch_s3_bucket,
                    Key=dest_key,
                    CopySource={"Bucket": store.bucket, "Key": source_key},
                    ContentType="video/mp4",
                )
            except Exception as copy_exc:
                logger.warning(
                    "Cross-bucket S3 copy failed for %s (%s); restoring and re-uploading",
                    project_id,
                    copy_exc,
                )
                restored = store.ensure_local_result_video(project_id, project)
                if not restored or not restored.exists():
                    raise BorschPublishError(
                        "Dubbed video not found",
                        status_code=404,
                        code="video_not_found",
                    ) from copy_exc
                store.client.upload_file(
                    str(restored),
                    settings.borsch_s3_bucket,
                    dest_key,
                    ExtraArgs={"ContentType": "video/mp4"},
                )
        else:
            restored = store.ensure_local_result_video(project_id, project)
            if not restored or not restored.exists():
                raise BorschPublishError("Dubbed video not found", status_code=404, code="video_not_found")
            store.client.upload_file(
                str(restored),
                settings.borsch_s3_bucket,
                dest_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
        store.client.head_object(Bucket=settings.borsch_s3_bucket, Key=dest_key)
        return dest_key
    except BorschPublishError:
        raise
    except Exception as exc:
        logger.exception("S3 copy to ИИ Борщ failed for %s", project_id)
        raise BorschPublishError(f"Could not copy video to ИИ Борщ S3: {exc}") from exc


def _delete_borsch_object(key: str) -> None:
    try:
        settings = get_settings()
        store = get_object_storage()
        store.client.delete_object(Bucket=settings.borsch_s3_bucket, Key=key)
    except Exception as exc:
        logger.warning("Could not clean up ИИ Борщ object %s: %s", key, exc)


def _register_in_borsch(vid: str, source_key: str, metadata: Dict[str, Any], admin_token: str) -> str:
    payload = {
        "vid": vid,
        "source_key": source_key,
        "title": metadata["title"],
        "channel": metadata["channel"],
        "summary": metadata["summary"],
        "why_hn": metadata["why_hn"],
        "published": datetime.now(timezone.utc).isoformat(),
        "content_type": metadata["content_type"],
        "level": metadata["level"],
        "shelf_life": metadata["shelf_life"],
        "primary_topic": metadata["primary_topic"],
        "tags": metadata["tags"],
        "category_ids": metadata["category_ids"],
        "views": 0,
        "likes": 0,
        "score": 0,
        "reason": metadata["reason"],
        "license": metadata["license"],
        "source_url": metadata.get("source_url"),
        "featured": False,
    }
    try:
        response = requests.post(
            _api_url("/api/admin/videos/register"),
            json=payload,
            headers={"X-Admin-Token": admin_token},
            timeout=90,
        )
    except requests.RequestException as exc:
        raise BorschPublishError(f"ИИ Борщ register request failed: {exc}") from exc
    if response.status_code == 401:
        raise BorschAuthError()
    if not response.ok:
        raise BorschPublishError(_request_error(response, "ИИ Борщ register failed"))
    data = response.json()
    return str(data.get("status") or "processing")


def _fallback_upload(
    project_id: str,
    project: Project,
    metadata: Dict[str, Any],
    admin_token: str,
) -> tuple[str, str]:
    settings = get_settings()
    local = _result_local_path(project_id, project)
    if not local or not local.exists():
        store = get_object_storage()
        local = store.ensure_local_result_video(project_id, project)
    if not local or not local.exists():
        raise BorschPublishError("Dubbed video not found for upload", status_code=404, code="video_not_found")
    if local.stat().st_size > settings.borsch_max_fallback_upload_bytes:
        raise BorschPublishError(
            "Video is too large for the fallback upload. Direct S3 copy failed.",
            code="video_too_large",
        )
    data = {
        "title": metadata["title"],
        "channel": metadata["channel"],
        "summary": metadata["summary"],
        "why_hn": metadata["why_hn"],
        "published": datetime.now(timezone.utc).isoformat(),
        "content_type": metadata["content_type"],
        "level": metadata["level"],
        "shelf_life": metadata["shelf_life"],
        "primary_topic": metadata["primary_topic"],
        "tags": ", ".join(metadata["tags"]),
        "category_ids": ", ".join(metadata["category_ids"]),
        "views": "0",
        "likes": "0",
        "score": "0",
        "reason": metadata["reason"],
        "license": metadata["license"],
        "source_url": metadata.get("source_url") or "",
    }
    try:
        with local.open("rb") as handle:
            response = requests.post(
                _api_url("/api/admin/videos"),
                data=data,
                files={"file": (local.name, handle, "video/mp4")},
                headers={"X-Admin-Token": admin_token},
                timeout=(15, settings.borsch_publish_timeout),
            )
    except requests.RequestException as exc:
        raise BorschPublishError(f"Fallback upload failed: {exc}") from exc
    if response.status_code == 401:
        raise BorschAuthError()
    if not response.ok:
        raise BorschPublishError(_request_error(response, "Fallback upload failed"))
    payload = response.json()
    return str(payload["vid"]), str(payload.get("status") or "processing")


def publish_project_to_borsch(project_id: str, admin_token: str) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.borsch_publish_enabled:
        raise BorschPublishError("Publishing is disabled", status_code=404, code="publish_disabled")

    validate_borsch_token(admin_token)
    project, metadata, source_url = generate_metadata(project_id, admin_token)
    metadata["source_url"] = source_url

    last_error: Optional[Exception] = None
    for _ in range(3):
        vid = secrets.token_urlsafe(8)
        source_key = ""
        try:
            source_key = _copy_result_to_borsch(project_id, project, vid, admin_token=admin_token)
            status = _register_in_borsch(vid, source_key, metadata, admin_token)
            state = {
                "vid": vid,
                "status": status,
                "url": f"{settings.borsch_public_url}/watch/{vid}",
                "publishedAt": _now_iso(),
                "metadata": metadata,
                "sourceUrl": source_url,
            }
            db = get_db_session(fresh=True)
            try:
                fresh = db.query(Project).filter(Project.id == project_id).first()
                if fresh:
                    config = dict(fresh.config or {})
                    config["borsch"] = state
                    fresh.config = config
                    flag_modified(fresh, "config")
                    db.commit()
            finally:
                db.close()
            return {"enabled": True, "published": True, **state}
        except BorschAuthError:
            if source_key:
                _delete_borsch_object(source_key)
            raise
        except BorschPublishError as exc:
            if source_key:
                _delete_borsch_object(source_key)
            last_error = exc
            # A collision is the only error worth retrying with a new id.
            if "already exists" not in str(exc).lower():
                break
        except Exception as exc:
            if source_key:
                _delete_borsch_object(source_key)
            last_error = exc
            break

    # Direct S3 copy failed: try the normal multipart API for smaller files.
    try:
        vid, status = _fallback_upload(project_id, project, metadata, admin_token)
        state = {
            "vid": vid,
            "status": status,
            "url": f"{settings.borsch_public_url}/watch/{vid}",
            "publishedAt": _now_iso(),
            "metadata": metadata,
            "sourceUrl": source_url,
        }
        db = get_db_session(fresh=True)
        try:
            fresh = db.query(Project).filter(Project.id == project_id).first()
            if fresh:
                config = dict(fresh.config or {})
                config["borsch"] = state
                fresh.config = config
                flag_modified(fresh, "config")
                db.commit()
        finally:
            db.close()
        return {"enabled": True, "published": True, **state}
    except Exception as fallback_exc:
        if isinstance(fallback_exc, BorschAuthError):
            raise fallback_exc
        if isinstance(last_error, BorschAuthError):
            raise last_error
        message = str(last_error or fallback_exc)
        raise BorschPublishError(
            f"Could not publish to ИИ Борщ: {message}; fallback: {fallback_exc}",
            code=getattr(last_error, "code", "borsch_publish_failed"),
        ) from fallback_exc


def get_publish_state(project_id: str, admin_token: Optional[str] = None) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.borsch_publish_enabled:
        return {"enabled": False, "published": False}

    db = get_db_session(fresh=True)
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise BorschPublishError("Project not found", status_code=404, code="project_not_found")
        config = dict(project.config or {})
        state = config.get("borsch")
        if not isinstance(state, dict) or not state.get("vid"):
            return {"enabled": True, "published": False}

        vid = str(state["vid"])
        if admin_token:
            try:
                response = requests.get(
                    _api_url(f"/api/admin/videos/{vid}"),
                    headers={"X-Admin-Token": admin_token},
                    timeout=15,
                )
                if response.status_code == 200:
                    remote = response.json()
                    remote_status = str(remote.get("status") or state.get("status") or "processing")
                    if remote_status != state.get("status"):
                        state["status"] = remote_status
                        config["borsch"] = state
                        project.config = config
                        flag_modified(project, "config")
                        db.commit()
                elif response.status_code == 404:
                    state["status"] = "missing"
            except requests.RequestException:
                pass

        return {"enabled": True, "published": True, **state}
    finally:
        db.close()
