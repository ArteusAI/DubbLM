"""Publish a dubbed project to the ИИ Борщ video library.

DubbLM only uploads the finished video and the aligned transcript.  ИИ Борщ
generates the public card (title, summary, tags, categories) and the timeline
from that transcript, so the video never has to be transcribed again.

The feature is intentionally opt-in: ``BORSCH_PUBLISH_ENABLED`` must be true.
The admin token is supplied by the browser on the first publish and is never
persisted on the server.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy.orm.attributes import flag_modified

from ..config import get_settings
from ..database.models import Project, ProjectStatus, Segment
from ..database.session import get_db_session
from ..services.object_storage import get_object_storage
from ..services.project_manager import ProjectManager

logger = logging.getLogger(__name__)

_SRT_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")
_TTS_TAG_RE = re.compile(r"\[[^\[\]]{1,40}\]")
_ANGLE_TAG_RE = re.compile(r"<[^<>]{1,40}>")
_PIPE_TAG_RE = re.compile(r"\|([^|]{1,60})\|")
_MAX_TRANSCRIPT_CHARS = 200000
_MAX_TRANSCRIPT_SEGMENTS = 20000


class BorschPublishError(Exception):
    """Base error for the publish flow."""

    def __init__(self, message: str, *, status_code: int = 502, code: str = "borsch_publish_failed"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class BorschAuthError(BorschPublishError):
    def __init__(self, message: str = "Invalid ИИ Борщ admin token"):
        super().__init__(message, status_code=401, code="borsch_invalid_token")


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


def _clean_text(value: Any) -> str:
    """Remove TTS/emotion markup before sending text to ИИ Борщ."""
    text = str(value or "")
    text = _TTS_TAG_RE.sub(" ", text)
    text = _ANGLE_TAG_RE.sub(" ", text)
    text = _PIPE_TAG_RE.sub(lambda match: f" {match.group(1)} ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _srt_seconds(parts: tuple[str, str, str, str]) -> float:
    hours, minutes, seconds, millis = parts
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis.ljust(3, "0")[:3]) / 1000.0
    )


def _parse_srt(path: Path) -> List[Dict[str, Any]]:
    """Parse an SRT file into the timed chunks accepted by ИИ Борщ."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff")
    except OSError as exc:
        logger.warning("Could not read subtitles %s: %s", path, exc)
        return []

    out: List[Dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        index = 0
        if lines[0].strip().isdigit():
            index = 1
        if index >= len(lines):
            continue
        times = _SRT_TIME_RE.findall(lines[index])
        if len(times) < 2:
            continue
        start = max(0.0, _srt_seconds(times[0]))
        end = max(start + 0.01, _srt_seconds(times[1]))
        body = _clean_text(" ".join(lines[index + 1 :]))
        if not body:
            continue
        out.append(
            {
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "text": body,
            }
        )
    out.sort(key=lambda item: item["start_s"])
    return out


def _segments_from_db(project_id: str) -> List[Dict[str, Any]]:
    """Fallback transcript builder that keeps speaker labels in the text."""
    db = get_db_session(fresh=True)
    try:
        rows = (
            db.query(Segment)
            .filter(Segment.project_id == project_id)
            .order_by(Segment.sequence.asc())
            .all()
        )
        out: List[Dict[str, Any]] = []
        for segment in rows:
            if segment.is_muted:
                continue
            text = _clean_text(segment.translated_text)
            if not text:
                continue
            start = max(0.0, float(segment.start_time or 0))
            end = max(start + 0.01, float(segment.end_time or start))
            out.append(
                {
                    "start_s": round(start, 3),
                    "end_s": round(end, 3),
                    "text": text,
                    "speaker": str(segment.speaker or "").strip(),
                }
            )
        out.sort(key=lambda item: item["start_s"])
        return out
    finally:
        db.close()


def _build_publish_transcript(project_id: str, project: Project) -> Dict[str, Any]:
    """Return the transcript payload for ИИ Борщ.

    The rendered SRT is preferred because its timings already match the final
    MP4.  If it is missing, project segments are used as a safe fallback.
    """
    target_lang = str((project.config or {}).get("targetLang") or "ru").strip().lower() or "ru"

    segments: List[Dict[str, Any]] = []
    from_srt = False
    try:
        store = get_object_storage()
        srt_path = store.ensure_local_subtitle(project_id, project, target_lang, "srt")
        if srt_path and srt_path.exists():
            segments = _parse_srt(srt_path)
            from_srt = bool(segments)
    except Exception as exc:
        logger.warning("Could not prepare subtitles for %s: %s", project_id, exc)

    if not segments:
        segments = _segments_from_db(project_id)

    if not segments:
        return {"transcript": "", "transcript_segments": [], "transcript_language": target_lang}

    if from_srt:
        transcript = "\n".join(item["text"] for item in segments)
    else:
        transcript = "\n".join(
            f"{item['speaker']}: {item['text']}" if item.get("speaker") else item["text"]
            for item in segments
        )

    public_segments = [
        {
            "start_s": item["start_s"],
            "end_s": item["end_s"],
            "text": item["text"],
        }
        for item in segments[:_MAX_TRANSCRIPT_SEGMENTS]
    ]
    return {
        "transcript": transcript[:_MAX_TRANSCRIPT_CHARS],
        "transcript_segments": public_segments,
        "transcript_language": target_lang,
    }


def _metadata_language(project: Project) -> str:
    settings = get_settings()
    configured = (settings.borsch_metadata_language or "auto").strip().lower()
    if configured and configured != "auto":
        return configured
    target = (project.config or {}).get("targetLang") or "ru"
    return str(target).strip().lower() or "ru"


def _source_url(project: Project) -> Optional[str]:
    config = project.config or {}
    for key in ("downloadUrl", "sourceUrl", "url"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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


def _upload_result_to_borsch(local: Path, dest_key: str) -> None:
    """Upload a large result video with retries and a longer read timeout.

    The default boto3 60-second read timeout is too short for multi-gigabyte
    multipart uploads to Hetzner Object Storage.
    """
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    settings = get_settings()
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region or "fsn1",
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=30,
            read_timeout=300,
            retries={"max_attempts": 10, "mode": "standard"},
            max_pool_connections=16,
        ),
    )
    client.upload_file(
        str(local),
        settings.borsch_s3_bucket,
        dest_key,
        ExtraArgs={"ContentType": "video/mp4"},
        Config=TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=32 * 1024 * 1024,
            max_concurrency=4,
            use_threads=True,
        ),
    )


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
            _upload_result_to_borsch(local, dest_key)
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
                _upload_result_to_borsch(restored, dest_key)
        else:
            restored = store.ensure_local_result_video(project_id, project)
            if not restored or not restored.exists():
                raise BorschPublishError("Dubbed video not found", status_code=404, code="video_not_found")
            _upload_result_to_borsch(restored, dest_key)
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


def _publish_payload(project: Project, transcript_payload: Dict[str, Any]) -> Dict[str, Any]:
    settings = get_settings()
    return {
        "title": project.name or "Без названия",
        "channel": settings.borsch_channel,
        "published": _now_iso(),
        "source_url": _source_url(project),
        "license": "Права на исходное видео принадлежат автору; дубляж выполнен DubbLM.",
        "auto_metadata": True,
        "metadata_language": _metadata_language(project),
        "transcript": transcript_payload["transcript"],
        "transcript_language": transcript_payload["transcript_language"],
        "transcript_segments": transcript_payload["transcript_segments"],
    }


def _ensure_remote_metadata_supported(data: Dict[str, Any]) -> None:
    if data.get("auto_metadata") is not True:
        raise BorschPublishError(
            "ИИ Борщ не подтвердил генерацию метаданных из переданного транскрипта. "
            "Обновите ИИ Борщ до версии с поддержкой auto_metadata.",
            code="borsch_metadata_not_supported",
        )


def _register_in_borsch(
    vid: str,
    source_key: str,
    project: Project,
    transcript_payload: Dict[str, Any],
    admin_token: str,
) -> Dict[str, Any]:
    payload = {"vid": vid, "source_key": source_key, **_publish_payload(project, transcript_payload)}
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
    _ensure_remote_metadata_supported(data)
    return data


def _fallback_upload(
    project_id: str,
    project: Project,
    transcript_payload: Dict[str, Any],
    admin_token: str,
) -> tuple[str, Dict[str, Any]]:
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
    payload = _publish_payload(project, transcript_payload)
    data = {
        "title": payload["title"],
        "channel": payload["channel"],
        "published": payload["published"],
        "source_url": payload["source_url"] or "",
        "license": payload["license"],
        "auto_metadata": "true",
        "metadata_language": payload["metadata_language"],
        "transcript": payload["transcript"],
        "transcript_language": payload["transcript_language"],
        "transcript_segments": json.dumps(payload["transcript_segments"], ensure_ascii=False),
        "views": "0",
        "likes": "0",
        "score": "0",
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
    result = response.json()
    _ensure_remote_metadata_supported(result)
    return str(result["vid"]), result


def _get_project(project_id: str) -> Project:
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
        return project
    finally:
        db.close()


def _store_publish_state(project_id: str, state: Dict[str, Any]) -> None:
    db = get_db_session(fresh=True)
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return
        config = dict(project.config or {})
        config["borsch"] = state
        project.config = config
        flag_modified(project, "config")
        db.commit()
    finally:
        db.close()


def _state_from_result(
    vid: str,
    status: str,
    project: Project,
    remote: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    state: Dict[str, Any] = {
        "vid": vid,
        "status": status,
        "url": f"{settings.borsch_public_url}/watch/{vid}",
        "publishedAt": _now_iso(),
        "sourceUrl": _source_url(project),
        "metadataStatus": (remote or {}).get("metadata_status") or ("pending" if status == "processing" else None),
        "timelineStatus": (remote or {}).get("timeline_status"),
    }
    if remote:
        metadata = _remote_metadata(remote)
        if metadata:
            state["metadata"] = metadata
        timeline = _remote_timeline(remote)
        if timeline:
            state["timeline"] = timeline
    return state


def publish_project_to_borsch(project_id: str, admin_token: str) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.borsch_publish_enabled:
        raise BorschPublishError("Publishing is disabled", status_code=404, code="publish_disabled")

    validate_borsch_token(admin_token)
    project = _get_project(project_id)
    transcript_payload = _build_publish_transcript(project_id, project)
    if not transcript_payload["transcript"]:
        raise BorschPublishError(
            "Project has no translated transcript to publish",
            status_code=400,
            code="empty_transcript",
        )

    last_error: Optional[Exception] = None
    for _ in range(3):
        vid = secrets.token_urlsafe(8)
        source_key = ""
        try:
            source_key = _copy_result_to_borsch(project_id, project, vid, admin_token=admin_token)
            remote = _register_in_borsch(
                vid,
                source_key,
                project,
                transcript_payload,
                admin_token,
            )
            state = _state_from_result(
                vid,
                str(remote.get("status") or "processing"),
                project,
                remote,
            )
            _store_publish_state(project_id, state)
            return {"enabled": True, "published": True, **state}
        except BorschAuthError:
            if source_key:
                _delete_borsch_object(source_key)
            raise
        except BorschPublishError as exc:
            if source_key:
                _delete_borsch_object(source_key)
            last_error = exc
            if exc.code == "borsch_metadata_not_supported":
                raise
            # A collision is the only error worth retrying with a new id.
            if "already exists" not in str(exc).lower():
                break
        except Exception as exc:
            if source_key:
                _delete_borsch_object(source_key)
            last_error = exc
            break

    # Direct S3 copy failed: try the normal multipart API for smaller files.
    if getattr(last_error, "code", "") == "borsch_metadata_not_supported":
        raise last_error  # type: ignore[misc]

    try:
        vid, remote = _fallback_upload(project_id, project, transcript_payload, admin_token)
        state = _state_from_result(
            vid,
            str(remote.get("status") or "processing"),
            project,
            remote,
        )
        _store_publish_state(project_id, state)
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


def _remote_metadata(remote: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if remote.get("metadata_status") != "ready":
        return None
    return {
        "title": remote.get("title") or "",
        "channel": remote.get("channel") or "",
        "summary": remote.get("summary") or "",
        "why_hn": remote.get("why_hn") or "",
        "content_type": remote.get("content_type") or "",
        "level": remote.get("level") or "",
        "shelf_life": remote.get("shelf_life") or "",
        "primary_topic": remote.get("primary_topic") or "",
        "tags": remote.get("tags") or [],
        "category_ids": remote.get("category_ids") or [],
        "reason": remote.get("reason") or "",
        "license": remote.get("license") or "",
        "source_url": remote.get("source_url"),
    }


def _remote_timeline(remote: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = remote.get("timeline")
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            start = max(0, int(round(float(item.get("start_s") or 0))))
        except (TypeError, ValueError):
            start = 0
        title = str(item.get("title") or "").strip()
        if title:
            out.append({"start_s": start, "title": title[:160]})
    return out[:40]


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
                    state["status"] = str(
                        remote.get("status") or state.get("status") or "processing"
                    )
                    state["metadataStatus"] = remote.get("metadata_status")
                    state["timelineStatus"] = remote.get("timeline_status")
                    state["metadataError"] = remote.get("metadata_error")
                    state["hasTranscript"] = bool(remote.get("has_transcript"))
                    state["transcriptSegments"] = int(
                        remote.get("transcript_segments_count") or 0
                    )
                    metadata = _remote_metadata(remote)
                    if metadata:
                        state["metadata"] = metadata
                    timeline = _remote_timeline(remote)
                    if timeline:
                        state["timeline"] = timeline
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
