"""S3-compatible object storage (Hetzner Object Storage, MinIO, AWS S3)."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import quote

from ..config import Settings, get_settings
from ..database.models import Project
from .project_manager import ProjectManager

logger = logging.getLogger(__name__)


def format_content_disposition(disposition: str, filename: str) -> str:
    """Format Content-Disposition header safely according to RFC 5987 / RFC 6266.

    Ensures non-ASCII / non-Latin-1 characters are properly percent-encoded in
    filename* and sanitized in the ASCII-only fallback filename.
    """
    clean_name = (Path(filename).name if filename else "file").replace("\r", "").replace("\n", "")
    encoded_name = quote(clean_name, safe="")
    ext = Path(clean_name).suffix
    stem_ascii = Path(clean_name).stem.encode("ascii", "ignore").decode("ascii").replace('"', '').strip()
    if not stem_ascii:
        ascii_fallback = f"download{ext}" if ext else "download"
    else:
        ascii_fallback = f"{stem_ascii}{ext}"

    if encoded_name != clean_name or '"' in clean_name:
        return f'{disposition}; filename="{ascii_fallback}"; filename*=utf-8\'\'{encoded_name}'
    return f'{disposition}; filename="{clean_name}"'


class ObjectStorage:
    """Thin wrapper around boto3 S3 client for project media archives."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None

    def is_enabled(self) -> bool:
        s = self.settings
        if not s.s3_enabled:
            return False
        return bool(s.s3_endpoint_url and s.s3_access_key and s.s3_secret_key and s.s3_bucket)

    @property
    def client(self):
        if self._client is None:
            if not self.is_enabled():
                raise RuntimeError("S3 object storage is not configured")
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 is required for S3 storage. Install with: pip install boto3"
                ) from exc

            s = self.settings
            self._client = boto3.client(
                "s3",
                endpoint_url=s.s3_endpoint_url,
                region_name=s.s3_region or "fsn1",
                aws_access_key_id=s.s3_access_key,
                aws_secret_access_key=s.s3_secret_key,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            )
        return self._client

    @property
    def bucket(self) -> str:
        return self.settings.s3_bucket or ""

    def object_key(self, project_id: str, kind: str, filename: str) -> str:
        """Build S3 key: {prefix}/{project_id}/{kind}/{filename}."""
        parts = [p for p in (self.settings.s3_prefix, project_id, kind, filename) if p]
        return "/".join(parts)

    def project_prefix(self, project_id: str) -> str:
        parts = [p for p in (self.settings.s3_prefix, project_id) if p]
        return "/".join(parts) + "/"

    def upload_file(self, local_path: Path, key: str) -> None:
        logger.info("S3 upload %s -> s3://%s/%s", local_path, self.bucket, key)
        self.client.upload_file(str(local_path), self.bucket, key)

    def download_file(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("S3 download s3://%s/%s -> %s", self.bucket, key, local_path)
        self.client.download_file(self.bucket, key, str(local_path))

    def head_object(self, key: str) -> dict[str, Any]:
        return self.client.head_object(Bucket=self.bucket, Key=key)

    def object_exists(self, key: str) -> bool:
        try:
            self.head_object(key)
            return True
        except Exception as exp:
            try:
                from botocore.exceptions import ClientError
            except ImportError:
                raise exp
            if isinstance(exp, ClientError):
                code = exp.response.get("Error", {}).get("Code", "")
                if code in ("404", "NoSuchKey", "NotFound"):
                    return False
            raise

    def get_object_size(self, key: str) -> int:
        return int(self.head_object(key)["ContentLength"])

    def presign_get_url(
        self,
        key: str,
        *,
        expires_in: int | None = None,
        filename: str | None = None,
        content_type: str = "video/mp4",
        as_attachment: bool = False,
    ) -> str:
        """Generate a time-limited HTTPS URL for downloading/viewing an object."""
        if not self.is_enabled():
            raise RuntimeError("S3 is not enabled")
        ttl = expires_in if expires_in is not None else int(self.settings.s3_presign_expires)
        # Cap at 7 days (common limit for S3-compatible providers)
        ttl = max(60, min(ttl, 7 * 24 * 3600))
        name = filename or Path(key).name
        disposition = "attachment" if as_attachment else "inline"
        params: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "ResponseContentType": content_type,
            "ResponseContentDisposition": format_content_disposition(disposition, name),
        }
        url = self.client.generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=ttl,
        )
        return url

    def stream_range(
        self,
        key: str,
        start: int | None = None,
        end: int | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> Iterator[bytes]:
        """Yield object bytes, optionally limited by inclusive byte range."""
        extra: dict[str, str] = {}
        if start is not None:
            if end is not None:
                extra["Range"] = f"bytes={start}-{end}"
            else:
                extra["Range"] = f"bytes={start}-"
        response = self.client.get_object(Bucket=self.bucket, Key=key, **extra)
        body = response["Body"]
        try:
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            body.close()

    def list_keys(self, prefix: str, max_keys: int = 1000) -> list[str]:
        """List object keys under prefix (paginated, up to max_keys total)."""
        keys: list[str] = []
        continuation: str | None = None
        while len(keys) < max_keys:
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": prefix,
                "MaxKeys": min(1000, max_keys - len(keys)),
            }
            if continuation:
                kwargs["ContinuationToken"] = continuation
            listing = self.client.list_objects_v2(**kwargs)
            for obj in listing.get("Contents") or []:
                keys.append(obj["Key"])
            if not listing.get("IsTruncated"):
                break
            continuation = listing.get("NextContinuationToken")
        return keys

    def discover_storage_meta(self, project_id: str) -> dict[str, Any] | None:
        """Infer storage metadata from objects already on S3 under the project prefix.

        Used when local media was cleaned but config.storage was never committed.
        """
        if not self.is_enabled():
            return None
        prefix = self.project_prefix(project_id)
        try:
            keys = self.list_keys(prefix, max_keys=200)
        except Exception as exc:
            logger.warning("Failed to list S3 objects for %s: %s", project_id, exc)
            return None
        if not keys:
            return None

        storage: dict[str, Any] = {
            "provider": "s3",
            "bucket": self.bucket,
            "discovered": True,
        }
        result_keys: list[str] = []
        srt_keys: list[str] = []
        result_video_key: str | None = None
        video_exts = {".mp4", ".mkv", ".webm", ".mov"}

        for key in keys:
            name = Path(key).name
            lower = name.lower()
            rel = key[len(prefix):] if key.startswith(prefix) else key

            if rel.startswith("source/") or "/source/" in f"/{rel}":
                if not storage.get("sourceKey"):
                    storage["sourceKey"] = key
                    storage["sourceFilename"] = name
            elif rel.startswith("report/") or name in {"report.md", "report.json"}:
                if name.endswith(".md"):
                    storage["reportKey"] = key
                elif name.endswith(".json"):
                    storage["reportJsonKey"] = key
            elif rel.startswith("results/") or True:
                # keys under project may be results/* directly if structure differs
                if any(lower.endswith(ext) for ext in video_exts):
                    result_video_key = key
                    result_keys.append(key)
                elif lower.endswith(".srt") or lower.endswith(".vtt"):
                    srt_keys.append(key)
                elif "/results/" in key:
                    result_keys.append(key)

        # Prefer video in results/ over other paths
        videos = [k for k in keys if any(k.lower().endswith(ext) for ext in video_exts)]
        results_videos = [k for k in videos if "/results/" in k]
        if results_videos:
            # Prefer language-tagged result over source copies
            result_video_key = sorted(results_videos, key=lambda k: (0 if "_ru." in k or k.count("_") > 1 else 1, k))[0]
        elif videos and not result_video_key:
            result_video_key = videos[0]

        if result_video_key:
            storage["resultVideoKey"] = result_video_key
        if result_keys:
            storage["resultKeys"] = result_keys
        if srt_keys:
            storage["resultSrtKeys"] = srt_keys
        elif True:
            storage["resultSrtKeys"] = [
                k for k in keys if k.endswith(".srt") or k.endswith(".vtt")
            ]

        if not storage.get("sourceKey"):
            source_keys = [k for k in keys if "/source/" in k]
            if source_keys:
                storage["sourceKey"] = source_keys[0]
                storage["sourceFilename"] = Path(source_keys[0]).name

        if not storage.get("resultVideoKey") and not storage.get("sourceKey"):
            return None
        return storage

    def resolve_result_video_key(self, project_id: str, project: Project | None = None) -> str | None:
        """Return S3 key for dubbed video: config.storage or discover from bucket."""
        storage = _storage_meta(project)
        key = storage.get("resultVideoKey")
        if key:
            return key
        discovered = self.discover_storage_meta(project_id)
        if discovered:
            return discovered.get("resultVideoKey")
        return None

    def resolve_source_key(self, project_id: str, project: Project | None = None) -> str | None:
        storage = _storage_meta(project)
        key = storage.get("sourceKey")
        if key:
            return key
        discovered = self.discover_storage_meta(project_id)
        if discovered:
            return discovered.get("sourceKey")
        return None

    def delete_prefix(self, prefix: str) -> int:
        """Delete all objects under prefix. Returns number of deleted objects."""
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        deleted = 0
        continuation: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if continuation:
                kwargs["ContinuationToken"] = continuation
            listing = self.client.list_objects_v2(**kwargs)
            contents = listing.get("Contents") or []
            if contents:
                objects = [{"Key": obj["Key"]} for obj in contents]
                self.client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": objects, "Quiet": True},
                )
                deleted += len(objects)
            if not listing.get("IsTruncated"):
                break
            continuation = listing.get("NextContinuationToken")
        if deleted:
            logger.info("S3 deleted %s objects under %s", deleted, prefix)
        return deleted

    def archive_project_media(self, project_id: str) -> dict[str, Any]:
        """Upload source + results (+ report) for a finished project.

        Returns storage metadata to persist on project.config['storage'].
        Raises on upload failure (caller should keep local files).
        """
        if not self.is_enabled():
            raise RuntimeError("S3 is not enabled")

        pm = ProjectManager(project_id)
        storage: dict[str, Any] = {
            "provider": "s3",
            "bucket": self.bucket,
            "uploadedAt": datetime.now(timezone.utc).isoformat(),
        }

        source = pm.get_source_video_path()
        if source and source.exists():
            key = self.object_key(project_id, "source", source.name)
            self.upload_file(source, key)
            if not self.object_exists(key):
                raise RuntimeError(f"S3 upload verification failed for {key}")
            storage["sourceKey"] = key
            storage["sourceFilename"] = source.name

        result_keys: list[str] = []
        srt_keys: list[str] = []
        result_video_key: str | None = None
        if pm.results_dir.exists():
            for path in sorted(pm.results_dir.iterdir()):
                if not path.is_file():
                    continue
                key = self.object_key(project_id, "results", path.name)
                self.upload_file(path, key)
                if not self.object_exists(key):
                    raise RuntimeError(f"S3 upload verification failed for {key}")
                if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
                    result_video_key = key
                    result_keys.append(key)
                elif path.suffix.lower() in {".srt", ".vtt"}:
                    srt_keys.append(key)
                else:
                    result_keys.append(key)

        if result_video_key:
            storage["resultVideoKey"] = result_video_key
        if result_keys:
            storage["resultKeys"] = result_keys
        if srt_keys:
            storage["resultSrtKeys"] = srt_keys

        report_md = pm.artifacts_dir / "report.md"
        if report_md.exists():
            key = self.object_key(project_id, "report", "report.md")
            self.upload_file(report_md, key)
            storage["reportKey"] = key
        report_json = pm.artifacts_dir / "report.json"
        if report_json.exists():
            key = self.object_key(project_id, "report", "report.json")
            self.upload_file(report_json, key)
            storage["reportJsonKey"] = key

        if not storage.get("sourceKey") and not storage.get("resultVideoKey"):
            raise RuntimeError("Nothing to archive to S3 (no source or result video)")

        return storage

    def cleanup_local_after_archive(
        self,
        project_id: str,
        *,
        keep_debug_artifacts: bool | None = None,
    ) -> dict[str, Any]:
        """Remove local media after a successful S3 archive.

        Never raises on individual file permission errors — returns stats
        with ``errors`` list so callers can still commit storage metadata.
        """
        pm = ProjectManager(project_id)
        keep = (
            self.settings.keep_debug_artifacts
            if keep_debug_artifacts is None
            else keep_debug_artifacts
        )
        removed = 0
        freed = 0
        errors: list[str] = []

        def _rm_file(path: Path) -> None:
            nonlocal removed, freed
            if not path.exists() or not path.is_file():
                return
            size = 0
            try:
                size = path.stat().st_size
            except OSError:
                pass
            try:
                path.unlink()
                removed += 1
                freed += size
                return
            except OSError:
                pass
            try:
                path.chmod(0o666)
                path.unlink()
                removed += 1
                freed += size
            except OSError as exc:
                errors.append(f"{path}: {exc}")
                logger.warning("Failed to remove local file %s: %s", path, exc)

        def _rm_tree(path: Path) -> None:
            nonlocal removed, freed, errors
            if not path.exists():
                return
            try:
                for child in path.rglob("*"):
                    if child.is_file():
                        try:
                            freed += child.stat().st_size
                            removed += 1
                        except OSError:
                            pass
            except OSError:
                pass
            try:
                shutil.rmtree(path, ignore_errors=True)
            except OSError as exc:
                errors.append(f"{path}/: {exc}")
                logger.warning("Failed to remove tree %s: %s", path, exc)
            if path.exists():
                # Best-effort walk if rmtree could not fully remove
                try:
                    for child in sorted(path.rglob("*"), reverse=True):
                        try:
                            if child.is_file() or child.is_symlink():
                                child.unlink()
                            elif child.is_dir():
                                child.rmdir()
                        except OSError as exc:
                            errors.append(f"{child}: {exc}")
                    if path.exists():
                        try:
                            path.rmdir()
                        except OSError as exc:
                            errors.append(f"{path}: {exc}")
                except OSError as exc:
                    errors.append(f"{path}: {exc}")

        if pm.uploads_dir.exists():
            for path in list(pm.uploads_dir.iterdir()):
                if path.is_file() or path.is_symlink():
                    _rm_file(path)
                elif path.is_dir():
                    _rm_tree(path)

        if pm.results_dir.exists():
            for path in list(pm.results_dir.iterdir()):
                if path.is_file() or path.is_symlink():
                    _rm_file(path)
                elif path.is_dir():
                    _rm_tree(path)

        if not keep:
            if pm.artifacts_dir.exists():
                _rm_tree(pm.artifacts_dir)
            if pm.cache_dir.exists():
                _rm_tree(pm.cache_dir)
            logger.info(
                "Removed local media for project %s (files≈%s, freed≈%s, errors=%s)",
                project_id,
                removed,
                freed,
                len(errors),
            )
        else:
            logger.info("KEEP_DEBUG_ARTIFACTS=true — keeping artifacts for %s", project_id)

        try:
            pm.ensure_directories()
        except OSError as exc:
            errors.append(f"ensure_directories: {exc}")

        return {
            "removed_files": removed,
            "freed_bytes": freed,
            "errors": errors,
            "kept_debug": keep,
        }

    def ensure_local_source(self, project_id: str, project: Project | None = None) -> Optional[Path]:
        """Ensure source video exists locally; restore from S3 if needed."""
        pm = ProjectManager(project_id)
        local = pm.get_source_video_path()
        if local and local.exists():
            return local

        storage = _storage_meta(project)
        key = storage.get("sourceKey")
        filename = storage.get("sourceFilename")
        if not key or not self.is_enabled():
            return local if local and local.exists() else None

        if not filename:
            filename = Path(key).name
        target = pm.uploads_dir / filename
        pm.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.download_file(key, target)
        return target if target.exists() else None

    def ensure_local_result_video(
        self,
        project_id: str,
        project: Project | None = None,
    ) -> Optional[Path]:
        """Ensure dubbed result video exists locally; restore from S3 if needed."""
        pm = ProjectManager(project_id)
        storage = _storage_meta(project)
        target_lang = (project.config or {}).get("targetLang", "ru") if project else "ru"

        local = pm.get_result_video_path(target_lang)
        if local.exists():
            return local
        videos = list(pm.results_dir.glob("*.mp4")) if pm.results_dir.exists() else []
        if videos:
            return videos[0]

        key = storage.get("resultVideoKey")
        if not key:
            discovered = self.discover_storage_meta(project_id)
            key = (discovered or {}).get("resultVideoKey")
        if not key or not self.is_enabled():
            return None

        filename = Path(key).name
        target = pm.results_dir / filename
        pm.results_dir.mkdir(parents=True, exist_ok=True)
        self.download_file(key, target)
        return target if target.exists() else None

    def ensure_local_subtitle(
        self,
        project_id: str,
        project: Project | None,
        language: str,
        sub_type: str = "srt",
    ) -> Optional[Path]:
        pm = ProjectManager(project_id)
        local = pm.get_result_subtitles_path(language, sub_type)
        if local.exists():
            return local
        if pm.results_dir.exists():
            matches = list(pm.results_dir.glob(f"*_{language}.{sub_type}"))
            if matches:
                return matches[0]

        storage = _storage_meta(project)
        for key in storage.get("resultSrtKeys") or []:
            name = Path(key).name
            if f"_{language}.{sub_type}" in name or (
                name.endswith(f".{sub_type}") and language in name
            ):
                target = pm.results_dir / name
                pm.results_dir.mkdir(parents=True, exist_ok=True)
                if self.is_enabled():
                    self.download_file(key, target)
                    if target.exists():
                        return target
        return None

    def ensure_local_report(
        self,
        project_id: str,
        project: Project | None = None,
        *,
        kind: str = "md",
    ) -> Optional[Path]:
        pm = ProjectManager(project_id)
        path = pm.artifacts_dir / ("report.md" if kind == "md" else "report.json")
        if path.exists():
            return path

        storage = _storage_meta(project)
        key = storage.get("reportKey" if kind == "md" else "reportJsonKey")
        if not key or not self.is_enabled():
            return None

        path.parent.mkdir(parents=True, exist_ok=True)
        self.download_file(key, path)
        return path if path.exists() else None

    def delete_project(self, project_id: str) -> None:
        if not self.is_enabled():
            return
        try:
            self.delete_prefix(self.project_prefix(project_id))
        except Exception as exc:
            logger.warning("Failed to delete S3 objects for %s: %s", project_id, exc)


def _storage_meta(project: Project | None) -> dict[str, Any]:
    if not project:
        return {}
    config = project.config or {}
    storage = config.get("storage") or {}
    return storage if isinstance(storage, dict) else {}


def get_object_storage(settings: Settings | None = None) -> ObjectStorage:
    return ObjectStorage(settings=settings)


def archive_and_cleanup_project(project_id: str) -> dict[str, Any] | None:
    """Archive media to S3 and free local disk. Returns storage meta or None if disabled."""
    store = get_object_storage()
    if not store.is_enabled():
        return None
    storage = store.archive_project_media(project_id)
    store.cleanup_local_after_archive(project_id)
    return storage
