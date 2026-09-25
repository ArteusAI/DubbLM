#!/usr/bin/env python3
"""Migrate existing project media (source + results + report) to S3.

Uploads local files for projects still on disk, writes
``project.config["storage"]``, then deletes local uploads/results
(and artifacts/cache unless KEEP_DEBUG_ARTIFACTS=true).

Usage:
  python tools/migrate_to_s3.py --dry-run
  python tools/migrate_to_s3.py
  python tools/migrate_to_s3.py -p proj_xxx --force

  # Second pass: free leftover local media after a partial migration
  # (Permission denied / storage saved but files remain)
  sudo .venv/bin/python tools/migrate_to_s3.py --cleanup-only --dry-run
  sudo .venv/bin/python tools/migrate_to_s3.py --cleanup-only --verify-s3

Requires S3_* env vars in .env (see env.example).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from sqlalchemy.orm.attributes import flag_modified  # noqa: E402
from tqdm import tqdm  # noqa: E402
from tqdm.contrib.logging import logging_redirect_tqdm  # noqa: E402

from src.api.config import get_settings  # noqa: E402
from src.api.database.models import Project, ProjectStatus  # noqa: E402
from src.api.database.session import get_session_factory, init_db  # noqa: E402
from src.api.services.object_storage import get_object_storage  # noqa: E402
from src.api.services.project_manager import ProjectManager  # noqa: E402
from src.api.services.video_info import compute_project_size  # noqa: E402

logger = logging.getLogger("migrate_to_s3")


def _human_size(num: float | int) -> str:
    size = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{int(size)}B" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{num}B"


def _human_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _short_name(name: str, max_len: int = 40) -> str:
    if len(name) <= max_len:
        return name
    half = (max_len - 1) // 2
    return f"{name[:half]}…{name[-(max_len - half - 1):]}"


@dataclass
class FileJob:
    path: Path
    kind: str  # source | results | report
    size: int
    key: str


@dataclass
class ProjectPlan:
    project_id: str
    name: str
    status: str
    files: list[FileJob] = field(default_factory=list)
    skip_reason: str | None = None

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


@dataclass
class ProjectResult:
    project_id: str
    name: str
    status: str  # ok | skipped | dry_run | error
    reason: str = ""
    bytes_uploaded: int = 0
    files_uploaded: int = 0
    duration_sec: float = 0.0
    error: str = ""


class TransferProgress:
    """boto3 upload Callback → tqdm bar."""

    def __init__(self, pbar: tqdm):
        self._pbar = pbar

    def __call__(self, bytes_amount: int) -> None:
        self._pbar.update(bytes_amount)


def _list_project_files(store, project_id: str) -> list[FileJob]:
    pm = ProjectManager(project_id)
    jobs: list[FileJob] = []

    source = pm.get_source_video_path()
    if source and source.exists() and source.is_file():
        jobs.append(
            FileJob(
                path=source,
                kind="source",
                size=source.stat().st_size,
                key=store.object_key(project_id, "source", source.name),
            )
        )

    if pm.results_dir.exists():
        for path in sorted(pm.results_dir.iterdir()):
            if path.is_file():
                jobs.append(
                    FileJob(
                        path=path,
                        kind="results",
                        size=path.stat().st_size,
                        key=store.object_key(project_id, "results", path.name),
                    )
                )

    for name in ("report.md", "report.json"):
        path = pm.artifacts_dir / name
        if path.exists() and path.is_file():
            jobs.append(
                FileJob(
                    path=path,
                    kind="report",
                    size=path.stat().st_size,
                    key=store.object_key(project_id, "report", name),
                )
            )
    return jobs


def _should_skip(project: Project, force: bool) -> str | None:
    storage = (project.config or {}).get("storage") or {}
    if (storage.get("sourceKey") or storage.get("resultVideoKey")) and not force:
        return "already has config.storage (use --force)"
    return None


def build_plans(projects: list[Project], store, *, force: bool) -> list[ProjectPlan]:
    plans: list[ProjectPlan] = []
    with tqdm(
        total=len(projects),
        desc="Scanning projects",
        unit="proj",
        dynamic_ncols=True,
        leave=True,
    ) as bar:
        for project in projects:
            plan = ProjectPlan(
                project_id=project.id,
                name=project.name or project.id,
                status=project.status.value if project.status else "?",
            )
            skip = _should_skip(project, force)
            if skip:
                plan.skip_reason = skip
            else:
                plan.files = _list_project_files(store, project.id)
                if not plan.files:
                    plan.skip_reason = "no local source/results/report"
            plans.append(plan)
            bar.set_postfix_str(_short_name(plan.name, 28), refresh=False)
            bar.update(1)
    return plans


def print_plan_summary(plans: list[ProjectPlan], *, dry_run: bool) -> list[ProjectPlan]:
    active = [p for p in plans if not p.skip_reason]
    skipped = [p for p in plans if p.skip_reason]
    total_bytes = sum(p.total_bytes for p in active)
    total_files = sum(len(p.files) for p in active)

    tqdm.write("")
    tqdm.write("=" * 72)
    tqdm.write(f"  Mode:     {'DRY-RUN' if dry_run else 'MIGRATE'}")
    tqdm.write(f"  Projects: {len(plans)} total · {len(active)} to process · {len(skipped)} skip")
    tqdm.write(f"  Files:    {total_files}")
    tqdm.write(f"  Volume:   {_human_size(total_bytes)}")
    tqdm.write("=" * 72)
    if active:
        tqdm.write("  Largest projects:")
        for plan in sorted(active, key=lambda p: p.total_bytes, reverse=True)[:12]:
            tqdm.write(
                f"    {_human_size(plan.total_bytes):>9}  {plan.project_id}  "
                f"{_short_name(plan.name, 34)}  ({len(plan.files)}f, {plan.status})"
            )
        if len(active) > 12:
            tqdm.write(f"    … +{len(active) - 12} more")
    tqdm.write("")
    return active


def upload_file(store, job: FileJob, file_bar: tqdm) -> None:
    file_bar.reset(total=max(job.size, 1))
    file_bar.set_description_str(f"↑ {_short_name(job.path.name, 42)}")
    file_bar.n = 0
    file_bar.refresh()

    kwargs: dict[str, Any] = {}
    if job.size > 0:
        kwargs["Callback"] = TransferProgress(file_bar)

    store.client.upload_file(str(job.path), store.bucket, job.key, **kwargs)

    if file_bar.n < file_bar.total:
        file_bar.update(file_bar.total - file_bar.n)

    if not store.object_exists(job.key):
        raise RuntimeError(f"upload verification failed: {job.key}")


def archive_with_progress(
    store,
    plan: ProjectPlan,
    *,
    overall_bar: tqdm,
    file_bar: tqdm,
    progress_mode: str,
) -> dict[str, Any]:
    """Upload all files for a project; advance overall_bar per file/bytes."""
    storage: dict[str, Any] = {
        "provider": "s3",
        "bucket": store.bucket,
        "uploadedAt": datetime.now(timezone.utc).isoformat(),
    }
    result_keys: list[str] = []
    srt_keys: list[str] = []
    result_video_key: str | None = None

    for job in plan.files:
        upload_file(store, job, file_bar)

        if progress_mode == "bytes":
            overall_bar.update(job.size)
        elif progress_mode == "files":
            overall_bar.update(1)

        if job.kind == "source":
            storage["sourceKey"] = job.key
            storage["sourceFilename"] = job.path.name
        elif job.kind == "report":
            if job.path.name.endswith(".md"):
                storage["reportKey"] = job.key
            else:
                storage["reportJsonKey"] = job.key
        else:
            suffix = job.path.suffix.lower()
            if suffix in {".mp4", ".mkv", ".webm", ".mov"}:
                result_video_key = job.key
                result_keys.append(job.key)
            elif suffix in {".srt", ".vtt"}:
                srt_keys.append(job.key)
            else:
                result_keys.append(job.key)

    if result_video_key:
        storage["resultVideoKey"] = result_video_key
    if result_keys:
        storage["resultKeys"] = result_keys
    if srt_keys:
        storage["resultSrtKeys"] = srt_keys

    if not any(storage.get(k) for k in ("sourceKey", "resultVideoKey", "reportKey")):
        raise RuntimeError("nothing uploaded")
    return storage


def migrate_plan(
    plan: ProjectPlan,
    store,
    *,
    dry_run: bool,
    skip_cleanup: bool,
    keep_debug: bool | None,
    overall_bar: tqdm,
    file_bar: tqdm,
    progress_mode: str,
) -> ProjectResult:
    t0 = time.monotonic()

    if plan.skip_reason:
        return ProjectResult(
            project_id=plan.project_id,
            name=plan.name,
            status="skipped",
            reason=plan.skip_reason,
        )

    overall_bar.set_postfix_str(_short_name(plan.name, 28), refresh=True)

    if dry_run:
        for job in plan.files:
            file_bar.reset(total=max(job.size, 1))
            file_bar.set_description_str(f"· {_short_name(job.path.name, 42)}")
            file_bar.n = job.size or 1
            file_bar.refresh()
            if progress_mode == "bytes":
                overall_bar.update(job.size)
            elif progress_mode == "files":
                overall_bar.update(1)
            time.sleep(0)  # let UI refresh

        if progress_mode == "projects":
            overall_bar.update(1)

        return ProjectResult(
            project_id=plan.project_id,
            name=plan.name,
            status="dry_run",
            reason=f"{len(plan.files)} files · {_human_size(plan.total_bytes)}",
            bytes_uploaded=plan.total_bytes,
            files_uploaded=len(plan.files),
            duration_sec=time.monotonic() - t0,
        )

    tqdm.write(
        f"→ [{len(plan.files)} files · {_human_size(plan.total_bytes)}] "
        f"{plan.project_id}  {_short_name(plan.name, 40)}"
    )

    try:
        storage_meta = archive_with_progress(
            store,
            plan,
            overall_bar=overall_bar,
            file_bar=file_bar,
            progress_mode=progress_mode,
        )
    except Exception as exc:
        logger.error("[%s] upload failed: %s", plan.project_id, exc, exc_info=True)
        if progress_mode == "projects":
            overall_bar.update(1)
        return ProjectResult(
            project_id=plan.project_id,
            name=plan.name,
            status="error",
            error=str(exc),
            duration_sec=time.monotonic() - t0,
        )

    session_factory = get_session_factory()
    db = session_factory()
    try:
        row = db.query(Project).filter(Project.id == plan.project_id).first()
        if not row:
            if progress_mode == "projects":
                overall_bar.update(1)
            return ProjectResult(
                project_id=plan.project_id,
                name=plan.name,
                status="error",
                error="project missing from DB after upload",
                duration_sec=time.monotonic() - t0,
            )

        config = dict(row.config or {})
        config["storage"] = storage_meta
        row.config = config
        flag_modified(row, "config")
        # Always persist storage first — cleanup permission errors must not roll it back
        db.commit()

        cleanup_note = ""
        if not skip_cleanup:
            file_bar.reset(total=1)
            file_bar.set_description_str("cleanup local")
            file_bar.n = 0
            file_bar.refresh()
            stats = store.cleanup_local_after_archive(
                plan.project_id,
                keep_debug_artifacts=keep_debug if keep_debug is not None else None,
            )
            file_bar.update(1)
            try:
                row.total_size = compute_project_size(ProjectManager(plan.project_id))
                db.commit()
            except Exception:
                pass
            if stats.get("errors"):
                cleanup_note = f"  · cleanup partial ({len(stats['errors'])} errors)"
                for err in stats["errors"][:5]:
                    logger.warning("[%s] cleanup: %s", plan.project_id, err)
            else:
                cleanup_note = "  · cleaned local"
    except Exception as exc:
        db.rollback()
        logger.error("[%s] DB update failed: %s", plan.project_id, exc)
        if progress_mode == "projects":
            overall_bar.update(1)
        return ProjectResult(
            project_id=plan.project_id,
            name=plan.name,
            status="error",
            error=f"DB: {exc}",
            bytes_uploaded=plan.total_bytes,
            files_uploaded=len(plan.files),
            duration_sec=time.monotonic() - t0,
        )
    finally:
        db.close()

    if progress_mode == "projects":
        overall_bar.update(1)

    elapsed = time.monotonic() - t0
    speed = plan.total_bytes / elapsed if elapsed > 0 else 0
    tqdm.write(
        f"✓ {plan.project_id}  {_human_size(plan.total_bytes)} in {_human_duration(elapsed)}"
        f"  (~{_human_size(speed)}/s)"
        f"{cleanup_note if not skip_cleanup else ''}"
    )
    return ProjectResult(
        project_id=plan.project_id,
        name=plan.name,
        status="ok",
        bytes_uploaded=plan.total_bytes,
        files_uploaded=len(plan.files),
        duration_sec=elapsed,
        reason=storage_meta.get("resultVideoKey") or storage_meta.get("sourceKey") or "",
    )


def print_final_report(results: list[ProjectResult], wall_sec: float, *, verbose: bool) -> None:
    by: dict[str, list[ProjectResult]] = {"ok": [], "skipped": [], "dry_run": [], "error": []}
    for r in results:
        by.setdefault(r.status, []).append(r)

    uploaded = sum(r.bytes_uploaded for r in by["ok"] + by["dry_run"])
    files = sum(r.files_uploaded for r in by["ok"] + by["dry_run"])
    ok_bytes = sum(r.bytes_uploaded for r in by["ok"])
    ok_time = sum(r.duration_sec for r in by["ok"])
    avg_speed = (ok_bytes / ok_time) if ok_time > 0 else 0

    tqdm.write("")
    tqdm.write("=" * 72)
    tqdm.write("  MIGRATION REPORT")
    tqdm.write("=" * 72)
    tqdm.write(f"  Wall time:   {_human_duration(wall_sec)}")
    tqdm.write(
        f"  Results:     ok={len(by['ok'])}  skipped={len(by['skipped'])}  "
        f"dry_run={len(by['dry_run'])}  errors={len(by['error'])}"
    )
    tqdm.write(f"  Files:       {files}")
    tqdm.write(f"  Data:        {_human_size(uploaded)}")
    if by["ok"]:
        tqdm.write(f"  Avg upload:  {_human_size(avg_speed)}/s")
    tqdm.write("-" * 72)

    if by["error"]:
        tqdm.write("  FAILED:")
        for r in by["error"]:
            tqdm.write(f"    ✗ {r.project_id}  {_short_name(r.name, 28)}  {r.error}")
        tqdm.write("-" * 72)

    show = by["ok"] or by["dry_run"]
    if show:
        label = "COMPLETED" if by["ok"] else "WOULD MIGRATE"
        tqdm.write(f"  {label}:")
        for r in sorted(show, key=lambda x: x.bytes_uploaded, reverse=True):
            tqdm.write(
                f"    {'✓' if r.status == 'ok' else '·'} {r.project_id}  "
                f"{_human_size(r.bytes_uploaded):>9}  "
                f"{_human_duration(r.duration_sec):>8}  "
                f"{r.files_uploaded:>3}f  {_short_name(r.name, 28)}"
            )

    if by["skipped"]:
        if verbose or len(by["skipped"]) <= 25:
            tqdm.write("  SKIPPED:")
            for r in by["skipped"]:
                tqdm.write(f"    · {r.project_id}  {r.reason}")
        else:
            tqdm.write(f"  SKIPPED: {len(by['skipped'])} (pass -v to list)")

    tqdm.write("=" * 72)


def _local_tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    try:
        if path.is_file():
            return path.stat().st_size
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _project_local_footprint(project_id: str) -> dict[str, int]:
    pm = ProjectManager(project_id)
    return {
        "uploads": _local_tree_size(pm.uploads_dir),
        "results": _local_tree_size(pm.results_dir),
        "artifacts": _local_tree_size(pm.artifacts_dir),
        "cache": _local_tree_size(pm.cache_dir),
    }


def _has_storage(project: Project) -> bool:
    storage = (project.config or {}).get("storage") or {}
    return bool(storage.get("sourceKey") or storage.get("resultVideoKey") or storage.get("reportKey"))


def _s3_has_project_objects(store, project_id: str) -> bool:
    """True if any S3 object exists under the project prefix."""
    if not store.is_enabled():
        return False
    try:
        prefix = store.project_prefix(project_id)
        listing = store.client.list_objects_v2(Bucket=store.bucket, Prefix=prefix, MaxKeys=1)
        return bool(listing.get("Contents"))
    except Exception as exc:
        logger.warning("S3 list failed for %s: %s", project_id, exc)
        return False


def run_cleanup_only(args, store, projects: list[Project]) -> int:
    """Clean local media for projects already archived to S3."""
    wall0 = time.monotonic()
    force_local = args.cleanup_all_local
    results: list[ProjectResult] = []
    candidates: list[tuple[Project, dict[str, int], int, str]] = []

    with logging_redirect_tqdm():
        with tqdm(
            total=len(projects),
            desc="Scanning dirty local",
            unit="proj",
            dynamic_ncols=True,
        ) as bar:
            for project in projects:
                footprint = _project_local_footprint(project.id)
                total = sum(footprint.values())
                bar.set_postfix_str(_short_name(project.name or project.id, 24), refresh=False)
                bar.update(1)

                if total <= 0:
                    results.append(
                        ProjectResult(
                            project_id=project.id,
                            name=project.name or project.id,
                            status="skipped",
                            reason="already clean (no local media)",
                        )
                    )
                    continue

                if force_local:
                    candidates.append((project, footprint, total, "forced"))
                    continue

                if _has_storage(project):
                    candidates.append((project, footprint, total, "config.storage"))
                    continue

                # Upload may have succeeded earlier while DB commit of storage failed
                if store.is_enabled() and _s3_has_project_objects(store, project.id):
                    candidates.append((project, footprint, total, "s3-prefix"))
                    continue

                results.append(
                    ProjectResult(
                        project_id=project.id,
                        name=project.name or project.id,
                        status="skipped",
                        reason=(
                            f"no S3 proof+storage, local still {_human_size(total)} "
                            f"(use --cleanup-all-local to force, or re-run migrate)"
                        ),
                    )
                )

        total_bytes = sum(t for _, _, t, _ in candidates)
        tqdm.write("")
        tqdm.write("=" * 72)
        tqdm.write(f"  Mode:     CLEANUP-ONLY{' (DRY-RUN)' if args.dry_run else ''}")
        tqdm.write(f"  Targets:  {len(candidates)} projects with leftover local media")
        tqdm.write(f"  Volume:   {_human_size(total_bytes)}")
        if force_local:
            tqdm.write("  Policy:   --cleanup-all-local (no S3 safety check)")
        else:
            tqdm.write("  Policy:   config.storage OR existing objects under S3 project prefix")
        tqdm.write("=" * 72)
        for project, footprint, total, reason in sorted(candidates, key=lambda x: x[2], reverse=True)[:15]:
            tqdm.write(
                f"    {_human_size(total):>9}  {project.id}  [{reason}]  "
                f"up={_human_size(footprint['uploads'])} "
                f"res={_human_size(footprint['results'])} "
                f"art={_human_size(footprint['artifacts'])} "
                f"cache={_human_size(footprint['cache'])}"
            )
        if len(candidates) > 15:
            tqdm.write(f"    … +{len(candidates) - 15} more")
        tqdm.write("")

        if not candidates:
            print_final_report(results, time.monotonic() - wall0, verbose=args.verbose)
            return 0

        with tqdm(
            total=len(candidates),
            desc="Cleanup",
            unit="proj",
            dynamic_ncols=True,
        ) as bar:
            for project, footprint, total, why in candidates:
                t0 = time.monotonic()
                bar.set_postfix_str(_short_name(project.name or project.id, 28), refresh=True)

                if args.dry_run:
                    results.append(
                        ProjectResult(
                            project_id=project.id,
                            name=project.name or project.id,
                            status="dry_run",
                            reason=(
                                f"[{why}] would free ~{_human_size(total)} "
                                f"(up={_human_size(footprint['uploads'])} "
                                f"res={_human_size(footprint['results'])} "
                                f"art={_human_size(footprint['artifacts'])} "
                                f"cache={_human_size(footprint['cache'])})"
                            ),
                            bytes_uploaded=total,
                            duration_sec=time.monotonic() - t0,
                        )
                    )
                    bar.update(1)
                    continue

                if args.verify_s3 and store.is_enabled() and why != "forced":
                    storage = (project.config or {}).get("storage") or {}
                    keys = [k for k in (storage.get("sourceKey"), storage.get("resultVideoKey")) if k]
                    if not keys:
                        # Fall back to non-empty prefix
                        if not _s3_has_project_objects(store, project.id):
                            results.append(
                                ProjectResult(
                                    project_id=project.id,
                                    name=project.name or project.id,
                                    status="error",
                                    error="--verify-s3: nothing found on S3",
                                )
                            )
                            bar.update(1)
                            continue
                    else:
                        missing = []
                        for key in keys:
                            try:
                                if not store.object_exists(key):
                                    missing.append(key)
                            except Exception as exc:
                                missing.append(f"{key} ({exc})")
                        if missing:
                            results.append(
                                ProjectResult(
                                    project_id=project.id,
                                    name=project.name or project.id,
                                    status="error",
                                    error=f"S3 missing keys, refuse cleanup: {missing}",
                                )
                            )
                            bar.update(1)
                            continue

                stats = store.cleanup_local_after_archive(project.id, keep_debug_artifacts=None)
                remaining = sum(_project_local_footprint(project.id).values())

                db = get_session_factory()()
                try:
                    row = db.query(Project).filter(Project.id == project.id).first()
                    if row:
                        try:
                            row.total_size = compute_project_size(ProjectManager(project.id))
                            db.commit()
                        except Exception:
                            db.rollback()
                finally:
                    db.close()

                elapsed = time.monotonic() - t0
                err_n = len(stats.get("errors") or [])
                freed = stats.get("freed_bytes", 0)
                if remaining > 1024 * 1024:  # >1MB leftover
                    msg = f"freed~{_human_size(freed)} remain~{_human_size(remaining)} errors={err_n}"
                    results.append(
                        ProjectResult(
                            project_id=project.id,
                            name=project.name or project.id,
                            status="error",
                            error=msg,
                            bytes_uploaded=freed,
                            duration_sec=elapsed,
                        )
                    )
                    tqdm.write(f"⚠ {project.id}  {msg}  (try sudo if Permission denied)")
                    if args.verbose:
                        for err in (stats.get("errors") or [])[:10]:
                            tqdm.write(f"    {err}")
                else:
                    results.append(
                        ProjectResult(
                            project_id=project.id,
                            name=project.name or project.id,
                            status="ok",
                            reason=f"[{why}] freed {_human_size(freed)}",
                            bytes_uploaded=freed,
                            duration_sec=elapsed,
                        )
                    )
                    tqdm.write(
                        f"✓ {project.id}  freed {_human_size(freed)} in {_human_duration(elapsed)}"
                    )
                bar.update(1)

        print_final_report(results, time.monotonic() - wall0, verbose=args.verbose)

    return 1 if any(r.status == "error" for r in results) else 0


def run(args) -> int:
    settings = get_settings()
    store = get_object_storage()

    # cleanup-only can run without S3 when not verifying, but keep_debug still from settings
    if not args.cleanup_only:
        if not args.dry_run and not store.is_enabled():
            logger.error(
                "S3 is not configured. Set S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET in .env"
            )
            return 1

        if not args.dry_run:
            try:
                import boto3  # noqa: F401
            except ImportError:
                logger.error(
                    "boto3 is not installed in this Python environment.\n"
                    "  %s -m pip install boto3\n"
                    "or: %s -m pip install boto3==1.38.1",
                    sys.executable,
                    sys.executable,
                )
                return 1
            try:
                _ = store.client
            except Exception as exc:
                logger.error("Failed to create S3 client: %s", exc)
                return 1
    elif args.verify_s3 and not args.dry_run:
        if not store.is_enabled():
            logger.error("--verify-s3 requires S3 credentials")
            return 1
        try:
            _ = store.client
        except Exception as exc:
            logger.error("Failed to create S3 client: %s", exc)
            return 1

    tqdm.write(
        f"S3: bucket={settings.s3_bucket}  endpoint={settings.s3_endpoint_url}\n"
        f"    prefix={settings.s3_prefix}  keep_debug={settings.keep_debug_artifacts}  "
        f"enabled={store.is_enabled()}"
    )

    init_db()
    db = get_session_factory()()
    try:
        q = db.query(Project).order_by(Project.created_at.asc())
        if args.projects:
            q = q.filter(Project.id.in_(args.projects))
        if args.only_dubbed:
            q = q.filter(Project.status == ProjectStatus.DUBBED)
        if args.include_status:
            statuses = []
            for name in args.include_status:
                try:
                    statuses.append(ProjectStatus(name.lower()))
                except ValueError:
                    logger.error("Unknown status: %s", name)
                    return 1
            q = q.filter(Project.status.in_(statuses))
        projects = q.all()
    finally:
        db.close()

    if not projects:
        tqdm.write("No projects matched")
        return 0

    if args.cleanup_only:
        return run_cleanup_only(args, store, projects)

    wall0 = time.monotonic()
    progress_mode = args.unit

    with logging_redirect_tqdm():
        plans = build_plans(projects, store, force=args.force)
        active = print_plan_summary(plans, dry_run=args.dry_run)

        results: list[ProjectResult] = [
            ProjectResult(
                project_id=p.project_id,
                name=p.name,
                status="skipped",
                reason=p.skip_reason or "",
            )
            for p in plans
            if p.skip_reason
        ]

        if active:
            if progress_mode == "bytes":
                overall_kw = dict(
                    total=sum(p.total_bytes for p in active) or 1,
                    desc="Overall",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )
            elif progress_mode == "files":
                overall_kw = dict(
                    total=sum(len(p.files) for p in active) or 1,
                    desc="Overall",
                    unit="file",
                )
            else:
                overall_kw = dict(total=len(active), desc="Overall", unit="proj")

            with tqdm(
                **overall_kw,
                dynamic_ncols=True,
                position=0,
                leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            ) as overall_bar, tqdm(
                total=1,
                desc="file",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                position=1,
                leave=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
            ) as file_bar:
                for idx, plan in enumerate(active, 1):
                    overall_bar.set_description(f"Overall {idx}/{len(active)}")
                    res = migrate_plan(
                        plan,
                        store,
                        dry_run=args.dry_run,
                        skip_cleanup=args.skip_cleanup,
                        keep_debug=None,
                        overall_bar=overall_bar,
                        file_bar=file_bar,
                        progress_mode=progress_mode,
                    )
                    results.append(res)
                    if res.status == "error":
                        tqdm.write(f"✗ {plan.project_id}: {res.error}")

        print_final_report(results, time.monotonic() - wall0, verbose=args.verbose)

    return 1 if any(r.status == "error" for r in results) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate existing DubbLM project media to S3")
    parser.add_argument("--project", "-p", action="append", dest="projects", help="Project id (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="Plan only, no upload/delete")
    parser.add_argument("--force", action="store_true", help="Re-upload even if config.storage set")
    parser.add_argument("--skip-cleanup", action="store_true", help="Upload only; keep local files")
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Do not upload; only delete local leftovers for projects already on S3 (config.storage)",
    )
    parser.add_argument(
        "--cleanup-all-local",
        action="store_true",
        help="With --cleanup-only: wipe local media even without config.storage (dangerous)",
    )
    parser.add_argument(
        "--verify-s3",
        action="store_true",
        help="With --cleanup-only: refuse cleanup if source/result key is missing on S3",
    )
    parser.add_argument("--only-dubbed", action="store_true", help="Only status=dubbed")
    parser.add_argument(
        "--include-status",
        nargs="*",
        default=None,
        help="Limit to statuses (e.g. dubbed transcribed)",
    )
    parser.add_argument(
        "--unit",
        choices=("bytes", "files", "projects"),
        default="bytes",
        help="Overall progress unit (default: bytes)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true", help="Progress bars + report only")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

