"""Persist API cost rows across split worker stages."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ..core.log_config import get_logger

logger = get_logger(__name__)

SNAPSHOT_VERSION = 1
SNAPSHOT_ORDER = ("transcription", "llm", "dub")
DEFAULT_IMPORT_SNAPSHOTS = ("transcription", "llm")

SNAPSHOT_CATEGORIES = {
    "transcription": {"primary_transcription", "transcription"},
    "llm": {
        "context_analysis",
        "translation",
        "refinement",
        "editor_pass",
        "emotion_enrichment",
    },
    "dub": {
        "tts_synthesis",
        "tts_batch_alignment",
        "tts_content_validation",
        "voice_cloning",
    },
}

SNAPSHOT_STAGES = {
    "transcription": {"transcription"},
    "llm": {"context_analysis", "translation", "emotion_analysis"},
    "dub": {"speech_synthesis"},
}


def filter_api_cost_rows(rows: Iterable[Dict[str, Any]], snapshot_name: str) -> List[Dict[str, Any]]:
    """Return cost rows that belong to one logical worker-stage snapshot."""
    categories = SNAPSHOT_CATEGORIES.get(snapshot_name, set())
    stages = SNAPSHOT_STAGES.get(snapshot_name, set())
    filtered: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        category = str(row.get("category") or "")
        stage_key = str(row.get("stage_key") or "")
        if category in categories or stage_key in stages:
            filtered.append(_copy_jsonable(row))
    return filtered


def load_cost_snapshot_rows(
    debug_dir: Path | str,
    snapshots: Sequence[str] = SNAPSHOT_ORDER,
) -> List[Dict[str, Any]]:
    """Load API cost rows from persisted snapshots in deterministic order."""
    loaded: List[Dict[str, Any]] = []
    for snapshot_name in snapshots:
        path = _snapshot_path(debug_dir, snapshot_name)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load cost snapshot %s: %s", path, exc)
            continue
        rows = payload.get("api_costs") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            continue
        loaded.extend(row for row in rows if isinstance(row, dict))
    return loaded


def import_cost_snapshots(
    debug_dir: Path | str,
    cost_tracker: Any,
    snapshots: Sequence[str] = DEFAULT_IMPORT_SNAPSHOTS,
) -> int:
    """Import persisted cost rows into a live CostTracker."""
    rows = load_cost_snapshot_rows(debug_dir, snapshots=snapshots)
    if not rows:
        return 0
    importer = getattr(cost_tracker, "import_api_cost_rows", None)
    if not callable(importer):
        return 0
    return int(importer(rows) or 0)


def write_cost_snapshot(
    debug_dir: Path | str,
    snapshot_name: str,
    rows: Iterable[Dict[str, Any]],
) -> Path:
    """Overwrite one snapshot with exactly the supplied rows."""
    path = _snapshot_path(debug_dir, snapshot_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_rows = [_copy_jsonable(row) for row in rows or [] if isinstance(row, dict)]
    payload = {
        "version": SNAPSHOT_VERSION,
        "snapshot": snapshot_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cost_usd": sum(float(row.get("cost_usd") or 0.0) for row in clean_rows),
        "api_costs": clean_rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_cost_snapshot_from_tracker(
    debug_dir: Path | str,
    snapshot_name: str,
    cost_tracker: Any,
) -> int:
    """Overwrite one snapshot with matching rows from a live CostTracker."""
    get_rows = getattr(cost_tracker, "get_api_cost_rows", None)
    rows = list(get_rows() or []) if callable(get_rows) else []
    filtered = filter_api_cost_rows(rows, snapshot_name)
    write_cost_snapshot(debug_dir, snapshot_name, filtered)
    return len(filtered)


def _snapshot_path(debug_dir: Path | str, snapshot_name: str) -> Path:
    return Path(debug_dir) / "cost_snapshots" / f"{snapshot_name}.json"


def _copy_jsonable(row: Dict[str, Any]) -> Dict[str, Any]:
    """Make a plain JSON-compatible copy and normalize numeric cost."""
    copied = json.loads(json.dumps(row, ensure_ascii=False, default=str))
    if isinstance(copied, dict):
        copied["cost_usd"] = float(copied.get("cost_usd") or 0.0)
    return copied
