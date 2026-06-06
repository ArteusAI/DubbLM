"""Summary report builder for the dubbing pipeline.

Produces two artefacts at the end of a run:
- ``artifacts/report.json`` — structured data for programmatic consumption.
- ``artifacts/report.md``   — human-readable Markdown (GFM tables) used by
  the frontend ``ResultView`` and downloadable as-is.

The builder aggregates:
- Stage timings from :class:`PerformanceTracker`.
- Monetary costs from :class:`CostTracker`.
- Per-segment TTS telemetry drained from every initialised TTS wrapper.
- An artifact scan over ``artifacts/`` and ``results/`` with URLs that point
  at the generic ``/projects/{id}/artifacts/{path}`` endpoint.
"""

from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.log_config import get_logger
from ...tts.models import BatchSynthesisReport, SegmentSynthesisReport

logger = get_logger(__name__)


STAGE_LABELS: Dict[str, str] = {
    "extract_audio": "Extract audio",
    "diarization": "Diarization",
    "speaker_audio": "Speaker audio",
    "transcription": "Transcription",
    "segment_normalization": "Segment normalization",
    "context_analysis": "Context analysis",
    "translation": "Translation",
    "emotion_analysis": "Emotion analysis",
    "speech_synthesis": "Speech synthesis",
    "background_audio": "Background audio",
    "audio_normalization": "Audio normalization",
    "video_creation": "Video creation",
}


class ReportBuilder:
    """Build and serialise the per-run summary report."""

    REPORT_JSON_PATH = Path("artifacts") / "report.json"
    REPORT_MD_PATH = Path("artifacts") / "report.md"
    ARTIFACT_DIRS = (Path("artifacts"), Path("results"))
    SKIP_DIR_NAMES = {"audio_chunks", "su_audio_chunks", "speakers_audio", "chunks", "cost_snapshots"}
    SKIP_FILES = {"report.json", "report.md"}

    def __init__(
        self,
        performance_tracker,
        cost_tracker,
        tts_systems: Dict[str, Any],
        segments_metadata: List[Dict[str, Any]],
        config: Dict[str, Any],
        project_id: Optional[str] = None,
        api_prefix: str = "/api/v1",
    ) -> None:
        self.performance_tracker = performance_tracker
        self.cost_tracker = cost_tracker
        self.tts_systems = tts_systems or {}
        self.segments_metadata = segments_metadata or []
        self.config = config or {}
        self.project_id = project_id
        self.api_prefix = api_prefix.rstrip("/")

    def build_and_write(self) -> Optional[Path]:
        """Build the report and persist both JSON + Markdown. Returns md path."""
        try:
            payload = self._build_payload()
        except Exception as exc:
            logger.warning(f"ReportBuilder: failed to assemble payload: {exc}", exc_info=True)
            return None

        try:
            self.REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self.REPORT_JSON_PATH.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)
        except Exception as exc:
            logger.warning(f"ReportBuilder: failed to write JSON report: {exc}")

        md_text = self._render_markdown(payload)
        try:
            self.REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.REPORT_MD_PATH.write_text(md_text, encoding="utf-8")
            logger.info(f"Summary report written to {self.REPORT_MD_PATH}")
            return self.REPORT_MD_PATH
        except Exception as exc:
            logger.warning(f"ReportBuilder: failed to write Markdown report: {exc}")
            return None

    def _build_payload(self) -> Dict[str, Any]:
        stages = self._collect_stages()
        segments = self._collect_segments()
        batches = self._collect_batches()
        artifacts = self._scan_artifacts()
        transcription_usage = self._collect_transcription_usage()
        api_costs = self._collect_api_costs()

        total_time = float(self.performance_tracker.metrics.get("total", 0.0))
        api_total = sum(float(row.get("cost_usd") or 0.0) for row in api_costs)
        total_cost = api_total if api_costs else float(self.performance_tracker.costs.get("total", 0.0))

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "project_id": self.project_id,
            "project": {
                "source_language": self.config.get("source_language"),
                "target_language": self.config.get("target_language"),
                "tts_system": self.config.get("tts_system"),
                "tts_model": self.config.get("tts_model"),
                "tts_fallback_model": self.config.get("tts_fallback_model"),
                "transcription_system": self.config.get("transcription_system"),
                "transcription_model": self.config.get("whisper_model"),
                "input": self.config.get("input"),
                "output": self.config.get("output"),
            },
            "totals": {
                "time_seconds": total_time,
                "cost_usd": total_cost,
                "segments": len(segments),
                "failed_segments": sum(1 for s in segments if not s.get("success", True)),
                "batches": len(batches),
                "batches_with_fallback": sum(
                    1 for b in batches if not b.get("success", True) or b.get("fallback_segment_count", 0) > 0
                ),
            },
            "stages": stages,
            "api_costs": api_costs,
            "transcription_usage": transcription_usage,
            "segments": segments,
            "batches": batches,
            "artifacts": artifacts,
        }

    def _collect_stages(self) -> List[Dict[str, Any]]:
        metrics = self.performance_tracker.metrics
        costs = dict(getattr(self.performance_tracker, "costs", {}) or {})
        tracker_costs = {}
        get_costs = getattr(self.cost_tracker, "get_costs_by_step", None)
        if callable(get_costs):
            try:
                tracker_costs = get_costs() or {}
            except Exception as exc:
                logger.debug(f"ReportBuilder: failed to collect tracker costs: {exc}")
        for key, value in tracker_costs.items():
            if key != "total":
                costs[key] = float(value or 0.0)
        mem_peak = getattr(self.performance_tracker, "memory_peak_mb", {}) or {}
        mem_delta = getattr(self.performance_tracker, "memory_delta_mb", {}) or {}
        errors = getattr(self.performance_tracker, "step_errors", {}) or {}
        order = [k for k in STAGE_LABELS.keys() if k in metrics or k in costs]
        stages: List[Dict[str, Any]] = []
        for key in order:
            time_sec = float(metrics.get(key, 0.0))
            cost_usd = float(costs.get(key, 0.0))
            error = errors.get(key)
            if time_sec <= 0 and cost_usd <= 0 and not error:
                continue
            stages.append({
                "key": key,
                "label": STAGE_LABELS.get(key, key),
                "time_seconds": time_sec,
                "cost_usd": cost_usd,
                "memory_peak_mb": float(mem_peak.get(key, 0.0)),
                "memory_delta_mb": float(mem_delta.get(key, 0.0)),
                "error": error,
            })
        return stages

    def _collect_api_costs(self) -> List[Dict[str, Any]]:
        get_rows = getattr(self.cost_tracker, "get_api_cost_rows", None)
        if not callable(get_rows):
            return []
        try:
            rows = list(get_rows() or [])
        except Exception as exc:
            logger.debug(f"ReportBuilder: failed to collect API costs: {exc}")
            return []

        out: List[Dict[str, Any]] = []
        for row in rows:
            stage_key = str(row.get("stage_key") or "")
            category = str(row.get("category") or "")
            out.append({
                "stage_key": stage_key,
                "stage_label": STAGE_LABELS.get(stage_key, stage_key.replace("_", " ").title()),
                "category": category,
                "category_label": row.get("category_label") or category.replace("_", " ").title(),
                "provider": row.get("provider") or "unknown",
                "model": row.get("model") or "default",
                "usage": row.get("usage") or {},
                "rates": row.get("rates") or {},
                "pricing_source": row.get("pricing_source") or (row.get("rates") or {}).get("pricing_source") or "unknown",
                "cost_usd": float(row.get("cost_usd") or 0.0),
            })
        out.sort(key=lambda r: (r["stage_key"], r["category"], r["provider"], r["model"]))
        return out

    def _collect_segments(self) -> List[Dict[str, Any]]:
        reports: List[SegmentSynthesisReport] = []
        for tts in self.tts_systems.values():
            drain = getattr(tts, "drain_segment_reports", None)
            if callable(drain):
                try:
                    reports.extend(drain() or [])
                except Exception as exc:
                    logger.debug(f"ReportBuilder: drain failed for {tts}: {exc}")

        # Deduplicate: per segment_index keep the last successful attempt if any,
        # otherwise the last record. Wrappers may emit cache-hit + synthesis for
        # the same index across prepass/main; the final one represents the state.
        by_index: Dict[int, SegmentSynthesisReport] = {}
        for rep in reports:
            existing = by_index.get(rep.segment_index)
            if existing is None:
                by_index[rep.segment_index] = rep
                continue
            # Prefer success over failure, otherwise prefer latest.
            if rep.success and not existing.success:
                by_index[rep.segment_index] = rep
            elif rep.success == existing.success:
                by_index[rep.segment_index] = rep

        # Merge with segments_metadata so ungrouped/skipped segments still show up.
        metadata_by_index: Dict[int, Dict[str, Any]] = {}
        for meta in self.segments_metadata:
            idx = meta.get("index")
            if idx is not None:
                metadata_by_index[idx] = meta

        merged_indices = sorted(set(by_index.keys()) | set(metadata_by_index.keys()))
        out: List[Dict[str, Any]] = []
        for idx in merged_indices:
            rep = by_index.get(idx)
            meta = metadata_by_index.get(idx, {})
            seg_dict = meta.get("segment_dict") or {}

            text = (
                (rep.text if rep else None)
                or meta.get("chosen_text")
                or seg_dict.get("translation")
                or ""
            )
            group_id = (
                (rep.group_id if rep else None)
                or seg_dict.get("group_id")
            )
            entry = {
                "segment_index": idx,
                "speaker": (rep.speaker if rep else None) or seg_dict.get("speaker"),
                "group_id": group_id,
                "text": text,
                "text_preview": _shorten(text, 80),
                "requested_model": rep.requested_model if rep else None,
                "actual_model": (rep.actual_model if rep else None) or meta.get("tts_system"),
                "attempts": rep.attempts if rep else 0,
                "used_fallback": bool(rep.used_fallback) if rep else False,
                "success": bool(rep.success) if rep else True,
                "duration_seconds": float(rep.duration_seconds) if rep else 0.0,
                "output_path": (rep.output_path if rep else None) or meta.get("output_path"),
                "error": rep.error if rep else None,
            }
            out.append(entry)
        return out

    def _collect_batches(self) -> List[Dict[str, Any]]:
        batch_reports: List[BatchSynthesisReport] = []
        for tts in self.tts_systems.values():
            drain = getattr(tts, "drain_batch_reports", None)
            if not callable(drain):
                continue
            try:
                batch_reports.extend(drain() or [])
            except Exception as exc:
                logger.debug(f"ReportBuilder: batch drain failed for {tts}: {exc}")

        out: List[Dict[str, Any]] = []
        for rep in batch_reports:
            seg_indices = list(rep.segment_indices or [])
            if seg_indices:
                first = seg_indices[0] + 1
                last = seg_indices[-1] + 1
                segments_label = f"#{first}" if first == last else f"#{first}..#{last}"
            else:
                segments_label = "—"
            out.append({
                "batch_index": rep.batch_index,
                "mode": rep.mode,
                "segment_indices": seg_indices,
                "segments_label": segments_label,
                "segment_count": len(seg_indices),
                "speakers": list(rep.speakers or []),
                "attempts": int(rep.attempts or 0),
                "success": bool(rep.success),
                "fallback_segment_count": int(rep.fallback_segment_count or 0),
                "duration_seconds": float(rep.duration_seconds or 0.0),
                "reason": rep.reason or "",
            })
        out.sort(key=lambda b: b["batch_index"])
        return out

    def _collect_transcription_usage(self) -> List[Dict[str, Any]]:
        usage = getattr(self.cost_tracker, "usage", {}) or {}
        details = ((usage.get("transcription") or {}).get("details") or {})
        out: List[Dict[str, Any]] = []
        for key, raw in details.items():
            categories = raw.get("categories") or {}
            audio_seconds = float(raw.get("audio_seconds") or 0.0)
            if audio_seconds <= 0:
                continue
            out.append({
                "key": key,
                "provider": raw.get("provider") or key.split(":", 1)[0],
                "model": raw.get("model") or key.split(":", 1)[-1],
                "audio_seconds": audio_seconds,
                "audio_hours": audio_seconds / 3600.0,
                "cost_usd": float(raw.get("cost_usd") or 0.0),
                "categories": {
                    category: {
                        "audio_seconds": float(seconds or 0.0),
                        "audio_hours": float(seconds or 0.0) / 3600.0,
                    }
                    for category, seconds in sorted(categories.items())
                    if float(seconds or 0.0) > 0
                },
            })
        out.sort(key=lambda row: row["audio_seconds"], reverse=True)
        return out

    def _scan_artifacts(self) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        for root in self.ARTIFACT_DIRS:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if path.name in self.SKIP_FILES:
                    continue
                if any(part in self.SKIP_DIR_NAMES for part in path.parts):
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                rel = path.as_posix()
                entry = {
                    "name": path.name,
                    "path": rel,
                    "size_bytes": size,
                    "media_type": mimetypes.guess_type(path.name)[0],
                }
                if self.project_id:
                    entry["url"] = (
                        f"{self.api_prefix}/projects/{self.project_id}/artifacts/{rel}"
                    )
                found.append(entry)
        return found

    def _render_markdown(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        project = payload.get("project", {}) or {}
        totals = payload.get("totals", {}) or {}
        name = self.config.get("project_name") or self.project_id or "run"

        lines.append(f"# Dubbing Report — {name}")
        src = project.get("source_language") or "?"
        tgt = project.get("target_language") or "?"
        lines.append(f"**Source:** `{src}` → **Target:** `{tgt}`  ")
        lines.append(
            f"**Total time:** {_fmt_duration(totals.get('time_seconds', 0))}  "
            f"**Total cost:** ${float(totals.get('cost_usd', 0)):.4f}  "
            f"**Segments:** {totals.get('segments', 0)} "
            f"(failed: {totals.get('failed_segments', 0)})"
        )
        lines.append("")

        lines.append("## API Costs")
        lines.append("")
        api_costs = payload.get("api_costs") or []
        if api_costs:
            lines.append("| Stage | Category | Provider | Model | Usage | Rate | Cost |")
            lines.append("|---|---|---|---|---:|---|---:|")
            for row in api_costs:
                lines.append(
                    f"| {_escape_cell(str(row.get('stage_label') or '—'))} "
                    f"| {_escape_cell(str(row.get('category_label') or '—'))} "
                    f"| {_escape_cell(str(row.get('provider') or '—'))} "
                    f"| {_escape_cell(str(row.get('model') or '—'))} "
                    f"| {_escape_cell(_format_api_usage(row.get('usage') or {}))} "
                    f"| {_escape_cell(_format_api_rate(row.get('rates') or {}, row.get('pricing_source')))} "
                    f"| ${float(row.get('cost_usd') or 0.0):.4f} |"
                )
            total_api_cost = sum(float(row.get("cost_usd") or 0.0) for row in api_costs)
            lines.append(
                f"| **Total** |  |  |  |  |  | **${total_api_cost:.4f}** |"
            )
        else:
            lines.append("_No paid API usage recorded._")
        lines.append("")

        lines.append("## Stages")
        lines.append("")
        stages = payload.get("stages") or []
        if stages:
            show_mem = any((st.get("memory_peak_mb") or 0) > 0 for st in stages)
            header_cells = ["Stage", "Time", "Cost"]
            align_cells = ["---", "---:", "---:"]
            if show_mem:
                header_cells.append("Peak RSS / Δ")
                align_cells.append("---:")
            header_cells.append("Notes")
            align_cells.append("---")
            lines.append("| " + " | ".join(header_cells) + " |")
            lines.append("|" + "|".join(align_cells) + "|")
            for st in stages:
                row = [
                    st["label"],
                    _fmt_duration(st["time_seconds"]),
                    f"${st['cost_usd']:.4f}",
                ]
                if show_mem:
                    peak = float(st.get("memory_peak_mb") or 0.0)
                    delta = float(st.get("memory_delta_mb") or 0.0)
                    row.append(
                        f"{peak:.0f} MB / +{delta:.0f} MB" if peak > 0 else "—"
                    )
                err = st.get("error")
                row.append(f"❌ {_escape_cell(err)}" if err else "")
                lines.append("| " + " | ".join(row) + " |")
        else:
            lines.append("_No stage timings recorded._")
        lines.append("")

        lines.append("## Transcription Usage")
        lines.append("")
        transcription_usage = payload.get("transcription_usage") or []
        if transcription_usage:
            total_transcription_hours = sum(float(row.get("audio_hours", 0.0)) for row in transcription_usage)
            category_order = [
                "primary_transcription",
                "tts_batch_alignment",
                "tts_content_validation",
            ]
            category_totals = {
                "primary_transcription": sum(
                    _category_hours(row.get("categories") or {}, "primary_transcription")
                    for row in transcription_usage
                ),
                "tts_batch_alignment": sum(
                    _category_hours(row.get("categories") or {}, "tts_batch_alignment")
                    for row in transcription_usage
                ),
                "tts_content_validation": sum(
                    _category_hours(row.get("categories") or {}, "tts_content_validation")
                    for row in transcription_usage
                ),
            }
            lines.append(f"**Total ASR audio processed:** {total_transcription_hours:.2f} h")
            lines.append(
                "**By stage:** "
                + ", ".join(
                    f"{_humanize_transcription_category(category)} {_fmt_hours(category_totals[category])}"
                    for category in category_order
                )
            )
            lines.append("")
            lines.append("| Provider | Model | Primary | Batch Alignment | TTS Validation | Total | Cost |")
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            for row in transcription_usage:
                categories = row.get("categories") or {}
                lines.append(
                    f"| {_escape_cell(str(row.get('provider') or '—'))} "
                    f"| {_escape_cell(str(row.get('model') or '—'))} "
                    f"| {_fmt_hours((categories.get('primary_transcription') or {}).get('audio_hours', 0.0))} "
                    f"| {_fmt_hours((categories.get('tts_batch_alignment') or {}).get('audio_hours', 0.0))} "
                    f"| {_fmt_hours((categories.get('tts_content_validation') or {}).get('audio_hours', 0.0))} "
                    f"| {_fmt_hours(row.get('audio_hours', 0.0))} "
                    f"| ${float(row.get('cost_usd', 0.0)):.4f} |"
                )
        else:
            lines.append("_No transcription usage recorded._")
        lines.append("")

        lines.append("## Artifacts")
        lines.append("")
        artifacts = payload.get("artifacts") or []
        if artifacts:
            for art in artifacts:
                url = art.get("url") or art["path"]
                size_kb = max(1, art["size_bytes"] // 1024)
                lines.append(f"- [{art['path']}]({url}) — {size_kb} KB")
        else:
            lines.append("_No artifacts found._")
        lines.append("")

        lines.append("## TTS Batches")
        lines.append("")
        batches = payload.get("batches") or []
        if batches:
            totals = payload.get("totals") or {}
            ms_batches = [b for b in batches if b["mode"] == "multi_speaker"]
            ms_clean = sum(
                1 for b in ms_batches
                if b.get("success") and b.get("fallback_segment_count", 0) == 0
            )
            ms_partial = sum(
                1 for b in ms_batches
                if 0 < b.get("fallback_segment_count", 0) < b.get("segment_count", 0)
            )
            ms_failed = sum(
                1 for b in ms_batches
                if not b.get("success")
                and b.get("fallback_segment_count", 0) >= b.get("segment_count", 0)
            )
            ss_batches = sum(1 for b in batches if b["mode"] == "single_speaker")
            solo_batches = sum(1 for b in batches if b["mode"] == "single_segment")
            summary_bits: List[str] = [
                f"**Total batches:** {len(batches)}",
                f"**Multi-speaker:** {len(ms_batches)} "
                f"(clean {ms_clean}, partial fallback {ms_partial}, failed {ms_failed})",
                f"**Single-speaker runs:** {ss_batches}",
                f"**Solo segments:** {solo_batches}",
                f"**Batches with fallback:** {totals.get('batches_with_fallback', 0)}",
            ]
            lines.append("  \n".join(summary_bits))
            lines.append("")
            lines.append("| Batch | Segments | Mode | Speakers | Attempts | Time | Result |")
            lines.append("|---:|---|---|---|---:|---:|---|")
            for b in batches:
                mode_label = {
                    "multi_speaker": "multi-speaker",
                    "single_speaker": "single-speaker",
                    "single_segment": "solo",
                }.get(b["mode"], b["mode"])
                speakers_cell = ", ".join(b.get("speakers") or []) or "—"
                fallback = b.get("fallback_segment_count", 0)
                if b["mode"] != "multi_speaker":
                    result = "—"
                elif b.get("success") and fallback == 0:
                    result = "ok"
                elif fallback and fallback < b.get("segment_count", 0):
                    result = f"partial ↘ {fallback}/{b['segment_count']} fell back"
                else:
                    result = f"failed ❌ ({fallback}/{b.get('segment_count', 0)} fell back)"
                result_cell = _format_batch_result_cell(result, b.get("reason", ""))
                lines.append(
                    f"| {b['batch_index'] + 1} "
                    f"| {b.get('segments_label', '—')} ({b.get('segment_count', 0)}) "
                    f"| {mode_label} "
                    f"| {_escape_cell(speakers_cell)} "
                    f"| {b.get('attempts', 0)} "
                    f"| {_fmt_duration(b.get('duration_seconds', 0))} "
                    f"| {_escape_cell(result_cell)} |"
                )
        else:
            lines.append("_No TTS batch telemetry recorded._")
        lines.append("")

        lines.append("## Segments")
        lines.append("")
        segments = payload.get("segments") or []
        if not segments:
            lines.append("_No segments processed._")
        else:
            grouped: Dict[Optional[str], List[Dict[str, Any]]] = {}
            for seg in segments:
                grouped.setdefault(seg.get("group_id"), []).append(seg)

            # Render groups first (preserving original order), then ungrouped rows.
            group_keys = [k for k in grouped.keys() if k is not None]
            group_keys.sort(key=lambda k: grouped[k][0].get("segment_index", 0))
            for gid in group_keys:
                rows = grouped[gid]
                lines.append(f"### Group `{gid}` ({len(rows)} segments)")
                lines.append("")
                lines.extend(_render_segment_table(rows))
                lines.append("")

            ungrouped = grouped.get(None) or []
            if ungrouped:
                lines.append("### Ungrouped")
                lines.append("")
                lines.extend(_render_segment_table(ungrouped))
                lines.append("")

        lines.append("---")
        lines.append(f"_Generated at {payload.get('generated_at')}_")
        return "\n".join(lines) + "\n"


def _format_api_usage(usage: Dict[str, Any]) -> str:
    parts: List[str] = []
    audio_seconds = float(usage.get("audio_seconds") or 0.0)
    if audio_seconds > 0:
        parts.append(f"{_fmt_hours(audio_seconds / 3600.0)} audio")

    input_tokens = float(usage.get("input_tokens") or 0.0)
    output_tokens = float(usage.get("output_tokens") or 0.0)
    reasoning_tokens = float(usage.get("reasoning_tokens") or 0.0)
    token_parts: List[str] = []
    if input_tokens > 0:
        token_parts.append(f"in {input_tokens:,.0f} tok")
    if output_tokens > 0:
        token_parts.append(f"out {output_tokens:,.0f} tok")
    if reasoning_tokens > 0:
        token_parts.append(f"reasoning {reasoning_tokens:,.0f} tok")
    if token_parts:
        parts.append(", ".join(token_parts))

    input_chars = float(usage.get("input_characters") or 0.0)
    if input_chars > 0:
        parts.append(f"{input_chars:,.0f} chars")

    voice_count = float(usage.get("voice_count") or 0.0)
    if voice_count > 0:
        parts.append(f"{voice_count:,.0f} voice")

    return "; ".join(parts) if parts else "—"


def _format_api_rate(rates: Dict[str, Any], source: Optional[str]) -> str:
    parts: List[str] = []
    input_rate = float(rates.get("input_per_1m_tokens") or 0.0)
    output_rate = float(rates.get("output_per_1m_tokens") or 0.0)
    reasoning_rate = float(rates.get("reasoning_per_1m_tokens") or 0.0)
    if input_rate > 0:
        parts.append(f"${input_rate:.4g}/M input tok")
    if output_rate > 0:
        parts.append(f"${output_rate:.4g}/M output tok")
    if reasoning_rate > 0 and abs(reasoning_rate - output_rate) > 1e-12:
        parts.append(f"${reasoning_rate:.4g}/M reasoning tok")

    per_min = float(rates.get("per_min") or 0.0)
    if per_min > 0:
        parts.append(f"${per_min:.4g}/min")
    diarization_per_min = float(rates.get("speaker_diarization_per_min") or 0.0)
    if diarization_per_min > 0:
        parts.append(f"+${diarization_per_min:.4g}/min diarization")

    per_audio_min = rates.get("per_audio_min")
    if per_audio_min is not None and float(per_audio_min or 0.0) > 0:
        parts.append(f"${float(per_audio_min):.4g}/audio min")

    per_chars = float(rates.get("per_1m_chars") or 0.0)
    if per_chars > 0:
        parts.append(f"${per_chars:.4g}/M chars")

    per_voice = float(rates.get("voice_clone_per_voice") or 0.0)
    if per_voice > 0:
        parts.append(f"${per_voice:.4g}/voice")

    tokens_per_second = float(rates.get("audio_output_tokens_per_second") or 0.0)
    if tokens_per_second > 0:
        parts.append(f"{tokens_per_second:.0f} audio tok/s")

    rate_text = ", ".join(parts) if parts else "unknown"
    source_text = source or rates.get("pricing_source")
    if source_text:
        rate_text = f"{rate_text} ({source_text})"
    return rate_text


def _render_segment_table(rows: List[Dict[str, Any]]) -> List[str]:
    out = ["| # | Text | Model | Attempts | Time |", "|---:|---|---|---:|---:|"]
    for row in rows:
        model = row.get("actual_model") or "—"
        if row.get("used_fallback"):
            model = f"{model} (fallback)"
        success_mark = "" if row.get("success", True) else " ❌"
        time_sec = float(row.get("duration_seconds", 0) or 0)
        text_cell = (row.get("text_preview") or "").replace("|", "\\|").replace("\n", " ")
        out.append(
            f"| {int(row.get('segment_index', -1)) + 1} "
            f"| {text_cell} "
            f"| {model} "
            f"| {int(row.get('attempts', 0))} "
            f"| {time_sec:.2f}s{success_mark} |"
        )
    return out


def _escape_cell(text: Optional[str]) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def _format_batch_result_cell(result: str, reason: Optional[str]) -> str:
    reason_text = (reason or "").strip()
    if not reason_text:
        return result
    if result == "—":
        return reason_text
    return f"{result} — {reason_text}"


def _shorten(text: str, limit: int) -> str:
    cleaned = (text or "").replace("\n", " ").strip()
    if len(cleaned) <= limit:
        return cleaned
    head = cleaned[: max(10, limit // 2 - 2)]
    tail = cleaned[-(limit - len(head) - 3):]
    return f"{head}...{tail}"


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0))
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins, secs = divmod(seconds, 60)
    if mins < 60:
        return f"{int(mins)}m {secs:05.2f}s"
    hours, mins = divmod(mins, 60)
    return f"{int(hours)}h {int(mins)}m {secs:05.2f}s"


def _fmt_hours(hours: float) -> str:
    hours = max(0.0, float(hours or 0.0))
    if hours < 0.01:
        return f"{hours * 60.0:.2f} min"
    return f"{hours:.2f} h"


def _category_hours(categories: Dict[str, Any], *names: str) -> float:
    return sum(
        float((categories.get(name) or {}).get("audio_hours", 0.0))
        for name in names
    )


def _humanize_transcription_category(category: str) -> str:
    mapping = {
        "primary_transcription": "primary",
        "tts_batch_alignment": "batch alignment",
        "tts_content_validation": "tts validation",
        "transcription": "transcription",
    }
    return mapping.get(category, category.replace("_", " "))


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)
