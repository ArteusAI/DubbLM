#!/usr/bin/env python3
"""Build a podcast episode from a video.

This is intentionally independent from the dubbing pipeline: it extracts audio,
runs a fresh AssemblyAI transcription + diarization, cuts the episode by source
timecodes, translates the kept lines with OpenRouter, and synthesizes the result
with Gemini 3.8 TTS.  There is no segment grouping, no timing-driven text
variants and no video time stretching.

Typical usage::

    /home/fpv/.venv/bin/python tools/podcast_from_video.py \
        --video /tmp/opencode/hinton_source.webm \
        --config podcast_configs/hinton_sleep.yaml \
        --output-dir artifacts/podcast_hinton_sleep \
        --dry-run

Then, after reviewing the script::

    ... --audition
    ... --render
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import yaml
from dotenv import load_dotenv
from json_repair import repair_json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

import assemblyai as aai  # noqa: E402


def _load_prompt_builder():
    """Load the dependency-free Gemini 3.8 prompt helper without importing src.tts."""
    path = REPO_ROOT / "src" / "tts" / "gemini38_tags.py"
    spec = importlib.util.spec_from_file_location("_podcast_gemini38_tags", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prompt helper: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules, so the
    # module must be registered before execution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_gemini38_prompt


build_gemini38_prompt = _load_prompt_builder()

LOG = logging.getLogger("podcast")
SAMPLE_RATE = 24000
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "podcast-ru-v2"

FILLER_RE = re.compile(
    r"^(?:yeah|yep|yes|right|okay|ok|sure|mm[- ]?hmm|mhm|uh[- ]?huh|"
    r"i see|got it|exactly|absolutely|of course|thanks|thank you|wow|"
    r"interesting|sounds good|that'?s right|all right|great|good|nice)"
    r"[\s.!?,]*$",
    re.IGNORECASE,
)
LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    id: str
    start: float
    end: float
    speaker: str
    text: str
    source_text: str = ""
    source_index: int = -1
    source_id: str = ""
    section: str = ""
    section_start: bool = False
    inserted: bool = False
    translated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        return cls(
            id=str(data["id"]),
            start=float(data.get("start", 0.0) or 0.0),
            end=float(data.get("end", 0.0) or 0.0),
            speaker=str(data.get("speaker", "SPEAKER_A")),
            text=str(data.get("text", "")),
            source_text=str(data.get("source_text", "")),
            source_index=int(data.get("source_index", -1) or -1),
            source_id=str(data.get("source_id", "")),
            section=str(data.get("section", "")),
            section_start=bool(data.get("section_start", False)),
            inserted=bool(data.get("inserted", False)),
            translated=bool(data.get("translated", False)),
        )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def run_command(cmd: Sequence[str], *, desc: str = "") -> None:
    if desc:
        LOG.info("%s", desc)
    LOG.debug("$ %s", " ".join(str(part) for part in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(str(p) for p in cmd)}\n"
            f"{result.stderr[-4000:]}"
        )


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def normalize_speaker_label(raw: Any) -> str:
    label = str(raw or "").strip()
    if not label:
        return "SPEAKER_UNKNOWN"
    match = re.match(r"(?i)^speaker[\s_-]*([a-z0-9]+)$", label)
    if match:
        return f"SPEAKER_{match.group(1).upper()}"
    if len(label) == 1 and label.isalpha():
        return f"SPEAKER_{label.upper()}"
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") or "UNKNOWN"
    return f"SPEAKER_{cleaned.upper()}"


def short_preview(text: str, limit: int = 120) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def extract_audio(video: Path, audio_path: Path, *, force: bool = False) -> None:
    if audio_path.exists() and audio_path.stat().st_size > 0 and not force:
        LOG.info("Using cached audio: %s", audio_path)
        return
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        desc=f"Extracting audio from {video.name}",
    )
    LOG.info("Audio: %s (%.1f sec)", audio_path, ffprobe_duration(audio_path))


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def _segments_from_utterances(transcript: Any, video_duration: float) -> List[Segment]:
    rows: List[Tuple[float, float, str, str]] = []
    utterances = getattr(transcript, "utterances", None) or []
    for utterance in utterances:
        text = " ".join(str(getattr(utterance, "text", "") or "").split())
        if not text:
            continue
        rows.append(
            (
                float(getattr(utterance, "start", 0) or 0) / 1000.0,
                float(getattr(utterance, "end", 0) or 0) / 1000.0,
                normalize_speaker_label(getattr(utterance, "speaker", "")),
                text,
            )
        )

    if not rows:
        # Fallback: rebuild utterances from word-level timings.
        current: Optional[Dict[str, Any]] = None
        words = getattr(transcript, "words", None) or []
        for word in words:
            text = str(getattr(word, "text", "") or "").strip()
            if not text:
                continue
            start = float(getattr(word, "start", 0) or 0) / 1000.0
            end = float(getattr(word, "end", 0) or 0) / 1000.0
            speaker = normalize_speaker_label(getattr(word, "speaker", ""))
            if current is None or speaker != current["speaker"] or start - current["end"] > 1.5:
                if current and current["words"]:
                    rows.append(
                        (
                            current["start"],
                            current["end"],
                            current["speaker"],
                            " ".join(current["words"]),
                        )
                    )
                current = {"start": start, "end": end, "speaker": speaker, "words": []}
            if current is None:
                continue
            current["words"].append(text)
            current["end"] = end
        if current and current["words"]:
            rows.append(
                (
                    current["start"],
                    current["end"],
                    current["speaker"],
                    " ".join(current["words"]),
                )
            )

    rows.sort(key=lambda item: item[0])
    segments: List[Segment] = []
    for idx, (start, end, speaker, text) in enumerate(rows, start=1):
        if end <= start:
            end = min(video_duration, start + 0.5)
        segments.append(
            Segment(
                id=f"seg_{idx:04d}",
                start=max(0.0, start),
                end=max(start + 0.1, min(video_duration, end)),
                speaker=speaker,
                text=text,
                source_text=text,
                source_index=idx - 1,
                source_id=f"seg_{idx:04d}",
            )
        )
    return segments


def transcribe_video(
    audio_path: Path,
    *,
    video_duration: float,
    source_lang: str,
    speaker_count: int,
    asr_model: str,
    cache_path: Path,
    force: bool = False,
) -> List[Segment]:
    if cache_path.exists() and not force:
        cached = read_json(cache_path)
        segments = [Segment.from_dict(item) for item in cached.get("segments", [])]
        if segments:
            LOG.info("Using cached transcript: %s (%d segments)", cache_path, len(segments))
            return segments

    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set in .env")
    aai.settings.api_key = api_key

    model = aai.SpeechModel.best if str(asr_model).lower() == "best" else aai.SpeechModel.nano
    config = aai.TranscriptionConfig(
        speech_model=model,
        language_code=source_lang,
        speaker_labels=True,
        speakers_expected=speaker_count,
        punctuate=True,
        format_text=True,
        disfluencies=False,
    )
    LOG.info("Running AssemblyAI transcription (%s, language=%s, speakers=%s)…", model.value, source_lang, speaker_count)
    started = time.time()
    transcript = aai.Transcriber().transcribe(str(audio_path), config=config)
    error = getattr(transcript, "error", None)
    if error:
        raise RuntimeError(f"AssemblyAI transcription failed: {error}")

    segments = _segments_from_utterances(transcript, video_duration)
    if not segments:
        raise RuntimeError("AssemblyAI returned no transcript segments")

    write_json(
        cache_path,
        {
            "video_duration": video_duration,
            "source_language": source_lang,
            "asr_model": asr_model,
            "speaker_count": speaker_count,
            "transcript_id": getattr(transcript, "id", None),
            "segments": [segment.to_dict() for segment in segments],
        },
    )
    LOG.info(
        "Transcription complete in %.1f sec: %d segments, %.1f min",
        time.time() - started,
        len(segments),
        sum(seg.end - seg.start for seg in segments) / 60.0,
    )
    return segments


# ---------------------------------------------------------------------------
# Speaker remapping
# ---------------------------------------------------------------------------


def remap_speakers(segments: List[Segment], *, speaker_count: int, swap: bool) -> List[Segment]:
    durations: Dict[str, float] = {}
    first_seen: Dict[str, float] = {}
    for segment in segments:
        durations[segment.speaker] = durations.get(segment.speaker, 0.0) + max(
            0.0, segment.end - segment.start
        )
        first_seen.setdefault(segment.speaker, segment.start)

    speakers = sorted(durations, key=lambda item: first_seen.get(item, 0.0))
    if len(speakers) > speaker_count:
        ranked = sorted(speakers, key=lambda item: durations.get(item, 0.0), reverse=True)
        LOG.warning(
            "AssemblyAI found %d speakers, keeping the two with most speech: %s",
            len(speakers),
            ranked[:speaker_count],
        )
        speakers = sorted(ranked[:speaker_count], key=lambda item: first_seen.get(item, 0.0))

    if len(speakers) < 2:
        raise RuntimeError(
            "Expected at least two speakers. Use --speaker-count or pass a different video."
        )

    canonical = ["SPEAKER_A", "SPEAKER_B"]
    mapping: Dict[str, str] = {}
    for index, raw_speaker in enumerate(speakers[:2]):
        mapping[raw_speaker] = canonical[index]
    if swap:
        mapping = {raw: ("SPEAKER_B" if value == "SPEAKER_A" else "SPEAKER_A") for raw, value in mapping.items()}

    for segment in segments:
        segment.speaker = mapping.get(segment.speaker, mapping[speakers[0]])

    for raw_speaker, canonical_speaker in mapping.items():
        total = durations.get(raw_speaker, 0.0)
        count = sum(1 for segment in segments if segment.speaker == canonical_speaker)
        LOG.info(
            "Speaker %s -> %s: %.1f sec across %d segments",
            raw_speaker,
            canonical_speaker,
            total,
            count,
        )
    return segments


# ---------------------------------------------------------------------------
# Edit plan
# ---------------------------------------------------------------------------


def parse_keep_ranges(config: Dict[str, Any], video_duration: float) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    for item in config.get("edit", {}).get("keep_ranges", []):
        start = float(item["start"])
        end_raw = item.get("end")
        end = video_duration if end_raw is None else float(end_raw)
        if end <= start:
            raise ValueError(f"Invalid keep range: {item}")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("No edit.keep_ranges configured")
    return ranges


def section_for_time(config: Dict[str, Any], start: float) -> str:
    for section in config.get("edit", {}).get("sections", []):
        section_start = float(section["start"])
        section_end_raw = section.get("end")
        section_end = float("inf") if section_end_raw is None else float(section_end_raw)
        if section_start <= start < section_end:
            return str(section.get("name", ""))
    return ""


def is_filler_backchannel(segment: Segment) -> bool:
    duration = max(0.0, segment.end - segment.start)
    if duration > 2.2:
        return False
    words = segment.text.split()
    if len(words) > 5:
        return False
    return bool(FILLER_RE.match(segment.text.strip()))


def select_segments(
    segments: List[Segment],
    *,
    config: Dict[str, Any],
    video_duration: float,
    boundary_overlap: float,
) -> List[Segment]:
    ranges = parse_keep_ranges(config, video_duration)
    section_starts = [float(value) for value in config.get("edit", {}).get("section_starts", [])]
    tolerance = float(config.get("edit", {}).get("section_start_tolerance", 1.5))
    drop_fillers = bool(config.get("edit", {}).get("drop_filler_backchannels", True))

    selected: List[Segment] = []
    for segment in segments:
        duration = max(segment.end - segment.start, 0.001)
        best_overlap = 0.0
        for range_start, range_end in ranges:
            overlap = min(segment.end, range_end) - max(segment.start, range_start)
            if overlap > 0:
                best_overlap = max(best_overlap, overlap / duration)
        if best_overlap < boundary_overlap:
            continue
        if drop_fillers and is_filler_backchannel(segment):
            LOG.info("Dropping backchannel: %s", short_preview(segment.text, 80))
            continue
        # Use the midpoint so a segment that starts a fraction of a second
        # before a section boundary is assigned to the intended section.
        segment_midpoint = (segment.start + segment.end) / 2.0
        segment.section = section_for_time(config, segment_midpoint)
        selected.append(segment)

    if not selected:
        raise RuntimeError("No segments left after applying keep_ranges")

    # Mark the first segment of configured section starts.
    for range_start, range_end in ranges:
        first = next(
            (segment for segment in selected if range_start - tolerance <= segment.start < range_end),
            None,
        )
        if first is None:
            continue
        if any(abs(range_start - value) <= tolerance for value in section_starts):
            first.section_start = True
        LOG.info(
            "Range %.1f–%.1f: %d segment(s); first: [%.1f] %s",
            range_start,
            range_end,
            sum(1 for segment in selected if range_start <= segment.start < range_end),
            first.start,
            short_preview(first.text),
        )

    LOG.info("Selected %d/%d segments", len(selected), len(segments))
    return selected


def _inserted_segment(
    *,
    segment_id: str,
    speaker: str,
    text: str,
    section: str,
) -> Segment:
    return Segment(
        id=segment_id,
        start=0.0,
        end=0.0,
        speaker=speaker,
        text=text.strip(),
        source_text="",
        inserted=True,
        translated=True,
        section=section,
    )


def build_script(segments: List[Segment], config: Dict[str, Any]) -> List[Segment]:
    edit = config.get("edit", {})
    intro_cfg = edit.get("intro")
    outro_cfg = edit.get("outro")
    bridges = {
        float(item["before_range_start"]): item
        for item in edit.get("bridges", [])
        if item.get("before_range_start") is not None
    }

    script: List[Segment] = []
    if intro_cfg:
        script.append(
            _inserted_segment(
                segment_id="intro",
                speaker=str(intro_cfg.get("speaker", "SPEAKER_A")),
                text=str(intro_cfg.get("text", "")),
                section="Интро",
            )
        )

    inserted_bridges: set[float] = set()
    for segment in segments:
        for bridge_at, bridge in bridges.items():
            if bridge_at in inserted_bridges:
                continue
            if segment.start >= bridge_at - float(edit.get("section_start_tolerance", 1.5)):
                script.append(
                    _inserted_segment(
                        segment_id=f"bridge_{len(inserted_bridges) + 1}",
                        speaker=str(bridge.get("speaker", "SPEAKER_A")),
                        text=str(bridge.get("text", "")),
                        section=segment.section or "Переход",
                    )
                )
                inserted_bridges.add(bridge_at)
        script.append(segment)

    if outro_cfg:
        script.append(
            _inserted_segment(
                segment_id="outro",
                speaker=str(outro_cfg.get("speaker", "SPEAKER_A")),
                text=str(outro_cfg.get("text", "")),
                section="Финал",
            )
        )
    return script


# ---------------------------------------------------------------------------
# OpenRouter translation
# ---------------------------------------------------------------------------


def _translation_prompt(
    batch: Sequence[Segment],
    *,
    config: Dict[str, Any],
    previous: Sequence[Segment],
    following: Sequence[Segment],
    target_lang: str,
) -> str:
    speakers = config.get("speakers", {})
    speaker_lines = []
    for speaker_id, meta in speakers.items():
        speaker_lines.append(
            f"- {speaker_id} = {meta.get('name', speaker_id)} "
            f"({meta.get('gender', 'unknown')})."
        )

    glossary = config.get("glossary") or {}
    glossary_text = json.dumps(glossary, ensure_ascii=False, indent=2)

    # Pass the full previous batch so terminology, names and the conversational
    # thread stay consistent across batch boundaries.
    prev_context = [
        {"id": item.id, "speaker": item.speaker, "text": item.text}
        for item in previous
        if item.text
    ]
    next_context = [
        {"id": item.id, "speaker": item.speaker, "text": item.source_text or item.text}
        for item in following[:2]
    ]

    payload = [
        {
            "id": item.id,
            "speaker": item.speaker,
            "text": item.source_text or item.text,
            "section_start": item.section_start,
            "section": item.section,
        }
        for item in batch
    ]

    return f"""
You are a senior Russian literary translator and podcast editor. Translate an
English interview into natural spoken Russian for a science podcast.

SPEAKERS:
{chr(10).join(speaker_lines)}
- SPEAKER_A is female: use feminine grammatical forms for her.
- SPEAKER_B is male: use masculine grammatical forms for him.

STYLE:
- Native Russian, written for a Russian-speaking listener. It must sound like
  a Russian podcast, not like a translated English text.
- Clear, warm, conversational and immediately understandable; never a
  newsreader, never a stiff lecture, never a calque or a subtitle.
- Prefer short, speakable sentences and natural Russian rhythm. Avoid literal
  English word order, unnecessary pronouns and awkward passive constructions.
- You may freely rephrase sentences, idioms and metaphors for naturalness.
  Preserve the meaning, facts, arguments, examples, numbers and tone exactly.
  Do not add facts, explanations or examples that are not in the source.
- If a literal phrase sounds unnatural in Russian, translate its sense rather
  than its words.
- Preserve the guest's precise, thoughtful, slightly dry manner.
- Remove only obvious filler, false starts and repetitions. Do not summarize,
  do not cut substantive content.
- Do not merge or split segments: one input segment must produce exactly one
  output segment with the same id and speaker.
- There are no timing constraints. Do not shorten or expand anything to fit
  a video or an audio duration.
- Keep names and terms natural. Use the glossary below.

GLOSSARY:
{glossary_text}

SPECIAL RULES:
- If a segment has "section_start": true, it is the first line of a new section.
  The previous topic was cut from the podcast. Omit any opening sentence or
  clause that refers back to it (for example "Then that's a whole different
  issue" or "Let's talk about dreams"). Start directly with the substantive
  content of that segment.
- Do not translate the PREVIOUS or NEXT context; they are only for continuity.
- Return only valid JSON, no markdown and no commentary.

PREVIOUS TRANSLATED BATCH (do not translate; use it for terminology, names
and continuity):
{json.dumps(prev_context, ensure_ascii=False, indent=2)}

NEXT SOURCE LINES AFTER THIS BATCH (do not translate):
{json.dumps(next_context, ensure_ascii=False, indent=2)}

SEGMENTS TO TRANSLATE:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return exactly:
{{"segments":[{{"id":"...","speaker":"SPEAKER_A","text":"..."}}]}}
""".strip()


def _extract_openrouter_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(payload)[:1000]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        content = "\n".join(parts)
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def openrouter_complete(
    prompt: str,
    *,
    model: str,
    effort: str,
    max_tokens: int,
    timeout: int,
) -> Tuple[str, Dict[str, Any]]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a meticulous Russian translator and podcast editor. "
                    "Return only the requested JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }
    if effort and effort.lower() != "none":
        payload["reasoning"] = {"effort": effort, "exclude": True}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://dubblm.local",
        "X-Title": "DubbLM Podcast Translator",
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"OpenRouter {response.status_code}: {response.text[:2000]}"
                )
            data = response.json()
            usage = data.get("usage") or {}
            LOG.info(
                "OpenRouter %s: prompt=%s completion=%s reasoning=%s",
                model,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            )
            return _extract_openrouter_text(data), data
        except Exception as exc:  # noqa: BLE001 - retry network/provider failures
            last_error = exc
            if attempt >= 3:
                break
            delay = 5 * attempt
            LOG.warning("OpenRouter attempt %d failed: %s — retrying in %ss", attempt, exc, delay)
            time.sleep(delay)
    raise RuntimeError(f"OpenRouter translation failed: {last_error}") from last_error


def _parse_translation_json(raw: str) -> List[Dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except Exception:
        data = repair_json(text, return_objects=True)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        segments = data.get("segments")
        if isinstance(segments, list):
            return segments
        # Some models wrap the array under another key.
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
    raise ValueError("Translation response is not a JSON object with a segments array")


def validate_translation(
    parsed: List[Dict[str, Any]],
    batch: Sequence[Segment],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            errors.append("One of the translated items is not an object.")
            continue
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            errors.append("An item has an empty id.")
            continue
        if item_id in by_id:
            errors.append(f"Duplicate id: {item_id}")
        by_id[item_id] = item

    expected_ids = [item.id for item in batch]
    missing = [item_id for item_id in expected_ids if item_id not in by_id]
    extra = [item_id for item_id in by_id if item_id not in set(expected_ids)]
    if missing:
        errors.append(f"Missing ids: {missing}")
    if extra:
        errors.append(f"Unexpected ids: {extra}")

    normalized: List[Dict[str, Any]] = []
    for expected in batch:
        item = by_id.get(expected.id)
        if not item:
            continue
        speaker = str(item.get("speaker", "")).strip()
        if speaker != expected.speaker:
            errors.append(
                f"{expected.id}: speaker changed from {expected.speaker} to {speaker or 'empty'}"
            )
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            errors.append(f"{expected.id}: empty translation")
            continue
        source_len = max(1, len(expected.source_text or expected.text))
        ratio = len(text) / source_len
        if ratio < 0.35:
            errors.append(
                f"{expected.id}: translation looks too short ({ratio:.2f} of source)"
            )
        if ratio > 3.0:
            errors.append(
                f"{expected.id}: translation looks too long ({ratio:.2f} of source)"
            )
        latin = len(LATIN_RE.findall(text))
        cyrillic = len(CYRILLIC_RE.findall(text))
        if cyrillic > 20 and latin / max(1, latin + cyrillic) > 0.45:
            errors.append(f"{expected.id}: too much Latin script in the translation")
        normalized.append({"id": expected.id, "speaker": expected.speaker, "text": text})
    return normalized, errors


def translate_segments(
    segments: Sequence[Segment],
    *,
    config: Dict[str, Any],
    target_lang: str,
    cache_path: Path,
    force: bool = False,
) -> Dict[str, str]:
    translation_cfg = config.get("translation", {})
    model = str(translation_cfg.get("model", "openai/gpt-6.1-sol"))
    effort = str(translation_cfg.get("effort", "max"))
    max_tokens = int(translation_cfg.get("max_tokens", 128000))
    timeout = int(translation_cfg.get("timeout_seconds", 600))
    batch_segments = int(translation_cfg.get("batch_segments", 6))
    batch_chars = int(translation_cfg.get("batch_chars", 9000))

    source_hash = json_hash(
        {
            "prompt_version": PROMPT_VERSION,
            "target_lang": target_lang,
            "model": model,
            "effort": effort,
            "segments": [
                {
                    "id": item.id,
                    "speaker": item.speaker,
                    "text": item.source_text or item.text,
                    "section_start": item.section_start,
                }
                for item in segments
            ],
        }
    )
    cache_file = cache_path / f"{source_hash}.json"
    if cache_file.exists() and not force:
        cached = read_json(cache_file)
        LOG.info("Using cached translation: %s", cache_file)
        return {str(key): str(value) for key, value in cached.items()}

    if target_lang.lower() == str(config.get("source_lang", "en")).lower():
        LOG.info("Source and target languages are the same; skipping translation")
        return {item.id: item.source_text or item.text for item in segments}

    translated: Dict[str, str] = {}
    translated_segments: List[Segment] = []

    batches: List[List[Segment]] = []
    batch_starts: List[int] = []
    current: List[Segment] = []
    current_chars = 0
    current_start = 0
    for index, segment in enumerate(segments):
        segment_chars = len(segment.source_text or segment.text)
        if current and (
            len(current) >= batch_segments or current_chars + segment_chars > batch_chars
        ):
            batches.append(current)
            batch_starts.append(current_start)
            current = []
            current_chars = 0
            current_start = index
        current.append(segment)
        current_chars += segment_chars
    if current:
        batches.append(current)
        batch_starts.append(current_start)

    LOG.info(
        "Translating %d segments in %d batch(es) with %s (effort=%s)…",
        len(segments),
        len(batches),
        model,
        effort,
    )

    previous_batch: List[Segment] = []
    for batch_index, batch in enumerate(batches, start=1):
        batch_start = batch_starts[batch_index - 1]
        batch_end = batch_start + len(batch)
        following = segments[batch_end : batch_end + 2]
        prompt = _translation_prompt(
            batch,
            config=config,
            previous=previous_batch,
            following=following,
            target_lang=target_lang,
        )
        LOG.info(
            "Batch %d/%d: %d segment(s), %d chars",
            batch_index,
            len(batches),
            len(batch),
            sum(len(item.source_text or item.text) for item in batch),
        )
        raw, _ = openrouter_complete(
            prompt,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        parsed = _parse_translation_json(raw)
        normalized, errors = validate_translation(parsed, batch)
        if errors:
            repair_prompt = (
                "Your previous answer failed validation. Fix it and return only JSON.\n"
                "Errors:\n- " + "\n- ".join(errors) + "\n\n"
                "Original request:\n" + prompt + "\n\n"
                "Your previous answer:\n" + raw + "\n\n"
                "Return exactly the requested JSON with every id from SEGMENTS TO TRANSLATE."
            )
            LOG.warning("Batch %d validation failed: %s", batch_index, "; ".join(errors[:5]))
            raw, _ = openrouter_complete(
                repair_prompt,
                model=model,
                effort=effort,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            parsed = _parse_translation_json(raw)
            normalized, errors = validate_translation(parsed, batch)
        if errors:
            failure_path = cache_path / f"failed_batch_{batch_index:02d}.txt"
            failure_path.write_text(raw, encoding="utf-8")
            raise RuntimeError(
                f"Translation batch {batch_index} failed validation after repair: "
                + "; ".join(errors[:10])
                + f" (raw response saved to {failure_path})"
            )

        for item, original in zip(normalized, batch):
            translated[original.id] = item["text"]
            original.text = item["text"]
            original.translated = True
            translated_segments.append(original)

        # The next batch gets the full translation of this batch as context.
        previous_batch = list(batch)
        LOG.info("Batch %d/%d translated", batch_index, len(batches))

    write_json(cache_file, translated)
    LOG.info("Translation saved: %s", cache_file)
    return translated


# ---------------------------------------------------------------------------
# Script rendering
# ---------------------------------------------------------------------------


def apply_postprocess(script: Sequence[Segment], config: Dict[str, Any]) -> List[Segment]:
    """Apply non-translation edits: tail cuts and small text fixes.

    These run after translation so they do not invalidate the translation
    cache (the cache key is based on the selected source segments).
    """
    edit = config.get("edit", {})
    drop_ranges: List[Tuple[float, float]] = []
    for item in edit.get("postprocess_drop_ranges", []) or []:
        start = float(item["start"])
        end_raw = item.get("end")
        # A postprocess range without an end means "to the end of the source".
        end = float("inf") if end_raw is None else float(end_raw)
        drop_ranges.append((start, end))

    replacements: List[Tuple[str, str]] = []
    for item in edit.get("text_replacements", []) or []:
        old = str(item.get("old", ""))
        new = str(item.get("new", ""))
        if old:
            replacements.append((old, new))

    result: List[Segment] = []
    dropped = 0
    for segment in script:
        if not segment.inserted and drop_ranges:
            midpoint = (segment.start + segment.end) / 2.0
            if any(start <= midpoint < end for start, end in drop_ranges):
                LOG.info("Postprocess drop: %s", short_preview(segment.text, 90))
                dropped += 1
                continue
        for old, new in replacements:
            if old in segment.text:
                segment.text = segment.text.replace(old, new)
        result.append(segment)
    if dropped:
        LOG.info("Postprocess removed %d segment(s)", dropped)
    return result


def speaker_names(config: Dict[str, Any]) -> Dict[str, str]:
    return {
        speaker_id: str(meta.get("name", speaker_id))
        for speaker_id, meta in (config.get("speakers") or {}).items()
    }


def script_markdown(
    script: Sequence[Segment],
    *,
    config: Dict[str, Any],
    video: Path,
    target_lang: str,
) -> str:
    names = speaker_names(config)
    source_lang = config.get("source_lang", "en")
    lines = [
        "# Подкаст",
        "",
        f"- Видео: `{video}`",
        f"- Язык: `{source_lang}` → `{target_lang}`",
        f"- Перевод: `{config.get('translation', {}).get('model')}` "
        f"(`{(config.get('translation') or {}).get('effort')}`)",
        f"- TTS: `{config.get('tts', {}).get('model')}`",
        "",
    ]
    current_section = None
    for segment in script:
        if segment.section and segment.section != current_section:
            current_section = segment.section
            lines.extend(["", f"## {current_section}", ""])
        name = names.get(segment.speaker, segment.speaker)
        lines.append(f"**{name}:** {segment.text}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_script_outputs(
    script: Sequence[Segment],
    *,
    output_dir: Path,
    config: Dict[str, Any],
    video: Path,
    target_lang: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "podcast_script.json", [segment.to_dict() for segment in script])
    (output_dir / "podcast_script.md").write_text(
        script_markdown(script, config=config, video=video, target_lang=target_lang),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


def trim_pcm_silence(pcm: bytes, *, threshold: int = 180, pad_ms: int = 40) -> bytes:
    import numpy as np

    if not pcm:
        return b""
    samples = np.frombuffer(pcm, dtype="<i2")
    if samples.size == 0:
        return b""
    absolute = np.abs(samples.astype(np.int32))
    nonzero = np.flatnonzero(absolute > threshold)
    if nonzero.size == 0:
        return b""
    pad = int(SAMPLE_RATE * pad_ms / 1000)
    start = max(0, int(nonzero[0]) - pad)
    end = min(samples.size, int(nonzero[-1]) + pad + 1)
    return samples[start:end].tobytes()


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


class Gemini38TTSClient:
    """Minimal Gemini 3.8 Interactions API client for the podcast script.

    This deliberately does not import the heavy DubbLM TTS wrapper (voice
    matching, torch, pydub, speaker references).  It only needs
    ``google-genai>=2.0.0`` and returns raw 24 kHz mono PCM.
    """

    def __init__(self, *, model: str, fallback_model: str, api_key: Optional[str] = None) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "google-genai is not installed. Install requirements-podcast.txt "
                "or run the script inside the DubbLM container."
            ) from exc

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set in .env")

        version = str(getattr(genai, "__version__", "0"))
        try:
            major_version = int(version.split(".", 1)[0])
        except ValueError:
            major_version = 0
        if major_version < 2:
            raise RuntimeError(
                f"google-genai {version} uses the removed legacy Interactions API. "
                "Install google-genai>=2.25.0 (requirements-podcast.txt) or run "
                "inside the DubbLM container."
            )

        self._genai = genai
        self.client = genai.Client(api_key=key)
        if not hasattr(self.client, "interactions"):
            raise RuntimeError(
                f"google-genai {version} has no Interactions API. "
                "Install google-genai>=2.25.0 (requirements-podcast.txt)."
            )
        self.model = model
        self.fallback_model = fallback_model
        self.sample_rate = SAMPLE_RATE

    def _attempts_for(self, model: str) -> int:
        return 4 if model == self.model else 2

    def _build_request(self, text: str, style: str, voice: str, model: str) -> Dict[str, Any]:
        content: Dict[str, Any] = {"type": "text", "text": text}
        if style:
            content["annotations"] = [{"type": "speech_metadata", "style": style}]
        return {
            "model": model,
            "input": [{"type": "user_input", "content": [content]}],
            "response_format": {
                "type": "audio",
                "mime_type": "audio/l16",
                "sample_rate": self.sample_rate,
            },
            "generation_config": {"speech_config": [{"voice": voice}]},
            "store": False,
        }

    @staticmethod
    def _extract_audio_bytes(interaction: Any) -> bytes:
        audio = getattr(interaction, "output_audio", None)
        if audio is None and isinstance(interaction, dict):
            audio = interaction.get("output_audio")
        data_b64 = getattr(audio, "data", None) if audio is not None else None
        if data_b64 is None and isinstance(audio, dict):
            data_b64 = audio.get("data")
        if not data_b64:
            return b""
        try:
            raw = base64.b64decode(data_b64)
        except Exception:
            return b""
        return Gemini38TTSClient._normalize_audio_bytes(raw)

    @staticmethod
    def _normalize_audio_bytes(raw: bytes) -> bytes:
        if not raw:
            return b""
        if raw[:4] == b"RIFF":
            try:
                with wave.open(io.BytesIO(raw), "rb") as handle:
                    if handle.getsampwidth() != 2:
                        return b""
                    return handle.readframes(handle.getnframes())
            except Exception:
                return b""
        return raw

    def synthesize_chunk(
        self,
        text: str,
        style: str,
        voice: str,
        model: Optional[str] = None,
    ) -> bytes:
        if not text or not text.strip():
            return b""
        effective_model = model or self.model
        attempts = self._attempts_for(effective_model)
        payload = self._build_request(text, style, voice, effective_model)
        for attempt in range(1, attempts + 1):
            try:
                interaction = self.client.interactions.create(**payload)
                raw = self._extract_audio_bytes(interaction)
                if raw:
                    return raw
                raise RuntimeError("Interaction returned empty audio")
            except Exception as exc:  # noqa: BLE001 - retry provider errors
                if attempt >= attempts:
                    LOG.error(
                        "Gemini TTS failed after %d attempt(s) with model '%s': %s",
                        attempts,
                        effective_model,
                        exc,
                    )
                    return b""
                delay = 2.0 ** attempt
                LOG.warning(
                    "Gemini TTS attempt %d/%d failed (model=%s): %s — retrying in %.1fs",
                    attempt,
                    attempts,
                    effective_model,
                    exc,
                    delay,
                )
                time.sleep(delay)
        return b""


def _tts_client(config: Dict[str, Any]) -> Gemini38TTSClient:
    tts_cfg = config.get("tts", {})
    return Gemini38TTSClient(
        model=str(tts_cfg.get("model", "gemini-3.8-flash-tts")),
        fallback_model=str(tts_cfg.get("fallback_model", "gemini-3.8-flash-lite-tts")),
    )


def _podcast_style(scene_prompt: str, speaker_style: str, max_chars: int) -> str:
    """Build the full scene + character style without the 300-char per-part cap.

    ``gemini38_tags.sanitize_style`` intentionally caps individual custom
    prompts at 300 characters to avoid voice drift in dubbing.  A podcast scene
    needs a little more room, so we build the two parts ourselves and apply the
    requested total cap.
    """
    parts: List[str] = []
    for raw in (scene_prompt, speaker_style):
        cleaned = " ".join(str(raw or "").split()).strip().rstrip(":")
        if cleaned:
            parts.append(cleaned)
    style = ", ".join(parts)
    if len(style) <= max_chars:
        return style
    cut = style.rfind(" ", 0, max_chars + 1)
    return (style[: cut if cut > 0 else max_chars]).rstrip(" ,.;:-")


def synthesize_line(
    client: Gemini38TTSClient,
    *,
    text: str,
    voice: str,
    speaker_style: str,
    scene_prompt: str,
    style_max_chars: int,
    output_path: Path,
    model: str,
    fallback_model: str,
) -> float:
    prompt = build_gemini38_prompt(
        text,
        prompt_prefix=scene_prompt,
        speaker_prompt=speaker_style,
        emotion=None,
        emotion_note=None,
        max_style_chars=style_max_chars,
    )
    # Override the helper's per-part 300-char truncation with the full podcast
    # scene + character direction.
    prompt.style = _podcast_style(scene_prompt, speaker_style, style_max_chars)
    if not prompt.text.strip():
        raise RuntimeError("TTS text is empty")
    pcm = client.synthesize_chunk(prompt.text, prompt.style, voice, model=model)
    if not pcm:
        LOG.warning("Primary TTS model returned empty audio; trying fallback %s", fallback_model)
        pcm = client.synthesize_chunk(prompt.text, prompt.style, voice, model=fallback_model)
    if not pcm:
        raise RuntimeError(f"TTS returned empty audio for line: {short_preview(text)}")
    pcm = trim_pcm_silence(pcm)
    if not pcm:
        raise RuntimeError(f"TTS returned silent audio for line: {short_preview(text)}")
    write_wav(output_path, pcm)
    return wav_duration(output_path)


def synthesize_script(
    script: Sequence[Segment],
    *,
    config: Dict[str, Any],
    tts_dir: Path,
    force: bool = False,
) -> List[Path]:
    client = _tts_client(config)
    speakers = config.get("speakers", {})
    scene_prompt = str(config.get("scene_prompt", "")).strip()
    tts_cfg = config.get("tts", {})
    model = str(tts_cfg.get("model", "gemini-3.8-flash-tts"))
    fallback_model = str(tts_cfg.get("fallback_model", "gemini-3.8-flash-lite-tts"))
    style_max_chars = int(tts_cfg.get("style_max_chars", 600))

    tts_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for index, segment in enumerate(script, start=1):
        output_path = tts_dir / f"line_{index:03d}_{segment.id}.wav"
        if output_path.exists() and output_path.stat().st_size > 1000 and not force:
            LOG.info("Using cached TTS line %d/%d: %s", index, len(script), output_path.name)
            paths.append(output_path)
            continue
        meta = speakers.get(segment.speaker, {})
        voice = str(meta.get("voice", "Kore"))
        style = str(meta.get("style", ""))
        LOG.info(
            "TTS %d/%d [%s, %s]: %s",
            index,
            len(script),
            segment.speaker,
            voice,
            short_preview(segment.text, 100),
        )
        duration = synthesize_line(
            client,
            text=segment.text,
            voice=voice,
            speaker_style=style,
            scene_prompt=scene_prompt,
            style_max_chars=style_max_chars,
            output_path=output_path,
            model=model,
            fallback_model=fallback_model,
        )
        LOG.info("  -> %.1f sec", duration)
        paths.append(output_path)
    return paths


def render_audition(
    script: Sequence[Segment],
    *,
    config: Dict[str, Any],
    output_dir: Path,
    voices: Sequence[str],
    max_chars: int,
    tts_dir: Path,
) -> List[Path]:
    candidates = [
        segment
        for segment in script
        if segment.speaker == "SPEAKER_B" and not segment.inserted and len(segment.text) > 200
    ]
    if not candidates:
        candidates = [segment for segment in script if segment.speaker == "SPEAKER_B"]
    if not candidates:
        raise RuntimeError("No SPEAKER_B lines to audition")
    line = max(candidates[:3], key=lambda item: len(item.text))
    text = line.text.strip()
    if len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        text = text[: cut if cut > 0 else max_chars].rstrip(" ,.;:-") + "…"
    speaker_style = str(
        (config.get("speakers", {}).get("SPEAKER_B") or {}).get("style", "")
    )
    scene_prompt = str(config.get("scene_prompt", "")).strip()
    tts_cfg = config.get("tts", {})
    model = str(tts_cfg.get("model", "gemini-3.8-flash-tts"))
    fallback_model = str(tts_cfg.get("fallback_model", "gemini-3.8-flash-lite-tts"))
    style_max_chars = int(tts_cfg.get("style_max_chars", 600))
    client = _tts_client(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    tts_dir.mkdir(parents=True, exist_ok=True)
    results: List[Path] = []
    for voice in voices:
        voice = voice.strip()
        if not voice:
            continue
        wav_path = tts_dir / f"audition_{voice}.wav"
        LOG.info("Audition: voice=%s, line=%s", voice, short_preview(text, 100))
        synthesize_line(
            client,
            text=text,
            voice=voice,
            speaker_style=speaker_style,
            scene_prompt=scene_prompt,
            style_max_chars=style_max_chars,
            output_path=wav_path,
            model=model,
            fallback_model=fallback_model,
        )
        ogg_path = output_dir / f"audition_{voice}.ogg"
        export_audio(wav_path, ogg_path, title=f"Podcast audition — {voice}", artist="DubbLM")
        results.append(ogg_path)
        LOG.info("Audition saved: %s", ogg_path)
    return results


# ---------------------------------------------------------------------------
# Assembly and export
# ---------------------------------------------------------------------------


def _read_wav_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise RuntimeError(f"Expected mono 16-bit WAV: {path}")
        return handle.readframes(handle.getnframes())


def _silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * max(0.0, seconds))


def assemble_audio(
    script: Sequence[Segment],
    tts_paths: Sequence[Path],
    *,
    master_path: Path,
) -> List[Dict[str, Any]]:
    if len(script) != len(tts_paths):
        raise RuntimeError("Script and TTS line counts do not match")
    master_path.parent.mkdir(parents=True, exist_ok=True)
    combined = bytearray()
    timeline: List[Dict[str, Any]] = []
    current_time = 0.0
    previous_section = None
    previous_speaker = None

    for index, (segment, path) in enumerate(zip(script, tts_paths)):
        if index == 0:
            gap = 0.15
        elif segment.section != previous_section:
            gap = 0.65
        elif segment.speaker != previous_speaker:
            gap = 0.28
        else:
            gap = 0.18

        if gap:
            combined.extend(_silence(gap))
            current_time += gap
        pcm = _read_wav_pcm(path)
        start = current_time
        combined.extend(pcm)
        duration = len(pcm) / (SAMPLE_RATE * 2)
        current_time += duration
        timeline.append(
            {
                "id": segment.id,
                "speaker": segment.speaker,
                "section": segment.section,
                "start": round(start, 3),
                "end": round(current_time, 3),
                "duration": round(duration, 3),
                "inserted": segment.inserted,
                "text": segment.text,
            }
        )
        previous_section = segment.section
        previous_speaker = segment.speaker

    combined.extend(_silence(0.35))
    with wave.open(str(master_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(bytes(combined))
    LOG.info("Master audio: %s (%.1f sec)", master_path, wav_duration(master_path))
    return timeline


def export_audio(
    source: Path,
    output: Path,
    *,
    title: str,
    artist: str = "DubbLM Podcast",
    album: str = "Самое интересное в ИИ",
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".ogg":
        codec = ["-c:a", "libopus", "-b:a", "160k"]
        metadata = [
            "-metadata:s:a:0",
            f"title={title}",
            "-metadata:s:a:0",
            f"artist={artist}",
            "-metadata:s:a:0",
            f"album={album}",
        ]
    elif suffix == ".mp3":
        codec = ["-c:a", "libmp3lame", "-b:a", "192k"]
        metadata = [
            "-metadata",
            f"title={title}",
            "-metadata",
            f"artist={artist}",
            "-metadata",
            f"album={album}",
        ]
    else:
        raise ValueError(f"Unsupported output format: {output.suffix}")
    run_command(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ac",
            "1",
            "-ar",
            "48000",
            *codec,
            *metadata,
            str(output),
        ],
        desc=f"Exporting {output.name}",
    )
    LOG.info("Exported: %s (%.1f sec, %.1f MB)", output, ffprobe_duration(output), output.stat().st_size / 1e6)


def _format_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def write_chapters(
    timeline: Sequence[Dict[str, Any]],
    *,
    config: Dict[str, Any],
    output_dir: Path,
    intro_title: str = "Интро",
) -> None:
    chapters = [{"title": intro_title, "start": 0.0}]
    seen = {intro_title}
    for item in timeline:
        section = str(item.get("section") or "").strip()
        if not section or section in seen:
            continue
        seen.add(section)
        chapters.append({"title": section, "start": float(item["start"])})

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "podcast_chapters.json", chapters)
    lines = [f"{_format_time(item['start'])}  {item['title']}" for item in chapters]
    (output_dir / "podcast_chapters.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_config(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML object: {path}")

    if args.source_lang:
        config["source_lang"] = args.source_lang
    if args.target_lang:
        config["target_lang"] = args.target_lang

    config.setdefault("speakers", {})
    if args.voice_a:
        config["speakers"].setdefault("SPEAKER_A", {})["voice"] = args.voice_a
    if args.voice_b:
        config["speakers"].setdefault("SPEAKER_B", {})["voice"] = args.voice_b
    if args.scene_prompt:
        config["scene_prompt"] = args.scene_prompt

    translation = config.setdefault("translation", {})
    if args.translation_model:
        translation["model"] = args.translation_model
    if args.translation_effort:
        translation["effort"] = args.translation_effort
    if args.translation_max_tokens:
        translation["max_tokens"] = args.translation_max_tokens

    tts = config.setdefault("tts", {})
    if args.tts_model:
        tts["model"] = args.tts_model
    if args.tts_fallback_model:
        tts["fallback_model"] = args.tts_fallback_model

    if args.boundary_overlap is not None:
        config.setdefault("edit", {})["boundary_overlap"] = args.boundary_overlap

    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a podcast from a video with fresh ASR, OpenRouter translation and Gemini 3.8 TTS."
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "podcast_configs" / "hinton_sleep.yaml"),
        help="YAML edit/speaker/TTS config",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "podcast"),
        help="Output directory",
    )
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "cache" / "podcast"))
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", default=None)
    parser.add_argument("--speaker-count", type=int, default=2)
    parser.add_argument("--swap-speakers", action="store_true")
    parser.add_argument("--asr-model", default="best", choices=["best", "nano"])
    parser.add_argument("--boundary-overlap", type=float, default=None)
    parser.add_argument("--voice-a", default=None)
    parser.add_argument("--voice-b", default=None)
    parser.add_argument("--scene-prompt", default=None)
    parser.add_argument("--translation-model", default=None)
    parser.add_argument("--translation-effort", default=None)
    parser.add_argument("--translation-max-tokens", type=int, default=None)
    parser.add_argument("--tts-model", default=None)
    parser.add_argument("--tts-fallback-model", default=None)
    parser.add_argument("--format", choices=["ogg", "mp3", "both"], default="ogg")
    parser.add_argument("--force-transcribe", action="store_true")
    parser.add_argument("--force-translate", action="store_true")
    parser.add_argument("--force-tts", action="store_true")
    parser.add_argument("--transcribe-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Translate and write the script, but do not synthesize")
    parser.add_argument("--audition", action="store_true", help="Render a short SPEAKER_B audition instead of the full episode")
    parser.add_argument(
        "--audition-voices",
        default="Alnilam,Algieba",
        help="Comma-separated Gemini voices for --audition",
    )
    parser.add_argument("--audition-chars", type=int, default=900)
    parser.add_argument("--render", action="store_true", help="Synthesize the full podcast")
    parser.add_argument("--keep-audio", action="store_true", help="Keep per-line WAV files")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        parser.error(f"Video not found: {video}")
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path, args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir).expanduser().resolve() / hash_file(video)[:12]
    cache_root.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    video_duration = ffprobe_duration(video)
    LOG.info("Video: %s (%.1f sec)", video, video_duration)
    LOG.info("Config: %s", config_path)
    LOG.info("Cache: %s", cache_root)

    audio_path = cache_root / "audio.wav"
    extract_audio(video, audio_path, force=args.force_transcribe)

    transcript_path = cache_root / "transcript.json"
    segments = transcribe_video(
        audio_path,
        video_duration=video_duration,
        source_lang=str(config.get("source_lang", "en")),
        speaker_count=args.speaker_count,
        asr_model=args.asr_model,
        cache_path=transcript_path,
        force=args.force_transcribe,
    )
    segments = remap_speakers(segments, speaker_count=args.speaker_count, swap=args.swap_speakers)
    write_json(
        output_dir / "podcast_original_transcript.json",
        [segment.to_dict() for segment in segments],
    )
    LOG.info("Original transcript: %s", output_dir / "podcast_original_transcript.json")

    if args.transcribe_only:
        return 0

    selected = select_segments(
        segments,
        config=config,
        video_duration=video_duration,
        boundary_overlap=float(config.get("edit", {}).get("boundary_overlap", 0.45)),
    )
    script = build_script(selected, config)

    target_lang = str(config.get("target_lang", "ru"))
    source_segments = [segment for segment in script if not segment.inserted]
    translated = translate_segments(
        source_segments,
        config=config,
        target_lang=target_lang,
        cache_path=cache_root,
        force=args.force_translate,
    )
    for segment in source_segments:
        if segment.id in translated:
            segment.text = translated[segment.id]
            segment.translated = True

    script = apply_postprocess(script, config)

    write_script_outputs(
        script,
        output_dir=output_dir,
        config=config,
        video=video,
        target_lang=target_lang,
    )
    write_json(output_dir / "podcast_transcript.json", [segment.to_dict() for segment in script])
    LOG.info("Script written to %s", output_dir / "podcast_script.md")

    if args.dry_run or not (args.audition or args.render):
        LOG.info("Dry run complete. Review %s", output_dir / "podcast_script.md")
        return 0

    if args.audition:
        voices = [voice.strip() for voice in args.audition_voices.split(",") if voice.strip()]
        render_audition(
            script,
            config=config,
            output_dir=output_dir,
            voices=voices,
            max_chars=args.audition_chars,
            tts_dir=work_dir / "audition",
        )
        return 0

    tts_paths = synthesize_script(
        script,
        config=config,
        tts_dir=work_dir / "tts",
        force=args.force_tts,
    )
    master_path = work_dir / "podcast_master.wav"
    timeline = assemble_audio(script, tts_paths, master_path=master_path)
    write_json(output_dir / "podcast_timeline.json", timeline)
    write_chapters(timeline, config=config, output_dir=output_dir)

    output_cfg = config.get("output", {})
    title = str(output_cfg.get("title") or video.stem)
    artist = str(output_cfg.get("artist") or "DubbLM Podcast")
    album = str(output_cfg.get("album") or "Самое интересное в ИИ")
    outputs: List[Path] = []
    if args.format in {"ogg", "both"}:
        outputs.append(output_dir / "podcast.ogg")
        export_audio(master_path, outputs[-1], title=title, artist=artist, album=album)
    if args.format in {"mp3", "both"}:
        outputs.append(output_dir / "podcast.mp3")
        export_audio(master_path, outputs[-1], title=title, artist=artist, album=album)

    if not args.keep_audio:
        shutil.rmtree(work_dir / "tts", ignore_errors=True)

    LOG.info("Done. Outputs:")
    for output in outputs:
        LOG.info("  %s", output)
    LOG.info("  %s", output_dir / "podcast_script.md")
    LOG.info("  %s", output_dir / "podcast_chapters.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
