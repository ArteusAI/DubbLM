"""Gemini-assisted speaker labeling for ASR transcripts.

AssemblyAI diarization can collapse different characters onto one speaker on
hard material (rapid, overlapping dialogue in music-heavy trailers). This
module asks Gemini (which hears the audio) to assign speaker labels to an
existing timestamped transcript, then re-aligns those labels onto the precise
word timestamps produced by the ASR so dubbing timing is never compromised.

The labeler only decides *who speaks when*; it must not change the words or the
timestamps, which stay anchored to the ASR output.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from typing import Any, Dict, List, Optional

from src.dubbing.core.log_config import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "google/gemini-2.5-pro"
MAX_INLINE_BYTES = 14 * 1024 * 1024
_TOKEN_RE = re.compile(r"[^\w']+", re.UNICODE)

LABELING_PROMPT = """You are a dubbing director. Below is an automatic transcript of the attached movie-trailer audio with timestamps.

Your job: assign the correct speaker to every line and split lines that contain more than one speaker.

Rules:
- There are at most {max_speakers} distinct human voices. Use labels SPEAKER_A..SPEAKER_{max_letter} only.
- The SAME voice must always get the SAME label (judge by voice timbre, not by content).
- If one transcript line mixes two speakers, split it into consecutive parts, keep the exact words, and estimate the split time between their start and end.
- Keep the given start/end timestamps for unsplit lines; for split lines divide the segment time proportionally.
- Never change the words. Never invent or drop lines.
- Ignore music, sound effects, singing and non-verbal sounds.
- Return ONLY JSON, no markdown: {{"segments":[{{"start":0.0,"end":1.2,"speaker":"SPEAKER_A","text":"..."}}]}}

Transcript lines:
{lines_json}"""


def _normalize_tokens(text: str) -> List[str]:
    tokens = []
    for raw in str(text or "").split():
        token = _TOKEN_RE.sub("", raw.lower())
        if token:
            tokens.append(token)
    return tokens


def label_speakers_with_gemini(
    audio_file: str,
    lines: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_speakers: int = 4,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Ask Gemini to assign speaker labels to timestamped transcript lines.

    Returns a list of ``{"start", "end", "speaker", "text"}`` dicts in order.
    Raises RuntimeError when the model is unavailable or returns unusable data.
    """
    if not lines:
        return []

    try:
        from src.utils.openrouter_client import (
            audio_input_part,
            chat_completion,
            extract_message_text,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"OpenRouter client is unavailable: {exc}") from exc

    if not (api_key or os.environ.get("OPENROUTER_API_KEY")):
        raise RuntimeError("OPENROUTER_API_KEY is required for speaker labeling")

    max_speakers = max(1, min(int(max_speakers or 4), 26))
    max_letter = chr(ord("A") + max_speakers - 1)

    with open(audio_file, "rb") as handle:
        audio_bytes = handle.read()
    audio_format = _guess_audio_format(audio_file)

    prompt = LABELING_PROMPT.format(
        max_speakers=max_speakers,
        max_letter=max_letter,
        lines_json=json.dumps(lines, ensure_ascii=False),
    )
    normalized_model = str(model or DEFAULT_MODEL)
    if not normalized_model.startswith("google/"):
        normalized_model = f"google/{normalized_model}"

    payload = chat_completion(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    audio_input_part(audio_bytes, audio_format),
                ],
            }
        ],
        model=normalized_model,
        temperature=0.0,
        response_format={"type": "json_object"},
        api_key=api_key,
    )
    parsed = _parse_label_response(extract_message_text(payload))
    if not parsed:
        raise RuntimeError("Gemini speaker labeling returned no usable segments")
    return parsed


def _guess_mime_type(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".mp3"):
        return "audio/mp3"
    if lowered.endswith((".m4a", ".mp4")):
        return "audio/mp4"
    if lowered.endswith(".webm"):
        return "audio/webm"
    if lowered.endswith(".ogg"):
        return "audio/ogg"
    if lowered.endswith(".flac"):
        return "audio/flac"
    return "audio/wav"


def _guess_audio_format(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".mp3"):
        return "mp3"
    if lowered.endswith((".m4a", ".mp4")):
        return "m4a"
    if lowered.endswith(".webm"):
        return "webm"
    if lowered.endswith(".ogg"):
        return "ogg"
    if lowered.endswith(".flac"):
        return "flac"
    return "wav"


def _parse_label_response(text: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
        if not match:
            logger.warning("Gemini speaker labeling response is not JSON: %s", text[:200])
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Gemini speaker labeling JSON could not be repaired")
            return []

    if isinstance(data, dict):
        data = data.get("segments", [])
    if not isinstance(data, list):
        return []

    segments: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text_value = str(item.get("text") or "").strip()
        speaker = str(item.get("speaker") or "").strip()
        if not text_value or not speaker:
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        segments.append(
            {"start": start, "end": end, "speaker": speaker, "text": text_value}
        )
    return segments


def align_labeled_segments_to_words(
    labeled_segments: List[Dict[str, Any]],
    words: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Re-anchor Gemini speaker labels onto precise ASR word timestamps.

    ``words`` must be the full ordered ASR word list with ``word``, ``start``
    and ``end`` (seconds). The Gemini text is matched order-preservingly to the
    ASR tokens; each labeled segment inherits the exact span of its matched
    words. Unmatched segments are dropped rather than trusted blindly.
    """
    if not labeled_segments or not words:
        return []

    asr_tokens: List[str] = []
    asr_spans: List[Dict[str, Any]] = []
    for word in words:
        for token in _normalize_tokens(word.get("word", "")):
            asr_tokens.append(token)
            asr_spans.append({"start": float(word["start"]), "end": float(word["end"])})
    if not asr_tokens:
        return []

    gemini_tokens: List[str] = []
    token_ranges: List[tuple[int, int, Dict[str, Any]]] = []
    for segment in labeled_segments:
        tokens = _normalize_tokens(segment.get("text", ""))
        if not tokens:
            continue
        start_idx = len(gemini_tokens)
        gemini_tokens.extend(tokens)
        token_ranges.append((start_idx, len(gemini_tokens), segment))

    if not gemini_tokens:
        return []

    matcher = difflib.SequenceMatcher(None, gemini_tokens, asr_tokens, autojunk=False)
    token_map: Dict[int, int] = {}
    for g_start, a_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            token_map[g_start + offset] = a_start + offset

    segments: List[Dict[str, Any]] = []
    for g_start, g_end, segment in token_ranges:
        matched_asr_indices = [token_map[i] for i in range(g_start, g_end) if i in token_map]
        if not matched_asr_indices:
            logger.debug(
                "Gemini speaker segment has no ASR match, dropping: %s", segment.get("text", "")[:60]
            )
            continue
        first, last = matched_asr_indices[0], matched_asr_indices[-1]
        span = asr_spans[first : last + 1]
        matched_words = _asr_words_for_span(words, first, last)
        segments.append(
            {
                "speaker": segment["speaker"],
                "start": span[0]["start"],
                "end": span[-1]["end"],
                "text": _join_asr_words(words, first, last),
                "words": [dict(word, speaker=segment["speaker"]) for word in matched_words],
            }
        )

    segments.sort(key=lambda item: item["start"])
    return segments


def _asr_words_for_span(words: List[Dict[str, Any]], first: int, last: int) -> List[Dict[str, Any]]:
    """Return the ASR words matching the token span [first, last] (inclusive)."""
    result: List[Dict[str, Any]] = []
    index = 0
    for word in words:
        tokens = _normalize_tokens(word.get("word", ""))
        if not tokens:
            continue
        word_first = index
        word_last = index + len(tokens) - 1
        index += len(tokens)
        if word_last < first:
            continue
        if word_first > last:
            break
        result.append(word)
    return result


def _join_asr_words(words: List[Dict[str, Any]], first: int, last: int) -> str:
    return " ".join(str(word.get("word", "")).strip() for word in _asr_words_for_span(words, first, last))
