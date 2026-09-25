"""Dialogue-context style hints shared by the dubbing pipeline and TTS wrappers.

The previous translated line can be attached to the next segment as a short
quoted delivery context. It is passed to Gemini 3.8 TTS inside
``speech_metadata.style`` only, so the model interprets it as acting direction
rather than reading it aloud.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from src.tts.gemini38_tags import strip_tts_markup

DEFAULT_CONTEXT_STYLE_MAX_CHARS = 140


def normalize_quote_text(text: Optional[str], max_chars: int) -> str:
    """Strip markup/whitespace and truncate on a word boundary."""
    if not text:
        return ""
    cleaned = strip_tts_markup(str(text))
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[: max_chars + 1]
    cut = truncated.rfind(" ")
    quote = (truncated[:cut] if cut > 0 else truncated[:max_chars]).strip()
    return quote.rstrip(" ,;:-") + "…"


def build_context_style(
    previous_text: Optional[str],
    previous_speaker: Optional[str],
    current_speaker: Optional[str],
    max_chars: int = DEFAULT_CONTEXT_STYLE_MAX_CHARS,
) -> Optional[str]:
    """Build a quoted delivery-context hint for the current segment."""
    if max_chars is None or int(max_chars) <= 0:
        return None
    quote = normalize_quote_text(previous_text, int(max_chars))
    if not quote:
        return None
    if previous_speaker and current_speaker and previous_speaker != current_speaker:
        return f'Replying to {previous_speaker}: "{quote}" — match the conversational energy'
    return f'Continuing the previous line: "{quote}" — keep the same delivery'


def context_cache_token(context_style: Optional[str]) -> str:
    """Short stable hash of the context hint for cache keys."""
    return hashlib.md5((context_style or "").encode()).hexdigest()[:8]
