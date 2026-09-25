"""Tag and style helpers for Gemini 3.8 Flash TTS (Interactions API).

Gemini 3.8 treats the request ``text`` as a verbatim transcript:

* sustained delivery lives in ``speech_metadata.style``;
* only momentary human vocalisations/pauses stay inline as angle-bracket
  tags (``<sigh>``, ``<short pause>``, ...).

DubbLM's editor and translator still emit the legacy ``[tag]`` dialect, so
this module converts legacy markup into the 3.8 dialect and builds a compact
style string. It is intentionally dependency-free so it can be unit tested
without the Google SDK.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Official 3.8 tag vocabulary (documented in the prompting guide)
# ---------------------------------------------------------------------------

INLINE_VOCAL_TAGS = frozenset(
    {
        "argh",
        "breath",
        "heavy breath",
        "exhales",
        "cackle",
        "cheer",
        "chuckle",
        "chuckles",
        "cough",
        "cry",
        "gasp",
        "giggle",
        "groan",
        "growl",
        "grunt",
        "grr",
        "hiss",
        "laugh",
        "laughter",
        "moan",
        "pant",
        "pff",
        "phew",
        "scream",
        "shout",
        "shriek",
        "sigh",
        "sighs",
        "sneeze",
        "snicker",
        "snort",
        "sob",
        "throat-clearing",
        "tsk",
        "whimper",
        "whispers",
        "whispering",
        "yawn",
        "short pause",
        "long pause",
    }
)

# Legacy [tag] -> inline angle-bracket tag (point-in-time event).
INLINE_TAG_MAP: Dict[str, str] = {
    "sigh": "sigh",
    "sighs": "sigh",
    "laugh": "laugh",
    "laughs": "laugh",
    "laughter": "laugh",
    "laughing": "laugh",
    "giggle": "giggle",
    "giggles": "giggle",
    "chuckle": "chuckle",
    "chuckles": "chuckle",
    "gasp": "gasp",
    "cough": "cough",
    "sob": "sob",
    "cry": "sob",
    "crying": "sob",
    "cries": "sob",
    "sneeze": "sneeze",
    "yawn": "yawn",
    "pant": "pant",
    "pants": "pant",
    "whimper": "whimper",
    "groan": "groan",
    "grunt": "grunt",
    "tsk": "tsk",
    "phew": "phew",
    "pff": "pff",
    "short pause": "short pause",
    "medium pause": "short pause",
    "long pause": "long pause",
    "pause": "short pause",
}

# Legacy [tag] -> sustained style phrase for speech_metadata.style.
STYLE_TAG_MAP: Dict[str, str] = {
    "sarcasm": "sarcastic",
    "sarcastic": "sarcastic",
    "robotic": "monotone and flat",
    "shouting": "shouting",
    "yelling": "shouting",
    "whisper": "whispering",
    "whispers": "whispering",
    "whispering": "whispering",
    "extremely fast": "speaking rapidly",
    "fast": "speaking rapidly",
    "quickly": "speaking rapidly",
    "slowly": "speaking slowly",
    "excited": "excited",
    "amazed": "amazed",
    "curious": "curious",
    "serious": "serious",
    "tired": "tired",
    "trembling": "trembling",
    "panicked": "panicked",
    "mischievously": "mischievous",
    "mischievous": "mischievous",
    "angry": "angry tone",
    "sad": "sad",
    "happy": "happy",
    "warm": "warm",
    "calm": "calm",
    "nervous": "nervous",
}

# Filler tags become literal disfluency words (per the prompting guide the
# transcript should keep natural spoken disfluencies as plain text).
FILLER_TAG_MAP: Dict[str, str] = {
    "uhm": "uhm",
    "um": "um",
    "uh": "uh",
    "erm": "erm",
    "hmm": "hmm",
    "hm": "hm",
    "eh": "eh",
}

_LEGACY_TAG_RE = re.compile(r"\[([^\[\]]{1,40})\]")
_PIPE_RE = re.compile(r"\|([^|]{1,60})\|")
_ANGLE_TAG_RE = re.compile(r"<([^<>]{1,40})>")
_QUOTED_FRAGMENT_RE = re.compile(r'["\u201c\u00ab]([^"\u201d\u00bb]{4,400})["\u201d\u00bb]')
_WORD_RE = re.compile(r"[a-zA-Z\u00c0-\u024f\u0400-\u04ff0-9']{3,}")

# Meta-instructions from earlier Gemini prompt dialects that must never leak
# into the 3.8 style field or transcript.
_META_STYLE_PATTERNS: Sequence[re.Pattern] = (
    re.compile(r"(?i)synthesize only the transcript[^.]*\.\s*"),
    re.compile(r"(?i)do not read these directions\.?\s*"),
    re.compile(r"(?i)speak only the transcript text[^.]*\.\s*"),
    re.compile(r"(?i)transcript to synthesize starts below\.?\s*"),
    re.compile(r"(?i)do not switch speaker identity[^.]*\.\s*"),
    re.compile(r"(?i)maintain identical timbre[^.]*\.\s*"),
)

# Legacy baked style prompts -> compact 3.8 style phrases.
_PRESET_STYLE_MAP: Dict[str, str] = {
    "speak with natural conversational energy, clear articulation": "natural conversational energy, clear articulation",
    "speak calmly and clearly, with measured pacing and an authoritative tone": "calm, measured pacing, authoritative tone",
    "speak slowly and deliberately, with a suspenseful, deeply resonant tone": "slow, deliberate, suspenseful, deeply resonant tone",
    "deliver it like a modern news explainer: confident, crisp broadcast narration with forward momentum, energetic narrative cadence, clear factual emphasis, bright but serious tone, no shouting or parody, short clean pauses between clauses": "confident crisp broadcast narration, forward momentum, clear factual emphasis, bright but serious tone",
}


@dataclass
class Gemini38Prompt:
    """Ready-to-send payload pieces for the Interactions API."""

    text: str
    style: str


def convert_legacy_tags(text: str) -> Tuple[str, List[str]]:
    """Convert legacy ``[tag]`` markup into the Gemini 3.8 dialect.

    Returns ``(converted_text, style_hints)`` where ``style_hints`` is an
    ordered, de-duplicated list of sustained style phrases. Unknown tags are
    dropped so they are never spoken aloud.
    """
    if not text:
        return "", []

    style_hints: List[str] = []
    seen_hints: set = set()

    def _replace(match: re.Match) -> str:
        raw = match.group(1).strip().lower()
        if not raw:
            return ""
        if raw in INLINE_TAG_MAP:
            return f"<{INLINE_TAG_MAP[raw]}>"
        if raw in STYLE_TAG_MAP:
            phrase = STYLE_TAG_MAP[raw]
            if phrase not in seen_hints:
                seen_hints.add(phrase)
                style_hints.append(phrase)
            return ""
        if raw in FILLER_TAG_MAP:
            return FILLER_TAG_MAP[raw]
        # Unknown legacy tag: drop rather than risk it being read aloud.
        return ""

    converted = _LEGACY_TAG_RE.sub(_replace, text)

    # Per-line synthesis has no second speaker: unwrap backchannel pipes so the
    # model reads the words naturally instead of the pipe characters.
    converted = _PIPE_RE.sub(lambda m: m.group(1), converted)

    # Collapse whitespace artifacts left by removed tags.
    converted = re.sub(r"[ \t]{2,}", " ", converted)
    converted = re.sub(r"\s+([,.!?;:])", r"\1", converted)
    converted = converted.strip()

    return converted, style_hints


def sanitize_angle_tags(text: str) -> str:
    """Normalize inline angle-bracket tags to the official 3.8 vocabulary.

    Keeps only tags documented in the prompting guide, maps known aliases
    (e.g. ``<medium pause>`` -> ``<short pause>``) and drops unknown tags so
    they are never sent to the model. The enricher normally emits compliant
    tags, but LLM output can drift; this is the final guard before the API.
    """
    if not text:
        return ""

    def _replace(match: re.Match) -> str:
        raw = match.group(1).strip().lower()
        if not raw:
            return ""
        if raw in INLINE_TAG_MAP:
            return f"<{INLINE_TAG_MAP[raw]}>"
        if raw in INLINE_VOCAL_TAGS:
            return f"<{raw}>"
        return ""

    cleaned = _ANGLE_TAG_RE.sub(_replace, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def strip_tts_markup(text: str) -> str:
    """Remove all markup (legacy, angle-bracket tags, pipes) for comparisons."""
    if not text:
        return ""
    cleaned = _LEGACY_TAG_RE.sub(" ", text)
    cleaned = _ANGLE_TAG_RE.sub(" ", cleaned)
    cleaned = _PIPE_RE.sub(lambda m: f" {m.group(1)} ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def sanitize_style(style: Optional[str]) -> str:
    """Normalize a legacy style prompt into a compact 3.8 style string."""
    if not style:
        return ""
    cleaned = str(style).strip()
    for pattern in _META_STYLE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.endswith(":"):
        cleaned = cleaned[:-1].strip()
    key = cleaned.lower().rstrip(": ").strip()
    if key in _PRESET_STYLE_MAP:
        return _PRESET_STYLE_MAP[key]
    # Custom prompts: keep them, but short. Long "director's notes" are the
    # main cause of voice drift on 3.8.
    return cleaned[:300]


def extract_quoted_fragments(text: Optional[str]) -> List[str]:
    """Return every double-quoted fragment (used for context quotes)."""
    if not text:
        return []
    return [match.group(1).strip() for match in _QUOTED_FRAGMENT_RE.finditer(str(text)) if match.group(1).strip()]


def _content_words(text: str) -> set:
    return {match.group(0) for match in _WORD_RE.finditer(text.lower())}


def quoted_fragment_leak(
    asr_text: Optional[str],
    expected_text: Optional[str],
    context_style: Optional[str],
    *,
    min_words: int = 3,
) -> Optional[str]:
    """Detect a context quote being spoken instead of merely interpreted.

    Returns the leaked fragment (for logging/retry decisions) or ``None``.
    The check fires when at least ``min_words`` content words of a quoted
    context fragment appear in the ASR transcript but are not part of the
    expected line.
    """
    fragments = extract_quoted_fragments(context_style)
    if not fragments:
        return None
    asr_words = _content_words(strip_tts_markup(asr_text or ""))
    if not asr_words:
        return None
    expected_words = _content_words(strip_tts_markup(expected_text or ""))
    for fragment in fragments:
        quoted_words = _content_words(fragment)
        if len(quoted_words) < min_words:
            continue
        spoken = quoted_words & asr_words
        if len(spoken) >= min_words and not spoken.issubset(expected_words):
            return fragment
    return None


def build_style(
    *parts: Optional[str],
    style_hints: Optional[Iterable[str]] = None,
    max_chars: int = 280,
) -> str:
    """Join sustained style parts into a de-duplicated, length-capped string."""
    ordered: List[str] = []
    seen: set = set()

    def _add(raw: Optional[str]) -> None:
        if not raw:
            return
        for chunk in re.split(r"[;\n]+", str(raw)):
            piece = chunk.strip().rstrip(":").strip()
            if not piece:
                continue
            key = piece.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(piece)

    for part in parts:
        sanitized = sanitize_style(part)
        _add(sanitized)
    if style_hints:
        for hint in style_hints:
            _add(hint)

    joined_parts: List[str] = []
    current_length = 0
    for piece in ordered:
        candidate_length = current_length + (2 if joined_parts else 0) + len(piece)
        if candidate_length > max_chars:
            # Drop later parts (least important, e.g. context quotes) instead
            # of cutting the leading direction mid-word.
            continue
        joined_parts.append(piece)
        current_length = candidate_length

    if not joined_parts and ordered:
        piece = ordered[0][: max_chars + 1]
        cut = piece.rfind(" ")
        joined_parts.append((piece[:cut] if cut > 0 else piece[:max_chars]).strip())

    return ", ".join(joined_parts)


def build_gemini38_prompt(
    transcript: str,
    *,
    prompt_prefix: Optional[str] = None,
    speaker_prompt: Optional[str] = None,
    segment_style: Optional[str] = None,
    emotion: Optional[str] = None,
    context_style: Optional[str] = None,
    max_style_chars: int = 280,
) -> Gemini38Prompt:
    """Prepare the verbatim transcript + style for one segment."""
    converted_text, style_hints = convert_legacy_tags(transcript or "")
    # Guards against out-of-spec tags coming from the emotion enricher or
    # legacy text before the request reaches the model.
    converted_text = sanitize_angle_tags(converted_text)

    emotion_style = ""
    if emotion and str(emotion).strip().lower() not in {"", "neutral", "none"}:
        emotion_style = str(emotion).strip().lower()

    style = build_style(
        prompt_prefix,
        speaker_prompt,
        segment_style,
        emotion_style,
        # Context quote goes last so a length cap drops it before the actual
        # acting direction for the current line.
        context_style,
        style_hints=style_hints,
        max_chars=max_style_chars,
    )
    return Gemini38Prompt(text=converted_text, style=style)
