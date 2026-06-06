"""Canonical TTS style prompts for Gemini and helpers to resolve them.

A "style" is a short one-line instruction the Gemini TTS model reads before
each synthesis batch. The baked-in styles cover the most common content
types; ``custom`` falls back to a user-supplied prompt; ``auto`` is resolved
during context analysis by the LLM translator.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


PODCAST_PROMPT = (
    "Speak with natural conversational energy, clear articulation:"
)

LECTURE_PROMPT = (
    "Speak calmly and clearly, with measured pacing and an authoritative tone:"
)

GOTHIC_PROMPT = (
    "Speak slowly and deliberately, with a suspenseful, deeply resonant tone:"
)

NEWS_PROMPT = (
    "Synthesize only the transcript below in the language it is written; do not read these directions. "
    "Deliver it like a modern news explainer: confident, crisp broadcast narration with forward momentum, "
    "energetic narrative cadence, clear factual emphasis, bright but serious tone, no shouting or parody, "
    "short clean pauses between clauses:"
)


TTS_STYLE_PROMPTS: Dict[str, str] = {
    "podcast": PODCAST_PROMPT,
    "lecture": LECTURE_PROMPT,
    "gothic": GOTHIC_PROMPT,
    "news": NEWS_PROMPT,
}

BAKED_STYLE_IDS: Iterable[str] = tuple(TTS_STYLE_PROMPTS.keys())
ALL_STYLE_IDS: Iterable[str] = (*BAKED_STYLE_IDS, "custom", "auto")

DEFAULT_STYLE = "podcast"


def resolve_tts_prompt_prefix(
    style: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    resolved_style: Optional[str] = None,
) -> str:
    """Return the final prompt prefix for a given style selection.

    ``custom`` yields the user-supplied prompt (falling back to the default
    baked style when empty). ``auto`` defers to ``resolved_style`` picked by
    the LLM context analyser, again falling back to the default.
    """
    effective = (style or DEFAULT_STYLE).lower()

    if effective == "custom":
        if custom_prompt and custom_prompt.strip():
            return custom_prompt
        return TTS_STYLE_PROMPTS[DEFAULT_STYLE]

    if effective == "auto":
        picked = (resolved_style or DEFAULT_STYLE).lower()
        return TTS_STYLE_PROMPTS.get(picked, TTS_STYLE_PROMPTS[DEFAULT_STYLE])

    return TTS_STYLE_PROMPTS.get(effective, TTS_STYLE_PROMPTS[DEFAULT_STYLE])


def get_blocked_voices_for_style(
    style: Optional[str] = None,
    resolved_style: Optional[str] = None,
) -> List[str]:
    """Return voices that must be excluded from auto-matching for the style.

    No style currently blocks any voices (the previous breathy-voice blocklist
    for ``podcast`` has been lifted), so this always returns an empty list.
    Kept as a stable hook so callers don't need to branch.
    """
    return []
