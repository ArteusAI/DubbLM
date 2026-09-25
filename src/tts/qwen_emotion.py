"""Qwen-Audio-3.0-TTS emotion/style helpers for OpenRouter.

Uses only control patterns documented on:
https://funaudiollm.github.io/qwen-audio-3.0-tts/

OpenRouter flash/plus ignore the top-level ``instructions`` field, so style
must be encoded in the speech ``input`` text (inline tags and/or NL prefixes).
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence, Set

from src.dubbing.core.log_config import get_logger
from src.utils.llm_call import robust_llm_call

logger = get_logger(__name__)

# Official fine-grained tags from the Qwen-Audio-3.0-TTS demo page.
QWEN_DEMO_TAGS: tuple[str, ...] = (
    "excited",
    "panicked",
    "laughing",
    "amazed",
    "angry",
    "mischievously",
    "giggles",
    "sarcastic, speaking slowly",
)

# Spoken tokens we do not want leaked into audio when a tag is mis-read aloud.
TAG_SPEAK_RISK_WORDS: frozenset[str] = frozenset(
    {
        "excited",
        "panicked",
        "laughing",
        "amazed",
        "angry",
        "mischievously",
        "giggles",
        "sarcastic",
        "giggle",
        "laugh",
        "panic",
    }
)

# Map common pipeline emotion labels -> demo tags.
EMOTION_TO_QWEN_TAG: dict[str, str] = {
    "happy": "excited",
    "joy": "excited",
    "joyful": "excited",
    "excited": "excited",
    "amused": "laughing",
    "laugh": "laughing",
    "laughing": "laughing",
    "angry": "angry",
    "anger": "angry",
    "mad": "angry",
    "furious": "angry",
    "fear": "panicked",
    "fearful": "panicked",
    "scared": "panicked",
    "panic": "panicked",
    "panicked": "panicked",
    "surprised": "amazed",
    "surprise": "amazed",
    "amazed": "amazed",
    "shocked": "amazed",
    "mischievous": "mischievously",
    "mischievously": "mischievously",
    "playful": "mischievously",
    "sarcastic": "sarcastic, speaking slowly",
    "sarcasm": "sarcastic, speaking slowly",
}

# Soft NL instructions (demo "Instructed Voice Generation"). Used when tags are
# skipped or for style_prompt waveforms. Kept short so cache keys stay stable.
EMOTION_TO_NL_INSTRUCTION: dict[str, str] = {
    "happy": "Say this happily and warmly.",
    "joy": "Say this happily and warmly.",
    "joyful": "Say this happily and warmly.",
    "excited": "Say this with excited energy.",
    "sad": "Sadly, read the following.",
    "sadness": "Sadly, read the following.",
    "angry": "Say this angrily.",
    "anger": "Say this angrily.",
    "fear": "Say this in a fearful, hesitant tone.",
    "fearful": "Say this in a fearful, hesitant tone.",
    "scared": "Say this in a fearful, hesitant tone.",
    "panicked": "Say this in a panicked hurry.",
    "surprised": "Say the following in shock.",
    "shock": "Say the following in shock.",
    "amazed": "Say this with amazement.",
    "disgust": "Speak with revulsion.",
    "disgusted": "Speak with revulsion.",
    "sarcastic": "Read this sarcastically, speaking slowly.",
    "calm": "Say this in a slow, calming voice.",
    "neutral": "",
}

_TAG_RE = re.compile(r"\[([^\[\]]+)\]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def normalize_emotion_label(emotion: Optional[str]) -> str:
    if not emotion:
        return ""
    return re.sub(r"\s+", " ", emotion.strip().lower())


def strip_markup_tags(text: str) -> str:
    """Remove all square-bracket tags from text (for duration estimate / ASR compare)."""
    if not text:
        return text
    cleaned = _TAG_RE.sub(" ", text)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def extract_markup_tags(text: str) -> List[str]:
    return [m.group(1).strip() for m in _TAG_RE.finditer(text or "")]


def is_allowed_qwen_tag(tag_body: str) -> bool:
    body = (tag_body or "").strip().lower()
    allowed = {t.lower() for t in QWEN_DEMO_TAGS}
    return body in allowed


def sanitize_qwen_markup(text: str) -> str:
    """Keep only demo-whitelisted tags; drop unknown bracket markup."""
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        if is_allowed_qwen_tag(body):
            # Canonical casing from catalog when possible
            for canonical in QWEN_DEMO_TAGS:
                if canonical.lower() == body.lower():
                    return f"[{canonical}]"
            return f"[{body}]"
        logger.debug(f"Dropping non-demo Qwen tag: [{body}]")
        return ""

    cleaned = _TAG_RE.sub(_replace, text)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def qwen_tag_for_emotion(emotion: Optional[str]) -> Optional[str]:
    key = normalize_emotion_label(emotion)
    if not key or key in {"neutral", "none", "n/a"}:
        return None
    tag = EMOTION_TO_QWEN_TAG.get(key)
    if tag:
        return tag
    # Direct tag label allowed through.
    if is_allowed_qwen_tag(key):
        return key
    return None


def nl_instruction_for_emotion(emotion: Optional[str]) -> str:
    key = normalize_emotion_label(emotion)
    if not key:
        return ""
    return EMOTION_TO_NL_INSTRUCTION.get(key, "")


def apply_leading_tag(text: str, tag_body: Optional[str]) -> str:
    if not text or not tag_body or not is_allowed_qwen_tag(tag_body):
        return text
    existing = [t.lower() for t in extract_markup_tags(text)]
    if tag_body.lower() in existing:
        return text
    # Prefer a single leading emotion tag.
    body = next(t for t in QWEN_DEMO_TAGS if t.lower() == tag_body.lower())
    stripped = text.lstrip()
    if stripped.startswith("["):
        return f"[{body}] {text}"
    return f"[{body}] {text}"


def apply_nl_instruction_prefix(text: str, instruction: str) -> str:
    """Embed a free-style NL instruction into input text (OpenRouter-compatible)."""
    if not text or not instruction:
        return text
    instr = instruction.strip()
    if not instr:
        return text
    # Avoid double-prefix if already present.
    if text.lstrip().lower().startswith(instr.lower()):
        return text
    if not instr.endswith((".", "!", "?", ":", "。", "！", "？")):
        instr = instr.rstrip(":").strip() + "."
    return f"{instr} {text}"


def style_prompt_to_instruction(style_prompt: Optional[str]) -> str:
    if not style_prompt:
        return ""
    cleaned = style_prompt.strip()
    if not cleaned:
        return ""
    # Common DubbLM convention ends with ':' for Gemini — strip for Qwen NL form.
    cleaned = cleaned.rstrip(":").strip()
    if not cleaned:
        return ""
    # Already imperative instructions pass through.
    lower = cleaned.lower()
    if lower.startswith(("say ", "speak ", "read ", "please ", "with ")):
        return cleaned if cleaned.endswith((".", "!", "?")) else cleaned + "."
    return f"Speak with this delivery: {cleaned}."


def prepare_qwen_speech_text(
    text: str,
    *,
    emotion: Optional[str] = None,
    style_prompt: Optional[str] = None,
    prefer_tags: bool = True,
) -> str:
    """Deterministic (no-LLM) adaptation of text for Qwen OpenRouter TTS."""
    if not text:
        return text

    body = sanitize_qwen_markup(text)
    prefixes: List[str] = []

    style_instr = style_prompt_to_instruction(style_prompt)
    if style_instr:
        prefixes.append(style_instr)

    tag = qwen_tag_for_emotion(emotion) if prefer_tags else None
    if prefer_tags and tag:
        body = apply_leading_tag(body, tag)
    else:
        nl = nl_instruction_for_emotion(emotion)
        if nl:
            prefixes.append(nl if nl.endswith((".", "!", "?")) else nl.rstrip(".") + ".")

    if prefixes:
        # NL instructions first, then tagged spoken body.
        return sanitize_qwen_markup(" ".join(prefixes) + " " + body)
    return sanitize_qwen_markup(body)


def spoken_tag_leak_hits(transcript: str, speech_text: str) -> List[str]:
    """Return tag-risk words present in ASR transcript but not in plain speech body."""
    if not transcript:
        return []
    plain = strip_markup_tags(speech_text).lower()
    plain_tokens = set(re.findall(r"[a-zA-Z]+", plain))
    transcript_tokens = set(re.findall(r"[a-zA-Z]+", transcript.lower()))

    active_risk: Set[str] = set()
    for tag in extract_markup_tags(speech_text):
        for part in re.findall(r"[a-zA-Z]+", tag.lower()):
            if part in TAG_SPEAK_RISK_WORDS:
                active_risk.add(part)

    hits = []
    for word in sorted(active_risk):
        if word in transcript_tokens and word not in plain_tokens:
            hits.append(word)
    return hits


class QwenEmotionEnricher:
    """LLM enricher that injects Qwen demo-page tags only."""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.llm = None

    def initialize(self) -> None:
        try:
            from llama_index.llms.gemini import Gemini
        except ImportError:
            logger.error("llama_index.llms.gemini not available for Qwen emotion enrichment")
            return

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY/GEMINI_API_KEY not found for Qwen emotion enrichment")
            return

        try:
            self.llm = Gemini(model=self.model, temperature=self.temperature)
            logger.info(f"Qwen emotion enricher LLM initialized: {self.model}")
        except Exception as exc:
            logger.error(f"Failed to initialize Qwen emotion enricher LLM: {exc}")
            self.llm = None

    @property
    def enrichment_prompt(self) -> str:
        tags_csv = ", ".join(f"[{t}]" for t in QWEN_DEMO_TAGS)
        return f"""You enrich dialogue lines for Qwen-Audio-3.0-TTS (OpenRouter).

Allowed inline tags ONLY (from the official Qwen demo). Never invent other tags:
{tags_csv}

Rules:
- Preserve original wording and meaning. Only add/adjust square-bracket tags.
- Use tags sparingly. Most lines should stay untagged.
- Prefer at most one leading emotion/style tag per line.
- You MAY combine demo tags when the demo does (e.g. [excited] ... [laughing], [mischievously][giggles]).
- Do NOT use Gemini-only tags ([uhm], [robotic], [short pause], [whispering], etc.).
- Do NOT add free-form natural-language stage directions outside tags.
- If the text is already well-tagged with allowed tags, keep or lightly improve them.
- Return ONLY the enriched current line. No explanations.

{{context_section}}
Current text to enrich:
{{text}}
"""

    def enrich_text(self, text: str, previous_segments: Optional[Sequence[str]] = None) -> str:
        if not text or not text.strip():
            return text
        if not self.llm:
            logger.debug("Qwen emotion enricher: LLM unavailable, skipping")
            return prepare_qwen_speech_text(text)

        context_section = ""
        if previous_segments:
            lines = "\n".join(f"- {seg}" for seg in list(previous_segments)[-5:])
            context_section = f"Previous conversation context:\n{lines}\n"

        prompt = self.enrichment_prompt.format(context_section=context_section, text=text)
        try:
            response = robust_llm_call(self.llm.complete, prompt)
            response_text = response.text if hasattr(response, "text") else str(response)
            if not response_text:
                return prepare_qwen_speech_text(text)

            enriched = sanitize_qwen_markup(response_text.strip())
            if not enriched:
                return prepare_qwen_speech_text(text)
            if len(enriched) > max(len(text) * 3, len(text) + 80):
                logger.warning("Qwen emotion enricher: output too long, using deterministic prep")
                return prepare_qwen_speech_text(text)

            # Keep plain-body close to original (tags aside).
            if strip_markup_tags(enriched).lower() and not _bodies_roughly_match(text, enriched):
                logger.warning("Qwen emotion enricher: body drifted too far, using deterministic prep")
                return prepare_qwen_speech_text(text)

            return enriched
        except Exception as exc:
            logger.error(f"Qwen emotion enrichment failed: {exc}")
            return prepare_qwen_speech_text(text)


def _bodies_roughly_match(original: str, enriched: str) -> bool:
    """Loose check that enrichment didn't rewrite the spoken line."""
    a = re.sub(r"\W+", "", strip_markup_tags(original).lower())
    b = re.sub(r"\W+", "", strip_markup_tags(enriched).lower())
    if not a or not b:
        return False
    # Allow minor punctuation/spacing differences; require high character overlap.
    if a in b or b in a:
        return True
    # Token Jaccard
    ta, tb = set(a), set(b)
    if not ta or not tb:
        return False
    # character multi-set-ish via set of digrams
    da = {a[i : i + 3] for i in range(max(1, len(a) - 2))}
    db = {b[i : i + 3] for i in range(max(1, len(b) - 2))}
    inter = len(da & db)
    union = len(da | db) or 1
    return (inter / union) >= 0.55
