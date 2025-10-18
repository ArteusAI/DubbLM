from __future__ import annotations

import math
import os
import subprocess
from typing import Optional, Dict, Any

import tiktoken

from ..debug.cost_tracker import CostTracker
from .log_config import get_logger
from translation.prompts import (
    REFINEMENT_PROMPTS,
    CONTEXT_ANALYSIS_PROMPT_TEMPLATE,
    TRANSLATION_PROMPT_TEMPLATE,
)


logger = get_logger(__name__)


class CostEstimator:
    """Estimate pipeline costs without running the full dubbing flow."""

    _AVERAGE_SPOKEN_TOKENS_PER_SECOND = 3.0
    _TRANSLATION_CHUNK_TOKEN_TARGET = 1200  # rough heuristic

    def __init__(self, config: Any):
        self._config = config
        self._tracker = CostTracker(config)
        self._prompt_token_cache: Dict[str, int] = {}

    def estimate(self) -> Dict[str, float]:
        """Estimate total and per-step costs. Returns a dict of costs."""
        video_path = self._config.get("input")
        duration_seconds = self._probe_video_duration(video_path)
        self._tracker.set_audio_duration(duration_seconds)

        logger.info(
            "Video duration: %.2f seconds (%.2f minutes)",
            duration_seconds,
            duration_seconds / 60.0,
        )

        self._estimate_transcription(duration_seconds)

        estimated_tokens = max(
            0.0, duration_seconds * self._AVERAGE_SPOKEN_TOKENS_PER_SECOND
        )
        logger.info(
            "Estimating translation tokens: %.0f (assumes %.2f tokens/second)",
            estimated_tokens,
            self._AVERAGE_SPOKEN_TOKENS_PER_SECOND,
        )

        self._estimate_translation(estimated_tokens)
        self._estimate_tts(duration_seconds, estimated_tokens)

        self._tracker.write_cost_summary()
        costs = self._tracker.get_estimated_costs()

        logger.info("Estimated cost (USD):")
        for step in ("transcription", "translation", "speech_synthesis", "total"):
            logger.info(
                "  %s: $%.4f", step.replace("_", " ").title(), costs.get(step, 0.0)
            )

        return costs

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _probe_video_duration(self, video_path: str) -> float:
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]

        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffprobe is required for cost estimation but was not found in PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"ffprobe failed to determine video duration: "
                f"{exc.output.decode().strip()}"
            ) from exc

        try:
            return float(output.decode().strip())
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Could not parse video duration from ffprobe output.") from exc

    def _estimate_transcription(self, duration_seconds: float) -> None:
        provider = self._resolve_transcription_provider()
        try:
            self._tracker.estimate_transcription_cost(provider, duration_seconds)
        except ValueError as exc:
            logger.warning("Skipping transcription cost estimation: %s", exc)

    def _estimate_translation(self, estimated_tokens: float) -> None:
        provider = (self._config.get("llm_provider") or "gemini").lower()
        model_name = self._resolve_llm_model_name(provider)
        encoding_model = model_name or self._default_model_for_provider(provider)
        chunk_count = self._estimate_chunk_count(estimated_tokens)

        context_prompt_tokens = self._get_context_prompt_tokens(encoding_model)
        translation_prompt_tokens = (
            chunk_count * self._get_translation_prompt_tokens(encoding_model)
        )

        translation_input_tokens = (
            estimated_tokens + context_prompt_tokens + translation_prompt_tokens
        )
        try:
            self._tracker.estimate_translation_cost(
                provider, model_name, translation_input_tokens, estimated_tokens
            )
        except ValueError as exc:
            logger.warning("Skipping translation cost estimation: %s", exc)

        refinement_provider = self._resolve_refinement_provider(provider)
        refinement_model = self._resolve_refinement_model_name(
            refinement_provider, model_name
        )
        refinement_encoding_model = (
            refinement_model or self._default_model_for_provider(refinement_provider)
        )
        refinement_prompt_tokens = (
            chunk_count * self._get_refinement_prompt_tokens(refinement_encoding_model)
        )
        refinement_input_tokens = estimated_tokens + refinement_prompt_tokens
        try:
            self._tracker.estimate_translation_cost(
                refinement_provider,
                refinement_model,
                refinement_input_tokens,
                estimated_tokens,
            )
        except ValueError as exc:
            logger.warning(
                "Skipping refinement translation cost estimation: %s", exc
            )

    def _estimate_tts(self, duration_seconds: float, estimated_tokens: float) -> None:
        tts_system = (self._config.get("tts_system") or "").lower()
        if tts_system not in ("gemini", "openai"):
            logger.info(
                "TTS system '%s' has no pricing configured; skipping TTS estimate.",
                tts_system,
            )
            return

        model_name = self._resolve_tts_model_name(tts_system)
        try:
            self._tracker.estimate_tts_cost(
                tts_system,
                model_name,
                estimated_tokens,
                expected_audio_seconds=duration_seconds,
                expected_output_tokens=estimated_tokens,
            )
        except ValueError as exc:
            logger.warning("Skipping TTS cost estimation: %s", exc)

    # ------------------------------------------------------------------ #
    # Resolvers
    # ------------------------------------------------------------------ #

    def _resolve_transcription_provider(self) -> str:
        system = (self._config.get("transcription_system") or "whisper").lower()
        if system in ("assemblyai", "openai", "whisper", "whisperx", "pyannote_openai"):
            return "openai" if system == "pyannote_openai" else system
        return system

    def _resolve_llm_model_name(self, provider: str) -> Optional[str]:
        model = self._config.get("llm_model_name")
        if model:
            return model
        return self._default_model_for_provider(provider)

    def _resolve_refinement_provider(self, default_provider: str) -> str:
        value = self._config.get("refinement_llm_provider")
        if isinstance(value, str) and value.strip():
            return value.lower()
        return default_provider

    def _resolve_refinement_model_name(
        self, provider: str, default_model: Optional[str]
    ) -> Optional[str]:
        model = self._config.get("refinement_model_name")
        if model:
            return model
        if provider == (self._config.get("llm_provider") or "gemini").lower():
            return default_model
        return self._default_model_for_provider(provider)

    def _resolve_tts_model_name(self, system: str) -> Optional[str]:
        model = self._config.get("tts_model")
        if model:
            return model
        if system == "gemini":
            fallback = self._config.get("tts_fallback_model")
            return fallback or "gemini-2.5-pro-preview-tts"
        if system == "openai":
            return "tts-1"
        return None

    def _estimate_chunk_count(self, estimated_tokens: float) -> int:
        return max(
            1,
            math.ceil(
                estimated_tokens / float(self._TRANSLATION_CHUNK_TOKEN_TARGET)
            ),
        )

    # ------------------------------------------------------------------ #
    # Prompt token helpers
    # ------------------------------------------------------------------ #

    def _count_tokens(self, text: str, model_name: Optional[str]) -> int:
        if not text:
            return 0

        encoding = None
        if model_name:
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                encoding = None
        if encoding is None:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _get_context_prompt_tokens(self, model_name: Optional[str]) -> int:
        cache_key = f"context::{model_name}"
        if cache_key in self._prompt_token_cache:
            return self._prompt_token_cache[cache_key]

        source_language = self._config.get("source_language") or "source"
        target_language = self._config.get("target_language") or "target"

        glossary_section = ""
        glossary = self._config.get("glossary")
        if glossary:
            glossary_entries = "\n".join(
                [f'- "{term}" → "{translation}"' for term, translation in glossary.items()]
            )
            glossary_section = f"""
# Translation glossary (MUST be followed. Adapt for grammar):
<glossary>
{glossary_entries}
</glossary>

CRITICAL: You MUST use the translations from the glossary for all listed terms.
IMPORTANT: The glossary provides base forms of translations. When using a term from the glossary, you MUST adapt it to fit the grammatical context (e.g., case, gender, number, verb conjugation) of the sentence in the target language "{target_language}".
"""

        additional_context_section = ""
        prompt_prefix = self._config.get("translation_prompt_prefix") or self._config.get(
            "prompt_prefix"
        )
        if prompt_prefix:
            additional_context_section = f"""
# Additional context (optional):
{prompt_prefix}
"""

        prompt = CONTEXT_ANALYSIS_PROMPT_TEMPLATE.format(
            source_language=source_language,
            target_language=target_language,
            glossary_section=glossary_section,
            additional_context_section=additional_context_section,
            transcript_body="",
        )

        tokens = self._count_tokens(prompt, model_name)
        self._prompt_token_cache[cache_key] = tokens
        return tokens

    def _get_translation_prompt_tokens(self, model_name: Optional[str]) -> int:
        cache_key = f"translation::{model_name}"
        if cache_key in self._prompt_token_cache:
            return self._prompt_token_cache[cache_key]

        domain = "general"
        tone = "neutral"
        themes = "conversation"
        terminology = "terminology"

        glossary_section = ""
        glossary = self._config.get("glossary")
        if glossary:
            glossary_entries = "\n".join(
                [f'- "{term}" → "{translation}"' for term, translation in glossary.items()]
            )
            target_language = self._config.get("target_language") or "target"
            glossary_section = f"""
# Translation glossary (MUST be followed. Adapt for grammar):
<glossary>
{glossary_entries}
</glossary>

CRITICAL: You MUST use the translations from the glossary for all listed terms.
IMPORTANT: The glossary provides base forms of translations. When using a term from the glossary, you MUST adapt it to fit the grammatical context (e.g., case, gender, number, verb conjugation) of the sentence in the target language "{target_language}".
"""

        prompt_prefix = self._config.get("translation_prompt_prefix") or self._config.get(
            "prompt_prefix"
        )
        custom_section = ""
        if prompt_prefix:
            custom_section = f"""
# Additional context (optional):
{prompt_prefix}
"""

        source_language = self._config.get("source_language") or "source"
        target_language = self._config.get("target_language") or "target"

        prompt = TRANSLATION_PROMPT_TEMPLATE.format(
            domain=domain,
            source_language=source_language,
            target_language=target_language,
            glossary_section=glossary_section,
            custom_section=custom_section,
            tone=tone,
            themes=themes,
            terminology=terminology,
            context_before="",
            text_to_translate="",
            context_after="",
            summary_section="",
        )

        tokens = self._count_tokens(prompt, model_name)
        self._prompt_token_cache[cache_key] = tokens
        return tokens

    def _get_refinement_prompt_tokens(self, model_name: Optional[str]) -> int:
        persona = self._config.get("refinement_persona", "normal")
        template = REFINEMENT_PROMPTS.get(persona) or REFINEMENT_PROMPTS["normal"]
        cache_key = f"refinement::{persona}::{model_name}"
        if cache_key in self._prompt_token_cache:
            return self._prompt_token_cache[cache_key]

        placeholder_values = {
            "domain": "general",
            "tone": "neutral",
            "themes": "",
            "terminology": "",
            "source_language": self._config.get("source_language") or "source",
            "target_language": self._config.get("target_language") or "target",
            "dialogue_summary": "",
            "glossary_section": "",
            "previous_chunk_context": "",
            "original_conversation_text": "",
            "next_chunk_context": "",
            "translated_conversation_text": "",
        }

        prompt = template.format(**placeholder_values)
        tokens = self._count_tokens(prompt, model_name)
        self._prompt_token_cache[cache_key] = tokens
        return tokens

    @staticmethod
    def _default_model_for_provider(provider: str) -> Optional[str]:
        if provider == "gemini":
            return "models/gemini-2.5-flash-preview-04-17"
        if provider == "openrouter":
            return "anthropic/claude-3.7-sonnet:thinking"
        return None
