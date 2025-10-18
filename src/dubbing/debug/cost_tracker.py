"""Cost tracking and estimation for the Smart Dubbing pipeline.

This module provides a CostTracker that estimates and records costs for
external services used in the pipeline, such as transcription APIs,
LLM-based translation, and cloud TTS providers.

Pricing is configurable via DubbingConfig under the `pricing` key. All
values default to 0, so if you don't configure pricing, costs will be 0.

Example pricing config (YAML):

pricing:
  transcription:
    assemblyai:
      per_min: 0.0025
    openai:
      per_min: 0.006
  translation:
    gemini:
      input_per_1m_tokens: 3.00
      output_per_1m_tokens: 6.00
    openrouter:
      models:
        anthropic/claude-3.7-sonnet:thinking:
          input_per_1m_tokens: 3.50
          output_per_1m_tokens: 7.00
  tts:
    gemini:
      models:
        gemini-2.5-flash-preview-tts:
          input_per_1m_tokens: 0.50
          output_per_1m_tokens: 10.00
        gemini-2.5-pro-preview-tts:
          input_per_1m_tokens: 1.00
          output_per_1m_tokens: 20.00
      per_audio_min: null
    openai:
      input_per_1m_tokens: 0.6
      output_per_1m_tokens: 12
      per_audio_min: null
"""

from __future__ import annotations

from typing import Dict, Optional, Any, Tuple

from ..core.log_config import get_logger

logger = get_logger(__name__)


def _get_nested(d: Dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    cur: Any = d or {}
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


class CostTracker:
    """Tracks estimated and calculated costs for pipeline steps."""

    def __init__(self, config: Any):
        # Raw pricing configuration (dict-like access via DubbingConfig)
        self.pricing: Dict[str, Any] = config.get("pricing", {}) or {}

        # Accumulators
        self.estimate: Dict[str, float] = {
            "transcription": 0.0,
            "translation": 0.0,
            "tts": 0.0,
            "total": 0.0,
        }
        self.actual: Dict[str, float] = {
            "transcription": 0.0,
            "translation": 0.0,
            "tts": 0.0,
            "total": 0.0,
        }

        # Context helpers
        self.audio_duration_sec: Optional[float] = None

        # For detailed usage tracking (best-effort)
        self.usage: Dict[str, Dict[str, float]] = {
            "translation": {"input_tokens": 0.0, "output_tokens": 0.0, "details": {}},
            "tts": {"input_tokens": 0.0, "output_tokens": 0.0, "audio_seconds": 0.0, "details": {}},
            "transcription": {"audio_seconds": 0.0},
        }

    # ----- Helpers -----
    def set_audio_duration(self, seconds: Optional[float]) -> None:
        self.audio_duration_sec = seconds

    def _get_translation_rates(self, provider: str, model: Optional[str]) -> Tuple[float, float]:
        """Retrieve input/output pricing (per million tokens) for a translation provider/model combo."""
        translation_cfg = self.pricing.get("translation")
        if not isinstance(translation_cfg, dict):
            raise ValueError("Translation pricing configuration is missing.")

        provider_cfg = translation_cfg.get(provider)
        if not isinstance(provider_cfg, dict):
            raise ValueError(f"Missing translation pricing for provider '{provider}'.")

        if model:
            models_cfg = provider_cfg.get("models")
            if isinstance(models_cfg, dict):
                model_cfg = models_cfg.get(model)
                if not model_cfg and ":" in model:
                    base_model = model.split(":")[0]
                    model_cfg = models_cfg.get(base_model)
                if isinstance(model_cfg, dict):
                    in_rate = model_cfg.get("input_per_1m_tokens")
                    out_rate = model_cfg.get("output_per_1m_tokens")
                    if in_rate is None or out_rate is None:
                        raise ValueError(
                            f"Translation pricing for {provider} model '{model}' must define input/output per_1m_tokens."
                        )
                    return float(in_rate), float(out_rate)

        in_rate = provider_cfg.get("input_per_1m_tokens")
        out_rate = provider_cfg.get("output_per_1m_tokens")
        if in_rate is None or out_rate is None:
            raise ValueError(f"Translation pricing for provider '{provider}' must define input/output per_1m_tokens.")
        return float(in_rate), float(out_rate)

    def _get_tts_rates(self, provider: str, model: Optional[str]) -> Tuple[float, float, Optional[float]]:
        """Retrieve TTS pricing (per million tokens and per audio minute)."""
        tts_cfg = self.pricing.get("tts")
        if not isinstance(tts_cfg, dict):
            raise ValueError("TTS pricing configuration is missing.")

        provider_cfg = tts_cfg.get(provider)
        if not isinstance(provider_cfg, dict):
            raise ValueError(f"Missing TTS pricing for provider '{provider}'.")

        input_rate = provider_cfg.get("input_per_1m_tokens")
        output_rate = provider_cfg.get("output_per_1m_tokens")

        if model:
            models_cfg = provider_cfg.get("models")
            if isinstance(models_cfg, dict):
                model_cfg = models_cfg.get(model)
                if not model_cfg and ":" in model:
                    base_model = model.split(":")[0]
                    model_cfg = models_cfg.get(base_model)
                if isinstance(model_cfg, dict):
                    input_rate = model_cfg.get("input_per_1m_tokens", input_rate)
                    output_rate = model_cfg.get("output_per_1m_tokens", output_rate)

        if input_rate is None and output_rate is None:
            raise ValueError(f"TTS pricing for provider '{provider}' must define at least one of input_per_1m_tokens or output_per_1m_tokens.")
        per_audio_min = provider_cfg.get("per_audio_min")
        return (
            float(input_rate or 0.0),
            float(output_rate or 0.0),
            None if per_audio_min is None else float(per_audio_min),
        )

    # ----- Transcription -----
    def estimate_transcription_cost(self, provider: str, audio_seconds: Optional[float]) -> float:
        if not audio_seconds or audio_seconds <= 0:
            return 0.0
        per_min = float(_get_nested(self.pricing, "transcription", provider, "per_min", default=0.0))
        cost = per_min * (audio_seconds / 60.0)
        self.estimate["transcription"] += cost
        self.estimate["total"] += cost
        self.usage["transcription"]["audio_seconds"] += float(audio_seconds or 0.0)
        logger.info(f"Estimated transcription cost [{provider}]: ${cost:.4f} for {audio_seconds/60.0:.2f} min")
        return cost

    def add_transcription_actual(self, provider: str, audio_seconds: Optional[float]) -> float:
        if not audio_seconds or audio_seconds <= 0:
            return 0.0
        per_min = float(_get_nested(self.pricing, "transcription", provider, "per_min", default=0.0))
        cost = per_min * (audio_seconds / 60.0)
        self.actual["transcription"] += cost
        self.actual["total"] += cost
        self.usage["transcription"]["audio_seconds"] += float(audio_seconds)
        logger.info(f"Calculated transcription cost [{provider}]: ${cost:.4f}")
        return cost

    # ----- Translation (LLM) -----
    def estimate_translation_cost(self, provider: str, model: Optional[str], input_tokens: float, expected_output_tokens: Optional[float] = None) -> float:
        if input_tokens <= 0:
            return 0.0
        if expected_output_tokens is None:
            expected_output_tokens = input_tokens

        in_rate, out_rate = self._get_translation_rates(provider, model)
        cost = in_rate * (float(input_tokens) / 1_000_000.0) + out_rate * (float(expected_output_tokens) / 1_000_000.0)

        self.estimate["translation"] += cost
        self.estimate["total"] += cost
        self.usage["translation"]["input_tokens"] += float(input_tokens or 0.0)
        self.usage["translation"]["output_tokens"] += float(expected_output_tokens or 0.0)
        detail_map = self.usage["translation"].setdefault("details", {})
        detail_entry = detail_map.setdefault(f"{provider}:{model or 'default'}", {"input_tokens": 0.0, "output_tokens": 0.0})
        detail_entry["input_tokens"] += float(input_tokens or 0.0)
        detail_entry["output_tokens"] += float(expected_output_tokens or 0.0)
        logger.info(
            f"Estimated translation cost [{provider}#{model or 'default'}]: "
            f"${cost:.4f} (in≈{input_tokens:.0f} tok, out≈{expected_output_tokens:.0f} tok)"
        )
        return cost

    def add_translation_actual(self, provider: str, model: Optional[str], input_tokens: float, output_tokens: float) -> float:
        input_tokens = float(input_tokens or 0.0)
        output_tokens = float(output_tokens or 0.0)

        in_rate, out_rate = self._get_translation_rates(provider, model)
        cost = in_rate * (input_tokens / 1_000_000.0) + out_rate * (output_tokens / 1_000_000.0)
        self.actual["translation"] += cost
        self.actual["total"] += cost
        self.usage["translation"]["input_tokens"] += input_tokens
        self.usage["translation"]["output_tokens"] += output_tokens

        if provider and model:
            detail_map = self.usage["translation"].setdefault("details", {})
            detail_key = f"{provider}:{model}"
            entry = detail_map.setdefault(detail_key, {"input_tokens": 0.0, "output_tokens": 0.0})
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens

        logger.info(
            f"Calculated translation cost [{provider}#{model or 'default'}]: "
            f"${cost:.4f} (in≈{input_tokens:.0f}, out≈{output_tokens:.0f})"
        )
        return cost

    # ----- TTS -----
    def estimate_tts_cost(
        self,
        provider: str,
        model: Optional[str],
        input_tokens: float,
        expected_audio_seconds: Optional[float] = None,
        expected_output_tokens: Optional[float] = None
    ) -> float:
        input_rate, output_rate, per_audio_min = self._get_tts_rates(provider, model)

        cost_input = input_rate * (float(input_tokens or 0.0) / 1_000_000.0) if input_rate else 0.0
        cost_output = output_rate * (float(expected_output_tokens or 0.0) / 1_000_000.0) if output_rate else 0.0
        cost_audio = 0.0
        if expected_audio_seconds and per_audio_min:
            cost_audio = float(per_audio_min) * (float(expected_audio_seconds) / 60.0)

        cost = cost_input + cost_output + cost_audio
        self.estimate["tts"] += cost
        self.estimate["total"] += cost
        self.usage["tts"]["input_tokens"] += float(input_tokens or 0.0)
        self.usage["tts"]["output_tokens"] += float(expected_output_tokens or 0.0)
        self.usage["tts"]["audio_seconds"] += float(expected_audio_seconds or 0.0)
        detail_map = self.usage["tts"].setdefault("details", {})
        detail_entry = detail_map.setdefault(f"{provider}:{model or 'default'}", {"input_tokens": 0.0, "output_tokens": 0.0, "audio_seconds": 0.0})
        detail_entry["input_tokens"] += float(input_tokens or 0.0)
        detail_entry["output_tokens"] += float(expected_output_tokens or 0.0)
        detail_entry["audio_seconds"] += float(expected_audio_seconds or 0.0)
        logger.info(f"Estimated TTS cost [{provider}#{model or 'default'}]: ${cost:.4f}")
        return cost

    def add_tts_actual(
        self,
        provider: str,
        model: Optional[str] = None,
        input_tokens: float = 0.0,
        output_tokens: float = 0.0,
        audio_seconds: float = 0.0
    ) -> float:
        input_rate, output_rate, per_audio_min = self._get_tts_rates(provider, model)

        cost_input = input_rate * (float(input_tokens or 0.0) / 1_000_000.0) if input_rate else 0.0
        cost_output = output_rate * (float(output_tokens or 0.0) / 1_000_000.0) if output_rate else 0.0
        cost_audio = float(per_audio_min) * (float(audio_seconds or 0.0) / 60.0) if (audio_seconds and per_audio_min) else 0.0

        cost = cost_input + cost_output + cost_audio
        self.actual["tts"] += cost
        self.actual["total"] += cost
        self.usage["tts"]["input_tokens"] += float(input_tokens or 0.0)
        self.usage["tts"]["output_tokens"] += float(output_tokens or 0.0)
        self.usage["tts"]["audio_seconds"] += float(audio_seconds or 0.0)

        if provider:
            detail_key = f"{provider}:{model or 'default'}"
            details = self.usage["tts"].setdefault("details", {})
            entry = details.setdefault(detail_key, {"input_tokens": 0.0, "output_tokens": 0.0, "audio_seconds": 0.0})
            entry["input_tokens"] += float(input_tokens or 0.0)
            entry["output_tokens"] += float(output_tokens or 0.0)
            entry["audio_seconds"] += float(audio_seconds or 0.0)

        logger.info(f"Calculated TTS cost [{provider}#{model or 'default'}]: ${cost:.4f}")
        return cost

    # ----- Summary -----
    def get_costs_by_step(self) -> Dict[str, float]:
        """Return actual costs mapped to pipeline step names."""
        return {
            "transcription": self.actual.get("transcription", 0.0),
            "translation": self.actual.get("translation", 0.0),
            "speech_synthesis": self.actual.get("tts", 0.0),
            "total": self.actual.get("total", 0.0),
        }

    def get_estimated_costs(self) -> Dict[str, float]:
        """Return estimated costs mapped to pipeline step names."""
        return {
            "transcription": self.estimate.get("transcription", 0.0),
            "translation": self.estimate.get("translation", 0.0),
            "speech_synthesis": self.estimate.get("tts", 0.0),
            "total": self.estimate.get("total", 0.0),
        }

    def write_cost_summary(self) -> None:
        """Log a summary of estimated vs calculated costs."""
        logger.info("Cost summary:")
        lines = []
        lines.append(f"{'Step':<18} {'Estimated($)':>14} {'Calculated($)':>16}")
        lines.append("-" * 50)
        for step in ("transcription", "translation", "tts"):
            est = self.estimate.get(step, 0.0)
            act = self.actual.get(step, 0.0)
            lines.append(f"{step.title():<18} {est:>14.4f} {act:>16.4f}")
        lines.append("-" * 50)
        lines.append(f"{'Total':<18} {self.estimate.get('total', 0.0):>14.4f} {self.actual.get('total', 0.0):>16.4f}")

        # Usage snippet (best-effort)
        t_in = self.usage["translation"].get("input_tokens", 0.0)
        t_out = self.usage["translation"].get("output_tokens", 0.0)
        tts_in = self.usage["tts"].get("input_tokens", 0.0)
        tts_out = self.usage["tts"].get("output_tokens", 0.0)
        tts_sec = self.usage["tts"].get("audio_seconds", 0.0)
        tr_sec = self.usage["transcription"].get("audio_seconds", 0.0)

        lines.append("")
        lines.append("Usage details (approx):")
        lines.append(f" - Translation tokens: in≈{t_in:.0f}, out≈{t_out:.0f}")
        lines.append(f" - TTS tokens: in≈{tts_in:.0f}, out≈{tts_out:.0f}, audio≈{tts_sec/60.0:.2f} min")
        lines.append(f" - Transcription audio: ≈{tr_sec/60.0:.2f} min")

        for line in lines:
            logger.info(line)
