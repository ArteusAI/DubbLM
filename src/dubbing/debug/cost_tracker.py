"""Cost tracking and estimation for the Smart Dubbing pipeline.

This module provides a CostTracker that estimates and records costs for
external services used in the pipeline, such as transcription APIs,
LLM-based translation, and cloud TTS providers.

Pricing is configurable via DubbingConfig under the `pricing` key. When no
override is provided, the tracker falls back to a small built-in catalog for
the paid APIs DubbLM uses. OpenRouter model pricing is resolved from the
public `/api/v1/models` endpoint when possible, with built-in fallback rates
for common presets.

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
          reasoning_per_1m_tokens: 14.00
  tts:
    gemini:
      models:
        gemini-2.5-pro-preview-tts:
          input_per_1m_tokens: 0.50
          output_per_1m_tokens: 10.00
      per_audio_min: null
    openai:
      input_per_1m_tokens: 0.6
      output_per_1m_tokens: 12
      per_audio_min: null
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict, Optional, Any, Tuple, List

from ..core.log_config import get_logger

logger = get_logger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_TIMEOUT_SECONDS = 5

LOCAL_TRANSCRIPTION_PROVIDERS = {"whisper", "whisperx"}

CATEGORY_STAGE_MAP: Dict[str, str] = {
    "primary_transcription": "transcription",
    "transcription": "transcription",
    "context_analysis": "context_analysis",
    "translation": "translation",
    "refinement": "translation",
    "editor_pass": "translation",
    "emotion_enrichment": "emotion_analysis",
    "tts_synthesis": "speech_synthesis",
    "tts_batch_alignment": "speech_synthesis",
    "tts_content_validation": "speech_synthesis",
    "voice_cloning": "speech_synthesis",
}

CATEGORY_LABELS: Dict[str, str] = {
    "primary_transcription": "Primary transcription",
    "transcription": "Transcription",
    "context_analysis": "Context analysis",
    "translation": "Translation",
    "refinement": "Refinement",
    "editor_pass": "Editor pass",
    "emotion_enrichment": "Emotion enrichment",
    "tts_synthesis": "TTS synthesis",
    "tts_batch_alignment": "TTS batch alignment ASR",
    "tts_content_validation": "TTS validation ASR",
    "voice_cloning": "Voice cloning",
}

MODEL_ALIASES: Dict[str, str] = {
    "gemini-flash-latest": "gemini-2.5-flash",
    "gemini-flash-lite-latest": "gemini-2.5-flash-lite",
}

# Rates are USD. Token rates are per 1M tokens unless noted otherwise.
DEFAULT_PRICING_CATALOG: Dict[str, Any] = {
    "transcription": {
        "assemblyai": {
            "per_min": 0.15 / 60.0,
            "models": {
                "best": {"per_min": 0.21 / 60.0},
                "universal-3-pro": {"per_min": 0.21 / 60.0},
                "nano": {"per_min": 0.15 / 60.0},
                "universal": {"per_min": 0.15 / 60.0},
                "universal-2": {"per_min": 0.15 / 60.0},
            },
            "addons": {
                "speaker_diarization_per_min": 0.02 / 60.0,
            },
        },
        "openai": {
            "per_min": 0.006,
            "models": {
                "whisper-1": {"per_min": 0.006},
                "gpt-4o-transcribe": {"per_min": 0.006},
                "gpt-4o-transcribe-diarize": {"per_min": 0.006},
                "gpt-4o-mini-transcribe": {"per_min": 0.003},
            },
        },
    },
    "translation": {
        "gemini": {
            "models": {
                "gemini-2.5-pro": {
                    "input_per_1m_tokens": 1.25,
                    "output_per_1m_tokens": 10.0,
                    "reasoning_per_1m_tokens": 10.0,
                },
                "gemini-2.5-flash": {
                    "input_per_1m_tokens": 0.30,
                    "output_per_1m_tokens": 2.50,
                    "reasoning_per_1m_tokens": 2.50,
                },
                "gemini-2.5-flash-lite": {
                    "input_per_1m_tokens": 0.10,
                    "output_per_1m_tokens": 0.40,
                    "reasoning_per_1m_tokens": 0.40,
                },
                "gemini-2.5-flash-lite-preview-09-2025": {
                    "input_per_1m_tokens": 0.10,
                    "output_per_1m_tokens": 0.40,
                    "reasoning_per_1m_tokens": 0.40,
                },
                "gemini-2.0-flash": {
                    "input_per_1m_tokens": 0.10,
                    "output_per_1m_tokens": 0.40,
                    "reasoning_per_1m_tokens": 0.40,
                },
                "gemini-2.0-flash-lite": {
                    "input_per_1m_tokens": 0.01875,
                    "output_per_1m_tokens": 0.075,
                    "reasoning_per_1m_tokens": 0.075,
                },
            }
        },
        "openrouter": {
            "models": {
                "openai/gpt-5.4": {
                    "input_per_1m_tokens": 2.50,
                    "output_per_1m_tokens": 15.00,
                    "reasoning_per_1m_tokens": 15.00,
                },
                "openai/gpt-5.4-mini": {
                    "input_per_1m_tokens": 0.75,
                    "output_per_1m_tokens": 4.50,
                    "reasoning_per_1m_tokens": 4.50,
                },
                "openai/gpt-5.5": {
                    "input_per_1m_tokens": 5.00,
                    "output_per_1m_tokens": 30.00,
                    "reasoning_per_1m_tokens": 30.00,
                },
                "openai/gpt-5.5-pro": {
                    "input_per_1m_tokens": 30.00,
                    "output_per_1m_tokens": 180.00,
                    "reasoning_per_1m_tokens": 180.00,
                },
            }
        },
    },
    "tts": {
        "gemini": {
            "models": {
                "gemini-2.5-pro-preview-tts": {
                    "input_per_1m_tokens": 1.00,
                    "output_per_1m_tokens": 20.00,
                    "audio_output_tokens_per_second": 25.0,
                },
                "gemini-2.5-flash-preview-tts": {
                    "input_per_1m_tokens": 0.50,
                    "output_per_1m_tokens": 10.00,
                    "audio_output_tokens_per_second": 25.0,
                },
                "gemini-3.1-flash-tts-preview": {
                    "input_per_1m_tokens": 1.00,
                    "output_per_1m_tokens": 20.00,
                    "audio_output_tokens_per_second": 25.0,
                },
            }
        },
        "openai": {
            "models": {
                "gpt-4o-mini-tts": {
                    "input_per_1m_tokens": 0.60,
                    "output_per_1m_tokens": 12.00,
                    "per_audio_min": 0.015,
                },
                "tts-1": {"per_1m_chars": 15.00},
                "tts-1-hd": {"per_1m_chars": 30.00},
            }
        },
        "minimax": {
            "voice_clone_per_voice": 3.00,
            "models": {
                "speech-02-hd": {"per_1m_chars": 100.00},
                "speech-2.6-hd": {"per_1m_chars": 100.00},
                "speech-2.8-hd": {"per_1m_chars": 100.00},
                "speech-02-turbo": {"per_1m_chars": 60.00},
                "speech-2.6-turbo": {"per_1m_chars": 60.00},
                "speech-2.8-turbo": {"per_1m_chars": 60.00},
            }
        },
    },
}


def _get_nested(d: Dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    cur: Any = d or {}
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _model_candidates(model: Optional[str]) -> List[str]:
    if not model:
        return []
    raw = str(model).strip()
    candidates = [raw]
    alias = MODEL_ALIASES.get(raw)
    if alias and alias not in candidates:
        candidates.append(alias)
    if ":" in raw:
        base = raw.split(":", 1)[0]
        if base and base not in candidates:
            candidates.append(base)
    return candidates


def _stage_for_category(category: str) -> str:
    return CATEGORY_STAGE_MAP.get(category, category)


def _label_for_category(category: str) -> str:
    return CATEGORY_LABELS.get(category, category.replace("_", " ").title())


class CostTracker:
    """Tracks estimated and calculated costs for pipeline steps."""

    def __init__(self, config: Any):
        # Raw pricing configuration (dict-like access via DubbingConfig)
        self.pricing: Dict[str, Any] = config.get("pricing", {}) or {}
        self._openrouter_pricing_cache: Dict[str, Optional[Dict[str, float]]] = {}

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
        self.usage: Dict[str, Dict[str, Any]] = {
            "translation": {"input_tokens": 0.0, "output_tokens": 0.0, "reasoning_tokens": 0.0, "details": {}},
            "tts": {"input_tokens": 0.0, "output_tokens": 0.0, "input_characters": 0.0, "audio_seconds": 0.0, "details": {}},
            "transcription": {"audio_seconds": 0.0, "details": {}},
        }
        self.api_costs: List[Dict[str, Any]] = []

    # ----- Helpers -----
    def set_audio_duration(self, seconds: Optional[float]) -> None:
        self.audio_duration_sec = seconds

    def _lookup_provider_cfg(self, catalog: Dict[str, Any], group: str, provider: str) -> Optional[Dict[str, Any]]:
        group_cfg = catalog.get(group)
        if not isinstance(group_cfg, dict):
            return None
        provider_cfg = group_cfg.get(provider)
        return provider_cfg if isinstance(provider_cfg, dict) else None

    def _lookup_model_cfg(self, provider_cfg: Dict[str, Any], model: Optional[str]) -> Optional[Dict[str, Any]]:
        models_cfg = provider_cfg.get("models")
        if not isinstance(models_cfg, dict):
            return None
        for candidate in _model_candidates(model):
            model_cfg = models_cfg.get(candidate)
            if isinstance(model_cfg, dict):
                return model_cfg
        return None

    def _resolve_translation_rate_info(self, provider: str, model: Optional[str]) -> Dict[str, Any]:
        provider_key = str(provider or "unknown").lower()

        config_info = self._translation_rate_info_from_catalog(self.pricing, provider_key, model, "config")
        if config_info:
            return config_info

        if provider_key == "openrouter":
            live_info = self._fetch_openrouter_rate_info(model)
            if live_info:
                return live_info

        builtin_info = self._translation_rate_info_from_catalog(
            DEFAULT_PRICING_CATALOG,
            provider_key,
            model,
            "builtin",
        )
        if builtin_info:
            return builtin_info

        return {
            "input_per_1m_tokens": 0.0,
            "output_per_1m_tokens": 0.0,
            "reasoning_per_1m_tokens": 0.0,
            "pricing_source": "unknown",
        }

    def _translation_rate_info_from_catalog(
        self,
        catalog: Dict[str, Any],
        provider: str,
        model: Optional[str],
        source: str,
    ) -> Optional[Dict[str, Any]]:
        provider_cfg = self._lookup_provider_cfg(catalog, "translation", provider)
        if not provider_cfg:
            return None

        model_cfg = self._lookup_model_cfg(provider_cfg, model) or {}
        input_rate = model_cfg.get("input_per_1m_tokens", provider_cfg.get("input_per_1m_tokens"))
        output_rate = model_cfg.get("output_per_1m_tokens", provider_cfg.get("output_per_1m_tokens"))
        reasoning_rate = model_cfg.get(
            "reasoning_per_1m_tokens",
            provider_cfg.get("reasoning_per_1m_tokens", output_rate),
        )
        if input_rate is None and output_rate is None and reasoning_rate is None:
            return None
        return {
            "input_per_1m_tokens": _to_float(input_rate),
            "output_per_1m_tokens": _to_float(output_rate),
            "reasoning_per_1m_tokens": _to_float(reasoning_rate),
            "pricing_source": source,
        }

    def _fetch_openrouter_rate_info(self, model: Optional[str]) -> Optional[Dict[str, Any]]:
        if not model:
            return None
        model_key = str(model)
        if model_key in self._openrouter_pricing_cache:
            cached = self._openrouter_pricing_cache[model_key]
            if cached is None:
                return None
            return {**cached, "pricing_source": "openrouter-live"}

        try:
            with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=OPENROUTER_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            for item in payload.get("data", []):
                if not isinstance(item, dict):
                    continue
                ids = {str(item.get("id") or ""), str(item.get("canonical_slug") or "")}
                if model_key not in ids:
                    continue
                pricing = item.get("pricing") or {}
                prompt = _to_float(pricing.get("prompt"), -1.0)
                completion = _to_float(pricing.get("completion"), -1.0)
                if prompt < 0 or completion < 0:
                    self._openrouter_pricing_cache[model_key] = None
                    return None
                resolved = {
                    "input_per_1m_tokens": prompt * 1_000_000.0,
                    "output_per_1m_tokens": completion * 1_000_000.0,
                    "reasoning_per_1m_tokens": completion * 1_000_000.0,
                }
                self._openrouter_pricing_cache[model_key] = resolved
                return {**resolved, "pricing_source": "openrouter-live"}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            logger.debug("OpenRouter pricing lookup failed for %s: %s", model_key, exc)

        self._openrouter_pricing_cache[model_key] = None
        return None

    def _resolve_transcription_rate_info(self, provider: str, model: Optional[str]) -> Dict[str, Any]:
        provider_key = str(provider or "unknown").lower()

        config_info = self._transcription_rate_info_from_catalog(self.pricing, provider_key, model, "config")
        if config_info:
            return config_info

        builtin_info = self._transcription_rate_info_from_catalog(
            DEFAULT_PRICING_CATALOG,
            provider_key,
            model,
            "builtin",
        )
        if builtin_info:
            return builtin_info

        source = "local" if provider_key in LOCAL_TRANSCRIPTION_PROVIDERS else "unknown"
        return {"per_min": 0.0, "speaker_diarization_per_min": 0.0, "pricing_source": source}

    def _transcription_rate_info_from_catalog(
        self,
        catalog: Dict[str, Any],
        provider: str,
        model: Optional[str],
        source: str,
    ) -> Optional[Dict[str, Any]]:
        provider_cfg = self._lookup_provider_cfg(catalog, "transcription", provider)
        if not provider_cfg:
            return None
        model_cfg = self._lookup_model_cfg(provider_cfg, model) or {}
        addons = provider_cfg.get("addons") if isinstance(provider_cfg.get("addons"), dict) else {}
        per_min = model_cfg.get("per_min", provider_cfg.get("per_min"))
        if per_min is None:
            return None
        return {
            "per_min": _to_float(per_min),
            "speaker_diarization_per_min": _to_float((addons or {}).get("speaker_diarization_per_min")),
            "pricing_source": source,
        }

    def _resolve_tts_rate_info(self, provider: str, model: Optional[str]) -> Dict[str, Any]:
        provider_key = str(provider or "unknown").lower()

        config_info = self._tts_rate_info_from_catalog(self.pricing, provider_key, model, "config")
        if config_info:
            return config_info

        builtin_info = self._tts_rate_info_from_catalog(
            DEFAULT_PRICING_CATALOG,
            provider_key,
            model,
            "builtin",
        )
        if builtin_info:
            return builtin_info

        return {
            "input_per_1m_tokens": 0.0,
            "output_per_1m_tokens": 0.0,
            "per_audio_min": None,
            "per_1m_chars": 0.0,
            "audio_output_tokens_per_second": 0.0,
            "pricing_source": "unknown",
        }

    def _tts_rate_info_from_catalog(
        self,
        catalog: Dict[str, Any],
        provider: str,
        model: Optional[str],
        source: str,
    ) -> Optional[Dict[str, Any]]:
        provider_cfg = self._lookup_provider_cfg(catalog, "tts", provider)
        if not provider_cfg:
            return None
        model_cfg = self._lookup_model_cfg(provider_cfg, model) or {}
        keys = (
            "input_per_1m_tokens",
            "output_per_1m_tokens",
            "per_audio_min",
            "per_1m_chars",
            "audio_output_tokens_per_second",
            "voice_clone_per_voice",
        )
        if not any(key in model_cfg or key in provider_cfg for key in keys):
            return None
        per_audio_min = model_cfg.get("per_audio_min", provider_cfg.get("per_audio_min"))
        return {
            "input_per_1m_tokens": _to_float(model_cfg.get("input_per_1m_tokens", provider_cfg.get("input_per_1m_tokens"))),
            "output_per_1m_tokens": _to_float(model_cfg.get("output_per_1m_tokens", provider_cfg.get("output_per_1m_tokens"))),
            "per_audio_min": None if per_audio_min is None else _to_float(per_audio_min),
            "per_1m_chars": _to_float(model_cfg.get("per_1m_chars", provider_cfg.get("per_1m_chars"))),
            "audio_output_tokens_per_second": _to_float(
                model_cfg.get(
                    "audio_output_tokens_per_second",
                    provider_cfg.get("audio_output_tokens_per_second"),
                )
            ),
            "voice_clone_per_voice": _to_float(model_cfg.get("voice_clone_per_voice", provider_cfg.get("voice_clone_per_voice"))),
            "pricing_source": source,
        }

    def _record_api_cost(
        self,
        *,
        category: str,
        provider: str,
        model: Optional[str],
        usage: Dict[str, float],
        rates: Dict[str, Any],
        cost_usd: float,
    ) -> None:
        stage_key = _stage_for_category(category)
        self.api_costs.append({
            "stage_key": stage_key,
            "category": category,
            "category_label": _label_for_category(category),
            "provider": provider,
            "model": model or "default",
            "usage": {k: float(v or 0.0) for k, v in usage.items() if float(v or 0.0) != 0.0},
            "rates": rates,
            "pricing_source": rates.get("pricing_source", "unknown"),
            "cost_usd": float(cost_usd or 0.0),
        })

    def import_api_cost_rows(self, rows: List[Dict[str, Any]]) -> int:
        """Import previously recorded API cost rows into this tracker.

        Worker stages can run in separate Celery task invocations. Importing
        rows keeps the final report's tracker representative of the whole
        project pipeline instead of only the last task.
        """
        imported = 0
        for raw in rows or []:
            if not isinstance(raw, dict):
                continue
            category = str(raw.get("category") or "")
            stage_key = str(raw.get("stage_key") or _stage_for_category(category) or "unknown")
            provider = str(raw.get("provider") or "unknown")
            model = str(raw.get("model") or "default")
            usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
            rates = raw.get("rates") if isinstance(raw.get("rates"), dict) else {}
            cost = _to_float(raw.get("cost_usd"))

            row = {
                "stage_key": stage_key,
                "category": category,
                "category_label": raw.get("category_label") or _label_for_category(category),
                "provider": provider,
                "model": model,
                "usage": {
                    str(key): float(value or 0.0)
                    for key, value in usage.items()
                    if isinstance(value, (int, float)) and float(value or 0.0) != 0.0
                },
                "rates": dict(rates),
                "pricing_source": raw.get("pricing_source") or rates.get("pricing_source", "unknown"),
                "cost_usd": cost,
            }
            self.api_costs.append(row)
            self._accumulate_imported_row(row)
            imported += 1
        return imported

    def _accumulate_imported_row(self, row: Dict[str, Any]) -> None:
        stage_key = str(row.get("stage_key") or "")
        category = str(row.get("category") or "")
        provider = str(row.get("provider") or "unknown")
        model = str(row.get("model") or "default")
        usage = row.get("usage") or {}
        cost = float(row.get("cost_usd") or 0.0)

        if stage_key == "speech_synthesis":
            self.actual["tts"] += cost
        elif stage_key == "transcription":
            self.actual["transcription"] += cost
        else:
            self.actual["translation"] += cost
        self.actual["total"] += cost

        audio_seconds = float(usage.get("audio_seconds") or 0.0)
        if audio_seconds > 0 and (
            stage_key == "transcription"
            or category in {"tts_batch_alignment", "tts_content_validation"}
        ):
            self.usage["transcription"]["audio_seconds"] += audio_seconds
            detail_map = self.usage["transcription"].setdefault("details", {})
            detail_key = f"{provider}:{model}"
            detail_entry = detail_map.setdefault(
                detail_key,
                {
                    "provider": provider,
                    "model": model,
                    "audio_seconds": 0.0,
                    "cost_usd": 0.0,
                    "categories": {},
                },
            )
            detail_entry["audio_seconds"] += audio_seconds
            detail_entry["cost_usd"] += cost
            categories = detail_entry.setdefault("categories", {})
            categories[category] = float(categories.get(category, 0.0)) + audio_seconds

        input_tokens = float(usage.get("input_tokens") or 0.0)
        output_tokens = float(usage.get("output_tokens") or 0.0)
        reasoning_tokens = float(usage.get("reasoning_tokens") or 0.0)
        if stage_key in {"context_analysis", "translation", "emotion_analysis"}:
            self.usage["translation"]["input_tokens"] += input_tokens
            self.usage["translation"]["output_tokens"] += output_tokens
            self.usage["translation"]["reasoning_tokens"] += reasoning_tokens
            detail_map = self.usage["translation"].setdefault("details", {})
            detail_key = f"{provider}:{model}"
            entry = detail_map.setdefault(
                detail_key,
                {"input_tokens": 0.0, "output_tokens": 0.0, "reasoning_tokens": 0.0},
            )
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens
            entry["reasoning_tokens"] += reasoning_tokens
            entry["cost_usd"] = float(entry.get("cost_usd", 0.0)) + cost

        if category == "tts_synthesis":
            self.usage["tts"]["input_tokens"] += input_tokens
            self.usage["tts"]["output_tokens"] += output_tokens
            self.usage["tts"]["input_characters"] += float(usage.get("input_characters") or 0.0)
            self.usage["tts"]["audio_seconds"] += audio_seconds
            details = self.usage["tts"].setdefault("details", {})
            detail_key = f"{provider}:{model}"
            entry = details.setdefault(
                detail_key,
                {
                    "input_tokens": 0.0,
                    "output_tokens": 0.0,
                    "input_characters": 0.0,
                    "audio_seconds": 0.0,
                    "cost_usd": 0.0,
                },
            )
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens
            entry["input_characters"] += float(usage.get("input_characters") or 0.0)
            entry["audio_seconds"] += audio_seconds
            entry["cost_usd"] += cost

    def _get_translation_rates(self, provider: str, model: Optional[str]) -> Tuple[float, float, float]:
        """Retrieve input/output/reasoning pricing (per million tokens) for a translation provider/model combo.
        
        Returns (0.0, 0.0, 0.0) if pricing configuration is missing.
        """
        info = self._resolve_translation_rate_info(provider, model)
        return (
            float(info.get("input_per_1m_tokens") or 0.0),
            float(info.get("output_per_1m_tokens") or 0.0),
            float(info.get("reasoning_per_1m_tokens") or 0.0),
        )

    def _get_tts_rates(self, provider: str, model: Optional[str]) -> Tuple[float, float, Optional[float]]:
        """Retrieve TTS pricing (per million tokens and per audio minute).
        
        Returns (0.0, 0.0, None) if pricing configuration is missing.
        """
        info = self._resolve_tts_rate_info(provider, model)
        per_audio_min = info.get("per_audio_min")
        return (
            float(info.get("input_per_1m_tokens") or 0.0),
            float(info.get("output_per_1m_tokens") or 0.0),
            None if per_audio_min is None else float(per_audio_min),
        )

    # ----- Transcription -----
    def estimate_transcription_cost(self, provider: str, audio_seconds: Optional[float]) -> float:
        if not audio_seconds or audio_seconds <= 0:
            return 0.0
        rate_info = self._resolve_transcription_rate_info(provider, None)
        per_min = float(rate_info.get("per_min") or 0.0)
        cost = per_min * (audio_seconds / 60.0)
        self.estimate["transcription"] += cost
        self.estimate["total"] += cost
        self.usage["transcription"]["audio_seconds"] += float(audio_seconds or 0.0)
        logger.debug(f"Estimated transcription cost [{provider}]: ${cost:.4f} for {audio_seconds/60.0:.2f} min")
        return cost

    def add_transcription_usage(
        self,
        provider: str,
        audio_seconds: Optional[float],
        *,
        model: Optional[str] = None,
        category: str = "transcription",
        count_cost: bool = True,
        speaker_diarization: bool = False,
    ) -> float:
        """Record actual transcription/ASR usage with optional per-model detail."""
        if not audio_seconds or audio_seconds <= 0:
            return 0.0

        provider_key = str(provider or "unknown")
        model_key = str(model or "default")
        category_key = str(category or "transcription")
        seconds = float(audio_seconds)
        rate_info = self._resolve_transcription_rate_info(provider_key, model_key)
        per_min = float(rate_info.get("per_min") or 0.0)
        diarization_per_min = float(rate_info.get("speaker_diarization_per_min") or 0.0)
        billable_minutes = seconds / 60.0
        cost = 0.0
        if count_cost:
            cost = per_min * billable_minutes
            if speaker_diarization:
                cost += diarization_per_min * billable_minutes

        if count_cost:
            self.actual["transcription"] += cost
            self.actual["total"] += cost
            self._record_api_cost(
                category=category_key,
                provider=provider_key,
                model=model_key,
                usage={"audio_seconds": seconds},
                rates={
                    "per_min": per_min,
                    "speaker_diarization_per_min": diarization_per_min if speaker_diarization else 0.0,
                    "pricing_source": rate_info.get("pricing_source", "unknown"),
                },
                cost_usd=cost,
            )
        self.usage["transcription"]["audio_seconds"] += seconds

        detail_map = self.usage["transcription"].setdefault("details", {})
        detail_key = f"{provider_key}:{model_key}"
        detail_entry = detail_map.setdefault(
            detail_key,
            {
                "provider": provider_key,
                "model": model_key,
                "audio_seconds": 0.0,
                "cost_usd": 0.0,
                "categories": {},
            },
        )
        detail_entry["audio_seconds"] += seconds
        detail_entry["cost_usd"] += cost
        categories = detail_entry.setdefault("categories", {})
        categories[category_key] = float(categories.get(category_key, 0.0)) + seconds

        logger.debug(
            "Recorded transcription usage [%s#%s, %s]: %.2fs, cost=$%.4f",
            provider_key,
            model_key,
            category_key,
            seconds,
            cost,
        )
        return cost

    def add_transcription_actual(self, provider: str, audio_seconds: Optional[float]) -> float:
        return self.add_transcription_usage(provider, audio_seconds)

    # ----- Translation (LLM) -----
    def estimate_translation_cost(
        self,
        provider: str,
        model: Optional[str],
        input_tokens: float,
        expected_output_tokens: Optional[float] = None,
        expected_reasoning_tokens: Optional[float] = None,
    ) -> float:
        if input_tokens <= 0:
            return 0.0
        if expected_output_tokens is None:
            expected_output_tokens = input_tokens

        in_rate, out_rate, reasoning_rate = self._get_translation_rates(provider, model)
        reasoning_tokens = float(expected_reasoning_tokens or 0.0)

        cost = (
            in_rate * (float(input_tokens) / 1_000_000.0)
            + out_rate * (float(expected_output_tokens) / 1_000_000.0)
            + reasoning_rate * (reasoning_tokens / 1_000_000.0)
        )

        self.estimate["translation"] += cost
        self.estimate["total"] += cost
        self.usage["translation"]["input_tokens"] += float(input_tokens or 0.0)
        self.usage["translation"]["output_tokens"] += float(expected_output_tokens or 0.0)
        self.usage["translation"]["reasoning_tokens"] += reasoning_tokens
        detail_map = self.usage["translation"].setdefault("details", {})
        detail_entry = detail_map.setdefault(
            f"{provider}:{model or 'default'}",
            {"input_tokens": 0.0, "output_tokens": 0.0, "reasoning_tokens": 0.0},
        )
        detail_entry["input_tokens"] += float(input_tokens or 0.0)
        detail_entry["output_tokens"] += float(expected_output_tokens or 0.0)
        detail_entry["reasoning_tokens"] += reasoning_tokens
        logger.debug(
            f"Estimated translation cost [{provider}#{model or 'default'}]: "
            f"${cost:.4f} (in≈{input_tokens:.0f} tok, out≈{expected_output_tokens:.0f} tok, reasoning≈{reasoning_tokens:.0f} tok)"
        )
        return cost

    def add_translation_actual(
        self,
        provider: str,
        model: Optional[str],
        input_tokens: float,
        output_tokens: float,
        reasoning_tokens: float = 0.0,
        *,
        category: str = "translation",
    ) -> float:
        input_tokens = float(input_tokens or 0.0)
        output_tokens = float(output_tokens or 0.0)
        reasoning_tokens = float(reasoning_tokens or 0.0)

        rate_info = self._resolve_translation_rate_info(provider, model)
        in_rate = float(rate_info.get("input_per_1m_tokens") or 0.0)
        out_rate = float(rate_info.get("output_per_1m_tokens") or 0.0)
        reasoning_rate = float(rate_info.get("reasoning_per_1m_tokens") or 0.0)
        cost = (
            in_rate * (input_tokens / 1_000_000.0)
            + out_rate * (output_tokens / 1_000_000.0)
            + reasoning_rate * (reasoning_tokens / 1_000_000.0)
        )
        self.actual["translation"] += cost
        self.actual["total"] += cost
        self.usage["translation"]["input_tokens"] += input_tokens
        self.usage["translation"]["output_tokens"] += output_tokens
        self.usage["translation"]["reasoning_tokens"] += reasoning_tokens

        if provider and model:
            detail_map = self.usage["translation"].setdefault("details", {})
            detail_key = f"{provider}:{model}"
            entry = detail_map.setdefault(
                detail_key,
                {"input_tokens": 0.0, "output_tokens": 0.0, "reasoning_tokens": 0.0},
            )
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens
            entry["reasoning_tokens"] += reasoning_tokens
            entry["cost_usd"] = float(entry.get("cost_usd", 0.0)) + cost

        self._record_api_cost(
            category=category,
            provider=str(provider or "unknown"),
            model=model,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
            rates={
                "input_per_1m_tokens": in_rate,
                "output_per_1m_tokens": out_rate,
                "reasoning_per_1m_tokens": reasoning_rate,
                "pricing_source": rate_info.get("pricing_source", "unknown"),
            },
            cost_usd=cost,
        )

        logger.debug(
            f"Calculated translation cost [{provider}#{model or 'default'}]: "
            f"${cost:.4f} (in≈{input_tokens:.0f}, out≈{output_tokens:.0f}, reasoning≈{reasoning_tokens:.0f})"
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
        rate_info = self._resolve_tts_rate_info(provider, model)
        input_rate = float(rate_info.get("input_per_1m_tokens") or 0.0)
        output_rate = float(rate_info.get("output_per_1m_tokens") or 0.0)
        per_audio_min = rate_info.get("per_audio_min")
        output_tokens_per_second = float(rate_info.get("audio_output_tokens_per_second") or 0.0)

        cost_input = input_rate * (float(input_tokens or 0.0) / 1_000_000.0) if input_rate else 0.0
        output_tokens = float(expected_output_tokens or 0.0)
        if expected_audio_seconds and output_tokens_per_second:
            output_tokens = float(expected_audio_seconds) * output_tokens_per_second
        cost_output = output_rate * (output_tokens / 1_000_000.0) if output_rate else 0.0
        cost_audio = 0.0
        if expected_audio_seconds and per_audio_min and output_tokens <= 0:
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
        logger.debug(f"Estimated TTS cost [{provider}#{model or 'default'}]: ${cost:.4f}")
        return cost

    def add_tts_actual(
        self,
        provider: str,
        model: Optional[str] = None,
        input_tokens: float = 0.0,
        output_tokens: float = 0.0,
        audio_seconds: float = 0.0,
        input_characters: float = 0.0,
        *,
        category: str = "tts_synthesis",
    ) -> float:
        rate_info = self._resolve_tts_rate_info(provider, model)
        input_rate = float(rate_info.get("input_per_1m_tokens") or 0.0)
        output_rate = float(rate_info.get("output_per_1m_tokens") or 0.0)
        per_audio_min = rate_info.get("per_audio_min")
        per_1m_chars = float(rate_info.get("per_1m_chars") or 0.0)
        output_tokens_per_second = float(rate_info.get("audio_output_tokens_per_second") or 0.0)

        cost_input = input_rate * (float(input_tokens or 0.0) / 1_000_000.0) if input_rate else 0.0
        effective_output_tokens = float(output_tokens or 0.0)
        if effective_output_tokens <= 0 and audio_seconds and output_tokens_per_second:
            effective_output_tokens = float(audio_seconds) * output_tokens_per_second
        cost_output = output_rate * (effective_output_tokens / 1_000_000.0) if output_rate else 0.0
        cost_audio = 0.0
        if audio_seconds and per_audio_min and effective_output_tokens <= 0:
            cost_audio = float(per_audio_min) * (float(audio_seconds or 0.0) / 60.0)
        cost_chars = per_1m_chars * (float(input_characters or 0.0) / 1_000_000.0) if per_1m_chars else 0.0

        cost = cost_input + cost_output + cost_audio + cost_chars
        self.actual["tts"] += cost
        self.actual["total"] += cost
        self.usage["tts"]["input_tokens"] += float(input_tokens or 0.0)
        self.usage["tts"]["output_tokens"] += effective_output_tokens
        self.usage["tts"]["input_characters"] += float(input_characters or 0.0)
        self.usage["tts"]["audio_seconds"] += float(audio_seconds or 0.0)

        if provider:
            detail_key = f"{provider}:{model or 'default'}"
            details = self.usage["tts"].setdefault("details", {})
            entry = details.setdefault(detail_key, {
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "input_characters": 0.0,
                "audio_seconds": 0.0,
                "cost_usd": 0.0,
            })
            entry["input_tokens"] += float(input_tokens or 0.0)
            entry["output_tokens"] += effective_output_tokens
            entry["input_characters"] += float(input_characters or 0.0)
            entry["audio_seconds"] += float(audio_seconds or 0.0)
            entry["cost_usd"] += cost

        self._record_api_cost(
            category=category,
            provider=str(provider or "unknown"),
            model=model,
            usage={
                "input_tokens": float(input_tokens or 0.0),
                "output_tokens": effective_output_tokens,
                "input_characters": float(input_characters or 0.0),
                "audio_seconds": float(audio_seconds or 0.0),
            },
            rates={
                "input_per_1m_tokens": input_rate,
                "output_per_1m_tokens": output_rate,
                "per_audio_min": per_audio_min,
                "per_1m_chars": per_1m_chars,
                "audio_output_tokens_per_second": output_tokens_per_second,
                "pricing_source": rate_info.get("pricing_source", "unknown"),
            },
            cost_usd=cost,
        )

        logger.debug(f"Calculated TTS cost [{provider}#{model or 'default'}]: ${cost:.4f}")
        return cost

    def add_voice_clone_actual(
        self,
        provider: str,
        model: Optional[str] = None,
        voice_count: float = 1.0,
    ) -> float:
        rate_info = self._resolve_tts_rate_info(provider, model)
        per_voice = float(rate_info.get("voice_clone_per_voice") or 0.0)
        voices = float(voice_count or 0.0)
        if voices <= 0:
            return 0.0

        cost = per_voice * voices
        self.actual["tts"] += cost
        self.actual["total"] += cost
        self._record_api_cost(
            category="voice_cloning",
            provider=str(provider or "unknown"),
            model=model,
            usage={"voice_count": voices},
            rates={
                "voice_clone_per_voice": per_voice,
                "pricing_source": rate_info.get("pricing_source", "unknown"),
            },
            cost_usd=cost,
        )
        logger.debug(f"Calculated voice clone cost [{provider}#{model or 'default'}]: ${cost:.4f}")
        return cost

    # ----- Summary -----
    def get_api_cost_rows(self) -> List[Dict[str, Any]]:
        """Return detailed actual API cost rows for reports."""
        return list(self.api_costs)

    def get_costs_by_step(self) -> Dict[str, float]:
        """Return actual costs mapped to pipeline step names."""
        if not self.api_costs:
            return {
                "transcription": self.actual.get("transcription", 0.0),
                "translation": self.actual.get("translation", 0.0),
                "speech_synthesis": self.actual.get("tts", 0.0),
                "total": self.actual.get("total", 0.0),
            }

        costs: Dict[str, float] = {}
        for row in self.api_costs:
            stage_key = str(row.get("stage_key") or "unknown")
            costs[stage_key] = costs.get(stage_key, 0.0) + float(row.get("cost_usd") or 0.0)
        costs["total"] = sum(value for key, value in costs.items() if key != "total")
        return costs

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
        tts_chars = self.usage["tts"].get("input_characters", 0.0)
        tts_sec = self.usage["tts"].get("audio_seconds", 0.0)
        tr_sec = self.usage["transcription"].get("audio_seconds", 0.0)
        reasoning_tokens = self.usage["translation"].get("reasoning_tokens", 0.0)

        lines.append("")
        lines.append("Usage details (approx):")
        reasoning_segment = f", reasoning≈{reasoning_tokens:.0f}" if reasoning_tokens > 0 else ""
        lines.append(f" - Translation tokens: in≈{t_in:.0f}, out≈{t_out:.0f}{reasoning_segment}")
        char_segment = f", chars≈{tts_chars:.0f}" if tts_chars > 0 else ""
        lines.append(f" - TTS tokens: in≈{tts_in:.0f}, out≈{tts_out:.0f}{char_segment}, audio≈{tts_sec/60.0:.2f} min")
        lines.append(f" - Transcription audio: ≈{tr_sec/60.0:.2f} min")
        if self.api_costs:
            lines.append(" - API cost rows: %d" % len(self.api_costs))

        for line in lines:
            logger.info(line)
