"""Gemini 3.8 Flash TTS wrapper (Interactions API, per-segment synthesis).

Design differences from the legacy ``GeminiTTSWrapper``:

* **Interactions API** (``client.interactions.create``) instead of
  ``generate_content``. The request text is a *verbatim transcript* and
  delivery directions live in a ``speech_metadata`` annotation.
* **No ASR-based track separation.** Every segment is synthesized in its own
  request and written straight to its own file, so no diarizer or AssemblyAI
  alignment is involved.
* **Multi-sample speaker references.** Voice matching and voice-consistency
  validation use a profile built from several clips of the original track
  (centroid of the speaker's "average" voice), see ``speaker_reference``.
* **3.8 tag dialect** via ``gemini38_tags``: legacy ``[tag]`` markup is
  converted to inline angle-bracket tags plus a compact ``style`` string.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import shutil
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from .gemini_voice_catalog import (
    ALL_GEMINI_VOICES,
    GEMINI_VOICE_GENDERS,
    get_gemini_voice_gender,
    resolve_default_gemini_voice,
)
from .gemini38_tags import build_gemini38_prompt, quoted_fragment_leak, sanitize_angle_tags, strip_tts_markup
from .models import (
    DiarizationSegment,
    SegmentAlignment,
    SegmentSynthesisReport,
    TTSSegmentData,
)
from .speaker_reference import SpeakerReferenceBuilder, SpeakerReferenceProfile, best_voice_for_profile
from .tts_interface import TTSInterface
from .voice_sample_manager import AudioFileUtils, AudioValidator, VoiceSampleManager
from src.dubbing.core.log_config import get_logger
from src.utils.audio_embedder import AudioEmbedder
from src.utils.llm_call import robust_llm_call
from src.utils.sent_split import greedy_sent_split
from src.utils.speaker_gender import (
    build_effective_speaker_gender_map,
    normalize_speaker_metadata_map,
)
from src.utils.voice_matcher import VoiceMatcher

try:
    from google import genai

    GEMINI_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    genai = None  # type: ignore[assignment]
    GEMINI_AVAILABLE = False

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]
    TIKTOKEN_AVAILABLE = False

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz

    RAPIDFUZZ_AVAILABLE = True
except Exception:  # pragma: no cover
    _rapidfuzz_fuzz = None  # type: ignore[assignment]
    RAPIDFUZZ_AVAILABLE = False

try:
    from src.tts.local_whisper_validator import LocalWhisperContentValidator

    LOCAL_WHISPER_VALIDATOR_AVAILABLE = True
except Exception:  # pragma: no cover
    LocalWhisperContentValidator = None  # type: ignore[assignment]
    LOCAL_WHISPER_VALIDATOR_AVAILABLE = False

try:
    from src.tts.speaker_segment_diarizer import SpeakerSegmentDiarizer

    ASSEMBLYAI_VALIDATOR_AVAILABLE = True
except Exception:  # pragma: no cover
    SpeakerSegmentDiarizer = None  # type: ignore[assignment]
    ASSEMBLYAI_VALIDATOR_AVAILABLE = False

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GEMINI38_TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_GEMINI38_TTS_FALLBACK_MODEL = "gemini-3.8-flash-lite-tts"
GEMINI38_TTS_MODELS: Tuple[str, ...] = (
    DEFAULT_GEMINI38_TTS_MODEL,
    DEFAULT_GEMINI38_TTS_FALLBACK_MODEL,
)

DEFAULT_SAMPLES_DIR = (Path(__file__).parent / "samples" / "gemini38").resolve()
DURATION_STATS_FILE = DEFAULT_SAMPLES_DIR / "gemini38_voice_stats.json"
DURATION_ADJUSTMENTS_FILE = DEFAULT_SAMPLES_DIR / "gemini38_duration_adjustments.json"
SPEAKER_REFERENCE_DIR = Path("artifacts/speakers_ref")

SAMPLE_RATE = 24000
MAX_CHAR_LIMIT_PER_REQUEST = 30000
MIN_REFERENCE_DURATION_SECONDS = 3.0
SILENCE_THRESHOLD_FOR_REPHRASING = 0.03
MAX_REPHRASE_ATTEMPTS = 3

DURATION_SAMPLE_TEXTS = [
    "Hello, this is a quick test of voice speed and clarity.",
    "The weather today is absolutely wonderful. I hope you are enjoying this beautiful day.",
    "Technology has transformed our lives in ways we never imagined possible before.",
    "Once upon a time, in a distant land, there lived a wise old merchant who traveled extensively.",
    "Scientific research continues to reveal fascinating discoveries about our universe and its mysteries.",
    "Education empowers individuals to achieve their dreams and contribute meaningfully to society.",
    "The art of cooking combines creativity, technique, and passion to create memorable dining experiences.",
    "Communication skills are essential for success in both personal relationships and professional endeavors.",
    "Environmental conservation requires collective effort from governments, businesses, and individual citizens worldwide.",
    "Innovation drives progress across industries, from healthcare and transportation to entertainment and beyond.",
]
DURATION_SAMPLE_TEXT = " ".join(DURATION_SAMPLE_TEXTS)

_ASR_FILLER_RE = None  # populated lazily in _normalize_for_asr_match


class Gemini38TTSConfig(BaseModel):
    """Configuration for the Gemini 3.8 Flash TTS wrapper."""

    model: str = DEFAULT_GEMINI38_TTS_MODEL
    fallback_model: Optional[str] = DEFAULT_GEMINI38_TTS_FALLBACK_MODEL
    default_voice: str = "Kore"
    prompt_prefix: str = ""
    blocked_voices: List[str] = Field(default_factory=list)

    # Context continuity: the previous translated line is passed to the model
    # as a quoted delivery context inside speech_metadata.style (never spoken).
    enable_context_style: bool = True
    context_style_max_chars: int = 140
    style_max_chars: int = 320

    # Voice matching / multi-sample references
    enable_voice_matching: bool = True
    enable_extended_voice_library: bool = False
    extended_library_shortlist: int = 24
    embedding_model_device: Optional[str] = None
    ref_min_clip_seconds: float = 1.2
    ref_max_clip_seconds: float = 15.0
    ref_target_seconds: float = 16.0
    ref_max_clips: int = 8
    ref_min_clips: int = 3
    ref_min_rms_dbfs: float = -45.0
    ref_silence_thresh_db: float = -40.0

    # Retries
    max_retries: int = 10
    retry_delay_base: float = 2.0
    primary_model_max_retries: int = 4
    fallback_model_max_retries: int = 2

    # Audio validation
    enable_audio_validation: bool = True
    fade_detection_enabled: bool = True
    fade_window_size_ms: int = 500
    min_fade_db: float = 10.0
    fade_detection_percentile: int = 75
    max_total_silence_ratio: float = 3.5
    max_contiguous_silence_seconds: float = 80.0
    trailing_silence_grace_ratio: float = 0.05

    # Voice consistency validation
    enable_voice_consistency_validation: bool = True
    voice_similarity_threshold: float = 0.70
    voice_similarity_relaxed_threshold: float = 0.65
    min_voice_validation_duration_seconds: float = 0.5
    voice_reference_mode: str = "max"  # "max" | "voice_sample" | "speaker_ref"
    enable_voice_gender_relative_check: bool = True
    voice_gender_relative_margin: float = 0.0

    # Content validation (per-segment ASR round-trip; NOT diarization)
    enable_content_validation: bool = True
    content_validator_provider: str = "whisper"
    content_validator_whisper_model: str = "base"
    content_validator_whisper_compute_type: str = "int8"
    content_validator_whisper_cpu_threads: int = 2
    content_validator_speech_model: str = "nano"
    content_similarity_threshold: float = 70.0
    content_trailing_word_count: int = 3
    content_trailing_fuzzy_threshold: float = 70.0
    content_validation_min_expected_chars: int = 8

    # Emotion enrichment
    enable_emotion_enrichment: bool = False
    enable_llm_editor: bool = False
    emotion_enrichment_model: str = "gemini-2.5-pro"
    emotion_enrichment_temperature: float = 0.7

    # Duration statistics
    duration_smoothing_alpha: float = 0.35
    duration_stats_auto_save: bool = True
    duration_stats_save_interval: int = 20


class Gemini38APIClient:
    """Thin client around the Gemini Interactions API for TTS."""

    def __init__(self, config: Gemini38TTSConfig):
        self.config = config
        self.client: Optional[Any] = None
        self.current_model: str = config.model

    def initialize(self) -> None:
        if not GEMINI_AVAILABLE or genai is None:
            raise ImportError("Google GenAI SDK is not installed.")
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is required for Gemini 3.8 TTS.")
        self.client = genai.Client(api_key=api_key)
        if not hasattr(self.client, "interactions"):
            raise RuntimeError(
                "Installed google-genai SDK has no Interactions API. "
                "Upgrade to google-genai>=2.3.0 (voice library needs >=2.25.0)."
            )
        self.current_model = self.config.model
        logger.info("Gemini 3.8 TTS client initialized (model=%s)", self.current_model)

    def set_model(self, model: str) -> None:
        self.current_model = model

    def reset_model(self) -> None:
        self.current_model = self.config.model

    def _attempts_for(self, model: str) -> int:
        if model == self.config.model:
            return max(1, int(self.config.primary_model_max_retries))
        return max(1, int(self.config.fallback_model_max_retries))

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
                "sample_rate": SAMPLE_RATE,
            },
            "generation_config": {"speech_config": [{"voice": voice}]},
            # TTS interactions do not need server-side conversation state;
            # keeping them stateless avoids storing generated audio.
            "store": False,
        }

    def synthesize_chunk(self, text: str, style: str, voice: str, model: Optional[str] = None) -> bytes:
        """Synthesize one chunk and return raw 24kHz mono 16-bit PCM bytes."""
        if not text or not text.strip():
            return b""
        if self.client is None:
            raise RuntimeError("Gemini 3.8 TTS client is not initialized.")

        effective_model = model or self.current_model
        attempts = self._attempts_for(effective_model)
        payload = self._build_request(text, style, voice, effective_model)

        for attempt in range(attempts):
            try:
                interaction = self.client.interactions.create(**payload)
                raw = self._extract_audio_bytes(interaction)
                if raw:
                    return raw
                raise RuntimeError("Interaction returned empty audio")
            except Exception as exc:
                if attempt + 1 >= attempts:
                    logger.error(
                        "Gemini 3.8 TTS failed after %d attempt(s) with model '%s': %s",
                        attempts,
                        effective_model,
                        exc,
                    )
                    return b""
                delay = self.config.retry_delay_base ** attempt
                logger.warning(
                    "Gemini 3.8 TTS attempt %d/%d failed (model=%s): %s — retrying in %.1fs",
                    attempt + 1,
                    attempts,
                    effective_model,
                    exc,
                    delay,
                )
                time.sleep(delay)
        return b""

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
        return Gemini38APIClient._normalize_audio_bytes(raw)

    @staticmethod
    def _normalize_audio_bytes(raw: bytes) -> bytes:
        """Accept both raw PCM (audio/l16) and a WAV container (audio/wav)."""
        if not raw:
            return b""
        if raw[:4] == b"RIFF":
            try:
                with wave.open(io.BytesIO(raw), "rb") as wf:
                    if wf.getsampwidth() != 2:
                        return b""
                    return wf.readframes(wf.getnframes())
            except Exception:
                return b""
        return raw

    @staticmethod
    def extract_usage(interaction: Any) -> Dict[str, float]:
        """Best-effort usage extraction (shape may vary across SDK versions)."""
        usage = getattr(interaction, "usage", None)
        if usage is None and isinstance(interaction, dict):
            usage = interaction.get("usage")
        if usage is None:
            return {}
        result: Dict[str, float] = {}
        mapping = (
            ("input_tokens", "input_tokens"),
            ("prompt_token_count", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("candidates_token_count", "output_tokens"),
        )
        for source_key, target_key in mapping:
            value = getattr(usage, source_key, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(source_key)
            if value:
                result[target_key] = result.get(target_key, 0.0) + float(value)
        return result


class Gemini38EmotionEnricher:
    """Enriches transcripts with Gemini 3.8 inline vocal tags."""

    def __init__(self, config: Gemini38TTSConfig):
        self.config = config
        self.llm = None

    def initialize(self) -> None:
        try:
            from llama_index.llms.gemini import Gemini
        except ImportError:
            logger.error("llama_index.llms.gemini not available for emotion enrichment")
            return
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found for emotion enrichment")
            return
        try:
            self.llm = Gemini(
                model=self.config.emotion_enrichment_model,
                temperature=self.config.emotion_enrichment_temperature,
            )
            logger.info(
                "Gemini 3.8 emotion enrichment LLM initialized (model=%s)",
                self.config.emotion_enrichment_model,
            )
        except Exception as exc:
            logger.error("Failed to initialize emotion enrichment LLM: %s", exc)
            self.llm = None

    @property
    def enrichment_prompt(self) -> str:
        return (
            "You enrich a verbatim TTS transcript with momentary human vocal "
            "cues for Gemini 3.8 Flash TTS.\n\n"
            "The transcript text is spoken exactly as written. Delivery style "
            "(emotion, pace, volume) is configured separately, so do NOT add "
            "style words to the text.\n\n"
            "Allowed inline tags (angle brackets, placed exactly where the sound happens):\n"
            "<sigh>, <breath>, <heavy breath>, <exhales>, <cough>, <gasp>, <giggle>, "
            "<chuckle>, <laugh>, <sob>, <scream>, <shout>, <groan>, <grunt>, <hiss>, "
            "<pant>, <phew>, <snicker>, <snort>, <sneeze>, <throat-clearing>, <tsk>, "
            "<whimper>, <yawn>, <short pause>, <long pause>\n\n"
            "Guidelines:\n"
            "- Be sparing: most lines need zero tags.\n"
            "- Never invent tags outside the list above and never use square brackets.\n"
            "- Prefer <short pause> / <long pause> over filler sounds for rhythm.\n"
            "- Keep natural spoken disfluencies (uh, uhm, hm) as plain words when present.\n"
            "- Preserve the text exactly apart from added/removed tags.\n\n"
            "{context_section}\n"
            "Current text to enrich:\n{text}\n\n"
            "Return ONLY the enriched text."
        )

    def enrich_text(self, text: str, previous_segments: Optional[List[str]] = None) -> str:
        if not text or not text.strip():
            return text
        if not self.llm:
            return text
        try:
            context_section = ""
            if previous_segments:
                context_lines = "\n".join(f"- {segment}" for segment in previous_segments[-5:])
                context_section = f"Previous conversation context:\n{context_lines}\n"
            prompt = self.enrichment_prompt.format(context_section=context_section, text=text)
            response = robust_llm_call(self.llm.complete, prompt)
            response_text = getattr(response, "text", None) or (str(response) if response else "")
            enriched = (response_text or "").strip()
            if enriched and len(enriched) <= len(text) * 3:
                return enriched
            return text
        except Exception as exc:
            logger.error("Error during text enrichment: %s", exc, exc_info=True)
            return text


class Gemini38TTSWrapper(TTSInterface):
    """Gemini 3.8 Flash TTS wrapper with per-segment synthesis."""

    # Per-segment synthesis only: no multi-speaker batches, no diarizer.
    supports_segment_batching: bool = False

    def __init__(
        self,
        model: str = DEFAULT_GEMINI38_TTS_MODEL,
        fallback_model: str = DEFAULT_GEMINI38_TTS_FALLBACK_MODEL,
        default_voice: str = "Kore",
        embedding_model_device: Optional[str] = None,
        enable_voice_matching: bool = True,
        enable_extended_voice_library: bool = False,
        enable_audio_validation: bool = True,
        prompt_prefix: Optional[str] = None,
        blocked_voices: Optional[List[str]] = None,
        debug_tts: bool = False,
        enable_emotion_enrichment: bool = False,
        enable_llm_editor: bool = False,
        emotion_enrichment_model: Optional[str] = None,
        emotion_enrichment_temperature: Optional[float] = None,
        cost_tracker: Optional[Any] = None,
        translator: Optional[Any] = None,
        target_language: Optional[str] = None,
        speaker_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI SDK is not installed.")

        config_kwargs: Dict[str, Any] = {
            "model": model or DEFAULT_GEMINI38_TTS_MODEL,
            "fallback_model": fallback_model,
            "default_voice": default_voice,
            "embedding_model_device": embedding_model_device,
            "enable_voice_matching": enable_voice_matching,
            "enable_extended_voice_library": enable_extended_voice_library,
            "enable_audio_validation": enable_audio_validation,
            "prompt_prefix": prompt_prefix or "",
            "blocked_voices": list(blocked_voices or []),
            "enable_emotion_enrichment": enable_emotion_enrichment,
            "enable_llm_editor": enable_llm_editor,
            "emotion_enrichment_model": emotion_enrichment_model or "gemini-2.5-pro",
            "emotion_enrichment_temperature": (
                emotion_enrichment_temperature if emotion_enrichment_temperature is not None else 0.7
            ),
        }
        passthrough_keys = (
            "enable_context_style",
            "context_style_max_chars",
            "style_max_chars",
            "extended_library_shortlist",
            "ref_min_clip_seconds",
            "ref_max_clip_seconds",
            "ref_target_seconds",
            "ref_max_clips",
            "ref_min_clips",
            "ref_min_rms_dbfs",
            "ref_silence_thresh_db",
            "max_retries",
            "retry_delay_base",
            "primary_model_max_retries",
            "fallback_model_max_retries",
            "max_total_silence_ratio",
            "max_contiguous_silence_seconds",
            "trailing_silence_grace_ratio",
            "enable_voice_consistency_validation",
            "voice_similarity_threshold",
            "voice_similarity_relaxed_threshold",
            "min_voice_validation_duration_seconds",
            "voice_reference_mode",
            "enable_voice_gender_relative_check",
            "voice_gender_relative_margin",
            "enable_content_validation",
            "content_validator_provider",
            "content_validator_whisper_model",
            "content_validator_whisper_compute_type",
            "content_validator_whisper_cpu_threads",
            "content_validator_speech_model",
            "content_similarity_threshold",
            "content_trailing_word_count",
            "content_trailing_fuzzy_threshold",
            "content_validation_min_expected_chars",
            "duration_smoothing_alpha",
            "duration_stats_auto_save",
            "duration_stats_save_interval",
        )
        for key in passthrough_keys:
            if kwargs.get(key) is not None:
                config_kwargs[key] = kwargs[key]

        self.config = Gemini38TTSConfig(**config_kwargs)
        self.debug_save_rejected: bool = debug_tts
        self.target_language = target_language or "en"
        self.speaker_metadata = normalize_speaker_metadata_map(speaker_metadata)
        self.speaker_genders = build_effective_speaker_gender_map(self.speaker_metadata)

        self.cost_tracker = cost_tracker
        self.translator = translator

        self.api_client = Gemini38APIClient(self.config)

        audio_embedder = None
        if self.config.enable_voice_matching:
            audio_embedder = AudioEmbedder(device=embedding_model_device)
        self.voice_matcher = VoiceMatcher(audio_embedder, self.config.enable_voice_matching)

        self.voice_sample_manager = VoiceSampleManager(
            tts_provider="gemini38",
            voice_list=list(ALL_GEMINI_VOICES),
            samples_dir=DEFAULT_SAMPLES_DIR,
            stats_file=DURATION_STATS_FILE,
            adjustments_file=DURATION_ADJUSTMENTS_FILE,
            sample_text=DURATION_SAMPLE_TEXT,
            audio_embedder=audio_embedder,
            voice_matcher=self.voice_matcher,
            enable_voice_matching=self.config.enable_voice_matching,
            enable_audio_validation=self.config.enable_audio_validation,
            duration_smoothing_alpha=self.config.duration_smoothing_alpha,
            duration_stats_auto_save=self.config.duration_stats_auto_save,
            duration_stats_save_interval=self.config.duration_stats_save_interval,
            fade_detection_enabled=self.config.fade_detection_enabled,
            fade_window_size_ms=self.config.fade_window_size_ms,
            min_fade_db=self.config.min_fade_db,
            fade_detection_percentile=self.config.fade_detection_percentile,
        )

        self.speaker_reference_builder = SpeakerReferenceBuilder(
            audio_embedder,
            enabled=self.config.enable_voice_matching,
            min_clip_seconds=self.config.ref_min_clip_seconds,
            max_clip_seconds=self.config.ref_max_clip_seconds,
            target_seconds=self.config.ref_target_seconds,
            max_clips=self.config.ref_max_clips,
            min_clips=self.config.ref_min_clips,
            min_rms_dbfs=self.config.ref_min_rms_dbfs,
            silence_thresh_db=self.config.ref_silence_thresh_db,
        )

        self.voice_mapping: Dict[str, str] = {}
        self.voice_prompt_mapping: Dict[str, str] = {}
        self.emotion_enricher: Optional[Gemini38EmotionEnricher] = None

        self._speaker_profiles: Dict[str, SpeakerReferenceProfile] = {}
        self._speaker_profile_lock = threading.Lock()
        self._voice_embeddings_lock = threading.Lock()
        self._content_validator: Optional[Any] = None
        self._content_validator_lock = threading.Lock()
        self._extended_voice_names: Optional[List[str]] = None
        self._extended_voice_lock = threading.Lock()

        self._audio_cache: Dict[str, Tuple[str, float]] = {}
        self._cache_dir: Optional[str] = None
        self._fallback_output_paths: set = set()
        self._fallback_paths_lock = threading.Lock()

    # ------------------------------------------------------------------
    # TTSInterface plumbing
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self.api_client.client is not None

    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_mapping = mapping or {}
        logger.debug("Gemini 3.8 voice mapping set: %d entries", len(self.voice_mapping))

    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_prompt_mapping = mapping or {}
        logger.debug("Gemini 3.8 voice prompt mapping set: %d entries", len(self.voice_prompt_mapping))

    def clone_voice(self, audio_path: str, voice_id: str) -> str:
        raise NotImplementedError(
            "Gemini 3.8 voice replication requires a consent recording from the speaker "
            "and is intentionally not supported by DubbLM."
        )

    def initialize(self) -> None:
        self.api_client.initialize()
        self._cache_dir = tempfile.mkdtemp(prefix="gemini38_tts_cache_")

        if self.config.enable_emotion_enrichment and self.config.enable_llm_editor:
            logger.info("Skipping runtime emotion enrichment because the LLM editor pass is enabled.")
            self.config.enable_emotion_enrichment = False

        if self.config.enable_emotion_enrichment:
            try:
                self.emotion_enricher = Gemini38EmotionEnricher(self.config)
                self.emotion_enricher.initialize()
                if not self.emotion_enricher.llm:
                    self.config.enable_emotion_enrichment = False
                    self.emotion_enricher = None
            except Exception as exc:
                logger.error("Failed to initialize emotion enricher: %s", exc)
                self.config.enable_emotion_enrichment = False
                self.emotion_enricher = None

        if self.config.enable_voice_matching and not self.voice_matcher.audio_embedder:
            logger.warning("AudioEmbedder not initialized. Voice matching disabled.")
            self.config.enable_voice_matching = False

        def generate_gemini38_sample(voice_name: str, output_path: str) -> bool:
            try:
                audio_data = self.api_client.synthesize_chunk(DURATION_SAMPLE_TEXT, style="", voice=voice_name)
                if audio_data:
                    AudioFileUtils.save_wave_file(output_path, audio_data)
                    return True
                return False
            except Exception as exc:
                logger.error("Error generating sample for %s: %s", voice_name, exc)
                return False

        self.voice_sample_manager.set_sample_generator(generate_gemini38_sample)

        try:
            if not self.voice_sample_manager.generate_all_samples():
                logger.warning("Duration analysis initialization failed. Using default estimates.")
        except Exception as exc:
            logger.error("Error during duration analysis initialization: %s", exc)

        self.voice_sample_manager.load_biases()

    def cleanup(self) -> None:
        try:
            self.voice_sample_manager.save_duration_stats()
        except Exception as exc:
            logger.debug("Could not persist duration stats on cleanup: %s", exc)
        self._audio_cache.clear()
        if self._cache_dir and os.path.isdir(self._cache_dir):
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_fallback_output(self, output_path: str) -> bool:
        with self._fallback_paths_lock:
            return os.path.abspath(output_path) in self._fallback_output_paths

    def get_voice_duration_stats(self, voice_name: Optional[str] = None) -> Dict[str, Any]:
        return self.voice_sample_manager.get_voice_duration_stats(voice_name)

    def generate_voice_samples(self, force_regenerate: bool = False) -> bool:
        return self.voice_sample_manager.generate_all_samples(force_regenerate)

    def save_duration_stats(self) -> None:
        self.voice_sample_manager.save_duration_stats()

    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: Optional[str] = None,
    ) -> Optional[float]:
        language = language or self.target_language
        voice_name = segment_data.voice
        if not voice_name and segment_data.speaker:
            voice_name = self.voice_mapping.get(segment_data.speaker)
        if not voice_name:
            voice_name = self._get_gender_aware_default_voice(segment_data.speaker)
        voice_name = self._validate_voice_name(voice_name)
        return self.voice_sample_manager.estimate_duration(
            text=segment_data.text,
            voice_name=voice_name,
            language=language,
            style_prompt=segment_data.style_prompt,
            emotion=segment_data.emotion,
            speed=segment_data.speed,
            speaker_id=segment_data.speaker,
            apply_biases=True,
        )

    # ------------------------------------------------------------------
    # synthesize()
    # ------------------------------------------------------------------

    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> List[SegmentAlignment]:
        if not segments_data:
            return []
        language = language or self.target_language
        previous_texts = list(kwargs.get("previous_context") or [])

        usage_tracker: Dict[str, Any] = {"models": {}, "lock": threading.Lock()}
        temp_dir = tempfile.mkdtemp(prefix="gemini38_segments_")
        alignments: List[SegmentAlignment] = []
        total_segments = len(segments_data)

        try:
            for index, segment in enumerate(segments_data):
                alignment = self._process_segment(
                    segment,
                    index,
                    total_segments,
                    temp_dir,
                    language,
                    previous_texts,
                    usage_tracker,
                )
                if alignment is not None:
                    alignments.append(alignment)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self._flush_usage(usage_tracker)
        return alignments

    # ------------------------------------------------------------------
    # Per-segment processing
    # ------------------------------------------------------------------

    def _process_segment(
        self,
        segment: TTSSegmentData,
        segment_index: int,
        total_segments: int,
        temp_dir: str,
        language: str,
        context_segments: List[str],
        usage_tracker: Optional[Dict[str, Any]],
    ) -> Optional[SegmentAlignment]:
        segment_path = os.path.join(temp_dir, f"segment_{segment_index}_{segment.speaker}.wav")
        report_index = segment.segment_index if getattr(segment, "segment_index", None) is not None else segment_index
        report_group = getattr(segment, "group_id", None)

        cache_key = self._get_cache_key(segment, language)
        cached = self._audio_cache.get(cache_key)
        if cached is not None and os.path.exists(cached[0]):
            cached_path, cached_duration = cached
            voice_name = self._resolve_voice_for_segment(segment)
            valid, reason, similarity = self._validate_voice_consistency(
                cached_path,
                voice_name,
                speaker_id=segment.speaker,
                reference_audio_path=segment.reference_audio_path,
            )
            if not valid:
                logger.warning(
                    "Gemini 3.8: cached segment %d/%d failed re-validation (%s) — re-synthesizing",
                    segment_index + 1,
                    total_segments,
                    reason,
                )
                try:
                    os.remove(cached_path)
                except OSError:
                    pass
                self._audio_cache.pop(cache_key, None)
            else:
                shutil.copy(cached_path, segment_path)
                self._write_output(segment, segment_path)
                self._record_segment_report(
                    SegmentSynthesisReport(
                        segment_index=report_index,
                        speaker=segment.speaker,
                        text=segment.text,
                        requested_model=self.config.model,
                        actual_model=self.config.model,
                        attempts=0,
                        used_fallback=False,
                        success=True,
                        duration_seconds=cached_duration,
                        output_path=segment.output_path or segment_path,
                        group_id=report_group,
                        voice_similarity=similarity,
                        voice_validation=f"cached; {reason}",
                    )
                )
                return self._make_alignment(segment, cached_duration)

        try:
            text_used, model_used, is_valid = self._synthesize_segment(
                segment,
                segment_path,
                language,
                previous_segments=context_segments,
                usage_tracker=usage_tracker,
            )
            duration = AudioFileUtils.get_audio_duration_seconds(segment_path) or 0.0
            if is_valid:
                self._record_duration_stats(segment, text_used, duration, language)
            if duration and model_used:
                self._register_usage(usage_tracker, model_used, audio_seconds=duration)

            used_fallback = model_used is not None and model_used != self.config.model
            if self._cache_dir and is_valid and not used_fallback:
                cache_path = os.path.join(self._cache_dir, f"{cache_key}.wav")
                shutil.copy(segment_path, cache_path)
                self._audio_cache[cache_key] = (cache_path, duration)
            if used_fallback and segment.output_path:
                with self._fallback_paths_lock:
                    self._fallback_output_paths.add(os.path.abspath(segment.output_path))

            self._write_output(segment, segment_path)
            self._record_segment_report(
                SegmentSynthesisReport(
                    segment_index=report_index,
                    speaker=segment.speaker,
                    text=text_used or segment.text,
                    requested_model=self.config.model,
                    actual_model=model_used,
                    attempts=1 if is_valid else max(1, self.config.primary_model_max_retries),
                    used_fallback=bool(used_fallback),
                    success=bool(is_valid),
                    duration_seconds=duration,
                    output_path=segment.output_path or segment_path,
                    group_id=report_group,
                )
            )
            return self._make_alignment(segment, duration)
        except Exception as exc:
            logger.error(
                "Error synthesizing segment %d for speaker '%s': %s",
                segment_index + 1,
                segment.speaker,
                exc,
            )
            self._record_segment_report(
                SegmentSynthesisReport(
                    segment_index=report_index,
                    speaker=segment.speaker,
                    text=segment.text,
                    requested_model=self.config.model,
                    actual_model=None,
                    attempts=0,
                    used_fallback=False,
                    success=False,
                    duration_seconds=0.0,
                    output_path=None,
                    group_id=report_group,
                    error=str(exc),
                )
            )
            return None

    def _write_output(self, segment: TTSSegmentData, source_path: str) -> None:
        if not segment.output_path:
            return
        os.makedirs(os.path.dirname(segment.output_path), exist_ok=True)
        shutil.copy(source_path, segment.output_path)

    @staticmethod
    def _make_alignment(segment: TTSSegmentData, duration: float) -> SegmentAlignment:
        diarized = DiarizationSegment(
            start_time=0.0,
            end_time=duration,
            speaker=segment.speaker,
            text=segment.text,
            confidence=1.0,
        )
        return SegmentAlignment(
            original_segment=segment,
            diarized_segment=diarized,
            alignment_confidence=1.0,
        )

    # ------------------------------------------------------------------
    # Core synthesis with validation + retries
    # ------------------------------------------------------------------

    def _synthesize_segment(
        self,
        segment: TTSSegmentData,
        output_path: str,
        language: str,
        previous_segments: Optional[List[str]] = None,
        usage_tracker: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[str], bool]:
        working_text = segment.text or ""
        if self.config.enable_emotion_enrichment and self.emotion_enricher:
            working_text = self.emotion_enricher.enrich_text(working_text, previous_segments=previous_segments)
            # LLM enrichment can drift outside the documented tag vocabulary;
            # keep the text (and reports) compliant with the 3.8 spec.
            working_text = sanitize_angle_tags(working_text)

        context_style: Optional[str] = None
        if self.config.enable_context_style and segment.context_style and segment.context_style.strip():
            context_style = segment.context_style.strip()

        prompt = self._build_prompt(segment, working_text, context_style=context_style)
        voice_name = self._resolve_voice_for_segment(segment)

        models: List[str] = [self.config.model]
        if self.config.fallback_model and self.config.fallback_model != self.config.model:
            models.append(self.config.fallback_model)

        last_good_model: Optional[str] = None
        last_reason = "not attempted"

        for model in models:
            attempts = self.api_client._attempts_for(model)
            for _attempt in range(attempts):
                if usage_tracker is not None:
                    input_tokens = self._count_tokens(prompt.text) + self._count_tokens(prompt.style)
                    self._register_usage(usage_tracker, model, input_tokens=input_tokens)

                if len(prompt.text) > MAX_CHAR_LIMIT_PER_REQUEST:
                    audio_bytes = self._synthesize_by_sentence_split(prompt, voice_name, model)
                else:
                    audio_bytes = self.api_client.synthesize_chunk(
                        prompt.text, prompt.style, voice_name, model=model
                    )
                if not audio_bytes:
                    last_reason = f"empty audio from {model}"
                    continue

                AudioFileUtils.save_wave_file(output_path, audio_bytes)
                valid, reason = self._validate_audio(output_path)
                if valid:
                    voice_valid, voice_reason, _sim = self._validate_voice_consistency(
                        output_path,
                        voice_name,
                        speaker_id=segment.speaker,
                        reference_audio_path=segment.reference_audio_path,
                    )
                    if voice_valid:
                        content_valid, content_reason, _score = self._validate_segment_content(
                            output_path, strip_tts_markup(prompt.text), context_style=context_style
                        )
                        if content_valid:
                            return working_text, model, True
                        reason = content_reason
                        if context_style and "quote leak" in content_reason.lower():
                            logger.warning(
                                "Gemini 3.8 context quote leaked into speech for speaker '%s' — "
                                "retrying without context",
                                segment.speaker,
                            )
                            context_style = None
                            prompt = self._build_prompt(segment, working_text, context_style=None)
                    else:
                        reason = voice_reason
                last_good_model = model
                last_reason = reason
                logger.debug(
                    "Gemini 3.8 validation failed for speaker '%s' (model=%s): %s",
                    segment.speaker,
                    model,
                    reason,
                )

        # Rephrase loop: only meaningful when the model produced audio but the
        # validation rejected it (silence/content/voice drift).
        if self.translator and last_good_model:
            for attempt in range(MAX_REPHRASE_ATTEMPTS):
                if "silence" not in last_reason.lower() and "content" not in last_reason.lower():
                    break
                rephrased = self._rephrase_for_tts_clarity(working_text, language, reason=last_reason)
                if not rephrased:
                    break
                rephrased = sanitize_angle_tags(rephrased)
                prompt = self._build_prompt(segment, rephrased, context_style=context_style)
                audio_bytes = self.api_client.synthesize_chunk(
                    prompt.text, prompt.style, voice_name, model=self.config.model
                )
                if not audio_bytes:
                    continue
                AudioFileUtils.save_wave_file(output_path, audio_bytes)
                valid, reason = self._validate_audio(output_path)
                if valid:
                    logger.info(
                        "Gemini 3.8 rephrase attempt %d succeeded for speaker '%s'",
                        attempt + 1,
                        segment.speaker,
                    )
                    return rephrased, self.config.model, True
                working_text = rephrased
                last_reason = reason

        if last_good_model:
            logger.warning(
                "Gemini 3.8: keeping best invalid attempt for speaker '%s' (%s)",
                segment.speaker,
                last_reason,
            )
            return working_text, last_good_model, False
        return working_text, None, False

    def _build_prompt(self, segment: TTSSegmentData, text: str, context_style: Optional[str] = None):
        speaker_prompt = self.voice_prompt_mapping.get(segment.speaker, "")
        return build_gemini38_prompt(
            text,
            prompt_prefix=self.config.prompt_prefix,
            speaker_prompt=speaker_prompt,
            segment_style=segment.style_prompt,
            emotion=segment.emotion,
            context_style=context_style,
            max_style_chars=self.config.style_max_chars,
        )

    def _synthesize_by_sentence_split(self, prompt, voice_name: str, model: Optional[str] = None) -> bytes:
        chunks = greedy_sent_split(prompt.text, MAX_CHAR_LIMIT_PER_REQUEST)
        part_paths: List[str] = []
        temp_dir = tempfile.mkdtemp(prefix="gemini38_split_")
        try:
            for index, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                audio = self.api_client.synthesize_chunk(chunk, prompt.style, voice_name, model=model)
                if not audio:
                    continue
                part_path = os.path.join(temp_dir, f"part_{index}.wav")
                AudioFileUtils.save_wave_file(part_path, audio)
                part_paths.append(part_path)
            if not part_paths:
                return b""
            return AudioFileUtils.concatenate_audio_files(part_paths, sample_rate=SAMPLE_RATE)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _rephrase_for_tts_clarity(self, original_text: str, language: str, reason: str = "high silence content") -> Optional[str]:
        if not self.translator or not self.translator.is_available():
            return None
        try:
            return self.translator.adjust_segment_text_length(
                original_text=original_text,
                source_language=language,
                target_language=language,
                desired_ratio=1.0,
                target_char_count=len(original_text),
                context_info={
                    "domain": "tts_optimization",
                    "tone": "clear",
                    "themes": ["clarity", "pronunciation"],
                },
                max_attempts=2,
                tts_system="gemini38",
            )
        except Exception as exc:
            logger.error("Error rephrasing text: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _fade_config(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.fade_detection_enabled,
            "window_size_ms": self.config.fade_window_size_ms,
            "min_fade_db": self.config.min_fade_db,
            "percentile": self.config.fade_detection_percentile,
        }

    def _validate_audio(self, audio_path: str) -> Tuple[bool, str]:
        if not self.config.enable_audio_validation:
            return True, "Audio validation disabled"
        try:
            is_valid, reason, _ratio = AudioValidator.validate_audio_sample(
                audio_path,
                expected_min_duration=0.2,
                max_silence_ratio=0.2,
                trailing_silence_grace_ratio=self.config.trailing_silence_grace_ratio,
                fade_detection_config=self._fade_config(),
                max_total_silence_ratio=self.config.max_total_silence_ratio,
                max_contiguous_silence_seconds=self.config.max_contiguous_silence_seconds,
            )
            return bool(is_valid), reason
        except Exception as exc:
            logger.debug("Audio validation error: %s", exc)
            return True, f"validation error: {exc}"

    def _validate_voice_consistency(
        self,
        audio_path: str,
        expected_voice_name: str,
        speaker_id: Optional[str] = None,
        reference_audio_path: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[float]]:
        if not self.config.enable_voice_consistency_validation:
            return True, "Voice consistency check disabled", None
        if not self.config.enable_voice_matching or not self.voice_matcher.audio_embedder:
            return True, "Voice matching unavailable", None

        duration = AudioFileUtils.get_audio_duration_seconds(Path(audio_path))
        if duration is None or duration < self.config.min_voice_validation_duration_seconds:
            return True, "Clip too short for voice validation", None

        embedding = self.voice_matcher.extract_embedding_for_audio_file(audio_path)
        if embedding is None:
            return True, "Could not extract embedding", None

        sample_similarity: Optional[float] = None
        if expected_voice_name in self.voice_matcher.sample_embeddings:
            raw_similarity = self.voice_matcher.get_voice_similarity(embedding, expected_voice_name)
            if raw_similarity is not None:
                sample_similarity = float(raw_similarity)

        ref_similarity: Optional[float] = None
        if speaker_id:
            profile = self._speaker_profiles.get(speaker_id)
            if profile is not None and profile.centroid is not None:
                ref_similarity = float(
                    np.dot(
                        embedding / (np.linalg.norm(embedding) + 1e-12),
                        profile.centroid / (np.linalg.norm(profile.centroid) + 1e-12),
                    )
                )

        mode = (self.config.voice_reference_mode or "max").lower()
        if mode == "voice_sample":
            similarity = sample_similarity
        elif mode == "speaker_ref":
            similarity = ref_similarity
        else:
            candidates = [value for value in (sample_similarity, ref_similarity) if value is not None]
            similarity = max(candidates) if candidates else None

        if similarity is None:
            return True, "No reference embedding available", None

        if self.config.enable_voice_gender_relative_check and sample_similarity is not None and speaker_id:
            speaker_gender = self._get_effective_speaker_gender(speaker_id)
            expected_gender = get_gemini_voice_gender(expected_voice_name)
            if speaker_gender in {"male", "female"} and expected_gender in {"male", "female"}:
                worst_opposite = None
                for voice_name in self.voice_matcher.sample_embeddings:
                    voice_gender = get_gemini_voice_gender(voice_name)
                    if voice_gender not in {"male", "female"} or voice_gender == speaker_gender:
                        continue
                    sim = self.voice_matcher.get_voice_similarity(embedding, voice_name)
                    if sim is None:
                        continue
                    sim = float(sim)
                    if worst_opposite is None or sim > worst_opposite:
                        worst_opposite = sim
                if worst_opposite is not None and worst_opposite > sample_similarity + self.config.voice_gender_relative_margin:
                    return (
                        False,
                        f"Closer to opposite-gender voice ({worst_opposite:.3f} > {sample_similarity:.3f})",
                        similarity,
                    )

        if similarity < self.config.voice_similarity_relaxed_threshold:
            return False, f"Voice similarity {similarity:.3f} below relaxed threshold", similarity
        if similarity < self.config.voice_similarity_threshold:
            return True, f"Voice similarity {similarity:.3f} below target but accepted", similarity
        return True, f"Voice similarity {similarity:.3f}", similarity

    def _get_content_validator(self) -> Optional[Any]:
        if not self.config.enable_content_validation:
            return None
        provider = str(self.config.content_validator_provider or "whisper").lower()
        with self._content_validator_lock:
            if self._content_validator is not None:
                return self._content_validator
            try:
                if provider == "assemblyai":
                    if not ASSEMBLYAI_VALIDATOR_AVAILABLE or SpeakerSegmentDiarizer is None:
                        return None
                    self._content_validator = SpeakerSegmentDiarizer(
                        language_code=self.target_language,
                        speech_model=self.config.content_validator_speech_model,
                    )
                else:
                    if not LOCAL_WHISPER_VALIDATOR_AVAILABLE or LocalWhisperContentValidator is None:
                        return None
                    self._content_validator = LocalWhisperContentValidator(
                        language_code=self.target_language,
                        model_name=self.config.content_validator_whisper_model,
                        compute_type=self.config.content_validator_whisper_compute_type,
                        cpu_threads=self.config.content_validator_whisper_cpu_threads,
                    )
            except Exception as exc:
                logger.warning("Content validator init failed: %s. Content checks disabled.", exc)
                self.config.enable_content_validation = False
                self._content_validator = None
            return self._content_validator

    @staticmethod
    def _normalize_for_asr_match(text: str) -> str:
        if not text:
            return ""
        import re

        lowered = text.lower()
        lowered = re.sub(r"[^\w\s'à-ÿÀ-ßа-яА-ЯёЁ]", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    def _validate_segment_content(
        self,
        audio_path: str,
        expected_text: str,
        context_style: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[float]]:
        if not self.config.enable_content_validation:
            return True, "Content validation disabled", None
        if not RAPIDFUZZ_AVAILABLE or _rapidfuzz_fuzz is None:
            return True, "rapidfuzz not installed", None

        cleaned_expected = self._normalize_for_asr_match(expected_text)
        if len(cleaned_expected) < max(1, int(self.config.content_validation_min_expected_chars)):
            return True, "Expected text too short for ASR check", None

        validator = self._get_content_validator()
        if validator is None:
            return True, "Content validator unavailable", None

        validation_seconds = AudioFileUtils.get_audio_duration_seconds(Path(audio_path)) or 0.0
        if self.cost_tracker and validation_seconds > 0:
            provider = str(self.config.content_validator_provider or "whisper").lower()
            model_name = (
                self.config.content_validator_speech_model
                if provider == "assemblyai"
                else self.config.content_validator_whisper_model
            )
            self.cost_tracker.add_transcription_usage(
                provider,
                float(validation_seconds),
                model=str(model_name),
                category="tts_content_validation",
            )

        try:
            asr_text = validator.transcribe_text(audio_path)
        except Exception as exc:
            logger.warning("Content validation ASR failed: %s", exc)
            return True, f"ASR error: {exc}", None

        asr_normalized = self._normalize_for_asr_match(asr_text)
        if not asr_normalized:
            return False, "ASR produced empty text (speech not detected)", 0.0

        leaked_fragment = quoted_fragment_leak(asr_text, expected_text, context_style)
        if leaked_fragment:
            return (
                False,
                f"Context quote leak: quoted fragment '{leaked_fragment[:60]}' was spoken",
                0.0,
            )

        overall = float(_rapidfuzz_fuzz.token_set_ratio(cleaned_expected, asr_normalized))
        expected_words = cleaned_expected.split()
        tail_count = max(1, int(self.config.content_trailing_word_count))
        tail_text = " ".join(expected_words[-tail_count:])
        tail_score = float(_rapidfuzz_fuzz.partial_ratio(tail_text, asr_normalized))

        if tail_score < float(self.config.content_trailing_fuzzy_threshold):
            return (
                False,
                f"Trailing words missing (tail='{tail_text}', tail_score={tail_score:.1f}, overall={overall:.1f})",
                overall,
            )
        if overall < float(self.config.content_similarity_threshold):
            return (
                False,
                f"Content similarity {overall:.1f} below threshold {self.config.content_similarity_threshold:.1f}",
                overall,
            )
        return True, f"Content match (overall={overall:.1f}, tail={tail_score:.1f})", overall

    # ------------------------------------------------------------------
    # Voice resolution + multi-sample matching
    # ------------------------------------------------------------------

    def _canonicalize_voice_name(self, voice_name: Optional[str]) -> Optional[str]:
        if not voice_name:
            return None
        normalized_voices = {voice.lower(): voice for voice in ALL_GEMINI_VOICES}
        return normalized_voices.get(str(voice_name).strip().lower())

    def _validate_voice_name(self, voice_name: str) -> str:
        canonical = self._canonicalize_voice_name(voice_name)
        if canonical:
            return canonical
        logger.warning(
            "Voice '%s' is not in the prebuilt catalog. Using default '%s'.",
            voice_name,
            self.config.default_voice,
        )
        return self.config.default_voice

    def _get_effective_speaker_gender(self, speaker_id: Optional[str]) -> str:
        normalized = str(speaker_id or "").strip()
        if not normalized:
            return "unknown"
        return self.speaker_genders.get(normalized, "unknown")

    def _get_gender_aware_default_voice(self, speaker_id: Optional[str]) -> str:
        speaker_gender = self._get_effective_speaker_gender(speaker_id)
        fallback = resolve_default_gemini_voice(
            speaker_gender,
            preferred_voice=self.config.default_voice,
            blocked_voices=self.config.blocked_voices,
        )
        return self._validate_voice_name(fallback)

    def _build_voice_search_exclusions(self, speaker_id: Optional[str]) -> List[str]:
        exclusions: List[str] = []
        seen: set = set()
        for raw in list(self.voice_mapping.values()) + list(self.config.blocked_voices or []):
            canonical = self._canonicalize_voice_name(raw)
            if canonical and canonical not in seen:
                seen.add(canonical)
                exclusions.append(canonical)
        speaker_gender = self._get_effective_speaker_gender(speaker_id)
        if speaker_gender not in {"male", "female"}:
            return exclusions
        for voice_name, voice_gender in GEMINI_VOICE_GENDERS.items():
            if voice_gender != speaker_gender and voice_name not in seen:
                seen.add(voice_name)
                exclusions.append(voice_name)
        return exclusions

    def _get_or_build_profile(
        self,
        speaker_id: str,
        reference_audio_path: Optional[str],
        time_ranges: Optional[List[Tuple[float, float]]] = None,
        original_audio_path: Optional[str] = None,
    ) -> Optional[SpeakerReferenceProfile]:
        if not speaker_id:
            return None
        with self._speaker_profile_lock:
            existing = self._speaker_profiles.get(speaker_id)
        if existing is not None:
            return existing
        if not self.speaker_reference_builder.is_available():
            return None
        profile = self.speaker_reference_builder.build_profile(
            speaker_id,
            reference_audio_path,
            SPEAKER_REFERENCE_DIR,
            time_ranges=time_ranges,
            original_audio_path=original_audio_path,
        )
        if profile is not None:
            with self._speaker_profile_lock:
                self._speaker_profiles[speaker_id] = profile
        return profile

    def _ensure_extended_voice_embeddings(self, speaker_gender: str) -> None:
        if not self.config.enable_extended_voice_library:
            return
        if self.api_client.client is None:
            return
        with self._extended_voice_lock:
            if self._extended_voice_names is None:
                names: List[str] = []
                try:
                    response = self.api_client.client.voices.list(
                        type_=["prebuilt"],
                        page_size=min(1000, max(1, self.config.extended_library_shortlist)),
                    )
                    for voice in getattr(response, "voices", None) or []:
                        voice_id = getattr(voice, "id", None) or getattr(voice, "name", None)
                        voice_gender = getattr(voice, "gender", None)
                        if not voice_id:
                            continue
                        if speaker_gender in {"male", "female"} and voice_gender in {"male", "female"}:
                            if voice_gender != speaker_gender:
                                continue
                        names.append(str(voice_id))
                except Exception as exc:
                    logger.warning("Extended voice library list failed: %s", exc)
                self._extended_voice_names = names
            shortlist = self._extended_voice_names[: max(0, int(self.config.extended_library_shortlist))]

        for voice_name in shortlist:
            if not self._canonicalize_voice_name(voice_name):
                continue
            if voice_name in self.voice_matcher.sample_embeddings:
                continue
            with self._voice_embeddings_lock:
                if voice_name in self.voice_matcher.sample_embeddings:
                    continue
                try:
                    self.voice_sample_manager.regenerate_voice_sample(voice_name)
                except Exception as exc:
                    logger.debug("Could not build embedding for extended voice '%s': %s", voice_name, exc)

    def find_and_pin_voice_for_speaker(
        self,
        speaker_id: str,
        reference_audio_path: Union[str, Path],
        force_search: bool = False,
        time_ranges: Optional[List[Tuple[float, float]]] = None,
        original_audio_path: Optional[str] = None,
    ) -> Optional[str]:
        if not self.is_available():
            raise RuntimeError("Gemini 3.8 TTS not initialized.")

        speaker_gender = self._get_effective_speaker_gender(speaker_id)

        if not self.config.enable_voice_matching:
            return self.voice_mapping.get(speaker_id) or self._get_gender_aware_default_voice(speaker_id)
        if not force_search and speaker_id in self.voice_mapping:
            return self.voice_mapping[speaker_id]
        if not self.voice_matcher.audio_embedder:
            return self._get_gender_aware_default_voice(speaker_id)

        profile = self._get_or_build_profile(
            speaker_id,
            str(reference_audio_path) if reference_audio_path else None,
            time_ranges=time_ranges,
            original_audio_path=original_audio_path,
        )
        if profile is None:
            return self._get_gender_aware_default_voice(speaker_id)

        self._ensure_extended_voice_embeddings(speaker_gender)
        exclusions = self._build_voice_search_exclusions(speaker_id)
        best_voice = best_voice_for_profile(profile, self.voice_matcher, exclude_voices=exclusions)

        if best_voice:
            best_voice = self._canonicalize_voice_name(best_voice) or best_voice
            voice_gender = get_gemini_voice_gender(best_voice)
            if speaker_gender in {"male", "female"} and voice_gender in {"male", "female"} and voice_gender != speaker_gender:
                logger.warning(
                    "Rejecting voice '%s' for speaker '%s': gender mismatch (%s)",
                    best_voice,
                    speaker_id,
                    speaker_gender,
                )
                best_voice = None

        if best_voice:
            ref_duration = AudioFileUtils.get_audio_duration_seconds(Path(reference_audio_path))
            if ref_duration is None or ref_duration >= MIN_REFERENCE_DURATION_SECONDS:
                self.voice_mapping[speaker_id] = best_voice
                logger.info(
                    "Pinned speaker '%s' to voice '%s' via multi-sample reference (clips=%d, gender=%s)",
                    speaker_id,
                    best_voice,
                    profile.clip_count,
                    speaker_gender,
                )
            return best_voice

        fallback = self._get_gender_aware_default_voice(speaker_id)
        logger.warning(
            "No matching voice for speaker '%s'. Using gender-aware default '%s'.",
            speaker_id,
            fallback,
        )
        return fallback

    def _get_voice_for_speaker(self, speaker_id: str, segment_hint: TTSSegmentData) -> str:
        return (
            segment_hint.voice
            or self.voice_mapping.get(speaker_id)
            or self._get_gender_aware_default_voice(speaker_id)
        )

    def _resolve_voice_for_segment(self, segment_data: TTSSegmentData) -> str:
        speaker_id = segment_data.speaker or ""
        if segment_data.voice:
            return self._validate_voice_name(segment_data.voice)
        if speaker_id in self.voice_mapping:
            return self._validate_voice_name(self.voice_mapping[speaker_id])
        if self.config.enable_voice_matching and segment_data.reference_audio_path:
            matched = self.find_and_pin_voice_for_speaker(
                speaker_id,
                segment_data.reference_audio_path,
                time_ranges=getattr(segment_data, "speaker_time_ranges", None),
                original_audio_path=getattr(segment_data, "source_audio_path", None),
            )
            if matched:
                return matched
        return self._get_gender_aware_default_voice(speaker_id)

    # ------------------------------------------------------------------
    # Stats / usage / cache key
    # ------------------------------------------------------------------

    def _record_duration_stats(
        self,
        segment: TTSSegmentData,
        synthesized_text: Optional[str],
        duration_seconds: float,
        language: str,
    ) -> None:
        text_for_stats = (synthesized_text or segment.text or "").strip()
        voice_name = self._resolve_voice_for_segment(segment)
        self.voice_sample_manager.record_duration_stats(
            text=text_for_stats,
            voice_name=voice_name,
            duration_seconds=duration_seconds,
            language=language,
            style_prompt=segment.style_prompt,
            emotion=segment.emotion,
            speed=segment.speed,
            speaker_id=segment.speaker,
        )

    def _register_usage(
        self,
        usage_tracker: Optional[Dict[str, Any]],
        model_name: str,
        input_tokens: float = 0.0,
        audio_seconds: float = 0.0,
        output_tokens: float = 0.0,
    ) -> None:
        if not usage_tracker or not model_name:
            return
        models_map = usage_tracker.setdefault("models", {})
        lock = usage_tracker.get("lock")

        def _update(target: Dict[str, float]) -> None:
            if input_tokens:
                target["input_tokens"] = target.get("input_tokens", 0.0) + float(input_tokens)
            if output_tokens:
                target["output_tokens"] = target.get("output_tokens", 0.0) + float(output_tokens)
            if audio_seconds:
                target["audio_seconds"] = target.get("audio_seconds", 0.0) + float(audio_seconds)

        if lock:
            with lock:
                _update(models_map.setdefault(model_name, {}))
        else:
            _update(models_map.setdefault(model_name, {}))

    def _flush_usage(self, usage_tracker: Dict[str, Any]) -> None:
        if not self.cost_tracker:
            return
        models_map = usage_tracker.get("models") or {}
        for model_name, usage in models_map.items():
            self.cost_tracker.add_tts_actual(
                "gemini",
                model=model_name,
                input_tokens=usage.get("input_tokens", 0.0),
                output_tokens=usage.get("output_tokens", 0.0),
                audio_seconds=usage.get("audio_seconds", 0.0),
            )

    def _count_tokens(self, text: Optional[str]) -> int:
        if not text:
            return 0
        if not (TIKTOKEN_AVAILABLE and tiktoken is not None):
            return max(1, len(text) // 4)
        encoding = None
        try:
            encoding = tiktoken.encoding_for_model(self.config.model)
        except Exception:
            encoding = None
        if encoding is None:
            encoding = tiktoken.get_encoding("cl100k_base")
        try:
            return len(encoding.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def _get_cache_key(self, segment_data: TTSSegmentData, language: str) -> str:
        voice_name = (
            segment_data.voice
            or self.voice_mapping.get(segment_data.speaker)
            or self._get_gender_aware_default_voice(segment_data.speaker)
        )
        style_prompt = segment_data.style_prompt or self.voice_prompt_mapping.get(segment_data.speaker, "")
        key_data = {
            "output_path": segment_data.output_path or "",
            "text": segment_data.text,
            "speaker": segment_data.speaker,
            "voice": voice_name,
            "style_prompt": style_prompt,
            "prompt_prefix": self.config.prompt_prefix,
            "context_style": segment_data.context_style or "",
            "emotion": segment_data.emotion or "Neutral",
            "speed": segment_data.speed or 1.0,
            "language": language,
            "model": self.config.model,
            "dialect": "gemini38",
        }
        return hashlib.sha256(str(sorted(key_data.items())).encode()).hexdigest()
