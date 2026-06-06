from typing import Optional, Dict, Any, List, Union, Tuple
import os
import wave
import io
import time
import json
import re
import tempfile
import shutil
import hashlib
import math
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .models import (
    TTSSegmentData, 
    SegmentAlignment,
    DiarizationSegment,
    SegmentSynthesisReport,
    BatchSynthesisReport,
)
from .gemini_voice_catalog import (
    ALL_GEMINI_VOICES,
    is_gemini_voice_gender_compatible,
    resolve_default_gemini_voice,
)
from .voice_sample_manager import VoiceSampleManager, AudioFileUtils, TextAnalysisUtils, AudioValidator
from src.tts.tts_interface import TTSInterface
from src.utils.sent_split import greedy_sent_split, split_at_sentence_midpoint
from src.utils.audio_embedder import AudioEmbedder
from src.utils.batch_alignment import find_temporal_conflicts
from src.utils.speaker_gender import (
    build_effective_speaker_gender_map,
    normalize_speaker_metadata_map,
)
from src.utils.voice_matcher import VoiceMatcher
from pydantic import BaseModel, Field
from src.dubbing.core.log_config import get_logger
from src.dubbing.tts_styles import TTS_STYLE_PROMPTS

# Import Google GenAI dependencies, with error handling for missing packages
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Optional diarizer import (requires extra deps like rapidfuzz/assemblyai)
try:
    from src.tts.speaker_segment_diarizer import SpeakerSegmentDiarizer
    from src.tts.speaker_segment_diarizer.models import DialogueLine, SpeakerSegment
    DIARIZER_AVAILABLE = True
except Exception:
    SpeakerSegmentDiarizer = None  # type: ignore[assignment]
    DialogueLine = None  # type: ignore[assignment]
    SpeakerSegment = None  # type: ignore[assignment]
    DIARIZER_AVAILABLE = False

try:
    from src.tts.local_whisper_validator import LocalWhisperContentValidator
    LOCAL_WHISPER_VALIDATOR_AVAILABLE = True
except Exception:
    LocalWhisperContentValidator = None  # type: ignore[assignment]
    LOCAL_WHISPER_VALIDATOR_AVAILABLE = False

# Optional imports for voice matching features
try:
    import torch
    import torchaudio
    from pydub import AudioSegment
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    _rapidfuzz_fuzz = None  # type: ignore[assignment]
    RAPIDFUZZ_AVAILABLE = False

logger = get_logger(__name__)

# Constants
# Resolve samples directory relative to this file's location (src/tts/)
DEFAULT_SAMPLES_DIR = (Path(__file__).parent / "samples" / "gemini").resolve()

# Duration analysis sample texts combined into one comprehensive text
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
    "Innovation drives progress across industries, from healthcare and transportation to entertainment and beyond."
]

# Combine all sample texts into one comprehensive text for duration analysis
DURATION_SAMPLE_TEXT = " ".join(DURATION_SAMPLE_TEXTS)

EMBEDDING_CACHE_FILE = DEFAULT_SAMPLES_DIR / "gemini_voice_stats.json"
DURATION_STATS_FILE = DEFAULT_SAMPLES_DIR / "gemini_voice_stats.json"
# Separate adjustments (biases) file
DURATION_ADJUSTMENTS_FILE = DEFAULT_SAMPLES_DIR / "gemini_duration_adjustments.json"
MAX_CHAR_LIMIT_PER_REQUEST = 1024*30
SAMPLE_RATE = 24000
MAX_SENTENCE_SPLIT_DEPTH = 5
MIN_CHARS_FOR_SENTENCE_SPLIT = 80

# Single source of truth for default Gemini TTS model names used across the
# backend, presets, and cost estimator. Bump here to roll out a new default.
DEFAULT_GEMINI_TTS_MODEL = "gemini-2.5-pro-preview-tts"
DEFAULT_GEMINI_TTS_FALLBACK_MODEL = "gemini-2.5-flash-preview-tts"
EXPERIMENTAL_GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"


class GeminiTTSConfig(BaseModel):
    """Configuration for Gemini TTS."""
    model: str = DEFAULT_GEMINI_TTS_MODEL
    # Fallback model used only when primary fails all retries for a segment.
    # Resets to primary after each segment (per-segment reset semantics).
    fallback_model: Optional[str] = DEFAULT_GEMINI_TTS_FALLBACK_MODEL
    fallback_model_max_retries: int = 2
    default_voice: str = "Kore"
    embedding_model_device: Optional[str] = None
    enable_voice_matching: bool = True
    max_retries: int = 10
    retry_delay_base: float = 2.0
    prompt_prefix: str = TTS_STYLE_PROMPTS["podcast"]
    blocked_voices: List[str] = []  # Voice names excluded from auto-matching (style-driven policy)
    enable_audio_validation: bool = True  # Allow disabling validation for debugging
    enable_emotion_enrichment: bool = False  # Enable emotion enrichment using LLM
    enable_llm_editor: bool = False  # Skip runtime enrichment when editor already handled markup
    emotion_enrichment_model: str = "gemini-2.5-pro"  # Model for emotion enrichment
    emotion_enrichment_temperature: float = 0.7  # Temperature for emotion enrichment
    enable_laughter_enrichment: bool = False  # Include laughter tags in enrichment
    max_workers: int = 4  # Maximum number of parallel workers for segment synthesis
    duration_smoothing_alpha: float = 0.35  # EMA smoothing factor for runtime stats
    duration_stats_auto_save: bool = True  # Persist runtime stats automatically
    duration_stats_save_interval: int = 20  # Save after this many updates
    fade_detection_enabled: bool = True  # Enable volume fade detection for trailing silence
    fade_window_size_ms: int = 500  # Window size for fade analysis in milliseconds
    min_fade_db: float = 10.0  # Minimum dB drop to consider as significant fade
    fade_detection_percentile: int = 75  # Percentile for reference level calculation
    # Guardrails for pathological internal silence that trailing checks can miss.
    # NOTE: temporarily relaxed 10x to stop false-positives from rejecting valid slices.
    max_total_silence_ratio: float = 3.5  # was 0.35
    max_contiguous_silence_seconds: float = 80.0  # was 8.0
    # Allow slight trailing-silence exceedance without relaxing total/internal silence checks.
    trailing_silence_grace_ratio: float = 0.05  # was 0.005
    # Retry budget for the model for a segment/batch before giving up.
    primary_model_max_retries: int = 4
    # Validate generated segment voice against the cached sample voice embedding.
    enable_voice_consistency_validation: bool = True
    # Main quality threshold for cosine similarity to expected voice sample.
    voice_similarity_threshold: float = 0.70
    # Relaxed threshold allowed when we must pick the best invalid fallback.
    voice_similarity_relaxed_threshold: float = 0.65
    # Skip voice consistency checks for very short clips (unstable embeddings).
    min_voice_validation_duration_seconds: float = 1.0
    # Multi-speaker (2-speaker) batching configuration
    enable_multi_speaker: bool = True
    # Hard cap: keep under ~2/3 of the 32k context window (doc: TTS context is 32k)
    max_batch_tokens: int = 8000
    # Soft caps for quality drift control (configurable). Kept intentionally tight:
    # Gemini multi-speaker reliably degrades (merges/skips turns) past ~8 lines,
    # which then makes the diarizer unable to assign one ASR utterance per line.
    max_batch_chars: int = 2000
    max_batch_duration_seconds: float = 90.0
    # Soft target for batch size. The batcher may exceed this by up to
    # ``max_batch_lines_hard`` when ``cohesion_with_prev=="tight"`` keeps a line
    # glued to its neighbours; otherwise it flushes at this cap.
    max_batch_lines: int = 4
    # Absolute ceiling — never grow a batch above this, regardless of cohesion.
    max_batch_lines_hard: int = 6
    # Minimum batch size at which a "loose" cohesion hint is allowed to trigger
    # an early flush. Prevents cutting a 1-line batch just because the next line
    # is flagged loose.
    min_batch_lines_for_loose_cut: int = 2
    multi_speaker_retries: int = 3
    # Content validation: ASR round-trip check that TTS actually spoke all
    # expected words. Catches segments where speech was truncated at the end,
    # or where the model swallowed a chunk of the prompt.
    enable_content_validation: bool = True
    # ASR provider used for single-speaker round-trip content validation.
    # Keep AssemblyAI available for compatibility; local Whisper is the default
    # because it is cheap and good enough for coverage checks.
    content_validator_provider: str = "whisper"
    content_validator_whisper_model: str = "base"
    content_validator_whisper_compute_type: str = "int8"
    content_validator_whisper_cpu_threads: int = 2
    # AssemblyAI model used when content_validator_provider == "assemblyai".
    content_validator_speech_model: str = "nano"
    # Overall token_set_ratio (0..100) threshold for expected-vs-ASR.
    content_similarity_threshold: float = 70.0
    # Last N expected words must be findable in ASR text (truncation guard).
    content_trailing_word_count: int = 3
    # partial_ratio threshold for the trailing-tail fuzzy check (0..100).
    content_trailing_fuzzy_threshold: float = 70.0
    # Skip content validation for very short clips — ASR is unreliable there
    # and expected text is too small to give a meaningful fuzzy score.
    content_validation_min_expected_chars: int = 8








class GeminiAPIClient:
    """Handles Gemini API communication."""
    
    def __init__(self, config: GeminiTTSConfig):
        self.config = config
        self.client: Optional[genai.Client] = None
        self.current_model = config.model

    def initialize(self) -> None:
        """Initialize the Google GenAI client."""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not provided. Please set GOOGLE_API_KEY environment variable.")
        
        try:
            self.client = genai.Client(api_key=api_key)
            logger.info(f"Gemini TTS client initialized. Target model: {self.current_model}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google GenAI client: {str(e)}")

    def set_model(self, model_name: str) -> bool:
        """Set a specific model for subsequent synthesis attempts."""
        if not model_name:
            return False
        if self.current_model == model_name:
            return False
        old_model = self.current_model
        self.current_model = model_name
        logger.debug(f"Switched from model {old_model} to model {self.current_model}")
        return True

    def synthesize_chunk(self, content: str, speech_config: genai_types.SpeechConfig) -> bytes:
        """Make a single Gemini TTS API call and return PCM audio data."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized.")
        
        if len(content) > MAX_CHAR_LIMIT_PER_REQUEST:
            logger.warning(f"Warning: Content length ({len(content)} chars) exceeds {MAX_CHAR_LIMIT_PER_REQUEST}. Truncating.")
            content = content[:MAX_CHAR_LIMIT_PER_REQUEST]

        content_config = genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config
        )

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.current_model,
                    contents=content,
                    config=content_config,
                )
                
                if response and response.candidates and response.candidates[0].content:
                    for part in response.candidates[0].content.parts:
                        inline = getattr(part, "inline_data", None)
                        if inline is None or not getattr(inline, "data", None):
                            continue
                        mime = (inline.mime_type or "").lower()
                        # Accept any audio/* PCM-like mime. New gemini-3.1 TTS returns
                        # "audio/l16; rate=24000; channels=1", older gemini-2.5 TTS returned
                        # "audio/L16;codec=pcm;rate=24000". Both are raw PCM @ 24kHz mono.
                        if mime.startswith("audio/") and "l16" in mime:
                            return inline.data
                
                # Extract text without prompt for logging
                log_text = content.split('\n', 1)[-1] if '\n' in content else content
                log_preview = f"{log_text[:30]}...{log_text[-30:]}" if len(log_text) > 70 else log_text
                # Log finish_reason/mime to speed up diagnostics next time.
                try:
                    cand = response.candidates[0] if response and response.candidates else None
                    finish = getattr(cand, "finish_reason", None) if cand else None
                    mime_seen = [
                        getattr(getattr(p, "inline_data", None), "mime_type", None)
                        for p in (cand.content.parts if cand and cand.content else [])
                    ] if cand else []
                    text_seen = [
                        (getattr(p, "text", None) or "")[:80]
                        for p in (cand.content.parts if cand and cand.content else [])
                    ] if cand else []
                except Exception:
                    finish, mime_seen, text_seen = None, [], []
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries}: No audio data in response "
                    f"for text: {log_preview} (finish={finish}, mimes={mime_seen}, text_parts={text_seen})"
                )
                if attempt + 1 >= self.config.max_retries:
                    logger.error(f"Gemini API call failed after {self.config.max_retries} attempts.")
                    return b''

            except Exception as e:
                err_msg = str(e)
                logger.error(f"Attempt {attempt + 1}/{self.config.max_retries} failed: {err_msg}")
                if attempt + 1 >= self.config.max_retries:
                    logger.error(f"Gemini API call failed after {self.config.max_retries} attempts.")
                    return b''
                time.sleep(self.config.retry_delay_base ** attempt)
        
        return b''


class EmotionEnricher:
    """Enriches text with emotional markup tags using Gemini LLM."""

    def __init__(self, config: GeminiTTSConfig):
        """
        Initialize the EmotionEnricher.

        Args:
            config: GeminiTTS configuration with enrichment settings
        """
        self.config = config
        self.llm = None  # Will be initialized with llama_index Gemini

    def initialize(self) -> None:
        """Initialize the Gemini LLM for emotion enrichment using llama_index."""
        try:
            from llama_index.llms.gemini import Gemini
        except ImportError:
            logger.error("llama_index.llms.gemini not available for emotion enrichment")
            return

        import os
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found for emotion enrichment")
            return

        try:
            self.llm = Gemini(
                model=self.config.emotion_enrichment_model,
                temperature=self.config.emotion_enrichment_temperature
            )
            logger.info(f"Emotion enrichment LLM initialized with model: {self.config.emotion_enrichment_model}")
        except Exception as e:
            logger.error(f"Failed to initialize emotion enrichment LLM: {e}")
            self.llm = None

    @property
    def enrichment_prompt(self) -> str:
        """Get the enrichment prompt template."""

        # Mode 1 non-speech sounds documented as High reliability for gemini-2.5-pro-preview-tts.
        non_speech_sounds = "[sigh], [uhm]"
        if self.config.enable_laughter_enrichment:
            non_speech_sounds = "[sigh], [laughing], [uhm]"

        return f"""You are an expert at enriching text with expressive audio tags for text-to-speech synthesis with gemini-2.5-pro-preview-tts.

Your task is to analyze the given text IN CONTEXT of the conversation and add appropriate markup tags to make the speech sound more natural and emotionally expressive.

Available markup tags (only these — the model ignores or mis-speaks other tags):
1. Non-speech sounds (the tag becomes an audible vocalization): {non_speech_sounds}
2. Style modifiers that change the delivery of the following phrase:
   [sarcasm], [robotic], [shouting], [whispering], [extremely fast]
3. Pacing: [short pause] (~250ms), [medium pause] (~500ms), [long pause] (~1000ms+)

Guidelines:
- Use tags sparingly — subtlety is key. Most explanatory lines should contain zero non-speech sounds
- Do NOT invent other emotion tags (no [excited], [scared], [curious], [amazed], [crying], etc.).
  Mode 3 vocalized adjectives are intentionally excluded because the model often speaks the tag
  word itself; set emotional tone through the style prompt instead
- Place style modifiers at the very beginning of the phrase they should affect
- [shouting] / [whispering] / [extremely fast] work best when the text itself already implies
  the delivery (yelling, quiet aside, rapid disclaimer). Do not apply them to neutral lines
- Do not add [uhm] just to make the line sound casual. Use it only when the text clearly implies
  hesitation, stumbling, or a self-interruption
- Never add more than one non-speech sound tag to a single line
- Prefer pause markers over filler/breath tags for natural rhythm
- The input may already contain markup tags from an earlier editing step; keep useful existing
  tags, improve them if needed, and avoid duplicating or stacking similar tags
- If a pause or modifier tag is already present, you may reposition, replace, or remove it to
  improve delivery, but do not blindly add more tags on top
- Preserve the original text exactly, only add or adjust markup tags where appropriate

{{context_section}}

Current text to enrich:
{{text}}

Return ONLY the enriched version of the CURRENT text with markup tags. Do not add explanations, comments, or repeat the context."""

    def enrich_text(self, text: str, previous_segments: Optional[List[str]] = None) -> str:
        """
        Enrich text with emotional markup tags using conversation context.

        Args:
            text: The current text to enrich
            previous_segments: List of previous text segments for context (up to 3-5 recent segments)

        Returns:
            Text enriched with markup tags
        """
        if not text or not text.strip():
            logger.debug("Emotion enrichment: Empty text provided, skipping")
            return text

        try:
            logger.debug(f"Emotion enrichment: Starting enrichment for text: '{text[:50]}...'")

            # Build context section from previous segments
            context_section = ""
            if previous_segments and len(previous_segments) > 0:
                context_lines = "\n".join([f"- {seg}" for seg in previous_segments[-5:]])  # Use last 5 segments
                context_section = f"Previous conversation context:\n{context_lines}\n"
                logger.debug(f"Emotion enrichment: Using {len(previous_segments[-5:])} previous segments as context")
            else:
                logger.debug("Emotion enrichment: No previous segments for context")

            # Check if LLM is initialized
            if not self.llm:
                logger.warning("Emotion enrichment: LLM not initialized. Using original text.")
                return text

            # Build the prompt
            prompt = self.enrichment_prompt.format(
                context_section=context_section,
                text=text
            )
            logger.debug(f"Emotion enrichment: Calling API with model {self.config.emotion_enrichment_model}")

            # Call Gemini API for text enrichment using llama_index (same as translator)
            response = self.llm.complete(prompt)

            logger.debug(f"Emotion enrichment: Response received - type: {type(response)}, has text: {hasattr(response, 'text') if response else False}")

            # Extract response text (same pattern as llm_translator.py)
            if hasattr(response, "text"):
                response_text = response.text
            else:
                response_text = str(response)

            logger.debug(f"Emotion enrichment: Response text exists, length: {len(response_text) if response_text else 0}")

            if response_text:
                enriched_text = response_text.strip()
                logger.debug(f"Emotion enrichment: Enriched text after strip: '{enriched_text[:100]}...'")

                # Basic validation: ensure the enriched text is not empty and not too different
                if enriched_text and len(enriched_text) <= len(text) * 3:
                    logger.debug(f"Emotion enrichment SUCCESS:\n  Original: {text}\n  Enriched: {enriched_text}")
                    return enriched_text
                else:
                    logger.warning(f"Emotion enrichment: Text validation failed - empty: {not enriched_text}, too long: {len(enriched_text) > len(text) * 3}")
                    logger.warning(f"Enriched text seems invalid (too long or empty). Using original text.")
                    logger.debug(f"  Original: {text}\n  Enriched: {enriched_text}")
                    return text
            else:
                logger.warning("Emotion enrichment: Response text is empty/None. Using original text.")
                return text

        except Exception as e:
            logger.error(f"Error during text enrichment: {e}", exc_info=True)
            logger.error(f"Exception type: {type(e).__name__}, args: {e.args}")
            return text


class SpeechConfigBuilder:
    """Builds speech configurations for different scenarios."""
    
    @staticmethod
    def build_single_speaker_config(voice_name: str) -> genai_types.SpeechConfig:
        """Build speech config for single speaker synthesis."""
        return genai_types.SpeechConfig(
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        )

    @staticmethod
    def build_multi_speaker_config(speaker_to_voice: Dict[str, str]) -> genai_types.SpeechConfig:
        """Build speech config for multi-speaker synthesis (up to 2 speakers)."""
        speaker_items = list(speaker_to_voice.items())
        if not speaker_items:
            raise ValueError("speaker_to_voice must contain at least one entry")
        if len(speaker_items) > 2:
            raise ValueError("Gemini multi-speaker TTS supports up to 2 speakers per request")

        speaker_voice_configs = [
            genai_types.SpeakerVoiceConfig(
                speaker=speaker,
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=voice_name)
                ),
            )
            for speaker, voice_name in speaker_items
        ]

        return genai_types.SpeechConfig(
            multi_speaker_voice_config=genai_types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=speaker_voice_configs
            )
        )



class GeminiTTSWrapper(TTSInterface):
    """Google Gemini TTS wrapper with simplified single-segment synthesis."""

    supports_segment_batching: bool = True

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_TTS_MODEL,
        fallback_model: str = DEFAULT_GEMINI_TTS_FALLBACK_MODEL,
        default_voice: str = "Kore",
        embedding_model_device: Optional[str] = None,
        enable_voice_matching: bool = True,
        enable_audio_validation: bool = True,
        prompt_prefix: Optional[str] = None,
        blocked_voices: Optional[List[str]] = None,
        debug_tts: bool = False,
        enable_emotion_enrichment: bool = False,
        enable_llm_editor: bool = False,
        emotion_enrichment_model: Optional[str] = None,
        emotion_enrichment_temperature: Optional[float] = None,
        max_workers: Optional[int] = None,
        duration_smoothing_alpha: Optional[float] = None,
        duration_stats_auto_save: Optional[bool] = None,
        duration_stats_save_interval: Optional[int] = None,
        cost_tracker: Optional[Any] = None,
        translator: Optional[Any] = None,
        target_language: Optional[str] = None,
        speaker_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any
    ):
        """
        Initialize Gemini TTS wrapper.
        
        Args:
            translator: Optional LLMTranslator instance for text rephrasing when TTS produces
                       segments with excessive silence (>3%). If provided, the system will
                       automatically attempt to rephrase problematic segments up to 3 times
                       before aborting.
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI SDK is not installed.")

        config_kwargs: Dict[str, Any] = {
            "model": model,
            "fallback_model": fallback_model,
            "default_voice": default_voice,
            "embedding_model_device": embedding_model_device,
            "enable_voice_matching": enable_voice_matching,
            "enable_audio_validation": enable_audio_validation,
            "prompt_prefix": prompt_prefix or "",
            "blocked_voices": list(blocked_voices or []),
            "enable_emotion_enrichment": enable_emotion_enrichment,
            "enable_llm_editor": enable_llm_editor,
            "emotion_enrichment_model": emotion_enrichment_model or "gemini-2.5-pro",
            "emotion_enrichment_temperature": emotion_enrichment_temperature if emotion_enrichment_temperature is not None else 0.7,
        }
        if max_workers is not None:
            config_kwargs["max_workers"] = max_workers
        if duration_smoothing_alpha is not None:
            config_kwargs["duration_smoothing_alpha"] = duration_smoothing_alpha
        if duration_stats_auto_save is not None:
            config_kwargs["duration_stats_auto_save"] = duration_stats_auto_save
        if duration_stats_save_interval is not None:
            config_kwargs["duration_stats_save_interval"] = duration_stats_save_interval
        if kwargs.get("max_total_silence_ratio") is not None:
            config_kwargs["max_total_silence_ratio"] = kwargs["max_total_silence_ratio"]
        if kwargs.get("max_contiguous_silence_seconds") is not None:
            config_kwargs["max_contiguous_silence_seconds"] = kwargs["max_contiguous_silence_seconds"]
        if kwargs.get("rescue_model") is not None:
            config_kwargs["rescue_model"] = kwargs["rescue_model"]
        if kwargs.get("max_silence_ratio_for_invalid_fallback") is not None:
            config_kwargs["max_silence_ratio_for_invalid_fallback"] = kwargs["max_silence_ratio_for_invalid_fallback"]
        if kwargs.get("trailing_silence_grace_ratio") is not None:
            config_kwargs["trailing_silence_grace_ratio"] = kwargs["trailing_silence_grace_ratio"]
        if kwargs.get("primary_model_max_retries") is not None:
            config_kwargs["primary_model_max_retries"] = kwargs["primary_model_max_retries"]
        if kwargs.get("fallback_model_max_retries") is not None:
            config_kwargs["fallback_model_max_retries"] = kwargs["fallback_model_max_retries"]
        if kwargs.get("enable_voice_consistency_validation") is not None:
            config_kwargs["enable_voice_consistency_validation"] = kwargs["enable_voice_consistency_validation"]
        if kwargs.get("voice_similarity_threshold") is not None:
            config_kwargs["voice_similarity_threshold"] = kwargs["voice_similarity_threshold"]
        if kwargs.get("voice_similarity_relaxed_threshold") is not None:
            config_kwargs["voice_similarity_relaxed_threshold"] = kwargs["voice_similarity_relaxed_threshold"]
        if kwargs.get("min_voice_validation_duration_seconds") is not None:
            config_kwargs["min_voice_validation_duration_seconds"] = kwargs["min_voice_validation_duration_seconds"]
        for _content_key in (
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
        ):
            if kwargs.get(_content_key) is not None:
                config_kwargs[_content_key] = kwargs[_content_key]

        self.config = GeminiTTSConfig(**config_kwargs)
        # Save rejected/silent attempts when debugging is enabled
        self.debug_save_rejected: bool = debug_tts
        self.target_language = target_language or "en"
        self.speaker_metadata = normalize_speaker_metadata_map(speaker_metadata)
        self.speaker_genders = build_effective_speaker_gender_map(self.speaker_metadata)

        # Initialize components
        self.api_client = GeminiAPIClient(self.config)

        audio_embedder = None
        if enable_voice_matching:
            audio_embedder = AudioEmbedder(device=embedding_model_device)

        self.voice_matcher = VoiceMatcher(audio_embedder, enable_voice_matching)
        
        # Initialize VoiceSampleManager
        self.voice_sample_manager = VoiceSampleManager(
            tts_provider="gemini",
            voice_list=ALL_GEMINI_VOICES,
            samples_dir=DEFAULT_SAMPLES_DIR,
            stats_file=DURATION_STATS_FILE,
            adjustments_file=DURATION_ADJUSTMENTS_FILE,
            sample_text=DURATION_SAMPLE_TEXT,
            audio_embedder=audio_embedder,
            voice_matcher=self.voice_matcher,
            enable_voice_matching=enable_voice_matching,
            enable_audio_validation=enable_audio_validation,
            duration_smoothing_alpha=self.config.duration_smoothing_alpha,
            duration_stats_auto_save=self.config.duration_stats_auto_save,
            duration_stats_save_interval=self.config.duration_stats_save_interval,
            fade_detection_enabled=self.config.fade_detection_enabled,
            fade_window_size_ms=self.config.fade_window_size_ms,
            min_fade_db=self.config.min_fade_db,
            fade_detection_percentile=self.config.fade_detection_percentile,
        )

        # Emotion enricher (will be initialized after API client is ready)
        self.emotion_enricher: Optional[EmotionEnricher] = None
        self.diarizer: Optional[SpeakerSegmentDiarizer] = None
        self._content_validator: Optional[Any] = None
        self._content_validator_lock = threading.Lock()

        # Voice mappings
        self.voice_mapping: Dict[str, str] = {}
        self.voice_prompt_mapping: Dict[str, str] = {}

        # Audio cache similar to OpenAI
        self._audio_cache: Dict[str, tuple[str, float]] = {}  # Maps cache_key to (file_path, duration)
        self._cache_dir = None
        self.cost_tracker = cost_tracker
        self.translator = translator

        # Track output paths produced by fallback/rescue models (not primary).
        # smart_dubbing checks this to avoid caching fallback results at the
        # pipeline level, allowing them to be regenerated with the primary model.
        self._fallback_output_paths: set = set()
        self._fallback_paths_lock = threading.Lock()

    def is_fallback_output(self, output_path: str) -> bool:
        """Check if a given output path was produced by a fallback/rescue model."""
        with self._fallback_paths_lock:
            return os.path.abspath(output_path) in self._fallback_output_paths

    def _register_usage(
        self,
        usage_tracker: Optional[Dict[str, Any]],
        model_name: str,
        input_tokens: float = 0.0,
        audio_seconds: float = 0.0,
        output_tokens: float = 0.0
    ) -> None:
        """Thread-safe accumulation of TTS usage metrics per model."""
        if not usage_tracker or not model_name:
            return

        models_map = usage_tracker.setdefault("models", {})

        def _update(target: Dict[str, float]) -> None:
            if input_tokens:
                target["input_tokens"] = target.get("input_tokens", 0.0) + float(input_tokens)
            if output_tokens:
                target["output_tokens"] = target.get("output_tokens", 0.0) + float(output_tokens)
            if audio_seconds:
                target["audio_seconds"] = target.get("audio_seconds", 0.0) + float(audio_seconds)

        lock = usage_tracker.get("lock")
        if lock:
            with lock:
                model_usage = models_map.setdefault(model_name, {})
                _update(model_usage)
        else:
            model_usage = models_map.setdefault(model_name, {})
            _update(model_usage)

    def _count_tokens(self, text: Optional[str]) -> int:
        """Count tokens for billing, using tiktoken when possible."""
        if not text:
            return 0

        if not (TIKTOKEN_AVAILABLE and tiktoken is not None):
            raise RuntimeError("tiktoken is required for Gemini TTS token counting. Install the 'tiktoken' package.")

        encoding = None
        try:
            encoding = tiktoken.encoding_for_model(self.config.model)
        except Exception:
            encoding = None
        if encoding is None:
            encoding = tiktoken.get_encoding("cl100k_base")

        try:
            return len(encoding.encode(text))
        except Exception as exc:
            raise RuntimeError(f"Failed to tokenize text for model '{self.config.model}': {exc}")

    def _build_tts_prompt(self, prompt_parts: List[str], transcript: str) -> str:
        """Join style instructions with a clear transcript boundary."""
        cleaned_transcript = transcript.strip()
        cleaned_parts = [part.strip() for part in prompt_parts if part and part.strip()]
        if not cleaned_parts:
            return cleaned_transcript

        return (
            f"{' '.join(cleaned_parts)}\n\n"
            "Transcript to synthesize starts below. Speak only the transcript text, not the instructions.\n"
            f"{cleaned_transcript}"
        )

    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        """Set a mapping of speaker IDs to Gemini voice names."""
        self.voice_mapping = mapping
        logger.debug(f"Gemini voice mapping set: {len(mapping)} entries.")

    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        """Set a mapping of speaker IDs to voice style prompts."""
        self.voice_prompt_mapping = mapping
        logger.debug(f"Gemini voice prompt mapping set: {len(mapping)} entries.")

    def initialize(self) -> None:
        """Initialize the Gemini TTS system."""
        self.api_client.initialize()

        # Create cache directory
        self._cache_dir = tempfile.mkdtemp(prefix="gemini_tts_cache_")
        logger.debug(f"Gemini audio cache directory: {self._cache_dir}")

        if self.config.enable_multi_speaker:
            try:
                if not DIARIZER_AVAILABLE or SpeakerSegmentDiarizer is None:
                    raise RuntimeError("speaker_segment_diarizer dependencies are not available")
                self.diarizer = SpeakerSegmentDiarizer(language_code=self.target_language)
            except Exception as exc:
                logger.error(f"Failed to initialize SpeakerSegmentDiarizer: {exc}. Multi-speaker disabled.")
                self.config.enable_multi_speaker = False
                self.diarizer = None

        if self.config.enable_emotion_enrichment and self.config.enable_llm_editor:
            logger.info(
                "Skipping runtime emotion enrichment because the LLM editor pass is enabled."
            )
            self.config.enable_emotion_enrichment = False

        # Initialize emotion enricher if enabled
        if self.config.enable_emotion_enrichment:
            try:
                self.emotion_enricher = EmotionEnricher(self.config)
                self.emotion_enricher.initialize()
                if self.emotion_enricher.llm:
                    logger.info(f"Emotion enrichment enabled with model: {self.config.emotion_enrichment_model}")
                else:
                    logger.error("Failed to initialize emotion enricher LLM. Emotion enrichment disabled.")
                    self.config.enable_emotion_enrichment = False
                    self.emotion_enricher = None
            except Exception as e:
                logger.error(f"Failed to initialize emotion enricher: {e}. Emotion enrichment disabled.")
                self.config.enable_emotion_enrichment = False
                self.emotion_enricher = None
        else:
            logger.info("Emotion enrichment disabled by configuration.")

        if self.config.enable_voice_matching:
            if not self.voice_matcher.audio_embedder:
                logger.warning("AudioEmbedder not initialized. Voice matching disabled.")
                self.config.enable_voice_matching = False
        else:
            logger.info("Voice matching disabled by configuration.")

        # Set up sample generation callback for VoiceSampleManager
        def generate_gemini_sample(voice_name: str, output_path: str) -> bool:
            """Callback for VoiceSampleManager to generate Gemini samples."""
            try:
                speech_config = SpeechConfigBuilder.build_single_speaker_config(voice_name)
                audio_data = self.api_client.synthesize_chunk(DURATION_SAMPLE_TEXT, speech_config)
                if audio_data:
                    AudioFileUtils.save_wave_file(output_path, audio_data)
                    return True
                return False
            except Exception as e:
                logger.error(f"Error generating sample for {voice_name}: {e}")
                return False
        
        self.voice_sample_manager.set_sample_generator(generate_gemini_sample)

        # Initialize duration analysis
        try:
            logger.info("Initializing voice duration analysis...")
            success = self.voice_sample_manager.generate_all_samples()
            if success:
                logger.info("Duration analysis initialized successfully.")
            else:
                logger.warning("Warning: Duration analysis initialization failed. Using default estimates.")
        except Exception as e:
            logger.error(f"Error during duration analysis initialization: {e}")

        # Load persisted biases
        self.voice_sample_manager.load_biases()

    def _estimate_segment_duration_seconds(self, segment: TTSSegmentData, language: str) -> float:
        """Estimate segment duration (best-effort) for batching heuristics."""
        try:
            estimate = self.estimate_audio_segment_length(segment, language=language)
        except Exception:
            estimate = None
        if estimate is not None and estimate > 0:
            return float(estimate)
        text = (segment.text or "").strip()
        return max(len(text) / 15.0, 0.1)

    def _build_multi_speaker_batches(self, segments: List[TTSSegmentData], language: str) -> List[List[TTSSegmentData]]:
        """Greedily build maximal batches under 2-speaker + token/quality constraints."""
        if not segments:
            return []

        batches: List[List[TTSSegmentData]] = []
        current: List[TTSSegmentData] = []
        current_speakers: set[str] = set()
        current_tokens = 0
        current_chars = 0
        current_duration = 0.0

        def flush() -> None:
            nonlocal current, current_speakers, current_tokens, current_chars, current_duration
            if current:
                batches.append(current)
            current = []
            current_speakers = set()
            current_tokens = 0
            current_chars = 0
            current_duration = 0.0

        soft_line_cap = int(self.config.max_batch_lines)
        hard_line_cap = max(soft_line_cap, int(self.config.max_batch_lines_hard))
        min_loose_cut = max(1, int(self.config.min_batch_lines_for_loose_cut))

        for seg in segments:
            if not seg.speaker or not (seg.text or "").strip():
                continue

            seg_speaker = str(seg.speaker)
            seg_text = str(seg.text)
            seg_tokens = 0
            try:
                seg_tokens = self._count_tokens(seg_text)
            except Exception:
                seg_tokens = int(math.ceil(len(seg_text) / 4.0))
            seg_chars = len(seg_text)
            seg_duration = self._estimate_segment_duration_seconds(seg, language)
            cohesion = (getattr(seg, "cohesion_with_prev", None) or "normal").lower()
            if cohesion not in {"tight", "normal", "loose"}:
                cohesion = "normal"

            current_len = len(current)
            would_exceed_speakers = (seg_speaker not in current_speakers and len(current_speakers) >= 2)
            would_exceed_tokens = (current_tokens + seg_tokens) > int(self.config.max_batch_tokens)
            would_exceed_chars = (current_chars + seg_chars) > int(self.config.max_batch_chars)
            would_exceed_duration = (current_duration + seg_duration) > float(self.config.max_batch_duration_seconds)

            would_exceed_lines_hard = (current_len + 1) > hard_line_cap
            would_exceed_lines_soft = (
                (current_len + 1) > soft_line_cap and cohesion != "tight"
            )
            early_cut_on_loose = (
                current_len >= min_loose_cut and cohesion == "loose"
            )

            should_flush = bool(current) and (
                would_exceed_speakers
                or would_exceed_tokens
                or would_exceed_chars
                or would_exceed_duration
                or would_exceed_lines_hard
                or would_exceed_lines_soft
                or early_cut_on_loose
            )

            if should_flush:
                if early_cut_on_loose and not (
                    would_exceed_speakers
                    or would_exceed_tokens
                    or would_exceed_chars
                    or would_exceed_duration
                    or would_exceed_lines_hard
                    or would_exceed_lines_soft
                ):
                    logger.debug(
                        f"Gemini batcher: loose-hint early cut before '{seg_speaker}' "
                        f"(batch size {current_len}, soft cap {soft_line_cap})"
                    )
                elif (
                    (current_len + 1) > soft_line_cap
                    and not would_exceed_lines_hard
                    and cohesion == "tight"
                ):
                    # defensive: this branch is unreachable because
                    # `would_exceed_lines_soft` is False here, but log is cheap.
                    logger.debug(
                        f"Gemini batcher: flushing despite tight hint at '{seg_speaker}' "
                        f"(batch size {current_len}, other caps exceeded)"
                    )
                flush()
            elif (
                bool(current)
                and (current_len + 1) > soft_line_cap
                and cohesion == "tight"
            ):
                logger.debug(
                    f"Gemini batcher: extending past soft cap ({soft_line_cap}) on tight hint "
                    f"for '{seg_speaker}' (batch size {current_len} -> {current_len + 1}, hard cap {hard_line_cap})"
                )

            current.append(seg)
            current_speakers.add(seg_speaker)
            current_tokens += seg_tokens
            current_chars += seg_chars
            current_duration += seg_duration

        flush()
        return self._post_process_batches(batches, language)

    # Segments shorter than this (after strip) are treated as at-risk "Хм."-style
    # interjections that Gemini tends to drop when synthesized alone. We pull a
    # neighbour of a different speaker into their batch so synthesis goes through
    # the multi-speaker path with contextual turns.
    _SHORT_SEGMENT_CHAR_LIMIT: int = 30

    # Relative overflow (0..1) allowed when attaching a single segment to an
    # adjacent batch during ``_post_process_batches``. Tiny overflow is a good
    # trade: it converts a doomed solo-batch with a short interjection into a
    # multi-speaker batch without meaningfully changing prompt size.
    _POST_PASS_CAP_SLACK: float = 0.2

    def _batch_fits_caps(
        self,
        batch: List[TTSSegmentData],
        language: str,
        slack: float = 0.0,
    ) -> bool:
        """Return True if ``batch`` respects every configured multi-speaker cap.

        ``slack`` allows a small relative overflow (e.g. 0.2 = +20%) on numeric
        caps (chars/duration/tokens) and a +1 tolerance on ``max_batch_lines``.
        Used by post-pass merging to absorb one neighbour segment even when the
        resulting batch sits just above the configured limits.
        """
        if not batch:
            return True
        line_tolerance = 1 if slack > 0 else 0
        if len(batch) > int(self.config.max_batch_lines) + line_tolerance:
            return False
        scale = 1.0 + max(slack, 0.0)
        total_chars = sum(len(str(s.text or "")) for s in batch)
        if total_chars > int(self.config.max_batch_chars * scale):
            return False
        total_duration = sum(self._estimate_segment_duration_seconds(s, language) for s in batch)
        if total_duration > float(self.config.max_batch_duration_seconds) * scale:
            return False
        total_tokens = 0
        for s in batch:
            text = str(s.text or "")
            try:
                total_tokens += self._count_tokens(text)
            except Exception:
                total_tokens += int(math.ceil(len(text) / 4.0))
        if total_tokens > int(self.config.max_batch_tokens * scale):
            return False
        return True

    def _should_avoid_solo_batch(self, batch: List[TTSSegmentData]) -> bool:
        """Solo-speaker batches that contain at least one short line are at risk:
        Gemini drops them when the style prefix dominates a tiny content string.
        """
        speakers = {str(s.speaker) for s in batch if s.speaker}
        if len(speakers) != 1:
            return False
        return any(len(str(s.text or "").strip()) <= self._SHORT_SEGMENT_CHAR_LIMIT for s in batch)

    def _post_process_batches(
        self,
        batches: List[List[TTSSegmentData]],
        language: str,
    ) -> List[List[TTSSegmentData]]:
        """Avoid single-speaker batches containing short interjections.

        For each such batch, try to steal the first segment of the next batch
        (if different speaker) or donate our first segment to the previous
        batch (if different speaker), provided the receiving batch still fits
        all caps. Segment order is preserved end-to-end.
        """
        if len(batches) <= 1:
            return batches

        i = 0
        while i < len(batches):
            b = batches[i]
            if not self._should_avoid_solo_batch(b):
                i += 1
                continue
            solo_speaker = str(b[0].speaker) if b else None

            stole = False
            if i + 1 < len(batches):
                nxt = batches[i + 1]
                if nxt and str(nxt[0].speaker) != solo_speaker:
                    candidate = b + [nxt[0]]
                    if self._batch_fits_caps(candidate, language, slack=self._POST_PASS_CAP_SLACK):
                        batches[i] = candidate
                        batches[i + 1] = nxt[1:]
                        if not batches[i + 1]:
                            del batches[i + 1]
                        stole = True

            if not stole and i > 0:
                prev = batches[i - 1]
                if prev and str(prev[-1].speaker) != solo_speaker:
                    candidate = prev + [b[0]]
                    if self._batch_fits_caps(candidate, language, slack=self._POST_PASS_CAP_SLACK):
                        batches[i - 1] = candidate
                        batches[i] = b[1:]
                        if not batches[i]:
                            del batches[i]
                            continue

            i += 1

        return [b for b in batches if b]

    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: Optional[str] = None
    ) -> Optional[float]:
        """Estimate audio segment length using VoiceSampleManager."""
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
            apply_biases=True
        )

    def get_voice_duration_stats(self, voice_name: Optional[str] = None) -> Dict[str, Any]:
        """Get duration statistics for a specific voice or all voices."""
        return self.voice_sample_manager.get_voice_duration_stats(voice_name)

    def regenerate_duration_analysis(self, force_regenerate: bool = True) -> bool:
        """Regenerate duration analysis samples and statistics."""
        if not self.is_available():
            raise RuntimeError("Gemini TTS not initialized.")
        
        logger.info("Regenerating duration analysis...")
        return self.voice_sample_manager.generate_all_samples(force_regenerate)

    def generate_voice_samples(self, force_regenerate: bool = False) -> bool:
        """Generate voice samples, analyze durations, and compute embeddings."""
        return self.voice_sample_manager.generate_all_samples(force_regenerate)

    def _canonicalize_voice_name(self, voice_name: Optional[str]) -> Optional[str]:
        """Resolve a case-insensitive Gemini voice name to its canonical form."""
        if not voice_name:
            return None
        normalized_voices = {voice.lower(): voice for voice in ALL_GEMINI_VOICES}
        return normalized_voices.get(str(voice_name).strip().lower())

    def _validate_voice_name(self, voice_name: str) -> str:
        """Validate that a voice name is supported by Gemini TTS."""
        canonical_name = self._canonicalize_voice_name(voice_name)
        if canonical_name:
            return canonical_name
        
        logger.warning(f"Warning: Voice '{voice_name}' not supported. Using default '{self.config.default_voice}'.")
        return self.config.default_voice

    def _get_effective_speaker_gender(self, speaker_id: str) -> str:
        """Return the effective gender for a speaker or ``unknown``."""
        normalized_speaker = str(speaker_id or "").strip()
        if not normalized_speaker:
            return "unknown"
        return self.speaker_genders.get(normalized_speaker, "unknown")

    def _get_gender_aware_default_voice(self, speaker_id: str) -> str:
        """Resolve a stable default voice that respects the speaker gender."""
        speaker_gender = self._get_effective_speaker_gender(speaker_id)
        fallback_voice = resolve_default_gemini_voice(
            speaker_gender,
            preferred_voice=self.config.default_voice,
            blocked_voices=self.config.blocked_voices,
        )
        return self._validate_voice_name(fallback_voice)

    def _build_voice_search_exclusions(self, speaker_id: str) -> List[str]:
        """Build the exclusion list for similarity-based Gemini auto-pinning."""
        exclusions: List[str] = []
        seen: set[str] = set()

        for raw_voice_name in list(self.voice_mapping.values()) + list(self.config.blocked_voices or []):
            canonical_name = self._canonicalize_voice_name(raw_voice_name)
            if canonical_name and canonical_name not in seen:
                seen.add(canonical_name)
                exclusions.append(canonical_name)

        speaker_gender = self._get_effective_speaker_gender(speaker_id)
        if speaker_gender not in {"male", "female"}:
            return exclusions

        for voice_name in ALL_GEMINI_VOICES:
            if not is_gemini_voice_gender_compatible(voice_name, speaker_gender) and voice_name not in seen:
                seen.add(voice_name)
                exclusions.append(voice_name)

        return exclusions

    def _get_style_prompt_for_speaker(self, speaker_id: str, 
                                    segment_hint: Optional[TTSSegmentData] = None) -> str:
        """Determine the style prompt for a speaker."""
        speaker_prompt = self.voice_prompt_mapping.get(speaker_id, "").strip()
        segment_prompt = segment_hint.style_prompt.strip() if segment_hint and segment_hint.style_prompt else ""
        
        combined_parts = []
        if speaker_prompt:
            combined_parts.append(speaker_prompt)
        if segment_prompt:
            combined_parts.append(segment_prompt)
            
        style_prompt = " ".join(combined_parts)
        
        if not style_prompt and segment_hint:
            if segment_hint.emotion and segment_hint.emotion != "Neutral":
                # Get voice name for more natural prompts
                voice_name = self._get_voice_for_speaker(speaker_id, segment_hint)
                voice_name = self._validate_voice_name(voice_name)
                style_prompt = f"Make {voice_name} sound {segment_hint.emotion.lower()}"
        
        if style_prompt and not style_prompt.endswith(":"):
            style_prompt = style_prompt.strip() + ":"
        
        return style_prompt.strip() if style_prompt else ""

    def find_and_pin_voice_for_speaker(self, speaker_id: str, reference_audio_path: Union[str, Path],
                                     force_search: bool = False) -> Optional[str]:
        """Find the best matching Gemini voice for a reference audio and pin it to the speaker."""
        if not self.is_available():
            raise RuntimeError("Gemini TTS not initialized.")

        speaker_gender = self._get_effective_speaker_gender(speaker_id)

        if not self.config.enable_voice_matching:
            return self.voice_mapping.get(speaker_id) or self._get_gender_aware_default_voice(speaker_id)

        if not force_search and speaker_id in self.voice_mapping:
            return self.voice_mapping[speaker_id]

        if not self.voice_matcher.audio_embedder:
            return self._get_gender_aware_default_voice(speaker_id)

        ref_path = Path(reference_audio_path)
        if not ref_path.exists():
            logger.warning(f"Reference audio file not found: {reference_audio_path}")
            return self._get_gender_aware_default_voice(speaker_id)

        # Check if reference audio is long enough for pinning
        min_ref_duration = 3.0
        ref_duration = AudioFileUtils.get_audio_duration_seconds(ref_path)
        can_be_pinned = ref_duration is not None and ref_duration >= min_ref_duration

        if not can_be_pinned:
            logger.debug(f"Reference audio is too short for pinning. Using for current synthesis only.")

        # Determine optimal segment duration based on audio length
        segment_duration_ms = 3000  # Default 3 seconds
        if ref_duration is not None:
            # For longer audio, we can use longer segments for better voice characteristics
            if ref_duration >= 30:
                segment_duration_ms = 5000  # 5 seconds for long audio
            elif ref_duration >= 15:
                segment_duration_ms = 4000  # 4 seconds for medium audio
            # For audio >= 11 seconds, use default 3 seconds

        # Extract multiple embeddings from different parts of the audio
        reference_embeddings = self.voice_matcher.extract_multiple_embeddings(
            ref_path,
            num_segments=3,  # Extract from start, middle, and end
            segment_duration_ms=segment_duration_ms
        )
        
        if not reference_embeddings:
            logger.warning(f"Could not extract embeddings from reference audio.")
            return self._get_gender_aware_default_voice(speaker_id)
        
        logger.debug(f"Extracted {len(reference_embeddings)} embeddings from reference audio (duration: {ref_duration:.1f}s)")

        # Find best matching voice using voting across multiple segments
        # Exclude voices that are already pinned to other speakers to prevent duplicates,
        # plus any voices blocked by the current TTS style policy (e.g. breathy voices on podcast),
        # plus voices whose documented gender conflicts with the speaker metadata.
        exclude_voices = self._build_voice_search_exclusions(speaker_id)
        best_match_voice = self.voice_matcher.find_best_matching_voice_multi_segment(
            reference_embeddings,
            exclude_voices=exclude_voices
        )

        if best_match_voice:
            best_match_voice = self._validate_voice_name(best_match_voice)
            if not is_gemini_voice_gender_compatible(best_match_voice, speaker_gender):
                logger.warning(
                    "Rejecting Gemini voice '%s' for speaker '%s': gender mismatch (%s).",
                    best_match_voice,
                    speaker_id,
                    speaker_gender,
                )
                best_match_voice = None
            
        if best_match_voice:
            if can_be_pinned:
                self.voice_mapping[speaker_id] = best_match_voice
                logger.info(
                    "Matched and pinned by similarity speaker '%s' to voice '%s' (gender=%s)",
                    speaker_id,
                    best_match_voice,
                    speaker_gender,
                )
            
            return best_match_voice

        fallback_voice = self._get_gender_aware_default_voice(speaker_id)
        logger.warning(
            "Could not find matching voice for speaker '%s'. Using gender-aware default '%s' (gender=%s).",
            speaker_id,
            fallback_voice,
            speaker_gender,
        )
        return fallback_voice

    def _get_voice_for_speaker(self, speaker_id: str, segment_hint: TTSSegmentData) -> str:
        """Get voice name for a speaker."""
        return (
            segment_hint.voice or
            self.voice_mapping.get(speaker_id) or
            self._get_gender_aware_default_voice(speaker_id)
        )

    def _resolve_voice_for_segment(self, segment_data: TTSSegmentData) -> str:
        """Determine the validated voice name that will be used for the segment."""
        speaker_id = segment_data.speaker or ""
        raw_voice = (
            segment_data.voice or
            self.voice_mapping.get(speaker_id) or
            self._get_gender_aware_default_voice(speaker_id)
        )
        return self._validate_voice_name(raw_voice)

    def _voice_similarity_gap(self, similarity: Optional[float]) -> float:
        """Return non-negative gap to the configured voice similarity threshold."""
        if similarity is None:
            return 0.0
        return max(0.0, self.config.voice_similarity_threshold - similarity)

    def _validate_voice_consistency(
        self,
        audio_path: str,
        expected_voice_name: str
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Validate synthesized audio against the expected sample voice embedding.

        Returns:
            (is_valid, reason, similarity_score)
        """
        if not self.config.enable_voice_consistency_validation:
            return True, "Voice consistency check disabled", None

        if not self.config.enable_voice_matching:
            return True, "Voice matching disabled", None

        if not self.voice_matcher or not self.voice_matcher.audio_embedder:
            return True, "Voice matcher not available", None

        if expected_voice_name not in self.voice_matcher.sample_embeddings:
            logger.warning(
                f"Voice consistency check skipped: no sample embedding for '{expected_voice_name}'"
            )
            return True, f"No sample embedding for '{expected_voice_name}'", None

        duration_seconds = AudioFileUtils.get_audio_duration_seconds(Path(audio_path))
        if (
            duration_seconds is None
            or duration_seconds < self.config.min_voice_validation_duration_seconds
        ):
            return True, "Voice consistency skipped for short segment", None

        similarity = self.voice_matcher.get_audio_similarity_to_voice(audio_path, expected_voice_name)
        if similarity is None:
            logger.warning(
                f"Voice consistency check skipped: could not compute embedding similarity for '{audio_path}'"
            )
            return True, "Could not compute voice similarity", None

        threshold = max(0.0, min(1.0, self.config.voice_similarity_threshold))
        if similarity < threshold:
            return (
                False,
                f"Voice mismatch ({similarity:.3f} < {threshold:.3f}) for expected voice '{expected_voice_name}'",
                similarity,
            )

        return True, f"Voice match ok ({similarity:.3f} >= {threshold:.3f})", similarity

    def _record_duration_stats(
        self,
        segment: TTSSegmentData,
        synthesized_text: Optional[str],
        duration_seconds: float,
        language: str
    ) -> None:
        """Update duration statistics with real synthesis data."""
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
            speaker_id=segment.speaker
        )

    def save_duration_stats(self) -> None:
        """Persist current duration statistics to disk."""
        self.voice_sample_manager.save_duration_stats()

    def _get_cache_key(self, segment_data: TTSSegmentData, language: str) -> str:
        """Generate a unique cache key for a segment based on its properties."""
        # Include all relevant parameters that affect audio generation
        voice_name = (
            segment_data.voice or
            self.voice_mapping.get(segment_data.speaker) or
            self._get_gender_aware_default_voice(segment_data.speaker)
        )
        style_prompt = segment_data.style_prompt or self.voice_prompt_mapping.get(segment_data.speaker, "")
        
        # NOTE: `output_path` is included to guarantee per-segment uniqueness.
        # Without it, two distinct timeline segments that happen to share the
        # same (text, speaker, voice, emotion, ...) payload (e.g. short
        # fillers like "Да." or "[laughs]") would map to the same cache file.
        # In multi-speaker mode every line gets its own sliced audio, so
        # pooling them by text causes cross-segment audio leakage (segment N
        # ends up playing segment M's slice after an overwrite).
        key_data = {
            "output_path": segment_data.output_path or "",
            "text": segment_data.text,
            "speaker": segment_data.speaker,
            "voice": voice_name,
            "style_prompt": style_prompt,
            "prompt_prefix": self.config.prompt_prefix,
            "emotion": segment_data.emotion or "Neutral",
            "speed": segment_data.speed or 1.0,
            "language": language,
            "model": self.api_client.current_model
        }
        
        # Create hash of the key data
        key_str = str(sorted(key_data.items()))
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _rephrase_for_tts_clarity(
        self,
        original_text: str,
        language: str,
        reason: str = "high silence content"
    ) -> Optional[str]:
        """
        Rephrase text for better TTS synthesis while preserving length and meaning.
        
        Args:
            original_text: Text to rephrase
            language: Target language code
            reason: Reason for rephrasing (for logging)
        
        Returns:
            Rephrased text or None if translator not available
        """
        if not self.translator or not self.translator.is_available():
            logger.warning("Translator not available for rephrasing")
            return None
        
        try:
            logger.info(f"Rephrasing text for TTS clarity (reason: {reason})")
            logger.debug(f"Original text: {original_text}")
            
            rephrased = self.translator.adjust_segment_text_length(
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
                tts_system="gemini",
            )
            
            logger.debug(f"Rephrased text: {rephrased}")
            return rephrased
            
        except Exception as e:
            logger.error(f"Error rephrasing text: {e}")
            return None

    def _synthesize_by_sentence_split(
        self,
        segment_data: TTSSegmentData,
        temp_output_path: str,
        language: str,
        max_retries_per_model: int,
        previous_segments: Optional[List[str]],
        usage_tracker: Optional[Dict[str, Any]],
        attempts_counter: List[int],
        synth_start_ts: float,
        report_segment_index: Optional[int],
        report_group_id: Optional[str],
        primary_model_name: str,
        split_depth: int,
    ) -> Optional[Tuple[Optional[str], Optional[str], bool]]:
        """Synthesize by halving text at a sentence boundary when monolithic synthesis fails."""
        if split_depth >= MAX_SENTENCE_SPLIT_DEPTH:
            return None

        text = (segment_data.text or "").strip()
        if len(text) < MIN_CHARS_FOR_SENTENCE_SPLIT:
            return None

        parts = split_at_sentence_midpoint(text)
        if not parts:
            return None

        first_text, second_text = parts
        logger.warning(
            f"All synthesis attempts failed for {segment_data.speaker}. "
            f"Splitting at sentence boundary into two parts "
            f"({len(first_text)} + {len(second_text)} chars, depth={split_depth + 1})..."
        )

        part_paths: List[str] = []
        model_used: Optional[str] = None
        try:
            for part_index, part_text in enumerate((first_text, second_text)):
                part_path = f"{temp_output_path}_split{split_depth}_part{part_index}.wav"
                part_segment = segment_data.model_copy(update={"text": part_text})
                try:
                    part_text_used, part_model, part_ok = self._synthesize_single_segment(
                        part_segment,
                        part_path,
                        language,
                        max_retries_per_model=max_retries_per_model,
                        previous_segments=previous_segments,
                        usage_tracker=usage_tracker,
                        _split_depth=split_depth + 1,
                    )
                except RuntimeError:
                    logger.warning(
                        f"Sentence-split part {part_index + 1}/2 failed for {segment_data.speaker}."
                    )
                    return None

                if not part_ok or not os.path.exists(part_path):
                    logger.warning(
                        f"Sentence-split part {part_index + 1}/2 did not pass validation for "
                        f"{segment_data.speaker}."
                    )
                    return None

                part_paths.append(part_path)
                if part_model:
                    model_used = part_model

            combined_audio = AudioFileUtils.concatenate_audio_files(part_paths, sample_rate=SAMPLE_RATE)
            AudioFileUtils.save_wave_file(temp_output_path, combined_audio, rate=SAMPLE_RATE)

            self._record_segment_report(SegmentSynthesisReport(
                segment_index=report_segment_index if report_segment_index is not None else -1,
                speaker=segment_data.speaker,
                text=text,
                requested_model=primary_model_name,
                actual_model=model_used or primary_model_name,
                attempts=attempts_counter[0],
                used_fallback=False,
                success=True,
                duration_seconds=time.perf_counter() - synth_start_ts,
                output_path=temp_output_path,
                group_id=report_group_id,
            ))
            return text, model_used or primary_model_name, True
        finally:
            for part_path in part_paths:
                if part_path != temp_output_path and os.path.exists(part_path):
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass

    def _synthesize_single_segment(
        self,
        segment_data: TTSSegmentData,
        temp_output_path: str,
        language: str,
        max_retries_per_model: int = 3,
        previous_segments: Optional[List[str]] = None,
        usage_tracker: Optional[Dict[str, Any]] = None,
        _split_depth: int = 0,
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """Synthesizes a single segment and saves it to a temporary path with validation and retry logic.

        After the primary model exhausts its retry + rephrase budget, the fallback
        model (if configured and different from primary) gets one chance per segment
        before we accept the best invalid attempt. The API client is always reset
        back to the primary model before returning so subsequent segments start fresh.

        Returns:
            Tuple containing the text sent to the model, the model identifier used, and success flag.
            Success flag is True only if validation passed, False if using best attempt.
        """
        if not self.api_client.client:
            raise RuntimeError("Gemini client not initialized.")

        SILENCE_THRESHOLD_FOR_REPHRASING = 0.03
        MAX_REPHRASE_ATTEMPTS = 3
        primary_model_retries = max(max_retries_per_model, self.config.primary_model_max_retries)

        original_text = segment_data.text
        current_text = original_text

        synth_start_ts = time.perf_counter()
        attempts_counter: List[int] = [0]
        report_segment_index = getattr(segment_data, "segment_index", None)
        report_group_id = getattr(segment_data, "group_id", None)
        primary_model_name = self.config.model
        fallback_error: Optional[str] = None

        def _cleanup_paths(paths: List[Optional[str]]) -> None:
            for path in paths:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        def _quality_rank(
            best_silence: float,
            best_similarity: Optional[float],
            best_content_valid: bool,
        ) -> Tuple[float, float, float]:
            content_gap = 0.0 if best_content_valid else 1.0
            return (content_gap, self._voice_similarity_gap(best_similarity), best_silence)

        primary_best: Optional[Dict[str, Any]] = None

        try:
            for rephrase_attempt in range(MAX_REPHRASE_ATTEMPTS):
                segment_data.text = current_text

                cache_key = self._get_cache_key(segment_data, language)
                if cache_key in self._audio_cache:
                    cached_path, _ = self._audio_cache[cache_key]
                    if os.path.exists(cached_path):
                        shutil.copy(cached_path, temp_output_path)
                        logger.debug(f"  Gemini: Using cached audio for speaker {segment_data.speaker}")
                        self._record_segment_report(SegmentSynthesisReport(
                            segment_index=report_segment_index if report_segment_index is not None else -1,
                            speaker=segment_data.speaker,
                            text=current_text,
                            requested_model=primary_model_name,
                            actual_model=primary_model_name,
                            attempts=0,
                            used_fallback=False,
                            success=True,
                            duration_seconds=time.perf_counter() - synth_start_ts,
                            output_path=temp_output_path,
                            group_id=report_group_id,
                        ))
                        return None, None, True

                success, best_silence, best_similarity, best_path, best_text, model_used, best_content_valid = self._attempt_segment_synthesis(
                    segment_data, temp_output_path, language, primary_model_retries, max_silence_ratio=0.2,  # was 0.02 — 10x relaxed
                    previous_segments=previous_segments,
                    usage_tracker=usage_tracker,
                    attempts_counter=attempts_counter,
                )

                if success:
                    if best_path:
                        shutil.move(best_path, temp_output_path)
                    self._record_segment_report(SegmentSynthesisReport(
                        segment_index=report_segment_index if report_segment_index is not None else -1,
                        speaker=segment_data.speaker,
                        text=best_text or current_text,
                        requested_model=primary_model_name,
                        actual_model=model_used or primary_model_name,
                        attempts=attempts_counter[0],
                        used_fallback=bool(model_used and model_used != primary_model_name),
                        success=True,
                        duration_seconds=time.perf_counter() - synth_start_ts,
                        output_path=temp_output_path,
                        group_id=report_group_id,
                    ))
                    return best_text, model_used, True

                if best_path:
                    rank = _quality_rank(best_silence, best_similarity, best_content_valid)
                    if primary_best is None or rank < primary_best["rank"]:
                        if primary_best is not None:
                            _cleanup_paths([primary_best["path"]])
                        primary_best = {
                            "path": best_path,
                            "text": best_text,
                            "model": model_used,
                            "silence": best_silence,
                            "similarity": best_similarity,
                            "content_valid": best_content_valid,
                            "rank": rank,
                        }
                    else:
                        _cleanup_paths([best_path])

                needs_voice_retry = (
                    self.config.enable_voice_consistency_validation
                    and best_similarity is not None
                    and best_similarity < self.config.voice_similarity_threshold
                )
                needs_content_retry = (
                    self.config.enable_content_validation
                    and not best_content_valid
                )

                last_rephrase = rephrase_attempt >= MAX_REPHRASE_ATTEMPTS - 1

                if best_silence > SILENCE_THRESHOLD_FOR_REPHRASING and not last_rephrase:
                    logger.warning(
                        f"Silence ratio {best_silence:.2f} exceeds threshold {SILENCE_THRESHOLD_FOR_REPHRASING}. "
                        f"Attempting rephrasing (attempt {rephrase_attempt + 1}/{MAX_REPHRASE_ATTEMPTS})..."
                    )
                    rephrased_text = self._rephrase_for_tts_clarity(
                        original_text=current_text,
                        language=language,
                        reason=f"silence ratio {best_silence:.2f}"
                    )
                    if rephrased_text and rephrased_text != current_text:
                        current_text = rephrased_text
                        logger.info("Text rephrased successfully. Retrying synthesis...")
                        continue
                    logger.warning("Rephrasing failed or returned same text. Moving on.")
                elif needs_content_retry and not last_rephrase:
                    logger.warning(
                        f"Content validation failed for best attempt. Rephrasing to recover missing words "
                        f"(attempt {rephrase_attempt + 1}/{MAX_REPHRASE_ATTEMPTS})..."
                    )
                    rephrased_text = self._rephrase_for_tts_clarity(
                        original_text=current_text,
                        language=language,
                        reason="TTS output missing expected words (likely truncated)"
                    )
                    if rephrased_text and rephrased_text != current_text:
                        current_text = rephrased_text
                        logger.info("Text rephrased successfully. Retrying synthesis...")
                        continue
                    logger.warning("Rephrasing failed or returned same text. Moving on.")
                elif needs_voice_retry and not last_rephrase:
                    logger.warning(
                        f"Voice similarity {best_similarity:.3f} below threshold {self.config.voice_similarity_threshold:.3f}. "
                        f"Retrying synthesis (attempt {rephrase_attempt + 1}/{MAX_REPHRASE_ATTEMPTS})..."
                    )
                    continue

                # No rephrase will help (or we're out of attempts). Stop the outer loop;
                # fallback model (if any) handles the last-chance retry below.
                break

            fallback_model = self.config.fallback_model
            if fallback_model and fallback_model != self.config.model:
                segment_data.text = current_text
                logger.warning(
                    f"Primary model '{self.config.model}' exhausted for speaker {segment_data.speaker}. "
                    f"Trying fallback model '{fallback_model}'..."
                )
                self.api_client.set_model(fallback_model)
                success, fb_silence, fb_similarity, fb_path, fb_text, fb_model_used, fb_content_valid = self._attempt_segment_synthesis(
                    segment_data, temp_output_path, language,
                    self.config.fallback_model_max_retries, max_silence_ratio=0.2,
                    previous_segments=previous_segments,
                    usage_tracker=usage_tracker,
                    attempts_counter=attempts_counter,
                )

                if success and fb_path:
                    if primary_best is not None:
                        _cleanup_paths([primary_best["path"]])
                    shutil.move(fb_path, temp_output_path)
                    self._record_segment_report(SegmentSynthesisReport(
                        segment_index=report_segment_index if report_segment_index is not None else -1,
                        speaker=segment_data.speaker,
                        text=fb_text or current_text,
                        requested_model=primary_model_name,
                        actual_model=fb_model_used or fallback_model,
                        attempts=attempts_counter[0],
                        used_fallback=True,
                        success=True,
                        duration_seconds=time.perf_counter() - synth_start_ts,
                        output_path=temp_output_path,
                        group_id=report_group_id,
                    ))
                    return fb_text, fb_model_used, True

                if fb_path:
                    fb_rank = _quality_rank(fb_silence, fb_similarity, fb_content_valid)
                    if primary_best is None or fb_rank < primary_best["rank"]:
                        if primary_best is not None:
                            _cleanup_paths([primary_best["path"]])
                        primary_best = {
                            "path": fb_path,
                            "text": fb_text,
                            "model": fb_model_used,
                            "silence": fb_silence,
                            "similarity": fb_similarity,
                            "content_valid": fb_content_valid,
                            "rank": fb_rank,
                        }
                    else:
                        _cleanup_paths([fb_path])

            split_result = self._synthesize_by_sentence_split(
                segment_data=segment_data,
                temp_output_path=temp_output_path,
                language=language,
                max_retries_per_model=max_retries_per_model,
                previous_segments=previous_segments,
                usage_tracker=usage_tracker,
                attempts_counter=attempts_counter,
                synth_start_ts=synth_start_ts,
                report_segment_index=report_segment_index,
                report_group_id=report_group_id,
                primary_model_name=primary_model_name,
                split_depth=_split_depth,
            )
            if split_result is not None:
                if primary_best is not None:
                    _cleanup_paths([primary_best["path"]])
                return split_result

            if primary_best is None:
                self._record_segment_report(SegmentSynthesisReport(
                    segment_index=report_segment_index if report_segment_index is not None else -1,
                    speaker=segment_data.speaker,
                    text=current_text,
                    requested_model=primary_model_name,
                    actual_model=None,
                    attempts=attempts_counter[0],
                    used_fallback=bool(self.config.fallback_model and self.config.fallback_model != primary_model_name),
                    success=False,
                    duration_seconds=time.perf_counter() - synth_start_ts,
                    output_path=None,
                    group_id=report_group_id,
                    error="all attempts failed",
                ))
                raise RuntimeError(
                    f"Failed to synthesize segment for speaker {segment_data.speaker} after all attempts."
                )

            similarity = primary_best["similarity"]
            similarity_repr = "n/a" if similarity is None else f"{similarity:.3f}"
            logger.warning(
                f"Model failed validation. Using best attempt anyway "
                f"(model: {primary_best['model']}, silence: {primary_best['silence']:.2f}, "
                f"voice_similarity: {similarity_repr}, "
                f"content_valid: {primary_best['content_valid']})"
            )
            shutil.move(primary_best["path"], temp_output_path)
            self._record_segment_report(SegmentSynthesisReport(
                segment_index=report_segment_index if report_segment_index is not None else -1,
                speaker=segment_data.speaker,
                text=primary_best["text"] or current_text,
                requested_model=primary_model_name,
                actual_model=primary_best["model"] or primary_model_name,
                attempts=attempts_counter[0],
                used_fallback=bool(primary_best["model"] and primary_best["model"] != primary_model_name),
                success=False,
                duration_seconds=time.perf_counter() - synth_start_ts,
                output_path=temp_output_path,
                group_id=report_group_id,
                error=f"validation failed (silence={primary_best['silence']:.2f}, content_valid={primary_best['content_valid']})",
            ))
            return primary_best["text"], primary_best["model"], False
        finally:
            # Always restore primary model so subsequent segments start fresh.
            if self.api_client.current_model != self.config.model:
                self.api_client.set_model(self.config.model)

    def _attempt_segment_synthesis(self, segment_data: TTSSegmentData, temp_output_path: str,
                                 language: str, max_retries: int, max_silence_ratio: float = 0.2,  # was 0.02 — 10x relaxed
                                 previous_segments: Optional[List[str]] = None,
                                 usage_tracker: Optional[Dict[str, Any]] = None,
                                 attempts_counter: Optional[List[int]] = None) -> Tuple[bool, float, Optional[float], Optional[str], Optional[str], Optional[str], bool]:
        """
        Attempt segment synthesis with the current model.
        Returns a tuple of
        (success, silence_ratio, voice_similarity, best_attempt_path,
         best_attempt_text, model_used, best_content_valid).
        ``best_content_valid`` refers to the best_attempt picked (True when
        content validation was skipped/disabled or passed, False on mismatch).

        If ``attempts_counter`` is provided, each TTS call increments
        ``attempts_counter[0]`` so callers can aggregate per-segment telemetry
        across primary + fallback passes.
        """
        best_attempt_path: Optional[str] = None
        best_attempt_text: Optional[str] = None
        best_model: Optional[str] = None
        best_silence_ratio = float('inf')
        best_voice_similarity: Optional[float] = None
        best_content_valid: bool = True
        best_quality_rank: Tuple[float, float, float] = (float('inf'), float('inf'), float('inf'))
        speaker_id = segment_data.speaker
        text_to_synthesize = segment_data.text

        # Apply emotion enrichment if enabled
        if self.config.enable_emotion_enrichment and self.emotion_enricher:
            logger.debug(f"Emotion enrichment: Enabled for speaker {speaker_id}, enricher exists: {self.emotion_enricher is not None}")
            try:
                original_text = text_to_synthesize
                logger.debug(f"Emotion enrichment: Calling enrich_text with {len(previous_segments) if previous_segments else 0} context segments")
                text_to_synthesize = self.emotion_enricher.enrich_text(
                    text_to_synthesize,
                    previous_segments=previous_segments
                )
                # Only log if text actually changed
                if text_to_synthesize != original_text:
                    logger.info(f"Text was enriched for speaker {speaker_id}")
                    logger.debug(f"  Original: {original_text}")
                    logger.debug(f"  Enriched: {text_to_synthesize}")
                else:
                    logger.debug(f"Emotion enrichment: Text unchanged for speaker {speaker_id}")
            except Exception as e:
                logger.error(f"Emotion enrichment failed with exception: {e}", exc_info=True)
                logger.warning(f"Emotion enrichment failed: {e}. Using original text.")
        else:
            if self.config.enable_emotion_enrichment:
                logger.debug(f"Emotion enrichment: Enabled but enricher is None")
            else:
                logger.debug(f"Emotion enrichment: Disabled in config")

        for attempt in range(max_retries):
            temp_attempt_path = f"{temp_output_path}_attempt_{self.api_client.current_model}_{attempt}.wav"
            if attempts_counter is not None:
                attempts_counter[0] += 1

            try:
                logger.debug(f"  Gemini: Synthesizing segment for {speaker_id} (attempt {attempt + 1}/{max_retries}) with model {self.api_client.current_model}")
                voice_name = self._resolve_voice_for_segment(segment_data)
                current_model = self.api_client.current_model

                final_text = text_to_synthesize
                style_hint = self._get_style_prompt_for_speaker(speaker_id, segment_data)
                prompt_parts: List[str] = []
                if self.config.prompt_prefix:
                    prompt_parts.append(self.config.prompt_prefix)
                if style_hint:
                    prompt_parts.append(style_hint)
                if prompt_parts:
                    final_text = self._build_tts_prompt(prompt_parts, text_to_synthesize)

                logger.debug(f"  TTS Synthesis for [{speaker_id}]:")
                logger.debug(f"    Voice: {voice_name}")
                logger.debug(f"    Prompt prefix: {self.config.prompt_prefix or '(none)'}")
                logger.debug(f"    Speaker style hint: {style_hint or '(none)'}")
                if prompt_parts:
                    logger.debug(f"    Full prompt: {' '.join(prompt_parts)}")

                text_chunks = greedy_sent_split(final_text, MAX_CHAR_LIMIT_PER_REQUEST)
                segment_audio_files = []
                temp_dir_for_chunks = tempfile.mkdtemp(prefix="gemini_chunks_")

                try:
                    speech_config = SpeechConfigBuilder.build_single_speaker_config(voice_name)
                    all_chunks_successful = True
                    for i, chunk_text in enumerate(text_chunks):
                        chunk_file_path = os.path.join(temp_dir_for_chunks, f"chunk_{i}.wav")
                        token_count = self._count_tokens(chunk_text)
                        self._register_usage(usage_tracker, current_model, input_tokens=token_count)
                        audio_data = self.api_client.synthesize_chunk(chunk_text, speech_config)
                        if audio_data:
                            AudioFileUtils.save_wave_file(chunk_file_path, audio_data, rate=SAMPLE_RATE)
                            segment_audio_files.append(chunk_file_path)
                        else:
                            all_chunks_successful = False
                            break
                    
                    if not all_chunks_successful or not segment_audio_files:
                        continue

                    if len(segment_audio_files) > 1:
                        final_audio_data = AudioFileUtils.concatenate_audio_files(segment_audio_files)
                        AudioFileUtils.save_wave_file(temp_attempt_path, final_audio_data, rate=SAMPLE_RATE)
                    elif segment_audio_files:
                        shutil.move(segment_audio_files[0], temp_attempt_path)
                    else:
                        continue # No audio files were generated

                finally:
                    shutil.rmtree(temp_dir_for_chunks)

                if self.api_client.config.enable_audio_validation:
                    fade_config = {
                        'enabled': self.api_client.config.fade_detection_enabled,
                        'window_size_ms': self.api_client.config.fade_window_size_ms,
                        'min_fade_db': self.api_client.config.min_fade_db,
                        'percentile': self.api_client.config.fade_detection_percentile
                    }
                    is_valid, reason, silence_ratio = AudioValidator.validate_audio_sample(
                        temp_attempt_path, 
                        max_silence_ratio=max_silence_ratio,
                        trailing_silence_grace_ratio=self.config.trailing_silence_grace_ratio,
                        fade_detection_config=fade_config,
                        max_total_silence_ratio=self.config.max_total_silence_ratio,
                        max_contiguous_silence_seconds=self.config.max_contiguous_silence_seconds
                    )
                else:
                    # Keep synthesis available even when silence validation is explicitly disabled.
                    is_valid, reason, silence_ratio = True, "Audio validation disabled", 0.0

                voice_valid = True
                voice_reason = "Voice consistency check skipped"
                voice_similarity: Optional[float] = None
                if is_valid:
                    voice_valid, voice_reason, voice_similarity = self._validate_voice_consistency(
                        temp_attempt_path,
                        voice_name
                    )

                content_valid = True
                content_reason = "Content validation skipped"
                content_similarity: Optional[float] = None
                if is_valid and voice_valid:
                    content_valid, content_reason, content_similarity = self._validate_segment_content(
                        temp_attempt_path,
                        text_to_synthesize,
                    )

                # If invalid due to silence and debug saving enabled, persist rejected attempt
                if (not is_valid) and self.debug_save_rejected:
                    try:
                        reason_lower = (reason or "").lower()
                        if ("silence" in reason_lower) or ("no energy" in reason_lower) or ("flat/constant" in reason_lower):
                            base_path_for_debug = Path(segment_data.output_path) if getattr(segment_data, "output_path", None) else Path(temp_attempt_path)
                            debug_dir = base_path_for_debug.parent
                            debug_dir.mkdir(parents=True, exist_ok=True)
                            # Convert silence ratio to integer percent for postfix
                            silence_percent = int(round(max(0.0, min(1.0, silence_ratio)) * 100))
                            debug_name = f"{base_path_for_debug.stem}_attempt{attempt}_{self.api_client.current_model}_silence_{silence_percent}.wav"
                            debug_path = debug_dir / debug_name
                            shutil.copy(temp_attempt_path, debug_path)
                            logger.debug(f"Saved rejected silent attempt to {debug_path}")
                    except Exception as save_exc:
                        logger.warning(f"Could not save rejected silent attempt: {save_exc}")

                if (not voice_valid) and self.debug_save_rejected:
                    try:
                        base_path_for_debug = Path(segment_data.output_path) if getattr(segment_data, "output_path", None) else Path(temp_attempt_path)
                        debug_dir = base_path_for_debug.parent
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        similarity_suffix = "na" if voice_similarity is None else f"{int(round(max(0.0, min(1.0, voice_similarity)) * 100)):02d}"
                        debug_name = f"{base_path_for_debug.stem}_attempt{attempt}_{self.api_client.current_model}_voice_{similarity_suffix}.wav"
                        debug_path = debug_dir / debug_name
                        shutil.copy(temp_attempt_path, debug_path)
                        logger.debug(f"Saved rejected voice-mismatch attempt to {debug_path}")
                    except Exception as save_exc:
                        logger.warning(f"Could not save rejected voice-mismatch attempt: {save_exc}")

                if (not content_valid) and self.debug_save_rejected:
                    try:
                        base_path_for_debug = Path(segment_data.output_path) if getattr(segment_data, "output_path", None) else Path(temp_attempt_path)
                        debug_dir = base_path_for_debug.parent
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        sim_suffix = "na" if content_similarity is None else f"{int(round(max(0.0, min(100.0, content_similarity)))):02d}"
                        debug_name = f"{base_path_for_debug.stem}_attempt{attempt}_{self.api_client.current_model}_content_{sim_suffix}.wav"
                        debug_path = debug_dir / debug_name
                        shutil.copy(temp_attempt_path, debug_path)
                        logger.debug(f"Saved rejected content-mismatch attempt to {debug_path}")
                    except Exception as save_exc:
                        logger.warning(f"Could not save rejected content-mismatch attempt: {save_exc}")

                # Content gap is the primary rank key so content-valid attempts always beat
                # content-invalid ones (a truncated segment with good silence is still worse
                # than a slightly-noisier one that actually contains all the words).
                if content_valid:
                    content_gap = 0.0
                else:
                    content_gap = max(0.01, 1.0 - (content_similarity or 0.0) / 100.0)
                quality_rank = (content_gap, self._voice_similarity_gap(voice_similarity), silence_ratio)
                if quality_rank < best_quality_rank:
                    if best_attempt_path and os.path.exists(best_attempt_path):
                        try:
                            os.remove(best_attempt_path)
                        except OSError as e:
                            logger.warning(f"Could not remove old best_attempt_path: {e}")
                    best_quality_rank = quality_rank
                    best_silence_ratio = silence_ratio
                    best_voice_similarity = voice_similarity
                    best_content_valid = content_valid
                    best_attempt_path = temp_attempt_path
                    best_attempt_text = final_text
                    best_model = current_model
                elif temp_attempt_path != best_attempt_path:
                    try:
                        os.remove(temp_attempt_path)
                    except OSError as e:
                        logger.warning(f"Could not remove temp_attempt_path: {e}")

                if is_valid and voice_valid and content_valid:
                    best_model = current_model
                    return True, silence_ratio, voice_similarity, best_attempt_path, final_text, current_model, True

                failure_reasons: List[str] = []
                if not is_valid:
                    failure_reasons.append(reason)
                if not voice_valid:
                    failure_reasons.append(voice_reason)
                if not content_valid:
                    failure_reasons.append(content_reason)
                logger.debug(
                    f"Segment validation failed for {speaker_id} (attempt {attempt + 1}): "
                    f"{'; '.join([r for r in failure_reasons if r])}"
                )

            except Exception as e:
                logger.error(f"Error synthesizing segment for {speaker_id} (attempt {attempt + 1}): {e}")
                time.sleep(2)

        return False, best_silence_ratio, best_voice_similarity, best_attempt_path, best_attempt_text, best_model, best_content_valid

    # Tags like [excited], [laughs], [sighs] are TTS style cues; they are synthesized
    # as audio behavior but ASR never transcribes them back, so they would otherwise
    # inflate the expected text and wreck fuzzy matching with real transcripts.
    _TTS_MARKUP_RE = re.compile(r"[\[\(][^\]\)]{1,40}[\]\)]")

    @classmethod
    def _strip_tts_markup(cls, text: str) -> str:
        """Remove inline TTS style tags (``[excited]``, ``(laughs)`` ...) and collapse whitespace."""
        if not text:
            return ""
        cleaned = cls._TTS_MARKUP_RE.sub(" ", text)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _normalize_for_asr_match(text: str) -> str:
        """Lower-case and strip non-alphanumeric so fuzzy match ignores punctuation."""
        if not text:
            return ""
        cleaned = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return re.sub(r"\s+", " ", cleaned).strip().lower()

    def _get_content_validator(self) -> Optional[Any]:
        """Return an ASR helper for round-trip content checks (reuse or lazy-init)."""
        if not self.config.enable_content_validation:
            return None
        provider = str(self.config.content_validator_provider or "whisper").lower()
        if provider == "assemblyai" and self.diarizer is not None:
            return self.diarizer
        with self._content_validator_lock:
            if self._content_validator is not None:
                return self._content_validator
            try:
                if provider == "assemblyai":
                    if not DIARIZER_AVAILABLE or SpeakerSegmentDiarizer is None:
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
                logger.warning(f"Content validator init failed: {exc}. Content checks disabled.")
                self.config.enable_content_validation = False
                self._content_validator = None
            return self._content_validator

    def _validate_segment_content(
        self,
        audio_path: str,
        expected_text: str,
    ) -> Tuple[bool, str, Optional[float]]:
        """Round-trip ASR check: did TTS actually produce all expected words.

        Returns ``(is_valid, reason, overall_similarity)``. ``overall_similarity``
        is rapidfuzz ``token_set_ratio`` (0..100) or ``None`` when skipped.

        Specifically catches segments where speech breaks off at the end by
        fuzzy-matching the last N expected words against the full ASR text.
        """
        if not self.config.enable_content_validation:
            return True, "Content validation disabled", None
        if not RAPIDFUZZ_AVAILABLE or _rapidfuzz_fuzz is None:
            return True, "rapidfuzz not installed", None

        cleaned_expected = self._normalize_for_asr_match(self._strip_tts_markup(expected_text))
        if len(cleaned_expected) < max(1, int(self.config.content_validation_min_expected_chars)):
            return True, "Expected text too short for ASR check", None

        validator = self._get_content_validator()
        if validator is None:
            return True, "Content validator unavailable", None

        validation_audio_seconds = AudioFileUtils.get_audio_duration_seconds(Path(audio_path)) or 0.0
        if self.cost_tracker and validation_audio_seconds > 0:
            provider = str(self.config.content_validator_provider or "whisper").lower()
            model_name = (
                self.config.content_validator_speech_model
                if provider == "assemblyai"
                else self.config.content_validator_whisper_model
            )
            self.cost_tracker.add_transcription_usage(
                provider,
                float(validation_audio_seconds),
                model=str(model_name),
                category="tts_content_validation",
            )

        try:
            asr_text = validator.transcribe_text(audio_path)
        except Exception as exc:
            logger.warning(f"Content validation ASR failed: {exc}")
            return True, f"ASR error: {exc}", None

        asr_normalized = self._normalize_for_asr_match(asr_text)
        if not asr_normalized:
            return False, "ASR produced empty text (speech not detected)", 0.0

        overall = float(_rapidfuzz_fuzz.token_set_ratio(cleaned_expected, asr_normalized))

        # Trailing-tail check catches cutoffs where the last few words are missing.
        expected_words = cleaned_expected.split()
        tail_count = max(1, int(self.config.content_trailing_word_count))
        tail_text = " ".join(expected_words[-tail_count:])
        tail_score = float(_rapidfuzz_fuzz.partial_ratio(tail_text, asr_normalized))

        if tail_score < float(self.config.content_trailing_fuzzy_threshold):
            return False, (
                f"Trailing words missing (tail='{tail_text}', "
                f"tail_score={tail_score:.1f} < {self.config.content_trailing_fuzzy_threshold:.1f}, "
                f"overall={overall:.1f})"
            ), overall

        if overall < float(self.config.content_similarity_threshold):
            return False, (
                f"Content similarity {overall:.1f} below threshold "
                f"{self.config.content_similarity_threshold:.1f}"
            ), overall

        return True, f"Content match (overall={overall:.1f}, tail={tail_score:.1f})", overall

    @staticmethod
    def _group_segments_by_matched_line(segments: List[SpeakerSegment]) -> Dict[int, List[SpeakerSegment]]:
        grouped: Dict[int, List[SpeakerSegment]] = {}
        for seg in segments:
            if seg.matched_line_idx is None:
                continue
            grouped.setdefault(int(seg.matched_line_idx), []).append(seg)
        for idx in grouped:
            grouped[idx].sort(key=lambda s: s.start_ms)
        return grouped

    @staticmethod
    def _find_temporal_conflicts(
        best_per_line: Dict[int, SpeakerSegment],
        overlap_tolerance_ms: int = 120,
    ) -> Dict[int, str]:
        return find_temporal_conflicts(best_per_line, overlap_tolerance_ms)

    def _build_multi_speaker_prompt(
        self,
        batch: List[TTSSegmentData],
        speakers: List[str],
    ) -> str:
        """Build the multi-speaker transcript prompt (Speaker: line per turn)."""
        preamble_parts: List[str] = []
        if self.config.prompt_prefix:
            preamble_parts.append(self.config.prompt_prefix)

        if len(speakers) == 2:
            preamble_parts.append(f"TTS the following conversation between {speakers[0]} and {speakers[1]}:")
        else:
            preamble_parts.append("TTS the following conversation:")

        style_lines: List[str] = []
        for spk in speakers:
            hint = (self.voice_prompt_mapping.get(spk) or "").strip()
            if hint:
                style_lines.append(f"Make {spk} sound like this: {hint}")
        if style_lines:
            preamble_parts.append("\n".join(style_lines))

        transcript_lines = [f"{seg.speaker}: {seg.text.strip()}" for seg in batch]
        return self._build_tts_prompt(preamble_parts, "\n".join(transcript_lines))

    def _synthesize_multi_speaker_batch(
        self,
        batch: List[TTSSegmentData],
        batch_indices: List[int],
        language: str,
        usage_tracker: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[int, SegmentAlignment], List[int], int]:
        """Synthesize a 2-speaker batch, diarize, slice, validate.

        Returns ``(alignments_by_global_index, invalid_global_indices, attempts_used)``
        with a best-effort partial result. The caller is expected to
        re-synthesize the invalid indices via single-speaker synthesis. Raises
        only when no attempt produced a usable API response at all (e.g. empty
        audio, diarizer crash).
        """
        if not DIARIZER_AVAILABLE or not self.diarizer or DialogueLine is None or SpeakerSegment is None:
            raise RuntimeError("SpeakerSegmentDiarizer not initialized.")
        if not batch:
            return {}, [], 0

        speakers = sorted({str(s.speaker) for s in batch if s.speaker})
        if len(speakers) != 2:
            raise ValueError("Multi-speaker batch must contain exactly 2 unique speakers")

        speaker_to_voice = {
            spk: self._validate_voice_name(
                self.voice_mapping.get(spk) or self._get_gender_aware_default_voice(spk)
            )
            for spk in speakers
        }
        speech_config = SpeechConfigBuilder.build_multi_speaker_config(speaker_to_voice)
        expected_lines = [
            DialogueLine(
                speaker=str(seg.speaker),
                text=self._strip_tts_markup(str(seg.text)),
            )
            for seg in batch
        ]

        best_alignments: Dict[int, SegmentAlignment] = {}
        best_invalid_global: Optional[List[int]] = None
        last_err: Optional[Exception] = None
        attempts_used = 0
        max_attempts = max(1, int(self.config.multi_speaker_retries) + 1)
        for attempt in range(max_attempts):
            attempts_used = attempt + 1
            tmp_dir = tempfile.mkdtemp(prefix="gemini_multi_speaker_")
            batch_wav_path = os.path.join(tmp_dir, f"batch_attempt_{attempt}.wav")
            try:
                prompt = self._build_multi_speaker_prompt(batch, speakers)
                token_count = 0
                try:
                    token_count = self._count_tokens(prompt)
                except Exception:
                    token_count = int(math.ceil(len(prompt) / 4.0))
                self._register_usage(usage_tracker, self.api_client.current_model, input_tokens=token_count)

                audio_data = self.api_client.synthesize_chunk(prompt, speech_config)
                if not audio_data:
                    raise RuntimeError("Gemini multi-speaker returned empty audio bytes")
                AudioFileUtils.save_wave_file(batch_wav_path, audio_data, rate=SAMPLE_RATE)

                batch_duration = AudioFileUtils.get_audio_duration_seconds(Path(batch_wav_path)) or 0.0
                if batch_duration:
                    self._register_usage(usage_tracker, self.api_client.current_model, audio_seconds=batch_duration)
                    if self.cost_tracker:
                        diarizer_model = getattr(self.diarizer, "_speech_model", "best") or "best"
                        self.cost_tracker.add_transcription_usage(
                            "assemblyai",
                            float(batch_duration),
                            model=str(diarizer_model),
                            category="tts_batch_alignment",
                        )

                diar = self.diarizer.align_by_text(batch_wav_path, expected_lines)
                grouped = self._group_segments_by_matched_line(diar.segments)

                # Pre-select a single best-scoring text span per expected line.
                # The alignment itself is text-only; speaker identity comes from
                # the known batch line, not from ASR speaker labels.
                best_per_line: Dict[int, SpeakerSegment] = {}
                for local_i in range(len(batch)):
                    group = grouped.get(local_i)
                    if not group:
                        continue
                    if len(group) > 1:
                        logger.debug(
                            f"Gemini MS slice #{batch_indices[local_i] + 1} "
                            f"({batch[local_i].speaker}): text aligner returned {len(group)} ASR "
                            f"span candidates for this line; keeping best-scoring one "
                            f"(scores: {[round(float(g.fuzzy_score), 1) for g in group]})"
                        )
                    best_per_line[local_i] = max(
                        group, key=lambda s: (float(s.fuzzy_score), float(s.confidence))
                    )

                utt_owner: Dict[tuple, int] = {}
                colliding: set = set()
                for local_i, best in best_per_line.items():
                    key = (int(best.start_ms), int(best.end_ms), str(best.asr_speaker_label))
                    if key in utt_owner:
                        colliding.add(utt_owner[key])
                        colliding.add(local_i)
                    else:
                        utt_owner[key] = local_i
                temporal_conflicts = self._find_temporal_conflicts(best_per_line)

                candidate_alignments: Dict[int, SegmentAlignment] = {}
                candidate_slice_paths: Dict[int, str] = {}
                candidate_durations: Dict[int, float] = {}
                invalid_local_indices: List[int] = []
                for local_i, seg in enumerate(batch):
                    global_i = batch_indices[local_i]
                    if local_i not in best_per_line:
                        logger.info(
                            f"Gemini MS slice #{global_i + 1} ({seg.speaker}): text aligner found "
                            f"no matching ASR span for this line (attempt {attempt + 1}/{max_attempts})"
                        )
                        invalid_local_indices.append(local_i)
                        continue
                    if local_i in colliding:
                        logger.info(
                            f"Gemini MS slice #{global_i + 1} ({seg.speaker}): text aligner assigned "
                            f"the same ASR span to multiple lines in this batch — "
                            f"marking invalid to avoid duplicate audio on the timeline "
                            f"(attempt {attempt + 1}/{max_attempts})"
                        )
                        invalid_local_indices.append(local_i)
                        continue
                    if local_i in temporal_conflicts:
                        logger.info(
                            f"Gemini MS slice #{global_i + 1} ({seg.speaker}): "
                            f"{temporal_conflicts[local_i]} — marking invalid to avoid "
                            f"overlapping or repeated audio on the timeline "
                            f"(attempt {attempt + 1}/{max_attempts})"
                        )
                        invalid_local_indices.append(local_i)
                        continue
                    best = best_per_line[local_i]
                    start_ms = int(best.start_ms)
                    end_ms = int(best.end_ms)

                    wav_bytes = self.diarizer.slice_audio(batch_wav_path, [SpeakerSegment(
                        index=local_i,
                        speaker=str(seg.speaker),
                        start_ms=start_ms,
                        end_ms=end_ms,
                        asr_text=str(best.asr_text or "").strip(),
                        matched_line_idx=local_i,
                        matched_text=str(seg.text).strip(),
                        fuzzy_score=float(best.fuzzy_score),
                        asr_speaker_label=str(best.asr_speaker_label),
                        confidence=float(best.confidence),
                    )])[0]

                    slice_path = os.path.join(tmp_dir, f"slice_{local_i}_{seg.speaker}.wav")
                    with open(slice_path, "wb") as f:
                        f.write(wav_bytes)

                    fade_config = {
                        'enabled': self.api_client.config.fade_detection_enabled,
                        'window_size_ms': self.api_client.config.fade_window_size_ms,
                        'min_fade_db': self.api_client.config.min_fade_db,
                        'percentile': self.api_client.config.fade_detection_percentile
                    }
                    is_valid, audio_reason, _silence_ratio = AudioValidator.validate_audio_sample(
                        slice_path,
                        expected_min_duration=0.2,  # text aligner already matched content; keep a tiny floor only
                        max_silence_ratio=0.5,  # was 0.05 — temporarily 10x to stop MS slice false-positives
                        trailing_silence_grace_ratio=self.config.trailing_silence_grace_ratio,
                        fade_detection_config=fade_config,
                        max_total_silence_ratio=self.config.max_total_silence_ratio,
                        max_contiguous_silence_seconds=self.config.max_contiguous_silence_seconds
                    )
                    voice_name = self._resolve_voice_for_segment(seg)
                    voice_valid, voice_reason, sim = (True, "", None)
                    if is_valid:
                        voice_valid, voice_reason, sim = self._validate_voice_consistency(slice_path, voice_name)

                    if not is_valid:
                        logger.info(
                            f"Gemini MS slice #{global_i + 1} ({seg.speaker}, voice={voice_name}): "
                            f"audio validation failed — {audio_reason} "
                            f"(attempt {attempt + 1}/{max_attempts})"
                        )
                        invalid_local_indices.append(local_i)
                        continue
                    if not voice_valid:
                        sim_str = f"{sim:.3f}" if sim is not None else "n/a"
                        logger.info(
                            f"Gemini MS slice #{global_i + 1} ({seg.speaker}, voice={voice_name}): "
                            f"voice mismatch — {voice_reason} (similarity={sim_str}) "
                            f"(attempt {attempt + 1}/{max_attempts})"
                        )
                        invalid_local_indices.append(local_i)
                        continue

                    duration = AudioFileUtils.get_audio_duration_seconds(Path(slice_path)) or 0.0
                    diarized_segment = DiarizationSegment(
                        start_time=float(start_ms) / 1000.0,
                        end_time=float(end_ms) / 1000.0,
                        speaker=str(seg.speaker),
                        text=str(seg.text),
                        confidence=1.0,
                    )
                    candidate_alignments[local_i] = SegmentAlignment(
                        original_segment=seg,
                        diarized_segment=diarized_segment,
                        alignment_confidence=1.0,
                    )
                    candidate_slice_paths[local_i] = slice_path
                    candidate_durations[local_i] = duration

                alignments: Dict[int, SegmentAlignment] = {
                    batch_indices[local_i]: alignment
                    for local_i, alignment in candidate_alignments.items()
                    if local_i not in invalid_local_indices
                }
                for local_i in sorted(candidate_alignments):
                    if local_i in invalid_local_indices:
                        continue
                    seg = batch[local_i]
                    global_i = batch_indices[local_i]
                    slice_path = candidate_slice_paths[local_i]
                    duration = candidate_durations.get(local_i, 0.0)
                    if seg.output_path:
                        os.makedirs(os.path.dirname(seg.output_path), exist_ok=True)
                        shutil.copy(slice_path, seg.output_path)

                    self._record_segment_report(SegmentSynthesisReport(
                        segment_index=(
                            getattr(seg, "segment_index", None)
                            if getattr(seg, "segment_index", None) is not None
                            else global_i
                        ),
                        speaker=seg.speaker,
                        text=seg.text,
                        requested_model=self.config.model,
                        actual_model=self.config.model,
                        attempts=attempt + 1,
                        used_fallback=False,
                        success=True,
                        duration_seconds=duration,
                        output_path=seg.output_path or slice_path,
                        group_id=getattr(seg, "group_id", None),
                    ))

                    # Populate wrapper-level audio cache only after all batch
                    # validations have passed; otherwise a contaminated slice can
                    # be reused by the single-speaker fallback in the same run.
                    if self._cache_dir:
                        try:
                            cache_key = self._get_cache_key(seg, language)
                            cache_path = os.path.join(self._cache_dir, f"{cache_key}.wav")
                            shutil.copy(slice_path, cache_path)
                            self._audio_cache[cache_key] = (cache_path, duration)
                        except Exception as cache_exc:
                            logger.debug(f"Failed to cache multi-speaker slice: {cache_exc}")

                    if duration:
                        self._record_duration_stats(seg, None, duration, language)

                invalid_global = [batch_indices[i] for i in invalid_local_indices]
                if not invalid_global:
                    return alignments, [], attempts_used

                if best_invalid_global is None or len(invalid_global) < len(best_invalid_global):
                    best_alignments = alignments
                    best_invalid_global = invalid_global

                last_err = RuntimeError(
                    f"Multi-speaker validation failed for {len(invalid_local_indices)}/{len(batch)} "
                    f"slice(s) (local indices: {invalid_local_indices})"
                )

                # Early bail-out: if after a couple of attempts more than half of the
                # batch is still unmatched, further retries rarely recover anything
                # (Gemini keeps merging/skipping turns). Cut losses and send the
                # invalid slices to the single-speaker fallback immediately.
                invalid_ratio = len(invalid_local_indices) / max(1, len(batch))
                if attempt + 1 >= 2 and invalid_ratio > 0.5 and attempt + 1 < max_attempts:
                    idx_range = (
                        f"#{batch_indices[0] + 1}..{batch_indices[-1] + 1}"
                        if batch_indices else "#?"
                    )
                    logger.warning(
                        f"Gemini batch {idx_range}: invalid ratio {invalid_ratio * 100:.0f}% "
                        f"after {attempt + 1}/{max_attempts} attempt(s) — early fallback"
                    )
                    break
            except Exception as exc:
                last_err = exc
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if best_invalid_global is not None:
            idx_range = (
                f"#{batch_indices[0] + 1}..{batch_indices[-1] + 1}"
                if batch_indices else "#?"
            )
            logger.warning(
                f"Gemini batch {idx_range}: multi-speaker after {attempts_used} attempt(s) "
                f"— kept {len(best_alignments)} valid slice(s), {len(best_invalid_global)} "
                f"slice(s) need single-speaker fallback "
                f"(global indices: {[g + 1 for g in best_invalid_global]})"
            )
            return best_alignments, best_invalid_global, attempts_used

        raise RuntimeError(f"Multi-speaker batch failed after retries: {last_err}")

    def _process_single_segment(
        self,
        segment: TTSSegmentData,
        segment_index: int,
        total_segments: int,
        temp_dir: str,
        language: str,
        context_segments: List[str],
        usage_tracker: Optional[Dict[str, Any]] = None
    ) -> Optional[SegmentAlignment]:
        """
        Process a single segment: check cache or synthesize.

        Args:
            segment: Segment data to process
            segment_index: Index of this segment
            total_segments: Total number of segments
            temp_dir: Temporary directory for audio files
            language: Target language code
            context_segments: List of previous segment texts for emotion enrichment context
            usage_tracker: Shared usage tracker for cost accounting

        Returns:
            SegmentAlignment if successful, None otherwise
        """
        segment_file_path = os.path.join(temp_dir, f"segment_{segment_index}_{segment.speaker}.wav")

        report_segment_index = (
            getattr(segment, "segment_index", None)
            if getattr(segment, "segment_index", None) is not None
            else segment_index
        )
        report_group_id = getattr(segment, "group_id", None)

        # Check if we already have this segment in cache
        cache_key = self._get_cache_key(segment, language)
        if cache_key in self._audio_cache:
            cached_path, duration = self._audio_cache[cache_key]
            if os.path.exists(cached_path):
                logger.debug(f"Gemini: Using cached segment {segment_index+1}/{total_segments} for speaker '{segment.speaker}'")
                shutil.copy(cached_path, segment_file_path)

                # Save to output path if specified
                if segment.output_path:
                    os.makedirs(os.path.dirname(segment.output_path), exist_ok=True)
                    shutil.copy(segment_file_path, segment.output_path)
                    logger.debug(f"Saved segment audio to {segment.output_path}")

                self._record_segment_report(SegmentSynthesisReport(
                    segment_index=report_segment_index,
                    speaker=segment.speaker,
                    text=segment.text,
                    requested_model=self.config.model,
                    actual_model=self.config.model,
                    attempts=0,
                    used_fallback=False,
                    success=True,
                    duration_seconds=0.0,
                    output_path=segment.output_path or segment_file_path,
                    group_id=report_group_id,
                ))

                # Create alignment using cached duration
                diarized = DiarizationSegment(
                    start_time=0.0,
                    end_time=duration,
                    speaker=segment.speaker,
                    text=segment.text,
                    confidence=1.0
                )
                return SegmentAlignment(
                    original_segment=segment,
                    diarized_segment=diarized,
                    alignment_confidence=1.0
                )

        # If not cached, synthesize normally
        logger.debug(f"Gemini: Synthesizing segment {segment_index+1}/{total_segments} for speaker '{segment.speaker}'")
        try:
            synthesized_text, model_used, is_valid = self._synthesize_single_segment(
                segment, segment_file_path, language,
                previous_segments=context_segments,
                usage_tracker=usage_tracker
            )

            # Get duration of the synthesized audio
            duration = AudioFileUtils.get_audio_duration_seconds(segment_file_path) or 0.0
            if is_valid:
                self._record_duration_stats(segment, synthesized_text, duration, language)
            else:
                logger.debug(
                    f"Skipping duration stats update for invalid segment "
                    f"{segment_index+1}/{total_segments} (speaker='{segment.speaker}')"
                )
            if duration and model_used:
                self._register_usage(usage_tracker, model_used, audio_seconds=duration)

            # Cache the generated audio only if synthesis was successful (passed validation)
            # and was produced by the primary model (not fallback/rescue).
            # Fallback results are intentionally not cached so they can be
            # regenerated with the primary model on the next run.
            used_fallback = (
                model_used is not None
                and model_used != self.config.model
            )
            if self._cache_dir and is_valid and not used_fallback:
                cache_path = os.path.join(self._cache_dir, f"{cache_key}.wav")
                shutil.copy(segment_file_path, cache_path)
                self._audio_cache[cache_key] = (cache_path, duration)
                logger.debug(f"Cached valid segment for speaker '{segment.speaker}'")
            elif used_fallback:
                logger.debug(
                    f"Skipping cache for fallback segment (speaker '{segment.speaker}', "
                    f"model '{model_used}') to allow primary-model retry next time"
                )
            elif not is_valid:
                logger.debug(f"Skipping cache for invalid segment (speaker '{segment.speaker}') to allow retry next time")

            # Track fallback output paths so the pipeline cache can skip them too
            if used_fallback and segment.output_path:
                with self._fallback_paths_lock:
                    self._fallback_output_paths.add(os.path.abspath(segment.output_path))

            # Save to output path if specified
            if segment.output_path:
                os.makedirs(os.path.dirname(segment.output_path), exist_ok=True)
                shutil.copy(segment_file_path, segment.output_path)
                logger.debug(f"Saved segment audio to {segment.output_path}")

            # Create alignment
            diarized = DiarizationSegment(
                start_time=0.0,
                end_time=duration,
                speaker=segment.speaker,
                text=segment.text,
                confidence=1.0
            )
            return SegmentAlignment(
                original_segment=segment,
                diarized_segment=diarized,
                alignment_confidence=1.0
            )

        except Exception as e_segment:
            logger.error(f"Error synthesizing segment {segment_index+1} for speaker '{segment.speaker}': {e_segment}")
            self._record_segment_report(SegmentSynthesisReport(
                segment_index=report_segment_index,
                speaker=segment.speaker,
                text=segment.text,
                requested_model=self.config.model,
                actual_model=None,
                attempts=0,
                used_fallback=False,
                success=False,
                duration_seconds=0.0,
                output_path=None,
                group_id=report_group_id,
                error=str(e_segment),
            ))
            return None

    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: Optional[str] = None,
        **kwargs: Any
    ) -> List[SegmentAlignment]:
        """
        Synthesize speech for each segment individually using parallel processing.

        Args:
            segments_data: List of segments to synthesize
            language: Target language code (uses target_language from init if not specified)
            **kwargs: Additional synthesis parameters
                - previous_context: List[str] - Previous segment texts for emotion enrichment
                - progress_callback: Optional callable(current, total, text=None) invoked as
                  batches complete. current/total reflect synthesized segments, not batches.

        Returns:
            List of segment alignments (in original order)
        """
        progress_callback = kwargs.get("progress_callback")
        if not self.is_available():
            raise RuntimeError("Gemini TTS not initialized.")
        
        language = language or self.target_language

        # Validate input
        if not segments_data:
            logger.warning("Warning: No segments provided.")
            return []

        valid_segments = [seg for seg in segments_data if seg.speaker and seg.text]
        if not valid_segments:
            logger.warning("Warning: No valid segments found.")
            return []

        # Auto-pin voices for unmapped speakers with reference audio
        if self.config.enable_voice_matching and self.voice_matcher.audio_embedder:
            speaker_to_ref_path = {}
            for segment in valid_segments:
                if (segment.speaker and segment.speaker not in self.voice_mapping and
                    segment.speaker not in speaker_to_ref_path and segment.reference_audio_path):
                    speaker_to_ref_path[segment.speaker] = segment.reference_audio_path

            for speaker_id, ref_path in speaker_to_ref_path.items():
                logger.debug(f"Auto-pinning voice for speaker '{speaker_id}'")
                self.find_and_pin_voice_for_speaker(speaker_id, ref_path)

        temp_dir = tempfile.mkdtemp(prefix="gemini_segments_")
        usage_tracker: Dict[str, Any] = {"models": {}, "lock": threading.Lock()}

        # Preserve original position within valid_segments
        seg_to_index: Dict[int, int] = {id(seg): i for i, seg in enumerate(valid_segments)}

        batches: List[List[TTSSegmentData]] = []
        multi_speaker_mode = self.config.enable_multi_speaker and self.diarizer is not None
        if multi_speaker_mode:
            batches = self._build_multi_speaker_batches(valid_segments, language)
        else:
            batches = [[seg] for seg in valid_segments]

        multi_batches = [b for b in batches if len({str(s.speaker) for s in b if s.speaker}) >= 2]
        single_batches_count = len(batches) - len(multi_batches)
        if multi_speaker_mode and multi_batches:
            logger.info(
                f"Gemini synthesize: {len(valid_segments)} segment(s) grouped into {len(batches)} batch(es) — "
                f"{len(multi_batches)} multi-speaker, {single_batches_count} single-speaker"
            )
        else:
            logger.info(
                f"Gemini synthesize: {len(valid_segments)} segment(s) — "
                f"{'multi-speaker disabled, ' if not multi_speaker_mode else ''}running as single-segment synthesis"
            )

        results: Dict[int, Optional[SegmentAlignment]] = {}
        results_lock = threading.Lock()

        def process_batch(batch_idx: int, batch: List[TTSSegmentData]) -> None:
            batch_indices = [seg_to_index[id(seg)] for seg in batch]
            unique_speakers = sorted({str(s.speaker) for s in batch if s.speaker})
            if batch_indices:
                idx_range = (
                    f"#{batch_indices[0] + 1}" if len(batch_indices) == 1
                    else f"#{min(batch_indices) + 1}..{max(batch_indices) + 1}"
                )
            else:
                idx_range = "#-"

            batch_started = time.perf_counter()
            batch_mode = "single_segment" if len(batch) <= 1 else "single_speaker"
            batch_attempts = 0
            batch_success = True
            batch_fallback_count = 0
            batch_reason = ""
            max_ms_attempts = max(1, int(self.config.multi_speaker_retries) + 1)

            invalid_after_ms: Optional[List[int]] = None
            if multi_speaker_mode and len(unique_speakers) == 2:
                batch_mode = "multi_speaker"
                logger.info(
                    f"Gemini batch {idx_range}: multi-speaker synth "
                    f"({len(batch)} lines, speakers: {' + '.join(unique_speakers)})"
                )
                try:
                    batch_alignments, invalid_global, ms_attempts = self._synthesize_multi_speaker_batch(
                        batch=batch,
                        batch_indices=batch_indices,
                        language=language,
                        usage_tracker=usage_tracker,
                    )
                    batch_attempts = ms_attempts
                    with results_lock:
                        for idx, alignment in batch_alignments.items():
                            results[idx] = alignment
                    if not invalid_global:
                        batch_reason = (
                            f"multi-speaker ok in {ms_attempts}/{max_ms_attempts} attempt(s)"
                        )
                        self._record_batch_report(BatchSynthesisReport(
                            batch_index=batch_idx,
                            mode=batch_mode,
                            segment_indices=list(batch_indices),
                            speakers=list(unique_speakers),
                            attempts=batch_attempts,
                            success=True,
                            fallback_segment_count=0,
                            duration_seconds=time.perf_counter() - batch_started,
                            reason=batch_reason,
                        ))
                        return
                    invalid_after_ms = invalid_global
                    batch_success = False
                    batch_fallback_count = len(invalid_global)
                    batch_reason = (
                        f"partial multi-speaker: {len(invalid_global)}/{len(batch)} "
                        f"slice(s) fell back to single-speaker after "
                        f"{ms_attempts}/{max_ms_attempts} attempt(s)"
                    )
                    logger.info(
                        f"Gemini batch {idx_range}: re-synth {len(invalid_global)} slice(s) "
                        f"via single-speaker "
                        f"(global indices: {[g + 1 for g in invalid_global]})"
                    )
                except Exception as exc:
                    batch_attempts = max_ms_attempts
                    batch_success = False
                    batch_fallback_count = len(batch)
                    batch_reason = (
                        f"multi-speaker raised {type(exc).__name__}: {exc}"
                    )[:240]
                    logger.warning(
                        f"Gemini batch {idx_range}: multi-speaker failed, falling back to single-speaker "
                        f"synthesis ({len(batch)} segment(s)). Error: {exc}"
                    )
            elif len(batch) > 1:
                batch_attempts = 1
                batch_reason = (
                    f"single-speaker run of {len(batch)} segment(s) "
                    f"(speaker: {unique_speakers[0] if unique_speakers else '?'})"
                )
                logger.info(
                    f"Gemini batch {idx_range}: single-speaker run "
                    f"({len(batch)} lines, speaker: {unique_speakers[0] if unique_speakers else '?'})"
                )
            else:
                batch_attempts = 1
                batch_reason = "single-segment synthesis"

            for idx, seg in zip(batch_indices, batch):
                if invalid_after_ms is not None and idx not in invalid_after_ms:
                    continue
                alignment = self._process_single_segment(
                    segment=seg,
                    segment_index=idx,
                    total_segments=len(valid_segments),
                    temp_dir=temp_dir,
                    language=language,
                    context_segments=[],
                    usage_tracker=usage_tracker,
                )
                with results_lock:
                    results[idx] = alignment

            self._record_batch_report(BatchSynthesisReport(
                batch_index=batch_idx,
                mode=batch_mode,
                segment_indices=list(batch_indices),
                speakers=list(unique_speakers),
                attempts=batch_attempts,
                success=batch_success,
                fallback_segment_count=batch_fallback_count,
                duration_seconds=time.perf_counter() - batch_started,
                reason=batch_reason,
            ))

        total_segments = len(valid_segments)
        completed_segments = 0
        progress_lock = threading.Lock()

        if progress_callback:
            try:
                progress_callback(0, total_segments, None)
            except Exception as cb_exc:
                logger.debug(f"Progress callback raised at start: {cb_exc}")

        try:
            max_workers = min(self.config.max_workers, max(1, len(batches)))
            logger.debug(f"Gemini: Starting batch synthesis with {max_workers} workers for {len(batches)} batches")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {
                    executor.submit(process_batch, i, b): b
                    for i, b in enumerate(batches)
                }
                for future in as_completed(future_to_batch):
                    future.result()
                    if progress_callback:
                        batch_segments = future_to_batch[future]
                        segments_in_batch = len(batch_segments)
                        preview_text: Optional[str] = None
                        for seg in batch_segments:
                            if seg.text:
                                preview_text = seg.text
                                break
                        with progress_lock:
                            completed_segments += segments_in_batch
                            current = min(completed_segments, total_segments)
                        try:
                            progress_callback(current, total_segments, preview_text)
                        except Exception as cb_exc:
                            logger.debug(f"Progress callback raised: {cb_exc}")

            alignments = [results.get(i) for i in range(len(valid_segments)) if results.get(i) is not None]

            if self.cost_tracker and usage_tracker:
                lock = usage_tracker.get("lock")
                if lock:
                    with lock:
                        models_usage = dict(usage_tracker.get("models", {}))
                else:
                    models_usage = dict(usage_tracker.get("models", {}))
                for model_name, metrics in models_usage.items():
                    input_tok = metrics.get("input_tokens", 0.0)
                    output_tok = metrics.get("output_tokens", 0.0)
                    audio_sec = metrics.get("audio_seconds", 0.0)
                    if input_tok or output_tok or audio_sec:
                        self.cost_tracker.add_tts_actual(
                            "gemini",
                            model=model_name,
                            input_tokens=input_tok,
                            output_tokens=output_tok,
                            audio_seconds=audio_sec,
                        )

            return alignments
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def is_available(self) -> bool:
        """Check if Gemini TTS is available and initialized."""
        return GEMINI_AVAILABLE and self.api_client.client is not None
    
    def clone_voice(self, audio_path: str, voice_id: str) -> str:
        """Voice cloning is not supported by Gemini TTS API."""
        raise NotImplementedError("Voice cloning is not supported by Gemini TTS. Use Minimax TTS for voice cloning capabilities.")

    def cleanup(self) -> None:
        """Clean up resources."""
        # Persist any pending duration statistics before tearing down
        try:
            self.save_duration_stats()
        except Exception as e:
            logger.error(f"Failed to save duration stats during cleanup: {e}")

        # Clean up cache directory
        if self._cache_dir and os.path.exists(self._cache_dir):
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir = None
        self._audio_cache.clear()
        
        # Clear voice matcher
        if self.voice_matcher:
            self.voice_matcher.clear()
            self.voice_matcher = None
