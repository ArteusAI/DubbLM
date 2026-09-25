from typing import Optional, Dict, Any, List, Union, Sequence
import os
import time
import tempfile
import shutil
import hashlib
from pathlib import Path

from .models import TTSSegmentData, SegmentAlignment, DiarizationSegment, SegmentSynthesisReport
from .voice_sample_manager import VoiceSampleManager, AudioValidator
from .openrouter_voice_catalog import (
    ALL_OPENROUTER_VOICES,
    OPENROUTER_VOICE_META,
    default_voice_for,
    is_qwen_tts_model,
    max_char_limit_for_model,
    voices_for_model,
)
from .qwen_emotion import (
    QwenEmotionEnricher,
    apply_nl_instruction_prefix,
    nl_instruction_for_emotion,
    prepare_qwen_speech_text,
    sanitize_qwen_markup,
    spoken_tag_leak_hits,
    strip_markup_tags,
    style_prompt_to_instruction,
)
from src.tts.tts_interface import TTSInterface
from src.utils.sent_split import greedy_sent_split
from src.utils.audio_embedder import AudioEmbedder
from src.utils.voice_matcher import VoiceMatcher
from src.dubbing.core.log_config import get_logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logger = get_logger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_TTS_MODEL = "qwen/qwen-audio-3.0-tts-flash"
OPENROUTER_TTS_MODELS = [
    "qwen/qwen-audio-3.0-tts-flash",
    "qwen/qwen-audio-3.0-tts-plus",
    "x-ai/grok-voice-tts-1.0",
]

DEFAULT_SAMPLES_DIR = (Path(__file__).parent / "samples" / "openrouter").resolve()
VOICE_STATS_FILE = DEFAULT_SAMPLES_DIR / "openrouter_voice_stats.json"
VOICE_SAMPLE_TEXT = """
Hello, this is a voice sample for analysis. The weather today is absolutely wonderful.
Technology has transformed our lives in remarkable ways. I hope you're having a great day.
Let me share some interesting facts about science and nature with you.
"""

MAX_CHAR_LIMIT = 2000  # Qwen default; use max_char_limit_for_model() at runtime


class OpenRouterTTSWrapper(TTSInterface):
    """OpenRouter TTS via OpenAI-compatible /audio/speech endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_OPENROUTER_TTS_MODEL,
        default_voice: Optional[str] = None,
        embedding_model_device: Optional[str] = None,
        enable_voice_matching: bool = True,
        cost_tracker: Optional[Any] = None,
        target_language: Optional[str] = None,
        enable_audio_validation: bool = True,
        max_silence_ratio: float = 0.5,
        fade_detection_enabled: bool = True,
        fade_window_size_ms: int = 500,
        min_fade_db: float = 10.0,
        fade_detection_percentile: int = 75,
        api_key: Optional[str] = None,
        enable_emotion_enrichment: bool = False,
        enable_llm_editor: bool = False,
        emotion_enrichment_model: Optional[str] = None,
        emotion_enrichment_temperature: Optional[float] = None,
        enable_content_validation: bool = False,
        content_validator_provider: str = "whisper",
        content_validator_whisper_model: str = "base",
        content_validator_whisper_compute_type: str = "int8",
        content_validator_whisper_cpu_threads: int = 2,
        **kwargs: Any
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Use 'pip install openai'.")
        if not PYDUB_AVAILABLE:
            raise ImportError("Pydub package not installed. Use 'pip install pydub'.")

        self.model = model or DEFAULT_OPENROUTER_TTS_MODEL
        self.target_language = target_language or "en"
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.client: Optional[OpenAI] = None
        self.voice_mapping: Dict[str, str] = {}
        self.voice_prompt_mapping: Dict[str, str] = {}
        self._audio_cache: Dict[str, tuple[str, float]] = {}
        self._cache_dir: Optional[str] = None
        self.cost_tracker = cost_tracker

        if default_voice:
            self.default_voice = self._validate_voice_name(default_voice)
        else:
            self.default_voice = default_voice_for(
                model=self.model,
                language=self.target_language,
            )

        self.enable_audio_validation = enable_audio_validation
        self.max_silence_ratio = max_silence_ratio
        self.fade_detection_enabled = fade_detection_enabled
        self.fade_window_size_ms = fade_window_size_ms
        self.min_fade_db = min_fade_db
        self.fade_detection_percentile = fade_detection_percentile

        self.enable_voice_matching = enable_voice_matching
        self.embedding_model_device = embedding_model_device
        self.audio_embedder: Optional[AudioEmbedder] = None
        self.voice_matcher: Optional[VoiceMatcher] = None
        self.voice_sample_manager: Optional[VoiceSampleManager] = None

        # Emotion enrichment (Qwen demo tags + NL prefixes in input text).
        self.enable_emotion_enrichment = bool(enable_emotion_enrichment)
        self.enable_llm_editor = bool(enable_llm_editor)
        self.emotion_enrichment_model = emotion_enrichment_model or "gemini-2.5-pro"
        self.emotion_enrichment_temperature = (
            float(emotion_enrichment_temperature)
            if emotion_enrichment_temperature is not None
            else 0.7
        )
        self.emotion_enricher: Optional[QwenEmotionEnricher] = None

        self.enable_content_validation = bool(enable_content_validation)
        self.content_validator_provider = content_validator_provider
        self.content_validator_whisper_model = content_validator_whisper_model
        self.content_validator_whisper_compute_type = content_validator_whisper_compute_type
        self.content_validator_whisper_cpu_threads = content_validator_whisper_cpu_threads
        self._content_validator = None

    def _voices_for_model(self, model: Optional[str] = None) -> List[str]:
        return voices_for_model(model or self.model)

    def _uses_qwen_markup(self) -> bool:
        return is_qwen_tts_model(self.model)

    def _max_char_limit(self) -> int:
        return max_char_limit_for_model(self.model)

    def _model_for_voice(self, voice_name: str) -> str:
        """Pick the OpenRouter model that owns this voice ID."""
        meta = OPENROUTER_VOICE_META.get(voice_name) or {}
        models = meta.get("models") or []
        if self.model in models:
            return self.model
        if models:
            return models[0]
        # Current-model voices / unknown IDs use the active model.
        if voice_name in self._voices_for_model(self.model):
            return self.model
        return self.model

    def regenerate_voice_samples(self, force_regenerate: bool = True) -> bool:
        """Regenerate OpenRouter voice samples and embeddings for matching."""
        if not self.is_available():
            raise RuntimeError("OpenRouter TTS not initialized.")
        if not self.voice_sample_manager:
            logger.warning("VoiceSampleManager not initialized.")
            return False
        logger.info("Regenerating OpenRouter voice samples and embeddings...")
        return self.voice_sample_manager.generate_all_samples(force_regenerate)

    def _register_usage(
        self,
        usage_tracker: Optional[Dict[str, Any]],
        model_name: str,
        input_characters: float = 0.0,
        audio_seconds: float = 0.0,
    ) -> None:
        if usage_tracker is None or not model_name:
            return
        models_map = usage_tracker.setdefault("models", {})
        model_usage = models_map.setdefault(model_name, {})
        if input_characters:
            model_usage["input_characters"] = model_usage.get("input_characters", 0.0) + float(input_characters)
        if audio_seconds:
            model_usage["audio_seconds"] = model_usage.get("audio_seconds", 0.0) + float(audio_seconds)

    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_mapping = mapping
        logger.debug(f"OpenRouter voice mapping set: {len(mapping)} entries.")

    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_prompt_mapping = mapping
        logger.debug(f"OpenRouter voice prompt mapping set: {len(mapping)} entries.")

    def _validate_voice_name(self, voice_name: str) -> str:
        fallback = (
            self.default_voice
            if hasattr(self, "default_voice")
            else default_voice_for(model=self.model, language=getattr(self, "target_language", None))
        )
        if not voice_name:
            return fallback

        allowed = self._voices_for_model(self.model)
        normalized = {v.lower(): v for v in allowed}
        all_normalized = {v.lower(): v for v in ALL_OPENROUTER_VOICES}

        key = voice_name.strip().lower()
        if key in normalized:
            return normalized[key]
        if key in all_normalized:
            logger.warning(
                f"Voice '{voice_name}' is for another OpenRouter TTS model. "
                f"Falling back to '{fallback}' for model '{self.model}'."
            )
            return fallback

        # DashScope names (Alek/Cherry/…) are not accepted by OpenRouter Qwen endpoints.
        logger.warning(
            f"Voice '{voice_name}' is not supported on OpenRouter TTS model '{self.model}'. "
            f"Using '{fallback}'."
        )
        return fallback

    def initialize(self) -> None:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        try:
            self.client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/anomalyco/DubbLM",
                    "X-Title": "DubbLM",
                },
            )
            self._cache_dir = tempfile.mkdtemp(prefix="openrouter_tts_cache_")
            logger.info(
                f"OpenRouter TTS initialized with model: {self.model}, default voice: {self.default_voice}"
            )

            if self.enable_emotion_enrichment and not self._uses_qwen_markup():
                logger.info(
                    f"Skipping OpenRouter emotion enrichment for non-Qwen model '{self.model}'."
                )
                self.enable_emotion_enrichment = False

            if self.enable_emotion_enrichment and self.enable_llm_editor:
                logger.info(
                    "Skipping runtime OpenRouter emotion enrichment because LLM editor is enabled."
                )
                self.enable_emotion_enrichment = False

            if self.enable_emotion_enrichment:
                try:
                    self.emotion_enricher = QwenEmotionEnricher(
                        model=self.emotion_enrichment_model,
                        temperature=self.emotion_enrichment_temperature,
                    )
                    self.emotion_enricher.initialize()
                    if self.emotion_enricher.llm:
                        logger.info(
                            f"OpenRouter Qwen emotion enrichment enabled "
                            f"(model={self.emotion_enrichment_model})"
                        )
                    else:
                        logger.warning(
                            "Qwen emotion enricher LLM unavailable; "
                            "falling back to deterministic tag/NL preparation."
                        )
                except Exception as exc:
                    logger.error(f"Failed to init Qwen emotion enricher: {exc}")
                    self.emotion_enricher = None
            else:
                logger.info("OpenRouter emotion enrichment disabled by configuration.")

            # Anti-speak is only relevant when Qwen demo tags may appear in input.
            if (
                self.enable_content_validation
                and self._uses_qwen_markup()
                and self.content_validator_provider == "whisper"
            ):
                try:
                    from src.tts.local_whisper_validator import LocalWhisperContentValidator
                    self._content_validator = LocalWhisperContentValidator(
                        language_code=self.target_language,
                        model_name=self.content_validator_whisper_model,
                        compute_type=self.content_validator_whisper_compute_type,
                        cpu_threads=self.content_validator_whisper_cpu_threads,
                    )
                    logger.info("OpenRouter tag anti-speak whisper validator enabled.")
                except Exception as exc:
                    logger.warning(f"Content validator unavailable for OpenRouter TTS: {exc}")
                    self._content_validator = None

            if self.enable_voice_matching:
                try:
                    # Only match/pin against voices supported by the active model.
                    # MP3 samples for other OpenRouter Qwen models may already exist on disk
                    # and will be used when that model is selected.
                    current_model_voices = self._voices_for_model(self.model)

                    self.audio_embedder = AudioEmbedder(device=self.embedding_model_device)
                    self.voice_matcher = VoiceMatcher(
                        audio_embedder=self.audio_embedder,
                        enable_matching=True,
                    )
                    self.voice_sample_manager = VoiceSampleManager(
                        tts_provider="openrouter",
                        voice_list=current_model_voices,
                        samples_dir=DEFAULT_SAMPLES_DIR,
                        stats_file=VOICE_STATS_FILE,
                        adjustments_file=DEFAULT_SAMPLES_DIR / "openrouter_duration_adjustments.json",
                        sample_text=VOICE_SAMPLE_TEXT,
                        audio_embedder=self.audio_embedder,
                        voice_matcher=self.voice_matcher,
                        enable_voice_matching=True,
                        enable_audio_validation=False,
                    )

                    def generate_sample(voice_name: str, output_path: str) -> bool:
                        try:
                            sample_model = self._model_for_voice(voice_name)
                            response = self.client.audio.speech.create(
                                model=sample_model,
                                voice=voice_name,
                                input=VOICE_SAMPLE_TEXT,
                                response_format="mp3",
                            )
                            response.write_to_file(output_path)
                            return True
                        except Exception as exc:
                            logger.error(f"Error generating OpenRouter sample for {voice_name}: {exc}")
                            return False

                    self.voice_sample_manager.set_sample_generator(generate_sample)
                    self.voice_sample_manager.generate_all_samples()
                except Exception as exc:
                    logger.warning(f"Failed to initialize OpenRouter voice matching: {exc}")
                    self.enable_voice_matching = False
                    self.audio_embedder = None
                    self.voice_matcher = None
                    self.voice_sample_manager = None
            else:
                logger.info("OpenRouter voice matching disabled by configuration.")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize OpenRouter TTS client: {exc}") from exc

    def find_and_pin_voice_for_speaker(
        self,
        speaker_id: str,
        reference_audio_path: Union[str, Path],
        force_search: bool = False,
        exclude_voices: Optional[List[str]] = None,
    ) -> Optional[str]:
        if not self.is_available():
            raise RuntimeError("OpenRouter TTS not initialized.")

        if not self.enable_voice_matching or not self.voice_matcher:
            return self.voice_mapping.get(speaker_id, self.default_voice)

        if not force_search and speaker_id in self.voice_mapping:
            return self.voice_mapping[speaker_id]

        if not reference_audio_path:
            return self.default_voice

        ref_path = Path(reference_audio_path)
        if not ref_path.exists():
            logger.warning(f"Reference audio file not found: {reference_audio_path}. Using default.")
            return self.default_voice

        reference_embeddings = self.voice_matcher.extract_multiple_embeddings(
            ref_path,
            num_segments=3,
            segment_duration_ms=3000,
        )
        if not reference_embeddings:
            return self.default_voice

        best_match_voice = self.voice_matcher.find_best_matching_voice_multi_segment(
            reference_embeddings,
            exclude_voices=exclude_voices,
        )
        if best_match_voice:
            self.voice_mapping[speaker_id] = best_match_voice
            logger.info(f"Matched and pinned speaker '{speaker_id}' to voice '{best_match_voice}'")
            return best_match_voice

        return self.default_voice

    def _get_cache_key(
        self,
        segment_data: TTSSegmentData,
        language: str,
        speech_text: Optional[str] = None,
    ) -> str:
        voice_name = segment_data.voice or self.voice_mapping.get(segment_data.speaker, self.default_voice)
        style_prompt = segment_data.style_prompt or self.voice_prompt_mapping.get(segment_data.speaker, "")
        key_data = {
            "text": speech_text if speech_text is not None else segment_data.text,
            "speaker": segment_data.speaker,
            "voice": voice_name,
            "style_prompt": style_prompt,
            "emotion": segment_data.emotion or "Neutral",
            "speed": segment_data.speed or 1.0,
            "language": language,
            "model": self.model,
            "emotion_enrichment": bool(self.enable_emotion_enrichment),
        }
        key_str = str(sorted(key_data.items()))
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _prepare_speech_text(
        self,
        segment_data: TTSSegmentData,
        previous_segments: Optional[Sequence[str]] = None,
    ) -> str:
        """Build OpenRouter input text; Qwen markup only for qwen/* models."""
        raw = segment_data.text or ""

        # Grok and other non-Qwen models: plain text only (strip provider markup leaks).
        if not self._uses_qwen_markup():
            prepared = strip_markup_tags(raw).strip() or raw.strip()
            if prepared != raw:
                logger.debug(
                    f"OpenRouter speech text stripped for [{segment_data.speaker}] "
                    f"(model={self.model}): '{raw[:60]}' -> '{prepared[:80]}'"
                )
            return prepared

        style_prompt = segment_data.style_prompt or self.voice_prompt_mapping.get(
            segment_data.speaker, ""
        )

        if self.enable_emotion_enrichment and self.emotion_enricher and self.emotion_enricher.llm:
            enriched = self.emotion_enricher.enrich_text(
                raw, previous_segments=list(previous_segments or [])
            )
            # Layer style_prompt even after LLM enrich (as NL prefix).
            style_instr = style_prompt_to_instruction(style_prompt)
            if style_instr:
                enriched = apply_nl_instruction_prefix(enriched, style_instr)
            # If segment has an explicit emotion without a matching leading tag, add it.
            prepared = prepare_qwen_speech_text(
                enriched,
                emotion=segment_data.emotion,
                style_prompt=None,  # already applied
                prefer_tags=True,
            )
        elif self.enable_emotion_enrichment or style_prompt or segment_data.emotion:
            prepared = prepare_qwen_speech_text(
                raw,
                emotion=segment_data.emotion,
                style_prompt=style_prompt,
                prefer_tags=True,
            )
        else:
            # Still sanitize any pre-existing tags from translation/editor.
            prepared = sanitize_qwen_markup(raw)

        if prepared != raw:
            logger.debug(
                f"OpenRouter speech text prepared for [{segment_data.speaker}]: "
                f"'{raw[:60]}' -> '{prepared[:80]}'"
            )
        return prepared

    def _fallback_nl_only_text(self, segment_data: TTSSegmentData) -> str:
        """Strip tags; keep only NL instruction prefixes (safer if tags are spoken)."""
        plain = strip_markup_tags(segment_data.text or "")
        style_prompt = segment_data.style_prompt or self.voice_prompt_mapping.get(
            segment_data.speaker, ""
        )
        prepared = plain
        style_instr = style_prompt_to_instruction(style_prompt)
        if style_instr:
            prepared = apply_nl_instruction_prefix(prepared, style_instr)
        nl = nl_instruction_for_emotion(segment_data.emotion)
        if nl:
            prepared = apply_nl_instruction_prefix(prepared, nl)
        return prepared.strip()

    def _tag_leak_detected(self, audio_path: str, speech_text: str) -> List[str]:
        if not self._content_validator:
            return []
        try:
            transcript = self._content_validator.transcribe_text(audio_path)
        except Exception as exc:
            logger.debug(f"OpenRouter anti-speak ASR failed: {exc}")
            return []
        return spoken_tag_leak_hits(transcript, speech_text)

    def _synthesize_single_segment(
        self,
        segment_data: TTSSegmentData,
        temp_output_path: str,
        language: str,
        usage_tracker: Optional[Dict[str, Any]] = None,
        previous_segments: Optional[Sequence[str]] = None,
    ) -> None:
        if not self.client:
            raise RuntimeError("OpenRouter client not initialized.")

        synth_start_ts = time.perf_counter()
        report_segment_index = getattr(segment_data, "segment_index", None)
        report_group_id = getattr(segment_data, "group_id", None)
        total_attempts = 0

        speech_text = self._prepare_speech_text(segment_data, previous_segments=previous_segments)
        used_tag_fallback = False

        cache_key = self._get_cache_key(segment_data, language, speech_text=speech_text)
        if cache_key in self._audio_cache:
            cached_path, _ = self._audio_cache[cache_key]
            if os.path.exists(cached_path):
                shutil.copy(cached_path, temp_output_path)
                self._record_segment_report(SegmentSynthesisReport(
                    segment_index=report_segment_index if report_segment_index is not None else -1,
                    speaker=segment_data.speaker,
                    text=speech_text,
                    requested_model=self.model,
                    actual_model=self.model,
                    attempts=0,
                    used_fallback=False,
                    success=True,
                    duration_seconds=time.perf_counter() - synth_start_ts,
                    output_path=temp_output_path,
                    group_id=report_group_id,
                ))
                return

        speaker_id = segment_data.speaker
        voice_name = segment_data.voice or self.voice_mapping.get(speaker_id, self.default_voice)
        voice_name = self._validate_voice_name(voice_name)

        logger.debug(
            f"  TTS Synthesis for [{speaker_id}]: voice={voice_name}, model={self.model}, "
            f"text='{speech_text[:80]}'"
        )

        max_validation_retries = 3 if self.enable_audio_validation else 1
        # Reserve an extra pass for anti-speak NL fallback.
        if self.enable_emotion_enrichment or self._content_validator is not None:
            max_validation_retries = max(max_validation_retries, 2)
        best_attempt_path: Optional[str] = None
        best_silence_ratio = float("inf")
        best_speech_text = speech_text

        for validation_attempt in range(max_validation_retries):
            total_attempts += 1
            segment_audio_files: List[str] = []
            temp_dir_for_chunks = tempfile.mkdtemp(prefix="openrouter_chunks_")
            temp_attempt_path = f"{temp_output_path}_attempt_{validation_attempt}.mp3"
            attempt_text = speech_text

            char_limit = self._max_char_limit()
            if len(attempt_text) > char_limit:
                logger.warning(
                    f"  OpenRouter: Text for speaker {speaker_id} ({len(attempt_text)} chars) "
                    f"exceeds limit ({char_limit}). Splitting into chunks."
                )
                text_chunks = greedy_sent_split(attempt_text, char_limit)
            else:
                text_chunks = [attempt_text]

            try:
                for i, chunk_text in enumerate(text_chunks):
                    chunk_file_path = os.path.join(temp_dir_for_chunks, f"chunk_{i}.mp3")
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            speed = segment_data.speed if segment_data.speed and segment_data.speed > 0 else 1.0
                            create_kwargs: Dict[str, Any] = {
                                "model": self.model,
                                "voice": voice_name,
                                "input": chunk_text,
                                "response_format": "mp3",
                            }
                            if speed != 1.0:
                                create_kwargs["speed"] = float(speed)

                            response = self.client.audio.speech.create(**create_kwargs)
                            response.write_to_file(chunk_file_path)
                            self._register_usage(
                                usage_tracker,
                                self.model,
                                input_characters=len(chunk_text or ""),
                            )
                            segment_audio_files.append(chunk_file_path)
                            break
                        except Exception as chunk_exc:
                            logger.error(
                                f"    OpenRouter: Attempt {attempt + 1}/{max_attempts} "
                                f"for chunk {i + 1} failed: {chunk_exc}"
                            )
                            if attempt + 1 >= max_attempts:
                                raise RuntimeError(
                                    f"OpenRouter TTS failed for chunk {i + 1} after {max_attempts} attempts: {chunk_exc}"
                                ) from chunk_exc
                            time.sleep(1.5 ** attempt)

                if not segment_audio_files:
                    raise RuntimeError(f"No audio chunks were generated for speaker {speaker_id}.")

                if len(segment_audio_files) == 1:
                    shutil.copy(segment_audio_files[0], temp_attempt_path)
                else:
                    combined = AudioSegment.empty()
                    for audio_file_path in segment_audio_files:
                        combined += AudioSegment.from_mp3(audio_file_path)
                    combined.export(temp_attempt_path, format="mp3")
            finally:
                if os.path.exists(temp_dir_for_chunks):
                    shutil.rmtree(temp_dir_for_chunks, ignore_errors=True)

            # Anti-speak: if the model read tag words aloud, retry without tags.
            if not used_tag_fallback:
                leaks = self._tag_leak_detected(temp_attempt_path, attempt_text)
                if leaks:
                    logger.warning(
                        f"  OpenRouter: tag anti-speak hit for [{speaker_id}] "
                        f"(leaked={leaks}). Retrying with NL-only prefix."
                    )
                    speech_text = self._fallback_nl_only_text(segment_data)
                    used_tag_fallback = True
                    best_speech_text = speech_text
                    if os.path.exists(temp_attempt_path):
                        try:
                            os.remove(temp_attempt_path)
                        except OSError:
                            pass
                    continue

            if self.enable_audio_validation:
                fade_config = {
                    "enabled": self.fade_detection_enabled,
                    "window_size_ms": self.fade_window_size_ms,
                    "min_fade_db": self.min_fade_db,
                    "percentile": self.fade_detection_percentile,
                }
                is_valid, reason, silence_ratio = AudioValidator.validate_audio_sample(
                    temp_attempt_path,
                    max_silence_ratio=self.max_silence_ratio,
                    fade_detection_config=fade_config,
                )

                if silence_ratio < best_silence_ratio:
                    if best_attempt_path and os.path.exists(best_attempt_path):
                        try:
                            os.remove(best_attempt_path)
                        except OSError:
                            pass
                    best_silence_ratio = silence_ratio
                    best_attempt_path = temp_attempt_path
                elif os.path.exists(temp_attempt_path) and temp_attempt_path != best_attempt_path:
                    try:
                        os.remove(temp_attempt_path)
                    except OSError:
                        pass

                if is_valid:
                    shutil.copy(best_attempt_path, temp_output_path)
                    if best_attempt_path and os.path.exists(best_attempt_path):
                        try:
                            os.remove(best_attempt_path)
                        except OSError:
                            pass
                    self._record_segment_report(SegmentSynthesisReport(
                        segment_index=report_segment_index if report_segment_index is not None else -1,
                        speaker=segment_data.speaker,
                        text=best_speech_text,
                        requested_model=self.model,
                        actual_model=self.model,
                        attempts=total_attempts,
                        used_fallback=used_tag_fallback,
                        success=True,
                        duration_seconds=time.perf_counter() - synth_start_ts,
                        output_path=temp_output_path,
                        group_id=report_group_id,
                    ))
                    return
                logger.warning(
                    f"  OpenRouter: Audio validation failed for speaker {speaker_id} "
                    f"(attempt {validation_attempt + 1}/{max_validation_retries}): {reason}"
                )
                if validation_attempt + 1 < max_validation_retries:
                    time.sleep(1.0)
            else:
                shutil.copy(temp_attempt_path, temp_output_path)
                if os.path.exists(temp_attempt_path):
                    try:
                        os.remove(temp_attempt_path)
                    except OSError:
                        pass
                self._record_segment_report(SegmentSynthesisReport(
                    segment_index=report_segment_index if report_segment_index is not None else -1,
                    speaker=segment_data.speaker,
                    text=best_speech_text,
                    requested_model=self.model,
                    actual_model=self.model,
                    attempts=total_attempts,
                    used_fallback=used_tag_fallback,
                    success=True,
                    duration_seconds=time.perf_counter() - synth_start_ts,
                    output_path=temp_output_path,
                    group_id=report_group_id,
                ))
                return

        if best_attempt_path and os.path.exists(best_attempt_path):
            shutil.copy(best_attempt_path, temp_output_path)
            try:
                os.remove(best_attempt_path)
            except OSError:
                pass
            self._record_segment_report(SegmentSynthesisReport(
                segment_index=report_segment_index if report_segment_index is not None else -1,
                speaker=segment_data.speaker,
                text=best_speech_text,
                requested_model=self.model,
                actual_model=self.model,
                attempts=total_attempts,
                used_fallback=used_tag_fallback,
                success=False,
                duration_seconds=time.perf_counter() - synth_start_ts,
                output_path=temp_output_path,
                group_id=report_group_id,
                error=f"validation failed (silence_ratio={best_silence_ratio:.2f})",
            ))
            return

        self._record_segment_report(SegmentSynthesisReport(
            segment_index=report_segment_index if report_segment_index is not None else -1,
            speaker=segment_data.speaker,
            text=best_speech_text,
            requested_model=self.model,
            actual_model=None,
            attempts=total_attempts,
            used_fallback=used_tag_fallback,
            success=False,
            duration_seconds=time.perf_counter() - synth_start_ts,
            output_path=None,
            group_id=report_group_id,
            error="all attempts failed",
        ))
        raise RuntimeError(
            f"OpenRouter TTS failed to generate valid audio for speaker {speaker_id} "
            f"after {max_validation_retries} attempts"
        )

    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: Optional[str] = None,
        **kwargs: Any
    ) -> List[SegmentAlignment]:
        if not self.client:
            raise RuntimeError("OpenRouter client not initialized. Call initialize() first.")
        if not segments_data:
            return []

        language = language or self.target_language
        previous_context: List[str] = list(kwargs.get("previous_context") or [])
        assigned_voices: List[str] = []

        if self.enable_voice_matching and self.voice_matcher:
            speakers_to_pin = []
            seen = set()
            for segment in segments_data:
                if (
                    segment.speaker
                    and segment.speaker not in self.voice_mapping
                    and segment.speaker not in seen
                    and segment.reference_audio_path
                ):
                    speakers_to_pin.append((segment.speaker, segment.reference_audio_path))
                    seen.add(segment.speaker)

            for speaker_id, ref_path in speakers_to_pin:
                pinned = self.find_and_pin_voice_for_speaker(
                    speaker_id, ref_path, exclude_voices=assigned_voices
                )
                if pinned and pinned != self.default_voice:
                    assigned_voices.append(pinned)

        temp_dir = tempfile.mkdtemp(prefix="openrouter_segments_")
        alignments: List[SegmentAlignment] = []
        usage_tracker: Dict[str, Any] = {"models": {}}

        try:
            for i, segment in enumerate(segments_data):
                segment_file_path = os.path.join(temp_dir, f"segment_{i}_{segment.speaker}.mp3")

                logger.info(
                    f"OpenRouter: Synthesizing segment {i + 1}/{len(segments_data)} "
                    f"for speaker '{segment.speaker}'"
                )
                try:
                    speech_text_preview = self._prepare_speech_text(
                        segment, previous_segments=previous_context
                    )
                    cache_key = self._get_cache_key(
                        segment, language, speech_text=speech_text_preview
                    )
                    if cache_key in self._audio_cache:
                        cached_path, duration = self._audio_cache[cache_key]
                        if os.path.exists(cached_path):
                            shutil.copy(cached_path, segment_file_path)
                            if segment.output_path:
                                os.makedirs(os.path.dirname(segment.output_path) or ".", exist_ok=True)
                                shutil.copy(segment_file_path, segment.output_path)
                            alignments.append(SegmentAlignment(
                                original_segment=segment,
                                diarized_segment=DiarizationSegment(
                                    start_time=0.0,
                                    end_time=duration,
                                    speaker=segment.speaker,
                                    text=segment.text,
                                    confidence=1.0,
                                ),
                                alignment_confidence=1.0,
                            ))
                            if segment.text:
                                previous_context.append(segment.text)
                            continue

                    self._synthesize_single_segment(
                        segment,
                        segment_file_path,
                        language,
                        usage_tracker=usage_tracker,
                        previous_segments=previous_context,
                    )
                    audio_segment = AudioSegment.from_mp3(segment_file_path)
                    duration = len(audio_segment) / 1000.0
                    self._register_usage(usage_tracker, self.model, audio_seconds=duration)

                    if self._cache_dir:
                        cache_path = os.path.join(self._cache_dir, f"{cache_key}.mp3")
                        shutil.copy(segment_file_path, cache_path)
                        self._audio_cache[cache_key] = (cache_path, duration)

                    if segment.output_path:
                        os.makedirs(os.path.dirname(segment.output_path) or ".", exist_ok=True)
                        shutil.copy(segment_file_path, segment.output_path)

                    alignments.append(SegmentAlignment(
                        original_segment=segment,
                        diarized_segment=DiarizationSegment(
                            start_time=0.0,
                            end_time=duration,
                            speaker=segment.speaker,
                            text=segment.text,
                            confidence=1.0,
                        ),
                        alignment_confidence=1.0,
                    ))
                    if segment.text:
                        previous_context.append(segment.text)
                except Exception as segment_exc:
                    logger.error(
                        f"Error synthesizing segment {i + 1} for speaker '{segment.speaker}': {segment_exc}"
                    )

            if self.cost_tracker:
                models_usage = usage_tracker.get("models", {})
                for model_name, metrics in models_usage.items():
                    input_chars = metrics.get("input_characters", 0.0)
                    audio_sec = metrics.get("audio_seconds", 0.0)
                    if input_chars or audio_sec:
                        self.cost_tracker.add_tts_actual(
                            "openrouter",
                            model=model_name,
                            audio_seconds=audio_sec,
                            input_characters=input_chars,
                        )
            return alignments
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and PYDUB_AVAILABLE and self.client is not None

    def clone_voice(self, audio_path: str, voice_id: str) -> str:
        raise NotImplementedError(
            "Voice cloning is not supported by OpenRouter Qwen TTS. Use Minimax TTS for cloning."
        )

    def cleanup(self) -> None:
        if self._cache_dir and os.path.exists(self._cache_dir):
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir = None
        self._audio_cache.clear()
        if self.voice_matcher:
            self.voice_matcher.clear()
            self.voice_matcher = None
        self.audio_embedder = None
        self.voice_sample_manager = None

    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: Optional[str] = None,
    ) -> Optional[float]:
        if not segment_data.text or not segment_data.text.strip():
            return 0.0

        language = language or self.target_language
        # Tags / NL instruction prefixes must not dominate length estimation.
        plain_text = strip_markup_tags(segment_data.text)

        if self.voice_sample_manager:
            voice_name = segment_data.voice or self.voice_mapping.get(
                segment_data.speaker, self.default_voice
            )
            voice_name = self._validate_voice_name(voice_name)
            return self.voice_sample_manager.estimate_duration(
                text=plain_text,
                voice_name=voice_name,
                language=language,
                style_prompt=segment_data.style_prompt,
                emotion=segment_data.emotion,
                speed=segment_data.speed,
                speaker_id=segment_data.speaker,
                apply_biases=True,
            )

        import re
        words = re.findall(r"\b\w+\b", plain_text.lower())
        estimated_duration = (len(words) / 175.0) * 60.0
        if segment_data.speed and segment_data.speed > 0:
            estimated_duration /= segment_data.speed
        return estimated_duration
