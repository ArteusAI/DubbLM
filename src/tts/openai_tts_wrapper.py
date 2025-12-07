from typing import Optional, Dict, Any, List, Union
import os
import time
import tempfile
import shutil
import hashlib
import json
import numpy as np
from pathlib import Path
from collections import Counter

from .models import TTSSegmentData, SegmentAlignment, DiarizationSegment 
from .voice_sample_manager import VoiceSampleManager, AudioFileUtils, AudioValidator
from src.tts.tts_interface import TTSInterface
from src.utils.sent_split import greedy_sent_split
from src.utils.audio_embedder import AudioEmbedder
from src.utils.voice_matcher import VoiceMatcher
from src.dubbing.core.log_config import get_logger

# Import OpenAI and Pydub dependencies, with error handling
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

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False

logger = get_logger(__name__)

# Constants
ALL_OPENAI_VOICES: List[str] = [
    "alloy", 
    "ash", 
    "coral", 
    "echo",
    "fable", 
    "onyx", 
    "nova", 
    "sage", 
    "shimmer", 
]

# Resolve samples directory relative to this file's location (src/tts/)
DEFAULT_SAMPLES_DIR = (Path(__file__).parent / "samples" / "openai").resolve()
EMBEDDING_CACHE_FILE = DEFAULT_SAMPLES_DIR / "openai_voice_embeddings.json"

# Sample text for voice analysis - shorter than Gemini's as we'll generate actual audio
VOICE_SAMPLE_TEXT = """
Hello, this is a voice sample for analysis. The weather today is absolutely wonderful. 
Technology has transformed our lives in remarkable ways. I hope you're having a great day.
Let me share some interesting facts about science and nature with you.
"""

class OpenAITTSWrapper(TTSInterface):
    """
    OpenAI TTS wrapper with voice matching capabilities.
    """
    
    def __init__(
        self,
        model: str = "tts-1",
        default_voice: str = "alloy",
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
        **kwargs: Any
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Use 'pip install openai'.")
        if not PYDUB_AVAILABLE:
            raise ImportError("Pydub package not installed. Use 'pip install pydub'.")
            
        self.model = model
        self.default_voice = self._validate_voice_name(default_voice)
        self.target_language = target_language or "en"
        self.client: Optional[OpenAI] = None
        self.voice_mapping: Dict[str, str] = {}
        self.voice_prompt_mapping: Dict[str, str] = {}
        self._audio_cache: Dict[str, tuple[str, float]] = {}
        self._cache_dir = None
        self.cost_tracker = cost_tracker

        # Audio validation settings
        self.enable_audio_validation = enable_audio_validation
        self.max_silence_ratio = max_silence_ratio
        self.fade_detection_enabled = fade_detection_enabled
        self.fade_window_size_ms = fade_window_size_ms
        self.min_fade_db = min_fade_db
        self.fade_detection_percentile = fade_detection_percentile

        # Voice matching components
        self.enable_voice_matching = enable_voice_matching
        self.embedding_model_device = embedding_model_device
        self.audio_embedder: Optional[AudioEmbedder] = None
        self.voice_matcher: Optional[VoiceMatcher] = None
        self.voice_sample_manager: Optional[VoiceSampleManager] = None

    def _register_usage(
        self,
        usage_tracker: Optional[Dict[str, Any]],
        model_name: str,
        input_tokens: float = 0.0,
        audio_seconds: float = 0.0,
        output_tokens: float = 0.0
    ) -> None:
        """Accumulate token/audio usage in a simple dictionary tracker."""
        if usage_tracker is None or not model_name:
            return

        models_map = usage_tracker.setdefault("models", {})
        model_usage = models_map.setdefault(model_name, {})

        if input_tokens:
            model_usage["input_tokens"] = model_usage.get("input_tokens", 0.0) + float(input_tokens)
        if output_tokens:
            model_usage["output_tokens"] = model_usage.get("output_tokens", 0.0) + float(output_tokens)
        if audio_seconds:
            model_usage["audio_seconds"] = model_usage.get("audio_seconds", 0.0) + float(audio_seconds)

    def _count_tokens(self, text: Optional[str]) -> int:
        """Count tokens for billing using tiktoken."""
        if not text:
            return 0

        if not (TIKTOKEN_AVAILABLE and tiktoken is not None):
            raise RuntimeError("tiktoken is required for OpenAI TTS token counting. Install the 'tiktoken' package.")

        encoding = None
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except Exception:
            encoding = None
        if encoding is None:
            encoding = tiktoken.get_encoding("cl100k_base")

        try:
            return len(encoding.encode(text))
        except Exception as exc:
            raise RuntimeError(f"Failed to tokenize text for model '{self.model}': {exc}")

    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_mapping = mapping
        logger.debug(f"OpenAI voice mapping set: {len(mapping)} entries. Preview: {dict(list(mapping.items())[:3])}")
        
    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        # OpenAI's standard TTS models (tts-1, tts-1-hd) do not support explicit style prompts via API parameter.
        # Style/emotion is typically influenced by the input text itself or potentially voice choice.
        # This mapping is stored but might be used to prepend to text if a future model supports it.
        self.voice_prompt_mapping = mapping
        logger.debug(f"OpenAI voice prompt mapping set: {len(mapping)} entries. Note: OpenAI tts-1 model family does not use explicit API style prompts.")
    
    def _validate_voice_name(self, voice_name: str) -> str:
        """Validate that a voice name is supported by OpenAI TTS."""
        normalized_voices = {v.lower(): v for v in ALL_OPENAI_VOICES}
        
        if voice_name.lower() in normalized_voices:
            return normalized_voices[voice_name.lower()]
        
        logger.warning(f"Warning: Voice '{voice_name}' not supported. Using default 'alloy'.")
        return "alloy"
        
    def initialize(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        try:
            self.client = OpenAI(api_key=api_key)
            # Create cache directory
            self._cache_dir = tempfile.mkdtemp(prefix="openai_tts_cache_")
            logger.info(f"OpenAI TTS initialized with model: {self.model} and default voice: {self.default_voice}")
            logger.debug(f"Audio cache directory: {self._cache_dir}")
            
            # Initialize voice matching if enabled
            if self.enable_voice_matching:
                try:
                    self.audio_embedder = AudioEmbedder(device=self.embedding_model_device)
                    self.voice_matcher = VoiceMatcher(
                        audio_embedder=self.audio_embedder,
                        enable_matching=True
                    )
                    logger.debug("Voice matching enabled with AudioEmbedder")
                    
                    # Initialize VoiceSampleManager
                    self.voice_sample_manager = VoiceSampleManager(
                        tts_provider="openai",
                        voice_list=ALL_OPENAI_VOICES,
                        samples_dir=DEFAULT_SAMPLES_DIR,
                        stats_file=EMBEDDING_CACHE_FILE,  # OpenAI uses the same file for embeddings and stats
                        adjustments_file=DEFAULT_SAMPLES_DIR / "openai_duration_adjustments.json",
                        sample_text=VOICE_SAMPLE_TEXT,
                        audio_embedder=self.audio_embedder,
                        voice_matcher=self.voice_matcher,
                        enable_voice_matching=True,
                        enable_audio_validation=False,  # OpenAI samples don't need validation
                    )
                    
                    # Set up sample generation callback
                    def generate_openai_sample(voice_name: str, output_path: str) -> bool:
                        """Callback for VoiceSampleManager to generate OpenAI samples."""
                        try:
                            response = self.client.audio.speech.create(
                                model=self.model,
                                voice=voice_name,
                                input=VOICE_SAMPLE_TEXT,
                                response_format="mp3"
                            )
                            response.write_to_file(output_path)
                            return True
                        except Exception as e:
                            logger.error(f"Error generating sample for {voice_name}: {e}")
                            return False
                    
                    self.voice_sample_manager.set_sample_generator(generate_openai_sample)
                    
                    # Load or generate voice samples and embeddings
                    self.voice_sample_manager.generate_all_samples()
                except Exception as e:
                    logger.warning(f"Warning: Failed to initialize voice matching: {e}")
                    self.enable_voice_matching = False
                    self.audio_embedder = None
                    self.voice_matcher = None
                    self.voice_sample_manager = None
            else:
                logger.info("Voice matching disabled by configuration.")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def find_and_pin_voice_for_speaker(self, speaker_id: str, reference_audio_path: Union[str, Path],
                                     force_search: bool = False,
                                     exclude_voices: Optional[List[str]] = None) -> Optional[str]:
        """Find the best matching OpenAI voice for a reference audio and pin it to the speaker."""
        if not self.is_available():
            raise RuntimeError("OpenAI TTS not initialized.")

        if not self.enable_voice_matching or not self.voice_matcher:
            return self.voice_mapping.get(speaker_id, self.default_voice)

        if not force_search and speaker_id in self.voice_mapping:
            return self.voice_mapping[speaker_id]

        if not reference_audio_path:
            logger.warning(f"Reference audio path not provided for speaker '{speaker_id}'. Using default.")
            return self.default_voice

        ref_path = Path(reference_audio_path)
        if not ref_path.exists():
            logger.warning(f"Reference audio file not found: {reference_audio_path}. Using default.")
            return self.default_voice

        # Extract multiple embeddings from different parts of the audio
        reference_embeddings = self.voice_matcher.extract_multiple_embeddings(
            ref_path,
            num_segments=3,  # Extract from start, middle, and end
            segment_duration_ms=3000  # 3 seconds per segment
        )
        
        if not reference_embeddings:
            logger.warning(f"Could not extract embeddings from reference audio.")
            return self.default_voice
        
        logger.debug(f"Extracted {len(reference_embeddings)} embeddings from reference audio")

        # Find best matching voice using voting across multiple segments
        best_match_voice = self.voice_matcher.find_best_matching_voice_multi_segment(
            reference_embeddings,
            exclude_voices=exclude_voices
        )

        if best_match_voice:
            self.voice_mapping[speaker_id] = best_match_voice
            logger.info(f"Matched and pinned speaker '{speaker_id}' to voice '{best_match_voice}'")
            return best_match_voice

        logger.warning(f"Could not find matching voice for speaker '{speaker_id}'. Using default.")
        return self.default_voice
    
    def regenerate_voice_samples(self, force_regenerate: bool = True) -> bool:
        """Regenerate voice samples and embeddings."""
        if not self.is_available():
            raise RuntimeError("OpenAI TTS not initialized.")
        
        if not self.voice_sample_manager:
            logger.warning("VoiceSampleManager not initialized.")
            return False
        
        logger.info("Regenerating OpenAI voice samples and embeddings...")
        return self.voice_sample_manager.generate_all_samples(force_regenerate)
    
    def _get_cache_key(self, segment_data: TTSSegmentData, language: str) -> str:
        """Generate a unique cache key for a segment based on its properties."""
        # Include all relevant parameters that affect audio generation
        voice_name = segment_data.voice or self.voice_mapping.get(segment_data.speaker, self.default_voice)
        style_prompt = segment_data.style_prompt or self.voice_prompt_mapping.get(segment_data.speaker, "")
        
        # Create a unique key
        key_data = {
            "text": segment_data.text,
            "speaker": segment_data.speaker,
            "voice": voice_name,
            "style_prompt": style_prompt,
            "emotion": segment_data.emotion or "Neutral",
            "speed": segment_data.speed or 1.0,
            "language": language,
            "model": self.model
        }
        
        # Create hash of the key data
        key_str = str(sorted(key_data.items()))
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _synthesize_single_segment(
        self,
        segment_data: TTSSegmentData,
        temp_output_path: str,
        language: str, # OpenAI generally auto-detects language from input text
        usage_tracker: Optional[Dict[str, Any]] = None
    ) -> None:
        """Synthesizes a single segment and saves it to a temporary path."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized.")

        # Check cache first
        cache_key = self._get_cache_key(segment_data, language)
        if cache_key in self._audio_cache:
            cached_path, _ = self._audio_cache[cache_key]
            if os.path.exists(cached_path):
                shutil.copy(cached_path, temp_output_path)
                logger.debug(f"  OpenAI: Using cached audio for speaker {segment_data.speaker}")
                return

        speaker_id = segment_data.speaker
        text_to_synthesize = segment_data.text
        
        # Determine voice: per-segment -> global mapping -> default
        voice_name = segment_data.voice or self.voice_mapping.get(speaker_id, self.default_voice)
        voice_name = self._validate_voice_name(voice_name)  # Ensure it's a valid OpenAI voice
        
        final_text = text_to_synthesize

        # OpenAI API character limit is 4096 for tts-1 models.
        MAX_CHAR_LIMIT = 4096
        text_chunks_for_openai: List[str] = []

        if len(final_text) > MAX_CHAR_LIMIT:
            logger.warning(f"  OpenAI: Text for speaker {speaker_id} ({len(final_text)} chars) exceeds limit. Splitting into chunks.")
            text_chunks_for_openai = greedy_sent_split(final_text, MAX_CHAR_LIMIT)
        else:
            text_chunks_for_openai.append(final_text)
        
        # Retry loop for audio validation
        max_validation_retries = 3 if self.enable_audio_validation else 1
        best_attempt_path: Optional[str] = None
        best_silence_ratio = float('inf')
        
        for validation_attempt in range(max_validation_retries):
            segment_audio_files = []
            temp_dir_for_chunks = tempfile.mkdtemp(prefix="openai_chunks_")
            temp_attempt_path = f"{temp_output_path}_attempt_{validation_attempt}.mp3"

            try:
                for i, chunk_text in enumerate(text_chunks_for_openai):
                    chunk_file_path = os.path.join(temp_dir_for_chunks, f"chunk_{i}.mp3")
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            response = self.client.audio.speech.create(
                                model=self.model,
                                voice=voice_name,
                                input=chunk_text,
                                response_format="mp3",
                            )
                            response.write_to_file(chunk_file_path)
                            token_count = self._count_tokens(chunk_text)
                            self._register_usage(usage_tracker, self.model, input_tokens=token_count)
                            segment_audio_files.append(chunk_file_path)
                            if len(text_chunks_for_openai) > 1:
                                logger.debug(f"    OpenAI: Synthesized chunk {i+1}/{len(text_chunks_for_openai)} for {speaker_id}")
                            break
                        except Exception as e_chunk:
                            logger.error(f"    OpenAI: Attempt {attempt + 1}/{max_attempts} for chunk {i+1} failed: {e_chunk}")
                            if attempt + 1 >= max_attempts:
                                raise RuntimeError(f"OpenAI TTS failed for chunk {i+1} after {max_attempts} attempts: {e_chunk}")
                            time.sleep(1.5 ** attempt)
                
                if not segment_audio_files:
                    raise RuntimeError(f"No audio chunks were generated for speaker {speaker_id}.")
                
                # Combine chunks
                if len(segment_audio_files) == 1:
                    shutil.copy(segment_audio_files[0], temp_attempt_path)
                else:
                    combined_chunk_audio = AudioSegment.empty()
                    for audio_file_path in segment_audio_files:
                        combined_chunk_audio += AudioSegment.from_mp3(audio_file_path)
                    combined_chunk_audio.export(temp_attempt_path, format="mp3")
                    logger.debug(f"  OpenAI: Combined {len(segment_audio_files)} chunks for speaker {speaker_id}")

            finally:
                if os.path.exists(temp_dir_for_chunks):
                    shutil.rmtree(temp_dir_for_chunks, ignore_errors=True)

            # Validate audio if enabled
            if self.enable_audio_validation:
                fade_config = {
                    'enabled': self.fade_detection_enabled,
                    'window_size_ms': self.fade_window_size_ms,
                    'min_fade_db': self.min_fade_db,
                    'percentile': self.fade_detection_percentile
                }
                is_valid, reason, silence_ratio = AudioValidator.validate_audio_sample(
                    temp_attempt_path,
                    max_silence_ratio=self.max_silence_ratio,
                    fade_detection_config=fade_config
                )

                # Track best attempt
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
                    logger.debug(f"  OpenAI: Audio validation passed for speaker {speaker_id} (silence ratio: {silence_ratio:.2%})")
                    shutil.copy(best_attempt_path, temp_output_path)
                    if best_attempt_path and os.path.exists(best_attempt_path):
                        try:
                            os.remove(best_attempt_path)
                        except OSError:
                            pass
                    return
                else:
                    logger.warning(f"  OpenAI: Audio validation failed for speaker {speaker_id} (attempt {validation_attempt + 1}/{max_validation_retries}): {reason}")
                    if validation_attempt + 1 < max_validation_retries:
                        time.sleep(1.0)
            else:
                # No validation, just use the generated audio
                shutil.copy(temp_attempt_path, temp_output_path)
                if os.path.exists(temp_attempt_path):
                    try:
                        os.remove(temp_attempt_path)
                    except OSError:
                        pass
                return

        # All validation attempts failed, use the best one
        if best_attempt_path and os.path.exists(best_attempt_path):
            logger.warning(f"  OpenAI: Using best attempt for speaker {speaker_id} with silence ratio {best_silence_ratio:.2%}")
            shutil.copy(best_attempt_path, temp_output_path)
            try:
                os.remove(best_attempt_path)
            except OSError:
                pass
        else:
            raise RuntimeError(f"OpenAI TTS failed to generate valid audio for speaker {speaker_id} after {max_validation_retries} attempts")

    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: Optional[str] = None,
        **kwargs: Any
    ) -> List[SegmentAlignment]:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Call initialize() first.")
        if not segments_data:
            logger.warning("Warning: No segments provided to OpenAITTSWrapper.synthesize.")
            return []
        
        language = language or self.target_language

        assigned_voices: List[str] = [] # Keep track of voices assigned to speakers

        # Auto-pin voices for unmapped speakers with reference audio
        if self.enable_voice_matching and self.voice_matcher:
            # Create a unique list of speakers needing voice pinning
            speakers_to_pin = []
            seen_speakers_for_pinning = set()
            for segment in segments_data:
                if (segment.speaker and 
                    segment.speaker not in self.voice_mapping and 
                    segment.speaker not in seen_speakers_for_pinning and 
                    segment.reference_audio_path):
                    speakers_to_pin.append((segment.speaker, segment.reference_audio_path))
                    seen_speakers_for_pinning.add(segment.speaker)
            
            for speaker_id, ref_path in speakers_to_pin:
                logger.debug(f"Auto-pinning voice for speaker '{speaker_id}'")
                # Pass already assigned voices to exclude them from search
                pinned_voice = self.find_and_pin_voice_for_speaker(speaker_id, ref_path, exclude_voices=assigned_voices)
                if pinned_voice and pinned_voice != self.default_voice:
                    assigned_voices.append(pinned_voice)

        temp_dir = tempfile.mkdtemp(prefix="openai_segments_")
        alignments = []
        usage_tracker: Dict[str, Any] = {"models": {}}

        try:
            for i, segment in enumerate(segments_data):
                segment_file_path = os.path.join(temp_dir, f"segment_{i}_{segment.speaker}.mp3")
                
                # Check if we already have this segment in cache
                cache_key = self._get_cache_key(segment, language)
                if cache_key in self._audio_cache:
                    cached_path, duration = self._audio_cache[cache_key]
                    if os.path.exists(cached_path):
                        logger.debug(f"OpenAI: Using cached segment {i+1}/{len(segments_data)} for speaker '{segment.speaker}'")
                        shutil.copy(cached_path, segment_file_path)
                        
                        # Save to output path if specified
                        if segment.output_path:
                            os.makedirs(os.path.dirname(segment.output_path), exist_ok=True)
                            shutil.copy(segment_file_path, segment.output_path)
                            logger.debug(f"Saved segment audio to {segment.output_path}")
                        
                        # Create alignment using cached duration
                        diarized = DiarizationSegment(
                            start_time=0.0,
                            end_time=duration,
                            speaker=segment.speaker,
                            text=segment.text,
                            confidence=1.0
                        )
                        alignments.append(SegmentAlignment(
                            original_segment=segment,
                            diarized_segment=diarized,
                            alignment_confidence=1.0
                        ))
                        continue
                
                # If not cached, synthesize normally
                logger.info(f"OpenAI: Synthesizing segment {i+1}/{len(segments_data)} for speaker '{segment.speaker}'")
                try:
                    self._synthesize_single_segment(segment, segment_file_path, language, usage_tracker=usage_tracker)
                    
                    # Get duration of the synthesized audio
                    audio_segment = AudioSegment.from_mp3(segment_file_path)
                    duration = len(audio_segment) / 1000.0  # Convert to seconds
                    self._register_usage(usage_tracker, self.model, audio_seconds=duration)
                    
                    # Cache the generated audio
                    if self._cache_dir:
                        cache_path = os.path.join(self._cache_dir, f"{cache_key}.mp3")
                        shutil.copy(segment_file_path, cache_path)
                        self._audio_cache[cache_key] = (cache_path, duration)
                        logger.debug(f"OpenAI: Cached generated audio. Actual duration: {duration:.2f}s")
                    
                    # Save to output path if specified
                    if segment.output_path:
                        # Ensure output directory exists
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
                    alignments.append(SegmentAlignment(
                        original_segment=segment,
                        diarized_segment=diarized,
                        alignment_confidence=1.0
                    ))
                    
                except Exception as e_segment:
                    logger.error(f"Error synthesizing segment {i+1} for speaker '{segment.speaker}': {e_segment}")
                    # Continue with other segments
            
            logger.info(f"OpenAI: Synthesized {len(alignments)} segments successfully")
            if self.cost_tracker:
                models_usage = usage_tracker.get("models", {}) if usage_tracker else {}
                for model_name, metrics in models_usage.items():
                    input_tok = metrics.get("input_tokens", 0.0)
                    output_tok = metrics.get("output_tokens", 0.0)
                    audio_sec = metrics.get("audio_seconds", 0.0)
                    if input_tok or output_tok or audio_sec:
                        self.cost_tracker.add_tts_actual(
                            "openai",
                            model=model_name,
                            input_tokens=input_tok,
                            output_tokens=output_tok,
                            audio_seconds=audio_sec
                        )
            return alignments

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and PYDUB_AVAILABLE and self.client is not None
    
    def get_voice_samples_info(self) -> Dict[str, Any]:
        """Get information about available voice samples and embeddings."""
        info = {
            "samples_directory": str(DEFAULT_SAMPLES_DIR),
            "embeddings_cache": str(EMBEDDING_CACHE_FILE),
            "voice_matching_enabled": self.enable_voice_matching,
            "available_voices": ALL_OPENAI_VOICES,
            "loaded_embeddings": list(self.voice_matcher.sample_embeddings.keys()) if self.voice_matcher else [],
            "voice_mappings": self.voice_mapping
        }
        
        # Check which sample files exist
        existing_samples = []
        if DEFAULT_SAMPLES_DIR.exists():
            for voice in ALL_OPENAI_VOICES:
                sample_path = DEFAULT_SAMPLES_DIR / f"{voice}.mp3"
                if sample_path.exists():
                    existing_samples.append(voice)
        
        info["existing_samples"] = existing_samples
        info["missing_samples"] = [v for v in ALL_OPENAI_VOICES if v not in existing_samples]
        
        return info
    
    def clone_voice(self, audio_path: str, voice_id: str) -> str:
        """Voice cloning is not supported by OpenAI TTS API."""
        raise NotImplementedError("Voice cloning is not supported by OpenAI TTS. Use Minimax TTS for voice cloning capabilities.")
        
    def cleanup(self) -> None:
        # No specific cloud resources to clean other than local temp files handled by synthesize.
        # Clean up cache directory
        if self._cache_dir and os.path.exists(self._cache_dir):
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir = None
        self._audio_cache.clear()
        
        # Clear voice matcher
        if self.voice_matcher:
            self.voice_matcher.clear()
            self.voice_matcher = None
        
        # Clean up AudioEmbedder if needed
        if self.audio_embedder:
            self.audio_embedder = None

    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: Optional[str] = None
    ) -> Optional[float]:
        """
        Estimate audio segment length using unified Gemini duration estimation algorithm.
        
        Args:
            segment_data: TTSSegmentData object containing text and voice parameters
            language: Target language code (uses target_language from init if not specified)
            
        Returns:
            Estimated duration in seconds
        """
        if not segment_data.text or not segment_data.text.strip():
            return 0.0
        
        language = language or self.target_language
        
        # Use VoiceSampleManager for unified estimation if available
        if self.voice_sample_manager:
            voice_name = segment_data.voice or self.voice_mapping.get(segment_data.speaker, self.default_voice)
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
        
        # Fallback if VoiceSampleManager not available
        import re
        words = re.findall(r'\b\w+\b', segment_data.text.lower())
        word_count = len(words)
        base_wpm = 175.0
        estimated_duration = (word_count / base_wpm) * 60
        if segment_data.speed and segment_data.speed > 0:
            estimated_duration /= segment_data.speed
        return max(0.1, estimated_duration) 
