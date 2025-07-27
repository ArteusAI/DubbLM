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
from tts.tts_interface import TTSInterface
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

logger = get_logger(__name__)

# Constants
ALL_OPENAI_VOICES: List[str] = [
    "alloy", 
    "ash", 
    "ballad", 
    "coral", 
    "echo",
    "fable", 
    "onyx", 
    "nova", 
    "sage", 
    "shimmer", 
    "verse"
]

DEFAULT_SAMPLES_DIR = Path("tts/samples/openai")
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
        model: str = "tts-1", # OpenAI models: tts-1, tts-1-hd
        default_voice: str = "alloy", # OpenAI voices: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
        embedding_model_device: Optional[str] = None,
        enable_voice_matching: bool = True,
        **kwargs: Any
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Use 'pip install openai'.")
        if not PYDUB_AVAILABLE:
            raise ImportError("Pydub package not installed. Use 'pip install pydub'.")
            
        self.model = model
        self.default_voice = self._validate_voice_name(default_voice)
        self.client: Optional[OpenAI] = None
        self.voice_mapping: Dict[str, str] = {}
        self.voice_prompt_mapping: Dict[str, str] = {} # OpenAI TTS API v1 doesn't directly use text prompts for voice style like Gemini.
                                                       # Emotion can be hinted in the input text if model supports it implicitly.
        self._audio_cache: Dict[str, tuple[str, float]] = {}  # Maps cache_key to (file_path, duration)
        self._cache_dir = None
        
        # Voice matching components
        self.enable_voice_matching = enable_voice_matching
        self.embedding_model_device = embedding_model_device
        self.audio_embedder: Optional[AudioEmbedder] = None
        self.voice_matcher: Optional[VoiceMatcher] = None

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
                    
                    # Load or generate voice samples and embeddings
                    self._initialize_voice_samples()
                except Exception as e:
                    logger.warning(f"Warning: Failed to initialize voice matching: {e}")
                    self.enable_voice_matching = False
                    self.audio_embedder = None
                    self.voice_matcher = None
            else:
                logger.info("Voice matching disabled by configuration.")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _initialize_voice_samples(self) -> None:
        """Initialize voice samples and embeddings for voice matching."""
        DEFAULT_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing embeddings
        if EMBEDDING_CACHE_FILE.exists():
            try:
                with open(EMBEDDING_CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    # Convert lists back to numpy arrays
                    embeddings = {
                        voice: np.array(embedding) 
                        for voice, embedding in cache_data.items()
                    }
                    if self.voice_matcher:
                        self.voice_matcher.set_sample_embeddings(embeddings)
                    logger.debug(f"Loaded {len(embeddings)} voice embeddings from cache")
                    return
            except Exception as e:
                logger.error(f"Error loading embeddings cache: {e}. Will regenerate.")
        
        # Generate samples and embeddings
        logger.debug("Generating voice samples and embeddings...")
        self._generate_voice_samples()
    
    def _generate_voice_samples(self, force_regenerate: bool = False) -> bool:
        """Generate voice samples and compute embeddings for all OpenAI voices."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized.")
        
        if not self.voice_matcher:
            logger.error("VoiceMatcher not available. Cannot generate embeddings.")
            return False
        
        embeddings = {}
        
        for voice_name in ALL_OPENAI_VOICES:
            sample_path = DEFAULT_SAMPLES_DIR / f"{voice_name}.mp3"
            
            # Generate sample if it doesn't exist or force regenerate
            if force_regenerate or not sample_path.exists():
                logger.debug(f"Generating sample for voice: {voice_name}")
                try:
                    response = self.client.audio.speech.create(
                        model=self.model,
                        voice=voice_name,
                        input=VOICE_SAMPLE_TEXT,
                        response_format="mp3"
                    )
                    response.write_to_file(str(sample_path))
                except Exception as e:
                    logger.error(f"Error generating sample for {voice_name}: {e}")
                    continue
            
            # Extract embedding if sample exists
            if sample_path.exists():
                embedding = self.voice_matcher.extract_embedding_for_audio_file(sample_path)
                if embedding is not None:
                    embeddings[voice_name] = embedding
                    logger.debug(f"Extracted embedding for {voice_name}")
                else:
                    logger.error(f"Failed to extract embedding for {voice_name}")
        
        # Save embeddings to cache
        if embeddings:
            self.voice_matcher.set_sample_embeddings(embeddings)
            
            try:
                # Convert numpy arrays to lists for JSON serialization
                cache_data = {
                    voice: embedding.tolist() 
                    for voice, embedding in embeddings.items()
                }
                with open(EMBEDDING_CACHE_FILE, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                logger.info(f"Saved {len(embeddings)} embeddings to cache")
            except Exception as e:
                logger.error(f"Error saving embeddings cache: {e}")
        
        return True
    
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
        
        logger.info("Regenerating OpenAI voice samples and embeddings...")
        return self._generate_voice_samples(force_regenerate)
    
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
        language: str # OpenAI generally auto-detects language from input text
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
        
        # OpenAI tts-1 doesn't use explicit style prompts. Emotion/style is part of input text.
        # However, if a style_prompt is provided in TTSSegmentData or via global mapping,
        # we can prepend it to the text as a hint, though its effect varies.
        final_text = text_to_synthesize
        # style_hint = segment_data.style_prompt or self.voice_prompt_mapping.get(speaker_id)
        # if style_hint:
        #     final_text = f"{style_hint.strip()} {text_to_synthesize}"
        #     logger.info(f"  OpenAI: Prepending style hint for speaker {speaker_id}: '{style_hint.strip()}'")
        # elif segment_data.emotion and segment_data.emotion != "Neutral":
        #     # Basic emotion hinting by prepending (effect is model-dependent)
        #     final_text = f"(Speaking in a {segment_data.emotion.lower()} tone) {text_to_synthesize}"
        #     logger.info(f"  OpenAI: Prepending emotion hint for speaker {speaker_id}: '{segment_data.emotion.lower()}'")

        # OpenAI API character limit is 4096 for tts-1 models.
        # If text is longer, it needs to be chunked.
        MAX_CHAR_LIMIT = 4096
        text_chunks_for_openai: List[str] = []

        if len(final_text) > MAX_CHAR_LIMIT:
            logger.warning(f"  OpenAI: Text for speaker {speaker_id} ({len(final_text)} chars) exceeds limit. Splitting into chunks.")
            # greedy_sent_split aims for MAX_CHAR_LIMIT per chunk
            text_chunks_for_openai = greedy_sent_split(final_text, MAX_CHAR_LIMIT)
        else:
            text_chunks_for_openai.append(final_text)
        
        segment_audio_files = []
        temp_dir_for_chunks = tempfile.mkdtemp(prefix="openai_chunks_")

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
                            response_format="mp3", # OpenAI supports mp3, opus, aac, flac
                            # speed=segment_data.speed # OpenAI tts-1 supports speed from 0.25 to 4.0
                        )
                        response.write_to_file(chunk_file_path)
                        segment_audio_files.append(chunk_file_path)
                        if len(text_chunks_for_openai) > 1:
                            logger.debug(f"    OpenAI: Synthesized chunk {i+1}/{len(text_chunks_for_openai)} for {speaker_id}")
                        break # Success for this chunk
                    except Exception as e_chunk:
                        logger.error(f"    OpenAI: Attempt {attempt + 1}/{max_attempts} for chunk {i+1} failed: {e_chunk}")
                        if attempt + 1 >= max_attempts:
                            raise RuntimeError(f"OpenAI TTS failed for chunk {i+1} after {max_attempts} attempts: {e_chunk}")
                        time.sleep(1.5 ** attempt)
            
            # Concatenate chunks if multiple were created for this segment
            if not segment_audio_files:
                raise RuntimeError(f"No audio chunks were generated for speaker {speaker_id}.")
            
            if len(segment_audio_files) == 1:
                shutil.copy(segment_audio_files[0], temp_output_path)
            else:
                combined_chunk_audio = AudioSegment.empty()
                for audio_file_path in segment_audio_files:
                    combined_chunk_audio += AudioSegment.from_mp3(audio_file_path)
                combined_chunk_audio.export(temp_output_path, format="mp3")
                logger.debug(f"  OpenAI: Combined {len(segment_audio_files)} chunks for speaker {speaker_id} into {temp_output_path}")

        finally:
            if os.path.exists(temp_dir_for_chunks):
                shutil.rmtree(temp_dir_for_chunks, ignore_errors=True)

    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: str = "en", # Global language hint, OpenAI mostly auto-detects
        **kwargs: Any
    ) -> List[SegmentAlignment]:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Call initialize() first.")
        if not segments_data:
            logger.warning("Warning: No segments provided to OpenAITTSWrapper.synthesize.")
            return []

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
                    self._synthesize_single_segment(segment, segment_file_path, language)
                    
                    # Get duration of the synthesized audio
                    audio_segment = AudioSegment.from_mp3(segment_file_path)
                    duration = len(audio_segment) / 1000.0  # Convert to seconds
                    
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
        language: str = "en"
    ) -> Optional[float]:
        """
        Generate audio for the segment and return its actual duration.
        The generated audio is cached for reuse in synthesize() method.
        
        Args:
            segment_data: TTSSegmentData object containing text and voice parameters
            language: Target language code (e.g., "en")
            
        Returns:
            Actual duration in seconds of the generated audio
        """
        if not segment_data.text or not segment_data.text.strip():
            return 0.0
        
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Call initialize() first.")
        
        # Check cache first
        cache_key = self._get_cache_key(segment_data, language)
        if cache_key in self._audio_cache:
            cached_path, duration = self._audio_cache[cache_key]
            if os.path.exists(cached_path):
                logger.debug(f"OpenAI: Returning cached duration for speaker {segment_data.speaker}: {duration:.2f}s")
                return duration
        
        # Generate audio to get actual duration
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            logger.debug(f"OpenAI: Generating audio to estimate duration for speaker {segment_data.speaker}")
            self._synthesize_single_segment(segment_data, temp_path, language)
            
            # Get actual duration
            audio_segment = AudioSegment.from_mp3(temp_path)
            duration = len(audio_segment) / 1000.0  # Convert to seconds
            
            # Cache the generated audio
            if self._cache_dir:
                cache_path = os.path.join(self._cache_dir, f"{cache_key}.mp3")
                shutil.copy(temp_path, cache_path)
                self._audio_cache[cache_key] = (cache_path, duration)
                logger.debug(f"OpenAI: Cached generated audio. Actual duration: {duration:.2f}s")
            
            return duration
            
        except Exception as e:
            logger.error(f"Error estimating audio length: {e}")
            # Fall back to estimation if generation fails
            import re
            words = re.findall(r'\b\w+\b', segment_data.text.lower())
            word_count = len(words)
            base_wpm = 175.0
            estimated_duration = (word_count / base_wpm) * 60
            if segment_data.speed and segment_data.speed > 0:
                estimated_duration /= segment_data.speed
            return max(0.1, estimated_duration)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path) 