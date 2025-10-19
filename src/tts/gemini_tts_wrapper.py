from typing import Optional, Dict, Any, List, Union, Tuple
import os
import wave
import time
import json
import re
import tempfile
import shutil
import hashlib
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .models import (
    TTSSegmentData, 
    VoiceDurationStats, 
    VoiceDurationDatabase,
    SegmentAlignment,
    DiarizationSegment
)
from tts.tts_interface import TTSInterface
from src.utils.sent_split import greedy_sent_split
from src.utils.audio_embedder import AudioEmbedder
from src.utils.voice_matcher import VoiceMatcher
from pydantic import BaseModel, Field
from src.dubbing.core.log_config import get_logger

# Import Google GenAI dependencies, with error handling for missing packages
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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

logger = get_logger(__name__)

# Constants
ALL_GEMINI_VOICES: List[str] = [
    "Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Aoede",
    "Autonoe", "Callirrhoe", "Charon", "Despina", "Enceladus", "Erinome",
    "Fenrir", "Gacrux", "Iapetus", "Kore", "Laomedeia", "Leda", "Orus",
    "Puck", "Pulcherrima", "Rasalgethi", "Sadachbia", "Sadaltager",
    "Schedar", "Sulafat", "Umbriel", "Vindemiatrix", "Zephyr", "Zubenelgenubi"
]
DEFAULT_SAMPLES_DIR = Path("tts/samples/gemini")

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
MAX_CHAR_LIMIT_PER_REQUEST = 1024*30
SAMPLE_RATE = 24000


class GeminiTTSConfig(BaseModel):
    """Configuration for Gemini TTS."""
    model: str = "gemini-2.5-pro-preview-tts"
    fallback_model: str = "gemini-2.5-flash-preview-tts"
    default_voice: str = "Enceladus"
    embedding_model_device: Optional[str] = None
    enable_voice_matching: bool = True
    max_retries: int = 10
    retry_delay_base: float = 2.0
    prompt_prefix: str = "Read aloud in a calm, articulate manner with natural pacing and avoid unnecessary emotions or dramatic emphasis:"
    enable_audio_validation: bool = True  # Allow disabling validation for debugging
    enable_emotion_enrichment: bool = False  # Enable emotion enrichment using LLM
    emotion_enrichment_model: str = "gemini-2.5-pro"  # Model for emotion enrichment
    emotion_enrichment_temperature: float = 0.7  # Temperature for emotion enrichment
    max_workers: int = 4  # Maximum number of parallel workers for segment synthesis
    duration_smoothing_alpha: float = 0.35  # EMA smoothing factor for runtime stats
    duration_stats_auto_save: bool = True  # Persist runtime stats automatically
    duration_stats_save_interval: int = 20  # Save after this many updates


class AudioFileUtils:
    """Utility class for audio file operations."""
    
    @staticmethod
    def save_wave_file(filename: str, pcm_data: bytes, channels: int = 1, 
                      rate: int = SAMPLE_RATE, sample_width: int = 2) -> None:
        """Saves PCM audio data to a WAV file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)

    @staticmethod
    def get_audio_duration_seconds(audio_path: Union[str, Path]) -> Optional[float]:
        """Gets the duration of an audio file in seconds."""
        try:
            if PYTORCH_AVAILABLE and hasattr(torchaudio, 'info'):
                info = torchaudio.info(str(audio_path))
                return info.num_frames / info.sample_rate
            elif PYTORCH_AVAILABLE and 'AudioSegment' in globals():
                audio_segment = AudioSegment.from_file(str(audio_path))
                return audio_segment.duration_seconds
            else:
                logger.warning("Warning: Cannot determine audio duration. torchaudio.info or pydub not fully available.")
                return None
        except Exception as e:
            logger.error(f"Error getting duration for {audio_path}: {e}")
            return None


class AudioValidator:
    """Validates the quality of generated audio samples."""
    
    @staticmethod
    def validate_audio_sample(audio_path: Union[str, Path], 
                            expected_min_duration: float = 1.0,
                            silence_threshold_db: float = -40.0,
                            max_silence_ratio: float = 0.03) -> tuple[bool, str, float]:
        """
        Validate audio by checking only trailing silence at the end of the segment.
        
        Args:
            audio_path: Path to the audio file
            expected_min_duration: Minimum expected duration in seconds
            silence_threshold_db: Threshold below which audio is considered silence (in dB)
            max_silence_ratio: Maximum allowed ratio of trailing silence vs total duration (0.1 = 10%)
            
        Returns:
            Tuple of (is_valid, reason, trailing_silence_ratio)
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                return False, "Audio file does not exist", 1.0
            
            # Check file size (empty or very small files are bad)
            file_size = audio_path.stat().st_size
            if file_size < 1000:  # Less than 1KB is likely empty
                return False, f"Audio file too small ({file_size} bytes)", 1.0
            
            # Get duration
            duration = AudioFileUtils.get_audio_duration_seconds(audio_path)
            if duration is None:
                return False, "Could not determine audio duration", 1.0
            
            if duration < expected_min_duration:
                return False, f"Audio too short ({duration:.2f}s < {expected_min_duration:.2f}s)", 1.0
            
            # Analyze audio content for silence
            if LIBROSA_AVAILABLE:
                return AudioValidator._validate_with_librosa(audio_path, silence_threshold_db, max_silence_ratio)
            elif PYTORCH_AVAILABLE:
                return AudioValidator._validate_with_pytorch(audio_path, silence_threshold_db, max_silence_ratio)
            else:
                # Fallback: just check duration and file size
                logger.warning("Advanced audio validation not available. Using basic checks only.")
                return True, "Basic validation passed (advanced libraries not available)", 0.0
        
        except Exception as e:
            logger.error(f"Error validating audio sample {audio_path}: {e}")
            return False, f"Validation error: {str(e)}", 1.0
    
    @staticmethod
    def _validate_with_librosa(audio_path: Path, silence_threshold_db: float, 
                              max_silence_ratio: float) -> tuple[bool, str, float]:
        """Validate audio using librosa."""
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=None)
            
            if len(y) == 0:
                return False, "Audio file contains no data", 1.0
            
            # Check for completely flat audio first (all zeros or constant value)
            if np.std(y) < 1e-6:
                return False, "Audio appears to be flat/constant", 1.0
            
            # Use RMS energy for better silence detection
            # Calculate RMS energy in small windows (100ms)
            hop_length = int(sr * 0.1)  # 100ms windows
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            
            # Convert RMS to dB, use dynamic reference based on the audio
            # Use 90th percentile as reference instead of max to avoid outliers
            ref_level = np.percentile(rms, 90)
            if ref_level == 0:
                ref_level = np.max(rms)
            if ref_level == 0:
                return False, "Audio contains no energy", 1.0
                
            rms_db = librosa.amplitude_to_db(rms, ref=ref_level)
            
            # Adaptive silence threshold based on dynamic range
            dynamic_range = np.max(rms_db) - np.min(rms_db)
            if dynamic_range < 6:  # Less than 6dB dynamic range suggests poor audio
                adaptive_threshold = np.min(rms_db) + 1  # Very lenient
            else:
                # Use threshold relative to the dynamic range
                adaptive_threshold = max(silence_threshold_db, np.min(rms_db) + dynamic_range * 0.1)
            
            # Compute trailing silence ratio only
            total_frames = len(rms_db)
            if total_frames == 0:
                return False, "Audio contains no analyzable frames", 1.0
            silence_mask = rms_db < adaptive_threshold
            non_silent_indices = np.where(~silence_mask)[0]
            if non_silent_indices.size == 0:
                trailing_silence_ratio = 1.0
            else:
                last_non_silent = int(non_silent_indices[-1])
                trailing_silent_frames = max(0, total_frames - (last_non_silent + 1))
                trailing_silence_ratio = trailing_silent_frames / total_frames
            
            # Debug info
            logger.debug(f"Audio analysis - Dynamic range: {dynamic_range:.1f}dB, "
                        f"Adaptive threshold: {adaptive_threshold:.1f}dB, "
                        f"Trailing silence ratio: {trailing_silence_ratio:.2%}")
            
            if trailing_silence_ratio > max_silence_ratio:
                return False, f"Too much trailing silence ({trailing_silence_ratio:.2%} > {max_silence_ratio:.2%}, threshold: {adaptive_threshold:.1f}dB)", trailing_silence_ratio
            
            return True, f"Audio validation passed (trailing silence: {trailing_silence_ratio:.2%}, dynamic range: {dynamic_range:.1f}dB)", trailing_silence_ratio
            
        except Exception as e:
            return False, f"Librosa validation error: {str(e)}", 1.0
    
    @staticmethod
    def _validate_with_pytorch(audio_path: Path, silence_threshold_db: float, 
                              max_silence_ratio: float) -> tuple[bool, str, float]:
        """Validate audio using PyTorch/torchaudio."""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            if waveform.numel() == 0:
                return False, "Audio file contains no data", 1.0
            
            # Get first channel if stereo
            if waveform.shape[0] > 1:
                waveform = waveform[0:1]
            
            # Check for completely flat audio first
            if torch.std(waveform) < 1e-6:
                return False, "Audio appears to be flat/constant", 1.0
            
            # For frame-by-frame analysis, split into chunks (100ms)
            chunk_size = sample_rate // 10  # 0.1 second chunks
            num_chunks = waveform.shape[1] // chunk_size
            
            if num_chunks == 0:
                return False, "Audio too short for chunk analysis", 1.0
            
            # Calculate RMS for each chunk and get dynamic range
            chunk_rms_values = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, waveform.shape[1])
                chunk = waveform[:, start_idx:end_idx]
                
                chunk_rms = torch.sqrt(torch.mean(chunk ** 2))
                chunk_rms_values.append(chunk_rms.item())
            
            chunk_rms_tensor = torch.tensor(chunk_rms_values)
            
            # Use 90th percentile as reference to avoid outliers
            ref_level = torch.quantile(chunk_rms_tensor, 0.9).item()
            if ref_level == 0:
                ref_level = torch.max(chunk_rms_tensor).item()
            if ref_level == 0:
                return False, "Audio contains no energy", 1.0
            
            # Convert to dB
            chunk_db_values = 20 * torch.log10(chunk_rms_tensor + 1e-8) - 20 * torch.log10(torch.tensor(ref_level))
            
            # Adaptive silence threshold based on dynamic range
            dynamic_range = (torch.max(chunk_db_values) - torch.min(chunk_db_values)).item()
            if dynamic_range < 6:  # Less than 6dB dynamic range suggests poor audio
                adaptive_threshold = torch.min(chunk_db_values).item() + 1  # Very lenient
            else:
                # Use threshold relative to the dynamic range
                adaptive_threshold = max(silence_threshold_db, torch.min(chunk_db_values).item() + dynamic_range * 0.1)
            
            # Compute trailing silence ratio only
            silence_mask = (chunk_db_values < adaptive_threshold)
            non_silent_indices = torch.nonzero(~silence_mask, as_tuple=False).flatten()
            if non_silent_indices.numel() == 0:
                trailing_silence_ratio = 1.0
            else:
                last_non_silent = int(non_silent_indices[-1].item())
                trailing_silent_chunks = max(0, num_chunks - (last_non_silent + 1))
                trailing_silence_ratio = trailing_silent_chunks / num_chunks
            
            # Debug info
            logger.debug(f"Audio analysis - Dynamic range: {dynamic_range:.1f}dB, "
                        f"Adaptive threshold: {adaptive_threshold:.1f}dB, "
                        f"Trailing silence ratio: {trailing_silence_ratio:.2%}")
            
            if trailing_silence_ratio > max_silence_ratio:
                return False, f"Too much trailing silence ({trailing_silence_ratio:.2%} > {max_silence_ratio:.2%}, threshold: {adaptive_threshold:.1f}dB)", trailing_silence_ratio
            
            return True, f"Audio validation passed (trailing silence: {trailing_silence_ratio:.2%}, dynamic range: {dynamic_range:.1f}dB)", trailing_silence_ratio
            
        except Exception as e:
            return False, f"PyTorch validation error: {str(e)}", 1.0


class TextAnalysisUtils:
    """Utility class for text analysis operations."""
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text, handling punctuation and multiple spaces."""
        words = re.findall(r'\b\w+\b', text.lower())
        return len(words)
    
    @staticmethod
    def count_characters(text: str, include_spaces: bool = True) -> int:
        """Count characters in text."""
        if include_spaces:
            return len(text)
        else:
            return len(re.sub(r'\s', '', text))
    
    @staticmethod
    def estimate_speech_complexity(text: str) -> float:
        """
        Estimate speech complexity factor (1.0 = normal, >1.0 = more complex/slower).
        Factors: punctuation density, word length, sentence structure.
        """
        if not text.strip():
            return 1.0
        
        # Count punctuation marks that typically cause pauses
        pause_punctuation = re.findall(r'[.!?,:;…]', text)
        punctuation_density = len(pause_punctuation) / len(text)
        
        # Average word length
        words = re.findall(r'\b\w+\b', text)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Sentence count
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Calculate complexity factor
        complexity = 1.0
        complexity += punctuation_density * 0.5  # Punctuation adds pauses
        complexity += max(0, (avg_word_length - 5) * 0.02)  # Longer words slow down speech
        complexity += max(0, (avg_sentence_length - 15) * 0.01)  # Long sentences may be read slower
        
        return min(complexity, 2.0)  # Cap at 2x normal complexity


class GeminiAPIClient:
    """Handles Gemini API communication."""
    
    def __init__(self, config: GeminiTTSConfig):
        self.config = config
        self.client: Optional[genai.Client] = None
        self.current_model = config.model
        self.fallback_model = config.fallback_model
        # Indicates whether we've permanently switched to the fallback model due to quota limits
        self.permanent_fallback = False

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

    def switch_to_fallback_model(self) -> bool:
        """Switch to fallback model for generation."""
        if self.current_model != self.fallback_model:
            old_model = self.current_model
            self.current_model = self.fallback_model
            logger.debug(f"Switched from model {old_model} to fallback model {self.current_model}")
            return True
        return False

    def reset_to_original_model(self) -> None:
        """Reset to the originally configured model unless we've permanently fallen back."""
        if self.permanent_fallback:
            # Do not reset if we've permanently switched due to quota exhaustion
            logger.debug("Permanent fallback active – not resetting to original model.")
            return
        self.current_model = self.config.model
        logger.debug(f"Reset to original model: {self.current_model}")

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
                        if (hasattr(part, 'inline_data') and 
                            part.inline_data.mime_type == "audio/L16;codec=pcm;rate=24000"):
                            return part.inline_data.data
                
                # Extract text without prompt for logging
                log_text = content.split('\n', 1)[-1] if '\n' in content else content
                log_preview = f"{log_text[:30]}...{log_text[-30:]}" if len(log_text) > 70 else log_text
                logger.warning(f"Attempt {attempt + 1}/{self.config.max_retries}: No audio data in response for text: {log_preview}")
                if attempt + 1 >= self.config.max_retries:
                    logger.error(f"Gemini API call failed after {self.config.max_retries} attempts.")
                    return b''

            except Exception as e:
                # Detect quota exhaustion errors and switch to fallback model permanently
                err_msg = str(e)
                logger.error(f"Attempt {attempt + 1}/{self.config.max_retries} failed: {err_msg}")
                if ("RESOURCE_EXHAUSTED" in err_msg) or ("429" in err_msg):
                    self.config.model = self.fallback_model
                    self.permanent_fallback = True
                    if self.switch_to_fallback_model():
                        # Retry immediately with fallback model
                        logger.info("Retrying with fallback model after quota exhaustion.")
                        continue
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
        return """You are an expert at enriching text with emotional markup tags for text-to-speech synthesis.

Your task is to analyze the given text IN CONTEXT of the conversation and add appropriate markup tags to make the speech sound more natural and emotionally expressive.

Available markup tags:
1. Non-speech sounds: [sigh], [laughing], [chuckle], [uhm], [gasp], [clearing throat]
2. Style modifiers: [sarcasm], [robotic], [shouting], [whispering], [excited], [calm]
3. Vocalized emotions: [scared], [curious], [bored], [angry], [happy], [sad], [surprised]
4. Pacing: [short pause], [medium pause], [long pause]

Guidelines:
- Use tags sparingly and only where they genuinely enhance the delivery
- Consider the conversational context to understand the emotional tone
- Place style modifiers at the beginning of relevant phrases
- Use pauses for natural rhythm and emphasis
- Non-speech sounds should feel natural and contextually appropriate
- DO NOT overuse tags - subtlety is key
- Preserve the original text exactly, only add markup tags where appropriate

{context_section}

Current text to enrich:
{text}

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


class SampleManager:
    """Manages voice sample generation, duration analysis, and embedding computation."""
    
    def __init__(self, api_client: GeminiAPIClient, voice_matcher: VoiceMatcher):
        self.api_client = api_client
        self.voice_matcher = voice_matcher
        self.duration_database = VoiceDurationDatabase()
    
    def save_duration_stats(self, preserve_embeddings: bool = True) -> None:
        """Persist current duration statistics (and optionally embeddings) to disk."""
        try:
            embeddings_data: Dict[str, Any] = {}
            if preserve_embeddings and DURATION_STATS_FILE.exists():
                try:
                    with open(DURATION_STATS_FILE, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        embeddings_data = existing_data.get("embeddings", {})
                except Exception as load_exc:
                    logger.warning(f"Could not load existing duration stats for merge: {load_exc}")

            combined_data = {
                "voice_stats": self.duration_database.dict(),
                "embeddings": embeddings_data
            }

            DURATION_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(DURATION_STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved runtime duration stats for {len(self.duration_database.voice_stats)} voices")
        except Exception as save_exc:
            logger.error(f"Error saving runtime duration stats: {save_exc}")

    def generate_sample_with_validation(self, voice_name: str, sample_file_path: Path,
                                      max_retries_per_model: int = 3) -> bool:
        """
        Generate a single voice sample with validation and model fallback.
        
        Args:
            voice_name: Name of the voice to generate sample for
            sample_file_path: Path where to save the sample
            max_retries_per_model: Maximum retries per model before fallback
            
        Returns:
            True if successful, False otherwise
        """
        if not self.api_client.client:
            raise RuntimeError("Gemini client not initialized.")
        
        # Ensure the API client is using the original model
        self.api_client.reset_to_original_model()
        original_model = self.api_client.current_model
        
        # Try with original model
        success = self._attempt_sample_generation(voice_name, sample_file_path, max_retries_per_model, max_silence_ratio=0.1)
        
        if not success:
            # Switch to fallback model and try again
            if self.api_client.switch_to_fallback_model():
                logger.info(f"Attempting sample generation for {voice_name} with fallback model")
                success = self._attempt_sample_generation(voice_name, sample_file_path, max_retries_per_model, max_silence_ratio=0.2)
                
                # Reset to original model after attempts
                self.api_client.reset_to_original_model()
        
        return success

    def _attempt_sample_generation(self, voice_name: str, sample_file_path: Path,
                                 max_retries: int, max_silence_ratio: float = 0.1) -> bool:
        """
        Attempt sample generation with the current model.
        
        Args:
            voice_name: Name of the voice
            sample_file_path: Path to save the sample
            max_retries: Maximum number of attempts
            
        Returns:
            True if successful, False otherwise
        """
        speech_config = SpeechConfigBuilder.build_single_speaker_config(voice_name)
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Generating sample for {voice_name} (attempt {attempt + 1}/{max_retries}) with model {self.api_client.current_model}")
                
                # Generate audio
                audio_data = self.api_client.synthesize_chunk(DURATION_SAMPLE_TEXT, speech_config)
                
                if not audio_data:
                    logger.warning(f"No audio data received for {voice_name} (attempt {attempt + 1})")
                    continue
                
                # Save to file
                AudioFileUtils.save_wave_file(str(sample_file_path), audio_data)
                
                # Validate the generated sample (if validation is enabled)
                if self.api_client.config.enable_audio_validation:
                    is_valid, reason, silence_ratio = AudioValidator.validate_audio_sample(
                        sample_file_path,
                        expected_min_duration=1.0,  # Expect at least 5 seconds for our sample text
                        silence_threshold_db=-40.0,
                        max_silence_ratio=max_silence_ratio
                    )
                else:
                    # Skip validation - just check if file exists and has reasonable size
                    file_size = sample_file_path.stat().st_size if sample_file_path.exists() else 0
                    is_valid = file_size > 10000  # At least 10KB
                    reason = f"Validation disabled, file size: {file_size} bytes"
                    silence_ratio = 0.0 # No silence ratio if validation is off
                
                if is_valid:
                    logger.debug(f"Generated valid sample for {voice_name}: {reason}")
                    return True
                else:
                    logger.warning(f"Invalid sample for {voice_name} (attempt {attempt + 1}): {reason}")
                    # Remove invalid file
                    if sample_file_path.exists():
                        sample_file_path.unlink()
                    
                    # If this is not the last attempt, continue to retry
                    if attempt + 1 < max_retries:
                        time.sleep(1.0)  # Brief delay before retry
                        continue
                    
            except Exception as e:
                logger.error(f"Error generating sample for {voice_name} (attempt {attempt + 1}): {e}")
                # Remove potentially corrupted file
                if sample_file_path.exists():
                    try:
                        sample_file_path.unlink()
                    except:
                        pass
                
                # If this is not the last attempt, continue to retry
                if attempt + 1 < max_retries:
                    time.sleep(1.0)  # Brief delay before retry
                    continue
        
        logger.error(f"Failed to generate valid sample for {voice_name} after {max_retries} attempts with model {self.api_client.current_model}")
        return False

    def generate_all_samples(self, force_regenerate: bool = False) -> bool:
        """
        Generate all samples, analyze durations, and compute embeddings in one process.
        
        Args:
            force_regenerate: Force regeneration of all samples and stats
            
        Returns:
            True if successful, False otherwise
        """
        if not self.api_client.client:
            raise RuntimeError("Gemini client not initialized.")

        logger.info("Starting comprehensive sample generation and analysis...")
        DEFAULT_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load existing stats if available and not forcing regeneration
        stats_loaded = False
        existing_embeddings: Dict[str, np.ndarray] = {}
        if not force_regenerate and DURATION_STATS_FILE.exists():
            try:
                with open(DURATION_STATS_FILE, 'r', encoding='utf-8') as f:
                    combined_data = json.load(f)
                    if 'voice_stats' in combined_data:
                        self.duration_database = VoiceDurationDatabase(**combined_data['voice_stats'])
                        logger.debug(f"Loaded existing duration statistics for {len(self.duration_database.voice_stats)} voices")
                        stats_loaded = True
                    if 'embeddings' in combined_data:
                        # Convert lists back to numpy arrays
                        existing_embeddings = {
                            voice_name: np.array(embedding_list) 
                            for voice_name, embedding_list in combined_data['embeddings'].items()
                        }
                        logger.debug(f"Loaded {len(existing_embeddings)} voice embeddings from cache")
            except Exception as e:
                logger.error(f"Error loading combined data: {e}. Will regenerate.")

        # Generate samples and analyze durations
        voices_processed = 0
        for voice_name in ALL_GEMINI_VOICES:
            sample_file_path = DEFAULT_SAMPLES_DIR / f"{voice_name}.wav"
            voice_stats = self.duration_database.get_or_create_stats(voice_name)
            
            # Check if we need to generate/analyze samples for this voice
            has_audio_file = sample_file_path.exists()
            has_stats = stats_loaded and voice_stats.total_samples > 0
            
            if force_regenerate or (not has_audio_file and not has_stats):
                # Need to generate new audio file with validation and retry logic
                logger.debug(f"Generating audio sample for {voice_name}")
                try:
                    success = self.generate_sample_with_validation(voice_name, sample_file_path)
                    if success:
                        has_audio_file = True
                        logger.debug(f"Generated valid audio sample for {voice_name}")
                    else:
                        logger.error(f"Failed to generate valid sample for {voice_name} after all retry attempts")
                        
                except Exception as e:
                    logger.error(f"Error generating sample for {voice_name}: {e}")
            
            # Analyze duration if we have audio but no stats (or forcing regeneration)
            if has_audio_file and (force_regenerate or not has_stats):
                logger.debug(f"Analyzing duration for existing {voice_name} sample")
                try:
                    # Reset stats for this voice if regenerating or no stats
                    voice_stats.total_samples = 0
                    voice_stats.total_words = 0
                    voice_stats.total_characters = 0
                    voice_stats.total_duration_seconds = 0.0
                    
                    # Analyze duration from existing file
                    duration = AudioFileUtils.get_audio_duration_seconds(sample_file_path)
                    if duration:
                        words = TextAnalysisUtils.count_words(DURATION_SAMPLE_TEXT)
                        characters = TextAnalysisUtils.count_characters(DURATION_SAMPLE_TEXT)
                        
                        # Update statistics
                        voice_stats.update_stats(words, characters, duration)
                        
                        logger.debug(f"Analyzed {voice_name}: {voice_stats.words_per_minute:.1f} WPM, {voice_stats.characters_per_second:.1f} CPS")
                    else:
                        logger.warning(f"Could not determine duration for {voice_name}")
                        
                except Exception as e:
                    logger.error(f"Error analyzing sample for {voice_name}: {e}")
            
            voices_processed += 1
        
        # Prepare duration statistics for saving (will save combined data later)
        logger.debug(f"Processed duration statistics for {voices_processed} voices")
        
        # Generate embeddings if voice matching is enabled
        if self.voice_matcher.enable_matching and self.voice_matcher.audio_embedder:
            logger.debug("Processing voice embeddings...")
            
            # Start with existing embeddings
            current_embeddings = existing_embeddings.copy()
            
            # Compute embeddings for voices that need them
            voices_needing_embeddings = []
            if force_regenerate:
                voices_needing_embeddings = ALL_GEMINI_VOICES
                current_embeddings = {}  # Clear existing if forcing regeneration
                logger.debug("Force regenerating all embeddings")
            else:
                for voice_name in ALL_GEMINI_VOICES:
                    if voice_name not in current_embeddings:
                        voices_needing_embeddings.append(voice_name)
                
                if voices_needing_embeddings:
                    logger.debug(f"Computing embeddings for {len(voices_needing_embeddings)} missing voice(s): {voices_needing_embeddings}")
                else:
                    logger.debug("All voice embeddings already available")
            
            if voices_needing_embeddings:
                for voice_name in voices_needing_embeddings:
                    sample_file_path = DEFAULT_SAMPLES_DIR / f"{voice_name}.wav"
                    if sample_file_path.exists():
                        embedding = self.voice_matcher.extract_embedding_for_audio_file(sample_file_path)
                        if embedding is not None:
                            current_embeddings[voice_name] = embedding
                    else:
                        logger.warning(f"No audio file found for {voice_name}, cannot compute embedding")
            
            self.voice_matcher.set_sample_embeddings(current_embeddings)
        else:
            # No embeddings support, use empty dict
            current_embeddings: Dict[str, np.ndarray] = {}
        
        # Save combined data (duration stats and embeddings)
        try:
            combined_data = {
                "voice_stats": self.duration_database.dict(),
                "embeddings": {}
            }
            
            # Add embeddings if available
            if current_embeddings:
                # Convert numpy arrays to lists for JSON serialization
                combined_data["embeddings"] = {
                    voice_name: embedding.tolist() 
                    for voice_name, embedding in current_embeddings.items()
                }
            
            with open(DURATION_STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            
            stats_count = len(self.duration_database.voice_stats)
            embeddings_count = len(current_embeddings)
            logger.info(f"Saved combined data: {stats_count} voice stats, {embeddings_count} embeddings")
            
        except Exception as e:
            logger.error(f"Error saving combined data: {e}")
            return False
        
        logger.info("Sample analysis completed successfully!")
        return True


class GeminiTTSWrapper(TTSInterface):
    """Google Gemini TTS wrapper with simplified single-segment synthesis."""

    def __init__(
        self,
        model: str = "gemini-2.5-pro-preview-tts",
        fallback_model: str = "gemini-2.5-flash-preview-tts",
        default_voice: str = "Kore",
        embedding_model_device: Optional[str] = None,
        enable_voice_matching: bool = True,
        enable_audio_validation: bool = True,
        prompt_prefix: Optional[str] = None,
        debug_tts: bool = False,
        enable_emotion_enrichment: bool = False,
        emotion_enrichment_model: Optional[str] = None,
        emotion_enrichment_temperature: Optional[float] = None,
        max_workers: Optional[int] = None,
        duration_smoothing_alpha: Optional[float] = None,
        duration_stats_auto_save: Optional[bool] = None,
        duration_stats_save_interval: Optional[int] = None,
        cost_tracker: Optional[Any] = None,
    ):
        """Initialize Gemini TTS wrapper."""
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
            "enable_emotion_enrichment": enable_emotion_enrichment,
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

        self.config = GeminiTTSConfig(**config_kwargs)
        # Save rejected/silent attempts when debugging is enabled
        self.debug_save_rejected: bool = debug_tts

        # Initialize components
        self.api_client = GeminiAPIClient(self.config)

        audio_embedder = None
        if enable_voice_matching:
            audio_embedder = AudioEmbedder(device=embedding_model_device)

        self.voice_matcher = VoiceMatcher(audio_embedder, enable_voice_matching)
        self.sample_manager = SampleManager(self.api_client, self.voice_matcher)

        # Emotion enricher (will be initialized after API client is ready)
        self.emotion_enricher: Optional[EmotionEnricher] = None

        # Voice mappings
        self.voice_mapping: Dict[str, str] = {}
        self.voice_prompt_mapping: Dict[str, str] = {}

        # Audio cache similar to OpenAI
        self._audio_cache: Dict[str, tuple[str, float]] = {}  # Maps cache_key to (file_path, duration)
        self._cache_dir = None
        self._duration_stats_lock = threading.Lock()
        self._stats_update_counter = 0
        self.cost_tracker = cost_tracker

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

        # Initialize duration analysis
        try:
            logger.info("Initializing voice duration analysis...")
            success = self.sample_manager.generate_all_samples()
            if success:
                logger.info("Duration analysis initialized successfully.")
            else:
                logger.warning("Warning: Duration analysis initialization failed. Using default estimates.")
        except Exception as e:
            logger.error(f"Error during duration analysis initialization: {e}")

    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: str = "en"
    ) -> Optional[float]:
        """
        Estimate the duration in seconds for a given text segment.

        Args:
            segment_data: TTSSegmentData object containing text and voice parameters
            language: Target language code (e.g., "en")

        Returns:
            Estimated duration in seconds, or None if estimation is not possible
        """
        if not segment_data.text or not segment_data.text.strip():
            return 0.0

        # Determine the voice to use
        voice_name = segment_data.voice
        if not voice_name and segment_data.speaker:
            voice_name = self.voice_mapping.get(segment_data.speaker)
        if not voice_name:
            voice_name = self.config.default_voice

        voice_name = self._validate_voice_name(voice_name)

        # Get voice statistics
        voice_stats = self.sample_manager.duration_database.get_or_create_stats(voice_name)

        # Analyze text
        text = segment_data.text.strip()
        word_count = TextAnalysisUtils.count_words(text)
        char_count = TextAnalysisUtils.count_characters(text)
        complexity_factor = TextAnalysisUtils.estimate_speech_complexity(text)

        logger.debug(
            f"Length estimation for voice '{voice_name}': {word_count} words, {char_count} chars, "
            f"complexity={complexity_factor:.2f}, stats_samples={voice_stats.total_samples}"
        )

        # Estimate using both word-based and character-based methods
        if voice_stats.words_per_minute > 0:
            word_based_duration = (word_count / voice_stats.words_per_minute) * 60
        else:
            word_based_duration = (word_count / 150.0) * 60  # Fallback WPM

        if voice_stats.characters_per_second > 0:
            char_based_duration = char_count / voice_stats.characters_per_second
        else:
            char_based_duration = char_count / 12.5  # Fallback CPS

        # Use the average of both methods if we have data, otherwise use word-based
        if voice_stats.total_samples > 0:  # We have sample data
            estimated_duration = (word_based_duration + char_based_duration) / 2
            logger.debug(
                f"  Using avg: word_based={word_based_duration:.2f}s (WPM={voice_stats.words_per_minute:.1f}), "
                f"char_based={char_based_duration:.2f}s (CPS={voice_stats.characters_per_second:.1f})"
            )
        else:  # No data, use fallback
            estimated_duration = word_based_duration
            logger.debug(
                f"  Using fallback: word_based={word_based_duration:.2f}s (no stats available)"
            )

        # Apply complexity factor
        estimated_duration *= complexity_factor

        # Apply speed factor if specified
        if segment_data.speed and segment_data.speed > 0:
            estimated_duration /= segment_data.speed
            logger.debug(f"  Applied speed factor {segment_data.speed:.2f}")

        logger.debug(f"  Final estimated duration: {estimated_duration:.2f}s")
        return max(0.1, estimated_duration)  # Minimum 0.1 seconds

    def get_voice_duration_stats(self, voice_name: Optional[str] = None) -> Dict[str, Any]:
        """Get duration statistics for a specific voice or all voices."""
        if voice_name:
            voice_name = self._validate_voice_name(voice_name)
            stats = self.sample_manager.duration_database.get_or_create_stats(voice_name)
            return {
                "voice_name": stats.voice_name,
                "words_per_minute": stats.words_per_minute,
                "characters_per_second": stats.characters_per_second,
                "total_samples": stats.total_samples,
                "has_data": stats.total_samples > 0
            }
        else:
            return {
                voice_name: {
                    "words_per_minute": stats.words_per_minute,
                    "characters_per_second": stats.characters_per_second,
                    "total_samples": stats.total_samples,
                    "has_data": stats.total_samples > 0
                }
                for voice_name, stats in self.sample_manager.duration_database.voice_stats.items()
            }

    def regenerate_duration_analysis(self, force_regenerate: bool = True) -> bool:
        """Regenerate duration analysis samples and statistics."""
        if not self.is_available():
            raise RuntimeError("Gemini TTS not initialized.")
        
        logger.info("Regenerating duration analysis...")
        return self.sample_manager.generate_all_samples(force_regenerate)

    def generate_voice_samples(self, force_regenerate: bool = False) -> bool:
        """Generate voice samples, analyze durations, and compute embeddings."""
        return self.sample_manager.generate_all_samples(force_regenerate)

    def _validate_voice_name(self, voice_name: str) -> str:
        """Validate that a voice name is supported by Gemini TTS."""
        normalized_voices = {v.lower(): v for v in ALL_GEMINI_VOICES}
        
        if voice_name.lower() in normalized_voices:
            return normalized_voices[voice_name.lower()]
        
        logger.warning(f"Warning: Voice '{voice_name}' not supported. Using default '{self.config.default_voice}'.")
        return self.config.default_voice

    def _get_style_prompt_for_speaker(self, speaker_id: str, 
                                    segment_hint: Optional[TTSSegmentData] = None) -> str:
        """Determine the style prompt for a speaker."""
        style_prompt = self.voice_prompt_mapping.get(speaker_id, "")
        
        if not style_prompt and segment_hint:
            style_prompt = segment_hint.style_prompt
            if not style_prompt and segment_hint.emotion and segment_hint.emotion != "Neutral":
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

        if not self.config.enable_voice_matching:
            return self.voice_mapping.get(speaker_id, self.config.default_voice)

        if not force_search and speaker_id in self.voice_mapping:
            return self.voice_mapping[speaker_id]

        if not self.voice_matcher.audio_embedder:
            return self._validate_voice_name(self.config.default_voice)

        ref_path = Path(reference_audio_path)
        if not ref_path.exists():
            logger.warning(f"Reference audio file not found: {reference_audio_path}")
            return self._validate_voice_name(self.config.default_voice)

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
            return self._validate_voice_name(self.config.default_voice)
        
        logger.debug(f"Extracted {len(reference_embeddings)} embeddings from reference audio (duration: {ref_duration:.1f}s)")

        # Find best matching voice using voting across multiple segments
        # Exclude voices that are already pinned to other speakers to prevent duplicates
        exclude_voices = list(self.voice_mapping.values())
        best_match_voice = self.voice_matcher.find_best_matching_voice_multi_segment(
            reference_embeddings,
            exclude_voices=exclude_voices
        )

        if best_match_voice:
            best_match_voice = self._validate_voice_name(best_match_voice)
            
            if can_be_pinned:
                self.voice_mapping[speaker_id] = best_match_voice
                logger.info(f"Matched and pinned by similarity speaker '{speaker_id}' to voice '{best_match_voice}'")
            
            return best_match_voice

        logger.warning(f"Could not find matching voice for speaker '{speaker_id}'. Using default.")
        return self._validate_voice_name(self.config.default_voice)

    def _get_voice_for_speaker(self, speaker_id: str, segment_hint: TTSSegmentData) -> str:
        """Get voice name for a speaker."""
        return (segment_hint.voice or 
                self.voice_mapping.get(speaker_id) or 
                self.config.default_voice)

    def _resolve_voice_for_segment(self, segment_data: TTSSegmentData) -> str:
        """Determine the validated voice name that will be used for the segment."""
        speaker_id = segment_data.speaker or ""
        raw_voice = (
            segment_data.voice or
            self.voice_mapping.get(speaker_id) or
            self.config.default_voice
        )
        return self._validate_voice_name(raw_voice)

    def _record_duration_stats(
        self,
        segment: TTSSegmentData,
        synthesized_text: Optional[str],
        duration_seconds: float
    ) -> None:
        """Update duration statistics with real synthesis data."""
        if duration_seconds <= 0.05:
            return

        text_for_stats = (synthesized_text or segment.text or "").strip()
        if not text_for_stats:
            return

        words = TextAnalysisUtils.count_words(text_for_stats)
        characters = TextAnalysisUtils.count_characters(text_for_stats)
        if words == 0 and characters == 0:
            return

        speed_factor = segment.speed if segment.speed and segment.speed > 0 else 1.0
        normalized_duration = duration_seconds / speed_factor
        if normalized_duration <= 0.05:
            return

        voice_name = self._resolve_voice_for_segment(segment)

        # Get current stats before update for comparison
        voice_stats = self.sample_manager.duration_database.get_or_create_stats(voice_name)
        old_wpm = voice_stats.words_per_minute
        old_cps = voice_stats.characters_per_second
        old_samples = voice_stats.total_samples

        smoothing_alpha = self.config.duration_smoothing_alpha
        if smoothing_alpha is not None and smoothing_alpha <= 0:
            smoothing_alpha = None

        with self._duration_stats_lock:
            self.sample_manager.duration_database.update_voice_stats(
                voice_name=voice_name,
                words=words,
                characters=characters,
                duration=normalized_duration,
                smoothing_alpha=smoothing_alpha
            )
            self._stats_update_counter += 1
            self._maybe_persist_duration_stats_locked()

        # Get updated stats for comparison
        new_wpm = voice_stats.words_per_minute
        new_cps = voice_stats.characters_per_second

        logger.info(
            f"Stats correction for voice '{voice_name}': "
            f"WPM {old_wpm:.1f} -> {new_wpm:.1f}, CPS {old_cps:.1f} -> {new_cps:.1f} "
            f"(samples: {old_samples} -> {voice_stats.total_samples}, alpha={smoothing_alpha})"
        )
        logger.debug(
            f"  From synthesis: {words} words, {characters} chars, {duration_seconds:.2f}s "
            f"(normalized {normalized_duration:.2f}s, speed={speed_factor:.2f})"
        )

    def _save_duration_stats_locked(self) -> None:
        """Save duration statistics to disk. Caller must hold _duration_stats_lock."""
        try:
            self.sample_manager.save_duration_stats(preserve_embeddings=True)
            logger.debug("Persisted Gemini duration stats to disk")
        except Exception as e:
            logger.error(f"Failed to persist Gemini duration stats: {e}")

    def _maybe_persist_duration_stats_locked(self) -> None:
        """Auto-save duration stats when configured threshold is met."""
        if not self.config.duration_stats_auto_save:
            return

        interval = self.config.duration_stats_save_interval
        if not interval or interval <= 0:
            return

        if self._stats_update_counter >= interval:
            self._save_duration_stats_locked()
            self._stats_update_counter = 0

    def save_duration_stats(self) -> None:
        """Persist current duration statistics to disk."""
        with self._duration_stats_lock:
            self._save_duration_stats_locked()
            self._stats_update_counter = 0

    def _get_cache_key(self, segment_data: TTSSegmentData, language: str) -> str:
        """Generate a unique cache key for a segment based on its properties."""
        # Include all relevant parameters that affect audio generation
        voice_name = segment_data.voice or self.voice_mapping.get(segment_data.speaker, self.config.default_voice)
        style_prompt = segment_data.style_prompt or self.voice_prompt_mapping.get(segment_data.speaker, "")
        
        # Create a unique key
        key_data = {
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

    def _synthesize_single_segment(
        self,
        segment_data: TTSSegmentData,
        temp_output_path: str,
        language: str,
        max_retries_per_model: int = 3,
        previous_segments: Optional[List[str]] = None,
        usage_tracker: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Synthesizes a single segment and saves it to a temporary path with validation and retry logic.

        Returns:
            Tuple containing the text sent to the model and the model identifier used.
        """
        if not self.api_client.client:
            raise RuntimeError("Gemini client not initialized.")

        # Check cache first
        cache_key = self._get_cache_key(segment_data, language)
        if cache_key in self._audio_cache:
            cached_path, _ = self._audio_cache[cache_key]
            if os.path.exists(cached_path):
                shutil.copy(cached_path, temp_output_path)
                logger.debug(f"  Gemini: Using cached audio for speaker {segment_data.speaker}")
                return None, None

        # Ensure the API client is using the original model
        self.api_client.reset_to_original_model()

        # Try with original model
        success, primary_silence, primary_best_path, primary_text, primary_model = self._attempt_segment_synthesis(
            segment_data, temp_output_path, language, max_retries_per_model, max_silence_ratio=0.01,
            previous_segments=previous_segments,
            usage_tracker=usage_tracker
        )

        if success:
            if primary_best_path:
                shutil.move(primary_best_path, temp_output_path)
            return primary_text, primary_model

        # If primary model fails, try the fallback model
        fallback_best_path = None
        fallback_text: Optional[str] = None
        if self.api_client.switch_to_fallback_model():
            logger.info(f"Attempting synthesis for speaker {segment_data.speaker} with fallback model")
            success, fallback_silence, fallback_best_path, fallback_text, fallback_model = self._attempt_segment_synthesis(
                segment_data, temp_output_path, language, max_retries_per_model, max_silence_ratio=0.05,
                previous_segments=previous_segments,
                usage_tracker=usage_tracker
            )
            self.api_client.reset_to_original_model()

            if success:
                if fallback_best_path:
                    shutil.move(fallback_best_path, temp_output_path)
                if primary_best_path and os.path.exists(primary_best_path):
                    os.remove(primary_best_path)
                return fallback_text, fallback_model

        # Both models failed, compare the best attempts
        if primary_best_path and fallback_best_path:
            if primary_silence <= fallback_silence:
                logger.warning(f"Both models failed validation. Using best attempt from primary model (silence: {primary_silence:.2f})")
                shutil.move(primary_best_path, temp_output_path)
                os.remove(fallback_best_path)
                return primary_text, primary_model
            else:
                logger.warning(f"Both models failed validation. Using best attempt from fallback model (silence: {fallback_silence:.2f})")
                shutil.move(fallback_best_path, temp_output_path)
                os.remove(primary_best_path)
                return fallback_text, fallback_model

        # Handle cases where one of the models didn't produce any output
        if primary_best_path:
            logger.warning(f"Fallback model failed. Using best attempt from primary model (silence: {primary_silence:.2f})")
            shutil.move(primary_best_path, temp_output_path)
            return primary_text, primary_model
        if fallback_best_path:
            logger.warning(f"Primary model failed. Using best attempt from fallback model (silence: {fallback_silence:.2f})")
            shutil.move(fallback_best_path, temp_output_path)
            return fallback_text, fallback_model

        raise RuntimeError(f"Failed to synthesize segment for speaker {segment_data.speaker} after all attempts.")

    def _attempt_segment_synthesis(self, segment_data: TTSSegmentData, temp_output_path: str,
                                 language: str, max_retries: int, max_silence_ratio: float = 0.01,
                                 previous_segments: Optional[List[str]] = None,
                                 usage_tracker: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, Optional[str], Optional[str], Optional[str]]:
        """
        Attempt segment synthesis with the current model.
        Returns a tuple of (success, silence_ratio, best_attempt_path, best_attempt_text, model_used).
        """
        best_attempt_path: Optional[str] = None
        best_attempt_text: Optional[str] = None
        best_model: Optional[str] = None
        best_silence_ratio = float('inf')
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

            try:
                logger.debug(f"  Gemini: Synthesizing segment for {speaker_id} (attempt {attempt + 1}/{max_retries}) with model {self.api_client.current_model}")
                voice_name = self._resolve_voice_for_segment(segment_data)
                current_model = self.api_client.current_model

                final_text = text_to_synthesize
                style_hint = self._get_style_prompt_for_speaker(speaker_id, segment_data)
                prompt_parts = []
                if self.config.prompt_prefix:
                    prompt_parts.append(self.config.prompt_prefix)
                if style_hint:
                    prompt_parts.append(style_hint)
                if prompt_parts:
                    final_text = f"{' '.join(prompt_parts)}\n\n{text_to_synthesize}"

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
                    is_valid, reason, silence_ratio = AudioValidator.validate_audio_sample(
                        temp_attempt_path, 
                        max_silence_ratio=max_silence_ratio
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

                    if silence_ratio < best_silence_ratio:
                        if best_attempt_path and os.path.exists(best_attempt_path):
                            try:
                                os.remove(best_attempt_path)
                            except OSError as e:
                                logger.warning(f"Could not remove old best_attempt_path: {e}")
                        best_silence_ratio = silence_ratio
                        best_attempt_path = temp_attempt_path
                        best_attempt_text = final_text
                        best_model = current_model
                    elif temp_attempt_path != best_attempt_path:
                        try:
                            os.remove(temp_attempt_path)
                        except OSError as e:
                            logger.warning(f"Could not remove temp_attempt_path: {e}")

                    if is_valid:
                        best_model = current_model
                        return True, silence_ratio, best_attempt_path, final_text, current_model
                    else:
                        logger.debug(f"Segment validation failed for {speaker_id} (attempt {attempt + 1}): {reason}")
                else:
                    # If validation is disabled, we can't determine the best path, so we just return the first successful one.
                    best_model = current_model
                    return True, 0.0, temp_attempt_path, final_text, current_model

            except Exception as e:
                logger.error(f"Error synthesizing segment for {speaker_id} (attempt {attempt + 1}): {e}")
                time.sleep(2)

        return False, best_silence_ratio, best_attempt_path, best_attempt_text, best_model

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
        logger.info(f"Gemini: Synthesizing segment {segment_index+1}/{total_segments} for speaker '{segment.speaker}'")
        try:
            synthesized_text, model_used = self._synthesize_single_segment(
                segment, segment_file_path, language,
                previous_segments=context_segments,
                usage_tracker=usage_tracker
            )

            # Get duration of the synthesized audio
            duration = AudioFileUtils.get_audio_duration_seconds(segment_file_path) or 0.0
            self._record_duration_stats(segment, synthesized_text, duration)
            if duration and model_used:
                self._register_usage(usage_tracker, model_used, audio_seconds=duration)

            # Cache the generated audio
            if self._cache_dir:
                cache_path = os.path.join(self._cache_dir, f"{cache_key}.wav")
                shutil.copy(segment_file_path, cache_path)
                self._audio_cache[cache_key] = (cache_path, duration)

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
            return None

    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: str = "en",
        **kwargs: Any
    ) -> List[SegmentAlignment]:
        """
        Synthesize speech for each segment individually using parallel processing.

        Args:
            segments_data: List of segments to synthesize
            language: Target language code
            **kwargs: Additional synthesis parameters
                - previous_context: List[str] - Previous segment texts for emotion enrichment

        Returns:
            List of segment alignments (in original order)
        """
        if not self.is_available():
            raise RuntimeError("Gemini TTS not initialized.")

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

        # Thread-safe context tracking for emotion enrichment
        context_lock = threading.Lock()
        usage_tracker: Dict[str, Any] = {"models": {}, "lock": threading.Lock()}

        # Use external context if provided (for resynthesis), otherwise build from segments
        external_context = kwargs.get('previous_context', None)
        context_segments: List[str] = external_context if external_context is not None else []

        def get_context_for_segment(segment_idx: int) -> List[str]:
            """Get context segments for a given segment index (thread-safe)."""
            with context_lock:
                if external_context is not None:
                    # Use provided external context (for resynthesis)
                    return external_context
                else:
                    # Build context from previous segments in current batch
                    start_idx = max(0, segment_idx - 5)
                    return [valid_segments[i].text for i in range(start_idx, segment_idx)]

        def update_context(segment_text: str) -> None:
            """Update context with completed segment (thread-safe)."""
            with context_lock:
                # Only update context if we're building it ourselves (not using external)
                if external_context is None:
                    context_segments.append(segment_text)

        try:
            # Results dict to preserve order: {segment_index: alignment}
            results: Dict[int, Optional[SegmentAlignment]] = {}
            results_lock = threading.Lock()

            def process_segment_wrapper(idx: int, seg: TTSSegmentData) -> None:
                """Wrapper to process segment and store result with index."""
                context = get_context_for_segment(idx) if self.config.enable_emotion_enrichment else []
                alignment = self._process_single_segment(
                    segment=seg,
                    segment_index=idx,
                    total_segments=len(valid_segments),
                    temp_dir=temp_dir,
                    language=language,
                    context_segments=context,
                    usage_tracker=usage_tracker
                )

                with results_lock:
                    results[idx] = alignment

                # Update context after successful synthesis
                if alignment and self.config.enable_emotion_enrichment:
                    update_context(seg.text)

            # Use ThreadPoolExecutor for parallel processing
            max_workers = min(self.config.max_workers, len(valid_segments))
            logger.info(f"Gemini: Starting parallel synthesis with {max_workers} workers for {len(valid_segments)} segments")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_segment_wrapper, i, segment): i
                    for i, segment in enumerate(valid_segments)
                }

                # Wait for all to complete and handle any exceptions
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()  # This will raise exception if task failed
                    except Exception as e:
                        logger.error(f"Unexpected error in parallel segment {idx}: {e}")
                        with results_lock:
                            results[idx] = None

            # Collect results in original order
            alignments = [results.get(i) for i in range(len(valid_segments)) if results.get(i) is not None]

            logger.info(f"Gemini: Synthesized {len(alignments)}/{len(valid_segments)} segments successfully")
            if self.cost_tracker and usage_tracker:
                models_usage: Dict[str, Dict[str, float]] = {}
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
                            audio_seconds=audio_sec
                        )
            return alignments

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def is_available(self) -> bool:
        """Check if Gemini TTS is available and initialized."""
        return GEMINI_AVAILABLE and self.api_client.client is not None

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
        
        # Clean up AudioEmbedder if needed
        if self.sample_manager and self.sample_manager.voice_matcher:
            self.sample_manager.voice_matcher = None
