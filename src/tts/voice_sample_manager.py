from typing import Optional, Dict, Any, List, Union, Tuple, Callable
import os
import wave
import time
import json
import re
import tempfile
import math
from pathlib import Path
import numpy as np
import threading

from .models import VoiceDurationStats, VoiceDurationDatabase
from src.utils.audio_embedder import AudioEmbedder
from src.utils.voice_matcher import VoiceMatcher
from src.dubbing.core.log_config import get_logger

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

logger = get_logger(__name__)

SAMPLE_RATE = 24000


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

    @staticmethod
    def concatenate_audio_files(audio_files: List[str], sample_rate: int = SAMPLE_RATE) -> bytes:
        """Concatenate multiple WAV files into a single PCM data stream."""
        if not audio_files:
            return b''
        
        if len(audio_files) == 1:
            with wave.open(audio_files[0], 'rb') as wf:
                return wf.readframes(wf.getnframes())
        
        combined_frames = b''
        for audio_file in audio_files:
            try:
                with wave.open(audio_file, 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    combined_frames += frames
            except Exception as e:
                logger.error(f"Error reading audio file {audio_file}: {e}")
        
        return combined_frames


class AudioValidator:
    """Validates the quality of generated audio samples."""

    @staticmethod
    def _max_consecutive_true(mask: Any) -> int:
        """Return longest run of truthy values in a 1D boolean-like mask."""
        if hasattr(mask, "tolist"):
            values = mask.tolist()
        else:
            values = list(mask)

        max_run = 0
        current_run = 0
        for value in values:
            if bool(value):
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0
        return max_run
    
    @staticmethod
    def validate_audio_sample(audio_path: Union[str, Path], 
                            expected_min_duration: float = 1.0,
                            silence_threshold_db: float = -40.0,
                            max_silence_ratio: float = 0.03,
                            trailing_silence_grace_ratio: float = 0.0,
                            fade_detection_config: Optional[Dict[str, Any]] = None,
                            absolute_silence_threshold_db: float = -38.0,
                            max_total_silence_ratio: float = 0.35,
                            max_contiguous_silence_seconds: float = 8.0) -> tuple[bool, str, float]:
        """
        Validate audio by checking trailing and pathological internal silence.
        
        Args:
            audio_path: Path to the audio file
            expected_min_duration: Minimum expected duration in seconds
            silence_threshold_db: Threshold below which audio is considered silence (in dB)
            max_silence_ratio: Maximum allowed ratio of trailing silence vs total duration (0.1 = 10%)
            trailing_silence_grace_ratio: Additional tolerance for trailing silence only.
                Example: 0.005 means allow +0.5pp above max_silence_ratio.
            fade_detection_config: Optional dict with fade detection settings:
                - enabled: bool - Enable fade detection
                - window_size_ms: int - Window size for fade analysis
                - min_fade_db: float - Minimum dB drop to consider as fade
                - percentile: int - Percentile for reference level
            absolute_silence_threshold_db: Absolute dBFS threshold for silence mask
            max_total_silence_ratio: Maximum allowed ratio of silence across whole segment
            max_contiguous_silence_seconds: Maximum allowed length of uninterrupted silence
            
        Returns:
            Tuple of (is_valid, reason, diagnostic_silence_ratio)
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                return False, "Audio file does not exist", 1.0
            
            file_size = audio_path.stat().st_size
            if file_size < 1000:
                return False, f"Audio file too small ({file_size} bytes)", 1.0
            
            duration = AudioFileUtils.get_audio_duration_seconds(audio_path)
            if duration is None:
                return False, "Could not determine audio duration", 1.0
            
            if duration < expected_min_duration:
                return False, f"Audio too short ({duration:.2f}s < {expected_min_duration:.2f}s)", 1.0
            
            if LIBROSA_AVAILABLE:
                return AudioValidator._validate_with_librosa(
                    audio_path,
                    silence_threshold_db,
                    max_silence_ratio,
                    trailing_silence_grace_ratio,
                    fade_detection_config,
                    absolute_silence_threshold_db,
                    max_total_silence_ratio,
                    max_contiguous_silence_seconds
                )
            elif PYTORCH_AVAILABLE:
                return AudioValidator._validate_with_pytorch(
                    audio_path,
                    silence_threshold_db,
                    max_silence_ratio,
                    trailing_silence_grace_ratio,
                    fade_detection_config,
                    absolute_silence_threshold_db,
                    max_total_silence_ratio,
                    max_contiguous_silence_seconds
                )
            else:
                logger.warning("Advanced audio validation not available. Using basic checks only.")
                return True, "Basic validation passed (advanced libraries not available)", 0.0
        
        except Exception as e:
            logger.error(f"Error validating audio sample {audio_path}: {e}")
            return False, f"Validation error: {str(e)}", 1.0
    
    @staticmethod
    def _validate_with_librosa(audio_path: Path, silence_threshold_db: float,
                              max_silence_ratio: float, trailing_silence_grace_ratio: float = 0.0,
                              fade_detection_config: Optional[Dict[str, Any]] = None,
                              absolute_silence_threshold_db: float = -38.0,
                              max_total_silence_ratio: float = 0.35,
                              max_contiguous_silence_seconds: float = 8.0) -> tuple[bool, str, float]:
        """Validate audio using librosa with fade detection."""
        try:
            fade_config = {
                'enabled': True,
                'window_size_ms': 500,
                'min_fade_db': 10.0,
                'percentile': 75
            }
            if fade_detection_config:
                fade_config.update(fade_detection_config)
            
            y, sr = librosa.load(str(audio_path), sr=None)
            
            if len(y) == 0:
                return False, "Audio file contains no data", 1.0
            
            if np.std(y) < 1e-6:
                return False, "Audio appears to be flat/constant", 1.0
            
            hop_length = int(sr * 0.1)
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            
            ref_level = np.percentile(rms, 90)
            if ref_level == 0:
                ref_level = np.max(rms)
            if ref_level == 0:
                return False, "Audio contains no energy", 1.0
                
            rms_db = librosa.amplitude_to_db(rms, ref=ref_level)
            # Absolute level in dBFS: robust against pathological files where relative
            # normalization can hide near-silent content.
            rms_dbfs = librosa.amplitude_to_db(rms + 1e-12, ref=1.0)
            
            total_frames = len(rms_db)
            if total_frames == 0:
                return False, "Audio contains no analyzable frames", 1.0

            dynamic_range = np.max(rms_db) - np.min(rms_db)
            # Use absolute dBFS mask to detect extended near-silent sections.
            global_silence_mask = rms_dbfs < absolute_silence_threshold_db
            total_silence_ratio = float(np.mean(global_silence_mask))
            frame_duration_sec = hop_length / float(sr)
            max_contiguous_silence_sec = AudioValidator._max_consecutive_true(global_silence_mask) * frame_duration_sec
            
            if fade_config['enabled']:
                window_size_ms = fade_config['window_size_ms']
                window_size_frames = max(1, int((window_size_ms / 1000.0) / 0.1))
                
                if total_frames < window_size_frames * 2:
                    adaptive_threshold = max(silence_threshold_db, np.min(rms_db) + dynamic_range * 0.1)
                    silence_mask = rms_db < adaptive_threshold
                    non_silent_indices = np.where(~silence_mask)[0]
                    trailing_silence_ratio = 1.0 if non_silent_indices.size == 0 else max(0, total_frames - (int(non_silent_indices[-1]) + 1)) / total_frames
                else:
                    trailing_start_idx = total_frames - window_size_frames
                    main_content = rms_db[:trailing_start_idx]
                    trailing_content = rms_db[trailing_start_idx:]
                    
                    reference_level_db = np.percentile(main_content, fade_config['percentile'])
                    trailing_level_db = np.mean(trailing_content)
                    fade_db = reference_level_db - trailing_level_db
                    has_significant_fade = fade_db > fade_config['min_fade_db']
                    
                    if has_significant_fade:
                        fade_threshold_db = reference_level_db - fade_config['min_fade_db']
                        faded_mask = rms_db < fade_threshold_db
                        non_faded_indices = np.where(~faded_mask)[0]
                        if non_faded_indices.size == 0:
                            trailing_silence_ratio = 1.0
                        else:
                            last_non_faded = int(non_faded_indices[-1])
                            trailing_silent_frames = max(0, total_frames - (last_non_faded + 1))
                            trailing_silence_ratio = trailing_silent_frames / total_frames
                    else:
                        adaptive_threshold = max(silence_threshold_db, np.min(rms_db) + dynamic_range * 0.1)
                        silence_mask = rms_db < adaptive_threshold
                        non_silent_indices = np.where(~silence_mask)[0]
                        trailing_silence_ratio = 1.0 if non_silent_indices.size == 0 else max(0, total_frames - (int(non_silent_indices[-1]) + 1)) / total_frames
                    
                    logger.debug(f"Audio analysis - Dynamic range: {dynamic_range:.1f}dB, "
                                f"Reference level: {reference_level_db:.1f}dB, "
                                f"Trailing level: {trailing_level_db:.1f}dB, "
                                f"Fade: {fade_db:.1f}dB, "
                                f"Significant fade: {has_significant_fade}, "
                                f"Trailing silence ratio: {trailing_silence_ratio:.2%}, "
                                f"Total silence ratio: {total_silence_ratio:.2%}, "
                                f"Max contiguous silence: {max_contiguous_silence_sec:.2f}s")
            else:
                if dynamic_range < 6:
                    adaptive_threshold = np.min(rms_db) + 1
                else:
                    adaptive_threshold = max(silence_threshold_db, np.min(rms_db) + dynamic_range * 0.1)
                
                silence_mask = rms_db < adaptive_threshold
                non_silent_indices = np.where(~silence_mask)[0]
                if non_silent_indices.size == 0:
                    trailing_silence_ratio = 1.0
                else:
                    last_non_silent = int(non_silent_indices[-1])
                    trailing_silent_frames = max(0, total_frames - (last_non_silent + 1))
                    trailing_silence_ratio = trailing_silent_frames / total_frames
                
                logger.debug(f"Audio analysis - Dynamic range: {dynamic_range:.1f}dB, "
                            f"Adaptive threshold: {adaptive_threshold:.1f}dB, "
                            f"Absolute silence threshold: {absolute_silence_threshold_db:.1f}dBFS, "
                            f"Trailing silence ratio: {trailing_silence_ratio:.2%}, "
                            f"Total silence ratio: {total_silence_ratio:.2%}, "
                            f"Max contiguous silence: {max_contiguous_silence_sec:.2f}s")
            
            diagnostic_ratio = max(trailing_silence_ratio, total_silence_ratio)

            if total_silence_ratio > max_total_silence_ratio:
                return False, (
                    f"Too much total silence ({total_silence_ratio:.2%} > {max_total_silence_ratio:.2%})"
                ), diagnostic_ratio
            
            if max_contiguous_silence_sec > max_contiguous_silence_seconds:
                return False, (
                    f"Silence block too long ({max_contiguous_silence_sec:.2f}s > {max_contiguous_silence_seconds:.2f}s)"
                ), diagnostic_ratio
            
            effective_max_silence_ratio = max_silence_ratio + max(0.0, trailing_silence_grace_ratio)
            if trailing_silence_ratio > effective_max_silence_ratio:
                return False, (
                    f"Too much trailing silence ({trailing_silence_ratio:.2%} > {effective_max_silence_ratio:.2%})"
                ), diagnostic_ratio
            
            return True, f"Audio validation passed (trailing silence: {trailing_silence_ratio:.2%})", diagnostic_ratio
            
        except Exception as e:
            return False, f"Librosa validation error: {str(e)}", 1.0
    
    @staticmethod
    def _validate_with_pytorch(audio_path: Path, silence_threshold_db: float,
                              max_silence_ratio: float, trailing_silence_grace_ratio: float = 0.0,
                              fade_detection_config: Optional[Dict[str, Any]] = None,
                              absolute_silence_threshold_db: float = -38.0,
                              max_total_silence_ratio: float = 0.35,
                              max_contiguous_silence_seconds: float = 8.0) -> tuple[bool, str, float]:
        """Validate audio using PyTorch/torchaudio with fade detection."""
        try:
            fade_config = {
                'enabled': True,
                'window_size_ms': 500,
                'min_fade_db': 10.0,
                'percentile': 75
            }
            if fade_detection_config:
                fade_config.update(fade_detection_config)
            
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            if waveform.numel() == 0:
                return False, "Audio file contains no data", 1.0
            
            if waveform.shape[0] > 1:
                waveform = waveform[0:1]
            
            if torch.std(waveform) < 1e-6:
                return False, "Audio appears to be flat/constant", 1.0
            
            chunk_size = sample_rate // 10
            num_chunks = waveform.shape[1] // chunk_size
            
            if num_chunks == 0:
                return False, "Audio too short for chunk analysis", 1.0
            
            chunk_rms_values = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, waveform.shape[1])
                chunk = waveform[:, start_idx:end_idx]
                
                chunk_rms = torch.sqrt(torch.mean(chunk ** 2))
                chunk_rms_values.append(chunk_rms.item())
            
            chunk_rms_tensor = torch.tensor(chunk_rms_values)
            
            ref_level = torch.quantile(chunk_rms_tensor, 0.9).item()
            if ref_level == 0:
                ref_level = torch.max(chunk_rms_tensor).item()
            if ref_level == 0:
                return False, "Audio contains no energy", 1.0
            
            chunk_db_values = 20 * torch.log10(chunk_rms_tensor + 1e-8) - 20 * torch.log10(torch.tensor(ref_level))
            # Absolute chunk level in dBFS for robust silence detection.
            chunk_dbfs_values = 20 * torch.log10(chunk_rms_tensor + 1e-8)
            dynamic_range = (torch.max(chunk_db_values) - torch.min(chunk_db_values)).item()
            global_silence_mask = (chunk_dbfs_values < absolute_silence_threshold_db)
            total_silence_ratio = float(torch.mean(global_silence_mask.float()).item())
            chunk_duration_sec = chunk_size / float(sample_rate)
            max_contiguous_silence_sec = AudioValidator._max_consecutive_true(global_silence_mask) * chunk_duration_sec
            
            if fade_config['enabled']:
                window_size_ms = fade_config['window_size_ms']
                window_size_chunks = max(1, int((window_size_ms / 1000.0) / 0.1))
                
                if num_chunks < window_size_chunks * 2:
                    adaptive_threshold = max(silence_threshold_db, torch.min(chunk_db_values).item() + dynamic_range * 0.1)
                    silence_mask = (chunk_db_values < adaptive_threshold)
                    non_silent_indices = torch.nonzero(~silence_mask, as_tuple=False).flatten()
                    trailing_silence_ratio = 1.0 if non_silent_indices.numel() == 0 else max(0, num_chunks - (int(non_silent_indices[-1].item()) + 1)) / num_chunks
                else:
                    trailing_start_idx = num_chunks - window_size_chunks
                    main_content = chunk_db_values[:trailing_start_idx]
                    trailing_content = chunk_db_values[trailing_start_idx:]
                    
                    reference_level_db = torch.quantile(main_content, fade_config['percentile'] / 100.0).item()
                    trailing_level_db = torch.mean(trailing_content).item()
                    fade_db = reference_level_db - trailing_level_db
                    has_significant_fade = fade_db > fade_config['min_fade_db']
                    
                    if has_significant_fade:
                        fade_threshold_db = reference_level_db - fade_config['min_fade_db']
                        faded_mask = chunk_db_values < fade_threshold_db
                        non_faded_indices = torch.nonzero(~faded_mask, as_tuple=False).flatten()
                        if non_faded_indices.numel() == 0:
                            trailing_silence_ratio = 1.0
                        else:
                            last_non_faded = int(non_faded_indices[-1].item())
                            trailing_silent_chunks = max(0, num_chunks - (last_non_faded + 1))
                            trailing_silence_ratio = trailing_silent_chunks / num_chunks
                    else:
                        adaptive_threshold = max(silence_threshold_db, torch.min(chunk_db_values).item() + dynamic_range * 0.1)
                        silence_mask = (chunk_db_values < adaptive_threshold)
                        non_silent_indices = torch.nonzero(~silence_mask, as_tuple=False).flatten()
                        trailing_silence_ratio = 1.0 if non_silent_indices.numel() == 0 else max(0, num_chunks - (int(non_silent_indices[-1].item()) + 1)) / num_chunks
                    
                    logger.debug(f"Audio analysis - Dynamic range: {dynamic_range:.1f}dB, "
                                f"Reference level: {reference_level_db:.1f}dB, "
                                f"Trailing level: {trailing_level_db:.1f}dB, "
                                f"Fade: {fade_db:.1f}dB, "
                                f"Significant fade: {has_significant_fade}, "
                                f"Trailing silence ratio: {trailing_silence_ratio:.2%}, "
                                f"Total silence ratio: {total_silence_ratio:.2%}, "
                                f"Max contiguous silence: {max_contiguous_silence_sec:.2f}s")
            else:
                if dynamic_range < 6:
                    adaptive_threshold = torch.min(chunk_db_values).item() + 1
                else:
                    adaptive_threshold = max(silence_threshold_db, torch.min(chunk_db_values).item() + dynamic_range * 0.1)
                
                silence_mask = (chunk_db_values < adaptive_threshold)
                non_silent_indices = torch.nonzero(~silence_mask, as_tuple=False).flatten()
                if non_silent_indices.numel() == 0:
                    trailing_silence_ratio = 1.0
                else:
                    last_non_silent = int(non_silent_indices[-1].item())
                    trailing_silent_chunks = max(0, num_chunks - (last_non_silent + 1))
                    trailing_silence_ratio = trailing_silent_chunks / num_chunks
                
                logger.debug(f"Audio analysis - Dynamic range: {dynamic_range:.1f}dB, "
                            f"Adaptive threshold: {adaptive_threshold:.1f}dB, "
                            f"Absolute silence threshold: {absolute_silence_threshold_db:.1f}dBFS, "
                            f"Trailing silence ratio: {trailing_silence_ratio:.2%}, "
                            f"Total silence ratio: {total_silence_ratio:.2%}, "
                            f"Max contiguous silence: {max_contiguous_silence_sec:.2f}s")
            
            diagnostic_ratio = max(trailing_silence_ratio, total_silence_ratio)

            if total_silence_ratio > max_total_silence_ratio:
                return False, (
                    f"Too much total silence ({total_silence_ratio:.2%} > {max_total_silence_ratio:.2%})"
                ), diagnostic_ratio
            
            if max_contiguous_silence_sec > max_contiguous_silence_seconds:
                return False, (
                    f"Silence block too long ({max_contiguous_silence_sec:.2f}s > {max_contiguous_silence_seconds:.2f}s)"
                ), diagnostic_ratio
            
            effective_max_silence_ratio = max_silence_ratio + max(0.0, trailing_silence_grace_ratio)
            if trailing_silence_ratio > effective_max_silence_ratio:
                return False, (
                    f"Too much trailing silence ({trailing_silence_ratio:.2%} > {effective_max_silence_ratio:.2%})"
                ), diagnostic_ratio
            
            return True, f"Audio validation passed (trailing silence: {trailing_silence_ratio:.2%})", diagnostic_ratio
            
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
        
        pause_punctuation = re.findall(r'[.!?,:;…]', text)
        punctuation_density = len(pause_punctuation) / len(text)
        
        words = re.findall(r'\b\w+\b', text)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        complexity = 1.0
        complexity += punctuation_density * 0.5
        complexity += max(0, (avg_word_length - 5) * 0.02)
        complexity += max(0, (avg_sentence_length - 15) * 0.01)
        
        return min(complexity, 2.0)

    @staticmethod
    def estimate_punctuation_pause_seconds(text: str) -> float:
        """Estimate additional pause time from punctuation."""
        if not text:
            return 0.0

        commas = len(re.findall(r",", text))
        terminals = len(re.findall(r"[\.!\?]", text))
        mids = len(re.findall(r"[:;…]", text))
        dashes = len(re.findall(r"[—-]", text))

        return commas * 0.12 + terminals * 0.26 + mids * 0.18 + dashes * 0.14

    @staticmethod
    def style_emotion_speed_multiplier(style_prompt: Optional[str], emotion: Optional[str]) -> float:
        """Return multiplicative speed factor based on style/emotion."""
        style_prompt_l = (style_prompt or "").lower()
        emotion_l = (emotion or "").lower()

        emotion_map: Dict[str, float] = {
            "angry": 0.93,
            "excited": 0.94,
            "happy": 0.97,
            "sad": 1.08,
            "bored": 1.06,
            "surprised": 0.96,
            "scared": 1.03,
        }

        style_map: Dict[str, float] = {
            "whisper": 1.05,
            "whispering": 1.05,
            "shout": 0.95,
            "shouting": 0.95,
            "sarcasm": 1.02,
            "robotic": 0.98,
            "calm": 1.00,
        }

        factor = 1.0

        for key, val in emotion_map.items():
            if key in emotion_l:
                factor *= val
                break

        for key, val in style_map.items():
            if key in style_prompt_l:
                factor *= val
                break

        return max(0.8, min(1.25, factor))


class VoiceSampleManager:
    """
    TTS-agnostic voice sample manager for generating samples, computing embeddings,
    tracking duration statistics, and estimating audio length.
    """
    
    def __init__(
        self,
        tts_provider: str,
        voice_list: List[str],
        samples_dir: Path,
        stats_file: Path,
        adjustments_file: Path,
        sample_text: str,
        audio_embedder: Optional[AudioEmbedder] = None,
        voice_matcher: Optional[VoiceMatcher] = None,
        enable_voice_matching: bool = True,
        enable_audio_validation: bool = True,
        duration_smoothing_alpha: float = 0.35,
        duration_stats_auto_save: bool = True,
        duration_stats_save_interval: int = 20,
        fade_detection_enabled: bool = True,
        fade_window_size_ms: int = 500,
        min_fade_db: float = 10.0,
        fade_detection_percentile: int = 75,
    ):
        """
        Initialize VoiceSampleManager.
        
        Args:
            tts_provider: TTS provider name ("gemini", "openai", "minimax")
            voice_list: List of available voice names
            samples_dir: Directory for storing voice samples
            stats_file: File path for storing stats and embeddings
            adjustments_file: File path for storing adaptive biases
            sample_text: Text to use for generating voice samples
            audio_embedder: AudioEmbedder instance for voice matching
            voice_matcher: VoiceMatcher instance for voice similarity
            enable_voice_matching: Whether to enable voice matching
            enable_audio_validation: Whether to validate generated audio
            duration_smoothing_alpha: EMA smoothing factor for runtime stats
            duration_stats_auto_save: Persist runtime stats automatically
            duration_stats_save_interval: Save after this many updates
            fade_detection_enabled: Enable volume fade detection
            fade_window_size_ms: Window size for fade analysis
            min_fade_db: Minimum dB drop to consider as significant fade
            fade_detection_percentile: Percentile for reference level calculation
        """
        self.tts_provider = tts_provider
        self.voice_list = voice_list
        self.samples_dir = Path(samples_dir)
        self.stats_file = Path(stats_file)
        self.adjustments_file = Path(adjustments_file)
        self.sample_text = sample_text
        
        self.audio_embedder = audio_embedder
        self.voice_matcher = voice_matcher
        self.enable_voice_matching = enable_voice_matching
        self.enable_audio_validation = enable_audio_validation
        
        self.duration_smoothing_alpha = duration_smoothing_alpha
        self.duration_stats_auto_save = duration_stats_auto_save
        self.duration_stats_save_interval = duration_stats_save_interval
        
        self.fade_detection_enabled = fade_detection_enabled
        self.fade_window_size_ms = fade_window_size_ms
        self.min_fade_db = min_fade_db
        self.fade_detection_percentile = fade_detection_percentile
        
        self.duration_database = VoiceDurationDatabase()
        self._sample_generator: Optional[Callable[[str, str], bool]] = None
        
        # Adaptive duration bias tracking
        self._biases: Dict[str, Dict[str, float]] = {"speaker": {}, "voice": {}, "language": {}}
        self._bias_lock = threading.Lock()
        self._bias_alpha: float = 0.6
        
        # Stats update tracking
        self._duration_stats_lock = threading.Lock()
        self._stats_update_counter = 0
        
        self.samples_dir.mkdir(parents=True, exist_ok=True)
    
    def set_sample_generator(self, generator_fn: Callable[[str, str], bool]) -> None:
        """
        Set callback for generating audio samples.
        
        Args:
            generator_fn: Function that takes (voice_name, output_path) and returns success bool
        """
        self._sample_generator = generator_fn
    
    def generate_all_samples(self, force_regenerate: bool = False) -> bool:
        """
        Generate all samples, analyze durations, and compute embeddings.
        
        Args:
            force_regenerate: Force regeneration of all samples and stats
            
        Returns:
            True if successful, False otherwise
        """
        if not self._sample_generator:
            logger.error("Sample generator not set. Call set_sample_generator() first.")
            return False
        
        logger.info(f"Starting comprehensive sample generation for {self.tts_provider}...")
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing stats
        stats_loaded = False
        existing_embeddings: Dict[str, np.ndarray] = {}
        if not force_regenerate and self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    combined_data = json.load(f)
                    if 'voice_stats' in combined_data:
                        self.duration_database = VoiceDurationDatabase(**combined_data['voice_stats'])
                        logger.debug(f"Loaded existing duration statistics for {len(self.duration_database.voice_stats)} voices")
                        stats_loaded = True
                    if 'embeddings' in combined_data:
                        existing_embeddings = {
                            voice_name: np.array(embedding_list) 
                            for voice_name, embedding_list in combined_data['embeddings'].items()
                        }
                        logger.debug(f"Loaded {len(existing_embeddings)} voice embeddings from cache")
                        
                        # Validate embedding dimensions match current model
                        if existing_embeddings and self.audio_embedder:
                            expected_dim = self._get_expected_embedding_dim()
                            if expected_dim:
                                first_embedding = next(iter(existing_embeddings.values()))
                                cached_dim = len(first_embedding)
                                if cached_dim != expected_dim:
                                    logger.warning(
                                        f"Cached embeddings dimension ({cached_dim}) doesn't match "
                                        f"current model ({expected_dim}). Regenerating all embeddings."
                                    )
                                    existing_embeddings = {}
            except Exception as e:
                logger.error(f"Error loading combined data: {e}. Will regenerate.")
        
        # Generate samples and analyze durations
        voices_processed = 0
        for voice_name in self.voice_list:
            sample_file_path = self.samples_dir / f"{voice_name}.{self._get_audio_extension()}"
            voice_stats = self.duration_database.get_or_create_stats(voice_name)
            
            has_audio_file = sample_file_path.exists()
            has_stats = stats_loaded and voice_stats.total_samples > 0
            
            if force_regenerate or (not has_audio_file and not has_stats):
                logger.debug(f"Generating audio sample for {voice_name}")
                try:
                    success = self._sample_generator(voice_name, str(sample_file_path))
                    if success:
                        has_audio_file = True
                        logger.debug(f"Generated audio sample for {voice_name}")
                    else:
                        logger.error(f"Failed to generate sample for {voice_name}")
                except Exception as e:
                    logger.error(f"Error generating sample for {voice_name}: {e}")
            
            # Analyze duration
            if has_audio_file and (force_regenerate or not has_stats):
                logger.debug(f"Analyzing duration for {voice_name}")
                try:
                    voice_stats.total_samples = 0
                    voice_stats.total_words = 0
                    voice_stats.total_characters = 0
                    voice_stats.total_duration_seconds = 0.0
                    
                    duration = AudioFileUtils.get_audio_duration_seconds(sample_file_path)
                    if duration:
                        words = TextAnalysisUtils.count_words(self.sample_text)
                        characters = TextAnalysisUtils.count_characters(self.sample_text)
                        voice_stats.update_stats(words, characters, duration)
                        logger.debug(f"Analyzed {voice_name}: {voice_stats.words_per_minute:.1f} WPM, {voice_stats.characters_per_second:.1f} CPS")
                    else:
                        logger.warning(f"Could not determine duration for {voice_name}")
                except Exception as e:
                    logger.error(f"Error analyzing sample for {voice_name}: {e}")
            
            voices_processed += 1
        
        logger.debug(f"Processed duration statistics for {voices_processed} voices")
        
        # Generate embeddings
        if self.enable_voice_matching and self.voice_matcher and self.audio_embedder:
            logger.debug("Processing voice embeddings...")
            current_embeddings = existing_embeddings.copy()
            
            voices_needing_embeddings = []
            if force_regenerate:
                voices_needing_embeddings = self.voice_list
                current_embeddings = {}
                logger.debug("Force regenerating all embeddings")
            else:
                for voice_name in self.voice_list:
                    if voice_name not in current_embeddings:
                        voices_needing_embeddings.append(voice_name)
                
                if voices_needing_embeddings:
                    logger.debug(f"Computing embeddings for {len(voices_needing_embeddings)} missing voice(s)")
                else:
                    logger.debug("All voice embeddings already available")
            
            if voices_needing_embeddings:
                for voice_name in voices_needing_embeddings:
                    sample_file_path = self.samples_dir / f"{voice_name}.{self._get_audio_extension()}"
                    if sample_file_path.exists():
                        embedding = self.voice_matcher.extract_embedding_for_audio_file(sample_file_path)
                        if embedding is not None:
                            current_embeddings[voice_name] = embedding
                    else:
                        logger.warning(f"No audio file found for {voice_name}, cannot compute embedding")
            
            self.voice_matcher.set_sample_embeddings(current_embeddings)
        else:
            current_embeddings: Dict[str, np.ndarray] = {}
        
        # Save combined data
        try:
            combined_data = {
                "voice_stats": self.duration_database.dict(),
                "embeddings": {}
            }
            
            if current_embeddings:
                combined_data["embeddings"] = {
                    voice_name: embedding.tolist() 
                    for voice_name, embedding in current_embeddings.items()
                }
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            
            stats_count = len(self.duration_database.voice_stats)
            embeddings_count = len(current_embeddings)
            logger.info(f"Saved combined data: {stats_count} voice stats, {embeddings_count} embeddings")
            
        except Exception as e:
            logger.error(f"Error saving combined data: {e}")
            return False
        
        logger.info("Sample analysis completed successfully!")
        return True
    
    def _get_audio_extension(self) -> str:
        """Get appropriate audio extension for the TTS provider."""
        if self.tts_provider == "gemini":
            return "wav"
        elif self.tts_provider == "openai":
            return "mp3"
        elif self.tts_provider == "minimax":
            return "mp3"
        else:
            return "wav"
    
    def _get_expected_embedding_dim(self) -> Optional[int]:
        """Get expected embedding dimension from current AudioEmbedder."""
        if not self.audio_embedder:
            return None
        
        # Resemblyzer produces 256-dimensional embeddings
        if hasattr(self.audio_embedder, 'embedding_type'):
            if self.audio_embedder.embedding_type == "resemblyzer":
                return 256
        
        return None
    
    def estimate_duration(
        self,
        text: str,
        voice_name: str,
        language: str = "en",
        style_prompt: Optional[str] = None,
        emotion: Optional[str] = None,
        speed: Optional[float] = None,
        speaker_id: Optional[str] = None,
        apply_biases: bool = True
    ) -> Optional[float]:
        """
        Estimate the duration in seconds for a given text using Gemini's algorithm.
        
        Args:
            text: Text to estimate duration for
            voice_name: Voice name to use for estimation
            language: Target language code
            style_prompt: Optional style prompt
            emotion: Optional emotion
            speed: Optional speed factor
            speaker_id: Optional speaker ID for bias tracking
            apply_biases: Whether to apply adaptive biases
            
        Returns:
            Estimated duration in seconds, or None if estimation is not possible
        """
        if not text or not text.strip():
            return 0.0
        
        voice_stats = self.duration_database.get_or_create_stats(voice_name)
        
        text = text.strip()
        word_count = TextAnalysisUtils.count_words(text)
        char_count = TextAnalysisUtils.count_characters(text)
        complexity_factor = TextAnalysisUtils.estimate_speech_complexity(text)
        punctuation_pause_sec = TextAnalysisUtils.estimate_punctuation_pause_seconds(text)
        
        logger.debug(
            f"Length estimation for voice '{voice_name}': {word_count} words, {char_count} chars, "
            f"complexity={complexity_factor:.2f}, stats_samples={voice_stats.total_samples}"
        )
        
        # Estimate using both word- and character-based methods
        if voice_stats.words_per_minute > 0:
            word_based_duration = (word_count / voice_stats.words_per_minute) * 60.0
        else:
            word_based_duration = (word_count / 150.0) * 60.0
        
        if voice_stats.characters_per_second > 0:
            char_based_duration = char_count / voice_stats.characters_per_second
        else:
            char_based_duration = char_count / 12.5
        
        # Mixing weight
        w_char = 1.0 / (1.0 + math.exp((char_count - 120.0) / 25.0))
        w_word = 1.0 - w_char
        
        base_duration = w_char * char_based_duration + w_word * word_based_duration
        
        if voice_stats.total_samples > 0:
            estimated_duration = base_duration
            logger.debug(
                f"  Mixed estimate: words={word_based_duration:.2f}s, chars={char_based_duration:.2f}s, "
                f"w_char={w_char:.2f} → base={base_duration:.2f}s"
            )
        else:
            estimated_duration = base_duration
        
        # Apply complexity and punctuation
        estimated_duration = max(0.0, estimated_duration * complexity_factor + punctuation_pause_sec)
        
        # Language multiplier
        lang = (language or "").lower()
        if lang.startswith("ru"):
            estimated_duration *= 1.12
        elif lang.startswith("en"):
            estimated_duration *= 1.00
        
        # Style/emotion multiplier
        style_factor = TextAnalysisUtils.style_emotion_speed_multiplier(style_prompt, emotion)
        estimated_duration *= style_factor
        
        # Adaptive biases
        if apply_biases:
            bias_lang = self._get_bias("language", lang or "")
            bias_voice = self._get_bias("voice", voice_name)
            bias_speaker = self._get_bias("speaker", speaker_id or "")
            bias_total = bias_lang * bias_voice * bias_speaker
            bias_total = max(0.7, min(1.3, bias_total))
            estimated_duration *= bias_total
        
        # Speed factor
        if speed and speed > 0:
            estimated_duration /= speed
            logger.debug(f"  Applied speed factor {speed:.2f}")
        
        logger.debug(f"  Final estimated duration: {estimated_duration:.2f}s")
        return max(0.1, estimated_duration)
    
    def record_duration_stats(
        self,
        text: str,
        voice_name: str,
        duration_seconds: float,
        language: str,
        style_prompt: Optional[str] = None,
        emotion: Optional[str] = None,
        speed: Optional[float] = None,
        speaker_id: Optional[str] = None
    ) -> None:
        """Update duration statistics with real synthesis data."""
        if duration_seconds <= 0.05:
            return
        
        text = text.strip()
        if not text:
            return
        
        words = TextAnalysisUtils.count_words(text)
        characters = TextAnalysisUtils.count_characters(text)
        if words == 0 and characters == 0:
            return
        
        speed_factor = speed if speed and speed > 0 else 1.0
        normalized_duration = duration_seconds / speed_factor
        if normalized_duration <= 0.05:
            return
        
        voice_stats = self.duration_database.get_or_create_stats(voice_name)
        old_wpm = voice_stats.words_per_minute
        old_cps = voice_stats.characters_per_second
        old_samples = voice_stats.total_samples
        
        smoothing_alpha = self.duration_smoothing_alpha
        if smoothing_alpha is not None and smoothing_alpha <= 0:
            smoothing_alpha = None
        
        with self._duration_stats_lock:
            self.duration_database.update_voice_stats(
                voice_name=voice_name,
                words=words,
                characters=characters,
                duration=normalized_duration,
                smoothing_alpha=smoothing_alpha
            )
            self._stats_update_counter += 1
            self._maybe_persist_duration_stats_locked()
        
        new_wpm = voice_stats.words_per_minute
        new_cps = voice_stats.characters_per_second
        
        logger.debug(
            f"Stats correction for voice '{voice_name}': "
            f"WPM {old_wpm:.1f} -> {new_wpm:.1f}, CPS {old_cps:.1f} -> {new_cps:.1f} "
            f"(samples: {old_samples} -> {voice_stats.total_samples}, alpha={smoothing_alpha})"
        )
        
        # Update adaptive biases
        try:
            predicted_unbiased = self.estimate_duration(
                text=text,
                voice_name=voice_name,
                language=language,
                style_prompt=style_prompt,
                emotion=emotion,
                speed=None,
                speaker_id=None,
                apply_biases=False
            ) or 0.0
            
            if predicted_unbiased > 0:
                target_ratio = normalized_duration / predicted_unbiased
                lang_key = (language or '').lower()
                self._update_bias('language', lang_key, target_ratio)
                self._update_bias('voice', voice_name, target_ratio)
                self._update_bias('speaker', speaker_id or '', target_ratio)
                
                logger.debug(
                    f"Bias updates (target={target_ratio:.3f}) → "
                    f"lang={self._get_bias('language', lang_key):.3f}, "
                    f"voice={self._get_bias('voice', voice_name):.3f}, "
                    f"speaker={self._get_bias('speaker', speaker_id or ''):.3f}"
                )
                
                with self._duration_stats_lock:
                    self._maybe_persist_duration_stats_locked()
        except Exception as e:
            logger.debug(f"Adaptive bias update failed: {e}")
    
    def _get_bias(self, scope: str, key: str) -> float:
        """Get bias factor for scope/key (default 1.0)."""
        if not key:
            return 1.0
        try:
            with self._bias_lock:
                return float(self._biases.get(scope, {}).get(key, 1.0))
        except Exception:
            return 1.0
    
    def _update_bias(self, scope: str, key: str, target_ratio: float) -> None:
        """EMA update of bias toward target_ratio (actual/predicted)."""
        if not key:
            return
        try:
            with self._bias_lock:
                old = float(self._biases.setdefault(scope, {}).get(key, 1.0))
                alpha = self._bias_alpha if self._bias_alpha > 0 else 0.6
                new = (1.0 - alpha) * old + alpha * float(target_ratio)
                new = max(0.7, min(1.3, new))
                self._biases[scope][key] = new
        except Exception as e:
            logger.debug(f"Bias update failed for {scope}:{key}: {e}")
    
    def save_duration_stats(self) -> None:
        """Persist current duration statistics to disk."""
        with self._duration_stats_lock:
            self._save_duration_stats_locked()
            self._stats_update_counter = 0
    
    def _save_duration_stats_locked(self) -> None:
        """Save duration statistics and adjustments to disk. Caller must hold lock."""
        # Save stats with embeddings
        embeddings_data: Dict[str, Any] = {}
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    embeddings_data = existing_data.get("embeddings", {})
            except Exception:
                embeddings_data = {}
        
        combined_data = {
            "voice_stats": self.duration_database.dict(),
            "embeddings": embeddings_data,
        }
        
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved runtime duration stats for {len(self.duration_database.voice_stats)} voices")
        
        # Save biases
        self.adjustments_file.parent.mkdir(parents=True, exist_ok=True)
        with self._bias_lock:
            biases_copy = {
                'speaker': dict(self._biases.get('speaker', {})),
                'voice': dict(self._biases.get('voice', {})),
                'language': dict(self._biases.get('language', {})),
            }
        with open(self.adjustments_file, 'w', encoding='utf-8') as f:
            json.dump({'biases': biases_copy}, f, indent=2, ensure_ascii=False)
        logger.debug("Persisted duration adjustments")
    
    def _maybe_persist_duration_stats_locked(self) -> None:
        """Auto-save duration stats when configured threshold is met."""
        if not self.duration_stats_auto_save:
            return
        
        interval = self.duration_stats_save_interval
        if not interval or interval <= 0:
            return
        
        if self._stats_update_counter >= interval:
            self._save_duration_stats_locked()
            self._stats_update_counter = 0
    
    def load_biases(self) -> None:
        """Load persisted biases from adjustments file."""
        if self.adjustments_file.exists():
            try:
                with open(self.adjustments_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                biases = data.get('biases', {}) or {}
                if isinstance(biases, dict):
                    with self._bias_lock:
                        for scope in ("speaker", "voice", "language"):
                            if isinstance(biases.get(scope), dict):
                                self._biases[scope].update({str(k): float(v) for k, v in biases[scope].items()})
                    logger.debug(
                        f"Loaded biases: speakers={len(self._biases['speaker'])}, "
                        f"voices={len(self._biases['voice'])}, languages={len(self._biases['language'])}"
                    )
            except Exception as e:
                logger.warning(f"Could not load biases: {e}")
    
    def get_voice_duration_stats(self, voice_name: Optional[str] = None) -> Dict[str, Any]:
        """Get duration statistics for a specific voice or all voices."""
        if voice_name:
            stats = self.duration_database.get_or_create_stats(voice_name)
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
                for voice_name, stats in self.duration_database.voice_stats.items()
            }
