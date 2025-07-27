import os
from typing import Optional

import numpy as np
from pydub import AudioSegment

from src.dubbing.core.log_config import get_logger

# Optional imports for voice embedding
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False

logger = get_logger(__name__)


class AudioEmbedder:
    """Handles audio embedding generation using Resemblyzer for voice similarity."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the AudioEmbedder with Resemblyzer.

        Args:
            device: Compute device (not used by Resemblyzer, kept for compatibility).
        """
        self.voice_encoder: Optional[VoiceEncoder] = None
        self.embedding_type = "resemblyzer"
        self._initialize_voice_encoder()

    def _initialize_voice_encoder(self):
        """Initialize the Resemblyzer voice encoder."""
        if not RESEMBLYZER_AVAILABLE:
            logger.error("Resemblyzer is not available. Please install it with: pip install resemblyzer")
            return
        
        try:
            self.voice_encoder = VoiceEncoder()
            logger.debug("Initialized Resemblyzer voice encoder")
        except Exception as e:
            logger.error(f"Failed to initialize Resemblyzer voice encoder: {e}")
            self.voice_encoder = None

    def extract_embedding(self, audio_segment: AudioSegment) -> Optional[np.ndarray]:
        """
        Extract embedding from a given AudioSegment using Resemblyzer.

        Args:
            audio_segment: The Pydub AudioSegment to process.

        Returns:
            A numpy array representing the 256-dimensional voice embedding, or None if extraction fails.
        """
        if not RESEMBLYZER_AVAILABLE or self.voice_encoder is None:
            logger.error("Resemblyzer voice encoder not initialized")
            return None

        if len(audio_segment) < 500:  # Minimum 500ms needed
            logger.warning("Skipping embedding due to insufficient audio length")
            return None

        try:
            # Convert AudioSegment to numpy array
            # Resemblyzer expects 16kHz mono audio
            audio_16khz = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # Convert to numpy array and normalize
            audio_data = np.array(audio_16khz.get_array_of_samples(), dtype=np.float32)
            
            # Normalize audio data to [-1, 1] range
            if audio_data.dtype == np.int16:
                audio_data = audio_data / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data / 2147483648.0
            elif audio_data.max() > 1.0 or audio_data.min() < -1.0:
                # If audio is not already normalized
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Extract embedding using Resemblyzer
            embedding = self.voice_encoder.embed_utterance(audio_data)
            
            logger.debug(f"Extracted Resemblyzer embedding with shape: {embedding.shape}")
            return embedding

        except Exception as e:
            logger.error(f"Error extracting Resemblyzer embedding: {e}")
            return None

    def extract_embedding_from_file(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract embedding directly from an audio file using Resemblyzer's preprocessing.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            A numpy array representing the voice embedding, or None if extraction fails
        """
        if not RESEMBLYZER_AVAILABLE or self.voice_encoder is None:
            logger.error("Resemblyzer voice encoder not initialized")
            return None
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None
        
        try:
            # Use Resemblyzer's preprocessing function
            wav = preprocess_wav(audio_path)
            
            if len(wav) < 8000:  # Less than 0.5 seconds at 16kHz
                logger.warning(f"Audio file {audio_path} is too short for embedding extraction")
                return None
            
            # Extract embedding
            embedding = self.voice_encoder.embed_utterance(wav)
            
            logger.debug(f"Extracted Resemblyzer embedding from file with shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding from file {audio_path}: {e}")
            return None 