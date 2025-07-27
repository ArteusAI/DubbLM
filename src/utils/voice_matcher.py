"""Voice matching utility for TTS systems."""

from typing import Optional, Dict, List, Union
from pathlib import Path
import numpy as np
from collections import Counter

from src.utils.audio_embedder import AudioEmbedder
from src.dubbing.core.log_config import get_logger

# Optional imports for voice matching features
try:
    from scipy.spatial.distance import cosine
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logger = get_logger(__name__)


class VoiceMatcher:
    """Handles voice matching and embedding operations for TTS systems."""
    
    def __init__(self, audio_embedder: Optional[AudioEmbedder] = None, enable_matching: bool = True):
        """
        Initialize VoiceMatcher.
        
        Args:
            audio_embedder: AudioEmbedder instance for extracting embeddings
            enable_matching: Whether to enable voice matching functionality
        """
        self.audio_embedder = audio_embedder
        self.enable_matching = enable_matching and SCIPY_AVAILABLE and PYDUB_AVAILABLE
        self.sample_embeddings: Dict[str, np.ndarray] = {}
        
        if enable_matching and not SCIPY_AVAILABLE:
            logger.warning("Warning: scipy not available. Voice matching disabled.")
            self.enable_matching = False
        
        if enable_matching and not PYDUB_AVAILABLE:
            logger.warning("Warning: pydub not available. Voice matching disabled.")
            self.enable_matching = False

    def extract_embedding_for_audio_file(self, audio_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Extracts speaker embedding from a given audio file using AudioEmbedder."""
        if not self.enable_matching or not self.audio_embedder:
            logger.debug("Voice matching features disabled or AudioEmbedder not initialized.")
            return None
        
        p_audio_path = Path(audio_path)
        if not p_audio_path.exists():
            logger.error(f"Audio file not found for embedding extraction: {p_audio_path}")
            return None

        audio_duration = self._get_audio_duration_seconds(p_audio_path)
        MIN_AUDIO_DURATION_S = 0.5
        if audio_duration is None or audio_duration < MIN_AUDIO_DURATION_S:
            logger.warning(f"Audio file {p_audio_path.name} is too short ({audio_duration}s) or duration unknown.")
            return None

        try:
            # Try to use direct file processing if available (more efficient with Resemblyzer)
            if hasattr(self.audio_embedder, 'extract_embedding_from_file'):
                embedding = self.audio_embedder.extract_embedding_from_file(str(p_audio_path))
            else:
                # Fallback to AudioSegment approach
                audio_segment = AudioSegment.from_file(str(p_audio_path))
                embedding = self.audio_embedder.extract_embedding(audio_segment)
            
            if embedding is not None:
                logger.debug(f"Successfully extracted Resemblyzer embedding for {p_audio_path.name}.")
            else:
                logger.warning(f"Failed to extract embedding for {p_audio_path.name}.")
            return embedding
        
        except Exception as e:
            logger.error(f"Error processing audio file {p_audio_path.name} for embedding: {e}")
            return None

    def extract_multiple_embeddings(self, audio_path: Union[str, Path], 
                                  num_segments: int = 3, 
                                  segment_duration_ms: int = 3000) -> List[np.ndarray]:
        """
        Extract multiple embeddings from different parts of an audio file.
        
        Args:
            audio_path: Path to the audio file
            num_segments: Number of segments to extract (default 3)
            segment_duration_ms: Duration of each segment in milliseconds (default 3000)
            
        Returns:
            List of embeddings extracted from different parts of the audio
        """
        if not self.enable_matching or not self.audio_embedder:
            return []
        
        p_audio_path = Path(audio_path)
        if not p_audio_path.exists():
            logger.error(f"Audio file not found: {p_audio_path}")
            return []
        
        try:
            # Load the full audio
            audio = AudioSegment.from_file(str(p_audio_path))
            audio_duration_ms = len(audio)
            
            # Check if audio is long enough
            min_duration_ms = segment_duration_ms * num_segments
            if audio_duration_ms < min_duration_ms:
                logger.warning(f"Audio too short ({audio_duration_ms}ms) for {num_segments} segments of {segment_duration_ms}ms each")
                # Fall back to single embedding
                embedding = self.audio_embedder.extract_embedding(audio)
                return [embedding] if embedding is not None else []
            
            embeddings = []
            
            # For 3 segments: extract from start, middle, and end
            if num_segments == 3:
                # Start segment
                start_segment = audio[:segment_duration_ms]
                
                # Middle segment
                middle_start = (audio_duration_ms - segment_duration_ms) // 2
                middle_segment = audio[middle_start:middle_start + segment_duration_ms]
                
                # End segment
                end_segment = audio[-segment_duration_ms:]
                
                segments = [("start", start_segment), ("middle", middle_segment), ("end", end_segment)]
            else:
                # For other numbers, distribute segments evenly
                segments = []
                step = (audio_duration_ms - segment_duration_ms) // max(1, num_segments - 1)
                
                for i in range(num_segments):
                    start_ms = i * step
                    segment = audio[start_ms:start_ms + segment_duration_ms]
                    segments.append((f"segment_{i}", segment))
            
            # Extract embeddings for each segment
            for segment_name, segment_audio in segments:
                embedding = self.audio_embedder.extract_embedding(segment_audio)
                if embedding is not None:
                    embeddings.append(embedding)
                    logger.debug(f"Extracted embedding from {segment_name} of {p_audio_path.name}")
                else:
                    logger.warning(f"Failed to extract embedding from {segment_name}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error extracting multiple embeddings from {p_audio_path.name}: {e}")
            return []

    def find_best_matching_voice(self, reference_embedding: np.ndarray) -> Optional[str]:
        """Finds the best matching voice from sample embeddings."""
        if not self.sample_embeddings:
            return None

        best_match_voice = None
        best_distance = float('inf')

        for voice_name, sample_embedding in self.sample_embeddings.items():
            try:
                distance = cosine(reference_embedding, sample_embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_match_voice = voice_name
            except Exception as e:
                logger.error(f"Error calculating distance for {voice_name}: {e}")
                continue

        return best_match_voice
    
    def find_best_matching_voice_multi_segment(self, reference_embeddings: List[np.ndarray], exclude_voices: Optional[List[str]] = None) -> Optional[str]:
        """
        Find the best matching voice using multiple reference embeddings.
        Uses voting to determine which voice prevails across segments.
        
        Args:
            reference_embeddings: List of embeddings from different segments
            exclude_voices: A list of voice names to exclude from consideration.
            
        Returns:
            The voice name that best matches across all segments
        """
        if not self.sample_embeddings or not reference_embeddings:
            return None
        
        current_sample_embeddings = {
            voice: emb for voice, emb in self.sample_embeddings.items()
            if not exclude_voices or voice not in exclude_voices
        }

        if not current_sample_embeddings:
            logger.warning("Warning: All available voices are excluded or no sample embeddings available.")
            return None

        # For each reference embedding, find the best matching voice
        voice_votes = Counter()
        voice_distances = {}  # Store average distances for tie-breaking
        
        for ref_embedding in reference_embeddings:
            best_voice = None
            best_distance = float('inf')
            
            # Find the best matching voice for this embedding
            for voice_name, sample_embedding in current_sample_embeddings.items():
                try:
                    distance = cosine(ref_embedding, sample_embedding)
                    if distance < best_distance:
                        best_distance = distance
                        best_voice = voice_name
                except Exception as e:
                    logger.error(f"Error calculating distance for {voice_name}: {e}")
                    continue
            
            if best_voice:
                voice_votes[best_voice] += 1
                # Accumulate distances for averaging
                if best_voice not in voice_distances:
                    voice_distances[best_voice] = []
                voice_distances[best_voice].append(best_distance)
        
        if not voice_votes:
            return None
        
        # Find the voice(s) with the most votes
        max_votes = max(voice_votes.values())
        top_voices = [voice for voice, votes in voice_votes.items() if votes == max_votes]
        
        # If there's a clear winner, return it
        if len(top_voices) == 1:
            winner = top_voices[0]
            avg_distance = sum(voice_distances[winner]) / len(voice_distances[winner])
            logger.debug(f"Voice '{winner}' won with {max_votes}/{len(reference_embeddings)} votes (avg distance: {avg_distance:.3f})")
            return winner
        
        # If there's a tie, use average distance as tie-breaker
        best_voice = None
        best_avg_distance = float('inf')
        
        for voice in top_voices:
            avg_distance = sum(voice_distances[voice]) / len(voice_distances[voice])
            if avg_distance < best_avg_distance:
                best_avg_distance = avg_distance
                best_voice = voice
        
        logger.debug(f"Voice '{best_voice}' won tie-breaker with {max_votes}/{len(reference_embeddings)} votes (avg distance: {best_avg_distance:.3f})")
        return best_voice

    def _get_audio_duration_seconds(self, audio_path: Union[str, Path]) -> Optional[float]:
        """Gets the duration of an audio file in seconds."""
        try:
            # Try using torchaudio if available
            try:
                import torchaudio
                info = torchaudio.info(str(audio_path))
                return info.num_frames / info.sample_rate
            except (ImportError, AttributeError):
                pass
            
            # Fall back to pydub
            if PYDUB_AVAILABLE:
                audio_segment = AudioSegment.from_file(str(audio_path))
                return audio_segment.duration_seconds
            
            logger.warning("Warning: Cannot determine audio duration. No suitable audio library available.")
            return None
            
        except Exception as e:
            logger.error(f"Error getting duration for {audio_path}: {e}")
            return None

    def set_sample_embeddings(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Set the sample embeddings for voice matching."""
        self.sample_embeddings = embeddings
        logger.debug(f"Set {len(embeddings)} sample embeddings for voice matching")

    def clear(self) -> None:
        """Clear all stored embeddings."""
        self.sample_embeddings.clear() 