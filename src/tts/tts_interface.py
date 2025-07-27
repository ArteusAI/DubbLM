from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from .models import TTSSegmentData, SegmentAlignment # Import the Pydantic models

class TTSInterface(ABC):
    """
    Abstract base class for text-to-speech systems.
    All TTS implementations should inherit from this class.
    """
    
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the TTS system with provider-specific arguments."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def initialize(self) -> None:
        """Perform any necessary setup for the TTS client (e.g., API connections, model loading)."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def synthesize(
        self,
        segments_data: List[TTSSegmentData], # Use the Pydantic model here
        language: str = "en", # Global language for the synthesis batch (if applicable)
        # Removed global reference_audio_path and reference_text
        **kwargs: Any # For any other global parameters a specific TTS system might need
    ) -> List[SegmentAlignment]:
        """
        Synthesize speech for a list of text segments.

        Args:
            segments_data: A list of TTSSegmentData objects. Each object contains
                           speaker ID, text, and optional per-segment parameters like
                           emotion, speed, voice override, style_prompt,
                           reference_audio_path, reference_text, and output_path.
            language: Target language code (e.g., "en"). Applied globally if the TTS supports it.
            **kwargs: Additional global parameters for the specific TTS system.
            
        Returns:
            List of SegmentAlignment objects mapping original segments to synthesized audio
        """
        pass # Specific implementation in derived classes
    
    @abstractmethod
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
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the TTS system is available and properly initialized."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        """Set a global mapping of speaker IDs to voice names for the TTS system."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        """Set a global mapping of speaker IDs to voice style prompts for the TTS system."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up any resources or temporary files used by the TTS system.
        This should be called when the TTS system is no longer needed.
        """
        pass # Specific implementation in derived classes 