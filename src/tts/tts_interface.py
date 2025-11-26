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
        segments_data: List[TTSSegmentData],
        language: Optional[str] = None,
        **kwargs: Any
    ) -> List[SegmentAlignment]:
        """
        Synthesize speech for a list of text segments.

        Args:
            segments_data: A list of TTSSegmentData objects. Each object contains
                           speaker ID, text, and optional per-segment parameters like
                           emotion, speed, voice override, style_prompt,
                           reference_audio_path, reference_text, and output_path.
            language: Target language code (e.g., "en"). If not provided, uses target_language from init.
            **kwargs: Additional global parameters for the specific TTS system.
            
        Returns:
            List of SegmentAlignment objects mapping original segments to synthesized audio
        """
        pass
    
    @abstractmethod
    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: Optional[str] = None
    ) -> Optional[float]:
        """
        Estimate the duration in seconds for a given text segment.
        
        Args:
            segment_data: TTSSegmentData object containing text and voice parameters
            language: Target language code (uses target_language from init if not specified)
            
        Returns:
            Estimated duration in seconds, or None if estimation is not possible
        """
        pass
    
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
    def clone_voice(self, audio_path: str, voice_id: str) -> str:
        """
        Clone a voice using the TTS provider's API.
        
        Args:
            audio_path: Path to reference audio file
            voice_id: Desired ID for the cloned voice
            
        Returns:
            voice_id: The ID of the cloned voice
        """
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up any resources or temporary files used by the TTS system.
        This should be called when the TTS system is no longer needed.
        """
        pass # Specific implementation in derived classes 