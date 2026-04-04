"""
Factory for creating transcription and diarization services.
"""
from typing import Optional, Literal

from transcription.transcription_interface import TranscriptionInterface

class TranscriptionFactory:
    """Factory for creating transcription and diarization services."""
    
    @staticmethod
    def create_transcriber(
        transcription_system: Literal["pyannote_openai", "whisper", "openai", "whisperx", "assemblyai"],
        source_language: str,
        device: Optional[str] = None,
        **kwargs
    ) -> TranscriptionInterface:
        """
        Create a transcription service implementation based on the specified system.
        
        Args:
            transcription_system: The transcription system to use
            source_language: Source language code (e.g., 'en')
            device: Compute device ('cuda' or 'cpu')
            **kwargs: Additional parameters for the specific implementation
            
        Returns:
            An initialized transcription service implementation
            
        Raises:
            ValueError: If the specified transcription system is not supported
        """
        if transcription_system == "whisperx":
            from transcription.whisperx_transcriber import WhisperXTranscriber

            return WhisperXTranscriber(
                source_language=source_language,
                device=device,
                **kwargs
            )
        elif transcription_system in {"pyannote_openai", "openai", "whisper"}:
            from transcription.pyannote_openai_transcriber import PyAnnoteOpenAITranscriber

            backend_kwargs = dict(kwargs)
            if transcription_system in {"openai", "whisper"}:
                backend_kwargs["transcription_system"] = transcription_system

            return PyAnnoteOpenAITranscriber(
                source_language=source_language,
                device=device,
                **backend_kwargs
            )
        elif transcription_system == "assemblyai":
            from transcription.assemblyai_transcriber import AssemblyAITranscriber

            return AssemblyAITranscriber(
                source_language=source_language,
                device=device,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported transcription system: {transcription_system}")
