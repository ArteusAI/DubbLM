from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel, Field

class TTSSegmentData(BaseModel):
    """Data model for a single text segment to be synthesized."""
    speaker: str = Field(..., description="Identifier for the speaker (e.g., \"SPEAKER_00\").")
    text: str = Field(..., description="The text to be synthesized for this speaker.")
    
    emotion: Optional[str] = Field(None, description="Emotion for this segment (e.g., \"Happy\", \"Sad\").")
    speed: Optional[float] = Field(None, description="Speed factor for this segment (e.g., 1.0 for normal, 1.2 for faster).")
    voice: Optional[str] = Field(None, description="Specific voice name to use for this segment, overriding any global speaker-to-voice mappings.")
    style_prompt: Optional[str] = Field(None, description="Specific style prompt for this segment, overriding any global speaker-to-style_prompt mappings.")
    reference_audio_path: Optional[str] = Field(None, description="Path to a reference audio file for voice cloning for this specific segment/speaker.")
    reference_text: Optional[str] = Field(None, description="Text corresponding to the reference_audio_path, if required by the TTS system.")
    output_path: Optional[str] = Field(None, description="Path to save the synthesized audio for this specific segment.")

    class Config:
        extra = 'allow' # Allow other kwargs to be passed through if a TTS system needs them beyond this model


class VoiceDurationStats(BaseModel):
    """Statistics for voice duration estimation."""
    voice_name: str = Field(..., description="Name of the voice")
    words_per_minute: float = Field(..., description="Average words per minute for this voice")
    characters_per_second: float = Field(..., description="Average characters per second for this voice")
    total_samples: int = Field(default=0, description="Number of samples used to calculate statistics")
    total_words: int = Field(default=0, description="Total words in all samples")
    total_characters: int = Field(default=0, description="Total characters in all samples")
    total_duration_seconds: float = Field(default=0.0, description="Total duration of all samples in seconds")
    
    def update_stats(self, words: int, characters: int, duration: float) -> None:
        """Update statistics with new sample data."""
        self.total_samples += 1
        self.total_words += words
        self.total_characters += characters
        self.total_duration_seconds += duration
        
        # Recalculate averages
        if self.total_duration_seconds > 0:
            self.words_per_minute = (self.total_words / self.total_duration_seconds) * 60
            self.characters_per_second = self.total_characters / self.total_duration_seconds


class VoiceDurationDatabase(BaseModel):
    """Database of voice duration statistics."""
    voice_stats: Dict[str, VoiceDurationStats] = Field(default_factory=dict)
    
    def get_or_create_stats(self, voice_name: str) -> VoiceDurationStats:
        """Get existing stats or create new ones for a voice."""
        if voice_name not in self.voice_stats:
            self.voice_stats[voice_name] = VoiceDurationStats(
                voice_name=voice_name,
                words_per_minute=150.0,  # Default fallback
                characters_per_second=12.5  # Default fallback
            )
        return self.voice_stats[voice_name]
    
    def update_voice_stats(self, voice_name: str, words: int, characters: int, duration: float) -> None:
        """Update statistics for a voice with new sample data."""
        stats = self.get_or_create_stats(voice_name)
        stats.update_stats(words, characters, duration)


class DiarizationSegment(BaseModel):
    """Segment detected during diarization of synthesized audio."""
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds") 
    speaker: str = Field(..., description="Detected speaker ID")
    text: str = Field(..., description="Transcribed text for this segment")
    confidence: float = Field(default=1.0, description="Confidence score for this segment")
    
    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end_time - self.start_time


class SegmentAlignment(BaseModel):
    """Alignment between original segment and diarized segment."""
    original_segment: TTSSegmentData = Field(..., description="Original segment from input")
    diarized_segment: DiarizationSegment = Field(..., description="Corresponding diarized segment")
    alignment_confidence: float = Field(..., description="Confidence of this alignment")

