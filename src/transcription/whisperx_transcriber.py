"""
Implementation of transcription and diarization using WhisperX.
"""
import os
import time
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING

from transcription.transcription_interface import BaseTranscriber
from src.dubbing.core.log_config import get_logger

if TYPE_CHECKING:
    from src.dubbing.core.cache_manager import CacheManager

logger = get_logger(__name__)


class WhisperXTranscriber(BaseTranscriber):
    """Transcription and diarization service using WhisperX."""
    
    def __init__(
        self,
        source_language: str,
        device: Optional[str] = None,
        whisperx_model: str = "large-v3",
        cache_manager: Optional['CacheManager'] = None,
        **kwargs
    ):
        """
        Initialize the WhisperX transcription service.
        
        Args:
            source_language: Source language code (e.g., 'en')
            device: Compute device ('cuda' or 'cpu')
            whisperx_model: WhisperX model size to use for transcription
            cache_manager: Cache manager instance for organized caching
            **kwargs: Additional parameters
        """
        super().__init__(source_language, device, **kwargs)
        self.whisperx_model = whisperx_model
        self.cache_manager = cache_manager
        
        # Lazy loading of whisperx to avoid initial import overhead
        self._whisperx = None
    
    @property
    def whisperx(self):
        """Lazy-load whisperx module when needed."""
        if self._whisperx is None:
            import whisperx
            self._whisperx = whisperx
        return self._whisperx
    
    @property
    def name(self) -> str:
        """Return the name of the transcription service implementation."""
        return "WhisperX"
    
    def diarize_and_transcribe(
        self,
        audio_file: str,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]]]:
        """
        Perform speaker diarization and transcription on the audio file using WhisperX.
        WhisperX integrates both transcription and diarization in a single pipeline.
        
        Args:
            audio_file: Path to the audio file
            cache_key: Optional cache key for caching results
            use_cache: Whether to use cached results if available
            
        Returns:
            Tuple containing:
            - Dictionary mapping time ranges to speaker IDs
            - List of transcription segments with timing information
        """

        # If no cache_key is provided, generate one
        if cache_key is None:
            cache_key = self._generate_cache_key(audio_file, f"_{self.whisperx_model}")
            
        step_name = "whisperx_diarization_transcription"
        
        # Check if results are cached
        if use_cache and self.cache_manager and self.cache_manager.cache_exists(step_name, cache_key):
            logger.debug("Loading WhisperX diarization and transcription from cache...")
            cached_results = self.cache_manager.load_from_cache(step_name, cache_key)
            
            # Store for debug
            self.debug_data["diarization"] = cached_results["diarization"]
            self.debug_data["transcription"] = cached_results["transcription"]
            
            return cached_results["diarization"], cached_results["transcription"]
        
        logger.info(f"Running WhisperX transcription and diarization on {audio_file}...")
        logger.debug(f"Using model: {self.whisperx_model} on {self.device} device")
        
        try:
            # Step 1: Transcribe with WhisperX
            model = self.whisperx.load_model(
                self.whisperx_model,
                self.device,
                compute_type="float16" if self.device == "cuda" else "float32",
                language=self.source_language
            )
            
            # Transcribe audio
            result = model.transcribe(
                audio_file,
                batch_size=16,
                language=self.source_language
            )
            
            # Step 2: Align whisper output
            model_a, metadata = self.whisperx.load_align_model(
                language_code=self.source_language,
                device=self.device
            )
            
            result = self.whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio_file,
                self.device,
                return_char_alignments=False
            )
            
            # Step 3: Assign speaker labels (diarization)
            diarize_model = self.whisperx.DiarizationPipeline(
                use_auth_token=os.environ.get("HF_TOKEN", None),
                device=self.device
            )
            
            diarize_segments = diarize_model(audio_file)
            result = self.whisperx.assign_word_speakers(diarize_segments, result)
            
            # Process the result into our expected output formats
            speakers_rolls = self._process_diarization_result(result)
            transcription = self._process_whisperx_transcript(result)
            
            # Store for debug
            self.debug_data["diarization"] = speakers_rolls
            self.debug_data["transcription"] = transcription
            
            # Save results to cache
            if use_cache and self.cache_manager:
                cache_data = {
                    "diarization": speakers_rolls,
                    "transcription": transcription
                }
                self.cache_manager.save_to_cache(step_name, cache_key, cache_data)
            
            return speakers_rolls, transcription
            
        except Exception as e:
            logger.error(f"Error in WhisperX processing: {e}", exc_info=True)
            # Return empty results in case of error
            return {}, []
    
    def _process_diarization_result(self, result: Dict) -> Dict[Tuple[float, float], str]:
        """
        Process WhisperX diarization result into a dictionary of time ranges to speaker IDs.
        
        Args:
            result: WhisperX result with speaker information
            
        Returns:
            Dictionary mapping time ranges to speaker IDs
        """
        speakers_rolls = {}
        
        # Extract segments from the result
        segments = result.get("segments", [])
        
        for segment in segments:
            # Get the speaker for this segment
            speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            
            if start < end:  # Ensure valid time range
                speakers_rolls[(start, end)] = speaker
                logger.debug(f"Speaker {speaker}: from {start:.2f}s to {end:.2f}s")
                
        return speakers_rolls
    
    def _process_whisperx_transcript(self, result: Dict) -> List[Dict[str, Any]]:
        """
        Process WhisperX transcript into a list of segments with timing information.
        
        Args:
            result: WhisperX result with transcript information
            
        Returns:
            List of transcript segments with timing information
        """
        records = []
        
        # Extract segments from the result
        segments = result.get("segments", [])
        
        for segment in segments:
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
            
            if text and start < end:  # Ensure valid segment
                records.append({
                    "text": text,
                    "start": start,
                    "end": end,
                    "speaker": speaker  # Include speaker information directly
                })
        
        # Sort records by start time to ensure chronological order
        records.sort(key=lambda x: x["start"])
        
        return records 