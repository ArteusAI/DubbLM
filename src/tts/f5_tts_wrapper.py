from typing import Optional, Dict, Any, Union, List, Tuple
import os
from importlib.resources import files
import shutil
from pydub import AudioSegment
from pydub.silence import detect_nonsilent, split_on_silence
import glob
import tempfile
import hashlib
from datetime import datetime, timezone

from tts.tts_interface import TTSInterface
from .models import TTSSegmentData, SegmentAlignment, DiarizationSegment
from src.dubbing.core.log_config import get_logger

# Import torch for device detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import F5TTS dependencies, with error handling for missing packages
try:
    from f5_tts.api import F5TTS as F5TTSOriginal
    from f5_tts.infer.utils_infer import load_model
    from omegaconf import OmegaConf
    from hydra.utils import get_class
    from f5_tts.synthesize import CSynthesizer
    from f5_tts.utils.file_utils import cargar_configuracion
    from f5_tts.utils.audio_utils import array_a_wav #, normalizar_audio # normalizar_audio not used here
    from f5_tts.texto.cleaners import russian_cleaners # Example cleaner
    F5TTS_AVAILABLE = True
except ImportError:
    F5TTS_AVAILABLE = False

# Import Whisper ASR, with error handling for missing packages
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

logger = get_logger(__name__)


def get_default_device() -> str:
    """
    Auto-detect the best available device.
    
    Returns:
        String with device name ('cuda' or 'cpu')
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def remove_silence_edges(audio_segment, silence_thresh=-50, min_silence_len=100):
    """
    Remove silence from the beginning and end of an audio segment.
    
    Args:
        audio_segment: The audio segment to process
        silence_thresh: The silence threshold in dB
        min_silence_len: Minimum length of silence in ms
        
    Returns:
        AudioSegment with silence removed from edges
    """
    if len(audio_segment) < min_silence_len * 2:
        return audio_segment
        
    non_silent = detect_nonsilent(
        audio_segment, 
        min_silence_len=min_silence_len, 
        silence_thresh=silence_thresh
    )
    
    if not non_silent:
        return audio_segment
    
    start_trim = non_silent[0][0]
    end_trim = non_silent[-1][1]
    
    return audio_segment[start_trim:end_trim]


class F5TTSWrapper(TTSInterface):
    """
    F5 TTS wrapper. Handles a list of TTSSegmentData by synthesizing each and concatenating.
    Uses per-segment reference audio and text.
    Includes logic to create reference samples.
    """
    # Class-level cache for transcribed reference audio texts to avoid re-transcribing
    _transcribed_ref_text_cache: Dict[str, str] = {}
    
    def __init__(
        self,
        config_path: str = "F5_tts/configs/base.json", 
        model_path: str = "F5_tts/logs/base/pretrained_base.pth", 
        device: str = "cuda",
        whisper_model_name: str = "medium", # For transcribing reference audio if text not provided
        **kwargs: Any
    ):
        if not F5TTS_AVAILABLE:
            raise ImportError("F5 TTS package not found. Ensure it's installed and in PYTHONPATH.")
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper ASR package not installed. Use 'pip install whisper'.")

        self.config_path = config_path
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        self.synthesizer: Optional[CSynthesizer] = None
        self.hps: Optional[Any] = None 
        self.voice_mapping: Dict[str, Tuple[str, str]] = {} # speaker_id -> (ref_audio_path, ref_text)
        self.voice_prompt_mapping: Dict[str, str] = {} # F5 doesn't use text prompts for style

        self.whisper_model_name = whisper_model_name
        self.asr_model: Optional[Any] = None
        self._temp_files_created: List[str] = [] # Track files for cleanup

    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        # Adapting to store tuple (ref_audio_path, ref_text_path_or_literal_text)
        # For F5, this should ideally map to a tuple of (ref_audio_path, ref_text)
        # This method from interface expects Dict[str,str]. We can't change interface here.
        # User must ensure that if using this, the string is interpretable or points to a text file.
        # The primary way to pass ref_audio/text is via TTSSegmentData.
        logger.debug(f"F5 voice mapping (Dict[str,str]) stored: {len(mapping)} entries. This is for advanced/fallback use.")
        # self.voice_mapping = mapping # Re-evaluate how to best use this if needed

    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_prompt_mapping = mapping
        logger.debug(f"F5 voice prompt mapping stored: {len(mapping)} entries. Note: F5 derives style from reference audio.")
        
    def initialize(self) -> None:
        try:
            self.hps = cargar_configuracion(self.config_path)
            # Ensure text cleaners are appropriate for the model/language
            self.hps.data.text_cleaners = ["russian_cleaners"] # Example, adjust if needed
            self.synthesizer = CSynthesizer(self.hps, self.model_path, self.device)
            logger.info(f"F5 TTS initialized with model: {self.model_path} on device: {self.device}")

            if WHISPER_AVAILABLE:
                try:
                    self.asr_model = whisper.load_model(self.whisper_model_name, device=self.device)
                    logger.debug(f"Whisper ASR model '{self.whisper_model_name}' loaded for F5 reference transcription.")
                except Exception as e_asr:
                    logger.warning(f"Warning: Failed to load Whisper ASR model '{self.whisper_model_name}': {e_asr}. Reference transcription will not be available.")
                    self.asr_model = None
            else:
                logger.warning("Warning: Whisper ASR package not available. F5 reference transcription from audio only will not work.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize F5 TTS: {str(e)}")
    
    def _transcribe_ref_audio(self, audio_path: str) -> str:
        if not self.asr_model:
            # logger.info(f"  F5 transcribe: ASR model not available, cannot transcribe {audio_path}")
            return "" # Return empty if no ASR
        
        cache_key = hashlib.md5(audio_path.encode() + str(os.path.getmtime(audio_path)).encode()).hexdigest()
        if cache_key in F5TTSWrapper._transcribed_ref_text_cache:
            # logger.info(f"  F5 transcribe: Using cached transcription for {audio_path}")
            return F5TTSWrapper._transcribed_ref_text_cache[cache_key]

        logger.debug(f"  F5 transcribe: Transcribing reference audio {audio_path}...")
        try:
            result = self.asr_model.transcribe(audio_path, language=self.hps.data.language.split('_')[0], fp16=False)
            text = result["text"].strip()
            if not text:
                logger.warning(f"  F5 transcribe: Warning - Transcription of {audio_path} resulted in empty text.")
            F5TTSWrapper._transcribed_ref_text_cache[cache_key] = text
            return text
        except Exception as e_transcribe:
            logger.error(f"  F5 transcribe: Error transcribing {audio_path}: {e_transcribe}")
            return "" # Return empty on error

    def _create_temp_file(self, suffix=".wav") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="f5_temp_")
        os.close(fd) # Close the file descriptor as we just need the path
        self._temp_files_created.append(path)
        return path
    
    def create_reference_sample(
        self, 
        speaker_id: str, 
        original_full_audio_path: str,
        speaker_segments_timestamps: List[Tuple[float, float]], # List of (start, end) for this speaker
        target_duration_s: float = 7.0, # Aim for ~7s reference for F5
        min_segment_len_s: float = 2.0
    ) -> Tuple[Optional[str], Optional[str]]:
        """Creates a reference audio sample for a speaker for F5 TTS.
        Args:
            speaker_id: Identifier of the speaker.
            original_full_audio_path: Path to the full original audio file.
            speaker_segments_timestamps: List of (start_time, end_time) tuples for this speaker's segments.
            target_duration_s: Ideal target duration for the reference sample.
            min_segment_len_s: Minimum duration of a segment to be considered.
        Returns:
            Tuple (path_to_reference_wav, reference_text). Paths are temporary.
        """
        if not speaker_segments_timestamps:
            logger.warning(f"  F5 ref creation: No segments for speaker {speaker_id}, cannot create reference.")
            return None, None

        best_segment_audio = AudioSegment.empty()
        try:
            full_audio = AudioSegment.from_file(original_full_audio_path)
        except Exception as e:
            logger.error(f"  F5 ref creation: Error loading full audio {original_full_audio_path}: {e}")
            return None, None

        # Find a good segment or combine segments to reach target_duration_s
        # Prioritize single segments close to target_duration_s
        candidate_segments = []
        for start_s, end_s in speaker_segments_timestamps:
            dur_s = end_s - start_s
            if dur_s >= min_segment_len_s:
                candidate_segments.append({"start": start_s, "end": end_s, "duration": dur_s})
        
        if not candidate_segments:
            logger.warning(f"  F5 ref creation: No suitable long segments for speaker {speaker_id}.")
            return None, None

        # Sort by closeness to target_duration_s, then by duration (longer is better if equally close)
        candidate_segments.sort(key=lambda s: (abs(s["duration"] - target_duration_s), -s["duration"]))
        
        chosen_segment_info = candidate_segments[0]
        ref_audio_chunk = full_audio[int(chosen_segment_info["start"]*1000):int(chosen_segment_info["end"]*1000)]

        # Trim or pad to be closer to target_duration_s, remove silence edges
        # ref_audio_chunk = ref_audio_chunk[:int(target_duration_s * 1000)] # Simple cut for now
        # A more sophisticated approach would be to find a segment of target_duration_s
        # or combine smaller ones, then remove silence. 
        # For now, let's just use the best single segment found.

        # Remove silence from edges
        if len(ref_audio_chunk) > 200: # Only if long enough to have meaningful silence
            non_silent_ranges = detect_nonsilent(ref_audio_chunk, min_silence_len=100, silence_thresh=-45)
            if non_silent_ranges:
                start_trim = non_silent_ranges[0][0]
                end_trim = non_silent_ranges[-1][1]
                ref_audio_chunk = ref_audio_chunk[start_trim:end_trim]

        if len(ref_audio_chunk) == 0:
            logger.warning(f"  F5 ref creation: Chosen segment for {speaker_id} became empty after silence removal.")
            return None, None

        temp_ref_path = self._create_temp_file(suffix=f"_spk_{speaker_id}_ref.wav")
        ref_audio_chunk.export(temp_ref_path, format="wav")
        
        ref_text = self._transcribe_ref_audio(temp_ref_path)
        if not ref_text:
            logger.warning(f"  F5 ref creation: Failed to get/transcribe text for reference audio {temp_ref_path} for speaker {speaker_id}. Voice cloning may fail or be poor.")
            # Fallback: use a placeholder if transcription fails completely but audio exists
            ref_text = "Reference speech for voice cloning." 

        logger.debug(f"  F5 ref creation: Created reference for {speaker_id}: {os.path.basename(temp_ref_path)}, Text: '{ref_text[:50]}...'")
        return temp_ref_path, ref_text

    def _synthesize_single_segment(
        self,
        segment_data: TTSSegmentData,
        temp_output_path: str,
        language: str 
    ) -> None:
        if not self.synthesizer or not self.hps:
            raise RuntimeError("F5 TTS synthesizer not initialized.")

        text_to_synthesize = segment_data.text
        ref_audio_p = segment_data.reference_audio_path
        ref_text_p = segment_data.reference_text

        if not ref_audio_p or not os.path.isfile(ref_audio_p):
            raise ValueError(f"F5 TTS requires a valid reference_audio_path for speaker {segment_data.speaker}, but got '{ref_audio_p}'")
        if not ref_text_p:
            logger.warning(f"  F5: Warning - Reference text not provided for speaker {segment_data.speaker} and segment. Transcribing ref audio.")
            ref_text_p = self._transcribe_ref_audio(ref_audio_p)
            if not ref_text_p: # Still no text
                logger.warning(f"  F5: Critical Warning - Failed to obtain reference text for {ref_audio_p}. Using placeholder. Quality will be impacted.")
                ref_text_p = "Sample audio for reference."
        
        # F5 specific cleaning - ensure it's done before synthesis
        cleaned_text = russian_cleaners(text_to_synthesize) # Assuming russian_cleaners, adjust if model uses others
        if not cleaned_text.strip():
            logger.warning(f"  F5: Warning - Text for speaker {segment_data.speaker} became empty after cleaning: '{text_to_synthesize}'. Synthesizing silence.")
            AudioSegment.silent(duration=100).export(temp_output_path, format="wav") # 100ms silence
            return

        try:
            audio_array_np = self.synthesizer.synthesize(
                texto=cleaned_text, # Use cleaned text
                ref_audio=ref_audio_p,
                ref_texto=ref_text_p,
                # locutor_id is not typically used with ref_audio for XTTS-like voice cloning in F5
            )
            
            if audio_array_np is None or not isinstance(audio_array_np, np.ndarray) or audio_array_np.size == 0:
                raise RuntimeError(f"F5 TTS returned no audio data for speaker {segment_data.speaker}.")

            sample_rate = self.hps.data.sampling_rate
            array_a_wav(audio_array_np, temp_output_path, sr=sample_rate)
        except Exception as e_f5_synth:
            raise RuntimeError(f"F5 TTS synthesis failed for speaker {segment_data.speaker}: {e_f5_synth}")
    
    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: str = "ru", # F5 default is often Russian, ensure this matches model
        **kwargs: Any
    ) -> List[SegmentAlignment]:
        if not self.synthesizer:
            raise RuntimeError("F5 TTS synthesizer not initialized.")
        if not segments_data:
            logger.warning("Warning: No segments provided to F5TTSWrapper.synthesize.")
            return []

        alignments = []
        # Ensure dubbing_tool provides these if it wants per-segment ref audio creation by F5 wrapper
        original_full_audio_path_from_kwargs = kwargs.get("original_full_audio_path")
        speaker_timestamps_from_kwargs = kwargs.get("speaker_timestamps_map") # Dict[speaker_id, List[Tuple[float,float]]]

        try:
            for i, segment in enumerate(segments_data):
                temp_segment_file_path = self._create_temp_file(suffix=f"_seg_{i}_{segment.speaker}.wav")
                logger.debug(f"F5: Synthesizing segment {i+1}/{len(segments_data)} for speaker '{segment.speaker}'")
                
                current_segment_data = segment
                # If reference audio/text are not in segment, try to create/fetch them
                if not current_segment_data.reference_audio_path or not current_segment_data.reference_text:
                    if original_full_audio_path_from_kwargs and speaker_timestamps_from_kwargs and segment.speaker in speaker_timestamps_from_kwargs:
                        logger.debug(f"  F5: Ref audio/text missing for segment {i+1} (speaker {segment.speaker}). Attempting to create.")
                        ref_audio, ref_text = self.create_reference_sample(
                            speaker_id=segment.speaker,
                            original_full_audio_path=original_full_audio_path_from_kwargs,
                            speaker_segments_timestamps=speaker_timestamps_from_kwargs[segment.speaker]
                        )
                        # Update the segment data model for this call if ref created. 
                        # This is a mutable update to a Pydantic model field, which is fine.
                        if ref_audio and ref_text:
                            current_segment_data.reference_audio_path = ref_audio
                            current_segment_data.reference_text = ref_text
                        else:
                            logger.warning(f"  F5: Failed to create reference for segment of speaker {segment.speaker}. Synthesis may fail or be poor.")
                    else:
                        logger.warning(f"  F5: Ref audio/text missing for seg {i+1} ({segment.speaker}), and insufficient data to create it. Needs reference_audio_path and reference_text in TTSSegmentData.")
                        # Allow _synthesize_single_segment to raise error if still missing critical refs

                try:
                    self._synthesize_single_segment(current_segment_data, temp_segment_file_path, language)
                    
                    # Get duration of the synthesized audio
                    audio_segment = AudioSegment.from_wav(temp_segment_file_path)
                    duration = len(audio_segment) / 1000.0  # Convert to seconds
                    
                    # Save to output path if specified
                    if segment.output_path:
                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(segment.output_path), exist_ok=True)
                        shutil.copy(temp_segment_file_path, segment.output_path)
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
                    
                except Exception as e_segment_synth:
                    logger.error(f"Error synthesizing F5 segment {i+1} for speaker '{segment.speaker}': {e_segment_synth}. Skipping.")
            
            logger.info(f"F5: Synthesized {len(alignments)} segments successfully")
            return alignments

        finally:
            # Cleanup all tracked temporary files at the end
            for f_path in self._temp_files_created:
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except Exception:
                        pass # Ignore if already removed or error during removal
            self._temp_files_created.clear()
    
    def is_available(self) -> bool:
        return F5TTS_AVAILABLE and WHISPER_AVAILABLE and self.synthesizer is not None

    def cleanup(self) -> None:
        if self.synthesizer is not None:
            logger.debug("F5 TTS cleanup: Releasing synthesizer and ASR references.")
            del self.synthesizer
            self.synthesizer = None
            if self.hps: del self.hps
            self.hps = None
            if self.asr_model: del self.asr_model
            self.asr_model = None
            
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.debug("  Attempted torch.cuda.empty_cache() for F5 TTS.")
                except Exception as e_cuda_clean:
                    logger.warning(f"  Warning during CUDA cache clear for F5: {e_cuda_clean}")
        # Clear tracked temp files that might have been missed
        for f_path in self._temp_files_created:
            if os.path.exists(f_path):
                try: os.remove(f_path)
                except Exception: pass
        self._temp_files_created.clear()
        
    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: str = "ru"
    ) -> Optional[float]:
        """
        Estimate the duration in seconds for a given text segment.
        
        Args:
            segment_data: TTSSegmentData object containing text and voice parameters
            language: Target language code (e.g., "ru")
            
        Returns:
            Estimated duration in seconds, or None if estimation is not possible
        """
        if not segment_data.text or not segment_data.text.strip():
            return 0.0

        # Simple estimation based on character count and average speaking rate
        # F5 TTS typically speaks at around 120-180 words per minute depending on language
        text = segment_data.text.strip()
        
        # Count words
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        
        # Estimate duration (assuming ~150 WPM average for Russian/multilingual)
        base_wpm = 150.0
        estimated_duration = (word_count / base_wpm) * 60
        
        # Apply speed factor if specified
        if segment_data.speed and segment_data.speed > 0:
            estimated_duration /= segment_data.speed
        
        return max(0.1, estimated_duration)  # Minimum 0.1 seconds
        