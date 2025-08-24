"""Main SmartDubbing orchestrator class."""

# --- Suppress noisy library logs BEFORE any imports ---
import os
# Set TensorFlow log level to suppress INFO and WARNING messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Changed to 3 for even more suppression
# Disable oneDNN custom operations log.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress CUDA registration warnings
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
# --- End of suppression block ---

import time
import torch
import warnings
import shutil
import subprocess
from typing import Dict, List, Tuple, Optional, Any, Literal, Union
from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment

# Disable all warnings for a cleaner output.
warnings.filterwarnings("ignore")

# Import our components
from .config import DubbingConfig
from .cache_manager import CacheManager
from ..audio.audio_processor import AudioProcessor
from ..audio.speaker_processor import SpeakerProcessor
from ..video.video_processor import VideoProcessor
from ..debug.performance_tracker import PerformanceTracker
from ..debug.debug_generator import DebugGenerator
from ..debug.reporter import SpeakerReporter
from ..utils.subtitle_utils import SubtitleManager
from .log_config import get_logger

# Import existing factories and interfaces
from tts.tts_factory import TTSFactory
from translation.translator_factory import TranslatorFactory
from transcription.transcription_factory import TranscriptionFactory
from speechbrain.inference.interfaces import foreign_class

# Disable warnings
warnings.filterwarnings("ignore")

# Get logger
logger = get_logger(__name__)


class SmartDubbing:
    """
    A video dubbing system that transcribes, translates, and synthesizes speech for videos.
    Uses context-aware translation to produce more natural-sounding results.
    
    This is the main orchestrator class that coordinates all components.
    """
    
    def __init__(self, config: DubbingConfig):
        """
        Initialize the SmartDubbing system.
        
        Args:
            config: Configuration object containing all settings
        """
        self.config = config
        self.tts_system_mapping = self.config.get('tts_system_mapping') or {}
        self.voice_prompt = self.config.get('voice_prompt') or {}
        
        # Initialize core components
        self.cache_manager = CacheManager(
            use_cache=not config.get('no_cache', False),
            input_file=config.get('input')
        )
        self.performance_tracker = PerformanceTracker()
        
        # Initialize processors
        self.audio_processor = AudioProcessor(self.cache_manager, self.performance_tracker)
        self.speaker_processor = SpeakerProcessor(self.cache_manager, self.performance_tracker)
        self.video_processor = VideoProcessor(self.performance_tracker)
        
        # Initialize utilities
        self.subtitle_manager = SubtitleManager()
        self.debug_generator = DebugGenerator()
        self.speaker_reporter = SpeakerReporter(self.performance_tracker)
        
        # Set device
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.torch_device = self._get_torch_device(device)
        
        # Initialize debug data container
        self.debug_data = {
            "diarization": None,
            "transcription": None,
            "translation": None,
            "speed_ratios": {},
            "voices": {},
            "speaker_groups": {}
        }
        
        # Initialize real segment positions for pause removal
        self.real_segment_positions = []
        
        # Initialize pause adjustments for subtitle timing
        self.pause_adjustments = []
        
        # Initialize translator
        self._initialize_translator()
        
        # Initialize TTS systems
        self._initialize_tts_systems()
        
        # Initialize transcriber
        self._initialize_transcriber()
        
        logger.info(f"Initialized SmartDubbing with {self.device} device")
        logger.debug(f"Using {config.get('tts_system', 'coqui')} TTS system")
        logger.debug(f"Using {config.get('transcription_system', 'whisper')} transcription system")
        
        if config.get('start_time') is not None or config.get('duration') is not None:
            start_str = f"from {config.get('start_time')}s" if config.get('start_time') is not None else "from beginning"
            duration_str = f"for {config.get('duration')}s" if config.get('duration') is not None else "to the end"
            logger.debug(f"Processing video segment {start_str} {duration_str}")
        
        if config.get('use_cache', True):
            logger.debug("Caching enabled: will use cached results when available")
            
        if config.get('debug_info', False):
            logger.debug("Debug mode enabled: will generate a debug video with speaker labels")
        
        if config.get('debug_diarize_only', False):
            logger.debug("Debug diarize-only mode enabled: will exit after diarization and transcription with debug video")
    
    def _get_torch_device(self, device_str: Optional[str] = None) -> torch.device:
        """Helper method to get a proper torch.device object."""
        if device_str is None:
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device_str)
    
    def _initialize_translator(self) -> None:
        """Initialize translator based on configuration."""
        self.translator = None
        try:
            self.translator = TranslatorFactory.create_translator(
                translator_type=self.config.get('translator_type', 'llm'),
                llm_provider=self.config.get('llm_provider', 'gemini'),
                model_name=self.config.get('llm_model_name'),
                temperature=self.config.get('llm_temperature', 0.5),
                refinement_llm_provider=self.config.get('refinement_llm_provider'),
                refinement_model_name=self.config.get('refinement_model_name'),
                refinement_temperature=self.config.get('refinement_temperature', 1.0),
                refinement_max_tokens=self.config.get('refinement_max_tokens'),
                refinement_persona=self.config.get('refinement_persona', 'normal'),
                glossary=self.config.get('glossary'),
                cache_manager=self.cache_manager
            )
            logger.debug(f"Using {self.config.get('translator_type', 'llm')} translator")
        except Exception as e:
            logger.warning(f"Failed to initialize translator: {e}")
    
    def _initialize_tts_systems(self) -> None:
        """Initialize TTS systems based on configuration."""
        self.tts_systems = {}
        self.default_tts = None
        
        try:
            tts_instance = TTSFactory.create_tts(
                tts_system=self.config.get('tts_system', 'coqui'),
                device=self.device,
                voice_config=self.config.get('voice_name'),
                voice_prompt=self.config.get('voice_prompt', {}),
                prompt_prefix=self.config.get('tts_prompt_prefix'),
                enable_voice_matching=self.config.get('voice_auto_selection', True),
                debug_tts=self.config.get('debug_tts', False)
            )
            self.tts_systems[self.config.get('tts_system', 'coqui')] = tts_instance
            self.default_tts = tts_instance
            logger.debug(f"Initialized {self.config.get('tts_system', 'coqui')} TTS system")
        except Exception as e:
            logger.warning(f"Failed to initialize TTS: {e}")
    
    def _initialize_transcriber(self) -> None:
        """Initialize transcriber based on configuration."""
        self.transcriber = None
        try:
            self.transcriber = TranscriptionFactory.create_transcriber(
                transcription_system=self.config.get('transcription_system', 'whisper'),
                source_language=self.config.get('source_language'),
                device=self.device,
                whisper_model=self.config.get('whisper_model', 'large-v3'),
                cache_manager=self.cache_manager
            )
            logger.debug(f"Initialized {self.transcriber.name} transcriber")
        except Exception as e:
            logger.warning(f"Failed to initialize transcriber: {e}")
    
    def run_pipeline(self, save_original_subtitles: bool = False, save_translated_subtitles: bool = False) -> str:
        """Run the full dubbing pipeline."""
        pipeline_start_time = time.perf_counter()
        self.performance_tracker.start_timing("total")
        
        logger.info(f"Starting dubbing process for {self.config.get('input')}")
        output_video_path = ""
        
        try:
            # Extract audio from video
            audio_file = self.audio_processor.extract_audio(
                self.config.get('input'),
                self.config.get('start_time'),
                self.config.get('duration')
            )
            
            # Perform speaker diarization and transcription
            speakers_rolls, transcription = self.diarize_and_transcribe(audio_file)
            if speakers_rolls is None or len(speakers_rolls) == 0:
                raise ValueError("No speakers found in the video")
            
            # If debug_diarize_only is True, generate debug video and exit early
            if self.config.get('debug_diarize_only', False):
                return self._handle_debug_diarize_only(audio_file, speakers_rolls)
            
            # Extract audio for each speaker
            self.speaker_processor.extract_speaker_audio(audio_file, speakers_rolls)
            
            # Translate segments
            translated_segments = self.translate_segments(transcription, audio_file)
            
            # Save debug TSV
            self.subtitle_manager.save_debug_tsv(translated_segments)
            
            # Save subtitles if requested (only if pause removal is disabled)
            remove_pauses_enabled = self.config.get('remove_pauses', True)
            if not remove_pauses_enabled:
                if save_original_subtitles:
                    self.subtitle_manager.save_subtitles(translated_segments, "original", self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language')))
                
                if save_translated_subtitles:
                    self.subtitle_manager.save_subtitles(translated_segments, "translation", self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language')))
            
            # Analyze emotions (if enabled)
            if self.config.get('enable_emotion_analysis', True):
                translated_segments = self.analyze_emotions(translated_segments, audio_file)
            else:
                logger.debug("Emotion analysis disabled")
                for segment in translated_segments:
                    segment["emotion"] = "Neutral"
            
            # Synthesize speech
            translated_audio_path = self.synthesize_speech(translated_segments, speakers_rolls, audio_file)
            
            # Save translated samples
            self.speaker_processor.save_translated_samples(translated_segments, audio_file)
            
            # Process background audio if needed
            background_audio_path = None
            if self.config.get('keep_background', False):
                background_audio_path = self.audio_processor.process_background_audio(audio_file)
            
            # Generate final debug video if needed
            if self.config.get('debug_info', False):
                self.debug_generator.generate_debug_video(
                    self.config.get('input'),
                    self.debug_data,
                    self.config.get('start_time'),
                    self.config.get('duration'),
                    self.audio_processor.get_total_duration()
                )
            
            # Combine with video (includes pause removal if enabled)
            output_video_path, pause_adjustments = self.video_processor.combine_audio_with_video(
                video_path=self.config.get('input'),
                translated_audio_path=translated_audio_path,
                background_audio_path=background_audio_path,
                watermark_path=self.config.get('watermark_path'),
                watermark_text=self.config.get('watermark_text'),
                include_original_audio=self.config.get('include_original_audio', False),
                output_file=self.config.get('output'),
                start_time=self.config.get('start_time'),
                duration=self.config.get('duration'),
                keep_original_audio_ranges=self.config.get('keep_original_audio_ranges'),
                source_language=self.config.get('source_language'),
                target_language=self.config.get('target_language'),
                normalize_audio=self.config.get('normalize_audio', True),
                use_two_pass_encoding=self.config.get('use_two_pass_encoding', True),
                remove_pauses=self.config.get('remove_pauses', True),
                min_pause_duration=self.config.get('min_pause_duration', 3),
                preserve_pause_duration=self.config.get('preserve_pause_duration', 1.5),
                keyframe_buffer=self.config.get('keyframe_buffer', 0.2),
                ffmpeg_batch_size=self.config.get('ffmpeg_batch_size', 50),
                dubbed_volume=self.config.get('dubbed_volume', 1.0)
            )
            
            # Store pause adjustments for potential future use
            self.pause_adjustments = pause_adjustments
            
            # Save subtitles after pause processing if pause removal is enabled
            if remove_pauses_enabled and (save_original_subtitles or save_translated_subtitles):
                if pause_adjustments:
                    logger.info("Adjusting subtitle timestamps based on pause modifications...")
                    
                    if save_original_subtitles:
                        adjusted_original_segments = self.adjust_subtitle_timestamps(translated_segments, pause_adjustments)
                        self.subtitle_manager.save_subtitles(adjusted_original_segments, "original", self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language')))
                        adjusted_path = self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language'))
                        logger.info(f"Saved pause-corrected original subtitles to {adjusted_path}")
                    
                    if save_translated_subtitles:
                        adjusted_translated_segments = self.adjust_subtitle_timestamps(translated_segments, pause_adjustments)
                        self.subtitle_manager.save_subtitles(adjusted_translated_segments, "translation", self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language')))
                        adjusted_path = self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language'))
                        logger.info(f"Saved pause-corrected translated subtitles to {adjusted_path}")
                else:
                    # No pause adjustments made, but pause removal was enabled - save original timestamps
                    logger.info("No pause adjustments needed, saving subtitles with original timestamps...")
                    
                    if save_original_subtitles:
                        self.subtitle_manager.save_subtitles(translated_segments, "original", self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language')))
                        subtitle_path = self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language'))
                        logger.info(f"Saved original subtitles to {subtitle_path}")
                    
                    if save_translated_subtitles:
                        self.subtitle_manager.save_subtitles(translated_segments, "translation", self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language')))
                        subtitle_path = self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language'))
                        logger.info(f"Saved translated subtitles to {subtitle_path}")
            
            # Overall pipeline metrics
            total_elapsed = time.perf_counter() - pipeline_start_time
            logger.info("Dubbing process completed!")
            self.performance_tracker.record_metric("total", total_elapsed)
            
            # Write performance summary
            self.performance_tracker.write_performance_summary(self.audio_processor.get_total_duration())
            
        except Exception as e:
            logger.error(f"Error in dubbing pipeline: {e}", exc_info=True)
            raise
        finally:
            # Clean up
            self._cleanup()
        
        return output_video_path
    
    def _handle_debug_diarize_only(self, audio_file: str, speakers_rolls: Dict) -> str:
        """Handle debug diarize-only mode."""
        logger.info("Debug diarization only mode: Generating debug video after diarization and exiting")
        
        # Set debug_info to True to ensure debug video generation works
        original_debug_info = self.config.get('debug_info', False)
        self.config.set('debug_info', True)
        
        # Extract audio for each speaker
        self.speaker_processor.extract_speaker_audio(audio_file, speakers_rolls)
        
        # Generate debug video
        self.debug_generator.generate_debug_video(
            self.config.get('input'),
            self.debug_data,
            self.config.get('start_time'),
            self.config.get('duration'),
            self.audio_processor.get_total_duration()
        )
        
        # Create debug TSV of original transcription
        self.subtitle_manager.save_debug_tsv(self.debug_data["transcription"])
        
        # Reset debug_info to original value
        self.config.set('debug_info', original_debug_info)
        
        # Return path to debug video
        debug_video_path = "artifacts/debug/dubbing_debug.mp4"
        logger.info(f"Debug video generated: {debug_video_path}")
        
        # Write partial performance summary
        self.performance_tracker.record_metric("total", time.perf_counter() - self.performance_tracker._start_times.get("total", 0))
        self.performance_tracker.write_performance_summary(self.audio_processor.get_total_duration())
        
        return debug_video_path
    
    def generate_diarization_report(self) -> Tuple[str, str]:
        """Generate a report of identified speakers and their voice samples."""
        report_start_time = time.perf_counter()
        logger.info("Starting speaker report generation...")

        # Extract audio from video
        audio_file = self.audio_processor.extract_audio(
            self.config.get('input'),
            self.config.get('start_time'),
            self.config.get('duration')
        )

        # Perform speaker diarization and transcription
        speakers_rolls, transcription = self.diarize_and_transcribe(audio_file)
        if speakers_rolls is None or len(speakers_rolls) == 0:
            raise ValueError("No speakers found in the video during diarization.")

        # Extract audio samples for each speaker
        speaker_audio_paths = self.speaker_processor.extract_speaker_audio(audio_file, speakers_rolls)
        if not speaker_audio_paths:
            raise ValueError("Could not extract audio samples for speakers.")

        # Create the speaker report
        report_file_path, samples_dir_path = self.speaker_reporter.create_speaker_report(
            speaker_audio_paths, transcription, self.config.get('input')
        )
        
        # Write performance summary for this specific operation
        self.performance_tracker.record_metric("total_report_generation", time.perf_counter() - report_start_time)
        self.performance_tracker.record_metric("video_duration", self.audio_processor.get_total_duration() or 0)
        self.performance_tracker.write_performance_summary_for_report()
        
        return report_file_path, samples_dir_path
    
    def diarize_and_transcribe(self, audio_file: str) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]:
        """Perform speaker diarization and transcription."""
        if not self.transcriber:
            raise ValueError("Transcriber not initialized")
            
        # Generate cache key
        cache_key = self.cache_manager.generate_cache_key(
            audio_file,
            self.config.get('source_language'),
            self.config.get('target_language'),
            self.config.get('whisper_model', 'large-v3'),
            self.config.get('start_time'),
            self.config.get('duration')
        )
        
        # Perform diarization and transcription
        speakers_rolls, transcription = self.transcriber.diarize_and_transcribe(
            audio_file=audio_file,
            cache_key=cache_key,
            use_cache=self.cache_manager.use_cache
        )
        
        # Store for debug
        self.debug_data["diarization"] = speakers_rolls
        self.debug_data["transcription"] = transcription
        
        # Save transcription to file
        self._save_transcription_file(transcription)
        
        return speakers_rolls, transcription
    
    def translate_segments(self, transcription: List[Dict], audio_file: str) -> List[Dict]:
        """Translate segments using the translator."""
        cache_key = f"{self.cache_manager.generate_cache_key(audio_file, self.config.get('source_language'), self.config.get('target_language'), self.config.get('whisper_model', 'large-v3'), self.config.get('start_time'), self.config.get('duration'))}_{self.config.get('target_language')}"
        step_name = "translation"
        
        translated_segments = None
        # Check if results are cached
        if self.cache_manager.cache_exists(step_name, cache_key):
            logger.debug("Loading translations from cache...")
            translated_segments = self.cache_manager.load_from_cache(step_name, cache_key)
            if translated_segments is not None:
                self.performance_tracker.record_metric("translation", 0.0)
            else:
                logger.warning("Found corrupted translation cache, re-translating.")

        if translated_segments is None:
            # Start timing
            self.performance_tracker.start_timing("translation")
            
            if self.translator and self.translator.is_available():
                translated_segments = self.translator.translate(
                    segments=transcription,
                    source_language=self.config.get('source_language'),
                    target_language=self.config.get('target_language'),
                    refinement_persona=self.config.get('refinement_persona', 'normal'),
                    debug=self.debug_data
                )
            else:
                raise ValueError("No translator available")
            
            # Save results to cache
            self.cache_manager.save_to_cache(step_name, cache_key, translated_segments)
            
            # End timing
            elapsed_time = self.performance_tracker.end_timing("translation")
            logger.info(f"Finished translation in {elapsed_time:.2f} seconds (≈ {elapsed_time/60:.2f} minutes)")
        
        # Store for debug
        self.debug_data["translation"] = translated_segments
        
        return translated_segments
    
    def analyze_emotions(self, segments: List[Dict], audio_file: str) -> List[Dict]:
        """Analyze emotions in the audio for each segment."""
        if not segments:
            return []

        cache_key = self.cache_manager.generate_cache_key(
            audio_file, "", "", ""  # Simple cache key for emotions
        )
        step_name = "emotions"
        
        # Check if results are cached
        if self.cache_manager.cache_exists(step_name, cache_key):
            logger.debug("Loading emotion analysis from cache...")
            cached_segments = self.cache_manager.load_from_cache(step_name, cache_key)
            if cached_segments is not None:
                return cached_segments
            logger.warning("Found corrupted emotion cache, re-analyzing.")
        
        logger.info("Analyzing speech emotions...")
        self.performance_tracker.start_timing("emotion_analysis")
        
        # Initialize the emotion classifier
        classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": self.torch_device}
        )
        
        # Emotion mapping
        emotion_dict = {
            'neu': 'Neutral',
            'ang': 'Angry',
            'hap': 'Happy',
            'sad': 'Sad',
            'None': None
        }
        
        # Process each segment
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_file)
        for segment in segments:
            try:
                start = int(segment["start"] * 1000)
                end = int(segment["end"] * 1000)
                
                segment_audio = audio[start:end]
                segment_audio.export("artifacts/audio/temp_segment.wav", format="wav")
                
                out_prob, score, index, text_lab = classifier.classify_file("artifacts/audio/temp_segment.wav")
                segment["emotion"] = emotion_dict[text_lab[0]]
                
                os.remove("artifacts/audio/temp_segment.wav")
            except Exception as e:
                logger.warning(f"Error analyzing emotion: {e}")
                segment["emotion"] = "Neutral"
        
        # Save results to cache
        self.cache_manager.save_to_cache(step_name, cache_key, segments)
        
        # End timing
        self.performance_tracker.end_timing("emotion_analysis")
        
        return segments
    
    def synthesize_speech(self, segments: List[Dict], speakers_rolls: Dict, audio_file: str) -> str:
        """
        Synthesize speech for translated segments with optimized batching and estimation.
        
        Args:
            segments: List of transcript segments with translations
            speakers_rolls: Dictionary mapping time ranges to speaker IDs
            audio_file: Path to the audio file for cache key
            
        Returns:
            Path to the output audio file
        """
        if not segments:
            raise ValueError("Cannot synthesize speech with no segments.")

        # Start timing
        self.performance_tracker.start_timing("speech_synthesis")
        
        cache_key = f"{self.cache_manager.generate_cache_key(audio_file, self.config.get('source_language'), self.config.get('target_language'), self.config.get('whisper_model', 'large-v3'), self.config.get('start_time'), self.config.get('duration'))}_{self.config.get('target_language')}_{self.config.get('tts_system')}"
        step_name = "synthesized_speech"
        
        # Check if results are cached
        if self.cache_manager.cache_exists(step_name, cache_key):
            logger.debug("Loading synthesized speech from cache...")
            # Copy the cached output audio
            cached_audio_path = self.cache_manager.get_cache_path(step_name) / f"{cache_key}.wav"
            if cached_audio_path.exists():
                output_path = "artifacts/audio/output.wav"
                shutil.copy(cached_audio_path, output_path)
                self.performance_tracker.end_timing("speech_synthesis")
                return output_path
        
        logger.info(f"Synthesizing translated speech using multiple TTS systems...")
        
        # Create segment cache directory if needed
        segment_cache_path = self.cache_manager.get_cache_path("segment_synthesis")
        base_cache_prefix = self.cache_manager.generate_cache_key(audio_file, self.config.get('source_language'), self.config.get('target_language'), self.config.get('whisper_model', 'large-v3'), self.config.get('start_time'), self.config.get('duration'))
        
        # Check if any TTS systems are initialized
        if not self.tts_systems:
            raise ValueError("No TTS systems are initialized properly")
        
        # Import TTSSegmentData for synthesis
        from tts.models import TTSSegmentData
        
        # Define comfort ratio constants
        COMFORT_MIN_ADJUSTMENT_RATIO = 0.75
        COMFORT_MAX_ADJUSTMENT_RATIO = 1.15
        
        # Group segments by TTS system for batch processing
        segments_by_tts_system = {}
        segment_to_tts_mapping = {}
        
        for i, segment_dict in enumerate(segments):
            speaker = segment_dict["speaker"]
            tts_system = self._get_tts_system_for_speaker(speaker)
            
            if tts_system not in segments_by_tts_system:
                segments_by_tts_system[tts_system] = []
            
            segments_by_tts_system[tts_system].append((i, segment_dict))
            segment_to_tts_mapping[i] = tts_system
        
        logger.debug(f"Segments grouped by TTS system: {[(tts_sys, len(segs)) for tts_sys, segs in segments_by_tts_system.items()]}")
        
        # First pass: Determine the best text version for each segment using estimation
        segments_to_synthesize_by_tts = {tts_sys: [] for tts_sys in segments_by_tts_system.keys()}
        segments_metadata = []
        
        # Initialize progress tracking
        total_segments = len(segments)
        completed_segments = 0
        
        logger.info(f"Estimating audio durations to select optimal text versions for {total_segments} segments...")
        
        for tts_system, segment_list in segments_by_tts_system.items():
            # Get the shared TTS instance for this system type
            tts_instance = self.tts_systems.get(tts_system)
            if not tts_instance:
                logger.warning(f"Warning: TTS system {tts_system} not available, using default")
                tts_instance = self.default_tts
            
            logger.debug(f"Processing {len(segment_list)} segments with {tts_system} TTS ...")
            
            for i, segment_dict in segment_list:
                # Determine speaker for this segment
                speaker = segment_dict["speaker"]
                
                # Get the voice prompt for this speaker if available
                segment_style_prompt = self.voice_prompt.get(speaker, None)
                
                # Get voice name if available (for OpenAI/Gemini TTS)
                voice_name = None
                voice_config = self.config.get('voice_name')
                if isinstance(voice_config, dict):
                    voice_name = voice_config.get(speaker, next(iter(voice_config.values()), "default"))
                elif isinstance(voice_config, str):
                    voice_name = voice_config
                
                # Store the voice information for debug
                if self.config.get('debug_info', False):
                    self.debug_data["voices"][i] = {
                        "speaker": speaker,
                        "voice": voice_name,
                        "style_prompt": segment_style_prompt,
                        "tts_system": tts_system
                    }
                
                # Check cache first
                import hashlib
                translation_hash = hashlib.md5(segment_dict["translation"].encode()).hexdigest()[:8]
                voice_prompt_hash = hashlib.md5((segment_style_prompt or "").encode()).hexdigest()[:8]
                segment_cache_key = f"{base_cache_prefix}_{tts_system}_{i}_{speaker}_{translation_hash}_{voice_prompt_hash}"
                current_segment_output_path = f"artifacts/audio_chunks/{i}.wav"
                os.makedirs(os.path.dirname(current_segment_output_path), exist_ok=True)
                
                segment_cached_file_path = segment_cache_path / f"{segment_cache_key}.wav"
                
                # Try to use cached segment
                if self.cache_manager.use_cache and segment_cached_file_path.exists():
                    try:
                        cached_audio_info = AudioSegment.from_file(segment_cached_file_path)
                        if len(cached_audio_info) > 0:
                            logger.debug(f"Using valid cached segment {i+1}/{len(segments)} ({tts_system})")
                            shutil.copy(segment_cached_file_path, current_segment_output_path)
                            segment_dict['synthesized_speech_len'] = len(cached_audio_info) / 1000.0
                            segment_dict['synthesized_speech_file'] = current_segment_output_path
                            continue
                        else:
                            os.remove(segment_cached_file_path)
                    except Exception as e:
                        try:
                            os.remove(segment_cached_file_path)
                        except:
                            pass
                
                # Prepare base TTSSegmentData
                tts_segment_data_args = {
                    "speaker": speaker,
                    "text": segment_dict["translation"],
                    "emotion": segment_dict.get("emotion", "Neutral"),
                    "style_prompt": segment_style_prompt,
                    "reference_audio_path": None,
                    "reference_text": None,
                    "voice": voice_name,
                    "speed": 1.0
                }
                
                # Generic reference audio path for systems that might use it
                potential_ref_audio_for_speaker = f"artifacts/speakers_audio/{speaker}.wav"
                if os.path.exists(potential_ref_audio_for_speaker):
                    tts_segment_data_args["reference_audio_path"] = potential_ref_audio_for_speaker
                
                # Calculate original duration and estimate current translation
                original_duration = segment_dict["end"] - segment_dict["start"]
                
                # Create TTSSegmentData for estimation
                segment_data_model = TTSSegmentData(**tts_segment_data_args)
                
                # Estimate duration for normal translation
                estimated_duration_normal = tts_instance.estimate_audio_segment_length(
                    segment_data_model,
                    language=self.config.get('target_language')
                )
                
                best_text = segment_dict["translation"]
                best_estimated_duration = estimated_duration_normal
                best_ratio = float('inf')
                best_deviation = float('inf')
                best_track_type = "translation"  # Track which translation variant was selected
                
                if estimated_duration_normal is None or estimated_duration_normal <= 0:
                    logger.warning(f"Segment {i+1}: Duration estimation failed for normal translation, using it directly.")
                else:
                    ratio_normal = original_duration / estimated_duration_normal
                    deviation_normal = self._calculate_percentage_deviation(
                        ratio_normal,
                        COMFORT_MIN_ADJUSTMENT_RATIO,
                        COMFORT_MAX_ADJUSTMENT_RATIO,
                    )
                    best_ratio = ratio_normal
                    best_deviation = deviation_normal
                    logger.debug(
                        f"Segment {i+1} ({tts_system}): Normal translation - Estimated duration: {estimated_duration_normal:.2f}s, Ratio: {ratio_normal:.2f}, Deviation: {deviation_normal:.2%}"
                    )
                    
                    # If normal is perfect, no need to check alternatives
                    if deviation_normal == 0.0:
                        logger.debug(f"  Normal translation is within comfort zone. Selecting it.")
                    else:
                        alternatives = []
                        # Decide which alternatives to consider based on whether we need to shorten or lengthen
                        if ratio_normal < COMFORT_MIN_ADJUSTMENT_RATIO:
                            # Synthesized audio longer than original – try shorter variants first
                            if "very_short_translation" in segment_dict:
                                alternatives.append(("very_short_translation", segment_dict["very_short_translation"]))
                            if "short_translation" in segment_dict:
                                alternatives.append(("short_translation", segment_dict["short_translation"]))
                        elif ratio_normal > COMFORT_MAX_ADJUSTMENT_RATIO:
                            # Synthesized audio shorter than original – try longer variant
                            if "long_translation" in segment_dict:
                                alternatives.append(("long_translation", segment_dict["long_translation"]))
                            # Fallback to original text if long not available (handled later)
                        
                        if alternatives:
                            logger.debug(f"  Normal translation is outside comfort. Estimating {len(alternatives)} alternative(s)...")
                        
                        # Preference: when remove_pauses enabled and non-zero deviation remains, prefer negative deviation (shorter) in tie
                        prefer_shorter = self.config.get('remove_pauses', True)
                        def deviation_key(dev: float) -> tuple:
                            # Primary: minimal absolute deviation; Secondary: prefer negative when enabled
                            return (abs(dev), 0 if (prefer_shorter and dev < 0) else 1)

                        for alt_key, alt_text_content in alternatives:
                            alt_segment_data_args = {**tts_segment_data_args, "text": alt_text_content}
                            alt_segment_data_model = TTSSegmentData(**alt_segment_data_args)
                            estimated_duration_alt = tts_instance.estimate_audio_segment_length(
                                alt_segment_data_model,
                                language=self.config.get('target_language')
                            )
                            
                            if estimated_duration_alt is None or estimated_duration_alt <= 0:
                                logger.warning(f"    {alt_key.replace('_', ' ').title()}: Estimation failed.")
                                continue
                            
                            ratio_alt = original_duration / estimated_duration_alt
                            deviation_alt = self._calculate_percentage_deviation(
                                ratio_alt,
                                COMFORT_MIN_ADJUSTMENT_RATIO,
                                COMFORT_MAX_ADJUSTMENT_RATIO,
                            )
                            logger.debug(f"    {alt_key.replace('_', ' ').title()} - Estimated duration: {estimated_duration_alt:.2f}s, Ratio: {ratio_alt:.2f}, Deviation: {deviation_alt:.2%}")
                            
                            # Update if this alternative is better per deviation_key
                            if deviation_key(deviation_alt) < deviation_key(best_deviation):
                                best_deviation = deviation_alt
                                best_ratio = ratio_alt
                                best_text = alt_text_content
                                best_estimated_duration = estimated_duration_alt
                                best_track_type = alt_key  # Track the selected variant
                                logger.debug(
                                    f"      New best: {alt_key.replace('_', ' ').title()} (Deviation: {best_deviation:.2%})"
                                )
                                if best_deviation == 0.0:
                                    break
                
                logger.debug(f"  Selected for synthesis: '{best_text[:50]}...' (Ratio: {best_ratio:.2f}, Deviation: {best_deviation:.2%})")
                
                # Prepare segment for synthesis with chosen text
                final_segment_data = TTSSegmentData(**{**tts_segment_data_args, "text": best_text, "output_path": current_segment_output_path})
                segments_to_synthesize_by_tts[tts_system].append(final_segment_data)
                segments_metadata.append({
                    "index": i,
                    "segment_dict": segment_dict,
                    "cache_path": segment_cached_file_path,
                    "output_path": current_segment_output_path,
                    "chosen_text": best_text,
                    "estimated_ratio": best_ratio,
                    "tts_system": tts_system,
                    "segment_data_args": tts_segment_data_args,
                    "selected_track_type": best_track_type  # Store the selected track type
                })
        
        # Second pass: Batch synthesize all segments by TTS system
        for tts_system, segments_to_synthesize in segments_to_synthesize_by_tts.items():
            if not segments_to_synthesize:
                continue
            
            # Get the shared TTS instance for this system type
            tts_instance = self.tts_systems.get(tts_system)
            if not tts_instance:
                logger.warning(f"Warning: TTS system {tts_system} not available, skipping segments")
                continue
            
            logger.info(f"Synthesizing {len(segments_to_synthesize)} segments with {tts_system} TTS ...")
            
            try:
                # Synthesize all segments for this TTS system in one call
                segment_alignments = tts_instance.synthesize(
                    segments_data=segments_to_synthesize,
                    language=self.config.get('target_language')
                )
                
                # Process results and update segment metadata
                for segment_data in segments_to_synthesize:
                    # Find corresponding metadata
                    metadata = next((m for m in segments_metadata if m["output_path"] == segment_data.output_path), None)
                    if not metadata:
                        continue
                    
                    segment_dict = metadata["segment_dict"]
                    output_path = metadata["output_path"]
                    
                    # Update progress tracking
                    completed_segments += 1
                    logger.info(f"Processing segment {completed_segments}/{total_segments} (Speaker: {segment_dict['speaker']})")
                    
                    # Check if file was created successfully
                    if os.path.exists(output_path):
                        audio_info = AudioSegment.from_file(output_path)
                        segment_dict['synthesized_speech_len'] = len(audio_info) / 1000.0
                        segment_dict['synthesized_speech_file'] = output_path
                        
                        # Cache the synthesized segment
                        if self.cache_manager.use_cache and len(audio_info) > 0:
                            try:
                                shutil.copy(output_path, metadata["cache_path"])
                                logger.debug(f"Cached synthesized segment {metadata['index']+1} ({tts_system})")
                            except Exception as e:
                                logger.error(f"Error caching segment {metadata['index']+1}: {e}")

                        # Validate actual duration and resynthesize if needed
                        original_dur = segment_dict["end"] - segment_dict["start"]
                        actual_dur = segment_dict['synthesized_speech_len']
                        ratio = original_dur / actual_dur if actual_dur > 0 else 1.0
                        deviation = abs(original_dur - actual_dur) / original_dur if original_dur > 0 else 0.0
                        if not (COMFORT_MIN_ADJUSTMENT_RATIO <= ratio <= COMFORT_MAX_ADJUSTMENT_RATIO):
                            logger.debug(f"Segment {metadata['index']+1}: duration mismatch after synthesis (ratio={ratio:.2f}, dev={deviation:.2%}). Trying alternatives...")
                            self._resynthesize_segment(
                                metadata,
                                tts_instance,
                                COMFORT_MIN_ADJUSTMENT_RATIO,
                                COMFORT_MAX_ADJUSTMENT_RATIO,
                                current_ratio=ratio,
                            )
                    else:
                        logger.warning(f"Warning: No audio file created for segment {metadata['index']+1} ({tts_system})")
                        segment_dict['synthesized_speech_len'] = 0
                        segment_dict['synthesized_speech_file'] = None
                        # Create empty file to prevent downstream errors
                        AudioSegment.silent(duration=0).export(output_path, format="wav")
                
                logger.debug(f"Batch synthesis completed for {len(segments_to_synthesize)} segments with {tts_system}")
                
            except Exception as e:
                logger.error(f"Batch synthesis failed for {tts_system}: {e}. Falling back to individual synthesis...")
                
                # Fallback: synthesize individually
                for segment_data in segments_to_synthesize:
                    metadata = next((m for m in segments_metadata if m["output_path"] == segment_data.output_path), None)
                    if not metadata:
                        continue
                    
                    segment_dict = metadata["segment_dict"]
                    output_path = metadata["output_path"]
                    
                    # Update progress tracking for fallback synthesis
                    completed_segments += 1
                    logger.info(f"Processing segment {completed_segments}/{total_segments} (Fallback - Speaker: {segment_dict['speaker']})")
                    
                    try:
                        tts_instance.synthesize(
                            segments_data=[segment_data],
                            language=self.config.get('target_language')
                        )
                        
                        if os.path.exists(output_path):
                            audio_info = AudioSegment.from_file(output_path)
                            segment_dict['synthesized_speech_len'] = len(audio_info) / 1000.0
                            segment_dict['synthesized_speech_file'] = output_path
                            
                            # Cache the synthesized segment
                            if self.cache_manager.use_cache and len(audio_info) > 0:
                                shutil.copy(output_path, metadata["cache_path"])
                        
                        # After fallback individual synthesis, validate duration again
                        original_dur = segment_dict["end"] - segment_dict["start"]
                        actual_dur = segment_dict.get('synthesized_speech_len', 0)
                        ratio = original_dur / actual_dur if actual_dur > 0 else 1.0
                        deviation = abs(original_dur - actual_dur) / original_dur if original_dur > 0 else 0.0
                        if not (COMFORT_MIN_ADJUSTMENT_RATIO <= ratio <= COMFORT_MAX_ADJUSTMENT_RATIO):
                            logger.info(f"Segment {metadata['index']+1}: duration mismatch after fallback synthesis (ratio={ratio:.2f}, dev={deviation:.2%}). Trying alternatives...")
                            self._resynthesize_segment(
                                metadata,
                                tts_instance,
                                COMFORT_MIN_ADJUSTMENT_RATIO,
                                COMFORT_MAX_ADJUSTMENT_RATIO,
                                current_ratio=ratio,
                            )
                        
                        logger.debug(f"Synthesized segment {metadata['index']+1}/{len(segments)} individually ({tts_system})")
                        
                    except Exception as e_synth:
                        logger.error(f"Failed to synthesize segment {metadata['index']+1} ({tts_system}): {e_synth}")
                        segment_dict['synthesized_speech_len'] = 0
                        segment_dict['synthesized_speech_file'] = None
                        AudioSegment.silent(duration=0).export(output_path, format="wav")
        
        # Adjust timing and combine audio segments
        combined_audio, real_segment_positions = self._adjust_and_combine_audio_grouped(segments)
        output_path = "artifacts/audio/output.wav"
        combined_audio.export(output_path, format="wav")
        
        # Store real segment positions for later use in pause removal
        self.real_segment_positions = real_segment_positions
        
        # Log information about real vs original timing
        if real_segment_positions:
            total_real_duration = real_segment_positions[-1]["end"] - real_segment_positions[0]["start"]
            total_original_duration = max(s["original_end"] for s in real_segment_positions) - min(s["original_start"] for s in real_segment_positions)
            logger.debug(f"Real segments timing: {len(real_segment_positions)} segments, "
                        f"Real duration: {total_real_duration:.2f}s, Original duration: {total_original_duration:.2f}s")
        
        # Save to cache
        if self.cache_manager.use_cache:
            # Save the output audio
            shutil.copy(output_path, self.cache_manager.get_cache_path(step_name) / f"{cache_key}.wav")
        
        # Generate track usage report
        track_usage_stats = {}
        for metadata in segments_metadata:
            track_type = metadata.get("selected_track_type", "translation")
            track_usage_stats[track_type] = track_usage_stats.get(track_type, 0) + 1
        
        # Log the track usage report
        logger.debug("Voice sample track usage report:")
        total_samples = sum(track_usage_stats.values())
        for track_type, count in sorted(track_usage_stats.items()):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            logger.debug(f"  {track_type}: {count} samples ({percentage:.1f}%)")
        logger.debug(f"Total voice samples processed: {total_samples}")
        
        # Log completion summary
        logger.info(f"Speech synthesis completed! Processed {total_segments} segments successfully.")
        
        # End timing
        self.performance_tracker.end_timing("speech_synthesis")
        
        return output_path
    
    def _save_transcription_file(self, transcription: List[Dict]) -> None:
        """Save transcription to a readable text file."""
        from src.utils.time_utils import format_seconds_to_hms
        
        transcription_output_path = "artifacts/transcription.txt"
        os.makedirs(os.path.dirname(transcription_output_path), exist_ok=True)
        
        try:
            with open(transcription_output_path, 'w', encoding='utf-8') as f:
                for segment in transcription:
                    start_seconds = segment['start']
                    end_seconds = segment['end']
                    formatted_time = format_seconds_to_hms(start_seconds)
                    formatted_time_end = format_seconds_to_hms(end_seconds)
                    
                    f.write(f"[{formatted_time}-{formatted_time_end}] {segment['speaker']}: {segment['text']}\n")
            logger.info(f"Transcription saved to {transcription_output_path}")
        except Exception as e:
            logger.warning(f"Failed to save transcription to file: {e}")
    
    def _cleanup(self) -> None:
        """Clean up temporary files and TTS systems."""
        logger.info("Cleaning up temporary files...")
        try:
            # Call cleanup through the TTS systems
            for tts_system, tts_instance in self.tts_systems.items():
                if tts_instance:
                    try:
                        tts_instance.cleanup()
                        logger.info(f"Cleaned up {tts_system} TTS system")
                    except Exception as cleanup_e:
                        logger.warning(f"Warning: Error cleaning up {tts_system} TTS: {cleanup_e}")
            
            # Clean up temporary directories
            for temp_dir in ["artifacts/audio_chunks", "artifacts/su_audio_chunks"]:
                if os.path.exists(temp_dir):
                    for temp_file in os.listdir(temp_dir):
                        if temp_file.startswith("temp_") or temp_file.startswith("group_"):
                            try:
                                os.remove(os.path.join(temp_dir, temp_file))
                            except Exception:
                                pass
            
            logger.info("Cleanup completed.")
        except Exception as e:
            logger.warning(f"Warning: Error during cleanup: {e}")
    
    def _get_tts_system_for_speaker(self, speaker_id: str) -> str:
        """
        Get the TTS system to use for a specific speaker.
        
        Args:
            speaker_id: Speaker ID
            
        Returns:
            TTS system name for the speaker
        """
        # Check for explicit speaker mapping first
        if speaker_id in self.tts_system_mapping:
            return self.tts_system_mapping[speaker_id]
        
        # Check for wildcard mapping
        if "*" in self.tts_system_mapping:
            return self.tts_system_mapping["*"]
        
        # Fall back to default TTS system
        return self.config.get('tts_system', 'coqui')
    
    def _calculate_percentage_deviation(self, ratio: float, min_ratio_comfort: float, max_ratio_comfort: float) -> float:
        """
        Calculates the signed percentage deviation of a given ratio from the comfort zone.
        
        Args:
            ratio: The speech ratio (original_duration / synthesized_duration).
            min_ratio_comfort: The minimum acceptable ratio for comfort.
            max_ratio_comfort: The maximum acceptable ratio for comfort.
            
        Returns:
            0.0 if the ratio is within the comfort zone.
            Positive value if the synthesized segment is longer than comfortable (ratio < min).
            Negative value if the synthesized segment is shorter than comfortable (ratio > max).
        """
        if ratio >= min_ratio_comfort and ratio <= max_ratio_comfort:
            return 0.0
        elif ratio < min_ratio_comfort:
            if min_ratio_comfort == 0:
                return float('inf')  # Avoid division by zero
            # Synthesized audio is longer than original → ratio is too small → positive deviation
            return (min_ratio_comfort - ratio) / min_ratio_comfort
        else:  # ratio > max_ratio_comfort
            if max_ratio_comfort == 0:
                return float('inf')  # Avoid division by zero
            # Synthesized audio is shorter than original → ratio is too large → negative deviation
            return -((ratio - max_ratio_comfort) / max_ratio_comfort)

    def _resynthesize_segment(
        self,
        metadata: Dict[str, Any],
        tts_instance,
        min_ratio: float,
        max_ratio: float,
        current_ratio: Optional[float] = None,
    ) -> None:
        """Attempt to resynthesize a segment using alternative translations,
        focusing on minimizing deviation from the target ratio range.

        Args:
            metadata: Metadata dictionary for the segment.
            tts_instance: The TTS instance used for synthesis.
            min_ratio: Minimum acceptable ratio original/actual.
            max_ratio: Maximum acceptable ratio original/actual.
            current_ratio: Current ratio to help prioritize alternatives.
        """

        from tts.models import TTSSegmentData

        segment_dict = metadata["segment_dict"]
        original_duration = segment_dict["end"] - segment_dict["start"]
        output_path = metadata["output_path"]
        base_args = metadata["segment_data_args"]
        
        # Log resynthesis attempt
        logger.info(f"Resynthesizing segment {metadata['index']+1} (Speaker: {segment_dict['speaker']}) for better duration matching...")

        # Decide search direction based on how the current ratio deviates
        if current_ratio is None and segment_dict.get("synthesized_speech_len", 0) > 0:
            current_ratio = original_duration / max(segment_dict["synthesized_speech_len"], 1e-6)

        if current_ratio is not None:
            if current_ratio < min_ratio:
                # synthesized audio longer than original – prioritize shorter variants
                candidate_keys = ["very_short_translation", "short_translation", "translation", "long_translation"]
            elif current_ratio > max_ratio:
                # synthesized audio shorter than original – prioritize longer variants
                candidate_keys = ["long_translation", "translation", "short_translation", "very_short_translation"]
            else:
                candidate_keys = ["very_short_translation", "short_translation", "translation", "long_translation"]
        else:
            candidate_keys = ["very_short_translation", "short_translation", "translation", "long_translation"]

        # Calculate current deviation to ensure we only accept improvements
        current_deviation = float('inf')
        if current_ratio is not None:
            current_deviation = self._calculate_percentage_deviation(current_ratio, min_ratio, max_ratio)
            logger.debug(f"Current ratio: {current_ratio:.2f}, current deviation: {current_deviation:.2%}")
        
        best_alternative = None
        best_deviation_from_range = current_deviation  # Start with current deviation as baseline
        best_ratio = None
        best_key = None

        # Preference for shorter audio when pause removal is enabled
        prefer_shorter = self.config.get('remove_pauses', True)
        def deviation_key(dev: float) -> tuple:
            # Primary: minimal absolute deviation; Secondary: prefer negative when enabled
            return (abs(dev), 0 if (prefer_shorter and dev < 0) else 1)

        # Keep track of already tried texts to avoid duplicate synthesis
        tried_texts = {metadata["chosen_text"]}

        # Try all alternatives and find the one with minimum deviation from target range
        for key in candidate_keys:
            if key not in segment_dict:
                continue

            alt_text = segment_dict[key]
            # Skip already used text or duplicate text
            if alt_text in tried_texts:
                continue
            
            # Add this text to tried set
            tried_texts.add(alt_text)

            # Create temporary output path for this alternative
            temp_output_path = f"{output_path}.temp_{key}"
            new_segment_data = TTSSegmentData(**{**base_args, "text": alt_text, "output_path": temp_output_path})

            try:
                tts_instance.synthesize(
                    segments_data=[new_segment_data],
                    language=self.config.get('target_language')
                )

                if not os.path.exists(temp_output_path):
                    continue

                audio_info = AudioSegment.from_file(temp_output_path)
                actual_duration = len(audio_info) / 1000.0

                if actual_duration == 0:
                    os.remove(temp_output_path)
                    continue

                ratio = original_duration / actual_duration
                
                # Calculate deviation from target range
                deviation_from_range = self._calculate_percentage_deviation(ratio, min_ratio, max_ratio)
                
                logger.debug(f"Alternative '{key}': ratio={ratio:.2f}, deviation_from_range={deviation_from_range:.2%}")

                # Check if this is the best alternative so far (consider signed deviation preference)
                if deviation_key(deviation_from_range) < deviation_key(best_deviation_from_range):
                    # Clean up previous best alternative if exists
                    if best_alternative and os.path.exists(best_alternative):
                        os.remove(best_alternative)
                    
                    best_alternative = temp_output_path
                    best_deviation_from_range = deviation_from_range
                    best_ratio = ratio
                    best_key = key
                    
                    logger.debug(f"New best alternative: '{key}' with deviation {deviation_from_range:.2%}")
                else:
                    # Clean up this alternative since it's not the best
                    os.remove(temp_output_path)

            except Exception as e:
                logger.error(f"Alternative synthesis failed for segment {metadata['index']+1} with '{key}': {e}")
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

        # If deviation remains large (>15%), try LLM-based text length adjustment
        try:
            LLM_DEVIATION_THRESHOLD = 0.15
            # Compute absolute deviation key for comparison
            if self.translator and self.translator.is_available() and deviation_key(current_deviation) > deviation_key(0.0) and abs(current_deviation) > LLM_DEVIATION_THRESHOLD:
                baseline_text = metadata.get("chosen_text") or segment_dict.get("translation", "")
                if baseline_text:
                    # Aim for center of comfort zone (prefer near 1.0), compute duration factor
                    target_ratio = 1.0
                    actual_duration = segment_dict.get("synthesized_speech_len", 0) or 1e-6
                    desired_duration = original_duration / max(target_ratio, 1e-6)
                    duration_factor = max(0.2, min(2.0, desired_duration / max(actual_duration, 1e-6)))

                    # Ask LLM to adjust text length
                    adjusted_text = self.translator.adjust_segment_text_length(
                        original_text=baseline_text,
                        source_language=self.config.get('source_language'),
                        target_language=self.config.get('target_language'),
                        desired_ratio=duration_factor,
                        target_char_count=int(len(baseline_text) * duration_factor),
                        context_info=None,
                        max_attempts=2,
                    )

                    if adjusted_text and adjusted_text.strip() and adjusted_text.strip() != baseline_text.strip():
                        # Estimate duration and synthesize to temp file
                        temp_output_path = f"{output_path}.temp_llm_adjusted"
                        from tts.models import TTSSegmentData
                        new_segment_data = TTSSegmentData(**{**base_args, "text": adjusted_text, "output_path": temp_output_path})

                        try:
                            tts_instance.synthesize(
                                segments_data=[new_segment_data],
                                language=self.config.get('target_language')
                            )

                            if os.path.exists(temp_output_path):
                                audio_info = AudioSegment.from_file(temp_output_path)
                                actual_duration_llm = len(audio_info) / 1000.0
                                if actual_duration_llm > 0:
                                    ratio_llm = original_duration / actual_duration_llm
                                    deviation_llm = self._calculate_percentage_deviation(ratio_llm, min_ratio, max_ratio)
                                    logger.info(f"LLM-adjusted alternative: ratio={ratio_llm:.2f}, deviation_from_range={deviation_llm:.2%}")

                                    if deviation_key(deviation_llm) < deviation_key(best_deviation_from_range):
                                        # Clean up previous best alternative if exists
                                        if best_alternative and os.path.exists(best_alternative):
                                            os.remove(best_alternative)

                                        best_alternative = temp_output_path
                                        best_deviation_from_range = deviation_llm
                                        best_ratio = ratio_llm
                                        best_key = "llm_adjusted"
                                        # Also update chosen text on success path later
                                        metadata["_llm_adjusted_text"] = adjusted_text
                                    else:
                                        # Not better; remove temp
                                        os.remove(temp_output_path)
                        except Exception as e:
                            logger.error(f"LLM-adjusted synthesis failed for segment {metadata['index']+1}: {e}")
                            if os.path.exists(temp_output_path):
                                try:
                                    os.remove(temp_output_path)
                                except Exception:
                                    pass
        except Exception as e:
            logger.warning(f"LLM adjustment step encountered an error: {e}")

        # Use the best alternative found only if it's actually better than current
        if best_alternative and os.path.exists(best_alternative) and deviation_key(best_deviation_from_range) < deviation_key(current_deviation):
            # Move the best alternative to the final output path
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(best_alternative, output_path)
            
            # Update segment data
            audio_info = AudioSegment.from_file(output_path)
            segment_dict["synthesized_speech_len"] = len(audio_info) / 1000.0
            segment_dict["synthesized_speech_file"] = output_path
            if best_key == "llm_adjusted":
                # Persist the adjusted text
                segment_dict["translation"] = metadata.get("_llm_adjusted_text", segment_dict.get("translation"))
                metadata["chosen_text"] = segment_dict["translation"]
            else:
                metadata["chosen_text"] = segment_dict[best_key]
            metadata["selected_track_type"] = best_key  # Update the selected track type

            # Update cache if needed
            if self.cache_manager.use_cache and len(audio_info) > 0:
                try:
                    shutil.copy(output_path, metadata["cache_path"])
                except Exception:
                    pass

            logger.info(f"Resynthesis successful for segment {metadata['index']+1} using '{best_key}' "
                        f"(improved from {current_deviation:.2%} to {best_deviation_from_range:.2%} deviation)")
        else:
            # Clean up the best alternative if it exists but isn't better
            if best_alternative and os.path.exists(best_alternative):
                os.remove(best_alternative)
            
            if current_ratio is not None:
                logger.info(f"No alternative found that improves deviation for segment {metadata['index']+1} "
                           f"(current: {current_deviation:.2%}). Keeping original synthesis.")
            else:
                logger.warning(f"No suitable alternative translation could improve duration for segment {metadata['index']+1}.")


    
    def _adjust_and_combine_audio_grouped(self, segments: List[Dict]) -> Tuple[AudioSegment, List[Dict]]:
        """
        Adjusts timing and combines audio segments with optimizations for speaker continuity.
        
        This method:
        1. Groups segments by speaker
        2. Optimizes each speaker's segments in continuous groups
        3. Creates separate audio tracks for each speaker
        4. Mixes all speaker tracks together
        
        Args:
            segments: List of transcript segments with translations and speaker info
            
        Returns:
            Tuple of (Combined AudioSegment with proper timing, List of real segment positions)
        """
        if not segments:
            return AudioSegment.empty(), []
        
        logger.info("Grouping segments by speaker and optimizing timing...")
        
        SPLITTING_PAUSE_THRESHOLD_SECONDS = 3
        MAX_GROUP_DURATION_SECONDS = 60
        LIMIT_MIN_ADJUSTMENT_RATIO = 0.5
        LIMIT_MAX_ADJUSTMENT_RATIO = 1.15
        
        # Get all unique speakers
        all_speakers = set(segment["speaker"] for segment in segments)
        logger.debug(f"Found {len(all_speakers)} unique speakers")
        
        # Create empty audio tracks for each speaker (full duration)
        total_duration_ms = int(max(segment["end"] for segment in segments) * 1000) + 1000  # Add 1s padding
        speaker_tracks = {speaker: AudioSegment.silent(duration=total_duration_ms) for speaker in all_speakers}
        
        # Track real segment positions for pause calculation
        real_segment_positions = []
        
        # For debug: store all speaker groups for later use in debug video
        speaker_groups_info = {}
        
        # Process each speaker's segments separately
        for speaker in sorted(all_speakers):
            # Get segments for this speaker
            speaker_segments = [segment for segment in segments if segment["speaker"] == speaker]
            logger.debug(f"Processing {len(speaker_segments)} segments for speaker {speaker}")
            
            # Group segments by continuous speech
            speaker_groups = []
            current_group = []
            
            for i, segment in enumerate(speaker_segments):
                start_new_group = False
                
                if not current_group:
                    start_new_group = True
                elif i > 0:
                    prev_segment = speaker_segments[i-1]
                    pause_duration = segment["start"] - prev_segment["end"]
                    
                    if pause_duration > SPLITTING_PAUSE_THRESHOLD_SECONDS:
                        start_new_group = True
                
                if current_group and segment["end"] - current_group[0]["start"] > MAX_GROUP_DURATION_SECONDS:
                    start_new_group = True
                
                if start_new_group and current_group:
                    speaker_groups.append(current_group)
                    current_group = []
                
                current_group.append(segment)
            
            # Add the last group
            if current_group:
                speaker_groups.append(current_group)
            
            logger.debug(f"Divided speaker {speaker} into {len(speaker_groups)} continuous groups")
            
            # Store groups information for debug
            if self.config.get('debug_info', False):
                speaker_groups_info[speaker] = speaker_groups
                self.debug_data["speaker_groups"] = speaker_groups_info
            
            # Process each group for this speaker
            for group_idx, group in enumerate(speaker_groups):
                group_start_time_ms = group[0]["start"] * 1000
                group_end_time_ms = group[-1]["end"] * 1000
                target_duration_ms = int(group_end_time_ms - group_start_time_ms)
                
                # Combine all synthesized audio in this group
                combined_group_audio = AudioSegment.empty()
                group_segment_positions = []  # Track individual segment positions within the group
                
                for i, segment in enumerate(group):
                    # Add pause between segments if not the first segment
                    segment_start_in_group_ms = len(combined_group_audio)
                    
                    if i > 0:
                        prev_segment_end = group[i-1]["end"]
                        current_segment_start = segment["start"]
                        pause_duration_ms = int((current_segment_start - prev_segment_end) * 1000)
                        if pause_duration_ms > 0:
                            combined_group_audio += AudioSegment.silent(duration=pause_duration_ms)
                            segment_start_in_group_ms = len(combined_group_audio)
                    
                    # Load segment audio
                    segment_file = segment.get('synthesized_speech_file', f"artifacts/audio_chunks/{segments.index(segment)}.wav")
                    if os.path.exists(segment_file):
                        segment_audio = AudioSegment.from_file(segment_file)
                    else:
                        # Fallback: create silence with original duration
                        duration_ms = int((segment["end"] - segment["start"]) * 1000)
                        segment_audio = AudioSegment.silent(duration=duration_ms)
                    
                    combined_group_audio += segment_audio
                    segment_end_in_group_ms = len(combined_group_audio)
                    
                    # Store the segment position within the group (before speed adjustment)
                    group_segment_positions.append({
                        "segment": segment,
                        "start_in_group_ms": segment_start_in_group_ms,
                        "end_in_group_ms": segment_end_in_group_ms,
                        "original_index": segments.index(segment)
                    })
                
                # Calculate required speed adjustment for the entire group
                actual_duration_ms = len(combined_group_audio)
                ratio = target_duration_ms / actual_duration_ms if actual_duration_ms > 0 else 1.0
                
                # Clamp ratio to maintain natural speech
                ratio_clamped = min(max(ratio, LIMIT_MIN_ADJUSTMENT_RATIO), LIMIT_MAX_ADJUSTMENT_RATIO)
                
                # Apply speed adjustment to the entire group if needed
                adjusted_group_audio = combined_group_audio
                if abs(ratio_clamped - 1.0) > 0.01:
                    try:
                        # Save the combined group audio to a temporary file
                        tmp_in = f"artifacts/audio_chunks/group_{speaker}_{group_idx}.wav"
                        tmp_out = f"artifacts/su_audio_chunks/group_{speaker}_{group_idx}.wav"
                        os.makedirs(os.path.dirname(tmp_out), exist_ok=True)
                        combined_group_audio.export(tmp_in, format="wav")
                        
                        # Store ratio for debugging
                        if self.config.get('debug_info', False):
                            for segment in group:
                                self.debug_data.setdefault("speed_ratios", {})[segments.index(segment)] = ratio_clamped
                        
                        # Apply tempo filter
                        tempo = 1.0 / ratio_clamped
                        cmd = f"ffmpeg -y -i {tmp_in} -filter:a atempo={tempo} -vn {tmp_out}"
                        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        if result.returncode == 0:
                            adjusted_group_audio = AudioSegment.from_file(tmp_out)
                        else:
                            logger.warning(f"Warning: Speed adjustment failed for group {speaker}_{group_idx}")
                    except Exception as exc:
                        logger.warning(f"Warning: Speed adjustment failed for group {speaker}_{group_idx}: {exc}")
                
                # Calculate real segment positions after speed adjustment
                final_group_duration_ms = len(adjusted_group_audio)
                position_ms = int(group_start_time_ms)
                
                for seg_pos in group_segment_positions:
                    # Apply speed ratio to get actual positions in final audio
                    real_start_ms = position_ms + (seg_pos["start_in_group_ms"] * ratio_clamped)
                    real_end_ms = position_ms + (seg_pos["end_in_group_ms"] * ratio_clamped)
                    
                    real_segment_positions.append({
                        "start": real_start_ms / 1000.0,
                        "end": real_end_ms / 1000.0,
                        "speaker": seg_pos["segment"]["speaker"],
                        "text": seg_pos["segment"]["text"],
                        "translation": seg_pos["segment"]["translation"],
                        "original_index": seg_pos["original_index"],
                        "original_start": seg_pos["segment"]["start"],
                        "original_end": seg_pos["segment"]["end"]
                    })
                
                # Place the adjusted group at the correct position in the speaker's track
                speaker_track = speaker_tracks[speaker]
                
                # Extend track if needed
                if position_ms + len(adjusted_group_audio) > len(speaker_track):
                    extension = position_ms + len(adjusted_group_audio) - len(speaker_track)
                    speaker_track += AudioSegment.silent(duration=extension)
                
                # Overlay the adjusted group audio at the correct position
                speaker_tracks[speaker] = speaker_track.overlay(
                    adjusted_group_audio,
                    position=position_ms
                )
                
                from src.utils.time_utils import format_seconds_to_srt
                duration_ms = len(adjusted_group_audio)
                duration_minutes = duration_ms / (1000 * 60)
                logger.debug(f"Processed group {group_idx+1}/{len(speaker_groups)} for speaker {speaker}: "
                      f"{len(group)} segments, Speed ratio: {ratio_clamped:.2f}, "
                      f"Time: {format_seconds_to_srt(group_start_time_ms / 1000)}-{format_seconds_to_srt((group_start_time_ms + duration_ms) / 1000)}, "
                      f"Duration: {duration_minutes:.2f} minutes")
        
        # Mix all speaker tracks together
        logger.info("Mixing all speaker tracks together...")
        final_audio = AudioSegment.silent(duration=total_duration_ms)
        
        for speaker, track in speaker_tracks.items():
            # Overlay each speaker track onto the final audio
            final_audio = final_audio.overlay(track)
            logger.debug(f"Added speaker {speaker}'s track to the mix")
        
        # If we processed the full video, pad to match original length if necessary
        if self.config.get('start_time') is None and self.config.get('duration') is None:
            try:
                total_original_ms = len(AudioSegment.from_file(self.config.get('input')))
                if len(final_audio) < total_original_ms:
                    final_audio += AudioSegment.silent(duration=total_original_ms - len(final_audio))
            except Exception as e:
                logger.warning(f"Warning: Could not pad audio to match original length: {e}")
        
        # Sort real segment positions by start time
        real_segment_positions.sort(key=lambda x: x["start"])
        
        return final_audio, real_segment_positions 

    def _get_subtitle_path(self, subtitle_type: str, input_path: str, language: str) -> str:
        """Generate subtitle path based on input file and language.
        
        Args:
            subtitle_type: "original" or "translation"
            input_path: Path to the input video file
            language: Language code (source for original, target for translation)
            
        Returns:
            Path for the subtitle file in current working directory
        """
        input_file = Path(input_path)
        filename = f"{input_file.stem}_{language}.srt"
        return filename  # Save in current working directory

    def adjust_subtitle_timestamps(self, segments: List[Dict], pause_adjustments: List[Dict[str, float]]) -> List[Dict]:
        """Adjust subtitle timestamps based on pause adjustments from video processing.
        
        Args:
            segments: List of subtitle segments with start/end times
            pause_adjustments: List of pause adjustments from video processing
            
        Returns:
            List of segments with adjusted timestamps
        """
        if not pause_adjustments:
            logger.debug("No pause adjustments to apply to subtitles")
            return segments
        
        logger.info(f"Adjusting subtitle timestamps based on {len(pause_adjustments)} pause modifications")
        
        adjusted_segments = []
        for segment in segments:
            adjusted_segment = segment.copy()
            
            # Calculate cumulative time offset for this segment's timestamps
            start_offset = 0.0
            end_offset = 0.0
            
            for adjustment in pause_adjustments:
                # If the segment starts after this pause was shortened, apply the full offset
                if segment['start'] >= adjustment['original_end']:
                    start_offset = adjustment['cumulative_offset']
                # If the segment starts during this pause, apply partial offset
                elif segment['start'] >= adjustment['original_start']:
                    # Segment starts within the pause - calculate partial offset
                    if segment['start'] <= adjustment['original_start'] + (adjustment['original_end'] - adjustment['original_start'] - adjustment['time_removed']):
                        # Segment starts in the preserved part of the pause
                        start_offset = adjustment['cumulative_offset'] - adjustment['time_removed']
                    else:
                        # Segment would have started in the removed part - move to end of preserved pause
                        start_offset = adjustment['cumulative_offset'] - adjustment['time_removed']
                        adjusted_segment['start'] = adjustment['original_start'] + (adjustment['original_end'] - adjustment['original_start'] - adjustment['time_removed'])
                
                # Apply same logic for end time
                if segment['end'] >= adjustment['original_end']:
                    end_offset = adjustment['cumulative_offset']
                elif segment['end'] >= adjustment['original_start']:
                    if segment['end'] <= adjustment['original_start'] + (adjustment['original_end'] - adjustment['original_start'] - adjustment['time_removed']):
                        end_offset = adjustment['cumulative_offset'] - adjustment['time_removed']
                    else:
                        end_offset = adjustment['cumulative_offset'] - adjustment['time_removed']
                        adjusted_segment['end'] = adjustment['original_start'] + (adjustment['original_end'] - adjustment['original_start'] - adjustment['time_removed'])
            
            # Apply the calculated offsets
            adjusted_segment['start'] = max(0, adjusted_segment['start'] - start_offset)
            adjusted_segment['end'] = max(adjusted_segment['start'], adjusted_segment['end'] - end_offset)
            
            adjusted_segments.append(adjusted_segment)
        
        logger.debug(f"Adjusted timestamps for {len(adjusted_segments)} subtitle segments")
        return adjusted_segments 