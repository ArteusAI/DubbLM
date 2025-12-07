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
import random
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Literal, Union
from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Disable all warnings for a cleaner output.
warnings.filterwarnings("ignore")

# Import our components
from .config import DubbingConfig
from .cache_manager import CacheManager
from .segment_optimizer import SegmentOptimizer
from ..audio.audio_processor import AudioProcessor
from ..audio.speaker_processor import SpeakerProcessor
from ..audio.time_stretcher import TimeStretcher
from ..video.video_processor import VideoProcessor
from ..debug.performance_tracker import PerformanceTracker
from ..debug.debug_generator import DebugGenerator
from ..debug.reporter import SpeakerReporter
from ..debug.cost_tracker import CostTracker
from ..utils.subtitle_utils import SubtitleManager
from .log_config import get_logger
from .cost_estimator import CostEstimator

# Import existing factories and interfaces
from src.tts.tts_factory import TTSFactory
from src.translation.translator_factory import TranslatorFactory
from src.transcription.transcription_factory import TranscriptionFactory

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
        
        # Speakers to mute (remove entirely from output)
        self.muted_speakers = set()
        mute_cfg = self.config.get('mute_speakers')
        if isinstance(mute_cfg, str) and mute_cfg.strip():
            self.muted_speakers = {mute_cfg.strip()}
        elif isinstance(mute_cfg, (list, tuple, set)):
            self.muted_speakers = {str(s).strip() for s in mute_cfg if isinstance(s, (str,)) and str(s).strip()}
        if self.muted_speakers:
            logger.info(f"Muted speakers: {sorted(self.muted_speakers)}")

        # Set device
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.torch_device = self._get_torch_device(device)

        # Initialize core components
        self.cache_manager = CacheManager(
            use_cache=not config.get('no_cache', False),
            input_file=config.get('input')
        )
        self.performance_tracker = PerformanceTracker()
        self.cost_tracker = CostTracker(config)
        self.segment_optimizer = SegmentOptimizer(config, tts_system=config.get('tts_system'))
        
        # Initialize processors
        self.audio_processor = AudioProcessor(self.cache_manager, self.performance_tracker)
        self.speaker_processor = SpeakerProcessor(self.cache_manager, self.performance_tracker)
        self.video_processor = VideoProcessor(self.performance_tracker)
        
        # Initialize time stretcher with preferred method
        time_stretch_method = config.get('time_stretch_method', 'auto')
        self.time_stretcher = TimeStretcher(preferred_method=time_stretch_method)
        
        # Initialize utilities
        self.subtitle_manager = SubtitleManager()
        self.debug_generator = DebugGenerator()
        self.speaker_reporter = SpeakerReporter(self.performance_tracker)
        
        
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

    def _apply_speaker_filter(self, segments: List[Dict]) -> List[Dict]:
        """Return segments with muted speakers removed (if configured)."""
        if not segments or not self.muted_speakers:
            return segments
        filtered = [s for s in segments if s.get("speaker") not in self.muted_speakers]
        if self.config.get('debug_info', False):
            removed = len(segments) - len(filtered)
            logger.debug(f"Speaker filter applied: muted={sorted(self.muted_speakers)} removed={removed} kept={len(filtered)}")
        return filtered
    
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
                translation_prompt_prefix=self.config.get('translation_prompt_prefix'),
                glossary=self.config.get('glossary'),
                cache_manager=self.cache_manager,
                cost_tracker=self.cost_tracker,
                enable_emotion_enrichment=self.config.get('enable_emotion_enrichment', False),
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
                debug_tts=self.config.get('debug_tts', False),
                model=self.config.get('tts_model'),
                fallback_model=self.config.get('tts_fallback_model'),
                enable_emotion_enrichment=self.config.get('enable_emotion_enrichment', False),
                emotion_enrichment_model=self.config.get('emotion_enrichment_model'),
                emotion_enrichment_temperature=self.config.get('emotion_enrichment_temperature'),
                max_workers=self.config.get('max_workers', 4),
                cost_tracker=self.cost_tracker,
                translator=self.translator,
                target_language=self.config.get('target_language', 'en')  # Pass target language for language-specific TTS configuration
            )
            self.tts_systems[self.config.get('tts_system', 'coqui')] = tts_instance
            self.default_tts = tts_instance
            logger.debug(f"Initialized {self.config.get('tts_system', 'coqui')} TTS system with target language: {self.config.get('target_language', 'en')}")
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
                cache_manager=self.cache_manager,
                cost_tracker=self.cost_tracker,
                speakers_expected=self.config.get('speakers_expected')
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
            self.cost_tracker.set_audio_duration(self.audio_processor.get_total_duration())

            # Perform cost estimation before starting the actual pipeline work
            try:
                logger.info("Estimating pipeline costs before processing...")
                estimator = CostEstimator(self.config, cost_tracker=self.cost_tracker)
                estimated_costs = estimator.estimate(write_summary=False)
                estimated_total = estimated_costs.get('total', 0.0)
                logger.info(f"Estimated total cost: ${estimated_total:.4f}")
                logger.debug("Cost estimation completed")
            except Exception as est_exc:
                logger.warning(f"Failed to estimate costs: {est_exc}")

            # Perform speaker diarization and transcription
            speakers_rolls, transcription = self.diarize_and_transcribe(audio_file)
            if speakers_rolls is None or len(speakers_rolls) == 0:
                raise ValueError("No speakers found in the video")
            
            # If debug_diarize_only is True, generate debug video and exit early
            if self.config.get('debug_diarize_only', False):
                return self._handle_debug_diarize_only(audio_file, speakers_rolls)
            
            # Extract audio for each speaker
            self.speaker_processor.extract_speaker_audio(audio_file, speakers_rolls)
            
            # Clone voices if requested
            if self.config.get('clone_voice'):
                voice_mapping = self.clone_speakers_voices(speakers_rolls, audio_file)
                logger.info("=" * 70)
                logger.info("VOICE CLONING COMPLETED")
                logger.info("=" * 70)
                logger.info(f"Voice mapping saved to: artifacts/cloned_voices/voice_mapping.json")
                logger.info(f"Test samples saved to: artifacts/cloned_voices/")
                logger.info("")
                logger.info("To use cloned voices in future runs:")
                logger.info("  1. Note the cloned voice IDs from the mapping file")
                logger.info("  2. Use them with --voice_name option or in your config file")
                logger.info("=" * 70)
                
                # Exit early after cloning
                return ""
            
            # Translate segments
            translated_segments = self.translate_segments(transcription, audio_file)

            # Apply optional speaker mute filter for downstream steps
            segments_for_output = self._apply_speaker_filter(translated_segments)
            
            # Save debug TSV
            self.subtitle_manager.save_debug_tsv(segments_for_output)
            
            # Save subtitles if requested (only if pause processing is disabled)
            pause_removal_mode = self.config.get('pause_removal', 'disabled')
            if pause_removal_mode == 'disabled':
                if save_original_subtitles:
                    self.subtitle_manager.save_subtitles(segments_for_output, "original", self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language')))
                
                if save_translated_subtitles:
                    self.subtitle_manager.save_subtitles(segments_for_output, "translation", self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language')))
            
            # Synthesize speech or generate silence if no segments remain after muting
            if segments_for_output and len(segments_for_output) > 0:
                translated_audio_path = self.synthesize_speech(segments_for_output, speakers_rolls, audio_file)
                
                # Check if synthesis was skipped due to exit_before_synthesis flag
                if not translated_audio_path:
                    logger.info("Pipeline stopped before speech synthesis as requested.")
                    # Write partial performance summary
                    self.performance_tracker.record_metric("total", time.perf_counter() - pipeline_start_time)
                    self.performance_tracker.set_costs(self.cost_tracker.get_costs_by_step())
                    self.performance_tracker.write_performance_summary(self.audio_processor.get_total_duration())
                    self.cost_tracker.write_cost_summary()
                    return ""
            else:
                logger.info("All segments filtered by mute_speakers; generating silent audio track...")
                total_duration_sec = self.audio_processor.get_total_duration() or 0
                silent_ms = int(max(0, total_duration_sec) * 1000)
                silent_audio = AudioSegment.silent(duration=silent_ms)
                os.makedirs("artifacts/audio", exist_ok=True)
                translated_audio_path = "artifacts/audio/output.wav"
                silent_audio.export(translated_audio_path, format="wav")
            
            # Save translated samples
            self.speaker_processor.save_translated_samples(segments_for_output, audio_file)
            
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
            
            # Determine original audio keep ranges when including original audio and muting speakers
            keep_original_audio_ranges = self.config.get('keep_original_audio_ranges')
            if keep_original_audio_ranges is None and self.config.get('include_original_audio', False) and self.muted_speakers:
                try:
                    # Keep only ranges where non-muted speakers talk
                    keep_original_audio_ranges = [
                        (start, end) for (start, end), spk in (speakers_rolls or {}).items() if spk not in self.muted_speakers
                    ]
                    if keep_original_audio_ranges:
                        logger.info(f"Computed keep_original_audio_ranges excluding muted speakers ({len(keep_original_audio_ranges)} ranges)")
                except Exception:
                    # Fallback silently if structure is unexpected
                    keep_original_audio_ranges = self.config.get('keep_original_audio_ranges')

            # Combine with video (includes pause processing if enabled)
            segments_opt = self.config.get('segments_optimization', {})
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
                keep_original_audio_ranges=keep_original_audio_ranges,
                source_language=self.config.get('source_language'),
                target_language=self.config.get('target_language'),
                normalize_audio=self.config.get('normalize_audio', True),
                use_two_pass_encoding=self.config.get('use_two_pass_encoding', True),
                pause_removal=self.config.get('pause_removal', 'disabled'),
                min_pause_duration=segments_opt.get('min_pause_duration', 3),
                preserve_pause_duration=segments_opt.get('preserve_pause_duration', 1.5),
                video_speed_min=segments_opt.get('video_speed_min', 0.75),
                video_speed_comfortable=segments_opt.get('video_speed_comfortable', 1.25),
                video_speed_max=segments_opt.get('video_speed_max', 1.5),
                segment_positions=self.real_segment_positions,
                keyframe_buffer=self.config.get('keyframe_buffer', 0.2),
                ffmpeg_batch_size=self.config.get('ffmpeg_batch_size', 50),
                dubbed_volume=self.config.get('dubbed_volume', 1.0),
                background_volume=self.config.get('background_volume', 0.562341),
                upscale_factor=self.config.get('upscale_factor', 1.0),
                upscale_sharpen=self.config.get('upscale_sharpen', True)
            )
            
            # Store pause adjustments for potential future use
            self.pause_adjustments = pause_adjustments
            
            # Save subtitles after pause processing if pause removal/speedup is enabled
            if pause_removal_mode != 'disabled' and (save_original_subtitles or save_translated_subtitles):
                if pause_adjustments:
                    logger.info("Adjusting subtitle timestamps based on pause modifications...")
                    
                    if save_original_subtitles:
                        adjusted_original_segments = self.adjust_subtitle_timestamps(segments_for_output, pause_adjustments)
                        self.subtitle_manager.save_subtitles(adjusted_original_segments, "original", self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language')))
                        adjusted_path = self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language'))
                        logger.info(f"Saved pause-corrected original subtitles to {adjusted_path}")
                    
                    if save_translated_subtitles:
                        adjusted_translated_segments = self.adjust_subtitle_timestamps(segments_for_output, pause_adjustments)
                        self.subtitle_manager.save_subtitles(adjusted_translated_segments, "translation", self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language')))
                        adjusted_path = self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language'))
                        logger.info(f"Saved pause-corrected translated subtitles to {adjusted_path}")
                else:
                    # No pause adjustments made, but pause removal was enabled - save original timestamps
                    logger.info("No pause adjustments needed, saving subtitles with original timestamps...")
                    
                    if save_original_subtitles:
                        self.subtitle_manager.save_subtitles(segments_for_output, "original", self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language')))
                        subtitle_path = self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language'))
                        logger.info(f"Saved original subtitles to {subtitle_path}")
                    
                    if save_translated_subtitles:
                        self.subtitle_manager.save_subtitles(segments_for_output, "translation", self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language')))
                        subtitle_path = self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language'))
                        logger.info(f"Saved translated subtitles to {subtitle_path}")
            
            # Overall pipeline metrics
            total_elapsed = time.perf_counter() - pipeline_start_time
            logger.info("Dubbing process completed!")
            self.performance_tracker.record_metric("total", total_elapsed)
            self.performance_tracker.set_costs(self.cost_tracker.get_costs_by_step())
            
            # Write performance summary
            self.performance_tracker.write_performance_summary(self.audio_processor.get_total_duration())
            self.cost_tracker.write_cost_summary()
            
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
        self.performance_tracker.set_costs(self.cost_tracker.get_costs_by_step())
        self.performance_tracker.write_performance_summary(self.audio_processor.get_total_duration())
        self.cost_tracker.write_cost_summary()
        
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
        self.cost_tracker.set_audio_duration(self.audio_processor.get_total_duration())

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
        self.performance_tracker.set_costs(self.cost_tracker.get_costs_by_step())
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

        # Optimize segments after diarization
        transcription = self.segment_optimizer.optimize_post_diarization(transcription)

        # Store for debug
        self.debug_data["diarization"] = speakers_rolls
        self.debug_data["transcription"] = transcription
        
        # Save transcription to file
        self._save_transcription_file(transcription)
        
        return speakers_rolls, transcription
    
    def clone_speakers_voices(self, speakers_rolls: Dict, audio_file: str) -> Dict[str, str]:
        """
        Clone voices for specified speakers.
        
        Args:
            speakers_rolls: Dictionary mapping time ranges to speaker IDs
            audio_file: Path to the audio file
            
        Returns:
            Dictionary mapping speaker IDs to cloned voice IDs
        """
        import json
        
        # Parse comma-separated speaker list
        clone_voice_config = self.config.get('clone_voice')
        if not clone_voice_config:
            logger.warning("No speakers specified for voice cloning")
            return {}
        
        speakers_to_clone = [s.strip() for s in clone_voice_config.split(',') if s.strip()]
        if not speakers_to_clone:
            logger.warning("No valid speakers found in clone_voice config")
            return {}
        
        # Parse custom voice names if provided
        custom_voice_names = []
        voice_names_config = self.config.get('voice_names')
        if voice_names_config:
            custom_voice_names = [n.strip() for n in voice_names_config.split(',') if n.strip()]
            if len(custom_voice_names) != len(speakers_to_clone):
                logger.warning(
                    f"Number of voice names ({len(custom_voice_names)}) does not match "
                    f"number of speakers ({len(speakers_to_clone)}). "
                    f"Auto-generating voice IDs instead."
                )
                custom_voice_names = []
            else:
                logger.info(f"Using custom voice names: {', '.join(custom_voice_names)}")
        
        logger.info(f"Starting voice cloning for speakers: {', '.join(speakers_to_clone)}")
        
        # Get all unique speakers from diarization
        all_speakers = set(speakers_rolls.values())
        
        # Verify requested speakers exist
        missing_speakers = [s for s in speakers_to_clone if s not in all_speakers]
        if missing_speakers:
            logger.warning(f"Speakers not found in diarization: {', '.join(missing_speakers)}")
            logger.info(f"Available speakers: {', '.join(sorted(all_speakers))}")
        
        # Filter to only speakers that exist
        speakers_to_clone = [s for s in speakers_to_clone if s in all_speakers]
        if not speakers_to_clone:
            logger.error("None of the requested speakers were found in the diarization")
            return {}
        
        # Get the TTS system to use for cloning
        tts_system = self.config.get('tts_system', 'minimax')
        if tts_system != 'minimax':
            logger.warning(f"Voice cloning is only supported with Minimax TTS. Current TTS system: {tts_system}")
            logger.info("Please set --tts_system minimax to use voice cloning")
            return {}
        
        tts_instance = self.tts_systems.get(tts_system)
        if not tts_instance:
            logger.error(f"TTS system {tts_system} not initialized")
            return {}
        
        # Create output directory for cloned voices
        output_dir = Path("artifacts/cloned_voices")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        voice_mapping = {}
        
        # Clone each speaker's voice
        for idx, speaker in enumerate(speakers_to_clone):
            try:
                logger.info(f"Cloning voice for {speaker}...")
                
                # Get speaker audio file
                speaker_audio_path = f"artifacts/speakers_audio/{speaker}.wav"
                if not os.path.exists(speaker_audio_path):
                    logger.error(f"Speaker audio file not found: {speaker_audio_path}")
                    continue
                
                # Generate voice_id - use custom name if provided, otherwise auto-generate
                if custom_voice_names and idx < len(custom_voice_names):
                    voice_id = custom_voice_names[idx]
                    logger.info(f"Using custom voice ID: {voice_id}")
                else:
                    voice_id = f"cloned_{speaker}"
                    logger.info(f"Using auto-generated voice ID: {voice_id}")
                
                # Clone the voice
                try:
                    cloned_voice_id = tts_instance.clone_voice(speaker_audio_path, voice_id)
                    logger.info(f"Successfully cloned voice for {speaker} with ID: {cloned_voice_id}")
                    voice_mapping[speaker] = cloned_voice_id
                except Exception as clone_error:
                    logger.error(f"Failed to clone voice for {speaker}: {clone_error}")
                    continue
                
                # Generate test sample
                test_text = self._get_test_text_for_language(speaker)
                test_output_path = output_dir / f"{speaker}_test_sample.mp3"
                
                try:
                    from src.tts.models import TTSSegmentData
                    
                    test_segment = TTSSegmentData(
                        speaker=speaker,
                        text=test_text,
                        voice=cloned_voice_id,
                        output_path=str(test_output_path)
                    )
                    
                    logger.info(f"Generating test sample for {speaker}...")
                    tts_instance.synthesize(
                        segments_data=[test_segment],
                        language=self.config.get('target_language', 'en')
                    )
                    
                    if test_output_path.exists():
                        logger.info(f"Test sample saved to: {test_output_path}")
                    else:
                        logger.warning(f"Test sample generation may have failed for {speaker}")
                        
                except Exception as synth_error:
                    logger.error(f"Failed to generate test sample for {speaker}: {synth_error}")
                
            except Exception as e:
                logger.error(f"Error processing speaker {speaker}: {e}")
        
        # Save voice mapping to JSON
        if voice_mapping:
            mapping_file = output_dir / "voice_mapping.json"
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(voice_mapping, f, indent=2)
            logger.info(f"Voice mapping saved to: {mapping_file}")
            
            # Also log the mapping for easy reference
            logger.info("Voice cloning completed successfully!")
            logger.info("Cloned voice IDs:")
            for speaker, voice_id in voice_mapping.items():
                logger.info(f"  {speaker} -> {voice_id}")
        else:
            logger.warning("No voices were successfully cloned")
        
        return voice_mapping
    
    def _get_test_text_for_language(self, speaker_name: str) -> str:
        """Generate test text based on target language."""
        target_lang = self.config.get('target_language', 'en')
        
        # Map of language codes to test phrases
        test_phrases = {
            'en': f"Hello, this is a test of the cloned voice for {speaker_name}.",
            'es': f"Hola, esta es una prueba de la voz clonada para {speaker_name}.",
            'fr': f"Bonjour, ceci est un test de la voix clonée pour {speaker_name}.",
            'de': f"Hallo, dies ist ein Test der geklonten Stimme für {speaker_name}.",
            'it': f"Ciao, questo è un test della voce clonata per {speaker_name}.",
            'pt': f"Olá, este é um teste da voz clonada para {speaker_name}.",
            'ru': f"Привет, это тест клонированного голоса для {speaker_name}.",
            'zh': f"你好，这是{speaker_name}克隆语音的测试。",
            'ja': f"こんにちは、これは{speaker_name}のクローン音声のテストです。",
            'ko': f"안녕하세요, 이것은 {speaker_name}의 복제된 음성 테스트입니다."
        }
        
        return test_phrases.get(target_lang, test_phrases['en'])
    
    def translate_segments(
        self, 
        transcription: List[Dict], 
        audio_file: str,
        progress_callback: callable = None
    ) -> List[Dict]:
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
                    debug=self.debug_data,
                    enable_emotion_enrichment=self.config.get('enable_emotion_enrichment', False),
                    progress_callback=progress_callback
                )
            else:
                raise ValueError("No translator available")
            
            # Save results to cache
            self.cache_manager.save_to_cache(step_name, cache_key, translated_segments)
            
            # End timing
            elapsed_time = self.performance_tracker.end_timing("translation")
            logger.info(f"Finished translation in {elapsed_time:.2f} seconds (≈ {elapsed_time/60:.2f} minutes)")

        # Optimize segments after translation
        initial_count = len(transcription)
        pre_opt_count = len(translated_segments)
        translated_segments = self.segment_optimizer.optimize_post_translation(translated_segments)

        # Log final translation pipeline statistics
        logger.info(
            f"Translation pipeline: {initial_count} input segments → {pre_opt_count} translated segments → "
            f"{len(translated_segments)} final optimized segments"
        )

        # Store for debug
        self.debug_data["translation"] = translated_segments

        return translated_segments
    
    def synthesize_speech(
        self, 
        segments: List[Dict], 
        speakers_rolls: Dict, 
        audio_file: str,
        progress_callback: callable = None,
        grouping_progress_callback: callable = None
    ) -> str:
        """
        Synthesize speech for translated segments with optimized batching and estimation.
        
        Args:
            segments: List of transcript segments with translations
            speakers_rolls: Dictionary mapping time ranges to speaker IDs
            audio_file: Path to the audio file for cache key
            progress_callback: Optional callback for TTS progress updates (current, total, text)
            grouping_progress_callback: Optional callback for grouping/overlay progress (current, total, message)
            
        Returns:
            Path to the output audio file
        """
        if not segments:
            raise ValueError("Cannot synthesize speech with no segments.")

        # Start timing
        self.performance_tracker.start_timing("speech_synthesis")
        
        cache_key = f"{self.cache_manager.generate_cache_key(audio_file, self.config.get('source_language'), self.config.get('target_language'), self.config.get('whisper_model', 'large-v3'), self.config.get('start_time'), self.config.get('duration'))}_{self.config.get('target_language')}_{self.config.get('tts_system')}"
        step_name = "synthesized_speech"
        
        # Check for editable segments file first
        segments_step_name = "segments_for_synthesis"
        editable_data = self.cache_manager.load_segments_json(segments_step_name, cache_key, "_editable")
        
        if editable_data and editable_data.get("segments"):
            logger.info("=" * 70)
            logger.info("USING EDITABLE SEGMENTS FILE")
            logger.info("=" * 70)
            
            edited_segments = editable_data.get("segments", [])
            
            # Count force resynthesis flags
            force_resynth_count = sum(1 for seg in edited_segments if seg.get("force_resynthesize", False))
            
            logger.info(f"Loaded {len(edited_segments)} segments from editable file")
            if force_resynth_count > 0:
                logger.info(f"Force resynthesis enabled for {force_resynth_count} segment(s)")
            logger.info("=" * 70)
            
            # Use edited segments
            segments = edited_segments

        # Prepare segments for saving (add force_resynthesize field, remove words to reduce size)
        segments_to_save = []
        for seg in segments:
            seg_copy = seg.copy()
            seg_copy["force_resynthesize"] = seg_copy.get("force_resynthesize", False)
            # Initialize chosen_text and selected_track_type if not present
            if "chosen_text" not in seg_copy:
                seg_copy["chosen_text"] = ""
            if "selected_track_type" not in seg_copy:
                seg_copy["selected_track_type"] = ""
            #seg_copy.pop("words", None)
            segments_to_save.append(seg_copy)
        
        metadata = {
            "source_language": self.config.get('source_language'),
            "target_language": self.config.get('target_language'),
            "tts_system": self.config.get('tts_system')
        }
        editable_path = self.cache_manager.save_segments_json(
            segments_step_name, cache_key, segments_to_save, metadata, "_editable"
        )
        
        # If exit_before_synthesis is True, save editable copy and exit
        if self.config.get('exit_before_synthesis', False):            
            logger.info("=" * 70)
            logger.info("EXITING BEFORE SPEECH SYNTHESIS")
            logger.info("=" * 70)
            logger.info(f"Segments saved to: {editable_path}")
            logger.info("")
            logger.info("To edit translations:")
            logger.info(f"  1. Open and edit: {editable_path}")
            logger.info(f"  2. Modify 'translation' field(s) as needed")
            logger.info(f"  3. Set 'force_resynthesize': true for segments to force regeneration")
            logger.info(f"  4. Rerun without --exit_before_synthesis flag")
            logger.info("")
            logger.info("The pipeline will automatically use your edited file.")
            logger.info("=" * 70)
            
            # End timing
            self.performance_tracker.end_timing("speech_synthesis")
            
            # Return empty string to signal early exit
            return ""
        
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
        from src.tts.models import TTSSegmentData
        
        # Define comfort ratio constants
        # Allow comfort zone override from config (optionally language-specific)
        COMFORT_MIN_ADJUSTMENT_RATIO = self.config.get('segments_optimization', {}).get('comfort_min_adjustment_ratio', 0.85)
        COMFORT_MAX_ADJUSTMENT_RATIO = self.config.get('segments_optimization', {}).get('comfort_max_adjustment_ratio', 1.15)
        use_enriched_for_tts = self.config.get('enable_emotion_enrichment', False)
        
        # Iteratively estimate and synthesize segments to leverage dynamic duration stats
        segments_metadata: List[Dict[str, Any]] = []
        total_segments = len(segments)
        target_language = self.config.get('target_language')

        # Track estimation accuracy
        estimation_stats = {
            "total_segments": 0,
            "accurate_estimations": 0,  # Within comfort zone
            "inaccurate_estimations": 0,  # Outside comfort zone, needed resynthesis
        }

        logger.info(f"Iteratively synthesizing {total_segments} segments with dynamic duration feedback...")

        # Prepare speaker queues while preserving original order
        speaker_to_indices: Dict[str, List[int]] = {}
        for idx, seg in enumerate(segments):
            speaker_id = seg["speaker"]
            speaker_to_indices.setdefault(speaker_id, []).append(idx)

        max_workers_config = self.config.get('max_workers', 4)
        if not max_workers_config or max_workers_config <= 0:
            max_workers_config = 1
        max_workers = min(max_workers_config, max(1, len(speaker_to_indices)))

        logger.debug(f"Using up to {max_workers} worker(s) for parallel synthesis across speakers")

        # Shared synchronization primitives
        debug_data_lock = threading.Lock()
        metadata_lock = threading.Lock()
        progress_lock = threading.Lock()
        progress_state = {"completed": 0}

        # Ensure each TTS instance has a synthesis lock to avoid concurrent model switching
        tts_locks: Dict[str, threading.Lock] = {}
        for system_name, tts_instance in self.tts_systems.items():
            lock = getattr(tts_instance, "_synthesis_lock", None)
            if lock is None:
                lock = threading.Lock()
                setattr(tts_instance, "_synthesis_lock", lock)
            tts_locks[system_name] = lock

        def increment_progress(segment_index: int, speaker_id: str, note: str, text: str = "") -> None:
            with progress_lock:
                progress_state["completed"] += 1
                completed = progress_state["completed"]
            logger.info(
                f"Completed segment {segment_index+1}/{total_segments} for speaker '{speaker_id}' "
                f"- {note}"
            )
            if progress_callback:
                progress_callback(completed, total_segments, text)

        def select_best_text_variant(
            segment_index: int,
            segment_dict: Dict[str, Any],
            tts_segment_data_args: Dict[str, Any],
            tts_instance,
            tts_system: str,
            original_duration: float
        ) -> Tuple[str, float, float, str]:
            base_text = segment_dict.get("translation", "")
            best_text = base_text
            best_track_type = "translation"
            best_ratio = 1.0
            best_deviation = 0.0

            try:
                normal_duration = tts_instance.estimate_audio_segment_length(
                    TTSSegmentData(**tts_segment_data_args),
                    language=target_language
                )
            except Exception as estimate_exc:
                logger.warning(
                    f"Segment {segment_index+1} ({tts_system}): duration estimation failed for base translation: {estimate_exc}"
                )
                return best_text, best_ratio, best_deviation, best_track_type

            if not normal_duration or normal_duration <= 0:
                logger.warning(
                    f"Segment {segment_index+1} ({tts_system}): Invalid duration estimate for base translation. Using as-is."
                )
                return best_text, best_ratio, best_deviation, best_track_type

            ratio_normal = original_duration / normal_duration if normal_duration > 0 else 1.0
            deviation_normal = self._calculate_percentage_deviation(
                ratio_normal,
                COMFORT_MIN_ADJUSTMENT_RATIO,
                COMFORT_MAX_ADJUSTMENT_RATIO,
            )
            best_ratio = ratio_normal
            best_deviation = deviation_normal

            logger.debug(
                f"Segment {segment_index+1} ({tts_system}): Normal translation - Estimated duration: {normal_duration:.2f}s, "
                f"Ratio: {ratio_normal:.2f}, Deviation: {deviation_normal:.2%}"
            )

            if deviation_normal == 0.0:
                logger.debug("  Normal translation is within comfort zone. Selecting it.")
                return best_text, best_ratio, best_deviation, best_track_type

            alternatives: List[Tuple[str, str]] = []
            if ratio_normal < COMFORT_MIN_ADJUSTMENT_RATIO:
                if segment_dict.get("very_short_translation"):
                    alternatives.append(("very_short_translation", segment_dict["very_short_translation"]))
                if segment_dict.get("short_translation"):
                    alternatives.append(("short_translation", segment_dict["short_translation"]))
            elif ratio_normal > COMFORT_MAX_ADJUSTMENT_RATIO:
                if segment_dict.get("long_translation"):
                    alternatives.append(("long_translation", segment_dict["long_translation"]))

            if alternatives:
                logger.debug(f"  Evaluating {len(alternatives)} alternative(s) for segment {segment_index+1}...")
                for alt_key, alt_text in alternatives:
                    if not alt_text:
                        continue
                    alt_args = {**tts_segment_data_args, "text": alt_text}
                    try:
                        alt_duration = tts_instance.estimate_audio_segment_length(
                            TTSSegmentData(**alt_args),
                            language=target_language
                        )
                    except Exception as alt_exc:
                        logger.warning(
                            f"Segment {segment_index+1} ({tts_system}): duration estimation failed for {alt_key}: {alt_exc}"
                        )
                        continue

                    if not alt_duration or alt_duration <= 0:
                        continue

                    alt_ratio = original_duration / alt_duration if alt_duration > 0 else 1.0
                    alt_deviation = self._calculate_percentage_deviation(
                        alt_ratio,
                        COMFORT_MIN_ADJUSTMENT_RATIO,
                        COMFORT_MAX_ADJUSTMENT_RATIO,
                    )

                    logger.debug(
                        f"    Alternative {alt_key.replace('_', ' ')} - Estimated duration: {alt_duration:.2f}s, "
                        f"Ratio: {alt_ratio:.2f}, Deviation: {alt_deviation:.2%}"
                    )

                    if alt_deviation < best_deviation:
                        best_text = alt_text
                        best_ratio = alt_ratio
                        best_deviation = alt_deviation
                        best_track_type = alt_key
                        logger.debug(
                            f"      New best: {alt_key.replace('_', ' ').title()} (Deviation: {best_deviation:.2%})"
                        )
                        if best_deviation == 0.0:
                            break

            logger.debug(
                f"  Selected for synthesis: '{best_text[:50]}...' (Ratio: {best_ratio:.2f}, Deviation: {best_deviation:.2%})"
            )
            return best_text, best_ratio, best_deviation, best_track_type

        def handle_segment(segment_index: int) -> Optional[Dict[str, Any]]:
            segment_dict = segments[segment_index]
            speaker = segment_dict["speaker"]
            tts_system = self._get_tts_system_for_speaker(speaker)
            tts_instance = self.tts_systems.get(tts_system)
            if not tts_instance:
                logger.warning(f"TTS system {tts_system} not available, falling back to default")
                tts_instance = self.default_tts
            if not tts_instance:
                raise ValueError(f"No TTS instance available for speaker '{speaker}' (system {tts_system})")

            segment_style_prompt = self.voice_prompt.get(speaker, None)

            voice_name = None
            voice_config = self.config.get('voice_name')
            if isinstance(voice_config, dict):
                voice_name = voice_config.get(speaker, next(iter(voice_config.values()), "default"))
            elif isinstance(voice_config, str):
                voice_name = voice_config

            if self.config.get('debug_info', False):
                with debug_data_lock:
                    self.debug_data["voices"][segment_index] = {
                        "speaker": speaker,
                        "voice": voice_name,
                        "style_prompt": segment_style_prompt,
                        "tts_system": tts_system
                    }

            active_segment = segment_dict.copy()
            # Prefer previously chosen text for cache key to maximize cache hits across edits
            preferred_text = (segment_dict.get("chosen_text") or active_segment.get("translation", ""))
            voice_prompt_hash = hashlib.md5((segment_style_prompt or "").encode()).hexdigest()[:8]

            if use_enriched_for_tts:
                enrichment_overrides = {
                    "translation": "emotion_enriched_translation",
                    "very_short_translation": "emotion_enriched_very_short_translation",
                    "short_translation": "emotion_enriched_short_translation",
                    "long_translation": "emotion_enriched_long_translation",
                }
                for base_key, enriched_key in enrichment_overrides.items():
                    enriched_val = segment_dict.get(enriched_key)
                    if enriched_val:
                        active_segment[base_key] = enriched_val

            def make_cache_key(text_value: str) -> str:
                text_hash = hashlib.md5((text_value or "").encode()).hexdigest()[:8]
                return f"{base_cache_prefix}_{tts_system}_{segment_index}_{speaker}_{text_hash}_{voice_prompt_hash}"

            # Default cache key/path (used when saving new audio)
            segment_cache_key = make_cache_key(preferred_text)
            current_segment_output_path = f"artifacts/audio_chunks/{segment_index}.wav"
            os.makedirs(os.path.dirname(current_segment_output_path), exist_ok=True)
            segment_cached_file_path = segment_cache_path / f"{segment_cache_key}.wav"

            # Build candidate cached paths to allow reuse of prior variants (chosen/translation/short/long/very_short)
            candidate_texts = []
            if segment_dict.get("chosen_text"):
                candidate_texts.append(segment_dict.get("chosen_text", ""))
            # If a specific track type was selected earlier, include its text
            selected_track_type = segment_dict.get("selected_track_type")
            if selected_track_type and isinstance(selected_track_type, str) and active_segment.get(selected_track_type):
                candidate_texts.append(active_segment.get(selected_track_type, ""))
            # Fallbacks
            for key in ("translation", "short_translation", "long_translation", "very_short_translation"):
                if active_segment.get(key):
                    candidate_texts.append(active_segment.get(key, ""))
            # De-duplicate while preserving order
            seen_texts = set()
            candidate_texts = [t for t in candidate_texts if not (t in seen_texts or seen_texts.add(t))]

            # Check cache only if force_resynthesize is not set; try multiple candidate keys to maximize hits
            force_resynth = segment_dict.get("force_resynthesize", False)
            if self.cache_manager.use_cache and not force_resynth:
                cached_path_to_use = None
                for text_variant in candidate_texts:
                    candidate_key = make_cache_key(text_variant)
                    candidate_path = segment_cache_path / f"{candidate_key}.wav"
                    if candidate_path.exists():
                        try:
                            cached_audio_info = AudioSegment.from_file(candidate_path)
                            if len(cached_audio_info) > 0:
                                cached_path_to_use = candidate_path
                                break
                            else:
                                os.remove(candidate_path)
                        except Exception:
                            try:
                                os.remove(candidate_path)
                            except Exception:
                                pass

                if cached_path_to_use is not None:
                    shutil.copy(cached_path_to_use, current_segment_output_path)
                    cached_audio_info = AudioSegment.from_file(current_segment_output_path)
                    segment_dict['synthesized_speech_len'] = len(cached_audio_info) / 1000.0
                    segment_dict['synthesized_speech_file'] = current_segment_output_path
                    # Keep existing chosen_text/selected_track_type if present; otherwise infer from preferred_text
                    if not segment_dict.get('chosen_text'):
                        segment_dict['chosen_text'] = preferred_text
                    if not segment_dict.get('selected_track_type'):
                        segment_dict['selected_track_type'] = 'translation'
                    increment_progress(segment_index, speaker, "cache hit", preferred_text)
                    return None

            tts_segment_data_args = {
                "speaker": speaker,
                "text": active_segment["translation"],
                "emotion": segment_dict.get("emotion", "Neutral"),
                "style_prompt": segment_style_prompt,
                "reference_audio_path": None,
                "reference_text": None,
                "voice": voice_name,
                "speed": 1.0
            }

            potential_ref_audio_for_speaker = f"artifacts/speakers_audio/{speaker}.wav"
            if os.path.exists(potential_ref_audio_for_speaker):
                tts_segment_data_args["reference_audio_path"] = potential_ref_audio_for_speaker

            original_duration = segment_dict["end"] - segment_dict["start"]

            best_text, best_ratio, best_deviation, best_track_type = select_best_text_variant(
                segment_index,
                active_segment,
                tts_segment_data_args,
                tts_instance,
                tts_system,
                original_duration
            )

            final_segment_data = TTSSegmentData(
                **{**tts_segment_data_args, "text": best_text, "output_path": current_segment_output_path}
            )

            previous_texts: List[str] = []
            if self.config.get('enable_emotion_enrichment', False):
                for j in range(max(0, segment_index - 5), segment_index):
                    prev_segment = segments[j]
                    prev_text = prev_segment.get('emotion_enriched_translation') or prev_segment.get('translation', '')
                    if prev_text:
                        previous_texts.append(prev_text)

            metadata = {
                "index": segment_index,
                "segment_dict": segment_dict,
                "cache_path": segment_cached_file_path,
                "output_path": current_segment_output_path,
                "chosen_text": best_text,
                "estimated_ratio": best_ratio,
                "tts_system": tts_system,
                "segment_data_args": tts_segment_data_args,
                "selected_track_type": best_track_type,
                "previous_context": previous_texts,
            }

            with metadata_lock:
                segments_metadata.append(metadata)

            text = active_segment["translation"]
            text_snippet = f"{text[:20]}...{text[-20:]}" if len(text) > 80 else text
            seg_info = "segment" if total_segments == 1 else f"segment {segment_index+1}/{total_segments}"
            logger.info(f"Processing {seg_info} (Speaker: {speaker}, TTS: {tts_system}): \"{text_snippet}\"")

            tts_lock = tts_locks.get(tts_system)
            if tts_lock is None:
                tts_lock = threading.Lock()
                tts_locks[tts_system] = tts_lock
                setattr(tts_instance, "_synthesis_lock", tts_lock)

            try:
                with tts_lock:
                    tts_instance.synthesize(
                        segments_data=[final_segment_data],
                        language=target_language,
                        previous_context=previous_texts
                    )
            except Exception as synth_exc:
                logger.error(f"Failed to synthesize segment {segment_index+1} ({tts_system}): {synth_exc}")
                segment_dict['synthesized_speech_len'] = 0
                segment_dict['synthesized_speech_file'] = None
                AudioSegment.silent(duration=0).export(current_segment_output_path, format="wav")
                increment_progress(segment_index, speaker, "synthesis failed", best_text)
                return metadata

            if not os.path.exists(current_segment_output_path):
                logger.warning(f"Warning: No audio file created for segment {segment_index+1} ({tts_system})")
                segment_dict['synthesized_speech_len'] = 0
                segment_dict['synthesized_speech_file'] = None
                AudioSegment.silent(duration=0).export(current_segment_output_path, format="wav")
                increment_progress(segment_index, speaker, "missing output", best_text)
                return metadata

            try:
                audio_info = AudioSegment.from_file(current_segment_output_path)
            except Exception as audio_exc:
                logger.error(f"Failed to load synthesized audio for segment {segment_index+1}: {audio_exc}")
                segment_dict['synthesized_speech_len'] = 0
                segment_dict['synthesized_speech_file'] = None
                AudioSegment.silent(duration=0).export(current_segment_output_path, format="wav")
                increment_progress(segment_index, speaker, "audio load failed", best_text)
                return metadata

            segment_dict['synthesized_speech_len'] = len(audio_info) / 1000.0
            segment_dict['synthesized_speech_file'] = current_segment_output_path
            segment_dict['chosen_text'] = best_text
            segment_dict['selected_track_type'] = best_track_type

            if self.cache_manager.use_cache and len(audio_info) > 0:
                try:
                    shutil.copy(current_segment_output_path, segment_cached_file_path)
                except Exception as cache_exc:
                    logger.error(f"Error caching segment {segment_index+1}: {cache_exc}")

            original_dur = segment_dict["end"] - segment_dict["start"]
            actual_dur = segment_dict['synthesized_speech_len']
            ratio = original_dur / actual_dur if actual_dur > 0 else 1.0
            deviation = abs(original_dur - actual_dur) / original_dur if original_dur > 0 else 0.0

            # Track estimation accuracy
            with metadata_lock:
                estimation_stats["total_segments"] += 1

            if not (COMFORT_MIN_ADJUSTMENT_RATIO <= ratio <= COMFORT_MAX_ADJUSTMENT_RATIO):
                # Estimation was inaccurate - outside comfort zone
                with metadata_lock:
                    estimation_stats["inaccurate_estimations"] += 1

                logger.info(
                    f"[MISS] Segment duration estimation MISS - "
                    f"Ratio={ratio:.2f} (expected {COMFORT_MIN_ADJUSTMENT_RATIO:.2f}-{COMFORT_MAX_ADJUSTMENT_RATIO:.2f}), "
                    f"Deviation={deviation:.1%}, Original={original_dur:.2f}s, Actual={actual_dur:.2f}s. "
                    f"Resynthesizing..."
                )
                with tts_lock:
                    self._resynthesize_segment(
                        metadata,
                        tts_instance,
                        COMFORT_MIN_ADJUSTMENT_RATIO,
                        COMFORT_MAX_ADJUSTMENT_RATIO,
                        current_ratio=ratio,
                        segments=segments,
                    )
            else:
                # Estimation was accurate - within comfort zone
                with metadata_lock:
                    estimation_stats["accurate_estimations"] += 1

                logger.debug(
                    f"[HIT] Segment estimation HIT - "
                    f"Ratio={ratio:.2f}, Deviation={deviation:.1%}, Original={original_dur:.2f}s, Actual={actual_dur:.2f}s"
                )

            increment_progress(segment_index, speaker, "synthesized", best_text)
            return metadata

        def process_speaker_segments(speaker_id: str) -> None:
            indices = speaker_to_indices.get(speaker_id, [])
            for segment_index in indices:
                try:
                    handle_segment(segment_index)
                except Exception as segment_exc:
                    logger.error(
                        f"Unexpected error while processing segment {segment_index+1} "
                        f"for speaker '{speaker_id}': {segment_exc}"
                    )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_speaker_segments, speaker_id): speaker_id
                for speaker_id in speaker_to_indices.keys()
            }

            for future in as_completed(futures):
                speaker_id = futures[future]
                try:
                    future.result()
                except Exception as speaker_exc:
                    logger.error(f"Synthesis thread for speaker '{speaker_id}' failed: {speaker_exc}")

        # Ensure metadata is ordered for downstream reporting
        segments_metadata.sort(key=lambda item: item["index"])
        
        # Normalize all segments to average volume with fading
        logger.info("Normalizing segment volumes and applying fades...")
        segment_files = [
            seg.get('synthesized_speech_file') for seg in segments 
            if seg.get('synthesized_speech_file') and os.path.exists(seg.get('synthesized_speech_file'))
        ]
        if segment_files:
            self.audio_processor.normalize_segments_to_average(segment_files)
        else:
            logger.warning("No valid segment files found for normalization")
        
        # Adjust timing and combine audio segments
        combined_audio, real_segment_positions = self._adjust_and_combine_audio_grouped(
            segments, 
            progress_callback=grouping_progress_callback
        )
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

        # Log estimation accuracy statistics
        logger.info("=" * 70)
        logger.info("SEGMENT LENGTH ESTIMATION ACCURACY REPORT")
        logger.info("=" * 70)

        total_estimated = estimation_stats["total_segments"]
        accurate = estimation_stats["accurate_estimations"]
        inaccurate = estimation_stats["inaccurate_estimations"]

        if total_estimated > 0:
            accuracy_rate = (accurate / total_estimated) * 100
            miss_rate = (inaccurate / total_estimated) * 100

            logger.info(f"Total segments synthesized: {total_estimated}")
            logger.info(f"[HIT] Accurate estimations (within comfort zone): {accurate} ({accuracy_rate:.1f}%)")
            logger.info(f"[MISS] Inaccurate estimations (required resynthesis): {inaccurate} ({miss_rate:.1f}%)")
            logger.info(f"")
            logger.info(f"Overall Accuracy Rate: {accuracy_rate:.1f}%")
            logger.info(f"Overall Miss Rate: {miss_rate:.1f}%")
            logger.info(f"Comfort zone range: {COMFORT_MIN_ADJUSTMENT_RATIO:.2f} - {COMFORT_MAX_ADJUSTMENT_RATIO:.2f}")
        else:
            logger.info("No segments were synthesized")

        logger.info("=" * 70)

        # Log completion summary
        logger.info(f"Speech synthesis completed! Processed {total_segments} segments successfully.")

        # Save updated segments with chosen_text and selected_track_type fields
        segments_to_save = []
        for seg in segments:
            seg_copy = seg.copy()
            seg_copy["force_resynthesize"] = seg_copy.get("force_resynthesize", False)
            # Ensure chosen_text and selected_track_type are preserved
            if "chosen_text" not in seg_copy:
                seg_copy["chosen_text"] = ""
            if "selected_track_type" not in seg_copy:
                seg_copy["selected_track_type"] = ""
            seg_copy.pop("words", None)
            segments_to_save.append(seg_copy)
        
        metadata = {
            "source_language": self.config.get('source_language'),
            "target_language": self.config.get('target_language'),
            "tts_system": self.config.get('tts_system')
        }
        updated_path = self.cache_manager.save_segments_json(
            segments_step_name, cache_key, segments_to_save, metadata, "_editable"
        )
        logger.info(f"Updated segments file with synthesis results: {updated_path}")

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
        segments: Optional[List[Dict]] = None,
    ) -> None:
        """Attempt to resynthesize a segment using alternative translations,
        focusing on minimizing deviation from the target ratio range.

        Args:
            metadata: Metadata dictionary for the segment.
            tts_instance: The TTS instance used for synthesis.
            min_ratio: Minimum acceptable ratio original/actual.
            max_ratio: Maximum acceptable ratio original/actual.
            current_ratio: Current ratio to help prioritize alternatives.
            segments: Optional list of all segments for context extraction.
        """

        from src.tts.models import TTSSegmentData

        segment_dict = metadata["segment_dict"]
        original_duration = segment_dict["end"] - segment_dict["start"]
        output_path = metadata["output_path"]
        base_args = metadata["segment_data_args"]

        def get_variant_text(key: str) -> str:
            enriched_key = f"emotion_enriched_{key}"
            return segment_dict.get(enriched_key) or segment_dict.get(key) or ""
        
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

        # Preference for shorter audio when pause removal/speedup is enabled
        prefer_shorter = self.config.get('pause_removal', 'disabled') != 'disabled'
        def deviation_key(dev: float) -> tuple:
            # Primary: minimal absolute deviation; Secondary: prefer negative when enabled
            return (abs(dev), 0 if (prefer_shorter and dev < 0) else 1)

        # Keep track of already tried texts to avoid duplicate synthesis
        tried_texts = {metadata["chosen_text"]}

        # Try all alternatives and find the one with minimum deviation from target range
        for key in candidate_keys:
            alt_text = get_variant_text(key)
            # Skip empty text or already used text
            if not alt_text or not alt_text.strip() or alt_text in tried_texts:
                continue
            
            # Add this text to tried set
            tried_texts.add(alt_text)

            # Create temporary output path for this alternative with unique identifier
            unique_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            temp_output_path = f"{output_path}.temp_{key}_{unique_id}"
            new_segment_data = TTSSegmentData(**{**base_args, "text": alt_text, "output_path": temp_output_path})

            try:
                tts_instance.synthesize(
                    segments_data=[new_segment_data],
                    language=self.config.get('target_language'),
                    previous_context=metadata.get('previous_context', [])
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

                    logger.info(f"Triggering LLM text adjustment for segment {metadata['index']+1}: "
                               f"deviation={abs(current_deviation):.2%} (threshold={LLM_DEVIATION_THRESHOLD:.2%}), "
                               f"duration_factor={duration_factor:.2f}, "
                               f"original_duration={original_duration:.2f}s, actual_duration={actual_duration:.2f}s")
                    logger.debug(f"Baseline text for adjustment: '{baseline_text[:80]}{'...' if len(baseline_text) > 80 else ''}'")

                    # Ask LLM to adjust text length
                    adjusted_text = self.translator.adjust_segment_text_length(
                        original_text=baseline_text,
                        source_language=self.config.get('source_language'),
                        target_language=self.config.get('target_language'),
                        desired_ratio=duration_factor,
                        target_char_count=int(len(baseline_text) * duration_factor),
                        context_info=None,
                        max_attempts=2,
                        tts_system=metadata.get("tts_system"),
                        segments=segments,
                        current_segment_index=metadata["index"],
                    )

                    if adjusted_text and adjusted_text.strip() and adjusted_text.strip() != baseline_text.strip():
                        logger.debug(f"LLM adjustment successful: '{adjusted_text[:80]}{'...' if len(adjusted_text) > 80 else ''}' "
                                   f"(length: {len(baseline_text)} → {len(adjusted_text)} chars)")
                        
                        # Synthesize the adjusted text to evaluate its duration
                        unique_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
                        temp_output_path = f"{output_path}.temp_llm_adjusted_{unique_id}"
                        from src.tts.models import TTSSegmentData
                        new_segment_data = TTSSegmentData(**{**base_args, "text": adjusted_text, "output_path": temp_output_path})

                        try:
                            tts_instance.synthesize(
                                segments_data=[new_segment_data],
                                language=self.config.get('target_language'),
                                previous_context=metadata.get('previous_context', [])
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
                    else:
                        logger.debug(f"LLM adjustment produced no change or empty result, skipping synthesis")
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
                segment_dict["chosen_text"] = segment_dict["translation"]
            else:
                chosen_val = get_variant_text(best_key)
                metadata["chosen_text"] = chosen_val
                segment_dict["chosen_text"] = chosen_val
            metadata["selected_track_type"] = best_key  # Update the selected track type
            segment_dict["selected_track_type"] = best_key

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


    
    def _adjust_and_combine_audio_grouped(
        self, 
        segments: List[Dict],
        progress_callback: callable = None
    ) -> Tuple[AudioSegment, List[Dict]]:
        """
        Adjusts timing and combines audio segments with optimizations for speaker continuity.
        
        This method:
        1. Groups segments by speaker
        2. Optimizes each speaker's segments in continuous groups
        3. Creates separate audio tracks for each speaker
        4. Mixes all speaker tracks together
        
        Args:
            segments: List of transcript segments with translations and speaker info
            progress_callback: Optional callback for progress updates (current, total, message)
            
        Returns:
            Tuple of (Combined AudioSegment with proper timing, List of real segment positions)
        """
        if not segments:
            return AudioSegment.empty(), []
        
        logger.info("Grouping segments by speaker and optimizing timing...")

        # Get parameters from config
        opt_cfg = self.config.get('segments_optimization', {})
        SPLITTING_PAUSE_THRESHOLD_SECONDS = opt_cfg.get('post_translation_merge_gap', 1.5)
        MAX_GROUP_DURATION_SECONDS = opt_cfg.get('max_segment_duration', 60)
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
        
        # First pass: count total groups for progress tracking
        total_groups = 0
        for speaker in sorted(all_speakers):
            speaker_segments = [(idx, segment) for idx, segment in enumerate(segments) if segment["speaker"] == speaker]
            current_group = []
            for i, (orig_idx, segment) in enumerate(speaker_segments):
                start_new_group = False
                if not current_group:
                    start_new_group = True
                elif i > 0:
                    prev_orig_idx, prev_segment = speaker_segments[i-1]
                    pause_duration = segment["start"] - prev_segment["end"]
                    if pause_duration > SPLITTING_PAUSE_THRESHOLD_SECONDS:
                        start_new_group = True
                if current_group and segment["end"] - current_group[0][1]["start"] > MAX_GROUP_DURATION_SECONDS:
                    start_new_group = True
                if start_new_group and current_group:
                    total_groups += 1
                    current_group = []
                current_group.append((orig_idx, segment))
            if current_group:
                total_groups += 1
        
        logger.info(f"Adjusting timing for {total_groups} segment groups...")
        
        # Track progress through groups
        processed_groups = 0
        
        # Process each speaker's segments separately
        for speaker in sorted(all_speakers):
            # Get segments for this speaker with their original indices
            speaker_segments = [(idx, segment) for idx, segment in enumerate(segments) if segment["speaker"] == speaker]
            logger.debug(f"Processing {len(speaker_segments)} segments for speaker {speaker}")
            
            # Group segments by continuous speech
            speaker_groups = []
            current_group = []
            
            for i, (orig_idx, segment) in enumerate(speaker_segments):
                start_new_group = False
                
                if not current_group:
                    start_new_group = True
                elif i > 0:
                    prev_orig_idx, prev_segment = speaker_segments[i-1]
                    pause_duration = segment["start"] - prev_segment["end"]
                    
                    if pause_duration > SPLITTING_PAUSE_THRESHOLD_SECONDS:
                        start_new_group = True
                
                if current_group and segment["end"] - current_group[0][1]["start"] > MAX_GROUP_DURATION_SECONDS:
                    start_new_group = True
                
                if start_new_group and current_group:
                    speaker_groups.append(current_group)
                    current_group = []
                
                current_group.append((orig_idx, segment))
            
            # Add the last group
            if current_group:
                speaker_groups.append(current_group)
            
            logger.debug(f"Divided speaker {speaker} into {len(speaker_groups)} continuous groups")
            
            # Store groups information for debug (extract just segments without indices)
            if self.config.get('debug_info', False):
                speaker_groups_info[speaker] = [[seg for _, seg in group] for group in speaker_groups]
                self.debug_data["speaker_groups"] = speaker_groups_info
            
            # Process each group for this speaker
            for group_idx, group in enumerate(speaker_groups):
                group_start_time_ms = group[0][1]["start"] * 1000
                group_end_time_ms = group[-1][1]["end"] * 1000
                target_duration_ms = int(group_end_time_ms - group_start_time_ms)
                
                # Combine all synthesized audio in this group
                combined_group_audio = AudioSegment.empty()
                group_segment_positions = []  # Track individual segment positions within the group
                
                for i, (orig_idx, segment) in enumerate(group):
                    # Add pause between segments if not the first segment
                    segment_start_in_group_ms = len(combined_group_audio)
                    
                    if i > 0:
                        _, prev_segment = group[i-1]
                        prev_segment_end = prev_segment["end"]
                        current_segment_start = segment["start"]
                        pause_duration_ms = int((current_segment_start - prev_segment_end) * 1000)
                        if pause_duration_ms > 0:
                            combined_group_audio += AudioSegment.silent(duration=pause_duration_ms)
                            segment_start_in_group_ms = len(combined_group_audio)
                    
                    # Load segment audio
                    segment_file = segment.get('synthesized_speech_file') or f"artifacts/audio_chunks/{orig_idx}.wav"
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
                        "original_index": orig_idx
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
                            for orig_idx, segment in group:
                                self.debug_data.setdefault("speed_ratios", {})[orig_idx] = ratio_clamped
                        
                        # Apply high-quality time-stretching
                        # This preserves pitch and timbre much better than simple speed change
                        tempo = 1.0 / ratio_clamped
                        
                        success = self.time_stretcher.stretch(tmp_in, tmp_out, tempo)
                        
                        if success:
                            adjusted_group_audio = AudioSegment.from_file(tmp_out)
                        else:
                            logger.warning(f"Warning: Speed adjustment failed for group {speaker}_{group_idx}")
                    except Exception as exc:
                        logger.warning(f"Warning: Speed adjustment failed for group {speaker}_{group_idx}: {exc}")
                
                # Enforce allowed overflow beyond the group's original timeframe
                overflow_tolerance = float(opt_cfg.get('group_overflow_tolerance', 1.0))
                overflow_tolerance = max(0.0, min(1.0, overflow_tolerance))

                original_group_span_ms = target_duration_ms
                adjusted_len_ms = len(adjusted_group_audio)
                if adjusted_len_ms > original_group_span_ms:
                    overflow_ms = adjusted_len_ms - original_group_span_ms
                    allowed_len_ms = original_group_span_ms + int(overflow_ms * overflow_tolerance)
                    if adjusted_len_ms > allowed_len_ms:
                        adjusted_group_audio = adjusted_group_audio[:allowed_len_ms]
                
                # Calculate real segment positions after speed adjustment
                final_group_duration_ms = len(adjusted_group_audio)
                position_ms = int(group_start_time_ms)
                
                for seg_pos in group_segment_positions:
                    # Apply speed ratio to get actual positions in final audio
                    real_start_ms = position_ms + (seg_pos["start_in_group_ms"] * ratio_clamped)
                    real_end_ms = position_ms + (seg_pos["end_in_group_ms"] * ratio_clamped)
                    # Clamp to the trimmed group end if overflow was restricted
                    group_final_end_ms = position_ms + final_group_duration_ms
                    if real_end_ms > group_final_end_ms:
                        real_end_ms = group_final_end_ms
                    if real_start_ms > group_final_end_ms:
                        real_start_ms = group_final_end_ms
                    
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
                
                # Update progress (report only at 25%, 50%, 75% milestones)
                processed_groups += 1
                if progress_callback and total_groups > 0:
                    progress_pct = (processed_groups * 100) // total_groups
                    prev_pct = ((processed_groups - 1) * 100) // total_groups
                    if progress_pct // 25 > prev_pct // 25 or processed_groups == total_groups:
                        progress_callback(processed_groups, total_groups, "Adjusting segment timing...")
        
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
        # Base filenames for source and target
        source_lang = self.config.get('source_language')
        target_lang = self.config.get('target_language')
        source_name = f"{input_file.stem}_{source_lang}.srt"
        target_name = f"{input_file.stem}_{target_lang}.srt"

        # If names coincide, disambiguate with explicit prefixes
        if source_name == target_name:
            if subtitle_type == "original":
                return f"source_{source_name}"
            else:
                return f"target_{target_name}"

        # Default: use requested language-specific filename
        return f"{input_file.stem}_{language}.srt"

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
