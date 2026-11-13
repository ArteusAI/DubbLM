"""Configuration management for the Smart Dubbing system."""

import os
import sys
import json
import argparse
from typing import Dict, Any
from pathlib import Path
import yaml
from .log_config import get_logger
from src.tts.gemini38_tts_wrapper import (
    DEFAULT_GEMINI38_TTS_MODEL,
    DEFAULT_GEMINI38_TTS_FALLBACK_MODEL,
)

logger = get_logger(__name__)

def _parse_time_to_seconds(time_str: str) -> float:
    """Parse time string (HH:MM:SS, MM:SS, or SS) into seconds."""
    parts = time_str.split(':')
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM:SS, MM:SS, or SS.")


class DubbingConfig:
    """Configuration class for the Smart Dubbing system."""
    
    def __init__(self):
        """Initialize configuration with default values."""
        self.defaults = {
            'whisper_model': 'large-v3',
            'keep_background': False,
            'start_time': None,
            'duration': None,
            'no_cache': False,
            'tts_system': 'gemini38',
            'tts_transport': 'openrouter',
            'tts_model': DEFAULT_GEMINI38_TTS_MODEL,
            'tts_fallback_model': DEFAULT_GEMINI38_TTS_FALLBACK_MODEL,
            'transcription_system': 'openrouter',
            'openrouter_stt_model': 'microsoft/mai-transcribe-2',
            'openrouter_stt_fallback_models': [
                'elevenlabs/scribe-v2',
                'deepgram/nova-3',
            ],
            'openrouter_stt_chunk_seconds': 600,
            'openrouter_stt_overlap_seconds': 15,
            # Gemini-assisted speaker labeling: keep AssemblyAI word timings but
            # let Gemini (which hears the audio) re-label who speaks. Helps on
            # rapid/overlapping trailer dialogue where AssemblyAI diarization
            # collapses different characters onto one speaker.
            'enable_gemini_speaker_labeling': False,
            'gemini_speaker_labeling_model': 'gemini-2.5-pro',
            'gemini_speaker_labeling_max_speakers': 4,
            'translator_type': 'llm',
            'llm_provider': 'openrouter',
            'llm_model_name': '~google/gemini-flash-latest',
            'llm_temperature': 0.5,
            'refinement_llm_provider': 'openrouter',
            'refinement_model_name': 'openai/gpt-6-sol',
            'refinement_temperature': 1.0,
            'refinement_max_tokens': None,
            'refinement_persona': 'normal',
            'enable_llm_editor': False,
            'enable_llm_text_adjustment': True,
            'editor_llm_provider': None,
            'editor_model_name': None,
            'editor_temperature': 1.0,
            'editor_reasoning_effort': 'none',
            'translation_prompt_prefix': None,
            'enable_speaker_gender_inference': True,
            'speaker_metadata': None,
            'voice_name': None,
            'debug_info': False,
            'debug_tts': False,
            'debug_diarize_only': False,
            'save_original_subtitles': False,
            'save_translated_subtitles': False,
            'reference_audio': None,
            'reference_text': None,
            'watermark_path': None,
            'watermark_text': None,
            'glossary': None,
            'voice_prompt': None,
            'voice_auto_selection': True,
            'enable_emotion_analysis': False,
            # Video+audio emotion analysis (Gemini VLM): one request sees the
            # trailer as a single stream (picture + original audio) and tells
            # how every line is performed. The result drives the TTS style.
            'enable_video_emotion_analysis': False,
            'video_emotion_analysis_model': 'gemini-flash-latest',
            'video_emotion_analysis_fps': 0.5,
            'video_emotion_analysis_direction_max_chars': 90,
            'video_emotion_analysis_language': 'en',
            # Voice advice (timbre per speaker) from the same VLM request.
            'enable_voice_advisor': False,
            'voice_advice_max_candidates': 8,
            'speaker_voice_profiles': {},
            'enable_laughter_detection': False,
            # Laughter handling mode: 'dub' (voice laughter via TTS tags) or
            # 'preserve' (keep the original laughter audio, rearrange the
            # translated speech so nothing overlaps it).
            'laughter_mode': 'dub',
            'laughter_preserve_pad_seconds': 0.15,
            'laughter_preserve_merge_gap': 0.25,
            'laughter_preserve_min_duration': 0.25,
            'laughter_preserve_boundary_snap_seconds': 1.5,
            'laughter_preserve_target_dbfs': -20.0,
            # Volume-envelope preserve mode: keep translation segments in
            # place, raise the original/background layer during laughter.
            'laughter_volume': 0.70,
            'laughter_volume_ramp_seconds': 0.25,
            # Non-verbal vocal event detection (AST AudioSet) for trailers and
            # drama: screams, sobbing, gasps, whispers, ...
            'enable_vocal_event_detection': False,
            'vocal_event_kinds': None,
            'vocal_event_threshold': 0.35,
            'vocal_event_min_seconds': 0.3,
            'vocal_event_merge_gap_seconds': 0.5,
            'vocal_event_max_per_segment': 3,
            # How non-laugh events are handled: 'dub' (TTS tags) or 'preserve'
            # (keep the original audio and split speech around it). Laughter is
            # controlled separately by laughter_mode.
            'vocal_events_mode': 'preserve',
            'vocal_event_window_seconds': 10.0,
            'vocal_event_hop_seconds': 1.0,
            'vocal_event_detection_mode': 'window',
            # Loudness based detection of shouted lines / screams. The AST
            # classifier stays silent on shouted dialogue, so the loudness
            # envelope of the vocals stem is used as a second signal.
            'enable_vocal_event_loudness_detection': False,
            'vocal_event_loudness_margin_db': 3.0,
            'vocal_event_loudness_floor_db': -25.0,
            'vocal_event_loudness_min_seconds': 0.1,
            'vocal_event_loudness_merge_gap_seconds': 0.3,
            'vocal_event_loudness_kind': 'scream',
            'vocal_event_probability_mode': 'softmax',
            'vocal_event_model_cache_dir': None,
            # Verify synthesized scream/sob/gasp tags against the audio with
            # AST. None = follow enable_vocal_event_detection.
            'enable_vocal_event_verification': None,
            'include_original_audio': False,
            'output': None, 
            'keep_original_audio_ranges': None,
            'tts_system_mapping': None,
            'tts_prompt_prefix': None,
            'blocked_voices': None,
            'pause_removal': 'disabled',  # 'cut' or 'disabled'
            'use_two_pass_encoding': True,
            'keyframe_buffer': 0.2,
            'dubbed_volume': 1.0,
            'background_volume': 0.562341,
            # "Background voice": the original voice mixed under the dubbed
            # track, quieter than the dub (0 = disabled).
            'background_voice_volume': 0.0,
            'normalize_audio': False,
            # Run ASR + diarization on the isolated vocals stem (UVR) instead of
            # the full mix. Much better speaker attribution on music-heavy
            # material (trailers). None = auto: follow keep_background, since the
            # separation is already paid for then and reused for the background.
            'transcribe_on_vocals': None,
            'enable_emotion_enrichment': False,
            # When true, the refinement LLM owns inline delivery tags and the
            # runtime Gemini 3.8 tag enricher is skipped (avoids double tags).
            'refiner_emotion_tags': False,
            # Request short variants even in video/audio_and_video stretch mode
            # so punchy-but-longer creative lines can avoid video slowdown.
            'creative_recovery_variants': False,
            'emotion_enrichment_model': '~google/gemini-flash-latest',
            'emotion_enrichment_temperature': 0.7,
            'enable_content_validation': True,
            'enable_context_style': True,
            'context_style_max_chars': 140,
            'content_validator_provider': 'whisper',
            'content_validator_whisper_model': 'base',
            'content_validator_whisper_compute_type': 'int8',
            'content_validator_whisper_cpu_threads': 2,
            'content_validator_speech_model': 'nano',
            'enable_content_repetition_check': True,
            'content_repetition_min_words': 4,
            'content_repetition_extra_allowance': 0,
            # Parallel ffmpeg workers for per-segment video speed adjustments.
            # None = auto: min(8, max(2, cpu_count//4)). x264 is cache/RAM-bandwidth
            # bound so oversubscribing CPU cores thrashes throughput and exhausts RAM.
            'video_segment_workers': None,
            'max_workers': 4,
            'estimate_cost': False,
            'speakers_expected': None,
            'exit_before_synthesis': False,
            'clone_voice': None,
            'voice_names': None,
            # Stateless voice cloning for models that support reference audio
            # (currently Fish Audio Drama 3 through OpenRouter). The API
            # resolves the per-model default before building this config.
            'enable_voice_cloning': False,
            # Segment stretch mode: audio | audio_and_video | video
            'segment_stretch': 'audio_and_video',
            # Segment optimization settings (grouped)
            'segments_optimization': {
                # Merging after diarization (tight gap for better transcription/translation)
                'post_diarization_merge_gap': 0.3,
                # Merging after translation (relaxed gap for TTS optimization)
                'post_translation_merge_gap': 1.5,
                # Segment length constraints
                'max_segment_before_translate_chars': 420,
                'max_segment_duration': 60,
                'min_segment_duration': 0.5,
                'max_segment_to_synth_tokens': 1024,
                # Repair diarization over-fragmentation before TTS voice assignment.
                # auto: repair only when speakerCount is set and diarization
                # produced more labels than expected. true/false forces behavior.
                'repair_speaker_fragmentation': 'auto',
                'speaker_fragment_expected_speaker_margin': 2,
                'speaker_fragment_max_duration': 3.0,
                'speaker_fragment_max_words': 10,
                'speaker_fragment_max_gap': 0.75,
                'speaker_fragment_repair_passes': 3,
                # Merge adjacent same-speaker segments to reduce TTS calls. Keep
                # this off to synthesize strictly one recognized turn per segment
                # (safer when diarization is noisy).
                'merge_same_speaker_segments': True,
                # Video pause processing
                'min_pause_duration': 3,
                'preserve_pause_duration': 1.5,
                # Video quality settings
                'video_minterpolate_threshold': 0.75,  # Speed threshold for smooth slowdown (< threshold uses minterpolate)
                # Audio comfort zone for segment_stretch
                'comfort_min_adjustment_ratio': 0.85,  # Audio slowdown comfort limit
                'comfort_max_adjustment_ratio': 1.15,  # Audio speedup comfort limit
                # TTS grouping
                'group_overflow_tolerance': 1.0,
            }
        }
        
        # Required parameters that must come from CLI
        self.required_params = ['input', 'source_language', 'target_language']
        
        # Configuration data
        self.config = self.defaults.copy()
    
    def load_from_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as config_file:
                yaml_config = yaml.safe_load(config_file)
                if yaml_config:
                    # Remove input from YAML config if present - it must come from CLI
                    if 'input' in yaml_config:
                        logger.warning("Warning: 'input' parameter found in YAML config will be ignored. Input must be provided via CLI argument --input")
                        del yaml_config['input']
                    self.config.update(yaml_config)
            logger.info(f"Loaded configuration from {config_path}")
        elif config_path:
            logger.warning(f"Config file {config_path} not found, using defaults and CLI arguments")
    
    def load_from_cli(self, args: argparse.Namespace) -> None:
        """Load configuration from CLI arguments."""
        # Override config with CLI arguments that are not None and actually present
        arg_dict = vars(args)
        for key, value in arg_dict.items():
            if key != 'config' and hasattr(args, key) and value is not None:
                self.config[key] = value
    
    def validate(self) -> None:
        """Validate required configuration parameters."""
        # Check that input is provided via CLI
        if not self.config.get('input'):
            logger.error("Error: Input video file must be specified via --input argument")
            sys.exit(1)
        
        # Check other required parameters  
        for param in ['source_language', 'target_language']:
            if not self.config.get(param):
                logger.error(f"Error: {param.replace('_', ' ').title()} not specified")
                logger.error(f"Please provide it in the config file or with --{param}")
                sys.exit(1)
        
        # Check if video file exists
        input_file = self.config['input']
        if not os.path.exists(input_file):
            logger.error(f"Error: Video file '{input_file}' not found")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Absolute path would be: {os.path.abspath(input_file)}")
            # Try to suggest similar files
            input_dir = os.path.dirname(input_file) or '.'
            if os.path.exists(input_dir):
                similar_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
                if similar_files:
                    logger.error(f"Video files found in directory '{input_dir}':")
                    for f in similar_files[:5]:  # Show first 5 files
                        logger.error(f"  - {f}")
            sys.exit(1)
        
        # Generate output filename if not provided
        if not self.config.get('output'):
            input_path = Path(self.config['input'])
            target_lang = self.config['target_language']
            output_filename = f"{input_path.stem}_{target_lang}{input_path.suffix}"
            self.config['output'] = output_filename  # Save in current working directory
            logger.info(f"Auto-generated output filename: {self.config['output']}")
    
    def process_special_parameters(self) -> None:
        """Process special parameters that need parsing."""
        # Process voice_name parameter
        voice_name = self.config['voice_name']
        if isinstance(voice_name, str) and ',' in voice_name and ':' in voice_name:
            # Parse as a mapping of speakers to voices
            voice_mapping = {pair.split(':')[0]: pair.split(':')[1] for pair in voice_name.split(',') if ':' in pair}
            self.config['voice_name'] = voice_mapping
            logger.info(f"Using multiple voices: {voice_mapping}")
        elif isinstance(voice_name, dict):
            logger.info(f"Using multiple voices from config: {voice_name}")
        elif voice_name:
            # Single voice for all speakers
            logger.info(f"Using single voice: {voice_name}")
        
        # Process keep_original_audio_ranges
        if self.config.get('keep_original_audio_ranges'):
            parsed_ranges = []
            for range_str in self.config['keep_original_audio_ranges']:
                try:
                    start_str, end_str = range_str.split('-')
                    start_seconds = _parse_time_to_seconds(start_str.strip())
                    end_seconds = _parse_time_to_seconds(end_str.strip())
                    if start_seconds >= end_seconds:
                        logger.warning(f"Warning: Invalid range {range_str} in keep_original_audio_ranges (start >= end). Skipping.")
                        continue
                    parsed_ranges.append((start_seconds, end_seconds))
                except ValueError as e:
                    logger.warning(f"Warning: Could not parse range '{range_str}' in keep_original_audio_ranges: {e}. Skipping.")
            
            self.config['keep_original_audio_ranges'] = parsed_ranges if parsed_ranges else None
            if parsed_ranges:
                logger.info(f"Parsed {len(parsed_ranges)} keep-original-audio ranges: {parsed_ranges}")
        
        # Process tts_system_mapping parameter
        tts_system_mapping = self.config.get('tts_system_mapping')
        if isinstance(tts_system_mapping, str):
            # Parse JSON string from command line
            try:
                tts_system_mapping = json.loads(tts_system_mapping)
                logger.info(f"Parsed TTS system mapping from JSON: {tts_system_mapping}")
                self.config['tts_system_mapping'] = tts_system_mapping
            except json.JSONDecodeError as e:
                logger.warning(f"Warning: Could not parse tts_system_mapping JSON '{tts_system_mapping}': {e}. Ignoring.")
                self.config['tts_system_mapping'] = None
        elif isinstance(tts_system_mapping, dict):
            logger.info(f"Using TTS system mapping from config: {tts_system_mapping}")
        else:
            self.config['tts_system_mapping'] = None

        # Clamp group_overflow_tolerance to [0.0, 1.0]
        tol = self.config.get('group_overflow_tolerance')
        try:
            tol_f = float(1.0 if tol is None else tol)
            if tol_f < 0.0 or tol_f > 1.0:
                logger.warning("Warning: group_overflow_tolerance must be between 0 and 1. Clamping to valid range.")
            tol_f = max(0.0, min(1.0, tol_f))
            self.config['group_overflow_tolerance'] = tol_f
        except Exception:
            logger.warning("Warning: Invalid group_overflow_tolerance value. Falling back to 1.0.")
            self.config['group_overflow_tolerance'] = 1.0
        
        # Validate speakers_expected parameter
        speakers_expected = self.config.get('speakers_expected')
        if speakers_expected is not None:
            try:
                speakers_expected = int(speakers_expected)
                if speakers_expected < 1:
                    logger.warning(f"Warning: speakers_expected must be >= 1. Got {speakers_expected}. Ignoring.")
                    self.config['speakers_expected'] = None
                else:
                    self.config['speakers_expected'] = speakers_expected
                    logger.info(f"Expected number of speakers for diarization: {speakers_expected}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Warning: Invalid speakers_expected value '{speakers_expected}': {e}. Ignoring.")
                self.config['speakers_expected'] = None

        # Normalize laughter_mode
        laughter_mode = str(self.config.get('laughter_mode') or 'dub').strip().lower()
        if laughter_mode not in ('dub', 'preserve'):
            logger.warning(f"Warning: Invalid laughter_mode '{laughter_mode}'. Falling back to 'dub'.")
            laughter_mode = 'dub'
        self.config['laughter_mode'] = laughter_mode

        # Normalize vocal_events_mode
        vocal_events_mode = str(self.config.get('vocal_events_mode') or 'preserve').strip().lower()
        if vocal_events_mode not in ('dub', 'preserve'):
            logger.warning(
                f"Warning: Invalid vocal_events_mode '{vocal_events_mode}'. Falling back to 'preserve'."
            )
            vocal_events_mode = 'preserve'
        self.config['vocal_events_mode'] = vocal_events_mode

        # Normalize voice cloning (YAML may supply a string).
        clone_enabled = self.config.get('enable_voice_cloning')
        if isinstance(clone_enabled, str):
            self.config['enable_voice_cloning'] = clone_enabled.strip().lower() in (
                '1', 'true', 'yes', 'on', 'enabled'
            )
        elif clone_enabled is not None:
            self.config['enable_voice_cloning'] = bool(clone_enabled)

        # Normalize transcribe_on_vocals (tri-state: None = auto).
        transcribe_on_vocals = self.config.get('transcribe_on_vocals')
        if transcribe_on_vocals is None:
            self.config['transcribe_on_vocals'] = None
        elif isinstance(transcribe_on_vocals, str):
            normalized = transcribe_on_vocals.strip().lower()
            if normalized in ('', 'auto', 'none', 'default'):
                self.config['transcribe_on_vocals'] = None
            else:
                self.config['transcribe_on_vocals'] = normalized in ('true', '1', 'yes', 'on', 'enabled')
        else:
            self.config['transcribe_on_vocals'] = bool(transcribe_on_vocals)

    def transcribe_on_vocals_enabled(self) -> bool:
        """Resolve whether ASR/diarization should run on the isolated vocals stem.

        ``transcribe_on_vocals`` is a tri-state: True/False force the behavior,
        while None follows ``keep_background`` because the UVR separation is
        already performed (and cached) for the background track in that case.
        """
        explicit = self.config.get('transcribe_on_vocals')
        if explicit is not None:
            return bool(explicit)
        return bool(self.config.get('keep_background'))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all CLI options."""
        parser = argparse.ArgumentParser(description='Smart Video Dubbing Tool')
        
        parser.add_argument('--config', type=str, help='Path to YAML configuration file', default='dubbing_config.yml')
        parser.add_argument('--input', type=str, required=True, help='Path to the video file (required)')
        parser.add_argument('--source_language', type=str, help='Video source language')
        parser.add_argument('--target_language', type=str, help='Video target language')
        parser.add_argument('--whisper_model', type=str, help='Whisper model size for transcription')
        parser.add_argument('--keep_background', action='store_true', default=argparse.SUPPRESS, help='Keep the background audio in the output')
        parser.add_argument('--start_time', type=float, help='Start time in seconds to begin processing')
        parser.add_argument('--duration', type=float, help='Duration in seconds to process')
        parser.add_argument('--no_cache', action='store_true', default=argparse.SUPPRESS, help='Disable caching of pipeline steps')
        parser.add_argument('--tts_system', type=str, choices=['coqui', 'openai', 'f5_tts', 'gemini', 'gemini38', 'minimax', 'openrouter', 'f5'], help="Text-to-speech system to use ('gemini' is deprecated, use 'gemini38')")
        parser.add_argument('--tts_model', type=str, help='Model name for the selected TTS provider')
        parser.add_argument('--tts_fallback_model', type=str, help='Fallback model name for the TTS provider (used by Gemini)')
        parser.add_argument('--tts_transport', type=str, choices=['openrouter', 'google'], help='Transport for Gemini 3.8 TTS')
        parser.add_argument('--transcription_system', type=str, choices=['openai', 'whisperx', 'assemblyai', 'openrouter'], help='Transcription system to use')
        parser.add_argument('--openrouter_stt_model', type=str, help='Speech-to-text model served through OpenRouter')
        parser.add_argument('--openrouter_stt_chunk_seconds', type=float, help='Audio chunk length for OpenRouter STT')
        parser.add_argument('--openrouter_stt_overlap_seconds', type=float, help='Overlap between OpenRouter STT chunks')
        parser.add_argument('--transcribe_on_vocals', type=lambda x: (str(x).lower() == 'true'), help='Run ASR/diarization on the isolated vocals stem (True/False; default: follow keep_background)')
        parser.add_argument('--enable_gemini_speaker_labeling', type=lambda x: (str(x).lower() == 'true'), help='Let Gemini re-label speakers on the ASR transcript while keeping ASR word timings (True/False)')
        parser.add_argument('--gemini_speaker_labeling_model', type=str, help='Gemini model used for speaker labeling (e.g. gemini-2.5-pro)')
        parser.add_argument('--gemini_speaker_labeling_max_speakers', type=int, help='Maximum number of distinct voices for Gemini speaker labeling')
        parser.add_argument('--translator_type', type=str, choices=['llm'], help='Translator type to use')
        parser.add_argument('--llm_provider', type=str, choices=['gemini', 'openrouter'], help='LLM provider to use')
        parser.add_argument('--llm_model_name', type=str, help='Model name for the LLM')
        parser.add_argument('--llm_temperature', type=float, help='Temperature for LLM generation')
        parser.add_argument('--refinement_llm_provider', type=str, choices=['gemini', 'openrouter'], help='LLM provider to use for refinement')
        parser.add_argument('--refinement_model_name', type=str, help='Model name for refinement')
        parser.add_argument('--refinement_temperature', type=float, help='Temperature for refinement')
        parser.add_argument('--refinement_max_tokens', type=int, help='Maximum tokens for OpenRouter refinement')
        parser.add_argument('--refinement_persona', type=str, choices=['none', 'normal', 'casual_manager', 'child', 'housewife'], help='Persona for refinement prompt')
        parser.add_argument('--enable_llm_editor', type=lambda x: (str(x).lower() == 'true'), help='Enable an additional LLM editor pass after refinement (True/False)')
        parser.add_argument('--enable_llm_text_adjustment', type=lambda x: (str(x).lower() == 'true'), help='Enable LLM text rewrites during TTS timing/recovery (True/False)')
        parser.add_argument('--editor_llm_provider', type=str, choices=['gemini', 'openrouter'], help='LLM provider to use for the optional editor pass')
        parser.add_argument('--editor_model_name', type=str, help='Model name for the optional editor pass')
        parser.add_argument('--editor_temperature', type=float, help='Temperature for the optional editor pass')
        parser.add_argument('--editor_reasoning_effort', type=str, choices=['minimal', 'low', 'medium', 'high', 'xhigh', 'none'], help='Reasoning effort for the optional editor pass')
        parser.add_argument('--translation_prompt_prefix', type=str, help='Additional context to prepend to LLM translation prompts')
        parser.add_argument('--voice_name', type=str, help='Voice to use for TTS')
        parser.add_argument('--debug_info', action='store_true', default=argparse.SUPPRESS, help='Generate a debug video with speaker labels')
        parser.add_argument('--debug_tts', action='store_true', default=argparse.SUPPRESS, help='Enable TTS debugging (e.g., save rejected/silent attempts)')
        parser.add_argument('--save_original_subtitles', action='store_true', default=argparse.SUPPRESS, help='Save original language subtitles')
        parser.add_argument('--save_translated_subtitles', action='store_true', default=argparse.SUPPRESS, help='Save translated language subtitles')
        parser.add_argument('--reference_audio', type=str, help='Path to a reference audio file for f5_tts system')
        parser.add_argument('--reference_text', type=str, help='Text corresponding to the reference audio for f5_tts system')
        parser.add_argument('--watermark_path', type=str, help='Path to the watermark PNG image')
        parser.add_argument('--watermark_text', type=str, help='Text to display under the watermark')
        parser.add_argument('--voice_auto_selection', type=lambda x: (str(x).lower() == 'true'), help='Enable automatic voice selection for TTS (True/False)')
        parser.add_argument('--enable_context_style', type=lambda x: (str(x).lower() == 'true'), help='Pass the previous line as delivery context to Gemini 3.8 TTS (True/False)')
        parser.add_argument('--context_style_max_chars', type=int, help='Max characters of the quoted previous line used as TTS delivery context')
        parser.add_argument('--content_validator_provider', type=str, choices=['whisper', 'assemblyai'], help='ASR provider for TTS content validation')
        parser.add_argument('--content_validator_whisper_model', type=str, help='Local Whisper model for TTS content validation')
        parser.add_argument('--content_validator_whisper_compute_type', type=str, help='faster-whisper compute type for TTS content validation')
        parser.add_argument('--content_validator_whisper_cpu_threads', type=int, help='CPU threads for local Whisper TTS content validation')
        parser.add_argument('--content_validator_speech_model', type=str, help='AssemblyAI speech model for TTS content validation')
        parser.add_argument('--enable_content_repetition_check', type=lambda x: (str(x).lower() == 'true'), help='Fail TTS content validation when the ASR transcript repeats words not present in the expected text (True/False)')
        parser.add_argument('--content_repetition_min_words', type=int, help='N-gram size used for unexpected TTS repetition detection')
        parser.add_argument('--content_repetition_extra_allowance', type=int, help='Allowed extra ASR occurrences of an n-gram before flagging a TTS repetition')
        parser.add_argument('--enable_emotion_analysis', type=lambda x: (str(x).lower() == 'true'), help='Enable emotion analysis for speech synthesis (True/False)')
        parser.add_argument('--enable_video_emotion_analysis', type=lambda x: (str(x).lower() == 'true'), help='Analyze the original performance (video + audio) with a Gemini VLM and use it for the TTS style (True/False)')
        parser.add_argument('--video_emotion_analysis_model', type=str, help='Model used for the video+audio emotion analysis')
        parser.add_argument('--video_emotion_analysis_fps', type=float, help='Frames per second sampled from the video for the emotion analysis')
        parser.add_argument('--enable_voice_advisor', type=lambda x: (str(x).lower() == 'true'), help='Use the voice advice (timbre per speaker) of the delivery analysis for the voice choice and the TTS style (True/False)')
        parser.add_argument('--enable_emotion_enrichment', type=lambda x: (str(x).lower() == 'true'), help='Label per-segment emotions for TTS style guidance (True/False)')
        parser.add_argument('--refiner_emotion_tags', type=lambda x: (str(x).lower() == 'true'), help='Let the refinement LLM own inline delivery tags and skip the runtime emotion enricher (True/False)')
        parser.add_argument('--creative_recovery_variants', type=lambda x: (str(x).lower() == 'true'), help='Request short/long variants in video modes for creative timing recovery (True/False)')
        parser.add_argument('--enable_laughter_detection', type=lambda x: (str(x).lower() == 'true'), help='Detect laughter moments and voice them with TTS laughter tags (True/False)')
        parser.add_argument('--laughter_mode', type=str, choices=['dub', 'preserve'], help="Laughter handling: 'dub' voices laughter via TTS tags, 'preserve' keeps the original laughter audio")
        parser.add_argument('--enable_vocal_event_detection', type=lambda x: (str(x).lower() == 'true'), help='Detect non-verbal vocal events (screams, sobbing, gasps, ...) with the AST AudioSet model (True/False)')
        parser.add_argument('--vocal_events_mode', type=str, choices=['dub', 'preserve'], help="Non-verbal event handling: 'dub' voices them via TTS tags, 'preserve' keeps the original audio")
        parser.add_argument('--vocal_event_kinds', type=str, help='Comma-separated event kinds to detect (laugh, cry, scream, shout, whisper, gasp, pant, sigh, groan, grunt, whimper, breath, snort)')
        parser.add_argument('--vocal_event_threshold', type=float, help='Minimum per-frame AST probability for a vocal event')
        parser.add_argument('--vocal_event_max_per_segment', type=int, help='Maximum number of distinct vocal event tags per segment')
        parser.add_argument('--run_step', type=str, choices=['combine_video'], 
                            help='Run only a specific, advanced pipeline step. This is intended for debugging or resuming a failed run where prior steps have successfully created their expected output files in the default locations. \
                                  Example: --run_step combine_video (Assumes audio/output.wav and potentially audio/background.wav exist from prior steps). \
                                  Note: For most users, running the full pipeline or using --generate_speaker_report is recommended.')
        parser.add_argument('--include_original_audio', action='store_true', default=argparse.SUPPRESS, help='Include the original audio track in the final video')
        parser.add_argument('--output', type=str, help='Path to the output video file (default: input_name + target_language + extension in current directory)')
        parser.add_argument('--generate_speaker_report', action='store_true', default=argparse.SUPPRESS, help='Generate a report of identified speakers and their voice samples, then exit.')
        parser.add_argument('--estimate_cost', action='store_true', default=argparse.SUPPRESS,
                            help='Estimate pipeline cost using configured pricing and exit without running dubbing.')
        parser.add_argument('--tts_system_mapping', type=str, help='JSON string mapping speakers to TTS systems')
        parser.add_argument('--tts_prompt_prefix', type=str, help='Global prompt prefix for TTS generation instructions (mainly for Gemini TTS)')
        parser.add_argument('--pause_removal', type=str, choices=['cut', 'disabled'], 
                            help='Pause removal mode: "cut" (remove pauses), "disabled" (no processing)')
        parser.add_argument('--segment_stretch', type=str, choices=['audio', 'audio_and_video', 'video'],
                            help='Segment stretch mode: "audio" (audio speed only), "audio_and_video" (hybrid: audio within comfort + video), "video" (video speed only, no audio change)')
        parser.add_argument('--keyframe_buffer', default=0.2, type=float, help='Buffer around keyframes to preserve during pause removal (seconds)')
        parser.add_argument('--use_two_pass_encoding', type=lambda x: (str(x).lower() == 'true'), help='Use two-pass encoding for better video quality during re-encoding (True/False)')
        parser.add_argument('--dubbed_volume', type=float, help='Gain multiplier for translated track (e.g., 1.2 for +1.6 dB)')
        parser.add_argument('--normalize_audio', type=lambda x: (str(x).lower() == 'true'), help='Apply final loudness normalization (loudnorm) to the output mix (True/False)')
        parser.add_argument('--background_volume', type=float, help='Gain multiplier for background track when keep_background=true (e.g., 0.56 ≈ -5 dB)')
        parser.add_argument('--speakers_expected', type=int, help='Expected number of speakers for diarization (if not specified, auto-detect)')
        parser.add_argument('--exit_before_synthesis', action='store_true', default=argparse.SUPPRESS, 
                            help='Save translated segments to JSON and exit before speech synthesis. Edit the file, rename it to *_edited.json, and rerun to use edited translations.')
        parser.add_argument('--clone_voice', type=str, default=None,
                            help='Comma-separated list of speakers to clone (e.g., SPEAKER_A,SPEAKER_B). Voices will be cloned after diarization and test samples generated.')
        parser.add_argument('--enable_voice_cloning', type=lambda x: (str(x).lower() == 'true'), default=None,
                            help='Build a clean reference from the source audio for every speaker and use stateless voice cloning (Drama 3 / OpenRouter).')
        parser.add_argument('--voice_names', type=str, default=None,
                            help='Comma-separated list of custom voice names for cloned voices (e.g., alice_voice,bob_voice). Must match the order of --clone_voice. If not provided, auto-generates as cloned_{SPEAKER_ID}.')
        
        return parser


def create_config_from_args(args: argparse.Namespace) -> DubbingConfig:
    """Create and configure DubbingConfig from CLI arguments."""
    config = DubbingConfig()
    
    # Load YAML config first
    config.load_from_yaml(args.config)
    
    # Override with CLI arguments
    config.load_from_cli(args)
    
    # Validate configuration
    config.validate()
    
    # Process special parameters
    config.process_special_parameters()
    
    return config


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    config = DubbingConfig()
    return config._create_parser() 
