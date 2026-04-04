"""Configuration management for the Smart Dubbing system."""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import yaml
from .log_config import get_logger

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
            'tts_system': 'coqui',
            'tts_model': None,
            'tts_fallback_model': None,
            'omnivoice_space_id': 'archivartaunik/OmniVoice',
            'omnivoice_api_name': '/_clone_fn',
            'omnivoice_lang': 'Belarusian',
            'omnivoice_num_steps': 32,
            'omnivoice_guidance_scale': 2.0,
            'omnivoice_denoise': True,
            'omnivoice_speed': 1.0,
            'omnivoice_duration': 3.0,
            'omnivoice_preprocess_prompt': True,
            'omnivoice_postprocess_output': True,
            'transcription_system': 'whisper',
            'translator_type': 'llm',
            'llm_provider': 'gemini',
            'llm_model_name': None,
            'llm_temperature': 0.5,
            'refinement_llm_provider': None,
            'refinement_model_name': None,
            'refinement_temperature': 1.0,
            'refinement_max_tokens': None,
            'refinement_persona': 'normal',
            'translation_prompt_prefix': None,
            'voice_name': None,
            'debug_info': False,
            'debug_tts': False,
            'debug_diarize_only': False,
            'save_original_subtitles': False,
            'save_translated_subtitles': False,
            'reference_audio': None,
            'reference_text': None,
            'reference_audio_mapping': None,
            'reference_text_mapping': None,
            'watermark_path': None,
            'watermark_text': None,
            'glossary': None,
            'voice_prompt': None,
            'voice_auto_selection': True,
            'enable_emotion_analysis': False,
            'include_original_audio': False,
            'output': None, 
            'keep_original_audio_ranges': None,
            'tts_system_mapping': None,
            'tts_prompt_prefix': None,
            'remove_pauses': False,
            'min_pause_duration': 3,
            'use_two_pass_encoding': True,
            'keyframe_buffer': 0.2,
            'dubbed_volume': 1.0,
            'background_volume': 0.562341,
            'group_overflow_tolerance': 1.0,
            'segment_reference_min_duration': 2.0
        }
        
        # Required parameters that must come from CLI
        self.required_params = ['input', 'source_language', 'target_language']
        
        # Configuration data
        self.config = self.defaults.copy()
    
    def load_from_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as config_file:
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

        input_path = Path(input_file)
        project_dir = input_path.parent / input_path.stem
        artifacts_dir = project_dir / "artifacts"
        project_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.config["project_dir"] = str(project_dir)
        self.config["artifacts_dir"] = str(artifacts_dir)
        self.config["audio_artifacts_dir"] = str(artifacts_dir / "audio")
        self.config["speakers_audio_dir"] = str(artifacts_dir / "speakers_audio")
        self.config["audio_chunks_dir"] = str(artifacts_dir / "audio_chunks")
        self.config["su_audio_chunks_dir"] = str(artifacts_dir / "su_audio_chunks")
        self.config["debug_dir"] = str(artifacts_dir / "debug")
        self.config["translation_debug_dir"] = str(artifacts_dir / "debug" / "translation")
        self.config["translation_refinement_debug_dir"] = str(artifacts_dir / "debug" / "translation_refinement")
        self.config["speaker_report_dir"] = str(artifacts_dir / "speaker_report")
        self.config["translated_samples_dir"] = str(artifacts_dir / "translated_samples")
        self.config["transcription_path"] = str(artifacts_dir / "transcription.txt")
        self.config["timecodes_report_path"] = str(artifacts_dir / "timecodes.txt")
        self.config["translated_audio_path"] = str(artifacts_dir / "audio" / "output.wav")
        self.config["background_audio_path"] = str(artifacts_dir / "audio" / "background.wav")
        self.config["debug_video_path"] = str(artifacts_dir / "debug" / "dubbing_debug.mp4")
        self.config["temp_segment_audio_path"] = str(artifacts_dir / "audio" / "temp_segment.wav")
        self.config["temp_final_audio_path"] = str(artifacts_dir / "audio" / "temp_final_for_pause_analysis.wav")
        self.config["temp_video_with_cuts_path"] = str(artifacts_dir / "temp_video_with_cuts.mp4")
        
        # Generate output filename if not provided
        if not self.config.get('output'):
            target_lang = self.config['target_language']
            output_filename = f"{input_path.stem}_{target_lang}{input_path.suffix}"
            self.config['output'] = str(project_dir / output_filename)
            logger.info(f"Auto-generated output filename: {self.config['output']}")
        else:
            output_path = Path(self.config['output'])
            if not output_path.suffix:
                normalized_output_path = output_path.with_suffix(input_path.suffix)
                self.config['output'] = str(normalized_output_path)
                logger.info(
                    "Added missing output extension '%s': %s",
                    input_path.suffix,
                    self.config['output'],
                )
    
    def process_special_parameters(self) -> None:
        """Process special parameters that need parsing."""
        def _parse_mapping_parameter(name: str) -> None:
            value = self.config.get(name)
            if isinstance(value, str):
                try:
                    parsed_value = json.loads(value)
                    if isinstance(parsed_value, dict):
                        self.config[name] = parsed_value
                        logger.info(f"Parsed {name} from JSON: {parsed_value}")
                    else:
                        logger.warning(f"Warning: {name} must decode to a JSON object. Ignoring.")
                        self.config[name] = None
                except json.JSONDecodeError as e:
                    logger.warning(f"Warning: Could not parse {name} JSON '{value}': {e}. Ignoring.")
                    self.config[name] = None
            elif isinstance(value, dict):
                logger.info(f"Using {name} from config: {value}")
            else:
                self.config[name] = None

        duration = self.config.get('duration')
        if duration is not None:
            try:
                duration_value = float(duration)
                if duration_value <= 0:
                    logger.info("Ignoring non-positive duration value; processing until the end of the file.")
                    self.config['duration'] = None
                else:
                    self.config['duration'] = duration_value
            except (TypeError, ValueError):
                logger.warning("Warning: Invalid duration value. Ignoring it.")
                self.config['duration'] = None

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

        _parse_mapping_parameter('reference_audio_mapping')
        _parse_mapping_parameter('reference_text_mapping')

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

        # Validate segment_reference_min_duration
        ref_min_duration = self.config.get('segment_reference_min_duration')
        try:
            ref_min_duration_f = float(0.0 if ref_min_duration is None else ref_min_duration)
            if ref_min_duration_f < 0.0:
                logger.warning("Warning: segment_reference_min_duration cannot be negative. Using 0 seconds.")
                ref_min_duration_f = 0.0
            self.config['segment_reference_min_duration'] = ref_min_duration_f
        except (TypeError, ValueError):
            default_ref_duration = self.defaults['segment_reference_min_duration']
            logger.warning(
                "Warning: Invalid segment_reference_min_duration value. Falling back to default of %.2f seconds.",
                default_ref_duration
            )
            self.config['segment_reference_min_duration'] = default_ref_duration
    
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
        parser.add_argument('--tts_system', type=str, choices=['coqui', 'xtts', 'openai', 'f5_tts', 'gemini', 'bextts', 'omnivoice'], help='Text-to-speech system to use')
        parser.add_argument('--tts_model', type=str, help='Model name for the selected TTS provider')
        parser.add_argument('--tts_fallback_model', type=str, help='Fallback model name for the TTS provider (used by Gemini)')
        parser.add_argument('--omnivoice_space_id', type=str, help='Hugging Face Space ID for OmniVoice')
        parser.add_argument('--omnivoice_api_name', type=str, help='Gradio API endpoint for OmniVoice synthesis')
        parser.add_argument('--omnivoice_lang', type=str, help='Language setting for OmniVoice (default: Belarusian)')
        parser.add_argument('--omnivoice_num_steps', type=int, help='Number of OmniVoice generation steps')
        parser.add_argument('--omnivoice_guidance_scale', type=float, help='Guidance scale for OmniVoice synthesis')
        parser.add_argument('--omnivoice_denoise', type=lambda x: (str(x).lower() == 'true'), help='Enable OmniVoice denoising (True/False)')
        parser.add_argument('--omnivoice_speed', type=float, help='Default speaking speed for OmniVoice synthesis')
        parser.add_argument('--omnivoice_duration', type=float, help='Default OmniVoice duration control value')
        parser.add_argument('--omnivoice_preprocess_prompt', type=lambda x: (str(x).lower() == 'true'), help='Enable OmniVoice prompt preprocessing (True/False)')
        parser.add_argument('--omnivoice_postprocess_output', type=lambda x: (str(x).lower() == 'true'), help='Enable OmniVoice audio postprocessing (True/False)')
        parser.add_argument('--transcription_system', type=str, choices=['whisper', 'openai', 'pyannote_openai', 'whisperx', 'assemblyai'], help='Transcription system to use')
        parser.add_argument('--translator_type', type=str, choices=['llm'], help='Translator type to use')
        parser.add_argument('--llm_provider', type=str, choices=['gemini', 'openrouter'], help='LLM provider to use')
        parser.add_argument('--llm_model_name', type=str, help='Model name for the LLM')
        parser.add_argument('--llm_temperature', type=float, help='Temperature for LLM generation')
        parser.add_argument('--refinement_llm_provider', type=str, choices=['gemini', 'openrouter'], help='LLM provider to use for refinement')
        parser.add_argument('--refinement_model_name', type=str, help='Model name for refinement')
        parser.add_argument('--refinement_temperature', type=float, help='Temperature for refinement')
        parser.add_argument('--refinement_max_tokens', type=int, help='Maximum tokens for OpenRouter refinement')
        parser.add_argument('--refinement_persona', type=str, choices=['normal', 'casual_manager', 'child', 'housewife'], help='Persona for refinement prompt')
        parser.add_argument('--translation_prompt_prefix', type=str, help='Additional context to prepend to LLM translation prompts')
        parser.add_argument('--voice_name', type=str, help='Voice to use for TTS')
        parser.add_argument('--debug_info', action='store_true', default=argparse.SUPPRESS, help='Generate a debug video with speaker labels')
        parser.add_argument('--debug_tts', action='store_true', default=argparse.SUPPRESS, help='Enable TTS debugging (e.g., save rejected/silent attempts)')
        parser.add_argument('--save_original_subtitles', action='store_true', default=argparse.SUPPRESS, help='Save original language subtitles')
        parser.add_argument('--save_translated_subtitles', action='store_true', default=argparse.SUPPRESS, help='Save translated language subtitles')
        parser.add_argument('--reference_audio', type=str, help='Path to a reference audio file for f5_tts system')
        parser.add_argument('--reference_text', type=str, help='Text corresponding to the reference audio for f5_tts system')
        parser.add_argument('--reference_audio_mapping', type=str, help='JSON string mapping speakers to reference audio file paths')
        parser.add_argument('--reference_text_mapping', type=str, help='JSON string mapping speakers to reference transcript text')
        parser.add_argument('--watermark_path', type=str, help='Path to the watermark PNG image')
        parser.add_argument('--watermark_text', type=str, help='Text to display under the watermark')
        parser.add_argument('--voice_auto_selection', type=lambda x: (str(x).lower() == 'true'), help='Enable automatic voice selection for TTS (True/False)')
        parser.add_argument('--enable_emotion_analysis', type=lambda x: (str(x).lower() == 'true'), help='Enable emotion analysis for speech synthesis (True/False)')
        parser.add_argument('--run_step', type=str, choices=['combine_video'], 
                            help='Run only a specific, advanced pipeline step. This is intended for debugging or resuming a failed run where prior steps have successfully created their expected output files in the default locations. \
                                  Example: --run_step combine_video (Assumes audio/output.wav and potentially audio/background.wav exist from prior steps). \
                                  Note: For most users, running the full pipeline or using --generate_speaker_report is recommended.')
        parser.add_argument('--include_original_audio', action='store_true', default=argparse.SUPPRESS, help='Include the original audio track in the final video')
        parser.add_argument('--output', type=str, help='Path to the output video file (default: input_name + target_language + extension in current directory)')
        parser.add_argument('--generate_speaker_report', action='store_true', default=argparse.SUPPRESS, help='Generate a report of identified speakers and their voice samples, then exit.')
        parser.add_argument('--tts_system_mapping', type=str, help='JSON string mapping speakers to TTS systems')
        parser.add_argument('--tts_prompt_prefix', type=str, help='Global prompt prefix for TTS generation instructions (mainly for Gemini TTS)')
        parser.add_argument('--remove_pauses', type=lambda x: (str(x).lower() == 'true'), help='Remove small pauses from video while preserving keyframes (True/False)')
        parser.add_argument('--min_pause_duration', default=3, type=float, help='Minimum pause duration to consider for removal (seconds)')
        parser.add_argument('--keyframe_buffer', default=0.2, type=float, help='Buffer around keyframes to preserve during pause removal (seconds)')
        parser.add_argument('--use_two_pass_encoding', type=lambda x: (str(x).lower() == 'true'), help='Use two-pass encoding for better video quality during re-encoding (True/False)')
        parser.add_argument('--dubbed_volume', type=float, help='Gain multiplier for translated track (e.g., 1.2 for +1.6 dB)')
        parser.add_argument('--background_volume', type=float, help='Gain multiplier for background track when keep_background=true (e.g., 0.56 ≈ -5 dB)')
        parser.add_argument('--group_overflow_tolerance', type=float, help='Allowed overflow beyond group timeframe when combining segments (0..1, default 1.0)')
        parser.add_argument('--segment_reference_min_duration', type=float, help='Minimum segment length in seconds required to export dedicated reference audio clips')

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
