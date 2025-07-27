"""CLI entry point for the refactored Smart Dubbing system."""

# --- Suppress CUDA/GPU library warnings BEFORE any imports ---
import os
# Set environment variables to suppress noisy CUDA/GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
# --- End suppression block ---

import sys
import logging
from dotenv import load_dotenv

from ..core.config import create_argument_parser, create_config_from_args
from ..core.smart_dubbing import SmartDubbing
from ..core.log_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Main function to run the dubbing tool."""
    # Load environment variables
    load_dotenv()
    
    # Create argument parser and parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = create_config_from_args(args)
    
    # Print configuration info
    glossary = config.get('glossary', {})
    if glossary:
        logger.debug(f"Using translation glossary with {len(glossary)} entries")
    
    voice_prompt = config.get('voice_prompt', {})
    if voice_prompt:
        logger.debug(f"Using voice prompts for {len(voice_prompt)} speakers")
        for speaker, prompt in voice_prompt.items():
            p = f"{prompt[:50]}..." if len(prompt) > 50 else prompt
            logger.debug(f"  {speaker}: {p}")
    
    # Initialize and run the dubbing system
    try:
        dubber = SmartDubbing(config)
        
        if config.get('generate_speaker_report'):
            logger.info("Generating speaker report...")
            try:
                report_path, samples_path = dubber.generate_diarization_report()
                logger.info(f"Speaker report generated: {report_path}")
                logger.info(f"Voice samples copied to: {samples_path}")
            except Exception as e:
                logger.error(f"Error generating speaker report: {e}")
                sys.exit(1)
            sys.exit(0)
            
        elif config.get('run_step') == 'combine_video':
            logger.info("Running only the 'combine_audio_with_video' step...")
            
            expected_translated_audio = "artifacts/audio/output.wav"
            expected_background_audio = None
            
            if config.get('keep_background'):
                expected_background_audio = "artifacts/audio/background.wav"
                if not os.path.exists(expected_background_audio):
                    logger.warning(f"Expected background audio {expected_background_audio} not found. Proceeding without it.")
                    expected_background_audio = None

            if not os.path.exists(expected_translated_audio):
                logger.error(f"Error: Expected translated audio {expected_translated_audio} not found.")
                sys.exit(1)
            
            watermark_input_path = config.get('watermark_path')
            if watermark_input_path and not os.path.exists(watermark_input_path):
                logger.warning(f"Watermark image {watermark_input_path} not found. Proceeding without it.")
                watermark_input_path = None
            
            try:
                output_video_path = dubber.video_processor.combine_audio_with_video(
                    video_path=config.get('input'),
                    translated_audio_path=expected_translated_audio,
                    background_audio_path=expected_background_audio,
                    watermark_path=watermark_input_path,
                    watermark_text=config.get('watermark_text'),
                    include_original_audio=config.get('include_original_audio', False),
                    output_file=config.get('output'),
                    start_time=config.get('start_time'),
                    duration=config.get('duration'),
                    keep_original_audio_ranges=config.get('keep_original_audio_ranges'),
                    source_language=config.get('source_language'),
                    target_language=config.get('target_language')
                )
                logger.info(f"Video combination complete. Output saved to: {output_video_path}")
            except Exception as e:
                logger.error(f"Error during 'combine_video' step: {e}", exc_info=True)
                sys.exit(1)
            sys.exit(0)
            
        else:
            # Run the full pipeline
            output_path = dubber.run_pipeline(
                save_original_subtitles=config.get('save_original_subtitles', False),
                save_translated_subtitles=config.get('save_translated_subtitles', False)
            )
            logger.info(f"Video dubbing complete. Output saved to: {output_path}")
            
    except Exception as e:
        logger.error(f"Error in dubbing system: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 