"""CLI entry point for the Smart Dubbing system."""

# --- Suppress CUDA/GPU library warnings BEFORE any imports ---
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async CUDA operations
# --- End suppression block ---

import sys
import glob
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from ..core.config import create_argument_parser, create_config_from_args
from ..core.smart_dubbing import SmartDubbing
from ..core.log_config import setup_logging, get_logger
from ..core.cost_estimator import CostEstimator

# Setup logging
setup_logging()
logger = get_logger(__name__)


def _expand_input_paths(input_arg: Optional[str]) -> List[str]:
    """Expand glob patterns in --input argument."""
    if not input_arg:
        return []

    normalized = os.path.expanduser(input_arg)
    if glob.has_magic(normalized):
        matches = sorted(
            path for path in glob.glob(normalized, recursive=True) if os.path.isfile(path)
        )
        return matches

    return [normalized]


def _merge_costs(
    accumulated: Optional[Dict[str, float]], current: Dict[str, float]
) -> Dict[str, float]:
    """Merge per-file cost estimates into an aggregate dictionary."""
    if accumulated is None:
        return current.copy()

    for key, value in current.items():
        accumulated[key] = accumulated.get(key, 0.0) + float(value or 0.0)
    return accumulated


def _run_cost_estimation(config) -> Tuple[int, Optional[Dict[str, float]]]:
    """Run cost estimation for a single configuration."""
    logger.info(
        "Estimating pipeline cost (no dubbing will be performed) for %s...",
        config.get("input"),
    )
    try:
        estimator = CostEstimator(config)
        costs = estimator.estimate()
        return 0, costs
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to estimate costs: %s", exc, exc_info=True)
        return 1, None


def _run_dubbing(config) -> Tuple[int, Optional[Dict[str, float]]]:
    """Run the dubbing pipeline for a single configuration."""
    try:
        dubber = SmartDubbing(config)

        if config.get("generate_speaker_report"):
            logger.info("Generating speaker report for %s...", config.get("input"))
            try:
                report_path, samples_path = dubber.generate_diarization_report()
                logger.info("Speaker report generated: %s", report_path)
                logger.info("Voice samples copied to: %s", samples_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Error generating speaker report: %s", exc, exc_info=True)
                return 1, None
            return 0, None

        if config.get("run_step") == "combine_video":
            logger.info("Running only the 'combine_audio_with_video' step for %s...", config.get("input"))

            expected_translated_audio = "artifacts/audio/output.wav"
            expected_background_audio = None

            if config.get("keep_background"):
                expected_background_audio = "artifacts/audio/background.wav"
                if not os.path.exists(expected_background_audio):
                    logger.warning(
                        "Expected background audio %s not found. Proceeding without it.",
                        expected_background_audio,
                    )
                    expected_background_audio = None

            if not os.path.exists(expected_translated_audio):
                logger.error(
                    "Error: Expected translated audio %s not found.",
                    expected_translated_audio,
                )
                return 1, None

            watermark_input_path = config.get("watermark_path")
            if watermark_input_path and not os.path.exists(watermark_input_path):
                logger.warning(
                    "Watermark image %s not found. Proceeding without it.",
                    watermark_input_path,
                )
                watermark_input_path = None

            try:
                output_video_path = dubber.video_processor.combine_audio_with_video(
                    video_path=config.get("input"),
                    translated_audio_path=expected_translated_audio,
                    background_audio_path=expected_background_audio,
                    watermark_path=watermark_input_path,
                    watermark_text=config.get("watermark_text"),
                    include_original_audio=config.get("include_original_audio", False),
                    output_file=config.get("output"),
                    start_time=config.get("start_time"),
                    duration=config.get("duration"),
                    keep_original_audio_ranges=config.get("keep_original_audio_ranges"),
                    source_language=config.get("source_language"),
                    target_language=config.get("target_language"),
                    dubbed_volume=config.get("dubbed_volume", 1.0),
                    background_volume=config.get("background_volume", 0.562341),
                    upscale_factor=config.get("upscale_factor", 1.0),
                    upscale_sharpen=config.get("upscale_sharpen", True),
                )
                logger.info(
                    "Video combination complete. Output saved to: %s", output_video_path
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Error during 'combine_video' step: %s", exc, exc_info=True
                )
                return 1, None
            return 0, None

        output_path = dubber.run_pipeline(
            save_original_subtitles=config.get("save_original_subtitles", False),
            save_translated_subtitles=config.get("save_translated_subtitles", False),
        )
        logger.info(
            "Video dubbing complete. Output saved to: %s for input %s",
            output_path,
            config.get("input"),
        )
        return 0, None

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error in dubbing system: %s", exc, exc_info=True)
        return 1, None


def main():
    """Main function to run the dubbing tool or estimate costs."""
    load_dotenv()

    parser = create_argument_parser()
    args = parser.parse_args()

    original_input = args.input
    resolved_inputs = _expand_input_paths(original_input)
    if not resolved_inputs:
        if glob.has_magic(os.path.expanduser(original_input or "")):
            logger.error("No files matched the --input pattern: %s", original_input)
        else:
            logger.error("Input video file must be specified via --input argument.")
        sys.exit(1)

    aggregated_costs: Optional[Dict[str, float]] = None
    exit_code = 0
    multiple_inputs = len(resolved_inputs) > 1

    for idx, input_path in enumerate(resolved_inputs, start=1):
        args.input = input_path
        config = create_config_from_args(args)

        glossary = config.get("glossary", {})
        if glossary:
            logger.debug("Using translation glossary with %d entries", len(glossary))

        voice_prompt = config.get("voice_prompt", {})
        if voice_prompt:
            logger.debug("Using voice prompts for %d speakers", len(voice_prompt))
            for speaker, prompt in voice_prompt.items():
                truncated = f"{prompt[:50]}..." if len(prompt) > 50 else prompt
                logger.debug("  %s: %s", speaker, truncated)

        if multiple_inputs:
            logger.info("Processing file %d/%d: %s", idx, len(resolved_inputs), input_path)

        if config.get("estimate_cost"):
            run_code, costs = _run_cost_estimation(config)
            if costs:
                aggregated_costs = _merge_costs(aggregated_costs, costs)
        else:
            run_code, _ = _run_dubbing(config)

        if run_code != 0:
            exit_code = run_code
            if not multiple_inputs:
                break
            logger.error("Failed processing input: %s", input_path)

    if aggregated_costs and len(resolved_inputs) > 1:
        logger.info("Aggregate estimated costs across %d files:", len(resolved_inputs))
        for step in ("transcription", "translation", "speech_synthesis", "total"):
            logger.info("  %s: $%.4f", step.replace("_", " ").title(), aggregated_costs.get(step, 0.0))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
