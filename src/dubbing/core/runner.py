"""Reusable configuration + execution helpers for CLI and UI entry points."""

import argparse
import io
import json
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

from dotenv import load_dotenv

from .config import DubbingConfig
from .log_config import setup_logging


LOGGER = logging.getLogger(__name__)

_JSON_FIELDS = {
    "glossary",
    "tts_system_mapping",
    "voice_prompt",
    "reference_audio_mapping",
    "reference_text_mapping",
}
_LIST_TEXT_FIELDS = {"keep_original_audio_ranges"}


@dataclass
class DubbingJobResult:
    """Structured result returned to UI callbacks."""

    status: str
    logs: str
    output_file: Optional[str] = None
    report_file: Optional[str] = None


def _normalize_override_value(key: str, value: Any) -> Any:
    """Convert UI values into config-friendly Python values."""
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None

    if key == "duration" and value == 0:
        return None

    if key in _JSON_FIELDS and isinstance(value, str):
        return json.loads(value)

    if key in _LIST_TEXT_FIELDS and isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]

    return value


def build_config_from_overrides(overrides: dict[str, Any]) -> DubbingConfig:
    """Create a validated config object from plain override values."""
    config = DubbingConfig()
    config_path = overrides.get("config", "dubbing_config.yml")
    if config_path:
        config.load_from_yaml(str(config_path))

    normalized_overrides = {}
    for key, value in overrides.items():
        if key == "config":
            continue
        normalized_value = _normalize_override_value(key, value)
        if normalized_value is not None:
            normalized_overrides[key] = normalized_value

    config.load_from_cli(argparse.Namespace(**normalized_overrides))
    config.validate()
    config.process_special_parameters()
    return config


def _run_combine_video_step(dubber: Any, config: DubbingConfig) -> str:
    """Run the advanced combine-video path used by the CLI."""
    expected_translated_audio = config.get("translated_audio_path")
    expected_background_audio = None

    if config.get("keep_background"):
        expected_background_audio = config.get("background_audio_path")
        if not os.path.exists(expected_background_audio):
            LOGGER.warning(
                "Expected background audio %s not found. Proceeding without it.",
                expected_background_audio,
            )
            expected_background_audio = None

    if not os.path.exists(expected_translated_audio):
        raise FileNotFoundError(
            "run_step=combine_video requires existing artifacts from a previous full dubbing run. "
            f"Expected translated audio {expected_translated_audio} not found. "
            "If you are using the Gradio UI, clear 'Run step' and click 'Run DubbLM' to execute the full pipeline first."
        )

    watermark_input_path = config.get("watermark_path")
    if watermark_input_path and not os.path.exists(watermark_input_path):
        LOGGER.warning(
            "Watermark image %s not found. Proceeding without it.",
            watermark_input_path,
        )
        watermark_input_path = None

    return dubber.video_processor.combine_audio_with_video(
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


def _extract_output_path(step_result: Any) -> Optional[str]:
    """Normalize pipeline step results down to a file path for UI consumers."""
    if isinstance(step_result, tuple):
        if not step_result:
            return None
        return _extract_output_path(step_result[0])

    if step_result is None:
        return None

    return str(step_result)


def run_dubbing_job(
    overrides: dict[str, Any],
    *,
    dubbing_factory: Optional[Callable[[DubbingConfig], Any]] = None,
) -> DubbingJobResult:
    """Run the configured dubbing job and capture logs for the caller."""
    if not logging.getLogger().handlers:
        setup_logging()

    log_stream = io.StringIO()
    capture_handler = logging.StreamHandler(log_stream)
    capture_handler.setLevel(logging.DEBUG)
    capture_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))

    root_logger = logging.getLogger()
    root_logger.addHandler(capture_handler)

    try:
        load_dotenv()
        config = build_config_from_overrides(overrides)

        if dubbing_factory is None:
            from .smart_dubbing import SmartDubbing

            dubbing_factory = SmartDubbing

        dubber = dubbing_factory(config)

        if config.get("generate_speaker_report"):
            report_path, samples_path = dubber.generate_diarization_report()
            logging.getLogger(__name__).info("Speaker report generated: %s", report_path)
            return DubbingJobResult(
                status="Speaker report generated",
                logs=log_stream.getvalue(),
                output_file=str(samples_path),
                report_file=str(report_path),
            )

        if config.get("run_step") == "combine_video":
            output_path = _extract_output_path(_run_combine_video_step(dubber, config))
            logging.getLogger(__name__).info("Video combination complete: %s", output_path)
            return DubbingJobResult(
                status="Combine step completed",
                logs=log_stream.getvalue(),
                output_file=output_path,
            )

        output_path = dubber.run_pipeline(
            save_original_subtitles=config.get("save_original_subtitles", False),
            save_translated_subtitles=config.get("save_translated_subtitles", False),
        )
        logging.getLogger(__name__).info("Video dubbing complete: %s", output_path)
        return DubbingJobResult(
            status="Completed",
            logs=log_stream.getvalue(),
            output_file=str(output_path),
        )
    except Exception as exc:
        traceback.print_exc(file=log_stream)
        return DubbingJobResult(
            status=f"Failed: {exc}",
            logs=log_stream.getvalue(),
        )
    finally:
        root_logger.removeHandler(capture_handler)
