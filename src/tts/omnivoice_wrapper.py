"""Wrapper for the OmniVoice Hugging Face Space."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import DiarizationSegment, SegmentAlignment, TTSSegmentData
from .tts_interface import TTSInterface
from src.dubbing.core.log_config import get_logger

try:
    from gradio_client import Client, handle_file
    GRADIO_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency missing at runtime
    GRADIO_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency missing at runtime
    PYDUB_AVAILABLE = False

logger = get_logger(__name__)


class OmniVoiceWrapper(TTSInterface):
    """Text-to-speech wrapper around the public OmniVoice Space."""

    def __init__(
        self,
        space_id: str = "archivartaunik/OmniVoice",
        api_name: str = "/_clone_fn",
        default_reference_audio: Optional[str] = None,
        default_reference_text: Optional[str] = None,
        hf_token_env: str = "HF_TOKEN",
        lang: str = "Belarusian",
        num_steps: int = 32,
        guidance_scale: float = 2.0,
        denoise: bool = True,
        speed: float = 1.0,
        duration: float = 3.0,
        preprocess_prompt: bool = True,
        postprocess_output: bool = True,
        **_: Any,
    ) -> None:
        if not GRADIO_AVAILABLE:
            raise ImportError(
                "gradio-client package is required for OmniVoice integration. "
                "Install it with 'pip install gradio-client'."
            )
        if not PYDUB_AVAILABLE:
            raise ImportError(
                "pydub package is required to post-process OmniVoice audio. "
                "Install it with 'pip install pydub'."
            )

        self.space_id = space_id
        self.api_name = api_name
        self.default_reference_audio = default_reference_audio
        self.default_reference_text = default_reference_text
        self.hf_token_env = hf_token_env

        self.lang = lang
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.denoise = denoise
        self.speed = speed
        self.duration = duration
        self.preprocess_prompt = preprocess_prompt
        self.postprocess_output = postprocess_output

        self.voice_mapping: Dict[str, str] = {}
        self.voice_prompt_mapping: Dict[str, str] = {}

        self.client: Optional[Client] = None
        self._temp_dir: Optional[str] = None

    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_mapping = mapping or {}

    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_prompt_mapping = mapping or {}

    def initialize(self) -> None:
        token = os.getenv(self.hf_token_env)
        try:
            if token:
                logger.info(
                    "Initializing OmniVoice client with authenticated access to %s",
                    self.space_id,
                )
                self.client = Client(self.space_id, hf_token=token)
            else:
                logger.warning(
                    "%s environment variable not set - using anonymous Hugging Face access.",
                    self.hf_token_env,
                )
                self.client = Client(self.space_id)
        except Exception as exc:  # pragma: no cover - network failures at runtime
            raise RuntimeError(
                f"Failed to initialize OmniVoice client for space '{self.space_id}': {exc}"
            ) from exc

        self._temp_dir = tempfile.mkdtemp(prefix="omnivoice_segments_")
        logger.info("OmniVoice wrapper initialized for space %s", self.space_id)

    def _prepare_text(self, segment: TTSSegmentData) -> str:
        return segment.text

    def _resolve_reference_audio(self, segment: TTSSegmentData) -> Optional[str]:
        if segment.reference_audio_path:
            return segment.reference_audio_path
        if segment.speaker and segment.speaker in self.voice_mapping:
            return self.voice_mapping[segment.speaker]
        return self.default_reference_audio

    def _resolve_reference_text(self, segment: TTSSegmentData) -> Optional[str]:
        reference_text = segment.reference_text
        if isinstance(reference_text, str):
            return reference_text.strip()
        return ""

    def _resolve_duration(self, segment: TTSSegmentData) -> float:
        target_duration = getattr(segment, "target_duration", None)
        try:
            if target_duration is not None:
                target_duration = float(target_duration)
                if target_duration > 0:
                    return target_duration
        except (TypeError, ValueError):
            pass
        return self.duration

    def _extract_result_path(self, prediction: Any) -> Optional[Path]:
        if isinstance(prediction, str):
            return Path(prediction)
        if isinstance(prediction, (list, tuple)):
            for item in prediction:
                if isinstance(item, str) and item.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    return Path(item)
                if isinstance(item, dict):
                    candidate = item.get("path") or item.get("name")
                    if isinstance(candidate, str):
                        return Path(candidate)
        if isinstance(prediction, dict):
            candidate = prediction.get("path") or prediction.get("name")
            if isinstance(candidate, str):
                return Path(candidate)
        return None

    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: str = "Belarusian",
        **_: Any,
    ) -> List[SegmentAlignment]:
        if not self.client:
            raise RuntimeError("OmniVoice client not initialized. Call initialize() first.")
        if not segments_data:
            logger.warning("No segments provided to OmniVoice synthesis.")
            return []

        alignments: List[SegmentAlignment] = []

        for index, segment in enumerate(segments_data):
            prepared_text = self._prepare_text(segment)
            reference_audio = self._resolve_reference_audio(segment)
            reference_text = self._resolve_reference_text(segment)
            duration = self._resolve_duration(segment)

            if reference_audio and not Path(reference_audio).exists():
                logger.warning(
                    "Reference audio '%s' for speaker '%s' does not exist. Falling back to no clone reference.",
                    reference_audio,
                    segment.speaker,
                )
                reference_audio = None

            if not reference_audio:
                logger.error(
                    "OmniVoice requires reference audio for speaker '%s'. Skipping segment.",
                    segment.speaker,
                )
                continue

            speed = segment.speed if segment.speed is not None else self.speed

            logger.info(
                "OmniVoice: Synthesizing segment %d/%d for speaker '%s' (lang=%s)",
                index + 1,
                len(segments_data),
                segment.speaker,
                self.lang or language,
            )

            try:
                prediction = self.client.predict(
                    text=prepared_text,
                    lang=self.lang or language,
                    ref_aud=handle_file(reference_audio),
                    ref_text=reference_text,
                    ns=self.num_steps,
                    gs=self.guidance_scale,
                    dn=self.denoise,
                    sp=speed,
                    du=duration,
                    pp=self.preprocess_prompt,
                    po=self.postprocess_output,
                    api_name=self.api_name,
                )
            except Exception as exc:  # pragma: no cover - network/runtime errors
                logger.error("OmniVoice: Failed to synthesize segment %s: %s", segment.speaker, exc)
                continue

            result_path = self._extract_result_path(prediction)
            if not result_path or not result_path.exists():
                logger.error(
                    "OmniVoice output file not found for speaker '%s': %r",
                    segment.speaker,
                    prediction,
                )
                continue

            temp_output = Path(self._temp_dir) / f"segment_{index}.wav" if self._temp_dir else result_path
            try:
                if temp_output != result_path:
                    shutil.copy(result_path, temp_output)
                audio = AudioSegment.from_file(temp_output)
                duration = len(audio) / 1000.0
            except Exception as exc:  # pragma: no cover - audio parsing failures
                logger.error("OmniVoice: Unable to load synthesized audio: %s", exc)
                duration = 0.0

            if segment.output_path:
                try:
                    os.makedirs(os.path.dirname(segment.output_path), exist_ok=True)
                    shutil.copy(temp_output, segment.output_path)
                except Exception as exc:  # pragma: no cover - filesystem errors
                    logger.error(
                        "OmniVoice: Failed to save audio for speaker '%s' to %s: %s",
                        segment.speaker,
                        segment.output_path,
                        exc,
                    )

            diarized = DiarizationSegment(
                start_time=0.0,
                end_time=duration,
                speaker=segment.speaker,
                text=segment.text,
                confidence=1.0,
            )
            alignments.append(
                SegmentAlignment(
                    original_segment=segment,
                    diarized_segment=diarized,
                    alignment_confidence=1.0,
                )
            )

            if result_path.exists() and (not self._temp_dir or result_path.parent != Path(self._temp_dir)):
                try:
                    result_path.unlink()
                except OSError:
                    logger.debug("OmniVoice: Could not delete temporary file %s", result_path)

        return alignments

    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: str = "Belarusian",
    ) -> Optional[float]:
        if not segment_data.text:
            return 0.0
        words = segment_data.text.strip().split()
        if not words:
            return 0.0

        estimated = len(words) * 0.45
        speed = segment_data.speed if segment_data.speed not in (None, 0) else self.speed
        if speed and speed > 0:
            estimated /= speed
        return max(estimated, 0.8)

    def is_available(self) -> bool:
        return self.client is not None

    def cleanup(self) -> None:
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        self.client = None
