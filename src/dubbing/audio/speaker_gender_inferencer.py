"""Speaker gender inference using a local wav2vec2 classifier."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from src.dubbing.core.cache_manager import CacheManager
from src.dubbing.core.log_config import get_logger
from src.utils.speaker_gender import (
    DEFAULT_GENDER_CONFIDENCE_THRESHOLD,
    DEFAULT_SPEAKER_GENDER,
    DEFAULT_SPEAKER_GENDER_MODEL_ID,
)

logger = get_logger(__name__)


class _ModelHead(nn.Module):
    """Classification head copied from the model card usage example."""

    def __init__(self, config, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _AgeGenderModel(Wav2Vec2PreTrainedModel):
    """Age/gender model shape from the public model card."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = _ModelHead(config, 1)
        self.gender = _ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


class SpeakerGenderInferencer:
    """Infer speaker gender-like labels from extracted speaker audio."""

    LABELS = ("female", "male", "child")

    def __init__(
        self,
        *,
        cache_manager: Optional[CacheManager] = None,
        device: Optional[torch.device | str] = None,
        model_id: str = DEFAULT_SPEAKER_GENDER_MODEL_ID,
        confidence_threshold: float = DEFAULT_GENDER_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.cache_manager = cache_manager
        self.model_id = model_id
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._processor: Optional[Wav2Vec2Processor] = None
        self._model: Optional[_AgeGenderModel] = None

    def infer_speakers(self, speaker_audio_paths: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Infer metadata for all provided speakers."""
        logger.info(
            "Starting speaker gender inference: speakers=%d, model=%s, device=%s, threshold=%.2f",
            len(speaker_audio_paths),
            self.model_id,
            self.device,
            self.confidence_threshold,
        )
        results: Dict[str, Dict[str, Any]] = {}
        for speaker, audio_path in speaker_audio_paths.items():
            results[speaker] = self._infer_single_speaker(speaker, audio_path)
        return results

    def _infer_single_speaker(self, speaker_name: str, audio_path: str) -> Dict[str, Any]:
        """Infer metadata for one speaker audio file."""
        speaker_audio_path = Path(audio_path)
        if not speaker_audio_path.exists():
            logger.warning(
                "Speaker gender inference skipped: speaker=%s, reason=missing_audio, path=%s",
                speaker_name,
                speaker_audio_path,
            )
            return self._unknown_result(raw_label="missing_audio")

        cache_key = self._cache_key_for_audio(speaker_audio_path)
        if self.cache_manager and self.cache_manager.cache_exists("speaker_gender", cache_key):
            try:
                cached = self.cache_manager.load_from_cache("speaker_gender", cache_key)
                if isinstance(cached, dict):
                    logger.info(
                        "Speaker gender inference cache hit: speaker=%s, audio=%s, raw_label=%s, confidence=%.4f, inferred_gender=%s",
                        speaker_name,
                        speaker_audio_path.name,
                        cached.get("rawLabel"),
                        float(cached.get("inferredConfidence") or 0.0),
                        cached.get("inferredGender"),
                    )
                    return cached
            except Exception as exc:
                logger.warning("Failed to load speaker gender cache for %s: %s", speaker_audio_path.name, exc)

        try:
            self._ensure_model_loaded()
            signal, sampling_rate = librosa.load(str(speaker_audio_path), sr=16000, mono=True)
            if signal.size == 0:
                logger.warning(
                    "Speaker gender inference skipped: speaker=%s, reason=empty_audio, audio=%s",
                    speaker_name,
                    speaker_audio_path.name,
                )
                result = self._unknown_result(raw_label="empty_audio")
            else:
                duration_seconds = float(signal.size) / float(sampling_rate)
                logger.info(
                    "Calling speaker gender model: speaker=%s, audio=%s, duration=%.2fs, sampling_rate=%d, model=%s",
                    speaker_name,
                    speaker_audio_path.name,
                    duration_seconds,
                    sampling_rate,
                    self.model_id,
                )
                result = self._run_model(signal, sampling_rate)
                logger.info(
                    "Speaker gender inference result: speaker=%s, audio=%s, raw_label=%s, confidence=%.4f, inferred_gender=%s",
                    speaker_name,
                    speaker_audio_path.name,
                    result.get("rawLabel"),
                    float(result.get("inferredConfidence") or 0.0),
                    result.get("inferredGender"),
                )
        except Exception as exc:
            logger.warning("Speaker gender inference failed for %s: %s", speaker_audio_path.name, exc)
            result = self._unknown_result(raw_label="inference_error")

        if self.cache_manager:
            try:
                self.cache_manager.save_to_cache("speaker_gender", cache_key, result)
            except Exception as exc:
                logger.warning("Failed to save speaker gender cache for %s: %s", speaker_audio_path.name, exc)

        return result

    def _ensure_model_loaded(self) -> None:
        """Load processor/model lazily on the first inference request."""
        if self._processor is not None and self._model is not None:
            return

        logger.info(
            "Loading speaker gender model: model=%s, device=%s",
            self.model_id,
            self.device,
        )
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        from_pretrained_kwargs = {"token": token} if token else {}

        self._processor = Wav2Vec2Processor.from_pretrained(
            self.model_id,
            **from_pretrained_kwargs,
        )
        self._model = _AgeGenderModel.from_pretrained(
            self.model_id,
            **from_pretrained_kwargs,
        )
        self._model.to(self.device)
        self._model.eval()
        logger.info(
            "Speaker gender model loaded successfully: model=%s, device=%s",
            self.model_id,
            self.device,
        )

    def _run_model(self, signal, sampling_rate: int) -> Dict[str, Any]:
        """Run the classifier and map the result to our metadata schema."""
        assert self._processor is not None
        assert self._model is not None

        processed = self._processor(
            signal,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = processed["input_values"].to(self.device)

        with torch.no_grad():
            _, _, logits_gender = self._model(input_values)

        probs = logits_gender[0].detach().cpu().tolist()
        best_idx = max(range(len(probs)), key=lambda idx: probs[idx])
        raw_label = self.LABELS[best_idx]
        confidence = float(probs[best_idx])
        inferred_gender = (
            DEFAULT_SPEAKER_GENDER
            if raw_label == "child" or confidence < self.confidence_threshold
            else raw_label
        )

        return {
            "inferredGender": inferred_gender,
            "inferredConfidence": confidence,
            "rawLabel": raw_label,
            "modelId": self.model_id,
            "overrideGender": None,
        }

    def _cache_key_for_audio(self, audio_path: Path) -> str:
        """Build a stable cache key per speaker audio file and model settings."""
        md5_hash = hashlib.md5()
        with audio_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        raw = f"{md5_hash.hexdigest()}|{self.model_id}|{self.confidence_threshold:.2f}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _unknown_result(self, *, raw_label: str) -> Dict[str, Any]:
        """Return a standardized unknown result."""
        return {
            "inferredGender": DEFAULT_SPEAKER_GENDER,
            "inferredConfidence": 0.0,
            "rawLabel": raw_label,
            "modelId": self.model_id,
            "overrideGender": None,
        }
