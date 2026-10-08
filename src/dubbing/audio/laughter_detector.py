"""Laughter detection and verification based on omine-me/LaughterSegmentation.

Model and method: Taisei Omine, Kenta Akita, Reiji Tsuruno,
"Robust Laughter Segmentation with Automatic Diverse Data Synthesis",
Interspeech 2024. Code: https://github.com/omine-me/LaughterSegmentation

The pretrained checkpoint is downloaded lazily from the Hugging Face Hub
(``omine-me/LaughterSegmentation``) and is available for research use only.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.log_config import get_logger

logger = get_logger(__name__)

DETECTOR_VERSION = "omine-v1"
DEFAULT_MODEL_REPO = "omine-me/LaughterSegmentation"
DEFAULT_BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

SAMPLE_RATE = 16000
FRAME_THRESHOLD = 0.5
MIN_EVENT_SECONDS = 0.2
CONCAT_GAP_SECONDS = 0.2
INPUT_SECONDS = 7.0
OVERLAP_SECONDS = 2.0
BATCH_SIZE = 10


class LaughterDetectorUnavailable(RuntimeError):
    """Raised when the laughter model cannot be loaded."""


@dataclass(frozen=True)
class LaughterEvent:
    """A detected laughter interval in seconds."""

    start: float
    end: float
    score: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class LaughterMeasurement:
    """Frame-level outcome for one short clip (used for verification)."""

    max_prob: float
    seconds_over_half: float


def concat_close(events: Sequence[LaughterEvent], gap: float = CONCAT_GAP_SECONDS) -> List[LaughterEvent]:
    """Merge events separated by less than ``gap`` seconds."""
    merged: List[LaughterEvent] = []
    for event in sorted(events, key=lambda item: item.start):
        if merged and event.start - merged[-1].end < gap:
            previous = merged[-1]
            merged[-1] = LaughterEvent(
                start=previous.start,
                end=max(previous.end, event.end),
                score=max(previous.score, event.score),
            )
        else:
            merged.append(event)
    return merged


def remove_short(events: Sequence[LaughterEvent], min_length: float = MIN_EVENT_SECONDS) -> List[LaughterEvent]:
    """Drop events shorter than ``min_length`` seconds."""
    return [event for event in events if event.duration >= min_length]


def frames_to_events(
    probs: Sequence[float],
    frame_seconds: float,
    offset: float = 0.0,
    threshold: float = FRAME_THRESHOLD,
    min_length: float = MIN_EVENT_SECONDS,
    concat_gap: float = CONCAT_GAP_SECONDS,
) -> List[LaughterEvent]:
    """Convert per-frame laughter probabilities into merged time intervals."""
    events: List[LaughterEvent] = []
    run_start: Optional[int] = None
    run_max = 0.0

    for index, prob in enumerate(probs):
        value = float(prob)
        if value >= threshold:
            if run_start is None:
                run_start = index
                run_max = value
            else:
                run_max = max(run_max, value)
        elif run_start is not None:
            events.append(
                LaughterEvent(
                    start=offset + run_start * frame_seconds,
                    end=offset + index * frame_seconds,
                    score=run_max,
                )
            )
            run_start = None

    if run_start is not None:
        events.append(
            LaughterEvent(
                start=offset + run_start * frame_seconds,
                end=offset + len(probs) * frame_seconds,
                score=run_max,
            )
        )

    return remove_short(concat_close(events, concat_gap), min_length)


def amplify_silence(array: np.ndarray, sr: int, mul_fac: float = 5.0) -> np.ndarray:
    """Boost silent stretches before detection, as done by the reference code.

    Quiet laughter (chuckles under speech, laughter without voiced speech) is
    otherwise easy to miss. Failures are non-fatal: the original audio is used.
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_silence

        data = np.asarray(array, dtype=np.float32)
        clipped = np.clip(data, -1.0, 1.0)
        segment = AudioSegment(
            (clipped * 32767).astype("int16").tobytes(),
            sample_width=2,
            frame_rate=sr,
            channels=1,
        )
        silent_sections = detect_silence(segment, min_silence_len=270, silence_thresh=-35)

        samples_per_ms = sr // 1000
        for start_ms, end_ms in silent_sections:
            fade_len = int(sr * 0.15)
            start_sample = start_ms * samples_per_ms
            end_sample = end_ms * samples_per_ms
            if end_sample - start_sample > fade_len * 2:
                data[start_sample:start_sample + fade_len] *= np.linspace(1, mul_fac, fade_len)
                data[start_sample + fade_len:end_sample - fade_len] *= mul_fac
                if end_sample < len(data):
                    data[end_sample - fade_len:end_sample] *= np.linspace(mul_fac, 1, fade_len)
            else:
                data[start_sample:end_sample] *= mul_fac

        import librosa

        return librosa.util.normalize(data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Laughter silence amplification failed: %s", exc)
        return np.asarray(array, dtype=np.float32)


class LaughterDetector:
    """Frame-level laughter detector/verifier around the omine checkpoint."""

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL_REPO,
        base_model: str = DEFAULT_BASE_MODEL,
        device: Optional[str] = None,
        frame_threshold: float = FRAME_THRESHOLD,
        min_event_seconds: float = MIN_EVENT_SECONDS,
        concat_gap_seconds: float = CONCAT_GAP_SECONDS,
        input_seconds: float = INPUT_SECONDS,
        overlap_seconds: float = OVERLAP_SECONDS,
        batch_size: int = BATCH_SIZE,
        hf_cache_dir: Optional[str] = None,
        scorer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        if input_seconds <= overlap_seconds:
            raise ValueError("input_seconds must be greater than overlap_seconds")
        self.model_repo = model_repo
        self.base_model = base_model
        self.device = device
        self.frame_threshold = float(frame_threshold)
        self.min_event_seconds = float(min_event_seconds)
        self.concat_gap_seconds = float(concat_gap_seconds)
        self.input_seconds = float(input_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.batch_size = max(1, int(batch_size))
        self.hf_cache_dir = hf_cache_dir
        self._scorer = scorer
        self._model = None
        self._torch = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._scorer is not None:
            return
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from huggingface_hub import hf_hub_download
                from transformers import Wav2Vec2ForAudioFrameClassification

                if self.device:
                    device = torch.device(self.device)
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                logger.info("Loading laughter model %s (base=%s)", self.model_repo, self.base_model)
                model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
                    self.base_model,
                    num_labels=1,
                    problem_type="single_label_classification",
                )
                checkpoint_path = hf_hub_download(
                    repo_id=self.model_repo,
                    filename="model.safetensors",
                    cache_dir=self.hf_cache_dir,
                )
                import safetensors.torch as safetensors_torch

                state = safetensors_torch.load_file(checkpoint_path, device="cpu")
                state = {
                    key[len("audio_model."):]: value
                    for key, value in state.items()
                    if key.startswith("audio_model.")
                } or state
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing:
                    logger.warning("Laughter model missing keys: %s", missing[:5])
                if unexpected:
                    logger.warning("Laughter model unexpected keys: %s", unexpected[:5])

                model.to(device)
                model.eval()
                self._model = model
                self._torch = torch
                logger.info("Laughter model loaded on %s", device)
            except Exception as exc:
                raise LaughterDetectorUnavailable(str(exc)) from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _window_starts(self, total_samples: int) -> range:
        window = int(self.input_seconds * SAMPLE_RATE)
        stride = int((self.input_seconds - self.overlap_seconds) * SAMPLE_RATE)
        return range(0, max(total_samples, 1), stride)

    def _predict(self, waveform: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Return ``(probs, offset_seconds)`` per window, with padding applied."""
        self._ensure_model()
        window = int(self.input_seconds * SAMPLE_RATE)
        results: List[Tuple[np.ndarray, float]] = []
        starts = list(self._window_starts(len(waveform)))

        if self._scorer is not None:
            for start in starts:
                chunk = waveform[start:start + window]
                if len(chunk) < window:
                    chunk = np.pad(chunk, (0, window - len(chunk)))
                results.append((np.asarray(self._scorer(chunk), dtype=np.float32), start / SAMPLE_RATE))
            return results

        torch = self._torch
        for batch_start in range(0, len(starts), self.batch_size):
            batch_starts = starts[batch_start:batch_start + self.batch_size]
            chunks = []
            for start in batch_starts:
                chunk = waveform[start:start + window]
                if len(chunk) < window:
                    chunk = np.pad(chunk, (0, window - len(chunk)))
                chunks.append(chunk)
            tensor = torch.from_numpy(np.stack(chunks)).float().to(self._model.device)
            with torch.no_grad():
                logits = self._model(input_values=tensor).logits
            probs = torch.sigmoid(logits.to(torch.float32)).squeeze(-1).cpu().numpy()
            for index, start in enumerate(batch_starts):
                results.append((probs[index] if probs.ndim > 1 else probs, start / SAMPLE_RATE))
        return results

    def detect_from_waveform(self, waveform: np.ndarray, sr: int = SAMPLE_RATE) -> List[LaughterEvent]:
        """Detect laughter events in a mono waveform."""
        import librosa

        if sr != SAMPLE_RATE:
            waveform = librosa.resample(np.asarray(waveform, dtype=np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        waveform = amplify_silence(np.asarray(waveform, dtype=np.float32), SAMPLE_RATE)

        events: List[LaughterEvent] = []
        for probs, offset in self._predict(waveform):
            frame_count = len(probs)
            if frame_count == 0:
                continue
            frame_seconds = self.input_seconds / frame_count
            events.extend(
                frames_to_events(
                    probs,
                    frame_seconds,
                    offset=offset,
                    threshold=self.frame_threshold,
                    min_length=self.min_event_seconds,
                    concat_gap=self.concat_gap_seconds,
                )
            )
        return concat_close(events, self.concat_gap_seconds)

    def detect(self, audio_path: str) -> List[LaughterEvent]:
        """Detect laughter events in an audio file."""
        import librosa

        waveform, _sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        return self.detect_from_waveform(waveform, SAMPLE_RATE)

    def measure_from_waveform(self, waveform: np.ndarray, sr: int = SAMPLE_RATE) -> LaughterMeasurement:
        """Return the strongest laughter response of a (short) waveform."""
        import librosa

        if sr != SAMPLE_RATE:
            waveform = librosa.resample(np.asarray(waveform, dtype=np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        waveform = amplify_silence(np.asarray(waveform, dtype=np.float32), SAMPLE_RATE)
        max_prob = 0.0
        seconds_over = 0.0
        for probs, _offset in self._predict(waveform):
            if len(probs) == 0:
                continue
            frame_seconds = self.input_seconds / len(probs)
            max_prob = max(max_prob, float(np.max(probs)))
            seconds_over += float(np.count_nonzero(probs >= FRAME_THRESHOLD)) * frame_seconds
        return LaughterMeasurement(max_prob=max_prob, seconds_over_half=seconds_over)

    def measure(self, audio_path: str) -> LaughterMeasurement:
        """Return the strongest laughter response of a (short) clip.

        Used as the laughter verification signal for synthesized segments.
        """
        import librosa

        waveform, _sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        return self.measure_from_waveform(waveform, SAMPLE_RATE)

    def has_laughter_from_waveform(
        self,
        waveform: np.ndarray,
        threshold: float = 0.5,
        min_seconds: float = 0.04,
        sr: int = SAMPLE_RATE,
    ) -> Tuple[bool, LaughterMeasurement]:
        """Verify that a waveform contains audible laughter."""
        measurement = self.measure_from_waveform(waveform, sr)
        ok = measurement.max_prob >= threshold and measurement.seconds_over_half >= min_seconds
        return ok, measurement

    def has_laughter(self, audio_path: str, threshold: float = 0.5, min_seconds: float = 0.04) -> Tuple[bool, LaughterMeasurement]:
        """Verify that a clip contains audible laughter."""
        measurement = self.measure(audio_path)
        ok = measurement.max_prob >= threshold and measurement.seconds_over_half >= min_seconds
        return ok, measurement
