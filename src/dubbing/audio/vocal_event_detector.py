"""Non-verbal vocal event detection based on the AST AudioSet model.

Model: ``MIT/ast-finetuned-audioset-10-10-0.4593`` (Audio Spectrogram
Transformer, AudioSet / 527 classes).

Unlike window-level classifiers, this detector applies the AudioSet
classification head to each time patch of the AST encoder, which yields
frame-level probabilities with roughly 100 ms resolution. Frames are merged
into ``VocalEvent`` intervals per kind (screams, sobbing, gasps, ...).

Laughter keeps using the dedicated omine frame classifier
(:mod:`src.dubbing.audio.laughter_detector`) because it is more accurate for
that specific class; AST covers the remaining non-verbal vocalisations.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.log_config import get_logger

logger = get_logger(__name__)

DETECTOR_VERSION = "ast-v1"
DEFAULT_MODEL_REPO = "MIT/ast-finetuned-audioset-10-10-0.4593"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 10.0
HOP_SECONDS = 1.0
FRAME_THRESHOLD = 0.35
MIN_EVENT_SECONDS = 0.3
CONCAT_GAP_SECONDS = 0.5
BATCH_SIZE = 2
PROBABILITY_SOFTMAX = "softmax"
PROBABILITY_SIGMOID = "sigmoid"
DETECTION_MODE_WINDOW = "window"
DETECTION_MODE_FRAME = "frame"

# Canonical DubbLM event kinds (mirrors the Gemini 3.8 inline vocal vocabulary).
VOCAL_EVENT_KINDS: Tuple[str, ...] = (
    "laugh",
    "cry",
    "scream",
    "shout",
    "whisper",
    "gasp",
    "pant",
    "sigh",
    "groan",
    "grunt",
    "whimper",
    "breath",
    "snort",
)

# Kinds enabled when the config does not specify a list. Excludes noisy classes
# that commonly fire during ordinary speech (shout, groan, grunt, pant, breath,
# snort) and whisper, which is better handled by TTS style than event tagging.
DEFAULT_DETECTION_KINDS: Tuple[str, ...] = (
    "cry",
    "scream",
    "whimper",
    "gasp",
    "sigh",
)

# AudioSet class label -> canonical kind.
CLASS_KIND_MAP: Dict[str, str] = {
    "Laughter": "laugh",
    "Baby laughter": "laugh",
    "Giggle": "laugh",
    "Snicker": "laugh",
    "Belly laugh": "laugh",
    "Chuckle, chortle": "laugh",
    "Crying, sobbing": "cry",
    "Baby cry, infant cry": "cry",
    "Whimper": "cry",
    "Wail, moan": "cry",
    "Screaming": "scream",
    "Battle cry": "scream",
    "Shout": "shout",
    "Yell": "shout",
    "Children shouting": "shout",
    "Bellow": "shout",
    "Whoop": "shout",
    "Whispering": "whisper",
    "Sigh": "sigh",
    "Groan": "groan",
    "Grunt": "grunt",
    "Breathing": "breath",
    "Gasp": "gasp",
    "Pant": "pant",
    "Snort": "snort",
}

# Per-kind default thresholds on the pooled window score; unspecified kinds
# use ``FRAME_THRESHOLD``. Noisy AudioSet classes get a higher bar.
KIND_THRESHOLDS: Dict[str, float] = {
    "cry": 0.30,
    "scream": 0.30,
    "shout": 0.35,
    "whisper": 0.35,
    "gasp": 0.35,
    "pant": 0.45,
    "sigh": 0.45,
    "groan": 0.50,
    "grunt": 0.50,
    "whimper": 0.40,
    "breath": 0.50,
    "snort": 0.50,
}


class VocalEventDetectorUnavailable(RuntimeError):
    """Raised when the AST model cannot be loaded."""


@dataclass(frozen=True)
class VocalEvent:
    """A detected non-verbal vocal event interval in seconds."""

    kind: str
    start: float
    end: float
    score: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class VocalEventMeasurement:
    """Frame-level outcome for one short clip (used for verification)."""

    kind: str
    max_prob: float
    seconds_over_threshold: float


def concat_close(events: Sequence[VocalEvent], gap: float = CONCAT_GAP_SECONDS) -> List[VocalEvent]:
    """Merge same-kind events separated by less than ``gap`` seconds."""
    merged: List[VocalEvent] = []
    for event in sorted(events, key=lambda item: (item.start, item.kind)):
        if (
            merged
            and merged[-1].kind == event.kind
            and event.start - merged[-1].end < gap
        ):
            previous = merged[-1]
            merged[-1] = VocalEvent(
                kind=previous.kind,
                start=previous.start,
                end=max(previous.end, event.end),
                score=max(previous.score, event.score),
            )
        else:
            merged.append(event)
    return merged


def remove_short(events: Sequence[VocalEvent], min_length: float = MIN_EVENT_SECONDS) -> List[VocalEvent]:
    """Drop events shorter than ``min_length`` seconds."""
    return [event for event in events if event.duration >= min_length]


def frames_to_events(
    probs: Sequence[float],
    kind: str,
    frame_seconds: float,
    offset: float = 0.0,
    threshold: float = FRAME_THRESHOLD,
    min_length: float = MIN_EVENT_SECONDS,
    concat_gap: float = CONCAT_GAP_SECONDS,
) -> List[VocalEvent]:
    """Convert per-frame probabilities of one kind into merged intervals."""
    events: List[VocalEvent] = []
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
                VocalEvent(
                    kind=kind,
                    start=offset + run_start * frame_seconds,
                    end=offset + index * frame_seconds,
                    score=run_max,
                )
            )
            run_start = None

    if run_start is not None:
        events.append(
            VocalEvent(
                kind=kind,
                start=offset + run_start * frame_seconds,
                end=offset + len(probs) * frame_seconds,
                score=run_max,
            )
        )

    return remove_short(concat_close(events, concat_gap), min_length)


def _default_cache_dir() -> Optional[str]:
    """Prefer the repository-local Hugging Face cache when present.

    DubbLM keeps downloaded models under ``cache/huggingface`` (see the
    laughter model). Falling back to ``None`` lets huggingface_hub use the
    user-level cache or the ``HF_HOME`` environment variable.
    """
    import os

    for candidate in ("cache/huggingface", "cache/huggingface/hub"):
        if os.path.isdir(candidate):
            return candidate
    return None


class AstVocalEventDetector:
    """Frame-level non-verbal vocal event detector around the AST checkpoint."""

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL_REPO,
        device: Optional[str] = None,
        window_seconds: float = WINDOW_SECONDS,
        hop_seconds: float = HOP_SECONDS,
        frame_threshold: float = FRAME_THRESHOLD,
        kind_thresholds: Optional[Dict[str, float]] = None,
        min_event_seconds: float = MIN_EVENT_SECONDS,
        concat_gap_seconds: float = CONCAT_GAP_SECONDS,
        batch_size: int = BATCH_SIZE,
        probability_mode: str = PROBABILITY_SOFTMAX,
        detection_mode: str = DETECTION_MODE_WINDOW,
        kinds: Optional[Sequence[str]] = None,
        hf_cache_dir: Optional[str] = None,
        frame_scorer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        window_scorer: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
        stub_frame_seconds: float = 0.1,
    ):
        if hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive")
        self.model_repo = model_repo
        self.device = device
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.frame_threshold = float(frame_threshold)
        self.kind_thresholds = dict(kind_thresholds or {})
        self.min_event_seconds = float(min_event_seconds)
        self.concat_gap_seconds = float(concat_gap_seconds)
        self.batch_size = max(1, int(batch_size))
        self.probability_mode = str(probability_mode or PROBABILITY_SOFTMAX).strip().lower()
        self.detection_mode = str(detection_mode or DETECTION_MODE_WINDOW).strip().lower()
        requested = tuple(kinds) if kinds else VOCAL_EVENT_KINDS
        ordered_kinds: List[str] = []
        seen_kinds: set = set()
        for kind in requested:
            if kind in VOCAL_EVENT_KINDS and kind not in seen_kinds:
                seen_kinds.add(kind)
                ordered_kinds.append(kind)
        self.kinds: Tuple[str, ...] = tuple(ordered_kinds) or VOCAL_EVENT_KINDS
        self.hf_cache_dir = hf_cache_dir if hf_cache_dir is not None else _default_cache_dir()
        self._frame_scorer = frame_scorer
        self._window_scorer = window_scorer
        self._stub_frame_seconds = float(stub_frame_seconds)
        self._model = None
        self._feature_extractor = None
        self._torch = None
        self._kind_indices: Dict[str, List[int]] = {}
        self._model_frame_seconds = 0.1
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._frame_scorer is not None or self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import ASTFeatureExtractor, ASTForAudioClassification

                if self.device:
                    device = torch.device(self.device)
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                logger.info("Loading AST vocal event model %s", self.model_repo)
                feature_extractor = ASTFeatureExtractor.from_pretrained(
                    self.model_repo, cache_dir=self.hf_cache_dir
                )
                model = ASTForAudioClassification.from_pretrained(
                    self.model_repo, cache_dir=self.hf_cache_dir
                )
                model.to(device)
                model.eval()

                self._kind_indices = self._build_kind_indices(model.config.id2label)
                if not self._kind_indices:
                    raise VocalEventDetectorUnavailable(
                        "AST model exposes none of the configured vocal event classes"
                    )
                self._model = model
                self._feature_extractor = feature_extractor
                self._torch = torch
                logger.info(
                    "AST vocal event model loaded on %s (kinds: %s)",
                    device,
                    ", ".join(sorted(self._kind_indices)),
                )
            except VocalEventDetectorUnavailable:
                raise
            except Exception as exc:
                raise VocalEventDetectorUnavailable(str(exc)) from exc

    def _build_kind_indices(self, id2label: Dict[object, str]) -> Dict[str, List[int]]:
        indices: Dict[str, List[int]] = {}
        for raw_index, label in (id2label or {}).items():
            kind = CLASS_KIND_MAP.get(str(label))
            if kind and kind in self.kinds:
                indices.setdefault(kind, []).append(int(raw_index))
        return indices

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_waveform(audio_path: str) -> np.ndarray:
        """Load any ffmpeg-supported file as mono float32 at 16 kHz.

        Uses pydub/ffmpeg instead of librosa/soundfile so the detector has no
        extra native audio dependency beyond what the pipeline already needs.
        """
        from pydub import AudioSegment

        segment = AudioSegment.from_file(audio_path)
        segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
        samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
        if segment.sample_width == 2:
            samples = samples / 32768.0
        elif segment.sample_width == 4:
            samples = samples / 2147483648.0
        else:
            samples = samples / (2 ** (8 * segment.sample_width - 1))
        return samples.astype(np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _window_starts(self, total_samples: int) -> List[int]:
        window = int(self.window_seconds * SAMPLE_RATE)
        stride = max(1, int(self.hop_seconds * SAMPLE_RATE))
        if total_samples <= window:
            return [0]
        return list(range(0, total_samples, stride))

    def _frame_seconds(self) -> float:
        return self._stub_frame_seconds

    def _score_windows(self, waveform: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        """Return ``(probability_matrix, offset_seconds, frame_seconds)`` per window.

        The matrix has shape ``(time_frames, len(self.kinds))`` and holds
        per-kind probabilities for every time frame of the window.
        """
        self._ensure_model()
        window = int(self.window_seconds * SAMPLE_RATE)
        starts = self._window_starts(len(waveform))
        results: List[Tuple[np.ndarray, float, float]] = []

        if self._frame_scorer is not None:
            for start in starts:
                chunk = waveform[start:start + window]
                if len(chunk) < window:
                    chunk = np.pad(chunk, (0, window - len(chunk)))
                matrix = np.asarray(self._frame_scorer(chunk), dtype=np.float32)
                if matrix.ndim == 1:
                    matrix = matrix[:, None]
                results.append((matrix, start / SAMPLE_RATE, self._frame_seconds()))
            return results

        torch = self._torch
        model = self._model
        extractor = self._feature_extractor
        for batch_start in range(0, len(starts), self.batch_size):
            batch_starts = starts[batch_start:batch_start + self.batch_size]
            chunks = []
            for start in batch_starts:
                chunk = waveform[start:start + window]
                if len(chunk) < window:
                    chunk = np.pad(chunk, (0, window - len(chunk)))
                chunks.append(chunk)
            inputs = extractor(chunks, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model.audio_spectrogram_transformer(
                    inputs["input_values"], output_hidden_states=True
                )
                hidden = outputs.last_hidden_state
                matrix, frame_seconds = self._frames_from_hidden(hidden)
            for index, start in enumerate(batch_starts):
                results.append((matrix[index], start / SAMPLE_RATE, frame_seconds))
        return results

    def _frames_from_hidden(self, hidden) -> Tuple[np.ndarray, float]:
        """Apply the AudioSet head to every time patch of the AST encoder.

        AST token layout is ``[CLS, distillation, time x freq patches]``.
        Returns ``(probs, frame_seconds)`` where ``probs`` has shape
        ``(batch, time_frames, len(self.kinds))``.
        """
        model = self._model
        config = model.config
        num_freq_patches = (int(config.num_mel_bins) - int(config.patch_size)) // int(
            config.frequency_stride
        ) + 1
        num_time_patches = (int(config.max_length) - int(config.patch_size)) // int(
            config.time_stride
        ) + 1
        patch_tokens = hidden[:, 2:, :]
        expected = num_time_patches * num_freq_patches
        if patch_tokens.shape[1] != expected:
            # Shape drift across transformers versions: fall back to the
            # pooled classification (window-level, single frame).
            pooled = (hidden[:, 0] + hidden[:, 1]) / 2
            logits = model.classifier(pooled)
            probs = self._probabilities(logits)
            return self._select_kinds(probs)[:, None, :], self.window_seconds

        time_hidden = patch_tokens.reshape(
            patch_tokens.shape[0], num_time_patches, num_freq_patches, patch_tokens.shape[-1]
        ).mean(dim=2)
        flat = model.classifier.layernorm(time_hidden)
        logits = model.classifier.dense(flat)
        probs = self._probabilities(logits)
        hop_seconds = int(config.time_stride) * 160 / SAMPLE_RATE
        return self._select_kinds(probs), hop_seconds

    def _probabilities(self, logits):
        torch = self._torch
        if self.probability_mode == PROBABILITY_SIGMOID:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def _select_kinds(self, probs):
        torch = self._torch
        columns = []
        for kind in self.kinds:
            indices = self._kind_indices.get(kind) or []
            if indices:
                columns.append(probs[..., indices].amax(dim=-1))
            else:
                columns.append(torch.zeros(probs.shape[:-1], device=probs.device))
        return torch.stack(columns, dim=-1).to(torch.float32).cpu().numpy()

    def _threshold_for(self, kind: str) -> float:
        return float(self.kind_thresholds.get(kind, self.frame_threshold))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _score_windows_pooled(
        self, waveform: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Return ``(patch_probs, pooled_probs, offset_seconds)`` per window.

        ``patch_probs`` has shape ``(time_frames, len(self.kinds))`` and is used
        only to localise an event inside the peak window. ``pooled_probs`` has
        shape ``(len(self.kinds),)`` and drives detection: the window-level
        AudioSet head is far better calibrated than per-patch logits.
        """
        self._ensure_model()
        window = int(self.window_seconds * SAMPLE_RATE)
        starts = self._window_starts(len(waveform))
        results: List[Tuple[np.ndarray, np.ndarray, float]] = []

        if self._window_scorer is not None:
            for start in starts:
                chunk = waveform[start:start + window]
                if len(chunk) < window:
                    chunk = np.pad(chunk, (0, window - len(chunk)))
                patch_matrix, pooled = self._window_scorer(chunk)
                patch_matrix = np.asarray(patch_matrix, dtype=np.float32)
                pooled = np.asarray(pooled, dtype=np.float32).reshape(-1)
                if patch_matrix.ndim == 1:
                    patch_matrix = patch_matrix[:, None]
                results.append((patch_matrix, pooled, start / SAMPLE_RATE))
            return results

        torch = self._torch
        model = self._model
        extractor = self._feature_extractor
        for batch_start in range(0, len(starts), self.batch_size):
            batch_starts = starts[batch_start:batch_start + self.batch_size]
            chunks = []
            for start in batch_starts:
                chunk = waveform[start:start + window]
                if len(chunk) < window:
                    chunk = np.pad(chunk, (0, window - len(chunk)))
                chunks.append(chunk)
            inputs = extractor(chunks, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model.audio_spectrogram_transformer(
                    inputs["input_values"], output_hidden_states=True
                )
                hidden = outputs.last_hidden_state
                pooled_logits = model.classifier((hidden[:, 0] + hidden[:, 1]) / 2)
                pooled = self._select_kinds(self._probabilities(pooled_logits))
                patch, frame_seconds = self._frames_from_hidden(hidden)
                self._model_frame_seconds = frame_seconds
            for index, start in enumerate(batch_starts):
                results.append((patch[index], pooled[index], start / SAMPLE_RATE))
        return results

    def _localization_frame_seconds(self) -> float:
        if self._window_scorer is not None:
            return self._stub_frame_seconds
        return self._model_frame_seconds or 0.1

    def _detect_frame_mode(self, waveform: np.ndarray) -> List[VocalEvent]:
        all_events: List[VocalEvent] = []
        for matrix, offset, frame_seconds in self._score_windows(waveform):
            if matrix.size == 0:
                continue
            for column, kind in enumerate(self.kinds):
                if column >= matrix.shape[1]:
                    break
                all_events.extend(
                    frames_to_events(
                        matrix[:, column],
                        kind,
                        frame_seconds=frame_seconds,
                        offset=offset,
                        threshold=self._threshold_for(kind),
                        min_length=self.min_event_seconds,
                        concat_gap=self.concat_gap_seconds,
                    )
                )
        return concat_close(all_events, self.concat_gap_seconds)

    def _detect_window_mode(self, waveform: np.ndarray) -> List[VocalEvent]:
        windows = self._score_windows_pooled(waveform)
        if not windows:
            return []
        all_events: List[VocalEvent] = []
        for column, kind in enumerate(self.kinds):
            threshold = self._threshold_for(kind)
            run_start: Optional[int] = None
            run_peak: Optional[int] = None
            run_max = 0.0
            for index, (_patch, pooled, _offset) in enumerate(windows):
                value = float(pooled[column]) if column < pooled.shape[0] else 0.0
                if value >= threshold:
                    if run_start is None:
                        run_start = index
                        run_peak = index
                        run_max = value
                    elif value > run_max:
                        run_peak = index
                        run_max = value
                elif run_start is not None:
                    all_events.append(
                        self._localize_window_event(
                            kind, column, windows, run_start, index, run_peak or run_start, run_max
                        )
                    )
                    run_start = None
            if run_start is not None:
                all_events.append(
                    self._localize_window_event(
                        kind, column, windows, run_start, len(windows), run_peak or run_start, run_max
                    )
                )
        return concat_close(all_events, self.concat_gap_seconds)

    def _localize_window_event(
        self,
        kind: str,
        column: int,
        windows: List[Tuple[np.ndarray, np.ndarray, float]],
        run_start: int,
        run_end: int,
        peak_index: int,
        score: float,
    ) -> VocalEvent:
        """Place a window-level detection on the timeline using patch frames."""
        patch, _pooled, offset = windows[peak_index]
        frame_seconds = self._localization_frame_seconds()
        if patch.size and column < patch.shape[1] and patch.shape[0] > 0:
            series = np.asarray(patch[:, column], dtype=np.float32)
            peak_frame = int(np.argmax(series))
            peak_value = float(series[peak_frame])
            lo = peak_frame
            hi = peak_frame
            if peak_value > 0:
                half = peak_value * 0.5
                while lo > 0 and series[lo - 1] >= half:
                    lo -= 1
                while hi < len(series) - 1 and series[hi + 1] >= half:
                    hi += 1
            start = offset + lo * frame_seconds
            end = offset + (hi + 1) * frame_seconds
        else:
            start = offset
            end = offset + self.window_seconds

        # Clamp to the extent of the windows that formed this run so a long run
        # never inflates into an over-wide preserve range.
        run_start_offset = windows[run_start][2]
        run_end_offset = windows[run_end - 1][2] + self.window_seconds
        start = max(start, run_start_offset)
        if end < start + self.min_event_seconds:
            end = start + self.min_event_seconds
        end = min(end, run_end_offset)
        if end < start + self.min_event_seconds:
            end = start + self.min_event_seconds
        return VocalEvent(kind=kind, start=start, end=end, score=score)

    def detect_from_waveform(
        self, waveform: np.ndarray, sr: int = SAMPLE_RATE
    ) -> List[VocalEvent]:
        """Detect vocal events in a mono waveform."""
        if sr != SAMPLE_RATE:
            waveform = _resample(waveform, sr, SAMPLE_RATE)
        waveform = np.asarray(waveform, dtype=np.float32)
        if self._frame_scorer is not None or self.detection_mode == DETECTION_MODE_FRAME:
            return self._detect_frame_mode(waveform)
        return self._detect_window_mode(waveform)

    def detect(self, audio_path: str) -> List[VocalEvent]:
        """Detect vocal events in an audio file."""
        return self.detect_from_waveform(self._load_waveform(audio_path), SAMPLE_RATE)

    def measure_from_waveform(
        self,
        waveform: np.ndarray,
        kind: str,
        threshold: Optional[float] = None,
        sr: int = SAMPLE_RATE,
    ) -> VocalEventMeasurement:
        """Return the strongest response for ``kind`` in a (short) waveform."""
        if sr != SAMPLE_RATE:
            waveform = _resample(waveform, sr, SAMPLE_RATE)
        waveform = np.asarray(waveform, dtype=np.float32)
        effective_threshold = float(threshold if threshold is not None else self._threshold_for(kind))
        column_index = self.kinds.index(kind) if kind in self.kinds else None
        max_prob = 0.0
        seconds_over = 0.0

        if self._frame_scorer is not None or self.detection_mode == DETECTION_MODE_FRAME:
            for matrix, _offset, frame_seconds in self._score_windows(waveform):
                if matrix.size == 0 or column_index is None or column_index >= matrix.shape[1]:
                    continue
                column = matrix[:, column_index]
                max_prob = max(max_prob, float(np.max(column)))
                seconds_over += float(np.count_nonzero(column >= effective_threshold)) * frame_seconds
            return VocalEventMeasurement(
                kind=kind, max_prob=max_prob, seconds_over_threshold=seconds_over
            )

        for _patch, pooled, _offset in self._score_windows_pooled(waveform):
            if column_index is None or column_index >= pooled.shape[0]:
                continue
            value = float(pooled[column_index])
            max_prob = max(max_prob, value)
            if value >= effective_threshold:
                seconds_over += self.hop_seconds
        return VocalEventMeasurement(
            kind=kind, max_prob=max_prob, seconds_over_threshold=seconds_over
        )

    def measure(self, audio_path: str, kind: str, threshold: Optional[float] = None) -> VocalEventMeasurement:
        """Return the strongest response for ``kind`` in a clip."""
        return self.measure_from_waveform(self._load_waveform(audio_path), kind, threshold)

    def has_event(
        self,
        audio_path: str,
        kind: str,
        threshold: Optional[float] = None,
        min_seconds: float = 0.1,
    ) -> Tuple[bool, VocalEventMeasurement]:
        """Verify that a clip contains audible ``kind`` vocalisation."""
        measurement = self.measure(audio_path, kind, threshold)
        effective_threshold = float(threshold if threshold is not None else self._threshold_for(kind))
        ok = (
            measurement.max_prob >= effective_threshold
            and measurement.seconds_over_threshold >= min_seconds
        )
        return ok, measurement


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear resampling fallback (avoids the librosa dependency)."""
    data = np.asarray(waveform, dtype=np.float32)
    if orig_sr == target_sr or data.size == 0:
        return data
    duration = data.size / float(orig_sr)
    target_count = max(1, int(round(duration * target_sr)))
    source_positions = np.linspace(0.0, data.size - 1, num=target_count, dtype=np.float64)
    return np.interp(source_positions, np.arange(data.size), data).astype(np.float32)
