"""Loudness based detection of screams and shouted lines.

The AST AudioSet classifier does not recognise shouted dialogue: a short
shout inside an analysis window is averaged away and the class stays
"Speech" (verified on a real trailer - even a 2 s window centred on a shout
never scores "Screaming").  Trailer screams are usually just very loud
speech, so they are found from the loudness envelope of the (separated)
vocals track instead.

The threshold is relative to the material: the loud speech level of the file
(90th percentile of the frame levels) plus a margin.  That keeps emphatic
dialogue out while catching the shouted lines, and it works regardless of
the recording level.

The detector returns plain intervals that the existing vocal event
preservation machinery can splice back into the dubbed track.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.log_config import get_logger

logger = get_logger(__name__)

DETECTOR_VERSION = "loudness-v1"
DEFAULT_FRAME_SECONDS = 0.05
# Margin above the loud speech level of the file.
DEFAULT_MARGIN_DB = 4.0
# Absolute floor: a very quiet recording must not trigger on noise.
DEFAULT_FLOOR_DB = -25.0
DEFAULT_MIN_SECONDS = 0.1
DEFAULT_MERGE_GAP_SECONDS = 0.3
DEFAULT_REFERENCE_PERCENTILE = 90.0
# Frames this far below the peak are considered "normal" material and define
# the loud speech level.
DEFAULT_PEAK_EXCLUSION_DB = 6.0
DEFAULT_KIND = "scream"

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:  # pragma: no cover - pydub is a hard dependency elsewhere
    PYDUB_AVAILABLE = False


@dataclass
class LoudnessEvent:
    """A loud interval (scream / shout) in the source timeline."""

    kind: str
    start: float
    end: float
    score: float


def _overlaps(ranges: Sequence[Tuple[float, float]], start: float, end: float) -> bool:
    for range_start, range_end in ranges:
        if min(range_end, end) - max(range_start, start) > 0.05:
            return True
    return False


class LoudnessScreamDetector:
    """Find loud passages (screams, shouts) in an audio file.

    ``margin_db`` is added to the loud speech level of the file (90th
    percentile of the frame levels) to obtain the detection threshold.
    """

    def __init__(
        self,
        *,
        margin_db: float = DEFAULT_MARGIN_DB,
        floor_db: float = DEFAULT_FLOOR_DB,
        min_seconds: float = DEFAULT_MIN_SECONDS,
        merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
        frame_seconds: float = DEFAULT_FRAME_SECONDS,
        reference_percentile: float = DEFAULT_REFERENCE_PERCENTILE,
        peak_exclusion_db: float = DEFAULT_PEAK_EXCLUSION_DB,
        kind: str = DEFAULT_KIND,
    ):
        self.margin_db = float(margin_db)
        self.floor_db = float(floor_db)
        self.min_seconds = float(min_seconds)
        self.merge_gap_seconds = float(merge_gap_seconds)
        self.frame_seconds = float(frame_seconds)
        self.reference_percentile = float(reference_percentile)
        self.peak_exclusion_db = float(peak_exclusion_db)
        self.kind = kind

    def _frame_levels_db(self, audio_path: str) -> Optional[Tuple[np.ndarray, float]]:
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as exc:
            logger.warning(f"Loudness detector could not read {audio_path}: {exc}")
            return None

        try:
            # Decode in the native sample rate: the RMS of a short frame is
            # rate independent and resampling would only add conversion loss.
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            channels = max(1, int(audio.channels))
            if channels > 1:
                samples = samples[: (samples.size // channels) * channels]
                samples = samples.reshape(-1, channels).mean(axis=1)
            samples = samples * (1.0 / float(1 << (8 * audio.sample_width - 1)))
        except Exception as exc:
            logger.warning(f"Loudness detector could not decode {audio_path}: {exc}")
            return None

        if samples.size == 0:
            return None

        frame_rate = float(audio.frame_rate or 16000)
        frame_size = max(1, int(self.frame_seconds * frame_rate))
        frames = samples.size // frame_size
        if frames <= 0:
            return None

        framed = samples[: frames * frame_size].reshape(frames, frame_size)
        rms = np.sqrt(np.maximum((framed.astype(np.float64) ** 2).mean(axis=1), 1e-12))
        return 20.0 * np.log10(rms), frame_rate

    def _reference_level_db(self, levels_db: np.ndarray) -> float:
        """Loud speech level of the material, used as the reference.

        The reference is taken from everything that is *not* a peak, so a
        recording that is dominated by loud passages does not raise its own
        detection threshold.
        """
        if levels_db.size == 0:
            return self.floor_db
        peak = float(levels_db.max())
        non_peak = levels_db[levels_db <= peak - self.peak_exclusion_db]
        if non_peak.size < max(8, levels_db.size // 20):
            non_peak = levels_db
        return float(np.percentile(non_peak, self.reference_percentile))

    def detect(
        self,
        audio_path: str,
        exclude_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> List[LoudnessEvent]:
        """Return the loud intervals of ``audio_path``."""
        if not PYDUB_AVAILABLE:
            logger.warning("Loudness detector unavailable: pydub is not installed")
            return []

        measured = self._frame_levels_db(audio_path)
        if measured is None:
            return []
        levels_db, frame_rate = measured

        reference = self._reference_level_db(levels_db)
        threshold = max(self.floor_db, reference + self.margin_db)

        loud = np.where(levels_db >= threshold)[0]
        if loud.size == 0:
            logger.debug(
                "Loudness detector: nothing above %.1f dBFS (reference %.1f dBFS)",
                threshold,
                reference,
            )
            return []

        gap_frames = max(1, int(round(self.merge_gap_seconds / self.frame_seconds)))
        groups: List[List[int]] = []
        for index in loud:
            if groups and index - groups[-1][-1] <= gap_frames:
                groups[-1].append(index)
            else:
                groups.append([index])

        events: List[LoudnessEvent] = []
        for group in groups:
            start = group[0] * self.frame_seconds
            end = (group[-1] + 1) * self.frame_seconds
            # Epsilon: the float multiplication can land just below the limit.
            if end - start + 1e-6 < self.min_seconds:
                continue
            if exclude_ranges and _overlaps(exclude_ranges, start, end):
                continue
            peak = float(levels_db[group].max())
            events.append(
                LoudnessEvent(
                    kind=self.kind,
                    start=start,
                    end=end,
                    score=round(peak, 3),
                )
            )

        if events:
            logger.info(
                "Loudness detector found %d loud passage(s) (threshold %.1f dBFS, "
                "loud speech %.1f dBFS): %s",
                len(events),
                threshold,
                reference,
                ", ".join(f"{event.start:.1f}-{event.end:.1f}s" for event in events[:8]),
            )
        return events
