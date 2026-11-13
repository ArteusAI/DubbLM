"""Build clean speaker references for stateless voice cloning.

Drama 3 accepts one reference clip per request.  A single diarization segment
is often a poor reference: it can start or end in the middle of a word, contain
another speaker talking over the target, or be an emotional outlier.  This
module builds a reference from several short clips of the same speaker, taken
from different parts of the source, and cuts every clip inside a quiet gap.

The quiet-gap rule is deliberate.  Diarization boundaries are approximate; a
cut at the boundary can shave a consonant or leave the tail of another voice at
the edge of the reference.  Cutting in the middle of a detected silence gives
the reference a guard band on both sides and keeps overlapping voices out.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.dubbing.core.log_config import get_logger

logger = get_logger(__name__)

try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    PYDUB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency handling
    AudioSegment = None  # type: ignore[assignment]
    detect_nonsilent = None  # type: ignore[assignment]
    PYDUB_AVAILABLE = False


DEFAULT_REFERENCE_DIR = "artifacts/cloned_voices"
SAFE_SPEAKER_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class VoiceCloneReferenceConfig:
    """Quality and duration limits for one voice-cloning reference."""

    min_clip_seconds: float = 2.0
    max_clip_seconds: float = 10.0
    target_seconds: float = 20.0
    min_total_seconds: float = 8.0
    max_total_seconds: float = 30.0
    max_clips: int = 5
    min_clips: int = 2
    min_silence_ms: int = 200
    boundary_search_seconds: float = 1.2
    guard_ms: int = 60
    min_speech_ratio: float = 0.45
    min_rms_dbfs: float = -42.0
    clip_gap_ms: int = 120
    timeline_buckets: int = 3
    min_gap_between_clips_seconds: float = 3.0
    sample_rate: int = 24000
    sample_width: int = 2
    channels: int = 1


@dataclass
class VoiceCloneReference:
    """A ready-to-send speaker reference."""

    speaker: str
    reference_path: str
    duration_seconds: float
    clip_ranges: List[Tuple[float, float]] = field(default_factory=list)
    source_path: str = ""
    transcript: Optional[str] = None
    cache_token: str = ""
    data_uri: Optional[str] = None

    @property
    def clip_count(self) -> int:
        return len(self.clip_ranges)


def _safe_speaker(speaker: str) -> str:
    cleaned = SAFE_SPEAKER_RE.sub("_", str(speaker or "speaker")).strip("._-")
    return cleaned or "speaker"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_ranges(
    speakers_rolls: Mapping[Tuple[float, float], str],
) -> List[Tuple[float, float, str]]:
    ranges: List[Tuple[float, float, str]] = []
    for (start, end), speaker in (speakers_rolls or {}).items():
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        speaker_name = str(speaker or "").strip()
        if not speaker_name or end_f <= start_f:
            continue
        ranges.append((start_f, end_f, speaker_name))
    ranges.sort(key=lambda item: (item[0], item[1]))
    return ranges


class VoiceCloneReferenceBuilder:
    """Select clean clips and concatenate them into a cloning reference."""

    def __init__(self, config: Optional[VoiceCloneReferenceConfig] = None) -> None:
        self.config = config or VoiceCloneReferenceConfig()

    def is_available(self) -> bool:
        return bool(PYDUB_AVAILABLE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_all(
        self,
        source_audio_path: str,
        speakers_rolls: Mapping[Tuple[float, float], str],
        *,
        out_dir: str = DEFAULT_REFERENCE_DIR,
        transcript_segments: Optional[Sequence[Dict[str, Any]]] = None,
        all_speakers_rolls: Optional[Mapping[Tuple[float, float], str]] = None,
    ) -> Dict[str, VoiceCloneReference]:
        """Build a reference for every speaker in ``speakers_rolls``.

        A speaker is omitted from the result when there is not enough clean,
        non-overlapping speech.  The caller must then fall back to a library
        voice for that speaker instead of failing the whole dub.
        """
        if not self.is_available():
            logger.warning("Voice cloning: pydub is unavailable; cannot build references")
            return {}

        source = Path(source_audio_path)
        if not source.exists():
            logger.warning("Voice cloning: source audio not found: %s", source_audio_path)
            return {}

        try:
            audio = AudioSegment.from_file(source)
            audio = (
                audio.set_channels(self.config.channels)
                .set_sample_width(self.config.sample_width)
            )
        except Exception as exc:
            logger.warning("Voice cloning: cannot load source audio '%s': %s", source_audio_path, exc)
            return {}

        ranges = _normalize_ranges(speakers_rolls)
        if not ranges:
            return {}
        all_ranges = _normalize_ranges(all_speakers_rolls or speakers_rolls)
        by_speaker: Dict[str, List[Tuple[float, float]]] = {}
        for start, end, speaker in ranges:
            by_speaker.setdefault(speaker, []).append((start, end))

        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        references: Dict[str, VoiceCloneReference] = {}
        for speaker in sorted(by_speaker):
            try:
                reference = self._build_speaker(
                    audio,
                    source,
                    speaker,
                    by_speaker[speaker],
                    all_ranges,
                    output_dir,
                    transcript_segments=transcript_segments,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Voice cloning: failed to build reference for '%s': %s", speaker, exc)
                reference = None
            if reference is not None:
                references[speaker] = reference
                logger.info(
                    "Voice cloning reference for '%s': %d clip(s), %.1fs, ranges=%s",
                    speaker,
                    reference.clip_count,
                    reference.duration_seconds,
                    ", ".join(f"{start:.1f}-{end:.1f}s" for start, end in reference.clip_ranges),
                )
            else:
                logger.warning(
                    "Voice cloning: not enough clean non-overlapping speech for '%s'; "
                    "the library voice will be used for this speaker",
                    speaker,
                )
        return references

    # ------------------------------------------------------------------
    # Per-speaker construction
    # ------------------------------------------------------------------

    def _build_speaker(
        self,
        audio: "AudioSegment",
        source: Path,
        speaker: str,
        speaker_ranges: Sequence[Tuple[float, float]],
        all_ranges: Sequence[Tuple[float, float, str]],
        output_dir: Path,
        *,
        transcript_segments: Optional[Sequence[Dict[str, Any]]],
    ) -> Optional[VoiceCloneReference]:
        token = self._cache_token(source, speaker, speaker_ranges)
        safe_speaker = _safe_speaker(speaker)
        reference_path = output_dir / f"{safe_speaker}_reference.wav"
        metadata_path = output_dir / f"{safe_speaker}_reference.json"
        cached = self._load_cached(metadata_path, reference_path, token)
        if cached is not None:
            logger.debug("Voice cloning: reusing cached reference for '%s'", speaker)
            return cached

        candidates = self._candidate_clips(
            audio,
            speaker,
            speaker_ranges,
            all_ranges,
        )
        selected = self._select_clips(candidates, speaker_ranges)
        if not selected:
            return None

        total_seconds = sum(clip["duration"] for clip in selected)
        if len(selected) < self.config.min_clips:
            logger.debug(
                "Voice cloning: only %d clean clip(s) for '%s' (min %d)",
                len(selected),
                speaker,
                self.config.min_clips,
            )
        if total_seconds < self.config.min_total_seconds:
            logger.debug(
                "Voice cloning: only %.1fs of clean speech for '%s' (min %.1fs)",
                total_seconds,
                speaker,
                self.config.min_total_seconds,
            )
            return None

        combined = self._export_reference(selected, reference_path)
        transcript = self._build_transcript(selected, speaker, transcript_segments)
        reference = VoiceCloneReference(
            speaker=speaker,
            reference_path=str(reference_path),
            duration_seconds=len(combined) / 1000.0,
            clip_ranges=[(clip["start"], clip["end"]) for clip in selected],
            source_path=str(source),
            transcript=transcript,
            cache_token=token,
        )
        self._persist_metadata(metadata_path, reference, selected)
        return reference

    def _candidate_clips(
        self,
        audio: "AudioSegment",
        speaker: str,
        speaker_ranges: Sequence[Tuple[float, float]],
        all_ranges: Sequence[Tuple[float, float, str]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        source_duration = len(audio) / 1000.0
        search = max(0.2, float(self.config.boundary_search_seconds))
        guard = max(0.01, self.config.guard_ms / 1000.0)
        min_gap = max(guard * 2.0, self.config.min_silence_ms / 1000.0)

        for range_start, range_end in sorted(speaker_ranges):
            window_start = max(0.0, range_start - search)
            window_end = min(source_duration, range_end + search)
            if window_end - window_start <= 0.1:
                continue
            window = audio[int(window_start * 1000) : int(window_end * 1000)]
            if len(window) <= 0:
                continue

            try:
                threshold = self._silence_threshold(window)
                spans = detect_nonsilent(
                    window,
                    min_silence_len=max(80, self.config.min_silence_ms),
                    silence_thresh=threshold,
                )
            except Exception as exc:
                logger.debug("Voice cloning: silence detection failed at %.1fs: %s", range_start, exc)
                continue
            if not spans:
                continue

            absolute_spans = [
                (window_start + start_ms / 1000.0, window_start + end_ms / 1000.0)
                for start_ms, end_ms in spans
            ]
            # A long diarization turn (a monologue) contains many speech
            # islands.  Consider every island that overlaps the turn, not just
            # the longest one: otherwise a speaker with one long turn would get
            # a single short reference.
            for island_index, (speech_start, speech_end) in enumerate(absolute_spans):
                overlap = max(0.0, min(speech_end, range_end) - max(speech_start, range_start))
                if overlap <= 0.05:
                    continue

                left_boundary = (
                    absolute_spans[island_index - 1][1]
                    if island_index > 0
                    else window_start
                )
                right_boundary = (
                    absolute_spans[island_index + 1][0]
                    if island_index + 1 < len(absolute_spans)
                    else window_end
                )
                left_gap = speech_start - left_boundary
                right_gap = right_boundary - speech_end
                if left_gap < min_gap or right_gap < min_gap:
                    continue

                cut_start = (left_boundary + speech_start) / 2.0
                cut_end = (speech_end + right_boundary) / 2.0
                if speech_start - cut_start < guard or cut_end - speech_end < guard:
                    continue

                duration = cut_end - cut_start
                if (
                    duration < self.config.min_clip_seconds
                    or duration > self.config.max_clip_seconds
                ):
                    continue

                # The cut interval and its guard bands must not contain
                # another speaker's diarized turn.  This is the main
                # protection against cross-talk and overlapping dialogue in
                # the reference.
                if self._overlaps_other_speaker(
                    cut_start,
                    cut_end,
                    guard,
                    speaker,
                    all_ranges,
                ):
                    continue

                clip = audio[int(cut_start * 1000) : int(cut_end * 1000)]
                if len(clip) <= 0:
                    continue
                try:
                    if clip.dBFS < self.config.min_rms_dbfs:
                        continue
                except Exception:
                    pass

                speech_seconds = self._speech_seconds(
                    absolute_spans,
                    cut_start,
                    cut_end,
                )
                speech_ratio = speech_seconds / duration if duration > 0 else 0.0
                if speech_ratio < self.config.min_speech_ratio:
                    continue

                score = self._score_clip(duration, clip.dBFS, speech_ratio)
                candidates.append(
                    {
                        "clip": clip.fade_in(8).fade_out(8),
                        "start": cut_start,
                        "end": cut_end,
                        "duration": duration,
                        "score": score,
                        "speech_ratio": speech_ratio,
                    }
                )
        return candidates

    def _select_clips(
        self,
        candidates: Sequence[Dict[str, Any]],
        speaker_ranges: Sequence[Tuple[float, float]],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda clip: clip["start"])
        if len(ordered) <= self.config.max_clips:
            selected = []
            for clip in ordered:
                if not self._too_close(clip, selected):
                    selected.append(clip)
        else:
            first = min(start for start, _ in speaker_ranges)
            last = max(end for _, end in speaker_ranges)
            span = max(1e-6, last - first)
            bucket_count = max(1, int(self.config.timeline_buckets))
            buckets: List[List[Dict[str, Any]]] = [[] for _ in range(bucket_count)]
            for clip in ordered:
                position = _clamp((clip["start"] - first) / span, 0.0, 0.999999)
                index = min(bucket_count - 1, int(position * bucket_count))
                buckets[index].append(clip)
            for bucket in buckets:
                bucket.sort(key=lambda clip: clip["score"], reverse=True)

            selected = []
            total_seconds = 0.0
            cursors = [0] * bucket_count
            while len(selected) < self.config.max_clips:
                progressed = False
                for index in range(bucket_count):
                    bucket = buckets[index]
                    while cursors[index] < len(bucket):
                        clip = bucket[cursors[index]]
                        cursors[index] += 1
                        if self._too_close(clip, selected):
                            continue
                        selected.append(clip)
                        total_seconds += clip["duration"]
                        progressed = True
                        break
                    if (
                        len(selected) >= self.config.max_clips
                        or total_seconds >= self.config.target_seconds
                    ):
                        break
                if not progressed:
                    break
                if total_seconds >= self.config.target_seconds and len(selected) >= self.config.min_clips:
                    break

        selected.sort(key=lambda clip: clip["start"])
        total = sum(clip["duration"] for clip in selected)
        if total > self.config.max_total_seconds:
            limited: List[Dict[str, Any]] = []
            running = 0.0
            for clip in selected:
                if running + clip["duration"] > self.config.max_total_seconds and limited:
                    break
                limited.append(clip)
                running += clip["duration"]
            selected = limited
        return selected

    def _too_close(
        self,
        clip: Dict[str, Any],
        selected: Sequence[Dict[str, Any]],
    ) -> bool:
        for chosen in selected:
            if abs(clip["start"] - chosen["start"]) < self.config.min_gap_between_clips_seconds:
                return True
        return False

    def _export_reference(
        self,
        clips: Sequence[Dict[str, Any]],
        output_path: Path,
    ) -> "AudioSegment":
        combined = AudioSegment.empty()
        for index, clip in enumerate(clips):
            if index:
                combined += AudioSegment.silent(
                    duration=self.config.clip_gap_ms,
                    frame_rate=self.config.sample_rate,
                )
            combined += clip["clip"]
        combined = (
            combined.set_channels(self.config.channels)
            .set_sample_width(self.config.sample_width)
            .set_frame_rate(self.config.sample_rate)
        )
        combined.export(str(output_path), format="wav")
        return combined

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _silence_threshold(window: "AudioSegment") -> float:
        threshold = -40.0
        try:
            if window.dBFS > -60:
                threshold = min(-30.0, max(-50.0, window.dBFS - 18.0))
        except Exception:
            pass
        return threshold

    @staticmethod
    def _best_island(
        spans: Sequence[Tuple[float, float]],
        range_start: float,
        range_end: float,
    ) -> Optional[int]:
        best_index = None
        best_overlap = 0.0
        for index, (start, end) in enumerate(spans):
            overlap = max(0.0, min(end, range_end) - max(start, range_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index is not None and best_overlap > 0.05:
            return best_index
        # Diarization can be a few hundred milliseconds off; accept the
        # nearest island only when it is very close to the reported turn.
        nearest_index = None
        nearest_distance = 0.25
        for index, (start, end) in enumerate(spans):
            distance = min(abs(start - range_start), abs(end - range_end))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index
        return nearest_index

    @staticmethod
    def _overlaps_other_speaker(
        start: float,
        end: float,
        guard: float,
        speaker: str,
        all_ranges: Sequence[Tuple[float, float, str]],
    ) -> bool:
        expanded_start = start - guard
        expanded_end = end + guard
        for other_start, other_end, other_speaker in all_ranges:
            if other_speaker == speaker:
                continue
            if other_end > expanded_start and other_start < expanded_end:
                return True
        return False

    @staticmethod
    def _speech_seconds(
        spans: Sequence[Tuple[float, float]],
        start: float,
        end: float,
    ) -> float:
        total = 0.0
        for span_start, span_end in spans:
            overlap = max(0.0, min(span_end, end) - max(span_start, start))
            total += overlap
        return total

    def _score_clip(self, duration: float, dbfs: float, speech_ratio: float) -> float:
        ideal = max(self.config.min_clip_seconds, self.config.target_seconds / 4.0)
        duration_score = 1.0 - min(1.0, abs(duration - ideal) / max(ideal, 0.1))
        rms_score = _clamp((dbfs + 50.0) / 35.0, 0.0, 1.0)
        return 0.45 * duration_score + 0.25 * rms_score + 0.30 * speech_ratio

    def _cache_token(
        self,
        source: Path,
        speaker: str,
        speaker_ranges: Sequence[Tuple[float, float]],
    ) -> str:
        try:
            stat = source.stat()
            source_info = {
                "path": str(source.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        except Exception:
            source_info = {"path": str(source)}
        payload = {
            "source": source_info,
            "speaker": speaker,
            "ranges": [[round(start, 3), round(end, 3)] for start, end in speaker_ranges],
            "config": asdict(self.config),
        }
        return hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def _load_cached(
        self,
        metadata_path: Path,
        reference_path: Path,
        token: str,
    ) -> Optional[VoiceCloneReference]:
        if not metadata_path.exists() or not reference_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if payload.get("cache_token") != token:
                return None
            return VoiceCloneReference(
                speaker=payload["speaker"],
                reference_path=str(reference_path),
                duration_seconds=float(payload.get("duration_seconds") or 0.0),
                clip_ranges=[
                    (float(item[0]), float(item[1]))
                    for item in payload.get("clip_ranges") or []
                ],
                source_path=payload.get("source_path") or "",
                transcript=payload.get("transcript"),
                cache_token=token,
            )
        except Exception:
            return None

    def _persist_metadata(
        self,
        metadata_path: Path,
        reference: VoiceCloneReference,
        clips: Sequence[Dict[str, Any]],
    ) -> None:
        payload = {
            "speaker": reference.speaker,
            "reference_path": reference.reference_path,
            "duration_seconds": reference.duration_seconds,
            "clip_ranges": [[start, end] for start, end in reference.clip_ranges],
            "source_path": reference.source_path,
            "transcript": reference.transcript,
            "cache_token": reference.cache_token,
            "clips": [
                {
                    "start": clip["start"],
                    "end": clip["end"],
                    "duration": clip["duration"],
                    "score": clip["score"],
                    "speech_ratio": clip["speech_ratio"],
                }
                for clip in clips
            ],
        }
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _build_transcript(
        clips: Sequence[Dict[str, Any]],
        speaker: str,
        transcript_segments: Optional[Sequence[Dict[str, Any]]],
    ) -> Optional[str]:
        if not transcript_segments:
            return None
        parts: List[str] = []
        for clip in clips:
            start = clip["start"]
            end = clip["end"]
            texts: List[str] = []
            for segment in transcript_segments:
                if str(segment.get("speaker") or "") != speaker:
                    continue
                try:
                    seg_start = float(segment.get("start"))
                    seg_end = float(segment.get("end"))
                except (TypeError, ValueError):
                    continue
                midpoint = (seg_start + seg_end) / 2.0
                if start <= midpoint <= end:
                    text = str(
                        segment.get("text")
                        or segment.get("original_text")
                        or segment.get("originalText")
                        or ""
                    ).strip()
                    if text:
                        texts.append(text)
            if not texts:
                return None
            parts.append(" ".join(texts))
        return " ".join(parts).strip() or None
