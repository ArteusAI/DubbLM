"""Multi-sample speaker reference profiles for robust voice matching.

The Gemini API never receives reference audio directly: DubbLM uses it to
match a speaker to a prebuilt voice and to validate synthesized audio. A
single clip can be unrepresentative (shouting, whispering, an emotional
outlier), so this module builds a profile from several clips taken from
different parts of the original track, ranks them by similarity to the
speaker embedding centroid (the "average voice") and keeps the most typical
ones. The same profile also produces an averaged reference WAV that can be
reused by other providers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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


PCM_SAMPLE_WIDTH = 2
PCM_CHANNELS = 1
PCM_FRAME_RATE = 24000
CLIP_GAP_MS = 80


@dataclass
class SpeakerReferenceProfile:
    """Selected clips + embeddings that describe a speaker's average voice."""

    speaker: str
    centroid: Optional[np.ndarray]
    clip_embeddings: List[np.ndarray] = field(default_factory=list)
    reference_path: Optional[str] = None
    clip_count: int = 0
    total_seconds: float = 0.0
    clip_ranges: List[Tuple[float, float]] = field(default_factory=list)
    source: str = "reference_audio"


def _unit(vector: np.ndarray) -> Optional[np.ndarray]:
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return None
    return array / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings (0.0 when degenerate)."""
    ua = _unit(a)
    ub = _unit(b)
    if ua is None or ub is None:
        return 0.0
    return float(np.dot(ua, ub))


def _mean_embedding(embeddings: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    units = [u for u in (_unit(e) for e in embeddings) if u is not None]
    if not units:
        return None
    return np.mean(np.stack(units, axis=0), axis=0)


class SpeakerReferenceBuilder:
    """Selects representative clips and builds an averaged reference."""

    def __init__(
        self,
        audio_embedder: Optional[Any] = None,
        *,
        enabled: bool = True,
        min_clip_seconds: float = 1.2,
        max_clip_seconds: float = 15.0,
        target_seconds: float = 16.0,
        max_clips: int = 8,
        min_clips: int = 3,
        min_rms_dbfs: float = -45.0,
        silence_thresh_db: float = -40.0,
        min_silence_len_ms: int = 250,
    ) -> None:
        self.audio_embedder = audio_embedder
        self.enabled = enabled
        self.min_clip_seconds = float(min_clip_seconds)
        self.max_clip_seconds = float(max_clip_seconds)
        self.target_seconds = float(target_seconds)
        self.max_clips = max(1, int(max_clips))
        self.min_clips = max(1, int(min_clips))
        self.min_rms_dbfs = float(min_rms_dbfs)
        self.silence_thresh_db = float(silence_thresh_db)
        self.min_silence_len_ms = int(min_silence_len_ms)

    def is_available(self) -> bool:
        return bool(self.enabled and PYDUB_AVAILABLE and self.audio_embedder is not None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_profile(
        self,
        speaker: str,
        reference_audio_path: Optional[str],
        out_dir: Path,
        *,
        time_ranges: Optional[Sequence[Tuple[float, float]]] = None,
        original_audio_path: Optional[str] = None,
    ) -> Optional[SpeakerReferenceProfile]:
        """Build a profile from several clips of the original track.

        Preference order:
        1. ``original_audio_path`` + ``time_ranges`` (exact speaker segments
           from diarization — best source, no cross-speaker bleed);
        2. ``reference_audio_path`` (the concatenated per-speaker artifact),
           split into non-silent chunks.
        """
        if not self.is_available():
            return None

        clips: List[Dict[str, Any]] = []
        if original_audio_path and time_ranges:
            clips = self._clips_from_original(original_audio_path, list(time_ranges))
        if not clips and reference_audio_path:
            clips = self._clips_from_reference(reference_audio_path)
        if not clips:
            logger.debug("Speaker reference: no usable clips for '%s'", speaker)
            return None

        if len(clips) < self.min_clips:
            logger.debug(
                "Speaker reference: only %d usable clip(s) for '%s' (min %d); using them anyway",
                len(clips),
                speaker,
                self.min_clips,
            )

        embeddings = [clip["embedding"] for clip in clips]
        centroid = _mean_embedding(embeddings)
        if centroid is None:
            return None

        selected = self._select_typical_clips(clips, centroid)

        reference_path = None
        total_seconds = 0.0
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            reference_path = str(out_dir / f"{speaker}_avg.wav")
            total_seconds = self._export_averaged_reference(selected, reference_path)
            self._persist_profile(
                out_dir / f"{speaker}.json",
                speaker,
                centroid,
                selected,
                total_seconds,
            )
        except Exception as exc:  # pragma: no cover - filesystem safety
            logger.warning("Speaker reference: could not persist profile for '%s': %s", speaker, exc)

        profile = SpeakerReferenceProfile(
            speaker=speaker,
            centroid=centroid,
            clip_embeddings=[clip["embedding"] for clip in selected],
            reference_path=reference_path,
            clip_count=len(selected),
            total_seconds=total_seconds,
            clip_ranges=[(clip["start"], clip["end"]) for clip in selected],
            source="original" if original_audio_path and time_ranges else "reference_audio",
        )
        logger.info(
            "Speaker reference for '%s': %d clip(s), %.1fs, source=%s",
            speaker,
            profile.clip_count,
            profile.total_seconds,
            profile.source,
        )
        return profile

    # ------------------------------------------------------------------
    # Clip extraction
    # ------------------------------------------------------------------

    def _clips_from_original(
        self,
        original_audio_path: str,
        time_ranges: Sequence[Tuple[float, float]],
    ) -> List[Dict[str, Any]]:
        source_path = Path(original_audio_path)
        if not source_path.exists():
            return []
        try:
            audio = AudioSegment.from_file(source_path)
            audio = audio.set_channels(PCM_CHANNELS).set_frame_rate(PCM_FRAME_RATE).set_sample_width(PCM_SAMPLE_WIDTH)
        except Exception as exc:
            logger.warning("Speaker reference: cannot load original audio '%s': %s", original_audio_path, exc)
            return []

        clips: List[Dict[str, Any]] = []
        for start, end in time_ranges:
            if end <= start:
                continue
            clip = self._prepare_clip(audio[int(start * 1000) : int(end * 1000)], start)
            if clip is not None:
                clips.append(clip)
        return clips

    def _clips_from_reference(self, reference_audio_path: str) -> List[Dict[str, Any]]:
        source_path = Path(reference_audio_path)
        if not source_path.exists():
            return []
        try:
            audio = AudioSegment.from_file(source_path)
            audio = audio.set_channels(PCM_CHANNELS).set_frame_rate(PCM_FRAME_RATE).set_sample_width(PCM_SAMPLE_WIDTH)
        except Exception as exc:
            logger.warning("Speaker reference: cannot load reference audio '%s': %s", reference_audio_path, exc)
            return []

        threshold = self.silence_thresh_db
        try:
            if audio.dBFS > -60:
                threshold = min(-30.0, max(-50.0, audio.dBFS - 18.0))
        except Exception:
            pass

        try:
            spans = detect_nonsilent(
                audio,
                min_silence_len=self.min_silence_len_ms,
                silence_thresh=threshold,
            )
        except Exception as exc:
            logger.warning("Speaker reference: non-silent detection failed: %s", exc)
            spans = [(0, len(audio))]

        clips: List[Dict[str, Any]] = []
        for start_ms, end_ms in spans:
            clip = self._prepare_clip(audio[start_ms:end_ms], start_ms / 1000.0)
            if clip is not None:
                clips.append(clip)
        return clips

    def _prepare_clip(self, clip: "AudioSegment", start_seconds: float) -> Optional[Dict[str, Any]]:
        if clip is None or len(clip) <= 0:
            return None
        try:
            trimmed = self._trim_silence(clip)
        except Exception:
            trimmed = clip
        duration = len(trimmed) / 1000.0
        if duration < self.min_clip_seconds:
            return None
        if duration > self.max_clip_seconds:
            trimmed = trimmed[: int(self.max_clip_seconds * 1000)]
            duration = self.max_clip_seconds
        try:
            if trimmed.dBFS < self.min_rms_dbfs:
                return None
        except Exception:
            pass

        try:
            embedding = self.audio_embedder.extract_embedding(trimmed)
        except Exception as exc:
            logger.debug("Speaker reference: embedding failed for clip at %.1fs: %s", start_seconds, exc)
            return None
        if embedding is None:
            return None

        end_seconds = start_seconds + len(clip) / 1000.0
        return {
            "clip": trimmed,
            "embedding": np.asarray(embedding, dtype=np.float64),
            "start": float(start_seconds),
            "end": float(end_seconds),
        }

    def _trim_silence(self, clip: "AudioSegment") -> "AudioSegment":
        threshold = self.silence_thresh_db
        try:
            if clip.dBFS > -60:
                threshold = min(-30.0, max(-50.0, clip.dBFS - 18.0))
        except Exception:
            pass
        spans = detect_nonsilent(
            clip,
            min_silence_len=120,
            silence_thresh=threshold,
        )
        if not spans:
            return clip
        first_start, last_end = spans[0][0], spans[-1][1]
        if last_end - first_start <= 0:
            return clip
        return clip[first_start:last_end]

    # ------------------------------------------------------------------
    # Selection + export
    # ------------------------------------------------------------------

    def _select_typical_clips(
        self,
        clips: Sequence[Dict[str, Any]],
        centroid: np.ndarray,
    ) -> List[Dict[str, Any]]:
        scored = []
        for clip in clips:
            similarity = cosine_similarity(clip["embedding"], centroid)
            scored.append((similarity, clip))

        if len(scored) <= self.max_clips:
            selected = [clip for _, clip in scored]
            selected.sort(key=lambda c: c["start"])
            return selected

        first_start = min(clip["start"] for _, clip in scored)
        last_end = max(clip["end"] for _, clip in scored)
        span = max(1e-6, last_end - first_start)

        buckets: List[List[Tuple[float, Dict[str, Any]]]] = [[], [], []]
        for similarity, clip in scored:
            rel = (clip["start"] - first_start) / span
            index = 0 if rel < 1 / 3 else (1 if rel < 2 / 3 else 2)
            buckets[index].append((similarity, clip))
        for bucket in buckets:
            bucket.sort(key=lambda item: item[0], reverse=True)

        selected: List[Dict[str, Any]] = []
        total_seconds = 0.0
        bucket_order = sorted(range(len(buckets)), key=lambda i: -len(buckets[i]))
        cursor = {i: 0 for i in range(len(buckets))}
        while len(selected) < self.max_clips:
            progressed = False
            for index in bucket_order:
                bucket = buckets[index]
                if cursor[index] >= len(bucket):
                    continue
                similarity, clip = bucket[cursor[index]]
                cursor[index] += 1
                progressed = True
                selected.append(clip)
                total_seconds += len(clip["clip"]) / 1000.0
                if len(selected) >= self.max_clips or total_seconds >= self.target_seconds:
                    break
            if not progressed:
                break

        selected.sort(key=lambda c: c["start"])
        return selected

    def _export_averaged_reference(self, clips: Sequence[Dict[str, Any]], output_path: str) -> float:
        combined = AudioSegment.empty()
        for index, clip in enumerate(clips):
            if index:
                combined += AudioSegment.silent(duration=CLIP_GAP_MS, frame_rate=PCM_FRAME_RATE)
            combined += clip["clip"]
        combined = combined.set_channels(PCM_CHANNELS).set_frame_rate(PCM_FRAME_RATE).set_sample_width(PCM_SAMPLE_WIDTH)
        combined.export(output_path, format="wav")
        return len(combined) / 1000.0

    def _persist_profile(
        self,
        path: Path,
        speaker: str,
        centroid: np.ndarray,
        clips: Sequence[Dict[str, Any]],
        total_seconds: float,
    ) -> None:
        payload = {
            "speaker": speaker,
            "centroid": centroid.tolist(),
            "clip_count": len(clips),
            "total_seconds": total_seconds,
            "clip_ranges": [[clip["start"], clip["end"]] for clip in clips],
            "source": "gemini38_speaker_reference_v1",
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


def best_voice_for_profile(
    profile: SpeakerReferenceProfile,
    voice_matcher: Any,
    exclude_voices: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Pick the closest prebuilt voice for a speaker profile."""
    if profile is None:
        return None
    if voice_matcher is not None and getattr(voice_matcher, "sample_embeddings", None):
        if profile.clip_embeddings and hasattr(voice_matcher, "find_best_matching_voice_multi_segment"):
            try:
                voice = voice_matcher.find_best_matching_voice_multi_segment(
                    profile.clip_embeddings,
                    exclude_voices=list(exclude_voices) if exclude_voices else None,
                )
                if voice:
                    return voice
            except Exception as exc:
                logger.debug("Multi-segment voice matching failed: %s", exc)
        if profile.centroid is not None and hasattr(voice_matcher, "sample_embeddings"):
            best_voice = None
            best_similarity = -1.0
            excluded = {str(v).strip().lower() for v in (exclude_voices or [])}
            for voice_name, embedding in voice_matcher.sample_embeddings.items():
                if voice_name.lower() in excluded:
                    continue
                similarity = cosine_similarity(profile.centroid, np.asarray(embedding, dtype=np.float64))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_voice = voice_name
            if best_voice:
                logger.debug(
                    "Speaker reference: centroid matched '%s' (similarity=%.3f)",
                    best_voice,
                    best_similarity,
                )
                return best_voice
    return None
