"""OpenRouter speech-to-text + diarization.

The default model is ``microsoft/mai-transcribe-2`` served through OpenRouter's
Azure endpoint.  It returns word-level speaker labels; the transcriber rebuilds
dubbing segments from those words and reconciles local speaker ids across
chunks with Resemblyzer voice embeddings.

OpenRouter does not expose a ``maxSpeakers`` option for this model (Azure
rejects it), so the speaker count is controlled locally: when the project
specifies ``speakers_expected`` the local speaker clusters are merged down to
that count, otherwise similar clusters are merged by a cosine threshold.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.transcription.transcription_interface import BaseTranscriber
from src.dubbing.core.log_config import get_logger

logger = get_logger(__name__)

try:
    from pydub import AudioSegment
    from pydub.silence import detect_silence

    PYDUB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    AudioSegment = None  # type: ignore[assignment]
    detect_silence = None  # type: ignore[assignment]
    PYDUB_AVAILABLE = False

DEFAULT_MODEL = "microsoft/mai-transcribe-2"
DEFAULT_FALLBACK_MODELS: Tuple[str, ...] = (
    "elevenlabs/scribe-v2",
    "deepgram/nova-3",
)
DEFAULT_CHUNK_SECONDS = 600.0
DEFAULT_OVERLAP_SECONDS = 15.0
MIN_CHUNK_SECONDS = 30.0
SPEAKER_MERGE_THRESHOLD = 0.78
MIN_EMBEDDING_CLIP_SECONDS = 1.0
MAX_EMBEDDING_SECONDS_PER_SPEAKER = 20.0
MAX_EMBEDDING_CLIPS_PER_SPEAKER = 6
SILENCE_BOUNDARY_SEARCH_SECONDS = 8.0
SILENCE_MIN_LEN_MS = 250

_TOKEN_RE = re.compile(r"[^\w'’]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_PUNCT_BEFORE_RE = re.compile(r"\s+([,.;:!?…])")
_SENTENCE_END_RE = re.compile(r"[.!?…]['\"»]?$")


def _normalize_tokens(text: str) -> List[str]:
    value = str(text or "").lower()
    value = _TOKEN_RE.sub(" ", value)
    return [token for token in value.split() if token]


def _join_words(words: Sequence[Dict[str, Any]]) -> str:
    text = " ".join(str(word.get("word") or "").strip() for word in words)
    text = _SPACE_RE.sub(" ", text).strip()
    return _PUNCT_BEFORE_RE.sub(r"\1", text)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _average_embeddings(embeddings: Sequence[np.ndarray], weights: Sequence[float]) -> Optional[np.ndarray]:
    if not embeddings:
        return None
    stacked = np.stack([np.asarray(item, dtype=np.float64).reshape(-1) for item in embeddings])
    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weight_array.size != stacked.shape[0] or float(weight_array.sum()) <= 0:
        weight_array = np.ones(stacked.shape[0], dtype=np.float64)
    weight_array = weight_array / weight_array.sum()
    result = np.sum(stacked * weight_array[:, None], axis=0)
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 0 else result


class _SpeakerCluster:
    """One global speaker assembled from local per-chunk labels."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.embeddings: List[np.ndarray] = []
        self.weights: List[float] = []
        self.members: List[Tuple[int, str]] = []
        self.ranges: List[Tuple[float, float]] = []

    @property
    def embedding(self) -> Optional[np.ndarray]:
        return _average_embeddings(self.embeddings, self.weights)

    def add(
        self,
        chunk_index: int,
        local_speaker: str,
        embedding: Optional[np.ndarray],
        weight: float,
        ranges: Sequence[Tuple[float, float]],
    ) -> None:
        self.members.append((chunk_index, local_speaker))
        if embedding is not None:
            self.embeddings.append(np.asarray(embedding, dtype=np.float64).reshape(-1))
            self.weights.append(max(0.1, float(weight or 1.0)))
        self.ranges.extend((float(start), float(end)) for start, end in ranges)


class OpenRouterTranscriber(BaseTranscriber):
    """Transcription and diarization through OpenRouter."""

    def __init__(
        self,
        source_language: str,
        device: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        fallback_models: Optional[Sequence[str]] = None,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
        speakers_expected: Optional[int] = None,
        cache_manager: Optional[Any] = None,
        cost_tracker: Optional[Any] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Values may arrive with the config-file names through the factory.
        model = str(kwargs.pop("openrouter_stt_model", model) or model)
        fallback_models = kwargs.pop("openrouter_stt_fallback_models", fallback_models)
        chunk_seconds = float(kwargs.pop("openrouter_stt_chunk_seconds", chunk_seconds) or chunk_seconds)
        overlap_seconds = float(
            kwargs.pop("openrouter_stt_overlap_seconds", overlap_seconds) or overlap_seconds
        )
        super().__init__(source_language, device, **kwargs)
        self.model = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        if isinstance(fallback_models, str):
            raw_fallbacks = fallback_models.strip()
            if raw_fallbacks.startswith("["):
                try:
                    fallback_models = json.loads(raw_fallbacks)
                except Exception:
                    fallback_models = [item.strip() for item in raw_fallbacks.strip("[]").split(",")]
            else:
                fallback_models = [item.strip() for item in raw_fallbacks.split(",")]
        fallback = [str(item).strip() for item in (fallback_models or DEFAULT_FALLBACK_MODELS) if str(item).strip()]
        self.fallback_models = [item for item in fallback if item and item != self.model]
        self.chunk_seconds = max(MIN_CHUNK_SECONDS, float(chunk_seconds or DEFAULT_CHUNK_SECONDS))
        self.overlap_seconds = max(0.0, float(overlap_seconds or 0.0))
        self.speakers_expected = int(speakers_expected) if speakers_expected else None
        self.cache_manager = cache_manager
        self.cost_tracker = cost_tracker
        self.api_key = api_key
        self._embedder: Optional[Any] = None
        self._embedder_checked = False
        self._models_tried: List[str] = []

    @property
    def name(self) -> str:
        return f"OpenRouter ({self.model})"

    @property
    def step_name(self) -> str:
        return "openrouter_diarization_transcription"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diarize_and_transcribe(
        self,
        audio_file: str,
        cache_key: Optional[str] = None,
        use_cache: bool = True,
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]]]:
        if not audio_file:
            raise ValueError("audio_file is required")
        if str(audio_file).startswith(("http://", "https://")):
            raise ValueError("OpenRouterTranscriber requires a local audio file")

        model_signature = "|".join([self.model, *self.fallback_models])
        if cache_key is None:
            cache_key = self._generate_cache_key(
                audio_file,
                (
                    f"_or1_{model_signature}_c{int(self.chunk_seconds)}_"
                    f"o{int(self.overlap_seconds)}_s{self.speakers_expected or 0}"
                ),
            )

        if use_cache and self.cache_manager and self.cache_manager.cache_exists(self.step_name, cache_key):
            logger.info("Loading OpenRouter transcription from cache (model=%s)", self.model)
            cached = self.cache_manager.load_from_cache(self.step_name, cache_key)
            self.debug_data["diarization"] = cached.get("diarization") or {}
            self.debug_data["transcription"] = cached.get("transcription") or []
            return cached["diarization"], cached["transcription"]

        logger.info("Running OpenRouter transcription with %s on %s", self.model, audio_file)
        temp_dir = tempfile.mkdtemp(prefix="openrouter_stt_")
        converted_path: Optional[str] = None
        try:
            converted_path = self._ensure_mp3(audio_file, temp_dir)
            duration = self._audio_duration(converted_path)
            chunks = self._build_chunks(converted_path, duration, temp_dir)
            logger.info(
                "OpenRouter STT: %.1fs audio in %d chunk(s) (%.0fs + %.0fs overlap)",
                duration,
                len(chunks),
                self.chunk_seconds,
                self.overlap_seconds,
            )

            chunk_results: List[Dict[str, Any]] = []
            total_cost = 0.0
            total_seconds = 0.0
            for index, (chunk_start, chunk_end, chunk_path) in enumerate(chunks):
                result = self._transcribe_chunk_with_fallback(chunk_path, index, len(chunks))
                words = self._normalise_words(result.get("words") or [], chunk_start)
                segments = result.get("segments") or []
                if not words and segments:
                    words = self._words_from_segments(segments, chunk_start)
                if not words:
                    logger.warning("OpenRouter STT chunk %d returned no words", index)
                usage = result.get("usage") or {}
                try:
                    total_cost += float(usage.get("cost") or 0.0)
                except (TypeError, ValueError):
                    pass
                try:
                    total_seconds += float(usage.get("seconds") or (chunk_end - chunk_start))
                except (TypeError, ValueError):
                    total_seconds += max(0.0, chunk_end - chunk_start)
                chunk_results.append(
                    {
                        "index": index,
                        "start": chunk_start,
                        "end": chunk_end,
                        "path": chunk_path,
                        "words": words,
                    }
                )

            speaker_map = self._resolve_global_speakers(chunk_results)
            merged_words = self._merge_chunk_words(chunk_results, speaker_map)
            transcription = self._build_segments(merged_words)
            if not transcription:
                raise RuntimeError("OpenRouter transcription returned no usable segments")

            speakers_rolls = {
                (float(segment["start"]), float(segment["end"])): str(segment["speaker"])
                for segment in transcription
            }
            self.debug_data["diarization"] = speakers_rolls
            self.debug_data["transcription"] = transcription
            self.debug_data["openrouter"] = {
                "model": self.model,
                "fallback_models": self.fallback_models,
                "models_used": list(self._models_tried),
                "cost": total_cost,
                "seconds": total_seconds,
                "chunks": len(chunks),
            }

            if use_cache and self.cache_manager:
                self.cache_manager.save_to_cache(
                    self.step_name,
                    cache_key,
                    {"diarization": speakers_rolls, "transcription": transcription},
                )

            if self.cost_tracker:
                seconds = total_seconds or duration
                try:
                    self.cost_tracker.add_transcription_usage(
                        "openrouter",
                        float(seconds),
                        model=self.model,
                        category="primary_transcription",
                        speaker_diarization=True,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Could not record OpenRouter transcription cost: %s", exc)

            logger.info(
                "OpenRouter transcription completed: %d segment(s), %d speaker(s), %.2fs / $%.4f",
                len(transcription),
                len({segment["speaker"] for segment in transcription}),
                total_seconds,
                total_cost,
            )
            return speakers_rolls, transcription
        finally:
            self._cleanup_temp_dir(temp_dir, audio_file, converted_path)

    # ------------------------------------------------------------------
    # Audio preparation
    # ------------------------------------------------------------------

    def _ensure_mp3(self, audio_file: str, temp_dir: str) -> str:
        source = Path(audio_file)
        if not source.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        if source.suffix.lower() == ".mp3":
            return str(source)
        output = Path(temp_dir) / "input.mp3"
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "64k",
            str(output),
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 or not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(
                f"ffmpeg could not convert audio for OpenRouter STT: {(result.stderr or '').strip()[:500]}"
            )
        return str(output)

    @staticmethod
    def _audio_duration(audio_path: str) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            audio_path,
        ]
        try:
            value = subprocess.check_output(command, stderr=subprocess.PIPE, text=True).strip()
            return max(0.0, float(value))
        except Exception:
            if PYDUB_AVAILABLE:
                try:
                    return max(0.0, float(len(AudioSegment.from_file(audio_path)) / 1000.0))
                except Exception:
                    pass
        return 0.0

    def _build_chunks(self, audio_path: str, duration: float, temp_dir: str) -> List[Tuple[float, float, str]]:
        if duration <= 0:
            raise RuntimeError("Could not determine audio duration")
        chunks: List[Tuple[float, float, str]] = []
        start = 0.0
        index = 0
        while start < duration - 0.25:
            target_end = min(duration, start + self.chunk_seconds)
            end = target_end
            if target_end < duration - 1.0 and self.overlap_seconds > 0:
                end = self._snap_boundary(audio_path, target_end, duration)
            actual_end = min(duration, end + self.overlap_seconds if end < duration else end)
            chunk_path = str(Path(temp_dir) / f"chunk_{index:03d}.mp3")
            self._cut_audio(audio_path, start, actual_end - start, chunk_path)
            chunks.append((start, actual_end, chunk_path))
            if actual_end >= duration - 0.25:
                break
            next_start = end
            if next_start <= start + 0.1:
                next_start = min(duration, start + self.chunk_seconds)
            start = next_start
            index += 1
        if not chunks:
            chunk_path = str(Path(temp_dir) / "chunk_000.mp3")
            self._cut_audio(audio_path, 0.0, duration, chunk_path)
            chunks.append((0.0, duration, chunk_path))
        return chunks

    def _snap_boundary(self, audio_path: str, target: float, duration: float) -> float:
        """Move a chunk boundary to the middle of a nearby silence."""
        if not PYDUB_AVAILABLE:
            return target
        search = min(SILENCE_BOUNDARY_SEARCH_SECONDS, max(1.0, target / 2.0))
        window_start = max(0.0, target - search)
        window_duration = min(duration - window_start, search * 2.0)
        if window_duration <= 0.5:
            return target
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{window_start:.3f}",
            "-t",
            f"{window_duration:.3f}",
            "-i",
            audio_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            "pipe:1",
        ]
        try:
            payload = subprocess.check_output(command, stderr=subprocess.PIPE)
            window = AudioSegment.from_file(io.BytesIO(payload), format="wav")
            silences = detect_silence(
                window,
                min_silence_len=SILENCE_MIN_LEN_MS,
                silence_thresh=max(-50.0, window.dBFS - 18.0),
            )
        except Exception:
            return target
        candidates = [
            window_start + (start_ms + end_ms) / 2000.0
            for start_ms, end_ms in silences
            if abs((window_start + (start_ms + end_ms) / 2000.0) - target) <= search
        ]
        if not candidates:
            return target
        return min(candidates, key=lambda value: abs(value - target))

    @staticmethod
    def _cut_audio(audio_path: str, start: float, duration: float, output_path: str) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{max(0.1, duration):.3f}",
            "-i",
            audio_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "64k",
            output_path,
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(
                f"ffmpeg could not cut audio chunk: {(result.stderr or '').strip()[:500]}"
            )

    # ------------------------------------------------------------------
    # OpenRouter calls
    # ------------------------------------------------------------------

    def _transcribe_chunk_with_fallback(self, chunk_path: str, index: int, total: int) -> Dict[str, Any]:
        from src.utils.openrouter_client import OpenRouterError, transcribe_audio

        models = [self.model, *self.fallback_models]
        last_error: Optional[BaseException] = None
        audio_bytes = Path(chunk_path).read_bytes()
        for model in models:
            try:
                logger.info(
                    "OpenRouter STT chunk %d/%d: %s (%.1f MB)",
                    index + 1,
                    total,
                    model,
                    len(audio_bytes) / 1e6,
                )
                result = transcribe_audio(
                    audio_bytes,
                    "mp3",
                    model=model,
                    language=self.source_language,
                    response_format="verbose_json",
                    timestamp_granularities=("segment", "word"),
                    provider_options={"azure": {"diarization": {"enabled": True}}}
                    if model == self.model
                    else None,
                    api_key=self.api_key,
                )
                words = result.get("words") or []
                segments = result.get("segments") or []
                if not words and not segments:
                    raise OpenRouterError(f"{model} returned an empty transcript")
                self._models_tried.append(model)
                return result
            except Exception as exc:
                last_error = exc
                logger.warning("OpenRouter STT model %s failed on chunk %d: %s", model, index, exc)
        raise RuntimeError(f"All OpenRouter STT models failed: {last_error}")

    @staticmethod
    def _normalise_words(words: Sequence[Any], chunk_start: float) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for item in words:
            if not isinstance(item, dict):
                continue
            text = str(item.get("word") or item.get("text") or "").strip()
            if not text:
                continue
            try:
                start = float(item.get("start"))
                end = float(item.get("end"))
            except (TypeError, ValueError):
                continue
            if end < start:
                end = start
            speaker = item.get("speaker")
            if speaker is None:
                speaker = item.get("speaker_label")
            result.append(
                {
                    "word": text,
                    "start": chunk_start + start,
                    "end": chunk_start + max(start, end),
                    "speaker": str(speaker).strip() if speaker is not None else "",
                }
            )
        result.sort(key=lambda word: (word["start"], word["end"]))
        return result

    @staticmethod
    def _words_from_segments(segments: Sequence[Any], chunk_start: float) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            try:
                start = float(segment.get("start"))
                end = float(segment.get("end"))
            except (TypeError, ValueError):
                continue
            speaker = str(segment.get("speaker") or "").strip()
            tokens = text.split()
            if not tokens:
                continue
            step = (max(start, end) - start) / max(1, len(tokens))
            for offset, token in enumerate(tokens):
                result.append(
                    {
                        "word": token,
                        "start": chunk_start + start + step * offset,
                        "end": chunk_start + start + step * (offset + 1),
                        "speaker": speaker,
                    }
                )
        return result

    # ------------------------------------------------------------------
    # Speaker reconciliation
    # ------------------------------------------------------------------

    def _resolve_global_speakers(self, chunks: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, str], str]:
        records: List[Dict[str, Any]] = []
        embedder = self._get_embedder()
        for chunk in chunks:
            index = int(chunk["index"])
            words = chunk.get("words") or []
            local_ranges: Dict[str, List[Tuple[float, float]]] = {}
            local_word_counts: Dict[str, int] = {}
            local_runs: Dict[str, List[Tuple[float, float]]] = {}
            for word in words:
                speaker = self._normalise_speaker_id(word.get("speaker"), index)
                word["speaker"] = speaker
                local_word_counts[speaker] = local_word_counts.get(speaker, 0) + 1
                local_ranges.setdefault(speaker, []).append((float(word["start"]), float(word["end"])))
            for speaker, ranges in local_ranges.items():
                local_runs[speaker] = self._merge_ranges(ranges)
            for speaker in local_ranges:
                embedding = self._speaker_embedding(
                    chunk,
                    speaker,
                    local_runs.get(speaker) or [],
                    embedder,
                )
                records.append(
                    {
                        "chunk_index": index,
                        "local_speaker": speaker,
                        "embedding": embedding,
                        "weight": float(local_word_counts.get(speaker, 1)),
                        "ranges": local_runs.get(speaker) or [],
                    }
                )

        if not records:
            return {}

        global_speakers: List[_SpeakerCluster] = []
        if any(record.get("embedding") is not None for record in records):
            embedded_records = [record for record in records if record.get("embedding") is not None]
            unembedded = [record for record in records if record.get("embedding") is None]
            clusters = self._cluster_embeddings(embedded_records, self.speakers_expected)
            for members in clusters:
                cluster = _SpeakerCluster(self._speaker_label(len(global_speakers)))
                for record in members:
                    cluster.add(
                        int(record["chunk_index"]),
                        str(record["local_speaker"]),
                        record.get("embedding"),
                        float(record.get("weight") or 1.0),
                        record.get("ranges") or [],
                    )
                global_speakers.append(cluster)
            self._assign_unembedded_records(global_speakers, unembedded)
        else:
            global_speakers = self._cluster_by_overlap(records)

        mapping: Dict[Tuple[int, str], str] = {}
        for cluster in global_speakers:
            for chunk_index, local_speaker in cluster.members:
                mapping[(chunk_index, local_speaker)] = cluster.label
        return mapping

    def _get_embedder(self) -> Optional[Any]:
        if self._embedder_checked:
            return self._embedder
        self._embedder_checked = True
        try:
            from src.utils.audio_embedder import AudioEmbedder

            embedder = AudioEmbedder(device=self.device)
            if getattr(embedder, "voice_encoder", None) is not None:
                self._embedder = embedder
                logger.info("OpenRouter STT: speaker embedding merge enabled")
            else:
                logger.warning("OpenRouter STT: Resemblyzer unavailable, using overlap speaker matching")
        except Exception as exc:
            logger.warning("OpenRouter STT: speaker embeddings unavailable (%s)", exc)
            self._embedder = None
        return self._embedder

    def _speaker_embedding(
        self,
        chunk: Dict[str, Any],
        speaker: str,
        ranges: Sequence[Tuple[float, float]],
        embedder: Optional[Any],
    ) -> Optional[np.ndarray]:
        if embedder is None or not PYDUB_AVAILABLE or not ranges:
            return None
        try:
            audio = AudioSegment.from_file(chunk["path"])
        except Exception as exc:
            logger.debug("OpenRouter STT: could not load chunk audio: %s", exc)
            return None

        chunk_start = float(chunk["start"])
        candidates = sorted(ranges, key=lambda item: item[1] - item[0], reverse=True)
        embeddings: List[np.ndarray] = []
        weights: List[float] = []
        selected_seconds = 0.0
        for start, end in candidates:
            if len(embeddings) >= MAX_EMBEDDING_CLIPS_PER_SPEAKER:
                break
            if selected_seconds >= MAX_EMBEDDING_SECONDS_PER_SPEAKER:
                break
            duration = max(0.0, end - start)
            if duration < MIN_EMBEDDING_CLIP_SECONDS:
                continue
            local_start_ms = int(max(0.0, start - chunk_start) * 1000)
            local_end_ms = int(max(local_start_ms + 100, (end - chunk_start) * 1000))
            try:
                clip = audio[local_start_ms:local_end_ms]
            except Exception:
                continue
            if len(clip) < 500:
                continue
            try:
                embedding = embedder.extract_embedding(clip)
            except Exception as exc:
                logger.debug("OpenRouter STT: embedding failed: %s", exc)
                continue
            if embedding is None:
                continue
            embeddings.append(np.asarray(embedding, dtype=np.float64))
            weights.append(float(min(duration, 5.0)))
            selected_seconds += duration
        return _average_embeddings(embeddings, weights)

    def _cluster_embeddings(
        self,
        records: Sequence[Dict[str, Any]],
        expected_speakers: Optional[int],
    ) -> List[List[Dict[str, Any]]]:
        clusters: List[Dict[str, Any]] = [
            {
                "members": [dict(record)],
                "embedding": np.asarray(record["embedding"], dtype=np.float64),
                "weight": float(record.get("weight") or 1.0),
            }
            for record in records
        ]
        target = max(1, int(expected_speakers)) if expected_speakers else 1
        minimum_clusters = target if expected_speakers else 1

        while len(clusters) > minimum_clusters:
            best_pair: Optional[Tuple[int, int]] = None
            best_score = -1.0
            for left in range(len(clusters)):
                for right in range(left + 1, len(clusters)):
                    score = _cosine_similarity(clusters[left]["embedding"], clusters[right]["embedding"])
                    if score > best_score:
                        best_score = score
                        best_pair = (left, right)
            if best_pair is None:
                break
            if not expected_speakers and best_score < SPEAKER_MERGE_THRESHOLD:
                break
            left, right = best_pair
            first = clusters[left]
            second = clusters[right]
            weight = float(first["weight"]) + float(second["weight"])
            embedding = (
                np.asarray(first["embedding"], dtype=np.float64) * float(first["weight"])
                + np.asarray(second["embedding"], dtype=np.float64) * float(second["weight"])
            ) / max(1e-9, weight)
            norm = float(np.linalg.norm(embedding))
            if norm > 0:
                embedding = embedding / norm
            merged = {
                "members": [*first["members"], *second["members"]],
                "embedding": embedding,
                "weight": weight,
            }
            clusters = [item for index, item in enumerate(clusters) if index not in (left, right)]
            clusters.append(merged)

        # Keep chronological order for stable SPEAKER_A/B/C labels.
        clusters.sort(key=lambda cluster: min(
            (
                float((member.get("ranges") or [(0.0, 0.0)])[0][0])
                for member in cluster["members"]
            ),
            default=0.0,
        ))
        return [cluster["members"] for cluster in clusters]

    def _cluster_by_overlap(self, records: Sequence[Dict[str, Any]]) -> List[_SpeakerCluster]:
        clusters: List[_SpeakerCluster] = []
        for record in sorted(records, key=lambda item: (item["chunk_index"], item["local_speaker"])):
            ranges = record.get("ranges") or []
            best_cluster: Optional[_SpeakerCluster] = None
            best_overlap = 0.0
            for cluster in clusters:
                overlap = self._ranges_overlap(ranges, cluster.ranges)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cluster = cluster
            if best_cluster is None or best_overlap < 0.35:
                best_cluster = _SpeakerCluster(self._speaker_label(len(clusters)))
                clusters.append(best_cluster)
            best_cluster.add(
                int(record["chunk_index"]),
                str(record["local_speaker"]),
                None,
                float(record.get("weight") or 1.0),
                ranges,
            )
        return clusters

    def _assign_unembedded_records(
        self,
        clusters: List[_SpeakerCluster],
        records: Sequence[Dict[str, Any]],
    ) -> None:
        for record in sorted(records, key=lambda item: (item["chunk_index"], item["local_speaker"])):
            ranges = record.get("ranges") or []
            best_cluster: Optional[_SpeakerCluster] = None
            best_overlap = 0.0
            for cluster in clusters:
                overlap = self._ranges_overlap(ranges, cluster.ranges)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cluster = cluster
            if best_cluster is None or best_overlap < 0.35:
                best_cluster = _SpeakerCluster(self._speaker_label(len(clusters)))
                clusters.append(best_cluster)
            best_cluster.add(
                int(record["chunk_index"]),
                str(record["local_speaker"]),
                None,
                float(record.get("weight") or 1.0),
                ranges,
            )

    @staticmethod
    def _normalise_speaker_id(value: Any, chunk_index: int) -> str:
        if value is None or str(value).strip() == "":
            return "SPEAKER_0"
        text = str(value).strip()
        if text.upper().startswith("SPEAKER_"):
            return text.upper()
        return f"SPEAKER_{text}"

    @staticmethod
    def _speaker_label(index: int) -> str:
        if index < 26:
            return f"SPEAKER_{chr(ord('A') + index)}"
        return f"SPEAKER_{index}"

    @staticmethod
    def _merge_ranges(ranges: Sequence[Tuple[float, float]], max_gap: float = 0.8) -> List[Tuple[float, float]]:
        if not ranges:
            return []
        ordered = sorted((float(start), float(end)) for start, end in ranges if float(end) >= float(start))
        merged: List[Tuple[float, float]] = []
        for start, end in ordered:
            if not merged or start - merged[-1][1] > max_gap:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    @staticmethod
    def _ranges_overlap(
        left: Sequence[Tuple[float, float]],
        right: Sequence[Tuple[float, float]],
    ) -> float:
        total = 0.0
        for l_start, l_end in left:
            for r_start, r_end in right:
                total += max(0.0, min(l_end, r_end) - max(l_start, r_start))
        return total

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _merge_chunk_words(
        self,
        chunks: Sequence[Dict[str, Any]],
        speaker_map: Dict[Tuple[int, str], str],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for chunk in chunks:
            index = int(chunk["index"])
            chunk_start = float(chunk["start"])
            overlap_start = chunk_start + (self.overlap_seconds if index > 0 else 0.0)
            for word in chunk.get("words") or []:
                local = self._normalise_speaker_id(word.get("speaker"), index)
                global_speaker = speaker_map.get((index, local))
                if not global_speaker:
                    global_speaker = self._speaker_label(index)
                start = float(word["start"])
                end = float(word["end"])
                midpoint = (start + end) / 2.0
                if index > 0 and self.overlap_seconds > 0 and midpoint < overlap_start:
                    continue
                merged.append(
                    {
                        "word": str(word.get("word") or ""),
                        "start": start,
                        "end": end,
                        "speaker": global_speaker,
                    }
                )
        merged.sort(key=lambda word: (word["start"], word["end"]))
        return merged

    def _build_segments(self, words: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not words:
            return []
        segments: List[Dict[str, Any]] = []
        current: List[Dict[str, Any]] = []
        current_speaker: Optional[str] = None

        def flush() -> None:
            nonlocal current, current_speaker
            if not current:
                return
            text = _join_words(current)
            if text:
                segments.append(
                    {
                        "text": text,
                        "start": float(current[0]["start"]),
                        "end": float(current[-1]["end"]),
                        "speaker": current_speaker or "SPEAKER_A",
                        "words": list(current),
                    }
                )
            current = []

        for word in words:
            speaker = str(word.get("speaker") or "SPEAKER_A")
            start = float(word["start"])
            if current:
                gap = start - float(current[-1]["end"])
                previous_text = str(current[-1].get("word") or "")
                sentence_end = bool(_SENTENCE_END_RE.search(previous_text))
                if speaker != current_speaker or gap > 0.9 or (sentence_end and len(current) >= 8):
                    flush()
            if not current:
                current_speaker = speaker
            current.append(word)
            if float(current[-1]["end"]) - float(current[0]["start"]) > 28.0:
                flush()
        flush()
        return segments

    @staticmethod
    def _cleanup_temp_dir(temp_dir: str, source_audio: str, converted_path: Optional[str]) -> None:
        import shutil

        if converted_path and os.path.abspath(converted_path) == os.path.abspath(source_audio):
            converted_path = None
        shutil.rmtree(temp_dir, ignore_errors=True)
