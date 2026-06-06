"""Alignment of Gemini TTS multi-speaker audio via AssemblyAI + fuzzy match."""
from __future__ import annotations

import io
import os
import re
import tempfile
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import assemblyai as aai
from rapidfuzz import fuzz

from .models import DialogueLine, DiarizationResult, SpeakerSegment

AudioSource = Union[str, Path, bytes]


@dataclass(frozen=True)
class _WordToken:
    text: str
    raw_text: str
    start_ms: int
    end_ms: int
    confidence: float


class SpeakerSegmentDiarizer:
    """Transcribe TTS-generated audio and map ASR timings back to expected lines.

    ``align_by_text`` is the safe path for Gemini multi-speaker batching: it
    avoids ASR speaker diarization and aligns expected dialogue lines by text
    span in the full transcript. ``diarize`` is kept as the older speaker-label
    based helper for callers that explicitly need it.

    Parameters
    ----------
    api_key:
        AssemblyAI API key. Falls back to env ``ASSEMBLYAI_API_KEY``.
    language_code:
        BCP-47 language code (e.g. ``"en"``, ``"ru"``). ``None`` = auto-detect.
    speech_model:
        AssemblyAI model: ``"best"`` (default) or ``"nano"`` (cheaper/faster).
    used_line_penalty:
        Score penalty (0..100) applied per previous assignment of the same
        dialogue line. Grows linearly with reuse count so fresh candidates win
        once a popular line has absorbed a few utterances.
    max_utts_per_line:
        Hard cap on how many ASR utterances may map to a single dialogue line.
        Prevents one popular line from "eating" all utterances of a long
        monologue and starving its neighbours.
    """

    _TTS_MARKUP_RE = re.compile(r"[\[\(][^\]\)]{1,40}[\]\)]")
    _WORD_RE = re.compile(r"\w+", re.UNICODE)

    def __init__(
        self,
        api_key: str | None = None,
        language_code: str | None = None,
        speech_model: str = "best",
        used_line_penalty: int = 10,
        max_utts_per_line: int = 3,
    ) -> None:
        key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        if not key:
            raise ValueError(
                "AssemblyAI API key missing: pass api_key or set ASSEMBLYAI_API_KEY"
            )
        self._api_key = key
        self._language_code = language_code
        self._speech_model = speech_model
        self._used_line_penalty = used_line_penalty
        self._max_utts_per_line = max(1, int(max_utts_per_line))

    @classmethod
    def _clean_for_match(cls, text: str) -> str:
        """Normalize text for fuzzy matching.

        Strips inline TTS markup (``[excited]``, ``(laughs)`` ...) so tags
        never participate in ``token_set_ratio``, then drops punctuation and
        lowercases whitespace-collapsed output.
        """
        if not text:
            return ""
        stripped = cls._TTS_MARKUP_RE.sub(" ", text)
        stripped = stripped.replace("ё", "е").replace("Ё", "Е")
        no_punct = re.sub(r"[^\w\s]", " ", stripped, flags=re.UNICODE)
        return re.sub(r"\s+", " ", no_punct).strip().lower()

    @classmethod
    def _tokens_for_match(cls, text: str) -> list[str]:
        return cls._clean_for_match(text).split()

    # ---------------------------------------------------------------- API #
    def diarize(
        self,
        audio: AudioSource,
        expected_lines: Sequence[DialogueLine],
    ) -> DiarizationResult:
        """Return timings and speaker assignments for every ASR utterance.

        ``audio`` can be a path-like or raw WAV/audio bytes.
        """
        if not expected_lines:
            raise ValueError("expected_lines must contain at least one DialogueLine")

        transcript = self._transcribe(audio)
        utterances = transcript.utterances or []

        expected_clean = [self._clean_for_match(line.text) for line in expected_lines]
        used_counts: dict[int, int] = {}
        votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        raw: list[tuple] = []
        for utt in utterances:
            idx, score = self._best_match(
                self._clean_for_match(utt.text or ""),
                expected_clean,
                used_counts,
            )
            if idx is not None:
                used_counts[idx] = used_counts.get(idx, 0) + 1
                votes[utt.speaker][expected_lines[idx].speaker] += 1
            raw.append((utt, idx, score))

        mapping: dict[str, str] = {
            asr_label: max(inner.items(), key=lambda kv: kv[1])[0]
            for asr_label, inner in votes.items()
        }

        segments: list[SpeakerSegment] = []
        for i, (utt, idx, score) in enumerate(raw):
            real = mapping.get(utt.speaker, utt.speaker)
            matched = expected_lines[idx] if idx is not None else None
            segments.append(
                SpeakerSegment(
                    index=i,
                    speaker=real,
                    start_ms=int(utt.start),
                    end_ms=int(utt.end),
                    asr_text=utt.text or "",
                    matched_line_idx=idx,
                    matched_text=matched.text if matched else None,
                    fuzzy_score=float(score),
                    asr_speaker_label=utt.speaker,
                    confidence=float(getattr(utt, "confidence", 0.0) or 0.0),
                )
            )

        return DiarizationResult(
            segments=segments,
            speaker_mapping=mapping,
            full_text=transcript.text or "",
        )

    def align_by_text(
        self,
        audio: AudioSource,
        expected_lines: Sequence[DialogueLine],
    ) -> DiarizationResult:
        """Locate expected dialogue lines in the full ASR text without speaker diarization.

        Gemini already receives the intended speaker for each line. At this stage
        we only need timings for each expected text span in the mixed TTS track;
        relying on ASR speaker labels can falsely reject valid multi-speaker audio
        when the ASR provider does not split turns the same way Gemini spoke them.
        """
        if not expected_lines:
            raise ValueError("expected_lines must contain at least one DialogueLine")

        transcript = self._transcribe(audio, speaker_labels=False)
        word_tokens = self._word_tokens_from_transcript(getattr(transcript, "words", None) or [])
        segments = self._align_lines_to_word_tokens(expected_lines, word_tokens)

        return DiarizationResult(
            segments=segments,
            speaker_mapping={},
            full_text=transcript.text or "",
        )

    def transcribe_text(self, audio: AudioSource, *, speaker_labels: bool = False) -> str:
        """Return the plain ASR transcript for ``audio``.

        ``speaker_labels`` is off by default — diarization is billed as an
        add-on and is unnecessary for single-speaker content checks.
        """
        transcript = self._transcribe(audio, speaker_labels=speaker_labels)
        return (transcript.text or "").strip()

    def slice_audio(
        self,
        audio: AudioSource,
        segments: Sequence[SpeakerSegment],
    ) -> list[bytes]:
        """Cut ``audio`` into WAV byte-strings at each segment's timestamps.

        Works for any PCM WAV (sample rate / bit depth auto-detected).
        """
        pcm, params = self._read_wav(audio)
        return [self._cut_wav(pcm, params, s.start_ms, s.end_ms) for s in segments]

    @staticmethod
    def parse_dialogue(text: str) -> list[DialogueLine]:
        """Parse a ``Speaker: line`` formatted prompt into DialogueLine list."""
        out: list[DialogueLine] = []
        for m in re.finditer(r"^\s*(\w[\w\s\.\-]*?)\s*:\s*(.+)$", text, re.MULTILINE):
            out.append(DialogueLine(speaker=m.group(1).strip(), text=m.group(2).strip()))
        return out

    # ---------------------------------------------------------- internals #
    def _transcribe(self, audio: AudioSource, *, speaker_labels: bool = True) -> aai.Transcript:
        aai.settings.api_key = self._api_key

        model = (
            aai.SpeechModel.nano
            if self._speech_model.lower() == "nano"
            else aai.SpeechModel.best
        )
        kwargs: dict = {
            "speaker_labels": speaker_labels,
            "punctuate": True,
            "speech_model": model,
        }
        if self._language_code:
            kwargs["language_code"] = self._language_code
        config = aai.TranscriptionConfig(**kwargs)
        transcriber = aai.Transcriber(config=config)

        data: str | io.IOBase
        tmp: tempfile._TemporaryFileWrapper | None = None
        if isinstance(audio, (bytes, bytearray)):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio)
            tmp.flush()
            tmp.close()
            data = tmp.name
        else:
            data = str(audio)

        try:
            transcript = transcriber.transcribe(data)
        finally:
            if tmp is not None:
                Path(tmp.name).unlink(missing_ok=True)

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
        return transcript

    def _best_match(
        self,
        utt_text: str,
        expected_clean: Sequence[str],
        used_counts: dict[int, int],
    ) -> tuple[int | None, float]:
        """Pick the best dialogue line for an ASR utterance.

        Inputs are expected to be pre-normalized (see ``_clean_for_match``).
        Lines that already got ``max_utts_per_line`` assignments are removed
        from the candidate pool; remaining candidates are scored with a
        progressive penalty to spread utterances across lines.
        """
        if not utt_text.strip():
            return None, 0.0
        scored: list[tuple[float, int]] = []
        for i, line_text in enumerate(expected_clean):
            count = used_counts.get(i, 0)
            if count >= self._max_utts_per_line:
                continue
            score = float(fuzz.token_set_ratio(utt_text, line_text))
            score -= count * self._used_line_penalty
            scored.append((score, i))
        if not scored:
            return None, 0.0
        scored.sort(key=lambda s: s[0], reverse=True)
        best_score, best_i = scored[0]
        return best_i, best_score

    @classmethod
    def _word_tokens_from_transcript(cls, words: Sequence[object]) -> list[_WordToken]:
        tokens: list[_WordToken] = []
        for word in words:
            raw_text = str(getattr(word, "text", "") or getattr(word, "word", "") or "")
            pieces = cls._WORD_RE.findall(cls._clean_for_match(raw_text))
            if not pieces:
                continue
            start_ms = int(getattr(word, "start", 0) or 0)
            end_ms = int(getattr(word, "end", start_ms) or start_ms)
            confidence = float(getattr(word, "confidence", 0.0) or 0.0)
            for piece in pieces:
                tokens.append(_WordToken(
                    text=piece,
                    raw_text=raw_text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=confidence,
                ))
        return tokens

    @classmethod
    def _align_lines_to_word_tokens(
        cls,
        expected_lines: Sequence[DialogueLine],
        word_tokens: Sequence[_WordToken],
        *,
        min_score: float = 68.0,
    ) -> list[SpeakerSegment]:
        if not word_tokens:
            return []

        segments: list[SpeakerSegment] = []
        cursor = 0
        for line_idx, line in enumerate(expected_lines):
            expected_tokens = cls._tokens_for_match(line.text)
            if not expected_tokens:
                continue
            match = cls._find_best_text_span(
                expected_tokens=expected_tokens,
                word_tokens=word_tokens,
                cursor=cursor,
                min_score=min_score,
            )
            if match is None:
                continue

            start_idx, end_idx, score = match
            span_tokens = word_tokens[start_idx:end_idx]
            confidence_values = [t.confidence for t in span_tokens if t.confidence > 0]
            confidence = (
                sum(confidence_values) / len(confidence_values)
                if confidence_values else 0.0
            )
            asr_text = " ".join(t.raw_text for t in span_tokens).strip()
            segments.append(SpeakerSegment(
                index=len(segments),
                speaker=line.speaker,
                start_ms=span_tokens[0].start_ms,
                end_ms=span_tokens[-1].end_ms,
                asr_text=asr_text,
                matched_line_idx=line_idx,
                matched_text=line.text,
                fuzzy_score=float(score),
                asr_speaker_label="text",
                confidence=float(confidence),
            ))
            cursor = end_idx

        return segments

    @classmethod
    def _find_best_text_span(
        cls,
        *,
        expected_tokens: Sequence[str],
        word_tokens: Sequence[_WordToken],
        cursor: int,
        min_score: float,
    ) -> tuple[int, int, float] | None:
        expected_len = len(expected_tokens)
        remaining = len(word_tokens) - cursor
        if expected_len <= 0 or remaining <= 0:
            return None

        expected_text = " ".join(expected_tokens)
        if expected_len <= 4:
            min_len = 1
            max_len = min(remaining, expected_len + 5)
        else:
            min_len = max(1, int(expected_len * 0.55))
            max_len = min(remaining, max(min_len, int(expected_len * 1.45) + 4))

        best: tuple[int, int, float] | None = None
        last_start = len(word_tokens) - min_len
        for start in range(cursor, last_start + 1):
            local_max_len = min(max_len, len(word_tokens) - start)
            for length in range(min_len, local_max_len + 1):
                end = start + length
                candidate_text = " ".join(t.text for t in word_tokens[start:end])
                if not candidate_text:
                    continue

                ordered = float(fuzz.ratio(expected_text, candidate_text))
                partial = float(fuzz.partial_ratio(expected_text, candidate_text))
                token_set = float(fuzz.token_set_ratio(expected_text, candidate_text))
                length_ratio = min(expected_len, length) / max(expected_len, length)
                score = (
                    (0.45 * ordered) +
                    (0.35 * partial) +
                    (0.20 * token_set)
                ) * (0.75 + 0.25 * length_ratio)

                if best is None or score > best[2]:
                    best = (start, end, score)

        if best is None or best[2] < min_score:
            return None
        return best

    @staticmethod
    def _read_wav(audio: AudioSource) -> tuple[bytes, wave._wave_params]:
        if isinstance(audio, (bytes, bytearray)):
            with wave.open(io.BytesIO(bytes(audio)), "rb") as wf:
                return wf.readframes(wf.getnframes()), wf.getparams()
        with wave.open(str(audio), "rb") as wf:
            return wf.readframes(wf.getnframes()), wf.getparams()

    @staticmethod
    def _cut_wav(
        pcm: bytes,
        params: wave._wave_params,
        start_ms: int,
        end_ms: int,
    ) -> bytes:
        bytes_per_ms = params.framerate * params.sampwidth * params.nchannels // 1000
        s = max(0, start_ms * bytes_per_ms)
        e = min(len(pcm), end_ms * bytes_per_ms)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(params.nchannels)
            wf.setsampwidth(params.sampwidth)
            wf.setframerate(params.framerate)
            wf.writeframes(pcm[s:e])
        return buf.getvalue()
