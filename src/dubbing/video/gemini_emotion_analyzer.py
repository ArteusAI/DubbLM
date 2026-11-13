"""Video + audio emotion analysis with a Gemini VLM (single unified stream).

The dubbing actor has to reproduce the *original performance*, not just the
words.  To know how a line is performed we need both halves of the take at
once: the picture (face, body language, editing rhythm) and the voice (tone,
pace, volume, breathiness).  Gemini can consume both from a single video
input, so the trailer is sent as ONE stream - a downscaled proxy that keeps
the original audio track.

The transcript is supplied as a timestamped list of lines and the model
answers with one annotation per line:

    {"index": 3, "emotion": "tense", "intensity": 0.7,
     "direction": "restrained, rising urgency"}

``emotion`` is a canonical label (the same vocabulary the TTS wrappers
already understand), ``direction`` is a short director's note that is mixed
into the TTS instruction (Gemini 3.8 style / Fish Audio Drama 3 direction).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

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

# Emotion analysis windows are expanded to a quiet point around the first and
# last transcript line.  A hard cut at a segment edge can truncate a breath,
# laugh or syllable, and the VLM then reads the wrong emotion from the cut.
SILENCE_BOUNDARY_SEARCH_SECONDS = 1.5
SILENCE_BOUNDARY_PAD_SECONDS = 0.5
SILENCE_MIN_LEN_MS = 180

DEFAULT_MODEL = "gemini-flash-latest"
# Gemini accepts up to ~20 MB inline; 14 MB keeps room for the prompt.
MAX_INLINE_BYTES = 14 * 1024 * 1024
# Files API limit for a single video.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_DIRECTION_MAX_CHARS = 90
NEUTRAL_LABEL = "neutral"
# Lines whose delivery is flat (or that are not spoken at all) are neutral.
DEFAULT_MIN_INTENSITY = 0.15
# Gemini samples video audio at 1 token/s; a proxy does not need more.
DEFAULT_FPS = 0.5
DEFAULT_PROXY_WIDTH = 480
FILE_ACTIVE_TIMEOUT_SECONDS = 300
FILE_ACTIVE_POLL_SECONDS = 5

_WHITESPACE_RE = re.compile(r"\s+")
# Directives must never carry markup that a TTS model could read aloud.
_DIRECTION_STRIP_RE = re.compile(r"[\[\]<>\(\)\{\}]")
_DIRECTION_QUOTES_RE = re.compile(r"[\"“”«»']")
_MAX_DIRECTION_WORDS = 16


def _canonical_labels() -> Tuple[str, ...]:
    """Canonical emotion vocabulary shared with the TTS wrappers."""
    from src.tts.gemini38_tags import EMOTION_LABELS

    return tuple(EMOTION_LABELS)


# Aliases the model may produce for a canonical label.
_LABEL_ALIASES: Dict[str, str] = {
    "neutral": "neutral",
    "flat": "neutral",
    "calm": "neutral",
    "normal": "neutral",
    "happy": "happy",
    "joy": "joyful",
    "joyful": "joyful",
    "excited": "excited",
    "excitement": "excited",
    "sad": "sad",
    "sorrow": "sorrowful",
    "sorrowful": "sorrowful",
    "grief": "grieving",
    "grieving": "grieving",
    "anger": "angry",
    "angry": "angry",
    "furious": "furious",
    "rage": "furious",
    "irritated": "irritated",
    "annoyed": "irritated",
    "afraid": "fearful",
    "fearful": "fearful",
    "scared": "scared",
    "panic": "panicked",
    "panicked": "panicked",
    "surprise": "surprised",
    "surprised": "surprised",
    "shock": "shocked",
    "shocked": "shocked",
    "tense": "tense",
    "nervous": "tense",
    "awe": "awe",
    "awe-struck": "awe",
    "tender": "tender",
    "loving": "loving",
    "serious": "serious",
    "solemn": "solemn",
    "playful": "playful",
    "sarcastic": "sarcastic",
    "tired": "tired",
    "hopeful": "hopeful",
    "desperate": "desperate",
    "commanding": "commanding",
    "authoritative": "commanding",
    "triumphant": "triumphant",
    "crying": "crying",
    "crying-out": "crying",
    "whispering": "whispering",
    "whisper": "whispering",
}


@dataclass
class DeliveryAnalysis:
    """Result of one delivery analysis.

    ``lines`` describes how every transcript line is performed, ``speakers``
    describes the voice of every character (timbre, age, energy, accent).
    """

    lines: List[Dict[str, Any]] = field(default_factory=list)
    speakers: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"lines": self.lines, "speakers": self.speakers}

    @classmethod
    def from_dict(cls, data: Any) -> "DeliveryAnalysis":
        if not isinstance(data, dict):
            return cls()
        lines = data.get("lines")
        speakers = data.get("speakers")
        return cls(
            lines=lines if isinstance(lines, list) else [],
            speakers=speakers if isinstance(speakers, list) else [],
        )


@dataclass
class EmotionAnalysisConfig:
    """Knobs for the video+audio emotion analysis."""

    model: str = DEFAULT_MODEL
    fps: float = DEFAULT_FPS
    proxy_width: int = DEFAULT_PROXY_WIDTH
    proxy_crf: int = 30
    proxy_preset: str = "veryfast"
    proxy_audio_bitrate: str = "64k"
    direction_max_chars: int = DEFAULT_DIRECTION_MAX_CHARS
    language: str = "en"
    min_intensity: float = DEFAULT_MIN_INTENSITY
    temperature: float = 0.2
    # Long sources are analysed window by window so one request never has to
    # carry more than this many seconds of video.
    max_window_seconds: float = 600.0
    start_time: Optional[float] = None
    duration: Optional[float] = None
    api_key: Optional[str] = None
    cost_tracker: Optional[Any] = None
    cache_dir: Optional[str] = None
    use_proxy: bool = True


EMOTION_ANALYSIS_PROMPT = """You are a dubbing director and performance analyst.

Attached is ONE video stream of a trailer / promo clip: you can SEE the picture and HEAR the original audio (dialogue, music, effects) at the same time. Below is an automatic transcript of the spoken lines with timestamps - the original wording, in the language that is actually spoken in the video.

Your job: describe HOW each line is PERFORMED in the original, so a dubbing actor can reproduce the same delivery in another language.

For every line return:
- "emotion": exactly one label from this list: {labels}
- "intensity": how strong that emotion is, from 0.0 (flat) to 1.0 (extreme)
- "direction": a short director's note (max {max_words} words) that describes the delivery - tone, pace, volume, breath, effort. Example: "quiet, controlled, almost a whisper" or "loud, breathless, crying while shouting".

Rules:
- Use the timestamps to find the right moment in the video, then judge the ACTUAL performance there: facial expression, body language, camera/editing rhythm, and the voice itself (tone, pitch, pace, volume, breathiness, laughing, crying, screaming).
- The label describes the DELIVERY of the line, not the plot or the meaning of the sentence.
- Use "neutral" when the line is flat and unemotional, when the line is a shout/scream without words, or when the line is not spoken at all (music, effects).
- Several lines can share one shot, and the emotion can change inside a line - if it does, pick the dominant one.
- Never invent, merge, split or reword the lines. Never translate them.
- The note must be written in {language}.
- Return ONE entry per line, using the given "index". Return JSON only, no commentary.

Also describe the VOICE of every character (the timbre of the person, not the emotion of a line):
- "speaker": the speaker id used in the transcript
- "timbre": the quality of the voice itself, e.g. "low and slightly raspy", "bright and clean", "soft and breathy", "gravelly and weathered"
- "age": how old the voice sounds: "child", "teen", "young adult", "adult" or "older"
- "energy": how energetic and intense this character is, from 0.0 (laid back) to 1.0 (intense)
- "accent": the accent of the original speech, or "neutral"

The voice is a property of the character: describe it once per speaker, not per line.

Expected format:
{{"lines":[{{"index":0,"emotion":"tense","intensity":0.6,"direction":"restrained, low, controlled"}}],"speakers":[{{"speaker":"SPEAKER_A","timbre":"low, slightly raspy","age":"adult","energy":0.4,"accent":"neutral"}}]}}

Transcript lines:
{lines_json}"""


def normalize_emotion_label(value: Any) -> str:
    """Map a free-form model label onto the canonical vocabulary.

    Unknown labels are lowercased and kept as-is: ``style_for_emotion`` and the
    Drama 3 renderer fall back to the raw label, so a new word still reaches
    the TTS instruction instead of being dropped.
    """
    label = _WHITESPACE_RE.sub(" ", str(value or "").strip().lower())
    if not label:
        return NEUTRAL_LABEL
    if label in _LABEL_ALIASES:
        return _LABEL_ALIASES[label]
    canonical = _canonical_labels()
    if label in canonical:
        return label
    for candidate in canonical:
        if label in candidate or candidate in label:
            return candidate
    return label


def sanitize_direction(value: Any, max_chars: int = DEFAULT_DIRECTION_MAX_CHARS) -> str:
    """Clean the director's note so it can be embedded in a TTS instruction."""
    text = _WHITESPACE_RE.sub(" ", str(value or "").strip())
    if not text:
        return ""
    text = _DIRECTION_STRIP_RE.sub(" ", text)
    text = _DIRECTION_QUOTES_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip(" ,.;:-")
    if not text:
        return ""
    words = text.split()
    if len(words) > _MAX_DIRECTION_WORDS:
        words = words[:_MAX_DIRECTION_WORDS]
    text = " ".join(words)
    if len(text) > max_chars:
        text = text[:max_chars]
        cut = text.rfind(" ")
        if cut > 20:
            text = text[:cut]
    return text.strip(" ,.;:-")


def _segment_text(segment: Dict[str, Any]) -> str:
    """Original wording of the segment (what is actually said in the video).

    The original text is preferred over the translation: the model matches the
    lines against the audio it hears in the video, so the source wording lines
    up with the take while the translation would not.
    """
    for key in ("text", "original_text", "translation"):
        value = str(segment.get(key) or "").strip()
        if value:
            return value
    return ""


def build_transcript_payload(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact, timestamped view of the segments handed to the model."""
    payload: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments):
        text = _segment_text(segment)
        if not text:
            continue
        payload.append(
            {
                "index": index,
                "start": round(float(segment.get("start") or 0.0), 2),
                "end": round(float(segment.get("end") or 0.0), 2),
                "speaker": str(segment.get("speaker") or ""),
                "text": text,
            }
        )
    return payload


def parse_emotion_response(text: str) -> List[Dict[str, Any]]:
    """Parse and validate the JSON answer of the emotion analyzer."""
    return parse_delivery_response(text).lines


def parse_delivery_response(text: str) -> "DeliveryAnalysis":
    """Parse and validate the JSON answer of the delivery analysis.

    The answer has two parts: the per-line delivery (emotion + note) and the
    per-speaker voice profile (timbre, age, energy, accent).
    """
    if not text:
        return DeliveryAnalysis()

    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("Delivery analysis response is not valid JSON")
                return DeliveryAnalysis()

    if not isinstance(data, dict):
        return DeliveryAnalysis()

    lines_payload = None
    for key in ("lines", "segments", "annotations", "result"):
        value = data.get(key)
        if isinstance(value, list):
            lines_payload = value
            break

    if lines_payload is None and data and all(
        isinstance(value, str) for value in data.values()
    ):
        # A flat mapping {"0": "tense", ...} is also acceptable.
        lines_payload = [
            {"index": key, "emotion": value} for key, value in data.items()
        ]

    speakers_payload = data.get("speakers")
    if not isinstance(speakers_payload, list):
        speakers_payload = []

    return DeliveryAnalysis(
        lines=_parse_lines(lines_payload or []),
        speakers=_parse_speakers(speakers_payload),
    )


def _parse_lines(payload: List[Any]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        try:
            intensity = float(item.get("intensity", 0.0))
        except (TypeError, ValueError):
            intensity = 0.0
        parsed.append(
            {
                "index": index,
                "emotion": normalize_emotion_label(item.get("emotion")),
                "intensity": max(0.0, min(1.0, intensity)),
                "direction": str(item.get("direction") or "").strip(),
            }
        )
    return parsed


def _parse_speakers(payload: List[Any]) -> List[Dict[str, Any]]:
    """Parse the per-speaker voice profiles."""
    parsed: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker") or "").strip()
        if not speaker:
            continue
        try:
            energy = float(item.get("energy", 0.0))
        except (TypeError, ValueError):
            energy = 0.0
        parsed.append(
            {
                "speaker": speaker,
                "timbre": str(item.get("timbre") or "").strip(),
                "age": str(item.get("age") or "").strip(),
                "energy": max(0.0, min(1.0, energy)),
                "accent": str(item.get("accent") or "").strip(),
            }
        )
    return parsed


def _apply_annotations(
    segments: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    *,
    min_intensity: float,
    direction_max_chars: int,
) -> List[Dict[str, Any]]:
    """Write the validated annotations onto the segment dicts.

    The label is always kept (it is what drives the sustained TTS style). The
    delivery note is only kept for a delivery that is strong enough to need
    extra direction - a flat line would just add noise to the style string.
    """
    by_index = {item["index"]: item for item in annotations}
    for index, segment in enumerate(segments):
        annotation = by_index.get(index)
        if not annotation:
            segment["emotion"] = NEUTRAL_LABEL
            segment["emotion_intensity"] = 0.0
            segment["emotion_direction"] = ""
            continue

        emotion = annotation.get("emotion") or NEUTRAL_LABEL
        intensity = float(annotation.get("intensity") or 0.0)
        if emotion == NEUTRAL_LABEL:
            # A flat delivery: no style, no note.
            segment["emotion"] = NEUTRAL_LABEL
            segment["emotion_intensity"] = 0.0
            segment["emotion_direction"] = ""
            continue

        direction = sanitize_direction(
            annotation.get("direction"), max_chars=direction_max_chars
        )
        if intensity < min_intensity:
            direction = ""
        segment["emotion"] = emotion
        segment["emotion_intensity"] = round(intensity, 3)
        segment["emotion_direction"] = direction
    return segments


def _ffmpeg_binary() -> str:
    return os.environ.get("FFMPEG_BINARY", "ffmpeg")


def build_video_proxy(
    video_path: str,
    output_path: str,
    *,
    fps: float = DEFAULT_FPS,
    width: int = DEFAULT_PROXY_WIDTH,
    crf: int = 30,
    preset: str = "veryfast",
    audio_bitrate: str = "64k",
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> bool:
    """Build a small mp4 proxy that keeps the original audio track.

    The proxy is the "unified stream" the model receives: a few frames per
    second for the picture plus the untouched dialogue for the voice.  It also
    keeps the upload small enough for a 500 MB source video.
    """
    ffmpeg = _ffmpeg_binary()
    if not shutil.which(ffmpeg):
        logger.warning("ffmpeg is not available, the emotion analysis will use the source file")
        return False

    command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if start_time is not None and start_time > 0:
        command += ["-ss", f"{float(start_time):.3f}"]
    command += ["-i", video_path]
    if duration is not None and duration > 0:
        command += ["-t", f"{float(duration):.3f}"]
    command += [
        "-vf",
        f"fps={max(0.05, float(fps))},scale={int(width)}:-2",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        # Keep the dialogue: without audio there is no voice to judge.
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-ac",
        "1",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        result = subprocess.run(
            command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Emotion analysis proxy failed: %s", exc)
        return False

    if result.returncode != 0 or not os.path.exists(output_path):
        logger.warning(
            "Emotion analysis proxy failed (rc=%s): %s",
            result.returncode,
            (result.stderr or "").strip()[:400],
        )
        return False
    return True


def _upload_and_wait(client, path: str, mime_type: str = "video/mp4"):
    """Upload a file to the Files API and wait until it is processed."""
    uploaded = client.files.upload(file=path)
    name = getattr(uploaded, "name", None)
    if not name:
        return uploaded

    deadline = time.time() + FILE_ACTIVE_TIMEOUT_SECONDS
    while time.time() < deadline:
        try:
            state = getattr(client.files.get(name=name), "state", None)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.debug("Files API state check failed: %s", exc)
            state = None
        if state is None or str(state).upper() == "ACTIVE":
            return uploaded
        if str(state).upper() == "FAILED":
            raise RuntimeError(f"Gemini file processing failed for {name}")
        time.sleep(FILE_ACTIVE_POLL_SECONDS)
    raise RuntimeError(f"Gemini file {name} was not ready in time")


def _build_video_part(client, video_path: str, fps: float):
    """Return a Part carrying the video (inline when small, Files API otherwise)."""
    from google.genai import types

    size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
    mime_type = "video/mp4"

    if size <= MAX_INLINE_BYTES:
        with open(video_path, "rb") as handle:
            data = handle.read()
        return types.Part(
            inline_data=types.Blob(data=data, mime_type=mime_type),
            video_metadata=types.VideoMetadata(fps=float(fps)),
        )

    if size > MAX_UPLOAD_BYTES:
        raise RuntimeError(
            f"Video is too large for the Gemini Files API ({size} bytes)"
        )

    logger.info("Uploading %s bytes of video to the Gemini Files API", size)
    uploaded = _upload_and_wait(client, video_path, mime_type)
    uri = getattr(uploaded, "uri", None)
    if not uri:
        raise RuntimeError("Gemini file upload did not return a URI")
    return types.Part(
        file_data=types.FileData(file_uri=uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(fps=float(fps)),
    )


def _record_cost(config: EmotionAnalysisConfig, response) -> None:
    tracker = config.cost_tracker
    if tracker is None:
        return
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    input_tokens = float(getattr(usage, "prompt_token_count", 0.0) or 0.0)
    output_tokens = float(getattr(usage, "candidates_token_count", 0.0) or 0.0)
    reasoning_tokens = float(getattr(usage, "thoughts_token_count", 0.0) or 0.0)
    try:
        tracker.add_translation_actual(
            "gemini",
            config.model,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            category="emotion_video_analysis",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not record emotion analysis cost: %s", exc)


def _cache_key(video_path: str, segments: List[Dict[str, Any]], config: EmotionAnalysisConfig) -> str:
    try:
        stat = os.stat(video_path)
        video_signature = f"{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        video_signature = video_path
    segment_signature = hashlib.md5(
        json.dumps(
            [
                [index, round(float(s.get("start") or 0.0), 2), round(float(s.get("end") or 0.0), 2), _segment_text(s)[:80]]
                for index, s in enumerate(segments)
            ],
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]
    raw = (
        f"v2|{video_signature}|{config.model}|{config.fps}|"
        f"{config.start_time or 0.0:.2f}|{config.duration or 0.0:.2f}|{segment_signature}"
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(config: EmotionAnalysisConfig, key: str) -> Optional[str]:
    if not config.cache_dir:
        return None
    directory = os.path.join(config.cache_dir, "emotion_analysis")
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return None
    return os.path.join(directory, f"{key}.json")


def _load_cache(config: EmotionAnalysisConfig, key: str) -> Optional[DeliveryAnalysis]:
    path = _cache_path(config, key)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return DeliveryAnalysis.from_dict(data)


def _save_cache(
    config: EmotionAnalysisConfig,
    key: str,
    analysis: DeliveryAnalysis,
) -> None:
    path = _cache_path(config, key)
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(analysis.to_dict(), handle, ensure_ascii=False, indent=2)
    except OSError as exc:  # pragma: no cover - defensive
        logger.debug("Could not write the emotion analysis cache: %s", exc)


def _window_end(segments: List[Dict[str, Any]], indices: List[int]) -> float:
    """End timestamp of the last segment of a window."""
    return max(float(segments[index].get("end") or 0.0) for index in indices)


def choose_silence_boundary(
    timestamp: float,
    direction: str,
    silences: List[Tuple[float, float]],
    *,
    pad_seconds: float = SILENCE_BOUNDARY_PAD_SECONDS,
) -> float:
    """Pick a quiet boundary near ``timestamp``.

    ``direction='start'`` moves the window start to the nearest silence at or
    before the timestamp; ``direction='end'`` moves the end to the nearest
    silence at or after it.  If no silence is available, a small pad is used.
    """
    time_value = float(timestamp or 0.0)
    pad = max(0.05, float(pad_seconds or 0.0))
    clean = [
        (float(start), float(end))
        for start, end in (silences or [])
        if float(end) > float(start)
    ]
    if direction == "start":
        before = [
            (start, end)
            for start, end in clean
            if end <= time_value + 0.05
        ]
        if before:
            start, end = max(before, key=lambda item: item[1])
            return max(0.0, (start + end) / 2.0)
        return max(0.0, time_value - pad)

    after = [
        (start, end)
        for start, end in clean
        if start >= time_value - 0.05
    ]
    if after:
        start, end = min(after, key=lambda item: item[0])
        return (start + end) / 2.0
    return time_value + pad


def _read_audio_window(
    video_path: str,
    start: float,
    duration: float,
) -> Optional["AudioSegment"]:
    if not PYDUB_AVAILABLE or not video_path or duration <= 0:
        return None
    command = [
        "ffmpeg",
        "-v", "error",
        "-ss", f"{max(0.0, float(start)):.3f}",
        "-t", f"{float(duration):.3f}",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        "pipe:1",
    ]
    try:
        payload = subprocess.check_output(command, stderr=subprocess.PIPE)
        if not payload:
            return None
        return AudioSegment.from_file(io.BytesIO(payload), format="wav")
    except Exception as exc:
        logger.debug("Could not read emotion-analysis audio boundary: %s", exc)
        return None


def _snap_analysis_boundary(
    video_path: str,
    timestamp: float,
    direction: str,
) -> float:
    """Find a silent point around an emotion-analysis window boundary."""
    time_value = max(0.0, float(timestamp or 0.0))
    search = max(0.1, float(SILENCE_BOUNDARY_SEARCH_SECONDS))
    window_start = max(0.0, time_value - search)
    duration = search * 2.0
    window = _read_audio_window(video_path, window_start, duration)
    if window is None:
        pad = max(0.05, float(SILENCE_BOUNDARY_PAD_SECONDS))
        return max(0.0, time_value - pad) if direction == "start" else time_value + pad

    threshold = -40.0
    try:
        if window.dBFS > -60:
            threshold = min(-30.0, max(-50.0, window.dBFS - 18.0))
    except Exception:
        pass
    try:
        raw = detect_silence(
            window,
            min_silence_len=SILENCE_MIN_LEN_MS,
            silence_thresh=threshold,
        )
    except Exception as exc:
        logger.debug("Emotion-analysis silence detection failed: %s", exc)
        raw = []
    silences = [
        (window_start + start_ms / 1000.0, window_start + end_ms / 1000.0)
        for start_ms, end_ms in raw
    ]
    snapped = choose_silence_boundary(
        time_value,
        direction,
        silences,
        pad_seconds=SILENCE_BOUNDARY_PAD_SECONDS,
    )
    if direction == "start":
        return min(snapped, time_value)
    return max(snapped, time_value)


def _build_windows(
    segments: List[Dict[str, Any]],
    max_window_seconds: float,
) -> List[Tuple[float, float, List[int]]]:
    """Split the segments into windows that fit into one request.

    A long source (a full lecture, a whole episode) would overflow the
    context of a single request, so the analysis is done window by window.
    Returns ``(start, end, segment_indices)`` per window.
    """
    windows: List[Tuple[float, float, List[int]]] = []
    current: List[int] = []
    window_start = 0.0

    for index, segment in enumerate(segments):
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        if current and (end - window_start) > max_window_seconds:
            windows.append((window_start, _window_end(segments, current), current))
            current = []
        if not current:
            window_start = start
        current.append(index)

    if current:
        windows.append((window_start, _window_end(segments, current), current))
    return windows


def _analyze_window(
    client,
    video_path: str,
    segments: List[Dict[str, Any]],
    config: EmotionAnalysisConfig,
) -> DeliveryAnalysis:
    """Analyze one window of the video and return its delivery analysis."""
    from google.genai import types

    cache_key = _cache_key(video_path, segments, config)
    cached = _load_cache(config, cache_key)
    if cached is not None:
        logger.info("Reusing cached delivery analysis (%s)", cache_key)
        return cached

    temp_dir = tempfile.mkdtemp(prefix="emotion_proxy_")
    proxy_path = os.path.join(temp_dir, "emotion_proxy.mp4")
    analysis_path = video_path
    try:
        if config.use_proxy and build_video_proxy(
            video_path,
            proxy_path,
            fps=config.fps,
            width=config.proxy_width,
            crf=config.proxy_crf,
            preset=config.proxy_preset,
            audio_bitrate=config.proxy_audio_bitrate,
            start_time=config.start_time,
            duration=config.duration,
        ):
            analysis_path = proxy_path
            logger.info(
                "Delivery analysis proxy: %.1f MB (source %.1f MB)",
                os.path.getsize(proxy_path) / 1e6,
                os.path.getsize(video_path) / 1e6,
            )

        video_part = _build_video_part(client, analysis_path, config.fps)

        transcript = build_transcript_payload(segments)
        prompt = EMOTION_ANALYSIS_PROMPT.format(
            labels=", ".join(_canonical_labels()),
            max_words=_MAX_DIRECTION_WORDS,
            language=config.language or "en",
            lines_json=json.dumps(transcript, ensure_ascii=False),
        )

        response = client.models.generate_content(
            model=config.model,
            contents=[prompt, video_part],
            config=types.GenerateContentConfig(
                temperature=config.temperature,
                response_mime_type="application/json",
            ),
        )
        _record_cost(config, response)

        analysis = parse_delivery_response(response.text or "")
        if not analysis.lines:
            logger.warning("Delivery analysis returned no usable annotations")
            return analysis

        _save_cache(config, cache_key, analysis)
        return analysis
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def analyze_delivery(
    video_path: str,
    segments: List[Dict[str, Any]],
    config: Optional[EmotionAnalysisConfig] = None,
) -> DeliveryAnalysis:
    """Analyze the original performance of every segment and every speaker.

    The segments are updated in place with ``emotion``, ``emotion_intensity``
    and ``emotion_direction``.  The returned object also carries the voice
    profile of every speaker (timbre, age, energy, accent).
    """
    config = config or EmotionAnalysisConfig()
    if not segments:
        return DeliveryAnalysis()
    if not video_path or not os.path.exists(video_path):
        logger.warning("Delivery analysis skipped: source video %s is missing", video_path)
        return DeliveryAnalysis()

    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"google-genai is not installed: {exc}") from exc

    api_key = (
        config.api_key
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for the video emotion analysis")

    client = genai.Client(api_key=api_key)

    windows = _build_windows(segments, config.max_window_seconds)
    logger.info(
        "Delivery analysis: %d segment(s) in %d window(s) with %s",
        len(segments),
        len(windows),
        config.model,
    )

    lines: List[Dict[str, Any]] = []
    speakers: List[Dict[str, Any]] = []
    base_start = float(config.start_time or 0.0)
    for window_start, window_end, indices in windows:
        # Snap both edges to a quiet point before/after the transcript lines.
        # The VLM must not see a hard cut through a breath, laugh or syllable.
        snapped_start_abs = _snap_analysis_boundary(
            video_path, base_start + window_start, "start"
        )
        snapped_end_abs = _snap_analysis_boundary(
            video_path, base_start + window_end, "end"
        )
        analysis_start = max(0.0, snapped_start_abs - base_start)
        analysis_end = max(analysis_start + 0.05, snapped_end_abs - base_start)
        if analysis_end <= analysis_start:
            analysis_start, analysis_end = window_start, window_end
        logger.debug(
            "Emotion analysis window snapped: %.2f-%.2f -> %.2f-%.2f",
            window_start,
            window_end,
            analysis_start,
            analysis_end,
        )

        window_segments = []
        for index in indices:
            segment = segments[index]
            start = float(segment.get("start") or 0.0) - analysis_start
            end = float(segment.get("end") or 0.0) - analysis_start
            window_segments.append({**segment, "start": max(0.0, start), "end": max(0.0, end)})

        window_config = replace(
            config,
            start_time=base_start + analysis_start,
            duration=max(1.0, analysis_end - analysis_start),
        )
        window_analysis = _analyze_window(
            client, video_path, window_segments, window_config
        )
        for annotation in window_analysis.lines:
            local_index = annotation.get("index")
            if not isinstance(local_index, int) or local_index >= len(indices):
                continue
            lines.append({**annotation, "index": indices[local_index]})
        speakers.extend(window_analysis.speakers)

    _apply_annotations(
        segments,
        lines,
        min_intensity=config.min_intensity,
        direction_max_chars=config.direction_max_chars,
    )
    return DeliveryAnalysis(lines=lines, speakers=speakers)


def analyze_segment_emotions(
    video_path: str,
    segments: List[Dict[str, Any]],
    config: Optional[EmotionAnalysisConfig] = None,
) -> List[Dict[str, Any]]:
    """Analyze the performance of every segment with a Gemini VLM.

    Thin wrapper around :func:`analyze_delivery` for callers that only need
    the per-line delivery.
    """
    return analyze_delivery(video_path, segments, config).lines
