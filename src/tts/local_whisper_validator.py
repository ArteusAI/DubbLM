"""Local Whisper ASR helper for cheap TTS content validation."""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Union

AudioSource = Union[str, Path, bytes]


class LocalWhisperContentValidator:
    """Transcribe short TTS clips with faster-whisper on CPU.

    The class caches model instances process-wide because Gemini TTS validation
    can run for many segments in one job. Inference is serialized per model to
    avoid thread-safety surprises when segment synthesis workers validate in
    parallel.
    """

    _models: dict[tuple[str, str, str, int], object] = {}
    _model_locks: dict[tuple[str, str, str, int], threading.Lock] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        language_code: str | None = None,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 2,
    ) -> None:
        self.language_code = language_code
        self.model_name = model_name or "base"
        self.device = device or "cpu"
        self.compute_type = compute_type or "int8"
        self.cpu_threads = max(1, int(cpu_threads or 2))

    @property
    def provider(self) -> str:
        return "whisper"

    @property
    def model(self) -> str:
        return self.model_name

    def transcribe_text(self, audio: AudioSource, *, speaker_labels: bool = False) -> str:
        """Return the plain transcript for ``audio``.

        ``speaker_labels`` is accepted for interface compatibility with the
        AssemblyAI diarizer, but local content validation does not diarize.
        """
        del speaker_labels
        model = self._get_model()
        audio_path: str | None = None
        tmp_path: str | None = None

        if isinstance(audio, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio)
                tmp_path = tmp.name
                audio_path = tmp.name
        else:
            audio_path = str(audio)

        try:
            language = self._normalize_language(self.language_code)
            key = self._model_key()
            with self._model_locks[key]:
                segments, _info = model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=1,
                    best_of=1,
                    temperature=0,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    word_timestamps=False,
                )
                return " ".join(segment.text.strip() for segment in segments).strip()
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _model_key(self) -> tuple[str, str, str, int]:
        return (self.model_name, self.device, self.compute_type, self.cpu_threads)

    def _get_model(self):
        key = self._model_key()
        with self._cache_lock:
            if key not in self._models:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise RuntimeError(
                        "faster-whisper is required for local TTS content validation"
                    ) from exc
                self._models[key] = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    num_workers=1,
                )
                self._model_locks[key] = threading.Lock()
            return self._models[key]

    @staticmethod
    def _normalize_language(language_code: str | None) -> str | None:
        if not language_code or language_code == "auto":
            return None
        return language_code.split("-", 1)[0].strip() or None
