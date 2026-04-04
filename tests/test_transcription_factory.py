import sys
import types


def test_transcription_factory_imports_without_optional_backends():
    from transcription.transcription_factory import TranscriptionFactory

    assert TranscriptionFactory is not None


def test_transcription_factory_supports_whisper_alias():
    from transcription.transcription_factory import TranscriptionFactory

    fake_module = types.ModuleType("transcription.pyannote_openai_transcriber")

    class FakePyAnnoteOpenAITranscriber:
        def __init__(self, source_language, device=None, **kwargs):
            self.source_language = source_language
            self.device = device
            self.kwargs = kwargs

    fake_module.PyAnnoteOpenAITranscriber = FakePyAnnoteOpenAITranscriber
    original_module = sys.modules.get("transcription.pyannote_openai_transcriber")
    sys.modules["transcription.pyannote_openai_transcriber"] = fake_module

    try:
        transcriber = TranscriptionFactory.create_transcriber(
            transcription_system="whisper",
            source_language="en",
            device="cpu",
            whisper_model="large-v3",
        )
    finally:
        if original_module is None:
            sys.modules.pop("transcription.pyannote_openai_transcriber", None)
        else:
            sys.modules["transcription.pyannote_openai_transcriber"] = original_module

    assert isinstance(transcriber, FakePyAnnoteOpenAITranscriber)
    assert transcriber.kwargs["transcription_system"] == "whisper"


def test_smart_dubbing_imports_without_optional_speechbrain():
    import dubbing.core.smart_dubbing as smart_dubbing

    assert smart_dubbing is not None


def test_smart_dubbing_reports_transcriber_init_failure_with_context():
    from dubbing.core.smart_dubbing import SmartDubbing

    class ConfigStub(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = ConfigStub({"transcription_system": "assemblyai"})
    dubber.transcriber = None
    dubber.transcriber_init_error = ValueError("ASSEMBLYAI_API_KEY environment variable is required")

    try:
        dubber.diarize_and_transcribe("artifacts/audio/source.wav")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected diarize_and_transcribe to raise RuntimeError")

    assert "assemblyai" in message
    assert "ASSEMBLYAI_API_KEY" in message


def test_transcription_factory_passes_artifacts_root_to_pyannote_backend():
    from transcription.transcription_factory import TranscriptionFactory

    fake_module = types.ModuleType("transcription.pyannote_openai_transcriber")

    class FakePyAnnoteOpenAITranscriber:
        def __init__(self, source_language, device=None, **kwargs):
            self.source_language = source_language
            self.device = device
            self.kwargs = kwargs

    fake_module.PyAnnoteOpenAITranscriber = FakePyAnnoteOpenAITranscriber
    original_module = sys.modules.get("transcription.pyannote_openai_transcriber")
    sys.modules["transcription.pyannote_openai_transcriber"] = fake_module

    try:
        transcriber = TranscriptionFactory.create_transcriber(
            transcription_system="openai",
            source_language="en",
            device="cpu",
            artifacts_root="D:/tmp/Are/artifacts",
        )
    finally:
        if original_module is None:
            sys.modules.pop("transcription.pyannote_openai_transcriber", None)
        else:
            sys.modules["transcription.pyannote_openai_transcriber"] = original_module

    assert isinstance(transcriber, FakePyAnnoteOpenAITranscriber)
    assert transcriber.kwargs["artifacts_root"] == "D:/tmp/Are/artifacts"
