import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_omnivoice_prepare_text_ignores_style_prompt():
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.voice_prompt_mapping = {"SPEAKER_00": "warm and soft"}

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Толькі гэты тэкст.",
        style_prompt="do not use",
    )

    assert wrapper._prepare_text(segment) == "Толькі гэты тэкст."


def test_omnivoice_uses_segment_reference_text_for_predict(tmp_path, monkeypatch):
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    reference_audio = tmp_path / "ref.wav"
    reference_audio.write_bytes(b"fake-audio")
    generated_audio = tmp_path / "generated.wav"
    generated_audio.write_bytes(b"RIFFfakeWAVE")

    captured = {}

    class ClientStub:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return str(generated_audio)

    class FakeAudioSegment:
        def __len__(self):
            return 1000

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.client = ClientStub()
    wrapper.lang = "Belarusian"
    wrapper.num_steps = 32
    wrapper.guidance_scale = 2.0
    wrapper.denoise = True
    wrapper.speed = 1.0
    wrapper.duration = 3.0
    wrapper.preprocess_prompt = True
    wrapper.postprocess_output = True
    wrapper.api_name = "/_clone_fn"
    wrapper.default_reference_audio = None
    wrapper.default_reference_text = "fallback text"
    wrapper.voice_mapping = {}
    wrapper.voice_prompt_mapping = {}
    wrapper._temp_dir = str(tmp_path)

    monkeypatch.setattr(module, "handle_file", lambda path: f"handled:{path}")
    monkeypatch.setattr(module, "AudioSegment", type("AudioSegmentStub", (), {"from_file": staticmethod(lambda _path: FakeAudioSegment())}))

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Synth text",
        reference_audio_path=str(reference_audio),
        reference_text="Recognized original speech",
    )

    alignments = wrapper.synthesize([segment])

    assert len(alignments) == 1
    assert captured["ref_text"] == "Recognized original speech"


def test_omnivoice_does_not_fallback_to_default_reference_text(tmp_path, monkeypatch):
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    reference_audio = tmp_path / "ref.wav"
    reference_audio.write_bytes(b"fake-audio")
    generated_audio = tmp_path / "generated.wav"
    generated_audio.write_bytes(b"RIFFfakeWAVE")

    captured = {}

    class ClientStub:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return str(generated_audio)

    class FakeAudioSegment:
        def __len__(self):
            return 1000

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.client = ClientStub()
    wrapper.lang = "Belarusian"
    wrapper.num_steps = 32
    wrapper.guidance_scale = 2.0
    wrapper.denoise = True
    wrapper.speed = 1.0
    wrapper.duration = 3.0
    wrapper.preprocess_prompt = True
    wrapper.postprocess_output = True
    wrapper.api_name = "/_clone_fn"
    wrapper.default_reference_audio = None
    wrapper.default_reference_text = "fallback text"
    wrapper.voice_mapping = {}
    wrapper.voice_prompt_mapping = {}
    wrapper._temp_dir = str(tmp_path)

    monkeypatch.setattr(module, "handle_file", lambda path: f"handled:{path}")
    monkeypatch.setattr(module, "AudioSegment", type("AudioSegmentStub", (), {"from_file": staticmethod(lambda _path: FakeAudioSegment())}))

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Synth text",
        reference_audio_path=str(reference_audio),
        reference_text=None,
    )

    alignments = wrapper.synthesize([segment])

    assert len(alignments) == 1
    assert captured["ref_text"] == ""


def test_omnivoice_sends_empty_reference_text_when_value_is_blank(tmp_path, monkeypatch):
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    reference_audio = tmp_path / "ref.wav"
    reference_audio.write_bytes(b"fake-audio")
    generated_audio = tmp_path / "generated.wav"
    generated_audio.write_bytes(b"RIFFfakeWAVE")

    captured = {}

    class ClientStub:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return str(generated_audio)

    class FakeAudioSegment:
        def __len__(self):
            return 1000

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.client = ClientStub()
    wrapper.lang = "Belarusian"
    wrapper.num_steps = 32
    wrapper.guidance_scale = 2.0
    wrapper.denoise = True
    wrapper.speed = 1.0
    wrapper.duration = 3.0
    wrapper.preprocess_prompt = True
    wrapper.postprocess_output = True
    wrapper.api_name = "/_clone_fn"
    wrapper.default_reference_audio = None
    wrapper.default_reference_text = "fallback text"
    wrapper.voice_mapping = {}
    wrapper.voice_prompt_mapping = {}
    wrapper._temp_dir = str(tmp_path)

    monkeypatch.setattr(module, "handle_file", lambda path: f"handled:{path}")
    monkeypatch.setattr(module, "AudioSegment", type("AudioSegmentStub", (), {"from_file": staticmethod(lambda _path: FakeAudioSegment())}))

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Synth text",
        reference_audio_path=str(reference_audio),
        reference_text="   ",
    )

    alignments = wrapper.synthesize([segment])

    assert len(alignments) == 1
    assert captured["ref_text"] == ""


def test_omnivoice_uses_segment_target_duration_for_du(tmp_path, monkeypatch):
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    reference_audio = tmp_path / "ref.wav"
    reference_audio.write_bytes(b"fake-audio")
    generated_audio = tmp_path / "generated.wav"
    generated_audio.write_bytes(b"RIFFfakeWAVE")

    captured = {}

    class ClientStub:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return str(generated_audio)

    class FakeAudioSegment:
        def __len__(self):
            return 1000

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.client = ClientStub()
    wrapper.lang = "Belarusian"
    wrapper.num_steps = 32
    wrapper.guidance_scale = 2.0
    wrapper.denoise = True
    wrapper.speed = 1.0
    wrapper.duration = 3.0
    wrapper.preprocess_prompt = True
    wrapper.postprocess_output = True
    wrapper.api_name = "/_clone_fn"
    wrapper.default_reference_audio = None
    wrapper.default_reference_text = None
    wrapper.voice_mapping = {}
    wrapper.voice_prompt_mapping = {}
    wrapper._temp_dir = str(tmp_path)

    monkeypatch.setattr(module, "handle_file", lambda path: f"handled:{path}")
    monkeypatch.setattr(module, "AudioSegment", type("AudioSegmentStub", (), {"from_file": staticmethod(lambda _path: FakeAudioSegment())}))

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Synth text",
        reference_audio_path=str(reference_audio),
        reference_text="Recognized original speech",
        target_duration=1.75,
    )

    alignments = wrapper.synthesize([segment])

    assert len(alignments) == 1
    assert captured["du"] == 1.75
