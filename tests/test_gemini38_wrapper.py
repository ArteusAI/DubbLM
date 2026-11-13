from __future__ import annotations

import base64
import io
import os
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import src.tts.gemini38_tts_wrapper as mod
from src.dubbing.debug.cost_tracker import DEFAULT_PRICING_CATALOG
from src.tts.gemini38_tts_wrapper import (
    DEFAULT_GEMINI38_TTS_FALLBACK_MODEL,
    DEFAULT_GEMINI38_TTS_MODEL,
    Gemini38TTSWrapper,
)
from src.tts.models import TTSSegmentData
from src.tts.tts_factory import TTS_PROVIDERS


def _wav_bytes(seconds: float = 1.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * int(24000 * seconds))
    return buffer.getvalue()


class _FakeCostTracker:
    def __init__(self) -> None:
        self.tts_calls = []

    def add_tts_actual(self, provider, model=None, **kwargs):
        self.tts_calls.append((provider, model, kwargs))

    def add_transcription_usage(self, *args, **kwargs):  # pragma: no cover
        pass


class Gemini38WrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build_wrapper(self, raw_audio: bytes):
        requests = []

        class _Interactions:
            def create(self, **payload):
                requests.append(payload)
                return SimpleNamespace(
                    output_audio=SimpleNamespace(data=base64.b64encode(raw_audio).decode()),
                    usage=SimpleNamespace(input_tokens=7, output_tokens=11),
                )

        fake_client = SimpleNamespace(interactions=_Interactions())
        cost_tracker = _FakeCostTracker()
        wrapper = Gemini38TTSWrapper(
            tts_transport="google",
            enable_voice_matching=False,
            enable_audio_validation=False,
            enable_content_validation=False,
            enable_emotion_enrichment=False,
            cost_tracker=cost_tracker,
        )
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch.object(mod.genai, "Client", return_value=fake_client):
                with patch.object(mod.VoiceSampleManager, "generate_all_samples", return_value=True):
                    with patch.object(mod.AudioFileUtils, "get_audio_duration_seconds", return_value=1.0):
                        wrapper.initialize()
        wrapper.set_voice_mapping({"SPEAKER_A": "Kore"})
        return wrapper, requests, cost_tracker

    def test_provider_is_registered(self) -> None:
        self.assertIs(TTS_PROVIDERS["gemini38"], Gemini38TTSWrapper)
        self.assertFalse(Gemini38TTSWrapper.supports_segment_batching)

    def test_synthesize_uses_interactions_payload_and_writes_audio(self) -> None:
        wrapper, requests, cost_tracker = self._build_wrapper(_wav_bytes(1.0))
        output_path = self.tmp_path / "segment.wav"
        segment = TTSSegmentData(
            speaker="SPEAKER_A",
            text="Hello [sigh] world",
            style_prompt="whispered urgently",
            output_path=str(output_path),
        )

        alignments = None
        with patch.object(mod.AudioFileUtils, "get_audio_duration_seconds", return_value=1.0):
            alignments = wrapper.synthesize([segment])

        self.assertEqual(len(alignments), 1)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 44)

        # The first captured request is the actual segment synthesis; sample
        # generation for the seeded catalog should not have been triggered.
        self.assertEqual(len(requests), 1)
        payload = requests[0]
        self.assertEqual(payload["model"], DEFAULT_GEMINI38_TTS_MODEL)
        self.assertFalse(payload["store"])
        self.assertEqual(payload["response_format"]["mime_type"], "audio/l16")
        self.assertEqual(payload["generation_config"]["speech_config"], [{"voice": "Kore"}])
        content = payload["input"][0]["content"][0]
        self.assertEqual(content["type"], "text")
        self.assertEqual(content["text"], "Hello <sigh> world")
        self.assertEqual(content["annotations"][0]["type"], "speech_metadata")
        self.assertIn("whispered urgently", content["annotations"][0]["style"])

        self.assertTrue(cost_tracker.tts_calls)
        provider, model, usage = cost_tracker.tts_calls[0]
        self.assertEqual(provider, "gemini")
        self.assertEqual(model, DEFAULT_GEMINI38_TTS_MODEL)
        self.assertGreater(usage.get("audio_seconds", 0.0), 0.0)

    def test_clone_voice_is_not_supported(self) -> None:
        wrapper, _requests, _cost = self._build_wrapper(_wav_bytes(0.5))
        with self.assertRaises(NotImplementedError):
            wrapper.clone_voice("reference.wav", "voice_x")

    def test_pricing_catalog_contains_38_models(self) -> None:
        models = DEFAULT_PRICING_CATALOG["tts"]["gemini"]["models"]
        self.assertEqual(models[DEFAULT_GEMINI38_TTS_MODEL]["output_per_1m_tokens"], 18.00)
        self.assertEqual(models[DEFAULT_GEMINI38_TTS_FALLBACK_MODEL]["output_per_1m_tokens"], 12.00)
        self.assertEqual(models[DEFAULT_GEMINI38_TTS_MODEL]["audio_output_tokens_per_second"], 25.0)

    def test_context_quote_is_style_only_and_not_spoken(self) -> None:
        wrapper, requests, _cost = self._build_wrapper(_wav_bytes(1.0))
        output_path = self.tmp_path / "context.wav"
        quote = 'Replying to SPEAKER_B: "The server room went dark at midnight." — match the conversational energy'
        segment = TTSSegmentData(
            speaker="SPEAKER_A",
            text="Then we heard a fan spin up.",
            context_style=quote,
            output_path=str(output_path),
        )

        with patch.object(mod.AudioFileUtils, "get_audio_duration_seconds", return_value=1.0):
            wrapper.synthesize([segment])

        content = requests[0]["input"][0]["content"][0]
        self.assertNotIn("server room", content["text"])
        style = content["annotations"][0]["style"]
        self.assertIn("server room", style)
        self.assertIn("Replying to SPEAKER_B", style)

    def test_cache_key_includes_context_style(self) -> None:
        wrapper = Gemini38TTSWrapper(enable_voice_matching=False, enable_content_validation=False)
        base = {"speaker": "SPEAKER_A", "text": "Hello there", "voice": "Kore"}
        without_context = TTSSegmentData(**base)
        with_context = TTSSegmentData(**base, context_style='Replying to B: "previous line"')

        self.assertNotEqual(
            wrapper._get_cache_key(without_context, "en"),
            wrapper._get_cache_key(with_context, "en"),
        )
        self.assertEqual(
            wrapper._get_cache_key(without_context, "en"),
            wrapper._get_cache_key(TTSSegmentData(**base), "en"),
        )

    def test_content_validation_flags_context_quote_leak(self) -> None:
        class _StubValidator:
            def __init__(self, text: str) -> None:
                self.text = text

            def transcribe_text(self, audio_path: str) -> str:
                return self.text

        wrapper = Gemini38TTSWrapper(enable_voice_matching=False, enable_content_validation=True)
        quote = 'Replying to SPEAKER_B: "the server room went dark at midnight" — match energy'
        expected = "Then we heard a fan spin up."

        wrapper._content_validator = _StubValidator(
            "The server room went dark at midnight. Then we heard a fan spin up."
        )
        valid, reason, _score = wrapper._validate_segment_content(
            "dummy.wav", expected, context_style=quote
        )
        self.assertFalse(valid)
        self.assertIn("quote leak", reason.lower())

        wrapper._content_validator = _StubValidator(expected)
        valid, reason, _score = wrapper._validate_segment_content(
            "dummy.wav", expected, context_style=quote
        )
        self.assertTrue(valid, reason)


if __name__ == "__main__":
    unittest.main()
