from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    from src.tts.gemini38_tts_wrapper import Gemini38APIClient, Gemini38TTSConfig

    TTS_RUNTIME_AVAILABLE = True
except Exception:  # pragma: no cover - pydub/google sdk may be absent in a minimal venv
    Gemini38APIClient = None  # type: ignore[assignment]
    Gemini38TTSConfig = None  # type: ignore[assignment]
    TTS_RUNTIME_AVAILABLE = False


@unittest.skipUnless(TTS_RUNTIME_AVAILABLE, "TTS runtime is not installed")
class OpenRouterTTSTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Gemini38TTSConfig(
            model="google/gemini-3.8-flash-tts",
            tts_transport="openrouter",
        )
        self.client = Gemini38APIClient(self.config)
        self.client.client = "openrouter"

    def test_model_prefix_is_added(self) -> None:
        self.assertEqual(
            self.client._normalize_openrouter_model("gemini-3.8-flash-tts"),
            "google/gemini-3.8-flash-tts",
        )
        self.assertEqual(
            self.client._normalize_openrouter_model("google/gemini-3.8-flash-lite-tts"),
            "google/gemini-3.8-flash-lite-tts",
        )

    def test_openrouter_request_passes_voice_and_style(self) -> None:
        captured = {}

        def fake_synthesize(text, **kwargs):
            captured["text"] = text
            captured.update(kwargs)
            return b"\x01\x02" * 1000, {"content-type": "audio/pcm"}

        with patch("src.utils.openrouter_client.synthesize_speech", side_effect=fake_synthesize):
            audio = self.client._synthesize_openrouter(
                "Привет",
                "warm and clear",
                "Puck",
                "google/gemini-3.8-flash-tts",
            )

        self.assertEqual(len(audio), 2000)
        self.assertEqual(captured["model"], "google/gemini-3.8-flash-tts")
        self.assertEqual(captured["voice"], "Puck")
        self.assertEqual(captured["instructions"], "warm and clear")
        self.assertEqual(captured["response_format"], "pcm")

    def test_synthesize_chunk_uses_openrouter_transport(self) -> None:
        with patch.object(self.client, "_synthesize_openrouter", return_value=b"\x00\x01" * 100) as call:
            audio = self.client.synthesize_chunk("Hello", "calm", "Kore")
        self.assertEqual(len(audio), 200)
        call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
