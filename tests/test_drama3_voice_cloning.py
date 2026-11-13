from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydub.generators import Sine

from src.tts.models import TTSSegmentData
from src.tts.openrouter_tts_wrapper import (
    OpenRouterTTSWrapper,
    VoiceCloneRequestError,
)
from src.tts.openrouter_voice_catalog import FISH_DRAMA_TTS_MODEL
from src.tts.voice_clone_reference import VoiceCloneReference


def _reference(root: Path) -> VoiceCloneReference:
    path = root / "speaker_reference.wav"
    tone = Sine(220).to_audio_segment(duration=2000, volume=-12)
    tone = tone.set_frame_rate(24000).set_channels(1).set_sample_width(2)
    tone.export(str(path), format="wav")
    return VoiceCloneReference(
        speaker="SPEAKER_A",
        reference_path=str(path),
        duration_seconds=2.0,
        clip_ranges=[(0.0, 2.0)],
        source_path=str(path),
        transcript="This is a clean reference.",
        cache_token="abc123",
    )


def _wrapper() -> OpenRouterTTSWrapper:
    return OpenRouterTTSWrapper(
        model=FISH_DRAMA_TTS_MODEL,
        target_language="ru",
        enable_voice_matching=False,
        enable_voice_cloning=True,
        enable_content_validation=False,
    )


class Drama3VoiceCloningTests(unittest.TestCase):
    def test_clone_request_sends_reference_and_omits_voice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = _wrapper()
            reference = _reference(Path(tmp))

            with patch("src.tts.openrouter_tts_wrapper.requests.post") as post_mock:
                post_mock.return_value.status_code = 200
                post_mock.return_value.content = b"audio-bytes"
                audio = wrapper._request_speech_bytes(
                    "Hello there.",
                    "ignored-library-voice",
                    1.0,
                    clone_reference=reference,
                )

            self.assertEqual(audio, b"audio-bytes")
            payload = post_mock.call_args.kwargs["json"]
            self.assertNotIn("voice", payload)
            self.assertEqual(payload["model"], FISH_DRAMA_TTS_MODEL)
            self.assertEqual(len(payload["input_references"]), 2)
            self.assertEqual(payload["input_references"][0]["type"], "input_audio")
            self.assertTrue(
                payload["input_references"][0]["input_audio"]["data"].startswith(
                    "data:audio/"
                )
            )
            self.assertEqual(
                payload["input_references"][1],
                {"type": "text", "text": "This is a clean reference."},
            )

    def test_manual_voice_mapping_overrides_cloning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = _wrapper()
            wrapper.set_voice_clone_references({"SPEAKER_A": _reference(Path(tmp))})
            segment = TTSSegmentData(
                speaker="SPEAKER_A",
                text="Hello",
                voice="c" * 32,
            )
            self.assertIsNone(wrapper._clone_reference_for_segment(segment))

    def test_clone_reference_is_used_without_manual_voice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = _wrapper()
            reference = _reference(Path(tmp))
            wrapper.set_voice_clone_references({"SPEAKER_A": reference})
            segment = TTSSegmentData(speaker="SPEAKER_A", text="Hello")
            self.assertIs(wrapper._clone_reference_for_segment(segment), reference)

    def test_clone_failure_falls_back_to_library_voice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = _wrapper()
            wrapper.client = object()  # synthesize() only guards against None
            reference = _reference(Path(tmp))
            wrapper.set_voice_clone_references({"SPEAKER_A": reference})
            segment = TTSSegmentData(
                speaker="SPEAKER_A",
                text="Hello",
                output_path=str(Path(tmp) / "segment.wav"),
            )

            with patch.object(wrapper, "_synthesize_single_segment") as synth_mock:
                synth_mock.side_effect = [
                    VoiceCloneRequestError(400, "reference rejected"),
                    None,
                ]
                wrapper.synthesize([segment])

            self.assertEqual(synth_mock.call_count, 2)
            self.assertIs(
                synth_mock.call_args_list[0].kwargs["clone_reference"],
                reference,
            )
            self.assertIsNone(
                synth_mock.call_args_list[1].kwargs["clone_reference"]
            )
            self.assertIn("SPEAKER_A", wrapper._voice_clone_disabled_speakers)


if __name__ == "__main__":
    unittest.main()
