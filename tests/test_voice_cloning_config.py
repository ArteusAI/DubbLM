from __future__ import annotations

import unittest

from src.tts.openrouter_voice_catalog import (
    FISH_DRAMA_TTS_MODEL,
    resolve_voice_cloning_enabled,
)


class VoiceCloningConfigTests(unittest.TestCase):
    def test_drama_3_enables_cloning_when_unset(self) -> None:
        self.assertTrue(resolve_voice_cloning_enabled(None, FISH_DRAMA_TTS_MODEL))

    def test_explicit_false_is_respected(self) -> None:
        self.assertFalse(
            resolve_voice_cloning_enabled(False, FISH_DRAMA_TTS_MODEL)
        )

    def test_other_models_do_not_enable_cloning(self) -> None:
        self.assertFalse(
            resolve_voice_cloning_enabled(None, "qwen/qwen-audio-3.0-tts-flash")
        )


if __name__ == "__main__":
    unittest.main()
