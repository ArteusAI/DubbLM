from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.dubbing.audio.audio_processor import AudioProcessor, NoAudioStreamError


class ExtractAudioTests(unittest.TestCase):
    """Regression tests for ffmpeg-based audio extraction.

    Full extraction used to go through pydub, which decodes the whole track into
    memory and raises ``CouldntDecodeError("Unable to process >4GB files")`` for
    long recordings. It must always use ffmpeg and produce 16 kHz mono PCM.
    """

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        (self.tmpdir / "artifacts" / "audio").mkdir(parents=True)
        self.video = self.tmpdir / "source.webm"
        self.video.write_bytes(b"fake video")
        self._original_cwd = Path.cwd()
        # extract_audio writes to a relative path ("artifacts/audio/source.wav").
        os.chdir(self.tmpdir)
        self.addCleanup(os.chdir, self._original_cwd)

        with mock.patch.object(AudioProcessor, "_setup_directories"):
            self.processor = AudioProcessor(mock.MagicMock(), mock.MagicMock())
        self.processor._has_audio_stream = mock.Mock(return_value=True)
        self.processor._determine_video_duration = mock.Mock()

    def _fake_ffmpeg(self, cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"RIFF")
        return mock.Mock(returncode=0, stderr=b"")

    def test_full_extraction_uses_ffmpeg_with_16khz_mono(self) -> None:
        with mock.patch(
            "src.dubbing.audio.audio_processor.subprocess.run",
            side_effect=self._fake_ffmpeg,
        ) as run:
            audio_file = self.processor.extract_audio(str(self.video))

        self.assertEqual(audio_file, "artifacts/audio/source.wav")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("-i", cmd)
        self.assertEqual(cmd[cmd.index("-i") + 1], str(self.video))
        self.assertNotIn("-ss", cmd)
        self.assertNotIn("-t", cmd)
        self.assertEqual(cmd[cmd.index("-acodec") + 1], "pcm_s16le")
        self.assertEqual(cmd[cmd.index("-ar") + 1], "16000")
        self.assertEqual(cmd[cmd.index("-ac") + 1], "1")
        self.assertEqual(cmd[-1], "artifacts/audio/source.wav")

    def test_trimmed_extraction_passes_seek_and_duration(self) -> None:
        with mock.patch(
            "src.dubbing.audio.audio_processor.subprocess.run",
            side_effect=self._fake_ffmpeg,
        ) as run:
            self.processor.extract_audio(str(self.video), start_time=12.5, duration=60)

        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-ss") + 1], "12.5")
        self.assertEqual(cmd[cmd.index("-t") + 1], "60")

    def test_extraction_failure_raises_runtime_error(self) -> None:
        with mock.patch(
            "src.dubbing.audio.audio_processor.subprocess.run",
            return_value=mock.Mock(returncode=1, stderr=b"boom"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.processor.extract_audio(str(self.video))

        self.assertIn("boom", str(ctx.exception))

    def test_missing_audio_stream_raises(self) -> None:
        self.processor._has_audio_stream = mock.Mock(return_value=False)

        with mock.patch("src.dubbing.audio.audio_processor.subprocess.run") as run:
            with self.assertRaises(NoAudioStreamError):
                self.processor.extract_audio(str(self.video))

        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
