from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.api.services.project_manager import ProjectManager


class ProjectManagerTests(unittest.TestCase):
    def _make_manager(self, base_dir: Path) -> ProjectManager:
        manager = object.__new__(ProjectManager)
        manager.base_dir = base_dir
        return manager

    def test_cleanup_artifacts_preserves_tts_cache_when_requested(self) -> None:
        with TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "proj"
            artifacts_dir = base_dir / "artifacts"
            cache_dir = base_dir / "cache"

            (artifacts_dir / "audio").mkdir(parents=True, exist_ok=True)
            (cache_dir / "abc123" / "segment_synthesis").mkdir(parents=True, exist_ok=True)
            (cache_dir / "abc123" / "synthesized_speech").mkdir(parents=True, exist_ok=True)
            (cache_dir / "abc123" / "translation").mkdir(parents=True, exist_ok=True)

            (artifacts_dir / "audio" / "chunk.wav").write_bytes(b"a")
            (cache_dir / "abc123" / "segment_synthesis" / "seg.wav").write_bytes(b"a")
            (cache_dir / "abc123" / "synthesized_speech" / "mix.wav").write_bytes(b"a")
            (cache_dir / "abc123" / "translation" / "segments.pkl").write_bytes(b"a")

            pm = self._make_manager(base_dir)
            pm.cleanup_artifacts(preserve_tts_cache=True)

            self.assertFalse(artifacts_dir.exists())
            self.assertTrue((cache_dir / "abc123" / "segment_synthesis" / "seg.wav").exists())
            self.assertTrue((cache_dir / "abc123" / "synthesized_speech" / "mix.wav").exists())
            self.assertFalse((cache_dir / "abc123" / "translation").exists())

    def test_cleanup_artifacts_removes_all_cache_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "proj"
            artifacts_dir = base_dir / "artifacts"
            cache_dir = base_dir / "cache"

            artifacts_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

            pm = self._make_manager(base_dir)
            pm.cleanup_artifacts()

            self.assertFalse(artifacts_dir.exists())
            self.assertFalse(cache_dir.exists())

    def test_result_video_path_converts_webm_source_to_mp4(self) -> None:
        with TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "proj"
            uploads_dir = base_dir / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            (uploads_dir / "clip.webm").write_bytes(b"v")

            pm = self._make_manager(base_dir)
            path = pm.get_result_video_path("ru")

            self.assertEqual(path.name, "clip_ru.mp4")
            self.assertEqual(path.parent, base_dir / "results")

    def test_result_video_path_preserves_mp4_source_extension(self) -> None:
        with TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "proj"
            uploads_dir = base_dir / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            (uploads_dir / "clip.MP4").write_bytes(b"v")

            pm = self._make_manager(base_dir)
            path = pm.get_result_video_path("ru")

            self.assertEqual(path.name, "clip_ru.mp4")


if __name__ == "__main__":
    unittest.main()
