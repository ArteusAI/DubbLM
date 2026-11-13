"""Round-trip tests for the per-segment delivery analysis (emotion fields)."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

# The worker module pulls in the audio stack (librosa) at import time. The
# delivery analysis does not use it, so a stub keeps the test runnable on
# machines without the audio extras.
if "librosa" not in sys.modules:
    try:  # pragma: no cover - depends on the environment
        import librosa  # noqa: F401
    except ImportError:  # pragma: no cover - depends on the environment
        import importlib.machinery

        stub = types.ModuleType("librosa")
        stub.__spec__ = importlib.machinery.ModuleSpec("librosa", loader=None)
        stub.load = lambda *args, **kwargs: (None, 16000)
        sys.modules["librosa"] = stub

from src.api import config as api_config
from src.api.database import session as db_session
from src.api.database.models import Project, Segment
from src.api.workers.tasks import (
    _build_speaker_voice_profiles,
    _save_segments_to_db,
    _save_speaker_voice_profiles,
    _segment_to_dubbing_format,
    _update_segment_emotions_in_db,
)


class SegmentEmotionPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "test.db")
        self._env = mock.patch.dict(
            os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}
        )
        self._env.start()
        # The settings and the engine are lazy singletons: force fresh ones
        # for the temp DB.
        api_config._settings = None
        db_session._engine = None
        db_session._SessionLocal = None
        db_session.init_db()

    def tearDown(self) -> None:
        db_session._engine = None
        db_session._SessionLocal = None
        api_config._settings = None
        self._env.stop()
        self._tmpdir.cleanup()

    def _create_project(self) -> str:
        db = db_session.get_db_session()
        try:
            project = Project(id="proj_test", name="test", status="DRAFT", config={})
            db.add(project)
            db.commit()
            return project.id
        finally:
            db.close()

    def test_emotion_survives_the_database_round_trip(self) -> None:
        project_id = self._create_project()
        _save_segments_to_db(
            project_id,
            [
                {
                    "speaker": "SPEAKER_A",
                    "start": 0.0,
                    "end": 1.5,
                    "text": "We are not ready.",
                    "translation": "Мы не готовы.",
                    "emotion": "tense",
                    "emotion_intensity": 0.8,
                    "emotion_direction": "low, controlled, rising urgency",
                },
                {
                    "speaker": "SPEAKER_B",
                    "start": 1.5,
                    "end": 3.0,
                    "text": "Are you sure?",
                    "translation": "Ты уверен?",
                    "emotion": "neutral",
                    "emotion_intensity": 0.0,
                    "emotion_direction": "",
                },
            ],
        )

        db = db_session.get_db_session()
        try:
            segments = (
                db.query(Segment)
                .filter(Segment.project_id == project_id)
                .order_by(Segment.sequence)
                .all()
            )
            self.assertEqual(len(segments), 2)
            self.assertEqual(segments[0].emotion, "tense")
            self.assertAlmostEqual(segments[0].emotion_intensity, 0.8)
            self.assertEqual(
                segments[0].emotion_direction, "low, controlled, rising urgency"
            )

            dubbing_format = _segment_to_dubbing_format(segments[0])
            self.assertEqual(dubbing_format["emotion"], "tense")
            self.assertEqual(
                dubbing_format["emotion_direction"], "low, controlled, rising urgency"
            )
            self.assertEqual(dubbing_format["translation"], "Мы не готовы.")
        finally:
            db.close()

    def test_emotions_can_be_updated_after_the_fact(self) -> None:
        project_id = self._create_project()
        _save_segments_to_db(
            project_id,
            [
                {
                    "speaker": "SPEAKER_A",
                    "start": 0.0,
                    "end": 1.5,
                    "text": "We are not ready.",
                    "translation": "Мы не готовы.",
                }
            ],
        )

        _update_segment_emotions_in_db(
            project_id,
            [
                {
                    "emotion": "furious",
                    "emotion_intensity": 0.95,
                    "emotion_direction": "loud, breathless",
                }
            ],
        )

        db = db_session.get_db_session()
        try:
            segment = (
                db.query(Segment)
                .filter(Segment.project_id == project_id)
                .order_by(Segment.sequence)
                .first()
            )
            self.assertEqual(segment.emotion, "furious")
            self.assertEqual(segment.emotion_direction, "loud, breathless")
        finally:
            db.close()


class VoiceAdvicePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "test.db")
        self._env = mock.patch.dict(
            os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}
        )
        self._env.start()
        api_config._settings = None
        db_session._engine = None
        db_session._SessionLocal = None
        db_session.init_db()

    def tearDown(self) -> None:
        db_session._engine = None
        db_session._SessionLocal = None
        api_config._settings = None
        self._env.stop()
        self._tmpdir.cleanup()

    def _create_project(self) -> str:
        db = db_session.get_db_session()
        try:
            project = Project(id="proj_voice", name="test", status="DRAFT", config={})
            db.add(project)
            db.commit()
            return project.id
        finally:
            db.close()

    def test_the_profiles_are_built_from_the_analysis(self) -> None:
        from src.dubbing.video.gemini_emotion_analyzer import DeliveryAnalysis

        analysis = DeliveryAnalysis(
            lines=[{"index": 0, "emotion": "tense", "intensity": 0.5}],
            speakers=[
                {
                    "speaker": "SPEAKER_A",
                    "timbre": "deep, resonant baritone",
                    "age": "adult",
                    "energy": 0.5,
                    "accent": "neutral",
                }
            ],
        )
        config_data = {
            "speakerMetadata": {
                "SPEAKER_A": {"inferredGender": "male", "inferredConfidence": 0.9}
            }
        }
        profiles = _build_speaker_voice_profiles(analysis, config_data)
        self.assertIn("SPEAKER_A", profiles)
        self.assertEqual(profiles["SPEAKER_A"]["timbre"], "deep, resonant baritone")
        # The gender of the pipeline wins over the model.
        self.assertEqual(profiles["SPEAKER_A"]["gender"], "male")
        self.assertEqual(profiles["SPEAKER_A"]["age"], "adult")

    def test_the_profiles_are_stored_in_the_project_config(self) -> None:
        project_id = self._create_project()
        _save_speaker_voice_profiles(
            project_id,
            {
                "SPEAKER_A": {
                    "speaker": "SPEAKER_A",
                    "timbre": "low, slightly raspy",
                    "age": "adult",
                    "energy": 0.4,
                    "accent": "neutral",
                    "gender": "male",
                }
            },
        )

        db = db_session.get_db_session()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            profiles = (project.config or {}).get("speakerVoiceProfiles")
            self.assertIsNotNone(profiles)
            self.assertEqual(profiles["SPEAKER_A"]["timbre"], "low, slightly raspy")
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()