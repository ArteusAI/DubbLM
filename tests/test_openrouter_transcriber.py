from __future__ import annotations

import unittest

import numpy as np

try:
    from src.transcription.openrouter_transcriber import OpenRouterTranscriber

    TRANSCRIBER_AVAILABLE = True
except Exception:  # pragma: no cover - torch/pyannote may be absent in a minimal venv
    OpenRouterTranscriber = None  # type: ignore[assignment]
    TRANSCRIBER_AVAILABLE = False


@unittest.skipUnless(TRANSCRIBER_AVAILABLE, "transcription runtime is not installed")
class OpenRouterTranscriberTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transcriber = OpenRouterTranscriber(
            source_language="auto",
            model="microsoft/mai-transcribe-2",
            chunk_seconds=600,
            overlap_seconds=15,
        )

    def test_normalise_words_offsets_chunk_start(self) -> None:
        words = self.transcriber._normalise_words(
            [
                {"word": "Hello", "start": 1.0, "end": 1.4, "speaker": 0},
                {"word": "world", "start": 0.5, "end": 0.7, "speaker": 1},
            ],
            chunk_start=100.0,
        )
        self.assertEqual([word["word"] for word in words], ["world", "Hello"])
        self.assertEqual(words[0]["start"], 100.5)
        self.assertEqual(words[1]["end"], 101.4)

    def test_build_segments_splits_speaker_changes(self) -> None:
        words = [
            {"word": "Hello.", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_A"},
            {"word": "Hi", "start": 0.6, "end": 0.8, "speaker": "SPEAKER_B"},
            {"word": "there.", "start": 0.8, "end": 1.1, "speaker": "SPEAKER_B"},
        ]
        segments = self.transcriber._build_segments(words)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["speaker"], "SPEAKER_A")
        self.assertEqual(segments[1]["speaker"], "SPEAKER_B")
        self.assertEqual(segments[1]["text"], "Hi there.")

    def test_merge_chunk_words_dedupes_overlap(self) -> None:
        chunks = [
            {
                "index": 0,
                "start": 0.0,
                "end": 70.0,
                "words": [
                    {"word": "one", "start": 65.0, "end": 65.2, "speaker": "SPEAKER_0"},
                ],
            },
            {
                "index": 1,
                "start": 60.0,
                "end": 130.0,
                "words": [
                    {"word": "one", "start": 65.0, "end": 65.2, "speaker": "SPEAKER_0"},
                    {"word": "two", "start": 75.0, "end": 75.2, "speaker": "SPEAKER_0"},
                ],
            },
        ]
        mapping = {(0, "SPEAKER_0"): "SPEAKER_A", (1, "SPEAKER_0"): "SPEAKER_A"}
        merged = self.transcriber._merge_chunk_words(chunks, mapping)
        self.assertEqual([word["word"] for word in merged], ["one", "two"])

    def test_embedding_clusters_merge_similar_speakers(self) -> None:
        records = [
            {"chunk_index": 0, "local_speaker": "SPEAKER_0", "embedding": np.array([1.0, 0.0]), "weight": 10, "ranges": [(0.0, 10.0)]},
            {"chunk_index": 1, "local_speaker": "SPEAKER_0", "embedding": np.array([0.99, 0.01]), "weight": 10, "ranges": [(60.0, 70.0)]},
            {"chunk_index": 1, "local_speaker": "SPEAKER_1", "embedding": np.array([0.0, 1.0]), "weight": 10, "ranges": [(80.0, 90.0)]},
        ]
        clusters = self.transcriber._cluster_embeddings(records, expected_speakers=None)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0]) + len(clusters[1]), 3)

    def test_embedding_clusters_respect_expected_speaker_count(self) -> None:
        records = [
            {"chunk_index": 0, "local_speaker": "SPEAKER_0", "embedding": np.array([1.0, 0.0]), "weight": 10, "ranges": [(0.0, 10.0)]},
            {"chunk_index": 1, "local_speaker": "SPEAKER_0", "embedding": np.array([0.8, 0.2]), "weight": 10, "ranges": [(60.0, 70.0)]},
            {"chunk_index": 1, "local_speaker": "SPEAKER_1", "embedding": np.array([0.0, 1.0]), "weight": 10, "ranges": [(80.0, 90.0)]},
        ]
        clusters = self.transcriber._cluster_embeddings(records, expected_speakers=2)
        self.assertEqual(len(clusters), 2)


if __name__ == "__main__":
    unittest.main()
