from __future__ import annotations

import unittest
from unittest import mock

from src.transcription.assemblyai_transcriber import AssemblyAITranscriber
from src.transcription import gemini_speaker_labeler as gsl


def _w(word: str, start: float, end: float) -> dict:
    return {"word": word, "start": start, "end": end, "confidence": 0.9, "speaker": "SPEAKER_A"}


class AlignLabeledSegmentsTests(unittest.TestCase):
    def test_split_uses_asr_word_spans(self):
        words = [
            _w("What", 17.70, 18.00),
            _w("are", 18.00, 18.20),
            _w("you", 18.20, 18.40),
            _w("seeing?", 18.40, 19.80),
            _w("I'm", 19.80, 20.10),
            _w("running.", 20.10, 21.10),
        ]
        labeled = [
            {"start": 17.70, "end": 19.80, "speaker": "SPEAKER_A", "text": "What are you seeing?"},
            {"start": 19.80, "end": 21.10, "speaker": "SPEAKER_B", "text": "I'm running."},
        ]
        aligned = gsl.align_labeled_segments_to_words(labeled, words)
        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0]["speaker"], "SPEAKER_A")
        self.assertAlmostEqual(aligned[0]["start"], 17.70)
        self.assertAlmostEqual(aligned[0]["end"], 19.80)
        self.assertEqual(aligned[0]["text"], "What are you seeing?")
        self.assertEqual(aligned[1]["speaker"], "SPEAKER_B")
        self.assertAlmostEqual(aligned[1]["start"], 19.80)
        self.assertAlmostEqual(aligned[1]["end"], 21.10)
        self.assertTrue(all(w["speaker"] == "SPEAKER_B" for w in aligned[1]["words"]))

    def test_drops_segment_without_asr_match(self):
        words = [_w("Hello", 0.0, 0.5), _w("world", 0.5, 1.0)]
        labeled = [
            {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_A", "text": "Hello"},
            {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_B", "text": "zzz qqq"},
        ]
        aligned = gsl.align_labeled_segments_to_words(labeled, words)
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0]["text"], "Hello")


class ParseLabelResponseTests(unittest.TestCase):
    def test_accepts_wrapped_and_bare_json(self):
        wrapped = gsl._parse_label_response(
            '{"segments":[{"start":0,"end":1,"speaker":"SPEAKER_A","text":"Hi"}]}'
        )
        self.assertEqual(len(wrapped), 1)
        bare = gsl._parse_label_response('[{"start":0,"end":1,"speaker":"SPEAKER_B","text":"Yo"}]')
        self.assertEqual(len(bare), 1)
        self.assertEqual(bare[0]["speaker"], "SPEAKER_B")

    def test_skips_invalid_items(self):
        parsed = gsl._parse_label_response(
            '[{"start":0,"end":1,"speaker":"","text":"Hi"},{"speaker":"SPEAKER_A"}]'
        )
        self.assertEqual(parsed, [])


class ApplyGeminiLabelingTests(unittest.TestCase):
    def _transcriber(self):
        tr = object.__new__(AssemblyAITranscriber)
        tr.gemini_speaker_labeling_model = "gemini-2.5-pro"
        tr.gemini_speaker_labeling_max_speakers = 4
        return tr

    def test_replaces_transcription_with_labeled_segments(self):
        transcription = [
            {
                "speaker": "SPEAKER_A",
                "start": 0.0,
                "end": 1.0,
                "text": "One two",
                "words": [_w("One", 0.0, 0.5), _w("two", 0.5, 1.0)],
            }
        ]
        labeled = [
            {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_X", "text": "One"},
            {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_Y", "text": "two"},
        ]
        with mock.patch.object(gsl, "label_speakers_with_gemini", return_value=labeled):
            result = self._transcriber()._apply_gemini_speaker_labeling("audio.wav", transcription)
        self.assertEqual([s["speaker"] for s in result], ["SPEAKER_X", "SPEAKER_Y"])

    def test_keeps_original_on_failure(self):
        transcription = [
            {
                "speaker": "SPEAKER_A",
                "start": 0.0,
                "end": 1.0,
                "text": "One two",
                "words": [_w("One", 0.0, 0.5), _w("two", 0.5, 1.0)],
            }
        ]
        with mock.patch.object(gsl, "label_speakers_with_gemini", side_effect=RuntimeError("boom")):
            result = self._transcriber()._apply_gemini_speaker_labeling("audio.wav", transcription)
        self.assertEqual(result, transcription)


if __name__ == "__main__":
    unittest.main()
