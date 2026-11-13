from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.dubbing.core.config import DubbingConfig
from src.dubbing.core.segment_optimizer import SegmentOptimizer
from src.transcription.assemblyai_transcriber import AssemblyAITranscriber
from src.translation.llm_translator import LLMTranslator


def _word(text: str, start_ms: int, end_ms: int, speaker: str | None):
    return SimpleNamespace(text=text, start=start_ms, end=end_ms, speaker=speaker, confidence=0.9)


def _utterance(words, speaker: str = "A", text: str = ""):
    return SimpleNamespace(
        words=words,
        speaker=speaker,
        text=text or " ".join(w.text for w in words),
        start=words[0].start if words else 0,
        end=words[-1].end if words else 0,
        confidence=0.9,
    )


class _StubConfig:
    def __init__(self, segments_optimization=None):
        self._cfg = {"segments_optimization": segments_optimization or {}, "tts_system": "gemini38"}

    def get(self, key, default=None):
        return self._cfg.get(key, default)


class AssemblyAIWordSplitTests(unittest.TestCase):
    def setUp(self):
        self.transcriber = object.__new__(AssemblyAITranscriber)

    def test_split_words_groups_by_speaker(self):
        words = [
            _word("Hello", 0, 400, "A"),
            _word("there", 400, 800, "A"),
            _word("Hi", 800, 1100, "B"),
            _word("back", 1100, 1500, "A"),
        ]
        runs = self.transcriber._split_words_by_speaker(words, default_speaker="SPEAKER_A")
        self.assertEqual([len(r) for r in runs], [2, 1, 1])
        self.assertEqual(runs[0][0]["speaker"], "SPEAKER_A")
        self.assertEqual(runs[1][0]["speaker"], "SPEAKER_B")
        self.assertEqual(runs[2][0]["speaker"], "SPEAKER_A")

    def test_split_words_without_speaker_labels_uses_default(self):
        words = [_word("one", 0, 400, None), _word("two", 400, 800, None)]
        runs = self.transcriber._split_words_by_speaker(words, default_speaker="SPEAKER_C")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0][0]["speaker"], "SPEAKER_C")

    def test_process_utterance_splits_on_speaker_change(self):
        transcript = SimpleNamespace(
            utterances=[
                _utterance(
                    [
                        _word("Why", 0, 300, "B"),
                        _word("Hey", 300, 700, "B"),
                        _word("Gretchen", 700, 1400, "B"),
                        _word("I", 1400, 1600, "A"),
                        _word("can't", 1600, 1900, "A"),
                        _word("remember", 1900, 2600, "A"),
                    ],
                    speaker="B",
                )
            ]
        )
        records = self.transcriber._process_assemblyai_transcript(transcript)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["speaker"], "SPEAKER_B")
        self.assertEqual(records[0]["text"], "Why Hey Gretchen")
        self.assertEqual(records[1]["speaker"], "SPEAKER_A")
        self.assertEqual(records[1]["text"], "I can't remember")
        self.assertAlmostEqual(records[1]["start"], 1.4)
        self.assertEqual(records[1]["words"][0]["speaker"], "SPEAKER_A")


class TranscribeOnVocalsConfigTests(unittest.TestCase):
    def test_tri_state_follows_keep_background(self):
        cfg = DubbingConfig()
        cfg.config["transcribe_on_vocals"] = None
        cfg.config["keep_background"] = True
        self.assertTrue(cfg.transcribe_on_vocals_enabled())

        cfg.config["keep_background"] = False
        self.assertFalse(cfg.transcribe_on_vocals_enabled())

    def test_explicit_override_wins(self):
        cfg = DubbingConfig()
        cfg.config["keep_background"] = True
        cfg.config["transcribe_on_vocals"] = False
        self.assertFalse(cfg.transcribe_on_vocals_enabled())

        cfg.config["keep_background"] = False
        cfg.config["transcribe_on_vocals"] = True
        self.assertTrue(cfg.transcribe_on_vocals_enabled())


class TranslationSpeakerImmutabilityTests(unittest.TestCase):
    def test_diarized_speaker_is_preserved(self):
        translator = object.__new__(LLMTranslator)
        segments = [
            {"speaker": "SPEAKER_A", "start": 0.0, "end": 1.0, "text": "Hello"},
        ]
        pairs = [{"speaker": "SPEAKER_B", "text": "Привет"}]
        out = translator._segment_translations([{"segments": segments, "translated_pairs": pairs}])
        self.assertEqual(out[0]["speaker"], "SPEAKER_A")
        self.assertEqual(out[0]["translation"], "Привет")

    def test_speaker_filled_when_missing(self):
        translator = object.__new__(LLMTranslator)
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        pairs = [{"speaker": "SPEAKER_B", "text": "Привет"}]
        out = translator._segment_translations([{"segments": segments, "translated_pairs": pairs}])
        self.assertEqual(out[0]["speaker"], "SPEAKER_B")


class MergeSpeakerGuardTests(unittest.TestCase):
    def _segment(self, speaker, start, end, text, words=None):
        seg = {"speaker": speaker, "start": start, "end": end, "text": text}
        if words is not None:
            seg["words"] = words
        return seg

    def test_merge_blocked_on_word_speaker_boundary(self):
        opt = SegmentOptimizer(_StubConfig())
        a = self._segment("SPEAKER_A", 0.0, 2.0, "one", [{"word": "one", "speaker": "SPEAKER_A"}])
        b = self._segment("SPEAKER_A", 2.05, 4.0, "two", [{"word": "two", "speaker": "SPEAKER_B"}])
        merged = opt.merge_adjacent_segments([a, b], max_gap=0.3, phase="test")
        self.assertEqual(len(merged), 2)

    def test_merge_allowed_for_consistent_words(self):
        opt = SegmentOptimizer(_StubConfig())
        a = self._segment("SPEAKER_A", 0.0, 2.0, "one", [{"word": "one", "speaker": "SPEAKER_A"}])
        b = self._segment("SPEAKER_A", 2.05, 4.0, "two", [{"word": "two", "speaker": "SPEAKER_A"}])
        merged = opt.merge_adjacent_segments([a, b], max_gap=0.3, phase="test")
        self.assertEqual(len(merged), 1)

    def test_merge_can_be_disabled(self):
        opt = SegmentOptimizer(_StubConfig({"merge_same_speaker_segments": False}))
        a = self._segment("SPEAKER_A", 0.0, 2.0, "one")
        b = self._segment("SPEAKER_A", 2.05, 4.0, "two")
        merged = opt.merge_adjacent_segments([a, b], max_gap=0.3, phase="test")
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
