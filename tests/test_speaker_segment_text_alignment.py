from __future__ import annotations

import unittest

from src.tts.speaker_segment_diarizer.diarizer import SpeakerSegmentDiarizer, _WordToken
from src.tts.speaker_segment_diarizer.models import DialogueLine


def _tokens_from_text(text: str, *, start_ms: int = 0, step_ms: int = 120) -> list[_WordToken]:
    tokens: list[_WordToken] = []
    for idx, token in enumerate(SpeakerSegmentDiarizer._tokens_for_match(text)):
        tokens.append(_WordToken(
            text=token,
            raw_text=token,
            start_ms=start_ms + idx * step_ms,
            end_ms=start_ms + (idx + 1) * step_ms,
            confidence=0.95,
        ))
    return tokens


class SpeakerSegmentTextAlignmentTests(unittest.TestCase):
    def test_aligns_adjacent_lines_without_speaker_labels(self) -> None:
        first = (
            "Обучение с подкреплением ужасная штука. Мы строим не животных, "
            "а призраков. Гении сегодняшнего дня едва царапают поверхность "
            "того, на что способен человеческий разум."
        )
        second = (
            "Сегодня я беседую с Андреем Карпати. Андрей, почему вы говорите, "
            "что нас ждет именно десятилетие агентов, а не год агентов?"
        )
        word_tokens = _tokens_from_text(f"{first} {second}")

        segments = SpeakerSegmentDiarizer._align_lines_to_word_tokens(
            [
                DialogueLine(speaker="SPEAKER_A", text=first),
                DialogueLine(speaker="SPEAKER_B", text=second.replace("ждет", "ждёт")),
            ],
            word_tokens,
        )

        self.assertEqual([segment.speaker for segment in segments], ["SPEAKER_A", "SPEAKER_B"])
        self.assertEqual([segment.asr_speaker_label for segment in segments], ["text", "text"])
        self.assertLess(segments[0].end_ms, segments[1].end_ms)
        self.assertNotIn("сегодня", segments[0].asr_text.split())
        self.assertTrue(segments[1].asr_text.startswith("сегодня"))

    def test_missing_line_does_not_make_previous_span_consume_the_rest(self) -> None:
        first = "Первая реплика достаточно длинная и хорошо находится в транскрипте."
        missing_second = "Этой второй реплики в аудио вообще нет."
        word_tokens = _tokens_from_text(first)

        segments = SpeakerSegmentDiarizer._align_lines_to_word_tokens(
            [
                DialogueLine(speaker="SPEAKER_A", text=first),
                DialogueLine(speaker="SPEAKER_B", text=missing_second),
            ],
            word_tokens,
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].matched_line_idx, 0)
        self.assertEqual(segments[0].speaker, "SPEAKER_A")


if __name__ == "__main__":
    unittest.main()
