import unittest

from src.tts.qwen_emotion import (
    apply_leading_tag,
    prepare_qwen_speech_text,
    sanitize_qwen_markup,
    spoken_tag_leak_hits,
    strip_markup_tags,
    style_prompt_to_instruction,
)


class QwenEmotionTests(unittest.TestCase):
    def test_sanitize_keeps_demo_tags_only(self) -> None:
        text = "[excited] Hello [short pause] world [angry]!"
        cleaned = sanitize_qwen_markup(text)
        self.assertIn("[excited]", cleaned)
        self.assertIn("[angry]", cleaned)
        self.assertNotIn("short pause", cleaned)

    def test_prepare_with_emotion_tag(self) -> None:
        out = prepare_qwen_speech_text("Hello there.", emotion="angry")
        self.assertTrue(out.startswith("[angry]"))
        self.assertIn("Hello there.", out)

    def test_prepare_with_style_prompt_prefix(self) -> None:
        out = prepare_qwen_speech_text(
            "Hello there.",
            style_prompt="Urgent delivery:",
            emotion="angry",
        )
        self.assertTrue(out.startswith("Speak with this delivery: Urgent delivery."))
        self.assertIn("[angry]", out)
        self.assertIn("Hello there.", out)
        self.assertLess(out.index("Speak with this delivery"), out.index("[angry]"))
        self.assertLess(out.index("[angry]"), out.index("Hello there."))

    def test_strip_markup(self) -> None:
        self.assertEqual(strip_markup_tags("[angry] Hi  there"), "Hi there")

    def test_spoken_tag_leak_detection(self) -> None:
        speech = "[angry] Please leave now."
        # model spoke the control word
        hits = spoken_tag_leak_hits("angry please leave now", speech)
        self.assertIn("angry", hits)
        # clean delivery
        hits_ok = spoken_tag_leak_hits("please leave now", speech)
        self.assertEqual(hits_ok, [])

    def test_style_prompt_to_instruction(self) -> None:
        self.assertEqual(
            style_prompt_to_instruction("Warm narrator:"),
            "Speak with this delivery: Warm narrator.",
        )
        self.assertTrue(style_prompt_to_instruction("Say this calmly").startswith("Say this calmly"))

    def test_apply_leading_tag_idempotent(self) -> None:
        text = apply_leading_tag("Hi", "excited")
        again = apply_leading_tag(text, "excited")
        self.assertEqual(again.count("[excited]"), 1)


if __name__ == "__main__":
    unittest.main()
