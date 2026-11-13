from __future__ import annotations

import json
import unittest
from unittest import mock

from src.translation.llm_translator import LLMTranslator


def _make_segment(index: int, text: str = "Original", translation: str = "Current translation") -> dict:
    return {
        "speaker": f"SPEAKER_{index:02d}",
        "start": float(index),
        "end": float(index) + 0.8,
        "text": text,
        "translation": translation,
    }


def _make_translator(**kwargs) -> LLMTranslator:
    defaults = {
        "enable_cache": False,
        "enable_llm_editor": True,
        "editor_llm_provider": "openrouter",
        "editor_model_name": "openai/gpt-5.6-terra",
        "editor_reasoning_effort": "xhigh",
        "refinement_persona": "normal",
    }
    defaults.update(kwargs)
    return LLMTranslator(
        **defaults,
    )


class LlmEditorTests(unittest.TestCase):
    def test_emotion_enrichment_prompt_handles_existing_tags(self) -> None:
        translator = _make_translator(enable_emotion_enrichment=True)
        captured_prompts: list[str] = []

        class _StubResponse:
            def __init__(self, text: str) -> None:
                self.text = text

        class _StubLlm:
            def complete(self, prompt: str):
                captured_prompts.append(prompt)
                return _StubResponse(
                    json.dumps(
                        {
                            "translation": "[excited] Current translation [short pause]",
                        }
                    )
                )

        translator.refinement_llm = _StubLlm()
        enriched = translator._enrich_text_with_llm(
            {"translation": "[happy] Current translation"},
            previous_lines=[],
            target_language="ru",
        )

        self.assertEqual(enriched["translation"], "[excited] Current translation [short pause]")
        self.assertEqual(len(captured_prompts), 1)
        self.assertIn("The input may already contain markup tags from an earlier editing step", captured_prompts[0])
        self.assertIn("avoid duplicating or stacking near-identical tags", captured_prompts[0])
        self.assertIn("Do not add [uhm] unless the line clearly implies hesitation", captured_prompts[0])
        self.assertIn("Prefer [short pause], [long pause], or ordinary expressive tags", captured_prompts[0])

    def test_editor_gemini_guidance_discourages_filler_sounds(self) -> None:
        translator = _make_translator(tts_system="gemini")
        guidance = translator._build_editor_tts_guidance(
            [
                _make_segment(0, translation="First line"),
                _make_segment(1, translation="Second line"),
            ]
        )

        self.assertIn("target no more than 2 such filler/breath insertions per speaker in total", guidance)
        self.assertIn("Do not add [uhm] just to make the speech sound conversational", guidance)
        self.assertIn("Prefer pause markers over filler/breath tags for pacing and clarity", guidance)
        self.assertIn("Uninterrupted continuous speech is hard to follow in dubbing", guidance)
        self.assertIn("include at least one pause unless the line is very short", guidance)

    def test_refinement_tts_guidance_preserves_and_enhances_tags(self) -> None:
        translator = _make_translator(tts_system="gemini", segment_stretch="audio")
        guidance = translator._build_refinement_tts_guidance(
            [{"speaker": "SPEAKER_A", "text": "[excited] Hello [short pause]"}]
        )

        self.assertIn("# Gemini TTS delivery guidance", guidance)
        self.assertIn("MUST preserve every existing tag", guidance)
        self.assertIn("You SHOULD enhance delivery", guidance)
        self.assertIn("Do NOT strip, normalize away, or rewrite markup tags", guidance)

    def test_refinement_prompt_includes_tts_markup_guidance(self) -> None:
        translator = _make_translator(
            enable_llm_editor=False,
            tts_system="gemini",
            segment_stretch="audio",
        )
        captured_prompts: list[str] = []

        class _StubResponse:
            text = json.dumps(
                {
                    "translations": [
                        {
                            "speaker": "SPEAKER_A",
                            "text": "[excited] Refined line [short pause]",
                            "very_short": "[excited] Refined",
                            "short": "[excited] Refined line",
                            "long": "[excited] Refined line [medium pause]",
                            "cohesion_with_prev": "normal",
                        }
                    ]
                }
            )

        class _StubLlm:
            def complete(self, prompt: str):
                captured_prompts.append(prompt)
                return _StubResponse()

        translator.refinement_llm = _StubLlm()
        chunk = {
            "segments": [{"speaker": "SPEAKER_A", "text": "Hello world", "start": 0.0, "end": 1.0}],
            "translated_pairs": [
                {"speaker": "SPEAKER_A", "text": "[excited] Hello world [short pause]"}
            ],
        }

        with mock.patch("src.translation.llm_translator.json_repair") as mock_json_repair:
            mock_json_repair.loads.side_effect = json.loads
            translator._refine_translation(
                translated_chunks=[chunk],
                context_info={"domain": "general", "tone": "excited", "themes": []},
                source_language="en",
                target_language="ru",
                dialogue_summary="Test summary",
                debug=False,
            )

        self.assertEqual(len(captured_prompts), 1)
        self.assertIn("# Refinement-specific markup rules", captured_prompts[0])
        self.assertIn("[excited] Hello world [short pause]", captured_prompts[0])
        self.assertIn("MUST preserve every existing tag", captured_prompts[0])

    def test_translation_prompt_adds_gemini_delivery_guidance_without_editor(self) -> None:
        translator = _make_translator(
            enable_llm_editor=False,
            refinement_persona="none",
            tts_system="gemini",
        )
        prompts: list[str] = []

        class _StubResponse:
            text = json.dumps(
                {
                    "translations": [
                        {"speaker": "SPEAKER_00", "text": "[excited] Translated line"}
                    ]
                }
            )

        class _StubLlm:
            def complete(self, prompt: str):
                prompts.append(prompt)
                return _StubResponse()

        translator.llm = _StubLlm()
        chunk = {
            "text": "SPEAKER_00: Exciting news",
            "original_speaker_texts": [
                {"speaker": "SPEAKER_00", "text": "Exciting news"}
            ],
            "speakers": ["SPEAKER_00"],
        }

        translator._translate_single_chunk(
            chunk=chunk,
            i=0,
            chunks=[chunk],
            context_info={"domain": "general", "tone": "excited", "themes": []},
            source_language="en",
            target_language="ru",
            enable_cache=False,
        )

        self.assertEqual(len(prompts), 1)
        self.assertIn("# Gemini TTS delivery guidance", prompts[0])
        self.assertIn("Gemini TTS speakers in this chunk: SPEAKER_00.", prompts[0])
        self.assertIn("pause markers [short pause] or [long pause]", prompts[0])
        self.assertEqual(chunk["translated_pairs"][0]["text"], "[excited] Translated line")

    def test_editor_preserves_segment_count_and_speaker_order(self) -> None:
        translator = _make_translator(segment_stretch="audio")
        segments = [
            _make_segment(0, text="Original API note", translation="API stays untranslated"),
            _make_segment(1, text="Second line", translation="Second translated line"),
        ]

        payload = {
            "translations": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "API means application programming interface.",
                    "very_short": "API means application interface.",
                    "short": "API means application programming interface.",
                    "long": "API means application programming interface, a software integration layer.",
                },
                {
                    "speaker": "SPEAKER_01",
                    "text": "Second translated line, clarified.",
                    "very_short": "Second line.",
                    "short": "Second translated line.",
                    "long": "Second translated line, clarified for easier understanding.",
                },
            ]
        }

        translator._complete_editor_prompt = lambda prompt: (json.dumps(payload), {"usage": {}})  # type: ignore[method-assign]
        edited = translator._edit_segments_with_llm(
            segments,
            context_info={"terminology": ["API"]},
            source_language="en",
            target_language="ru",
            dialogue_summary="A short discussion.",
        )

        self.assertEqual(len(edited), len(segments))
        self.assertEqual([segment["speaker"] for segment in edited], ["SPEAKER_00", "SPEAKER_01"])
        self.assertTrue(edited[0]["translation"].startswith("API means"))
        self.assertTrue(edited[0]["very_short_translation"])
        self.assertTrue(edited[0]["short_translation"])
        self.assertTrue(edited[0]["long_translation"])

    def test_editor_reports_start_and_finish_progress(self) -> None:
        translator = _make_translator(segment_stretch="audio")
        segments = [_make_segment(0, translation="Pre-editor line")]
        progress_events: list[tuple[str, int, int, str | None]] = []

        translator._edit_batch_with_llm = lambda *args, **kwargs: [  # type: ignore[method-assign]
            {**segments[0], "translation": "Post-editor line"}
        ]

        edited = translator._edit_segments_with_llm(
            segments,
            context_info={},
            source_language="en",
            target_language="ru",
            dialogue_summary="A short discussion.",
            progress_callback=lambda phase, current, total, text=None: progress_events.append(  # type: ignore[call-arg]
                (phase, current, total, text)
            ),
        )

        self.assertEqual(edited[0]["translation"], "Post-editor line")
        self.assertEqual(
            progress_events,
            [
                ("editor", 0, 1, "Pre-editor line"),
                ("editor", 1, 1, "Post-editor line"),
            ],
        )

    def test_editor_falls_back_when_output_structure_is_invalid(self) -> None:
        translator = _make_translator(segment_stretch="audio")
        segments = [
            _make_segment(0, translation="First line"),
            _make_segment(1, translation="Second line"),
        ]
        original_translations = [segment["translation"] for segment in segments]

        invalid_payload = {
            "translations": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "",
                    "short": "",
                }
            ]
        }

        translator._complete_editor_prompt = lambda prompt: (json.dumps(invalid_payload), {"usage": {}})  # type: ignore[method-assign]
        edited = translator._edit_segments_with_llm(
            segments,
            context_info={},
            source_language="en",
            target_language="ru",
            dialogue_summary="Fallback path.",
        )

        self.assertEqual([segment["translation"] for segment in edited], original_translations)
        self.assertEqual(len(edited), len(segments))

    def test_editor_prompt_marks_first_use_explanation_in_full_dialogue(self) -> None:
        translator = _make_translator(segment_stretch="audio_and_video")
        segments = [
            _make_segment(0, text="API details", translation="API details remain"),
            _make_segment(1, text="More API details", translation="API appears again"),
        ]
        prompts: list[str] = []
        payload = {
            "translations": [
                {"speaker": "SPEAKER_00", "text": "API means application programming interface.", "long": "API means application programming interface."},
                {"speaker": "SPEAKER_01", "text": "API appears again.", "long": "API appears again in the discussion."}],
        }

        def _complete(prompt: str):
            prompts.append(prompt)
            return json.dumps(payload), {"usage": {}}

        translator._complete_editor_prompt = _complete  # type: ignore[method-assign]
        translator._edit_segments_with_llm(
            segments,
            context_info={"terminology": ["API"]},
            source_language="en",
            target_language="ru",
            dialogue_summary="Two mentions of the same term.",
        )

        self.assertEqual(len(prompts), 1)
        self.assertIn("Terms that still need first-use explanation in this full dialogue:\n- API", prompts[0])
        self.assertIn("Full dialogue slot data (order and speakers are fixed):", prompts[0])

    def test_editor_prompt_guides_same_speaker_repetition_cleanup(self) -> None:
        translator = _make_translator(segment_stretch="audio")
        prompt, _terms = translator._build_editor_prompt(
            [_make_segment(0, text="I I think we should start", translation="Я я думаю, нам стоит начать")],
            context_info={},
            source_language="en",
            target_language="ru",
            dialogue_summary="Speaker hesitates and restarts.",
            explained_terms=set(),
        )

        self.assertIn("remove accidental same-speaker repetitions caused by hesitation", prompt)
        self.assertIn("preserve deliberate repetition when it is used for emphasis", prompt)

    def test_editor_prompt_guides_transcription_speaker_split_repair(self) -> None:
        translator = _make_translator(segment_stretch="audio")
        prompt, _terms = translator._build_editor_prompt(
            [
                _make_segment(0, text="We need to deploy the", translation="Нам нужно развернуть"),
                _make_segment(1, text="service today", translation="сервис сегодня"),
            ],
            context_info={},
            source_language="en",
            target_language="ru",
            dialogue_summary="A phrase ending is assigned to the wrong speaker.",
            explained_terms=set(),
        )

        self.assertIn("repair obvious transcription speaker-split artifacts", prompt)
        self.assertIn("the next speaker only contains the final 1-2 words", prompt)
        self.assertIn("Do NOT merge real turn-taking", prompt)

    def test_editor_prompt_adds_gemini_delivery_guidance_for_gemini_slots(self) -> None:
        translator = _make_translator(
            segment_stretch="audio",
            tts_system="openai",
            tts_system_mapping={"SPEAKER_00": "gemini", "SPEAKER_01": "openai"},
        )
        segments = [
            _make_segment(0, text="Exciting news", translation="Exciting news translated"),
            _make_segment(1, text="Plain follow-up", translation="Plain follow-up translated"),
        ]
        prompts: list[str] = []
        payload = {
            "translations": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "[excited] Exciting news translated!",
                    "very_short": "Exciting news!",
                    "short": "[excited] Exciting news translated!",
                    "long": "[excited] Exciting news translated! [short pause]",
                },
                {
                    "speaker": "SPEAKER_01",
                    "text": "Plain follow-up translated.",
                    "very_short": "Plain follow-up.",
                    "short": "Plain follow-up translated.",
                    "long": "Plain follow-up translated, with context.",
                },
            ]
        }

        def _complete(prompt: str):
            prompts.append(prompt)
            return json.dumps(payload), {"usage": {}}

        translator._complete_editor_prompt = _complete  # type: ignore[method-assign]
        translator._edit_segments_with_llm(
            segments,
            context_info={},
            source_language="en",
            target_language="ru",
            dialogue_summary="Mixed TTS dialogue.",
        )

        self.assertEqual(len(prompts), 1)
        self.assertIn("# Gemini TTS delivery guidance", prompts[0])
        self.assertIn('"tts_system": "gemini"', prompts[0])
        self.assertIn("Pause markers: [short pause]", prompts[0])
        self.assertIn("Uninterrupted continuous speech is hard to follow in dubbing", prompts[0])
        self.assertIn("For non-Gemini slots, do not add Gemini markup tags or pause markers.", prompts[0])

    def test_editor_prompt_omits_gemini_delivery_guidance_without_gemini_tts(self) -> None:
        translator = _make_translator(segment_stretch="audio", tts_system="openai")
        segments = [_make_segment(0, text="Neutral line", translation="Neutral translation")]
        prompts: list[str] = []
        payload = {
            "translations": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Neutral translation.",
                    "very_short": "Neutral.",
                    "short": "Neutral translation.",
                    "long": "Neutral translation in full.",
                }
            ]
        }

        def _complete(prompt: str):
            prompts.append(prompt)
            return json.dumps(payload), {"usage": {}}

        translator._complete_editor_prompt = _complete  # type: ignore[method-assign]
        translator._edit_segments_with_llm(
            segments,
            context_info={},
            source_language="en",
            target_language="ru",
            dialogue_summary="Single line.",
        )

        self.assertEqual(len(prompts), 1)
        self.assertNotIn("# Gemini TTS delivery guidance", prompts[0])

    def test_translate_skips_translator_side_emotion_enrichment(self) -> None:
        translator = _make_translator(enable_emotion_enrichment=True)
        source_segments = [_make_segment(0, text="Original line", translation="Initial translation")]
        edited_segment = {**source_segments[0], "translation": "[warmly] Edited line"}

        translator.is_available = lambda: True  # type: ignore[method-assign]
        translator._analyze_context = lambda **kwargs: {  # type: ignore[method-assign]
            "text_summary": "",
            "chapters": [],
            "overall_summary": "",
        }
        translator._optimize_segments = lambda segments, **kwargs: [  # type: ignore[method-assign]
            segment.copy() for segment in segments
        ]
        translator._chunk_transcription = lambda segments: ["chunk"]  # type: ignore[method-assign]
        translator._translate_chunks = lambda **kwargs: ["translated chunk"]  # type: ignore[method-assign]
        translator._generate_timecodes_report = lambda **kwargs: None  # type: ignore[method-assign]
        translator._refine_translation = lambda **kwargs: ["refined chunk"]  # type: ignore[method-assign]
        translator._segment_translations = lambda refined_chunks: [  # type: ignore[method-assign]
            source_segments[0].copy()
        ]
        translator._edit_segments_with_llm = lambda segments, **kwargs: [  # type: ignore[method-assign]
            edited_segment.copy()
        ]
        translator._enrich_segments_with_emotion = lambda *args, **kwargs: self.fail(  # type: ignore[method-assign]
            "translator-side emotion enrichment should remain disabled"
        )

        translated = translator.translate(
            source_segments,
            source_language="en",
            target_language="ru",
        )

        self.assertEqual(translated[0]["translation"], "[warmly] Edited line")

    def test_translation_cache_key_changes_when_editor_mode_changes(self) -> None:
        without_editor = LLMTranslator(enable_cache=False, enable_llm_editor=False)
        with_editor = _make_translator()

        plain_key = without_editor._generate_cache_key("hello", "en", "ru")
        editor_key = with_editor._generate_cache_key("hello", "en", "ru")

        self.assertNotEqual(plain_key, editor_key)

    def test_translation_cache_key_changes_when_editor_temperature_changes(self) -> None:
        cool_editor = _make_translator(editor_temperature=0.2)
        warm_editor = _make_translator(editor_temperature=1.3)

        cool_key = cool_editor._generate_cache_key("hello", "en", "ru")
        warm_key = warm_editor._generate_cache_key("hello", "en", "ru")

        self.assertNotEqual(cool_key, warm_key)

    def test_translation_cache_key_changes_when_editor_tts_config_changes(self) -> None:
        openai_tts_editor = _make_translator(tts_system="openai")
        gemini_tts_editor = _make_translator(tts_system="gemini")

        openai_key = openai_tts_editor._generate_cache_key("hello", "en", "ru")
        gemini_key = gemini_tts_editor._generate_cache_key("hello", "en", "ru")

        self.assertNotEqual(openai_key, gemini_key)

    def test_translation_cache_key_changes_when_translation_tts_guidance_changes(self) -> None:
        openai_tts = LLMTranslator(enable_cache=False, enable_llm_editor=False, tts_system="openai")
        gemini_tts = LLMTranslator(enable_cache=False, enable_llm_editor=False, tts_system="gemini")

        openai_key = openai_tts._generate_cache_key("hello", "en", "ru")
        gemini_key = gemini_tts._generate_cache_key("hello", "en", "ru")

        self.assertNotEqual(openai_key, gemini_key)

    def test_editor_processes_full_dialogue_in_single_call(self) -> None:
        translator = _make_translator(segment_stretch="audio")
        segments = [
            _make_segment(index, text=f"Original line {index}", translation=f"Current translation {index}")
            for index in range(17)
        ]
        calls = {"count": 0}
        payload = {
            "translations": [
                {
                    "speaker": segment["speaker"],
                    "text": f"Edited {segment['translation']}",
                    "very_short": f"Edited {segment['translation']}",
                    "short": f"Edited {segment['translation']}",
                    "long": f"Edited {segment['translation']}",
                }
                for segment in segments
            ]
        }

        def _complete(prompt: str):
            calls["count"] += 1
            return json.dumps(payload), {"usage": {}}

        translator._complete_editor_prompt = _complete  # type: ignore[method-assign]
        edited = translator._edit_segments_with_llm(
            segments,
            context_info={},
            source_language="en",
            target_language="ru",
            dialogue_summary="A long discussion.",
        )

        self.assertEqual(calls["count"], 1)
        self.assertEqual(len(edited), len(segments))

    def test_xhigh_openrouter_editor_raises_token_budget_floor(self) -> None:
        translator = _make_translator()

        self.assertEqual(translator._resolve_editor_openrouter_max_tokens(), 65536)

    def test_editor_max_tokens_scales_with_prompt_size_and_caps_by_model(self) -> None:
        translator = _make_translator()
        translator._count_tokens = lambda model_name, text: 40000  # type: ignore[method-assign]

        self.assertEqual(translator._resolve_editor_openrouter_max_tokens("prompt body"), 128000)

    def test_editor_max_tokens_scales_with_prompt_size_without_known_model_cap(self) -> None:
        translator = _make_translator(editor_model_name="custom/editor-model")
        translator._count_tokens = lambda model_name, text: 20000  # type: ignore[method-assign]

        self.assertEqual(translator._resolve_editor_openrouter_max_tokens("prompt body"), 80000)

    def test_openrouter_missing_text_error_is_diagnostic(self) -> None:
        translator = _make_translator()

        with self.assertRaises(ValueError) as exc_info:
            translator._extract_openrouter_message_text(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": None},
                            "finish_reason": None,
                            "native_finish_reason": None,
                        }
                    ],
                    "usage": {
                        "completion_tokens": 0,
                        "reasoning_tokens": 1200,
                    },
                }
            )

        self.assertIn("Likely zero-completion", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
