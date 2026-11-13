"""Tests for speaker gender metadata helpers."""

import unittest

from src.utils.speaker_gender import (
    SPEAKER_GENDER_SIGNATURE_CONFIG_KEY,
    SPEAKER_METADATA_CONFIG_KEY,
    apply_speaker_gender_override,
    build_effective_gender_prompt_section,
    build_effective_speaker_gender_map,
    build_translation_gender_signature,
    is_speaker_gender_translation_stale,
    normalize_speaker_metadata_map,
    rename_speaker_keyed_map,
)


class SpeakerGenderUtilsTests(unittest.TestCase):
    def test_normalize_speaker_metadata_map_sanitizes_values(self) -> None:
        normalized = normalize_speaker_metadata_map(
            {
                " Speaker A ": {
                    "inferredGender": "FEMALE",
                    "inferredConfidence": "1.4",
                    "rawLabel": " female ",
                    "modelId": "",
                    "overrideGender": "invalid",
                },
                "": {"inferredGender": "male"},
            }
        )

        self.assertEqual(
            normalized,
            {
                "Speaker A": {
                    "inferredGender": "female",
                    "inferredConfidence": 1.0,
                    "rawLabel": "female",
                    "modelId": "audeering/wav2vec2-large-robust-24-ft-age-gender",
                    "overrideGender": "unknown",
                }
            },
        )

    def test_effective_gender_prefers_override(self) -> None:
        effective = build_effective_speaker_gender_map(
            {
                "SPEAKER_00": {
                    "inferredGender": "male",
                    "inferredConfidence": 0.91,
                    "overrideGender": "female",
                },
                "SPEAKER_01": {
                    "inferredGender": "female",
                    "inferredConfidence": 0.95,
                },
            }
        )

        self.assertEqual(
            effective,
            {
                "SPEAKER_00": "female",
                "SPEAKER_01": "female",
            },
        )

    def test_translation_signature_changes_with_effective_gender(self) -> None:
        segments = [
            {"speaker": "SPEAKER_00", "text": "hello"},
            {"speaker": "SPEAKER_01", "text": "world"},
        ]
        base_metadata = {
            "SPEAKER_00": {"inferredGender": "male", "inferredConfidence": 0.92},
            "SPEAKER_01": {"inferredGender": "unknown", "inferredConfidence": 0.0},
        }

        base_signature = build_translation_gender_signature(segments, base_metadata)
        updated_signature = build_translation_gender_signature(
            segments,
            apply_speaker_gender_override(base_metadata, "SPEAKER_00", "female"),
        )

        self.assertNotEqual(updated_signature, base_signature)

    def test_stale_detection_uses_stored_signature(self) -> None:
        segments = [
            {"speaker": "SPEAKER_00"},
            {"speaker": "SPEAKER_01"},
        ]
        metadata = {
            "SPEAKER_00": {"inferredGender": "male", "inferredConfidence": 0.92},
            "SPEAKER_01": {"inferredGender": "female", "inferredConfidence": 0.88},
        }
        signature = build_translation_gender_signature(segments, metadata)

        current_config = {
            "enableSpeakerGenderInference": True,
            SPEAKER_METADATA_CONFIG_KEY: metadata,
            SPEAKER_GENDER_SIGNATURE_CONFIG_KEY: signature,
        }
        stale_config = {
            **current_config,
            SPEAKER_METADATA_CONFIG_KEY: apply_speaker_gender_override(
                metadata,
                "SPEAKER_01",
                "unknown",
            ),
        }

        self.assertFalse(is_speaker_gender_translation_stale(current_config, segments))
        self.assertTrue(is_speaker_gender_translation_stale(stale_config, segments))

    def test_prompt_section_and_rename_helpers(self) -> None:
        metadata = {
            "SPEAKER_01": {"inferredGender": "female", "inferredConfidence": 0.9},
            "SPEAKER_00": {"inferredGender": "male", "inferredConfidence": 0.8},
        }

        prompt_section = build_effective_gender_prompt_section(
            metadata,
            speakers=["SPEAKER_01", "SPEAKER_00"],
        )
        renamed = rename_speaker_keyed_map(metadata, "SPEAKER_00", "HOST")

        self.assertIn("- SPEAKER_00 -> male", prompt_section)
        self.assertIn("- SPEAKER_01 -> female", prompt_section)
        self.assertLess(prompt_section.index("SPEAKER_00"), prompt_section.index("SPEAKER_01"))
        self.assertIn("HOST", renamed)
        self.assertNotIn("SPEAKER_00", renamed)


if __name__ == "__main__":
    unittest.main()
