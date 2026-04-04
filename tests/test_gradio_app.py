from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import yaml

import dubbing.ui.gradio_app as gradio_app
from dubbing.ui.gradio_app import DEFAULT_CONFIG_PATH, build_app, save_settings


def test_build_app_returns_gradio_blocks():
    app = build_app()

    assert isinstance(app, gr.Blocks)


def _component_value_by_label(app, label):
    for component in app.config["components"]:
        props = component.get("props", {})
        if props.get("label") == label:
            value = props.get("value")
            if isinstance(value, dict) and "data" in value:
                return value["data"]
            return value
    raise AssertionError(f"Component with label {label!r} not found")


def _component_props_by_label(app, label):
    for component in app.config["components"]:
        props = component.get("props", {})
        if props.get("label") == label:
            return props
    raise AssertionError(f"Component with label {label!r} not found")


def test_build_app_uses_yaml_defaults(tmp_path):
    config_path = tmp_path / "ui_defaults.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "source_language": "pl",
                "target_language": "be",
                "save_original_subtitles": True,
                "keep_background": True,
                "transcription_system": "assemblyai",
                "tts_system": "bextts",
                "debug_tts": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    app = build_app(config_path=str(config_path))

    assert _component_value_by_label(app, "Source language") == "pl"
    assert _component_value_by_label(app, "Target language") == "be"
    assert _component_value_by_label(app, "Save original subtitles") is True
    assert _component_value_by_label(app, "Keep background audio") is True
    assert _component_value_by_label(app, "Transcription system") == "assemblyai"
    assert _component_value_by_label(app, "TTS system") == "bextts"
    assert _component_value_by_label(app, "Debug TTS") is True


def test_build_app_lists_omnivoice_in_tts_system_choices():
    app = build_app()

    tts_props = _component_props_by_label(app, "TTS system")
    choice_values = [choice[1] if isinstance(choice, (list, tuple)) else choice for choice in tts_props["choices"]]

    assert "omnivoice" in choice_values


def test_build_app_explains_run_step_requires_existing_artifacts():
    app = build_app()

    run_step_props = _component_props_by_label(app, "Run step")

    assert "existing artifacts" in run_step_props["info"]
    assert "Run DubbLM" in run_step_props["info"]


def test_save_settings_writes_to_default_config_and_preserves_other_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    initial_config_path = tmp_path / DEFAULT_CONFIG_PATH
    initial_config_path.write_text(
        yaml.safe_dump(
            {
                "source_language": "en",
                "target_language": "be",
                "normalize_audio": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    status = save_settings(
        {
            "input": "clip.mp4",
            "output": "result.mp4",
            "config": "custom.yml",
            "run_step": "combine_video",
            "generate_speaker_report": True,
            "source_language": "de",
            "target_language": "uk",
            "keep_background": True,
            "save_translated_subtitles": True,
            "tts_system": "bextts",
        }
    )

    saved_data = yaml.safe_load(initial_config_path.read_text(encoding="utf-8"))

    assert status == f"Settings saved to {DEFAULT_CONFIG_PATH}"
    assert saved_data["source_language"] == "de"
    assert saved_data["target_language"] == "uk"
    assert saved_data["keep_background"] is True
    assert saved_data["save_translated_subtitles"] is True
    assert saved_data["tts_system"] == "bextts"
    assert saved_data["normalize_audio"] is True
    assert "input" not in saved_data
    assert "output" not in saved_data
    assert "config" not in saved_data
    assert "run_step" not in saved_data
    assert "generate_speaker_report" not in saved_data
    assert not (tmp_path / "custom.yml").exists()


def test_build_app_formats_structured_yaml_values_for_text_inputs(tmp_path):
    config_path = tmp_path / "structured_defaults.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "glossary": {"AI": "ШІ"},
                "voice_prompt": {"SPEAKER_00": "warm"},
                "tts_system_mapping": {"SPEAKER_00": "gemini"},
                "reference_audio_mapping": {
                    "SPEAKER_00": "D:/voices/speaker_00.wav",
                    "SPEAKER_01": "D:/voices/speaker_01.wav",
                },
                "reference_text_mapping": {
                    "SPEAKER_00": "First speaker reference",
                    "SPEAKER_01": "Second speaker reference",
                },
                "keep_original_audio_ranges": ["00:10-00:15", "01:02-01:08"],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    app = build_app(config_path=str(config_path))

    assert _component_value_by_label(app, "Glossary JSON") == '{\n  "AI": "ШІ"\n}'
    assert _component_value_by_label(app, "Voice prompt JSON") == '{\n  "SPEAKER_00": "warm"\n}'
    assert _component_value_by_label(app, "TTS system mapping JSON") == '{\n  "SPEAKER_00": "gemini"\n}'
    assert _component_value_by_label(app, "Speaker reference mappings") == [
        ["SPEAKER_00", "D:/voices/speaker_00.wav", "First speaker reference"],
        ["SPEAKER_01", "D:/voices/speaker_01.wav", "Second speaker reference"],
    ]
    assert _component_value_by_label(app, "Keep original audio ranges") == "00:10-00:15\n01:02-01:08"


def test_save_settings_parses_structured_text_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    status = save_settings(
        {
            "glossary": '{\n  "AI": "ШІ"\n}',
            "voice_prompt": '{"SPEAKER_00": "warm"}',
            "tts_system_mapping": '{"SPEAKER_00": "gemini"}',
            "speaker_reference_rows": [
                ["SPEAKER_00", "D:/voices/speaker_00.wav", "First speaker reference"],
                ["SPEAKER_01", "D:/voices/speaker_01.wav", "Second speaker reference"],
                ["", "", ""],
            ],
            "keep_original_audio_ranges": "00:10-00:15\n01:02-01:08",
        }
    )

    saved_data = yaml.safe_load((tmp_path / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))

    assert status == f"Settings saved to {DEFAULT_CONFIG_PATH}"
    assert saved_data["glossary"] == {"AI": "ШІ"}
    assert saved_data["voice_prompt"] == {"SPEAKER_00": "warm"}
    assert saved_data["tts_system_mapping"] == {"SPEAKER_00": "gemini"}
    assert saved_data["reference_audio_mapping"] == {
        "SPEAKER_00": "D:/voices/speaker_00.wav",
        "SPEAKER_01": "D:/voices/speaker_01.wav",
    }
    assert saved_data["reference_text_mapping"] == {
        "SPEAKER_00": "First speaker reference",
        "SPEAKER_01": "Second speaker reference",
    }
    assert saved_data["keep_original_audio_ranges"] == ["00:10-00:15", "01:02-01:08"]


def test_save_settings_drops_zero_duration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    save_settings(
        {
            "source_language": "en",
            "target_language": "be",
            "start_time": 0,
            "duration": 0,
        }
    )

    saved_data = yaml.safe_load((tmp_path / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))

    assert saved_data["start_time"] == 0
    assert "duration" not in saved_data


def test_save_speaker_reference_to_library_copies_audio_and_writes_metadata(tmp_path, monkeypatch):
    library_dir = tmp_path / "speaker_reference_library"
    monkeypatch.setattr(gradio_app, "DEFAULT_SPEAKER_REFERENCE_LIBRARY_PATH", str(library_dir))

    source_audio = tmp_path / "speaker01.wav"
    source_audio.write_bytes(b"fake-audio")

    saved_audio_path = gradio_app.save_speaker_reference_to_library(
        speaker_id="SPEAKER_01",
        source_audio_path=str(source_audio),
        reference_text="Second speaker reference",
    )

    meta_path = library_dir / "SPEAKER_01" / "meta.yml"

    assert saved_audio_path == str(library_dir / "SPEAKER_01" / "reference.wav")
    assert Path(saved_audio_path).read_bytes() == b"fake-audio"
    assert yaml.safe_load(meta_path.read_text(encoding="utf-8")) == {
        "speaker_id": "SPEAKER_01",
        "reference_audio_path": str(library_dir / "SPEAKER_01" / "reference.wav"),
        "reference_text": "Second speaker reference",
    }


def test_save_library_reference_updates_current_mapping_rows(tmp_path, monkeypatch):
    library_dir = tmp_path / "speaker_reference_library"
    monkeypatch.setattr(gradio_app, "DEFAULT_SPEAKER_REFERENCE_LIBRARY_PATH", str(library_dir))

    source_audio = tmp_path / "speaker00.wav"
    source_audio.write_bytes(b"fake-audio")

    status, updated_rows, library_rows, cleared_file, cleared_text, cleared_speaker = gradio_app._save_library_reference(
        "SPEAKER_00",
        str(source_audio),
        "First speaker reference",
        [["SPEAKER_99", "D:/voices/other.wav", "Other"]],
    )

    assert "saved" in status.lower()
    assert updated_rows == [
        ["SPEAKER_99", "D:/voices/other.wav", "Other"],
        [str("SPEAKER_00"), str(library_dir / "SPEAKER_00" / "reference.wav"), "First speaker reference"],
    ]
    assert library_rows == [
        ["SPEAKER_00", str(library_dir / "SPEAKER_00" / "reference.wav"), "First speaker reference"]
    ]
    assert cleared_file is None
    assert cleared_text == ""
    assert cleared_speaker == ""


def test_build_app_lists_saved_speaker_reference_library_entries(tmp_path, monkeypatch):
    library_dir = tmp_path / "speaker_reference_library"
    speaker_dir = library_dir / "SPEAKER_01"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    (speaker_dir / "reference.wav").write_bytes(b"fake-audio")
    (speaker_dir / "meta.yml").write_text(
        yaml.safe_dump(
            {
                "speaker_id": "SPEAKER_01",
                "reference_audio_path": str(speaker_dir / "reference.wav"),
                "reference_text": "Second speaker reference",
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gradio_app, "DEFAULT_SPEAKER_REFERENCE_LIBRARY_PATH", str(library_dir))

    app = build_app()

    assert _component_value_by_label(app, "Speaker reference library") == [
        ["SPEAKER_01", str(speaker_dir / "reference.wav"), "Second speaker reference"]
    ]


def test_select_library_row_stores_only_one_selected_entry():
    selected_row = gradio_app._store_selected_library_row(
        SimpleNamespace(
            row_value=["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Second speaker reference"],
            selected=True,
        )
    )

    assert selected_row == ["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Second speaker reference"]


def test_deselect_library_row_clears_selected_entry():
    selected_row = gradio_app._store_selected_library_row(
        SimpleNamespace(
            row_value=["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Second speaker reference"],
            selected=False,
        )
    )

    assert selected_row is None


def test_use_selected_library_row_updates_current_mappings():
    status, updated_rows = gradio_app._use_selected_library_reference(
        ["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Second speaker reference"],
        [["SPEAKER_99", "D:/voices/other.wav", "Other"]],
    )

    assert "added" in status.lower()
    assert updated_rows == [
        ["SPEAKER_99", "D:/voices/other.wav", "Other"],
        ["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Second speaker reference"],
    ]


def test_use_selected_library_row_replaces_existing_speaker_mapping():
    status, updated_rows = gradio_app._use_selected_library_reference(
        ["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Library text"],
        [["SPEAKER_01", "D:/old.wav", "Old text"]],
    )

    assert "updated" in status.lower()
    assert updated_rows == [["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Library text"]]
