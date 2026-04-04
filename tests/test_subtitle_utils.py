from pathlib import Path


def test_save_subtitles_works_without_nltk(tmp_path):
    from dubbing.utils.subtitle_utils import SubtitleManager

    output_path = tmp_path / "out.srt"
    segments = [
        {
            "start": 0.0,
            "end": 4.0,
            "text": "First sentence. Second sentence.",
            "translation": "First sentence. Second sentence.",
        }
    ]

    SubtitleManager().save_subtitles(segments, "original", str(output_path), max_chars_per_line=80)

    content = output_path.read_text(encoding="utf-8")

    assert output_path.exists()
    assert "First sentence." in content
    assert "Second sentence." in content
