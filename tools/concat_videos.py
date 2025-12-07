#!/usr/bin/env python3
"""CLI tool to concatenate videos with margins and top title/subtitle.

This script reads a JSON scenario describing a list of video segments and
concatenates them into a single output video, applying per-segment margins
that create black borders around the content and overlaying a bold title and
subtitle within the top border.

JSON scenario formats supported:

- Object with keys:
  {
    "title": "Main Title",
    "subtitle": "Sub Title",
    "text_color": "white",           # optional, ffmpeg color
    "segments": [
      {
        "file": "/path/to/1.mp4",
        "margin": 0.08,               # optional, uniform 0..1
        "margins": {                  # optional, sides in 0..1
          "top": 0.08,
          "bottom": 0.08,
          "left": 0.06,
          "right": 0.06
        },
        "bg_color": "black"          # optional, ffmpeg color
      },
      ...
    ]
  }

- Or a simple list of file paths: ["a.mp4", "b.mp4", ...]

Usage example:
  python tools/concat_videos.py scenario.json -o out.mp4 --width 1920 --height 1080 \
    --fps 30 --title "My Title" --subtitle "My Subtitle"
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Margins:
    """Normalized margins for a segment (fractions in [0, 1))."""

    top: float = 0.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 0.0

    @staticmethod
    def from_spec(spec: Dict[str, Any] | float | None) -> "Margins":
        """Create margins from either a float or dict spec.

        If spec is a number, applies uniform margins to all sides.
        If spec is a dict, reads keys: top, bottom, left, right.
        """

        if spec is None:
            return Margins()
        if isinstance(spec, (int, float)):
            val = float(spec)
            return Margins(val, val, val, val)
        if isinstance(spec, dict):
            return Margins(
                float(spec.get("top", 0.0) or 0.0),
                float(spec.get("bottom", 0.0) or 0.0),
                float(spec.get("left", 0.0) or 0.0),
                float(spec.get("right", 0.0) or 0.0),
            )
        raise ValueError("Invalid margins spec; must be float or dict")

    def clamped(self) -> "Margins":
        """Clamp all sides to [0, 0.49] to ensure content remains visible."""

        return Margins(
            top=max(0.0, min(0.49, self.top)),
            bottom=max(0.0, min(0.49, self.bottom)),
            left=max(0.0, min(0.49, self.left)),
            right=max(0.0, min(0.49, self.right)),
        )


@dataclass
class SegmentSpec:
    """Single input segment specification."""

    file: Path
    margins: Margins
    bg_color: str = "black"


@dataclass
class Scenario:
    """Scenario with optional global text and list of segments."""

    title: str = ""
    subtitle: str = ""
    text_color: str = "white"
    segments: List[SegmentSpec] = None  # type: ignore[assignment]


def check_tool_available(name: str) -> None:
    """Ensure external tool exists in PATH."""

    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool not found: {name}")


def run_command(args: List[str]) -> None:
    """Run subprocess and stream output; raise on failure."""

    process = subprocess.run(args, stdout=sys.stdout, stderr=sys.stderr, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(args)}")


def ffprobe_streams(path: Path) -> Dict[str, Any]:
    """Return ffprobe streams metadata for the given file."""

    args = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-print_format",
        "json",
        str(path),
    ]
    result = subprocess.run(args, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")
    return json.loads(result.stdout or "{}")


def ffprobe_duration_seconds(path: Path) -> float:
    """Return media duration in seconds using ffprobe; 0.0 on failure."""

    args = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(args, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def parse_scenario(json_path: Path) -> Scenario:
    """Parse scenario JSON into a structured Scenario object."""

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        segments = [
            SegmentSpec(file=Path(item), margins=Margins().clamped()) for item in data
        ]
        return Scenario(segments=segments)

    if not isinstance(data, dict):
        raise ValueError("Scenario JSON must be an object or list")

    title = str(data.get("title", "") or "")
    subtitle = str(data.get("subtitle", "") or "")
    text_color = str(data.get("text_color", "white") or "white")

    segs_raw = data.get("segments")
    if not isinstance(segs_raw, list) or not segs_raw:
        raise ValueError("Scenario must include non-empty 'segments' list or be a list of files")

    segments: List[SegmentSpec] = []
    for item in segs_raw:
        if isinstance(item, str):
            segments.append(SegmentSpec(file=Path(item), margins=Margins().clamped()))
            continue
        if not isinstance(item, dict) or "file" not in item:
            raise ValueError("Each segment must be a string path or object with 'file'")
        file_path = Path(item["file"])  # type: ignore[index]
        margin_spec = item.get("margins")
        if margin_spec is None and "margin" in item:
            margin_spec = item.get("margin")
        margins = Margins.from_spec(margin_spec).clamped()
        bg_color = str(item.get("bg_color", "black") or "black")
        segments.append(SegmentSpec(file=file_path, margins=margins, bg_color=bg_color))

    return Scenario(title=title, subtitle=subtitle, text_color=text_color, segments=segments)


def escape_drawtext(text: str) -> str:
    """Escape text for ffmpeg drawtext filter."""

    # Escape backslashes first
    escaped = text.replace("\\", "\\\\")
    # Escape single quotes and colons
    escaped = escaped.replace("'", "\\'").replace(":", "\\:")
    # Newlines
    escaped = escaped.replace("\n", "\\n")
    # Percent signs
    escaped = escaped.replace("%", "\\%")
    return escaped


def build_filtergraph(
    segments: List[SegmentSpec],
    output_w: int,
    output_h: int,
    fps: int,
    samplerate: int,
    title: str,
    subtitle: str,
    text_color: str,
    font: Optional[str],
    font_bold: Optional[str],
) -> Tuple[str, List[str]]:
    """Build the ffmpeg filtergraph and input arguments for all segments.

    Returns a tuple of (filter_complex, ffmpeg_input_args).
    """

    filter_parts: List[str] = []
    input_args: List[str] = []
    concat_inputs: List[str] = []

    # Title/subtitle sizes and spacing
    title_font_size = max(10, int(round(output_h * 0.055)))
    subtitle_font_size = max(8, int(round(output_h * 0.04)))
    spacing = max(2, int(round(output_h * 0.01)))

    for idx, seg in enumerate(segments):
        input_args += ["-i", str(seg.file)]

        # Probe streams for audio presence and duration
        streams = ffprobe_streams(seg.file)
        stream_list = streams.get("streams", []) or []
        has_audio = any(s.get("codec_type") == "audio" for s in stream_list)
        duration = ffprobe_duration_seconds(seg.file)

        # Compute geometry
        m = seg.margins
        content_w = max(2, int(round(output_w * (1.0 - m.left - m.right))))
        content_h = max(2, int(round(output_h * (1.0 - m.top - m.bottom))))
        left_off = max(0, int(round(output_w * m.left)))
        top_off = max(0, int(round(output_h * m.top)))
        top_border = top_off  # height portion at top

        v_in = f"[{idx}:v]"
        a_in = f"[{idx}:a]"

        # Video chain: fps -> scale -> pad to content -> pad to output -> drawtext(s)
        v_label = f"v{idx}"
        chain = [
            f"{v_in}fps=fps={fps}",
            f"scale=w={content_w}:h={content_h}:force_original_aspect_ratio=decrease",
            f"pad=w={content_w}:h={content_h}:x=(ow-iw)/2:y=(oh-ih)/2:color={seg.bg_color}",
            f"pad=w={output_w}:h={output_h}:x={left_off}:y={top_off}:color={seg.bg_color}",
        ]

        # Title drawtext
        if title:
            x_expr = "(w-text_w)/2"
            y_title = max(0, int(round(top_border * 0.35)))
            font_selector = (
                f"font='{font_bold}'" if font_bold and Path(font_bold).exists() else (f"font='{font or 'DejaVu Sans'}'" )
            )
            chain.append(
                "drawtext="
                + f"{font_selector}:text='{escape_drawtext(title)}':fontsize={title_font_size}:"
                + f"fontcolor={text_color}:x={x_expr}:y={y_title}:fix_bounds=1"
            )

        # Subtitle drawtext
        if subtitle:
            x_expr = "(w-text_w)/2"
            y_sub = max(0, int(round(top_border * 0.35 + title_font_size + spacing)))
            font_selector = (
                f"font='{font or 'DejaVu Sans'}'" if not (font and Path(font).exists()) else f"font='{font}'"
            )
            chain.append(
                "drawtext="
                + f"{font_selector}:text='{escape_drawtext(subtitle)}':fontsize={subtitle_font_size}:"
                + f"fontcolor={text_color}:x={x_expr}:y={y_sub}:fix_bounds=1"
            )

        chain_str = ",".join(chain) + f"[{v_label}]"
        filter_parts.append(chain_str)

        # Audio chain: normalize or synthesize if missing
        a_label = f"a{idx}"
        if has_audio:
            a_chain = f"{a_in}aformat=sample_fmts=s16:sample_rates={samplerate}:channel_layouts=stereo[{a_label}]"
        else:
            # Synthesize silent audio matching approximate duration
            dur = max(0.0, duration)
            a_chain = (
                f"anullsrc=r={samplerate}:cl=stereo,atrim=0:{dur},asetpts=N/SR/TB[{a_label}]"
            )
        filter_parts.append(a_chain)

        concat_inputs.append(f"[{v_label}][{a_label}]")

    # Concat
    concat_join = "".join(concat_inputs)
    filter_parts.append(
        f"{concat_join}concat=n={len(segments)}:v=1:a=1[vout][aout]"
    )

    filter_complex = ";".join(filter_parts)
    return filter_complex, input_args


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Concatenate videos with per-segment margins and top title/subtitle."
        )
    )
    parser.add_argument("scenario", type=Path, help="Path to scenario JSON")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output video file path")
    parser.add_argument("--width", type=int, default=1920, help="Output width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Output height (default: 1080)")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument("--samplerate", type=int, default=48000, help="Audio sample rate (default: 48000)")
    parser.add_argument("--crf", type=int, default=20, help="H.264 CRF (default: 20)")
    parser.add_argument("--preset", type=str, default="medium", help="H.264 preset (default: medium)")
    parser.add_argument("--title", type=str, default="", help="Override title text")
    parser.add_argument("--subtitle", type=str, default="", help="Override subtitle text")
    parser.add_argument("--text-color", type=str, default="white", help="Text color (default: white)")
    parser.add_argument("--font", type=str, default="", help="Font name/file for subtitle")
    parser.add_argument("--font-bold", type=str, default="", help="Font name/file for title (bold)")

    args = parser.parse_args(argv)

    check_tool_available("ffmpeg")
    check_tool_available("ffprobe")

    scenario = parse_scenario(args.scenario)
    title = args.title or scenario.title
    subtitle = args.subtitle or scenario.subtitle
    text_color = args.text_color or scenario.text_color or "white"

    if not scenario.segments:
        raise ValueError("No segments to process")

    # Build filtergraph
    filter_complex, input_args = build_filtergraph(
        segments=scenario.segments,
        output_w=args.width,
        output_h=args.height,
        fps=args.fps,
        samplerate=args.samplerate,
        title=title,
        subtitle=subtitle,
        text_color=text_color,
        font=args.font or None,
        font_bold=args.font_bold or None,
    )

    # Assemble ffmpeg command
    cmd: List[str] = [
        "ffmpeg",
        "-y",
        *input_args,
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-r",
        str(args.fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(args.output),
    ]

    run_command(cmd)


if __name__ == "__main__":
    main()


