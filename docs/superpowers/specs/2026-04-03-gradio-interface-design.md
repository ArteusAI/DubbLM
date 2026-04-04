# Gradio Interface Design

**Date:** 2026-04-03

## Goal

Add a simple Gradio interface on top of the existing Python pipeline so users can run DubbLM without manually composing CLI commands.

## Scope

- Use Gradio as a Python wrapper over the internal pipeline API.
- Keep the existing CLI behavior intact.
- Expose two tabs:
  - `Workflow` for the main end-to-end run.
  - `Settings` for the full set of advanced options.

## Architecture

- Introduce a shared runner module that accepts plain Python configuration overrides.
- Reuse the existing `DubbingConfig` defaults, YAML loading, validation, and special-parameter parsing.
- Keep the Gradio UI focused on value collection and result presentation.

## UI Structure

### Workflow

- Input video
- Source language
- Target language
- Config path
- Output path
- Run mode toggles
- Run button
- Status
- Logs
- Output files

### Settings

- Transcription settings
- Translation settings
- Refinement settings
- TTS settings
- Video/audio settings
- Debug settings

## Data Handling

- Empty strings should be normalized to `None` where appropriate.
- Structured text areas should support JSON for mappings and one-range-per-line input for time ranges.
- Output path should still auto-generate when omitted.

## Non-Goals

- No separate frontend stack.
- No subprocess shelling out to the CLI.
- No authentication or multi-user job queue.
