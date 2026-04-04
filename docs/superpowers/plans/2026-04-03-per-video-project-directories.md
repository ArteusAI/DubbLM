# Per-Video Project Directories Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save each dubbing run into a dedicated directory named after the input video and located next to that video.

**Architecture:** Compute a per-video `project_dir` and `artifacts_dir` during config validation, then pass those paths into pipeline components instead of relying on the global `artifacts/` root. Keep explicit user-provided `output` paths untouched while moving auto-generated outputs, subtitles, reports, and intermediate files under the per-video directory.

**Tech Stack:** Python, pathlib, pytest, existing DubbLM pipeline classes

---

## Chunk 1: Path Derivation

### Task 1: Add config tests for per-video directories

**Files:**
- Modify: `tests/test_runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_config_from_overrides_places_outputs_inside_project_dir(tmp_path):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runner.py::test_build_config_from_overrides_places_outputs_inside_project_dir -v`
Expected: FAIL because the config still points to the workspace root.

- [ ] **Step 3: Write minimal implementation**

Compute `project_dir` as `<input parent>/<input stem>` and derive `output` plus artifact paths from it.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runner.py::test_build_config_from_overrides_places_outputs_inside_project_dir -v`
Expected: PASS

## Chunk 2: Pipeline Consumers

### Task 2: Thread project-specific artifact roots through processors

**Files:**
- Modify: `src/dubbing/core/config.py`
- Modify: `src/dubbing/core/runner.py`
- Modify: `src/dubbing/core/smart_dubbing.py`
- Modify: `src/dubbing/audio/audio_processor.py`
- Modify: `src/dubbing/audio/speaker_processor.py`
- Modify: `src/dubbing/debug/debug_generator.py`
- Modify: `src/dubbing/debug/reporter.py`
- Modify: `src/dubbing/video/video_processor.py`
- Test: `tests/test_audio_processor.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_audio_processor_creates_directories_inside_custom_artifacts_root(tmp_path):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_audio_processor.py tests/test_runner.py -k "project_dir or artifacts_root" -v`
Expected: FAIL because processors still write into the global `artifacts/`.

- [ ] **Step 3: Write minimal implementation**

Pass the computed artifact root into each processor and replace hard-coded `artifacts/...` paths with derived paths.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_audio_processor.py tests/test_runner.py -k "project_dir or artifacts_root" -v`
Expected: PASS

## Chunk 3: Final Verification

### Task 3: Verify regression coverage

**Files:**
- Test: `tests/test_runner.py`
- Test: `tests/test_audio_processor.py`
- Test: `tests/test_gradio_app.py`

- [ ] **Step 1: Run targeted verification**

Run: `pytest tests/test_runner.py tests/test_audio_processor.py tests/test_gradio_app.py -v`
Expected: PASS

- [ ] **Step 2: Confirm no explicit-output regression**

Run: `pytest tests/test_runner.py -v`
Expected: Existing tests for explicit `output` continue to pass.
