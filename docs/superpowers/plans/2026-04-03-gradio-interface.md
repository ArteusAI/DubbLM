# Gradio Interface Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-tab Gradio UI that runs the existing DubbLM pipeline through shared Python execution logic.

**Architecture:** Extract reusable config normalization and pipeline execution into a shared runner module, then build a Gradio Blocks app on top of it. Keep the CLI entry point unchanged except for optional reuse of the shared runner in the future.

**Tech Stack:** Python, Gradio, pytest

---

## Chunk 1: Shared execution layer

### Task 1: Add test coverage for UI config normalization

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_runner.py`
- Create: `tests/test_gradio_app.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write minimal implementation**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Implement reusable config + pipeline runner

**Files:**
- Create: `src/dubbing/core/runner.py`

- [ ] **Step 1: Normalize UI values into config overrides**
- [ ] **Step 2: Reuse `DubbingConfig` load/validate/process flow**
- [ ] **Step 3: Return structured execution results for full pipeline and speaker report paths**
- [ ] **Step 4: Capture execution logs for the UI**

## Chunk 2: Gradio entry point

### Task 3: Add Gradio UI module and launcher

**Files:**
- Create: `src/dubbing/ui/__init__.py`
- Create: `src/dubbing/ui/gradio_app.py`
- Create: `gradio_app.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Build two-tab Blocks layout**
- [ ] **Step 2: Wire Workflow + Settings into shared runner**
- [ ] **Step 3: Expose a simple `main()` launcher**
- [ ] **Step 4: Add optional script entry point**

## Chunk 3: Verification

### Task 4: Run targeted tests

**Files:**
- Test: `tests/test_runner.py`
- Test: `tests/test_gradio_app.py`

- [ ] **Step 1: Run targeted pytest commands**
- [ ] **Step 2: Fix any failures**
- [ ] **Step 3: Re-run until green**
