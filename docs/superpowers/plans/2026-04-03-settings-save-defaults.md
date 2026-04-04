# Settings Save Defaults Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Gradio interface load its visible config-backed defaults from `dubbing_config.yml` and add a button that saves current settings back into that same file.

**Architecture:** Keep the existing two-tab UI, add small helper functions in the Gradio module for reading and writing YAML-backed settings, and reuse those helpers both for initial component values and for the save callback. Persist only real config fields exposed by the UI, while preserving unrelated YAML keys already present in the file.

**Tech Stack:** Python, Gradio, PyYAML, pytest

---

## Chunk 1: Test coverage

### Task 1: Cover YAML-backed UI defaults

**Files:**
- Modify: `tests/test_gradio_app.py`

- [ ] **Step 1: Write a failing test that builds the app from a temp YAML file**
- [ ] **Step 2: Assert visible component defaults come from that file**
- [ ] **Step 3: Run the targeted pytest command and verify the failure**

### Task 2: Cover settings persistence

**Files:**
- Modify: `tests/test_gradio_app.py`

- [ ] **Step 1: Write a failing test for saving UI values into `dubbing_config.yml`**
- [ ] **Step 2: Assert non-persistent workflow fields are excluded**
- [ ] **Step 3: Assert unrelated existing YAML keys are preserved**

## Chunk 2: UI/config implementation

### Task 3: Load defaults and persist settings

**Files:**
- Modify: `src/dubbing/ui/gradio_app.py`

- [ ] **Step 1: Add helpers to load config-backed defaults**
- [ ] **Step 2: Add helpers to save config-backed settings into `dubbing_config.yml`**
- [ ] **Step 3: Update visible component defaults to use YAML-backed values**
- [ ] **Step 4: Add a `Save settings` button wired to the save callback**

## Chunk 3: Verification

### Task 4: Run targeted tests

**Files:**
- Test: `tests/test_gradio_app.py`

- [ ] **Step 1: Run the new targeted pytest command**
- [ ] **Step 2: Fix any failures until green**
