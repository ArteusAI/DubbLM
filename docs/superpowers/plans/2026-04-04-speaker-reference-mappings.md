# Speaker Reference Mappings Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dynamic Gradio UI table for per-speaker reference audio/text mappings and apply those mappings during synthesis with higher priority than auto-extracted references.

**Architecture:** Extend the UI layer with a structured table component that is converted to config mappings before save/run. Persist the mappings in YAML as separate audio/text dictionaries, then teach runtime synthesis to consult those dictionaries before auto-generating per-segment references.

**Tech Stack:** Python, Gradio, YAML, pytest

---

## Chunk 1: Tests

### Task 1: UI persistence tests

**Files:**
- Modify: `tests/test_gradio_app.py`

- [ ] **Step 1: Write failing tests for loading speaker reference rows**
- [ ] **Step 2: Run targeted pytest command and confirm failure**
- [ ] **Step 3: Write failing tests for saving speaker reference rows into YAML mappings**
- [ ] **Step 4: Run targeted pytest command and confirm failure**

### Task 2: Runtime override test

**Files:**
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing test asserting manual speaker mappings override auto segment references**
- [ ] **Step 2: Run targeted pytest command and confirm failure**

## Chunk 2: Implementation

### Task 3: UI/config plumbing

**Files:**
- Modify: `src/dubbing/ui/gradio_app.py`
- Modify: `src/dubbing/core/config.py`
- Modify: `src/dubbing/core/runner.py`

- [ ] **Step 1: Add mapping defaults and parsing support**
- [ ] **Step 2: Add Gradio dataframe for speaker reference rows**
- [ ] **Step 3: Convert dataframe rows to/from YAML mapping fields during load/save/run**

### Task 4: Runtime synthesis override

**Files:**
- Modify: `src/dubbing/core/smart_dubbing.py`

- [ ] **Step 1: Resolve manual per-speaker reference mappings from config**
- [ ] **Step 2: Apply manual mappings before auto segment reference extraction**
- [ ] **Step 3: Keep existing fallback behavior when no manual mapping exists**

## Chunk 3: Verification

### Task 5: Focused verification

**Files:**
- None

- [ ] **Step 1: Run targeted pytest for Gradio tests**
- [ ] **Step 2: Run targeted pytest for runner/runtime tests**
- [ ] **Step 3: Run combined targeted pytest command**
