from pathlib import Path
import tomllib

import translation.llm_translator as llm_translator
from translation.llm_translator import LLMTranslator


def test_llm_translator_initializes_without_json_repair(monkeypatch):
    created = []

    def fake_create_llm(self, provider, model_name, temperature, max_tokens=None, purpose="default"):
        handle = object()
        created.append((purpose, handle))
        return handle

    monkeypatch.setattr(llm_translator, "JSON_REPAIR_AVAILABLE", False)
    monkeypatch.setattr(llm_translator, "json_repair", None, raising=False)
    monkeypatch.setattr(LLMTranslator, "_create_llm", fake_create_llm)

    translator = LLMTranslator(enable_cache=False)
    translator.initialize()

    assert [purpose for purpose, _ in created] == ["translation", "refinement"]
    assert translator.llm is created[0][1]
    assert translator.refinement_llm is created[1][1]


def test_pyproject_declares_json_repair_dependency():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("json-repair") for dep in dependencies)


def test_pyproject_declares_gemini_dependency_for_default_translator():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("llama-index-llms-gemini") for dep in dependencies)


def test_gemini_translator_accepts_legacy_gemini_api_key_env(monkeypatch):
    created = {}

    class FakeGemini:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(llm_translator, "GEMINI_AVAILABLE", True)
    monkeypatch.setattr(llm_translator, "Gemini", FakeGemini, raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-key")

    translator = LLMTranslator(enable_cache=False)

    llm = translator._create_llm(
        provider="gemini",
        model_name="models/gemini-2.5-flash",
        temperature=0.5,
        purpose="translation",
    )

    assert isinstance(llm, FakeGemini)
    assert created["api_key"] == "legacy-key"
    assert created["model"] == "models/gemini-2.5-flash"
