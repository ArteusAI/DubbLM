import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clear_tts_modules():
    for module_name in list(sys.modules):
        if module_name == "tts" or module_name.startswith("tts."):
            sys.modules.pop(module_name, None)


def test_tts_factory_import_does_not_eagerly_load_all_wrappers():
    _clear_tts_modules()

    importlib.import_module("tts.tts_factory")

    assert "tts.omnivoice_wrapper" not in sys.modules
    assert "tts.gemini_tts_wrapper" not in sys.modules
    assert "tts.openai_tts_wrapper" not in sys.modules
    assert "tts.xtts_local_wrapper" not in sys.modules


def test_tts_factory_lists_omnivoice_provider():
    _clear_tts_modules()

    factory = importlib.import_module("tts.tts_factory")

    assert "omnivoice" in factory.TTSFactory.get_available_providers()
