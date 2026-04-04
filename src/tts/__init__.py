from importlib import import_module

from .tts_factory import TTSFactory
from .tts_interface import TTSInterface


_LAZY_EXPORTS = {
    "F5TTSWrapper": ("tts.f5_tts_wrapper", "F5TTSWrapper"),
    "OpenAITTSWrapper": ("tts.openai_tts_wrapper", "OpenAITTSWrapper"),
    "GeminiTTSWrapper": ("tts.gemini_tts_wrapper", "GeminiTTSWrapper"),
    "BexTTSWrapper": ("tts.bextts_wrapper", "BexTTSWrapper"),
    "OmniVoiceWrapper": ("tts.omnivoice_wrapper", "OmniVoiceWrapper"),
    "XTTSLocalWrapper": ("tts.xtts_local_wrapper", "XTTSLocalWrapper"),
}

__all__ = [
    "TTSInterface",
    "F5TTSWrapper",
    "OpenAITTSWrapper",
    "GeminiTTSWrapper",
    "BexTTSWrapper",
    "OmniVoiceWrapper",
    "XTTSLocalWrapper",
    "TTSFactory",
]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'tts' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
