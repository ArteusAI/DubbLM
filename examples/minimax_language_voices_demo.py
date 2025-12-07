#!/usr/bin/env python3
"""
Demo script showing language-specific voice selection in MiniMax TTS.

This demonstrates how the MiniMax TTS wrapper now automatically selects
appropriate voices based on the target language.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tts.minimax_tts_wrapper import MinimaxTTSWrapper

def main():
    print("=" * 70)
    print("MiniMax TTS - Language-Specific Voice Selection Demo")
    print("=" * 70)
    print()
    
    # Get all supported languages
    supported_languages = MinimaxTTSWrapper.get_supported_languages()
    print(f"📋 Supported Languages ({len(supported_languages)}):")
    print(f"   {', '.join(sorted(supported_languages))}")
    print()
    
    # Show voice counts per language
    print("🎤 Voice Count per Language:")
    for lang in sorted(supported_languages):
        voices = MinimaxTTSWrapper.get_available_voices(lang)
        print(f"   {lang:4s}: {len(voices):3d} voices")
    print()
    
    # Example: Show voices for specific languages
    demo_languages = ["en", "es", "zh", "ja", "fr", "de", "ru"]
    
    for lang in demo_languages:
        voices = MinimaxTTSWrapper.get_available_voices(lang)
        if voices:
            print(f"\n🌍 {lang.upper()} - {len(voices)} voices:")
            print(f"   First 5: {', '.join(voices[:5])}")
            if len(voices) > 5:
                print(f"   ... and {len(voices) - 5} more")
    
    print()
    print("=" * 70)
    print("Usage Example:")
    print("=" * 70)
    print()
    
    # Show how to initialize with different languages
    examples = [
        ("en", "English"),
        ("es", "Spanish"),
        ("zh", "Chinese (Mandarin)"),
        ("ja", "Japanese"),
        ("fr", "French"),
    ]
    
    for lang_code, lang_name in examples:
        voices = MinimaxTTSWrapper.get_available_voices(lang_code)
        default_voice = voices[0] if voices else "N/A"
        
        print(f"# Initialize for {lang_name} ({lang_code})")
        print(f"tts = MinimaxTTSWrapper(")
        print(f"    target_language='{lang_code}',")
        print(f"    # Default voice will be: '{default_voice}'")
        print(f")")
        print()
    
    print("=" * 70)
    print("Key Features:")
    print("=" * 70)
    print()
    print("✨ Automatic voice selection based on target language")
    print("✨ Default voice automatically set per language")
    print("✨ Language-specific sample directories")
    print("✨ Support for 20+ languages")
    print("✨ Easy voice discovery via helper methods")
    print()
    print("📚 Reference: https://platform.minimax.io/docs/faq/system-voice-id")
    print()

if __name__ == "__main__":
    main()

