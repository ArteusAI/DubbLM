#!/usr/bin/env python3
"""
Demo showing language_boost feature in MiniMax TTS wrapper.

This demonstrates how the wrapper automatically includes the language_boost
parameter to enhance recognition for specific languages.
"""

from typing import Dict, List

# Language boost mapping (same as in minimax_tts_wrapper.py)
LANGUAGE_BOOST_MAPPING: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "pt": "Portuguese",
    "fr": "French",
    "id": "Indonesian",
    "de": "German",
    "ru": "Russian",
    "it": "Italian",
    "nl": "Dutch",
    "vi": "Vietnamese",
    "ar": "Arabic",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "th": "Thai",
    "pl": "Polish",
    "ro": "Romanian",
    "el": "Greek",
    "cs": "Czech",
    "fi": "Finnish",
    "hi": "Hindi"
}

# Simplified voice count mapping
VOICE_COUNTS = {
    "en": 45, "zh": 34, "pt": 47, "es": 6, "ru": 8, "id": 9,
    "fr": 6, "ja": 4, "it": 4, "th": 4, "pl": 4, "ro": 4,
    "de": 3, "cs": 3, "fi": 3, "el": 3, "hi": 3, "ar": 2,
    "ko": 2, "nl": 2, "tr": 2, "uk": 2, "vi": 1
}

def main():
    print("=" * 80)
    print("MiniMax TTS - Language Boost Feature")
    print("=" * 80)
    print()
    print("The language_boost parameter enhances recognition for specific minority")
    print("languages and dialects, improving synthesis quality.")
    print()
    print("Reference: https://platform.minimax.io/docs/api-reference/speech-t2a-http#body-language-boost")
    print()
    print("=" * 80)
    print("Language Boost Mapping")
    print("=" * 80)
    print()
    
    for lang_code in sorted(LANGUAGE_BOOST_MAPPING.keys()):
        boost_value = LANGUAGE_BOOST_MAPPING[lang_code]
        voice_count = VOICE_COUNTS.get(lang_code, 0)
        print(f"  {lang_code:4s} → language_boost: '{boost_value:15s}' ({voice_count:2d} voices)")
    
    print()
    print("=" * 80)
    print("How It Works")
    print("=" * 80)
    print()
    print("When you initialize the wrapper with a target_language:")
    print()
    print("  tts = MinimaxTTSWrapper(target_language='ru')")
    print()
    print("The API payload automatically includes:")
    print()
    print("  {")
    print('    "model": "speech-2.6-turbo",')
    print('    "text": "Ваш текст здесь",')
    print('    "language_boost": "Russian",    ← Automatically added!')
    print('    "voice_setting": {')
    print('      "voice_id": "Russian_ReliableMan",')
    print('      "speed": 1.0,')
    print('      ...')
    print('    },')
    print('    ...')
    print('  }')
    print()
    print("=" * 80)
    print("Benefits")
    print("=" * 80)
    print()
    print("✓ Improved recognition accuracy for target language")
    print("✓ Better handling of minority languages and dialects")
    print("✓ More natural speech synthesis")
    print("✓ Automatic - no manual configuration needed")
    print()
    print("=" * 80)
    print("Supported Languages by Region")
    print("=" * 80)
    print()
    
    # Group by region/family
    language_groups = {
        "European": ["en", "es", "pt", "fr", "de", "it", "nl", "ru", "uk", "pl", "ro", "el", "cs", "fi"],
        "Asian": ["zh", "ja", "ko", "th", "hi", "id", "vi"],
        "Middle Eastern": ["ar", "tr"]
    }
    
    for group_name, lang_codes in language_groups.items():
        print(f"  {group_name}:")
        for lang_code in lang_codes:
            if lang_code in LANGUAGE_BOOST_MAPPING:
                boost_value = LANGUAGE_BOOST_MAPPING[lang_code]
                print(f"    • {lang_code:4s} - {boost_value}")
        print()
    
    print("=" * 80)
    print("Example Usage")
    print("=" * 80)
    print()
    
    examples = [
        ("es", "Spanish", "Spanish_male_speech", "¡Hola! ¿Cómo estás?"),
        ("ru", "Russian", "Russian_ReliableMan", "Здравствуйте! Как дела?"),
        ("zh", "Chinese", "Chinese (Mandarin)_Reliable_Executive", "你好！你好吗？"),
        ("ja", "Japanese", "Japanese_IntellectualSenior", "こんにちは！お元気ですか？"),
        ("pt", "Portuguese", "Portuguese_LevelheadedAnchor", "Olá! Como está?"),
    ]
    
    for lang_code, lang_name, default_voice, sample_text in examples:
        boost_value = LANGUAGE_BOOST_MAPPING.get(lang_code, "N/A")
        
        print(f"# {lang_name} ({lang_code})")
        print(f"tts = MinimaxTTSWrapper(target_language='{lang_code}')")
        print(f"# → Automatically uses voice: {default_voice}")
        print(f"# → Automatically sets language_boost: '{boost_value}'")
        print(f"# Sample text: {sample_text}")
        print()
    
    print("=" * 80)
    print()
    print("💡 Tip: The language_boost parameter is automatically included in:")
    print("   • Voice sample generation (during initialization)")
    print("   • Text-to-speech synthesis (during runtime)")
    print("   • Voice cloning (when cloning custom voices)")
    print()
    print("🔧 Voice Cloning Example:")
    print()
    print("  # Initialize for Japanese")
    print("  tts = MinimaxTTSWrapper(target_language='ja', model='speech-2.6-hd')")
    print()
    print("  # Clone voice - automatically uses:")
    print("  #   • model='speech-2.6-hd'")
    print("  #   • language_boost='Japanese'")
    print("  #   • need_noise_reduction=True (always enabled)")
    print("  #   • need_volume_normalization=True (always enabled)")
    print("  voice_id = tts.clone_voice('japanese_sample.mp3', 'my_jp_voice')")
    print()
    print("  Audio enhancement ensures cleaner, more professional cloned voices!")
    print()

if __name__ == "__main__":
    main()
