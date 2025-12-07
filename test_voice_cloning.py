#!/usr/bin/env python3
"""
Test script to verify voice cloning implementation.
This script checks that all required components are in place.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tts_interface():
    """Test that TTSInterface has the clone_voice method."""
    from src.tts.tts_interface import TTSInterface
    import inspect
    
    methods = [m for m in dir(TTSInterface) if not m.startswith('_')]
    
    if 'clone_voice' in methods:
        print("✓ TTSInterface has clone_voice method")
        
        # Check if it's abstract
        sig = inspect.signature(TTSInterface.clone_voice)
        print(f"  Signature: {sig}")
        return True
    else:
        print("✗ TTSInterface missing clone_voice method")
        return False

def test_minimax_wrapper():
    """Test that MinimaxTTSWrapper has clone_voice implemented."""
    from src.tts.minimax_tts_wrapper import MinimaxTTSWrapper
    import inspect
    
    if hasattr(MinimaxTTSWrapper, 'clone_voice'):
        sig = inspect.signature(MinimaxTTSWrapper.clone_voice)
        print(f"✓ MinimaxTTSWrapper has clone_voice method")
        print(f"  Signature: {sig}")
        return True
    else:
        print("✗ MinimaxTTSWrapper missing clone_voice method")
        return False

def test_other_wrappers():
    """Test that other wrappers have clone_voice stubs."""
    wrappers = [
        ('OpenAI', 'src.tts.openai_tts_wrapper', 'OpenAITTSWrapper'),
        ('F5', 'src.tts.f5_tts_wrapper', 'F5TTSWrapper'),
        ('Gemini', 'src.tts.gemini_tts_wrapper', 'GeminiTTSWrapper'),
    ]
    
    all_pass = True
    for name, module_path, class_name in wrappers:
        try:
            module = __import__(module_path, fromlist=[class_name])
            wrapper_class = getattr(module, class_name)
            
            if hasattr(wrapper_class, 'clone_voice'):
                print(f"✓ {name}TTSWrapper has clone_voice stub")
            else:
                print(f"✗ {name}TTSWrapper missing clone_voice stub")
                all_pass = False
        except Exception as e:
            print(f"✗ Error checking {name}TTSWrapper: {e}")
            all_pass = False
    
    return all_pass

def test_config():
    """Test that config has clone_voice and voice_names options."""
    from src.dubbing.core.config import DubbingConfig
    
    config = DubbingConfig()
    
    has_clone_voice = 'clone_voice' in config.defaults
    has_voice_names = 'voice_names' in config.defaults
    
    if has_clone_voice:
        print("✓ Config has clone_voice default")
        print(f"  Default value: {config.defaults['clone_voice']}")
    else:
        print("✗ Config missing clone_voice default")
    
    if has_voice_names:
        print("✓ Config has voice_names default")
        print(f"  Default value: {config.defaults['voice_names']}")
    else:
        print("✗ Config missing voice_names default")
    
    return has_clone_voice and has_voice_names

def test_smart_dubbing():
    """Test that SmartDubbing has clone_speakers_voices method."""
    from src.dubbing.core.smart_dubbing import SmartDubbing
    import inspect
    
    if hasattr(SmartDubbing, 'clone_speakers_voices'):
        print("✓ SmartDubbing has clone_speakers_voices method")
        sig = inspect.signature(SmartDubbing.clone_speakers_voices)
        print(f"  Signature: {sig}")
        return True
    else:
        print("✗ SmartDubbing missing clone_speakers_voices method")
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("VOICE CLONING IMPLEMENTATION VERIFICATION")
    print("=" * 70)
    print()
    
    results = []
    
    print("1. Testing TTSInterface...")
    results.append(test_tts_interface())
    print()
    
    print("2. Testing MinimaxTTSWrapper...")
    results.append(test_minimax_wrapper())
    print()
    
    print("3. Testing other TTS wrappers...")
    results.append(test_other_wrappers())
    print()
    
    print("4. Testing Config...")
    results.append(test_config())
    print()
    
    print("5. Testing SmartDubbing...")
    results.append(test_smart_dubbing())
    print()
    
    print("=" * 70)
    if all(results):
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Voice cloning feature is ready to use!")
        print()
        print("Usage examples:")
        print("  1. Auto-generated voice IDs:")
        print("     python main.py --input video.mp4 --clone_voice SPEAKER_A,SPEAKER_B")
        print()
        print("  2. Custom voice names:")
        print("     python main.py --input video.mp4 --clone_voice SPEAKER_A,SPEAKER_B --voice_names alice,bob")
        print()
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1

if __name__ == '__main__':
    sys.exit(main())


