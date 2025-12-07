"""Example of high-quality time-stretching with timbre preservation."""

import os
from pydub import AudioSegment
from src.dubbing.audio.time_stretcher import TimeStretcher


def example_basic_usage():
    """Basic time-stretching example."""
    print("=" * 70)
    print("Example 1: Basic Time-Stretching")
    print("=" * 70)
    
    stretcher = TimeStretcher(preferred_method='auto')
    
    input_file = "test_audio.wav"
    output_file = "test_audio_stretched.wav"
    
    if not os.path.exists(input_file):
        print(f"Creating test audio file: {input_file}")
        silence = AudioSegment.silent(duration=5000)
        silence.export(input_file, format="wav")
    
    print(f"\nStretching audio to 80% speed (slower)...")
    success = stretcher.stretch(
        input_path=input_file,
        output_path=output_file,
        tempo_ratio=0.8
    )
    
    if success:
        print(f"✅ Success! Output saved to: {output_file}")
        
        original = AudioSegment.from_file(input_file)
        stretched = AudioSegment.from_file(output_file)
        
        print(f"\nOriginal duration: {len(original)/1000:.2f}s")
        print(f"Stretched duration: {len(stretched)/1000:.2f}s")
        print(f"Expected duration: {len(original)/1000/0.8:.2f}s")
    else:
        print("❌ Stretching failed")
    
    print()


def example_method_comparison():
    """Compare different stretching methods."""
    print("=" * 70)
    print("Example 2: Method Comparison")
    print("=" * 70)
    
    input_file = "test_audio.wav"
    
    if not os.path.exists(input_file):
        print(f"Creating test audio file: {input_file}")
        silence = AudioSegment.silent(duration=5000)
        silence.export(input_file, format="wav")
    
    methods = ['rubberband', 'atempo_chain', 'atempo']
    tempo_ratio = 0.75  # Slow down to 75% speed
    
    print(f"\nTesting all methods with tempo ratio: {tempo_ratio}")
    print()
    
    for method in methods:
        stretcher = TimeStretcher(preferred_method=method)
        output_file = f"test_audio_{method}.wav"
        
        print(f"Testing method: {method}")
        success = stretcher.stretch(
            input_path=input_file,
            output_path=output_file,
            tempo_ratio=tempo_ratio
        )
        
        if success:
            stretched = AudioSegment.from_file(output_file)
            print(f"  ✅ Success - Duration: {len(stretched)/1000:.2f}s")
            
            os.remove(output_file)
        else:
            print(f"  ❌ Failed (method may not be available)")
        
        print()


def example_extreme_ratios():
    """Test extreme tempo ratios."""
    print("=" * 70)
    print("Example 3: Extreme Tempo Ratios")
    print("=" * 70)
    
    input_file = "test_audio.wav"
    
    if not os.path.exists(input_file):
        print(f"Creating test audio file: {input_file}")
        silence = AudioSegment.silent(duration=5000)
        silence.export(input_file, format="wav")
    
    stretcher = TimeStretcher(preferred_method='auto')
    
    test_ratios = [
        (0.5, "Half speed (very slow)"),
        (0.75, "75% speed (slower)"),
        (1.0, "Normal speed"),
        (1.25, "125% speed (faster)"),
        (2.0, "Double speed (very fast)")
    ]
    
    print("\nTesting various tempo ratios:\n")
    
    for ratio, description in test_ratios:
        output_file = f"test_audio_{ratio:.2f}x.wav"
        
        print(f"{description} (ratio={ratio:.2f})")
        success = stretcher.stretch(
            input_path=input_file,
            output_path=output_file,
            tempo_ratio=ratio
        )
        
        if success:
            original = AudioSegment.from_file(input_file)
            stretched = AudioSegment.from_file(output_file)
            expected_duration = len(original) / 1000 / ratio
            actual_duration = len(stretched) / 1000
            
            print(f"  Original: {len(original)/1000:.2f}s")
            print(f"  Expected: {expected_duration:.2f}s")
            print(f"  Actual:   {actual_duration:.2f}s")
            print(f"  Accuracy: {(actual_duration/expected_duration)*100:.1f}%")
            print(f"  ✅ Success")
            
            os.remove(output_file)
        else:
            print(f"  ❌ Failed")
        
        print()


def example_audio_segment():
    """Example using AudioSegment objects directly."""
    print("=" * 70)
    print("Example 4: Working with AudioSegment Objects")
    print("=" * 70)
    
    stretcher = TimeStretcher(preferred_method='auto')
    
    print("\nCreating test audio segment...")
    audio = AudioSegment.silent(duration=3000)
    
    print(f"Original duration: {len(audio)/1000:.2f}s")
    
    print("\nStretching to 120% speed (faster)...")
    stretched = stretcher.stretch_audio_segment(
        audio=audio,
        tempo_ratio=1.2
    )
    
    if stretched:
        print(f"Stretched duration: {len(stretched)/1000:.2f}s")
        print(f"Expected duration: {len(audio)/1000/1.2:.2f}s")
        print("✅ Success!")
    else:
        print("❌ Failed")
    
    print()


def example_voice_timbre_preservation():
    """Demonstrate why RubberBand is better for voice."""
    print("=" * 70)
    print("Example 5: Voice Timbre Preservation")
    print("=" * 70)
    
    print("\nWhy RubberBand preserves voice timbre better:\n")
    
    print("1. Formant Preservation:")
    print("   - Formants are frequency peaks that define voice character")
    print("   - RubberBand --formant flag preserves these frequencies")
    print("   - Result: Voice sounds natural even at different speeds")
    print()
    
    print("2. Phase Vocoder:")
    print("   - Analyzes frequency spectrum of audio")
    print("   - Stretches time domain while preserving frequency domain")
    print("   - Result: No pitch shifting (voice doesn't sound chipmunk-like)")
    print()
    
    print("3. WSOLA (Waveform Similarity Overlap-Add):")
    print("   - Finds similar waveform patterns")
    print("   - Overlaps them smoothly when stretching")
    print("   - Result: No audible glitches or artifacts")
    print()
    
    print("Comparison with simple speed change:")
    print("  - Simple speed change: Changes both speed AND pitch")
    print("    Example: 2x faster = chipmunk voice")
    print("  - RubberBand: Changes only speed, preserves pitch")
    print("    Example: 2x faster = same voice, just faster")
    print()
    
    print("Installation:")
    print("  Ubuntu/Debian: sudo apt-get install rubberband-cli")
    print("  MacOS: brew install rubberband")
    print("  Arch: sudo pacman -S rubberband")
    print()


def cleanup():
    """Clean up test files."""
    test_files = [
        "test_audio.wav",
        "test_audio_stretched.wav"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    
    print("Test files cleaned up.")


if __name__ == "__main__":
    try:
        example_basic_usage()
        example_method_comparison()
        example_extreme_ratios()
        example_audio_segment()
        example_voice_timbre_preservation()
    finally:
        cleanup()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)

