import os
from pathlib import Path

from src.tts.minimax_tts_wrapper import MinimaxTTSWrapper
from src.tts.models import TTSSegmentData

from dotenv import load_dotenv
load_dotenv()  

def verify_minimax():
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("Error: MINIMAX_API_KEY not set.")
        return

    print("Initializing MinimaxTTSWrapper for Russian...")
    # Use target_language to automatically select appropriate Russian voices
    tts = MinimaxTTSWrapper(
        api_key=api_key,
        target_language="ru"  # Russian
    )
    tts.initialize()
    
    print(f"Default voice for Russian: {tts.default_voice}")
    print(f"Available Russian voices: {MinimaxTTSWrapper.get_available_voices('ru')}")
    
    # Test Synthesis
    print("\nTesting Synthesis...")
    segment = TTSSegmentData(
        speaker="Russian_AttractiveGuy",
        text="Обучение с подкреплением действительно отлично зарекомендовало себя в компьютерных играх. Но при этом оно также всё более активно развивается и набирает реальные обороты в оптимизации роботов, различных логистических систем и многих подобных практических областях. Вы изучите все эти применения в курсе. И последнее, что хочу сказать на сегодня: я очень надеюсь, что вы начнёте активно общаться, знакомиться с однокурсниками, заводить друзей, искать партнёров для совместных проектов и создавать учебные группы.",
        output_path="minimax_test.mp3"
    )
    
    alignments = tts.synthesize([segment])
    
    if alignments and os.path.exists("minimax_test.mp3"):
        print("Synthesis successful! Audio saved to minimax_test.mp3")
        print(f"Duration: {alignments[0].diarized_segment.end_time:.2f}s")
    else:
        print("Synthesis failed.")

    # Test Voice Cloning (Mock or Real if file exists)
    # Note: Voice cloning automatically uses:
    #   - target_language for language_boost ('Russian')
    #   - model specified during initialization ('speech-2.6-turbo')
    #   - noise_reduction=True (always enabled)
    #   - volume_normalization=True (always enabled)
    # print("\nTesting Voice Cloning...")
    # try:
    #     # Will automatically use:
    #     #   model='speech-2.6-turbo'
    #     #   language_boost='Russian'
    #     #   need_noise_reduction=True
    #     #   need_volume_normalization=True
    #     voice_id = tts.clone_voice("path/to/sample.mp3", "test_clone_voice_ru")
    #     print(f"Cloning successful! Voice ID: {voice_id}")
    #     print(f"Voice cloned with model: {tts.model}, language_boost: Russian")
    #     print(f"Audio enhancement: noise_reduction=True, volume_normalization=True")
    # except Exception as e:
    #     print(f"Cloning failed (expected if file missing): {e}")

if __name__ == "__main__":
    verify_minimax()
