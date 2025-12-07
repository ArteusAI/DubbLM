#!/usr/bin/env python3
"""
Overview of MiniMax TTS language-specific voice mapping.

This is a standalone script that shows the voice configuration
without requiring the full DubbLM environment.
"""

from typing import Dict, List

# Minimax language-specific voice mapping
# Source: https://platform.minimax.io/docs/faq/system-voice-id
LANGUAGE_VOICE_MAPPING: Dict[str, List[str]] = {
    "en": [  # English - 45 voices
        "English_expressive_narrator", "English_radiant_girl", "English_magnetic_voiced_man",
        "English_compelling_lady1", "English_Aussie_Bloke", "English_captivating_female1",
        "English_Upbeat_Woman", "English_Trustworth_Man", "English_CalmWoman",
        "English_UpsetGirl", "English_Gentle-voiced_man", "English_Whispering_girl",
        "English_Diligent_Man", "English_Graceful_Lady", "English_ReservedYoungMan",
        "English_PlayfulGirl", "English_ManWithDeepVoice", "English_MaturePartner",
        "English_FriendlyPerson", "English_MatureBoss", "English_Debator",
        "English_LovelyGirl", "English_Steadymentor", "English_Deep-VoicedGentleman",
        "English_Wiselady", "English_CaptivatingStoryteller", "English_DecentYoungMan",
        "English_SentimentalLady", "English_ImposingManner", "English_SadTeen",
        "English_PassionateWarrior", "English_WiseScholar", "English_Soft-spokenGirl",
        "English_SereneWoman", "English_ConfidentWoman", "English_PatientMan",
        "English_Comedian", "English_BossyLeader", "English_Strong-WilledBoy",
        "English_StressedLady", "English_AssertiveQueen", "English_AnimeCharacter",
        "English_Jovialman", "English_WhimsicalGirl", "English_Kind-heartedGirl"
    ],
    "zh": [  # Chinese (Mandarin) - 34 voices
        "Chinese (Mandarin)_Reliable_Executive", "Chinese (Mandarin)_News_Anchor",
        "Chinese (Mandarin)_Unrestrained_Young_Man", "Chinese (Mandarin)_Mature_Woman",
        "Arrogant_Miss", "Robot_Armor", "Chinese (Mandarin)_Kind-hearted_Antie",
        "Chinese (Mandarin)_HK_Flight_Attendant", "Chinese (Mandarin)_Humorous_Elder",
        "Chinese (Mandarin)_Gentleman", "Chinese (Mandarin)_Warm_Bestie",
        "Chinese (Mandarin)_Stubborn_Friend", "Chinese (Mandarin)_Sweet_Lady",
        "Chinese (Mandarin)_Southern_Young_Man", "Chinese (Mandarin)_Wise_Women",
        "Chinese (Mandarin)_Gentle_Youth", "Chinese (Mandarin)_Warm_Girl",
        "Chinese (Mandarin)_Male_Announcer", "Chinese (Mandarin)_Kind-hearted_Elder",
        "Chinese (Mandarin)_Cute_Spirit", "Chinese (Mandarin)_Radio_Host",
        "Chinese (Mandarin)_Lyrical_Voice", "Chinese (Mandarin)_Straightforward_Boy",
        "Chinese (Mandarin)_Sincere_Adult", "Chinese (Mandarin)_Gentle_Senior",
        "Chinese (Mandarin)_Crisp_Girl", "Chinese (Mandarin)_Pure-hearted_Boy",
        "Chinese (Mandarin)_Soft_Girl", "Chinese (Mandarin)_IntellectualGirl",
        "Chinese (Mandarin)_Warm_HeartedGirl", "Chinese (Mandarin)_Laid_BackGirl",
        "Chinese (Mandarin)_ExplorativeGirl", "Chinese (Mandarin)_Warm-HeartedAunt",
        "Chinese (Mandarin)_BashfulGirl"
    ],
    "ja": ["Japanese_IntellectualSenior", "Japanese_DecisivePrincess", "Japanese_LoyalKnight", "Japanese_DominantMan"],
    "ko": ["Korean_CalmWoman", "Korean_FriendlyGuy"],
    "es": ["Spanish_male_speech", "Spanish_Newscaster", "Spanish_PatientMan", "Spanish_CharmingLady", "Spanish_FriendlyGuy", "Spanish_ConfidentWoman"],
    "pt": [  # Portuguese - 44 voices
        "Portuguese_LevelheadedAnchor", "Portuguese_KindMan", "Portuguese_TolerantWoman",
        "Portuguese_PassionateSoul", "Portuguese_SensitiveWoman", "Portuguese_WarmMan",
        "Portuguese_HonestMan", "Portuguese_CalmGirl", "Portuguese_ConfidentMan",
        "Portuguese_CharmingLady", "Portuguese_BoldVoice", "Portuguese_SoftMan",
        "Portuguese_WarmLady", "Portuguese_ConsiderateMan", "Portuguese_ReliableMan",
        "Portuguese_CalmWoman", "Portuguese_GentleGirl", "Portuguese_IntellectualMan",
        "Portuguese_OptimisticLady", "Portuguese_FriendlyGuy", "Portuguese_SteadyMentor",
        "Portuguese_AmbitiousMan", "Portuguese_PositiveSoul", "Portuguese_AssertiveQueen",
        "Portuguese_WhimsicalGirl", "Portuguese_StressedLady", "Portuguese_FriendlyNeighbor",
        "Portuguese_CaringGirlfriend", "Portuguese_PowerfulSoldier", "Portuguese_FascinatingBoy",
        "Portuguese_RomanticHusband", "Portuguese_StrictBoss", "Portuguese_InspiringLady",
        "Portuguese_PlayfulSpirit", "Portuguese_ElegantGirl", "Portuguese_CompellingGirl",
        "Portuguese_PowerfulVeteran", "Portuguese_SensibleManager", "Portuguese_ThoughtfulLady",
        "Portuguese_TheatricalActor", "Portuguese_FragileBoy", "Portuguese_ChattyGirl",
        "Portuguese_Conscientiousinstructor", "Portuguese_RationalMan", "Portuguese_WiseScholar",
        "Portuguese_FrankLady", "Portuguese_DeterminedManager"
    ],
    "fr": ["French_Male_Speech_New", "French_Female_News Anchor", "French_CasualMan", "French_MovieLeadFemale", "French_FemaleAnchor", "French_MaleNarrator"],
    "id": ["Indonesian_SweetGirl", "Indonesian_ReservedYoungMan", "Indonesian_CharmingGirl", "Indonesian_CalmWoman", "Indonesian_ConfidentWoman", "Indonesian_CaringMan", "Indonesian_BossyLeader", "Indonesian_DeterminedBoy", "Indonesian_GentleGirl"],
    "de": ["German_FriendlyMan", "German_SweetLady", "German_PlayfulMan"],
    "ru": ["Russian_HandsomeChildhoodFriend", "Russian_BrightHeroine", "Russian_AmbitiousWoman", "Russian_ReliableMan", "Russian_CrazyQueen", "Russian_PessimisticGirl", "Russian_AttractiveGuy", "Russian_Bad-temperedBoy"],
    "it": ["Italian_BraveHeroine", "Italian_Narrator", "Italian_WanderingSorcerer", "Italian_DiligentLeader"],
    "nl": ["Dutch_kindhearted_girl", "Dutch_bossy_leader"],
    "vi": ["Vietnamese_kindhearted_girl"],
    "ar": ["Arabic_CalmWoman", "Arabic_FriendlyGuy"],
    "tr": ["Turkish_CalmWoman", "Turkish_Trustworthyman"],
    "uk": ["Ukrainian_CalmWoman", "Ukrainian_WiseScholar"],
    "th": ["Thai_male_1_sample8", "Thai_male_2_sample2", "Thai_female_1_sample1", "Thai_female_2_sample2"],
    "pl": ["Polish_male_1_sample4", "Polish_male_2_sample3", "Polish_female_1_sample1", "Polish_female_2_sample3"],
    "ro": ["Romanian_male_1_sample2", "Romanian_male_2_sample1", "Romanian_female_1_sample4", "Romanian_female_2_sample1"],
    "el": ["greek_male_1a_v1", "Greek_female_1_sample1", "Greek_female_2_sample3"],
    "cs": ["czech_male_1_v1", "czech_female_5_v7", "czech_female_2_v2"],
    "fi": ["finnish_male_3_v1", "finnish_male_1_v2", "finnish_female_4_v1"],
    "hi": ["hindi_male_1_v2", "hindi_female_2_v1", "hindi_female_1_v2"]
}

# Language names for display
LANGUAGE_NAMES = {
    "en": "English", "zh": "Chinese (Mandarin)", "ja": "Japanese", "ko": "Korean",
    "es": "Spanish", "pt": "Portuguese", "fr": "French", "id": "Indonesian",
    "de": "German", "ru": "Russian", "it": "Italian", "nl": "Dutch",
    "vi": "Vietnamese", "ar": "Arabic", "tr": "Turkish", "uk": "Ukrainian",
    "th": "Thai", "pl": "Polish", "ro": "Romanian", "el": "Greek",
    "cs": "Czech", "fi": "Finnish", "hi": "Hindi"
}

def main():
    print("=" * 80)
    print("MiniMax TTS - Language-Specific Voice Configuration")
    print("=" * 80)
    print()
    
    # Summary statistics
    total_voices = sum(len(voices) for voices in LANGUAGE_VOICE_MAPPING.values())
    print(f"📊 Summary:")
    print(f"   • Supported Languages: {len(LANGUAGE_VOICE_MAPPING)}")
    print(f"   • Total Voices: {total_voices}")
    print()
    
    # Voice count per language
    print("🎤 Voice Count per Language:")
    print()
    for lang_code in sorted(LANGUAGE_VOICE_MAPPING.keys()):
        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
        voice_count = len(LANGUAGE_VOICE_MAPPING[lang_code])
        default_voice = LANGUAGE_VOICE_MAPPING[lang_code][0]
        print(f"   {lang_code:4s} ({lang_name:20s}): {voice_count:3d} voices")
        print(f"        Default: {default_voice}")
    
    print()
    print("=" * 80)
    print("🌟 Key Features")
    print("=" * 80)
    print()
    print("✓ Automatic voice selection based on target language")
    print("✓ Default voice automatically set per language")
    print("✓ Automatic language_boost for enhanced recognition")
    print("✓ Language-specific sample directories")
    print("✓ Support for 23 languages")
    print("✓ 45+ English voices, 34+ Chinese voices, 47+ Portuguese voices")
    print()
    
    print("=" * 80)
    print("📖 Usage Example")
    print("=" * 80)
    print()
    print("# Before (hardcoded list):")
    print("tts = MinimaxTTSWrapper(default_voice='Russian_AttractiveGuy')")
    print()
    print("# After (language-specific):")
    print("tts = MinimaxTTSWrapper(target_language='es')  # Spanish")
    print("# Automatically uses: Spanish_male_speech")
    print()
    print("tts = MinimaxTTSWrapper(target_language='ja')  # Japanese")
    print("# Automatically uses: Japanese_IntellectualSenior")
    print()
    print("tts = MinimaxTTSWrapper(target_language='zh')  # Chinese")
    print("# Automatically uses: Chinese (Mandarin)_Reliable_Executive")
    print()
    
    print("=" * 80)
    print("🔗 Reference")
    print("=" * 80)
    print()
    print("https://platform.minimax.io/docs/faq/system-voice-id")
    print()

if __name__ == "__main__":
    main()

