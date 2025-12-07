from typing import Optional, Dict, Any, List, Union, Tuple
import os
import requests
import json
import tempfile
import shutil
import time
from pathlib import Path
import numpy as np

from .models import TTSSegmentData, SegmentAlignment, DiarizationSegment
from .voice_sample_manager import VoiceSampleManager, AudioFileUtils
from src.tts.tts_interface import TTSInterface
from src.utils.audio_embedder import AudioEmbedder
from src.utils.voice_matcher import VoiceMatcher
from src.dubbing.core.log_config import get_logger
from pydub import AudioSegment

logger = get_logger(__name__)

# Minimax language-specific voice mapping
# Source: https://platform.minimax.io/docs/faq/system-voice-id
LANGUAGE_VOICE_MAPPING: Dict[str, List[str]] = {
    "en": [  # English
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
    "zh": [  # Chinese (Mandarin)
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
    "ja": [  # Japanese
        "Japanese_IntellectualSenior", "Japanese_DecisivePrincess", "Japanese_LoyalKnight",
        "Japanese_DominantMan"
    ],
    "ko": [  # Korean
        "Korean_CalmWoman", "Korean_FriendlyGuy"
    ],
    "es": [  # Spanish
        "Spanish_male_speech", "Spanish_Newscaster", "Spanish_PatientMan",
        "Spanish_CharmingLady", "Spanish_FriendlyGuy", "Spanish_ConfidentWoman"
    ],
    "pt": [  # Portuguese
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
    "fr": [  # French
        "French_Male_Speech_New", "French_Female_News Anchor", "French_CasualMan",
        "French_MovieLeadFemale", "French_FemaleAnchor", "French_MaleNarrator"
    ],
    "id": [  # Indonesian
        "Indonesian_SweetGirl", "Indonesian_ReservedYoungMan", "Indonesian_CharmingGirl",
        "Indonesian_CalmWoman", "Indonesian_ConfidentWoman", "Indonesian_CaringMan",
        "Indonesian_BossyLeader", "Indonesian_DeterminedBoy", "Indonesian_GentleGirl"
    ],
    "de": [  # German
        "German_FriendlyMan", "German_SweetLady", "German_PlayfulMan"
    ],
    "ru": [  # Russian
        "Russian_ReliableMan", "Russian_AttractiveGuy", "Russian_HandsomeChildhoodFriend", 
        "Russian_BrightHeroine", "Russian_AmbitiousWoman",
        "Russian_CrazyQueen", "Russian_PessimisticGirl",
        "Russian_Bad-temperedBoy"
    ],
    "it": [  # Italian
        "Italian_BraveHeroine", "Italian_Narrator", "Italian_WanderingSorcerer",
        "Italian_DiligentLeader"
    ],
    "nl": [  # Dutch
        "Dutch_kindhearted_girl", "Dutch_bossy_leader"
    ],
    "vi": [  # Vietnamese
        "Vietnamese_kindhearted_girl"
    ],
    "ar": [  # Arabic
        "Arabic_CalmWoman", "Arabic_FriendlyGuy"
    ],
    "tr": [  # Turkish
        "Turkish_CalmWoman", "Turkish_Trustworthyman"
    ],
    "uk": [  # Ukrainian
        "Ukrainian_CalmWoman", "Ukrainian_WiseScholar"
    ],
    "th": [  # Thai
        "Thai_male_1_sample8", "Thai_male_2_sample2", "Thai_female_1_sample1",
        "Thai_female_2_sample2"
    ],
    "pl": [  # Polish
        "Polish_male_1_sample4", "Polish_male_2_sample3", "Polish_female_1_sample1",
        "Polish_female_2_sample3"
    ],
    "ro": [  # Romanian
        "Romanian_male_1_sample2", "Romanian_male_2_sample1", "Romanian_female_1_sample4",
        "Romanian_female_2_sample1"
    ],
    "el": [  # Greek
        "greek_male_1a_v1", "Greek_female_1_sample1", "Greek_female_2_sample3"
    ],
    "cs": [  # Czech
        "czech_male_1_v1", "czech_female_5_v7", "czech_female_2_v2"
    ],
    "fi": [  # Finnish
        "finnish_male_3_v1", "finnish_male_1_v2", "finnish_female_4_v1"
    ],
    "hi": [  # Hindi
        "hindi_male_1_v2", "hindi_female_2_v1", "hindi_female_1_v2"
    ]
}

# Default voices per language (first voice in each list)
DEFAULT_VOICE_PER_LANGUAGE: Dict[str, str] = {
    lang: voices[0] for lang, voices in LANGUAGE_VOICE_MAPPING.items()
}

# Language boost mapping for improved recognition
# Maps language codes to MiniMax language_boost parameter values
# Source: https://platform.minimax.io/docs/api-reference/speech-t2a-http#body-language-boost
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

# Resolve samples directory relative to this file's location (src/tts/)
DEFAULT_SAMPLES_DIR = (Path(__file__).parent / "samples" / "minimax").resolve()
VOICE_SAMPLE_TEXT = """
Hello, this is a voice sample for analysis. The weather today is absolutely wonderful. 
Technology has transformed our lives in remarkable ways. I hope you're having a great day.
Let me share some interesting facts about science and nature with you.
"""

class MinimaxTTSWrapper(TTSInterface):
    """
    Minimax TTS wrapper with voice cloning capabilities.
    """
    
    BASE_URL = "https://api.minimax.io/v1"
    
    def __init__(
        self,
        model: str = "speech-02-hd",
        default_voice: Optional[str] = None,
        target_language: str = "en",
        api_key: Optional[str] = None,
        group_id: Optional[str] = None,
        embedding_model_device: Optional[str] = None,
        enable_voice_matching: bool = True,
        cost_tracker: Optional[Any] = None,
        **kwargs: Any
    ):
        self.model = model
        self.target_language = target_language
        
        # Set default voice based on target language
        if default_voice:
            self.default_voice = default_voice
        else:
            self.default_voice = DEFAULT_VOICE_PER_LANGUAGE.get(
                target_language, 
                "English_CaptivatingStoryteller"  # Fallback to English
            )
        
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        self.group_id = group_id or os.environ.get("MINIMAX_GROUP_ID")
        
        if not self.api_key:
            raise ValueError("Minimax API key not provided. Please set MINIMAX_API_KEY environment variable.")
            
        self.voice_mapping: Dict[str, str] = {}
        self.voice_prompt_mapping: Dict[str, str] = {}
        self._audio_cache: Dict[str, float] = {}
        self.cost_tracker = cost_tracker
        
        # Voice matching components
        self.enable_voice_matching = enable_voice_matching
        self.embedding_model_device = embedding_model_device
        self.audio_embedder: Optional[AudioEmbedder] = None
        self.voice_matcher: Optional[VoiceMatcher] = None
        self.voice_sample_manager: Optional[VoiceSampleManager] = None

    def initialize(self) -> None:
        """Initialize the Minimax TTS client and voice matching."""
        if not self.api_key:
            raise ValueError("Minimax API key is missing.")
        logger.info(f"Minimax TTS initialized with model: {self.model}, target language: {self.target_language}, default voice: {self.default_voice}")
        
        # Initialize voice matching if enabled
        if self.enable_voice_matching:
            try:
                self.audio_embedder = AudioEmbedder(device=self.embedding_model_device)
                self.voice_matcher = VoiceMatcher(
                    audio_embedder=self.audio_embedder,
                    enable_matching=True
                )
                logger.debug("Voice matching enabled with AudioEmbedder")
                
                # Get voice list for target language
                voice_list = LANGUAGE_VOICE_MAPPING.get(
                    self.target_language, 
                    LANGUAGE_VOICE_MAPPING.get("en", [])  # Fallback to English
                )
                
                if not voice_list:
                    logger.warning(f"No voices found for language '{self.target_language}'. Using English voices.")
                    voice_list = LANGUAGE_VOICE_MAPPING["en"]
                
                logger.info(f"Using {len(voice_list)} voices for language '{self.target_language}'")
                
                # Initialize VoiceSampleManager
                self.voice_sample_manager = VoiceSampleManager(
                    tts_provider="minimax",
                    voice_list=voice_list,
                    samples_dir=DEFAULT_SAMPLES_DIR / self.target_language,
                    stats_file=DEFAULT_SAMPLES_DIR / self.target_language / "minimax_voice_stats.json",
                    adjustments_file=DEFAULT_SAMPLES_DIR / self.target_language / "minimax_duration_adjustments.json",
                    sample_text=VOICE_SAMPLE_TEXT,
                    audio_embedder=self.audio_embedder,
                    voice_matcher=self.voice_matcher,
                    enable_voice_matching=True,
                    enable_audio_validation=False,  # Minimax samples don't need validation
                )
                
                # Set up sample generation callback
                def generate_minimax_sample(voice_name: str, output_path: str) -> bool:
                    """Callback for VoiceSampleManager to generate Minimax samples."""
                    try:
                        url = f"{self.BASE_URL}/t2a_v2"
                        headers = {
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        # Get language_boost for target language
                        language_boost = LANGUAGE_BOOST_MAPPING.get(self.target_language)
                        logger.debug(f"Language boost for {self.target_language}: {language_boost}")
                        
                        payload = {
                            "model": self.model,
                            "text": VOICE_SAMPLE_TEXT,
                            "stream": False,
                            "voice_setting": {
                                "voice_id": voice_name,
                                "speed": 1.0,
                                "vol": 1,
                                "pitch": 0
                            },
                            "audio_setting": {
                                "sample_rate": 32000,
                                "bitrate": 128000,
                                "format": "mp3",
                                "channel": 1
                            }
                        }
                        
                        # Add language_boost if available for target language
                        if language_boost:
                            payload["language_boost"] = language_boost
                        
                        response = requests.post(url, headers=headers, json=payload)
                        response.raise_for_status()
                        
                        result = response.json()
                        if result.get("base_resp", {}).get("status_code") != 0:
                            logger.error(f"Minimax sample generation failed: {result.get('base_resp', {}).get('status_msg')}")
                            return False
                        
                        audio_hex = result.get("data", {}).get("audio")
                        if not audio_hex:
                            logger.error("Minimax response missing audio data")
                            return False
                        
                        audio_bytes = bytes.fromhex(audio_hex)
                        with open(output_path, 'wb') as f:
                            f.write(audio_bytes)
                        return True
                    except Exception as e:
                        logger.error(f"Error generating sample for {voice_name}: {e}")
                        return False
                
                self.voice_sample_manager.set_sample_generator(generate_minimax_sample)
                
                # Load or generate voice samples and embeddings
                self.voice_sample_manager.generate_all_samples()
            except Exception as e:
                logger.warning(f"Warning: Failed to initialize voice matching: {e}")
                self.enable_voice_matching = False
                self.audio_embedder = None
                self.voice_matcher = None
                self.voice_sample_manager = None
        else:
            logger.info("Voice matching disabled by configuration.")

    def is_available(self) -> bool:
        return bool(self.api_key)
    
    @staticmethod
    def get_available_voices(language: str) -> List[str]:
        """
        Get available voices for a specific language.
        
        Args:
            language: Language code (e.g., 'en', 'es', 'zh', 'ja')
            
        Returns:
            List of voice IDs available for the language
        """
        return LANGUAGE_VOICE_MAPPING.get(language, [])
    
    @staticmethod
    def get_supported_languages() -> List[str]:
        """
        Get list of all supported language codes.
        
        Returns:
            List of supported language codes
        """
        return list(LANGUAGE_VOICE_MAPPING.keys())

    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_mapping = mapping
        logger.debug(f"Minimax voice mapping set: {len(mapping)} entries.")

    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_prompt_mapping = mapping
        logger.debug(f"Minimax voice prompt mapping set: {len(mapping)} entries.")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.voice_matcher:
            self.voice_matcher.clear()
            self.voice_matcher = None
        if self.audio_embedder:
            self.audio_embedder = None
    
    def find_and_pin_voice_for_speaker(self, speaker_id: str, reference_audio_path: Union[str, Path],
                                     force_search: bool = False) -> Optional[str]:
        """Find the best matching Minimax voice for a reference audio and pin it to the speaker."""
        if not self.enable_voice_matching or not self.voice_matcher:
            return self.voice_mapping.get(speaker_id, self.default_voice)
        
        if not force_search and speaker_id in self.voice_mapping:
            return self.voice_mapping[speaker_id]
        
        ref_path = Path(reference_audio_path)
        if not ref_path.exists():
            logger.warning(f"Reference audio file not found: {reference_audio_path}")
            return self.default_voice
        
        # Extract multiple embeddings from different parts of the audio
        reference_embeddings = self.voice_matcher.extract_multiple_embeddings(
            ref_path,
            num_segments=3,
            segment_duration_ms=3000
        )
        
        if not reference_embeddings:
            logger.warning(f"Could not extract embeddings from reference audio.")
            return self.default_voice
        
        logger.debug(f"Extracted {len(reference_embeddings)} embeddings from reference audio")
        
        # Find best matching voice using voting across multiple segments
        exclude_voices = list(self.voice_mapping.values())
        best_match_voice = self.voice_matcher.find_best_matching_voice_multi_segment(
            reference_embeddings,
            exclude_voices=exclude_voices
        )
        
        if best_match_voice:
            self.voice_mapping[speaker_id] = best_match_voice
            logger.info(f"Matched and pinned speaker '{speaker_id}' to voice '{best_match_voice}'")
            return best_match_voice
        
        logger.warning(f"Could not find matching voice for speaker '{speaker_id}'. Using default.")
        return self.default_voice

    def _upload_file(self, file_path: str, purpose: str = "voice_clone") -> str:
        """
        Upload a file to Minimax for voice cloning.
        
        Args:
            file_path: Path to the file to upload.
            purpose: Purpose of the file (voice_clone, prompt_audio, etc.)
            
        Returns:
            file_id: The ID of the uploaded file.
        """
        url = f"{self.BASE_URL}/files/upload"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            # Content-Type is set automatically by requests for multipart/form-data
        }
        
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {"purpose": purpose}
                
                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                
                result = response.json()
                if result.get("base_resp", {}).get("status_code") != 0:
                     raise RuntimeError(f"Minimax upload failed: {result.get('base_resp', {}).get('status_msg')}")
                
                file_id = result.get("file", {}).get("file_id")
                if not file_id:
                    raise RuntimeError("Minimax upload response missing file_id")
                
                logger.info(f"Uploaded file {file_path} to Minimax, file_id: {file_id}")
                return file_id
                
        except Exception as e:
            logger.error(f"Error uploading file to Minimax: {e}")
            raise

    def _limit_audio_duration(self, audio_path: str, max_duration_seconds: int = 300) -> str:
        """
        Limit audio file duration to max_duration_seconds (default 5 minutes).
        If audio exceeds the limit, create a trimmed temporary file.
        
        Args:
            audio_path: Path to the audio file.
            max_duration_seconds: Maximum duration in seconds (default 300 = 5 minutes).
            
        Returns:
            Path to the audio file (original or trimmed temporary file).
        """
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            duration_ms = len(audio_segment)
            duration_seconds = duration_ms / 1000.0
            
            if duration_seconds <= max_duration_seconds:
                logger.debug(f"Audio duration {duration_seconds:.2f}s is within limit ({max_duration_seconds}s)")
                return audio_path
            
            logger.warning(f"Audio duration {duration_seconds:.2f}s exceeds limit ({max_duration_seconds}s). Trimming to {max_duration_seconds}s.")
            
            max_duration_ms = max_duration_seconds * 1000
            trimmed_audio = audio_segment[:max_duration_ms]
            
            temp_file = tempfile.NamedTemporaryFile(suffix=Path(audio_path).suffix, delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            trimmed_audio.export(temp_path, format=Path(audio_path).suffix.lstrip('.'))
            logger.info(f"Created trimmed audio file: {temp_path} (duration: {max_duration_seconds}s)")
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Error processing audio duration: {e}. Using original file.")
            return audio_path

    def clone_voice(self, audio_path: str, voice_id: str) -> str:
        """
        Clone a voice using Minimax API with automatic audio enhancement.
        
        Automatically applies noise reduction and volume normalization for optimal
        voice cloning quality. Also uses target language's language_boost parameter
        for improved recognition.
        
        Args:
            audio_path: Path to the reference audio file (mp3, m4a, wav).
                       Audio duration: 10 seconds to 5 minutes.
                       File size: max 20 MB.
            voice_id: Desired ID for the cloned voice (8-256 chars, must start with letter).
            
        Returns:
            voice_id: The ID of the cloned voice (same as input if successful).
            
        Note:
            - Noise reduction and volume normalization are always enabled
            - Uses model and language_boost from instance configuration
            - Audio is automatically trimmed to 5 minutes if longer
        """
        # 1. Limit audio duration to 5 minutes
        limited_audio_path = self._limit_audio_duration(audio_path, max_duration_seconds=300)
        temp_file_created = limited_audio_path != audio_path
        
        try:
            # 2. Upload file
            file_id = self._upload_file(limited_audio_path, purpose="voice_clone")
            
            # 3. Clone voice
            url = f"{self.BASE_URL}/voice_clone"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Get language_boost for target language
            language_boost = LANGUAGE_BOOST_MAPPING.get(self.target_language)
            
            payload = {
                "file_id": file_id,
                "voice_id": voice_id,
                "model": self.model,  # Include model for consistency
                "need_noise_reduction": True,  # Enable noise reduction for better voice quality
                "need_volume_normalization": True,  # Enable volume normalization for consistent audio levels
                # Optional: clone_prompt could be added here if we had a prompt audio/text
            }
            
            # Add language_boost if available for target language
            if language_boost:
                payload["language_boost"] = language_boost
                logger.debug(f"Using language_boost: {language_boost} for voice cloning")
            
            logger.debug("Voice cloning with noise_reduction=True and volume_normalization=True")
            
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                if result.get("base_resp", {}).get("status_code") != 0:
                     raise RuntimeError(f"Minimax voice clone failed: {result.get('base_resp', {}).get('status_msg')}")
                
                logger.info(f"Successfully cloned voice {voice_id} using file {file_id} with model {self.model}")
                return voice_id
                
            except Exception as e:
                logger.error(f"Error cloning voice with Minimax: {e}")
                raise
        finally:
            if temp_file_created and os.path.exists(limited_audio_path):
                try:
                    os.unlink(limited_audio_path)
                    logger.debug(f"Cleaned up temporary trimmed audio file: {limited_audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {limited_audio_path}: {e}")

    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: Optional[str] = None,
        **kwargs: Any
    ) -> List[SegmentAlignment]:
        language = language or self.target_language
        
        # Auto-pin voices for unmapped speakers with reference audio
        if self.enable_voice_matching and self.voice_matcher:
            speaker_to_ref_path = {}
            for segment in segments_data:
                if (segment.speaker and segment.speaker not in self.voice_mapping and
                    segment.speaker not in speaker_to_ref_path and segment.reference_audio_path):
                    speaker_to_ref_path[segment.speaker] = segment.reference_audio_path
            
            for speaker_id, ref_path in speaker_to_ref_path.items():
                logger.debug(f"Auto-pinning voice for speaker '{speaker_id}'")
                self.find_and_pin_voice_for_speaker(speaker_id, ref_path)
        
        alignments = []
        
        for segment in segments_data:
            # Determine voice
            voice_id = segment.voice or self.voice_mapping.get(segment.speaker, self.default_voice)

            # Prepare request
            url = f"{self.BASE_URL}/t2a_v2"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Get language_boost for target language
            language_boost = LANGUAGE_BOOST_MAPPING.get(self.target_language)
            
            payload = {
                "model": self.model,
                "text": segment.text,
                "stream": False,
                "voice_setting": {
                    "voice_id": voice_id,
                    #"speed": segment.speed or 1.0,
                    "vol": 1,
                    "pitch": 0
                },
                "audio_setting": {
                    "sample_rate": 32000,
                    "bitrate": 128000,
                    "format": "mp3",
                    "channel": 1
                }
            }
            
            # Add language_boost if available for target language
            if language_boost:
                payload["language_boost"] = language_boost
                logger.debug(f"Using language_boost: {language_boost} for target language: {self.target_language}")
            
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                if result.get("base_resp", {}).get("status_code") != 0:
                     logger.error(f"Minimax synthesis failed: {result.get('base_resp', {}).get('status_msg')}")
                     continue
                
                audio_hex = result.get("data", {}).get("audio")
                if not audio_hex:
                    logger.error("Minimax response missing audio data")
                    continue
                    
                audio_bytes = bytes.fromhex(audio_hex)
                
                # Save to file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    temp_path = tmp_file.name
                
                # Get duration
                audio_segment = AudioSegment.from_mp3(temp_path)
                duration = len(audio_segment) / 1000.0
                
                # Handle output path
                if segment.output_path:
                    if os.path.dirname(segment.output_path):
                        os.makedirs(os.path.dirname(segment.output_path), exist_ok=True)
                    shutil.move(temp_path, segment.output_path)
                    final_path = segment.output_path
                else:
                    final_path = temp_path # Keep temp file if no output path
                
                # Create alignment
                diarized = DiarizationSegment(
                    start_time=0.0,
                    end_time=duration,
                    speaker=segment.speaker,
                    text=segment.text,
                    confidence=1.0
                )
                alignments.append(SegmentAlignment(
                    original_segment=segment,
                    diarized_segment=diarized,
                    alignment_confidence=1.0
                ))
                
            except Exception as e:
                logger.error(f"Error synthesizing segment for speaker {segment.speaker}: {e}")
                
        return alignments

    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: Optional[str] = None
    ) -> Optional[float]:
        """
        Estimate audio segment length using unified Gemini duration estimation algorithm.
        
        Args:
            segment_data: TTSSegmentData object containing text and voice parameters
            language: Target language code (uses target_language from init if not specified)
            
        Returns:
            Estimated duration in seconds
        """
        if not segment_data.text or not segment_data.text.strip():
            return 0.0
        
        language = language or self.target_language
        
        # Use VoiceSampleManager for unified estimation if available
        if self.voice_sample_manager:
            voice_id = segment_data.voice or self.voice_mapping.get(segment_data.speaker, self.default_voice)
            
            return self.voice_sample_manager.estimate_duration(
                text=segment_data.text,
                voice_name=voice_id,
                language=language,
                style_prompt=segment_data.style_prompt,
                emotion=segment_data.emotion,
                speed=segment_data.speed,
                speaker_id=segment_data.speaker,
                apply_biases=True
            )
        
        # Fallback if VoiceSampleManager not available
        words = len(segment_data.text.split())
        duration = words / 2.5
        return max(1.0, duration)
