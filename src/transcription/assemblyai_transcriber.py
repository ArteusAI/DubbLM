"""
Implementation of transcription and diarization using AssemblyAI API.
"""
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING

from src.transcription.transcription_interface import BaseTranscriber
from src.dubbing.core.log_config import get_logger

if TYPE_CHECKING:
    from src.dubbing.core.cache_manager import CacheManager

logger = get_logger(__name__)


class AssemblyAITranscriber(BaseTranscriber):
    """Transcription and diarization service using AssemblyAI API."""
    
    def __init__(
        self,
        source_language: str,
        device: Optional[str] = None,
        speech_model: str = "best",
        cache_manager: Optional['CacheManager'] = None,
        convert_to_mp3: bool = True,
        mp3_bitrate: str = "256k",
        mp3_size_threshold_mb: int = 20,
        cost_tracker: Optional[Any] = None,
        speakers_expected: Optional[int] = None,
        enable_gemini_speaker_labeling: bool = False,
        gemini_speaker_labeling_model: Optional[str] = None,
        gemini_speaker_labeling_max_speakers: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the AssemblyAI transcription service.
        
        Args:
            source_language: Source language code (e.g., 'en')
            device: Compute device (not used for API-based service)
            speech_model: AssemblyAI speech model to use ('best', 'nano')
            cache_manager: Cache manager instance for organized caching
            convert_to_mp3: Whether to convert audio files to MP3 before upload
            mp3_bitrate: MP3 bitrate for conversion (e.g., '128k', '192k', '256k')
            mp3_size_threshold_mb: Only convert files larger than this size in MB
            speakers_expected: Expected number of speakers for diarization (None for auto-detect)
            **kwargs: Additional parameters
        """
        super().__init__(source_language, device, **kwargs)
        self.speech_model = speech_model
        self.cache_manager = cache_manager
        self.convert_to_mp3 = convert_to_mp3
        self.mp3_bitrate = mp3_bitrate
        self.mp3_size_threshold_mb = mp3_size_threshold_mb
        self.cost_tracker = cost_tracker
        self.speakers_expected = speakers_expected
        self.enable_gemini_speaker_labeling = bool(enable_gemini_speaker_labeling)
        from src.transcription.gemini_speaker_labeler import DEFAULT_MODEL as _GEMINI_LABEL_MODEL

        self.gemini_speaker_labeling_model = gemini_speaker_labeling_model or _GEMINI_LABEL_MODEL
        self.gemini_speaker_labeling_max_speakers = int(gemini_speaker_labeling_max_speakers or 4)
        
        # Get API key from environment
        self.api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY environment variable is required")
        
        # Lazy loading of assemblyai to avoid initial import overhead
        self._aai = None
    
    @property
    def aai(self):
        """Lazy-load assemblyai module when needed."""
        if self._aai is None:
            try:
                import assemblyai as aai
                aai.settings.api_key = self.api_key
                self._aai = aai
            except ImportError:
                raise ImportError(
                    "AssemblyAI package not found. Install it with: pip install assemblyai"
                )
        return self._aai
    
    @property
    def name(self) -> str:
        """Return the name of the transcription service implementation."""
        return "AssemblyAI"
    
    def _convert_to_mp3(self, audio_file: str) -> str:
        """
        Convert audio file to MP3 format to reduce upload size.
        Only converts files larger than the threshold size.
        
        Args:
            audio_file: Path to the input audio file
            
        Returns:
            Path to the converted MP3 file (temporary file) or original file
        """
        input_path = Path(audio_file)
        
        # Check if file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Get file size in MB
        file_size_bytes = input_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # If file is smaller than threshold, return original
        if file_size_mb < self.mp3_size_threshold_mb:
            logger.debug(
                f"File {audio_file} is {file_size_mb:.1f}MB, below {self.mp3_size_threshold_mb}MB threshold. "
                "Skipping MP3 conversion."
            )
            return audio_file
        
        # If already MP3, return original file
        if input_path.suffix.lower() == '.mp3':
            logger.debug(f"File {audio_file} is already MP3 format ({file_size_mb:.1f}MB)")
            return audio_file
        
        # Create temporary MP3 file
        temp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        temp_mp3_path = temp_mp3.name
        temp_mp3.close()
        
        try:
            import subprocess
            
            # Convert to MP3 using ffmpeg
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-codec:a', 'libmp3lame',
                '-b:a', self.mp3_bitrate,
                '-ac', '1',  # Convert to mono
                '-y',  # Overwrite output file
                temp_mp3_path
            ]
            
            logger.info(
                f"Converting {audio_file} ({file_size_mb:.1f}MB) to MP3 with bitrate {self.mp3_bitrate}"
            )
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check if conversion was successful
            if not Path(temp_mp3_path).exists() or Path(temp_mp3_path).stat().st_size == 0:
                raise RuntimeError("MP3 conversion failed - output file is empty or missing")
            
            original_size = input_path.stat().st_size
            converted_size = Path(temp_mp3_path).stat().st_size
            compression_ratio = (1 - converted_size / original_size) * 100
            converted_size_mb = converted_size / (1024 * 1024)
            
            logger.info(
                f"Audio converted to MP3: {file_size_mb:.1f}MB → {converted_size_mb:.1f}MB "
                f"({compression_ratio:.1f}% reduction)"
            )
            
            return temp_mp3_path
            
        except subprocess.CalledProcessError as e:
            # Clean up temp file on error
            if Path(temp_mp3_path).exists():
                Path(temp_mp3_path).unlink()
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")
        except FileNotFoundError:
            # Clean up temp file on error
            if Path(temp_mp3_path).exists():
                Path(temp_mp3_path).unlink()
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg to enable audio conversion: "
                "https://ffmpeg.org/download.html"
            )
    
    def diarize_and_transcribe(
        self,
        audio_file: str,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]]]:
        """
        Perform speaker diarization and transcription on the audio file using AssemblyAI.
        
        Args:
            audio_file: Path to the audio file
            cache_key: Optional cache key for caching results
            use_cache: Whether to use cached results if available
            
        Returns:
            Tuple containing:
            - Dictionary mapping time ranges to speaker IDs
            - List of transcription segments with timing information
        """
        # If no cache_key is provided, generate one
        # Include speakers_expected in cache key to differentiate results
        if cache_key is None:
            speakers_suffix = f"_spk{self.speakers_expected}" if self.speakers_expected else ""
            cache_key = self._generate_cache_key(audio_file, f"_{self.speech_model}{speakers_suffix}")
            
        step_name = "assemblyai_diarization_transcription"
        
        # Check if results are cached
        if use_cache and self.cache_manager and self.cache_manager.cache_exists(step_name, cache_key):
            logger.info(f"Loading AssemblyAI diarization and transcription from cache (key includes lang={self.source_language}, speakers={self.speakers_expected})...")
            cached_results = self.cache_manager.load_from_cache(step_name, cache_key)
            
            # Store for debug
            self.debug_data["diarization"] = cached_results["diarization"]
            self.debug_data["transcription"] = cached_results["transcription"]
            
            return cached_results["diarization"], cached_results["transcription"]
        
        logger.info(f"Running AssemblyAI transcription and diarization on {audio_file}...")
        logger.debug(f"Using speech model: {self.speech_model}")
        logger.debug(f"Source language configured: {self.source_language}")
        
        # Convert to MP3 if enabled to reduce upload size
        processed_audio_file = audio_file
        temp_mp3_file = None
        
        # Only try MP3 conversion for local files (not URLs)
        is_local_file = not audio_file.startswith(('http://', 'https://'))
        
        if self.convert_to_mp3 and is_local_file:
            try:
                processed_audio_file = self._convert_to_mp3(audio_file)
                if processed_audio_file != audio_file:
                    temp_mp3_file = processed_audio_file
            except Exception as e:
                logger.warning(f"MP3 conversion failed, using original file: {e}")
                processed_audio_file = audio_file
        elif not is_local_file:
            logger.debug(f"Skipping MP3 conversion for URL: {audio_file}")
        
        try:
            # Configure transcription settings
            # Use None for language_code when auto-detect is enabled
            language_code = None if self.source_language == 'auto' else self.source_language
            logger.info(f"AssemblyAI language_code: {language_code} (source_language={self.source_language})")
            
            # Warn if auto-detecting language (might fail for some languages)
            if language_code is None:
                logger.debug("Language auto-detection enabled. If transcription quality is poor, try specifying the source language explicitly.")
            
            config_params = {
                'speech_model': getattr(self.aai.SpeechModel, self.speech_model),
                'speaker_labels': True,  # Enable speaker diarization
                'language_code': language_code,
                'word_boost': None,  # Can be configured if needed
                'boost_param': "default"  # Can be configured if needed
            }
            
            # Add speakers_expected if specified
            if self.speakers_expected is not None:
                config_params['speakers_expected'] = self.speakers_expected
                logger.info(f"Using explicit speaker count for diarization: {self.speakers_expected}")
            
            config = self.aai.TranscriptionConfig(**config_params)
            
            # Create transcriber and submit job
            transcriber = self.aai.Transcriber(config=config)
            transcript = transcriber.transcribe(processed_audio_file)
            
            # Check for transcription errors
            if transcript.status == "error":
                raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
            
            # Process transcript and split overly long segments (>420 chars)
            transcription = self._process_assemblyai_transcript(transcript)

            # Optional Gemini-assisted speaker labeling: AssemblyAI diarization
            # can collapse different characters on hard material. Gemini hears
            # the audio and re-labels the transcript; word timings stay anchored
            # to the AssemblyAI output.
            if self.enable_gemini_speaker_labeling and transcription:
                transcription = self._apply_gemini_speaker_labeling(
                    processed_audio_file, transcription
                )

            # Build speakers_rolls mapping from the (potentially) split transcription
            speakers_rolls = {
                (seg["start"], seg["end"]): seg["speaker"] for seg in transcription
            }
            
            # Store for debug
            self.debug_data["diarization"] = speakers_rolls
            self.debug_data["transcription"] = transcription
            
            # Cache the results
            if use_cache and self.cache_manager:
                results_to_cache = {
                    "diarization": speakers_rolls,
                    "transcription": transcription
                }
                self.cache_manager.save_to_cache(step_name, cache_key, results_to_cache)

            if self.cost_tracker:
                audio_seconds = getattr(transcript, "audio_duration", None)
                if not audio_seconds and transcription:
                    audio_seconds = max((seg.get("end", 0.0) for seg in transcription), default=0.0)
                if audio_seconds:
                    speech_model = getattr(self, "speech_model", None) or "best"
                    self.cost_tracker.add_transcription_usage(
                        "assemblyai",
                        float(audio_seconds),
                        model=str(speech_model),
                        category="primary_transcription",
                        speaker_diarization=True,
                    )
            
            logger.info(f"AssemblyAI transcription completed successfully")
            logger.debug(f"Identified {len(set(speakers_rolls.values()))} speakers")
            logger.debug(f"Generated {len(transcription)} transcription segments")
            
            return speakers_rolls, transcription
            
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {e}")
            raise RuntimeError(f"AssemblyAI transcription failed: {e}")
        finally:
            # Clean up temporary MP3 file if created
            if temp_mp3_file and Path(temp_mp3_file).exists():
                try:
                    Path(temp_mp3_file).unlink()
                    logger.debug(f"Cleaned up temporary MP3 file: {temp_mp3_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary MP3 file {temp_mp3_file}: {e}")
        
    def _process_diarization_result(self, transcript) -> Dict[Tuple[float, float], str]:
        """
        Process AssemblyAI transcript to extract speaker diarization information.
        
        Args:
            transcript: AssemblyAI transcript object with speaker information
            
        Returns:
            Dictionary mapping time ranges to speaker IDs
        """
        speakers_rolls = {}
        
        # AssemblyAI provides utterances with speaker labels
        if hasattr(transcript, 'utterances') and transcript.utterances:
            for utterance in transcript.utterances:
                start_time = utterance.start / 1000.0  # Convert ms to seconds
                end_time = utterance.end / 1000.0      # Convert ms to seconds
                speaker_id = f"SPEAKER_{utterance.speaker}"
                
                speakers_rolls[(start_time, end_time)] = speaker_id
        
        # If no utterances with speakers, fall back to words with speaker labels
        elif hasattr(transcript, 'words') and transcript.words:
            current_speaker = None
            current_start = None
            current_end = None
            
            for word in transcript.words:
                word_start = word.start / 1000.0  # Convert ms to seconds
                word_end = word.end / 1000.0      # Convert ms to seconds
                word_speaker = f"SPEAKER_{word.speaker}" if word.speaker else "SPEAKER_00"
                
                if current_speaker != word_speaker:
                    # Save previous segment if exists
                    if current_speaker and current_start is not None and current_end is not None:
                        speakers_rolls[(current_start, current_end)] = current_speaker
                    
                    # Start new segment
                    current_speaker = word_speaker
                    current_start = word_start
                    current_end = word_end
                else:
                    # Extend current segment
                    current_end = word_end
            
            # Save last segment
            if current_speaker and current_start is not None and current_end is not None:
                speakers_rolls[(current_start, current_end)] = current_speaker
        
        logger.debug(f"Extracted {len(speakers_rolls)} speaker segments")
        return speakers_rolls

    def _apply_gemini_speaker_labeling(
        self, audio_file: str, transcription: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-label speakers with Gemini while keeping ASR word timings."""
        from src.transcription.gemini_speaker_labeler import (
            align_labeled_segments_to_words,
            label_speakers_with_gemini,
        )

        lines = [
            {
                "start": round(float(segment["start"]), 3),
                "end": round(float(segment["end"]), 3),
                "text": segment["text"],
            }
            for segment in transcription
        ]
        if str(audio_file).startswith(("http://", "https://")):
            logger.warning("Gemini speaker labeling skipped: remote audio URL is not supported")
            return transcription
        words = [word for segment in transcription for word in (segment.get("words") or [])]
        if not words:
            logger.warning("Gemini speaker labeling skipped: ASR produced no word timestamps")
            return transcription

        try:
            labeled = label_speakers_with_gemini(
                audio_file,
                lines,
                model=self.gemini_speaker_labeling_model,
                max_speakers=self.gemini_speaker_labeling_max_speakers,
            )
            realigned = align_labeled_segments_to_words(labeled, words)
        except Exception as exc:
            logger.warning("Gemini speaker labeling failed, keeping AssemblyAI labels: %s", exc)
            return transcription

        if len(realigned) < 2:
            logger.warning(
                "Gemini speaker labeling produced %d segment(s); keeping AssemblyAI labels",
                len(realigned),
            )
            return transcription

        logger.info(
            "Gemini speaker labeling: %d ASR segment(s) -> %d labeled segment(s), %d speaker(s)",
            len(transcription),
            len(realigned),
            len({segment["speaker"] for segment in realigned}),
        )
        return realigned

    def _split_words_by_speaker(
        self, words, default_speaker: str
    ) -> List[List[Dict[str, Any]]]:
        """Split an utterance's words into consecutive same-speaker runs.

        AssemblyAI exposes a speaker label per word. A single utterance can
        still span a speaker change when diarization is unstable (e.g. on
        music-heavy audio), so we re-segment on word-level speaker boundaries.
        Returns a list of runs, each a list of word dicts including a
        ``speaker`` field. Empty/untimed words are skipped.
        """
        runs: List[List[Dict[str, Any]]] = []
        current_run: List[Dict[str, Any]] = []
        current_speaker: Optional[str] = None

        for word in words or []:
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            text = getattr(word, "text", None)
            if start is None or end is None or not text:
                continue

            raw_speaker = getattr(word, "speaker", None)
            speaker = f"SPEAKER_{raw_speaker}" if raw_speaker else default_speaker

            if current_speaker is None or speaker == current_speaker:
                current_speaker = speaker
            else:
                if current_run:
                    runs.append(current_run)
                current_run = []
                current_speaker = speaker

            current_run.append(
                {
                    "word": text,
                    "start": start / 1000.0,
                    "end": end / 1000.0,
                    "confidence": getattr(word, "confidence", None),
                    "speaker": speaker,
                }
            )

        if current_run:
            runs.append(current_run)
        return runs

    def _process_assemblyai_transcript(self, transcript) -> List[Dict[str, Any]]:
        """
        Process AssemblyAI transcript into a list of segments with timing information.
        
        Args:
            transcript: AssemblyAI transcript object
            
        Returns:
            List of transcript segments with timing information
        """
        records = []
        
        # Use utterances if available (preferred as they contain speaker info)
        if hasattr(transcript, 'utterances') and transcript.utterances:
            for utterance in transcript.utterances:
                default_speaker = f"SPEAKER_{utterance.speaker}"
                utt_start = utterance.start / 1000.0  # Convert ms to seconds
                utt_end = utterance.end / 1000.0      # Convert ms to seconds
                confidence = getattr(utterance, 'confidence', None)
                words = getattr(utterance, 'words', None) or []

                word_runs = self._split_words_by_speaker(words, default_speaker)
                if word_runs:
                    for run in word_runs:
                        text = " ".join(w["word"] for w in run).strip()
                        start = run[0]["start"]
                        end = run[-1]["end"]
                        if not text or start >= end:
                            continue
                        segment = {
                            "text": text,
                            "start": start,
                            "end": end,
                            "speaker": run[0].get("speaker") or default_speaker,
                            "words": run,
                        }
                        if confidence is not None:
                            segment["confidence"] = confidence
                        records.append(segment)
                    continue

                # No usable word timings - keep the utterance as a single segment.
                text = utterance.text.strip() if getattr(utterance, "text", None) else ""
                if text and utt_start < utt_end:  # Ensure valid segment
                    segment = {
                        "text": text,
                        "start": utt_start,
                        "end": utt_end,
                        "speaker": default_speaker
                    }
                    if confidence is not None:
                        segment["confidence"] = confidence
                    records.append(segment)
        
        # Fall back to sentences if utterances are not available
        elif hasattr(transcript, 'sentences') and transcript.sentences:
            for sentence in transcript.sentences:
                text = sentence.text.strip()
                start = sentence.start / 1000.0  # Convert ms to seconds
                end = sentence.end / 1000.0      # Convert ms to seconds
                confidence = getattr(sentence, 'confidence', None)
                
                if text and start < end:  # Ensure valid segment
                    segment = {
                        "text": text,
                        "start": start,
                        "end": end,
                        "speaker": "SPEAKER_00"  # Default speaker if no diarization info
                    }
                    
                    if confidence is not None:
                        segment["confidence"] = confidence
                    
                    records.append(segment)
        
        # Ultimate fall back to the full transcript
        else:
            if hasattr(transcript, 'text') and transcript.text:
                segment = {
                    "text": transcript.text.strip(),
                    "start": 0.0,
                    "end": getattr(transcript, 'audio_duration', 0.0) / 1000.0,
                    "speaker": "SPEAKER_00"
                }
                records.append(segment)
        
        # Sort records by start time to ensure chronological order
        records.sort(key=lambda x: x["start"])

        logger.debug(f"Generated {len(records)} transcript segments")

        return records 
