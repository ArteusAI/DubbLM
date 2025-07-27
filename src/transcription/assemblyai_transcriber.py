"""
Implementation of transcription and diarization using AssemblyAI API.
"""
import os
import time
from typing import Dict, List, Tuple, Any, Optional

from transcription.transcription_interface import BaseTranscriber
from src.dubbing.core.log_config import get_logger

logger = get_logger(__name__)


class AssemblyAITranscriber(BaseTranscriber):
    """Transcription and diarization service using AssemblyAI API."""
    
    def __init__(
        self,
        source_language: str,
        device: Optional[str] = None,
        speech_model: str = "best",
        **kwargs
    ):
        """
        Initialize the AssemblyAI transcription service.
        
        Args:
            source_language: Source language code (e.g., 'en')
            device: Compute device (not used for API-based service)
            speech_model: AssemblyAI speech model to use ('best', 'nano')
            **kwargs: Additional parameters
        """
        super().__init__(source_language, device, **kwargs)
        self.speech_model = speech_model
        
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
        if cache_key is None:
            cache_key = self._generate_cache_key(audio_file, f"_{self.speech_model}")
            
        step_name = "assemblyai_diarization_transcription"
        
        # Check if results are cached
        if use_cache and self._cache_exists(step_name, cache_key):
            logger.debug("Loading AssemblyAI diarization and transcription from cache...")
            cached_results = self._load_from_cache(step_name, cache_key)
            
            # Store for debug
            self.debug_data["diarization"] = cached_results["diarization"]
            self.debug_data["transcription"] = cached_results["transcription"]
            
            return cached_results["diarization"], cached_results["transcription"]
        
        logger.info(f"Running AssemblyAI transcription and diarization on {audio_file}...")
        logger.debug(f"Using speech model: {self.speech_model}")
        
        try:
            # Configure transcription settings
            config = self.aai.TranscriptionConfig(
                speech_model=getattr(self.aai.SpeechModel, self.speech_model),
                speaker_labels=True,  # Enable speaker diarization
                language_code=self.source_language,
                word_boost=None,  # Can be configured if needed
                boost_param="default"  # Can be configured if needed
            )
            
            # Create transcriber and submit job
            transcriber = self.aai.Transcriber(config=config)
            transcript = transcriber.transcribe(audio_file)
            
            # Check for transcription errors
            if transcript.status == "error":
                raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
            
            # Process transcript and split overly long segments (>420 chars)
            transcription = self._process_assemblyai_transcript(transcript)

            # Build speakers_rolls mapping from the (potentially) split transcription
            speakers_rolls = {
                (seg["start"], seg["end"]): seg["speaker"] for seg in transcription
            }
            
            # Store for debug
            self.debug_data["diarization"] = speakers_rolls
            self.debug_data["transcription"] = transcription
            
            # Cache the results
            if use_cache:
                results_to_cache = {
                    "diarization": speakers_rolls,
                    "transcription": transcription
                }
                self._save_to_cache(step_name, cache_key, results_to_cache)
            
            logger.info(f"AssemblyAI transcription completed successfully")
            logger.debug(f"Identified {len(set(speakers_rolls.values()))} speakers")
            logger.debug(f"Generated {len(transcription)} transcription segments")
            
            return speakers_rolls, transcription
            
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {e}")
            raise RuntimeError(f"AssemblyAI transcription failed: {e}")
        
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
                text = utterance.text.strip()
                start = utterance.start / 1000.0  # Convert ms to seconds
                end = utterance.end / 1000.0      # Convert ms to seconds
                speaker = f"SPEAKER_{utterance.speaker}"
                confidence = getattr(utterance, 'confidence', None)
                
                if text and start < end:  # Ensure valid segment
                    segment = {
                        "text": text,
                        "start": start,
                        "end": end,
                        "speaker": speaker
                    }
                    
                    if confidence is not None:
                        segment["confidence"] = confidence
                        
                    # Add word-level details if available
                    if hasattr(utterance, 'words') and utterance.words:
                        words = []
                        for word in utterance.words:
                            word_data = {
                                "word": word.text,
                                "start": word.start / 1000.0,
                                "end": word.end / 1000.0,
                                "confidence": word.confidence
                            }
                            words.append(word_data)
                        segment["words"] = words
                    
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

        # After initial record creation, ensure each segment does not exceed MAX_CHARS
        MAX_CHARS = 420
        processed_records: List[Dict[str, Any]] = []

        for record in records:
            # If segment is short enough or word-level timestamps are unavailable – keep as is
            if len(record["text"]) <= MAX_CHARS or "words" not in record:
                processed_records.append(record)
                continue

            # Otherwise split using the helper
            processed_records.extend(self._split_long_segment(record, MAX_CHARS))

        # Ensure chronological order after splitting
        processed_records.sort(key=lambda x: x["start"])

        logger.debug(
            f"Generated {len(processed_records)} transcript segments after splitting "
            f"(initial: {len(records)})"
        )

        return processed_records 

    def _split_long_segment(self, segment: Dict[str, Any], max_chars: int) -> List[Dict[str, Any]]:
        """Split a single transcription segment into smaller chunks not exceeding
        ``max_chars`` characters using word-level timestamps.

        The algorithm tries to split on sentence boundaries (., !, ?). If including the
        remainder of the last sentence would exceed the limit only slightly (\<50 chars),
        it is **kept** inside the current block to preserve sentence integrity.

        Args:
            segment: Original transcription segment with "words" list.
            max_chars: Character limit per chunk.

        Returns:
            List of new segments derived from the original one.
        """
        words: List[Dict[str, Any]] = segment["words"]

        chunks: List[Dict[str, Any]] = []

        idx = 0
        while idx < len(words):
            char_count = 0
            last_sentence_break = -1  # index of last word ending with sentence punctuation
            start_idx = idx
            split_end_idx = None  # will be determined below

            # Grow the chunk word-by-word
            while idx < len(words):
                word_text: str = words[idx]["word"]
                # plus one for space if not first word in chunk
                projected = char_count + (1 if char_count > 0 else 0) + len(word_text)

                # Track sentence boundary
                if word_text.endswith((".", "!", "?")):
                    last_sentence_break = idx

                # If adding the word would exceed the limit
                if projected > max_chars:
                    # Decide on split position
                    if last_sentence_break >= start_idx:
                        # Compute leftover length to decide whether to keep remainder
                        leftover_words = words[last_sentence_break + 1 :]
                        leftover_text_len = len(" ".join(w["word"] for w in leftover_words))
                        if leftover_text_len < 50:
                            # Keep remainder – allow slight overflow
                            idx += 1  # include current word as well
                            char_count = projected
                            # Continue consuming until end to keep sentence integrity
                            while idx < len(words):
                                w_text = words[idx]["word"]
                                char_count += 1 + len(w_text)
                                idx += 1
                            split_end_idx = len(words) - 1
                            break  # Whole segment consumed
                        else:
                            # Split at last_sentence_break
                            split_end_idx = last_sentence_break
                            break
                    # No suitable sentence break or leftover large – split at previous word
                    split_end_idx = idx - 1 if idx > start_idx else idx
                    break

                # Otherwise, add the word and continue
                char_count = projected
                idx += 1
            else:
                # Reached end of words without exceeding limit
                split_end_idx = len(words) - 1

            # Fallback if split_end_idx was not set inside the loop
            if split_end_idx is None:
                split_end_idx = len(words) - 1 if idx >= len(words) else idx - 1

            # Build chunk with words [start_idx, split_end_idx]
            chunk_words = words[start_idx : split_end_idx + 1]
            chunk_text = " ".join(w["word"] for w in chunk_words)
            chunk_start = chunk_words[0]["start"]
            chunk_end = chunk_words[-1]["end"]

            new_segment = {
                "text": chunk_text,
                "start": chunk_start,
                "end": chunk_end,
                "speaker": segment["speaker"],
                "words": chunk_words,
            }

            # Propagate confidence if available
            if "confidence" in segment:
                new_segment["confidence"] = segment["confidence"]

            chunks.append(new_segment)

            # Prepare for next loop iteration
            idx = split_end_idx + 1

        return chunks 