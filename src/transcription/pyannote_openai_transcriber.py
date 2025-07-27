"""
Implementation of transcription and diarization using PyAnnote and OpenAI.
"""
import os
import time
import pickle
import shutil
import hashlib
import subprocess
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import whisper
from types import SimpleNamespace
import torch
from pydub import AudioSegment
from pydub.silence import split_on_silence
from speechbrain.pretrained import EncoderClassifier
from scipy.spatial.distance import cosine
from speechbrain.dataio.dataio import read_audio
from src.utils.sent_split import greedy_sent_split
from transcription.transcription_interface import BaseTranscriber
from src.utils.audio_embedder import AudioEmbedder
from src.dubbing.core.log_config import get_logger

logger = get_logger(__name__)

# Constants for audio chunking
TARGET_CHUNK_LENGTH_MINUTES = 15
MIN_CHUNK_LENGTH_MINUTES = 5
ADJACENT_SPEAKER_SEGMENTS_MAX_GAP_SECONDS = 0.3

class PyAnnoteOpenAITranscriber(BaseTranscriber):
    """Transcription and diarization service using PyAnnote and OpenAI/Whisper."""
    
    def __init__(
        self,
        source_language: str,
        device: Optional[str] = None,
        whisper_model: str = "large-v3",
        transcription_system: str = "openai",
        **kwargs
    ):
        """
        Initialize the PyAnnote + OpenAI/Whisper transcription service.
        
        Args:
            source_language: Source language code (e.g., 'en')
            device: Compute device ('cuda' or 'cpu')
            whisper_model: Whisper model size to use for transcription
            transcription_system: System to use for transcription ('whisper' or 'openai')
            **kwargs: Additional parameters
        """
        super().__init__(source_language, device, **kwargs)
        self.whisper_model = whisper_model
        self.transcription_system = transcription_system
        
        # Initialize OpenAI client if using OpenAI transcription
        if self.transcription_system == "openai":
            from openai import OpenAI
            self.openai_client = OpenAI()
        else:
            self.openai_client = None
            
        # Initialize speaker embedding model
        self.audio_embedder = AudioEmbedder(device=self.torch_device)
        
        # Dictionary to store speaker embeddings across chunks
        self.speaker_embeddings = {}
        
        # Dictionary to map original speaker IDs to new IDs
        self.speaker_mapping = {}
    
    @property
    def name(self) -> str:
        """Return the name of the transcription service implementation."""
        return "PyAnnoteOpenAI"
    
    def diarize_and_transcribe(
        self,
        audio_file: str,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]]]:
        """
        Perform speaker diarization and transcription on the audio file.
        
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
            cache_key = self._generate_cache_key(audio_file, f"_{self.whisper_model}")
            
        step_name = "chunked_processing"
        
        # Check if results are cached
        if use_cache and self._cache_exists(step_name, cache_key):
            logger.debug("Loading chunked processing results from cache...")
            results = self._load_from_cache(step_name, cache_key)
            return results[0], results[1]
        
        # Split audio into lesser chunks at silent points
        logger.info(f"Splitting audio into ~{TARGET_CHUNK_LENGTH_MINUTES}-minute chunks at silent points...")
        chunks = self._split_audio_into_chunks(audio_file)
        
        # Initialize variables to collect results
        all_speakers_rolls = {}
        all_transcriptions = []
        chunk_results = []
        
        # Process each chunk
        for i, (chunk_path, start_offset) in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Generate cache key for this chunk
            chunk_cache_key = f"{cache_key}_chunk_{i}"
            
            # Process the chunk
            speakers_rolls, transcription = self._process_audio_chunk(chunk_path, chunk_cache_key, use_cache)
            
            # Adjust timestamps by adding start_offset
            adjusted_speakers_rolls = {}
            for (start, end), speaker in speakers_rolls.items():
                adjusted_speakers_rolls[(start + start_offset, end + start_offset)] = speaker
            
            for segment in transcription:
                segment['start'] += start_offset
                segment['end'] += start_offset
                
                # Also adjust word-level timestamps if they exist
                if 'words' in segment and segment['words']:
                    for word in segment['words']:
                        if 'start' in word:
                            word['start'] += start_offset
                        if 'end' in word:
                            word['end'] += start_offset

            # Store results for this chunk
            chunk_results.append((adjusted_speakers_rolls, transcription, chunk_path, start_offset))
            
            # Collect all results
            all_speakers_rolls.update(adjusted_speakers_rolls)
            all_transcriptions.extend(transcription)

                # Match speakers across chunks and create consistent IDs
        all_speakers_rolls, all_transcriptions = self._match_speakers_across_chunks(chunk_results)
        
        # Rename speakers based on speaking time
        all_speakers_rolls, all_transcriptions = self._rename_speakers_by_speech_duration(all_speakers_rolls, all_transcriptions)
        
        # Sort transcription by start time
        all_transcriptions.sort(key=lambda x: x['start'])

        
        # Sort transcription by start time
        all_transcriptions.sort(key=lambda x: x['start'])

        # Print final speakers and transcription summary
        unique_speakers = set(all_speakers_rolls.values())
        logger.info(f"Number of speakers identified: {len(unique_speakers)}")
        
        for segment in all_transcriptions:
            text = segment['text']
            truncated_text = text[:30] + ('...' + text[-30:] if len(text) > 60 else text[30:])
            timestamp_start = self._seconds_to_hhmmss_ms(segment['start'])
            timestamp_end = self._seconds_to_hhmmss_ms(segment['end'])
            logger.debug(f"[{timestamp_start}-{timestamp_end}] {segment['speaker']}: '{truncated_text}'")
    
        # Save results to cache
        if use_cache:
            self._save_to_cache(step_name, cache_key, (all_speakers_rolls, all_transcriptions))
        
        return all_speakers_rolls, all_transcriptions
    
    def _process_audio_chunk(
        self,
        chunk_path: str,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]]]:
        """
        Process a single audio chunk using segment-based transcription.
        
        Args:
            chunk_path: Path to the audio chunk
            cache_key: Optional cache key for caching results
            use_cache: Whether to use cached results if available
            
        Returns:
            Tuple containing:
            - Dictionary mapping time ranges to speaker IDs
            - List of transcription segments with timing information
        """
        # First, check if we have cached results
        step_name = "segment_transcription"
        if use_cache and cache_key and self._cache_exists(step_name, cache_key):
            logger.debug("Loading segment-based transcription from cache...")
            return self._load_from_cache(step_name, cache_key)
            
        # Diarize the chunk to get speaker segments
        speakers_rolls = self._perform_diarization(chunk_path, cache_key, use_cache)
        
        # Load the full audio file once
        full_audio = AudioSegment.from_file(chunk_path)
        
        # Initialize list to store all transcription segments
        all_transcriptions = []
        
        # Create temp directory for segment audio files
        temp_dir = Path("artifacts/audio/segment_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each speaker segment
        logger.debug(f"Processing {len(speakers_rolls)} speaker segments...")
        for i, ((start_time, end_time), speaker) in enumerate(speakers_rolls.items()):
            # Skip very short segments (less than 0.5 seconds)
            if end_time - start_time < 0.5:
                logger.debug(f"Skipping segment {i+1} (too short: {end_time - start_time:.2f}s)")
                continue
                
            logger.debug(f"Processing segment {i+1}/{len(speakers_rolls)}: Speaker {speaker} ({start_time:.2f}s - {end_time:.2f}s)")
            
            # Extract audio for this segment
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            segment_audio = full_audio[start_ms:end_ms]
            
            # Export segment to temporary file
            segment_path = temp_dir / f"segment_{i}_{start_time:.2f}_{end_time:.2f}.wav"
            segment_audio.export(segment_path, format="wav")
            
            try:
                # Transcribe just this segment
                segment_transcription = self._transcribe_audio(str(segment_path), use_cache=False)
                
                # Adjust timestamps to match original timeline
                for record in segment_transcription:
                    record['start'] += start_time
                    record['end'] += start_time
                    record['speaker'] = speaker
                    
                    # Also adjust word-level timestamps if they exist
                    if 'words' in record and record['words']:
                        for word in record['words']:
                            if 'start' in word:
                                word['start'] += start_time
                            if 'end' in word:
                                word['end'] += start_time
                
                # Add to collection
                all_transcriptions.extend(segment_transcription)
                
            except Exception as e:
                logger.error(f"Error transcribing segment {i+1}: {e}")
            
            # Clean up temporary file
            segment_path.unlink(missing_ok=True)
        
        # Sort transcriptions by start time
        all_transcriptions.sort(key=lambda x: x['start'])
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            logger.error(f"Error removing temporary directory: {e}")
        
        # Cache the results
        if use_cache and cache_key:
            self._save_to_cache(step_name, cache_key, (speakers_rolls, all_transcriptions))
        
        return speakers_rolls, all_transcriptions
    
    def _split_audio_into_chunks(self, audio_file: str) -> List[Tuple[str, float]]:
        """
        Split audio file into chunks at silent points using a block-based approach.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            List of tuples containing (chunk_path, start_offset)
        """
        # Create output directory
        os.makedirs("artifacts/audio/chunks", exist_ok=True)
        
        # Load audio file with pydub
        logger.debug(f"Loading audio file: {audio_file}")
        audio = AudioSegment.from_file(audio_file)
        
        # Target and minimum chunk lengths in milliseconds
        target_length_ms = TARGET_CHUNK_LENGTH_MINUTES * 60 * 1000
        min_length_ms = MIN_CHUNK_LENGTH_MINUTES * 60 * 1000
        
        # If audio is shorter than target length, return it as is
        if len(audio) <= target_length_ms:
            chunk_path = "artifacts/audio/chunks/chunk_0.wav"
            audio.export(chunk_path, format="wav")
            return [(chunk_path, 0)]
            
        # Minimum silence length (in ms) and silence threshold (in dB)
        min_silence_len = 750  # 750 mseconds
        silence_thresh = -35  # dB
        
        # Split the entire audio into blocks at silent points
        logger.debug("Detecting silence points and splitting into blocks...")
        blocks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=True, 
        )
        
        logger.debug(f"Split audio into {len(blocks)} blocks")
        
        # If split_on_silence didn't find enough silence and returned just one block
        if len(blocks) <= 1:
            logger.info("Few silence points detected. Returning whole audio without chunking.")
            chunk_path = "artifacts/audio/chunks/chunk_0.wav"
            audio.export(chunk_path, format="wav")
            return [(chunk_path, 0)]
        
        # Now combine blocks into chunks that don't exceed target_length_ms
        chunks = []  # Will store tuples of (path, offset_seconds, duration_ms)
        current_chunk = AudioSegment.empty()
        current_chunk_ms = 0
        next_chunk_start_offset_ms = 0 # Track the start time for the *next* chunk
        chunk_index = 0
        
        for i, block in enumerate(blocks):
            block_length_ms = len(block)
            
            # If adding this block would exceed the target length and we already have content
            if current_chunk_ms > 0 and current_chunk_ms + block_length_ms > target_length_ms:
                # Export current chunk
                chunk_path = f"artifacts/audio/chunks/chunk_{chunk_index}.wav"
                current_chunk.export(chunk_path, format="wav")
                # Store path, offset (using the offset calculated *before* this chunk) and duration
                chunks.append((chunk_path, next_chunk_start_offset_ms / 1000.0, current_chunk_ms))
                
                # Update the offset for the next chunk
                next_chunk_start_offset_ms += current_chunk_ms
                
                # Start a new chunk with this block
                current_chunk = block
                current_chunk_ms = block_length_ms
                # No need to update offset here, it's already set for the next export
                chunk_index += 1
            else:
                # Add block to current chunk
                if current_chunk_ms == 0:
                    current_chunk = block
                else:
                    current_chunk += block
                current_chunk_ms += block_length_ms
        
        # Add the last chunk if there's anything left
        if current_chunk_ms > 0:
            last_chunk_path = f"artifacts/audio/chunks/chunk_{chunk_index}.wav"
            current_chunk.export(last_chunk_path, format="wav")
            chunks.append((last_chunk_path, next_chunk_start_offset_ms / 1000.0, current_chunk_ms))
        
        # Check if the last chunk is too small
        if len(chunks) > 1:
            last_chunk_path, last_offset, last_chunk_ms = chunks[-1]
            
            if last_chunk_ms < min_length_ms:
                logger.debug(f"Last chunk ({last_chunk_ms/1000.0:.1f}s) is smaller than minimum length ({MIN_CHUNK_LENGTH_MINUTES*60}s), merging with previous chunk")
                # Remove the last chunk from the list
                chunks.pop()
                
                # Get the previous chunk info
                prev_chunk_path, prev_offset, prev_chunk_ms = chunks[-1]
                
                # Instead of reading both files, just read the last chunk and append to previous
                last_chunk = AudioSegment.from_file(last_chunk_path)
                # Read previous chunk - we have to read this one since we need to modify it
                prev_chunk = AudioSegment.from_file(prev_chunk_path)
                
                # Combine with the last chunk
                combined_chunk = prev_chunk + last_chunk
                
                # Export the combined chunk
                combined_chunk.export(prev_chunk_path, format="wav")
                
                # Update the duration in our tracking data
                chunks[-1] = (prev_chunk_path, prev_offset, prev_chunk_ms + last_chunk_ms)
                
                # Delete the last chunk file
                if os.path.exists(last_chunk_path):
                    os.remove(last_chunk_path)
        
        # If we only have one chunk after processing, see if we can just use the original audio
        if len(chunks) == 1:
            logger.info("Only produced one chunk after processing, returning original audio")
            chunk_path = "artifacts/audio/chunks/chunk_0.wav" 
            audio.export(chunk_path, format="wav")
            return [(chunk_path, 0)]
        
        logger.debug(f"Created {len(chunks)} chunks")
        for i, (_, offset, duration_ms) in enumerate(chunks):
            logger.debug(f"Chunk {i}: {duration_ms/1000.0:.1f}s, offset: {offset:.1f}s")
        
        # Return only the path and offset (without duration) to maintain the expected return type
        return [(path, offset) for path, offset, _ in chunks]
    
    def _extract_speaker_embeddings(
        self,
        audio_file: str,
        speakers_rolls: Dict[Tuple[float, float], str],
        start_offset: float
    ) -> Dict[str, np.ndarray]:
        """
        Extract speaker embeddings for each speaker in the audio file.
        
        Args:
            audio_file: Path to the audio file
            speakers_rolls: Dictionary mapping time ranges to speaker IDs
            
        Returns:
            Dictionary mapping speaker IDs to embeddings
        """
        
        # Get all unique speakers
        speakers = set(speakers_rolls.values())
        
        # Organize speech segments by speaker
        speaker_segments = {speaker: [] for speaker in speakers}
        for (start, end), speaker in speakers_rolls.items():
            speaker_segments[speaker].append((start, end))
        
        # Extract embedding for each speaker
        embeddings = {}
        audio = AudioSegment.from_file(audio_file)
        
        for speaker, segments in speaker_segments.items():
            # Get the longest segments for this speaker (more reliable embeddings)
            segments.sort(key=lambda x: x[1] - x[0], reverse=True)
            
            # Use the top 5 longest segments or all if fewer than 5
            top_segments = segments[:min(5, len(segments))]
            
            # Extract audio for each segment and concatenate
            speaker_audio = AudioSegment.empty()
            # Sort segments by start time to process them chronologically
            for start, end in sorted(top_segments, key=lambda x: x[0]):
                # Convert seconds to milliseconds
                start_ms = int((start - start_offset) * 1000) # start_offset is relative to chunk, not whole audio
                end_ms = int((end - start_offset) * 1000)
                
                # Extract segment and add to collection
                segment_audio = audio[start_ms:end_ms]
                speaker_audio += segment_audio
            
            # Skip if speaker audio is too short (handled by AudioEmbedder now)
            # if len(speaker_audio) < 500: # Minimum 500ms needed
            #     print(f"Warning: Skipping embedding for speaker {speaker} due to insufficient audio length.")
            #     continue

            # Extract embedding using AudioEmbedder
            embedding = self.audio_embedder.extract_embedding(speaker_audio)

            if embedding is not None:
                embeddings[speaker] = embedding
                logger.debug(f"Extracted embedding by {self.audio_embedder.embedding_type} for speaker {speaker} with shape: {embedding.shape}")
            else:
                logger.warning(f"Warning: Failed to extract embedding for speaker {speaker}.")
                # Assign a default or skip this speaker
                embeddings[speaker] = np.zeros(192) # Assuming typical embedding size, adjust if needed

        return embeddings
    
    def _perform_diarization(
        self,
        audio_file: str,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[Tuple[float, float], str]:
        """
        Perform speaker diarization on the audio file.
        
        Args:
            audio_file: Path to the audio file
            cache_key: Optional cache key for caching results
            use_cache: Whether to use cached results if available
            
        Returns:
            Dictionary mapping time ranges to speaker IDs
        """
        # Start timing
        start_time = time.perf_counter()

        # If no cache_key is provided, generate one
        if cache_key is None:
            cache_key = self._generate_cache_key(audio_file, f"_{self.whisper_model}")
            
        step_name = "diarization"
        
        # Check if results are cached
        if use_cache and self._cache_exists(step_name, cache_key):
            logger.debug("Loading speaker diarization from cache...")
            speakers_rolls = self._load_from_cache(step_name, cache_key)
            
            # Store for debug
            self.debug_data["diarization"] = speakers_rolls
            
            return speakers_rolls
        
        logger.debug("Initializing speaker diarization pipeline...")
        hf_token = os.environ.get("HF_TOKEN")
        
        # AVOID TO USE pyannote/speaker-diarization-3.0 - 3.1 due DER regression for long audio segments
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                            use_auth_token=hf_token).to(self.torch_device)
        
        logger.debug("Running speaker diarization...")
        diarization = pipeline(audio_file)
        
        speakers_rolls = {}
        # Check for overlapping speech segments
        overlapping_segments = []
        all_segments = []
        
        # First collect all segments
        for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
            if abs(speech_turn.end - speech_turn.start) > 0.75:
                all_segments.append((speech_turn.start, speech_turn.end, speaker))
        
        # Then detect intersections
        for i, (start1, end1, speaker1) in enumerate(all_segments):
            for j, (start2, end2, speaker2) in enumerate(all_segments[i+1:], i+1):
                # Check if segments overlap
                if max(start1, start2) < min(end1, end2):
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    overlap_duration = overlap_end - overlap_start
                    if overlap_duration > 1.5:  # Only report overlaps > 100ms
                        overlapping_segments.append((overlap_start, overlap_end, speaker1, speaker2))
                        logger.warning(f"WARNING: Overlapping speech detected from {overlap_start:.2f}s to {overlap_end:.2f}s "
                              f"between Speaker {speaker1} and Speaker {speaker2} ({overlap_duration:.2f}s)")
        
        # Add all valid segments to speakers_rolls
        for start, end, speaker in all_segments:
            logger.debug(f"Speaker {speaker}: from {start:.2f}s to {end:.2f}s")
            speakers_rolls[(start, end)] = speaker
        
        # Merge adjacent segments from the same speaker with short pauses
        logger.debug("Merging adjacent segments from the same speaker...")
        speakers_rolls = self._merge_adjacent_speaker_segments(speakers_rolls, ADJACENT_SPEAKER_SEGMENTS_MAX_GAP_SECONDS)
        
        # Store for debug
        self.debug_data["diarization"] = speakers_rolls
        
        # Save results to cache
        if use_cache:
            self._save_to_cache(step_name, cache_key, speakers_rolls)
        
        return speakers_rolls
    
    def _merge_adjacent_speaker_segments(self, speakers_rolls: Dict[Tuple[float, float], str], max_pause: float) -> Dict[Tuple[float, float], str]:
        """
        Merge adjacent speech segments from the same speaker when the pause between them is less than max_pause.
        Performs multiple passes until no more merges are possible.
        
        Args:
            speakers_rolls: Dictionary mapping time ranges to speaker IDs
            max_pause: Maximum pause in seconds to consider for merging
            
        Returns:
            Dictionary with merged segments
        """
        if not speakers_rolls:
            return {}
            
        # Convert speakers_rolls to a list of (start, end, speaker) tuples
        segments = [(start, end, speaker) for (start, end), speaker in speakers_rolls.items()]
        
        # Sort segments by start time
        segments.sort(key=lambda x: x[0])
        
        total_merged = 0
        pass_count = 0
        
        # Keep merging until no more merges happen
        while True:
            pass_count += 1
            merged_in_this_pass = 0
            i = 0
            
            while i < len(segments) - 1:
                start1, end1, speaker1 = segments[i]
                start2, end2, speaker2 = segments[i + 1]
                gap = start2 - end1
                
                if speaker1 == speaker2 and gap <= max_pause:
                    # Log merges with different characteristics
                    if gap < 0:
                        logger.debug(f"Merging overlapping segments: {end1:.2f}-{start2:.2f} (overlap: {-gap:.2f}s)")
                    elif gap == 0:
                        logger.debug(f"Merging exactly adjacent segments at {end1:.2f}s")
                    
                    # Merge segments
                    segments[i] = (start1, end2, speaker1)
                    # Remove the merged segment
                    segments.pop(i + 1)
                    merged_in_this_pass += 1
                else:
                    i += 1
            
            total_merged += merged_in_this_pass
            
            # If no merges in this pass, we're done
            if merged_in_this_pass == 0:
                break
                
            # Safety check - don't go into infinite loop
            if pass_count >= 10:
                logger.info("Reached maximum number of merge passes (10)")
                break
        
        if total_merged > 0:
            logger.debug(f"Merged {total_merged} adjacent segments in {pass_count} passes (pauses < {max_pause:.1f}s)")
            
        # Convert back to dictionary
        merged_speakers_rolls = {(start, end): speaker for start, end, speaker in segments}
        
        return merged_speakers_rolls
    
    def _transcribe_audio(
        self,
        audio_file: str,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Transcribe audio using either Whisper or OpenAI.
        
        Args:
            audio_file: Path to the audio file
            cache_key: Optional cache key for caching results
            use_cache: Whether to use cached results if available
            
        Returns:
            List of transcription segments with timestamps
        """
        # If no cache_key is provided, generate one
        if cache_key is None:
            cache_key = self._generate_cache_key(audio_file, f"_{self.whisper_model}")
            
        cache_key = f"{cache_key}_{self.transcription_system}"
        step_name = "transcription"
        
        # Check if results are cached
        if use_cache and self._cache_exists(step_name, cache_key):
            logger.debug("Loading transcription from cache...")
            return self._load_from_cache(step_name, cache_key)
        
        logger.info(f"Transcribing audio using {self.transcription_system}...")
        
        if self.transcription_system == "whisper":
            # Use local Whisper model
            logger.debug(f"Loading Whisper model: {self.whisper_model}")
            model = whisper.load_model(self.whisper_model, device=self.torch_device)
            
            transcript = model.transcribe(
                audio=audio_file,
                word_timestamps=True,
            )
            
            # Process and organize transcription
            records = self._process_transcript(transcript)
            
        elif self.transcription_system == "openai":
            # Check if OpenAI client is initialized
            if not self.openai_client:
                raise ValueError("OpenAI client not initialized. Please provide a valid OpenAI API key.")
            
            # Convert audio to 16kHz 64kbit MP3 before sending to API
            logger.debug("Converting audio to 16kHz 64kbit MP3 format...")
            temp_dir = Path("artifacts/audio/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a temporary filename for the converted audio
            temp_mp3_path = temp_dir / f"temp_whisper_{os.path.basename(audio_file)}.mp3"
            
            try:
                # Load the audio file with pydub
                audio_segment = AudioSegment.from_file(audio_file)
                
                # Convert to mono 16kHz 
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                
                # Export as MP3 with 64kbit bitrate
                audio_segment.export(
                    temp_mp3_path,
                    format="mp3",
                    bitrate="64k",
                    parameters=["-ac", "1", "-ar", "16000"]
                )
                
                logger.debug(f"Converted audio saved to {temp_mp3_path}")
                
                # Use OpenAI's speech-to-text API with the converted file
                max_attempts = 3
                transcript = None
                
                for attempt in range(max_attempts):
                    try:
                        with open(temp_mp3_path, "rb") as audio:
                            transcript = self.openai_client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio,
                                language=self.source_language,
                                response_format="verbose_json",
                                timestamp_granularities=["word"]
                            )
                        break
                    except Exception as e:
                        if attempt < max_attempts - 1:
                            logger.warning(f"OpenAI transcription error: {e}, retrying ({attempt+1}/{max_attempts})...")
                            time.sleep(1)  # Add a small delay before retrying
                        else:
                            raise RuntimeError(f"Failed to transcribe with OpenAI after {max_attempts} attempts: {e}")
                
                if not transcript:
                    raise RuntimeError("Failed to get a valid transcription from OpenAI")
            
            finally:
                # Clean up the temporary file
                if temp_mp3_path.exists():
                    temp_mp3_path.unlink()
                    logger.debug(f"Removed temporary file: {temp_mp3_path}")
            
            # Convert OpenAI's format to our internal format
            records = self._process_transcript(transcript)
        
        else:
            raise ValueError(f"Unsupported transcription system: {self.transcription_system}")
        
        # Save results to cache
        if use_cache:
            self._save_to_cache(step_name, cache_key, records)
        
        # Store for debug
        self.debug_data["transcription"] = records
        
        return records
    
    def _process_transcript(self, transcript: Dict) -> List[Dict]:
        """
        Process the raw transcript into a structured format.
        
        Args:
            transcript: Raw transcript from Whisper
            
        Returns:
            List of structured transcript segments
        """
        # Extract word-level timestamps
        time_stamped = []
        full_text = []

        if hasattr(transcript, 'words'):
            # if openai api case
            for word in transcript.words:
                time_stamped.append([word.word, word.start, word.end])

            full_text = transcript.text
        else:
            # local-whisper case
            for segment in transcript['segments']:
                for word in segment['words']:
                    time_stamped.append([word['word'], word['start'], word['end']])
                    full_text.append(word['word'])

            full_text = "".join(full_text)
        
        # Tokenize into sentences
        sentences = greedy_sent_split(full_text, 360)
        
        # Map sentences to timestamps using a linear pass through the word list
        records: List[Dict[str, Any]] = []
        word_idx = 0  # Index into time_stamped
        total_tokens = len(time_stamped)

        for sentence in sentences:
            # Count *spoken* words in the sentence
            sentence_word_count = len(sentence.split())

            if word_idx >= total_tokens:
                break  # No more timestamps available

            start_time = time_stamped[word_idx][1]

            # Consume time_stamped entries until we have covered the sentence word count
            words_consumed = 0
            last_word_idx = word_idx

            while last_word_idx < total_tokens and words_consumed < sentence_word_count:
                token_text = time_stamped[last_word_idx][0]
                # A single timestamp entry might contain multiple space‑separated words (rare but possible)
                words_in_token = len(token_text.split())
                words_consumed += words_in_token
                last_word_idx += 1

            # Adjust because the loop exits after incrementing last_word_idx once more
            last_word_idx -= 1

            end_time = time_stamped[last_word_idx][2]

            records.append({
                "text": sentence,
                "start": start_time,
                "end": end_time,
            })

            # Prepare for next sentence
            word_idx = last_word_idx + 1

        # In rare cases where the token counts mismatch, ensure records are sorted
        records.sort(key=lambda x: x["start"])
        
        return records
    
    def _match_speakers_across_chunks(
        self,
        chunk_results: List[Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]], str, float]] #TODO: refactor this, use models
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]]]:
        """
        Match speakers across chunks using their embeddings with a greedy algorithm.
        
        Args:
            chunk_results: List of tuples containing (speakers_rolls, transcription, audio_path, start_offset)
                for each processed chunk
                
        Returns:
            Tuple containing:
            - Combined speakers_rolls with consistent speaker IDs
            - Combined transcription with consistent speaker IDs
        """
        if not chunk_results:
            return {}, []
            
        logger.info("Matching speakers across chunks...")
        
        # Extract embeddings for each speaker in each chunk
        chunk_embeddings = []
        for i, (speakers_rolls, _, audio_path, start_offset) in enumerate(chunk_results):
            logger.debug(f"Extracting speaker embeddings for chunk {i+1}/{len(chunk_results)}...")
            embeddings = self._extract_speaker_embeddings(audio_path, speakers_rolls, start_offset)
            chunk_embeddings.append(embeddings)
        
        # Initialize global speaker mapping and combined results
        global_speaker_map = {}  # Maps (chunk_idx, speaker_id) to global_speaker_id
        combined_speakers_rolls = {}
        combined_transcription = []
        
        # Threshold for considering speakers as the same person
        # Lower values are more strict (more likely to create new speakers)
        similarity_threshold = 0.15  # Cosine distance threshold (lower is more similar)
        
        # Process each chunk and map speakers to global IDs
        for chunk_idx, ((speakers_rolls, transcription, _, _), embeddings) in enumerate(zip(chunk_results, chunk_embeddings)):
            # For each speaker in this chunk
            for speaker in set(speakers_rolls.values()):
                # Skip speakers with no embeddings
                if speaker not in embeddings:
                    logger.warning(f"Warning: No embedding for speaker {speaker} in chunk {chunk_idx+1}")
                    # Assign a new global ID
                    global_speaker_map[(chunk_idx, speaker)] = f"S_{len(global_speaker_map)}"
                    continue
                
                # Get this speaker's embedding
                speaker_embedding = embeddings[speaker]
                
                # Find best match among previously processed speakers
                best_match = None
                best_distance = float('inf')
                
                # Check all previous chunks
                for prev_chunk_idx in range(chunk_idx):
                    prev_embeddings = chunk_embeddings[prev_chunk_idx]
                    
                    # For each speaker in the previous chunk
                    for prev_speaker, prev_embedding in prev_embeddings.items():
                        # Skip if no embedding
                        if prev_speaker not in prev_embeddings:
                            continue
                            
                        # Calculate cosine distance (lower is more similar)
                        distance = cosine(speaker_embedding, prev_embedding)
                        
                        # Update best match if this is better
                        if distance < best_distance:
                            best_distance = distance
                            best_match = (prev_chunk_idx, prev_speaker)
                
                # Decide whether to map to an existing speaker or create a new one
                if best_match and best_distance < similarity_threshold:
                    # Get the global ID of the best match
                    global_id = global_speaker_map[best_match]
                    global_speaker_map[(chunk_idx, speaker)] = global_id
                    logger.debug(f"Matched speaker {speaker} in chunk {chunk_idx+1} to global speaker {global_id} (distance: {best_distance:.3f})")
                else:
                    # Create a new global speaker ID
                    global_id = f"S_{len(global_speaker_map)}"
                    global_speaker_map[(chunk_idx, speaker)] = global_id
                    logger.debug(f"Created new global speaker {global_id} for speaker {speaker} in chunk {chunk_idx+1}" + 
                          (f" (closest match distance: {best_distance:.3f})" if best_match else ""))
        
        # Now update all speakers_rolls and transcription with the global IDs
        for chunk_idx, (speakers_rolls, transcription, _, start_offset) in enumerate(chunk_results):
            # Update speakers_rolls
            for time_range, speaker in speakers_rolls.items():
                if (chunk_idx, speaker) in global_speaker_map:
                    global_id = global_speaker_map[(chunk_idx, speaker)]
                    combined_speakers_rolls[time_range] = global_id
                else:
                    # Fallback for speakers without mapping
                    combined_speakers_rolls[time_range] = f"UNKNOWN_{speaker}"
            
            # Update transcription
            for segment in transcription:
                if 'speaker' in segment and segment['speaker'] is not None:
                    if (chunk_idx, segment['speaker']) in global_speaker_map:
                        segment['speaker'] = global_speaker_map[(chunk_idx, segment['speaker'])]
                    else:
                        # Fallback for speakers without mapping
                        segment['speaker'] = f"UNKNOWN_{segment['speaker']}"
                combined_transcription.append(segment)
        
        # Sort combined transcription by start time
        combined_transcription.sort(key=lambda x: x['start'])
        
        logger.info(f"Speaker matching complete. Identified {len(set(combined_speakers_rolls.values()))} unique speakers across all chunks.")
        
        return combined_speakers_rolls, combined_transcription
    
    def _rename_speakers_by_speech_duration(
        self,
        speakers_rolls: Dict[Tuple[float, float], str],
        transcription: List[Dict[str, Any]]
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]]]:
        """
        Rename speakers based on their total speaking time.
        
        Args:
            speakers_rolls: Dictionary mapping time ranges to speaker IDs
            transcription: List of transcription segments with speaker IDs
            
        Returns:
            Tuple containing:
            - Updated speakers_rolls with renamed speakers
            - Updated transcription with renamed speakers
        """
        # Calculate total speaking time for each speaker
        speaking_time = {}
        for (start, end), speaker in speakers_rolls.items():
            duration = end - start
            speaking_time[speaker] = speaking_time.get(speaker, 0) + duration
        
        # Sort speakers by speaking time (descending)
        sorted_speakers = sorted(speaking_time.items(), key=lambda x: x[1], reverse=True)
        
        # Create mapping from old speaker IDs to new ones (speaker0, speaker1, etc.)
        speaker_map = {
            old_id: f"SPEAKER_{i:02d}" 
            for i, (old_id, _) in enumerate(sorted_speakers)
        }
        
        # Update speakers_rolls
        renamed_speakers_rolls = {
            time_range: speaker_map[speaker]
            for time_range, speaker in speakers_rolls.items()
        }
        
        # Update transcription
        renamed_transcription = transcription.copy()
        for segment in renamed_transcription:
            if 'speaker' in segment and segment['speaker'] in speaker_map:
                segment['speaker'] = speaker_map[segment['speaker']]

        # Print report of how much time each speaker spoke
        logger.debug("\nSpeaker speaking time report:")
        for speaker, duration in sorted_speakers:
            renamed_speaker = speaker_map[speaker]
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            percentage = (duration / sum(time for _, time in speaking_time.items())) * 100
            logger.debug(f"{renamed_speaker}: {minutes:02d}:{seconds:02d} minutes ({percentage:.1f}% of total speaking time)")
        
        return renamed_speakers_rolls, renamed_transcription

    ############################################################
    #TODO: move all below to libs!! 
    ############################################################
    
    def _generate_cache_key(self, audio_file: str, suffix: str = "") -> str:
        """
        Generate a unique cache key based on MD5 hash of the audio file and processing parameters.
        
        Args:
            audio_file: Path to the audio file
            suffix: Optional suffix to add to the key
            
        Returns:
            Unique cache key string
        """
        # Generate MD5 hash of audio file
        md5_hash = hashlib.md5()
        with open(audio_file, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        # Include processing parameters in the key to ensure uniqueness
        params = f"{self.source_language}_{self.whisper_model}"
        
        # Combine file hash and parameters
        return f"{md5_hash.hexdigest()}_{params}{suffix}"
    
    def _get_cache_path(self, step_name: str) -> Path:
        """
        Get the cache directory path for a specific pipeline step.
        
        Args:
            step_name: Name of the pipeline step
            
        Returns:
            Path to the cache directory
        """
        cache_dir = Path("cache") / step_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def _cache_exists(self, step_name: str, cache_key: str) -> bool:
        """
        Check if cached results exist for a specific step and key.
        
        Args:
            step_name: Name of the pipeline step
            cache_key: Cache key
            
        Returns:
            True if cache exists, False otherwise
        """
        cache_file = self._get_cache_path(step_name) / f"{cache_key}.pkl"
        return cache_file.exists()
    
    def _save_to_cache(self, step_name: str, cache_key: str, data: Any) -> None:
        """
        Save data to cache.
        
        Args:
            step_name: Name of the pipeline step
            cache_key: Cache key
            data: Data to cache
        """
        cache_file = self._get_cache_path(step_name) / f"{cache_key}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
    
    def _load_from_cache(self, step_name: str, cache_key: str) -> Any:
        """
        Load data from cache.
        
        Args:
            step_name: Name of the pipeline step
            cache_key: Cache key
            
        Returns:
            Cached data
        """
        cache_file = self._get_cache_path(step_name) / f"{cache_key}.pkl"
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    def _seconds_to_hhmmss(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string in HH:MM:SS format
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _seconds_to_hhmmss_ms(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS.mmm format with milliseconds.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string in HH:MM:SS.mmm format
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        whole_seconds = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"

