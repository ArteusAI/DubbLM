"""Emotion detection using YAMNet audio event classification."""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from pydub import AudioSegment

from src.dubbing.core.log_config import get_logger

logger = get_logger(__name__)


class EmotionType(Enum):
    """Emotion types detected from audio events."""
    NEUTRAL = "Neutral"
    HAPPY_LAUGHTER = "Happy/Laughter"
    SAD_CRYING = "Sad/Crying"
    ANGRY = "Angry"
    SURPRISED = "Surprised"


class EmotionDetector:
    """
    Detects emotions in audio using YAMNet audio event classification.
    Maps audio events (laughter, crying, etc.) to emotional states.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize YAMNet emotion detector.
        
        Args:
            device: Device for computation (CPU/GPU)
        """
        self.device = device
        self.model = None
        self.class_names = None
        self._emotion_mapping = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load YAMNet model from TensorFlow Hub."""
        try:
            logger.info("Loading YAMNet model for emotion detection...")
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            
            # Load class names
            class_map_path = self.model.class_map_path().numpy().decode('utf-8')
            self.class_names = self._load_class_names(class_map_path)
            
            # Build emotion mapping from YAMNet classes
            self._build_emotion_mapping()
            
            logger.info("YAMNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YAMNet model: {e}")
            raise
    
    def _load_class_names(self, class_map_csv: str) -> List[str]:
        """Load YAMNet class names from CSV."""
        import csv
        from io import StringIO
        
        # Download class map if needed
        if class_map_csv.startswith('http'):
            import urllib.request
            with urllib.request.urlopen(class_map_csv) as response:
                class_map_content = response.read().decode('utf-8')
        else:
            with open(class_map_csv, 'r') as f:
                class_map_content = f.read()
        
        class_names = []
        reader = csv.DictReader(StringIO(class_map_content))
        for row in reader:
            class_names.append(row['display_name'])
        
        return class_names
    
    def _build_emotion_mapping(self):
        """Map YAMNet audio event classes to emotion types."""
        self._emotion_mapping = {}

        laughter_keywords = [
            'laugh', 'laughter', 'laughing', 'giggle', 'chuckle', 'chortle',
            'snicker', 'snigger', 'cackle', 'titter', 'guffaw'
        ]

        for idx, class_name in enumerate(self.class_names):
            class_lower = class_name.lower()

            # Happy/Laughter
            if any(word in class_lower for word in laughter_keywords):
                self._emotion_mapping[idx] = EmotionType.HAPPY_LAUGHTER

            # Sad/Crying
            elif any(word in class_lower for word in ['cry', 'sob', 'whimper', 'wail']):
                self._emotion_mapping[idx] = EmotionType.SAD_CRYING

            # Angry
            elif any(word in class_lower for word in ['scream', 'shout', 'yell', 'roar', 'growl']):
                self._emotion_mapping[idx] = EmotionType.ANGRY

            # Surprised
            elif any(word in class_lower for word in ['gasp']):
                self._emotion_mapping[idx] = EmotionType.SURPRISED

            # Default to Neutral
            else:
                self._emotion_mapping[idx] = EmotionType.NEUTRAL
        
        logger.debug(f"Built emotion mapping for {len(self.class_names)} YAMNet classes")
    
    def _load_audio_for_yamnet(self, audio_file: str, start_time: float, end_time: float) -> np.ndarray:
        """
        Load and prepare audio segment for YAMNet processing.
        
        Args:
            audio_file: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Audio waveform as numpy array (16kHz mono, float32)
        """
        # Load audio segment
        audio = AudioSegment.from_file(audio_file)
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        segment = audio[start_ms:end_ms]
        
        # Convert to mono and resample to 16kHz
        segment = segment.set_channels(1)
        segment = segment.set_frame_rate(16000)
        
        # Convert to numpy array (float32, normalized to [-1, 1])
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2**15)  # Normalize 16-bit audio
        
        return samples
    
    def _detect_emotions_in_window(
        self,
        waveform: np.ndarray,
        confidence_threshold: float = 0.3,
        debug_segment_info: Optional[str] = None
    ) -> List[Tuple[EmotionType, float]]:
        """Run YAMNet on a window and extract emotions with top-k non-neutral focus.

        The method first searches for non-neutral events (e.g., laughter) among
        top-k class probabilities per frame using a lower threshold. If any
        non-neutral emotions are found, they are returned. Otherwise, a neutral
        fallback is used.

        Args:
            waveform: Audio waveform (16kHz mono)
            confidence_threshold: Minimum confidence for neutral/top-1 fallback
            debug_segment_info: Optional string for debug logging (e.g., "segment X")

        Returns:
            List of (emotion, confidence) tuples
        """
        if len(waveform) == 0:
            return [(EmotionType.NEUTRAL, 1.0)]

        try:
            scores, embeddings, spectrogram = self.model(waveform)
            scores = scores.numpy()

            # Prefer picking non-neutral events even when not top-1
            top_k = 10
            non_neutral_threshold = 0.05  # even lower to catch subtle events like laughter
            non_neutral_detections: List[Tuple[EmotionType, float]] = []

            # Collect top-k class names and probabilities for debug
            debug_top_classes: Dict[str, float] = {}

            for frame_idx, frame_scores in enumerate(scores):
                # Get top-k indices for this frame
                if top_k >= len(frame_scores):
                    top_indices = np.argsort(frame_scores)[::-1]
                else:
                    top_indices = np.argpartition(frame_scores, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(frame_scores[top_indices])[::-1]]

                # Log top classes for first frame if debug enabled
                if frame_idx == 0 and debug_segment_info:
                    for rank, idx in enumerate(top_indices[:5]):
                        class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
                        prob = float(frame_scores[idx])
                        debug_top_classes[class_name] = max(debug_top_classes.get(class_name, 0.0), prob)

                for idx in top_indices:
                    prob = float(frame_scores[idx])
                    if prob < non_neutral_threshold:
                        continue
                    emotion = self._emotion_mapping.get(idx, EmotionType.NEUTRAL)
                    if emotion != EmotionType.NEUTRAL:
                        non_neutral_detections.append((emotion, prob))

            # Log debug info for this segment
            if debug_segment_info and debug_top_classes:
                top_5_str = ", ".join([f"{name}:{prob:.3f}" for name, prob in sorted(debug_top_classes.items(), key=lambda x: -x[1])[:5]])
                logger.debug(f"{debug_segment_info} - YAMNet top-5: {top_5_str} | Non-neutral found: {len(non_neutral_detections)}")

            if non_neutral_detections:
                return non_neutral_detections

            # Fallback: consider top-1 with original threshold (may be Neutral)
            fallback_detections: List[Tuple[EmotionType, float]] = []
            for frame_scores in scores:
                top_idx = int(np.argmax(frame_scores))
                top_prob = float(frame_scores[top_idx])
                if top_prob >= confidence_threshold:
                    fallback_detections.append((self._emotion_mapping.get(top_idx, EmotionType.NEUTRAL), top_prob))

            if fallback_detections:
                return fallback_detections

            return [(EmotionType.NEUTRAL, 1.0)]

        except Exception as e:
            logger.warning(f"Error in YAMNet inference: {e}")
            return [(EmotionType.NEUTRAL, 1.0)]
    
    def _get_dominant_emotion(
        self, 
        emotions: List[Tuple[EmotionType, float]]
    ) -> Tuple[EmotionType, float]:
        """
        Get the dominant emotion from a list of detections.
        
        Args:
            emotions: List of (emotion, confidence) tuples
            
        Returns:
            Dominant emotion and average confidence
        """
        if not emotions:
            return EmotionType.NEUTRAL, 1.0
        
        # Count occurrences and sum confidences
        emotion_stats = {}
        for emotion, confidence in emotions:
            if emotion not in emotion_stats:
                emotion_stats[emotion] = {'count': 0, 'total_confidence': 0.0}
            emotion_stats[emotion]['count'] += 1
            emotion_stats[emotion]['total_confidence'] += confidence
        
        # Find dominant emotion (by count, then by avg confidence)
        dominant_emotion = max(
            emotion_stats.items(),
            key=lambda x: (x[1]['count'], x[1]['total_confidence'] / x[1]['count'])
        )
        
        emotion_type = dominant_emotion[0]
        avg_confidence = dominant_emotion[1]['total_confidence'] / dominant_emotion[1]['count']
        
        return emotion_type, avg_confidence
    
    def _expand_segment_with_gaps(
        self, 
        segment: Dict, 
        prev_segment: Optional[Dict], 
        next_segment: Optional[Dict]
    ) -> Tuple[float, float]:
        """
        Expand segment time range to include gaps before and after.
        
        Args:
            segment: Current segment
            prev_segment: Previous segment (or None)
            next_segment: Next segment (or None)
            
        Returns:
            Expanded (start_time, end_time) tuple
        """
        start = segment['start']
        end = segment['end']
        
        # Include gap before
        if prev_segment:
            gap_start = prev_segment['end']
            start = gap_start
        
        # Include gap after
        if next_segment:
            gap_end = next_segment['start']
            end = gap_end
        
        return start, end
    
    def analyze_segments(
        self, 
        segments: List[Dict], 
        audio_file: str,
        confidence_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Analyze emotions in audio segments at both segment and word level.
        
        Args:
            segments: List of segments with 'start', 'end', 'words' fields
            audio_file: Path to audio file
            confidence_threshold: Minimum confidence for emotion detection
            
        Returns:
            Segments enriched with emotion data
        """
        if not segments:
            return segments
        
        logger.info(f"Analyzing emotions for {len(segments)} segments...")
        
        enriched_segments = []
        
        for i, segment in enumerate(segments):
            try:
                # Get expanded time range including gaps
                prev_segment = segments[i - 1] if i > 0 else None
                next_segment = segments[i + 1] if i < len(segments) - 1 else None
                
                expanded_start, expanded_end = self._expand_segment_with_gaps(
                    segment, prev_segment, next_segment
                )
                
                # Load expanded audio
                waveform = self._load_audio_for_yamnet(audio_file, expanded_start, expanded_end)
                
                # Detect emotions in the expanded segment
                debug_info = f"Segment {i+1}/{len(segments)} [{segment['start']:.2f}s-{segment['end']:.2f}s, {segment.get('speaker', 'UNKNOWN')}]"
                emotions_detected = self._detect_emotions_in_window(
                    waveform, confidence_threshold, debug_segment_info=debug_info
                )
                
                # Get dominant emotion for the segment
                dominant_emotion, avg_confidence = self._get_dominant_emotion(emotions_detected)
                
                # Create enriched segment
                enriched_segment = segment.copy()
                enriched_segment['emotion'] = dominant_emotion.value
                enriched_segment['emotion_confidence'] = avg_confidence
                
                # Analyze word-level emotions if words are available
                if 'words' in segment and segment['words']:
                    enriched_words = []
                    for word in segment['words']:
                        word_start = word['start']
                        word_end = word['end']
                        
                        # Load audio for this word
                        try:
                            word_waveform = self._load_audio_for_yamnet(
                                audio_file, word_start, word_end
                            )
                            word_emotions = self._detect_emotions_in_window(
                                word_waveform, confidence_threshold
                            )
                            word_emotion, word_confidence = self._get_dominant_emotion(word_emotions)
                            
                            enriched_word = word.copy()
                            enriched_word['emotion'] = word_emotion.value
                            enriched_word['emotion_confidence'] = word_confidence
                            enriched_words.append(enriched_word)
                            
                        except Exception as e:
                            logger.warning(f"Error analyzing word emotion: {e}")
                            enriched_word = word.copy()
                            enriched_word['emotion'] = EmotionType.NEUTRAL.value
                            enriched_word['emotion_confidence'] = 1.0
                            enriched_words.append(enriched_word)
                    
                    enriched_segment['words'] = enriched_words
                
                enriched_segments.append(enriched_segment)
                
            except Exception as e:
                logger.warning(f"Error analyzing segment {i} emotion: {e}")
                # Fall back to neutral
                enriched_segment = segment.copy()
                enriched_segment['emotion'] = EmotionType.NEUTRAL.value
                enriched_segment['emotion_confidence'] = 1.0
                enriched_segments.append(enriched_segment)
        
        # Build breakdown by emotion type with segment time ranges and speakers
        breakdown: Dict[str, List[str]] = {et.value: [] for et in EmotionType}
        for seg in enriched_segments:
            emotion_label = seg.get('emotion', EmotionType.NEUTRAL.value)
            start = float(seg.get('start', 0.0))
            end = float(seg.get('end', start))
            speaker = seg.get('speaker', 'UNKNOWN')
            breakdown.setdefault(emotion_label, []).append(f"{start:.2f}s-{end:.2f}s ({speaker})")

        logger.debug("Emotion breakdown by type (segment-level):")
        for et in EmotionType:
            items = breakdown.get(et.value, [])
            joined = ", ".join(items)
            logger.debug(f"{et.value} [{joined}]")

        logger.info("Emotion analysis completed")
        return enriched_segments

