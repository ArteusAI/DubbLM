"""Centralized segment optimization logic for the dubbing pipeline."""

from typing import List, Dict, Any, Optional
from src.dubbing.core.log_config import get_logger

logger = get_logger(__name__)


class SegmentOptimizer:
    """Handles segment merging, splitting, and optimization at different pipeline stages."""

    def __init__(self, config, tts_system: Optional[str] = None):
        """
        Initialize the SegmentOptimizer.

        Args:
            config: DubbingConfig instance
            tts_system: TTS system name (e.g., 'gemini', 'coqui'), defaults to config value
        """
        self.config = config
        self.opt_cfg = config.get('segments_optimization', {})
        self.tts_system = tts_system or config.get('tts_system', 'coqui')

        # Load parameters with defaults
        self.post_diarization_merge_gap = self.opt_cfg.get('post_diarization_merge_gap', 0.3)
        self.post_translation_merge_gap = self.opt_cfg.get('post_translation_merge_gap', 1.5)
        self.max_segment_chars = self.opt_cfg.get('max_segment_chars', 420)
        self.max_segment_duration = self.opt_cfg.get('max_segment_duration', 60)
        self.min_segment_duration = self.opt_cfg.get('min_segment_duration', 0.5)

        logger.debug(f"SegmentOptimizer initialized with: "
                    f"tts_system={self.tts_system}, "
                    f"post_diarization_gap={self.post_diarization_merge_gap}s, "
                    f"post_translation_gap={self.post_translation_merge_gap}s, "
                    f"max_chars={self.max_segment_chars}, "
                    f"max_duration={self.max_segment_duration}s")

    def _get_translation_separator(self, gap: float) -> str:
        """
        Get separator for merged translation segments based on TTS system and gap duration.

        For Gemini TTS, uses pause markers to create more natural speech:
        - gap < 0.25s  → [short pause] (~250ms, similar to comma)
        - 0.25s ≤ gap < 0.5s → [medium pause] (~500ms, similar to sentence break)
        - gap ≥ 0.5s → [long pause] (~1000ms+, for dramatic effect)

        Args:
            gap: Time gap between segments in seconds

        Returns:
            Appropriate separator string
        """
        # For Gemini TTS, use pause markers based on official spec
        if self.tts_system == 'gemini':
            if gap < 0.25:
                return " [short pause] "
            elif gap < 0.6:
                return " [medium pause] "
            else:
                return " [long pause] "

        # For other TTS systems, use simple space
        return " "

    def merge_adjacent_segments(
        self,
        segments: List[Dict[str, Any]],
        max_gap: float,
        phase: str = "unknown"
    ) -> List[Dict[str, Any]]:
        """
        Merge adjacent segments from the same speaker when the gap is small.

        Args:
            segments: List of transcript segments with speaker, start, end, and text
            max_gap: Maximum gap in seconds to consider for merging
            phase: Pipeline phase for logging (e.g., "post_diarization", "post_translation")

        Returns:
            List of merged segments
        """
        if not segments:
            return []

        logger.debug(f"Merging adjacent segments ({phase}): {len(segments)} segments, max_gap={max_gap}s")

        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])

        merged = []
        current_segment = None

        for segment in sorted_segments:
            if current_segment is None:
                # Start first segment
                current_segment = segment.copy()
                continue

            # Check if we can merge with current segment
            same_speaker = segment.get('speaker') == current_segment.get('speaker')
            gap = segment['start'] - current_segment['end']
            can_merge = same_speaker and gap <= max_gap

            if can_merge:
                # Merge into current segment
                old_end = current_segment['end']
                current_segment['end'] = segment['end']

                # Concatenate original text with space (no markers for original text)
                current_text = current_segment.get('text', '').strip()
                new_text = segment.get('text', '').strip()
                current_segment['text'] = f"{current_text} {new_text}".strip()

                # Merge translations with TTS-aware separator
                if 'translation' in segment and 'translation' in current_segment:
                    # Use Gemini pause markers for translations
                    separator = self._get_translation_separator(gap)
                    current_trans = current_segment.get('translation', '').strip()
                    new_trans = segment.get('translation', '').strip()
                    current_segment['translation'] = f"{current_trans}{separator}{new_trans}".strip()

                    # Log when pause commands are added
                    if '[' in separator and 'pause]' in separator:
                        pause_type = separator.strip()
                        logger.debug(f"Added {pause_type} (gap={gap:.2f}s) merging segments at "
                                    f"{old_end:.2f}s-{segment['start']:.2f}s, speaker={segment.get('speaker', 'unknown')}")

                # Log segment modification
                logger.debug(f"Merged segment ({phase}): [{old_end:.2f}s-{segment['start']:.2f}s] "
                            f"gap={gap:.2f}s, speaker={segment.get('speaker', 'unknown')}, "
                            f"new_duration={current_segment['end'] - current_segment['start']:.2f}s")

                # Update confidence (take minimum if present)
                if 'confidence' in segment and 'confidence' in current_segment:
                    current_segment['confidence'] = min(
                        current_segment.get('confidence', 1.0),
                        segment.get('confidence', 1.0)
                    )

                # Merge word-level timestamps if present
                if 'words' in segment and 'words' in current_segment:
                    current_segment['words'].extend(segment['words'])
            else:
                # Can't merge - save current and start new
                merged.append(current_segment)
                current_segment = segment.copy()

        # Add last segment
        if current_segment is not None:
            merged.append(current_segment)

        merged_count = len(sorted_segments) - len(merged)
        if merged_count > 0:
            logger.debug(f"Merged {merged_count} segments ({phase}): {len(sorted_segments)} → {len(merged)}")

        return merged

    def split_long_segments(
        self,
        segments: List[Dict[str, Any]],
        max_chars: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Split segments that exceed character limit using word-level timestamps.

        Args:
            segments: List of transcript segments
            max_chars: Maximum characters per segment (uses config default if None)

        Returns:
            List of segments with long ones split
        """
        if max_chars is None:
            max_chars = self.max_segment_chars

        if not segments:
            return []

        logger.debug(f"Splitting long segments: max_chars={max_chars}")

        result = []
        split_count = 0

        for segment in segments:
            text = segment.get('text', '')

            # If segment is within limit or has no word-level data, keep as-is
            if len(text) <= max_chars or 'words' not in segment or not segment['words']:
                result.append(segment)
                continue

            # Split using word-level timestamps
            logger.debug(f"Splitting long segment: {len(text)} chars (max={max_chars}), "
                        f"duration={segment['end'] - segment['start']:.2f}s, "
                        f"speaker={segment.get('speaker', 'unknown')}, "
                        f"start={segment['start']:.2f}s")
            split_segments = self._split_long_segment(segment, max_chars)
            result.extend(split_segments)
            split_count += len(split_segments) - 1

            # Log each split chunk
            for i, chunk in enumerate(split_segments, 1):
                chunk_text = chunk.get('text', '')
                logger.debug(f"  Split chunk {i}/{len(split_segments)}: {len(chunk_text)} chars, "
                            f"{chunk['start']:.2f}s-{chunk['end']:.2f}s")

        if split_count > 0:
            logger.debug(f"Split {split_count} long segments: {len(segments)} → {len(result)}")

        return result

    def _split_long_segment(
        self,
        segment: Dict[str, Any],
        max_chars: int
    ) -> List[Dict[str, Any]]:
        """
        Split a single segment into smaller chunks using word-level timestamps.

        Tries to split on sentence boundaries (., !, ?) when possible.

        Args:
            segment: Original segment with 'words' list
            max_chars: Character limit per chunk

        Returns:
            List of new segments derived from the original
        """
        words: List[Dict[str, Any]] = segment['words']
        chunks: List[Dict[str, Any]] = []

        idx = 0
        while idx < len(words):
            char_count = 0
            last_sentence_break = -1
            start_idx = idx
            split_end_idx = None

            # Grow the chunk word-by-word
            while idx < len(words):
                word_text: str = words[idx]['word']
                # Add space if not first word in chunk
                projected = char_count + (1 if char_count > 0 else 0) + len(word_text)

                # Track sentence boundary
                if word_text.rstrip().endswith(('.', '!', '?')):
                    last_sentence_break = idx

                # If adding the word would exceed limit
                if projected > max_chars:
                    # Decide on split position
                    if last_sentence_break >= start_idx:
                        # Compute leftover length to decide whether to keep remainder
                        leftover_words = words[last_sentence_break + 1:]
                        leftover_text_len = len(' '.join(w['word'] for w in leftover_words))

                        if leftover_text_len < 50:
                            # Keep remainder – allow slight overflow
                            idx += 1
                            char_count = projected
                            # Continue consuming until end to keep sentence integrity
                            while idx < len(words):
                                w_text = words[idx]['word']
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

            # Fallback if split_end_idx was not set
            if split_end_idx is None:
                split_end_idx = len(words) - 1 if idx >= len(words) else idx - 1

            # Build chunk with words [start_idx, split_end_idx]
            chunk_words = words[start_idx:split_end_idx + 1]
            chunk_text = ' '.join(w['word'] for w in chunk_words)
            chunk_start = chunk_words[0]['start']
            chunk_end = chunk_words[-1]['end']

            new_segment = {
                'text': chunk_text,
                'start': chunk_start,
                'end': chunk_end,
                'speaker': segment['speaker'],
                'words': chunk_words,
            }

            # Propagate other fields if present
            if 'confidence' in segment:
                new_segment['confidence'] = segment['confidence']
            if 'emotion' in segment:
                new_segment['emotion'] = segment['emotion']

            chunks.append(new_segment)

            # Prepare for next iteration
            idx = split_end_idx + 1

        return chunks

    def filter_short_segments(
        self,
        segments: List[Dict[str, Any]],
        min_duration: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter out segments that are too short.

        Args:
            segments: List of transcript segments
            min_duration: Minimum duration in seconds (uses config default if None)

        Returns:
            Filtered list of segments
        """
        if min_duration is None:
            min_duration = self.min_segment_duration

        if not segments:
            return []

        filtered = [
            seg for seg in segments
            if (seg.get('end', 0) - seg.get('start', 0)) >= min_duration
        ]

        removed = len(segments) - len(filtered)
        if removed > 0:
            logger.debug(f"Filtered {removed} short segments (< {min_duration}s): "
                        f"{len(segments)} → {len(filtered)}")

        return filtered

    def optimize_post_diarization(
        self,
        transcription: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimize segments after diarization and transcription.

        This performs:
        1. Merging of adjacent segments from same speaker (tight gap)
        2. Splitting of overly long segments
        3. Filtering of very short segments

        Args:
            transcription: Raw transcription from diarization

        Returns:
            Optimized transcription segments
        """
        logger.info("Optimizing segments after diarization...")

        # Step 1: Merge adjacent segments with tight gap
        merged = self.merge_adjacent_segments(
            transcription,
            max_gap=self.post_diarization_merge_gap,
            phase="post_diarization"
        )

        # Step 2: Split long segments
        split = self.split_long_segments(merged)

        # Step 3: Filter very short segments
        filtered = self.filter_short_segments(split)

        logger.info(f"Post-diarization optimization: {len(transcription)} → {len(filtered)} segments")

        return filtered

    def optimize_post_translation(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimize segments after translation.

        This performs:
        1. Merging of adjacent segments from same speaker (relaxed gap)

        We use a more relaxed gap here since the text is already translated
        and we want to optimize for TTS synthesis efficiency.

        Args:
            segments: Translated segments

        Returns:
            Optimized segments
        """
        logger.info("Optimizing segments after translation...")

        # Merge with relaxed gap for TTS optimization
        merged = self.merge_adjacent_segments(
            segments,
            max_gap=self.post_translation_merge_gap,
            phase="post_translation"
        )

        logger.info(f"Post-translation optimization: {len(segments)} → {len(merged)} segments")

        return merged
