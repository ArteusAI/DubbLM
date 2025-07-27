"""Subtitle utilities for the Smart Dubbing system."""

import os
from typing import List, Dict
from nltk.tokenize import sent_tokenize

from src.utils.time_utils import format_seconds_to_srt, format_seconds_to_hms
from src.utils.sent_split import greedy_sent_split
from src.dubbing.core.log_config import get_logger

logger = get_logger(__name__)


class SubtitleManager:
    """Manages subtitle generation and formatting for the Smart Dubbing system."""
    
    def save_subtitles(self, segments: List[Dict], subtitle_type: str, output_path: str, max_chars_per_line: int = 160) -> None:
        """Save optimized subtitles in SRT format.
        
        Args:
            segments: List of transcript segments with timing information
            subtitle_type: Type of subtitles to save ('original' or 'translation')
            output_path: Path to save the SRT file
            max_chars_per_line: Maximum characters per subtitle line (default: 160)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare optimized entries
        optimized_entries = []
        
        for segment in segments:
            # Skip segments without translation if needed
            if subtitle_type == "translation" and "translation" not in segment:
                continue
            
            # Get timestamps and text
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"] if subtitle_type == "original" else segment["translation"]
            duration = end_time - start_time
            
            # Split text into sentences
            sentences = sent_tokenize(text)
            # Add a period at the end of each sentence if it doesn't already have ending punctuation
            for i in range(len(sentences)):
                sentence = sentences[i].strip()
                if sentence and not sentence[-1] in ['.', '!', '?', ':', ';']:
                    sentences[i] = sentence + '.'
            
            if len(sentences) == 1 and len(text) <= max_chars_per_line:
                # If it's already a single short sentence, keep it as is
                optimized_entries.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                })
            else:
                # Distribute the time proportionally based on sentence length
                total_chars = sum(len(s) for s in sentences)
                current_time = start_time
                
                for sentence in sentences:
                    # Skip empty sentences
                    if not sentence.strip():
                        continue
                        
                    # Calculate the time proportion for this sentence
                    sentence_duration = (len(sentence) / total_chars) * duration if total_chars > 0 else duration / len(sentences)
                    sentence_end_time = min(current_time + sentence_duration, end_time)
                    
                    # Split long sentences using greedy_sent_split instead of word-by-word
                    if len(sentence) > max_chars_per_line:
                        lines = greedy_sent_split(sentence, max_chars_per_line)
                        # Create separate entry for each line with evenly distributed time
                        line_duration = sentence_duration / len(lines)
                        line_start_time = current_time
                        
                        for line in lines:
                            line_end_time = min(line_start_time + line_duration, end_time)
                            optimized_entries.append({
                                'start_time': line_start_time,
                                'end_time': line_end_time,
                                'text': line
                            })
                            line_start_time = line_end_time
                        
                        current_time = line_start_time
                    else:
                        # For short sentences, keep as a single entry
                        optimized_entries.append({
                            'start_time': current_time,
                            'end_time': sentence_end_time,
                            'text': sentence
                        })
                        current_time = sentence_end_time
        
        # Write the optimized SRT file
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(optimized_entries):
                f.write(f"{i+1}\n")
                f.write(f"{format_seconds_to_srt(entry['start_time'])} --> {format_seconds_to_srt(entry['end_time'])}\n")
                f.write(f"{entry['text']}\n\n")
        
        logger.debug(f"Saved optimized {subtitle_type} subtitles to {output_path} ({len(optimized_entries)} entries)")

    def save_debug_tsv(self, segments: List[Dict], output_dir: str = "artifacts/debug") -> None:
        """Save original and translated text for each segment in a TSV file.
        
        Args:
            segments: List of transcript segments with translations
            output_dir: Directory to save the TSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        tsv_path = os.path.join(output_dir, "translations.tsv")
        
        with open(tsv_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("time\toriginal\ttranslation\n")
            
            # Write each segment
            for segment in segments:
                start_time_str = format_seconds_to_srt(segment["start"])
                original_text = segment["text"]
                # Check if translation exists (might not in debug_diarize_only mode)
                translation = segment.get("translation", "")
                
                # Escape any tab characters in the text
                original_text = original_text.replace("\t", " ")
                translation = translation.replace("\t", " ")
                
                f.write(f"{start_time_str}\t{original_text}\t{translation}\n")
        
        logger.debug(f"Saved transcription/translation TSV to {tsv_path}")
    
    def _parse_srt_time(self, time_str: str) -> float:
        """Parse SRT time format (HH:MM:SS,mmm) to seconds.
        
        Args:
            time_str: SRT format time string
            
        Returns:
            Time in seconds
        """
        hours, minutes, rest = time_str.split(':')
        seconds, milliseconds = rest.split(',')
        
        return (int(hours) * 3600 + 
                int(minutes) * 60 + 
                int(seconds) + 
                int(milliseconds) / 1000)
    
    def combine_subtitle_files(self, subtitle_files: List[str], output_file: str) -> None:
        """Combine multiple SRT files into a single SRT file.
        
        Args:
            subtitle_files: List of SRT files to combine
            output_file: Path to save the combined SRT file
        """
        entries = []
        
        # Read all subtitle entries from all files
        for file_path in subtitle_files:
            if not os.path.exists(file_path):
                continue
                
            current_entry = {}
            entry_lines = []
            line_count = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    line_count += 1
                    
                    if not line and entry_lines:
                        # Parse the entry
                        try:
                            index = int(entry_lines[0])
                            times = entry_lines[1].split(' --> ')
                            start_time = self._parse_srt_time(times[0])
                            end_time = self._parse_srt_time(times[1])
                            text = '\n'.join(entry_lines[2:])
                            
                            entries.append({
                                'index': index,
                                'start_time': start_time,
                                'end_time': end_time,
                                'text': text
                            })
                        except (ValueError, IndexError) as e:
                            logger.error(f"Error parsing subtitle entry in {file_path}: {e}")
                            
                        entry_lines = []
                    elif line:
                        entry_lines.append(line)
        
        # Sort entries by start time
        entries.sort(key=lambda x: x['start_time'])
        
        # Write combined SRT file
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(entries):
                f.write(f"{i+1}\n")
                f.write(f"{format_seconds_to_srt(entry['start_time'])} --> {format_seconds_to_srt(entry['end_time'])}\n")
                f.write(f"{entry['text']}\n\n") 