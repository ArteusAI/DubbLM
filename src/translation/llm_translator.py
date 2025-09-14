from typing import List, Dict, Any, Optional, Tuple, Set, Literal, TYPE_CHECKING
import time
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import math

try:
    import json_repair
except ImportError:
    json_repair = None

from translation.translation_interface import TranslationInterface
from translation.prompts import REFINEMENT_PROMPTS, LENGTH_ADJUST_PROMPT
from src.dubbing.core.log_config import get_logger

if TYPE_CHECKING:
    from src.dubbing.core.cache_manager import CacheManager

# Import LLM-related dependencies, with error handling for missing packages
try:
    from llama_index.llms.gemini import Gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Try to import OpenRouter for model access
try:
    from llama_index.llms.openrouter import OpenRouter
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

# Try to import json_repair for handling JSON parsing errors
try:
    import json_repair
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False

logger = get_logger(__name__)


class LLMTranslator(TranslationInterface):
    """
    Language Model-based translator implementing TranslationInterface.
    Uses LLMs like Gemini or OpenRouter models to perform context-aware translations.
    """
    
    def __init__(
        self,
        llm_provider: Literal["gemini", "openrouter"] = "gemini",
        model_name: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 16384,
        refinement_llm_provider: Optional[Literal["gemini", "openrouter"]] = None,
        refinement_model_name: Optional[str] = None,
        refinement_temperature: float = 1.0,
        refinement_max_tokens: Optional[int] = None,
        debug: bool = False,
        cache_dir: Optional[str] = "cache/translation_cache",
        enable_cache: bool = True,
        glossary: Optional[Dict[str, str]] = None,
        refinement_persona: str = "manager",
        cache_manager: Optional['CacheManager'] = None,
        prompt_prefix: Optional[str] = None,
    ):
        """
        Initialize LLM translator.
        
        Args:
            llm_provider: LLM provider to use ("gemini" or "openrouter")
            model_name: LLM model name
                - For Gemini: Default is "models/gemini-2.5-flash-preview-04-17"
                - For OpenRouter: Models like "anthropic/claude-3.7-sonnet:thinking"
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens in the response (for OpenRouter)
            refinement_llm_provider: LLM provider to use for refinement (defaults to same as llm_provider)
            refinement_model_name: LLM model name for refinement (defaults to same as model_name)
            refinement_temperature: Temperature for refinement (defaults to 1.0)
            refinement_max_tokens: Maximum number of tokens for refinement (defaults to same as max_tokens)
            debug: Enable debug mode to write prompts and results to files
            cache_dir: Directory to store translation cache (deprecated, use cache_manager instead)
            enable_cache: Whether to enable translation caching
            glossary: Dictionary mapping source terms to their desired translations
            refinement_persona: The persona to use for the refinement prompt ("manager", "child", "tractor_driver")
            cache_manager: Cache manager instance for organized caching
        """
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = debug
        
        # Set default model names based on provider
        if model_name:
            self.model_name = model_name
        elif llm_provider == "gemini":
            self.model_name = "models/gemini-2.5-flash-preview-04-17"
        elif llm_provider == "openrouter":
            self.model_name = "anthropic/claude-3.7-sonnet:thinking"
        
        # Configure refinement model settings
        self.refinement_llm_provider = refinement_llm_provider or llm_provider
        self.refinement_model_name = refinement_model_name or self.model_name
        self.refinement_temperature = refinement_temperature
        self.refinement_max_tokens = refinement_max_tokens or max_tokens
        
        # Initialize caching settings
        self.enable_cache = enable_cache
        self.cache_manager = cache_manager
        self.cache_dir = cache_dir  # Keep for backward compatibility when cache_manager is None
        self.translation_cache = {}  # In-memory cache
        
        # Initialize glossary
        self.glossary = glossary or {}
        
        # Set refinement persona
        self.refinement_persona = refinement_persona
        
        # Optional custom prompt prefix to inject additional context
        self.prompt_prefix = (prompt_prefix or "").strip() or None
        
        self.llm = None
        self.refinement_llm = None
        
    def initialize(self) -> None:
        """Initialize the LLM translation system."""
        import os
        
        if not JSON_REPAIR_AVAILABLE:
            raise ImportError("json_repair is not installed. Please install it.")
        
        # Initialize translation LLM
        self.llm = self._create_llm(
            provider=self.llm_provider,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            purpose="translation"
        )
            
        # Initialize refinement LLM
        self.refinement_llm = self._create_llm(
            provider=self.refinement_llm_provider,
            model_name=self.refinement_model_name,
            temperature=self.refinement_temperature,
            max_tokens=self.refinement_max_tokens,
            purpose="refinement"
        )
        
        # Initialize cache directory if caching is enabled
        if self.enable_cache:
            if self.cache_manager:
                # Use organized cache structure through CacheManager
                self._load_cache()
            elif self.cache_dir:
                # Fallback to legacy cache directory structure
                os.makedirs(self.cache_dir, exist_ok=True)
                self._load_cache()
            
    def _create_llm(
        self, 
        provider: str, 
        model_name: str, 
        temperature: float, 
        max_tokens: Optional[int] = None,
        purpose: str = "default"
    ) -> Any:
        """
        Create and return an LLM instance based on the provided parameters.
        
        Args:
            provider: LLM provider ('gemini' or 'openrouter')
            model_name: Name of the model to use
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens (for OpenRouter)
            purpose: Purpose of this LLM ('translation' or 'refinement')
            
        Returns:
            An initialized LLM instance
            
        Raises:
            ValueError: If the provider is not supported
            RuntimeError: If initialization fails
        """
        import os
        
        if provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Gemini dependencies are not installed. Please install llama-index.")
            
            # Get API key from environment
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key not found in environment. Set GOOGLE_API_KEY environment variable.")
            
            try:
                # Initialize Gemini LLM
                llm = Gemini(model=model_name, temperature=temperature)
                logger.debug(f"Initialized Gemini {purpose} LLM with model: {model_name}, temperature={temperature}")
                return llm
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Gemini LLM for {purpose}: {str(e)}")
                
        elif provider == "openrouter":
            if not OPENROUTER_AVAILABLE:
                raise ImportError("OpenRouter dependencies are not installed. Please install llama-index-llms-openrouter.")
            
            # Get API key from environment
            openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                raise ValueError("OpenRouter API key not found in environment. Set OPENROUTER_API_KEY environment variable.")
            
            try:
                # Initialize OpenRouter LLM
                llm = OpenRouter(
                    api_key=openrouter_api_key,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                logger.debug(f"Initialized OpenRouter {purpose} LLM with model: {model_name}, temperature={temperature}")
                return llm
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenRouter LLM for {purpose}: {str(e)}")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Use 'gemini' or 'openrouter'.")
            
    def _load_cache(self) -> None:
        """Load translation cache from disk if it exists."""
        if not self.enable_cache:
            return
        
        cache_file = None
        if self.cache_manager:
            # Use organized cache structure
            cache_dir = self.cache_manager.get_cache_path("translation_cache")
            cache_file = cache_dir / "translation_cache.json"
        elif self.cache_dir:
            # Fallback to legacy cache directory structure
            cache_file = os.path.join(self.cache_dir, "translation_cache.json")
        else:
            return
            
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.translation_cache = json.load(f)
                logger.debug(f"Loaded translation cache with {len(self.translation_cache)} entries")
            except Exception as e:
                logger.error(f"Error loading translation cache: {e}")
                self.translation_cache = {}
                
    def _save_cache(self) -> None:
        """Save translation cache to disk."""
        if not self.enable_cache or not self.translation_cache:
            return
        
        cache_file = None
        if self.cache_manager:
            # Use organized cache structure
            cache_dir = self.cache_manager.get_cache_path("translation_cache")
            cache_file = cache_dir / "translation_cache.json"
        elif self.cache_dir:
            # Fallback to legacy cache directory structure
            cache_file = os.path.join(self.cache_dir, "translation_cache.json")
        else:
            return
            
        if cache_file:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.translation_cache, f, ensure_ascii=False, indent=2)
                logger.debug(f"Saved translation cache with {len(self.translation_cache)} entries")
            except Exception as e:
                logger.error(f"Error saving translation cache: {e}")
    
    def _generate_cache_key(self, chunk_text: str, source_language: str, target_language: str) -> str:
        """
        Generate a unique cache key for a chunk based on its content and languages.
        
        Args:
            chunk_text: Text content of the chunk
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            A unique hash string to use as cache key
        """
        # Combine the parameters that fully define a translation
        # Include prompt prefix in the key so cache varies when extra context changes
        cache_data = f"{chunk_text}|{source_language}|{target_language}|{self.llm_provider}|{self.model_name}|{self.temperature}|{(self.prompt_prefix or '')}"
        # Create a hash of this data
        return hashlib.md5(cache_data.encode("utf-8")).hexdigest()
    
    def _format_time(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = math.floor(seconds / 3600)
        minutes = math.floor((seconds % 3600) / 60)
        secs = math.floor(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _analyze_context(
        self,
        transcription: List[Dict],
        source_language: str,
        target_language: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze the context of the transcribed text and generate an initial summary in one step.
        
        Args:
            transcription: List of transcription segments
            source_language: Source language code
            target_language: Target language code
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with context information and initial summary
        """
        if not self.is_available():
            raise RuntimeError("LLM translator is not initialized. Call initialize() first.")
        
        # Format transcription with timestamps and speakers
        formatted_lines = []
        for segment in transcription:
            start_time_str = self._format_time(segment["start"])
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment["text"]
            formatted_lines.append(f"[{start_time_str}] {speaker}: {text}")
            
        all_text = "\n".join(formatted_lines)

        glossary_section = ""
        if self.glossary:
            glossary_entries = "\n".join([f"- \"{term}\" → \"{translation}\"" for term, translation in self.glossary.items()])
            glossary_section = f"""
# Translation glossary (MUST be followed. Adapt for grammar):
<glossary>
{glossary_entries}
</glossary>

CRITICAL: You MUST use the translations from the glossary for all listed terms.
IMPORTANT: The glossary provides base forms of translations. When using a term from the glossary, you MUST adapt it to fit the grammatical context (e.g., case, gender, number, verb conjugation) of the sentence in the target language "{target_language}". For example, if the glossary says "cat" -> "кошка", and the sentence requires the genitive case, you should use "кошки", not "кошка". Do not just insert the glossary term verbatim if it violates grammatical rules.
"""
        
        # Optional additional context section
        additional_context_section = f"""
# Additional context (optional):
{self.prompt_prefix}
""" if self.prompt_prefix else ""

        # Combine context analysis and initial summarization in one prompt
        combined_prompt = f"""
Analyze the following transcript in "{source_language}" language and provide:

1. Context Analysis:
   - The general topic or domain (e.g., medical, technical, casual conversation)
   - Any specialized terminology or jargon (e.g., AI, machine learning, deep learning, etc. list all of them)
   - The overall tone or style of speech
   - Key themes or subjects discussed

2. Transcript Summary:
   - Identify logical chapters/sections based on topic shifts.
   - For long transcripts (e.g., over 30 minutes), aim to create chapters that cover approximately 15-20 minutes of content each, while still following logical topic shifts.
   - For each chapter/section, provide a clear title and brief summary (2-3 sentences).
   - Assign approximate timecodes for each chapter (use format "HH:MM:SS" for start_time).
   - Write a comprehensive overall summary (3-5 sentences) describing the main topics and flow of the content, suitable for use as a video description.

{glossary_section}
{additional_context_section}

IMPORTANT: Create the summary in "{target_language}" language.

Transcript (format: [HH:MM:SS] SPEAKER: text):
<transcript>
{all_text}
</transcript>

IMPORTANT: Create the summary in "{target_language}" language.

Provide your analysis in JSON format with these keys: 
domain, terminology, tone, themes, chapters, overall_summary

For chapters, include title, summary, and start_time for each chapter.

Example JSON output:
{{
    "domain": "technology",
    "terminology": ["API", "LLM", "vector database"],
    "tone": "informative",
    "themes": ["artificial intelligence", "software development"],
    "chapters": [
        {{
            "title": "Introduction to AI",
            "summary": "Brief overview of artificial intelligence concepts.",
            "start_time": "00:00:00"
        }},
        {{
            "title": "Machine Learning Applications",
            "summary": "Discussion of real-world applications of machine learning.",
            "start_time": "00:18:45" 
        }},
        {{
            "title": "Future Trends",
            "summary": "Exploring upcoming advancements in AI.",
            "start_time": "00:35:10"
        }}
    ],
    "overall_summary": "This video provides a comprehensive overview of artificial intelligence, starting with fundamental concepts, moving into practical machine learning applications across various industries, and concluding with a look at future trends and the potential impact of AI development."
}}
"""
        
        try:
            result = self.llm.complete(combined_prompt)
            if hasattr(result, "text"):
                response_text = result.text
            else:
                response_text = str(result)
            
            # Extract JSON from the response
            try:
                combined_info = json_repair.loads(response_text)
            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"Error parsing combined analysis JSON: {e}")
                combined_info = {
                    "domain": "general",
                    "terminology": [],
                    "tone": "neutral",
                    "themes": ["conversation"],
                    "chapters": [
                        {
                            "title": "Conversation",
                            "summary": "General discussion",
                            "start_time": "00:00:00"
                        }
                    ],
                    "overall_summary": "A general conversation."
                }
                
            # Validate the structure
            required_keys = ["domain", "terminology", "tone", "themes", "chapters", "overall_summary"]
            for key in required_keys:
                if key not in combined_info:
                    if key in ["domain", "tone", "overall_summary"]:
                        combined_info[key] = "general" if key == "domain" else "neutral" if key == "tone" else "A conversation."
                    else:
                        combined_info[key] = [] if key in ["terminology", "themes"] else [{"title": "Conversation", "summary": "General discussion"}]
            
            # Create a text-only summary for easy passing to other methods
            text_summary = combined_info.get("overall_summary", "")
            text_summary += "\n\nChapters:\n"
            
            for chapter in combined_info.get("chapters", []):
                text_summary += f"- {chapter.get('title')}: {chapter.get('summary')}\n"
            
            combined_info["text_summary"] = text_summary
            
            # Process chapters to add default start/end time if not present
            for chapter in combined_info.get("chapters", []):
                if "start_time" not in chapter:
                    chapter["start_time"] = "00:00:00"
            
            logger.debug(f"Generated combined context analysis and summary in {target_language}")
            return combined_info
            
        except Exception as e:
            logger.error(f"Error in combined context analysis and summary: {e}", exc_info=True)
            # Fallback without LLM
            combined_info = {
                "domain": "general",
                "terminology": [],
                "tone": "neutral",
                "themes": ["conversation"],
                "chapters": [
                    {
                        "title": "Conversation",
                        "summary": "General discussion",
                        "start_time": "00:00:00"
                    }
                ],
                "overall_summary": "A general conversation."
            }
            # Create a text-only summary for the fallback case
            text_summary = "A general conversation.\n\nChapters:\n- Conversation: General discussion"
            combined_info["text_summary"] = text_summary
            
            return combined_info

    def translate(
        self,
        segments: List[Dict],
        context_info: Optional[Dict[str, Any]] = None,
        source_language: str = "en",
        target_language: str = "ru",
        **kwargs
    ) -> List[Dict]:
        """
        Translate segments with context awareness.
        
        This method handles the entire translation pipeline:
        1. Analyzing context and generating initial summary in source language (if not provided)
        2. Optimizing segments (e.g., merging adjacent segments from same speaker) with small max_chars
        3. Chunking segments into logical groups
        4. Translating chunks with context and source-language summary
        5. Generating a human-readable timecodes report (using source language summary)
        6. Refining the full translation for coherence and flow
        7. Decomposing translated chunks back to segment level
        8. Optimizing segments again with larger max_chars
        
        Args:
            segments: List of transcript segments
            context_info: Optional pre-generated context and summary information
            source_language: Source language code
            target_language: Target language code
            **kwargs: Additional parameters
            
        Returns:
            List of segments with translations added
        """
        if not self.is_available():
            raise RuntimeError("LLM translator is not initialized. Call initialize() first.")
            
        # Start timing for performance metrics
        start_time = time.perf_counter()
        
        # 1. Perform combined context analysis and initial summarization
        logger.info("Performing combined context analysis and initial summarization...")
        context_info = self._analyze_context(
            transcription=segments,
            source_language=source_language,
            target_language=target_language,
            **kwargs
        )
        
        # Extract source language summary for use in translation
        source_summary = context_info.get("text_summary", "")
        summary_json = {
            "chapters": context_info.get("chapters", []),
            "overall_summary": context_info.get("overall_summary", "")
        }
        
        # 2. First optimization pass - small chunks for better translation
        max_chars = 180
        logger.debug(f"First optimization pass - merging adjacent segments (max_chars={max_chars})...")
        optimized_segments = self._optimize_segments(segments, max_gap_seconds=0.3, max_chars=max_chars)
        
        # 3. Chunk optimized segments for translation
        logger.debug("Chunking optimized segments for translation...")
        chunks = self._chunk_transcription(optimized_segments)
        
        # 4. Translate chunks with the source language summary for context
        logger.info("Translating chunks with context awareness and initial summary...")
        translated_chunks = self._translate_chunks(
            chunks=chunks, 
            context_info=context_info, 
            source_language=source_language, 
            target_language=target_language,
            source_summary=source_summary,  # Pass the source language summary here
            **kwargs
        )
        
        # 5. Generate a human-readable timecodes report using the source language summary
        report_path = kwargs.get("timecodes_report_path", "artifacts/timecodes.txt")
        self._generate_timecodes_report(
            summary_json=summary_json,
            report_path=report_path
        )
        
        # 6. Refine the full translation for coherence and flow
        logger.info("Refining the full translation using source language summary...")
        refined_chunks = self._refine_translation(
            translated_chunks=translated_chunks,
            context_info=context_info,
            source_language=source_language,
            target_language=target_language,
            dialogue_summary=source_summary,  # Use source language summary directly
            **kwargs
        )
        
        # 7. Segment refined translations back to individual segments
        logger.debug("Decomposing refined translated chunks back to segment level...")
        translated_segments = self._segment_translations(refined_chunks)
        
        # 8. Second optimization pass - larger chunks for final output
        max_chars = 600
        logger.debug(f"Second optimization pass - merging translated segments (max_chars={max_chars})...")
        final_segments = self._optimize_segments(translated_segments, max_gap_seconds=0.3, max_chars=max_chars)
        
        # Report performance metrics
        elapsed_time = time.perf_counter() - start_time
        logger.debug(f"Completed translation pipeline (including refinement) in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {len(segments)} segments → {len(optimized_segments)} initial optimized segments → {len(chunks)} chunks → {len(refined_chunks)} refined chunks → {len(translated_segments)} translated segments → {len(final_segments)} final segments")
        
        return final_segments
    
    def _optimize_segments(self, segments: List[Dict], max_gap_seconds: float = 0.3, max_chars: int = 500) -> List[Dict]:
        """
        Optimize segments by merging adjacent ones from the same speaker.
        
        Args:
            segments: List of segments with translations
            max_gap_seconds: Maximum gap between segments to consider for merging
            max_chars: Maximum character length for merged segments
            
        Returns:
            List of optimized segments
        """
        optimized_segments = []
        current_segment = None
        
        # Sort segments by start time to ensure proper ordering
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        
        for segment in sorted_segments:
            if current_segment is None:
                # First segment
                current_segment = segment.copy()
            elif (segment["speaker"] == current_segment["speaker"] and 
                  segment["start"] - current_segment["end"] <= max_gap_seconds and
                  len(current_segment["text"]) + len(segment["text"]) + 1 <= max_chars):  # +1 for the space
                # Same speaker, close enough in time, and won't exceed max character limit
                merged_segment = {
                    "start": current_segment["start"],
                    "end": segment["end"],
                    "speaker": current_segment["speaker"],
                    "text": current_segment["text"] + " " + segment["text"],
                }
                # Explicitly handle translation merging
                merged_translation = ""
                if "translation" in current_segment:
                    merged_translation += current_segment["translation"]
                if "translation" in segment:
                    # Add space only if both parts exist and merged_translation is not empty
                    if merged_translation:
                        merged_translation += " "
                    merged_translation += segment["translation"]
                
                if merged_translation:
                     merged_segment["translation"] = merged_translation
                
                # Copy any other fields from the current segment, prioritizing existing values
                for key, value in current_segment.items():
                    if key not in merged_segment:
                        merged_segment[key] = value
                
                current_segment = merged_segment
            else:
                # Different speaker, too far apart, or max chars reached - add current segment and start a new one
                optimized_segments.append(current_segment)
                current_segment = segment.copy()
        
        # Add the last segment if we have one
        if current_segment is not None:
            optimized_segments.append(current_segment)
        
        # Log the optimization results
        original_count = len(segments)
        optimized_count = len(optimized_segments)
        reduction = original_count - optimized_count
        reduction_percent = (reduction / original_count) * 100 if original_count > 0 else 0
        
        logger.debug(f"Optimized segments: {original_count} → {optimized_count} ({reduction} merged, {reduction_percent:.1f}% reduction)")
        
        return optimized_segments
    
    def _chunk_transcription(self, segments: List[Dict]) -> List[Dict]:
        """
        Chunk the transcription into logical groups for translation.
        
        Args:
            segments: List of transcription segments with speaker information
            
        Returns:
            List of chunks, each containing segments and speaker information
        """
        # Group segments by speaker as much as possible
        max_chunk_size = 10  # Maximum segments per chunk
        max_speaker_switch = 3  # Maximum number of speaker switches per chunk
        
        chunks = []
        current_chunk = []
        current_speakers = set()
        
        for segment in segments:
            # Check if adding this segment would exceed our limits
            would_exceed_size = len(current_chunk) >= max_chunk_size
            
            # Count potential speaker switches if we add this segment
            speaker_switches = 0
            if current_chunk:
                previous_speakers = [s["speaker"] for s in current_chunk]
                for i in range(1, len(previous_speakers)):
                    if previous_speakers[i] != previous_speakers[i-1]:
                        speaker_switches += 1
                
                # Check if adding this segment would create a new switch
                if segment["speaker"] != current_chunk[-1]["speaker"]:
                    speaker_switches += 1
            
            would_exceed_switches = speaker_switches > max_speaker_switch
            
            # Start a new chunk if needed
            if would_exceed_size or would_exceed_switches:
                if current_chunk:
                    chunks.append({
                        "segments": current_chunk.copy(),
                        "start": current_chunk[0]["start"],
                        "end": current_chunk[-1]["end"],
                        "speakers": list(current_speakers)
                    })
                    current_chunk = []
                    current_speakers = set()
            
            # Add segment to current chunk
            current_chunk.append(segment)
            current_speakers.add(segment["speaker"])
            
        # Add any remaining segments
        if current_chunk:
            chunks.append({
                "segments": current_chunk,
                "start": current_chunk[0]["start"],
                "end": current_chunk[-1]["end"],
                "speakers": list(current_speakers)
            })
        
        logger.debug(f"Split transcript into {len(chunks)} chunks for translation")
        
        return chunks
    
    def _translate_chunks(
        self,
        chunks: List[Dict],
        context_info: Dict[str, Any],
        source_language: str,
        target_language: str,
        source_summary: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Translate chunks with context awareness.
        
        Args:
            chunks: List of transcript chunks
            context_info: Dictionary with context information
            source_language: Source language code
            target_language: Target language code
            source_summary: Optional initial summary in source language
            **kwargs: Additional parameters
            
        Returns:
            List of chunks with translations added
        """
        # Enable debug mode if specified in kwargs, overriding instance setting
        debug = kwargs.get("debug", self.debug)
        debug_dir = kwargs.get("debug_dir", "artifacts/debug/translation")
        
        # Get cache settings from kwargs, falling back to instance settings
        enable_cache = kwargs.get("enable_cache", self.enable_cache)
        cache_stats = {"hits": 0, "misses": 0}
        
        # Create debug directory if needed
        if debug:
            # Create debug directory
            Path(debug_dir).mkdir(parents=True, exist_ok=True)

            # Create a timestamp for this translation session
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(debug_dir, f"session_{timestamp}")
            os.makedirs(session_dir)
            
            logger.debug(f"Debug mode enabled. Writing debug info to {session_dir}")
        else:
            session_dir = None
        
        # Preprocess all chunks to prepare their text representation
        for chunk in chunks:
            # Construct chunk text in speaker:text format
            chunk_text = ""
            original_speaker_texts = []
            
            for segment in chunk["segments"]:
                speaker = segment["speaker"]
                text = segment["text"]
                chunk_text += f"{speaker}: {text}\n"
                original_speaker_texts.append({"speaker": speaker, "text": text})
            
            chunk_text = chunk_text.strip()
            
            # Store the original text in the chunk for reference
            chunk["text"] = chunk_text
            chunk["original_speaker_texts"] = original_speaker_texts
        
        for i, chunk in enumerate(chunks):
            self._translate_single_chunk(
                chunk=chunk,
                i=i,
                chunks=chunks,
                context_info=context_info,
                source_language=source_language,
                target_language=target_language,
                source_summary=source_summary,  # Pass the source language summary
                enable_cache=enable_cache,
                debug=debug,
                session_dir=session_dir,
                cache_stats=cache_stats
            )
            
            logger.info(f"Translated chunk {i+1}/{len(chunks)} in {chunk.get('execution_time', 0):.2f}s")
            
        # Print cache statistics
        if enable_cache:
            self._save_cache()
            
        return chunks
        
    def _translate_single_chunk(
        self,
        chunk: Dict,
        i: int,
        chunks: List[Dict],
        context_info: Dict[str, Any],
        source_language: str,
        target_language: str,
        source_summary: Optional[str] = None,
        enable_cache: bool = True,
        debug: bool = False,
        session_dir: Optional[str] = None,
        cache_stats: Dict[str, int] = None,
        recursion_depth: int = 0
    ) -> None:
        """
        Translate a single chunk with context awareness.
        
        Args:
            chunk: The chunk to translate
            i: Index of the chunk in the chunks list
            chunks: List of all transcript chunks
            context_info: Dictionary with context information
            source_language: Source language code
            target_language: Target language code
            source_summary: Optional initial summary in source language
            enable_cache: Whether to use cache
            debug: Debug mode flag
            session_dir: Debug session directory
            cache_stats: Dictionary to track cache hits/misses
            recursion_depth: Recursion depth for splitting chunks on failure
            
        Returns:
            None - The chunk is modified in place
        """
        # Prevent infinite recursion
        max_recursion_depth = 4
        if recursion_depth > max_recursion_depth:
            logger.error(f"Max recursion depth reached ({recursion_depth}), using original text")
            # It needs to set 'translated_pairs' here as well for the caller
            chunk["translated_pairs"] = chunk["original_speaker_texts"]
            raise Exception("Max recursion depth reached")
            
        if cache_stats is None:
            cache_stats = {"hits": 0, "misses": 0}
            
        chunk_text = chunk["text"]
        original_speaker_texts = chunk["original_speaker_texts"]
        
        # Check cache first if enabled
        if enable_cache:
            cache_key = self._generate_cache_key(chunk_text, source_language, target_language)
            cached_result = self.translation_cache.get(cache_key)
            
            if cached_result:
                cache_stats["hits"] += 1
                logger.debug(f"Cache hit for chunk {i+1}: using cached translation")
                
                # Apply cached translation data to the chunk
                chunk["translated_pairs"] = cached_result["translated_pairs"]
                chunk["translation"] = cached_result["translation"]
                chunk["execution_time"] = 0.0  # No execution time for cache hits
                
                # Skip rest of processing since we have a cache hit
                return
            else:
                cache_stats["misses"] += 1
        
        # Get surrounding context
        context_before = chunks[i-2].get("translation", chunks[i-2]["text"]) if i > 1 else ""
        context_before += "\n" + (chunks[i-1].get("translation", chunks[i-1]["text"]) if i > 0 else "")

        context_after = chunks[i+1]["text"] if i < len(chunks)-1 else ""
        
        # Include source summary in the prompt if available
        summary_section = ""
        if source_summary:
            summary_section = f"""
# Initial content summary:
<initial_summary>
{source_summary}
</initial_summary>
"""

        # Add glossary section to the prompt if glossary exists
        glossary_section = ""
        if self.glossary:
            glossary_entries = "\n".join([f"- \"{term}\" → \"{translation}\"" for term, translation in self.glossary.items()])
            glossary_section = f"""
# Translation glossary (MUST be followed. Adapt for grammar):
<glossary>
{glossary_entries}
</glossary>

CRITICAL: You MUST use the translations from the glossary for all listed terms.
IMPORTANT: The glossary provides base forms of translations. When using a term from the glossary, you MUST adapt it to fit the grammatical context (e.g., case, gender, number, verb conjugation) of the sentence in the target language "{target_language}". For example, if the glossary says "cat" -> "кошка", and the sentence requires the genitive case, you should use "кошки", not "кошка". Do not just insert the glossary term verbatim if it violates grammatical rules.
"""
        
        # Optional additional context section for translation prompt
        custom_section = f"""
# Additional context (optional):
{self.prompt_prefix}
""" if self.prompt_prefix else ""

        # Build prompt with context information
        prompt = f"""
You are a professional translator specializing in {context_info['domain']} content.

Translate the following transcript of a conversation from '{source_language}' language to '{target_language}' language.

Preserve the meaning, tone, and style of the original.

# General rules:
1. Do not translate proper names, brand names, and abbreviations (e.g., LLM, API, etc.) - instead
2. Translatable terms: AI -> ИИ
3. Number and Date Conversion: Convert all digits and numbers to their written form in the target language as they would be naturally spoken aloud.
4. Remember this translation will be used for audio dubbing, so ensure the text flows naturally when spoken
5. Maintain the speaker identifiers exactly as given
6. Preserve the conversational flow and natural dialogue tone - don't make it sound too formal or robotic
7. Keep the emotional tone of the original speech (excited, concerned, questioning, etc.)
8. Ensure NO details or nuances from the original text are lost in translation - preserve
9. Pay special attention to {context_info['domain']} terminology. all information, examples, technical concepts, and specific details accurately.
10. Preserve the original structure of the conversation, number of lines, and number of speakers.

{glossary_section}
{custom_section}

# Special handling for filler words and conciseness:
1. Remove any filler words (e.g., 'um', 'uh', 'like', 'you know', 'Итак...' etc.) from the translation to make it sound more fluent and professional. 
2. When removing filler words results in a significantly shorter translation, use the freed-up space to:
   - Expand on technical concepts for better clarity
   - Add natural connecting phrases to improve flow
   - Provide slightly more context where the original might be too terse
   - Make implicit ideas more explicit when appropriate
3. The goal is a natural-sounding translation that conveys the full meaning, not just a direct word-for-word conversion
4. Balance conciseness with comprehensiveness - the translation should be clear and complete

# When you detect humor, jokes, puns, or wordplay:
1. Try to preserve the humor in the target language
2. If a direct translation would lose the humor, adapt it to an equivalent joke in the target language
3. If a cultural reference wouldn't make sense, replace it with a similar reference that would be understood by speakers of the target language
4. For wordplay that can't be directly translated, focus on preserving the comedic effect rather than the exact words

# Length considerations for audio dubbing:
1. Try to maintain a similar length between the original and translated text
2. This is crucial for audio dubbing, as the translated speech needs to fit within the same time constraints as the original
3. If the translation would naturally be much longer:
   - Look for more concise ways to express the same ideas
   - Remove redundancies while preserving all information
   - Use more compact phrasing where possible
4. If the translation would naturally be much shorter:
   - Add natural filler phrases that enhance clarity
   - Expand slightly on concepts where appropriate
   - Use more descriptive language while maintaining the original meaning
5. The goal is to have the translated audio match the timing of the original speech as closely as possible

# Translation considerations:
- Domain: {context_info['domain']}
- Tone: {context_info['tone']}
- Key themes: {', '.join(context_info['themes'])}
- Technical terms: {', '.join(context_info['terminology'])}
{summary_section}

# Following is the context of the conversation:

Context before:
<context_before>
{context_before}
</context_before>

Text to translate:
<text_to_translate>
{chunk_text}
</text_to_translate>

Context after:
<context_after>
{context_after}
</context_after>

CRITICAL: Output tanslation should contain same number of rows and original speaker names. If phrase is not translatable, leave blank.

IMPORTANT: Respond in JSON format with an array of objects containing speaker and translated text:
{{
    "translations": [
        {{"speaker": "SPEAKER_00", "text": "translated text 1"}},
        {{"speaker": "SPEAKER_01", "text": "translated text 2"}},
        ...
    ]
}}
"""

        # Start timer for this chunk's translation
        chunk_start_time = time.perf_counter()
        
        # Call LLM for translation
        max_attempts = 5
        translated_pairs = None
        translation_text = ""
        translation_success = False
        
        for attempt in range(max_attempts):
            try:
                translation = self.llm.complete(prompt)
                if hasattr(translation, "text"):
                    translation_text = translation.text.strip()
                else:
                    translation_text = str(translation).strip()
                
                # Check if translation is empty
                if not translation_text:
                    if attempt < max_attempts - 1:
                        logger.warning(f"Empty translation received, retrying ({attempt+1}/{max_attempts})...")
                        if debug:
                            self._write_attempt_debug(session_dir, i, attempt, chunks, prompt, translation_text, None, 
                                                     chunk_text, f"Empty translation received")
                        continue
                    else:
                        # Will handle failure after loop
                        logger.error(f"Failed to get translation after {max_attempts} attempts, empty response")
                        break  # Will trigger the fallback splitting mechanism
                
                try:
                    # Try to parse JSON from LLM response
                    repaired_json = json_repair.loads(translation_text)
                            
                    translated_pairs = repaired_json.get("translations", [])
                        
                    # Validate all translations match the target language
                    validation_errors = []
                    has_invalid_translations = False
                    
                    # Validate number of pairs matches original
                    if len(translated_pairs) != len(original_speaker_texts):
                        error_msg = f"Count mismatch: {len(translated_pairs)} translations vs {len(original_speaker_texts)} original lines"
                        validation_errors.append(error_msg)
                        logger.error(error_msg)
                        has_invalid_translations = True
                    
                    if not has_invalid_translations:
                        # Validate speakers match in the same order
                        speaker_mismatch_count = 0
                        for j, (orig, trans) in enumerate(zip(original_speaker_texts, translated_pairs)):
                            if j >= len(translated_pairs):
                                break
                            if orig["speaker"] != trans.get("speaker"):
                                speaker_mismatch_count += 1
                                if speaker_mismatch_count > 0: #FIXME: remove this check
                                    error_msg = f"Too many speaker mismatches: expected max 1, found {speaker_mismatch_count}"
                                    validation_errors.append(error_msg)
                                    logger.error(error_msg)
                                    has_invalid_translations = True
                                    break
                                else:
                                    # Allow one speaker mismatch, but record it
                                    logger.warning(f"Speaker correction at line {j+1}: replacing '{orig['speaker']}' with '{trans.get('speaker')}'")
                                    # Update the original speaker with the LLM-assigned speaker
                                    original_speaker_texts[j]["speaker"] = trans.get("speaker")
                                                
                    if has_invalid_translations:
                        if debug:
                            self._write_attempt_debug(session_dir, i, attempt, chunks, prompt, translation_text, 
                                                   translated_pairs, chunk_text, "\n".join(validation_errors))
                        
                        if attempt < max_attempts - 1:
                            logger.warning(f"Invalid translation, retrying ({attempt+1}/{max_attempts})...")
                            continue
                        else:
                            # Will handle failure after loop
                            break
                    
                    # If we reached here, translation was successful
                    translation_success = True
                    break
                except Exception as exc:
                    if debug:
                        self._write_attempt_debug(session_dir, i, attempt, chunks, prompt, translation_text, None, 
                                               chunk_text, f"Exception: {str(exc)}")
                    
                    if attempt < max_attempts - 1:
                        logger.warning(f"Chunk translation error: {str(exc)}, retrying ({attempt+1}/{max_attempts})...")
                        continue
                    else:
                        # Will handle failure after loop
                        logger.error(f"Failed to translate chunk: {str(exc)}")
                        break
            
            except Exception as e:
                if debug:
                    self._write_attempt_debug(session_dir, i, attempt, chunks, prompt, "API ERROR", None, 
                                           chunk_text, f"API Error: {str(e)}")
                
                if attempt < max_attempts - 1:
                    logger.warning(f"API error: {str(e)}, retrying ({attempt+1}/{max_attempts})...")
                else:
                    # Will handle failure after loop
                    logger.error(f"Failed to get translation after {max_attempts} attempts due to API errors")
                    break
        
        # If translation failed and we have enough content to split, use the fallback splitting mechanism
        if not translation_success and len(original_speaker_texts) > 1 and recursion_depth < max_recursion_depth:
            logger.warning(f"Translation failed for chunk {i+1}. Splitting chunk and retrying...")
            
            # Create two half-chunks
            mid_idx = len(original_speaker_texts) // 2
            
            first_half = {
                "text": "\n".join([f"{pair['speaker']}: {pair['text']}" for pair in original_speaker_texts[:mid_idx]]),
                "original_speaker_texts": original_speaker_texts[:mid_idx],
                "segments": chunk.get("segments", [])[:mid_idx] if "segments" in chunk else []
            }
            
            second_half = {
                "text": "\n".join([f"{pair['speaker']}: {pair['text']}" for pair in original_speaker_texts[mid_idx:]]),
                "original_speaker_texts": original_speaker_texts[mid_idx:],
                "segments": chunk.get("segments", [])[mid_idx:] if "segments" in chunk else []
            }
            
            # Translate each half
            logger.info(f"Translating first half ({len(first_half['original_speaker_texts'])} segments)...")
            self._translate_single_chunk(
                chunk=first_half,
                i=i,  # Use same index for context
                chunks=chunks,
                context_info=context_info,
                source_language=source_language,
                target_language=target_language,
                source_summary=source_summary,
                enable_cache=enable_cache,
                debug=debug,
                session_dir=session_dir,
                cache_stats=cache_stats,
                recursion_depth=recursion_depth + 1
            )
            
            logger.info(f"Translating second half ({len(second_half['original_speaker_texts'])} segments)...")
            self._translate_single_chunk(
                chunk=second_half,
                i=i,  # Use same index for context
                chunks=chunks,
                context_info=context_info,
                source_language=source_language,
                target_language=target_language,
                source_summary=source_summary,
                enable_cache=enable_cache,
                debug=debug,
                session_dir=session_dir,
                cache_stats=cache_stats,
                recursion_depth=recursion_depth + 1
            )
            
            translated_pairs = first_half["translated_pairs"] + second_half["translated_pairs"]
            
            # Create combined translation
            formatted_translation = "\n".join([f"{pair['speaker']}: {pair['text']}" for pair in translated_pairs])
            
            # Save combined results to original chunk
            chunk["translated_pairs"] = translated_pairs
            chunk["translation"] = formatted_translation
            
            # Calculate combined execution time
            chunk["execution_time"] = first_half.get("execution_time", 0) + second_half.get("execution_time", 0)
            
            # Log success
            logger.info(f"Successfully translated chunk {i+1} by splitting it into two parts")
            translation_success = True
        
        # If still not successful, use original text as fallback
        if not translation_success:
            translated_pairs = original_speaker_texts
            chunk["translated_pairs"] = translated_pairs
            
            # Create formatted version with original text
            formatted_translation = "\n".join([f"{pair['speaker']}: {pair['text']}" for pair in translated_pairs])
            chunk["translation"] = formatted_translation
            
            # Record end time for this chunk
            chunk_end_time = time.perf_counter()
            chunk_execution_time = chunk_end_time - chunk_start_time
            chunk["execution_time"] = chunk_execution_time
            
            raise Exception("Translation failed")
        
        # If we get here through normal successful translation (not split)
        if translation_success and translated_pairs:
            # Record end time for this chunk
            chunk_end_time = time.perf_counter()
            chunk_execution_time = chunk_end_time - chunk_start_time
            chunk["execution_time"] = chunk_execution_time

            # Store the translated pairs in the chunk
            chunk["translated_pairs"] = translated_pairs
            
            if recursion_depth > 0:
                return
                
            # Write debug information if debug mode is enabled
            if debug:
                # Write final successful result using the same debug helper function
                success_message = f"Successfully translated chunk in {chunk_execution_time:.2f} seconds"
                self._write_attempt_debug(
                    session_dir, 
                    i, 
                    "success", 
                    chunks, 
                    prompt, 
                    translation_text, 
                    translated_pairs, 
                    chunk_text, 
                    success_message
                )
                        
            # Also create a formatted version for compatibility with the rest of the pipeline
            formatted_translation = "\n".join([f"{pair['speaker']}: {pair['text']}" for pair in translated_pairs])
            chunk["translation"] = formatted_translation
            
            # Cache the successful translation if caching is enabled
            if enable_cache:
                cache_key = self._generate_cache_key(chunk_text, source_language, target_language)
                
                # Store translation in cache
                self.translation_cache[cache_key] = {
                    "translated_pairs": translated_pairs,
                    "translation": chunk["translation"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                self._save_cache()
    
    def _write_attempt_debug(self, session_dir, chunk_idx, attempt, chunks, prompt, response_text, parsed_result, 
                           original_text, error_message=None):
        """
        Write debug information for a translation attempt or final result.
        
        Args:
            session_dir: Debug directory path
            chunk_idx: Index of the current chunk
            attempt: Current attempt number or "success" for successful translations
            chunks: List of all chunks
            prompt: Prompt sent to the LLM
            response_text: Raw response from the LLM
            parsed_result: Parsed translation pairs (if successful)
            original_text: Original chunk text
            error_message: Any error message or success message to include
        """
        try:
            # Create a descriptive filename
            if attempt == "success":
                chunk_debug_file = os.path.join(session_dir, f"chunk_{chunk_idx+1}_success.txt")
                header = f"=== CHUNK {chunk_idx+1}/{len(chunks)} (FINAL RESULT) ==="
            else:
                chunk_debug_file = os.path.join(session_dir, f"chunk_{chunk_idx+1}_attempt_{attempt+1}.txt")
                header = f"=== CHUNK {chunk_idx+1}/{len(chunks)} (ATTEMPT {attempt+1}) ==="
            
            with open(chunk_debug_file, "w", encoding="utf-8") as f:
                f.write(f"{header}\n")
                
                if error_message:
                    f.write(f"=== {'SUCCESS' if attempt == 'success' else 'ERROR'} INFORMATION ===\n")
                    f.write(error_message)
                    f.write("\n\n")
                
                f.write("=== PROMPT ===\n")
                f.write(prompt)
                f.write("\n\n")
                
                f.write("=== RAW RESPONSE ===\n")
                f.write(response_text)
                f.write("\n\n")
                
                f.write("=== PARSED RESULT ===\n")
                if parsed_result:
                    for pair in parsed_result:
                        f.write(f"{pair.get('speaker', 'UNKNOWN')}: {pair.get('text', '')}\n")
                else:
                    f.write("Failed to parse valid translation pairs\n")
                
                f.write("\n=== ORIGINAL TEXT ===\n")
                f.write(original_text)
                
                # Add comparison if we have both original text and translation result
                if parsed_result and original_text:
                    f.write("\n\n=== SIDE-BY-SIDE COMPARISON ===\n")
                    original_pairs = []
                    for line in original_text.split('\n'):
                        if ':' in line:
                            speaker, text = line.split(':', 1)
                            original_pairs.append({"speaker": speaker.strip(), "text": text.strip()})
                    
                    # Only show comparison if we have matching counts
                    if len(original_pairs) == len(parsed_result):
                        for i, (orig, trans) in enumerate(zip(original_pairs, parsed_result)):
                            f.write(f"Line {i+1}:\n")
                            f.write(f"  Original ({orig['speaker']}): {orig['text']}\n")
                            f.write(f"  Translated ({trans.get('speaker', 'UNKNOWN')}): {trans.get('text', '')}\n\n")
            
            description = "success" if attempt == "success" else f"attempt {attempt+1}"
            logger.debug(f"Wrote debug info for chunk {chunk_idx+1} {description} to {chunk_debug_file}")
        except Exception as e:
            logger.error(f"Error writing debug file for chunk {chunk_idx+1}: {str(e)}")
    
    def _refine_translation(
        self,
        translated_chunks: List[Dict],
        context_info: Dict[str, Any],
        source_language: str,
        target_language: str,
        dialogue_summary: Optional[str] = None,
        refinement_persona: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Refine the entire translated conversation for better flow and consistency.
        Processes chunks in batches to stay within 32k token limits.

        Args:
            translated_chunks: List of chunks with initial translations
            context_info: Dictionary with context information
            source_language: Source language code
            target_language: Target language code
            dialogue_summary: Summary of the entire dialogue for context
            refinement_persona: The persona to use for the refinement prompt (overrides instance setting)
            **kwargs: Additional parameters (e.g., debug)

        Returns:
            List of chunks with refined translations and shorter alternatives
        """
        # Enable debug mode if specified in kwargs, overriding instance setting
        debug = kwargs.get("debug", self.debug)
        debug_dir = kwargs.get("debug_dir", "artifacts/debug/translation_refinement")
        session_dir = None # Initialize session_dir
        
        # Determine which persona to use
        persona = refinement_persona or self.refinement_persona
        if persona not in REFINEMENT_PROMPTS:
            logger.warning(f"Warning: Persona '{persona}' not found. Defaulting to 'manager'.")
            persona = "manager"
        
        # Add a depth parameter to track recursion depth for split operations
        depth = kwargs.get("_depth", 0)
        # Maximum recursion depth to prevent infinite splits
        max_depth = kwargs.get("max_depth", 2)
        # Maximum number of chunks to process in a single batch (to stay under 32k tokens)
        max_batch_size = kwargs.get("max_batch_size", 1)

        # Store all comparison data for the final report
        all_refinement_comparisons = []

        # Create debug directory if needed
        if debug:
            # Create debug directory
            Path(debug_dir).mkdir(parents=True, exist_ok=True)

            # Create a timestamp for this refinement session
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(debug_dir, f"refinement_session_{timestamp}_depth{depth}")
            os.makedirs(session_dir, exist_ok=True) # Ensure the session-specific directory exists

            logger.debug(f"Refinement Debug mode enabled. Writing debug info to {session_dir}")

        # For very small inputs, just return the original to avoid unnecessary processing
        if len(translated_chunks) <= 1:
            return translated_chunks
            
        # Start timer for total refinement process
        total_start_time = time.perf_counter()
        
        # Process chunks in batches to stay within token limits
        refined_chunks = []
        
        # Split chunks into batches
        batches = []
        current_batch = []
        
        for chunk in translated_chunks:
            current_batch.append(chunk)
            
            # Start a new batch if we've reached the max batch size
            if len(current_batch) >= max_batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # Add any remaining chunks to the last batch
        if current_batch:
            batches.append(current_batch)
            
        logger.debug(f"Split {len(translated_chunks)} chunks into {len(batches)} batches for refinement")
        
        # Add glossary section to the refinement prompt if glossary exists
        glossary_section = ""
        if self.glossary:
            glossary_entries = "\n".join([f"- \"{term}\" → \"{translation}\"" for term, translation in self.glossary.items()])
            glossary_section = f"""
# Translation glossary (MUST be followed. Adapt for grammar):
<glossary>
{glossary_entries}
</glossary>

CRITICAL: You MUST use the translations from the glossary for all listed terms.
IMPORTANT: The glossary provides base forms of translations. When using a term from the glossary, you MUST adapt it to fit the grammatical context (e.g., case, gender, number, verb conjugation) of the sentence in the target language "{target_language}". For example, if the glossary says "cat" -> "кошка", and the sentence requires the genitive case, you should use "кошки", not "кошка". Do not just insert the glossary term verbatim if it violates grammatical rules.
"""
        
        # Process each batch separately
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Refining and rephrasing translation chunk {batch_idx + 1}/{len(batches)}...")
            
            # Combine all translated pairs in this batch into a single list for the prompt
            all_translated_pairs = [pair for chunk in batch for pair in chunk.get("translated_pairs", [])]
            
            # Extract original text to create original-translation pairs
            all_original_texts = []
            idx = 0
            
            for chunk in batch:
                for segment in chunk.get("segments", []):
                    if idx < len(all_translated_pairs):
                        all_original_texts.append({
                            "speaker": segment["speaker"],
                            "text": segment["text"]
                        })
                        idx += 1
            
            # Get previous chunk context (from the previous batch if needed)
            previous_chunk_context = ""
            if batch_idx > 0:
                # Get the last chunk from the previous batch
                previous_batch = batches[batch_idx - 1]
                if previous_batch:
                    previous_chunk = previous_batch[-1]  # Last chunk of previous batch
                    if "translation" in previous_chunk:
                        previous_chunk_context = f"# Context from previous segment:\n{previous_chunk['translation']}\n\n"

            # Get next chunk context (from the next batch if available)
            next_chunk_context = ""
            if batch_idx < len(batches) - 1:
                # Get the first chunk from the next batch
                next_batch = batches[batch_idx + 1]
                if next_batch:
                    next_chunk = next_batch[0]  # First chunk of next batch
                    if "text" in next_chunk:
                        next_chunk_context = f"# Context from next segment:\n{next_chunk['text']}\n\n"
            
            # Create paired conversation format with original and translation
            original_conversation_text = ""
            for orig in all_original_texts:
                original_conversation_text += f"{orig['speaker']}: {orig['text']}\n"

            translated_conversation_text = ""
            for trans in all_translated_pairs:
                translated_conversation_text += f"{trans['speaker']}: {trans['text']}\n"

                
            # Build the refinement prompt
            base_prompt = REFINEMENT_PROMPTS[persona]
            
            refinement_prompt = base_prompt.format(
                dialogue_summary=dialogue_summary or "No summary available.",
                glossary_section=glossary_section,
                domain=context_info.get('domain', 'general'),
                tone=context_info.get('tone', 'neutral'),
                themes=', '.join(context_info.get('themes', [])),
                terminology=', '.join(context_info.get('terminology', [])),
                previous_chunk_context=previous_chunk_context,
                original_conversation_text=original_conversation_text,
                next_chunk_context=next_chunk_context,
                translated_conversation_text=translated_conversation_text,
                source_language=source_language,
                target_language=target_language
            )

            # Start timer for batch refinement
            batch_start_time = time.perf_counter()

            # Call LLM for refinement
            max_attempts = 5
            refined_pairs = None
            llm_response_text = ""
            refinement_success = False  # Track if refinement succeeded

            for attempt in range(max_attempts):
                try:
                    # Prepend custom context if provided
                    effective_refinement_prompt = (
                        f"# Additional context (optional):\n{self.prompt_prefix}\n\n{refinement_prompt}"
                        if self.prompt_prefix else refinement_prompt
                    )
                    refinement_response = self.refinement_llm.complete(effective_refinement_prompt)
                    if hasattr(refinement_response, "text"):
                        llm_response_text = refinement_response.text.strip()
                    else: # openrouter
                        llm_response_text = str(refinement_response).strip()

                    if not llm_response_text:
                        if attempt < max_attempts - 1:
                            logger.warning(f"Empty refinement response received, retrying ({attempt+1}/{max_attempts})...")
                            continue
                        else:
                            logger.error(f"Failed to get refinement after {max_attempts} attempts (empty response).")
                            break  # Will trigger the fallback splitting mechanism
                    
                    try:
                        # Parse JSON from LLM response
                        repaired_json = json_repair.loads(llm_response_text)
                        refined_pairs = repaired_json.get("translations", [])

                        # Basic validation
                        if not refined_pairs or not isinstance(refined_pairs, list):
                            raise ValueError("Invalid or empty 'translations' array in response.")
                        
                        if len(refined_pairs) != len(all_translated_pairs):
                            raise ValueError(f"Refined pairs count ({len(refined_pairs)}) does not match original pairs count ({len(all_translated_pairs)}).")

                        # Validate individual pairs
                        has_invalid_pairs = False
                        for i, pair in enumerate(refined_pairs):
                            if not isinstance(pair, dict) or "speaker" not in pair or "text" not in pair:
                                logger.warning(f"Invalid refined pair structure at index {i}: {pair}")
                                has_invalid_pairs = True
                                # Attempt to fix by using original data
                                refined_pairs[i] = all_translated_pairs[i] 
                            elif pair["speaker"] != all_translated_pairs[i]["speaker"]:
                                logger.warning(f"Speaker mismatch at index {i}. Expected {all_translated_pairs[i]['speaker']}, got {pair['speaker']}. Correcting.")
                                refined_pairs[i]["speaker"] = all_translated_pairs[i]["speaker"] # Enforce original speaker
                                has_invalid_pairs = True # Consider it invalid for retry logic
                                
                            # Validate shorter versions exist, or create them
                            if "short" not in pair or not pair["short"]:
                                logger.debug(f"Missing 'short' version at index {i}. Using main text as fallback.")
                                refined_pairs[i]["short"] = pair.get("text", "")
                                has_invalid_pairs = True

                            if "long" not in pair or not pair["long"]:
                                logger.debug(f"Missing 'long' version at index {i}. Using main text as fallback.")
                                refined_pairs[i]["long"] = pair.get("text", "")
                                has_invalid_pairs = True
                            
                        if has_invalid_pairs and attempt < max_attempts - 1:
                            logger.warning(f"Invalid pairs found in refinement, retrying ({attempt+1}/{max_attempts})...")
                            continue

                        # If we get here, refinement was successful
                        refinement_success = True
                        break # Success

                    except Exception as json_error:
                        if attempt < max_attempts - 1:
                            logger.warning(f"Refinement JSON parsing error: {str(json_error)}, retrying ({attempt+1}/{max_attempts})...")
                            # Optionally log the problematic response text
                            if debug and session_dir:
                                try:
                                    error_file = os.path.join(session_dir, f"batch{batch_idx+1}_refinement_error_attempt_{attempt+1}.txt")
                                    with open(error_file, "w", encoding="utf-8") as f_err:
                                        f_err.write("=== ERROR ===\n")
                                        f_err.write(str(json_error))
                                        f_err.write("\n\n=== LLM RESPONSE ===\n")
                                        f_err.write(llm_response_text)
                                except Exception as log_e:
                                    logger.error(f"Error writing refinement error log: {log_e}")
                            continue
                        else:
                            logger.error(f"Failed to parse refinement JSON after {max_attempts} attempts: {str(json_error)}.")
                            break  # Will trigger the fallback splitting mechanism

                except Exception as e:
                    if attempt < max_attempts - 1:
                        logger.warning(f"Refinement API error: {str(e)}, retrying ({attempt+1}/{max_attempts})...")
                    else:
                        logger.error(f"Failed to get refinement after {max_attempts} attempts due to API errors.")
                        break  # Will trigger the fallback splitting mechanism
            
            # If refinement failed for this batch and it has more than one chunk, try splitting the batch
            if not refinement_success and len(batch) > 1 and depth < max_depth:
                logger.warning(f"Refinement failed for batch {batch_idx+1} at depth {depth}. Splitting batch and retrying...")
                
                # Calculate midpoint
                mid = len(batch) // 2
                
                # Split batch into two halves
                first_half = batch[:mid]
                second_half = batch[mid:]
                
                # Refine each half separately with incremented depth
                logger.info(f"Refining first half of batch {batch_idx+1} ({len(first_half)} chunks)...")
                refined_first_half = self._refine_translation(
                    translated_chunks=first_half,
                    context_info=context_info,
                    source_language=source_language,
                    target_language=target_language,
                    dialogue_summary=dialogue_summary,
                    refinement_persona=persona,
                    _depth=depth+1,
                    max_depth=max_depth,
                    max_batch_size=max_batch_size,
                    **kwargs
                )
                
                logger.info(f"Refining second half of batch {batch_idx+1} ({len(second_half)} chunks)...")
                refined_second_half = self._refine_translation(
                    translated_chunks=second_half,
                    context_info=context_info,
                    source_language=source_language,
                    target_language=target_language,
                    dialogue_summary=dialogue_summary,
                    refinement_persona=persona,
                    _depth=depth+1,
                    max_depth=max_depth,
                    max_batch_size=max_batch_size,
                    **kwargs
                )
                
                # Add the results of the recursive calls
                refined_chunks.extend(refined_first_half)
                refined_chunks.extend(refined_second_half)
                
                # Skip the rest of the loop for this batch
                continue
                
            # If refinement still failed or we've hit max depth, use the original chunks for this batch
            if not refinement_success:
                logger.error(f"Refinement failed for batch {batch_idx+1}. Using original translations.")
                refined_chunks.extend(batch)
                continue

            # Record end time for batch refinement
            batch_end_time = time.perf_counter()
            batch_execution_time = batch_end_time - batch_start_time
            logger.debug(f"Completed batch {batch_idx+1} refinement in {batch_execution_time:.2f} seconds")

            # Store comparison data for the comprehensive report
            if refined_pairs and len(refined_pairs) == len(all_translated_pairs):
                batch_comparisons = []
                for i, (orig, refined) in enumerate(zip(all_translated_pairs, refined_pairs)):
                    batch_comparisons.append({
                        "line_num": i+1,
                        "original": {"speaker": orig['speaker'], "text": orig['text']},
                        "refined": {
                            "speaker": refined['speaker'], 
                            "text": refined['text'],
                            "short": refined.get('short', refined['text']),
                            "long": refined.get('long', refined['text'])
                        }
                    })
                all_refinement_comparisons.extend(batch_comparisons)

            # Write debug information if debug mode is enabled
            if debug and session_dir:
                try:
                    refinement_debug_file = os.path.join(session_dir, f"batch{batch_idx+1}_refinement_log.txt")
                    with open(refinement_debug_file, "w", encoding="utf-8") as f:
                        f.write(f"=== BATCH {batch_idx+1} REFINEMENT PROCESS ===\n")
                        f.write(f"Execution time: {batch_execution_time:.2f} seconds\n\n")
                        
                        f.write("=== REFINEMENT PROMPT ===\n")
                        f.write(refinement_prompt)
                        f.write("\n\n")
                        
                        f.write("=== LLM RESPONSE ===\n")
                        f.write(llm_response_text)
                        f.write("\n\n")
                        
                        f.write("=== PARSED REFINED RESULT ===\n")
                        if refined_pairs:
                            for pair in refined_pairs:
                                f.write(f"{pair.get('speaker', 'UNKNOWN')}: {pair.get('text', '')}\n")
                                if "very_short" in pair:
                                    f.write(f"  VERY SHORT: {pair['very_short']}\n")
                                if "short" in pair:
                                    f.write(f"  SHORT: {pair['short']}\n")
                                if "long" in pair:
                                    f.write(f"  LONG: {pair['long']}\n")
                                f.write("\n")
                        else:
                            f.write("Failed to parse valid refined pairs\n")
                        
                        # Add side-by-side comparison between original translations and refined translations
                        if refined_pairs and len(refined_pairs) == len(all_translated_pairs):
                            f.write("\n\n=== SIDE-BY-SIDE COMPARISON ===\n")
                            for i, (orig, refined) in enumerate(zip(all_translated_pairs, refined_pairs)):
                                f.write(f"Line {i+1}:\n")
                                f.write(f"  Original ({orig['speaker']}): {orig['text']}\n")
                                f.write(f"  Refined ({refined['speaker']}): {refined['text']}\n")
                                if "very_short" in refined:
                                    f.write(f"  Very short: {refined['very_short']}\n")
                                if "short" in refined:
                                    f.write(f"  Short: {refined['short']}\n")
                                if "long" in refined:
                                    f.write(f"  Long: {refined['long']}\n")
                                f.write("\n")
                        
                    logger.debug(f"Wrote refinement debug info to {refinement_debug_file}")
                except Exception as e:
                    logger.error(f"Error writing refinement debug file: {str(e)}")

            # Now, redistribute the refined pairs back into the chunk structure for this batch
            batch_refined_chunks = []
            current_pair_index = 0
            for chunk in batch:
                num_segments_in_chunk = len(chunk.get("segments", [])) # Use original segment count
                if num_segments_in_chunk == 0 and "translated_pairs" in chunk:
                    num_segments_in_chunk = len(chunk["translated_pairs"]) # Fallback if segments are missing

                if current_pair_index + num_segments_in_chunk > len(refined_pairs):
                    logger.error(f"Error: Not enough refined pairs to redistribute into chunk structure. Skipping chunk.")
                    # Optionally handle this error more gracefully, e.g., reuse original chunk data
                    batch_refined_chunks.append(chunk) # Append original chunk as fallback
                    continue

                # Slice the refined pairs corresponding to this chunk
                chunk_refined_pairs = refined_pairs[current_pair_index : current_pair_index + num_segments_in_chunk]
                
                # Create a new chunk dictionary with refined data
                refined_chunk = chunk.copy() # Start with original chunk data
                refined_chunk["translated_pairs"] = chunk_refined_pairs
                
                # Update the 'translation' string representation as well
                formatted_translation = "\n".join([f"{pair['speaker']}: {pair['text']}" for pair in chunk_refined_pairs])
                refined_chunk["translation"] = formatted_translation
                
                # Add alternative versions in separate format for easy access
                formatted_very_short_translation = "\n".join([f"{pair['speaker']}: {pair.get('very_short', pair['text'])}" for pair in chunk_refined_pairs])
                refined_chunk["very_short_translation"] = formatted_very_short_translation
                
                formatted_short_translation = "\n".join([f"{pair['speaker']}: {pair.get('short', pair['text'])}" for pair in chunk_refined_pairs])
                refined_chunk["short_translation"] = formatted_short_translation
                
                # Add long version
                formatted_long_translation = "\n".join([f"{pair['speaker']}: {pair.get('long', pair['text'])}" for pair in chunk_refined_pairs])
                refined_chunk["long_translation"] = formatted_long_translation
                                
                batch_refined_chunks.append(refined_chunk)
                current_pair_index += num_segments_in_chunk
                
            # Add refined chunks from this batch to the overall results
            refined_chunks.extend(batch_refined_chunks)

        # Final check: Ensure the number of output chunks matches input
        if len(refined_chunks) != len(translated_chunks):
            logger.warning(f"Warning: Refined chunk count ({len(refined_chunks)}) differs from original ({len(translated_chunks)}). This might indicate an issue.")
            # Fallback to original chunks if structure is broken
            return translated_chunks
            
        # Record end time for total refinement process
        total_end_time = time.perf_counter()
        total_execution_time = total_end_time - total_start_time
        logger.info(f"Completed entire translation refinement at depth {depth} in {total_execution_time:.2f} seconds")

        # Write the comprehensive comparison report with all batches
        if debug and session_dir and all_refinement_comparisons and depth == 0:
            try:
                comprehensive_report_file = os.path.join(session_dir, "comprehensive_refinement_comparison.txt")
                with open(comprehensive_report_file, "w", encoding="utf-8") as f:
                    f.write(f"=== COMPREHENSIVE REFINEMENT COMPARISON REPORT ===\n")
                    f.write(f"Total execution time: {total_execution_time:.2f} seconds\n")
                    f.write(f"Processed {len(translated_chunks)} chunks in {len(batches)} batches\n\n")
                    
                    f.write("=== SIDE-BY-SIDE COMPARISON ===\n")
                    for item in all_refinement_comparisons:
                        f.write(f"Line {item['line_num']}:\n")
                        f.write(f"  Original ({item['original']['speaker']}): {item['original']['text']}\n")
                        f.write(f"  Refined ({item['refined']['speaker']}): {item['refined']['text']}\n")
                        if "very_short" in item['refined']:
                            f.write(f"  Very short: {item['refined']['very_short']}\n")
                        if "short" in item['refined']:
                            f.write(f"  Short: {item['refined']['short']}\n")
                        if "long" in item['refined']:
                            f.write(f"  Long: {item['refined']['long']}\n")
                        f.write("\n")
                
                logger.info(f"Wrote comprehensive refinement comparison report to {comprehensive_report_file}")
            except Exception as e:
                logger.error(f"Error writing comprehensive comparison report: {str(e)}")

        return refined_chunks
    
    def _segment_translations(self, chunks: List[Dict]) -> List[Dict]:
        """
        Split translated chunks back into segment-level translations.
        
        Args:
            chunks: List of chunks with translations
            
        Returns:
            List of segments with individual translations
        """
        all_segments = []
        
        for chunk in chunks:
            original_segments = chunk["segments"]
            translated_pairs = chunk["translated_pairs"]
            
            # Handle mismatch without complex alignment algorithm
            if len(original_segments) != len(translated_pairs):
                raise ValueError(f"Segment count mismatch: {len(original_segments)} original segments, {len(translated_pairs)} translations")
            else:
                # When counts match, use direct mapping
                for i, segment in enumerate(original_segments):
                    segment["translation"] = translated_pairs[i]["text"]
                    # Add alternative versions if available
                    if "very_short" in translated_pairs[i]:
                        segment["very_short_translation"] = translated_pairs[i]["very_short"]
                    if "short" in translated_pairs[i]:
                        segment["short_translation"] = translated_pairs[i]["short"]
                    if "long" in translated_pairs[i]:
                        segment["long_translation"] = translated_pairs[i]["long"]
                    # Use the potentially modified speaker from translated_pairs
                    # This preserves any allowed speaker replacement that happened during validation
                    segment["speaker"] = translated_pairs[i]["speaker"]
                    all_segments.append(segment)
        
        # Filter out segments with empty translations (after filling with original text)
        # This step shouldn't actually remove any segments if we've filled them all above
        filtered_segments = [segment for segment in all_segments if segment.get("translation") and segment["translation"].strip() != ""]
        
        logger.debug(f"Generated {len(filtered_segments)} segment-level translations")
        
        return filtered_segments
        
    def is_available(self) -> bool:
        """Check if the LLM translator is available and properly initialized."""
        if self.llm_provider == "gemini":
            return GEMINI_AVAILABLE and self.llm is not None and self.refinement_llm is not None
        elif self.llm_provider == "openrouter":
            return OPENROUTER_AVAILABLE and self.llm is not None and self.refinement_llm is not None
        return False
        
    def _generate_timecodes_report(self, summary_json: Dict[str, Any], report_path: str = "artifacts/timecodes.txt") -> None:
        """
        Generate a human-readable report of dialogue chapters with timecodes.
        
        Args:
            summary_json: The structured summary JSON with chapters
            report_path: Path where to save the report (default: artifacts/timecodes.txt)
            
        Returns:
            None
        """
        logger.debug(f"Generating human-readable timecodes report...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                total_summary = summary_json.get("overall_summary")
                if total_summary:
                    f.write(f"{total_summary}\n\n")

                # Write each chapter in a human-readable format
                for chapter in summary_json.get("chapters", []):
                    title = chapter.get("title", "Untitled Chapter")
                    start_time = chapter.get("start_time", "00:00:00")
                    summary = chapter.get("summary", "No summary available")
                    
                    # Format time to remove hours if they're 00
                    if start_time.startswith("00:"):
                        display_time = start_time[3:]  # Remove "00:"
                    else:
                        display_time = start_time
                        
                    f.write(f"{display_time} - {title}\n")
                    f.write(f"{summary}\n\n")
            
            logger.info(f"Human-readable timecodes report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error writing timecodes report: {e}")

    def adjust_segment_text_length(
        self,
        original_text: str,
        source_language: str,
        target_language: str,
        desired_ratio: float,
        target_char_count: Optional[int] = None,
        context_info: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3,
    ) -> str:
        """Adjust a segment's text length using the refinement LLM.

        The method preserves meaning and tone while lengthening or shortening the
        text approximately by the provided ratio for dubbing alignment.

        Args:
            original_text: Text to rewrite in target language.
            source_language: Source language code.
            target_language: Target language code.
            desired_ratio: Desired relative length factor compared to input text.
            target_char_count: Optional target character count hint.
            context_info: Optional context with 'domain' and 'tone'.
            max_attempts: Number of retries on transient failures.

        Returns:
            Rewritten text in target language. Falls back to original_text on failure.
        """
        if not self.is_available():
            return original_text

        # Clamp ratio to reasonable bounds to avoid extreme prompts
        safe_ratio = max(0.2, min(desired_ratio, 2.0))
        approx_target_chars = target_char_count if target_char_count is not None else max(1, int(len(original_text) * safe_ratio))

        # Build glossary section if available
        glossary_section = ""
        if self.glossary:
            glossary_entries = "\n".join([f"- \"{term}\" → \"{translation}\"" for term, translation in self.glossary.items()])
            glossary_section = f"""
# Translation glossary (MUST be followed. Adapt for grammar):
<glossary>
{glossary_entries}
</glossary>

CRITICAL: You MUST use the translations from the glossary for all listed terms.
IMPORTANT: The glossary provides base forms of translations. When using a term from the glossary, you MUST adapt it to fit the grammatical context (e.g., case, gender, number, verb conjugation) of the sentence in the target language \"{target_language}\".
"""

        domain = (context_info or {}).get("domain", "general")
        tone = (context_info or {}).get("tone", "neutral")

        prompt = LENGTH_ADJUST_PROMPT.format(
            source_language=source_language,
            target_language=target_language,
            desired_ratio=safe_ratio,
            target_char_count=approx_target_chars,
            glossary_section=glossary_section,
            domain=domain,
            tone=tone,
            original_text=original_text,
        )

        response_text = ""
        for attempt in range(max_attempts):
            try:
                response = self.refinement_llm.complete(prompt)
                response_text = response.text.strip() if hasattr(response, "text") else str(response).strip()
                if not response_text:
                    continue

                try:
                    repaired = json_repair.loads(response_text)
                    adjusted = repaired.get("text") if isinstance(repaired, dict) else None
                    if adjusted and isinstance(adjusted, str) and adjusted.strip():
                        return adjusted.strip()
                except Exception:
                    # Try a naive fallback: if response looks like raw text without JSON
                    if response_text and response_text.lstrip().startswith("{") is False:
                        return response_text
            except Exception:
                # Retry on transient issues
                continue

        # Fallback
        return original_text
