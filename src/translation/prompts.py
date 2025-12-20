import os
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PersonaInfo:
    """Metadata about a persona loaded from file."""
    id: str
    name: str
    description: str
    prompt: str


def _load_persona_from_file(file_path: Path) -> PersonaInfo:
    """Load a single persona from a text file."""
    content = file_path.read_text(encoding='utf-8')
    
    # Split header and prompt by '---' separator
    parts = content.split('---', 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid persona file format: {file_path}")
    
    header = parts[0].strip()
    prompt = parts[1].strip()
    
    # Parse header
    name = ""
    description = ""
    for line in header.split('\n'):
        line = line.strip()
        if line.startswith('name:'):
            name = line[5:].strip()
        elif line.startswith('description:'):
            description = line[12:].strip()
    
    persona_id = file_path.stem
    return PersonaInfo(id=persona_id, name=name, description=description, prompt=prompt)


def _load_all_personas() -> dict[str, str]:
    """Load all persona prompts from the personas directory."""
    personas_dir = Path(__file__).parent / 'personas'
    prompts = {}
    
    if not personas_dir.exists():
        return prompts
    
    for file_path in sorted(personas_dir.glob('*.txt')):
        try:
            persona = _load_persona_from_file(file_path)
            prompts[persona.id] = persona.prompt
        except Exception as e:
            logger.warning(f"Failed to load persona from {file_path}: {e}")
    
    return prompts


def get_available_personas() -> list[PersonaInfo]:
    """Get list of all available personas with their metadata."""
    personas_dir = Path(__file__).parent / 'personas'
    personas = []
    
    if not personas_dir.exists():
        return personas
    
    for file_path in sorted(personas_dir.glob('*.txt')):
        try:
            persona = _load_persona_from_file(file_path)
            personas.append(persona)
        except Exception as e:
            logger.warning(f"Failed to load persona from {file_path}: {e}")
    
    return personas


REFINEMENT_PROMPTS = _load_all_personas()
# Prompt used to adjust a single segment's text length while preserving meaning.
# The LLM should rewrite the text in the target language so that the spoken length
# approximately follows the requested ratio relative to the original text.
LENGTH_ADJUST_PROMPT = """
You are an expert dialogue editor optimizing text for audio dubbing. Rewrite the given line from '{source_language}' into '{target_language}' while preserving meaning, tone, and speaker voice, but adjust its speaking length.

# Goal
- Adjust the speaking length to approximately match the requested relative length factor.
- Relative length factor: {desired_ratio:.2f}× of the original text length.
- Target character count (approximate): {target_char_count} characters.

{pause_markers_section}

# Adjustment Strategy
{lengthening_guidance}
{shortening_guidance}

**Orthography and Diacritics:** Apply correct target-language orthography and diacritics. For example: in Russian, prefer the 'ё' - 'yo' letter where standard usage requires (not the plain 'e'); preserve accents in Romance languages (e.g., é, è, ñ, ç); use umlauts and ß in German; respect dotted/dotless I rules in Turkish (İ/i vs I/ı). Do not strip diacritics; use language-appropriate casing.

# Constraints
- Preserve all critical facts, numbers, names, and technical terms.
- Convert digits and dates to spoken-form appropriate for '{target_language}'.
- Maintain the original speaker's intent, tone, and register.
- Keep it natural for dubbing (flowing speech, not robotic).
- Do not add new claims or technical details.
- Prefer correct orthography and diacritics for the target language.
{glossary_section}

# Context (optional)
- Domain: {domain}
- Tone: {tone}
- Key themes: {themes}
- Technical terms: {terminology}

# Persona-specific constraints
{persona_requirements}

# Current segment original text (for reference)
<current_original>
{current_segment_original}
</current_original>

# Context before (previous segment)
{context_before_section}

# Context after (next segment)
{context_after_section}

# Original text (translation to adjust)
<original>
{original_text}
</original>

# Output strictly as JSON with a single key "text"
{{
  "text": "...rewritten line in {target_language}..."
}}
"""
CONTEXT_ANALYSIS_PROMPT_TEMPLATE = """
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
{transcript_body}
</transcript>

IMPORTANT: Create the summary in "{target_language}" language.

Provide your analysis in JSON format with these keys:
domain, terminology, tone, themes, chapters, overall_summary

For chapters, include title, summary, and start_time for each chapter.

Example JSON output:
{{"domain": "technology","terminology": ["API","LLM","vector database"],"tone": "informative","themes": ["artificial intelligence","software development"],"chapters": [],"overall_summary": ""}}
"""

TRANSLATION_PROMPT_TEMPLATE = """
You are a professional translator specializing in {domain} content.

Translate the following transcript of a conversation from '{source_language}' language to '{target_language}' language.

Preserve the meaning, tone, and style of the original.

# General rules:
1. Do not translate proper names, brand names, and abbreviations.
2. Translatable terms: AI -> ИИ
3. Number and Date Conversion: Convert all digits and numbers to their written form in the target language as they would be naturally spoken aloud.
4. Remember this translation will be used for audio dubbing, so ensure the text flows naturally when spoken
5. Maintain the speaker identifiers exactly as given
6. Preserve the conversational flow and natural dialogue tone - don't make it sound too formal or robotic
7. Keep the emotional tone of the original speech (excited, concerned, questioning, etc.)
8. Ensure NO details or nuances from the original text are lost in translation
9. Pay special attention to {domain} terminology. all information, examples, technical concepts, and specific details accurately.
10. Preserve the original structure of the conversation, number of lines, and number of speakers.

{glossary_section}
{custom_section}

# Special handling for filler words and conciseness:
1. Remove any filler words from the translation to make it sound more fluent and professional.
2. When removing filler words results in a significantly shorter translation, use the freed-up space to expand on technical concepts, add natural connecting phrases, or provide slightly more context.
3. The goal is a natural-sounding translation that conveys the full meaning, not just a direct word-for-word conversion.
4. Balance conciseness with comprehensiveness - the translation should be clear and complete.

# When you detect humor, jokes, puns, or wordplay:
1. Try to preserve the humor in the target language.
2. If a direct translation would lose the humor, adapt it to an equivalent joke in the target language.
3. If a cultural reference wouldn't make sense, replace it with a similar reference understood by target-language speakers.
4. For wordplay that can't be directly translated, focus on preserving the comedic effect rather than the exact words.

# Length considerations for audio dubbing:
1. Try to maintain a similar length between the original and translated text.
2. This is crucial for audio dubbing, as the translated speech needs to fit within the same time constraints as the original.
3. If the translation would naturally be much longer, look for more concise ways to express the same ideas.
4. If the translation would naturally be much shorter, add natural filler phrases that enhance clarity.
5. The goal is to have the translated audio match the timing of the original speech as closely as possible.

# Translation considerations:
- Domain: {domain}
- Tone: {tone}
- Key themes: {themes}
- Technical terms: {terminology}
{summary_section}

# Following is the context of the conversation:

Context before:
<context_before>
{context_before}
</context_before>

Text to translate:
<text_to_translate>
{text_to_translate}
</text_to_translate>

Context after:
<context_after>
{context_after}
</context_after>

CRITICAL: Output translation should contain same number of rows and original speaker names. If phrase is not translatable, leave blank.

IMPORTANT: Respond in JSON format following this exact structure:

{{
  "translations": [
    {{
      "speaker": "SPEAKER_ID",
      "text": "translated text here"
    }}
  ]
}}

JSON Requirements:
- Root object with single key "translations"
- Value is array of objects
- Each object has "speaker" and "text" fields
- Field name is "text" NOT "translation"
- Number of objects must match input lines
- Speaker IDs must match exactly
"""


# Instructions inserted into refinement persona prompts.
# These strings are *not* formatted with .format() (they are injected as-is into the persona template),
# so they can safely include JSON braces and examples.

ALTERNATIVE_VERSIONS_FULL = """
# Alternative versions (required)
For every line, produce 3 additional variants that preserve ALL facts and speaker intent:
- very_short: noticeably shorter than "text" (tight, but still complete and natural)
- short: slightly shorter than "text"
- long: slightly longer than "text" (adds natural connective phrasing, but NO new facts)

Constraints:
- "text" is the refined, naturally rephrased version optimized for understanding and native fluency in the target language.
- You may rephrase text across consecutive blocks in the translations list if it makes the dialogue flow more naturally or improves comprehension in the target language.
- All variants must keep the same meaning and details; do not add or remove any concrete information.
- Keep speaker IDs unchanged.
"""

ALTERNATIVE_VERSIONS_LONG_ONLY = """
# Alternative versions (required)
For every line, produce a "long" variant that is slightly longer than "text" while preserving ALL facts.

Constraints:
- "text" is the refined, naturally rephrased version optimized for understanding and native fluency in the target language.
- You may rephrase text across consecutive blocks in the translations list if it makes the dialogue flow more naturally or improves comprehension in the target language.
- "long" may add natural connective phrasing, but MUST NOT add new facts or omit details.
- Keep speaker IDs unchanged.
"""

JSON_OUTPUT_FORMAT_FULL = r"""
Return ONLY valid JSON (no markdown, no commentary, no code fences).

Schema (all keys required):
{
  "translations": [
    {
      "speaker": "SPEAKER_ID",
      "text": "refined line",
      "very_short": "very short variant",
      "short": "short variant",
      "long": "long variant"
    }
  ]
}

Rules:
- The number of items in "translations" MUST equal the number of input lines.
- "speaker" MUST exactly match the corresponding input speaker for that line.
- All fields MUST be strings (use "" if absolutely necessary; do not omit keys).

Few-shot examples (structure only):
Example 1 output:
{"translations":[{"speaker":"SPEAKER_A","text":"A.","very_short":"A.","short":"A.","long":"Well, A."}]}

Example 2 output:
{"translations":[{"speaker":"SPEAKER_A","text":"We ship tomorrow.","very_short":"Ship tomorrow.","short":"We ship tomorrow.","long":"Alright, we ship tomorrow."},{"speaker":"SPEAKER_B","text":"Got it.","very_short":"OK.","short":"Got it.","long":"Yep, got it."}]}
"""

JSON_OUTPUT_FORMAT_LONG_ONLY = r"""
Return ONLY valid JSON (no markdown, no commentary, no code fences).

Schema (all keys required):
{
  "translations": [
    {
      "speaker": "SPEAKER_ID",
      "text": "refined line",
      "long": "long variant"
    }
  ]
}

Rules:
- The number of items in "translations" MUST equal the number of input lines.
- "speaker" MUST exactly match the corresponding input speaker for that line.
- "long" should be slightly longer than "text" while preserving ALL facts (no new claims).
- All fields MUST be strings (use "" if absolutely necessary; do not omit keys).

Few-shot examples (structure only):
Example 1 output:
{"translations":[{"speaker":"SPEAKER_A","text":"Got it.","long":"Yes, got it."}]}

Example 2 output:
{"translations":[{"speaker":"SPEAKER_A","text":"We start at nine.","long":"Alright, we start at nine."},{"speaker":"SPEAKER_B","text":"Perfect.","long":"Perfect, that works."}]}
"""
