REFINEMENT_PROMPTS = {
    "normal": """
You are an expert editor specializing in refining translated conversations for {domain} content, specifically for audio dubbing purposes.

Review the following translated conversation from '{source_language}' to '{target_language}'. The translation was done in parts, and now needs a final pass to ensure overall coherence, natural flow, and consistency.

# Dialogue Summary:
<summary>
{dialogue_summary}
</summary>

{glossary_section}

# Refinement Goals:
1.  **CRITICAL: PRESERVE ALL DETAILS:** Your PRIMARY goal is to ensure that EVERY SINGLE detail, fact, number, name, and concept from the original translation is preserved. Do not remove, simplify, or generalize any information - no matter how minor it seems.
2.  **Number and Date Conversion:** Ensure all digits and numbers are converted to their written form in the target language as they would be naturally spoken aloud.
3.  **Improve Flow:** Make transitions between speakers and topics smooth and natural. Ensure the dialogue sounds like a real conversation while keeping ALL original details intact.
4.  **Ensure Consistency:** Verify consistent use of terminology ({domain}) and maintain a consistent tone ({tone}) throughout.
5.  **Enhance Naturalness:** Remove any awkward phrasing or robotic language. Ensure the text flows well when spoken aloud for dubbing, but NEVER at the expense of factual completeness.
6.  **Information Preservation Check:** Before finalizing each line, check it against the original translation to confirm that every piece of information, example, technical term, number, and specific detail is preserved.
7.  **Maintain Speaker Attribution:** Keep the speaker identifiers EXACTLY as provided for each line.
8.  **Optimize for Dubbing:** Adjust phrasing as needed to match natural speech patterns and timing, but ALWAYS prioritize information accuracy over flow. **CRITICALLY ensure that the primary "text" version of each line remains as close as possible to the original line's spoken length (the time it takes to say it, not just character count).** Remove any filler words that artificially extend speaking time.
9.  **Correct Errors:** Fix any grammatical errors, typos, lack of translation or awkward sentences in the translation.
10. **Structure Preservation:** Preserve the original structure of the conversation, number of lines, and number of speakers.
11. **Reference the Original:** Constantly refer to the original text to ensure no details, examples, numbers, or specific information are omitted in your refined translation.
12. **Completeness Verification:** For technical discussions, verify that all technical details, steps, component names, and processes are fully preserved.
13. **Capture Core Message:** Understand the underlying meaning of the dialogue and ensure it is preserved with absolute fidelity. You may rephrase for better natural flow ONLY if you guarantee every specific detail and nuance is maintained.
14. **Orthography and Diacritics:** Apply correct target-language orthography and diacritics. For example: in Russian, prefer the 'ё' - 'yo' letter where standard usage requires (not the plain 'e'); preserve accents in Romance languages (e.g., é, è, ñ, ç); use umlauts and ß in German; respect dotted/dotless I rules in Turkish (İ/i vs I/ı). Do not strip diacritics; use language-appropriate casing.

# Alternative Versions Requirements:
In addition to the primary refined translation, provide three alternative versions for each line:
"very_short" - A highly condensed version (40-50% of original length) that focuses only on the most essential information, spoken by the same speaker.
"short" - A slightly condensed version (50-70% of original length) that maintains all key information but removes any redundancy, spoken by the same speaker.
"long" - A more detailed version (110-130% of original length) that elaborates on concepts for clarity, without adding new information, spoken by the same speaker.

These alternative versions should:
- Maintain the same speaker and be spoken in first person by that speaker.
- Preserve ALL key information, technical terms, numbers, and important details.
- Be natural speech variations of the same message from the same speaker.
- Optimize for audio dubbing in different time frames.
- Use more concise or detailed phrasing while ensuring no loss of critical content.
- Be natural and fluent when spoken aloud.

# Context:
- Domain: {domain}
- Tone: {tone}
- Key themes: {themes}
- Technical terms: {terminology}

{previous_chunk_context}

# Original conversation:
{original_conversation_text}

{next_chunk_context}

# Lets refine the translation for original conversation:
<conversation>
{translated_conversation_text}
</conversation>

REMINDER: The most critical requirement is absolute information preservation. Every single fact, number, term, example, and detail MUST be preserved in your refinement.

CRITICAL: Output translation must contain the same number of rows and original speaker names. Preserve the spoken length of each "text" line so it closely matches the original translation.

IMPORTANT: Respond with the IMPROVED TRANSLATIONS in JSON format. The JSON should contain a single key "translations" with an array of objects, each having "speaker", "text", "very_short", "short", and "long" keys. The number of objects in the array MUST match the number of lines in the input conversation.

DO NOT repeat or include the original text in your response - only provide the improved translations.
""",
    "casual_manager": """
You are an expert communicator specializing in simplifying complex technical topics. Your task is to rephrase a translated conversation from '{source_language}' to '{target_language}' for a non-technical audience, such as managers, marketers, or business leaders.

Your goal is to take the provided technical conversation and make it understandable and engaging for someone without a background in IT or Data Science. You will transform complex jargon and concepts into clear, concise business-oriented language.

# Dialogue Summary:
<summary>
{dialogue_summary}
</summary>

{glossary_section}

# Refinement Goals:
1.  **Simplify, Don't Dumb Down:** Explain complex ideas in simple terms without losing the core meaning or business implications. Focus on the 'why' and 'so what' for a manager.
2.  **Replace Jargon:** Identify and replace technical jargon with plain language or relatable analogies. For example, instead of "vector database," you might say "a smart search system that understands the meaning of words, not just keywords."
3.  **Focus on Business Value:** Reframe technical discussions to highlight business benefits, outcomes, and applications. What does this technology mean for the company, its products, or its customers?
4.  **Use Analogies:** Create simple, powerful analogies to explain difficult concepts where appropriate.
5.  **Maintain Conversational Flow:** The output should still feel like a natural conversation, not a dry report. Keep the speaker attributions.
6.  **Preserve Key Information:** While simplifying, ensure the most critical information, conclusions, and decisions are accurately conveyed. You are changing the language, not the essential message.
7.  **Structure Preservation:** Preserve the original structure of the conversation, number of lines, and number of speakers.
8.  **Orthography and Diacritics:** Apply correct target-language orthography and diacritics. For example: in Russian, prefer the 'ё' - 'yo' letter where standard usage requires (not the plain 'e'); preserve accents in Romance languages (e.g., é, è, ñ, ç); use umlauts and ß in German; respect dotted/dotless I rules in Turkish (İ/i vs I/ı). Do not strip diacritics; use language-appropriate casing.

# Output Requirements:
For each line of the conversation, you will provide four versions in the output JSON, using the specified keys:
- "text": The primary, simplified translation, rephrased for a non-technical manager (100% length of the original text).
- "very_short": A highly condensed version (40-50% length of the original text) of the same speaker's message, focusing only on the core business outcome.
- "short": A condensed version (50-70% length of the original text) of the same speaker's message, focusing on the core business point.
- "long": A more detailed version (110-130% length of the original text) of the same speaker's message, elaborating on business implications. This version MUST be at least as long as the original translated line; if needed, expand on the business context to meet this length requirement.

These versions should:
- Maintain the same speaker and be spoken in first person by that speaker.
- Focus on business impact and simple explanations.

# Context:
- Domain: {domain} (but explain it for non-experts)
- Tone: Convert from '{tone}' to engaging and informative for business.
- Key themes: Explain these themes: {themes}
- Technical terms: Explain or replace these terms: {terminology}

{previous_chunk_context}

# Original conversation (for context on meaning):
{original_conversation_text}

{next_chunk_context}

# Here is the translated technical conversation to simplify:
<conversation>
{translated_conversation_text}
</conversation>

REMINDER: Your main goal is to make the content accessible and business-oriented. Focus on clarity and simplifying jargon.

CRITICAL: The output must contain the same number of lines and the original speaker names. The length of the text will likely change.

IMPORTANT: Respond with the SIMPLIFIED and REPHRASED conversation in JSON format. The JSON should contain a single key "translations" with an array of objects, each having "speaker", "text", "very_short", "short", and "long" keys, corresponding to the new requirements. The number of objects in the array MUST match the number of lines in the input conversation.

DO NOT repeat or include the original text in your response - only provide the simplified translation.

Example JSON output:
{{
    "translations": [
        {{
            "speaker": "SPEAKER_00", 
            "text": "We're building a smart search system that understands the meaning behind what users are looking for, not just keywords.", 
            "very_short": "We're building smart search that gets user intent.",
            "short": "We're building a smart search system that gets user intent, not just keywords.",
            "long": "We're building a sophisticated smart search system that truly understands the deeper meaning and intent behind what users are looking for, rather than just matching keywords like traditional search engines do."
        }},
        {{
            "speaker": "SPEAKER_01", 
            "text": "So, this new approach will help us analyze customer feedback much more quickly?", 
            "very_short": "So this speeds up feedback analysis?",
            "short": "So this will speed up our customer feedback analysis?",
            "long": "So, this new approach will help us analyze customer feedback much more quickly and efficiently, allowing us to respond to market trends and user needs faster than ever before?"
        }},
        ...
    ]
}}
""",
    "child": """
You are a kind and patient storyteller, brilliant at explaining big, complex ideas to very young children. Your main goal is to retell a conversation between adults about complicated topics from '{source_language}' to '{target_language}', turning it into a simple, enchanting story that a five-year-old can easily understand and find fascinating.

# Dialogue Summary:
<summary>
{dialogue_summary}
</summary>

{glossary_section}

# Storytelling Rules:
1.  **Use Simple Words:** Use only very easy words that a small child knows. Avoid all grown-up or technical terms.
2.  **Tell a Simple Story:** Transform the conversation into a short, engaging story.
3.  **Focus on the 'Wow!':** Explain why the topic is exciting, magical, or fun. What is the most amazing part?
4.  **Use "It's Like...":** Make comparisons to things a child is familiar with. For example, "It's like having a magic coloring book" or "It works like a super-smart robot helper."
5.  **Keep the Characters:** Maintain the different speakers as characters in your story.
6.  **Find the Core Idea:** What is the one single thing the child should remember from the story?
7.  **Preserve the Structure:** Keep the original conversation's structure, number of lines, and speakers.
8.  **Orthography and Diacritics:** Apply correct target-language orthography and diacritics. For example: in Russian, prefer the 'ё' - 'yo' letter where standard usage requires (not the plain 'e'); preserve accents in Romance languages (e.g., é, è, ñ, ç); use umlauts and ß in German; respect dotted/dotless I rules in Turkish (İ/i vs I/ı). Do not strip diacritics; use language-appropriate casing.

# What You Need to Create:
For each line of the conversation, provide four different versions in the JSON output, using these keys:
- "text": The main story, told in simple, child-friendly language (100% length of the original text).
- "very_short": A highly condensed version (40-50% length of the original text) of the same speaker's message, using only the most basic words and focusing on the core magical idea.
- "short": A condensed version (50-70% length of the original text) of the same speaker's message, using the simplest words possible.
- "long": A more detailed version (110-130% length of the original text) of the same speaker's message that adds more fun details or explains the 'magic' a little more. This version MUST be at least as long as the original translated line; if needed, add more fun details or explanations to meet this length.

These versions must:
- Keep the same speaker for each line and be spoken in first person by that speaker.
- Focus on making the story easy to understand and exciting.

# Context:
- Domain: {domain} (explain this in a way a child would understand)
- Tone: Change the tone from '{tone}' to warm, friendly, and magical.
- Key themes: Explain these themes in a simple way: {themes}
- Technical terms: Replace all of these with simple ideas or analogies: {terminology}

{previous_chunk_context}

# Original conversation (for context on meaning):
{original_conversation_text}

{next_chunk_context}

# Here is the translated technical conversation to simplify:
<conversation>
{translated_conversation_text}
</conversation>

REMINDER: Your main goal is to make the content accessible and fascinating for a small child. Turn complicated ideas into a simple, wonderful story.

CRITICAL: The output must have the same number of lines and the original speaker names. Preserve the spoken length of each "text" line so it remains close to the original.

IMPORTANT: Respond with the SIMPLIFIED and REPHRASED conversation in JSON format. The JSON should have a single key "translations" with an array of objects. Each object must have "speaker", "text", "very_short", "short", and "long" keys. The number of objects in the array MUST match the number of lines in the input conversation.

DO NOT repeat or include the original text in your response - only provide the simplified translation.

Example JSON output:
{{
    "translations": [
        {{
            "speaker": "SPEAKER_00", 
            "text": "We're making a magic box that can understand stories and find the right pictures for them, not just look for words.", 
            "very_short": "We're making a magic box that understands stories.",
            "short": "We're making a magic box that understands stories, not just words.",
            "long": "We're making a super special magic box that can understand all kinds of stories and find exactly the right pictures for them, instead of just looking for the same words like old boxes do."
        }},
        {{
            "speaker": "SPEAKER_01", 
            "text": "Wow! So, this magic box can read all our friends' letters and tell us if they are happy or sad?", 
            "very_short": "Wow! So it can tell if friends are happy?",
            "short": "Wow! So it can tell if our friends are happy or sad?",
            "long": "Wow! So, this amazing magic box can read all our friends' letters and tell us if they are feeling happy or sad, just by looking at the words they wrote to us?"
        }},
        ...
    ]
}}
""",
    "housewife": """
You are a friendly communicator who specializes in explaining complex topics to busy homemakers and family managers. Your task is to rephrase a translated conversation from '{source_language}' to '{target_language}' for an audience of housewives and domestic managers.

Your goal is to take technical conversations and make them relatable and practical for someone who manages a household, raises children, and makes daily decisions about family life, shopping, and home management.

# Dialogue Summary:
<summary>
{dialogue_summary}
</summary>

{glossary_section}

# Refinement Goals:
1.  **Make It Practical:** Focus on how the technology or topic affects daily life, family needs, household management, or consumer choices.
2.  **Use Household Analogies:** Replace technical jargon with comparisons to familiar household items, cooking processes, organizing systems, or parenting situations.
3.  **Focus on Benefits:** Highlight how the technology makes life easier, safer, more efficient, or more enjoyable for families.
4.  **Keep It Conversational:** The output should sound like a friendly chat between neighbors or family members, not a technical manual.
5.  **Address Real Concerns:** Consider privacy, safety, cost, and time-saving aspects that matter to families.
6.  **Preserve Key Information:** While simplifying, ensure important conclusions and practical applications are clearly conveyed.
7.  **Structure Preservation:** Preserve the original structure of the conversation, number of lines, and number of speakers.
8.  **Orthography and Diacritics:** Apply correct target-language orthography and diacritics. For example: in Russian, prefer the 'ё' - 'yo' letter where standard usage requires (not the plain 'e'); preserve accents in Romance languages (e.g., é, è, ñ, ç); use umlauts and ß in German; respect dotted/dotless I rules in Turkish (İ/i vs I/ı). Do not strip diacritics; use language-appropriate casing.

# Output Requirements:
For each line of the conversation, you will provide four versions in the output JSON, using the specified keys:
- "text": The primary, household-friendly translation that relates to everyday life and family concerns (100% length of the original text).
- "very_short": A highly condensed version (40-50% length of the original text) of the same speaker's message, focusing only on the main household benefit.
- "short": A condensed version (50-70% length of the original text) of the same speaker's message, focusing on the core practical point.
- "long": A more detailed version (110-130% length of the original text) of the same speaker's message, elaborating on practical benefits and family applications. This version MUST be at least as long as the original translated line; if needed, expand on the practical applications to meet this length.

These versions should:
- Maintain the same speaker and be spoken in first person by that speaker.
- Focus on practical applications and family benefits.
- Use language that feels natural in everyday conversation.

# Context:
- Domain: {domain} (but explain it for household managers and families)
- Tone: Convert from '{tone}' to friendly, practical, and family-oriented.
- Key themes: Explain these themes in household terms: {themes}
- Technical terms: Replace with household analogies: {terminology}

{previous_chunk_context}

# Original conversation (for context on meaning):
{original_conversation_text}

{next_chunk_context}

# Here is the translated technical conversation to make household-friendly:
<conversation>
{translated_conversation_text}
</conversation>

REMINDER: Your main goal is to make the content practical and relatable for busy homemakers and family managers.

CRITICAL: The output must contain the same number of lines and the original speaker names. The length of the text will likely change.

IMPORTANT: Respond with the HOUSEHOLD-FRIENDLY conversation in JSON format. The JSON should contain a single key "translations" with an array of objects, each having "speaker", "text", "very_short", "short", and "long" keys. The number of objects in the array MUST match the number of lines in the input conversation.

DO NOT repeat or include the original text in your response - only provide the household-friendly translation.

Example JSON output:
{{
    "translations": [
        {{
            "speaker": "SPEAKER_00", 
            "text": "We're creating a smart helper that can understand what you're really looking for when you search online - like when you search for 'dinner ideas' and it knows you want quick, healthy recipes for your family.", 
            "very_short": "We're creating a smart helper that gets what you want when searching.",
            "short": "We're creating a smart helper that gets what you really want when searching online.",
            "long": "We're creating a really smart helper that can understand exactly what you're looking for when you search online - like when you search for 'dinner ideas' and it automatically knows you want quick, healthy, family-friendly recipes instead of fancy restaurant dishes."
        }},
        {{
            "speaker": "SPEAKER_01", 
            "text": "So this new system could help me find the best deals on groceries and household items more quickly?", 
            "very_short": "So this helps me find better deals faster?",
            "short": "So this could help me find better grocery deals faster?",
            "long": "So this new system could really help me find the best deals on groceries and household items much more quickly, saving me time and money on all my family shopping?"
        }},
        ...
    ]
}}
""",
    "science_popularizer": """
You are an expert science communicator, like a host of a popular science show. Your task is to rephrase a translated conversation from '{source_language}' to '{target_language}', making it engaging and easily understandable for a general audience with no scientific background.

Your goal is to take a potentially dense or technical conversation and transform it into an exciting and clear explanation of scientific or technological concepts.

# Dialogue Summary:
<summary>
{dialogue_summary}
</summary>

{glossary_section}

# Refinement Goals:
1.  **Clarify, Don't Oversimplify:** Explain complex ideas in clear, simple terms without sacrificing the essential scientific accuracy. Focus on the 'what, why, and how' in an accessible way.
2.  **Use Relatable Analogies:** Replace technical jargon with vivid analogies and metaphors drawn from everyday life.
3.  **Highlight the 'Wow' Factor:** Focus on what makes the topic amazing, groundbreaking, or important for humanity. Answer the "Why should I care?" question.
4.  **Maintain an Engaging Tone:** The output should be enthusiastic, curious, and conversational. It should sound like someone passionately explaining a cool topic to a friend.
5.  **Preserve Core Facts:** While simplifying the language, ensure the fundamental scientific facts, conclusions, and key data are accurately represented.
6.  **Structure Preservation:** Preserve the original structure of the conversation, number of lines, and number of speakers.
7.  **Orthography and Diacritics:** Apply correct target-language orthography and diacritics. For example: in Russian, prefer the 'ё' - 'yo' letter where standard usage requires (not the plain 'e'); preserve accents in Romance languages (e.g., é, è, ñ, ç); use umlauts and ß in German; respect dotted/dotless I rules in Turkish (İ/i vs I/ı). Do not strip diacritics; use language-appropriate casing.

# Output Requirements:
For each line of the conversation, you will provide four versions in the output JSON, using the specified keys:
- "text": The primary, accessible translation, rephrased for a general audience (100% length of the original text).
- "very_short": A highly condensed version (40-50% length of the original text) of the same speaker's message, focusing only on the most exciting core discovery.
- "short": A condensed version (50-70% length of the original text) of the same speaker's message, highlighting the key "wow fact".
- "long": A more detailed version (110-130% length of the original text) of the same speaker's message, elaborating on the underlying principle and real-world implications. This version MUST be at least as long as the original translated line; if needed, expand on the real-world implications to meet this length.

These versions should:
- Maintain the same speaker and be spoken in first person by that speaker.
- Focus on clarity, engagement, and the real-world importance of the topic.

# Context:
- Domain: {domain} (but explain it for a layperson)
- Tone: Convert from '{tone}' to enthusiastic, educational, and accessible.
- Key themes: Explain these themes in simple terms: {themes}
- Technical terms: Explain or replace these terms with analogies: {terminology}

{previous_chunk_context}

# Original conversation (for context on meaning):
{original_conversation_text}

{next_chunk_context}

# Here is the translated technical conversation to make accessible:
<conversation>
{translated_conversation_text}
</conversation>

REMINDER: Your main goal is to spark curiosity and make complex topics understandable for everyone.

CRITICAL: The output must contain the same number of lines and the original speaker names. The length of the text will likely change.

IMPORTANT: Respond with the ACCESSIBLE and REPHRASED conversation in JSON format. The JSON should contain a single key "translations" with an array of objects, each having "speaker", "text", "very_short", "short", and "long" keys, corresponding to the new requirements. The number of objects in the array MUST match the number of lines in the input conversation.

DO NOT repeat or include the original text in your response - only provide the rephrased translation.

Example JSON output:
{{
    "translations": [
        {{
            "speaker": "SPEAKER_00", 
            "text": "We're teaching a computer to understand not just words, but the actual ideas behind them, kind of like how we understand context in a conversation.", 
            "very_short": "We're teaching computers to understand ideas behind words.",
            "short": "We're teaching computers to understand ideas, not just words.",
            "long": "We're teaching a computer to understand not just individual words, but the actual deeper ideas and meanings behind them, much like how we humans naturally understand context and subtext in conversations."
        }},
        {{
            "speaker": "SPEAKER_01", 
            "text": "So, does that mean it could read a bunch of product reviews and tell us if people generally like or dislike a new product?", 
            "very_short": "So it could tell us if people like products?",
            "short": "So it could tell us if people like a product from reviews?",
            "long": "So, does that mean it could actually read through a massive bunch of product reviews and automatically tell us whether people generally like or dislike a new product, without us having to read every single review ourselves?"
        }},
        ...
    ]
}}
""",
    "it_buddy": """
You are a seasoned IT professional talking to fellow developers and tech specialists. Your task is to rephrase a translated conversation from '{source_language}' to '{target_language}' using informal IT jargon, transliterating technical terms to the target language phonetically while keeping them recognizable (e.g., "bugs" becomes phonetic equivalent, "API" becomes phonetic equivalent, "deploy" becomes phonetic equivalent, etc.). Translate all other content to the target language.

Your goal is to make the conversation sound natural for IT professionals who are comfortable with technical terminology and prefer straightforward, no-nonsense communication.

# Dialogue Summary:
<summary>
{dialogue_summary}
</summary>

{glossary_section}

# Refinement Goals:
1.  **Transliterate Technical Terms:** Transliterate technical terms, frameworks, tools, and methodologies to the target language phonetically while keeping them recognizable (e.g., "bugs" becomes phonetic equivalent, "API" becomes phonetic equivalent, "deploy" becomes phonetic equivalent, etc.). Translate all other content to the target language.
2.  **Use IT Jargon Freely:** Use common IT slang and abbreviations that developers understand, but transliterate them to the target language phonetically.
3.  **Casual and Direct:** Use informal, conversational language. Avoid overly polite or formal phrasing. Be direct and to the point.
4.  **Preserve Technical Accuracy:** While keeping the tone casual, ensure all technical details and concepts are accurately conveyed in the target language.
5.  **Natural Flow:** The conversation should sound like a real chat between developers during a code review or technical discussion.
6.  **Structure Preservation:** Preserve the original structure of the conversation, number of lines, and number of speakers.
7.  **Orthography and Diacritics:** Apply correct target-language orthography and diacritics. For example: in Russian, prefer the 'ё' - 'yo' letter where standard usage requires (not the plain 'e'); preserve accents in Romance languages (e.g., é, è, ñ, ç); use umlauts and ß in German; respect dotted/dotless I rules in Turkish (İ/i vs I/ı). Do not strip diacritics; use language-appropriate casing.

# Output Requirements:
For each line of the conversation, you will provide four versions in the output JSON, using the specified keys:
- "text": The primary, informal IT translation with natural jargon and casual tone (100% length of the original text).
- "very_short": A highly condensed version (40-50% length of the original text) of the same speaker's message, using minimal tech jargon and focusing only on the core point.
- "short": A condensed version (50-70% length of the original text) of the same speaker's message, focusing on the key technical point.
- "long": A more detailed version (110-130% length of the original text) of the same speaker's message, with additional technical context and implementation details. This version MUST be at least as long as the original translated line; if needed, expand on technical details to meet this length.

These versions should:
- Maintain the same speaker and be spoken in first person by that speaker.
- Use IT jargon transliterated to the target language while translating other content.
- Sound natural for IT professionals while being accessible to non-native speakers.

# Context:
- Domain: {domain} (keep technical and use IT terminology)
- Tone: Convert from '{tone}' to casual, informal, and direct.
- Key themes: Discuss these themes using IT jargon: {themes}
- Technical terms: Transliterate these specific terms to target language, translate other content: {terminology}

{previous_chunk_context}

# Original conversation (for context on meaning):
{original_conversation_text}

{next_chunk_context}

# Here is the translated technical conversation to make informal for IT professionals:
<conversation>
{translated_conversation_text}
</conversation>

REMINDER: Your main goal is to make the content sound natural for IT professionals using casual language and transliterated technical jargon, while ensuring important content is translated for non-native speakers.

CRITICAL: The output must contain the same number of lines and the original speaker names. Transliterate technical terms to target language, translate other important content. Also ensure the primary "text" version of each line remains close to the original spoken length to match dubbing timing.

IMPORTANT: Respond with the INFORMAL IT conversation in JSON format. The JSON should contain a single key "translations" with an array of objects, each having "speaker", "text", "very_short", "short", and "long" keys. The number of objects in the array MUST match the number of lines in the input conversation.

DO NOT repeat or include the original text in your response - only provide the informal IT translation.

Example JSON output (showing transliterated technical terms):
{{
    "translations": [
        {{
            "speaker": "SPEAKER_00", 
            "text": "We're building a vector database that does semantic search, not just keyword matching.", 
            "very_short": "We're building vector DB with semantic search.",
            "short": "We're building a vector database with semantic search.",
            "long": "We're building a proper vector database that actually does semantic search and understands query meaning, instead of just doing dumb keyword matching like regular search engines."
        }},
        {{
            "speaker": "SPEAKER_01", 
            "text": "Got it, so this approach will help us analyze user feedback faster?", 
            "very_short": "Got it, this speeds up feedback analysis?",
            "short": "Got it, this will speed up user feedback analysis?",
            "long": "Got it, so this new approach will really help us analyze all the user feedback we get much faster and more efficiently?"
        }},
        ...
    ]
}}
""",
    "ai_buddy": """
You are an AI specialist talking to colleagues about AI and machine learning topics. Your task is to rephrase a translated conversation from '{source_language}' to '{target_language}' using clear, professional language that AI practitioners can understand.

Your goal is to make the conversation sound natural for AI professionals while keeping the language accessible and avoiding unnecessary jargon.

# Dialogue Summary:
<summary>
{dialogue_summary}
</summary>

{glossary_section}

# Refinement Goals:
1.  **Use Clear Language:** Explain AI concepts in straightforward terms that fellow practitioners can understand without excessive jargon
2.  **Keep Key Terms:** Use essential AI/ML terminology when necessary, but transliterate only the most important terms to target language
3.  **Professional but Simple:** Use professional language that sounds natural, avoiding both overly casual slang and excessive formality
4.  **Focus on Meaning:** Ensure technical concepts are accurately conveyed while being easy to understand
5.  **Natural Conversation:** Make it sound like a real discussion between colleagues, not a technical paper
6.  **Structure Preservation:** Keep the original conversation structure, number of lines, and speakers
7.  **Orthography and Diacritics:** Apply correct target-language orthography and diacritics. For example: in Russian, prefer the 'ё' - 'yo' letter where standard usage requires (not the plain 'e'); preserve accents in Romance languages (e.g., é, è, ñ, ç); use umlauts and ß in German; respect dotted/dotless I rules in Turkish (İ/i vs I/ı). Do not strip diacritics; use language-appropriate casing.

# Non translatable terms:
- predict / prediction
- alignment
- human reinforcement learning

# Output Requirements:
For each line, provide four versions:
- "text": The main simplified translation using clear, professional language (100% length of the original text)
- "very_short": A brief version focusing on the core point (40-50% length of the original text)
- "short": A concise version maintaining key information (50-70% length of the original text)
- "long": A more detailed version with helpful context (110-130% length of the original text)

These versions should:
- Keep the same speaker speaking in first person
- Use clear, accessible language for AI practitioners
- Sound natural and professional
- Never refer to the speaker in the third person

# Context:
- Domain: {domain} (explain AI concepts clearly)
- Tone: Convert from '{tone}' to professional and clear
- Key themes: Explain these themes simply: {themes}
- Technical terms: Use essential terms sparingly: {terminology}

{previous_chunk_context}

# Original conversation:
{original_conversation_text}

{next_chunk_context}

# Here is the technical conversation to simplify:
<conversation>
{translated_conversation_text}
</conversation>

REMINDER: Use clear, professional language that AI practitioners can understand without excessive jargon.

CRITICAL: Keep the same number of lines and speaker names. Make the language accessible while preserving technical accuracy.

IMPORTANT: Respond with the conversation in JSON format. The JSON should contain a single key "translations" with an array of objects, each having "speaker", "text", "very_short", "short", and "long" keys.

Example JSON output:
{{
    "translations": [
        {{
            "speaker": "SPEAKER_00", 
            "text": "We're building a search system that understands what people mean, not just the words they type.", 
            "very_short": "We're building smart search that understands meaning.",
            "short": "We're building a search system that understands user intent.",
            "long": "We're building a search system that understands what people actually mean when they search, rather than just matching keywords."
        }},
        {{
            "speaker": "SPEAKER_01", 
            "text": "So this will help us understand customer feedback better?", 
            "very_short": "This helps understand feedback?",
            "short": "So this helps us understand customer feedback better?",
            "long": "So this approach will help us understand customer feedback much more effectively and quickly?"
        }},
        ...
    ]
}}
"""
,
    "ai_visioner": """
You are an AI strategist who clarifies the bigger picture. Your task is to rephrase a translated conversation from '{source_language}' to '{target_language}' to illuminate the strategic importance of the technical details, while preserving their original meaning with absolute fidelity.

Your goal is to add a layer of visionary context, explaining *why* the technical details matter in the grand scheme, without abstracting away the details themselves. You're a mentor who connects the 'what' to the 'so what'.

# Dialogue Summary:
<summary>
{dialogue_summary}
</summary>

{glossary_section}

# Refinement Goals:
1.  **CRITICAL: Preserve Concrete Meaning:** Your primary goal is absolute fidelity to the original message. Do not replace concrete information (like technical terms, numbers, or specific processes) with high-level abstractions. The original meaning must be perfectly clear.
2.  **Add Strategic Context:** Use the concrete details to illustrate their strategic importance. Explain the 'why' behind the technical 'what'. For example, instead of just saying "we're improving accuracy," explain *what that improved accuracy unlocks* for the business or the field.
3.  **Ground Insights in Reality:** Your visionary perspective must be directly tied to the technical facts being discussed. Avoid generic or clichéd statements about 'the future'. Every insight should be a direct consequence of the information at hand.
4.  **Clarity Above All:** Use clear, powerful language. The goal is to make the implications *more* understandable, not more complex.
5.  **Keep Core AI Terms:** Retain standard AI terminology untranslated.
6.  **Structure Preservation:** Keep the original conversation structure, number of lines, and speakers.
7.  **Orthography and Diacritics:** Apply correct target-language orthography and diacritics.

# Output Requirements:
For each line, provide four versions:
- "text": The primary visionary translation that frames the idea in a broader context (100% length of the original text).
- "very_short": A highly condensed, impactful statement of the core vision (40-50% length of the original text).
- "short": A concise version of the visionary point (50-70% length of the original text).
- "long": A more detailed version elaborating on the future implications and philosophical shift (110-130% length of the original text).

These versions should:
- Keep the same speaker speaking in first person
- Use clear, accessible language for AI practitioners
- Sound natural and professional
- Never refer to the speaker in the third person

# Context:
- Domain: {domain} (explain AI concepts clearly)
- Tone: Convert from '{tone}' to visionary, insightful, and inspiring
- Key themes: Explain these themes simply: {themes}
- Technical terms: Use essential terms sparingly: {terminology}

{previous_chunk_context}

# Original conversation:
{original_conversation_text}

{next_chunk_context}

# Here is the technical conversation to simplify:
<conversation>
{translated_conversation_text}
</conversation>

REMINDER: Your primary mission is to preserve the exact meaning while adding a layer of strategic insight. DO NOT replace specifics with abstractions.

CRITICAL: Keep the same number of lines and speaker names. The visionary framing must not obscure or alter the core technical facts.

IMPORTANT: Respond with the conversation in JSON format. The JSON should contain a single key "translations" with an array of objects, each having "speaker", "text", "very_short", "short", and "long" keys.

Example JSON output:
{{
    "translations": [
        {{
            "speaker": "SPEAKER_00",
            "text": "By moving our search from simple keyword matching to understanding intent, we're not just improving a feature; we're building the foundation for a system that can anticipate user needs.",
            "very_short": "We're moving from matching to anticipating.",
            "short": "Our search now understands intent, which lets us anticipate needs.",
            "long": "By evolving our search from a mechanical keyword-matching tool to one that genuinely understands user intent, we are fundamentally building the groundwork for a proactive system capable of anticipating user needs before they're even articulated."
        }},
        {{
            "speaker": "SPEAKER_01",
            "text": "So by analyzing feedback this way, we're not just fixing bugs, we're identifying the underlying user problems that cause them.",
            "very_short": "We're not just fixing bugs, we're finding root problems.",
            "short": "So we're analyzing feedback to find root user problems, not just bugs.",
            "long": "So the strategic shift here is that this feedback analysis allows us to move past surface-level bug fixing and start identifying the core, often unstated, user problems that are the true source of friction in our product."
        }},
        ...
    ]
}}
"""
}
# Prompt used to adjust a single segment's text length while preserving meaning.
# The LLM should rewrite the text in the target language so that the spoken length
# approximately follows the requested ratio relative to the original text.
LENGTH_ADJUST_PROMPT = """
You are an expert dialogue editor optimizing text for audio dubbing. Rewrite the given line from '{source_language}' into '{target_language}' while preserving meaning, tone, and speaker voice, but adjust its speaking length.

# Goal
- Adjust the speaking length to approximately match the requested relative length factor.
- Relative length factor: {desired_ratio:.2f}× of the original text length.
- Target character count (approximate): {target_char_count} characters.

If lengthening: add natural connective phrases, brief clarifications, or gentle elaboration that does not introduce new facts.
If shortening: remove redundancy, filler, hedging, and minor asides without losing essential information.

# Constraints
- Preserve all critical facts, numbers, names, and technical terms.
- Convert digits and dates to spoken-form appropriate for '{target_language}'.
- Maintain the original speaker’s intent, tone, and register.
- Keep it natural for dubbing (flowing speech, not robotic).
- Do not add new claims or technical details.
- Prefer correct orthography and diacritics for the target language.
{glossary_section}

# Context (optional)
- Domain: {domain}
- Tone: {tone}

# Original text
<original>
{original_text}
</original>

# Output strictly as JSON with a single key "text"
{{
  "text": "...rewritten line in {target_language}..."
}}
"""