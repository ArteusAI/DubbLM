from typing import Optional, Dict, Any, Union
import os

from src.translation.translation_interface import TranslationInterface
from src.translation.llm_translator import LLMTranslator

# Import additional translator classes as they are developed
# from src.translation.other_translator import OtherTranslator


class TranslatorFactory:
    """
    Factory class to create translation systems.
    """
    
    @staticmethod
    def create_translator(
        translator_type: str,
        **kwargs
    ) -> TranslationInterface:
        """
        Create and return a translator based on the specified type.
        
        Args:
            translator_type: Type of translator to create ("llm" or other future types)
            **kwargs: Additional configuration parameters
            
        Returns:
            An instance of a class implementing TranslationInterface
            
        Raises:
            ValueError: If the translator type is not supported
        """
        if translator_type == "llm":
            # Get LLM provider (gemini or openrouter)
            llm_provider = kwargs.get("llm_provider", "gemini")
            cost_tracker = kwargs.get("cost_tracker")
            
            # Get translation model configuration parameters
            model_name = kwargs.get("model_name")
            temperature = kwargs.get("temperature", 0.5)
            max_tokens = kwargs.get("max_tokens", 16384)
            
            # Get refinement model configuration parameters
            refinement_llm_provider = kwargs.get("refinement_llm_provider")
            refinement_model_name = kwargs.get("refinement_model_name")
            refinement_temperature = kwargs.get("refinement_temperature", 1.0)
            refinement_max_tokens = kwargs.get("refinement_max_tokens")
            
            # Get glossary if provided
            glossary = kwargs.get("glossary")

            # Persona used during refinement
            refinement_persona = kwargs.get("refinement_persona")

            # Optional additional prompt prefix for translation prompts
            translation_prompt_prefix = kwargs.get("translation_prompt_prefix")
            
            # Get cache manager if provided
            cache_manager = kwargs.get("cache_manager")

            enable_emotion_enrichment = kwargs.get("enable_emotion_enrichment", False)
            enable_llm_editor = kwargs.get("enable_llm_editor", False)
            editor_llm_provider = kwargs.get("editor_llm_provider")
            editor_model_name = kwargs.get("editor_model_name")
            editor_temperature = kwargs.get("editor_temperature", 1.0)
            editor_reasoning_effort = kwargs.get("editor_reasoning_effort")
            tts_system = kwargs.get("tts_system")
            tts_system_mapping = kwargs.get("tts_system_mapping")
            
            # Segment stretch mode for alternative version generation
            segment_stretch = kwargs.get("segment_stretch", "audio_and_video")
            
            translator = LLMTranslator(
                llm_provider=llm_provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                refinement_llm_provider=refinement_llm_provider,
                refinement_model_name=refinement_model_name,
                refinement_temperature=refinement_temperature,
                refinement_max_tokens=refinement_max_tokens,
                glossary=glossary,
                refinement_persona=refinement_persona,
                prompt_prefix=translation_prompt_prefix,
                cache_manager=cache_manager,
                cost_tracker=cost_tracker,
                enable_emotion_enrichment=enable_emotion_enrichment,
                segment_stretch=segment_stretch,
                enable_llm_editor=enable_llm_editor,
                editor_llm_provider=editor_llm_provider,
                editor_model_name=editor_model_name,
                editor_temperature=editor_temperature,
                editor_reasoning_effort=editor_reasoning_effort,
                tts_system=tts_system,
                tts_system_mapping=tts_system_mapping,
            )
                
            # Initialize the translator
            translator.initialize()
            return translator
        else:
            raise ValueError(f"Unsupported translator type: {translator_type}") 
