from typing import Optional, Dict, Any, Union
import os

from translation.translation_interface import TranslationInterface
from translation.llm_translator import LLMTranslator

# Import additional translator classes as they are developed
# from translation.other_translator import OtherTranslator


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
            
            # Get translation model configuration parameters
            model_name = kwargs.get("model_name")
            temperature = kwargs.get("temperature", 0.5)
            max_tokens = kwargs.get("max_tokens", 1024)
            
            # Get refinement model configuration parameters
            refinement_llm_provider = kwargs.get("refinement_llm_provider")
            refinement_model_name = kwargs.get("refinement_model_name")
            refinement_temperature = kwargs.get("refinement_temperature", 1.0)
            refinement_max_tokens = kwargs.get("refinement_max_tokens")
            
            # Get glossary if provided
            glossary = kwargs.get("glossary")
            
            translator = LLMTranslator(
                llm_provider=llm_provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                refinement_llm_provider=refinement_llm_provider,
                refinement_model_name=refinement_model_name,
                refinement_temperature=refinement_temperature,
                refinement_max_tokens=refinement_max_tokens,
                glossary=glossary
            )
                
            # Initialize the translator
            translator.initialize()
            return translator
        else:
            raise ValueError(f"Unsupported translator type: {translator_type}") 