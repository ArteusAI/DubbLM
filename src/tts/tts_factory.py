import os
from typing import Dict, Any, Union, Callable, Optional, Type

from .tts_interface import TTSInterface
from .f5_tts_wrapper import F5TTSWrapper
from .openai_tts_wrapper import OpenAITTSWrapper
from .gemini_tts_wrapper import GeminiTTSWrapper

# Define available TTS providers
TTS_PROVIDERS: Dict[str, Type[TTSInterface]] = {
    "f5": F5TTSWrapper,
    "openai": OpenAITTSWrapper,
    "gemini": GeminiTTSWrapper,
}

class TTSConfig:
    """Configuration for TTS factory."""
    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        default_voice: Optional[str] = None,
        voice_mapping: Optional[Dict[str, str]] = None,
        voice_prompt_mapping: Optional[Dict[str, str]] = None,
        prompt_prefix: Optional[str] = None,
        **kwargs: Any
    ):
        self.provider = provider
        self.model = model
        self.default_voice = default_voice
        self.voice_mapping = voice_mapping if voice_mapping is not None else {}
        self.voice_prompt_mapping = voice_prompt_mapping if voice_prompt_mapping is not None else {}
        self.prompt_prefix = prompt_prefix
        self.kwargs = kwargs # Store any additional provider-specific args

class TTSFactory:
    """Factory for creating TTS instances."""
    
    @staticmethod
    def create_tts(
        tts_system: str,
        device: str = "cpu", # device is not used by OpenAI or Gemini but kept for interface consistency
        voice_config: Optional[Union[str, Dict[str, str]]] = None,
        voice_prompt: Optional[Dict[str, str]] = None,
        prompt_prefix: Optional[str] = None,
        **kwargs: Any # To catch any other potential args like model name for specific TTS
    ) -> TTSInterface:
        """
        Create and return a TTS object based on the specified system.
        
        Args:
            tts_system: TTS system to use ("coqui", "openai", "f5_tts", "gemini")
            device: Computing device to use ("cuda" or "cpu")
            voice_config: Voice configuration:
                - For OpenAI/Gemini TTS: Either a single voice name (str) to be used as default_voice,
                                       or a mapping of speaker IDs to voice names (Dict) for voice_mapping.
            voice_prompt: Dictionary mapping speaker IDs to voice prompts for detailed speech style.
            prompt_prefix: Global prompt prefix for TTS generation instructions (mainly for Gemini TTS).
            **kwargs: Additional arguments for specific TTS providers (e.g., model).
            
        Returns:
            An instance of a TTS class implementing TTSInterface
            
        Raises:
            ValueError: If the TTS system is not supported
            RuntimeError: If client creation or initialization fails.
        """
        provider_name_lower = tts_system.lower()
        
        if provider_name_lower not in TTS_PROVIDERS:
            raise ValueError(f"Unsupported TTS system: {tts_system}. Supported: {list(TTS_PROVIDERS.keys())}")

        # Prepare general config arguments
        config_args = {
            "provider": provider_name_lower,
            "voice_prompt_mapping": voice_prompt,
            "prompt_prefix": prompt_prefix,
        }
        config_args.update(kwargs) # Pass through other kwargs like model

        if isinstance(voice_config, str):
            config_args["default_voice"] = voice_config
        elif isinstance(voice_config, dict):
            config_args["voice_mapping"] = voice_config
        
        # For providers that use 'device' (Coqui, F5), add it to kwargs for TTSConfig
        if provider_name_lower in ["coqui", "f5"]: # F5TTSWrapper uses f5 not f5_tts as key in current factory
             # Correcting based on TTS_PROVIDERS keys: "f5" is F5TTSWrapper
            config_args.setdefault("kwargs", {})["device"] = device


        tts_config = TTSConfig(**config_args)
        
        return TTSFactory.create_tts_client(tts_config)
            
    @staticmethod
    def create_tts_client(config: TTSConfig) -> TTSInterface:
        """
        Create a TTS client based on the provided configuration.

        Args:
            config: TTSConfig object with provider and other settings.

        Returns:
            An instance of a TTSInterface implementation.

        Raises:
            ValueError: If the specified provider is not supported.
            RuntimeError: If client creation or initialization fails.
        """
        provider_name_lower = config.provider.lower()
        provider_class = TTS_PROVIDERS.get(provider_name_lower)
        
        # This check is technically redundant if create_tts already validates,
        # but good for direct create_tts_client calls.
        if not provider_class: 
            supported_providers = list(TTS_PROVIDERS.keys())
            raise ValueError(
                f"Unsupported TTS provider: '{config.provider}'. Supported: {supported_providers}"
            )

        init_args = {}
        if config.model is not None:
            init_args["model"] = config.model
        if config.default_voice is not None:
            init_args["default_voice"] = config.default_voice
        
        # Add prompt_prefix for Gemini TTS
        if provider_name_lower == "gemini" and config.prompt_prefix is not None:
            init_args["prompt_prefix"] = config.prompt_prefix
        
        # Pass through any additional kwargs from TTSConfig (like 'device' for Coqui/F5)
        init_args.update(config.kwargs)
        
        try:
            tts_client = provider_class(**init_args)
            tts_client.initialize()
            
            # voice_mapping and voice_prompt_mapping can be empty dicts if None initially
            if config.voice_mapping: # This check is fine as it's {} if not provided
                tts_client.set_voice_mapping(config.voice_mapping)
            if config.voice_prompt_mapping: # This is also fine
                tts_client.set_voice_prompt_mapping(config.voice_prompt_mapping)
                
            return tts_client
        except Exception as e:
            # Ensure we catch and re-raise with context
            import traceback
            print(f"Error details during TTS client creation for provider '{config.provider}':\n{traceback.format_exc()}")
            raise RuntimeError(
                f"Failed to create or initialize TTS client for provider '{config.provider}': {str(e)}"
            ) from e

    @staticmethod
    def get_available_providers() -> list[str]:
        """Returns a list of available TTS providers."""
        return list(TTS_PROVIDERS.keys())
            
