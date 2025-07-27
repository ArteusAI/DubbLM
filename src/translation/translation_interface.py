from abc import ABC, abstractmethod
from typing import List, Dict, Any


class TranslationInterface(ABC):
    """
    Abstract base class for translation systems.
    All translation implementations should inherit from this class.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the translation system with required resources."""
        pass
    
    @abstractmethod
    def translate(
        self,
        segments: List[Dict],
        source_language: str = "en",
        target_language: str = "ru",
        **kwargs
    ) -> List[Dict]:
        """
        Translate segments with context awareness.
        
        This method handles the entire translation pipeline:
        1. Analyzing context and generating initial summary in source language 
        2. Optimizing segments (e.g., merging adjacent segments from same speaker)
        3. Chunking segments into logical groups
        4. Translating chunks
        5. Decomposing translated chunks back to segment level
        
        Args:
            segments: List of transcript segments
            source_language: Source language code
            target_language: Target language code
            **kwargs: Additional parameters
            
        Returns:
            List of segments with translations added
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the translation system is available and properly initialized."""
        pass 