"""
Common interface for transcription and diarization services.
"""
import os
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import torch


class TranscriptionInterface(ABC):
    """Interface for transcription and diarization services."""
    
    @abstractmethod
    def __init__(
        self,
        source_language: str,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the transcription service.
        
        Args:
            source_language: Source language code (e.g., 'en')
            device: Compute device ('cuda' or 'cpu')
            **kwargs: Additional implementation-specific parameters
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the transcription service implementation."""
        pass
    
    @abstractmethod
    def diarize_and_transcribe(
        self,
        audio_file: str,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]]]:
        """
        Perform speaker diarization and transcription on the audio file.
        
        Args:
            audio_file: Path to the audio file
            cache_key: Optional cache key for caching results
            use_cache: Whether to use cached results if available
            
        Returns:
            Tuple containing:
            - Dictionary mapping time ranges to speaker IDs
            - List of transcription segments with timing information
        """
        pass
    
    @staticmethod
    def get_torch_device(device_str: Optional[str] = None) -> torch.device:
        """
        Helper method to get a proper torch.device object.
        
        Args:
            device_str: Optional device string ('cuda' or 'cpu')
            
        Returns:
            torch.device object
        """
        if device_str is None:
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return torch.device(device_str)


class BaseTranscriber(TranscriptionInterface):
    """Base class for transcription services with common caching functionality."""
    
    def __init__(
        self,
        source_language: str,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the base transcriber.
        
        Args:
            source_language: Source language code (e.g., 'en')
            device: Compute device ('cuda' or 'cpu')
            **kwargs: Additional implementation-specific parameters
        """
        self.source_language = source_language
        self.device = device
        self.torch_device = self.get_torch_device(device)
        self.debug_data = {
            "diarization": None,
            "transcription": None
        }
    
    def _generate_cache_key(self, audio_file: str, additional_params: str = "") -> str:
        """
        Generate a unique cache key based on MD5 hash of the audio file and processing parameters.
        
        Args:
            audio_file: Path to the audio file
            additional_params: Additional parameters to include in the key
            
        Returns:
            Unique cache key string
        """
        # Generate MD5 hash of audio file
        md5_hash = hashlib.md5()
        with open(audio_file, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        # Include processing parameters in the key to ensure uniqueness
        params = f"{self.source_language}{additional_params}"
        
        # Combine file hash and parameters
        return f"{md5_hash.hexdigest()}_{params}"
    
    def _get_cache_path(self, step_name: str) -> Path:
        """
        Get the cache directory path for a specific pipeline step.
        
        Args:
            step_name: Name of the pipeline step
            
        Returns:
            Path to the cache directory
        """
        cache_dir = Path("cache") / step_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def _cache_exists(self, step_name: str, cache_key: str) -> bool:
        """
        Check if cached results exist for a specific step and key.
        
        Args:
            step_name: Name of the pipeline step
            cache_key: Cache key
            
        Returns:
            True if cache exists, False otherwise
        """
        cache_file = self._get_cache_path(step_name) / f"{cache_key}.pkl"
        return cache_file.exists()
    
    def _save_to_cache(self, step_name: str, cache_key: str, data: Any) -> None:
        """
        Save data to cache.
        
        Args:
            step_name: Name of the pipeline step
            cache_key: Cache key
            data: Data to cache
        """
        cache_file = self._get_cache_path(step_name) / f"{cache_key}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
    
    def _load_from_cache(self, step_name: str, cache_key: str) -> Any:
        """
        Load data from cache.
        
        Args:
            step_name: Name of the pipeline step
            cache_key: Cache key
            
        Returns:
            Cached data
        """
        cache_file = self._get_cache_path(step_name) / f"{cache_key}.pkl"
        with open(cache_file, "rb") as f:
            return pickle.load(f) 