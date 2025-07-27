"""Cache management for the Smart Dubbing system."""

import os
import hashlib
import pickle
import shutil
from pathlib import Path
from typing import Any, Optional


class CacheManager:
    """Manages caching for the Smart Dubbing pipeline."""
    
    def __init__(self, use_cache: bool = True, input_file: Optional[str] = None):
        """Initialize the cache manager.
        
        Args:
            use_cache: Whether to enable caching
            input_file: Path to the original input file (for hash-based cache organization)
        """
        self.use_cache = use_cache
        self.input_file = input_file
        self.cache_root = Path("cache")
        self.cache_root.mkdir(exist_ok=True)
    
    def set_input_file(self, input_file: str) -> None:
        """Set the input file path for cache organization.
        
        Args:
            input_file: Path to the original input file
        """
        self.input_file = input_file
    
    def generate_input_hash(self, input_file_path: str) -> str:
        """Generate a short hash based on input file name and size.
        
        Args:
            input_file_path: Path to the input file
            
        Returns:
            Short hash string (8 characters)
        """
        try:
            # Get file name and size
            file_path = Path(input_file_path)
            file_name = file_path.name
            file_size = file_path.stat().st_size
            
            # Create hash from name + size
            hash_input = f"{file_name}_{file_size}"
            hash_obj = hashlib.md5(hash_input.encode())
            return hash_obj.hexdigest()[:8]
        except (OSError, FileNotFoundError):
            # Fallback: use just the filename if file doesn't exist or can't be accessed
            file_name = Path(input_file_path).name
            hash_obj = hashlib.md5(file_name.encode())
            return hash_obj.hexdigest()[:8]
    
    def generate_cache_key(self, 
                          audio_file: str,
                          source_language: str,
                          target_language: str,
                          whisper_model: str,
                          start_time: Optional[float] = None,
                          duration: Optional[float] = None) -> str:
        """Generate a unique cache key based on MD5 hash of the audio file and processing parameters.
        
        Args:
            audio_file: Path to the audio file
            source_language: Source language code
            target_language: Target language code
            whisper_model: Whisper model name
            start_time: Start time in seconds
            duration: Duration in seconds
            
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
        params = f"{source_language}_{target_language}_{whisper_model}"
        if start_time is not None:
            params += f"_start_{start_time}"
        if duration is not None:
            params += f"_dur_{duration}"
        
        # Combine file hash and parameters
        return f"{md5_hash.hexdigest()}_{params}"
    
    def get_cache_path(self, step_name: str) -> Path:
        """Get the cache directory path for a specific pipeline step.
        
        Args:
            step_name: Name of the pipeline step
            
        Returns:
            Path to the cache directory
        """
        if self.input_file:
            # Use input-hash-based directory structure: ./cache/<hash>/<step_name>/
            input_hash = self.generate_input_hash(self.input_file)
            cache_dir = self.cache_root / input_hash / step_name
        else:
            # Fallback to old structure if no input file is set
            cache_dir = self.cache_root / step_name
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def cache_exists(self, step_name: str, cache_key: str) -> bool:
        """Check if cached results exist for a specific step and key.
        
        Args:
            step_name: Name of the pipeline step
            cache_key: Cache key
            
        Returns:
            True if cache exists, False otherwise
        """
        if not self.use_cache:
            return False
            
        cache_file = self.get_cache_path(step_name) / f"{cache_key}.pkl"
        return cache_file.exists()
    
    def save_to_cache(self, step_name: str, cache_key: str, data: Any) -> None:
        """Save data to cache.
        
        Args:
            step_name: Name of the pipeline step
            cache_key: Cache key
            data: Data to cache
        """
        if not self.use_cache:
            return
            
        cache_file = self.get_cache_path(step_name) / f"{cache_key}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
    
    def load_from_cache(self, step_name: str, cache_key: str) -> Any:
        """Load data from cache.
        
        Args:
            step_name: Name of the pipeline step
            cache_key: Cache key
            
        Returns:
            Cached data
        """
        if not self.use_cache:
            return None
            
        cache_file = self.get_cache_path(step_name) / f"{cache_key}.pkl"
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    def copy_cached_files(self, step_name: str, cache_key: str, pattern: str, destination_dir: str) -> None:
        """Copy cached files matching a pattern to a destination directory.
        
        Args:
            step_name: Name of the pipeline step
            cache_key: Cache key
            pattern: File pattern to match (e.g., "*.wav")
            destination_dir: Destination directory
        """
        if not self.use_cache:
            return
            
        cache_path = self.get_cache_path(step_name)
        cache_dir = cache_path / cache_key
        
        if cache_dir.exists():
            os.makedirs(destination_dir, exist_ok=True)
            for cached_file in cache_dir.glob(pattern):
                shutil.copy(cached_file, f"{destination_dir}/{cached_file.name}")
    
    def save_file_to_cache(self, step_name: str, cache_key: str, source_file: str, cache_filename: str) -> None:
        """Save a file to cache.
        
        Args:
            step_name: Name of the pipeline step
            cache_key: Cache key
            source_file: Source file path
            cache_filename: Name for the cached file
        """
        if not self.use_cache:
            return
            
        cache_dir = self.get_cache_path(step_name)
        cache_file = cache_dir / cache_filename
        
        if os.path.exists(source_file):
            shutil.copy(source_file, cache_file)
    
    def load_file_from_cache(self, step_name: str, cache_key: str, cache_filename: str, destination_file: str) -> bool:
        """Load a file from cache.
        
        Args:
            step_name: Name of the pipeline step
            cache_key: Cache key
            cache_filename: Name of the cached file
            destination_file: Destination file path
            
        Returns:
            True if file was loaded successfully, False otherwise
        """
        if not self.use_cache:
            return False
            
        cache_dir = self.get_cache_path(step_name)
        cache_file = cache_dir / cache_filename
        
        if cache_file.exists():
            shutil.copy(cache_file, destination_file)
            return True
        return False
    
    def clear_cache(self, step_name: Optional[str] = None) -> None:
        """Clear cache for a specific step or all steps.
        
        Args:
            step_name: Name of the pipeline step to clear, or None to clear all
        """
        if step_name:
            cache_dir = self.get_cache_path(step_name)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        else:
            if self.cache_root.exists():
                shutil.rmtree(self.cache_root)
        
        # Recreate the cache directory
        self.cache_root.mkdir(exist_ok=True)
        if step_name:
            self.get_cache_path(step_name)
    
    def clear_input_cache(self, input_file: Optional[str] = None) -> None:
        """Clear cache for a specific input file.
        
        Args:
            input_file: Path to the input file, or None to use the current input file
        """
        if not input_file:
            input_file = self.input_file
        
        if input_file:
            input_hash = self.generate_input_hash(input_file)
            input_cache_dir = self.cache_root / input_hash
            if input_cache_dir.exists():
                shutil.rmtree(input_cache_dir) 