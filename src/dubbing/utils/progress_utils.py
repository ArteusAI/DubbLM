"""Progress utilities for capturing tqdm output."""

from typing import Callable, Optional
from contextlib import contextmanager
import sys
import tqdm


class CallbackTqdm(tqdm.tqdm):
    """Custom tqdm class that calls a progress callback on each update."""
    
    _progress_callback: Optional[Callable[[int, int, str], None]] = None
    _prefix: str = ""
    _last_reported: int = -1
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CallbackTqdm._last_reported = -1
        if CallbackTqdm._progress_callback and self.total:
            CallbackTqdm._progress_callback(0, self.total, CallbackTqdm._prefix)
            CallbackTqdm._last_reported = 0
    
    def update(self, n: int = 1):
        result = super().update(n)
        if CallbackTqdm._progress_callback and self.total:
            current = int(self.n)
            # Only report if progress changed
            if current != CallbackTqdm._last_reported:
                CallbackTqdm._progress_callback(current, self.total, CallbackTqdm._prefix)
                CallbackTqdm._last_reported = current
        return result


@contextmanager
def tqdm_progress_callback(
    callback: Optional[Callable[[int, int, str], None]] = None,
    prefix: str = ""
):
    """Context manager that patches tqdm to use a progress callback.
    
    Args:
        callback: Function called with (current, total, prefix) on each update
        prefix: Optional prefix string passed to the callback
        
    Usage:
        with tqdm_progress_callback(my_callback, "Processing"):
            separator.separate(audio_file)
    """
    if not callback:
        yield
        return
    
    # Store original tqdm class
    original_tqdm_class = tqdm.tqdm
    
    # Set up callback
    CallbackTqdm._progress_callback = callback
    CallbackTqdm._prefix = prefix
    
    # Patch tqdm module
    tqdm.tqdm = CallbackTqdm
    
    # Also patch in audio_separator modules that already imported tqdm
    patched_modules = []
    for mod_name, mod in list(sys.modules.items()):
        if mod and 'audio_separator' in mod_name:
            if hasattr(mod, 'tqdm') and mod.tqdm is original_tqdm_class:
                mod.tqdm = CallbackTqdm
                patched_modules.append((mod_name, mod))
    
    try:
        yield
    finally:
        # Restore original tqdm
        tqdm.tqdm = original_tqdm_class
        CallbackTqdm._progress_callback = None
        CallbackTqdm._prefix = ""
        
        # Restore patched modules
        for mod_name, mod in patched_modules:
            if hasattr(mod, 'tqdm'):
                mod.tqdm = original_tqdm_class

