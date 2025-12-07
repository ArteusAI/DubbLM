"""High-quality time-stretching for audio segments with voice timbre preservation."""

import os
import subprocess
import shutil
from typing import Optional, Literal
from pydub import AudioSegment

from ..core.log_config import get_logger

logger = get_logger(__name__)


class TimeStretcher:
    """High-quality time-stretching that preserves voice timbre."""
    
    METHODS = Literal["atempo", "atempo_chain", "auto"]
    
    def __init__(self, preferred_method: METHODS = "auto"):
        """Initialize time stretcher.

        Args:
            preferred_method: Preferred stretching method
                - "atempo": Fast but may affect timbre at high ratios
                - "atempo_chain": Chained atempo for better quality
                - "auto": Auto-select best available method
        """
        logger.info(f"Initializing TimeStretcher with preferred_method='{preferred_method}'")
        self.preferred_method = preferred_method
        # Backward compatibility: attribute kept for tests; RubberBand removed
        self._rubberband_available = False
        if self.preferred_method == "auto":
            logger.info("Auto-select mode enabled, will choose best method based on tempo ratio")
    
    # RubberBand support removed
    
    def stretch(
        self,
        input_path: str,
        output_path: str,
        tempo_ratio: float,
        method: Optional[METHODS] = None
    ) -> bool:
        """Apply time-stretching to audio file while preserving pitch and timbre.

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            tempo_ratio: Tempo ratio (< 1.0 = slower, > 1.0 = faster)
            method: Override the preferred method for this call

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Time-stretch requested: input='{input_path}', ratio={tempo_ratio:.3f}")

        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return False

        # Get input file info
        try:
            file_size = os.path.getsize(input_path)
            logger.debug(f"Input file size: {file_size} bytes")
        except Exception as e:
            logger.warning(f"Could not get input file size: {e}")

        if abs(tempo_ratio - 1.0) < 0.001:
            logger.info("Tempo ratio ~1.0, copying file without stretching")
            shutil.copy(input_path, output_path)
            return True

        method = method or self.preferred_method

        if method == "auto":
            method = self._select_best_method(tempo_ratio)
            logger.info(f"Auto-selected method: '{method}' for ratio {tempo_ratio:.3f}")
        elif method == "rubberband":
            logger.warning("Requested method 'rubberband' is disabled. Falling back to 'atempo_chain'.")
            method = "atempo_chain"

        logger.info(f"Applying time-stretch with method '{method}', ratio={tempo_ratio:.3f}")

        import time
        start_time = time.time()

        if method == "atempo_chain":
            success = self._stretch_atempo_chain(input_path, output_path, tempo_ratio)
        else:  # atempo
            success = self._stretch_atempo(input_path, output_path, tempo_ratio)

        elapsed = time.time() - start_time

        if success:
            try:
                output_size = os.path.getsize(output_path)
                logger.info(f"Time-stretch completed in {elapsed:.2f}s, output size: {output_size} bytes")
            except Exception:
                logger.info(f"Time-stretch completed in {elapsed:.2f}s")
        else:
            logger.error(f"Time-stretch failed after {elapsed:.2f}s")

        return success
    
    def _select_best_method(self, tempo_ratio: float) -> str:
        """Auto-select method optimized for speech quality.

        Prefers FFmpeg `atempo` for small changes and `atempo_chain` for larger
        ones.
        """
        delta = abs(tempo_ratio - 1.0)

        # Small change: single atempo yields very natural speech
        if delta <= 0.15:
            logger.debug(f"Selecting 'atempo' for small ratio change {tempo_ratio:.3f}")
            return "atempo"

        # Moderate to large change: chain atempo stages for better quality
        if delta <= 0.5:
            logger.debug(f"Selecting 'atempo_chain' for moderate ratio {tempo_ratio:.3f}")
            return "atempo_chain"

        # Very large change: stick to atempo_chain
        logger.debug(f"Selecting 'atempo_chain' for extreme ratio {tempo_ratio:.3f}")
        return "atempo_chain"
    
    # RubberBand implementation removed
    
    def _stretch_atempo(self, input_path: str, output_path: str, tempo_ratio: float) -> bool:
        """Apply simple atempo filter (fast but may affect timbre at high ratios).

        Note: atempo is limited to 0.5-2.0 range per filter.
        """
        logger.debug(f"Starting atempo stretch: ratio={tempo_ratio:.6f}")
        try:
            # Clamp to atempo limits
            if tempo_ratio < 0.5 or tempo_ratio > 2.0:
                logger.warning(f"Tempo ratio {tempo_ratio:.3f} outside atempo range [0.5-2.0], switching to chain method")
                return self._stretch_atempo_chain(input_path, output_path, tempo_ratio)

            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-filter:a", f"atempo={tempo_ratio:.6f}",
                "-vn",
                output_path
            ]

            logger.debug(f"atempo command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300
            )

            if result.returncode == 0 and os.path.exists(output_path):
                logger.info("atempo time-stretch successful")
                return True
            else:
                stderr_msg = result.stderr.decode().strip()
                logger.error(f"atempo failed with returncode={result.returncode}: {stderr_msg}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("atempo processing timed out after 300 seconds")
            return False
        except Exception as e:
            logger.error(f"atempo error: {e}", exc_info=True)
            return False
    
    def _stretch_atempo_chain(self, input_path: str, output_path: str, tempo_ratio: float) -> bool:
        """Apply chained atempo filters for better quality at extreme ratios.

        Breaks down large tempo changes into multiple smaller steps,
        which preserves quality better than a single large change.
        """
        logger.debug(f"Starting atempo chain stretch: ratio={tempo_ratio:.6f}")
        try:
            # Break down into chain of atempo filters (each limited to 0.5-2.0)
            filters = []
            remaining_ratio = tempo_ratio

            # Decompose ratio into multiple steps within atempo limits
            while abs(remaining_ratio - 1.0) > 0.001:
                if remaining_ratio > 2.0:
                    filters.append("atempo=2.0")
                    remaining_ratio /= 2.0
                elif remaining_ratio < 0.5:
                    filters.append("atempo=0.5")
                    remaining_ratio /= 0.5
                else:
                    filters.append(f"atempo={remaining_ratio:.6f}")
                    break

                # Safety limit
                if len(filters) > 10:
                    logger.error(f"Too many atempo stages required ({len(filters)}), aborting")
                    return False

            filter_chain = ",".join(filters)
            logger.info(f"Using {len(filters)}-stage atempo chain: {filter_chain}")

            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-filter:a", filter_chain,
                "-vn",
                output_path
            ]

            logger.debug(f"atempo chain command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300
            )

            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"atempo chain time-stretch successful ({len(filters)} stages)")
                return True
            else:
                stderr_msg = result.stderr.decode().strip()
                logger.error(f"atempo chain failed with returncode={result.returncode}: {stderr_msg}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("atempo chain processing timed out after 300 seconds")
            return False
        except Exception as e:
            logger.error(f"atempo chain error: {e}", exc_info=True)
            return False
    
    def stretch_audio_segment(
        self,
        audio: AudioSegment,
        tempo_ratio: float,
        method: Optional[METHODS] = None
    ) -> Optional[AudioSegment]:
        """Convenience method to stretch an AudioSegment object.

        Args:
            audio: Input AudioSegment
            tempo_ratio: Tempo ratio
            method: Optional method override

        Returns:
            Stretched AudioSegment or None if failed
        """
        import tempfile

        logger.info(f"Stretching AudioSegment: duration={len(audio)}ms, ratio={tempo_ratio:.3f}")

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:

                tmp_in_path = tmp_in.name
                tmp_out_path = tmp_out.name

            logger.debug(f"Created temp files: in={tmp_in_path}, out={tmp_out_path}")

            # Export to temp file
            logger.debug("Exporting AudioSegment to temp file...")
            audio.export(tmp_in_path, format="wav")

            # Apply stretching
            success = self.stretch(tmp_in_path, tmp_out_path, tempo_ratio, method)

            if success:
                logger.debug("Loading stretched audio from temp file...")
                result = AudioSegment.from_file(tmp_out_path)
                logger.info(f"AudioSegment stretch successful: new duration={len(result)}ms")
            else:
                logger.error("AudioSegment stretch failed")
                result = None

            # Cleanup
            logger.debug("Cleaning up temp files...")
            try:
                os.remove(tmp_in_path)
                os.remove(tmp_out_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {e}")

            return result

        except Exception as e:
            logger.error(f"Error in stretch_audio_segment: {e}", exc_info=True)
            return None

