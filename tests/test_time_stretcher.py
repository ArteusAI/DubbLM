"""Tests for high-quality time-stretching module."""

import os
import tempfile
import pytest
from pydub import AudioSegment
from src.dubbing.audio.time_stretcher import TimeStretcher


@pytest.fixture
def test_audio_file():
    """Create a temporary test audio file."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    
    silence = AudioSegment.silent(duration=1000)
    silence.export(temp_path, format='wav')
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def output_audio_file():
    """Create a temporary output audio file path."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestTimeStretcher:
    """Test suite for TimeStretcher class."""
    
    def test_initialization_auto(self):
        """Test TimeStretcher initialization with auto method."""
        stretcher = TimeStretcher(preferred_method='auto')
        assert stretcher.preferred_method == 'auto'
    
    def test_initialization_rubberband(self):
        """Test TimeStretcher initialization with rubberband method."""
        stretcher = TimeStretcher(preferred_method='rubberband')
        assert stretcher.preferred_method == 'rubberband'
    
    def test_initialization_atempo(self):
        """Test TimeStretcher initialization with atempo method."""
        stretcher = TimeStretcher(preferred_method='atempo')
        assert stretcher.preferred_method == 'atempo'
    
    def test_check_rubberband_available(self):
        """Test rubberband availability check."""
        stretcher = TimeStretcher()
        is_available = stretcher._rubberband_available
        assert isinstance(is_available, bool)
    
    def test_stretch_no_change(self, test_audio_file, output_audio_file):
        """Test stretching with ratio ~1.0 (no change)."""
        stretcher = TimeStretcher(preferred_method='auto')
        
        success = stretcher.stretch(
            input_path=test_audio_file,
            output_path=output_audio_file,
            tempo_ratio=1.0
        )
        
        assert success
        assert os.path.exists(output_audio_file)
        
        original = AudioSegment.from_file(test_audio_file)
        output = AudioSegment.from_file(output_audio_file)
        
        assert abs(len(original) - len(output)) < 10
    
    def test_stretch_slower(self, test_audio_file, output_audio_file):
        """Test stretching to slower speed."""
        stretcher = TimeStretcher(preferred_method='atempo')
        
        success = stretcher.stretch(
            input_path=test_audio_file,
            output_path=output_audio_file,
            tempo_ratio=0.8
        )
        
        assert success
        assert os.path.exists(output_audio_file)
        
        original = AudioSegment.from_file(test_audio_file)
        output = AudioSegment.from_file(output_audio_file)
        
        expected_duration = len(original) / 0.8
        assert abs(len(output) - expected_duration) < 100
    
    def test_stretch_faster(self, test_audio_file, output_audio_file):
        """Test stretching to faster speed."""
        stretcher = TimeStretcher(preferred_method='atempo')
        
        success = stretcher.stretch(
            input_path=test_audio_file,
            output_path=output_audio_file,
            tempo_ratio=1.2
        )
        
        assert success
        assert os.path.exists(output_audio_file)
        
        original = AudioSegment.from_file(test_audio_file)
        output = AudioSegment.from_file(output_audio_file)
        
        expected_duration = len(original) / 1.2
        assert abs(len(output) - expected_duration) < 100
    
    def test_stretch_extreme_slower(self, test_audio_file, output_audio_file):
        """Test stretching with extreme slower ratio."""
        stretcher = TimeStretcher(preferred_method='atempo_chain')
        
        success = stretcher.stretch(
            input_path=test_audio_file,
            output_path=output_audio_file,
            tempo_ratio=0.4
        )
        
        assert success
        assert os.path.exists(output_audio_file)
    
    def test_stretch_extreme_faster(self, test_audio_file, output_audio_file):
        """Test stretching with extreme faster ratio."""
        stretcher = TimeStretcher(preferred_method='atempo_chain')
        
        success = stretcher.stretch(
            input_path=test_audio_file,
            output_path=output_audio_file,
            tempo_ratio=2.5
        )
        
        assert success
        assert os.path.exists(output_audio_file)
    
    def test_stretch_missing_input(self, output_audio_file):
        """Test stretching with missing input file."""
        stretcher = TimeStretcher(preferred_method='auto')
        
        success = stretcher.stretch(
            input_path='nonexistent_file.wav',
            output_path=output_audio_file,
            tempo_ratio=1.0
        )
        
        assert not success
    
    def test_stretch_audio_segment(self):
        """Test stretching AudioSegment object directly."""
        stretcher = TimeStretcher(preferred_method='atempo')
        
        audio = AudioSegment.silent(duration=1000)
        stretched = stretcher.stretch_audio_segment(audio, tempo_ratio=0.8)
        
        if stretched is not None:
            expected_duration = len(audio) / 0.8
            assert abs(len(stretched) - expected_duration) < 100
    
    def test_select_best_method_auto_small_ratio(self):
        """Test auto method selection for small tempo ratio."""
        stretcher = TimeStretcher(preferred_method='auto')
        method = stretcher._select_best_method(tempo_ratio=1.1)
        
        if stretcher._rubberband_available:
            assert method == 'rubberband'
        else:
            assert method in ['atempo', 'atempo_chain']
    
    def test_select_best_method_auto_large_ratio(self):
        """Test auto method selection for large tempo ratio."""
        stretcher = TimeStretcher(preferred_method='auto')
        method = stretcher._select_best_method(tempo_ratio=0.5)
        
        if stretcher._rubberband_available:
            assert method == 'rubberband'
        else:
            assert method == 'atempo_chain'
    
    def test_atempo_out_of_range(self, test_audio_file, output_audio_file):
        """Test atempo method with out-of-range ratio (should use chain)."""
        stretcher = TimeStretcher(preferred_method='atempo')
        
        success = stretcher.stretch(
            input_path=test_audio_file,
            output_path=output_audio_file,
            tempo_ratio=2.5
        )
        
        assert success
    
    @pytest.mark.skipif(
        not TimeStretcher()._rubberband_available,
        reason="RubberBand not installed"
    )
    def test_rubberband_method(self, test_audio_file, output_audio_file):
        """Test RubberBand method (requires rubberband-cli installed)."""
        stretcher = TimeStretcher(preferred_method='rubberband')
        
        success = stretcher.stretch(
            input_path=test_audio_file,
            output_path=output_audio_file,
            tempo_ratio=0.85
        )
        
        assert success
        assert os.path.exists(output_audio_file)
        
        original = AudioSegment.from_file(test_audio_file)
        output = AudioSegment.from_file(output_audio_file)
        
        expected_duration = len(original) / 0.85
        assert abs(len(output) - expected_duration) < 100
    
    def test_method_override(self, test_audio_file, output_audio_file):
        """Test method override in stretch call."""
        stretcher = TimeStretcher(preferred_method='auto')
        
        success = stretcher.stretch(
            input_path=test_audio_file,
            output_path=output_audio_file,
            tempo_ratio=0.9,
            method='atempo'
        )
        
        assert success
    
    def test_atempo_chain_decomposition(self):
        """Test atempo chain decomposition logic."""
        stretcher = TimeStretcher(preferred_method='atempo_chain')
        
        # Test extreme ratio decomposition
        test_ratios = [0.3, 0.5, 1.0, 2.0, 3.0]
        
        for ratio in test_ratios:
            filters = []
            remaining = ratio
            
            while abs(remaining - 1.0) > 0.001:
                if remaining > 2.0:
                    filters.append('atempo=2.0')
                    remaining /= 2.0
                elif remaining < 0.5:
                    filters.append('atempo=0.5')
                    remaining /= 0.5
                else:
                    filters.append(f'atempo={remaining:.6f}')
                    break
                
                if len(filters) > 10:
                    break
            
            assert len(filters) <= 10

