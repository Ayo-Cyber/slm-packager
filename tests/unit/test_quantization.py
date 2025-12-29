"""Unit tests for quantization module."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from slm_packager.quantization.binary_manager import BinaryManager


@pytest.mark.unit
class TestBinaryManager:
    """Test the BinaryManager class."""
    
    def test_binary_manager_class_exists(self):
        """Test that Bin aryManager can be imported."""
        assert BinaryManager is not None
    
    @patch('slm_packager.quantization.binary_manager.Path.home')
    def test_get_quantize_binary_returns_path(self, mock_home, temp_dir):
        """Test that get_quantize_binary returns a Path."""
        mock_home.return_value = temp_dir
        
        # Mock the binary existence
        with patch.object(Path, 'exists', return_value=True):
            path = BinaryManager.get_quantize_binary()
            
            assert isinstance(path, Path)
            assert "quantize" in str(path).lower()


@pytest.mark.unit
class TestQuantizer:
    """Test the Quantizer class."""
    
    def test_quantizer_requires_binary_manager(self):
        """Test that Quantizer can be imported."""
        from slm_packager.quantization import Quantizer
        
        assert Quantizer is not None
