"""Unit tests for model downloader."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY
import shutil

from slm_packager.registry.downloader import ModelDownloader
from slm_packager.registry import ModelRegistry

@pytest.fixture
def mock_registry():
    with patch('slm_packager.registry.downloader.ModelRegistry') as mock:
        registry_instance = mock.return_value
        yield registry_instance

@pytest.fixture
def downloader(mock_registry, temp_dir):
    with patch('slm_packager.registry.downloader.Path.home') as mock_home:
        mock_home.return_value = temp_dir
        # Mock disk usage to always return plenty of space by default
        with patch('slm_packager.registry.downloader.shutil.disk_usage') as mock_disk:
            mock_disk.return_value = (1000, 500, 100 * (1024**3)) # 100GB free
            yield ModelDownloader()

@pytest.mark.unit
class TestModelDownloader:
    """Test the ModelDownloader class."""
    
    def test_pull_requires_huggingface_hub_for_downloaded_models(self, mock_registry, temp_dir):
        """GGUF/ONNX downloads should fail clearly without huggingface-hub."""
        with patch('slm_packager.registry.downloader.Path.home', return_value=temp_dir):
            downloader = ModelDownloader()

        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.repo = "test/repo"
        mock_model.format = "gguf"
        mock_model.runtime = "llama_cpp"
        mock_model.variants = {
            "q4_k_m": MagicMock(file="model.gguf", size="500MB")
        }
        downloader.registry.get_model.return_value = mock_model

        with patch('slm_packager.registry.downloader.HF_AVAILABLE', False):
            with pytest.raises(ImportError) as exc_info:
                downloader.pull("test-model", "q4_k_m")

            assert "huggingface-hub" in str(exc_info.value)

    def test_pull_pytorch_model_without_huggingface_hub_still_creates_config(self, mock_registry, temp_dir):
        """PyTorch registry entries should still create configs without download support."""
        with patch('slm_packager.registry.downloader.Path.home', return_value=temp_dir):
            downloader = ModelDownloader()

        mock_model = MagicMock()
        mock_model.name = "test-pytorch"
        mock_model.repo = "test/repo"
        mock_model.format = "pytorch"
        mock_model.runtime = "transformers"
        mock_model.variants = {
            "default": MagicMock(file="test-pytorch", size="50MB")
        }
        downloader.registry.get_model.return_value = mock_model
        downloader.registry.get_recommended_variant.return_value = "default"

        with patch('slm_packager.registry.downloader.HF_AVAILABLE', False):
            result = downloader.pull("test-pytorch")

        assert result == Path("test/repo")
        assert (downloader.configs_dir / "test-pytorch.yaml").exists()
    
    def test_init_creates_directories(self, mock_registry, temp_dir):
        """Test that __init__ creates necessary directories."""
        with patch('slm_packager.registry.downloader.Path.home') as mock_home:
            mock_home.return_value = temp_dir
            ModelDownloader()
            
            assert (temp_dir / ".slm" / "models").exists()
            assert (temp_dir / ".slm" / "configs").exists()
            
    def test_check_disk_space_success(self, downloader):
        """Test disk space check passes when enough space."""
        # 100GB free vs 1GB required
        assert downloader._check_disk_space(1.0) is True

    def test_check_disk_space_failure(self, downloader):
        """Test disk space check fails when not enough space."""
        with patch('slm_packager.registry.downloader.shutil.disk_usage') as mock_disk:
            # 0.5GB free
            mock_disk.return_value = (1000, 999.5, 0.5 * (1024**3))
            assert downloader._check_disk_space(1.0) is False

    @patch('slm_packager.registry.downloader.HF_AVAILABLE', True)
    @patch('slm_packager.registry.downloader.hf_hub_download')
    def test_pull_gguf_model(self, mock_hf_download, downloader):
        """Test pulling a GGUF model."""
        # Setup mock registry
        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.repo = "test/repo"
        mock_model.format = "gguf"
        mock_model.runtime = "llama_cpp"
        mock_model.variants = {
            "q4_k_m": MagicMock(file="model.gguf", size="500MB")
        }
        downloader.registry.get_model.return_value = mock_model
        
        # Setup mock download return
        expected_path = downloader.models_dir / "model.gguf"
        mock_hf_download.return_value = str(expected_path)
        
        # Pull
        result = downloader.pull("test-model", "q4_k_m")
        
        # Verify
        assert result == Path(expected_path)
        mock_hf_download.assert_called_once_with(
            repo_id="test/repo",
            filename="model.gguf",
            cache_dir=str(downloader.models_dir),
            resume_download=True
        )
        
        # Verify config created
        config_path = downloader.configs_dir / "test-model.yaml"
        assert config_path.exists()

    def test_pull_not_found_in_registry(self, downloader):
        """Test pulling a model not in registry."""
        downloader.registry.get_model.return_value = None
        
        with pytest.raises(ValueError, match="not found in registry"):
            downloader.pull("unknown-model")

    @patch('slm_packager.registry.downloader.hf_hub_download')
    def test_pull_disk_full(self, mock_hf_download, downloader):
        """Test pulling fails when disk is full."""
        # Setup mock registry
        mock_model = MagicMock()
        mock_model.variants = {
            "q4_k_m": MagicMock(file="model.gguf", size="10GB")
        }
        downloader.registry.get_model.return_value = mock_model
        
        # Mock full disk (only 1GB free)
        with patch('slm_packager.registry.downloader.shutil.disk_usage') as mock_disk:
            mock_disk.return_value = (1000, 999, 1 * (1024**3))
            
            with pytest.raises(IOError, match="Not enough disk space"):
                downloader.pull("test-model", "q4_k_m")
        
        mock_hf_download.assert_not_called()

    @patch('slm_packager.registry.downloader.HF_AVAILABLE', True)
    @patch('slm_packager.registry.downloader.hf_hub_download')
    def test_pull_network_error(self, mock_hf_download, downloader):
        """Test proper handling of network errors."""
        mock_model = MagicMock()
        mock_model.variants = {"q4_k_m": MagicMock(size="1MB")}
        downloader.registry.get_model.return_value = mock_model
        
        # Mock network error
        mock_hf_download.side_effect = Exception("Connection reset")
        
        with pytest.raises(RuntimeError, match="Failed to download model"):
            downloader.pull("test-model", "q4_k_m")

    def test_list_installed(self, downloader):
        """Test listing installed models."""
        # Create a fake config and model file
        config_path = downloader.configs_dir / "test.yaml"
        model_path = downloader.models_dir / "model.gguf"
        
        # Write dummy model file (1MB)
        model_path.write_bytes(b"0" * 1024 * 1024)
        
        # Write config using ConfigLoader manually or just mock it, 
        # but integration with ConfigLoader is better.
        # Since we don't want to depend on ConfigLoader specifics too much here, 
        # let's just create a file that looks like yaml
        config_content = f"""
model:
  name: test
  path: {model_path}
  format: gguf
runtime:
  type: llama_cpp
"""
        config_path.write_text(config_content)
        
        installed = downloader.list_installed()
        
        assert len(installed) == 1
        assert installed[0]['name'] == 'test'
        assert installed[0]['format'] == 'gguf'
