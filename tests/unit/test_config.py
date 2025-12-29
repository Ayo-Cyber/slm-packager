"""Unit tests for configuration models and loading."""
import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from slm_packager.config.models import (
    SLMConfig, ModelConfig, RuntimeConfig, GenerationParams,
    RuntimeType, DeviceType, QuantizationType
)
from slm_packager.config.loader import ConfigLoader


@pytest.mark.unit
class TestConfigModels:
    """Test Pydantic configuration models."""
    
    def test_model_config_valid(self):
        """Test creating a valid ModelConfig."""
        config = ModelConfig(
            name="test-model",
            path="/path/to/model",
            format="gguf",
            description="Test model"
        )
        
        assert config.name == "test-model"
        assert config.path == "/path/to/model"
        assert config.format == "gguf"
        assert config.description == "Test model"
    
    def test_model_config_optional_description(self):
        """Test ModelConfig with optional description."""
        config = ModelConfig(
            name="test",
            path="/path",
            format="gguf"
        )
        
        assert config.description is None
    
    def test_runtime_config_defaults(self):
        """Test RuntimeConfig with default values."""
        config = RuntimeConfig(type=RuntimeType.LLAMA_CPP)
        
        assert config.type == RuntimeType.LLAMA_CPP
        assert config.device == DeviceType.CPU
        assert config.threads == 4
        assert config.gpu_layers == 0
        assert config.context_size == 2048
    
    def test_runtime_config_custom_values(self):
        """Test RuntimeConfig with custom values."""
        config = RuntimeConfig(
            type=RuntimeType.TRANSFORMERS,
            device=DeviceType.CUDA,
            threads=8,
            gpu_layers=32,
            context_size=4096
        )
        
        assert config.threads == 8
        assert config.gpu_layers == 32
        assert config.context_size == 4096
    
    def test_runtime_config_validation_threads(self):
        """Test that threads must be >= 1."""
        with pytest.raises(ValidationError):
            RuntimeConfig(type=RuntimeType.LLAMA_CPP, threads=0)
    
    def test_runtime_config_validation_gpu_layers(self):
        """Test that gpu_layers must be >= 0."""
        with pytest.raises(ValidationError):
            RuntimeConfig(type=RuntimeType.LLAMA_CPP, gpu_layers=-1)
    
    def test_runtime_config_validation_context_size(self):
        """Test that context_size must be >= 512."""
        with pytest.raises(ValidationError):
            RuntimeConfig(type=RuntimeType.LLAMA_CPP, context_size=256)
    
    def test_generation_params_defaults(self):
        """Test GenerationParams default values."""
        params = GenerationParams()
        
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 40
        assert params.max_tokens == 512
        assert params.stop == []
        assert params.stream is False
    
    def test_generation_params_validation_temperature(self):
        """Test temperature must be in range [0.0, 2.0]."""
        # Valid values
        GenerationParams(temperature=0.0)
        GenerationParams(temperature=2.0)
        
        # Invalid values
        with pytest.raises(ValidationError):
            GenerationParams(temperature=-0.1)
        with pytest.raises(ValidationError):
            GenerationParams(temperature=2.1)
    
    def test_generation_params_validation_top_p(self):
        """Test top_p must be in range [0.0, 1.0]."""
        GenerationParams(top_p=0.0)
        GenerationParams(top_p=1.0)
        
        with pytest.raises(ValidationError):
            GenerationParams(top_p=1.1)
    
    def test_slm_config_complete(self, sample_gguf_config):
        """Test complete SLMConfig."""
        assert sample_gguf_config.model.name == "test-gguf"
        assert sample_gguf_config.runtime.type == RuntimeType.LLAMA_CPP
        assert sample_gguf_config.params.temperature == 0.7
    
    def test_runtime_type_enum(self):
        """Test RuntimeType enum values."""
        assert RuntimeType.LLAMA_CPP == "llama_cpp"
        assert RuntimeType.TRANSFORMERS == "transformers"
        assert RuntimeType.ONNX == "onnx"
    
    def test_device_type_enum(self):
        """Test DeviceType enum values."""
        assert DeviceType.CPU == "cpu"
        assert DeviceType.CUDA == "cuda"
        assert DeviceType.MPS == "mps"
    
    def test_quantization_type_enum(self):
        """Test QuantizationType enum values."""
        assert QuantizationType.Q4_K_M == "q4_k_m"
        assert QuantizationType.Q8_0 == "q8_0"


@pytest.mark.unit
class TestConfigLoader:
    """Test ConfigLoader for YAML serialization."""
    
    def test_load_valid_config(self, sample_config_yaml):
        """Test loading a valid YAML config."""
        config = ConfigLoader.load(sample_config_yaml)
        
        assert isinstance(config, SLMConfig)
        assert config.model.name == "test-gguf"
        assert config.model.format == "gguf"
        assert config.runtime.type == RuntimeType.LLAMA_CPP
    
    def test_save_config(self, temp_dir, sample_gguf_config):
        """Test saving a config to YAML."""
        output_path = temp_dir / "output.yaml"
        ConfigLoader.save(sample_gguf_config, output_path)
        
        assert output_path.exists()
        
        # Verify YAML is valid
        with open(output_path) as f:
            data = yaml.safe_load(f)
        
        assert data["model"]["name"] == "test-gguf"
        assert data["runtime"]["type"] == "llama_cpp"
    
    def test_save_and_load_roundtrip(self, temp_dir, sample_transformers_config):
        """Test that save → load preserves config."""
        path = temp_dir / "roundtrip.yaml"
        
        # Save
        ConfigLoader.save(sample_transformers_config, path)
        
        # Load
        loaded = ConfigLoader.load(path)
        
        assert loaded.model.name == sample_transformers_config.model.name
        assert loaded.model.path == sample_transformers_config.model.path
        assert loaded.runtime.type == sample_transformers_config.runtime.type
        assert loaded.params.temperature == sample_transformers_config.params.temperature
    
    def test_load_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML raises error."""
        invalid_yaml = temp_dir / "invalid.yaml"
        invalid_yaml.write_text("{ invalid yaml content")
        
        with pytest.raises(Exception):  # YAML parse error
            ConfigLoader.load(invalid_yaml)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load(Path("/nonexistent/config.yaml"))
