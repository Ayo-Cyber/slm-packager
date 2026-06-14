"""Shared pytest fixtures for SLM Packager tests."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from slm_packager.config.models import (
    DeviceType,
    GenerationParams,
    ModelConfig,
    RuntimeConfig,
    RuntimeType,
    SLMConfig,
)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that gets cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_models_dir(temp_dir):
    """Create a mock models directory structure."""
    models_dir = temp_dir / "models"
    configs_dir = temp_dir / "configs"
    models_dir.mkdir(parents=True)
    configs_dir.mkdir(parents=True)
    return {"models": models_dir, "configs": configs_dir}


@pytest.fixture
def sample_registry_data() -> Dict[str, Any]:
    """Provide sample model registry data for testing."""
    return {
        "version": "0.1.0",
        "models": {
            "test-model": {
                "name": "Test Model",
                "description": "A test model for unit tests",
                "format": "gguf",
                "runtime": "llama_cpp",
                "repo": "test/test-model",
                "variants": {
                    "q4_k_m": {
                        "file": "test-model.Q4_K_M.gguf",
                        "size": "100MB",
                        "speed": "fast",
                        "quality": "good",
                        "recommended": True,
                    },
                    "q8_0": {
                        "file": "test-model.Q8_0.gguf",
                        "size": "200MB",
                        "speed": "medium",
                        "quality": "excellent",
                        "recommended": False,
                    },
                },
            },
            "test-pytorch": {
                "name": "Test PyTorch Model",
                "description": "PyTorch test model",
                "format": "pytorch",
                "runtime": "transformers",
                "repo": "test/pytorch-model",
                "variants": {
                    "default": {
                        "file": "test-pytorch",
                        "size": "50MB",
                        "speed": "slow",
                        "quality": "good",
                        "recommended": True,
                    }
                },
            },
        },
    }


@pytest.fixture
def mock_registry_file(temp_dir, sample_registry_data):
    """Create a temporary registry JSON file."""
    registry_path = temp_dir / "models.json"
    with open(registry_path, "w") as f:
        json.dump(sample_registry_data, f)
    return registry_path


@pytest.fixture
def sample_gguf_config() -> SLMConfig:
    """Provide a sample GGUF model configuration."""
    return SLMConfig(
        model=ModelConfig(
            name="test-gguf",
            path="/path/to/test.gguf",
            format="gguf",
            description="Test GGUF model",
        ),
        runtime=RuntimeConfig(
            type=RuntimeType.LLAMA_CPP, device=DeviceType.CPU, threads=4, context_size=2048
        ),
        params=GenerationParams(temperature=0.7, max_tokens=512, stream=False),
    )


@pytest.fixture
def sample_transformers_config() -> SLMConfig:
    """Provide a sample Transformers model configuration."""
    return SLMConfig(
        model=ModelConfig(
            name="test-transformers",
            path="gpt2",
            format="pytorch",
            description="Test Transformers model",
        ),
        runtime=RuntimeConfig(type=RuntimeType.TRANSFORMERS, device=DeviceType.CPU),
        params=GenerationParams(temperature=0.8, max_tokens=256, stream=False),
    )


@pytest.fixture
def sample_config_yaml(temp_dir, sample_gguf_config) -> Path:
    """Create a sample YAML config file for testing."""
    from slm_packager.config.loader import ConfigLoader

    config_path = temp_dir / "test-config.yaml"
    ConfigLoader.save(sample_gguf_config, config_path)
    return config_path


@pytest.fixture
def mock_runtime():
    """Provide a mock runtime for testing without actual model loading."""
    mock = MagicMock()
    mock.is_loaded = True
    mock.load.return_value = None
    mock.unload.return_value = None
    mock.generate.return_value = "This is a test output from the mock runtime."
    return mock


@pytest.fixture
def mock_hf_download():
    """Mock huggingface_hub.hf_hub_download to avoid actual downloads."""
    with patch("slm_packager.registry.downloader.hf_hub_download") as mock:
        mock.return_value = "/fake/path/to/model.gguf"
        yield mock
