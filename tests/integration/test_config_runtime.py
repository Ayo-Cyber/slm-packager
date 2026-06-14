"""Integration tests for config loading and runtime initialization."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slm_packager.config.loader import ConfigLoader
from slm_packager.runtime import get_runtime


@pytest.mark.integration
class TestConfigToRuntime:
    """Test the integration between config loading and runtime initialization."""

    def test_load_config_and_create_runtime(self, sample_config_yaml):
        """Test loading YAML config and creating runtime."""
        config = ConfigLoader.load(sample_config_yaml)

        with patch("slm_packager.runtime.LlamaCppRuntime") as MockRuntime:
            mock_instance = MagicMock()
            MockRuntime.return_value = mock_instance

            runtime = get_runtime(config)

            assert runtime is not None
            MockRuntime.assert_called_once_with(config)

    def test_config_validation_errors_propagate(self, temp_dir):
        """Test that config validation errors are properly raised."""
        # Create invalid config YAML
        invalid_yaml = temp_dir / "invalid.yaml"
        invalid_yaml.write_text("""
model:
  name: test
  path: /path
  format: gguf
runtime:
  type: llama_cpp
  threads: -1  # Invalid: must be >= 1
""")

        with pytest.raises(Exception):  # Should raise validation error
            ConfigLoader.load(invalid_yaml)

    def test_runtime_switching(self, sample_gguf_config, sample_transformers_config):
        """Test switching between different runtime types."""
        # GGUF → llama.cpp
        with patch("slm_packager.runtime.LlamaCppRuntime") as MockLlama:
            mock_llama = MagicMock()
            MockLlama.return_value = mock_llama

            runtime1 = get_runtime(sample_gguf_config)
            MockLlama.assert_called_once()

        # PyTorch → Transformers
        with patch("slm_packager.runtime.TransformersRuntime") as MockTransformers:
            mock_transformers = MagicMock()
            MockTransformers.return_value = mock_transformers

            runtime2 = get_runtime(sample_transformers_config)
            MockTransformers.assert_called_once()

    def test_roundtrip_save_load_runtime(self, temp_dir, sample_gguf_config):
        """Test saving config, reloading it, and creating runtime."""
        config_path = temp_dir / "roundtrip.yaml"

        # Save
        ConfigLoader.save(sample_gguf_config, config_path)

        # Load
        loaded_config = ConfigLoader.load(config_path)

        # Create runtime from loaded config
        with patch("slm_packager.runtime.LlamaCppRuntime") as MockRuntime:
            mock_instance = MagicMock()
            MockRuntime.return_value = mock_instance

            runtime = get_runtime(loaded_config)

            # Verify config was preserved
            call_args = MockRuntime.call_args[0][0]
            assert call_args.model.name == sample_gguf_config.model.name
            assert call_args.runtime.type == sample_gguf_config.runtime.type
