"""Unit tests for runtime base classes and factory."""

from unittest.mock import MagicMock, patch

import pytest

from slm_packager.config.models import RuntimeType
from slm_packager.runtime import get_runtime
from slm_packager.runtime.base import BaseRuntime


@pytest.mark.unit
class TestBaseRuntime:
    """Test the abstract BaseRuntime class."""

    def test_base_runtime_is_abstract(self, sample_gguf_config):
        """Test that BaseRuntime cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRuntime(sample_gguf_config)

    def test_mock_runtime_implementation(self, sample_gguf_config):
        """Test a mock runtime implementation."""

        # Create a concrete implementation
        class MockRuntime(BaseRuntime):
            def load(self):
                self.model = "mock_model"

            def generate(self, prompt, params):
                return f"Generated from: {prompt}"

            def unload(self):
                self.model = None

        runtime = MockRuntime(sample_gguf_config)

        assert runtime.config == sample_gguf_config
        assert runtime.model is None
        assert not runtime.is_loaded

        runtime.load()
        assert runtime.is_loaded

        output = runtime.generate("test", sample_gguf_config.params)
        assert "test" in output

        runtime.unload()
        assert not runtime.is_loaded


@pytest.mark.unit
class TestRuntimeFactory:
    """Test the runtime factory function."""

    def test_get_llama_cpp_runtime(self, sample_gguf_config):
        """Test creating a llama.cpp runtime."""
        with patch("slm_packager.runtime.LlamaCppRuntime") as MockRuntime:
            mock_instance = MagicMock()
            MockRuntime.return_value = mock_instance

            runtime = get_runtime(sample_gguf_config)

            MockRuntime.assert_called_once_with(sample_gguf_config)
            assert runtime == mock_instance

    def test_get_transformers_runtime(self, sample_transformers_config):
        """Test creating a Transformers runtime."""
        with patch("slm_packager.runtime.TransformersRuntime") as MockRuntime:
            mock_instance = MagicMock()
            MockRuntime.return_value = mock_instance

            runtime = get_runtime(sample_transformers_config)

            MockRuntime.assert_called_once_with(sample_transformers_config)
            assert runtime == mock_instance

    @pytest.mark.skip("ONNX runtime is experimental - not fully implemented")
    def test_get_onnx_runtime(self, sample_gguf_config):
        """Test creating an ONNX runtime."""
        pass

    def test_get_runtime_invalid_type(self, sample_gguf_config):
        """Test that invalid runtime type raises error."""
        sample_gguf_config.runtime.type = "invalid_runtime"

        with pytest.raises((ValueError, KeyError, AttributeError)):
            get_runtime(sample_gguf_config)
