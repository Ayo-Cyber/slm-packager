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

    def test_runtime_modules_are_not_imported_eagerly(self):
        """Importing the package must not pull torch, llama_cpp, or onnxruntime.

        These are optional extras. If `slm_packager.runtime` imported them at import
        time, a lean install (`pip install slm-packager`) could not even run `slm
        list` — which is exactly the failure this lazy factory prevents.
        """
        import subprocess
        import sys

        # A clean interpreter, so nothing another test imported can mask a regression.
        code = (
            "import slm_packager.runtime, sys;"
            "heavy=[m for m in ('torch','llama_cpp','onnxruntime') if m in sys.modules];"
            "print(','.join(heavy))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "", f"eagerly imported: {result.stdout.strip()}"

    def test_missing_engine_names_the_extra_to_install(self, sample_gguf_config):
        """The error must tell the user which extra provides the missing engine."""
        import slm_packager.runtime as runtime_pkg

        with patch.object(
            runtime_pkg.importlib, "import_module", side_effect=ImportError("No module named x")
        ):
            with pytest.raises(ImportError) as exc:
                get_runtime(sample_gguf_config)

        message = str(exc.value)
        assert "slm-packager[gguf]" in message
        assert "No module named x" in message

    def test_every_runtime_type_maps_to_an_extra(self):
        """A new RuntimeType must come with an install target, or errors are useless."""
        from slm_packager.config.models import RuntimeType as RT
        from slm_packager.runtime import _RUNTIMES

        assert set(_RUNTIMES) == set(RT)
        for _module, _cls, extra in _RUNTIMES.values():
            assert extra in {"gguf", "torch", "onnx"}
