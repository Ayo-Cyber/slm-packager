"""Tests for the bug fixes and new features added during code review."""
import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from pydantic import ValidationError

from slm_packager.config.models import (
    SLMConfig, ModelConfig, RuntimeConfig, RuntimeType,
    DeviceType, GenerationParams
)
from slm_packager.config.loader import ConfigLoader


@pytest.mark.unit
class TestTrustRemoteCodeConfig:
    """Test the trust_remote_code field on RuntimeConfig."""

    def test_default_is_false(self):
        """trust_remote_code should default to False for security."""
        config = RuntimeConfig(type=RuntimeType.LLAMA_CPP)
        assert config.trust_remote_code is False

    def test_explicit_true(self):
        """trust_remote_code can be set to True explicitly."""
        config = RuntimeConfig(type=RuntimeType.TRANSFORMERS, trust_remote_code=True)
        assert config.trust_remote_code is True

    def test_roundtrip_yaml(self, temp_dir):
        """trust_remote_code should survive save/load roundtrip."""
        config = SLMConfig(
            model=ModelConfig(name="test", path="/test", format="pytorch"),
            runtime=RuntimeConfig(type=RuntimeType.TRANSFORMERS, trust_remote_code=True),
        )
        path = temp_dir / "test.yaml"
        ConfigLoader.save(config, path)
        loaded = ConfigLoader.load(path)
        assert loaded.runtime.trust_remote_code is True

    def test_roundtrip_yaml_default(self, temp_dir):
        """Default trust_remote_code=False should roundtrip correctly."""
        config = SLMConfig(
            model=ModelConfig(name="test", path="/test", format="gguf"),
            runtime=RuntimeConfig(type=RuntimeType.LLAMA_CPP),
        )
        path = temp_dir / "test.yaml"
        ConfigLoader.save(config, path)
        loaded = ConfigLoader.load(path)
        assert loaded.runtime.trust_remote_code is False


@pytest.mark.unit
class TestVersionSingleSource:
    """Test that the version string is single-sourced."""

    def test_version_exists(self):
        from slm_packager import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_server_uses_package_version(self):
        """server.py source should import __version__ from the package."""
        server_path = Path(__file__).parent.parent.parent / "slm_packager" / "api" / "server.py"
        source = server_path.read_text()
        assert "from .. import __version__" in source
        assert "version=__version__" in source


@pytest.mark.unit
class TestStreamingDetection:
    """Test that the streaming detection logic correctly distinguishes types."""

    def test_string_not_detected_as_stream(self):
        """Strings should not be treated as streaming iterators."""
        result = "hello world"
        # The server checks __next__ or __aiter__, strings have neither
        assert not hasattr(result, '__next__')
        assert not hasattr(result, '__aiter__')

    def test_generator_detected_as_stream(self):
        """Generators should be detected as streaming."""
        def gen():
            yield "chunk1"
            yield "chunk2"
        result = gen()
        assert hasattr(result, '__next__')

    def test_iterator_detected_as_stream(self):
        """Iterators should be detected as streaming."""
        result = iter(["chunk1", "chunk2"])
        assert hasattr(result, '__next__')

    def test_list_not_detected_as_stream(self):
        """Lists should not be treated as streaming (they lack __next__)."""
        result = ["chunk1", "chunk2"]
        assert not hasattr(result, '__next__')
        assert not hasattr(result, '__aiter__')


@pytest.mark.unit
class TestManagerFixes:
    """Test fixes to the ModelManager."""

    def test_lifecycle_typo_fixed(self):
        """Docstring should say 'lifecycle', not 'lifestyle'."""
        from slm_packager.api.manager import ModelManager
        assert "lifecycle" in ModelManager.__doc__
        assert "lifestyle" not in ModelManager.__doc__


@pytest.mark.unit
class TestModelDelete:
    """Test the model deletion functionality."""

    def test_delete_nonexistent_model(self, temp_dir):
        """Deleting a non-existent model should return False."""
        with patch.object(
            __import__('slm_packager.registry.downloader', fromlist=['ModelDownloader']).ModelDownloader,
            '__init__', lambda self: None
        ):
            from slm_packager.registry.downloader import ModelDownloader
            downloader = ModelDownloader.__new__(ModelDownloader)
            downloader.configs_dir = temp_dir / "configs"
            downloader.configs_dir.mkdir(parents=True, exist_ok=True)
            downloader.models_dir = temp_dir / "models"

            result = downloader.delete("nonexistent")
            assert result is False

    def test_delete_existing_model(self, temp_dir):
        """Deleting an existing model should remove config and return True."""
        from slm_packager.registry.downloader import ModelDownloader

        downloader = ModelDownloader.__new__(ModelDownloader)
        downloader.configs_dir = temp_dir / "configs"
        downloader.configs_dir.mkdir(parents=True, exist_ok=True)
        downloader.models_dir = temp_dir / "models"
        downloader.models_dir.mkdir(parents=True, exist_ok=True)

        # Create a fake model file and config
        model_file = temp_dir / "models" / "test.gguf"
        model_file.write_bytes(b"fake model data")

        config = SLMConfig(
            model=ModelConfig(name="test", path=str(model_file), format="gguf"),
            runtime=RuntimeConfig(type=RuntimeType.LLAMA_CPP),
        )
        config_path = downloader.configs_dir / "test.yaml"
        ConfigLoader.save(config, config_path)

        result = downloader.delete("test")
        assert result is True
        assert not config_path.exists()
        assert not model_file.exists()


@pytest.mark.unit
class TestDefaultBindAddress:
    """Test that the API server defaults to localhost."""

    def test_server_defaults_to_localhost(self):
        """start_server should default to 127.0.0.1, not 0.0.0.0."""
        import inspect
        from slm_packager.api.server import start_server
        sig = inspect.signature(start_server)
        assert sig.parameters['host'].default == "127.0.0.1"

    def test_cli_serve_defaults_to_localhost(self):
        """CLI serve command should default to 127.0.0.1."""
        from slm_packager.cli.main import serve
        # Click stores defaults in the params
        for param in serve.params:
            if param.name == "host":
                assert param.default == "127.0.0.1"
                break
        else:
            pytest.fail("No 'host' parameter found on serve command")


@pytest.mark.unit
class TestBenchmarkTokenCounting:
    """Test that the benchmarker uses tokenizer when available."""

    def test_count_tokens_with_tokenizer(self):
        """When runtime has a tokenizer, use it for accurate counts."""
        from slm_packager.evaluation.benchmark import Benchmarker

        config = SLMConfig(
            model=ModelConfig(name="test", path="/test", format="gguf"),
            runtime=RuntimeConfig(type=RuntimeType.LLAMA_CPP),
        )

        with patch('slm_packager.evaluation.benchmark.get_runtime') as mock_get:
            mock_runtime = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_runtime.tokenizer = mock_tokenizer
            mock_get.return_value = mock_runtime

            benchmarker = Benchmarker(config)
            count = benchmarker._count_tokens("hello world")
            assert count == 5
            mock_tokenizer.encode.assert_called_once_with("hello world")

    def test_count_tokens_fallback(self):
        """When runtime has no tokenizer, fall back to char estimate."""
        from slm_packager.evaluation.benchmark import Benchmarker

        config = SLMConfig(
            model=ModelConfig(name="test", path="/test", format="gguf"),
            runtime=RuntimeConfig(type=RuntimeType.LLAMA_CPP),
        )

        with patch('slm_packager.evaluation.benchmark.get_runtime') as mock_get:
            mock_runtime = MagicMock()
            mock_runtime.tokenizer = None
            mock_runtime.model = None
            mock_get.return_value = mock_runtime

            benchmarker = Benchmarker(config)
            # 20 chars // 4 = 5
            count = benchmarker._count_tokens("a" * 20)
            assert count == 5


@pytest.mark.unit
class TestSlmRmCommand:
    """Test the slm rm CLI command exists and has correct options."""

    def test_rm_command_registered(self):
        """The rm command should be registered on the CLI group."""
        from slm_packager.cli.main import cli
        command_names = [cmd for cmd in cli.commands]
        assert "rm" in command_names

    def test_rm_has_yes_flag(self):
        """The rm command should have a --yes/-y skip-confirmation flag."""
        from slm_packager.cli.main import rm
        param_names = [p.name for p in rm.params]
        assert "yes" in param_names
