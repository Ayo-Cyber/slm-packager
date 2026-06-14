"""Unit tests for model registry."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slm_packager.registry import ModelInfo, ModelRegistry, ModelVariant


@pytest.mark.unit
class TestModelRegistry:
    """Test the ModelRegistry class."""

    def test_init_loads_registry(self, mock_registry_file):
        """Test that registry initializes and loads JSON data."""
        with patch.object(Path, "__truediv__", return_value=mock_registry_file):
            registry = ModelRegistry()
            assert registry._registry is not None
            assert "models" in registry._registry
            assert "version" in registry._registry

    def test_get_model_valid(self, mock_registry_file):
        """Test getting a valid model from registry."""
        with patch.object(Path, "__truediv__", return_value=mock_registry_file):
            registry = ModelRegistry()
            model = registry.get_model("test-model")

            assert model is not None
            assert isinstance(model, ModelInfo)
            assert model.name == "Test Model"
            assert model.format == "gguf"
            assert model.runtime == "llama_cpp"
            assert model.repo == "test/test-model"
            assert len(model.variants) == 2

    def test_get_model_invalid(self, mock_registry_file):
        """Test getting a non-existent model returns None."""
        with patch.object(Path, "__truediv__", return_value=mock_registry_file):
            registry = ModelRegistry()
            model = registry.get_model("nonexistent-model")

            assert model is None

    def test_get_model_variants(self, mock_registry_file):
        """Test that model variants are properly loaded."""
        with patch.object(Path, "__truediv__", return_value=mock_registry_file):
            registry = ModelRegistry()
            model = registry.get_model("test-model")

            assert "q4_k_m" in model.variants
            assert "q8_0" in model.variants

            q4_variant = model.variants["q4_k_m"]
            assert isinstance(q4_variant, ModelVariant)
            assert q4_variant.file == "test-model.Q4_K_M.gguf"
            assert q4_variant.size == "100MB"
            assert q4_variant.recommended is True

    def test_list_models(self, mock_registry_file):
        """Test listing all available model names."""
        with patch.object(Path, "__truediv__", return_value=mock_registry_file):
            registry = ModelRegistry()
            models = registry.list_models()

            assert isinstance(models, list)
            assert len(models) == 2
            assert "test-model" in models
            assert "test-pytorch" in models

    def test_get_all_models(self, mock_registry_file):
        """Test getting all models as ModelInfo objects."""
        with patch.object(Path, "__truediv__", return_value=mock_registry_file):
            registry = ModelRegistry()
            all_models = registry.get_all_models()

            assert isinstance(all_models, dict)
            assert len(all_models) == 2
            assert "test-model" in all_models
            assert isinstance(all_models["test-model"], ModelInfo)

    def test_get_recommended_variant_exists(self, mock_registry_file):
        """Test getting the recommended variant for a model."""
        with patch.object(Path, "__truediv__", return_value=mock_registry_file):
            registry = ModelRegistry()
            variant = registry.get_recommended_variant("test-model")

            assert variant == "q4_k_m"  # This is marked as recommended

    def test_get_recommended_variant_no_recommendation(
        self, mock_registry_file, sample_registry_data
    ):
        """Test fallback when no variant is marked as recommended."""
        # Modify data to have no recommended variant
        sample_registry_data["models"]["test-model"]["variants"]["q4_k_m"]["recommended"] = False

        with patch.object(Path, "__truediv__", return_value=mock_registry_file):
            with patch.object(ModelRegistry, "_load_registry", return_value=sample_registry_data):
                registry = ModelRegistry()
                variant = registry.get_recommended_variant("test-model")

                # Should fallback to first variant
                assert variant in ["q4_k_m", "q8_0"]

    def test_get_recommended_variant_invalid_model(self, mock_registry_file):
        """Test getting recommended variant for non-existent model."""
        with patch.object(Path, "__truediv__", return_value=mock_registry_file):
            registry = ModelRegistry()
            variant = registry.get_recommended_variant("nonexistent")

            assert variant is None

    def test_pytorch_model_format(self, mock_registry_file):
        """Test that PyTorch models are correctly parsed."""
        with patch.object(Path, "__truediv__", return_value=mock_registry_file):
            registry = ModelRegistry()
            model = registry.get_model("test-pytorch")

            assert model.format == "pytorch"
            assert model.runtime == "transformers"
            assert "default" in model.variants
