from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

# numpy ships with the optional [onnx] extra, so these tests only apply when it is
# installed. Everything else in the suite must pass on a core-only install.
np = pytest.importorskip("numpy", reason="requires the [onnx] extra")

from slm_packager.config.models import GenerationParams, ModelConfig, RuntimeConfig, SLMConfig
from slm_packager.runtime.onnx import OnnxRuntime


@pytest.fixture
def mock_onnx_config(tmp_path):
    config = SLMConfig(
        model=ModelConfig(
            name="test-model",
            path=str(tmp_path / "model.onnx"),
            format="onnx",
            description="Test ONNX model",
        ),
        runtime=RuntimeConfig(type="onnx", device="cpu", threads=1),
    )
    return config


@pytest.fixture
def mock_ort_session():
    with patch("slm_packager.runtime.onnx.ort.InferenceSession") as mock:
        session = mock.return_value
        # Mock inputs
        input_mock = MagicMock()
        input_mock.name = "input_ids"
        session.get_inputs.return_value = [input_mock]
        session.get_outputs.return_value = []
        yield session


@pytest.fixture
def onnx_runtime(mock_onnx_config, mock_ort_session):
    # Mock existence of model file
    with patch("slm_packager.runtime.onnx.ONNX_AVAILABLE", True):
        with patch("pathlib.Path.exists", return_value=True):
            # Mock tokenizer
            with patch("slm_packager.runtime.onnx.AutoTokenizer.from_pretrained") as mock_tokenizer:
                runtime = OnnxRuntime(mock_onnx_config)
                runtime.load()
                return runtime


@pytest.mark.unit
class TestOnnxRuntime:
    def test_load_raises_import_error_when_onnx_deps_missing(self, mock_onnx_config):
        """Missing ONNX deps should surface as an ImportError before path checks."""
        runtime = OnnxRuntime(mock_onnx_config)

        with patch("slm_packager.runtime.onnx.ONNX_AVAILABLE", False):
            with pytest.raises(ImportError, match="onnxruntime"):
                runtime.load()

    def test_forward_success(self, onnx_runtime):
        """Test successful forward pass."""
        input_ids = np.array([[1, 2, 3]])
        attention_mask = np.ones_like(input_ids)

        # Mock session run return
        onnx_runtime.session.run.return_value = [np.random.randn(1, 3, 50257)]
        onnx_runtime.output_names = ["logits"]

        outputs = onnx_runtime._forward(input_ids, attention_mask)
        assert "logits" in outputs

    def test_forward_failure(self, onnx_runtime):
        """Test forward pass failure handling."""
        input_ids = np.array([[1, 2, 3]])
        attention_mask = np.ones_like(input_ids)

        # Mock failure
        onnx_runtime.session.run.side_effect = Exception("Runtime Error")

        with pytest.raises(RuntimeError, match="Model inference failed"):
            onnx_runtime._forward(input_ids, attention_mask)

    def test_sample_nan_handling(self, onnx_runtime):
        """Test that NaNs in logits are handled."""
        logits = np.array([-1.0, np.nan, 2.0])
        params = GenerationParams(temperature=1.0)

        # Should not raise error
        token = onnx_runtime._sample(logits, params)
        assert isinstance(token, (int, np.integer))

    def test_sample_zeros_probability(self, onnx_runtime):
        """Test fallback when probability sum is zero (underflow)."""
        logits = np.array([-1000.0, -1000.0])  # Very small numbers
        params = GenerationParams(temperature=1.0)

        # Should fall back to uniform distribution
        token = onnx_runtime._sample(logits, params)
        assert isinstance(token, (int, np.integer))

    def test_empty_prompt_warning(self, onnx_runtime):
        """Test handling of empty prompt."""
        params = GenerationParams()
        with patch("slm_packager.runtime.onnx.logger.warning") as mock_warn:
            result = onnx_runtime.generate("", params)
            assert result == ""
            mock_warn.assert_called_with("Received empty prompt")

    def test_unload_calls_gc(self, onnx_runtime):
        """Test that unload calls garbage collection."""
        with patch("gc.collect") as mock_gc:
            onnx_runtime.unload()
            mock_gc.assert_called_once()
            assert onnx_runtime.session is None
            assert onnx_runtime.tokenizer is None


@pytest.mark.unit
class TestGenerationParamHandling:
    """Params documented in the config schema must actually take effect."""

    def test_truncate_at_stop_picks_earliest_match(self):
        from slm_packager.runtime.base import _truncate_at_stop

        assert _truncate_at_stop("hello</s>world", ["</s>"]) == "hello"
        # Earliest wins regardless of the order given.
        assert _truncate_at_stop("a STOP b END c", ["END", "STOP"]) == "a "
        assert _truncate_at_stop("no match", ["</s>"]) == "no match"
        assert _truncate_at_stop("anything", None) == "anything"
        assert _truncate_at_stop("anything", []) == "anything"

    def test_onnx_temperature_zero_is_greedy(self, mock_onnx_config):
        """temperature=0 must take the argmax, not sample at temperature 1."""
        runtime = OnnxRuntime(mock_onnx_config)
        logits = np.array([0.1, 5.0, 0.2, 0.3])

        params = GenerationParams(temperature=0.0)
        # Deterministic across repeats, and always the highest-logit token.
        assert [runtime._sample(logits, params) for _ in range(5)] == [1] * 5

    def test_onnx_applies_repetition_penalty(self, mock_onnx_config):
        """Already-generated tokens must be penalised, not silently ignored."""
        runtime = OnnxRuntime(mock_onnx_config)
        logits = np.array([1.0, 5.0, 4.9, 0.5])
        params = GenerationParams(temperature=0.0, repetition_penalty=2.0)

        # Without history, token 1 wins outright.
        assert runtime._sample(logits, params, generated=None) == 1
        # Having produced token 1 already, its score halves (5.0 -> 2.5) and the
        # runner-up takes over.
        assert runtime._sample(logits, params, generated=[1]) == 2

    def test_onnx_repetition_penalty_pushes_negative_scores_down_too(self, mock_onnx_config):
        """A penalty must always reduce a token's chance, including negative logits."""
        runtime = OnnxRuntime(mock_onnx_config)
        logits = np.array([-1.0, -1.5])
        params = GenerationParams(temperature=0.0, repetition_penalty=2.0)

        # Token 0 leads; after penalising it (-1.0 -> -2.0) token 1 leads.
        assert runtime._sample(logits, params, generated=None) == 0
        assert runtime._sample(logits, params, generated=[0]) == 1
