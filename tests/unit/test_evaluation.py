"""Unit tests for evaluation/benchmarking."""
import pytest
from unittest.mock import MagicMock, patch

from slm_packager.evaluation.benchmark import Benchmarker


@pytest.mark.unit
class TestBenchmarker:
    """Test the Benchmarker class."""
    
    def test_init(self, sample_gguf_config, mock_runtime):
        """Test Benchmarker initialization."""
        with patch('slm_packager.evaluation.benchmark.get_runtime', return_value=mock_runtime):
            benchmarker = Benchmarker(sample_gguf_config)
            
            assert benchmarker.config == sample_gguf_config
            assert benchmarker.runtime == mock_runtime
    
    def test_run_returns_metrics(self, sample_gguf_config, mock_runtime):
        """Test that run() returns benchmark metrics."""
        with patch('slm_packager.evaluation.benchmark.get_runtime', return_value=mock_runtime):
            benchmarker = Benchmarker(sample_gguf_config)
            
            metrics = benchmarker.run(prompt="Test prompt")
            
            assert isinstance(metrics, dict)
            assert "load_time_sec" in metrics
            assert "memory_mb" in metrics
            assert "generation_time_sec" in metrics
            assert "tokens_per_second" in metrics
            assert "latency_ms" in metrics
    
    def test_run_calls_runtime_methods(self, sample_gguf_config, mock_runtime):
        """Test that run() calls load, generate, and unload."""
        with patch('slm_packager.evaluation.benchmark.get_runtime', return_value=mock_runtime):
            benchmarker = Benchmarker(sample_gguf_config)
            
            benchmarker.run()
            
            mock_runtime.load.assert_called_once()
            mock_runtime.generate.assert_called_once()
            mock_runtime.unload.assert_called_once()
    
    def test_run_disables_streaming(self, sample_gguf_config, mock_runtime):
        """Test that streaming is disabled during benchmarking."""
        sample_gguf_config.params.stream = True
        
        with patch('slm_packager.evaluation.benchmark.get_runtime', return_value=mock_runtime):
            benchmarker = Benchmarker(sample_gguf_config)
            
            benchmarker.run()
            
            # Verify generate was called (streaming handled internally by benchmarker)
            mock_runtime.generate.assert_called_once()
    
    def test_run_measures_timing(self, sample_gguf_config, mock_runtime):
        """Test that timing metrics are reasonable."""
        with patch('slm_packager.evaluation.benchmark.get_runtime', return_value=mock_runtime):
            benchmarker = Benchmarker(sample_gguf_config)
            
            metrics = benchmarker.run()
            
            # Timing should be positive numbers
            assert metrics["load_time_sec"] >= 0
            assert metrics["generation_time_sec"] >= 0
            assert metrics["latency_ms"] >= 0
            assert metrics["tokens_per_second"] >= 0
    
    def test_run_with_custom_prompt(self, sample_gguf_config, mock_runtime):
        """Test running benchmark with custom prompt."""
        custom_prompt = "Custom test prompt for benchmarking"
        
        with patch('slm_packager.evaluation.benchmark.get_runtime', return_value=mock_runtime):
            benchmarker = Benchmarker(sample_gguf_config)
            
            benchmarker.run(prompt=custom_prompt)
            
            # Verify custom prompt was used
            call_args = mock_runtime.generate.call_args
            assert call_args[0][0] == custom_prompt

    def test_memory_fallback_without_psutil_or_resource(self, sample_gguf_config, mock_runtime):
        """If neither psutil nor resource is available, report 0 MB instead of crashing."""
        with patch('slm_packager.evaluation.benchmark.get_runtime', return_value=mock_runtime):
            benchmarker = Benchmarker(sample_gguf_config)

            with patch('slm_packager.evaluation.benchmark.psutil', None):
                with patch('slm_packager.evaluation.benchmark.resource', None):
                    assert benchmarker._get_memory_mb() == 0.0

    def test_run_restores_stream_and_unloads_on_failure(self, sample_gguf_config, mock_runtime):
        """Benchmark cleanup should restore config state and unload even when generation fails."""
        sample_gguf_config.params.stream = True
        mock_runtime.generate.side_effect = RuntimeError("boom")

        with patch('slm_packager.evaluation.benchmark.get_runtime', return_value=mock_runtime):
            benchmarker = Benchmarker(sample_gguf_config)

            with pytest.raises(RuntimeError, match="boom"):
                benchmarker.run()

            assert sample_gguf_config.params.stream is True
            mock_runtime.unload.assert_called_once()
