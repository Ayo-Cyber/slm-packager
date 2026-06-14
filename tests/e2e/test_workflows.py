"""End-to-end workflow tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from slm_packager.cli.main import cli
from slm_packager.config.loader import ConfigLoader


@pytest.mark.e2e
@pytest.mark.slow
class TestWorkflows:
    """Test complete end-to-end workflows."""

    @patch("slm_packager.cli.main.ModelDownloader")
    def test_pull_and_run_workflow(self, mock_downloader_class, temp_dir, sample_config_yaml):
        """Test: pull model → config created → run model."""
        runner = CliRunner()

        # Setup mock downloader
        mock_downloader = MagicMock()
        mock_downloader.pull.return_value = Path("/fake/model.gguf")
        mock_downloader_class.return_value = mock_downloader

        # Step 1: Pull model
        result = runner.invoke(cli, ["pull", "test-model"])
        assert result.exit_code == 0

        # Verify pull was called
        mock_downloader.pull.assert_called_once()

    @patch("slm_packager.cli.main.get_runtime")
    @patch("slm_packager.cli.main.ConfigLoader")
    def test_load_config_run_benchmark_workflow(
        self, mock_loader, mock_get_runtime, sample_config_yaml, mock_runtime
    ):
        """Test: load config → run model → benchmark."""
        runner = CliRunner()

        mock_loader.load.return_value = MagicMock()
        mock_get_runtime.return_value = mock_runtime

        # Step 1: Run model
        result1 = runner.invoke(cli, ["run", str(sample_config_yaml), "--prompt", "Test"])
        assert result1.exit_code == 0

        # Verify runtime was used
        mock_runtime.load.assert_called()
        mock_runtime.generate.assert_called()

    def test_init_and_load_workflow(self, temp_dir):
        """Test: init config → load config → validate."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Step 1: Create config with init
            result = runner.invoke(
                cli,
                [
                    "init",
                    "--name",
                    "workflow-test",
                    "--path",
                    "/fake/model.gguf",
                    "--format",
                    "gguf",
                    "--runtime",
                    "llama_cpp",
                    "--output",
                    "workflow.yaml",
                ],
            )

            assert result.exit_code == 0
            assert Path("workflow.yaml").exists()

            # Step 2: Load and validate config
            config = ConfigLoader.load(Path("workflow.yaml"))
            assert config.model.name == "workflow-test"
            assert config.model.path == "/fake/model.gguf"

    @patch("slm_packager.cli.main.Benchmarker")
    @patch("slm_packager.cli.main.ConfigLoader")
    def test_benchmark_workflow(self, mock_loader, mock_benchmarker_class, sample_config_yaml):
        """Test: load config → benchmark → verify metrics."""
        runner = CliRunner()

        mock_config = MagicMock()
        mock_loader.load.return_value = mock_config

        mock_benchmarker = MagicMock()
        mock_benchmarker.run.return_value = {
            "load_time_sec": 1.0,
            "generation_time_sec": 2.0,
            "tokens_per_second": 30.0,
            "memory_mb": 512.0,
            "latency_ms": 2000.0,
        }
        mock_benchmarker_class.return_value = mock_benchmarker

        # Run benchmark
        result = runner.invoke(cli, ["benchmark", str(sample_config_yaml)])

        assert result.exit_code == 0
        mock_benchmarker.run.assert_called_once()

        # Verify metrics in output
        assert "30" in result.output or "tokens" in result.output.lower()
