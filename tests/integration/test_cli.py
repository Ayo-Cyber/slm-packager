"""Integration tests for CLI commands."""
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from pathlib import Path

from slm_packager.cli.main import cli


@pytest.mark.integration
class TestCLICommands:
    """Test CLI commands using Click's CliRunner."""
    
    def test_cli_help(self):
        """Test that CLI help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "SLM Packager CLI" in result.output or "Usage" in result.output
    
    def test_list_command(self, mock_registry_file):
        """Test 'slm list' command."""
        runner = CliRunner()
        
        with patch.object(Path, '__truediv__', return_value=mock_registry_file):
            result = runner.invoke(cli, ['list'])
            
            assert result.exit_code == 0
            # Should show available models
            assert "test-model" in result.output or "Test Model" in result.output
    
    def test_list_installed_command(self, temp_dir):
        """Test 'slm list --installed' command."""
        runner = CliRunner()
        
        with patch('slm_packager.cli.main.ModelDownloader') as MockDownloader:
            mock_downloader = MagicMock()
            mock_downloader.list_installed.return_value = []
            MockDownloader.return_value = mock_downloader
            
            result = runner.invoke(cli, ['list', '--installed'])
            
            assert result.exit_code == 0
            mock_downloader.list_installed.assert_called_once()
    
    def test_init_command(self, temp_dir):
        """Test 'slm init' command."""
        runner =CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, [
                'init',
                '--name', 'test-model',
                '--path', '/path/to/model.gguf',
                '--format', 'gguf',
                '--runtime', 'llama_cpp',
                '--output', 'test-config.yaml'
            ])
            
            assert result.exit_code == 0
            assert Path('test-config.yaml').exists()
    
    @patch('slm_packager.cli.main.ModelDownloader')
    def test_pull_command(self, mock_downloader_class, mock_registry_file):
        """Test 'slm pull' command."""
        runner = CliRunner()
        
        mock_downloader = MagicMock()
        mock_downloader.pull.return_value = Path("/fake/model.gguf")
        mock_downloader_class.return_value = mock_downloader
        
        result = runner.invoke(cli, ['pull', 'test-model'])
        
        # Check it was called with model_name and quant (no list_variants in basic call)
        assert result.exit_code == 0
        mock_downloader.pull.assert_called_once()
        call_args = mock_downloader.pull.call_args[0]
        assert call_args[0] == 'test-model'
    
    @patch('slm_packager.cli.main.ModelDownloader')
    def test_pull_with_quantization(self, mock_downloader_class):
        """Test 'slm pull' with quantization option."""
        runner = CliRunner()
        
        mock_downloader = MagicMock()
        mock_downloader.pull.return_value = Path("/fake/model.gguf")
        mock_downloader_class.return_value = mock_downloader
        
        result = runner.invoke(cli, ['pull', 'test-model', '--quant', 'q4_k_m'])
        
        mock_downloader.pull.assert_called_once()
        call_args = mock_downloader.pull.call_args
        assert call_args[0][0] == 'test-model'
        assert call_args[0][1] == 'q4_k_m'
    
    @patch('slm_packager.cli.main.get_runtime')
    @patch('slm_packager.cli.main.ConfigLoader')
    def test_run_command_with_config(self, mock_loader, mock_get_runtime, sample_config_yaml, mock_runtime):
        """Test 'slm run' command with config file."""
        runner = CliRunner()
        
        mock_loader.load.return_value = MagicMock()
        mock_get_runtime.return_value = mock_runtime
        
        result = runner.invoke(cli, ['run', str(sample_config_yaml), '--prompt', 'Hello'])
        
        assert result.exit_code == 0
        mock_runtime.load.assert_called_once()
        mock_runtime.generate.assert_called_once()
    
    @patch('slm_packager.cli.main.Benchmarker')
    @patch('slm_packager.cli.main.ConfigLoader')
    def test_benchmark_command(self, mock_loader, mock_benchmarker_class, sample_config_yaml):
        """Test 'slm benchmark' command."""
        runner = CliRunner()
        
        mock_loader.load.return_value = MagicMock()
        mock_benchmarker = MagicMock()
        mock_benchmarker.run.return_value = {
            "load_time_sec": 1.5,
            "generation_time_sec": 2.0,
            "tokens_per_second": 25.0,
            "memory_mb": 512.0,
            "latency_ms": 2000.0
        }
        mock_benchmarker_class.return_value = mock_benchmarker
        
        result = runner.invoke(cli, ['benchmark', str(sample_config_yaml)])
        
        assert result.exit_code == 0
        assert "1.5" in result.output  # load time
        assert "25" in result.output or "tokens" in result.output.lower()
    
    @patch('slm_packager.cli.main.start_server')
    def test_serve_command(self, mock_start_server):
        """Test 'slm serve' command."""
        runner = CliRunner()
        
        # Serve command starts a server, we just test it's called
        result = runner.invoke(cli, ['serve', '--host', '127.0.0.1', '--port', '8080'])
        
        assert result.exit_code == 0
        # Function is called with positional args in actual code
        mock_start_server.assert_called_once()
        call_args = mock_start_server.call_args[0]
        assert call_args[0] == '127.0.0.1'
        assert call_args[1] == 8080
