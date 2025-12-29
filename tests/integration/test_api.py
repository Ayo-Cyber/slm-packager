"""Integration tests for FastAPI server."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from slm_packager.api.server import app


@pytest.mark.integration
class TestAPIServer:
    """Test FastAPI server endpoints."""
    
    def test_health_endpoint(self):
        """Test /health endpoint."""
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_info_endpoint_no_model(self):
        """Test /info endpoint when no model is loaded."""
        client = TestClient(app)
        response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "model" in data
    
    @patch('slm_packager.api.server.ConfigLoader')
    @patch('slm_packager.api.server.get_runtime')
    def test_load_endpoint(self, mock_get_runtime, mock_loader, sample_config_yaml, mock_runtime):
        """Test POST /load endpoint."""
        client = TestClient(app)
        
        mock_loader.load.return_value = MagicMock()
        mock_get_runtime.return_value = mock_runtime
        
        response = client.post("/load", json={"config_path": str(sample_config_yaml)})
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_runtime.load.assert_called_once()
    
    @patch('slm_packager.api.server.runtime', new_callable=MagicMock)
    @patch('slm_packager.api.server.config', new_callable=MagicMock)
    def test_generate_endpoint_no_model(self, mock_config, mock_runtime_global):
        """Test POST /generate when no model is loaded."""
        # Simulate no model loaded
        mock_runtime_global.is_loaded = False
        
        client = TestClient(app)
        response = client.post("/generate", json={"prompt": "Hello"})
        
        assert response.status_code == 400
        assert "not loaded" in response.json()["detail"].lower()
    
    @patch('slm_packager.api.server.get_runtime')
    @patch('slm_packager.api.server.ConfigLoader')
    def test_generate_endpoint_success(self, mock_loader, mock_get_runtime, sample_gguf_config):
        """Test POST /generate with successful generation."""
        client = TestClient(app)
        
        # Load a model first
        mock_runtime = MagicMock()
        mock_runtime.is_loaded = True
        mock_runtime.generate.return_value = "Generated response"
        mock_get_runtime.return_value = mock_runtime
        mock_loader.load.return_value = sample_gguf_config
        
        # Load model
        client.post("/load", json={"config_path": "/fake/path.yaml"})
        
        # Patch the global runtime variable
        with patch('slm_packager.api.server.runtime', mock_runtime):
            with patch('slm_packager.api.server.config', sample_gguf_config):
                response = client.post("/generate", json={
                    "prompt": "Test prompt",
                    "params": None
                })
        
                assert response.status_code == 200
                assert "text" in response.json()
                assert response.json()["text"] == "Generated response"
    
    def test_generate_request_validation(self):
        """Test that generate request validates input."""
        client = TestClient(app)
        
        # Missing prompt
        response = client.post("/generate", json={})
        
        assert response.status_code == 422  # Validation error
    
    @patch('slm_packager.api.server.get_runtime')
    @patch('slm_packager.api.server.ConfigLoader')
    def test_info_endpoint_with_loaded_model(self, mock_loader, mock_get_runtime, sample_gguf_config):
        """Test /info endpoint after loading a model."""
        client = TestClient(app)
        
        mock_runtime = MagicMock()
        mock_get_runtime.return_value = mock_runtime
        mock_loader.load.return_value = sample_gguf_config
        
        # Load model
        client.post("/load", json={"config_path": "/fake/path.yaml"})
        
        # Check info
        with patch('slm_packager.api.server.config', sample_gguf_config):
            response = client.get("/info")
            
            assert response.status_code == 200
            data = response.json()
            assert "model" in data
    
    @patch('slm_packager.api.server.get_runtime')
    @patch('slm_packager.api.server.ConfigLoader')
    def test_generate_streaming(self, mock_loader, mock_get_runtime, sample_gguf_config):
        """Test POST /generate with streaming enabled."""
        client = TestClient(app)
        
        # Setup streaming mock
        mock_runtime = MagicMock()
        mock_runtime.is_loaded = True
        
        def mock_generate_stream(prompt, params):
            yield "Hello"
            yield " world"
            yield "!"
        
        mock_runtime.generate.return_value = mock_generate_stream("test", None)
        mock_get_runtime.return_value = mock_runtime
        
        # Config with streaming enabled
        streaming_config = sample_gguf_config
        streaming_config.params.stream = True
        mock_loader.load.return_value = streaming_config
        
        # Load model
        client.post("/load", json={"config_path": "/fake/path.yaml"})
        
        # Generate with streaming
        with patch('slm_packager.api.server.runtime', mock_runtime):
            with patch('slm_packager.api.server.config', streaming_config):
                response = client.post("/generate", json={
                    "prompt": "Test prompt"
                })
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    @patch('slm_packager.api.server.ConfigLoader')
    def test_load_endpoint_file_not_found(self, mock_loader):
        """Test /load endpoint with non-existent file."""
        client = TestClient(app)
        
        mock_loader.load.side_effect = FileNotFoundError("Config not found")
        
        response = client.post("/load", json={"config_path": "/nonexistent.yaml"})
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @patch('slm_packager.api.server.ConfigLoader')
    def test_load_endpoint_invalid_config(self, mock_loader):
        """Test /load endpoint with invalid configuration."""
        client = TestClient(app)
        
        mock_loader.load.side_effect = ValueError("Invalid config")
        
        response = client.post("/load", json={"config_path": "/invalid.yaml"})
        
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()
