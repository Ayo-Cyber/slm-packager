"""Integration tests for FastAPI server."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock

from slm_packager.api.server import app
from slm_packager.api.manager import ModelBusyError


@pytest.mark.integration
class TestAPIServer:
    """Test FastAPI server endpoints."""
    
    def test_health_endpoint(self):
        """Test /health endpoint."""
        with TestClient(app) as client:
            response = client.get("/health")
            
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
    
    def test_info_endpoint_no_model(self):
        """Test /info endpoint when no model is loaded."""
        with TestClient(app) as client:
            response = client.get("/info")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
    
    @patch('slm_packager.api.manager.ConfigLoader')
    @patch('slm_packager.api.manager.get_runtime')
    def test_load_endpoint(self, mock_get_runtime, mock_loader, sample_config_yaml, mock_runtime):
        """Test POST /load endpoint."""
        with TestClient(app) as client:
            mock_loader.load.return_value = MagicMock()
            mock_get_runtime.return_value = mock_runtime
            
            response = client.post("/load", json={"config_path": str(sample_config_yaml)})
            
            assert response.status_code == 200
            assert response.json()["status"] == "success"
            mock_runtime.load.assert_called_once()
    
    def test_generate_endpoint_no_model(self):
        """Test POST /generate when no model is loaded."""
        # Ensure fresh state (no model loaded)
        # Note: TestClient(app) makes a fresh lifespan context if not reused across tests?
        # Actually starlette/fastapi TestClient re-runs lifespan for each client context usually.
        
        with TestClient(app) as client:
            response = client.post("/generate", json={"prompt": "Hello"})
            
            assert response.status_code == 400
            assert "not loaded" in response.json()["detail"].lower()
    
    @patch('slm_packager.api.manager.get_runtime')
    @patch('slm_packager.api.manager.ConfigLoader')
    def test_generate_endpoint_success(self, mock_loader, mock_get_runtime, sample_gguf_config):
        """Test POST /generate with successful generation."""
        
        # Setup mocks
        mock_runtime = MagicMock()
        mock_runtime.is_loaded = True
        mock_runtime.generate.return_value = "Generated response"
        mock_get_runtime.return_value = mock_runtime
        mock_loader.load.return_value = sample_gguf_config
        
        with TestClient(app) as client:
            # Load model first
            client.post("/load", json={"config_path": "/fake/path.yaml"})
            
            # Generate
            response = client.post("/generate", json={
                "prompt": "Test prompt",
                "params": None
            })
            
            assert response.status_code == 200
            assert "text" in response.json()
            assert response.json()["text"] == "Generated response"
    
    def test_generate_request_validation(self):
        """Test that generate request validates input."""
        with TestClient(app) as client:
            # Missing prompt
            response = client.post("/generate", json={})
            
            assert response.status_code == 422  # Validation error
    
    @patch('slm_packager.api.manager.get_runtime')
    @patch('slm_packager.api.manager.ConfigLoader')
    def test_info_endpoint_with_loaded_model(self, mock_loader, mock_get_runtime, sample_gguf_config):
        """Test /info endpoint after loading a model."""
        
        mock_runtime = MagicMock()
        mock_get_runtime.return_value = mock_runtime
        mock_loader.load.return_value = sample_gguf_config
        
        with TestClient(app) as client:
            # Load model
            client.post("/load", json={"config_path": "/fake/path.yaml"})
            
            # Check info
            response = client.get("/info")
            
            assert response.status_code == 200
            data = response.json()
            assert "model" in data
    
    @patch('slm_packager.api.manager.get_runtime')
    @patch('slm_packager.api.manager.ConfigLoader')
    def test_generate_streaming(self, mock_loader, mock_get_runtime, sample_gguf_config):
        """Test POST /generate with streaming enabled."""
        
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
        
        with TestClient(app) as client:
            # Load model
            client.post("/load", json={"config_path": "/fake/path.yaml"})
            
            # Generate with streaming
            response = client.post("/generate", json={
                "prompt": "Test prompt"
            })
            
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
    
    @patch('slm_packager.api.manager.ConfigLoader')
    def test_load_endpoint_file_not_found(self, mock_loader):
        """Test /load endpoint with non-existent file."""
        
        mock_loader.load.side_effect = FileNotFoundError("Config not found")
        
        with TestClient(app) as client:
            response = client.post("/load", json={"config_path": "/nonexistent.yaml"})
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()
    
    @patch('slm_packager.api.manager.ConfigLoader')
    def test_load_endpoint_invalid_config(self, mock_loader):
        """Test /load endpoint with invalid configuration."""
        
        mock_loader.load.side_effect = ValueError("Invalid config")
        
        with TestClient(app) as client:
            response = client.post("/load", json={"config_path": "/invalid.yaml"})
            
            assert response.status_code == 400
            assert "invalid" in response.json()["detail"].lower()

    def test_generate_endpoint_returns_409_when_manager_busy(self):
        """Test POST /generate returns conflict while a model switch is in progress."""
        with patch('slm_packager.api.server.ModelManager.is_loaded', new_callable=PropertyMock, return_value=True):
            with patch(
                'slm_packager.api.server.ModelManager.generate',
                new_callable=AsyncMock,
                side_effect=ModelBusyError("Model is busy loading or unloading. Try again shortly.")
            ):
                with TestClient(app) as client:
                    response = client.post("/generate", json={"prompt": "Hello"})

                    assert response.status_code == 409
                    assert "busy" in response.json()["detail"].lower()
