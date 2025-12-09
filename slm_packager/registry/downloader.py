"""Download manager for pulling models from HuggingFace"""
from pathlib import Path
from typing import Optional
import sys

try:
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from ..config.models import SLMConfig, ModelConfig, RuntimeConfig
from ..config.loader import ConfigLoader
from . import ModelRegistry

class ModelDownloader:
    """Handles downloading models from HuggingFace"""
    
    def __init__(self):
        if not HF_AVAILABLE:
            raise ImportError(
                "❌ Model downloading requires 'huggingface-hub'\n"
                "💡 Install with: pip install huggingface-hub"
            )
        
        self.registry = ModelRegistry()
        self.models_dir = Path.home() / ".slm" / "models"
        self.configs_dir = Path.home() / ".slm" / "configs"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
    
    def pull(self, model_name: str, quantization: Optional[str] = None) -> Path:
        """
        Pull a model from the registry
        
        Args:
            model_name: Name of model in registry
            quantization: Specific quantization type (e.g., 'q4_k_m')
        
        Returns:
            Path to downloaded model file
        """
        # Get model info from registry
        model_info = self.registry.get_model(model_name)
        if not model_info:
            raise ValueError(
                f"❌ Model '{model_name}' not found in registry\n"
                "💡 See available models with: slm list"
            )
        
        # Determine quantization variant
        if quantization is None:
            quantization = self.registry.get_recommended_variant(model_name)
            print(f"📦 Using recommended quantization: {quantization}")
        
        if quantization not in model_info.variants:
            raise ValueError(
                f"❌ Quantization '{quantization}' not available for {model_name}\n"
                f"💡 Available: {', '.join(model_info.variants.keys())}"
            )
        
        variant = model_info.variants[quantization]
        
        print(f"\n📥 Downloading {model_info.name} ({quantization})")
        print(f"   Source: {model_info.repo}")
        print(f"   Size: {variant.size}")
        print(f"   Speed: {variant.speed}, Quality: {variant.quality}\n")
        
        # Handle based on format
        if model_info.format == "pytorch":
            # PyTorch/Transformers models don't need file download
            # They auto-download on first use with transformers
            print(f"✅ Model configured: {model_info.repo}")
            print(f"   (Will download automatically on first run)\n")
            
            # Create config pointing to HF repo
            config_path = self._create_config(
                model_name,
                model_info.repo,  # Use repo ID as path
                model_info,
                quantization
            )
            print(f"✅ Config created: {config_path}")
            print(f"\n🚀 Ready to use:")
            print(f"   slm run {model_name} --prompt \"Hello!\"")
            
            return Path(model_info.repo)
        
        # GGUF/ONNX models - download file from HuggingFace
        try:
            model_path = hf_hub_download(
                repo_id=model_info.repo,
                filename=variant.file,
                cache_dir=str(self.models_dir),
                resume_download=True
            )
            
            print(f"\n✅ Model downloaded to: {model_path}")
            
            # Create config
            config_path = self._create_config(
                model_name, 
                model_path, 
                model_info,
                quantization
            )
            print(f"✅ Config created: {config_path}")
            
            print(f"\n🚀 Ready to use:")
            print(f"   slm run {model_name} --prompt \"Hello!\"")
            
            return Path(model_path)
            
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to download model\n"
                f"   {str(e)}\n"
                "💡 Check:\n"
                "   - Internet connection\n"
                "   - HuggingFace availability\n"
                "   - Disk space"
            ) from e
    
    def _create_config(
        self, 
        model_name: str, 
        model_path: str,
        model_info,
        quantization: str
    ) -> Path:
        """Create a config file for the pulled model"""
        config = SLMConfig(
            model=ModelConfig(
                name=model_name,
                path=model_path,
                format=model_info.format,
                description=f"{model_info.name} ({quantization})"
            ),
            runtime=RuntimeConfig(
                type=model_info.runtime
            )
        )
        
        config_path = self.configs_dir / f"{model_name}.yaml"
        ConfigLoader.save(config, config_path)
        
        return config_path
    
    def list_installed(self) -> list:
        """List installed models"""
        configs = list(self.configs_dir.glob("*.yaml"))
        installed = []
        
        for config_path in configs:
            try:
                config = ConfigLoader.load(config_path)
                model_file = Path(config.model.path)
                if model_file.exists():
                    size = model_file.stat().st_size / (1024**3)  # GB
                    installed.append({
                        'name': config.model.name,
                        'path': config.model.path,
                        'size': f"{size:.2f}GB",
                        'format': config.model.format
                    })
            except Exception:
                continue
        
        return installed
