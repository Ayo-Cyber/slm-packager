"""Download manager for pulling models from HuggingFace"""
from pathlib import Path
from typing import Optional
import os
import shutil
import logging
import re

IMPORT_ERROR = ""
try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError, LocalEntryNotFoundError, RevisionNotFoundError
    HF_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR = str(e)
    HF_AVAILABLE = False

    def hf_hub_download(*args, **kwargs):
        raise ImportError(
            "❌ Model downloading requires 'huggingface-hub'\n"
            "💡 Install with: pip install huggingface-hub\n"
            f"   Error details: {IMPORT_ERROR}"
        )

    # Define dummy exceptions for try/except blocks to work if HF not available
    class RepositoryNotFoundError(Exception): pass
    class LocalEntryNotFoundError(Exception): pass
    class RevisionNotFoundError(Exception): pass

from ..config.models import SLMConfig, ModelConfig, RuntimeConfig
from ..config.loader import ConfigLoader
from . import ModelRegistry

logger = logging.getLogger(__name__)

class ModelDownloader:
    """Handles downloading models from HuggingFace"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.models_dir = Path.home() / ".slm" / "models"
        self.configs_dir = Path.home() / ".slm" / "configs"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        # Enable hf_transfer for faster downloads if available
        try:
            import hf_transfer
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            logger.info("🚀 Fast download enabled (hf_transfer)")
        except ImportError:
            logger.debug("hf_transfer not available, using standard download")
    
    def _parse_size_str(self, size_str: str) -> float:
        """Parse size string like '1.5GB' or '500MB' to GB float"""
        try:
            units = {'GB': 1.0, 'MB': 1/1024, 'KB': 1/(1024**2)}
            match = re.match(r"([\d\.]+)([A-Z]+)", size_str.upper())
            if match:
                value, unit = match.groups()
                return float(value) * units.get(unit, 1.0)
            return 0.5 # Default conservative estimate if unknown
        except Exception:
            return 0.5

    def _check_disk_space(self, required_gb: float) -> bool:
        """Check if there is enough disk space"""
        try:
            total, used, free = shutil.disk_usage(self.models_dir.parent)
            free_gb = free / (1024**3)
            # Add 10% buffer
            if free_gb < (required_gb * 1.1):
                return False
            return True
        except Exception:
            # If check fails (e.g. permissions), assume it's fine but log warning
            logger.warning("Could not verify disk space")
            return True

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
            msg = f"❌ Model '{model_name}' not found in registry. See available models with: slm list"
            logger.error(msg)
            raise ValueError(msg)
        
        # Determine quantization variant
        if quantization is None:
            quantization = self.registry.get_recommended_variant(model_name)
            logger.info(f"📦 Using recommended quantization: {quantization}")
        
        if quantization not in model_info.variants:
            msg = f"❌ Quantization '{quantization}' not available for {model_name}. Available: {', '.join(model_info.variants.keys())}"
            logger.error(msg)
            raise ValueError(msg)
        
        variant = model_info.variants[quantization]
        
        # Pre-flight check: Disk Space
        if hasattr(variant, 'size'):
            size_gb = self._parse_size_str(variant.size)
            if not self._check_disk_space(size_gb):
                msg = f"❌ Not enough disk space. Required: ~{size_gb:.2f}GB (plus buffer)"
                logger.error(msg)
                raise IOError(msg)

        logger.info(f"📥 Downloading {model_info.name} ({quantization})")
        logger.info(f"   Source: {model_info.repo}")
        logger.info(f"   Size: {variant.size}")
        
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") != "1":
            logger.info("💡 Tip: Install 'hf_transfer' for faster downloads: pip install hf_transfer")

        
        # Handle based on format
        if model_info.format == "pytorch":
            # PyTorch/Transformers models don't need file download
            # They auto-download on first use with transformers
            logger.info(f"✅ Model configured: {model_info.repo}")
            logger.info(f"   (Will download automatically on first run)")
            
            # Create config pointing to HF repo
            config_path = self._create_config(
                model_name,
                model_info.repo,  # Use repo ID as path
                model_info,
                quantization
            )
            logger.info(f"✅ Config created: {config_path}")
            print(f"\n🚀 Ready to use:\n   slm run {model_name} --prompt \"Hello!\"")
            
            return Path(model_info.repo)
        
        # GGUF/ONNX models - download file from HuggingFace
        try:
            if not HF_AVAILABLE:
                raise ImportError(
                    "❌ Model downloading requires 'huggingface-hub'\n"
                    "💡 Install with: pip install huggingface-hub\n"
                    f"   Error details: {IMPORT_ERROR}"
                )

            model_path = hf_hub_download(
                repo_id=model_info.repo,
                filename=variant.file,
                cache_dir=str(self.models_dir),
                resume_download=True
            )
            
            logger.info(f"✅ Model downloaded to: {model_path}")
            
            # Create config
            config_path = self._create_config(
                model_name, 
                model_path, 
                model_info,
                quantization
            )
            logger.info(f"✅ Config created: {config_path}")
            
            print(f"\n🚀 Ready to use:\n   slm run {model_name} --prompt \"Hello!\"")
            
            return Path(model_path)
            
        except ImportError:
            raise
        except (RepositoryNotFoundError, RevisionNotFoundError, LocalEntryNotFoundError) as e:
            msg = f"❌ Model file not found on HuggingFace: {str(e)}"
            logger.error(msg)
            raise ValueError(msg) from e
        except Exception as e:
            logger.exception("Download failed unexpectedly")
            raise RuntimeError(
                f"❌ Failed to download model: {str(e)}\n"
                "💡 Check internet connection, disk space, or try a different model."
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
    
    def delete(self, model_name: str) -> bool:
        """
        Delete an installed model and its config.

        Args:
            model_name: Name of model to delete

        Returns:
            True if model was deleted, False if not found
        """
        config_path = self.configs_dir / f"{model_name}.yaml"
        if not config_path.exists():
            return False

        # Load config to find the model file path
        try:
            config = ConfigLoader.load(config_path)
            model_file = Path(config.model.path)

            # Delete model file/directory
            if model_file.exists():
                if model_file.is_dir():
                    shutil.rmtree(model_file)
                    logger.info(f"Deleted model directory: {model_file}")
                else:
                    model_file.unlink()
                    logger.info(f"Deleted model file: {model_file}")

                    # Also clean up the parent HF cache directory if it's now empty
                    parent = model_file.parent
                    if parent != self.models_dir and not any(parent.iterdir()):
                        shutil.rmtree(parent, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Could not delete model file: {e}")

        # Always delete the config
        config_path.unlink()
        logger.info(f"Deleted config: {config_path}")
        return True

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
