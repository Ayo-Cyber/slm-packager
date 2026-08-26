"""Download manager for pulling models from HuggingFace"""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import click

IMPORT_ERROR = ""
try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

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
    class RepositoryNotFoundError(Exception):
        pass

    class LocalEntryNotFoundError(Exception):
        pass

    class RevisionNotFoundError(Exception):
        pass


from ..config.loader import ConfigLoader
from ..config.models import ModelConfig, RuntimeConfig, SLMConfig
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
            units = {"GB": 1.0, "MB": 1 / 1024, "KB": 1 / (1024**2)}
            match = re.match(r"([\d\.]+)([A-Z]+)", size_str.upper())
            if match:
                value, unit = match.groups()
                return float(value) * units.get(unit, 1.0)
            return 0.5  # Default conservative estimate if unknown
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
            raise ValueError(
                f"❌ Model '{model_name}' not found in registry. See available models with: slm list"
            )

        # Determine quantization variant
        if quantization is None:
            quantization = self.registry.get_recommended_variant(model_name)
            logger.info(f"📦 Using recommended quantization: {quantization}")

        if quantization not in model_info.variants:
            raise ValueError(
                f"❌ Quantization '{quantization}' not available for {model_name}. "
                f"Available: {', '.join(model_info.variants.keys())}"
            )

        variant = model_info.variants[quantization]

        # Pre-flight check: Disk Space
        if hasattr(variant, "size"):
            size_gb = self._parse_size_str(variant.size)
            if not self._check_disk_space(size_gb):
                msg = f"❌ Not enough disk space. Required: ~{size_gb:.2f}GB (plus buffer)"
                logger.error(msg)
                raise IOError(msg)

        logger.info(f"📥 Downloading {model_info.name} ({quantization})")
        logger.info(f"   Source: {model_info.repo}")
        logger.info(f"   Size: {variant.size}")

        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") != "1":
            logger.info(
                "💡 Tip: Install 'hf_transfer' for faster downloads: pip install hf_transfer"
            )

        # Handle based on format
        if model_info.format == "pytorch":
            # PyTorch/Transformers models don't need file download
            # They auto-download on first use with transformers
            logger.info(f"✅ Model configured: {model_info.repo}")
            logger.info(f"   (Will download automatically on first run)")

            # Create config pointing to HF repo
            config_path = self._create_config(
                model_name, model_info.repo, model_info, quantization  # Use repo ID as path
            )
            logger.info(f"✅ Config created: {config_path}")
            click.echo(f'\n🚀 Ready to use:\n   slm run {model_name} --prompt "Hello!"')

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
            )

            logger.info(f"✅ Model downloaded to: {model_path}")

            # Create config
            config_path = self._create_config(model_name, model_path, model_info, quantization)
            logger.info(f"✅ Config created: {config_path}")

            click.echo(f'\n🚀 Ready to use:\n   slm run {model_name} --prompt "Hello!"')

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

    def pull_from_repo(self, repo_id: str, filename: str, name: Optional[str] = None) -> Path:
        """
        Pull any GGUF/ONNX file directly from a HuggingFace repo without a registry entry.

        Args:
            repo_id:  HuggingFace repo ID, e.g. "TheBloke/Mistral-7B-GGUF"
            filename: File to download, e.g. "mistral-7b-v0.1.Q4_K_M.gguf"
            name:     Local alias for the model. Defaults to the repo name (part after "/").
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "❌ Model downloading requires 'huggingface-hub'\n"
                "💡 Install with: pip install huggingface-hub\n"
                f"   Error details: {IMPORT_ERROR}"
            )

        # Derive a safe local name from the repo if not provided
        local_name = name or repo_id.split("/")[-1].lower()

        # Auto-detect format and runtime from file extension
        ext = Path(filename).suffix.lower()
        if ext == ".gguf":
            fmt, runtime = "gguf", "llama_cpp"
        elif ext == ".onnx":
            fmt, runtime = "onnx", "onnx"
        else:
            fmt, runtime = "pytorch", "transformers"

        logger.info(f"📥 Downloading {filename} from {repo_id}")

        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.models_dir),
            )
        except (RepositoryNotFoundError, RevisionNotFoundError, LocalEntryNotFoundError) as e:
            raise ValueError(
                f"❌ File not found on HuggingFace: {repo_id}/{filename}\n"
                "💡 Check the repo ID and filename are correct."
            ) from e
        except Exception as e:
            logger.exception("Direct repo download failed")
            raise RuntimeError(
                f"❌ Failed to download {filename}: {str(e)}\n"
                "💡 Check your internet connection and try again."
            ) from e

        # Build and save config
        from ..config.models import ModelConfig, RuntimeConfig, SLMConfig

        config = SLMConfig(
            model=ModelConfig(
                name=local_name,
                path=model_path,
                format=fmt,
                description=f"{repo_id} — {filename}",
            ),
            runtime=RuntimeConfig(type=runtime),
        )
        config_path = self.configs_dir / f"{local_name}.yaml"
        ConfigLoader.save(config, config_path)

        logger.info(f"✅ Downloaded to: {model_path}")
        logger.info(f"✅ Config saved: {config_path}")
        click.echo(f'\n🚀 Ready to use:\n   slm run {local_name} --prompt "Hello!"')

        return Path(model_path)

    def _create_config(
        self, model_name: str, model_path: str, model_info, quantization: str
    ) -> Path:
        """Create a config file for the pulled model"""
        config = SLMConfig(
            model=ModelConfig(
                name=model_name,
                path=model_path,
                format=model_info.format,
                description=f"{model_info.name} ({quantization})",
            ),
            runtime=RuntimeConfig(type=model_info.runtime),
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
            model_path_str = config.model.path
            model_file = Path(model_path_str)

            if model_file.exists():
                # Local file or directory (GGUF / ONNX / local pytorch dir)
                if model_file.is_dir():
                    shutil.rmtree(model_file)
                    logger.info(f"Deleted model directory: {model_file}")
                elif self._hf_repo_id_for_path(model_file):
                    # Inside our hub cache: a snapshots/<rev>/<file> symlink whose
                    # target blob may be shared with other revisions. Deleting the
                    # symlink alone leaves the multi-GB blob behind, and deleting the
                    # blob by hand would break anything else pointing at it — so let
                    # the hub's refcounted deletion handle both.
                    self._delete_hf_cache(self._hf_repo_id_for_path(model_file))
                else:
                    # A path the user manages. Only ever remove the entry itself,
                    # never follow it out to a symlink target elsewhere on disk.
                    model_file.unlink()
                    logger.info(f"Deleted model file: {model_file}")
            elif "/" in model_path_str or not model_file.is_absolute():
                # HF repo ID (e.g. "gpt2" or "TheBloke/TinyLlama-GGUF") — delete from HF cache
                self._delete_hf_cache(model_path_str)
        except Exception as e:
            logger.warning(f"Could not delete model file: {e}")

        # Always delete the config
        config_path.unlink()
        logger.info(f"Deleted config: {config_path}")
        return True

    def _hf_repo_id_for_path(self, path: Path) -> Optional[str]:
        """Return the repo id if ``path`` lives inside our hub cache, else None.

        The hub lays out ``<cache>/models--<org>--<name>/snapshots/<rev>/<file>``, so
        the repo id is recoverable from the directory name.
        """
        try:
            resolved_dir = self.models_dir.resolve()
            candidates = [p for p in path.resolve().parents if p.name.startswith("models--")]
        except OSError:
            return None

        for repo_dir in candidates:
            if resolved_dir not in repo_dir.parents and repo_dir.parent != resolved_dir:
                continue
            parts = repo_dir.name.split("--")[1:]
            if parts:
                return "/".join(parts)
        return None

    def _delete_hf_cache(self, repo_id: str) -> None:
        """Remove a model from the HuggingFace hub cache."""
        if not HF_AVAILABLE:
            logger.warning("huggingface-hub not available; cannot clean HF cache")
            return
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir(self.models_dir)
            for repo in cache_info.repos:
                if repo.repo_id == repo_id:
                    # delete_revisions refcounts shared blobs, so revisions of other
                    # repos that reference the same file keep working.
                    delete_strategy = cache_info.delete_revisions(
                        *[rev.commit_hash for rev in repo.revisions]
                    )
                    delete_strategy.execute()
                    logger.info(
                        f"Deleted HF cache for repo: {repo_id} "
                        f"(freed ~{delete_strategy.expected_freed_size_str})"
                    )
                    return
            logger.debug(f"No HF cache found for repo: {repo_id}")
        except Exception as e:
            logger.warning(f"Could not clean HF cache for '{repo_id}': {e}")

    def list_installed(self) -> list:
        """List installed models"""
        configs = list(self.configs_dir.glob("*.yaml"))
        installed = []

        for config_path in configs:
            try:
                config = ConfigLoader.load(config_path)
                model_file = Path(config.model.path)

                if model_file.exists():
                    # Local file or directory
                    if model_file.is_dir():
                        size_bytes = sum(
                            f.stat().st_size for f in model_file.rglob("*") if f.is_file()
                        )
                    else:
                        size_bytes = model_file.stat().st_size
                    size_str = f"{size_bytes / (1024**3):.2f}GB"
                else:
                    # HF-managed repo (pytorch models store in HF cache)
                    size_str = "HF cache"

                installed.append(
                    {
                        "name": config.model.name,
                        "path": config.model.path,
                        "size": size_str,
                        "format": config.model.format,
                    }
                )
            except Exception:
                continue

        return installed
