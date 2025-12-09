import yaml
import json
from pathlib import Path
from typing import Union, Dict
from pydantic import ValidationError
from .models import SLMConfig

class ConfigLoader:
    @staticmethod
    def load(path: Union[str, Path]) -> SLMConfig:
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(
                f"❌ Config file not found: '{path}'\n"
                "💡 Create one with:\n"
                "   slm init\n"
                "\n"
                "   Or see examples/ directory for reference configs"
            )

        try:
            with open(path, "r") as f:
                if path.suffix in [".yaml", ".yml"]:
                    try:
                        data = yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        raise ValueError(
                            f"❌ Invalid YAML syntax in config file\n"
                            f"   File: {path}\n"
                            f"   Error: {str(e)}\n"
                            "💡 Check for:\n"
                            "   - Proper indentation (use spaces, not tabs)\n"
                            "   - Matching quotes\n"
                            "   - Valid YAML structure\n"
                            "\n"
                            "   See examples/ for reference configs"
                        ) from e
                elif path.suffix == ".json":
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"❌ Invalid JSON syntax in config file\n"
                            f"   File: {path}\n"
                            f"   Error: {str(e)}\n"
                            "💡 Check for:\n"
                            "   - Matching braces and brackets\n"
                            "   - Proper comma placement\n"
                            "   - Valid JSON structure"
                        ) from e
                else:
                    raise ValueError(
                        f"❌ Unsupported config format: '{path.suffix}'\n"
                        "💡 Use .yaml, .yml, or .json file extension"
                    )
        except PermissionError:
            raise PermissionError(
                f"❌ Permission denied reading config file: '{path}'\n"
                "💡 Check file permissions:\n"
                f"   chmod 644 {path}"
            )
        except Exception as e:
            raise RuntimeError(
                f"❌ Error reading config file: '{path}'\n"
                f"   {type(e).__name__}: {str(e)}"
            ) from e

        # Validate config structure
        try:
            return SLMConfig(**data)
        except ValidationError as e:
            # Format validation errors nicely
            error_details = []
            for error in e.errors():
                field = " -> ".join(str(x) for x in error['loc'])
                msg = error['msg']
                error_details.append(f"   • {field}: {msg}")
            
            raise ValueError(
                f"❌ Invalid config structure in: '{path}'\n"
                f"\n"
                f"Validation errors:\n" +
                "\n".join(error_details) +
                f"\n\n"
                f"💡 Check that all required fields are present:\n"
                f"   - model.name (string)\n"
                f"   - model.path (string)\n"
                f"   - model.format (gguf, onnx, or pytorch)\n"
                f"   - runtime.type (llama_cpp, onnx, or transformers)\n"
                f"\n"
                f"   See examples/ directory for reference configs"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"❌ Error creating config from file: '{path}'\n"
                f"   {type(e).__name__}: {str(e)}\n"
                "💡 Check that the config structure is correct"
            ) from e

    @staticmethod
    def save(config: SLMConfig, path: Union[str, Path]):
        path = Path(path)
        
        try:
            data = config.model_dump(mode="json")
        except Exception as e:
            raise RuntimeError(
                f"❌ Error serializing config\n"
                f"   {type(e).__name__}: {str(e)}"
            ) from e
        
        try:
            # Create parent directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w") as f:
                if path.suffix in [".yaml", ".yml"]:
                    yaml.dump(data, f, sort_keys=False, default_flow_style=False)
                elif path.suffix == ".json":
                    json.dump(data, f, indent=2)
                else:
                    raise ValueError(
                        f"❌ Unsupported config format: '{path.suffix}'\n"
                        "💡 Use .yaml, .yml, or .json file extension"
                    )
        except PermissionError:
            raise PermissionError(
                f"❌ Permission denied writing config file: '{path}'\n"
                "💡 Check:\n"
                f"   - You have write permission in {path.parent}\n"
                f"   - The directory exists and is writable"
            )
        except Exception as e:
            raise RuntimeError(
                f"❌ Error writing config file: '{path}'\n"
                f"   {type(e).__name__}: {str(e)}"
            ) from e

