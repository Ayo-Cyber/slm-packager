import yaml
import json
from pathlib import Path
from typing import Union, Dict
from .models import SLMConfig

class ConfigLoader:
    @staticmethod
    def load(path: Union[str, Path]) -> SLMConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError("Unsupported config format. Use .yaml or .json")

        return SLMConfig(**data)

    @staticmethod
    def save(config: SLMConfig, path: Union[str, Path]):
        path = Path(path)
        data = config.model_dump(mode="json")
        
        with open(path, "w") as f:
            if path.suffix in [".yaml", ".yml"]:
                yaml.dump(data, f, sort_keys=False)
            elif path.suffix == ".json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError("Unsupported config format. Use .yaml or .json")
