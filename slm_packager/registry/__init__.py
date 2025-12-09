"""Model registry management"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ModelVariant:
    """Represents a model variant (quantization type)"""
    file: str
    size: str
    speed: str
    quality: str
    recommended: bool = False

@dataclass
class ModelInfo:
    """Represents a model in the registry"""
    name: str
    description: str
    format: str
    runtime: str
    repo: str
    variants: Dict[str, ModelVariant]

class ModelRegistry:
    """Manages the model registry"""
    
    def __init__(self):
        self.registry_path = Path(__file__).parent / "models.json"
        self._registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load registry from JSON file"""
        with open(self.registry_path, 'r') as f:
            return json.load(f)
    
    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """Get model info by name"""
        if model_name not in self._registry['models']:
            return None
        
        data = self._registry['models'][model_name]
        variants = {
            k: ModelVariant(**v) 
            for k, v in data['variants'].items()
        }
        
        return ModelInfo(
            name=data['name'],
            description=data['description'],
            format=data['format'],
            runtime=data['runtime'],
            repo=data['repo'],
            variants=variants
        )
    
    def list_models(self) -> List[str]:
        """List all available model names"""
        return list(self._registry['models'].keys())
    
    def get_all_models(self) -> Dict[str, ModelInfo]:
        """Get all models as ModelInfo objects"""
        return {
            name: self.get_model(name)
            for name in self.list_models()
        }
    
    def get_recommended_variant(self, model_name: str) -> Optional[str]:
        """Get recommended quantization variant for a model"""
        model = self.get_model(model_name)
        if not model:
            return None
        
        for variant_name, variant in model.variants.items():
            if variant.recommended:
                return variant_name
        
        # Fallback to first variant
        return list(model.variants.keys())[0] if model.variants else None
