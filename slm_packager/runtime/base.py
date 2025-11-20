from abc import ABC, abstractmethod
from typing import Iterator, Union, Dict, Any
from ..config.models import SLMConfig, GenerationParams

class BaseRuntime(ABC):
    def __init__(self, config: SLMConfig):
        self.config = config
        self.model = None

    @abstractmethod
    def load(self):
        """Load the model into memory."""
        pass

    @abstractmethod
    def generate(self, prompt: str, params: GenerationParams) -> Union[str, Iterator[str]]:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def unload(self):
        """Unload the model and free resources."""
        pass

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
