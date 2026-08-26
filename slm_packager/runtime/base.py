from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Sequence, Union

from ..config.models import GenerationParams, SLMConfig


def _truncate_at_stop(text: str, stop: Optional[Sequence[str]]) -> str:
    """Cut ``text`` at the earliest stop sequence, if any occurs.

    Engines generally halt *after* emitting the stop string, so the caller would
    otherwise see it in the output.
    """
    if not stop or not text:
        return text

    cut = min((idx for s in stop if s and (idx := text.find(s)) != -1), default=-1)
    return text if cut == -1 else text[:cut]


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

    def apply_chat_template(self, prompt: str) -> Optional[str]:
        """Wrap ``prompt`` in the model's chat format, or return None if unavailable.

        Instruction-tuned models expect their own turn markers. Passing a bare
        prompt to one usually yields an immediate end-of-sequence — an empty
        response — so runtimes that can format the turn should do so.
        """
        return None
