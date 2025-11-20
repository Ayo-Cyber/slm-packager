from typing import Iterator, Union
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from .base import BaseRuntime
from ..config.models import SLMConfig, GenerationParams

class LlamaCppRuntime(BaseRuntime):
    def load(self):
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed. Please install it to use this runtime.")
        
        self.model = Llama(
            model_path=self.config.model.path,
            n_ctx=self.config.runtime.context_size,
            n_gpu_layers=self.config.runtime.gpu_layers,
            n_threads=self.config.runtime.threads,
            verbose=False
        )

    def generate(self, prompt: str, params: GenerationParams) -> Union[str, Iterator[str]]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        output = self.model(
            prompt,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            stop=params.stop,
            stream=params.stream
        )

        if params.stream:
            return self._stream_generator(output)
        else:
            return output["choices"][0]["text"]

    def _stream_generator(self, output_stream) -> Iterator[str]:
        for chunk in output_stream:
            text = chunk["choices"][0]["text"]
            yield text

    def unload(self):
        if self.model:
            del self.model
            self.model = None
