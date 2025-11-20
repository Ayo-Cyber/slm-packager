from typing import Iterator, Union
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    import torch
except ImportError:
    AutoModelForCausalLM = None
    
from threading import Thread
from .base import BaseRuntime
from ..config.models import SLMConfig, GenerationParams

class TransformersRuntime(BaseRuntime):
    def load(self):
        if AutoModelForCausalLM is None:
            raise ImportError("transformers and torch are required for Transformers runtime.")
            
        device_map = "auto" if self.config.runtime.device == "cuda" else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.path,
            device_map=device_map,
            torch_dtype="auto"
        )

    def generate(self, prompt: str, params: GenerationParams) -> Union[str, Iterator[str]]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        if params.stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                do_sample=True
            )
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            return self._stream_generator(streamer)
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]

    def _stream_generator(self, streamer) -> Iterator[str]:
        for new_text in streamer:
            yield new_text

    def unload(self):
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
