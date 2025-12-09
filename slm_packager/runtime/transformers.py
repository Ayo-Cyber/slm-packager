from typing import Iterator, Union
import sys

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    
try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

from threading import Thread
from .base import BaseRuntime
from ..config.models import SLMConfig, GenerationParams

class TransformersRuntime(BaseRuntime):
    def load(self):
        # Check for required dependencies
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "❌ Transformers runtime requires 'transformers' and 'torch' packages.\n"
                "💡 Install them with: pip install transformers torch\n"
                f"   Error details: {IMPORT_ERROR}"
            )
        
        if not ACCELERATE_AVAILABLE:
            raise ImportError(
                "❌ Transformers runtime requires 'accelerate' package for model loading.\n"
                "💡 Install it with: pip install accelerate\n"
                "   Or reinstall slm-packager: pip install -e ."
            )
        
        try:
            device_map = "auto" if self.config.runtime.device == "cuda" else "cpu"
            
            print(f"📥 Loading tokenizer from '{self.config.model.path}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.path,
                trust_remote_code=True
            )
            
            print(f"📥 Loading model from '{self.config.model.path}'...")
            print(f"   Device: {device_map}")
            print(f"   This may take a while for large models...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.path,
                device_map=device_map,
                torch_dtype="auto",
                trust_remote_code=True
            )
            
            print("✅ Model loaded successfully!")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"❌ Model not found: '{self.config.model.path}'\n"
                "💡 If using a HuggingFace model:\n"
                "   - Check the model ID is correct (format: 'namespace/repo-name')\n"
                "   - Ensure you have internet connection\n"
                "   - Try visiting: https://huggingface.co/{self.config.model.path}\n"
                "💡 If using a local model:\n"
                "   - Check the file path exists\n"
                "   - Use absolute path or relative to current directory"
            ) from e
        except OSError as e:
            if "401 Client Error" in str(e):
                raise RuntimeError(
                    f"❌ Authentication required for '{self.config.model.path}'\n"
                    "💡 This model requires HuggingFace authentication:\n"
                    "   1. Create account at https://huggingface.co\n"
                    "   2. Get token from https://huggingface.co/settings/tokens\n"
                    "   3. Run: huggingface-cli login\n"
                    "   4. Paste your token when prompted"
                ) from e
            elif "Repository Not Found" in str(e):
                raise ValueError(
                    f"❌ Model repository not found: '{self.config.model.path}'\n"
                    "💡 Check that:\n"
                    "   - The model ID is spelled correctly\n"
                    "   - The model exists on HuggingFace\n"
                    "   - Format is 'namespace/repo-name' (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
                ) from e
            else:
                raise RuntimeError(
                    f"❌ Error loading model: {str(e)}\n"
                    "💡 Check your internet connection and model path"
                ) from e
        except Exception as e:
            raise RuntimeError(
                f"❌ Unexpected error loading model:\n"
                f"   {type(e).__name__}: {str(e)}\n"
                "💡 Try:\n"
                "   - Using a different model\n"
                "   - Checking available disk space\n"
                "   - Ensuring sufficient RAM (need ~4GB+ for small models)"
            ) from e

    def generate(self, prompt: str, params: GenerationParams) -> Union[str, Iterator[str]]:
        if not self.is_loaded:
            raise RuntimeError(
                "❌ Model is not loaded. Call runtime.load() first.\n"
                "💡 If using CLI, this is a bug - please report it."
            )

        try:
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
        
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                "❌ GPU out of memory!\n"
                "💡 Try:\n"
                "   - Using a smaller model\n"
                "   - Setting device to 'cpu' in config\n"
                "   - Reducing context_size in config\n"
                "   - Using a quantized GGUF model instead"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"❌ Error during generation:\n"
                f"   {type(e).__name__}: {str(e)}\n"
                "💡 Check your generation parameters in the config"
            ) from e

    def _stream_generator(self, streamer) -> Iterator[str]:
        for new_text in streamer:
            yield new_text

    def unload(self):
        if self.model:
            self.model = None
        if self.tokenizer:
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

