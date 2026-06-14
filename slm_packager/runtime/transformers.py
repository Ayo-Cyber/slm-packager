import sys
from typing import Iterator, Union

IMPORT_ERROR = ""
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

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

from ..config.models import GenerationParams, SLMConfig
from .base import BaseRuntime


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
            # Determine device and configuration
            device_config = self._configure_device()

            print(f"📥 Loading tokenizer from '{self.config.model.path}'...")
            trust_remote = self.config.runtime.trust_remote_code
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.path, trust_remote_code=trust_remote
            )

            print(f"📥 Loading model from '{self.config.model.path}'...")
            print(f"   Device: {device_config['name']}")
            print(f"   This may take a while for large models...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.path,
                device_map=device_config["device_map"],
                torch_dtype=device_config["dtype"],
                trust_remote_code=trust_remote,
            )

            # Move to MPS if needed (device_map doesn't support MPS yet)
            if self.config.runtime.device == "mps":
                self.model = self.model.to(device_config["device"])
                print(f"✅ Model loaded successfully on Apple Silicon GPU (MPS)!")
            else:
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

    def _configure_device(self):
        """Configure device settings based on runtime config."""
        device = self.config.runtime.device

        # MPS (Apple Silicon GPU)
        if device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "❌ MPS requested but not available\n"
                    "💡 MPS requires:\n"
                    "   - macOS 12.3+\n"
                    "   - Apple Silicon (M1/M2/M3)\n"
                    "   - PyTorch 1.12+\n"
                    "\n   Falling back to CPU might work - set device: 'cpu' in config"
                )
            return {
                "device": torch.device("mps"),
                "device_map": None,  # MPS doesn't support device_map yet
                "dtype": torch.float32,  # MPS works best with float32
                "name": "MPS (Apple Silicon GPU)",
            }

        # CUDA (NVIDIA GPU)
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "❌ CUDA requested but not available\n"
                    "💡 Ensure:\n"
                    "   - NVIDIA GPU is present\n"
                    "   - CUDA drivers are installed\n"
                    "   - PyTorch CUDA version matches your CUDA version"
                )
            return {
                "device": None,
                "device_map": "auto",
                "dtype": torch.float16,  # FP16 for faster GPU inference
                "name": "CUDA (NVIDIA GPU)",
            }

        # CPU (default)
        else:
            return {
                "device": torch.device("cpu"),
                "device_map": None,
                "dtype": "auto",
                "name": "CPU",
            }

    def generate(self, prompt: str, params: GenerationParams) -> Union[str, Iterator[str]]:
        if not self.is_loaded:
            raise RuntimeError(
                "❌ Model is not loaded. Call runtime.load() first.\n"
                "💡 If using CLI, this is a bug - please report it."
            )

        try:
            # Determine device for inputs
            device = self.model.device if hasattr(self.model, "device") else "cpu"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            if params.stream:
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
                generation_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=params.max_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    top_k=params.top_k,
                    repetition_penalty=params.repetition_penalty,
                    do_sample=True,
                )
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                return self._stream_generator(streamer)
            else:
                input_length = inputs["input_ids"].shape[1]
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=params.max_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    top_k=params.top_k,
                    repetition_penalty=params.repetition_penalty,
                    do_sample=True,
                )
                # Strip prompt tokens, not prompt characters, to avoid truncation bugs
                return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

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
        if hasattr(self, "tokenizer") and self.tokenizer:
            self.tokenizer = None
        if self.model:
            self.model = None

        # Clear GPU cache only if torch was successfully imported
        if TRANSFORMERS_AVAILABLE:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
