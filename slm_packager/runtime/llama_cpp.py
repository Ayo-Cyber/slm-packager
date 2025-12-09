from typing import Iterator, Union
import logging
import sys
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError as e:
    LLAMA_CPP_AVAILABLE = False
    IMPORT_ERROR = str(e)

from .base import BaseRuntime
from ..config.models import SLMConfig, GenerationParams

logger = logging.getLogger(__name__)

class LlamaCppRuntime(BaseRuntime):
    def load(self):
        # Check for llama-cpp-python dependency
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama.cpp runtime requires 'llama-cpp-python' package.\n"
                "Install it with:\n"
                "   pip install llama-cpp-python\n"
                "\n"
                "   For Metal support (Apple Silicon M1/M2/M3):\n"
                "   CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python --no-cache-dir\n"
                "\n"
                "   For CUDA support (NVIDIA GPU):\n"
                "   CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python --no-cache-dir\n"
                f"\n   Error details: {IMPORT_ERROR}"
            )
        
        model_path = Path(self.config.model.path)
        
        # Check if model file exists
        if not model_path.exists():
            raise FileNotFoundError(
                f"GGUF model file not found: '{self.config.model.path}'\n"
                "Suggestions:\n"
                "   - Verify the file path is correct (use absolute or relative path)\n"
                "   - Ensure the .gguf file was fully downloaded (check file size)\n"
                f"   - Confirm you're running from the correct directory (current: {Path.cwd()})\n"
                "\n"
                "   Download GGUF models from:\n"
                "   https://huggingface.co/TheBloke (search for '[model name] GGUF')"
            )
        
        # Check if it's actually a file (not a directory)
        if model_path.is_dir():
            raise ValueError(
                f"Path is a directory, not a GGUF file: '{self.config.model.path}'\n"
                "For GGUF models:\n"
                "   - Point to the .gguf file directly\n"
                "   - Example: './models/tinyllama.Q4_K_M.gguf'\n"
                "\n"
                "   For HuggingFace models, use pytorch format and transformers runtime"
            )
        
        # Check file extension
        if not str(model_path).endswith('.gguf'):
            raise ValueError(
                f"File doesn't appear to be a GGUF model: '{self.config.model.path}'\n"
                "GGUF models must have .gguf extension\n"
                "   - Verify you downloaded the correct file\n"
                "   - For PyTorch models, use 'transformers' runtime instead\n"
                "   - For ONNX models, use 'onnx' runtime instead"
            )
        
        try:
            logger.info(f"Loading GGUF model from '{self.config.model.path}'")
            logger.debug(f"Context size: {self.config.runtime.context_size}")
            logger.debug(f"GPU layers: {self.config.runtime.gpu_layers}")
            logger.debug(f"Threads: {self.config.runtime.threads}")
            
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.runtime.context_size,
                n_gpu_layers=self.config.runtime.gpu_layers,
                n_threads=self.config.runtime.threads,
                verbose=False
            )
            
            logger.info("Model loaded successfully")
            
        except ValueError as e:
            error_str = str(e).lower()
            if "invalid" in error_str or "corrupt" in error_str:
                raise ValueError(
                    f"Invalid or corrupted GGUF file\n"
                    f"   Error: {str(e)}\n"
                    "Suggestions:\n"
                    "   - Re-download the model file\n"
                    "   - Verify the file isn't corrupted (check file size)\n"
                    "   - Download from a trusted source (TheBloke on HuggingFace)"
                ) from e
            else:
                raise RuntimeError(
                    f"Error loading GGUF model\n"
                    f"   {str(e)}\n"
                    "Suggestions:\n"
                    "   - Verify the model file is valid\n"
                    "   - Ensure you have enough RAM available\n"
                    "   - Check the quantization type is supported"
                ) from e
                
        except MemoryError as e:
            raise MemoryError(
                "Out of memory loading model!\n"
                "Suggestions:\n"
                "   - Use a smaller model\n"
                "   - Use more aggressive quantization (Q4_K_M instead of Q8_0)\n"
                "   - Reduce context_size in your config\n"
                "   - Close other applications\n"
                f"   - Current context size: {self.config.runtime.context_size}"
            ) from e
            
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error loading GGUF model\n"
                f"   {type(e).__name__}: {str(e)}\n"
                "Suggestions:\n"
                "   - Verify the model file is valid\n"
                "   - Ensure llama-cpp-python is correctly installed\n"
                "   - For GPU issues, check Metal/CUDA is available"
            ) from e

    def generate(self, prompt: str, params: GenerationParams) -> Union[str, Iterator[str]]:
        if not self.is_loaded:
            raise RuntimeError(
                "Model is not loaded. Call runtime.load() first.\n"
                "If using CLI, this is a bug - please report it."
            )

        try:
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
                
        except KeyError as e:
            raise RuntimeError(
                f"Unexpected model output format\n"
                f"   Missing key: {str(e)}\n"
                "This might be a bug - please report it with:\n"
                "   - Your model name\n"
                "   - The command you ran"
            ) from e
            
        except Exception as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or ("cuda" in error_str and "memory" in error_str):
                raise MemoryError(
                    "Out of memory during generation!\n"
                    "Suggestions:\n"
                    "   - Reduce max_tokens in your config\n"
                    "   - Reduce context_size\n"
                    "   - Use a smaller model or more aggressive quantization"
                ) from e
            else:
                raise RuntimeError(
                    f"Error during text generation\n"
                    f"   {type(e).__name__}: {str(e)}\n"
                    "Check your generation parameters in the config"
                ) from e

    def _stream_generator(self, output_stream) -> Iterator[str]:
        try:
            for chunk in output_stream:
                text = chunk["choices"][0]["text"]
                yield text
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"\nStream error: {str(e)}\n"

    def unload(self):
        if self.model:
            del self.model
            self.model = None
            logger.info("Model unloaded")
