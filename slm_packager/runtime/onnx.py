from typing import Iterator, Union
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
except ImportError as e:
    ONNX_AVAILABLE = False
    IMPORT_ERROR = str(e)

from .base import BaseRuntime
from ..config.models import SLMConfig, GenerationParams

class OnnxRuntime(BaseRuntime):
    def load(self):
        # Check for required dependencies
        if not ONNX_AVAILABLE:
            raise ImportError(
                "❌ ONNX runtime requires 'onnxruntime' and 'transformers' packages.\n"
                "💡 Install them with:\n"
                "   pip install onnxruntime transformers\n"
                "\n"
                "   For GPU support:\n"
                "   pip install onnxruntime-gpu transformers\n"
                f"\n   Error details: {IMPORT_ERROR}"
            )
        
        print("\n⚠️  WARNING: ONNX runtime support is currently limited in v0.1")
        print("   - Generation uses simplified loop without KV-cache")
        print("   - Performance may be slower than expected")
        print("   - This is a known limitation being improved\n")
        
        model_path = Path(self.config.model.path)
        
        # Check if model file exists
        if not model_path.exists():
            raise FileNotFoundError(
                f"❌ ONNX model file not found: '{self.config.model.path}'\n"
                "💡 Check that:\n"
                "   - The file path is correct\n"
                "   - The .onnx file was fully downloaded\n"
                f"   - You're running from the correct directory (current: {Path.cwd()})"
            )
        
        # Check file extension
        if not str(model_path).endswith('.onnx'):
            raise ValueError(
                f"❌ File doesn't appear to be an ONNX model: '{self.config.model.path}'\n"
                "💡 ONNX models must have .onnx extension\n"
                "   - For PyTorch models, use 'transformers' runtime\n"
                "   - For GGUF models, use 'llama_cpp' runtime"
            )
        
        try:
            print(f"📥 Loading tokenizer for '{self.config.model.name}'...")
            
            # Try to load tokenizer from model name (assumed to be HF repo ID)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.name,
                trust_remote_code=True
            )
            
            print("✅ Tokenizer loaded")
            
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to load tokenizer for '{self.config.model.name}'\n"
                f"   Error: {str(e)}\n"
                "💡 For ONNX models:\n"
                "   - Set model.name to the HuggingFace repo ID (e.g., 'microsoft/phi-2')\n"
                "   - Or provide a local path with tokenizer files\n"
                "   - Tokenizer is needed for text encoding/decoding"
            ) from e
        
        try:
            print(f"📥 Loading ONNX model from '{self.config.model.path}'...")
            
            sess_options = ort.SessionOptions()
            if self.config.runtime.threads > 0:
                sess_options.intra_op_num_threads = self.config.runtime.threads
                
            providers = ["CPUExecutionProvider"]
            if self.config.runtime.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options,
                providers=providers
            )
            self.model = self.session
            
            print("✅ ONNX model loaded")
            
        except Exception as e:
            error_str = str(e).lower()
            if "cuda" in error_str and "not available" in error_str:
                raise RuntimeError(
                    "❌ CUDA provider not available\n"
                    "💡 Try:\n"
                    "   - Set device to 'cpu' in your config\n"
                    "   - Install onnxruntime-gpu: pip install onnxruntime-gpu\n"
                    "   - Ensure CUDA is properly installed"
                ) from e
            else:
                raise RuntimeError(
                    f"❌ Error loading ONNX model\n"
                    f"   {type(e).__name__}: {str(e)}\n"
                    "💡 Check:\n"
                    "   - The .onnx file is valid\n"
                    "   - The model is compatible with onnxruntime"
                ) from e

    def generate(self, prompt: str, params: GenerationParams) -> Union[str, Iterator[str]]:
        if not self.is_loaded:
            raise RuntimeError(
                "❌ Model is not loaded. Call runtime.load() first.\n"
                "💡 If using CLI, this is a bug - please report it."
            )

        try:
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="np")
            
            # IMPORTANT NOTE: This is a placeholder implementation
            # Full ONNX generation requires complex KV-cache management
            # This is mentioned in the README as a known limitation
            
            warning_msg = (
                f"\n⚠️  ONNX generation not fully implemented in v0.1\n"
                f"   Input prompt: \"{prompt}\"\n"
                f"   Model: {self.config.model.name}\n"
                f"\n"
                f"   This runtime needs:\n"
                f"   - Proper KV-cache implementation\n"
                f"   - Full generation loop with sampling\n"
                f"\n"
                f"   For now, use 'transformers' runtime for full generation capability.\n"
                f"   See IMPROVEMENTS.md for roadmap.\n"
            )
            
            return warning_msg
            
        except Exception as e:
            raise RuntimeError(
                f"❌ Error during ONNX generation\n"
                f"   {type(e).__name__}: {str(e)}\n"
                "💡 ONNX runtime is currently experimental\n"
                "   - Consider using 'transformers' runtime for now\n"
                "   - See IMPROVEMENTS.md for current limitations"
            ) from e

    def unload(self):
        if self.session:
            self.session = None
        if self.model:
            self.model = None
        if self.tokenizer:
            self.tokenizer = None

