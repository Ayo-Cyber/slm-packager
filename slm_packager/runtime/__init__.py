from .base import BaseRuntime
from .llama_cpp import LlamaCppRuntime
from .onnx import OnnxRuntime
from .transformers import TransformersRuntime
from ..config.models import SLMConfig, RuntimeType

def get_runtime(config: SLMConfig) -> BaseRuntime:
    if config.runtime.type == RuntimeType.LLAMA_CPP:
        return LlamaCppRuntime(config)
    elif config.runtime.type == RuntimeType.ONNX:
        return OnnxRuntime(config)
    elif config.runtime.type == RuntimeType.TRANSFORMERS:
        return TransformersRuntime(config)
    else:
        raise ValueError(f"Unsupported runtime type: {config.runtime.type}")
