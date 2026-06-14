from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class RuntimeType(str, Enum):
    LLAMA_CPP = "llama_cpp"
    ONNX = "onnx"
    TRANSFORMERS = "transformers"

    def __str__(self):
        return self.value


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

    def __str__(self):
        return self.value


class FormatType(str, Enum):
    GGUF = "gguf"
    ONNX = "onnx"
    PYTORCH = "pytorch"

    def __str__(self):
        return self.value


class QuantizationType(str, Enum):
    # GGUF types
    Q4_0 = "q4_0"
    Q4_K_M = "q4_k_m"
    Q5_K_M = "q5_k_m"
    Q8_0 = "q8_0"
    # ONNX types
    INT8 = "int8"
    NONE = "none"

    def __str__(self):
        return self.value


class ModelConfig(BaseModel):
    name: str
    path: str
    description: Optional[str] = None
    format: FormatType = Field(..., description="Model format: gguf, onnx, pytorch")


class RuntimeConfig(BaseModel):
    type: RuntimeType
    device: DeviceType = DeviceType.CPU
    threads: int = Field(default=4, ge=1)
    gpu_layers: int = Field(default=0, ge=0)
    context_size: int = Field(default=2048, ge=512)
    trust_remote_code: bool = Field(
        default=False, description="Allow running custom code from model repos (security risk)"
    )


class GenerationParams(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    max_tokens: int = Field(default=512, ge=1)
    stop: List[str] = Field(default_factory=list)
    stream: bool = False
    repetition_penalty: float = Field(default=1.1, ge=1.0)


class SLMConfig(BaseModel):
    model: ModelConfig
    runtime: RuntimeConfig
    params: GenerationParams = Field(default_factory=GenerationParams)
    quantization: Optional[QuantizationType] = None

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v, info):
        # Basic validation logic could go here
        return v
