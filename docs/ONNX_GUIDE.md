# ONNX Runtime Guide

Using ONNX models with SLM Packager for optimized inference.

## Overview

ONNX Runtime provides optimized inference for ONNX models exported from frameworks like PyTorch or TensorFlow. As of v0.2, SLM Packager supports **proper ONNX text generation** with KV-cache through `onnxruntime-genai`.

## Installation

### Basic ONNX Support (CPU)
```bash
pip install -e ".[onnx]"
# Or manually:
pip install onnxruntime-genai
```

### GPU Support (CUDA)
```bash
pip install onnxruntime-genai-cuda
```

> **Note**: onnxruntime-genai requires Python 3.8-3.11

## Model Requirements

ONNX models for text generation must be in the **onnxruntime-genai format**:
- Directory containing `.onnx` model files
- `config.json` with model configuration
- Tokenizer files

### Supported Model Formats

ONNX models exported using `optimum` from HuggingFace models work well:

```bash
# Export a model to ONNX with optimum
pip install optimum[exporters]

optimum-cli export onnx \
  --model microsoft/phi-2 \
  --task text-generation-with-past \
  phi-2-onnx/
```

## Configuration

Example config for ONNX runtime:

```yaml
model:
  name: "microsoft/phi-2"
  format: "onnx"
  path: "./models/phi-2-onnx"  # Directory, not file

runtime:
  type: "onnx"
  device: "cpu"  # or "cuda"
  threads: 4
  context_size: 2048

params:
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  max_tokens: 512
  stream: true
```

## Usage

```bash
# Initialize config
slm init \
  --name "phi-2-onnx" \
  --path "./models/phi-2-onnx" \
  --format onnx \
  --runtime onnx \
  --output phi2-onnx.yaml

# Run inference
slm run phi2-onnx.yaml --prompt "Explain quantum computing"

# Benchmark
slm benchmark phi2-onnx.yaml
```

## Performance

ONNX Runtime with KV-cache provides:
- **Efficient inference**: Optimized operators
- **Low latency**: ~20-40 tokens/sec on CPU
- **Small memory footprint**: Comparable to GGUF
- **GPU acceleration**: Via CUDA execution provider

### Expected Performance (Approximate)

| Model Size | Device | Tokens/sec | Memory |
|------------|--------|------------|---------|
| 1-2B params | CPU | 20-30 | 2-3GB |
| 1-2B params | CUDA | 60-100 | 2GB VRAM |
| 7B params | CPU | 5-10 | 8-10GB |
| 7B params | CUDA | 30-50 | 7GB VRAM |

## Exporting Models to ONNX

### Using Optimum (Recommended)

```bash
# Install optimum
pip install "optimum[exporters,onnxruntime]"

# Export with KV-cache support
optimum-cli export onnx \
  --model microsoft/phi-2 \
  --task text-generation-with-past \
  --opset 14 \
  phi-2-onnx/

# Test the exported model
slm init --name phi-2 --path ./phi-2-onnx --format onnx --runtime onnx -o phi2.yaml
slm run phi2.yaml --prompt "Test"
```

### Supported Models

Models that export well to ONNX:
- ✅ GPT-2, GPT-Neo
- ✅ Phi-2, Phi-3
- ✅ LLaMA, Mistral (with optimum)
- ✅ TinyLlama, SmolLM

## GPU Acceleration

### CUDA Setup

```bash
# Install CUDA-enabled onnxruntime-genai
pip install onnxruntime-genai-cuda

# Verify CUDA is available
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Update config:
```yaml
runtime:
  type: "onnx"
  device: "cuda"  # Enable GPU
```

## Troubleshooting

### Model Not Found
**Error**: `ONNX model path not found`

**Solution**:
- Verify path points to directory (not .onnx file)
- Ensure directory contains config.json and model files
- Use absolute path or path relative to current directory

### Import Error
**Error**: `onnxruntime-genai not installed`

**Solution**:
```bash
pip install onnxruntime-genai
# Or for CUDA:
pip install onnxruntime-genai-cuda
```

### Generation Issues
**Error**: During text generation

**Solution**:
- Verify model was exported with `task=text-generation-with-past`
- Check model supports autoregressive generation
- Try with different generation parameters

### CUDA Not Available
**Error**: `CUDA provider not available`

**Solution**:
- Install `onnxruntime-genai-cuda` (not regular onnxruntime-genai)
- Verify CUDA installation
- Set `device: cpu` in config as fallback

## Comparison with Other Runtimes

| Feature | ONNX | llama.cpp | Transformers |
|---------|------|-----------|--------------|
| Speed (CPU) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Speed (GPU) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Memory | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Model Support | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Setup Complexity | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**Use ONNX when:**
- You want optimized inference without quantization
- You're deploying to production environments
- You need cross-platform compatibility
- You want GPU acceleration without large dependencies

**Use llama.cpp when:**
- You want the fastest CPU inference
- You're using GGUF quantized models
- Memory is constrained

**Use transformers when:**
- You want maximum model compatibility
- You're prototyping/experimenting
- You need latest model architectures

## Examples

### Streaming Generation
```python
from slm_packager.config import ConfigLoader
from slm_packager.runtime import get_runtime

# Load config
config = ConfigLoader.load("phi2-onnx.yaml")
config.params.stream = True

# Create runtime
runtime = get_runtime(config)
runtime.load()

# Stream tokens
for token in runtime.generate("Tell me a story about", config.params):
    print(token, end="", flush=True)

runtime.unload()
```

### Batch Processing
```python
prompts = [
    "Explain AI in simple terms",
    "What is machine learning?",
    "Define deep learning"
]

for prompt in prompts:
    output = runtime.generate(prompt, config.params)
    print(f"Q: {prompt}\nA: {output}\n")
```

## Links

- [onnxruntime-genai Documentation](https://github.com/microsoft/onnxruntime-genai)
- [Optimum ONNX Export Guide](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [SLM Packager GitHub](https://github.com/YOUR_USERNAME/slm-packager)
