# ONNX Runtime Guide

Using ONNX models with SLM Packager for optimized inference.

## Overview

ONNX Runtime provides optimized inference for ONNX models exported from frameworks
like PyTorch or TensorFlow. SLM Packager runs ONNX text generation with KV-cache on
plain `onnxruntime` plus `transformers` for tokenization, both provided by the
optional `[onnx]` extra.

!!! note "onnxruntime-genai is no longer used"
    Releases before v0.2.0 required `onnxruntime-genai`. It was dropped because the
    generation path didn't work reliably; do **not** install it. The ONNX runtime is
    still the least mature of the three — prefer llama.cpp for GGUF models.

## Installation

Install the ONNX extra — prebuilt wheels, no compiler needed:

```bash
pip install "slm-packager[onnx]"
```

### GPU Support (CUDA)

```bash
pip install onnxruntime-gpu

# Verify the CUDA provider is visible
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include: CUDAExecutionProvider
```

## Model Requirements

An ONNX model directory needs:
- One or more `.onnx` model files (exported with past-key-value support)
- `config.json` with the model configuration
- Tokenizer files loadable by `transformers.AutoTokenizer`

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
- **Optimized operators** via ONNX Runtime
- **GPU acceleration** via the CUDA execution provider

No throughput figures are published for this runtime: it is experimental and has no
benchmark coverage. Measure your own model with `slm benchmark`, and prefer llama.cpp
for GGUF or transformers for PyTorch if you have the choice.

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
# Replace the CPU build with the GPU build
pip uninstall -y onnxruntime
pip install onnxruntime-gpu

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
**Error**: `onnxruntime not installed` or `transformers not installed`

**Solution**: install the ONNX extra, which provides both:
```bash
pip install "slm-packager[onnx]"
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
- Install `onnxruntime-gpu` in place of `onnxruntime`
- Verify your CUDA installation
- Set `device: cpu` in config as a fallback

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

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [Optimum ONNX Export Guide](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [SLM Packager GitHub](https://github.com/Ayo-Cyber/slm-packager)
