# Example Configurations

This directory contains example configuration files for popular Small Language Models.

## Quick Start

To use any example:

```bash
# From the slm-packager root directory
slm run examples/tinyllama-example.yaml --prompt "Hello, how are you?"
```

## Available Examples

### tinyllama-example.yaml
- **Model**: TinyLlama 1.1B Chat
- **Runtime**: Transformers (PyTorch)
- **Device**: CPU
- **Best For**: Quick testing, learning, CPU-only machines
- **Memory**: ~2-4GB RAM

**Try it:**
```bash
slm run examples/tinyllama-example.yaml --prompt "Explain what a neural network is"
```

---

## Creating Your Own Config

### Option 1: Interactive
```bash
slm init
```

### Option 2: Manual
Copy and edit one of these examples:

```bash
cp examples/tinyllama-example.yaml my-model.yaml
# Edit my-model.yaml with your model details
slm run my-model.yaml
```

### Option 3: Command Line
```bash
slm init \
  --name my-model \
  --path path/to/model \
  --format pytorch \
  --runtime transformers \
  --output my-model.yaml
```

---

## Configuration Options

### Model Formats & Runtimes

| Format | Runtime | Use Case |
|--------|---------|----------|
| `gguf` | `llama_cpp` | Fast CPU inference, quantized models |
| `pytorch` | `transformers` | HuggingFace models, GPU support |
| `onnx` | `onnx` | Cross-platform, optimized inference |

### Device Options

- `cpu` - CPU inference (works everywhere)
- `cuda` - NVIDIA GPU (fastest for large models)
- `mps` - Apple Silicon (M1/M2/M3 Macs)

### Generation Parameters

- `temperature` (0.0-2.0): Lower = more focused, Higher = more creative
- `top_p` (0.0-1.0): Nucleus sampling threshold
- `top_k` (int): Consider top K tokens
- `max_tokens` (int): Maximum response length
- `stream` (bool): Stream output token-by-token

---

## Tips

1. **Start small**: Use TinyLlama (1.1B) before trying larger models
2. **GGUF is fastest**: For CPU, use GGUF format with llama_cpp runtime
3. **GPU acceleration**: Set `device: cuda` and `gpu_layers: 32` if you have NVIDIA GPU
4. **Reduce memory**: Lower `context_size` or use more quantization
5. **Tune temperature**: 0.1-0.3 for factual, 0.7-0.9 for creative

---

## More Models

You can use any HuggingFace model ID in the `path` field:

```yaml
# Phi-2 (2.7B)
path: microsoft/phi-2

# Qwen 1.8B
path: Qwen/Qwen-1_8B-Chat

# Gemma 2B
path: google/gemma-2b-it
```

For GGUF models, download from TheBloke's HuggingFace repos and use local path.
