# Runtimes

SLM Packager supports three inference runtimes. They all share the same interface (`load()`, `generate()`, `unload()`) — switching is a one-line change in your YAML config.

---

## Comparison

| Runtime | Format | Best For | Pros | Cons |
|---------|--------|----------|------|------|
| **llama.cpp** | GGUF | Production, CPU-constrained | Low memory, fast quantized inference, Metal/CUDA | GGUF format only |
| **transformers** | PyTorch (HF) | Development, latest models | Widest model support, MPS/CUDA, streaming | Higher memory usage |
| **ONNX** | `.onnx` | Cross-platform, optimized pipelines | Fast CPU inference, portable | Requires model export first |

---

## llama.cpp (GGUF)

Best choice for deploying quantized models on CPU or Apple Silicon Metal.

```yaml
runtime:
  type: llama_cpp
  device: cpu
  threads: 8
  gpu_layers: 0        # set >0 to offload to GPU
  context_size: 2048
```

**When to use:**
- You want the lowest possible memory footprint
- You're deploying on a machine without a GPU
- You have a GGUF model (from HuggingFace or `slm quantize`)
- You want quantization (Q4_K_M is a great default)

**Pull a GGUF model:**
```bash
slm pull tinyllama
slm pull Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --name qwen3-4b
```

---

## Transformers (PyTorch)

Best choice for development, experimentation, and models that don't have a GGUF variant yet.

```yaml
runtime:
  type: transformers
  device: cpu           # or mps, cuda
  trust_remote_code: false
```

**When to use:**
- You want to use a model straight from HuggingFace without format conversion
- You have Apple Silicon and want MPS acceleration
- You're fine-tuning or experimenting
- You need a model that only exists in PyTorch format

**Pull a PyTorch model:**
```bash
slm pull gpt2
```

**Enable MPS (Apple Silicon):**
```bash
slm init --name gpt2 --path gpt2 --format pytorch \
         --runtime transformers --device mps -o gpt2-mps.yaml
```

---

## ONNX Runtime

Best choice when you have an exported ONNX model or need to integrate with an existing ML pipeline.

```yaml
runtime:
  type: onnx
  device: cpu
```

**When to use:**
- You already have a `.onnx` model file
- You need a portable, optimized inference graph
- You're targeting a cross-platform deployment

**Export a model to ONNX first:**
```bash
pip install "optimum[exporters]"
optimum-cli export onnx --model gpt2 --task text-generation-with-past models/gpt2-onnx/
```

Then run it:
```bash
slm init --name gpt2-onnx --path models/gpt2-onnx --format onnx --runtime onnx
slm run gpt2-onnx --prompt "Hello!"
```

→ See the [ONNX Guide](ONNX_GUIDE.md) for the full export workflow.
