# SLM Packager

**Run any small language model locally — one command.**

SLM Packager is an open-source toolkit for running, packaging, and benchmarking Small Language Models (1B–7B parameters) across GGUF, PyTorch, and ONNX formats. One unified CLI. Three runtimes. Zero friction.

---

## Install

```bash
pip install slm-packager
```

## Quickstart

```bash
# Pull a model
slm pull tinyllama

# Run it
slm run tinyllama --prompt "Explain transformers in one sentence"

# Benchmark it
slm benchmark tinyllama
```

That's it. Model downloads, auto-configures, and runs.

---

## Why SLM Packager?

Running small language models means juggling different formats (GGUF, PyTorch, ONNX), runtimes (llama.cpp, transformers, onnxruntime), and configuration options. SLM Packager provides:

- **Unified interface** — one CLI and Python API for all runtimes
- **Auto-configuration** — models work out-of-the-box with sensible defaults
- **Any HuggingFace model** — pull any GGUF or ONNX file directly, no registry needed
- **GPU acceleration** — MPS on Apple Silicon, CUDA on NVIDIA, Metal via llama.cpp
- **Reproducibility** — YAML configs that describe exactly how a model should run
- **API server** — FastAPI-based serving with streaming support

---

## Architecture

```
┌─────────────────────────────────┐
│   CLI  ·  API Server            │
├─────────────────────────────────┤
│   Runtime Abstraction Layer     │
│   load()  ·  generate()  ·  unload()  │
├──────────┬──────────┬───────────┤
│ llama.cpp│Transformers│   ONNX  │
│  (GGUF)  │ (PyTorch) │ Runtime  │
└──────────┴──────────┴───────────┘
```

Three runtimes — one interface. Switch runtimes by changing one line in your YAML config.

---

## Real Benchmarks (M3 Pro · 18GB)

| Model | Runtime | Device | Tokens/sec |
|-------|---------|--------|-----------|
| GPT-2 124M | transformers | CPU | 53.16 |
| GPT-2 124M | transformers | MPS ⚡ | 28.06 |
| TinyLlama 1.1B | llama.cpp | CPU | 9.19 |
| Phi-2 2.7B | llama.cpp | CPU | 33.67 |
| Qwen3 4B | llama.cpp | CPU | 31.71 |

→ [Full benchmark details](benchmarks.md)

---

## Next Steps

- [Quick Start](QUICKSTART.md) — full walkthrough from install to serving
- [CLI Reference](cli-reference.md) — every command and flag
- [Runtimes](runtimes.md) — choosing between llama.cpp, transformers, and ONNX
- [GPU Acceleration](GPU_ACCELERATION.md) — MPS, CUDA, Metal setup
- [Benchmarks](benchmarks.md) — methodology and full results
