# SLM Packager

**Run any small language model locally — one command.**

SLM Packager is an open-source toolkit for running, packaging, and benchmarking Small Language Models (1B–7B parameters). One CLI over llama.cpp and transformers, plus an ONNX path for exported models — so switching runtime or quantization is a config change, not a rewrite. GGUF and PyTorch are the well-tested routes; **ONNX is experimental**.

---

## Install

Install the engine you need. Core is small and pure-Python (~50MB, seconds, no
compiler):

```bash
pip install "slm-packager[gguf]"    # GGUF via llama.cpp — best on CPU
pip install "slm-packager[torch]"   # PyTorch/HuggingFace via transformers
pip install "slm-packager[onnx]"    # ONNX Runtime (experimental)
pip install "slm-packager[all]"     # everything
```

| Extra | Pulls | Notes |
|-------|-------|-------|
| *(core)* | click, pydantic, fastapi, huggingface-hub | CLI, config, registry, API server |
| `[gguf]` | llama-cpp-python | **Builds from source** — needs cmake and a C/C++ toolchain |
| `[torch]` | torch, transformers, accelerate | Multi-GB download |
| `[onnx]` | onnxruntime, transformers, numpy | Prebuilt wheels, no compiler |
| `[speed]` | hf_transfer | Faster model downloads |

Core alone is enough for `slm list`, `slm pull`, `slm init`, and `slm serve`. Add an
engine when you want to actually generate; running a model without the matching
engine names the extra to install.

With `pipx` (no venv setup needed):
```bash
pipx install "slm-packager[gguf]"
```

!!! tip "macOS users"
    If `pip install` gives an *externally-managed-environment* error, use `pipx` instead:
    `brew install pipx && pipx install "slm-packager[gguf]"`

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

- [Quick Start](quickstart.md) — full walkthrough from install to serving
- [CLI Reference](cli-reference.md) — every command and flag
- [Runtimes](runtimes.md) — choosing between llama.cpp, transformers, and ONNX
- [GPU Acceleration](gpu-acceleration.md) — MPS, CUDA, Metal setup
- [Benchmarks](benchmarks.md) — methodology and full results
