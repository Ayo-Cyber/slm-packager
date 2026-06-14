# SLM Packager

**Run any small language model locally — one command.**

SLM Packager is an open-source toolkit for running, packaging, and benchmarking Small Language Models (1B–7B parameters) across GGUF, PyTorch, and ONNX formats. One unified CLI. Three runtimes. Zero friction.

[![PyPI](https://img.shields.io/pypi/v/slm-packager?color=blue&label=pypi)](https://pypi.org/project/slm-packager/)
[![CI](https://github.com/Ayo-Cyber/slm-packager/actions/workflows/test.yml/badge.svg)](https://github.com/Ayo-Cyber/slm-packager/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-117%20tests-brightgreen)](https://github.com/Ayo-Cyber/slm-packager/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ayo-cyber.github.io/slm-packager)

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

## Pull Any HuggingFace Model

Not in the registry? Pull any GGUF or ONNX directly:

```bash
slm pull Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --name qwen3-4b
slm run qwen3-4b --prompt "Hello!"
```

---

## Available Models

| Model | Size | Runtime | Best For |
|-------|------|---------|----------|
| `gpt2` | 500MB | transformers | Fast testing, MPS |
| `tinyllama` | 637MB | llama.cpp | CPU-efficient chat |
| `phi-2` | 1.6GB | llama.cpp | Reasoning tasks |
| `qwen-1.8b` | 1.1GB | llama.cpp | Multilingual chat |

```bash
slm list                          # all registry models
slm pull phi-2 --list-variants    # see quantization options
```

---

## Real Benchmarks (M3 Pro · 18GB)

| Model | Runtime | Device | Tokens/sec |
|-------|---------|--------|-----------|
| GPT-2 124M | transformers | CPU | 53.16 |
| GPT-2 124M | transformers | MPS ⚡ | 28.06 |
| TinyLlama 1.1B | llama.cpp | CPU | 9.19 |
| Phi-2 2.7B | llama.cpp | CPU | 33.67 |
| Qwen3 4B | llama.cpp | CPU | 31.71 |

Run your own: `slm benchmark <model>`

---

## GPU Acceleration

### Apple Silicon (MPS) — zero setup

```bash
slm init --name gpt2 --path gpt2 --format pytorch \
         --runtime transformers --device mps -o gpt2-mps.yaml
slm run gpt2-mps.yaml --prompt "Hello!"
```

### NVIDIA (CUDA)

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir
# then set gpu_layers in your YAML config
```

---

## CLI Reference

```bash
slm list                              # registry models
slm list --installed                  # downloaded models
slm pull <model>                      # download from registry
slm pull <hf-repo> <file> --name x   # pull any HF GGUF/ONNX
slm run <model> --prompt "..."        # generate text
slm benchmark <model>                 # speed + memory metrics
slm serve --port 8000                 # start FastAPI server
slm quantize input.gguf --type q4_k_m
slm init                              # create YAML config interactively
slm rm <model>                        # remove installed model
```

---

## API Server

```bash
slm serve --port 8000
```

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "params": {"max_tokens": 100}}'
```

Supports streaming (`"stream": true`) and async model loading.

---

## YAML Config

```yaml
model:
  name: my-model
  path: /path/to/model.gguf
  format: gguf

runtime:
  type: llama_cpp
  device: cpu
  threads: 8
  context_size: 2048

params:
  temperature: 0.7
  max_tokens: 512
  stream: true
```

---

## Documentation

Full docs at **[ayo-cyber.github.io/slm-packager](https://ayo-cyber.github.io/slm-packager)**

- [Quick Start](https://ayo-cyber.github.io/slm-packager/quickstart/)
- [CLI Reference](https://ayo-cyber.github.io/slm-packager/cli-reference/)
- [Runtimes](https://ayo-cyber.github.io/slm-packager/runtimes/)
- [GPU Acceleration](https://ayo-cyber.github.io/slm-packager/gpu-acceleration/)
- [Benchmarks](https://ayo-cyber.github.io/slm-packager/benchmarks/)

---

## Development

```bash
git clone https://github.com/Ayo-Cyber/slm-packager.git
cd slm-packager
pip install -e ".[dev]"
pytest
```

117 tests · 52% coverage · CI on every push

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

**Issues / Discussions:** [github.com/Ayo-Cyber/slm-packager](https://github.com/Ayo-Cyber/slm-packager)
