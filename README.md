# SLM Packager

**Run small language models locally with one consistent CLI.**

SLM Packager is an open-source toolkit for running, packaging, and benchmarking Small Language Models (1B–7B parameters). One CLI over llama.cpp and transformers, plus an ONNX path for exported models — so switching runtime or quantization is a config change, not a rewrite. GGUF and PyTorch are the well-tested routes; **ONNX is experimental**.

[![PyPI](https://img.shields.io/pypi/v/slm-packager?color=blue&label=pypi)](https://pypi.org/project/slm-packager/)
[![CI](https://github.com/Ayo-Cyber/slm-packager/actions/workflows/test.yml/badge.svg)](https://github.com/Ayo-Cyber/slm-packager/actions/workflows/test.yml)
[![Tests](https://img.shields.io/badge/tests-148%20passing-brightgreen)](https://github.com/Ayo-Cyber/slm-packager/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ayo-cyber.github.io/slm-packager)

---

## Install

Pick the engine you actually need — the core install is small and pure-Python
(~50MB, a few seconds, no compiler):

```bash
pip install "slm-packager[gguf]"    # GGUF via llama.cpp — best on CPU
pip install "slm-packager[torch]"   # PyTorch/HuggingFace via transformers
pip install "slm-packager[onnx]"    # ONNX Runtime (experimental)
pip install "slm-packager[all]"     # everything
```

With `pipx` (no venv setup needed on macOS, Linux, or Windows):

```bash
pipx install "slm-packager[gguf]"
```

| Extra | Pulls | Notes |
|-------|-------|-------|
| *(core)* | click, pydantic, fastapi, huggingface-hub | CLI, config, registry, API server. Seconds to install. |
| `[gguf]` | llama-cpp-python | **Builds from source** — needs cmake and a C/C++ toolchain |
| `[torch]` | torch, transformers, accelerate | Multi-GB download |
| `[onnx]` | onnxruntime, transformers, numpy | Prebuilt wheels, no compiler |
| `[speed]` | hf_transfer | Faster model downloads |

`[gguf]` is the only one needing a compiler: `xcode-select --install` on macOS,
`build-essential` + `cmake` on Debian/Ubuntu,
[MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) on
Windows. If you'd rather skip that, `[torch]` and `[onnx]` install from prebuilt
wheels.

You can install core first and add an engine later — `slm list`, `slm pull`,
`slm init` and `slm serve` all work without one, and running a model without the
matching engine tells you exactly which extra to install.

> macOS users: if `pip install` gives an "externally-managed-environment" error, use `pipx` (install with `brew install pipx`).

## Quickstart

```bash
# Install the GGUF engine (tinyllama is a GGUF model)
pip install "slm-packager[gguf]"

# Pull a model
slm pull tinyllama

# Run it
slm run tinyllama --prompt "Explain transformers in one sentence"

# Benchmark it
slm benchmark tinyllama
```

`pull` downloads the recommended quantization and writes a config; `run` loads it and
generates. Prompts are automatically wrapped in the model's own chat template (pass
`--raw` to skip), which is what stops an instruction-tuned model from replying to a
bare prompt with nothing at all.

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
| `qwen2.5-1.5b` | 1.1GB | llama.cpp | Multilingual chat |
| `gemma2-2b` | 1.6GB | llama.cpp | Quality at small size |
| `phi-2` | 1.6GB | llama.cpp | Reasoning tasks |
| `phi-3-mini` | 2.2GB | llama.cpp | Reasoning, longer context |
| `qwen2.5-3b` | 1.9GB | llama.cpp | Multilingual + coding |
| `mistral-7b` | 4.4GB | llama.cpp | Fast general-purpose |
| `llama3-8b` | 4.9GB | llama.cpp | Best instruction following |

```bash
slm list                          # all registry models
slm pull phi-2 --list-variants    # see quantization options
```

---

## Benchmarks (M3 Pro · 18GB)

| Model | Runtime | Device | Tokens/sec |
|-------|---------|--------|-----------|
| TinyLlama 1.1B Q4_K_M | llama.cpp | CPU | 142.9 |
| GPT-2 124M | transformers | CPU | 79.7 |
| GPT-2 124M | transformers | MPS | 77.0 |
| Phi-2 2.7B Q4_K_M | llama.cpp | CPU | 30.9 |
| Qwen3 4B Q4_K_M | llama.cpp | CPU | 27.4 |

Measured with `slm benchmark <model> --runs 5 --max-tokens 128` — median of 5 timed
runs after a discarded warmup, tokens counted with the model's own tokenizer. One
machine, so treat these as a starting point rather than a leaderboard; run your own.

Two things worth knowing: MPS and CPU are within noise for a 124M model, because the
GPU dispatch overhead isn't amortized at that size — expect MPS to pull ahead on
larger PyTorch models. And memory figures from `slm benchmark` are whole-process RSS
including Python and the framework, so they're an upper bound, not weight size.

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
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
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

The server starts with no model loaded, so load one first, then generate:

```bash
slm serve --port 8000
```

```bash
# 1. Load a model (required before generating)
curl -X POST http://localhost:8000/load \
  -H "Content-Type: application/json" \
  -d '{"config_path": "~/.slm/configs/tinyllama.yaml"}'

# 2. Generate
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "params": {"max_tokens": 100}}'
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/load` | POST | Load a model from a config path (`{"config_path": "..."}`) |
| `/generate` | POST | Generate text; returns `{"text": ...}` or an SSE stream |
| `/info` | GET | Active model's config, or `{"status": "no model loaded"}` |
| `/health` | GET | Liveness check |

Set `"stream": true` in `params` for Server-Sent Events. Streamed chunks arrive as
`data: {"text": "..."}`, failures as a named `error` event, and the stream ends with
`data: [DONE]`. Loading a new model waits for in-flight generations to finish;
requests during a switch get HTTP 409. Interactive docs: `http://localhost:8000/docs`.

Prompts are wrapped in the model's chat template by default, so `/generate` returns
the same thing `slm run` would. Pass `"raw": true` to send the prompt verbatim.
Generations are serialized per model, so concurrent requests queue rather than
corrupting shared model state.

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

148 tests · 59% coverage · CI on every push

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

**Issues / Discussions:** [github.com/Ayo-Cyber/slm-packager](https://github.com/Ayo-Cyber/slm-packager)
