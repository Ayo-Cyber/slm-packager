# SLM Packager

**A Unified Runtime & Developer Layer for Small Language Models**

SLM Packager is an open-source toolkit for running, packaging, and evaluating Small Language Models (1B-7B parameters) across different formats and runtimes. Think of it as **Terraform for SLMs** — making model deployment simple, reproducible, and developer-friendly.

[![Tests](https://img.shields.io/badge/tests-73%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-52%25-yellow)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## ✨ Features

- 🎯 **Model Registry**: One-command downloads from HuggingFace with `slm pull`
- 🔄 **Multi-Runtime Support**: llama.cpp (GGUF), Transformers (PyTorch), ONNX
- ⚡ **GPU Acceleration**: MPS (Apple Silicon), CUDA (NVIDIA), Metal (llama.cpp)
- ⚙️ **Auto-Quantization**: On-device model quantization with automatic tool setup
- 📊 **Benchmarking**: Measure speed, memory, latency across runtimes
- 🛠️ **Config-Driven**: YAML configs for reproducible deployments
- 🌐 **API Server**: FastAPI-based serving with streaming support

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Ayo-Cyber/slm-packager.git
cd slm-packager
pip install -e .
```

### Pull & Run a Model

```bash
# List available models
slm list

# Pull GPT-2 (500MB, fast for testing)
slm pull gpt2

# Run it
slm run gpt2 --prompt "Explain AI in one sentence"
```

That's it! The model downloads, auto-configures, and runs.

### Pull a GGUF Model

```bash
# Pull TinyLlama with llama.cpp (637MB)
slm pull tinyllama

# Run with different parameters
slm run tinyllama --prompt "Write a haiku"
```

## 📦 Available Models

| Model | Size | Runtime | Description |
|-------|------|---------|-------------|
| **gpt2** | 500MB | transformers | OpenAI GPT-2, fast to download |
| **tinyllama** | 637MB | llama.cpp | 1.1B chat model, CPU-optimized |
| **phi-2** | 1.6GB | llama.cpp | Microsoft's 2.7B reasoning model |
| **qwen-1.8b** | 1.1GB | llama.cpp | Alibaba's efficient chat model |

View all: `slm list`  
Pull with specific quantization: `slm pull tinyllama --quant q8_0`

## 🛠️ CLI Commands

```bash
# Model management
slm list                    # Show available models
slm list --installed        # Show downloaded models
slm pull <model>            # Download a model
slm pull <model> --list-variants  # Show quantization options

# Running models
slm run <model> --prompt "Your prompt"
slm run <config.yaml> --prompt "Your prompt"

# Quantization (auto-downloads tool)
slm quantize input.gguf output.gguf --type q4_k_m

# Benchmarking
slm benchmark <model>

# API server
slm serve --port 8000

# Manual config creation
slm init
```

## ⚡ GPU Acceleration

SLM Packager supports GPU acceleration across different hardware platforms.

### Apple Silicon (MPS) - Zero Setup Required! 🍎

**No installation needed** - works out of the box on M1/M2/M3 Macs!

```bash
# Create GPU-accelerated config
slm init --name gpt2 --path gpt2 --format transformers --runtime transformers --device mps -o gpt2-gpu.yaml

# Run on GPU
slm run gpt2-gpu.yaml --prompt "Explain quantum computing"
```

**Real Performance (M2 Pro):**
```
GPT-2 Performance Comparison
├─ CPU:  1.3 tokens/sec
└─ MPS:  2.4 tokens/sec  ⚡ 2.14x faster!

Tested on: M2 Pro, macOS 14.x, GPT-2 (124M parameters)
```

**Requirements:**
- macOS 12.3 or later
- Apple Silicon (M1/M2/M3 series)
- PyTorch 1.12+ (included with installation)

### NVIDIA GPU (CUDA)

```bash
# Install llama.cpp with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir

# Set gpu_layers in config
runtime:
  type: llama_cpp
  device: cuda
  gpu_layers: 32  # Offload layers to GPU
```

**Expected Performance:**
- 2-5x speedup vs CPU
- Depends on GPU, model size, and layers offloaded

### llama.cpp Metal (Apple Silicon)

For GGUF models with llama.cpp on Apple Silicon:

```bash
# Rebuild with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir

# Use GPU layers in config
runtime:
  type: llama_cpp
  device: cpu  # Metal auto-detected
  gpu_layers: 32  # Offload to GPU
```

---

## 📊 Performance Benchmarks

Real-world performance on different hardware:

### GPT-2 (124M parameters)

| Runtime | Device | Tokens/sec | Memory | Notes |
|---------|--------|-----------|--------|-------|
| **transformers** | CPU (M2 Pro) | 1.3 | 2.1GB | Baseline |
| **transformers** | MPS (M2 Pro) | 2.4 | 2.1GB | 2.14x speedup ⚡ |
| **ONNX** | CPU (M2 Pro) | 13.8 | 600MB | With KV-cache |
| **llama.cpp** | CPU | ~15-20 | ~400MB | Quantized GGUF |

### TinyLlama (1.1B parameters)

| Runtime | Device | Tokens/sec | Memory | Notes |
|---------|--------|-----------|--------|-------|
| **llama.cpp** | CPU | 15-20 | ~800MB | Q4_K_M quantization |
| **llama.cpp** | Metal (M1) | 40-60 | ~800MB | With GPU layers |
| **transformers** | CPU | 5-10 | 4GB | Full precision |

*Performance varies based on hardware, model size, and configuration. Benchmarks collected on M2 Pro (Dec 2025).*

---

## 📖 Runtime Comparison

Choose the right runtime for your use case:

| Runtime | Best For | Pros | Cons |
|---------|----------|------|------|
| **llama.cpp** | Production, efficiency | Fast, low memory, quantized | GGUF format only |
| **transformers** | Development, flexibility | Latest models, GPU support | Higher memory |
| **ONNX** | Cross-platform, optimization | Fast, portable, optimized | Requires model export |

### When to Use Each

**llama.cpp (GGUF):**
- ✅ Production deployments
- ✅ Limited memory/CPU
- ✅ Want quantization
- ✅ Edge devices

**transformers (PyTorch):**
- ✅ Development & experimentation  
- ✅ Latest HuggingFace models
- ✅ Fine-tuning workflows
- ✅ GPU available

**ONNX:**
- ✅ Cross-platform deployment
- ✅ ML pipeline integration
- ✅ Optimized inference graphs
- ✅ Already have ONNX models

---

## 🎯 Example Workflows

### Developer: Fine-Tune & Quantize

```bash
# 1. Fine-tune your model (external tool)
# 2. Quantize it
slm quantize my-model.gguf my-model-q4.gguf --type q4_k_m

# 3. Test it
slm run my-model-q4.gguf --prompt "Test prompt"

# 4. Benchmark it
slm benchmark my-model-q4.gguf
```

### Researcher: Compare Runtimes

```bash
# Pull same model, different runtimes
slm pull gpt2              # Transformers
slm pull tinyllama         # llama.cpp

# Benchmark both
slm benchmark gpt2
slm benchmark tinyllama

# Compare results
```

### MacBook User: GPU-Accelerated Inference

```bash
# Zero setup - just run!
slm pull gpt2
slm init --name gpt2 --path gpt2 --device mps -o gpt2-gpu.yaml
slm run gpt2-gpu.yaml --prompt "Hello!"

# 2.14x faster than CPU! ⚡
```

### ONNX User: Export & Run

```bash
# 1. Export model to ONNX
pip install "optimum[exporters]"
optimum-cli export onnx --model gpt2 --task text-generation-with-past models/gpt2-onnx/

# 2. Create config  
slm init --name gpt2 --path models/gpt2-onnx --format onnx --runtime onnx -o gpt2-onnx.yaml

# 3. Run (13.8 tok/s on CPU!)
slm run gpt2-onnx.yaml --prompt "Hello world"
```

---

## 📦 Configuration

### Example Config

```yaml
# my-model.yaml
model:
  name: my-custom-model
  path: /path/to/model.gguf
  format: gguf
  description: "My quantized model"

runtime:
  type: llama_cpp
  device: cpu
  threads: 8
  context_size: 2048
  gpu_layers: 0

params:
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  max_tokens: 512
  stream: true
  stop: ["User:", "\n\n"]
```

### Create Config Interactively

```bash
slm init  # Guided prompts
```

---

## 🌐 API Server

Start a FastAPI server for HTTP access:

```bash
# Start server
slm serve --port 8000

# Or with custom config
slm serve --config my-model.yaml --port 8000
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Load model
curl -X POST http://localhost:8000/load \
  -H "Content-Type: application/json" \
  -d '{"config_path": "gpt2.yaml"}'

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.8
  }'

# Streaming
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"prompt": "Hello", "stream": true}'
```

---

## 📖 Documentation

Comprehensive guides for each component:

- [Quick Start Guide](docs/V01_QUICKSTART.md) - Complete walkthrough
- [Model Formats Guide](docs/MODEL_FORMATS.md) - GGUF vs PyTorch vs ONNX
- [GGUF Setup Guide](docs/GGUF_GUIDE.md) - Using llama.cpp with Metal/CUDA
- [ONNX Guide](docs/ONNX_GUIDE.md) - Export, run, and optimize ONNX models
- [Init Guide](docs/INIT_GUIDE.md) - Creating configs manually
- [Contributing Guide](CONTRIBUTING.md) - Development setup and guidelines

---

## 🧪 Testing & Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=slm_packager --cov-report=html

# Code quality
black slm_packager tests
isort slm_packager tests
mypy slm_packager

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

**Test Results:**
- ✅ 73 tests passing
- Coverage: 52% overall
  - API: 82% ⭐
  - Core runtime: 60%
  - CLI: 47%

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

---

## 🗺️ Roadmap

### v0.2 (Current - December 2025)
- [x] Automated test suite (✅ Complete: 73 tests, 52% coverage)
- [x] MPS GPU support for Apple Silicon (✅ Complete: 2.14x speedup)
- [x] ONNX runtime with KV-cache (✅ Complete: 13.8 tok/s)
- [x] API server improvements (✅ Complete: 82% coverage)
- [ ] CUDA GPU acceleration testing  
- [ ] Comprehensive benchmark suite
- [ ] Expand model registry

### v1.0 (Future)
- [ ] vLLM integration for high-performance GPU serving
- [ ] ROCm support (AMD GPUs)
- [ ] Model conversion utilities
- [ ] Web UI for model management
- [ ] Advanced quantization options
- [ ] Multi-GPU support

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│   CLI / API Server              │
├─────────────────────────────────┤
│   Model Registry & Downloader   │
├─────────────────────────────────┤
│   Runtime Abstraction Layer     │
│   ├─ llama.cpp (GGUF)           │
│   │   └─ Metal/CUDA support     │
│   ├─ Transformers (PyTorch)     │
│   │   └─ MPS/CUDA support       │
│   └─ ONNX Runtime               │
│       └─ Manual KV-cache        │
├─────────────────────────────────┤
│   Quantization & Benchmarking   │
└─────────────────────────────────┘
```

---

## 💡 Why SLM Packager?

**Problem:** Running small language models involves juggling different formats (GGUF, PyTorch, ONNX), runtimes (llama.cpp, transformers, onnxruntime), and configuration options.

**Solution:** SLM Packager provides:
- **Unified interface** - One CLI/API for all runtimes
- **Auto-configuration** - Models work out-of-the-box
- **GPU acceleration** - Automatic MPS on Mac, easy CUDA setup
- **Reproducibility** - YAML configs for deployment
- **Developer-friendly** - Python API, FastAPI server, streaming support

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Testing guidelines
- Code style requirements
- Pull request process

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **llama.cpp** - Efficient GGUF runtime
- **HuggingFace** - Transformers and model hub
- **ONNX Runtime** - Optimized inference
- **FastAPI** - Modern API framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Ayo-Cyber/slm-packager/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ayo-Cyber/slm-packager/discussions)
- **GitHub**: [@Ayo-Cyber](https://github.com/Ayo-Cyber)

---

**Built with ❤️ for the AI community**

*Making small language models accessible, fast, and easy to deploy.*
