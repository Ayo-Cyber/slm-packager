# SLM Packager

**A Unified Runtime & Developer Layer for Small Language Models**

SLM Packager is an open-source toolkit for running, packaging, and evaluating Small Language Models (1B-7B parameters) across different formats and runtimes. Think of it as **Terraform for SLMs** — making model deployment simple, reproducible, and developer-friendly.

## ✨ Features

- 🎯 **Model Registry**: One-command downloads from HuggingFace with `slm pull`
- 🔄 **Multi-Runtime Support**: llama.cpp (GGUF), Transformers (PyTorch), ONNX
- ⚙️ **Auto-Quantization**: On-device model quantization with automatic tool setup
- 📊 **Benchmarking**: Measure speed, memory, latency across runtimes
- 🛠️ **Config-Driven**: YAML configs for reproducible deployments
- 🌐 **API Server**: FastAPI-based serving with streaming support

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/slm-packager.git
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

### Fast User: Just Run Models

```bash
slm pull tinyllama
slm run tinyllama --prompt "Hello!"
```

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│   CLI / API Server              │
├─────────────────────────────────┤
│   Model Registry & Downloader   │
├─────────────────────────────────┤
│   Runtime Abstraction Layer     │
│   ├─ llama.cpp (GGUF)           │
│   ├─ Transformers (PyTorch)     │
│   └─ ONNX Runtime               │
├─────────────────────────────────┤
│   Quantization & Benchmarking   │
└─────────────────────────────────┘
```

## 📖 Documentation

- [Quick Start Guide](docs/V01_QUICKSTART.md) - Complete walkthrough
- [Model Formats Guide](docs/MODEL_FORMATS.md) - GGUF vs PyTorch vs ONNX
- [GGUF Setup Guide](docs/GGUF_GUIDE.md) - Using llama.cpp
- [Init Guide](docs/INIT_GUIDE.md) - Creating configs manually

## 🔧 Advanced Usage

### Custom Config

```yaml
# my-model.yaml
model:
  name: my-custom-model
  path: /path/to/model.gguf
  format: gguf

runtime:
  type: llama_cpp
  device: cpu
  threads: 4

params:
  temperature: 0.7
  max_tokens: 512
  stream: true
```

```bash
slm run my-model.yaml --prompt "Test"
```

### API Server

```bash
# Start server
slm serve --port 8000

# Use API
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

## 🎯 Why SLM Packager?

**vs. Ollama:**
- ✅ Multi-runtime support (not locked to one backend)
- ✅ On-device quantization
- ✅ Config-driven (reproducible)
- ✅ Developer-focused (not consumer tool)

**vs. HuggingFace Transformers:**
- ✅ Supports GGUF (faster CPU inference)
- ✅ Unified API across formats
- ✅ Built-in benchmarking

**vs. llama.cpp:**
- ✅ Auto-downloads models
- ✅ No manual file management
- ✅ Config-based workflows

## 🚧 Known Limitations (v0.1)

- **ONNX Runtime**: Simplified generation without KV-cache (slow, experimental)
- **Apple Silicon**: Untested on MPS devices
- **Tests**: No automated test suite yet (manual testing only)

See [issues](https://github.com/YOUR_USERNAME/slm-packager/issues) for details.

## 🗺️ Roadmap

### v0.2 (Next)
- [ ] Automated test suite
- [ ] Fix ONNX runtime with proper KV-cache
- [ ] MPS/ROCm testing
- [ ] Expand model registry

### v1.0 (Future)
- [ ] vLLM integration (GPU serving)
- [ ] Advanced evaluation (perplexity, lm-eval-harness)
- [ ] Plugin system for custom runtimes
- [ ] Docker support
- [ ] Community model registry

## 🤝 Contributing

We welcome contributions! Whether it's:
- Adding models to the registry
- Fixing bugs
- Improving documentation
- Testing on different hardware

### Development Setup

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/slm-packager.git
cd slm-packager

# Install in dev mode
pip install -e ".[dev]"

# Make changes
# Run tests (when available)
# Submit PR
```

### Coding Guidelines
- Python 3.9+
- Follow PEP 8
- Use Pydantic for validation
- Add docstrings to new code
- Emoji-prefixed error messages (❌ errors, 💡 solutions)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built on top of:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF runtime
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - PyTorch models
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - ONNX inference
- [TheBloke](https://huggingface.co/TheBloke) - Pre-quantized models

---

**Made for developers who want Ollama-level UX with llama.cpp-level control.** 🚀

*Star ⭐ this repo if you find it useful!*
