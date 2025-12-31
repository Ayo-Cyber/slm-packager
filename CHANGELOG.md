# Changelog

All notable changes to SLM Packager will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-24

### Added

#### GPU Acceleration 🚀
- **MPS support for Apple Silicon** - Zero-setup GPU acceleration on M1/M2/M3 Macs
  - 2.14x speedup on M2 Pro (GPT-2: 1.3 → 2.4 tokens/sec)
  - Automatic device detection and tensor placement
  - Works with any PyTorch/transformers model
  - GPU cache management (MPS/CUDA)
- **Comprehensive GPU documentation** - New `docs/GPU_ACCELERATION.md` guide
  - Setup instructions for MPS, CUDA, and Metal
  - Performance benchmarks across platforms
  - Troubleshooting and optimization tips

#### ONNX Runtime Improvements 🔧
- **Complete ONNX runtime rewrite** - Now production-ready!
  - Manual KV-cache management for efficient generation
  - Works with ANY model exported via optimum
  - 13.8 tokens/sec on CPU (GPT-2)
  - Support for position_ids and dynamic KV-cache tensors
  - Token sampling with temperature, top-k, top-p
  - Streaming generation support
- **ONNX documentation** - Updated `docs/ONNX_GUIDE.md`
  - Model export instructions with optimum
  - Configuration examples
  - Performance comparison tables
  - Troubleshooting guide

#### Testing & Quality 🧪
- **API server improvements**
  - Enhanced error handling with specific exception types
  - Better error messages for debugging
  - Streaming functionality tests
  - Coverage: 79% → 82%
- **Test suite expansion**
  - 73 total tests passing
  - Overall coverage: 52%
  - Integration tests for streaming
  - Error path coverage

### Changed

- **README overhaul** - Comprehensive rewrite with:
  - Real performance benchmarks (M2 Pro, CUDA)
  - GPU acceleration section with examples
  - Runtime comparison table
  - Expanded example workflows
  - Better quick start guide
- **Documentation structure** - Organized guides by topic:
  - `GPU_ACCELERATION.md` - All GPU setup in one place
  - `GGUF_GUIDE.md` - Metal/CUDA instructions included
  - `ONNX_GUIDE.md` - Export and optimization guide

### Fixed

- **ONNX runtime** - Replaced non-functional onnxruntime-genai with standard onnxruntime
  - Fixed incompatibility with optimum ONNX exports
  - Proper KV-cache initialization and management
  - position_ids handling for GPT-2 and similar models
- **Transformers runtime** - Improved device handling
  - Fixed MPS tensor placement
  - Better error messages for device availability
  - Proper GPU cache clearing on unload

### Performance

Real-world benchmarks (December 2024):

**GPT-2 (124M parameters):**
- transformers + CPU: 1.3 tok/s
- transformers + MPS (M2 Pro): 2.4 tok/s (2.14x faster)
- ONNX + CPU: 13.8 tok/s
- llama.cpp GGUF + CPU: 15-20 tok/s

**TinyLlama (1.1B parameters):**
- llama.cpp + Metal (M1): 40-60 tok/s
- transformers + MPS (M2 Pro): 28 tok/s

## [0.1.0] - 2024-11-20

### Added

- Initial release
- Multi-runtime support (llama.cpp, transformers, ONNX placeholder)
- Model registry with HuggingFace integration
- CLI interface (`slm` command)
- FastAPI server with streaming
- Auto-quantization with llama.cpp tools
- Configuration system (YAML)
- Benchmarking utilities

### Runtime Support

- **llama.cpp** - GGUF models with CPU/GPU layers
- **transformers** - PyTorch models with basic CUDA support
- **ONNX** - Placeholder implementation

### Documentation

- Quick start guide
- Model formats guide
- GGUF setup guide
- Contributing guidelines

---

## Upgrade Guide

### From 0.1.0 to 0.2.0

**ONNX Runtime:** If you were using ONNX runtime in 0.1.0:
- Previous: Required onnxruntime-genai (didn't work)
- Now: Uses standard onnxruntime (works!)
- Action: No changes needed, just works better

**GPU Acceleration:** New features available:
- Mac users: Add `device: mps` to configs for 2x speedup
- NVIDIA users: Documentation now covers all CUDA setup
- No breaking changes to existing configs

**API:** Fully backward compatible
- All existing configs work unchanged
- New features are opt-in (device settings)

---

## Future Roadmap

See [README.md](README.md) for planned features in v1.0:
- vLLM integration
- ROCm support (AMD GPUs)
- Web UI
- Multi-GPU support
- Enhanced quantization

---

**Questions?** Open an issue on [GitHub](https://github.com/Ayo-Cyber/slm-packager/issues)
