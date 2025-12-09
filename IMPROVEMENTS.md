# SLM Packager - Improvement Plan

This document tracks known issues, limitations, and planned improvements for the SLM Packager project.

## ✅ Bug Fixes Applied

### December 8, 2025

#### Missing Dependency - psutil
- **Status**: ✅ Fixed
- **Issue**: `psutil` package was used by benchmarker module but not listed in `pyproject.toml` dependencies
- **Impact**: CLI would crash on import with `ModuleNotFoundError: No module named 'psutil'`
- **Fix**: Added `psutil>=5.9.0` to project dependencies in `pyproject.toml`

#### Missing Dependency - accelerate
- **Status**: ✅ Fixed
- **Issue**: `accelerate` package required by transformers runtime but not in dependencies
- **Impact**: Model loading failed with `ValueError: Using a device_map requires accelerate`
- **Fix**: Added `accelerate>=0.25.0` to project dependencies

#### Poor Error Handling
- **Status**: ✅ Improved
- **Issue**: Cryptic error messages that didn't help users troubleshoot issues
- **Impact**: Users couldn't understand what went wrong or how to fix it
- **Fix**: Rewrote transformers runtime with comprehensive error handling:
  - Clear emoji-prefixed messages (❌ for errors, 💡 for solutions)
  - Specific error cases for missing auth, wrong model IDs, OOM, etc.
  - Actionable suggestions for each failure scenario
  - Better progress messages during model loading

#### Incorrect Documentation Examples
- **Status**: ✅ Fixed
- **Issue**: Documentation used invalid HuggingFace model ID format (`microsoft/TinyLlama/...` instead of `TinyLlama/...`)
- **Impact**: Users following docs would get validation errors
- **Fix**: Corrected all model ID references to proper `namespace/repo-name` format

## 🐛 Known Issues & Limitations

### Runtime Issues

#### 1. ONNX Runtime - Simplified Generation Loop
- **Status**: Known Limitation
- **Issue**: Current ONNX adapter uses a simplified generation loop without proper KV-cache implementation
- **Impact**: ONNX model inference is significantly slower than it should be
- **Priority**: High
- **Next Steps**: Implement proper KV-cache mechanism for ONNX runtime

#### 2. Tokenizer Auto-Detection
- **Status**: Known Limitation  
- **Issue**: Basic tokenizer handling, especially for ONNX models
- **Impact**: May require manual tokenizer configuration in some cases
- **Priority**: Medium
- **Next Steps**: Improve auto-detection logic and add fallback strategies

### Quantization

#### 3. External Binary Dependency
- **Status**: Known Limitation
- **Issue**: Quantization wraps external tools (`quantize` binary for llama.cpp)
- **Impact**: Requires external dependencies to be installed separately
- **Priority**: Medium
- **Next Steps**: Implement native Python quantization or bundle binaries

### Testing & Quality

#### 4. Missing Test Suite
- **Status**: Not Implemented
- **Issue**: `/tests` directory exists but no tests implemented yet
- **Impact**: No automated testing, quality assurance relies on manual testing
- **Priority**: High
- **Next Steps**: 
  - Create pytest test suite
  - Add unit tests for each runtime adapter
  - Add integration tests for CLI commands
  - Add API endpoint tests

#### 5. Error Handling
- **Status**: Needs Improvement
- **Issue**: Error handling could be more robust throughout the codebase
- **Impact**: Less informative error messages, potential crashes
- **Priority**: Medium
- **Next Steps**: 
  - Add try-catch blocks with meaningful error messages
  - Implement proper error logging
  - Add validation at boundaries

#### 6. Documentation Gaps
- **Status**: Needs Improvement
- **Issue**: Minimal docstrings in some areas of the codebase
- **Impact**: Harder for contributors to understand code
- **Priority**: Low
- **Next Steps**: 
  - Add comprehensive docstrings
  - Create API documentation
  - Add usage examples

### Platform Support

#### 7. Limited Hardware Testing
- **Status**: Known Limitation
- **Issue**: While CUDA is supported in code, extensive testing across MPS and ROCm is pending
- **Impact**: Unknown stability on Apple Silicon (MPS) and AMD GPUs (ROCm)
- **Priority**: Medium
- **Next Steps**: 
  - Test on MPS devices
  - Test on ROCm devices
  - Add device-specific optimizations

### Architecture

#### 8. Empty Core Module
- **Status**: Placeholder
- **Issue**: `core/` directory is empty
- **Impact**: None currently - appears to be a placeholder
- **Priority**: Low
- **Next Steps**: Define purpose or remove directory

## 🗺️ Roadmap Features

### High Priority

- **vLLM Integration**: Add high-performance GPU serving adapter using vLLM
- **Model Registry**: Implement `slm pull` command to auto-download models from HuggingFace
- **Test Suite**: Comprehensive pytest test coverage

### Medium Priority

- **Advanced Evaluation**: 
  - Perplexity scoring
  - Integration with `lm-evaluation-harness`
  - More benchmarking metrics
- **Plugin System**: Allow users to write custom runtime adapters as Python plugins
- **Docker Support**: Auto-generate Dockerfiles for packaged models

### Low Priority

- **Web UI**: Optional web interface for model management
- **Model Conversion Tools**: Built-in tools to convert between formats
- **Batch Processing**: Support for batch inference

## 📋 Quick Wins (Easy Improvements)

1. Add more comprehensive docstrings
2. Improve error messages with context
3. Add input validation at API boundaries
4. Create example configurations for popular models
5. Add logging throughout the application
6. Create a CONTRIBUTING.md guide
7. Add pre-commit hooks for code quality

## 🔄 Status Legend

- **Not Implemented**: Feature/fix not started
- **Known Limitation**: Acknowledged issue, workaround exists
- **Needs Improvement**: Partially implemented but needs work
- **Placeholder**: Structure in place, no implementation

---

**Last Updated**: 2025-12-08  
**Version**: 0.1.0
