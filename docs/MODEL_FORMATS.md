# Understanding Model Formats & Runtimes

A beginner's guide to choosing the right model format and runtime for your use case.

## 🎯 Quick Decision Guide

**Choose based on your priority:**

| Priority | Best Choice | Runtime | Format |
|----------|------------|---------|--------|
| **Speed on CPU** | llama.cpp | `llama_cpp` | GGUF |
| **Flexibility** | Transformers | `transformers` | PyTorch |
| **Small size** | llama.cpp | `llama_cpp` | GGUF (quantized) |
| **GPU inference** | Transformers or llama.cpp | `transformers` or `llama_cpp` | PyTorch or GGUF |
| **Just testing** | Transformers | `transformers` | PyTorch |
| **Production CPU** | llama.cpp | `llama_cpp` | GGUF |

---

## 📦 Model Formats Explained

### GGUF (GPT-Generated Unified Format)

**What is it?**
- Binary format optimized for llama.cpp
- Supports aggressive quantization (4-bit, 5-bit, 8-bit)
- Single-file format with model + metadata

**Pros:**
- ⚡ **5-10x faster** on CPU than PyTorch
- 💾 **Much smaller** files (4-bit = ~4x smaller)
- 🚀 **Low memory usage** 
- 🔧 **CPU-optimized** with SIMD instructions

**Cons:**
- ⚠️ Limited to models supported by llama.cpp
- ⚠️ Need to download pre-converted GGUF files
- ⚠️ Quantization can slightly reduce quality

**Best For:**
- Production deployments on CPU
- Edge devices / laptops
- Memory-constrained environments
- Fast inference without GPU

**Example Models:**
- TinyLlama Q4_K_M (600MB vs 2.2GB)
- Phi-2 Q4_K_M (1.6GB vs 5GB)
- Llama-2-7B Q4_K_M (4GB vs 13GB)

---

### PyTorch (.bin, .safetensors)

**What is it?**
- Native PyTorch model format
- Full-precision or half-precision (FP32/FP16)
- Standard format on HuggingFace

**Pros:**
- ✅ **Most compatible** - works with all transformers models
- ✅ **Easy to use** - just provide HuggingFace model ID
- ✅ **Full ecosystem** - training, fine-tuning, etc.
- ✅ **No conversion** needed

**Cons:**
- 🐌 **Slower on CPU** than GGUF
- 💾 **Larger file sizes**
- 🧠 **Higher memory usage**
- ⚡ Requires GPU for good speed

**Best For:**
- Development and experimentation
- GPU-based deployments
- Fine-tuning models
- When you need full model flexibility

**Example Models:**
- GPT-2 (500MB)
- TinyLlama (2.2GB)
- Phi-2 (5GB)

---

### ONNX (Open Neural Network Exchange)

**What is it?**
- Cross-platform model format
- Optimized for inference
- Supports many frameworks

**Pros:**
- 🔄 **Cross-platform** (works everywhere)
- ⚡ **Optimized inference**
- 🛠️ **Framework agnostic**

**Cons:**
- ⚠️ **Current SLM Packager ONNX support is basic**
- ⚠️ Conversion can be tricky
- ⚠️ Not all models convert well

**Best For:**
- Production systems requiring cross-platform support
- When you need framework independence
- (Note: Currently experimental in SLM Packager)

---

## ⚙️ Runtime Comparison

### llama.cpp Runtime

**Speed:** ⭐⭐⭐⭐⭐ (Fastest on CPU)  
**Memory:** ⭐⭐⭐⭐⭐ (Most efficient)  
**Compatibility:** ⭐⭐⭐ (Growing but limited)

**When to use:**
- Running on CPU (especially Apple Silicon M1/M2/M3)
- Need fast inference without GPU
- Memory-constrained environments
- Production deployments

**Requires:** GGUF model files

---

### Transformers Runtime

**Speed:** ⭐⭐ (Slow on CPU, fast on GPU)  
**Memory:** ⭐⭐ (High usage)  
**Compatibility:** ⭐⭐⭐⭐⭐ (Works with everything)

**When to use:**
- Have a GPU available
- Need maximum compatibility
- Want easy access to HuggingFace models
- Development and testing

**Requires:** PyTorch model files or HuggingFace ID

---

### ONNX Runtime

**Speed:** ⭐⭐⭐ (Good)  
**Memory:** ⭐⭐⭐ (Moderate)  
**Compatibility:** ⭐⭐⭐ (Platform independent)

**When to use:**
- Need cross-platform deployment
- Want optimized inference
- (Note: Limited support in current version)

**Requires:** ONNX model files

---

## 📊 Performance Comparison (Example: TinyLlama)

| Format | Runtime | Size | CPU Speed | Memory | GPU Speed |
|--------|---------|------|-----------|--------|-----------|
| GGUF Q4_K_M | llama.cpp | 600MB | **35 tok/s** ⚡ | 800MB | 25 tok/s |
| GGUF Q8_0 | llama.cpp | 1.1GB | 30 tok/s | 1.2GB | 28 tok/s |
| PyTorch FP16 | transformers | 2.2GB | 7 tok/s 🐌 | 3GB | **150 tok/s** ⚡ |

*On M1 MacBook Pro. tok/s = tokens per second*

**Key Takeaway:** GGUF with llama.cpp is **5x faster** on CPU but transformers is **5x faster** on GPU.

---

## 🎓 Quantization Types (GGUF)

When you see model names like `Q4_K_M`, here's what it means:

### Common Quantization Types

| Type | Size | Quality | Speed | Use Case |
|------|------|---------|-------|----------|
| **Q4_0** | Smallest | Good | Fastest | Maximum speed, tight memory |
| **Q4_K_M** | Small | Better | Very fast | **Recommended balance** |
| **Q5_K_M** | Medium | Great | Fast | Better quality, still efficient |
| **Q8_0** | Large | Excellent | Moderate | Quality over size |
| **F16** | Largest | Perfect | Slower | When quality is critical |

**Recommendation:** Start with **Q4_K_M** - it's the sweet spot of speed, size, and quality.

---

## 🔍 Where to Find Models

### GGUF Models
- **TheBloke on HuggingFace** - Most popular GGUF converter
  - Example: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
  - Download individual `.gguf` files
  
### PyTorch Models
- **HuggingFace Model Hub**
  - Use model ID directly: `microsoft/phi-2`
  - Or download and use local path
  
### ONNX Models
- **HuggingFace** or **ONNX Model Zoo**
- Convert PyTorch models using `optimum` library

---

## 💡 Practical Recommendations

### For Learning & Testing
→ Use **PyTorch + Transformers** with small models (GPT-2, DistilGPT-2)

### For Production on CPU
→ Use **GGUF Q4_K_M + llama.cpp** for speed and efficiency

### For Production on GPU
→ Use **PyTorch + Transformers** (or vLLM for advanced use)

### For Maximum Speed on Apple Silicon (M1/M2/M3)
→ Use **GGUF + llama.cpp** with Metal acceleration

### For Edge Devices / Limited Memory
→ Use **GGUF Q4_0 + llama.cpp** for smallest footprint

---

## 🚀 Next Steps

1. Read [QUICKSTART.md](QUICKSTART.md) for setup instructions
2. Try both runtimes to see the difference
3. Benchmark your specific use case with `slm benchmark`
4. Choose the format that best fits your needs

**Questions?** Check the main [README.md](../README.md) or open an issue!
