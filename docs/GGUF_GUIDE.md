# GGUF Models - Quick Start Guide

This guide shows you how to use GGUF models with llama.cpp for **fast CPU inference**.

## 🚀 Why GGUF?

GGUF models with llama.cpp are **5-10x faster** on CPU than PyTorch models:

- **TinyLlama PyTorch**: ~7 tokens/sec on CPU, 2.2GB
- **TinyLlama GGUF Q4**: ~35 tokens/sec on CPU, 600MB

That's **5x faster** and **4x smaller!**

---

## 📥 Step-by-Step: Using GGUF Models

### Step 1: Download a GGUF Model

GGUF models are available from **TheBloke** on HuggingFace:

```bash
# Create models directory
mkdir -p models
cd models

# Download TinyLlama GGUF (Q4_K_M quantization - recommended)
# Using wget (macOS: brew install wget)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Or using curl
curl -L -o tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

cd ..
```

**Note:** This downloads ~600MB (much smaller than the 2.2GB PyTorch version!)

---

### Step 2: Initialize Config with `slm init`

```bash
slm init
```

**Answer the prompts for GGUF:**

```
Model Name: tinyllama-gguf
Model Path: ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
Model Format: gguf
Runtime: llama_cpp
```

This creates `slm.yaml`:

```yaml
model:
  name: tinyllama-gguf
  path: ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
  format: gguf
runtime:
  type: llama_cpp
  device: cpu
  threads: 4
params:
  temperature: 0.7
  max_tokens: 256
  stream: true
```

---

### Step 3: Run the Model

```bash
slm run slm.yaml --prompt "Explain quantum computing in simple terms"
```

**You should see:**
- Model loads in <1 second (vs 5-10 seconds for PyTorch)
- **Fast token generation** (~30-50 tokens/sec on decent CPU)
- Low memory usage

---

## 📊 Compare: GGUF vs PyTorch

Let's run a side-by-side comparison:

### GGUF (Fast!)

```bash
# Download the GGUF model (if not done already)
slm init --name tinyllama-gguf \
  --path ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --format gguf \
  --runtime llama_cpp \
  --output gguf-config.yaml

slm benchmark gguf-config.yaml
```

### PyTorch (Slower on CPU)

```bash
slm init --name tinyllama-pytorch \
  --path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --format pytorch \
  --runtime transformers \
  --output pytorch-config.yaml

slm benchmark pytorch-config.yaml
```

**Compare the results!** You should see:
- GGUF: ~30-50 tokens/sec, ~800MB memory
- PyTorch: ~5-10 tokens/sec, ~3GB memory

---

## 🎯 Popular GGUF Models to Try

### TinyLlama (1.1B) - Best for Testing
- **Size**: 600MB (Q4_K_M)
- **Use**: Learning, quick tests
- **Link**: [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)

```bash
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### Phi-2 (2.7B) - Better Quality
- **Size**: 1.6GB (Q4_K_M)
- **Use**: More capable reasoning
- **Link**: [TheBloke/phi-2-GGUF](https://huggingface.co/TheBloke/phi-2-GGUF)

```bash
wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf
```

### Llama-2-7B (7B) - Production Quality
- **Size**: 4GB (Q4_K_M)
- **Use**: Production chatbots
- **Link**: [TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)

```bash
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

---

## ⚙️ Understanding GGUF File Names

When you see: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`

- `tinyllama-1.1b-chat-v1.0` = Model name and version
- `Q4_K_M` = Quantization type
- `.gguf` = File format

### Quantization Types (from smallest to largest):

| Filename | Size | Quality | Speed | Recommended For |
|----------|------|---------|-------|-----------------|
| `.Q4_0.gguf` | Smallest | Good | Fastest | Tight memory constraints |
| `.Q4_K_M.gguf` | Small | Better | Very Fast | **Best balance** ⭐ |
| `.Q5_K_M.gguf` | Medium | Great | Fast | Better quality needed |
| `.Q8_0.gguf` | Large | Excellent | Moderate | Quality-focused |

**Start with Q4_K_M** - it's the sweet spot!

---

## 🖥️ Device-Specific Tips

### Apple Silicon (M1/M2/M3)
GGUF models are **incredibly fast** on Apple Silicon with Metal acceleration:

```yaml
runtime:
  type: llama_cpp
  device: mps  # Use Metal
  gpu_layers: 32  # Offload layers to GPU
```

### Windows/Linux with NVIDIA GPU
```yaml
runtime:
  type: llama_cpp
  device: cuda
  gpu_layers: 32
```

### CPU-Only (Any Platform)
```yaml
runtime:
  type: llama_cpp
  device: cpu
  threads: 8  # Adjust based on your CPU cores
```

---

## 🔧 Troubleshooting

### "llama.cpp not found" or compilation errors

The `llama-cpp-python` package needs to be compiled. Try:

```bash
# Force reinstall with compilation
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir

# For CUDA support on Linux/Windows
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir
```

### Model loads but is slow

- Increase `threads` in your config (e.g., `threads: 8`)
- On Apple Silicon, set `device: mps` and `gpu_layers: 32`
- On NVIDIA GPU, set `device: cuda` and `gpu_layers: 32`

### File not found

- Check the path in your config is correct
- Use absolute paths if relative paths don't work
- Ensure the `.gguf` file was fully downloaded

---

## 📚 Learn More

- [MODEL_FORMATS.md](MODEL_FORMATS.md) - Deep dive into formats and runtimes
- [QUICKSTART.md](QUICKSTART.md) - General getting started guide
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp) - The underlying engine

---

**Ready to see the speed difference?** Download a GGUF model and try it! 🚀
