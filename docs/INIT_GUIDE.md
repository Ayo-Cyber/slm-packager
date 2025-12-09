# SLM Init Walkthrough - Interactive Guide

This guide shows you **exactly** what to input during `slm init` for different scenarios.

## 🎯 Scenario 1: GGUF Model (Recommended for CPU Speed)

**Best for:** Fast CPU inference, production deployments, low memory

### What You'll Answer:

```bash
$ slm init

Model Name: tinyllama-fast
Model Path: ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
Model Format (gguf, onnx, pytorch): gguf
Runtime (llama_cpp, onnx, transformers): llama_cpp

Config saved to slm.yaml
```

### Explanation:

- **Model Name**: `tinyllama-fast` - Any friendly name you want
- **Model Path**: Path to your downloaded `.gguf` file
  - Can be relative: `./models/model.gguf`
  - Or absolute: `/Users/you/models/model.gguf`
- **Model Format**: `gguf` - The file format
- **Runtime**: `llama_cpp` - The engine to run GGUF files

### Before Running:

Make sure you've downloaded the GGUF file:
```bash
mkdir -p models
cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
cd ..
```

---

## 🎯 Scenario 2: HuggingFace Model (Easiest but Slower on CPU)

**Best for:** Quick testing, GPU deployments, development

### What You'll Answer:

```bash
$ slm init

Model Name: gpt2-test
Model Path: gpt2
Model Format (gguf, onnx, pytorch): pytorch
Runtime (llama_cpp, onnx, transformers): transformers

Config saved to slm.yaml
```

### Explanation:

- **Model Name**: `gpt2-test` - Your choice of name
- **Model Path**: `gpt2` - **HuggingFace model ID** (no download needed!)
  - Format: `namespace/repo-name` or just `model-name`
  - Examples: `gpt2`, `microsoft/phi-2`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Model Format**: `pytorch` - Standard HuggingFace format
- **Runtime**: `transformers` - Uses the transformers library

### What Happens:

Model auto-downloads from HuggingFace on first run (cached for future use)

---

## 🎯 Scenario 3: Local PyTorch Model

**Best for:** Using a model you've already downloaded or fine-tuned

### What You'll Answer:

```bash
$ slm init

Model Name: my-local-model
Model Path: /Users/you/models/my-model
Model Format (gguf, onnx, pytorch): pytorch
Runtime (llama_cpp, onnx, transformers): transformers

Config saved to slm.yaml
```

### Explanation:

- **Model Path**: Full path to directory containing model files
  - Must have: `config.json`, `pytorch_model.bin` (or `.safetensors`), tokenizer files

---

## 📋 Quick Reference Table

| Want | Model Format | Runtime | Model Path Example |
|------|--------------|---------|-------------------|
| **Fast CPU** | `gguf` | `llama_cpp` | `./models/model.Q4_K_M.gguf` |
| **HuggingFace** | `pytorch` | `transformers` | `gpt2` or `microsoft/phi-2` |
| **Local file** | `pytorch` | `transformers` | `/path/to/model/directory` |
| **ONNX** | `onnx` | `onnx` | `./models/model.onnx` |

---

## 🔍 Format & Runtime Must Match!

**Critical Rule:** Format and Runtime must be compatible:

✅ **Correct Combinations:**
- `gguf` + `llama_cpp`
- `pytorch` + `transformers`
- `onnx` + `onnx`

❌ **Wrong Combinations:**
- `gguf` + `transformers` ❌ Won't work!
- `pytorch` + `llama_cpp` ❌ Won't work!

---

## 🎓 Step-by-Step: Your First GGUF Model

Let's walk through setting up TinyLlama GGUF from scratch:

### Step 1: Download the Model

```bash
# Create directory
mkdir -p models

# Download GGUF file (600MB)
curl -L -o models/tinyllama.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

### Step 2: Initialize Config

```bash
slm init
```

**Type these answers:**
```
Model Name: tinyllama-gguf
Model Path: ./models/tinyllama.gguf
Model Format: gguf
Runtime: llama_cpp
```

### Step 3: Run It!

```bash
slm run slm.yaml --prompt "What is machine learning?"
```

**Result:** Fast inference at ~35 tokens/sec on CPU! 🚀

---

## 💡 Tips & Tricks

### Tip 1: Use Relative Paths

```bash
# Instead of:
Model Path: /Users/johndoe/Documents/models/model.gguf

# Use:
Model Path: ./models/model.gguf
```

Makes configs portable across machines!

### Tip 2: Non-Interactive Mode

Skip the prompts:

```bash
slm init \
  --name tinyllama-gguf \
  --path ./models/tinyllama.gguf \
  --format gguf \
  --runtime llama_cpp \
  --output tinyllama.yaml
```

### Tip 3: Multiple Configs

Create different configs for different models:

```bash
slm init --output gpt2.yaml        # GPT-2 config
slm init --output tinyllama.yaml   # TinyLlama config
slm init --output phi2.yaml        # Phi-2 config

# Then run any:
slm run gpt2.yaml
slm run tinyllama.yaml
```

---

## ❓ Common Questions

**Q: Where do I get GGUF models?**
A: TheBloke on HuggingFace - search for "[model name] GGUF"

**Q: What's the difference between Q4_K_M and Q8_0?**
A: Q4 is smaller/faster, Q8 is larger/better quality. Start with Q4_K_M.

**Q: Can I use the same config on different computers?**
A: Yes! Use relative paths (e.g., `./models/...`) and commit the config to git.

**Q: Do I need to download HuggingFace models first?**
A: No! Just use the model ID (e.g., `gpt2`) and it auto-downloads.

**Q: Which is faster: GGUF or PyTorch?**
A: GGUF is ~5-10x faster on CPU. PyTorch is faster on GPU.

---

## 📚 Next Steps

1. ✅ Create your first config with `slm init`
2. 📖 Read [GGUF_GUIDE.md](GGUF_GUIDE.md) for detailed GGUF setup
3. 📖 Read [MODEL_FORMATS.md](MODEL_FORMATS.md) to understand the differences
4. 🚀 Run `slm benchmark` to compare different configs

Happy experimenting! 🎯
