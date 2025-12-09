# SLM Packager - Guide

This guide walks you through initializing and running your first Small Language Model using SLM Packager.

## 📋 Understanding `slm init`

The `slm init` command creates a YAML configuration file that defines:
- **Model details**: Name, path, and format
- **Runtime settings**: Which inference engine to use
- **Generation parameters**: Temperature, max tokens, etc.

### Interactive Questions Explained

When you run `slm init`, you'll be asked:

1. **Model Name** 
   - A friendly name for your model (e.g., "phi-2", "tinyllama")
   - Used for identification and logging

2. **Model Path**
   - Absolute or relative path to the model file
   - Can be a local file or HuggingFace model ID
   - Examples: `./models/phi-2.gguf`, `microsoft/phi-2`

3. **Model Format**
   - Choose from: `gguf`, `onnx`, or `pytorch`
   - `gguf` = llama.cpp format (fastest for CPU)
   - `onnx` = ONNX format
   - `pytorch` = HuggingFace/PyTorch models

4. **Runtime**
   - Choose from: `llama_cpp`, `onnx`, or `transformers`
   - Must match your model format:
     - `gguf` → `llama_cpp`
     - `onnx` → `onnx`
     - `pytorch` → `transformers`

5. **Output** (optional)
   - Name of the config file (default: `slm.yaml`)

---

## 🚀 Example: TinyLlama (GGUF Format)

**TinyLlama** is a great starter model - it's small (~600MB), fast, and works well on CPU.

### Step 1: Download the Model

```bash
# Create a models directory
mkdir -p models
cd models

# Download TinyLlama GGUF from HuggingFace
# Using Q4_K_M quantization (good balance of speed/quality)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Go back to project root
cd ..
```

### Step 2: Initialize Configuration

```bash
slm init
```

**Answer the prompts:**
```
Model Name: tinyllama
Model Path: ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
Model Format: gguf
Runtime: llama_cpp
```

This creates `slm.yaml`:

```yaml
model:
  name: tinyllama
  path: ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
  description: null
  format: gguf
runtime:
  type: llama_cpp
  device: cpu
  threads: 4
  gpu_layers: 0
  context_size: 2048
params:
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  max_tokens: 512
  stop: []
  stream: false
quantization: null
```

### Step 3: Run the Model

```bash
slm run slm.yaml --prompt "What is the capital of France?"
```

Or interactively:

```bash
slm run slm.yaml

# You'll be prompted for a prompt
Enter prompt: Tell me a short story about a robot
```

---

## 🎯 Example: Phi-2 (PyTorch Format)

**Phi-2** is a 2.7B parameter model from Microsoft. Great for more complex tasks.

### Using HuggingFace Model (No Download Needed)

```bash
slm init
```

**Answer the prompts:**
```
Model Name: phi-2
Model Path: microsoft/phi-2
Model Format: pytorch
Runtime: transformers
```

This creates a config that loads directly from HuggingFace:

```yaml
model:
  name: phi-2
  path: microsoft/phi-2
  format: pytorch
runtime:
  type: transformers
  device: cpu  # Change to 'cuda' if you have a GPU
  threads: 4
  gpu_layers: 0
  context_size: 2048
params:
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  max_tokens: 512
  stop: []
  stream: false
```

### Run it:

```bash
slm run phi-2.yaml --prompt "Explain quantum computing in simple terms"
```

---

## 🔧 Non-Interactive Init

You can also skip the prompts using command-line options:

```bash
slm init \
  --name tinyllama \
  --path ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --format gguf \
  --runtime llama_cpp \
  --output tinyllama.yaml
```

---

## ⚙️ Customizing Your Config

After initialization, you can manually edit the YAML file to adjust:

### Runtime Settings

```yaml
runtime:
  type: llama_cpp
  device: cuda        # Use GPU (if available)
  threads: 8          # More CPU threads
  gpu_layers: 32      # Offload layers to GPU
  context_size: 4096  # Larger context window
```

### Generation Parameters

```yaml
params:
  temperature: 0.3    # Lower = more focused/deterministic
  top_p: 0.95         # Nucleus sampling
  top_k: 50           # Top-k sampling
  max_tokens: 1024    # Longer responses
  stop: ["</s>", "\n\n"]  # Stop sequences
  stream: true        # Stream output token by token
```

---

## 📊 Testing and Benchmarking

### Benchmark Performance

```bash
slm benchmark slm.yaml
```

Output:
```
Load Time: 2.34s
Generation Time: 1.45s
Memory Usage: 1234.56 MB
Latency: 23.45 ms
Estimated TPS: 45.67
```

### Start API Server

```bash
slm serve --port 8000
```

Then test with curl:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "params": {"temperature": 0.5, "max_tokens": 200}
  }'
```

---

## 🎓 Recommended Models for Beginners

| Model | Size | Format | Best For | Download Link |
|-------|------|--------|----------|---------------|
| **TinyLlama** | 1.1B | GGUF | CPU testing, learning | [HuggingFace](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) |
| **Phi-2** | 2.7B | PyTorch | Reasoning, coding | `microsoft/phi-2` (HF ID) |
| **Qwen-1.8B** | 1.8B | GGUF/PyTorch | Multilingual | [HuggingFace](https://huggingface.co/Qwen/Qwen-1_8B-Chat) |

---

## ❓ Troubleshooting

### "Model file not found"
- Check that the path in your config is correct
- Use absolute paths if relative paths don't work
- For HuggingFace models, ensure you have internet connection

### "Runtime not found" or import errors
- Ensure all dependencies are installed: `pip install -e .`
- For GGUF: `llama-cpp-python` must be compiled correctly
- For GPU: Make sure CUDA/Metal is installed

### Model runs slowly
- Try a GGUF format (faster than PyTorch on CPU)
- Use higher quantization (Q4_K_M or lower)
- Increase `threads` in runtime config
- For GPU: set `device: cuda` and `gpu_layers: 32`

### Out of memory
- Use a smaller model (TinyLlama instead of Phi-2)
- Reduce `context_size` in config
- Use higher quantization (Q4_K_M instead of Q8_0)

---

## 🎯 Next Steps

1. **Try different models**: Experiment with various sizes and formats
2. **Tune parameters**: Adjust temperature, top_p for different outputs
3. **Benchmark**: Compare performance across different configurations
4. **Build an app**: Use the API server to integrate into your application
5. **Contribute**: Check `IMPROVEMENTS.md` for ways to help

Happy experimenting! 🚀
