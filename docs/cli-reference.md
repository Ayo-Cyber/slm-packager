# CLI Reference

All commands are available via the `slm` command after installing `slm-packager`.

---

## `slm list`

List available or installed models.

```bash
slm list                  # show registry models
slm list --installed      # show downloaded models only
```

---

## `slm pull`

Download a model from the registry or directly from HuggingFace.

```bash
# Registry pull
slm pull tinyllama
slm pull phi-2 --quant q8_0
slm pull tinyllama --list-variants   # show available quantizations

# Direct HuggingFace pull (any GGUF or ONNX)
slm pull <repo-id> <filename>
slm pull Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --name qwen3-4b
slm pull TheBloke/Mistral-7B-GGUF mistral-7b-v0.1.Q4_K_M.gguf
```

**Options:**

| Flag | Description |
|------|-------------|
| `--quant` | Quantization variant (e.g. `q4_k_m`, `q8_0`) |
| `--list-variants` | List all quantization options for a registry model |
| `--name` | Local alias when pulling from a HF repo directly |

Models are stored in `~/.slm/models/` and configs in `~/.slm/configs/`.

---

## `slm run`

Run a model from a config file or by installed name.

```bash
slm run tinyllama --prompt "Hello!"
slm run my-model.yaml --prompt "Summarise this document"
slm run gpt2 --no-stream          # disable streaming
slm run tinyllama --raw           # skip chat template formatting
```

**Options:**

| Flag | Description |
|------|-------------|
| `--prompt`, `-p` | Prompt string. If omitted, you will be prompted interactively |
| `--stream/--no-stream` | Enable or disable streaming output (default: stream) |
| `--raw` | Disable automatic chat template formatting |

---

## `slm benchmark`

Measure load time, generation speed, memory usage, and latency.

```bash
slm benchmark tinyllama
slm benchmark my-model.yaml
```

Output:

```
📊 Benchmark Results:
   Load Time: 1.23s
   Generation Time: 8.45s
   Memory Usage: 812.30 MB
   Latency: 8450.00 ms
   Estimated TPS: 12.43
```

---

## `slm serve`

Start a FastAPI HTTP server for model inference.

```bash
slm serve                        # default: 127.0.0.1:8000
slm serve --port 8080
slm serve --host 0.0.0.0 --port 8000   # expose on network
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Host to bind to |
| `--port` | `8000` | Port to listen on |

See the API endpoints at `http://localhost:8000/docs` once the server is running.

---

## `slm init`

Create a YAML config file interactively or via flags.

```bash
slm init                         # guided prompts
slm init --name gpt2 --path gpt2 --format pytorch --runtime transformers --device mps -o gpt2-mps.yaml
```

**Options:**

| Flag | Description |
|------|-------------|
| `--name` | Model name |
| `--path` | Path to model file or HuggingFace repo ID |
| `--format` | `gguf`, `onnx`, or `pytorch` |
| `--runtime` | `llama_cpp`, `onnx`, or `transformers` |
| `--device` | `cpu`, `cuda`, or `mps` |
| `-o`, `--output` | Output config path (default: `slm.yaml`) |

---

## `slm quantize`

Quantize a GGUF or ONNX model file.

```bash
slm quantize model.gguf --type q4_k_m
slm quantize model.gguf output-q4.gguf --type q4_k_m
slm quantize model.onnx --type int8
```

**Supported types:** `q4_k_m`, `q4_0`, `q5_k_m`, `q8_0` (GGUF) · `int8` (ONNX)

---

## `slm rm`

Remove an installed model and its config.

```bash
slm rm tinyllama
slm rm tinyllama --yes    # skip confirmation prompt
```

---

## Config Schema

```yaml
model:
  name: my-model          # display name
  path: /path/to/model    # file path or HF repo ID
  format: gguf            # gguf | onnx | pytorch
  description: "optional"

runtime:
  type: llama_cpp         # llama_cpp | onnx | transformers
  device: cpu             # cpu | cuda | mps
  threads: 4              # CPU thread count (llama_cpp)
  gpu_layers: 0           # layers to offload to GPU
  context_size: 2048      # context window size
  trust_remote_code: false

params:
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  max_tokens: 512
  stream: true
  repetition_penalty: 1.1
  stop: []                # stop sequences
```
