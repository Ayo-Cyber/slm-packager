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

Measure load time, throughput, and memory use. Runs a discarded warmup pass, then
`--runs` timed generations, and reports the median tokens/sec. The prompt is wrapped
in the model's chat template so instruction-tuned models actually produce output.

```bash
slm benchmark tinyllama
slm benchmark my-model.yaml
slm benchmark tinyllama --runs 5 --max-tokens 256
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` / `-p` | built-in instruction prompt | Prompt to benchmark with |
| `--runs` | `3` | Timed runs to take the median of |
| `--max-tokens` | `128` | Tokens to generate per run |
| `--warmup` / `--no-warmup` | `--warmup` | Discard an initial untimed run |

Output:

```
📊 Benchmark Results:
   Load Time: 0.81s
   Memory Usage (process RSS): 1369.19 MB
   Tokens Generated: 700 over 5 run(s)
   Generation Time (mean): 0.98s
   Tokens/sec (median): 142.92
   Time per Token: 7.00 ms
```

`Memory Usage` is whole-process RSS, including Python and the framework — an upper
bound, not model weight size. If a model generates nothing for the given prompt, the
command fails rather than reporting a throughput measured over zero tokens.

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

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/load` | POST | Load a model: `{"config_path": "~/.slm/configs/model.yaml"}` |
| `/generate` | POST | Generate: `{"prompt": "...", "params": {...}, "raw": false}` |
| `/info` | GET | Active model config |
| `/health` | GET | Liveness check |

The server starts with **no model loaded** — call `/load` first or `/generate`
returns `400 Model not loaded`. Generations are serialized per model, and a
`/load` during an active generation waits for it to finish (concurrent requests
mid-switch get `409`).

`/generate` wraps the prompt in the model's chat template by default, so it returns
what `slm run` would for the same input. Send `"raw": true` to pass the prompt through
untouched.

!!! warning "Binding beyond localhost"
    There is no authentication on these endpoints, and `/load` accepts any
    config path on disk. Only use `--host 0.0.0.0` on a trusted network, behind
    a reverse proxy that handles auth.

See the interactive API docs at `http://localhost:8000/docs` once the server is running.

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

GGUF quantization downloads a `llama-quantize` binary from a pinned llama.cpp release
on first use. That path is **supported on macOS (Intel and Apple Silicon) and x86_64
Linux**; Windows and ARM Linux are not yet supported. Pulling a pre-quantized variant
avoids the whole issue:

```bash
slm pull tinyllama --quant q4_k_m
slm pull phi-2 --list-variants
```

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

**Generation parameters:**

| Key | Meaning |
|-----|---------|
| `temperature` | Sampling temperature. **`0` means greedy decoding** — the highest-probability token every step, so output is deterministic. `top_p`/`top_k` don't apply then. |
| `top_p` | Nucleus sampling threshold (ignored when `temperature: 0`) |
| `top_k` | Top-k sampling cutoff (ignored when `temperature: 0`) |
| `max_tokens` | Maximum tokens to generate |
| `repetition_penalty` | Above `1.0` discourages repeating tokens already produced |
| `stop` | Stop sequences. Generation ends at the first match and the stop text is **not** included in the output, streaming or not. Sequences spanning several tokens are handled. |
| `stream` | Emit tokens as they are produced |

All of these are honored by the llama.cpp and transformers runtimes. The
experimental ONNX runtime honors them too but is otherwise less exercised — see
[Runtimes](runtimes.md).
