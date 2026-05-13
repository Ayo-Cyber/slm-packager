# Benchmarks

Real-world performance numbers collected on an Apple M3 Pro with 18GB unified memory, running macOS. All benchmarks use the `slm benchmark` command, which runs a single generation pass with streaming disabled for accurate timing.

---

## Results

| Model | Params | Runtime | Device | Tokens/sec | Load Time | Memory |
|-------|--------|---------|--------|-----------|-----------|--------|
| GPT-2 | 124M | transformers | CPU | 53.16 | ~2s | ~600MB |
| GPT-2 | 124M | transformers | MPS ⚡ | 28.06 | ~3s | ~600MB |
| TinyLlama 1.1B | 1.1B | llama.cpp | CPU | 9.19 | ~1s | ~800MB |
| Phi-2 | 2.7B | llama.cpp | CPU | 33.67 | ~3s | ~1.8GB |
| Qwen3 4B | 4B | llama.cpp | CPU | 31.71 | ~2.6s | ~3.3GB |

---

## Run Your Own Benchmarks

```bash
# Benchmark an installed model
slm benchmark tinyllama

# Benchmark via config file
slm benchmark my-model.yaml
```

Output includes:

```
📊 Benchmark Results:
   Load Time: 2.60s
   Generation Time: 16.12s
   Memory Usage: 3270.45 MB
   Latency: 16115.85 ms
   Estimated TPS: 31.71
```

---

## Notes on the Numbers

**GPT-2 CPU vs MPS:** GPT-2 (124M) is small enough that MPS dispatch overhead outweighs the GPU compute benefit. For larger models (1B+), MPS gains become more significant.

**llama.cpp efficiency:** Phi-2 (2.7B) and Qwen3 (4B) both achieve ~32 TPS on CPU via Q4_K_M quantization — comparable throughput despite Qwen3 being 50% larger. llama.cpp's CPU optimizations are very effective for GGUF models.

**Token counting:** TPS is measured on the generated output using the model's own tokenizer (transformers) or `tokenize()` method (llama.cpp), not a character estimate.

---

## Hardware Reference

```
Machine:  Apple MacBook Pro
Chip:     M3 Pro
Memory:   18GB unified
OS:       macOS Sequoia
Python:   3.12
```

To submit benchmarks from your machine, open an [issue](https://github.com/Ayo-Cyber/slm-packager/issues) with your results.
