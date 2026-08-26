# Benchmarks

Performance numbers collected on an Apple M3 Pro with 18GB unified memory, running
macOS. Every figure below comes from:

```bash
slm benchmark <model> --runs 5 --max-tokens 128
```

That is: one discarded warmup run, then 5 timed runs, reporting the **median**
tokens/sec. Output tokens are counted with the model's own tokenizer (transformers)
or `tokenize()` (llama.cpp), never a character estimate.

---

## Results

| Model | Params | Runtime | Device | Tokens/sec (median) | Load Time |
|-------|--------|---------|--------|--------------------|-----------|
| TinyLlama 1.1B Chat Q4_K_M | 1.1B | llama.cpp | CPU | 142.9 | 0.8s |
| GPT-2 | 124M | transformers | CPU | 79.7 | 2.9s |
| GPT-2 | 124M | transformers | MPS | 77.0 | 2.9s |
| Phi-2 Q4_K_M | 2.7B | llama.cpp | CPU | 30.9 | 2.1s |
| Qwen3 4B Q4_K_M | 4B | llama.cpp | CPU | 27.4 | 4.9s |

These are single-machine numbers. Treat them as a starting point, not a leaderboard —
running `slm benchmark` on your own hardware is the point of the command.

---

## Run Your Own Benchmarks

```bash
# Benchmark an installed model
slm benchmark tinyllama

# More samples, longer generations
slm benchmark tinyllama --runs 5 --max-tokens 256

# Benchmark via config file
slm benchmark my-model.yaml
```

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

---

## Notes on the Numbers

**Throughput scales inversely with model size**, as you'd expect for
memory-bandwidth-bound decoding: TinyLlama (0.6GB of weights) runs ~4.6x faster than
Phi-2 (1.8GB) and ~5.2x faster than Qwen3 4B (2.5GB).

**GPT-2 CPU vs MPS is a wash** (79.7 vs 77.0 — within run-to-run noise). At 124M
parameters the per-step GPU dispatch overhead cancels the compute benefit. Expect MPS
to pull ahead on larger PyTorch models; don't expect it to help at this size. Note
also that the transformers runtime uses float32 on MPS and float16 on CUDA, which
caps the MPS ceiling.

**Memory is whole-process RSS** measured after load — it includes the Python
interpreter and the framework, so it is an upper bound rather than model weight size,
and it varies by 20%+ between runs. Use it for rough capacity planning only.

**`slm benchmark` applies the model's chat template** before generating. Without it,
instruction-tuned models answer an unformatted prompt with an immediate
end-of-sequence, and a "throughput" measured over zero generated tokens is
meaningless. (Numbers published before v0.2.3 were affected by exactly this: the
harness counted an empty generation as one token, which understated TinyLlama by
roughly 15x. They have been re-measured.)

---

## Hardware Reference

```
Machine:  Apple MacBook Pro
Chip:     M3 Pro
Memory:   18GB unified
OS:       macOS
Python:   3.12
```

To submit benchmarks from your machine, open an
[issue](https://github.com/Ayo-Cyber/slm-packager/issues) with your results and the
exact command you ran.
