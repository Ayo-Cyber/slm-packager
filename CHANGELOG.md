# Changelog

All notable changes to SLM Packager will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-22

### Changed — install now selects an engine (action required)

**Inference engines are optional extras.** `pip install slm-packager` previously
pulled PyTorch *and* compiled `llama-cpp-python` from source, so a first install took
5–15 minutes, needed several GB, and **failed outright** on any machine without cmake
and a C/C++ toolchain — the most common "it doesn't work" report. Core is now lean and
pure-Python: **~50MB, a few seconds, no compiler.**

```bash
pip install "slm-packager[gguf]"    # GGUF via llama.cpp
pip install "slm-packager[torch]"   # PyTorch/HuggingFace via transformers
pip install "slm-packager[onnx]"    # ONNX Runtime (experimental)
pip install "slm-packager[all]"     # everything
```

- **Upgrading:** a bare `pip install slm-packager` no longer installs an engine. Add
  the extra for the models you run. Nothing else about the CLI or configs changed.
- Core still covers `slm list`, `slm pull`, `slm init`, `slm rm`, and `slm serve`;
  only generating needs an engine, and attempting it without one now names the exact
  extra to install rather than failing on a raw `ModuleNotFoundError`.
- Runtime modules are imported **lazily**, so no engine is loaded until it is used.
  The codebase already had careful `ImportError` fallbacks throughout; eager imports
  and mandatory dependencies made them unreachable. They are now the real path, and a
  test asserts that importing `slm_packager.runtime` pulls in none of torch,
  `llama_cpp`, or `onnxruntime`.
- Dependency floors raised for security: `torch>=2.6` (the `torch.load`
  `weights_only` default) and `transformers>=4.48` (deserialization fixes).
- CI restructured to match: the version matrix installs test tooling only and runs in
  seconds, a dedicated job installs the engines, and a `lean-install` job asserts no
  engine leaks back into core and that a core-only install still works end to end.
  The macOS leg no longer compiles llama-cpp-python and is now a blocking check
  instead of advisory.

### Added
- **Chat templates are applied automatically for every runtime.** `BaseRuntime` gained
  `apply_chat_template()`; llama.cpp renders the template embedded in the GGUF metadata
  and transformers uses the tokenizer's. Previously only transformers models were
  formatted, so chat-tuned GGUF models received raw prompts and frequently replied with
  nothing at all. Pass `--raw` to `slm run` to opt out.
- `slm benchmark` gained `--prompt`, `--runs`, `--max-tokens`, and `--warmup/--no-warmup`.
- `scripts/check_registry.py` plus a weekly CI job that verifies every registry model
  still resolves on HuggingFace.

### Changed
- **`slm benchmark` is now statistically meaningful.** It performs a discarded warmup
  run, then medians over N timed runs with `max_tokens` pinned, and benchmarks against
  a copy of the params instead of mutating the caller's config. It reports
  `tokens_generated`, `ms_per_token`, and run count, and labels memory as process RSS.
- **Replaced registry entry `qwen-1.8b` with `qwen2.5-1.5b`.** Qwen retired the Qwen1
  GGUF repos, so `slm pull qwen-1.8b` failed with an HTTP 401. All 17 registry variants
  are now verified reachable.
- Republished all benchmark numbers in the README and docs (see Documentation below),
  and removed the unverifiable, mutually contradictory tables from the format and GPU
  guides in favour of one measured source.
- Dropped the deprecated `resume_download` argument, which `huggingface-hub` ignores
  and warns about on every download.

### Fixed — generation parameters are no longer silently ignored
- **`temperature: 0` now means greedy decoding.** The config schema accepts it, but
  the transformers runtime hardcoded `do_sample=True` and crashed with "`temperature`
  (=0.0) has to be a strictly positive float", while the ONNX runtime sampled at
  temperature 1.0 instead. Both now take the argmax, so output is deterministic, and
  `top_p`/`top_k` are omitted where they don't apply.
- **`stop` sequences work in the transformers runtime.** They were dropped entirely.
  Now forwarded as native `stop_strings` (which matches across token boundaries), and
  the stop text is trimmed from the result — engines halt *after* emitting it. The
  streaming path withholds partial matches so a stop sequence split across chunks is
  never shown to the caller.
- **`repetition_penalty` works in the ONNX runtime.** It was accepted and ignored.
  Penalties now apply to already-generated tokens before temperature scaling, dividing
  positive scores and multiplying negative ones so the penalty always pushes a token
  down.
- **ONNX stop sequences are matched against the decoded text**, not each token in
  isolation — a stop string spanning two tokens was previously never detected.
- The transformers streamer now sets `skip_special_tokens`, so special tokens no
  longer leak into streamed output.
- **`POST /generate` now applies the chat template, matching `slm run`.** The CLI
  formatted prompts and the API did not, so the same model and prompt produced
  different answers depending on the entry point — for a chat-tuned model the API
  continued the text instead of answering it. Send `"raw": true` to opt out.

### Fixed
- **`slm benchmark` no longer reports throughput for generations that produced no
  tokens.** `Llama.tokenize(b"")` returns a BOS token, so an empty generation counted
  as one token and yielded a number measuring only call overhead — this understated
  TinyLlama by roughly 15x (published 9.19 tok/s; actual 142.9). Empty output now
  raises with guidance instead.
- `token_count_method` reported `estimate` for llama.cpp even though it tokenizes
  exactly via the model, so the CLI showed a spurious accuracy warning.
- `slm rm` now deletes HuggingFace-cached models through the hub's refcounted
  `delete_revisions` API. The previous code called `scan_cache_info`, which does not
  exist in `huggingface-hub` — so cleanup silently did nothing — and a manual blob
  deletion could have removed a file shared with another model, or followed a
  user-managed symlink out to an unrelated file.
- Corrected the stale llama.cpp build flags in the runtime's own ImportError message
  (`-DLLAMA_METAL`/`-DLLAMA_CUBLAS` → `-DGGML_METAL`/`-DGGML_CUDA`).
- The `[onnx]` extra installed the unused `onnxruntime-genai`; it now installs what the
  ONNX runtime actually imports (`onnxruntime`, `transformers`, `numpy`). There is
  deliberately no `[gpu]` extra: `onnxruntime-gpu` has to *replace* `onnxruntime`,
  which pip cannot express — the ONNX guide documents the manual steps.
- `slm_packager.runtime.onnx` imported `numpy` unguarded, so on an install without it
  the whole runtime package failed to import and took the CLI down with it.

### Fixed (API concurrency)
- **Concurrent API generations no longer corrupt model state.** `/generate` requests
  ran inference in parallel against a single runtime instance; `llama_cpp.Llama` is not
  thread-safe (shared KV cache/context), so simultaneous requests could produce garbage
  output or crash the process. All runtime calls now go through a single-worker executor
  (so nothing runs simultaneously even if the awaiting request is cancelled) plus a lock
  held for a whole generation (so a second request cannot interleave between a stream's
  tokens). The lock is acquired and released within one frame and never held across a
  return, so an abandoned response — a client that disconnects before the stream body is
  read — cannot leak it and wedge the server.
- **Server no longer deadlocks when a streaming request fails during setup.** A failure
  in `runtime.generate()` on the streaming path leaked the active-generation counter, so
  the next `/load` or shutdown waited forever and every later request got HTTP 409 until
  restart.
- **A failed `/load` no longer unloads the model that was already serving.** Config
  parsing happened inside the switch, so a typo'd path or invalid YAML ran the failure
  cleanup and left the server with no model at all. The config is now read and validated
  before the active model is touched (which also moves that blocking file IO off the
  event loop).
- **Streaming errors are no longer emitted as generated text.** The llama.cpp runtime
  yielded `"Stream error: ..."` into the token stream, indistinguishable from model
  output; it now raises. The API emits a named SSE `error` event instead of a plain
  `data:` payload.
- **`slm rm` now actually frees disk space for downloaded models.** Files pulled into the
  HuggingFace cache are `snapshots/` symlinks into `blobs/`; deleting only the symlink
  left the multi-GB blob behind.
- **Crash inside the quantization error handler.** The `type` parameter shadowed the
  builtin, so `type(e).__name__` in the generic `except` raised `TypeError` and masked
  the real error. Renamed to `quant_type` in `Quantizer` and the `slm quantize` command.
- **`slm quantize` default output path.** `str.replace(".gguf", ...)` rewrote the first
  match anywhere in the path (e.g. a `.gguf` directory name); now uses `pathlib`.
- `examples/gpt2-mps.yaml` and two config blocks in the GPU guide used
  `format: "transformers"`, which is not a valid `FormatType` — corrected to `pytorch`.

### Changed
- `build-system` requires `setuptools>=77` (needed by the SPDX `license` field; source
  builds on older setuptools previously failed with a metadata error).
- Release workflow now gates publishing: tests and lint must pass, the git tag must match
  the `pyproject.toml` version, `twine check` must pass, and the built wheel is verified
  to contain the model registry before anything is uploaded to PyPI.

### Documentation
- API server docs now show the required `POST /load` step — the previous
  `serve` → `curl /generate` example returned `400 Model not loaded`. All four endpoints
  (`/load`, `/generate`, `/info`, `/health`) are documented, with a warning about binding
  beyond localhost.
- Install docs rewritten around the extras, with a table of what each one pulls and a
  note that only `[gguf]` needs a compiler.
- Renamed stale llama.cpp build flags in all install instructions:
  `-DLLAMA_CUBLAS=on` → `-DGGML_CUDA=on`, `-DLLAMA_METAL=on` → `-DGGML_METAL=on`.
- README model table now lists all 9 registry models instead of 4.
- **All benchmark numbers re-measured** with the corrected harness and the exact command
  published alongside them. Previous figures were affected by the empty-generation bug
  above, and the size ordering they implied (a 1.1B model slower than a 4B one) was an
  artifact, not a result.
- Removed the benchmark tables from the GPU and model-format guides. They were
  unverifiable, up to 8 months stale, and contradicted each other by as much as 41x for
  the same measurement; the guides now give qualitative guidance and link to the one
  measured table.
- Replaced the invented sample output in the quickstart (its "Generation Time: 1.45s /
  Latency: 23.45 ms" pairing was arithmetically impossible) with real captured output.
- Documented the ONNX runtime's actual limitations (no greedy decoding, ignores
  `repetition_penalty`, per-token stop matching, fp32-only KV cache) and marked it
  experimental everywhere, including softening the 0.2.0 "production-ready" claim.
- Documented that GGUF quantization is supported on macOS and x86_64 Linux only.
- Docs no longer advertise "one command" or "zero friction" — the install genuinely
  takes 5–15 minutes and running a model is two commands.

## [0.2.2] - 2026-06-14

### Fixed
- `slm rm` now correctly removes HuggingFace hub cache for PyTorch/transformers models (previously only deleted the config, leaving model weights on disk)
- `slm list --installed` now shows all installed models including HF-managed ones (previously silently omitted PyTorch models)
- Python 3.12 enum rendering: all `str` enums (`RuntimeType`, `DeviceType`, `FormatType`, `QuantizationType`) now correctly display their value in f-strings and CLI output (e.g. `transformers` instead of `RuntimeType.TRANSFORMERS`)
- `ModelConfig.format` is now a validated `FormatType` enum — invalid format strings are caught at config load time instead of at runtime
- Duplicate error output in `slm pull` — error message no longer printed twice to stderr
- `slm run` error tip now gives actionable `slm pull` + `slm run` guidance instead of incorrectly referencing `slm benchmark`

### Changed
- `slm run` delegates to shared `_resolve_config_path` helper (removes duplicated path resolution logic)
- `print()` calls in downloader replaced with `click.echo` for consistent CLI output
- `Benchmarker` is now a top-level import in CLI instead of a module-level `None` mutated at call time
- Full black + isort formatting pass across all source and test files

## [0.2.1] - 2026-06-05

### Added
- Arbitrary HuggingFace model pull: `slm pull <repo-id> <filename> --name <alias>` works for any GGUF or ONNX model on HuggingFace
- Article visuals (architecture diagram, benchmark chart) in `article_visuals/`

### Changed
- Benchmark token counting now uses the runtime's tokenizer when available (transformers) or the llama.cpp `tokenize()` method, falling back to a char-based estimate — fixes inflated TPS numbers on models without an accessible tokenizer

### Fixed
- `ModelManager` lifecycle orchestration: load/unload sequencing and repetition-penalty support
- `trust_remote_code` now configurable per-runtime in YAML configs
- Improved error messages across CLI commands

## [0.2.0] - 2025-12-24

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
- **Complete ONNX runtime rewrite** (still experimental — see 0.3.0 notes)
  - Manual KV-cache management for efficient generation
  - Works with models exported via optimum with past-key-values
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

Real-world benchmarks (December 2025):

**GPT-2 (124M parameters):**
- transformers + CPU: 1.3 tok/s
- transformers + MPS (M2 Pro): 2.4 tok/s (2.14x faster)
- ONNX + CPU: 13.8 tok/s
- llama.cpp GGUF + CPU: 15-20 tok/s

**TinyLlama (1.1B parameters):**
- llama.cpp + Metal (M1): 40-60 tok/s
- transformers + MPS (M2 Pro): 28 tok/s

## [0.1.0] - 2025-11-20

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
