import os
import time
from typing import Any, Dict

from ..config.models import SLMConfig
from ..runtime import get_runtime

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None


class Benchmarker:
    def __init__(self, config: SLMConfig):
        self.config = config
        self.runtime = get_runtime(config)
        self._last_count_exact = False

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the runtime's tokenizer if available, otherwise estimate.

        Sets ``self._last_count_exact`` so callers can report whether the number came
        from a real tokenizer or a character estimate.
        """
        self._last_count_exact = False

        if not text:
            # Tokenizers return a BOS token for empty input; that is not generated
            # output, and counting it turns an empty generation into a throughput
            # number measuring nothing.
            return 0

        runtime = self.runtime
        # transformers runtime
        if hasattr(runtime, "tokenizer") and runtime.tokenizer is not None:
            self._last_count_exact = True
            return len(runtime.tokenizer.encode(text))
        # llama_cpp runtime — model exposes tokenize()
        if hasattr(runtime, "model") and runtime.model is not None:
            if hasattr(runtime.model, "tokenize"):
                try:
                    count = len(runtime.model.tokenize(text.encode()))
                    self._last_count_exact = True
                    return count
                except Exception:
                    pass
        # Fallback: rough estimate. Returns 0 for empty output — callers must not
        # invent a token that wasn't generated.
        return len(text) // 4

    # An instruction, not a completion fragment. Chat-tuned models answer a
    # sentence like "The quick brown fox..." with an immediate EOS, producing zero
    # tokens — which makes any throughput number meaningless.
    DEFAULT_PROMPT = "Write a short paragraph explaining what a small language model is."

    def run(
        self,
        prompt: str = DEFAULT_PROMPT,
        runs: int = 3,
        warmup: bool = True,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        """
        Run a benchmark on the configured model.

        Generates ``runs`` times and reports the median throughput, after an
        optional discarded warmup pass. A single cold sample mostly measures
        allocation and page-cache effects, not throughput.
        """
        metrics: Dict[str, Any] = {}
        loaded = False

        if runs < 1:
            raise ValueError("runs must be at least 1")

        # Benchmark on a copy: streaming off for clean timing, and max_tokens pinned
        # so the result doesn't depend on how verbose the model happens to be.
        # Mutating self.config.params would leak these into the caller's config.
        params = self.config.params.model_copy(deep=True)
        params.stream = False
        params.max_tokens = max_tokens

        try:
            # Measure load time
            start_load = time.time()
            self.runtime.load()
            loaded = True
            metrics["load_time_sec"] = time.time() - start_load

            # Measure memory usage (RSS of this process — includes interpreter and
            # framework overhead, so it is an upper bound, not model weight size).
            metrics["memory_mb"] = self._get_memory_mb()

            # Chat-tuned models answer an unformatted prompt with an immediate
            # end-of-sequence, which would measure throughput over zero tokens.
            formatted = self.runtime.apply_chat_template(prompt)
            metrics["chat_template_applied"] = bool(formatted)
            if formatted:
                prompt = formatted

            if warmup:
                self.runtime.generate(prompt, params)

            samples = []
            for _ in range(runs):
                start_gen = time.time()
                output = self.runtime.generate(prompt, params)
                elapsed = time.time() - start_gen
                samples.append((elapsed, self._count_tokens(output), output))

            # A run that produced no tokens measures nothing; reporting a
            # throughput for it (the old behaviour) invents data.
            productive = [(e, n) for e, n, _ in samples if n > 0 and e > 0]
            if not productive:
                raise RuntimeError(
                    "Benchmark produced no tokens\n"
                    f"   The model returned empty output for: {prompt!r}\n"
                    "Suggestions:\n"
                    "   - Try a different prompt: slm benchmark <model> is prompt-sensitive\n"
                    "   - Verify the model generates at all: slm run <model> --prompt 'Hello'"
                )

            throughputs = sorted(n / e for e, n in productive)
            median_tps = throughputs[len(throughputs) // 2]

            total_time = sum(e for e, _ in productive)
            total_tokens = sum(n for _, n in productive)

            # Reflects how the last count was actually obtained — the llama.cpp path
            # tokenizes via the model, which is exact even though it has no
            # `.tokenizer` attribute.
            metrics["token_count_method"] = (
                "tokenizer" if getattr(self, "_last_count_exact", False) else "estimate"
            )
            metrics["runs"] = len(productive)
            metrics["tokens_generated"] = total_tokens
            metrics["generation_time_sec"] = total_time / len(productive)
            metrics["tokens_per_second"] = median_tps
            metrics["ms_per_token"] = 1000.0 / median_tps
            # Kept for backwards compatibility; this is mean wall-clock time for one
            # full generation, not time-to-first-token.
            metrics["latency_ms"] = metrics["generation_time_sec"] * 1000

            return metrics
        finally:
            if loaded:
                self.runtime.unload()

    def _get_memory_mb(self) -> float:
        """Measure RSS memory, using psutil when available and a stdlib fallback otherwise."""
        if psutil is not None:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024

        if resource is None:
            return 0.0

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.name == "posix" and "darwin" in os.uname().sysname.lower():
            return rss / 1024 / 1024
        return rss / 1024
