import time
import os
from typing import Dict, Any
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

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the runtime's tokenizer if available, otherwise estimate."""
        runtime = self.runtime
        # transformers runtime
        if hasattr(runtime, 'tokenizer') and runtime.tokenizer is not None:
            return len(runtime.tokenizer.encode(text))
        # llama_cpp runtime — model exposes tokenize()
        if hasattr(runtime, 'model') and runtime.model is not None:
            if hasattr(runtime.model, 'tokenize'):
                try:
                    return len(runtime.model.tokenize(text.encode()))
                except Exception:
                    pass
        # Fallback: estimate, minimum 1 to avoid zero TPS
        return max(len(text) // 4, 1)

    def run(self, prompt: str = "The quick brown fox jumps over the lazy dog.") -> Dict[str, Any]:
        """
        Run a benchmark on the configured model.
        """
        metrics = {}
        original_stream = self.config.params.stream
        loaded = False

        try:
            # Measure load time
            start_load = time.time()
            self.runtime.load()
            loaded = True
            metrics["load_time_sec"] = time.time() - start_load

            # Measure memory usage (RSS)
            metrics["memory_mb"] = self._get_memory_mb()

            # Measure generation
            start_gen = time.time()
            # Force non-streaming for accurate timing
            self.config.params.stream = False

            output = self.runtime.generate(prompt, self.config.params)

            metrics["generation_time_sec"] = time.time() - start_gen

            # Count tokens using the actual tokenizer when available
            num_tokens = self._count_tokens(output)
            has_tokenizer = hasattr(self.runtime, 'tokenizer') and self.runtime.tokenizer is not None
            metrics["token_count_method"] = "tokenizer" if has_tokenizer else "estimate"
            metrics["tokens_per_second"] = num_tokens / metrics["generation_time_sec"]
            metrics["latency_ms"] = metrics["generation_time_sec"] * 1000

            return metrics
        finally:
            self.config.params.stream = original_stream
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
