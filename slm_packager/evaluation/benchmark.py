import time
import psutil
import os
from typing import Dict, Any
from ..config.models import SLMConfig
from ..runtime import get_runtime

class Benchmarker:
    def __init__(self, config: SLMConfig):
        self.config = config
        self.runtime = get_runtime(config)

    def run(self, prompt: str = "The quick brown fox jumps over the lazy dog.") -> Dict[str, Any]:
        """
        Run a benchmark on the configured model.
        """
        metrics = {}
        
        # Measure load time
        start_load = time.time()
        self.runtime.load()
        metrics["load_time_sec"] = time.time() - start_load
        
        # Measure memory usage (RSS)
        process = psutil.Process(os.getpid())
        metrics["memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        # Measure generation
        start_gen = time.time()
        # Force non-streaming for accurate timing
        original_stream = self.config.params.stream
        self.config.params.stream = False
        
        output = self.runtime.generate(prompt, self.config.params)
        
        metrics["generation_time_sec"] = time.time() - start_gen
        
        # Restore config
        self.config.params.stream = original_stream
        
        # Calculate tokens per second (approximate)
        # In a real system, we'd use the tokenizer to count tokens
        num_chars = len(output)
        num_tokens = num_chars / 4.0 # Rough approximation
        metrics["tokens_per_second"] = num_tokens / metrics["generation_time_sec"]
        metrics["latency_ms"] = metrics["generation_time_sec"] * 1000
        
        self.runtime.unload()
        
        return metrics
