# Annotations are strings so `np.ndarray` in signatures is not evaluated at import
# time — this module must stay importable when numpy is not installed.
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

IMPORT_ERROR = ""
try:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    ONNX_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR = str(e)
    ONNX_AVAILABLE = False

    np = None  # type: ignore[assignment]

    class _SessionOptions:
        def __init__(self):
            self.intra_op_num_threads = 0

    def _missing_inference_session(*args, **kwargs):
        raise ImportError(
            "The ONNX runtime requires 'onnxruntime', 'transformers' and 'numpy'\n"
            "Install with:\n"
            "  pip install 'slm-packager[onnx]'\n"
            f"\nError: {IMPORT_ERROR}"
        )

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError(
                "The ONNX runtime requires 'onnxruntime', 'transformers' and 'numpy'\n"
                "Install with:\n"
                "  pip install 'slm-packager[onnx]'\n"
                f"\nError: {IMPORT_ERROR}"
            )

    ort = SimpleNamespace(
        SessionOptions=_SessionOptions,
        InferenceSession=_missing_inference_session,
    )

from ..config.models import GenerationParams, SLMConfig
from .base import BaseRuntime, _truncate_at_stop

logger = logging.getLogger(__name__)


class OnnxRuntime(BaseRuntime):
    """ONNX Runtime with manual KV-cache management for efficient generation."""

    def load(self):
        if not ONNX_AVAILABLE:
            raise ImportError(
                "The ONNX runtime requires 'onnxruntime', 'transformers' and 'numpy'\n"
                "Install with:\n"
                "  pip install 'slm-packager[onnx]'\n"
                f"\nError: {IMPORT_ERROR}"
            )

        model_path = Path(self.config.model.path)

        # Find .onnx file
        if model_path.is_dir():
            onnx_files = list(model_path.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(
                    f"No .onnx files found in {model_path}\n"
                    "Export a model first with optimum:\n"
                    "  optimum-cli export onnx --model gpt2 models/gpt2-onnx/"
                )
            model_file = onnx_files[0]
            logger.info(f"Found ONNX model: {model_file.name}")
        else:
            model_file = model_path

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        # Load tokenizer from directory containing model
        tokenizer_path = model_path if model_path.is_dir() else model_path.parent

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), local_files_only=True
            )
            logger.info("Tokenizer loaded from model directory")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer from {tokenizer_path}\n"
                "ONNX models need tokenizer files in the same directory\n"
                f"Error: {str(e)}"
            ) from e

        # Create ONNX session
        sess_options = ort.SessionOptions()
        if self.config.runtime.threads > 0:
            sess_options.intra_op_num_threads = self.config.runtime.threads

        providers = ["CPUExecutionProvider"]
        if self.config.runtime.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(str(model_file), sess_options, providers=providers)
            self.model = self.session  # For is_loaded check
            logger.info("ONNX session created")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create ONNX session\n"
                f"Error: {str(e)}\n"
                "Ensure the .onnx file is valid and compatible"
            ) from e

        # Inspect model I/O for KV-cache support
        self._inspect_model()

        logger.info(
            f"✅ ONNX model loaded ({self.num_layers} layers, KV-cache: {self.has_kv_cache})"
        )

    def _inspect_model(self):
        """Inspect model inputs/outputs to understand KV-cache structure."""
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Get input shapes for KV-cache initialization
        self.input_info = {inp.name: inp for inp in self.session.get_inputs()}

        # Find KV-cache tensor names
        self.past_names = sorted([n for n in self.input_names if "past" in n.lower()])
        self.present_names = sorted([n for n in self.output_names if "present" in n.lower()])

        self.num_layers = len(self.past_names) // 2 if self.past_names else 0
        self.has_kv_cache = len(self.past_names) > 0
        self.has_position_ids = "position_ids" in self.input_names

        logger.debug(f"Model I/O: {len(self.input_names)} inputs, {len(self.output_names)} outputs")
        if self.has_kv_cache:
            logger.debug(f"KV-cache: {self.num_layers} layers")
        if self.has_position_ids:
            logger.debug("Model requires position_ids")

    def _init_empty_kv_cache(self, batch_size=1):
        """Initialize empty KV-cache tensors for first pass."""
        if not self.has_kv_cache:
            return None

        cache = {}
        for past_name in self.past_names:
            # Get shape from input info: [batch, num_heads, 0, head_dim]
            inp = self.input_info[past_name]
            shape = [
                int(d) if isinstance(d, int) else batch_size if d == "batch" else 0
                for d in inp.shape
            ]
            # For GPT-2: [1, num_heads, 0, head_dim] - empty sequence
            # Actually we need proper shape - let's use (batch, heads, 0, dim)
            if len(shape) == 4:
                shape[0] = batch_size  # batch
                shape[2] = 0  # sequence length = 0 for empty cache
            cache[past_name] = np.zeros(shape, dtype=np.float32)
        return cache

    def _forward(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        past_kv: Dict[str, np.ndarray] = None,
        is_first_forward: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run model forward pass with optional KV-cache."""
        try:
            seq_len = input_ids.shape[1]
            batch_size = input_ids.shape[0]

            # Build input dict
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

            # Add position_ids if model requires it
            if self.has_position_ids:
                if is_first_forward:
                    # First forward: positions 0 to seq_len-1
                    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
                else:
                    # Subsequent: position is past_len + current position
                    past_len = attention_mask.shape[1] - 1
                    position_ids = np.array([[past_len]], dtype=np.int64)
                inputs["position_ids"] = position_ids

            # Add KV-cache
            if self.has_kv_cache:
                if is_first_forward:
                    # Initialize empty cache for first forward
                    cache = self._init_empty_kv_cache(batch_size)
                    inputs.update(cache)
                elif past_kv is not None:
                    inputs.update(past_kv)

            # Run inference
            outputs = self.session.run(self.output_names, inputs)

            # Convert to dict
            return {name: output for name, output in zip(self.output_names, outputs)}

        except Exception as e:
            logger.error(f"ONNX inference failed: {str(e)}")
            raise RuntimeError(f"Model inference failed: {str(e)}") from e

    def _extract_kv_cache(self, outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract KV-cache from outputs for next iteration."""
        if not self.has_kv_cache:
            return None

        cache = {}
        for present_name in self.present_names:
            # Map present.0.key -> past_key_values.0.key (or similar)
            past_name = present_name.replace("present", "past_key_values")
            if past_name not in self.past_names:
                # Handle different naming conventions
                past_name = present_name.replace("present", "past")
            cache[past_name] = outputs[present_name]

        return cache

    def _sample(
        self,
        logits: np.ndarray,
        params: GenerationParams,
        generated: Optional[Sequence[int]] = None,
    ) -> int:
        """Sample next token from logits distribution."""
        # Check for NaNs
        if np.isnan(logits).any():
            logger.warning("NaNs detected in logits, replacing with -inf")
            logits = np.nan_to_num(logits, nan=-np.inf)

        # Repetition penalty, applied to logits before any temperature scaling —
        # matching the convention used by transformers and llama.cpp.
        if generated and params.repetition_penalty and params.repetition_penalty != 1.0:
            logits = logits.astype(np.float64, copy=True)
            for token_id in set(generated):
                if 0 <= token_id < len(logits):
                    score = logits[token_id]
                    # Divide positive scores, multiply negative ones, so the penalty
                    # always pushes the token *down*.
                    logits[token_id] = (
                        score / params.repetition_penalty
                        if score > 0
                        else score * params.repetition_penalty
                    )

        # temperature == 0 means greedy decoding: take the most likely token rather
        # than sampling at temperature 1.0.
        if not params.temperature or params.temperature <= 0:
            return int(np.argmax(logits))

        # Apply temperature
        if params.temperature != 1.0:
            logits = logits / params.temperature

        # Convert to probabilities (with numerical stability)
        logits_max = np.max(logits)
        probs = np.exp(logits - logits_max)
        probs_sum = np.sum(probs)

        # Avoid division by zero
        if probs_sum == 0:
            logger.warning("Probability sum is zero, falling back to uniform")
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        # Top-k filtering
        if params.top_k > 0 and params.top_k < len(probs):
            top_k_idx = np.argsort(probs)[-params.top_k :]
            probs_filtered = np.zeros_like(probs)
            probs_filtered[top_k_idx] = probs[top_k_idx]
            probs = probs_filtered / np.sum(probs_filtered)

        # Top-p (nucleus) filtering
        if params.top_p < 1.0 and params.top_p > 0:
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, params.top_p)
            probs_filtered = np.zeros_like(probs)
            probs_filtered[sorted_idx[: cutoff + 1]] = probs[sorted_idx[: cutoff + 1]]
            if np.sum(probs_filtered) > 0:
                probs = probs_filtered / np.sum(probs_filtered)

        # Sample
        return np.random.choice(len(probs), p=probs)

    def generate(self, prompt: str, params: GenerationParams) -> Union[str, Iterator[str]]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call runtime.load() first")

        if not prompt:
            logger.warning("Received empty prompt")
            return "" if not params.stream else iter([])

        if params.stream:
            return self._generate_stream(prompt, params)
        else:
            return self._generate_text(prompt, params)

    def _generate_text(self, prompt: str, params: GenerationParams) -> str:
        """Generate text without streaming."""
        # Tokenize prompt
        encoded = self.tokenizer(prompt, return_tensors="np")
        input_ids = encoded["input_ids"]

        # Initialize attention mask
        attention_mask = np.ones_like(input_ids)

        # First forward pass (process prompt) - mark as first
        past_kv = None
        outputs = self._forward(input_ids, attention_mask, past_kv, is_first_forward=True)

        logits = outputs["logits"]  # [batch, seq_len, vocab_size]
        if self.has_kv_cache:
            past_kv = self._extract_kv_cache(outputs)

        # Sample first token
        next_token = self._sample(logits[0, -1, :], params)
        generated_tokens = [next_token]

        # Generate remaining tokens
        for _ in range(params.max_tokens - 1):
            # Check stopping conditions
            if next_token == self.tokenizer.eos_token_id:
                break
            if params.stop:
                # Check the decoded text so far, not the latest token alone: a stop
                # sequence usually spans several tokens and would never match here.
                text_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                if _truncate_at_stop(text_so_far, params.stop) != text_so_far:
                    break

            # Prepare next input
            input_ids = np.array([[next_token]], dtype=np.int64)
            attention_mask = np.concatenate(
                [attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1
            )

            # Forward pass - not first anymore
            outputs = self._forward(input_ids, attention_mask, past_kv, is_first_forward=False)
            logits = outputs["logits"]

            if self.has_kv_cache:
                past_kv = self._extract_kv_cache(outputs)

            # Sample next token
            next_token = self._sample(logits[0, -1, :], params, generated_tokens)
            generated_tokens.append(next_token)

        # Decode, trimming anything at or past a stop sequence
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return _truncate_at_stop(text, params.stop)

    def _generate_stream(self, prompt: str, params: GenerationParams) -> Iterator[str]:
        """Generate text with streaming."""
        # Tokenize prompt
        encoded = self.tokenizer(prompt, return_tensors="np")
        input_ids = encoded["input_ids"]
        attention_mask = np.ones_like(input_ids)

        # Process prompt
        past_kv = None
        outputs = self._forward(input_ids, attention_mask, past_kv, is_first_forward=True)
        logits = outputs["logits"]

        if self.has_kv_cache:
            past_kv = self._extract_kv_cache(outputs)

        # Generate and stream tokens
        generated_tokens: List[int] = []
        emitted = 0  # characters of the decoded text already yielded
        longest_stop = max((len(s) for s in params.stop if s), default=0)

        for _ in range(params.max_tokens):
            next_token = self._sample(logits[0, -1, :], params, generated_tokens)
            generated_tokens.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if params.stop:
                truncated = _truncate_at_stop(text, params.stop)
                if truncated != text:
                    # Emit up to the stop sequence and finish without ever showing it.
                    if len(truncated) > emitted:
                        yield truncated[emitted:]
                    return
                # Hold back a possible partial stop sequence straddling chunks.
                safe_len = max(len(text) - (longest_stop - 1), emitted)
            else:
                safe_len = len(text)

            if safe_len > emitted:
                yield text[emitted:safe_len]
                emitted = safe_len

            # Continue generation
            input_ids = np.array([[next_token]], dtype=np.int64)
            attention_mask = np.concatenate(
                [attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1
            )

            outputs = self._forward(input_ids, attention_mask, past_kv, is_first_forward=False)
            logits = outputs["logits"]

            if self.has_kv_cache:
                past_kv = self._extract_kv_cache(outputs)

        # Flush whatever was held back for stop-sequence matching.
        if generated_tokens:
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            text = _truncate_at_stop(text, params.stop)
            if len(text) > emitted:
                yield text[emitted:]

    def unload(self):
        import gc

        if hasattr(self, "session") and self.session:
            self.session = None
        if hasattr(self, "model") and self.model:
            self.model = None
        if hasattr(self, "tokenizer") and self.tokenizer:
            self.tokenizer = None
        gc.collect()
        logger.info("ONNX model unloaded")
