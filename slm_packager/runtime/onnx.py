from typing import Iterator, Union, Dict, Any
import logging
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
except ImportError as e:
    ONNX_AVAILABLE = False
    IMPORT_ERROR = str(e)

from .base import BaseRuntime
from ..config.models import SLMConfig, GenerationParams

logger = logging.getLogger(__name__)

class OnnxRuntime(BaseRuntime):
    """ONNX Runtime with manual KV-cache management for efficient generation."""
    
    def load(self):
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX runtime requires 'onnxruntime' and 'transformers'\n"
                "Install with:\n"
                "  pip install onnxruntime transformers\n"
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
                str(tokenizer_path),
                local_files_only=True
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
            self.session = ort.InferenceSession(
                str(model_file),
                sess_options,
                providers=providers
            )
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
        
        logger.info(f"✅ ONNX model loaded ({self.num_layers} layers, KV-cache: {self.has_kv_cache})")
    
    def _inspect_model(self):
        """Inspect model inputs/outputs to understand KV-cache structure."""
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Get input shapes for KV-cache initialization
        self.input_info = {inp.name: inp for inp in self.session.get_inputs()}
        
        # Find KV-cache tensor names
        self.past_names = sorted([n for n in self.input_names if 'past' in n.lower()])
        self.present_names = sorted([n for n in self.output_names if 'present' in n.lower()])
        
        self.num_layers = len(self.past_names) // 2 if self.past_names else 0
        self.has_kv_cache = len(self.past_names) > 0
        self.has_position_ids = 'position_ids' in self.input_names
        
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
            shape = [int(d) if isinstance(d, int) else batch_size if d == 'batch' else 0 
                     for d in inp.shape]
            # For GPT-2: [1, num_heads, 0, head_dim] - empty sequence
            # Actually we need proper shape - let's use (batch, heads, 0, dim)
            if len(shape) == 4:
                shape[0] = batch_size  # batch
                shape[2] = 0  # sequence length = 0 for empty cache
            cache[past_name] = np.zeros(shape, dtype=np.float32)
        return cache
    
    def _forward(self, input_ids: np.ndarray, attention_mask: np.ndarray, 
                 past_kv: Dict[str, np.ndarray] = None, is_first_forward: bool = False) -> Dict[str, np.ndarray]:
        """Run model forward pass with optional KV-cache."""
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        
        # Build input dict
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Add position_ids if model requires it
        if self.has_position_ids:
            if is_first_forward:
                # First forward: positions 0 to seq_len-1
                position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            else:
                # Subsequent: position is past_len + current position
                past_len = attention_mask.shape[1] - 1
                position_ids = np.array([[past_len]], dtype=np.int64)
            inputs['position_ids'] = position_ids
        
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
    
    def _extract_kv_cache(self, outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract KV-cache from outputs for next iteration."""
        if not self.has_kv_cache:
            return None
        
        cache = {}
        for present_name in self.present_names:
            # Map present.0.key -> past_key_values.0.key (or similar)
            past_name = present_name.replace('present', 'past_key_values')
            if past_name not in self.past_names:
                # Handle different naming conventions
                past_name = present_name.replace('present', 'past')
            cache[past_name] = outputs[present_name]
        
        return cache
    
    def _sample(self, logits: np.ndarray, params: GenerationParams) -> int:
        """Sample next token from logits distribution."""
        # Apply temperature
        if params.temperature > 0 and params.temperature != 1.0:
            logits = logits / params.temperature
        
        # Convert to probabilities (with numerical stability)
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        
        # Top-k filtering
        if params.top_k > 0 and params.top_k < len(probs):
            top_k_idx = np.argsort(probs)[-params.top_k:]
            probs_filtered = np.zeros_like(probs)
            probs_filtered[top_k_idx] = probs[top_k_idx]
            probs = probs_filtered / np.sum(probs_filtered)
        
        # Top-p (nucleus) filtering
        if params.top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, params.top_p)
            probs_filtered = np.zeros_like(probs)
            probs_filtered[sorted_idx[:cutoff+1]] = probs[sorted_idx[:cutoff+1]]
            if np.sum(probs_filtered) > 0:
                probs = probs_filtered / np.sum(probs_filtered)
        
        # Sample
        return np.random.choice(len(probs), p=probs)
    
    def generate(self, prompt: str, params: GenerationParams) -> Union[str, Iterator[str]]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call runtime.load() first")
        
        if params.stream:
            return self._generate_stream(prompt, params)
        else:
            return self._generate_text(prompt, params)
    
    def _generate_text(self, prompt: str, params: GenerationParams) -> str:
        """Generate text without streaming."""
        # Tokenize prompt
        encoded = self.tokenizer(prompt, return_tensors='np')
        input_ids = encoded['input_ids']
        
        # Initialize attention mask
        attention_mask = np.ones_like(input_ids)
        
        # First forward pass (process prompt) - mark as first
        past_kv = None
        outputs = self._forward(input_ids, attention_mask, past_kv, is_first_forward=True)
        
        logits = outputs['logits']  # [batch, seq_len, vocab_size]
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
                token_text = self.tokenizer.decode([next_token])
                if any(stop_seq in token_text for stop_seq in params.stop):
                    break
            
            # Prepare next input
            input_ids = np.array([[next_token]], dtype=np.int64)
            attention_mask = np.concatenate([
                attention_mask,
                np.ones((1, 1), dtype=np.int64)
            ], axis=1)
            
            # Forward pass - not first anymore
            outputs = self._forward(input_ids, attention_mask, past_kv, is_first_forward=False)
            logits = outputs['logits']
            
            if self.has_kv_cache:
                past_kv = self._extract_kv_cache(outputs)
            
            # Sample next token
            next_token = self._sample(logits[0, -1, :], params)
            generated_tokens.append(next_token)
        
        # Decode
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    def _generate_stream(self, prompt: str, params: GenerationParams) -> Iterator[str]:
        """Generate text with streaming."""
        # Tokenize prompt
        encoded = self.tokenizer(prompt, return_tensors='np')
        input_ids = encoded['input_ids']
        attention_mask = np.ones_like(input_ids)
        
        # Process prompt
        past_kv = None
        outputs = self._forward(input_ids, attention_mask, past_kv, is_first_forward=True)
        logits = outputs['logits']
        
        if self.has_kv_cache:
            past_kv = self._extract_kv_cache(outputs)
        
        # Generate and stream tokens
        for _ in range(params.max_tokens):
            next_token = self._sample(logits[0, -1, :], params)
            
            # Decode and yield
            token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
            yield token_text
            
            # Check stopping
            if next_token == self.tokenizer.eos_token_id:
                break
            if params.stop and any(stop_seq in token_text for stop_seq in params.stop):
                break
            
            # Continue generation
            input_ids = np.array([[next_token]], dtype=np.int64)
            attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
            
            outputs = self._forward(input_ids, attention_mask, past_kv, is_first_forward=False)
            logits = outputs['logits']
            
            if self.has_kv_cache:
                past_kv = self._extract_kv_cache(outputs)
    
    def unload(self):
        if hasattr(self, 'session') and self.session:
            self.session = None
        if hasattr(self, 'model') and self.model:
            self.model = None
        if hasattr(self, 'tokenizer') and self.tokenizer:
            self.tokenizer = None
        logger.info("ONNX model unloaded")
