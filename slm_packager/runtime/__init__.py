import importlib
from typing import TYPE_CHECKING, Tuple

from ..config.models import RuntimeType, SLMConfig
from .base import BaseRuntime

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .llama_cpp import LlamaCppRuntime
    from .onnx import OnnxRuntime
    from .transformers import TransformersRuntime

# Runtime modules are imported lazily. Each one pulls a large third-party engine
# (llama-cpp-python, torch, onnxruntime) that is an optional extra, so importing all
# three up front would make the CLI unusable on a lean install — and slow on a full
# one. Only the configured runtime gets imported.
_RUNTIMES = {
    RuntimeType.LLAMA_CPP: (".llama_cpp", "LlamaCppRuntime", "gguf"),
    RuntimeType.ONNX: (".onnx", "OnnxRuntime", "onnx"),
    RuntimeType.TRANSFORMERS: (".transformers", "TransformersRuntime", "torch"),
}

# Attribute name -> module, for `from slm_packager.runtime import LlamaCppRuntime`.
_LAZY_CLASSES = {
    "LlamaCppRuntime": ".llama_cpp",
    "OnnxRuntime": ".onnx",
    "TransformersRuntime": ".transformers",
}

__all__ = ["BaseRuntime", "get_runtime", *_LAZY_CLASSES]


def _load_runtime_class(module_name: str, class_name: str, extra: str):
    try:
        module = importlib.import_module(module_name, package=__name__)
    except ImportError as e:
        # The engine's own optional imports are guarded inside each module, so this
        # only fires when a hard dependency of the module is missing entirely.
        raise ImportError(
            f"The {extra!r} runtime is not installed.\n"
            f"Install it with:\n"
            f"   pip install 'slm-packager[{extra}]'\n"
            f"\n   Error details: {e}"
        ) from e
    return getattr(module, class_name)


def get_runtime(config: SLMConfig) -> BaseRuntime:
    try:
        module_name, class_name, extra = _RUNTIMES[config.runtime.type]
    except KeyError:
        raise ValueError(f"Unsupported runtime type: {config.runtime.type}")

    # Check this module's namespace first so the class can be substituted (tests
    # patch it here); otherwise import the engine on demand.
    runtime_class = globals().get(class_name)
    if runtime_class is None:
        runtime_class = _load_runtime_class(module_name, class_name, extra)
    return runtime_class(config)


def __getattr__(name: str):
    """Expose runtime classes lazily (PEP 562)."""
    module_name = _LAZY_CLASSES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    extra = next(e for _, (m, _c, e) in _RUNTIMES.items() if m == module_name)
    return _load_runtime_class(module_name, name, extra)
