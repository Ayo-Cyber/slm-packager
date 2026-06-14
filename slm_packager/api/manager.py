import asyncio
from typing import Any, AsyncIterator, Optional, Union

from ..config.loader import ConfigLoader
from ..config.models import GenerationParams, SLMConfig
from ..runtime import BaseRuntime, get_runtime


class ModelBusyError(RuntimeError):
    """Raised when the manager is temporarily unavailable during a model switch."""

    pass


class ModelManager:
    """
    Singleton-like manager for model state and lifecycle.
    Ensures thread-safe loading and unloading of models.
    """

    def __init__(self):
        self._runtime: Optional[BaseRuntime] = None
        self._config: Optional[SLMConfig] = None
        self._lock = asyncio.Lock()
        self._state_changed = asyncio.Condition(self._lock)
        self._active_generations = 0
        self._switching = False

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None and self._runtime.is_loaded

    @property
    def config(self) -> Optional[SLMConfig]:
        return self._config

    async def load(self, config_path: str) -> SLMConfig:
        """
        Load a model from a config file.
        This method is thread-safe and ensures only one model is loaded at a time.
        """
        async with self._state_changed:
            try:
                # Load config
                config = ConfigLoader.load(config_path)

                self._switching = True
                await self._state_changed.wait_for(lambda: self._active_generations == 0)

                # Unload existing if any
                if self._runtime:
                    # Run unload in executor to allow async context to breathe (if it takes time)
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self._runtime.unload)
                    self._runtime = None

                # Initialize new runtime
                runtime = get_runtime(config)

                # Run load in executor because it blocks (IO/CPU heavy)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, runtime.load)

                self._runtime = runtime
                self._config = config
                return config
            except Exception:
                # Ensure clean state on failure
                if self._runtime:
                    try:
                        self._runtime.unload()
                    except Exception:
                        pass
                self._runtime = None
                self._config = None
                raise
            finally:
                self._switching = False
                self._state_changed.notify_all()

    async def unload(self):
        """Unload the current model."""
        async with self._state_changed:
            self._switching = True
            try:
                await self._state_changed.wait_for(lambda: self._active_generations == 0)
                if self._runtime:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self._runtime.unload)
                    self._runtime = None
                    self._config = None
            finally:
                self._switching = False
                self._state_changed.notify_all()

    async def generate(
        self, prompt: str, params: Optional[GenerationParams] = None
    ) -> Union[str, Any]:
        """
        Generate text from the loaded model.
        Returns a string (non-stream) or a generator (stream).
        """
        # Snapshot runtime and config under lock to avoid race with concurrent /load.
        async with self._state_changed:
            if self._switching:
                raise ModelBusyError("Model is busy loading or unloading. Try again shortly.")

            runtime = self._runtime
            config = self._config

            if not runtime or not runtime.is_loaded:
                raise RuntimeError("Model not loaded. Call /load first.")

            # Use provided params or fall back to config default, but avoid sharing mutable state.
            effective_params = (params or config.params).model_copy(deep=True)
            self._active_generations += 1

        loop = asyncio.get_running_loop()

        try:
            if effective_params.stream:
                iterator = runtime.generate(prompt, effective_params)
                return self._stream_generation(iterator)

            try:
                return await loop.run_in_executor(None, runtime.generate, prompt, effective_params)
            finally:
                await self._release_generation()
        except Exception:
            raise

    async def _release_generation(self):
        async with self._state_changed:
            self._active_generations -= 1
            self._state_changed.notify_all()

    async def _stream_generation(self, iterator: Any) -> AsyncIterator[str]:
        """Wrap a streaming iterator so generation bookkeeping survives disconnects and errors."""
        loop = asyncio.get_running_loop()
        sentinel = object()

        try:
            if hasattr(iterator, "__aiter__"):
                async for chunk in iterator:
                    yield chunk
                return

            while True:
                chunk = await loop.run_in_executor(None, lambda: next(iterator, sentinel))
                if chunk is sentinel:
                    break
                yield chunk
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            await self._release_generation()
