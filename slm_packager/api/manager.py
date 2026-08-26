import asyncio
from concurrent.futures import ThreadPoolExecutor
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
        # Mutated only from the event loop thread, so a plain bool is sufficient.
        self._switching = False
        # Runtimes (llama_cpp.Llama in particular) are not thread-safe: concurrent
        # generations share KV cache/context state, so only one may touch the model
        # at a time. Two mechanisms enforce that, and both are needed:
        #
        # 1. A single-worker executor. Every runtime call goes through it, so work
        #    cannot run *simultaneously* even when the awaiting coroutine is
        #    cancelled — an abandoned run_in_executor future keeps running, and the
        #    next call queues behind it rather than entering the model alongside it.
        # 2. This lock, which additionally prevents *interleaving*: a stream holds
        #    internal model state between chunks, so another generation must not slip
        #    in between two of its tokens. load()/unload() take it too, so a model
        #    swap waits for in-flight generation instead of pulling the model out
        #    from under it.
        #
        # Two invariants keep it from leaking, both of which matter because Starlette
        # abandons a response body when a client disconnects:
        #   - it is acquired and released within a single frame, never held across a
        #     return, so an un-started generator holds nothing;
        #   - release is synchronous (Lock.__aexit__ never suspends), so it still
        #     happens when a generator is closed mid-iteration. An `await` that can
        #     suspend must never appear in those finally blocks.
        self._inference_lock = asyncio.Lock()
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="slm-inference"
        )

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
        loop = asyncio.get_running_loop()

        # Read and validate the config *before* touching the loaded model: a bad path
        # or invalid YAML must leave a working model serving, not unload it. Parsing
        # is blocking file IO, so keep it off the event loop and outside the lock.
        config = await loop.run_in_executor(None, ConfigLoader.load, config_path)

        # Holding the inference lock is what makes the swap safe: any in-flight
        # generation finishes first, and new ones queue behind us.
        async with self._inference_lock:
            self._switching = True
            try:
                # Unload existing if any, on the inference thread so it queues
                # behind any abandoned generation work still running there.
                if self._runtime:
                    await self._run_in_inference_thread(self._runtime.unload)
                    self._runtime = None
                    self._config = None

                # Initialize new runtime
                runtime = get_runtime(config)

                # Run load on the inference thread because it blocks (IO/CPU heavy)
                await self._run_in_inference_thread(runtime.load)

                self._runtime = runtime
                self._config = config
                return config
            except Exception:
                # Any previously loaded model was already unloaded above; drop a
                # half-initialized replacement so state stays consistent.
                if self._runtime:
                    try:
                        await self._run_in_inference_thread(self._runtime.unload)
                    except Exception:
                        pass
                self._runtime = None
                self._config = None
                raise
            finally:
                self._switching = False

    async def unload(self):
        """Unload the current model."""
        async with self._inference_lock:
            self._switching = True
            try:
                if self._runtime:
                    await self._run_in_inference_thread(self._runtime.unload)
                    self._runtime = None
                    self._config = None
            finally:
                self._switching = False

    def shutdown(self):
        """Release the inference thread. Call after the final unload()."""
        self._inference_executor.shutdown(wait=False)

    async def generate(
        self, prompt: str, params: Optional[GenerationParams] = None, raw: bool = False
    ) -> Union[str, Any]:
        """
        Generate text from the loaded model.
        Returns a string (non-stream) or a generator (stream).
        """
        # This snapshot needs no lock: it contains no await, so the event loop cannot
        # interleave a /load between the checks and the copy.
        if self._switching:
            raise ModelBusyError("Model is busy loading or unloading. Try again shortly.")

        runtime = self._runtime
        config = self._config

        if not runtime or not runtime.is_loaded:
            raise RuntimeError("Model not loaded. Call /load first.")

        # Use provided params or fall back to config default, but avoid sharing mutable state.
        effective_params = (params or config.params).model_copy(deep=True)

        if effective_params.stream:
            # Returned un-started: the generator takes the lock itself on first
            # iteration, so an endpoint that never iterates it (client disconnected
            # before the body was streamed) leaves nothing held.
            return self._stream_generation(runtime, prompt, effective_params, raw)

        return await self._generate_locked(runtime, prompt, effective_params, raw)

    async def _run_in_inference_thread(self, fn, *args) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._inference_executor, fn, *args)

    @staticmethod
    def _prepare_prompt(runtime: BaseRuntime, prompt: str, raw: bool) -> str:
        """Wrap the prompt in the model's chat format unless the caller opted out.

        Matches `slm run`: without this an instruction-tuned model continues the text
        instead of answering, and often returns nothing at all.
        """
        if raw:
            return prompt
        try:
            return runtime.apply_chat_template(prompt) or prompt
        except Exception:
            return prompt

    async def _generate_locked(
        self, runtime: BaseRuntime, prompt: str, params: GenerationParams, raw: bool = False
    ) -> str:
        """Run one non-streaming generation with exclusive access to the runtime."""
        async with self._inference_lock:
            prepared = await self._run_in_inference_thread(
                self._prepare_prompt, runtime, prompt, raw
            )
            return await self._run_in_inference_thread(runtime.generate, prepared, params)

    async def _stream_generation(
        self,
        runtime: BaseRuntime,
        prompt: str,
        params: GenerationParams,
        raw: bool = False,
    ) -> AsyncIterator[str]:
        """Stream one generation with exclusive access to the runtime.

        The lock is taken on first iteration and released by ``__aexit__`` in the same
        frame. Nothing in the exit path may await something that can suspend: this
        generator is routinely closed mid-iteration when a client disconnects, and a
        suspending await there never resumes — which would strand the lock and hang
        every later request and the shutdown path with it.
        """
        sentinel = object()

        async with self._inference_lock:
            iterator: Any = None
            try:
                prepared = await self._run_in_inference_thread(
                    self._prepare_prompt, runtime, prompt, raw
                )
                # Stream setup can do real work (tokenization, spawning a worker
                # thread), so it belongs on the inference thread and inside the lock.
                iterator = await self._run_in_inference_thread(runtime.generate, prepared, params)

                if hasattr(iterator, "__aiter__"):
                    async for chunk in iterator:
                        yield chunk
                    return

                while True:
                    chunk = await self._run_in_inference_thread(next, iterator, sentinel)
                    if chunk is sentinel:
                        break
                    yield chunk
            finally:
                # Synchronous only, per the note above. If the worker thread is still
                # inside this generator, close() raises "already executing"; the
                # abandoned iterator is then reclaimed by GC instead.
                close = getattr(iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
