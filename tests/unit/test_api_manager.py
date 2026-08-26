"""Unit tests for API model lifecycle management."""

import asyncio
import gc
import threading
from unittest.mock import MagicMock, patch

import pytest

from slm_packager.api.manager import ModelBusyError, ModelManager
from slm_packager.config.models import GenerationParams


@pytest.mark.unit
@pytest.mark.asyncio
class TestModelManager:
    async def test_non_stream_generation_releases_the_lock(self, sample_gguf_config):
        """Non-streaming generation should release manager state after success."""
        manager = ModelManager()

        runtime = MagicMock()
        runtime.is_loaded = True
        runtime.generate.return_value = "done"

        manager._runtime = runtime
        manager._config = sample_gguf_config

        result = await manager.generate("hello", GenerationParams(stream=False))

        assert result == "done"
        assert not manager._inference_lock.locked()

    async def test_load_waits_for_active_stream_before_switching(self, sample_gguf_config):
        """Active streams should finish before the current runtime is unloaded."""
        manager = ModelManager()

        old_runtime = MagicMock()
        old_runtime.is_loaded = True
        old_runtime.unload.return_value = None

        release_stream = threading.Event()

        def old_stream(*args, **kwargs):
            yield "chunk-1"
            release_stream.wait(timeout=1)
            yield "chunk-2"

        old_runtime.generate.side_effect = old_stream

        new_runtime = MagicMock()
        new_runtime.is_loaded = True
        new_runtime.load.return_value = None
        new_runtime.unload.return_value = None

        manager._runtime = old_runtime
        manager._config = sample_gguf_config

        stream = await manager.generate("hello", GenerationParams(stream=True))
        first_chunk = await stream.__anext__()
        assert first_chunk == "chunk-1"

        with patch("slm_packager.api.manager.ConfigLoader.load", return_value=sample_gguf_config):
            with patch("slm_packager.api.manager.get_runtime", return_value=new_runtime):
                load_task = asyncio.create_task(manager.load("/fake/config.yaml"))

                await asyncio.sleep(0.05)
                assert not load_task.done()
                old_runtime.unload.assert_not_called()

                release_stream.set()
                remaining = [chunk async for chunk in stream]
                assert remaining == ["chunk-2"]

                # Bounded: if the generation bookkeeping regresses this hangs forever,
                # and a named timeout failure beats a killed CI job.
                await asyncio.wait_for(load_task, timeout=5)

        old_runtime.unload.assert_called_once()
        new_runtime.load.assert_called_once()
        assert manager._runtime is new_runtime

    async def test_api_applies_chat_template_like_the_cli(self, sample_gguf_config):
        """`/generate` must format prompts the same way `slm run` does.

        Otherwise the same model and prompt give different answers depending on
        whether you go through the CLI or the API.
        """
        manager = ModelManager()

        runtime = MagicMock()
        runtime.is_loaded = True
        runtime.generate.return_value = "ok"
        runtime.apply_chat_template.side_effect = lambda p: f"<|user|>\n{p}\n<|assistant|>\n"

        manager._runtime = runtime
        manager._config = sample_gguf_config

        await manager.generate("hi", GenerationParams(stream=False))

        assert runtime.generate.call_args[0][0] == "<|user|>\nhi\n<|assistant|>\n"

    async def test_raw_bypasses_the_chat_template(self, sample_gguf_config):
        """Callers that want to control formatting themselves can opt out."""
        manager = ModelManager()

        runtime = MagicMock()
        runtime.is_loaded = True
        runtime.generate.return_value = "ok"
        runtime.apply_chat_template.side_effect = AssertionError("must not be called")

        manager._runtime = runtime
        manager._config = sample_gguf_config

        await manager.generate("hi", GenerationParams(stream=False), raw=True)

        assert runtime.generate.call_args[0][0] == "hi"

    async def test_templating_failure_falls_back_to_the_raw_prompt(self, sample_gguf_config):
        """A broken template must not fail the request."""
        manager = ModelManager()

        runtime = MagicMock()
        runtime.is_loaded = True
        runtime.generate.return_value = "ok"
        runtime.apply_chat_template.side_effect = RuntimeError("bad template")

        manager._runtime = runtime
        manager._config = sample_gguf_config

        assert await manager.generate("hi", GenerationParams(stream=False)) == "ok"
        assert runtime.generate.call_args[0][0] == "hi"

    async def test_failed_load_keeps_existing_model_serving(self, sample_gguf_config):
        """A bad config path must not unload the model that is already working."""
        manager = ModelManager()

        runtime = MagicMock()
        runtime.is_loaded = True
        runtime.generate.return_value = "still here"

        manager._runtime = runtime
        manager._config = sample_gguf_config

        with patch(
            "slm_packager.api.manager.ConfigLoader.load",
            side_effect=FileNotFoundError("no such config"),
        ):
            with pytest.raises(FileNotFoundError):
                await manager.load("/does/not/exist.yaml")

        runtime.unload.assert_not_called()
        assert manager.is_loaded
        assert manager._config is sample_gguf_config
        assert await manager.generate("hi", GenerationParams(stream=False)) == "still here"

    async def test_stream_setup_failure_releases_the_lock(self, sample_gguf_config):
        """A failure while setting up a stream must not leave the inference lock held.

        Setup is deferred to first iteration, so the error surfaces there (and reaches
        the client as a named SSE error event) rather than at call time.
        """
        manager = ModelManager()

        runtime = MagicMock()
        runtime.is_loaded = True
        runtime.generate.side_effect = ValueError("tokenization failed")

        manager._runtime = runtime
        manager._config = sample_gguf_config

        stream = await manager.generate("hello", GenerationParams(stream=True))

        with pytest.raises(ValueError, match="tokenization failed"):
            await stream.__anext__()

        assert not manager._inference_lock.locked()

        # The leak used to brick the server: unload would wait forever for the
        # in-flight generation to clear. It must complete promptly now.
        await asyncio.wait_for(manager.unload(), timeout=1)

    async def test_concurrent_generations_are_serialized(self, sample_gguf_config):
        """Two /generate calls must never run inference concurrently on one runtime."""
        manager = ModelManager()

        in_flight = 0
        max_in_flight = 0
        counter_lock = threading.Lock()

        def slow_generate(*args, **kwargs):
            nonlocal in_flight, max_in_flight
            with counter_lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            threading.Event().wait(timeout=0.05)
            with counter_lock:
                in_flight -= 1
            return "done"

        runtime = MagicMock()
        runtime.is_loaded = True
        runtime.generate.side_effect = slow_generate

        manager._runtime = runtime
        manager._config = sample_gguf_config

        params = GenerationParams(stream=False)
        results = await asyncio.gather(
            manager.generate("a", params),
            manager.generate("b", params),
            manager.generate("c", params),
        )

        assert results == ["done", "done", "done"]
        assert max_in_flight == 1
        assert not manager._inference_lock.locked()

    async def test_stream_holds_inference_lock_until_exhausted(self, sample_gguf_config):
        """A second generation must queue behind an in-progress stream, not interleave."""
        manager = ModelManager()

        runtime = MagicMock()
        runtime.is_loaded = True
        runtime.generate.side_effect = lambda *a, **k: iter(["chunk-1", "chunk-2"])

        manager._runtime = runtime
        manager._config = sample_gguf_config

        stream = await manager.generate("hello", GenerationParams(stream=True))
        assert await stream.__anext__() == "chunk-1"
        assert manager._inference_lock.locked()

        # A concurrent generation must not start while the stream is mid-flight.
        second = asyncio.create_task(manager.generate("other", GenerationParams(stream=False)))
        await asyncio.sleep(0.05)
        assert not second.done(), "second generation ran while a stream was in progress"

        remaining = [chunk async for chunk in stream]
        assert remaining == ["chunk-2"]
        assert not manager._inference_lock.locked()

        await asyncio.wait_for(second, timeout=1)
        assert not manager._inference_lock.locked()

    async def test_stream_closed_midway_releases_the_lock(self, sample_gguf_config):
        """Closing a partly-consumed stream must release the lock.

        This is what a client disconnecting mid-response does: Starlette calls
        aclose() on the body generator. If the exit path awaits anything that can
        suspend, it never resumes, the lock is stranded, and both /load and process
        shutdown hang forever.
        """
        manager = ModelManager()

        runtime = MagicMock()
        runtime.is_loaded = True
        runtime.generate.side_effect = lambda *a, **k: iter([f"chunk-{i}" for i in range(50)])

        manager._runtime = runtime
        manager._config = sample_gguf_config

        stream = await manager.generate("hello", GenerationParams(stream=True))
        assert await stream.__anext__() == "chunk-0"
        assert manager._inference_lock.locked()

        # Abandon it partway through, as a disconnect does.
        await stream.aclose()

        assert not manager._inference_lock.locked()

        # A model switch must still be able to proceed.
        with patch("slm_packager.api.manager.ConfigLoader.load", return_value=sample_gguf_config):
            with patch("slm_packager.api.manager.get_runtime", return_value=runtime):
                await asyncio.wait_for(manager.load("/fake/config.yaml"), timeout=2)

        await asyncio.wait_for(manager.unload(), timeout=2)

    async def test_abandoned_stream_does_not_wedge_the_manager(self, sample_gguf_config):
        """Dropping a stream without iterating it must not hold the lock or the counter.

        Starlette can abandon a StreamingResponse body when the client disconnects
        before it is consumed, so this is a normal event, not an edge case. Holding
        the inference lock across that boundary used to wedge every later request.
        """
        manager = ModelManager()

        runtime = MagicMock()
        runtime.is_loaded = True
        runtime.generate.side_effect = lambda *a, **k: iter(["chunk-1"])

        manager._runtime = runtime
        manager._config = sample_gguf_config

        stream = await manager.generate("hello", GenerationParams(stream=True))
        del stream  # never iterated — exactly what a pre-body disconnect produces
        gc.collect()

        assert not manager._inference_lock.locked()
        assert not manager._inference_lock.locked()

        # The manager must still be fully usable.
        result = await asyncio.wait_for(
            manager.generate("again", GenerationParams(stream=False)), timeout=2
        )
        assert result is not None
        await asyncio.wait_for(manager.unload(), timeout=2)

    async def test_generate_rejects_requests_during_switch(self, sample_gguf_config):
        """New generations should be rejected while a load/unload switch is in progress."""
        manager = ModelManager()
        runtime = MagicMock()
        runtime.is_loaded = True

        manager._runtime = runtime
        manager._config = sample_gguf_config
        manager._switching = True

        with pytest.raises(ModelBusyError, match="busy"):
            await manager.generate("hello")
