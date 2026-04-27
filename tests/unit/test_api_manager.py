"""Unit tests for API model lifecycle management."""
import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from slm_packager.api.manager import ModelManager, ModelBusyError
from slm_packager.config.models import GenerationParams


@pytest.mark.unit
@pytest.mark.asyncio
class TestModelManager:
    async def test_non_stream_generation_releases_active_count(self, sample_gguf_config):
        """Non-streaming generation should release manager state after success."""
        manager = ModelManager()

        runtime = MagicMock()
        runtime.is_loaded = True
        runtime.generate.return_value = "done"

        manager._runtime = runtime
        manager._config = sample_gguf_config

        result = await manager.generate("hello", GenerationParams(stream=False))

        assert result == "done"
        assert manager._active_generations == 0

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

                await load_task

        old_runtime.unload.assert_called_once()
        new_runtime.load.assert_called_once()
        assert manager._runtime is new_runtime

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
