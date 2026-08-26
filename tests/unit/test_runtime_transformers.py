"""Unit tests for the transformers runtime's parameter handling."""

from unittest.mock import MagicMock

import pytest

# torch/transformers ship with the optional [torch] extra.
pytest.importorskip("torch", reason="requires the [torch] extra")

from slm_packager.config.models import GenerationParams
from slm_packager.runtime.transformers import TransformersRuntime


@pytest.fixture
def runtime(sample_transformers_config):
    rt = TransformersRuntime(sample_transformers_config)
    rt.tokenizer = MagicMock()
    rt.model = MagicMock()
    return rt


@pytest.mark.unit
class TestSamplingKwargs:
    """`temperature: 0` is valid per the config schema and must mean greedy."""

    def test_temperature_zero_selects_greedy_decoding(self, runtime):
        kwargs = runtime._sampling_kwargs(GenerationParams(temperature=0.0))

        assert kwargs["do_sample"] is False
        # transformers raises if temperature=0 is passed as a sampling temperature,
        # and warns if top_p/top_k accompany do_sample=False.
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "top_k" not in kwargs

    def test_positive_temperature_samples(self, runtime):
        kwargs = runtime._sampling_kwargs(GenerationParams(temperature=0.8, top_p=0.9, top_k=40))

        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 40

    def test_repetition_penalty_and_max_tokens_always_passed(self, runtime):
        kwargs = runtime._sampling_kwargs(GenerationParams(max_tokens=64, repetition_penalty=1.2))

        assert kwargs["max_new_tokens"] == 64
        assert kwargs["repetition_penalty"] == 1.2

    def test_stop_sequences_are_forwarded_with_the_tokenizer(self, runtime):
        """Stop was previously dropped entirely by this runtime."""
        kwargs = runtime._sampling_kwargs(GenerationParams(stop=["</s>", "\n\n"]))

        assert kwargs["stop_strings"] == ["</s>", "\n\n"]
        # stop_strings requires a tokenizer to match multi-token sequences.
        assert kwargs["tokenizer"] is runtime.tokenizer

    def test_no_stop_means_no_stop_kwargs(self, runtime):
        kwargs = runtime._sampling_kwargs(GenerationParams(stop=[]))

        assert "stop_strings" not in kwargs
        assert "tokenizer" not in kwargs


@pytest.mark.unit
class TestStreamStopHandling:
    """The streamer emits the stop text, so the wrapper has to withhold it."""

    def test_stream_without_stop_passes_chunks_through(self, runtime):
        chunks = list(runtime._stream_generator(iter(["a", "b", "c"]), stop=None))

        assert chunks == ["a", "b", "c"]

    def test_stream_stops_before_the_stop_sequence(self, runtime):
        chunks = list(
            runtime._stream_generator(iter(["hello", " wor", "ld</s>", "more"]), ["</s>"])
        )

        assert "".join(chunks) == "hello world"
        assert "</s>" not in "".join(chunks)

    def test_stop_sequence_split_across_chunks_is_still_caught(self, runtime):
        """A per-chunk substring check would miss this."""
        chunks = list(runtime._stream_generator(iter(["done", "<", "/s", ">", "tail"]), ["</s>"]))

        assert "".join(chunks) == "done"

    def test_all_text_is_emitted_when_no_stop_occurs(self, runtime):
        chunks = list(runtime._stream_generator(iter(["abc", "def"]), ["</s>"]))

        assert "".join(chunks) == "abcdef"
