import os
import sys
from pathlib import Path

import click

from ..api import start_server
from ..config.loader import ConfigLoader
from ..config.models import (
    DeviceType,
    FormatType,
    ModelConfig,
    RuntimeConfig,
    RuntimeType,
    SLMConfig,
)
from ..evaluation import Benchmarker
from ..quantization import Quantizer
from ..registry import ModelRegistry
from ..registry.downloader import ModelDownloader
from ..runtime import get_runtime


@click.group()
def cli():
    """SLM Packager CLI"""
    pass


@cli.command()
@click.option("--name", prompt="Model Name", help="Name of the model")
@click.option("--path", prompt="Model Path", help="Path to the model file")
@click.option(
    "--format",
    type=click.Choice(["gguf", "onnx", "pytorch"]),
    prompt="Model Format",
    help="Model format",
)
@click.option(
    "--runtime",
    type=click.Choice(["llama_cpp", "onnx", "transformers"]),
    prompt="Runtime",
    help="Runtime to use",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cpu",
    show_default=True,
    help="Device to target",
)
@click.option("-o", "--output", default="slm.yaml", help="Output config file")
def init(name, path, format, runtime, device, output):
    """Initialize a new SLM config"""
    try:
        config = SLMConfig(
            model=ModelConfig(name=name, path=path, format=format),
            runtime=RuntimeConfig(type=runtime, device=device),
        )
        ConfigLoader.save(config, output)
        click.echo(f"✅ Config saved to {output}")

    except Exception as e:
        click.echo(f"\n❌ Error creating initialization config:", err=True)
        click.echo(f"   {str(e)}", err=True)
        click.echo(f"\n💡 Check your inputs and try again", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model_or_config")
@click.option("--prompt", "-p", help="Prompt to generate from")
@click.option("--stream/--no-stream", default=True, help="Stream output")
@click.option("--raw", is_flag=True, help="Disable auto-chat formatting")
def run(model_or_config, prompt, stream, raw):
    """Run a model from a config file or by name"""
    try:
        config_path = _resolve_config_path(model_or_config)
        config = ConfigLoader.load(config_path)

        # Override stream param if provided
        config.params.stream = stream

        click.echo(f"Loading model {config.model.name} with {config.runtime.type}...")

        # Get and load runtime
        runtime = get_runtime(config)
        runtime.load()

        # Get prompt if not provided
        if not prompt:
            prompt = click.prompt("Enter prompt")

        # Auto-apply the model's chat format. Instruction-tuned models answer a bare
        # prompt with an immediate end-of-sequence, so without this a chat GGUF
        # often returns nothing at all.
        if not raw:
            formatted = runtime.apply_chat_template(prompt)
            if formatted:
                prompt = formatted
                click.echo("ℹ️  Auto-formatting prompt with chat template (disable with --raw)")

        click.echo("-" * 20)

        # Generate
        if stream:
            for chunk in runtime.generate(prompt, config.params):
                click.echo(chunk, nl=False)
            click.echo()
        else:
            output = runtime.generate(prompt, config.params)
            click.echo(output)

        # Cleanup
        runtime.unload()

    except KeyboardInterrupt:
        click.echo("\n\n⚠️  Interrupted by user (Ctrl+C)", err=True)
        sys.exit(130)
    except (FileNotFoundError, ImportError, ValueError, RuntimeError, MemoryError) as e:
        click.echo(f"\n{str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Unexpected error running model:", err=True)
        click.echo(f"   {type(e).__name__}: {str(e)}", err=True)
        click.echo(f"\n💡 If this persists, please report it as a bug with:", err=True)
        click.echo(f"   - Your config file", err=True)
        click.echo(f"   - The command you ran", err=True)
        click.echo(f"   - Python version: {sys.version}", err=True)
        sys.exit(1)


def _resolve_config_path(model_or_config: str) -> Path:
    """Resolve either a direct config path or an installed model name."""
    input_path = Path(model_or_config)
    if input_path.exists():
        if input_path.suffix not in {".yaml", ".yml", ".json"}:
            raise ValueError(
                f"'{model_or_config}' is not a config file.\n"
                "💡 Pass a config path or an installed model name.\n"
                "   - Use an installed model: slm run tinyllama\n"
                "   - Or create a config: slm init --output my-model.yaml"
            )
        return input_path

    config_dir = Path.home() / ".slm" / "configs"
    potential_configs = [
        config_dir / f"{model_or_config}.yaml",
        config_dir / f"{model_or_config}.yml",
        config_dir / f"{model_or_config}.json",
    ]
    for potential_config in potential_configs:
        if potential_config.exists():
            return potential_config

    raise FileNotFoundError(
        f"Model or config not found: '{model_or_config}'\n"
        f"Tried:\n"
        f"   - Direct path: {input_path}\n"
        + "\n".join(f"   - Model config: {path}" for path in potential_configs)
        + "\n"
        "💡 Pull a model first: slm pull tinyllama\n"
        "   Then run it:        slm run tinyllama"
    )


@cli.command()
@click.argument("model_or_config")
@click.option("--prompt", "-p", default=None, help="Prompt to benchmark with")
@click.option("--runs", default=3, show_default=True, help="Timed runs to median over")
@click.option("--max-tokens", default=128, show_default=True, help="Tokens to generate per run")
@click.option("--warmup/--no-warmup", default=True, help="Discard an initial warmup run")
def benchmark(model_or_config, prompt, runs, max_tokens, warmup):
    """Benchmark a model"""
    try:
        config_path = _resolve_config_path(model_or_config)
        config = ConfigLoader.load(config_path)

        click.echo(f"Benchmarking {config.model.name}...")

        benchmarker = Benchmarker(config)
        kwargs = {"runs": runs, "max_tokens": max_tokens, "warmup": warmup}
        if prompt:
            kwargs["prompt"] = prompt
        metrics = benchmarker.run(**kwargs)

        click.echo(f"\n📊 Benchmark Results:")
        click.echo(f"   Load Time: {metrics['load_time_sec']:.2f}s")
        click.echo(f"   Memory Usage (process RSS): {metrics['memory_mb']:.2f} MB")
        click.echo(
            f"   Tokens Generated: {metrics['tokens_generated']}" f" over {metrics['runs']} run(s)"
        )
        click.echo(f"   Generation Time (mean): {metrics['generation_time_sec']:.2f}s")
        click.echo(f"   Tokens/sec (median): {metrics['tokens_per_second']:.2f}")
        click.echo(f"   Time per Token: {metrics['ms_per_token']:.2f} ms")
        if metrics["token_count_method"] == "estimate":
            click.echo("   ⚠️  Token count estimated (no tokenizer available)")

    except FileNotFoundError as e:
        click.echo(f"\n{str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Error during benchmarking:", err=True)
        click.echo(f"   {str(e)}", err=True)
        click.echo(f"\n💡 Try:", err=True)
        click.echo(f"   - Checking your config file is valid", err=True)
        click.echo(f"   - Ensuring the model loads correctly with 'slm run'", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_path")
@click.argument("output_path", required=False)
@click.option("--type", "quant_type", default="q4_k_m", help="Quantization type (q4_k_m, int8)")
def quantize(input_path, output_path, quant_type):
    """Quantize a model"""
    try:
        model_path = input_path
        source = Path(model_path)

        if not source.exists():
            click.echo(f"❌ Model file not found: '{model_path}'", err=True)
            click.echo(f"💡 Provide the full path to the model file", err=True)
            sys.exit(1)

        if source.suffix == ".gguf":
            output_path = output_path or str(source.with_name(f"{source.stem}-{quant_type}.gguf"))
            click.echo(f"Quantizing GGUF model to {quant_type}...")
            Quantizer.quantize_gguf(model_path, output_path, quant_type)
        elif source.suffix == ".onnx":
            output_path = output_path or str(source.with_name(f"{source.stem}-{quant_type}.onnx"))
            click.echo(f"Quantizing ONNX model to {quant_type}...")
            Quantizer.quantize_onnx(model_path, output_path, quant_type)
        else:
            click.echo(f"❌ Unsupported file extension: '{Path(model_path).suffix}'", err=True)
            click.echo(f"💡 Only .gguf and .onnx are supported", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"\n❌ Error during quantization:", err=True)
        click.echo(f"   {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def serve(host, port):
    """Start the API server"""
    try:
        click.echo(f"🚀 Starting API server on {host}:{port}")
        click.echo(f"   Press Ctrl+C to stop")
        start_server(host, port)
    except KeyboardInterrupt:
        click.echo(f"\n\n⚠️  Server stopped by user (Ctrl+C)")
        sys.exit(0)
    except OSError as e:
        if "Address already in use" in str(e):
            click.echo(f"\n❌ Port {port} is already in use", err=True)
            click.echo(f"💡 Try:", err=True)
            click.echo(f"   - Using a different port: slm serve --port 8001", err=True)
            click.echo(f"   - Finding and stopping the other process on port {port}", err=True)
        else:
            click.echo(f"\n❌ Error starting server:", err=True)
            click.echo(f"   {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Error starting server:", err=True)
        click.echo(f"   {type(e).__name__}: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model_name")
@click.argument("filename", required=False, default=None)
@click.option(
    "--quant", "--quantization", default=None, help="Quantization type (q4_k_m, q8_0, etc.)"
)
@click.option("--list-variants", is_flag=True, help="List available variants for this model")
@click.option("--name", default=None, help="Local alias for the model (used with HF repo pull)")
def pull(model_name, filename, quant, list_variants, name):
    """Pull a model from the registry or directly from HuggingFace.

    Registry pull:   slm pull tinyllama
    Direct HF pull:  slm pull TheBloke/Mistral-7B-GGUF mistral-7b-v0.1.Q4_K_M.gguf
    """
    try:
        downloader = ModelDownloader()

        # Direct HF repo pull — model_name contains a slash (e.g. "TheBloke/Mistral-7B-GGUF")
        if "/" in model_name:
            if not filename:
                click.echo(
                    "❌ A filename is required when pulling directly from a HuggingFace repo.",
                    err=True,
                )
                click.echo("💡 Usage: slm pull <repo-id> <filename>", err=True)
                click.echo(
                    "   Example: slm pull TheBloke/Mistral-7B-GGUF mistral-7b-v0.1.Q4_K_M.gguf",
                    err=True,
                )
                sys.exit(1)
            downloader.pull_from_repo(model_name, filename, name=name)
            return

        registry = ModelRegistry()

        # List variants if requested
        if list_variants:
            model = registry.get_model(model_name)
            if not model:
                click.echo(f"❌ Model '{model_name}' not found", err=True)
                click.echo(f"💡 See available models with: slm list", err=True)
                sys.exit(1)

            click.echo(f"\nAvailable variants for {model.name}:")
            for variant_name, variant in model.variants.items():
                recommended = " ⭐" if variant.recommended else ""
                click.echo(f"  • {variant_name} ({variant.size}){recommended}")
                click.echo(f"    Speed: {variant.speed}, Quality: {variant.quality}")
            sys.exit(0)

        # Registry pull
        downloader.pull(model_name, quant)

    except ValueError as e:
        click.echo(f"\n{str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Error pulling model:", err=True)
        click.echo(f"   {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model_name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def rm(model_name, yes):
    """Remove an installed model and its config"""
    try:
        downloader = ModelDownloader()

        # Check if model exists
        config_path = downloader.configs_dir / f"{model_name}.yaml"
        if not config_path.exists():
            click.echo(f"❌ Model '{model_name}' is not installed", err=True)
            click.echo("💡 See installed models with: slm list --installed", err=True)
            sys.exit(1)

        # Confirm deletion
        if not yes:
            if not click.confirm(f"Delete model '{model_name}' and its files?"):
                click.echo("Cancelled.")
                sys.exit(0)

        deleted = downloader.delete(model_name)
        if deleted:
            click.echo(f"✅ Model '{model_name}' removed")
        else:
            click.echo(f"❌ Failed to remove '{model_name}'", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"\n❌ Error removing model:", err=True)
        click.echo(f"   {str(e)}", err=True)
        sys.exit(1)


@cli.command("list")
@click.option("--installed", is_flag=True, help="Show only installed models")
def list_models(installed):
    """List available or installed models"""
    try:
        if installed:
            # List installed models
            downloader = ModelDownloader()
            models = downloader.list_installed()

            if not models:
                click.echo("\nNo models installed yet.")
                click.echo("💡 Pull a model with: slm pull tinyllama\n")
                sys.exit(0)

            click.echo("\n📦 Installed models:\n")
            for model in models:
                click.echo(f"  • {model['name']} ({model['size']})")
                click.echo(f"    Format: {model['format']}")
                click.echo(f"    Path: {model['path']}")
                click.echo()
        else:
            # List available models from registry
            registry = ModelRegistry()
            models = registry.get_all_models()

            click.echo("\n📋 Available models in registry:\n")
            for name, model in models.items():
                recommended = registry.get_recommended_variant(name)
                click.echo(f"  • {name} - {model.name}")
                click.echo(f"    {model.description}")
                click.echo(f"    Format: {model.format}, Runtime: {model.runtime}")
                click.echo(f"    Recommended: {recommended}")
                click.echo()

            click.echo("💡 Pull a model with: slm pull <model-name>")
            click.echo("💡 List variants with: slm pull <model-name> --list-variants\n")

    except Exception as e:
        click.echo(f"\n❌ Error listing models:", err=True)
        click.echo(f"   {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
