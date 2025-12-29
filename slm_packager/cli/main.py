import click
import os
import sys
from pathlib import Path
from ..config.models import SLMConfig, ModelConfig, RuntimeConfig, RuntimeType, DeviceType
from ..config.loader import ConfigLoader
from ..runtime import get_runtime
from ..api import start_server
from ..quantization import Quantizer
from ..evaluation import Benchmarker
from ..registry.downloader import ModelDownloader
from ..registry import ModelRegistry

@click.group()
def cli():
    """SLM Packager CLI"""
    pass

@cli.command()
@click.option("--name", prompt="Model Name", help="Name of the model")
@click.option("--path", prompt="Model Path", help="Path to the model file")
@click.option("--format", type=click.Choice(["gguf", "onnx", "pytorch"]), prompt="Model Format", help="Model format")
@click.option("--runtime", type=click.Choice(["llama_cpp", "onnx", "transformers"]), prompt="Runtime", help="Runtime to use")
@click.option("--output", default="slm.yaml", help="Output config file")
def init(name, path, format, runtime, output):
    """Initialize a new SLM config"""
    try:
        config = SLMConfig(
            model=ModelConfig(name=name, path=path, format=format),
            runtime=RuntimeConfig(type=runtime)
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
def run(model_or_config, prompt, stream):
    """Run a model from a config file or by name"""
    try:
        # Check if input is a model name or config path
        input_path = Path(model_or_config)
        
        # Try to resolve as config path first
        if input_path.exists():
            config_path = input_path
        else:
            # Try to resolve as model name from ~/.slm/configs/
            config_dir = Path.home() / ".slm" / "configs"
            potential_config = config_dir / f"{model_or_config}.yaml"
            
            if potential_config.exists():
                config_path = potential_config
            else:
                click.echo(f"\n❌ Model or config not found: '{model_or_config}'", err=True)
                click.echo(f"\nTried:", err=True)
                click.echo(f"   - Direct path: {input_path}", err=True)
                click.echo(f"   - Model config: {potential_config}", err=True)
                click.echo(f"\n💡 Suggestions:", err=True)
                click.echo(f"   - List installed models: slm list --installed", err=True)
                click.echo(f"   - Pull a model: slm pull gpt2", err=True)
                sys.exit(1)
        
        # Load config
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
        
    except FileNotFoundError as e:
        click.echo(f"\n{str(e)}", err=True)
        sys.exit(1)
    except ImportError as e:
        click.echo(f"\n{str(e)}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"\n{str(e)}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        click.echo(f"\n{str(e)}", err=True)
        sys.exit(1)
    except MemoryError as e:
        click.echo(f"\n{str(e)}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo(f"\n\n⚠️  Interrupted by user (Ctrl+C)", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"\n❌ Unexpected error running model:", err=True)
        click.echo(f"   {type(e).__name__}: {str(e)}", err=True)
        click.echo(f"\n💡 If this persists, please report it as a bug with:", err=True)
        click.echo(f"   - Your config file", err=True)
        click.echo(f"   - The command you ran", err=True)
        click.echo(f"   - Python version: {sys.version}", err=True)
        sys.exit(1)

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def benchmark(config_path):
    """Benchmark a model"""
    try:
        config = ConfigLoader.load(config_path)
        
        click.echo(f"Benchmarking {config.model.name}...")
        
        benchmarker = Benchmarker(config)
        metrics = benchmarker.run()
        
        click.echo(f"\n📊 Benchmark Results:")
        click.echo(f"   Load Time: {metrics['load_time_sec']:.2f}s")
        click.echo(f"   Generation Time: {metrics['generation_time_sec']:.2f}s")
        click.echo(f"   Memory Usage: {metrics['memory_mb']:.2f} MB")
        click.echo(f"   Latency: {metrics['latency_ms']:.2f} ms")
        click.echo(f"   Estimated TPS: {metrics['tokens_per_second']:.2f}")
        
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
@click.argument("model_name")
@click.option("--type", default="q4_k_m", help="Quantization type (q4_k_m, int8)")
def quantize(model_name, type):
    """Quantize a model"""
    try:
        # This is a simplified CLI that assumes model_name is a path for now
        # In a real app, we would look up the model in a registry
        model_path = model_name
        
        if not Path(model_path).exists():
            click.echo(f"❌ Model file not found: '{model_path}'", err=True)
            click.echo(f"💡 Provide the full path to the model file", err=True)
            sys.exit(1)
        
        if model_path.endswith(".gguf"):
            output_path = model_path.replace(".gguf", f"-{type}.gguf")
            click.echo(f"Quantizing GGUF model to {type}...")
            Quantizer.quantize_gguf(model_path, output_path, type)
        elif model_path.endswith(".onnx"):
            output_path = model_path.replace(".onnx", f"-{type}.onnx")
            click.echo(f"Quantizing ONNX model to {type}...")
            Quantizer.quantize_onnx(model_path, output_path, type)
        else:
            click.echo(f"❌ Unsupported file extension: '{Path(model_path).suffix}'", err=True)
            click.echo(f"💡 Only .gguf and .onnx are supported", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"\n❌ Error during quantization:", err=True)
        click.echo(f"   {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
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
@click.option("--quant", "--quantization", default=None, help="Quantization type (q4_k_m, q8_0, etc.)")
@click.option("--list-variants", is_flag=True, help="List available variants for this model")
def pull(model_name, quant, list_variants):
    """Pull a model from the registry"""
    try:
        downloader = ModelDownloader()
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
        
        # Pull model
        downloader.pull(model_name, quant)
        
    except ValueError as e:
        click.echo(f"\n{str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Error pulling model:", err=True)
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

