import click
import os
from pathlib import Path
from ..config.models import SLMConfig, ModelConfig, RuntimeConfig, RuntimeType, DeviceType
from ..config.loader import ConfigLoader
from ..runtime import get_runtime
from ..api import start_server
from ..quantization import Quantizer
from ..evaluation import Benchmarker

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
    config = SLMConfig(
        model=ModelConfig(name=name, path=path, format=format),
        runtime=RuntimeConfig(type=runtime)
    )
    ConfigLoader.save(config, output)
    click.echo(f"Config saved to {output}")

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--prompt", "-p", help="Prompt to generate from")
@click.option("--stream/--no-stream", default=True, help="Stream output")
def run(config_path, prompt, stream):
    """Run a model from a config file"""
    config = ConfigLoader.load(config_path)
    
    # Override stream param if provided
    config.params.stream = stream
    
    click.echo(f"Loading model {config.model.name} with {config.runtime.type}...")
    runtime = get_runtime(config)
    runtime.load()
    
    if not prompt:
        prompt = click.prompt("Enter prompt")
        
    click.echo("-" * 20)
    if stream:
        for chunk in runtime.generate(prompt, config.params):
            click.echo(chunk, nl=False)
        click.echo()
    else:
        output = runtime.generate(prompt, config.params)
        click.echo(output)
    
    runtime.unload()

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def benchmark(config_path):
    """Benchmark a model"""
    config = ConfigLoader.load(config_path)
    
    click.echo(f"Benchmarking {config.model.name}...")
    
    benchmarker = Benchmarker(config)
    metrics = benchmarker.run()
    
    click.echo(f"Load Time: {metrics['load_time_sec']:.2f}s")
    click.echo(f"Generation Time: {metrics['generation_time_sec']:.2f}s")
    click.echo(f"Memory Usage: {metrics['memory_mb']:.2f} MB")
    click.echo(f"Latency: {metrics['latency_ms']:.2f} ms")
    click.echo(f"Estimated TPS: {metrics['tokens_per_second']:.2f}")

@cli.command()
@click.argument("model_name")
@click.option("--type", default="q4_k_m", help="Quantization type (q4_k_m, int8)")
def quantize(model_name, type):
    """Quantize a model"""
    # This is a simplified CLI that assumes model_name is a path for now
    # In a real app, we would look up the model in a registry
    model_path = model_name
    
    if model_path.endswith(".gguf"):
        output_path = model_path.replace(".gguf", f"-{type}.gguf")
        Quantizer.quantize_gguf(model_path, output_path, type)
    elif model_path.endswith(".onnx"):
        output_path = model_path.replace(".onnx", f"-{type}.onnx")
        Quantizer.quantize_onnx(model_path, output_path, type)
    else:
        click.echo("Unsupported file extension. Only .gguf and .onnx are supported.")

@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def serve(host, port):
    """Start the API server"""
    click.echo(f"Starting API server on {host}:{port}")
    start_server(host, port)

if __name__ == "__main__":
    cli()
