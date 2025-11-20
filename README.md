# SLM Packager

**A Unified Runtime & Developer Layer for Small Language Models**

SLM Packager is an open-source toolchain designed to solve the fragmentation in the Small Language Model (SLM) ecosystem. It provides a unified interface for running, packaging, and evaluating models across different formats (GGUF, ONNX, PyTorch) and runtimes.

Our goal is to make SLMs behave like modular plugins — easy to load, evaluate, switch, and ship.

## 🚀 Current Status (v0.1)

We have built the foundational layer of the project. Here's what works today:

*   **Unified Configuration**: A single YAML schema to define model parameters, runtime settings, and generation options.
*   **Multi-Runtime Support**:
    *   `llama.cpp`: Full support for GGUF models with quantization.
    *   `Transformers`: Support for PyTorch models (CPU/GPU).
    *   `ONNX Runtime`: Basic support for ONNX models.
*   **Developer API**: A FastAPI-based server providing `/load`, `/generate` (streaming supported), and `/info` endpoints.
*   **CLI Tool**: A robust CLI for initializing configs, running models, and benchmarking.
*   **Benchmarking**: Built-in tools to measure Load Time, Tokens/Sec, Latency, and Memory Usage.

## 🚧 Known Issues & Limitations

As this is an early alpha (v0.1), there are some rough edges we are working on:

*   **ONNX Generation**: The current ONNX adapter uses a simplified generation loop. It lacks a proper KV-cache implementation, making it slower than it should be.
*   **Quantization**: We currently wrap external tools (`quantize` binary for llama.cpp). A more integrated native python approach is desired.
*   **Tokenizer Handling**: Auto-detection of tokenizers for ONNX models is basic.
*   **Device Support**: While CUDA is supported in code, extensive testing across different hardware (MPS, ROCm) is still pending.

## 🗺️ Roadmap

We are actively looking for contributors to help us build the next set of features:

*   **vLLM Integration**: Add a high-performance GPU serving adapter using vLLM.
*   **Advanced Evaluation**: Add perplexity scoring and integration with `lm-evaluation-harness`.
*   **Model Registry**: A `slm pull` command to automatically download models from HuggingFace or other sources.
*   **Plugin System**: Allow users to write custom runtime adapters as Python plugins.
*   **Docker Support**: Auto-generate Dockerfiles for packaged models.

## 🤝 Contributing

We welcome contributions! Whether it's fixing a bug, adding a feature, or improving documentation, we'd love your help.

### How to Contribute

1.  **Fork the Repository**: Click the "Fork" button on the top right.
2.  **Clone your Fork**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/slm-packager.git
    cd slm-packager
    ```
3.  **Set up Environment**:
    ```bash
    pip install -e ".[dev]"
    ```
4.  **Create a Branch**: `git checkout -b feature/my-new-feature`
5.  **Make Changes**: Write your code and tests.
6.  **Run Tests**: Ensure everything is working.
    ```bash
    pytest
    ```
7.  **Submit a Pull Request**: Open a PR describing your changes.

### Coding Style

*   We use **Python 3.9+**.
*   We follow **PEP 8** guidelines.
*   We use **Pydantic** for data validation.
*   Please include docstrings for new functions and classes.

## Installation

```bash
pip install .
```

## Usage

### Initialize a Config

```bash
slm init
```

### Run a Model

```bash
slm run phi-3
```

### Benchmark

```bash
slm benchmark phi-3
```

### Serve API

```bash
slm serve --port 8000
```

## License

MIT
