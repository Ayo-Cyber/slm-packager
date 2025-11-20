import subprocess
import os
from pathlib import Path
from typing import Optional

class Quantizer:
    @staticmethod
    def quantize_gguf(model_path: str, output_path: str, type: str = "q4_k_m"):
        """
        Quantize a GGUF model using llama.cpp's quantize tool.
        Assumes 'quantize' binary is in the PATH or accessible.
        """
        # In a real implementation, we might bundle the binary or find it.
        # For now, we assume the user has installed llama.cpp tools.
        cmd = ["quantize", model_path, output_path, type]
        
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully quantized to {output_path}")
        except FileNotFoundError:
            print("Error: 'quantize' command not found. Please install llama.cpp tools.")
        except subprocess.CalledProcessError as e:
            print(f"Quantization failed: {e}")

    @staticmethod
    def quantize_onnx(model_path: str, output_path: str, type: str = "int8"):
        """
        Quantize an ONNX model using onnxruntime.quantization.
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            print("Error: onnxruntime is not installed.")
            return

        print(f"Quantizing {model_path} to {output_path} with {type}...")
        
        quant_type = QuantType.QUInt8 if type == "int8" else QuantType.QInt8
        
        quantize_dynamic(
            model_input=Path(model_path),
            model_output=Path(output_path),
            weight_type=quant_type
        )
        print(f"Successfully quantized to {output_path}")
