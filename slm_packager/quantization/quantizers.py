import subprocess
import os
from pathlib import Path
from typing import Optional

from .binary_manager import BinaryManager

class Quantizer:
    @staticmethod
    def quantize_gguf(model_path: str, output_path: str, type: str = "q4_k_m"):
        """
        Quantize a GGUF model - downloads tool automatically if needed
        """
        try:
            # Auto-download binary (Terraform-style!)
            binary = BinaryManager.get_quantize_binary()
            
            print(f"📦 Quantizing {Path(model_path).name} to {type}...")
            print(f"   Output: {output_path}\n")
            
            cmd = [str(binary), model_path, output_path, type]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            print(result.stdout)
            print(f"\n✅ Quantized successfully: {output_path}")
            print(f"   You can now use: slm run {output_path}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"❌ Quantization failed\n"
                f"   {e.stderr}\n"
                "💡 Check that:\n"
                "   - Input model file exists\n"
                "   - You have enough disk space\n"
                "   - Quantization type is valid (q4_0, q4_k_m, q5_k_m, q8_0)"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"❌ Unexpected error during quantization\n"
                f"   {type(e).__name__}: {str(e)}\n"
                "💡 Alternatives:\n"
                "   - Download pre-quantized models: slm pull tinyllama --quant q4_k_m"
            ) from e

    @staticmethod
    def quantize_onnx(model_path: str, output_path: str, type: str = "int8"):
        """
        Quantize an ONNX model using onnxruntime.quantization.
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            raise ImportError(
                "❌ ONNX quantization requires 'onnxruntime'\n"
                "💡 Install with: pip install onnxruntime"
            )

        print(f"📦 Quantizing ONNX model to {type}...")
        
        quant_type = QuantType.QUInt8 if type == "int8" else QuantType.QInt8
        
        try:
            quantize_dynamic(
                model_input=Path(model_path),
                model_output=Path(output_path),
                weight_type=quant_type
            )
            print(f"✅ Successfully quantized to {output_path}")
        except Exception as e:
            raise RuntimeError(
                f"❌ ONNX quantization failed\n"
                f"   {str(e)}"
            ) from e

