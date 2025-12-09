"""Binary manager for auto-downloading quantization tools"""
import platform
import urllib.request
from pathlib import Path
from typing import Optional

class BinaryManager:
    """Manages downloading and caching of quantization binaries"""
    
    # llama.cpp releases with quantize binary
    BINARY_URLS = {
        "Darwin": {
            "arm64": "https://github.com/ggerganov/llama.cpp/releases/download/b3683/llama-b3683-bin-macos-arm64.zip",
            "x86_64": "https://github.com/ggerganov/llama.cpp/releases/download/b3683/llama-b3683-bin-macos-x64.zip"
        },
        "Linux": "https://github.com/ggerganov/llama.cpp/releases/download/b3683/llama-b3683-bin-ubuntu-x64.zip",
        "Windows": "https://github.com/ggerganov/llama.cpp/releases/download/b3683/llama-b3683-bin-win-avx2-x64.zip"
    }
    
    @classmethod
    def get_quantize_binary(cls) -> Path:
        """
        Get quantize binary, downloading if needed (Terraform-style)
        
        Returns:
            Path to quantize binary
        """
        binary_name = "quantize.exe" if platform.system() == "Windows" else "quantize"
        binary_path = Path.home() / ".slm" / "bin" / binary_name
        
        if binary_path.exists():
            return binary_path
        
        # First time - need to download
        print(f"\n📥 First-time setup: Downloading quantization tool...")
        print(f"   (This only happens once, then it's cached)\n")
        
        try:
            binary_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download and extract
            cls._download_and_extract(binary_path)
            
            # Make executable on Unix
            if platform.system() != "Windows":
                binary_path.chmod(0o755)
            
            print(f"\n✅ Quantization tool ready!\n")
            return binary_path
            
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to download quantization tool\n"
                f"   {str(e)}\n"
                "💡 Alternatively:\n"
                "   - Download pre-quantized models: slm pull tinyllama --quant q4_k_m\n"
                "   - Install llama.cpp manually and add to PATH"
            ) from e
    
    @classmethod
    def _download_and_extract(cls, binary_path: Path):
        """Download and extract binary from llama.cpp releases"""
        import zipfile
        import tempfile
        
        url = cls._get_url()
        
        # Download to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
            print(f"   Downloading from: {url}")
            urllib.request.urlretrieve(url, tmp.name, reporthook=cls._show_progress)
            zip_path = Path(tmp.name)
        
        # Extract quantize binary
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Find quantize binary in zip
                quantize_name = "quantize.exe" if platform.system() == "Windows" else "quantize"
                
                for member in zip_ref.namelist():
                    if member.endswith(quantize_name):
                        # Extract just this file
                        zip_ref.extract(member, binary_path.parent)
                        
                        # Move to expected location
                        extracted = binary_path.parent / member
                        extracted.replace(binary_path)
                        break
                else:
                    raise FileNotFoundError(f"quantize binary not found in {url}")
        finally:
            zip_path.unlink()  # Clean up zip file
    
    @classmethod
    def _get_url(cls) -> str:
        """Get download URL for current platform"""
        system = platform.system()
        
        if system == "Darwin":
            arch = platform.machine()
            return cls.BINARY_URLS["Darwin"][arch]
        
        if system not in cls.BINARY_URLS:
            raise OSError(
                f"❌ Unsupported platform: {system}\n"
                "💡 Supported: macOS, Linux, Windows"
            )
        
        return cls.BINARY_URLS[system]
    
    @classmethod
    def _show_progress(cls, block_num, block_size, total_size):
        """Show download progress"""
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
        
        if block_num % 50 == 0:  # Update every 50 blocks
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"   Downloaded: {mb_downloaded:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)", end='\r')
