"""
Example: Using SLM Packager as a Python Library (SDK)

This demonstrates how a developer handles the low-level runtime directly 
in their own Python application.
"""
from pathlib import Path
from slm_packager.config.loader import ConfigLoader
from slm_packager.runtime import get_runtime

# 1. Point to a model config
# (In a real app, you might generate this or bundle it)
config_path = Path.home() / ".slm" / "configs" / "tinyllama.yaml"

if not config_path.exists():
    print("❌ Config not found. Run 'slm pull tinyllama' first.")
    exit(1)

# 2. Load Configuration
print(f"📄 Loading config from {config_path}...")
config = ConfigLoader.load(config_path)

# 3. Initialize Runtime (this is the "Engine")
# This handles all the GGUF/PyTorch/ONNX abstraction for you
print(f"🚀 Initializing {config.runtime.type} engine...")
runtime = get_runtime(config)
runtime.load()

# 4. Integrate into your logic
print("\n--- Your Custom App Logic ---")
user_input = "Write a python function to add two numbers."

# Apply a chat template if needed (your app's responsibility in SDK mode)
prompt = f"<|user|>\n{user_input}</s>\n<|assistant|>"

# 5. Generate Response
print(f"User: {user_input}")
print("Assistant: ", end="", flush=True)

# Stream the output token by token
output_buffer = ""
for chunk in runtime.generate(prompt, config.params):
    print(chunk, end="", flush=True)
    output_buffer += chunk

print("\n-----------------------------")

# 6. Cleanup
runtime.unload()
print("\n✅ Done!")
