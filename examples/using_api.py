"""
Example: Building an App on top of the SLM API

This demonstrates how a separate service/app (Frontend, Slack Bot, etc.) 
interacts with the `slm serve` infrastructure.
"""
import requests
import json
import sys

API_URL = "http://localhost:8006"
MODEL_CONFIG = "/Users/admin/.slm/configs/tinyllama.yaml"

def main():
    # 1. Ensure the "Infrastructure" is ready
    print(f"🔌 Connecting to SLM Engine at {API_URL}...")
    try:
        health = requests.get(f"{API_URL}/health")
        if health.status_code != 200:
            print("❌ Server is offline. Run 'slm serve' first.")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect. Run 'slm serve --port 8006' in another terminal.")
        return

    # 2. Load the model via API
    # (The server manages the heavy model in memory, not this script)
    print("📥 Requesting model load...")
    requests.post(f"{API_URL}/load", json={"config_path": MODEL_CONFIG})

    # 3. Send a Generation Request
    prompt = "Explain what an API is in one sentence."
    formatted_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>"
    
    print(f"\n📝 Prompt: {prompt}")
    print("🤖 Response: ", end="", flush=True)

    # 4. Handle Streaming Response
    response = requests.post(
        f"{API_URL}/generate", 
        json={
            "prompt": formatted_prompt,
            "params": {"stream": True, "max_tokens": 100}
        }, 
        stream=True
    )

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data: '):
                data_str = decoded_line.replace('data: ', '')
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if "text" in data:
                        print(data["text"], end="", flush=True)
                except:
                    pass
    
    print("\n\n✅ Done! The heavy lifting was done by the Server.")

if __name__ == "__main__":
    main()
