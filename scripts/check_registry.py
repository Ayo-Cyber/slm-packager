#!/usr/bin/env python3
"""Verify every model in the registry still resolves on HuggingFace.

Model repos get renamed, gated, or retired (HF answers 401 for a gone repo, not
404), which silently breaks `slm pull <model>`. Run in CI on a schedule so the
registry can't rot unnoticed.

Usage: python scripts/check_registry.py
Exits non-zero if any variant is unreachable.
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "slm_packager" / "registry" / "models.json"
TIMEOUT = 30


def check(url: str) -> str:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=TIMEOUT
        ) as r:
            return str(r.status)
    except urllib.error.HTTPError as e:
        return str(e.code)
    except Exception as e:  # network/DNS/timeout
        return type(e).__name__


def main() -> int:
    data = json.loads(REGISTRY.read_text())
    broken = []

    for name, model in data["models"].items():
        repo = model.get("repo") or model.get("path")

        if model.get("format") != "gguf":
            # PyTorch/ONNX entries reference a repo, not a specific file.
            status = check(f"https://huggingface.co/api/models/{repo}")
            print(f"{status:>6}  {name}  ({repo})")
            if status != "200":
                broken.append(f"{name} -> {repo}")
            continue

        for quant, variant in model["variants"].items():
            url = f"https://huggingface.co/{repo}/resolve/main/{variant['file']}"
            status = check(url)
            print(f"{status:>6}  {name}:{quant}")
            if status != "200":
                broken.append(f"{name}:{quant} -> {url}")

    print()
    if broken:
        print(f"{len(broken)} unreachable registry target(s):")
        for entry in broken:
            print(f"  - {entry}")
        return 1

    print("All registry targets resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
