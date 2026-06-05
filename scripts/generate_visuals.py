"""Generate Substack article visuals for SLM Packager."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "article_visuals"
OUTPUT_DIR.mkdir(exist_ok=True)

BG = "#0d1117"
SURFACE = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"
TEAL = "#39d353"


# ── 1. ARCHITECTURE DIAGRAM ──────────────────────────────────────────────────

def draw_architecture():
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, color, label, sublabel=None, fontsize=13, radius=0.3):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle=f"round,pad=0,rounding_size={radius}",
                              linewidth=1.5, edgecolor=color,
                              facecolor=SURFACE)
        ax.add_patch(rect)
        cy = y + h / 2 + (0.15 if sublabel else 0)
        ax.text(x + w / 2, cy, label, color=color,
                fontsize=fontsize, fontweight="bold",
                ha="center", va="center", fontfamily="monospace")
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.25, sublabel, color=MUTED,
                    fontsize=9, ha="center", va="center", fontfamily="monospace")

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.5, mutation_scale=14))

    # Title
    ax.text(5, 9.4, "SLM Packager — Architecture",
            color=TEXT, fontsize=15, fontweight="bold",
            ha="center", va="center", fontfamily="monospace")

    # Top layer — user entry points
    box(0.5, 7.8, 3.8, 0.9, BLUE, "CLI", "slm run / pull / serve", fontsize=12)
    box(5.7, 7.8, 3.8, 0.9, BLUE, "API Server", "FastAPI + streaming", fontsize=12)

    # Arrows down to runtime abstraction
    arrow(2.4, 7.8, 2.4, 7.1)
    arrow(7.6, 7.8, 7.6, 7.1)

    # Middle — abstraction layer
    box(0.5, 6.0, 9.0, 1.0, GREEN, "Runtime Abstraction Layer",
        "load()  ·  generate()  ·  unload()", fontsize=13)

    # Arrows down to runtimes
    arrow(2.4, 6.0, 2.4, 5.2)
    arrow(5.0, 6.0, 5.0, 5.2)
    arrow(7.6, 6.0, 7.6, 5.2)

    # Three runtimes
    box(0.5, 4.1, 3.0, 1.0, ORANGE, "llama.cpp", "GGUF · Metal · CUDA", fontsize=11)
    box(3.6, 4.1, 2.8, 1.0, PURPLE, "Transformers", "PyTorch · MPS · CUDA", fontsize=11)
    box(6.6, 4.1, 2.9, 1.0, TEAL, "ONNX Runtime", "KV-cache · CPU opt.", fontsize=11)

    # Arrows to model formats
    arrow(2.0, 4.1, 2.0, 3.3)
    arrow(5.0, 4.1, 5.0, 3.3)
    arrow(8.0, 4.1, 8.0, 3.3)

    # Model formats
    box(0.5, 2.2, 3.0, 1.0, ORANGE, ".gguf", "Quantized weights", fontsize=11)
    box(3.6, 2.2, 2.8, 1.0, PURPLE, "HuggingFace", "PyTorch weights", fontsize=11)
    box(6.6, 2.2, 2.9, 1.0, TEAL, ".onnx", "Optimized graph", fontsize=11)

    # Bottom — config + registry
    box(1.5, 0.5, 3.2, 1.1, MUTED, "YAML Config", "slm.yaml · reproducible", fontsize=11)
    box(5.3, 0.5, 3.2, 1.1, MUTED, "Model Registry", "slm pull · HF hub", fontsize=11)

    plt.tight_layout()
    out = OUTPUT_DIR / "architecture.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"✅ Saved: {out}")


# ── 2. BENCHMARK BAR CHART ───────────────────────────────────────────────────

def draw_benchmark():
    labels = [
        "GPT-2\ntransformers CPU",
        "GPT-2\ntransformers MPS ⚡",
        "TinyLlama 1.1B\nllama.cpp CPU",
        "Phi-2 2.7B\nllama.cpp CPU",
        "Qwen3 4B\nllama.cpp CPU",
    ]
    values = [53.16, 28.06, 9.19, 33.67, 31.71]
    colors = [PURPLE, BLUE, TEAL, ORANGE, GREEN]
    highlights = [True, True, True, True, True]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.barh(labels, values, color=colors, height=0.5,
                   edgecolor=BG, linewidth=1.5)

    # Value labels
    for bar, val, hl in zip(bars, values, highlights):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val} tok/s", color=TEXT if hl else MUTED,
                va="center", fontsize=12,
                fontweight="bold" if hl else "normal",
                fontfamily="monospace")


    ax.set_xlabel("Tokens per second  (higher is better)",
                  color=MUTED, fontsize=11, fontfamily="monospace")
    ax.set_title("SLM Packager — Runtime Performance\nM3 Pro · 18GB · Real Benchmarks",
                 color=TEXT, fontsize=14, fontweight="bold",
                 fontfamily="monospace", pad=16)

    ax.tick_params(colors=TEXT, labelsize=11)
    ax.spines[:].set_visible(False)
    ax.xaxis.label.set_color(MUTED)
    for label in ax.get_yticklabels():
        label.set_fontfamily("monospace")
        label.set_color(TEXT)
    for label in ax.get_xticklabels():
        label.set_fontfamily("monospace")
        label.set_color(MUTED)

    ax.set_xlim(0, 70)
    ax.grid(axis="x", color=BORDER, linewidth=0.8, alpha=0.5)

    # Multiplier callouts
    multipliers = [None, None, None, None, None]
    for bar, mult in zip(bars, multipliers):
        if mult:
            ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    mult, color=BG, va="center", ha="center",
                    fontsize=10, fontweight="bold", fontfamily="monospace")

    plt.tight_layout()
    out = OUTPUT_DIR / "benchmark.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"✅ Saved: {out}")


if __name__ == "__main__":
    draw_architecture()
    draw_benchmark()
    print(f"\n📁 Both images saved to: {OUTPUT_DIR}/")
