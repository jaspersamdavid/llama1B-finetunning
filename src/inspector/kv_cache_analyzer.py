"""
KV Cache Analyzer — instrument the cache during single-token generation.

Tracks per-step:
  - cache size (tokens × layers × kv_heads × head_dim)
  - memory in MB
  - per-layer K and V vector norms for the latest cached token

Compares with-cache vs without-cache generation speed.
Visualises cache growth and per-layer K-norm heatmap.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

OUTPUTS_DIR = Path("outputs/kv_cache")


# ---------------------------------------------------------------------------
# Memory math
# ---------------------------------------------------------------------------

def cache_memory_bytes(seq_len: int, num_layers: int, num_kv_heads: int,
                       head_dim: int, bytes_per_value: int = 2) -> int:
    """Bytes used by the KV cache for `seq_len` cached tokens."""
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_value


def print_memory_breakdown(model) -> None:
    """Show the formula plus a table of cache size across sequence lengths."""
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    per_token = 2 * num_layers * num_kv_heads * head_dim * 2

    print(f"\n{'='*72}")
    print(f"  KV CACHE MEMORY FORMULA — Llama 3.2 1B")
    print(f"{'='*72}")
    print(f"  KV bytes = 2 (K+V) × {num_layers} (layers) × {num_kv_heads} (kv_heads) "
          f"× {head_dim} (head_dim) × seq_len × 2 (FP16 bytes)")
    print(f"           = {per_token:,} bytes per cached token")
    print(f"           = {per_token/1024:.1f} KB per token\n")

    print(f"  Cache size at various sequence lengths:")
    for seq_len in [10, 50, 100, 250, 500, 1000, 2000, 5000]:
        b = cache_memory_bytes(seq_len, num_layers, num_kv_heads, head_dim)
        print(f"    {seq_len:>5} tokens  →  {b/(1024*1024):>7.2f} MB")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Generation profiling
# ---------------------------------------------------------------------------

def _sync(device) -> None:
    """Block until all queued ops on device are done — needed for honest timing on MPS."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def profile_generation_with_cache(model, tokenizer, prompt: str,
                                  max_new_tokens: int = 50) -> dict:
    """Manual greedy generation loop with use_cache=True. Records per-step state."""
    device = next(model.parameters()).device
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    past_key_values = None
    steps = []
    generated_ids = []

    _sync(device)
    total_start = time.perf_counter()

    with torch.no_grad():
        for step in range(max_new_tokens):
            _sync(device)
            t0 = time.perf_counter()

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            _sync(device)
            t_step = time.perf_counter() - t0

            past_key_values = out.past_key_values
            next_id = out.logits[0, -1, :].argmax()
            generated_ids.append(next_id.item())

            # transformers 5.x: DynamicCache exposes .layers[i].keys / .values
            cache_layers = past_key_values.layers
            seq_len = cache_layers[0].keys.shape[-2]   # [batch, kv_heads, seq, head_dim]

            k_norms = []
            v_norms = []
            for layer_idx in range(num_layers):
                k_layer = cache_layers[layer_idx].keys
                v_layer = cache_layers[layer_idx].values
                k_latest = k_layer[0, :, -1, :]   # latest token, all kv_heads
                v_latest = v_layer[0, :, -1, :]
                k_norms.append(k_latest.norm(dim=-1).mean().item())
                v_norms.append(v_latest.norm(dim=-1).mean().item())

            mem_bytes = cache_memory_bytes(seq_len, num_layers, num_kv_heads, head_dim)

            steps.append({
                "step": step,
                "seq_len": seq_len,
                "memory_mb": mem_bytes / (1024 ** 2),
                "step_time_ms": t_step * 1000,
                "k_norms": k_norms,
                "v_norms": v_norms,
                "token": tokenizer.decode([next_id.item()]),
            })

            # Next iteration: feed only the new token. Cache covers the rest.
            input_ids = next_id.view(1, 1)
            attention_mask = torch.cat(
                [attention_mask,
                 torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )

    _sync(device)
    total = time.perf_counter() - total_start

    return {
        "prompt": prompt,
        "generated_text": tokenizer.decode(generated_ids),
        "total_time_s": total,
        "steps": steps,
        "config": {"num_layers": num_layers, "num_kv_heads": num_kv_heads,
                   "head_dim": head_dim},
    }


def profile_generation_without_cache(model, tokenizer, prompt: str,
                                     max_new_tokens: int = 50) -> dict:
    """
    Generate one token at a time with use_cache=False — re-feed the full
    sequence every step. For speed comparison only; no per-step state.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generated_ids = []

    _sync(device)
    total_start = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            next_id = out.logits[0, -1, :].argmax().view(1, 1)
            generated_ids.append(next_id.item())
            input_ids = torch.cat([input_ids, next_id], dim=1)
            attention_mask = torch.cat(
                [attention_mask,
                 torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )

    _sync(device)
    total = time.perf_counter() - total_start

    return {
        "prompt": prompt,
        "generated_text": tokenizer.decode(generated_ids),
        "total_time_s": total,
    }


def compare_speed(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> dict:
    """Side-by-side speed test with and without cache."""
    print(f"\n  Profiling WITH cache...")
    with_c = profile_generation_with_cache(model, tokenizer, prompt, max_new_tokens)
    print(f"  Profiling WITHOUT cache (slower)...")
    without_c = profile_generation_without_cache(model, tokenizer, prompt, max_new_tokens)

    speedup = without_c["total_time_s"] / with_c["total_time_s"]
    print(f"\n  {'─'*72}")
    print(f"  Generated {max_new_tokens} tokens for {repr(prompt[:50])}")
    print(f"  WITH cache:    {with_c['total_time_s']:.2f}s  "
          f"({max_new_tokens / with_c['total_time_s']:.1f} tok/s)")
    print(f"  WITHOUT cache: {without_c['total_time_s']:.2f}s  "
          f"({max_new_tokens / without_c['total_time_s']:.1f} tok/s)")
    print(f"  Speedup: {speedup:.2f}×")
    print(f"  {'─'*72}\n")
    return {"with_cache": with_c, "without_cache": without_c, "speedup": speedup}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_cache_growth(profile: dict, save_path: Path = None) -> None:
    """Cache memory and step time vs generation step (twin axes)."""
    steps = profile["steps"]
    step_idx = [s["step"] for s in steps]
    memory = [s["memory_mb"] for s in steps]
    times = [s["step_time_ms"] for s in steps]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    c1 = "#2c7be5"
    ax1.plot(step_idx, memory, color=c1, linewidth=2, marker="o", markersize=3,
             label="KV cache (MB)")
    ax1.set_xlabel("Generation step")
    ax1.set_ylabel("KV cache memory (MB)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    c2 = "#e74c3c"
    ax2.plot(step_idx, times, color=c2, linewidth=2, marker="s", markersize=3,
             alpha=0.7, label="Step time (ms)")
    ax2.set_ylabel("Step time (ms)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    plt.title(f"KV cache growth — {repr(profile['prompt'][:45])}")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


def plot_k_norm_heatmap(profile: dict, save_path: Path = None) -> None:
    """Layer × generation-step heatmap of K vector norms."""
    steps = profile["steps"]
    num_layers = profile["config"]["num_layers"]
    matrix = np.array([s["k_norms"] for s in steps]).T  # [layers, steps]

    xlabels = [f"S{s['step']}" if s["step"] % 5 == 0 else "" for s in steps]
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(
        matrix,
        cmap="viridis",
        xticklabels=xlabels,
        yticklabels=[f"L{i}" for i in range(num_layers)],
        ax=ax,
        cbar_kws={"label": "K vector norm (avg over kv_heads)"},
    )
    ax.set_title(f"K-vector norms by layer × step — {repr(profile['prompt'][:45])}")
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Layer")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


def plot_speed_comparison(speed: dict, save_path: Path = None) -> None:
    """Bar chart: with-cache vs without-cache total time."""
    labels = ["with cache", "without cache"]
    times = [speed["with_cache"]["total_time_s"], speed["without_cache"]["total_time_s"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, times, color=["#2c7be5", "#e74c3c"])
    ax.set_ylabel("Total generation time (s)")
    ax.set_title(f"Speed: with vs without KV cache  ·  speedup = {speed['speedup']:.2f}×")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, t, f"{t:.2f}s",
                ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.inspector.model_loader import load_model

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model()

    print("\n" + "█"*72)
    print("  KV CACHE ANALYZER — Day 4")
    print("█"*72)

    # 1. Pure-math memory breakdown
    print_memory_breakdown(model)

    # 2. Profile a long-ish generation with full state tracking
    prompt = "The sun appears yellow because"
    print(f"  Profiling generation: {repr(prompt)} (50 tokens, with cache)")
    profile = profile_generation_with_cache(model, tokenizer, prompt, max_new_tokens=50)
    print(f"\n  Generated: {repr(profile['generated_text'])}")
    print(f"  Total time: {profile['total_time_s']:.2f}s")
    print(f"  Final cache: {profile['steps'][-1]['memory_mb']:.2f} MB at "
          f"{profile['steps'][-1]['seq_len']} tokens")

    plot_cache_growth(profile, save_path=OUTPUTS_DIR / "cache_growth.png")
    plot_k_norm_heatmap(profile, save_path=OUTPUTS_DIR / "k_norms_heatmap.png")

    # 3. Speed comparison: with vs without cache
    speed = compare_speed(model, tokenizer, "Once upon a time", max_new_tokens=30)
    plot_speed_comparison(speed, save_path=OUTPUTS_DIR / "speed_comparison.png")

    print(f"\n✓ All outputs saved to {OUTPUTS_DIR}/")
