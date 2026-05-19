"""
Day 9 — KV Cache Profiler.

Establishes Track B baselines: how much attention each cached token
actually receives, what fraction of the cache is "dead weight", how
cache memory scales with sequence length.

Setup per prompt:
  1. Forward the prompt once (output_attentions=True, use_cache=True).
     Record initial intra-prompt attention.
  2. For 100 generation steps: pass the last new token + past_key_values
     back in, extract the per-layer attention vectors (shape
     [1, num_heads, 1, kv_len]), accumulate "attention received" by
     each cached position.
  3. Compute the "dead-token rate": fraction of cached positions whose
     cumulative attention is below 1% of the total attention budget
     for that prompt.

Outputs:
  - outputs/cache_profiles/day9_cache_memory_curve.png
  - outputs/cache_profiles/day9_attention_heatmap.png           (canonical prompt)
  - outputs/cache_profiles/day9_per_layer_dead_pct.png
  - outputs/cache_profiles/day9_memory_breakdown.png
  - outputs/cache_profiles/day9_dead_token_stats.csv
  - outputs/cache_profiles/day9_per_layer_attention.csv

Usage:
    cd ~/projects/llama1B-finetunning
    ~/venvs/llama-xray/bin/python -m src.kv_optimization.cache_profiler
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.inspector.model_loader import load_model

OUTPUTS_DIR = Path("outputs/cache_profiles")
TEST_PROMPTS_FILE = Path("data/test_prompts.json")
MAX_NEW_TOKENS = 100
DEAD_THRESHOLD_PCT = 1.0
CANONICAL_PROMPT_IDX = 4   # "The first president of the United States was" — longest factual
KV_BYTES_PER_TOKEN = 32 * 1024   # Day 4: 32 KB / token for Llama 3.2 1B FP16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts() -> list[str]:
    with open(TEST_PROMPTS_FILE) as f:
        d = json.load(f)
    return [p for cat in d.values() for p in cat]


# ---------------------------------------------------------------------------
# Profile one prompt
# ---------------------------------------------------------------------------

def profile_prompt(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    """
    Returns:
      attn_received  [num_layers, prompt_len + max_new_tokens]
                     — total attention each cached position received,
                     summed over heads AND over all queries (prompt
                     queries + every generation step's query).
      heatmap        [max_new_tokens, prompt_len + max_new_tokens]
                     — heatmap[step, pos] = attention from step's new-token
                     query to cached position `pos`, averaged across layers
                     and summed across heads. NaN where pos > kv_len at
                     that step.
      cache_lens     list of int, length max_new_tokens — kv-cache size
                     observed by attention at each generation step.
      prompt_len     int.
    """
    num_layers = model.config.num_hidden_layers
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    max_len = prompt_len + max_new_tokens

    attn_received = np.zeros((num_layers, max_len), dtype=np.float64)
    heatmap = np.full((max_new_tokens, max_len), np.nan, dtype=np.float64)
    cache_lens = []

    # ----- Prompt forward (queries = all prompt tokens) -----
    with torch.no_grad():
        out = model(**inputs, output_attentions=True, use_cache=True)

    # out.attentions[L]: [1, H, prompt_len, prompt_len]
    for L, attn in enumerate(out.attentions):
        received = attn.sum(dim=(0, 1, 2)).float().cpu().numpy()
        attn_received[L, :prompt_len] += received

    past_kv = out.past_key_values
    next_token = out.logits[0, -1].argmax().view(1, 1)

    # ----- Generation loop (1 query per step) -----
    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past_kv,
                output_attentions=True,
                use_cache=True,
            )

        kv_len = out.attentions[0].shape[-1]
        cache_lens.append(kv_len)

        layer_sum = np.zeros(kv_len, dtype=np.float64)
        for L, attn in enumerate(out.attentions):
            # attn: [1, H, 1, kv_len] — one query attending to kv_len cached
            received = attn.sum(dim=(0, 1, 2)).float().cpu().numpy()
            attn_received[L, :kv_len] += received
            layer_sum += received
        heatmap[step, :kv_len] = layer_sum / num_layers

        past_kv = out.past_key_values
        next_token = out.logits[0, -1].argmax().view(1, 1)

    return attn_received, heatmap, cache_lens, prompt_len


# ---------------------------------------------------------------------------
# Dead-token stats
# ---------------------------------------------------------------------------

def dead_token_stats(attn_received: np.ndarray, prompt_len: int,
                     n_tokens_total: int, threshold_pct: float = DEAD_THRESHOLD_PCT):
    """
    For each layer, returns (% of cached tokens that are 'dead',
    i.e. receive less than threshold_pct % of the total attention budget).
    Considers only positions 0 to n_tokens_total (the live cache range).
    Returns:
      dead_pct_per_layer  [num_layers]
      total_budget_per_layer [num_layers]
      attn_per_pos        [num_layers, n_tokens_total]
    """
    num_layers = attn_received.shape[0]
    attn = attn_received[:, :n_tokens_total]            # [L, T]
    total = attn.sum(axis=1, keepdims=True) + 1e-12      # [L, 1]
    pct_share = 100.0 * attn / total                     # [L, T]
    dead = pct_share < threshold_pct                     # bool [L, T]
    dead_pct_per_layer = 100.0 * dead.sum(axis=1) / n_tokens_total
    return dead_pct_per_layer, total.flatten(), pct_share


# ---------------------------------------------------------------------------
# Aggregation across prompts
# ---------------------------------------------------------------------------

def profile_all_prompts(model, tokenizer, prompts):
    """Returns per-prompt records + the canonical-prompt heatmap data."""
    records = []
    canonical_data = None

    for i, prompt in enumerate(tqdm(prompts, desc="  cache profile")):
        attn_received, heatmap, cache_lens, prompt_len = profile_prompt(
            model, tokenizer, prompt)
        n_total = prompt_len + MAX_NEW_TOKENS
        dead_pct_per_layer, total_budget, pct_share = dead_token_stats(
            attn_received, prompt_len, n_total)
        records.append({
            "prompt_idx": i,
            "prompt": prompt,
            "prompt_len": prompt_len,
            "n_total": n_total,
            "dead_pct_per_layer": dead_pct_per_layer,
            "dead_pct_avg": float(dead_pct_per_layer.mean()),
            "attn_received": attn_received,
            "pct_share": pct_share,
            "cache_lens": cache_lens,
        })
        if i == CANONICAL_PROMPT_IDX:
            canonical_data = {
                "prompt": prompt,
                "prompt_len": prompt_len,
                "heatmap": heatmap,
                "attn_received": attn_received,
                "pct_share": pct_share,
            }
    return records, canonical_data


# ---------------------------------------------------------------------------
# Memory breakdown
# ---------------------------------------------------------------------------

def memory_breakdown(model, seq_lengths):
    """KV cache vs model weights at multiple sequence lengths."""
    n_params = sum(p.numel() for p in model.parameters())
    weights_mb = n_params * 2 / (1024 ** 2)        # FP16

    rows = []
    for L in seq_lengths:
        kv_mb = L * KV_BYTES_PER_TOKEN / (1024 ** 2)
        total = weights_mb + kv_mb
        rows.append({
            "seq_len": L,
            "weights_mb": weights_mb,
            "kv_cache_mb": kv_mb,
            "total_mb": total,
            "kv_cache_pct_of_total": 100.0 * kv_mb / total,
        })
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def plot_cache_memory_curve(seq_lengths, kv_mb_list, save_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(seq_lengths, kv_mb_list, marker="o", linewidth=2, color="#1f77b4")
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("KV cache memory (MB)")
    ax.set_title("KV cache memory grows linearly with sequence length (32 KB / token)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def plot_attention_heatmap(canonical, save_path):
    heatmap = canonical["heatmap"]
    prompt_len = canonical["prompt_len"]

    fig, ax = plt.subplots(figsize=(13, 7))
    masked = np.ma.masked_invalid(heatmap)
    im = ax.imshow(masked, aspect="auto", cmap="magma", interpolation="nearest")
    ax.axvline(prompt_len - 0.5, color="cyan", linestyle="--", linewidth=1.5,
               label=f"prompt/gen boundary (pos {prompt_len})")
    ax.set_xlabel("Cached token position")
    ax.set_ylabel("Generation step")
    ax.set_title(f"Attention received by each cached token per generation step\n"
                 f"prompt: {canonical['prompt']!r}  (averaged across 16 layers, summed over 32 heads)")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax, label="Attention weight sum")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def plot_per_layer_dead_pct(records, save_path):
    num_layers = records[0]["dead_pct_per_layer"].shape[0]
    all_layer_dead = np.stack([r["dead_pct_per_layer"] for r in records])
    avg = all_layer_dead.mean(axis=0)
    std = all_layer_dead.std(axis=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(np.arange(num_layers), avg, yerr=std, capsize=4,
           color="#d62728", alpha=0.85, label=f"avg across {len(records)} prompts")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("% of cached tokens receiving <1% attention")
    ax.set_title("Per-layer 'dead cache' rate (the eviction-headroom signal)")
    ax.set_xticks(np.arange(num_layers))
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def plot_memory_breakdown(breakdown, save_path):
    seq_lens = [b["seq_len"] for b in breakdown]
    weights = [b["weights_mb"] for b in breakdown]
    kv = [b["kv_cache_mb"] for b in breakdown]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(seq_lens))
    width = 0.6
    ax.bar(x, weights, width, label="Model weights", color="#7f7f7f")
    ax.bar(x, kv, width, bottom=weights, label="KV cache", color="#1f77b4")

    for i, b in enumerate(breakdown):
        ax.text(x[i], b["total_mb"] + 30, f"{b['kv_cache_pct_of_total']:.1f}%",
                ha="center", fontsize=9, color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in seq_lens])
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory breakdown — model weights vs KV cache (% = cache share of total)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def write_dead_token_csv(records, save_path):
    fields = ["prompt_idx", "prompt", "prompt_len", "n_total", "dead_pct_avg"]
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in fields})
    print(f"  saved → {save_path}")


def write_per_layer_attention_csv(records, save_path):
    num_layers = records[0]["dead_pct_per_layer"].shape[0]
    all_dead = np.stack([r["dead_pct_per_layer"] for r in records])
    avg_dead = all_dead.mean(axis=0)
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "avg_dead_token_pct", "std_across_prompts"])
        for L in range(num_layers):
            w.writerow([L, f"{avg_dead[L]:.2f}", f"{all_dead[:, L].std():.2f}"])
    print(f"  saved → {save_path}")


def print_summary(records, breakdown, canonical):
    print("\n" + "=" * 90)
    print("  DAY 9 — KV CACHE PROFILING SUMMARY")
    print("=" * 90)

    dead = np.array([r["dead_pct_avg"] for r in records])
    print(f"  Prompts profiled: {len(records)}")
    print(f"  Tokens generated per prompt: {MAX_NEW_TOKENS}")
    print(f"  Dead-token threshold: <{DEAD_THRESHOLD_PCT}% of layer's attention budget")
    print(f"  Mean dead-token rate across prompts: {dead.mean():.1f}% (std {dead.std():.1f})")
    print(f"  Min / Max per prompt: {dead.min():.1f}% / {dead.max():.1f}%")

    # Per-layer averaged dead rate
    all_layer_dead = np.stack([r["dead_pct_per_layer"] for r in records])
    avg_layer_dead = all_layer_dead.mean(axis=0)
    print("\n  Per-layer dead-token rate (avg across prompts):")
    sorted_layers = np.argsort(avg_layer_dead)[::-1]
    print("    Most-dead layers (most eviction headroom):")
    for L in sorted_layers[:5]:
        print(f"      L{int(L):<3}  {avg_layer_dead[L]:.1f}% dead")
    print("    Least-dead layers (cache is dense here):")
    for L in sorted_layers[-5:][::-1]:
        print(f"      L{int(L):<3}  {avg_layer_dead[L]:.1f}% dead")

    print("\n  Memory breakdown at various sequence lengths:")
    print(f"  {'seq_len':>8} {'weights_mb':>12} {'kv_mb':>10} {'cache_share':>13}")
    for b in breakdown:
        print(f"  {b['seq_len']:>8} {b['weights_mb']:>12.1f} "
              f"{b['kv_cache_mb']:>10.1f} {b['kv_cache_pct_of_total']:>12.2f}%")

    # Canonical-prompt attention sink check
    if canonical is not None:
        share = canonical["pct_share"]                    # [num_layers, n_total]
        n_layers = share.shape[0]
        pos0_share = share[:, 0]
        print(f"\n  ATTENTION SINK CHECK — canonical prompt {canonical['prompt']!r}")
        print(f"  Position 0 (BOS) attention share, per layer:")
        for L in range(n_layers):
            tag = " ←   strong sink" if pos0_share[L] > 20.0 else ""
            print(f"    L{L:<3}  {pos0_share[L]:.1f}%{tag}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "█" * 72)
    print("  KV CACHE PROFILER — Day 9 (Track B, Phase 3A — opens Track B)")
    print("█" * 72)

    model, tokenizer = load_model()
    prompts = load_prompts()
    print(f"\n  Loaded {len(prompts)} test prompts, generating {MAX_NEW_TOKENS} tokens each.")

    records, canonical = profile_all_prompts(model, tokenizer, prompts)

    # Memory breakdown across sequence lengths
    breakdown = memory_breakdown(model, [50, 100, 200, 500, 1000, 2000, 4000])
    kv_curve_lengths = [50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    kv_curve_mb = [L * KV_BYTES_PER_TOKEN / (1024 ** 2) for L in kv_curve_lengths]

    print_summary(records, breakdown, canonical)

    # Plots + CSV
    plot_cache_memory_curve(kv_curve_lengths, kv_curve_mb,
                             OUTPUTS_DIR / "day9_cache_memory_curve.png")
    plot_attention_heatmap(canonical, OUTPUTS_DIR / "day9_attention_heatmap.png")
    plot_per_layer_dead_pct(records, OUTPUTS_DIR / "day9_per_layer_dead_pct.png")
    plot_memory_breakdown(breakdown, OUTPUTS_DIR / "day9_memory_breakdown.png")

    write_dead_token_csv(records, OUTPUTS_DIR / "day9_dead_token_stats.csv")
    write_per_layer_attention_csv(records, OUTPUTS_DIR / "day9_per_layer_attention.csv")

    # Save the canonical attention received array for downstream Day 10 use
    np.savez(OUTPUTS_DIR / "day9_canonical_attention.npz",
             attn_received=canonical["attn_received"],
             heatmap=canonical["heatmap"],
             prompt=np.array([canonical["prompt"]]),
             prompt_len=np.array([canonical["prompt_len"]]))
    print(f"  saved → {OUTPUTS_DIR / 'day9_canonical_attention.npz'}")

    print("\n" + "█" * 72)
    print("  DONE — Track B baseline established")
    print("█" * 72 + "\n")


if __name__ == "__main__":
    main()
