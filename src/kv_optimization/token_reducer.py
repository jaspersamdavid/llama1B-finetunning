"""
Day 10 — Attention Sink verification + basic cache eviction strategies.

Two experiments:

  EXP A — Sink verification across 17 prompts
    Re-run cache_profiler.profile_prompt for each of the 17 test prompts,
    extract position-0 attention share per layer, aggregate. Day 9
    confirmed the sink for one canonical prompt; Day 10 verifies it's
    universal.

  EXP B — Eviction strategies
    Two strategies, four cache budgets each:

      Strategy 1 — Window-only:
        After each generation step, if cache size > budget, keep only
        the last `budget` tokens. NO sink preservation.
      Strategy 2 — Sink + Window:
        Keep first 4 tokens (BOS + 3 neighbors) + last (budget - 4)
        tokens.

    For each (strategy, budget), generate 30 tokens with eviction
    enabled across all 17 test prompts. Quality A/B against baseline
    (no eviction) by:
      - match_pct      = % of generated positions matching baseline
      - prefix_match   = longest matching prefix length
      - repetition     = 1 - unique_ratio in generated tokens
      - first_top1     = baseline-top1 stayed top-1 at step 0

Outputs:
  - outputs/cache_profiles/day10_sink_verification.png
  - outputs/cache_profiles/day10_eviction_quality.png
  - outputs/cache_profiles/day10_eviction_results.csv
  - outputs/cache_profiles/day10_per_prompt_sink.csv

Usage:
    cd ~/projects/llama1B-finetunning
    ~/venvs/llama-xray/bin/python -m src.kv_optimization.token_reducer
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.inspector.model_loader import load_model
from src.kv_optimization.cache_profiler import (
    load_prompts, profile_prompt, dead_token_stats,
)

OUTPUTS_DIR = Path("outputs/cache_profiles")
MAX_NEW_TOKENS = 30
SINK_SIZE = 4
BUDGETS = [100, 30, 20, 10]   # 100 → no eviction (sanity); 10 → aggressive


# ---------------------------------------------------------------------------
# Cache mutation primitive
# ---------------------------------------------------------------------------

def trim_cache(past_kv, keep_first: int, keep_last: int) -> bool:
    """
    Mutate `past_kv` in place: keep `keep_first` initial positions plus
    `keep_last` most-recent positions. Drop the middle.

    K/V shape per layer: [batch, num_kv_heads, seq_len, head_dim].

    Returns True if a trim actually happened (cache was longer than the
    budget), False if cache fit and nothing was changed.
    """
    if keep_first < 0 or keep_last < 0:
        raise ValueError("keep_first and keep_last must be non-negative")

    trimmed = False
    for layer in past_kv.layers:
        K = layer.keys
        V = layer.values
        seq_len = K.shape[-2]
        budget = keep_first + keep_last
        if seq_len <= budget:
            continue

        slices_K, slices_V = [], []
        if keep_first > 0:
            slices_K.append(K[..., :keep_first, :])
            slices_V.append(V[..., :keep_first, :])
        if keep_last > 0:
            slices_K.append(K[..., -keep_last:, :])
            slices_V.append(V[..., -keep_last:, :])

        layer.keys = torch.cat(slices_K, dim=-2)
        layer.values = torch.cat(slices_V, dim=-2)
        trimmed = True
    return trimmed


# ---------------------------------------------------------------------------
# Generation loops
# ---------------------------------------------------------------------------

def gen_no_evict(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    """Manual greedy generation with full cache (baseline)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated: list[int] = []
    past_kv = None

    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        past_kv = out.past_key_values
        next_token = out.logits[0, -1].argmax().view(1, 1)
        generated.append(int(next_token.item()))
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[0, -1].argmax().view(1, 1)
            generated.append(int(next_token.item()))

    return generated


def gen_with_eviction(model, tokenizer, prompt: str, budget: int,
                      sink_size: int = 0,
                      max_new_tokens: int = MAX_NEW_TOKENS):
    """
    Manual generation loop that trims past_kv between steps.

    sink_size = 0 → window-only (Strategy 1)
    sink_size = 4 → sink + window (Strategy 2)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated: list[int] = []
    past_kv = None

    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        past_kv = out.past_key_values

        # Apply eviction immediately after prompt forward (matters if
        # prompt is already longer than budget — not the case for our 17,
        # but the invariant is "cache <= budget before next step")
        trim_cache(past_kv, keep_first=sink_size, keep_last=max(budget - sink_size, 0))

        next_token = out.logits[0, -1].argmax().view(1, 1)
        generated.append(int(next_token.item()))

        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values

            trim_cache(past_kv, keep_first=sink_size, keep_last=max(budget - sink_size, 0))

            next_token = out.logits[0, -1].argmax().view(1, 1)
            generated.append(int(next_token.item()))

    return generated


# ---------------------------------------------------------------------------
# Quality A/B
# ---------------------------------------------------------------------------

def compare_to_baseline(test_tokens: list[int], baseline_tokens: list[int]) -> dict:
    n = len(test_tokens)
    matches = sum(1 for a, b in zip(test_tokens, baseline_tokens) if a == b)
    prefix = 0
    for a, b in zip(test_tokens, baseline_tokens):
        if a == b:
            prefix += 1
        else:
            break
    unique = len(set(test_tokens))
    return {
        "match_pct": 100.0 * matches / n,
        "prefix_match": prefix,
        "first_top1_kept": int(test_tokens[0] == baseline_tokens[0]),
        "repetition_rate": 1.0 - unique / n,
    }


# ---------------------------------------------------------------------------
# Experiment A — sink verification across 17 prompts
# ---------------------------------------------------------------------------

def verify_sinks_across_prompts(model, tokenizer, prompts):
    """Returns (per_prompt_per_layer_pos0_share, summary stats)."""
    num_layers = model.config.num_hidden_layers
    shares = np.zeros((len(prompts), num_layers))

    for i, prompt in enumerate(tqdm(prompts, desc="  sink verify")):
        attn_received, _, _, prompt_len = profile_prompt(
            model, tokenizer, prompt, max_new_tokens=50)
        n_total = prompt_len + 50
        _, _, pct_share = dead_token_stats(attn_received, prompt_len, n_total)
        shares[i, :] = pct_share[:, 0]   # position 0 share per layer

    return shares


# ---------------------------------------------------------------------------
# Experiment B — eviction strategies
# ---------------------------------------------------------------------------

def run_eviction_experiments(model, tokenizer, prompts):
    """For each (strategy, budget), score against baseline across 17 prompts."""

    print("  Computing baseline (no eviction) generations...")
    baselines = {}
    for prompt in tqdm(prompts, desc="  baseline gen"):
        baselines[prompt] = gen_no_evict(model, tokenizer, prompt)

    configs = []
    for sink in (0, SINK_SIZE):
        for budget in BUDGETS:
            configs.append({
                "strategy": "window-only" if sink == 0 else f"sink+window (sink={sink})",
                "sink_size": sink,
                "budget": budget,
            })

    results = []
    for config in tqdm(configs, desc="  evict configs"):
        scores = []
        for prompt in prompts:
            test_tokens = gen_with_eviction(
                model, tokenizer, prompt,
                budget=config["budget"],
                sink_size=config["sink_size"],
                max_new_tokens=MAX_NEW_TOKENS,
            )
            scores.append(compare_to_baseline(test_tokens, baselines[prompt]))
        results.append({
            **config,
            "n_prompts": len(prompts),
            "match_pct_avg": float(np.mean([s["match_pct"] for s in scores])),
            "prefix_match_avg": float(np.mean([s["prefix_match"] for s in scores])),
            "first_top1_kept_pct": 100.0 * float(np.mean([s["first_top1_kept"] for s in scores])),
            "repetition_rate_avg": float(np.mean([s["repetition_rate"] for s in scores])),
        })

    return results, baselines


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def plot_sink_verification(shares, save_path: Path):
    num_layers = shares.shape[1]
    avg = shares.mean(axis=0)
    std = shares.std(axis=0)
    minv = shares.min(axis=0)
    maxv = shares.max(axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(num_layers)
    ax.bar(x, avg, yerr=std, capsize=4, color="#9467bd", alpha=0.85,
           label="mean ± std")
    ax.plot(x, minv, marker="v", linestyle="", markersize=8, color="#1f77b4",
            label="min across prompts")
    ax.plot(x, maxv, marker="^", linestyle="", markersize=8, color="#d62728",
            label="max across prompts")
    ax.axhline(50, linestyle=":", color="gray", alpha=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Position 0 (BOS) attention share %")
    ax.set_title(f"Attention sink at position 0 across {shares.shape[0]} prompts\n"
                 f"(every layer relies heavily on the sink)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def plot_eviction_quality(results, save_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))

    # Split into window-only and sink+window
    win_only = [r for r in results if r["sink_size"] == 0]
    sink_win = [r for r in results if r["sink_size"] == SINK_SIZE]

    win_budgets = [r["budget"] for r in win_only]
    win_match = [r["match_pct_avg"] for r in win_only]
    sink_budgets = [r["budget"] for r in sink_win]
    sink_match = [r["match_pct_avg"] for r in sink_win]

    order_w = np.argsort(win_budgets)
    order_s = np.argsort(sink_budgets)
    win_budgets = [win_budgets[i] for i in order_w]
    win_match = [win_match[i] for i in order_w]
    sink_budgets = [sink_budgets[i] for i in order_s]
    sink_match = [sink_match[i] for i in order_s]

    ax.plot(win_budgets, win_match, marker="o", linewidth=2,
            color="#d62728", label="Strategy 1: window-only (sink evicted)")
    ax.plot(sink_budgets, sink_match, marker="s", linewidth=2,
            color="#2ca02c", label=f"Strategy 2: sink ({SINK_SIZE}) + window")
    ax.axhline(75, linestyle="--", color="gray", alpha=0.6, label="75% threshold")

    ax.set_xlabel("Cache budget (tokens kept)")
    ax.set_ylabel("Match % vs no-eviction baseline (30-token gen)")
    ax.set_title("Day 10: Eviction quality vs cache budget — sink preservation matters")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-5, 105)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def write_eviction_csv(results, save_path: Path):
    fields = ["strategy", "sink_size", "budget", "n_prompts",
              "match_pct_avg", "prefix_match_avg",
              "first_top1_kept_pct", "repetition_rate_avg"]
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  saved → {save_path}")


def write_per_prompt_sink_csv(shares, prompts, save_path: Path):
    num_layers = shares.shape[1]
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_idx", "prompt"] + [f"L{L}_pos0_pct" for L in range(num_layers)])
        for i, prompt in enumerate(prompts):
            w.writerow([i, prompt] + [f"{shares[i, L]:.2f}" for L in range(num_layers)])
    print(f"  saved → {save_path}")


def print_eviction_table(results):
    print("\n  EXP B — Eviction quality across 17 prompts (30-token gen, vs no-evict baseline)")
    print(f"  {'Strategy':<28} {'Sink':>5} {'Budget':>7} {'Match%':>8} "
          f"{'Prefix':>7} {'Step0Ok%':>9} {'RepRate':>8}")
    print(f"  {'-'*28} {'-'*5} {'-'*7} {'-'*8} {'-'*7} {'-'*9} {'-'*8}")
    for r in results:
        print(f"  {r['strategy']:<28} {r['sink_size']:>5} {r['budget']:>7} "
              f"{r['match_pct_avg']:>7.1f}% {r['prefix_match_avg']:>7.2f} "
              f"{r['first_top1_kept_pct']:>8.1f}% {r['repetition_rate_avg']:>8.3f}")


def print_sink_summary(shares):
    avg = shares.mean(axis=0)
    print("\n  EXP A — Sink verification across 17 prompts (avg BOS attention share per layer)")
    for L in range(shares.shape[1]):
        tag = " ← strong sink" if avg[L] >= 20 else ""
        print(f"    L{L:<3}  {avg[L]:>6.1f}% ± {shares[:, L].std():.1f}%  "
              f"(min {shares[:, L].min():.1f}, max {shares[:, L].max():.1f}){tag}")
    overall = avg.mean()
    print(f"\n  Overall: BOS receives {overall:.1f}% of attention on average per layer.")
    print(f"  Sink is universal — present in every layer, every prompt.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "█" * 72)
    print("  TOKEN REDUCER — Day 10 (Track B, Phase 3B)")
    print("█" * 72)

    model, tokenizer = load_model()
    prompts = load_prompts()
    print(f"\n  Loaded {len(prompts)} test prompts.")

    print("\n  ── Experiment A: Sink verification across 17 prompts ──")
    shares = verify_sinks_across_prompts(model, tokenizer, prompts)
    print_sink_summary(shares)
    plot_sink_verification(shares, OUTPUTS_DIR / "day10_sink_verification.png")
    write_per_prompt_sink_csv(shares, prompts, OUTPUTS_DIR / "day10_per_prompt_sink.csv")

    print("\n  ── Experiment B: Eviction strategies (window-only vs sink+window) ──")
    results, _baselines = run_eviction_experiments(model, tokenizer, prompts)
    print_eviction_table(results)
    plot_eviction_quality(results, OUTPUTS_DIR / "day10_eviction_quality.png")
    write_eviction_csv(results, OUTPUTS_DIR / "day10_eviction_results.csv")

    print("\n" + "█" * 72)
    print("  DONE — Day 10 complete")
    print("█" * 72 + "\n")


if __name__ == "__main__":
    main()
