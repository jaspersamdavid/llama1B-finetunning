"""
Day 11 — Advanced eviction (importance, per-layer sizing) + KV quantization.

Three new techniques, all between-step (no monkey-patching needed):

  Strategy 3 — IMPORTANCE EVICTION
    Track cumulative attention received per cached token *live* during
    generation. When cache > budget, keep: sink[:sink_size] + the
    `budget - sink_size - recent_min` highest-cumulative-attention middle
    positions + last `recent_min` positions. Requires `output_attentions=True`
    at every step.

  Strategy 4 — PER-LAYER CACHE SIZING
    Use Day 9 dead-rate data to give each layer its own budget. Layers
    with high dead-rate (L1-L3, ~99% dead) get small budgets; layers
    with dense caches (L7-L9, ~87% dead) get larger budgets. Average
    budget ≈ Day 10's uniform-20 case for direct comparison.

  Strategy 5 — KV CACHE QUANTIZATION
    Simulate INT8 / INT4 quantization on cache K/V by round-tripping
    (FP16 → INT → FP16) each step. Memory savings are theoretical
    (we don't keep the INT form), but quality drop is real and
    measurable.

All strategies evaluated with the same 30-token greedy-match metric
from Day 10 against a no-eviction baseline.

Outputs:
  - outputs/cache_profiles/day11_advanced_eviction.csv
  - outputs/cache_profiles/day11_strategies_compared.png
  - outputs/cache_profiles/day11_quant_quality.png

Usage:
    cd ~/projects/llama1B-finetunning
    ~/venvs/llama-xray/bin/python -m src.kv_optimization.advanced_eviction
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.inspector.model_loader import load_model
from src.kv_optimization.cache_profiler import load_prompts
from src.kv_optimization.token_reducer import (
    gen_no_evict, gen_with_eviction, compare_to_baseline, trim_cache,
)

OUTPUTS_DIR = Path("outputs/cache_profiles")
MAX_NEW_TOKENS = 30
SINK_SIZE = 4
RECENT_MIN = 5

# Per-layer budgets derived from Day 9 dead-rate ordering.
# Average = 21.6 ≈ Day 10's uniform-20 case for fair comparison.
PER_LAYER_BUDGETS = {
    0: 25, 1: 10, 2: 10, 3: 10, 4: 10, 5: 25, 6: 35, 7: 35,
    8: 35, 9: 35, 10: 25, 11: 20, 12: 20, 13: 10, 14: 20, 15: 20,
}


# ---------------------------------------------------------------------------
# Strategy 3 — Importance eviction
# ---------------------------------------------------------------------------

class ImportanceState:
    """Tracks cumulative attention per cached token, per layer."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.cumulative = [np.zeros(0, dtype=np.float64) for _ in range(num_layers)]

    def update(self, attentions, kv_len: int) -> None:
        """Accumulate attention received by each cached position."""
        for L, attn in enumerate(attentions):
            received = attn.sum(dim=(0, 1, 2)).float().cpu().numpy()
            cur = self.cumulative[L]
            if cur.shape[0] < kv_len:
                cur = np.concatenate([cur, np.zeros(kv_len - cur.shape[0])])
            cur[:kv_len] += received
            self.cumulative[L] = cur

    def trim(self, past_kv, sink_size: int, recent_min: int, budget: int) -> bool:
        """Trim each layer to budget; keep sink + top-importance middle + recent."""
        trimmed = False
        for L in range(self.num_layers):
            layer = past_kv.layers[L]
            seq_len = layer.keys.shape[-2]
            if seq_len <= budget:
                continue

            n_middle = budget - sink_size - recent_min
            if n_middle <= 0:
                # Degenerate budget — fall back to sink + recent
                keep = list(range(sink_size)) + list(range(seq_len - (budget - sink_size), seq_len))
            else:
                middle_start = sink_size
                middle_end = seq_len - recent_min
                middle_scores = self.cumulative[L][middle_start:middle_end]
                if len(middle_scores) > n_middle:
                    middle_top = np.argsort(middle_scores)[-n_middle:] + middle_start
                else:
                    middle_top = np.arange(middle_start, middle_end)
                keep = (list(range(sink_size))
                        + sorted(middle_top.tolist())
                        + list(range(middle_end, seq_len)))

            keep_tensor = torch.tensor(keep, device=layer.keys.device, dtype=torch.long)
            layer.keys = layer.keys[..., keep_tensor, :]
            layer.values = layer.values[..., keep_tensor, :]
            # Realign cumulative to the new cache layout
            self.cumulative[L] = self.cumulative[L][keep]
            trimmed = True
        return trimmed


def gen_with_importance_eviction(model, tokenizer, prompt: str,
                                  budget: int,
                                  sink_size: int = SINK_SIZE,
                                  recent_min: int = RECENT_MIN,
                                  max_new_tokens: int = MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    state = ImportanceState(model.config.num_hidden_layers)
    generated: list[int] = []

    with torch.no_grad():
        out = model(**inputs, output_attentions=True, use_cache=True)
        past_kv = out.past_key_values
        kv_len = out.attentions[0].shape[-1]
        state.update(out.attentions, kv_len)
        state.trim(past_kv, sink_size, recent_min, budget)

        next_token = out.logits[0, -1].argmax().view(1, 1)
        generated.append(int(next_token.item()))

        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_token,
                        past_key_values=past_kv,
                        output_attentions=True,
                        use_cache=True)
            past_kv = out.past_key_values
            kv_len = out.attentions[0].shape[-1]
            state.update(out.attentions, kv_len)
            state.trim(past_kv, sink_size, recent_min, budget)

            next_token = out.logits[0, -1].argmax().view(1, 1)
            generated.append(int(next_token.item()))

    return generated


# ---------------------------------------------------------------------------
# Strategy 4 — Per-layer cache sizing
# ---------------------------------------------------------------------------

def trim_cache_per_layer(past_kv, sink_size: int, per_layer_budgets: dict) -> None:
    """Like trim_cache but each layer has its own budget."""
    for L, layer in enumerate(past_kv.layers):
        budget_L = per_layer_budgets[L]
        seq_len = layer.keys.shape[-2]
        if seq_len <= budget_L:
            continue
        keep_last = max(budget_L - sink_size, 0)
        slices_K, slices_V = [], []
        if sink_size > 0:
            slices_K.append(layer.keys[..., :sink_size, :])
            slices_V.append(layer.values[..., :sink_size, :])
        if keep_last > 0:
            slices_K.append(layer.keys[..., -keep_last:, :])
            slices_V.append(layer.values[..., -keep_last:, :])
        layer.keys = torch.cat(slices_K, dim=-2)
        layer.values = torch.cat(slices_V, dim=-2)


def gen_with_per_layer(model, tokenizer, prompt: str,
                       per_layer_budgets: dict,
                       sink_size: int = SINK_SIZE,
                       max_new_tokens: int = MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated: list[int] = []

    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        past_kv = out.past_key_values
        trim_cache_per_layer(past_kv, sink_size, per_layer_budgets)

        next_token = out.logits[0, -1].argmax().view(1, 1)
        generated.append(int(next_token.item()))

        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            trim_cache_per_layer(past_kv, sink_size, per_layer_budgets)

            next_token = out.logits[0, -1].argmax().view(1, 1)
            generated.append(int(next_token.item()))

    return generated


# ---------------------------------------------------------------------------
# Strategy 5 — Cache quantization (simulation)
# ---------------------------------------------------------------------------

def quantize_tensor(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-tensor quantization simulation. Round-trip FP→INT→FP."""
    if bits == 16:
        return t
    max_val = (1 << (bits - 1)) - 1   # 127 for INT8, 7 for INT4
    min_val = -(1 << (bits - 1))      # -128 / -8
    scale = t.abs().max() / max_val + 1e-12
    q = (t / scale).round().clamp(min_val, max_val)
    return (q * scale).to(t.dtype)


def quantize_cache(past_kv, bits: int) -> None:
    """Round-trip-quantize every layer's K/V (simulation only)."""
    if bits == 16:
        return
    for layer in past_kv.layers:
        layer.keys = quantize_tensor(layer.keys, bits)
        layer.values = quantize_tensor(layer.values, bits)


def gen_with_quantization(model, tokenizer, prompt: str,
                          bits: int,
                          max_new_tokens: int = MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated: list[int] = []

    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        past_kv = out.past_key_values
        quantize_cache(past_kv, bits)

        next_token = out.logits[0, -1].argmax().view(1, 1)
        generated.append(int(next_token.item()))

        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            quantize_cache(past_kv, bits)

            next_token = out.logits[0, -1].argmax().view(1, 1)
            generated.append(int(next_token.item()))

    return generated


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_one_strategy(model, tokenizer, prompts, baselines, name, run_fn):
    """run_fn(prompt) → list[int] of generated tokens."""
    scores = []
    for prompt in prompts:
        test = run_fn(prompt)
        scores.append(compare_to_baseline(test, baselines[prompt]))
    return {
        "strategy": name,
        "n_prompts": len(prompts),
        "match_pct_avg": float(np.mean([s["match_pct"] for s in scores])),
        "prefix_match_avg": float(np.mean([s["prefix_match"] for s in scores])),
        "first_top1_kept_pct": 100.0 * float(np.mean([s["first_top1_kept"] for s in scores])),
        "repetition_rate_avg": float(np.mean([s["repetition_rate"] for s in scores])),
    }


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "█" * 72)
    print("  ADVANCED EVICTION + QUANTIZATION — Day 11 (Track B, Phase 3C)")
    print("█" * 72)

    model, tokenizer = load_model()
    prompts = load_prompts()
    print(f"\n  Loaded {len(prompts)} test prompts.")

    print("\n  Computing baselines (no eviction)...")
    baselines = {}
    for prompt in tqdm(prompts, desc="  baseline"):
        baselines[prompt] = gen_no_evict(model, tokenizer, prompt)

    avg_per_layer = float(np.mean(list(PER_LAYER_BUDGETS.values())))
    total_per_layer = float(np.sum(list(PER_LAYER_BUDGETS.values())))
    uniform_20_total = 20 * 16
    print(f"\n  Per-layer budgets: avg = {avg_per_layer:.1f}, total = {total_per_layer:.0f}")
    print(f"  Uniform budget=20 total = {uniform_20_total} → per-layer saves "
          f"{100*(1-total_per_layer/uniform_20_total):.1f}% cache vs uniform-20")

    results = []

    # ----- Day 10 sink+window references for comparison -----
    print("\n  Reference 1/2: Day 10 sink+window @ budget=30, 20")
    for b in (30, 20):
        results.append(run_one_strategy(
            model, tokenizer, prompts, baselines,
            f"sink+window b={b} (Day 10)",
            lambda p, b=b: gen_with_eviction(model, tokenizer, p, budget=b, sink_size=SINK_SIZE),
        ))
        print(f"    sink+window b={b}: match% = {results[-1]['match_pct_avg']:.1f}, "
              f"rep = {results[-1]['repetition_rate_avg']:.3f}")

    # ----- Strategy 3: Importance eviction -----
    print("\n  Strategy 3: IMPORTANCE eviction (track cumulative attention live)")
    for b in (30, 20, 15):
        results.append(run_one_strategy(
            model, tokenizer, prompts, baselines,
            f"importance b={b}",
            lambda p, b=b: gen_with_importance_eviction(model, tokenizer, p,
                                                        budget=b,
                                                        sink_size=SINK_SIZE,
                                                        recent_min=RECENT_MIN),
        ))
        print(f"    importance b={b}: match% = {results[-1]['match_pct_avg']:.1f}, "
              f"rep = {results[-1]['repetition_rate_avg']:.3f}")

    # ----- Strategy 4: Per-layer sizing — DEFERRED -----
    # Per-layer cache sizing crashes with a tensor-shape mismatch in
    # `eager_attention_forward` because transformers' attention mask is
    # built using a SINGLE `cache_position` shared across all 16 layers.
    # When layers have different cache lengths, the mask is the wrong
    # shape for layers whose cache doesn't match the mask length.
    # This requires monkey-patching `LlamaAttention.forward` to use
    # per-layer cache shape — moving it into the Day 12 plan.
    print("\n  Strategy 4: PER-LAYER cache sizing — DEFERRED to Day 12 (needs monkey-patching)")
    print("    Reason: transformers shares cache_position across all 16 layers; per-layer")
    print("    shapes require a custom attention mask per layer. Documented finding.")

    # ----- Strategy 5: Quantization -----
    print("\n  Strategy 5: CACHE QUANTIZATION (round-trip simulation)")
    for bits in (8, 4):
        results.append(run_one_strategy(
            model, tokenizer, prompts, baselines,
            f"quantize INT{bits} (no evict)",
            lambda p, bits=bits: gen_with_quantization(model, tokenizer, p, bits=bits),
        ))
        print(f"    INT{bits}: match% = {results[-1]['match_pct_avg']:.1f}, "
              f"rep = {results[-1]['repetition_rate_avg']:.3f}")

    # ----- Print table + save -----
    print("\n" + "=" * 90)
    print("  DAY 11 — ALL STRATEGIES (30-token greedy match vs no-evict baseline)")
    print("=" * 90)
    print(f"  {'Strategy':<35} {'Match%':>8} {'Prefix':>8} {'Step0%':>8} {'RepRate':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(f"  {r['strategy']:<35} {r['match_pct_avg']:>7.1f}% "
              f"{r['prefix_match_avg']:>8.2f} {r['first_top1_kept_pct']:>7.1f}% "
              f"{r['repetition_rate_avg']:>8.3f}")
    print("=" * 90)

    # Save CSV
    fields = ["strategy", "n_prompts", "match_pct_avg", "prefix_match_avg",
              "first_top1_kept_pct", "repetition_rate_avg"]
    with open(OUTPUTS_DIR / "day11_advanced_eviction.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print(f"\n  saved → {OUTPUTS_DIR / 'day11_advanced_eviction.csv'}")

    # Strategy comparison plot
    plot_strategies(results, OUTPUTS_DIR / "day11_strategies_compared.png")
    plot_quantization_only(results, OUTPUTS_DIR / "day11_quant_quality.png")

    print("\n" + "█" * 72)
    print("  DONE — Day 11 complete")
    print("█" * 72 + "\n")


def plot_strategies(results, save_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    labels = [r["strategy"] for r in results]
    match = [r["match_pct_avg"] for r in results]
    colors = []
    for s in labels:
        if "sink+window" in s:
            colors.append("#2ca02c")
        elif "importance" in s:
            colors.append("#9467bd")
        elif "per-layer" in s:
            colors.append("#ff7f0e")
        else:
            colors.append("#1f77b4")
    x = np.arange(len(labels))
    ax.bar(x, match, color=colors, alpha=0.85)
    ax.axhline(75, linestyle="--", color="gray", alpha=0.6, label="75% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Match % vs no-evict baseline (30-token gen)")
    ax.set_title("Day 11: All strategies compared (sink+window vs importance vs per-layer vs quantization)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def plot_quantization_only(results, save_path):
    """Cache-quant strategies vs full FP16 baseline."""
    quant_results = [r for r in results if "quantize" in r["strategy"]]
    if not quant_results:
        return
    # Include implicit "FP16 / no quant" point at 100%
    labels = ["FP16 (baseline)"] + [r["strategy"] for r in quant_results]
    match = [100.0] + [r["match_pct_avg"] for r in quant_results]
    memory_savings = [0.0] + [50.0 if "INT8" in r["strategy"] else 75.0 for r in quant_results]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, match, 0.4, color="#1f77b4", label="Match %")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Match % vs FP16 baseline", color="#1f77b4")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, memory_savings, 0.4, color="#d62728", alpha=0.85,
            label="Cache memory savings (theoretical)")
    ax2.set_ylabel("Memory savings %", color="#d62728")
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title("Day 11: Cache quantization — quality drop vs theoretical memory savings")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


if __name__ == "__main__":
    main()
