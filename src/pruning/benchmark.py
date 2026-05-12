"""
Day 8 — Speed/memory benchmarks for pruning configurations + Track A summary.

For each of 8 representative pruning configs (full 16 layers down through
total-collapse 8 layers), measure:

  - num_params      → from `sum(p.numel() for p in model.parameters())`
  - memory_mb       → params × 2 bytes (FP16)
  - ttft_ms         → time to first generated token, median of 5 runs
  - gen_50_s        → time to generate 50 tokens, median of 3 runs
  - tokens_per_sec  → 50 / gen_50_s
  - speedup_x       → relative to full-16 baseline

We rely on `torch.mps.synchronize()` (or cuda equivalent) around every
timer to get honest wall-clock numbers on MPS. Quality numbers are
copied in from Day 6 / Day 7 runs — we do NOT re-evaluate quality here.

Outputs:
  - outputs/pruning_results/day8_benchmark.csv
  - outputs/pruning_results/day8_speed_vs_layers.png   (speed curve)
  - outputs/pruning_results/day8_pareto.png            (quality × speed scatter)
  - outputs/pruning_results/TRACK_A_SUMMARY.md         (portfolio-ready writeup)

Usage:
    cd ~/projects/llama1B-finetunning
    ~/venvs/llama-xray/bin/python -m src.pruning.benchmark
"""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.inspector.model_loader import load_model
from src.pruning.layer_pruner import LayerPruner

OUTPUTS_DIR = Path("outputs/pruning_results")
PROMPT = "The capital of France is"
GEN_TOKENS = 50
TTFT_RUNS = 5
GEN_RUNS = 3

# Quality numbers pulled from Day 6 / Day 7 CSVs — we don't re-measure here.
CONFIGS = [
    {"label": "Full 16 layers (baseline)",       "drop": [],                   "quality_pct": 100.0, "source": "baseline"},
    {"label": "Drop L4 alone (mid)",             "drop": [4],                  "quality_pct": 70.6,  "source": "Day 6 mid-single"},
    {"label": "Drop L12 alone (smart)",          "drop": [12],                 "quality_pct": 64.7,  "source": "Day 7 smart"},
    {"label": "Drop L15 alone (end)",            "drop": [15],                 "quality_pct": 52.9,  "source": "Day 6 end"},
    {"label": "Drop L12, L7 (smart 14)",         "drop": [12, 7],              "quality_pct": 41.2,  "source": "Day 7 smart"},
    {"label": "Drop L14, L15 (end 14)",          "drop": [14, 15],             "quality_pct": 23.5,  "source": "Day 6 end"},
    {"label": "Drop L12, L7, L6 (smart 13)",     "drop": [12, 7, 6],           "quality_pct": 41.2,  "source": "Day 7 smart"},
    {"label": "Drop L8-15 (end 8)",              "drop": [8, 9, 10, 11, 12, 13, 14, 15], "quality_pct": 0.0, "source": "Day 6 end"},
]


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------

def sync(device) -> None:
    """Honest wall-clock timing requires synchronizing async device queues."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_one(model, tokenizer) -> dict:
    """Measure params, memory, TTFT, generation time for current model state."""
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    num_params = sum(p.numel() for p in model.parameters())
    memory_mb = num_params * 2 / (1024 ** 2)   # FP16

    # Warm-up: avoid first-call kernel-compile overhead from polluting timing
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
    sync(model.device)

    # Time to first token: 1 new token, median of TTFT_RUNS
    ttft_times = []
    for _ in range(TTFT_RUNS):
        sync(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
        sync(model.device)
        ttft_times.append(time.perf_counter() - t0)
    ttft_ms = float(np.median(ttft_times)) * 1000

    # Full generation: 50 tokens, median of GEN_RUNS
    gen_times = []
    for _ in range(GEN_RUNS):
        sync(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=GEN_TOKENS, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
        sync(model.device)
        gen_times.append(time.perf_counter() - t0)
    gen_50_s = float(np.median(gen_times))
    tokens_per_sec = GEN_TOKENS / gen_50_s

    return {
        "num_params": num_params,
        "memory_mb": memory_mb,
        "ttft_ms": ttft_ms,
        "gen_50_s": gen_50_s,
        "tokens_per_sec": tokens_per_sec,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_csv(results, save_path: Path):
    fields = ["label", "source", "drop", "layers_remaining", "quality_pct",
              "num_params", "memory_mb", "ttft_ms", "gen_50_s",
              "tokens_per_sec", "speedup_x", "memory_saved_mb"]
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fields}
            if isinstance(row["drop"], list):
                row["drop"] = ",".join(str(x) for x in row["drop"])
            w.writerow(row)
    print(f"  saved → {save_path}")


def plot_speed_curve(results, save_path: Path):
    fig, ax1 = plt.subplots(figsize=(11, 6))

    layers = [r["layers_remaining"] for r in results]
    tps = [r["tokens_per_sec"] for r in results]
    quality = [r["quality_pct"] for r in results]

    # Sort by layers descending for visual clarity
    order = np.argsort(layers)[::-1]
    layers_sorted = [layers[i] for i in order]
    tps_sorted = [tps[i] for i in order]
    quality_sorted = [quality[i] for i in order]

    color1 = "#1f77b4"
    color2 = "#d62728"
    ax1.plot(layers_sorted, tps_sorted, marker="o", linewidth=2, color=color1, label="Tokens/sec")
    ax1.set_xlabel("Layers remaining")
    ax1.set_ylabel("Tokens / second", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(sorted(set(layers_sorted), reverse=True))
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(layers_sorted, quality_sorted, marker="s", linewidth=2, color=color2,
             linestyle="--", label="Quality (exact match %)")
    ax2.axhline(75, linestyle=":", color="gray", alpha=0.6)
    ax2.set_ylabel("Exact match %", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-5, 105)

    plt.title("Day 8: Speed × Quality vs Layers remaining")
    fig.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def plot_pareto(results, save_path: Path):
    """Pareto-style scatter: quality vs speedup. Top-right is the ideal corner."""
    fig, ax = plt.subplots(figsize=(11, 6))
    speedups = [r["speedup_x"] for r in results]
    quality = [r["quality_pct"] for r in results]
    labels = [r["label"] for r in results]

    ax.scatter(speedups, quality, s=120, color="#2ca02c", zorder=3, edgecolor="black")
    for x, y, label in zip(speedups, quality, labels):
        ax.annotate(label, xy=(x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color="black", alpha=0.85)

    ax.axhline(75, linestyle="--", color="gray", alpha=0.6, label="75% quality threshold")
    ax.axvline(1.0, linestyle=":", color="gray", alpha=0.4)
    ax.set_xlabel("Speedup vs full-16 baseline  →  faster")
    ax.set_ylabel("Quality (exact match %)  →  better")
    ax.set_title("Day 8: Quality–Speed Pareto. Top-right corner = ideal (empty here).")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_ylim(-5, 105)
    fig.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def print_table(results):
    print("\n" + "=" * 110)
    print("  DAY 8 BENCHMARK — all configurations")
    print("=" * 110)
    print(f"  {'Config':<33} {'#L':>3} {'Qual%':>6} {'Params':>13} {'Mem MB':>8} "
          f"{'TTFT ms':>9} {'Tok/s':>7} {'Speedup':>8}")
    print(f"  {'-'*33} {'-'*3} {'-'*6} {'-'*13} {'-'*8} {'-'*9} {'-'*7} {'-'*8}")
    for r in results:
        print(f"  {r['label']:<33} {r['layers_remaining']:>3} "
              f"{r['quality_pct']:>6.1f} {r['num_params']:>13,} "
              f"{r['memory_mb']:>8.1f} {r['ttft_ms']:>9.1f} "
              f"{r['tokens_per_sec']:>7.2f} {r['speedup_x']:>7.2f}x")
    print("=" * 110)


# ---------------------------------------------------------------------------
# Track A summary markdown
# ---------------------------------------------------------------------------

def write_track_a_summary(results, save_path: Path):
    baseline = results[0]

    def fmt_row(r):
        params_b = r["num_params"] / 1e9
        mem_saved = baseline["memory_mb"] - r["memory_mb"]
        return (f"| {r['label']} | {r['layers_remaining']} | "
                f"{r['quality_pct']:.1f}% | {r['tokens_per_sec']:.2f} | "
                f"{r['speedup_x']:.2f}× | {r['memory_mb']:.0f} MB | "
                f"{mem_saved:.0f} MB | {params_b:.3f} B |")

    md = f"""# Track A Summary — Layer Pruning on Llama 3.2 1B

> Generated by `src/pruning/benchmark.py` on Day 8. Quality numbers
> come from Day 6 / Day 7 runs; speed/memory numbers measured fresh
> here on Mac Mini M4 MPS, FP16, generating 50 tokens after the prompt
> `"{PROMPT}"`. Times are median of {GEN_RUNS} runs (gen) /
> {TTFT_RUNS} runs (TTFT) after a warm-up forward.

## Original goal

> Find the minimum number of layers that maintains ≥75% top-1
> exact-match across our 17-prompt test suite (factual / math / code /
> pattern / reasoning).

## Result (reframed)

For Llama 3.2 1B, **no sub-16 layer configuration holds 75% exact
match.** The 1B model is at the dense edge of what survives layer
pruning. The deliverable reframes to:

> Characterize the redundancy structure of Llama 3.2 1B and
> document the quality / speed / memory trade-off across
> representative pruning configurations.

## Method

| Day | Phase | Built | Used for |
|---|---|---|---|
| 5 | 2A | `layer_importance_scorer.py` — 3 methods (logit-lens KL, zero-out, cosine sim) | Ranking layers by predicted importance |
| 6 | 2B | `layer_pruner.py`, `eval_pruned.py` | Sequential end + single-mid removal experiments |
| 7 | 2C | `smart_pruning.py`, `SkippableLayer` | Smart removal (Day 5 ranking) + learnable skip via distillation |
| 8 | 2D | `benchmark.py`, this summary | Speed/memory benchmarks + Track A close-out |

## Findings — which layers matter

### Critical (do not remove)

- **L0, L1** — top-2 across multiple Day 5 methods; the input-routing
  layers. Removing either collapses the model.
- **L4** — Day 5 missed this (ranked it mid-importance), but
  **independent confirmation from Day 6 + Day 7** put it at the top:
  Day 6 (best single mid-removal at 71% exact match — closest to
  threshold among all single-layer ablations); Day 7 (highest learned
  skip weight 0.9967 — the layer the model wants to keep the most).
- **L14** — top-3 by zero-out impact on Day 5.

### Most "skippable" layer

- **L12** is consistently the least-load-bearing across all our
  methods: Day 5 combined importance bottom-3 (score 0.072), Day 7
  learned skip weight lowest (0.9595), and the only layer whose
  removal in end-progressive removal is preferred to dropping a
  later layer (drop L12-15 better than drop L13-15).

### Layer-interaction non-additivity

End-removal damage is **not monotone in N**. Dropping 3 trailing
layers (L13-15 → 12% exact match) is worse than dropping 4 trailing
layers (L12-15 → 24%). Layer effects do not add linearly; pruning
quality is path-dependent. This means independently-validated
removals **cannot be safely combined** — every combined patch
needs fresh end-to-end validation.

## Findings — speed × quality × memory table

| Config | # Layers | Quality | Tok/sec | Speedup | Memory | Memory saved | Params |
|---|---|---|---|---|---|---|---|
"""
    for r in results:
        md += fmt_row(r) + "\n"

    md += f"""

> Per-layer cost: ≈ {(baseline['memory_mb'] - results[-1]['memory_mb']) / len(results[-1]['drop']):.0f} MB and {(baseline['num_params'] - results[-1]['num_params']) / len(results[-1]['drop']) / 1e6:.1f} M params per decoder layer.

## Findings — strategy comparison

| Strategy | 15-layer best | 14-layer best | 12-layer best | Avg lift vs end |
|---|---|---|---|---|
| End-removal (Day 6 Exp 1) | drop L15 → 53% | drop L14-15 → 24% | drop L12-15 → 24% | — (baseline) |
| Mid-single (Day 6 Exp 2) | **drop L4 → 71%** | n/a | n/a | best 15-layer config |
| Smart (Day 7 Exp 3) | drop L12 → 65% | drop L12,7 → 41% | drop L12,7,6,9 → 12% | **+9pp avg** |
| Learnable skip (Day 7 Exp 4) | no layer skipped (all weights in [0.96, 1.00]) | — | — | — |

**Best 15-layer configuration overall:** drop L4 alone (Day 6
mid-single experiment) — 71% exact match. Still below threshold.

## Findings — the learnable skip experiment is itself informative

Even with full gradient access to 30 distillation steps against the
frozen 16-layer teacher, **the model could not be coerced to skip
any layer.** All 16 `skip_weights` converged in [0.96, 1.00]. The L1
sparsity penalty (λ = 0.05) was too weak to overcome the KL
gradient — but the gradient signal itself says every layer matters.

This is independent evidence (gradient-based, not ablation-based)
that the 1B model's layers are densely needed — and it agrees with
Day 6's direct-ablation experiments.

## Implications for Track B (Days 9-12)

Layer pruning has hit a ceiling at the 1B scale. **KV cache
reduction (Track B) operates at finer granularity** — we drop
individual cached *tokens* rather than whole *layers*. That's a
smaller intervention with more headroom. The 5%+ compute-savings
target for Track B is still realistic; the 75% / N<16 target for
Track A was not.

Specific carry-overs from Track A:

- **L12** is the most patch-friendly layer for aggressive cache
  eviction. Three methods agree.
- **L0, L1, L4** should get gentler cache budgets — they're
  load-bearing.
- **The verification kit** (exact-match across 17 prompts +
  cosine-sim per layer + KL Δ + repetition rate as collapse canary)
  carries forward unchanged to Day 10-12 patch validation.

## What transfers to bigger models

The pruning methodology is model-agnostic; the same code can run on
3B, 7B, or 70B Llama / Mistral / Gemma by changing the model id.
Larger models are well-known to be **more** prunable than 1B (more
redundant capacity from over-parameterization). The portfolio claim
on this project: *the techniques transfer; the specific 75% / N<16
result is a property of Llama 3.2 1B specifically.*
"""
    save_path.write_text(md)
    print(f"  saved → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "█" * 72)
    print("  DAY 8 — Pruning benchmarks + Track A summary (Phase 2D, closes Track A)")
    print("█" * 72)

    model, tokenizer = load_model()
    pruner = LayerPruner(model)

    results = []
    for config in tqdm(CONFIGS, desc="  benchmark configs"):
        pruner.prune(config["drop"])
        metrics = benchmark_one(model, tokenizer)
        pruner.reset()
        results.append({
            "label": config["label"],
            "source": config["source"],
            "drop": config["drop"],
            "layers_remaining": 16 - len(config["drop"]),
            "quality_pct": config["quality_pct"],
            **metrics,
        })

    # Compute speedup vs baseline (first config)
    baseline_tps = results[0]["tokens_per_sec"]
    for r in results:
        r["speedup_x"] = r["tokens_per_sec"] / baseline_tps
        r["memory_saved_mb"] = results[0]["memory_mb"] - r["memory_mb"]

    print_table(results)
    write_csv(results, OUTPUTS_DIR / "day8_benchmark.csv")
    plot_speed_curve(results, OUTPUTS_DIR / "day8_speed_vs_layers.png")
    plot_pareto(results, OUTPUTS_DIR / "day8_pareto.png")
    write_track_a_summary(results, OUTPUTS_DIR / "TRACK_A_SUMMARY.md")

    print("\n" + "█" * 72)
    print("  TRACK A COMPLETE — see outputs/pruning_results/TRACK_A_SUMMARY.md")
    print("█" * 72 + "\n")


if __name__ == "__main__":
    main()
