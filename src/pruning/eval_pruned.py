"""
Day 6 — Sequential Layer Removal experiments.

Two experiments:
  1. Remove from the end progressively: keep layers [0..N-1] for N in 16..8.
     8 configs total (16, 15, 14, 13, 12, 11, 10, 9, 8 layers remaining).

  2. Remove from the middle, single-layer ablation: keep 0-3 and 12-15,
     ablate one of L4..L11 at a time. 8 configs total.

Quality metrics (relative to the unpruned baseline on each prompt):
  - exact_match_pct: pruned top-1 == baseline top-1
  - top5_pct:        baseline top-1 still in pruned top-5
  - mean_baseline_prob: average prob the pruned model assigns to the
    baseline top-1 token
  - repetition_rate: in 15 generated tokens, fraction NOT unique
    (high = collapsed/looping output)

Output:
  - outputs/pruning_results/day6_results.csv         — per-config table
  - outputs/pruning_results/day6_end_curve.png       — exp 1 quality curve
  - outputs/pruning_results/day6_middle_ablation.png — exp 2 per-layer drop
  - outputs/pruning_results/day6_generations.txt     — sample text from each config

Usage:
    cd ~/projects/llama1B-finetunning
    ~/venvs/llama-xray/bin/python -m src.pruning.eval_pruned
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.inspector.model_loader import load_model
from src.pruning.layer_pruner import LayerPruner

OUTPUTS_DIR = Path("outputs/pruning_results")
TEST_PROMPTS_FILE = Path("data/test_prompts.json")
GEN_TOKENS = 15
QUALITY_THRESHOLD = 75.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts():
    with open(TEST_PROMPTS_FILE) as f:
        d = json.load(f)
    return [p for cat in d.values() for p in cat]


def compute_baselines(model, tokenizer, prompts):
    """Run baseline forward on each prompt; return (baseline_top1_ids, sample_gens)."""
    top1s = {}
    sample_gens = {}
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        top1 = torch.argmax(out.logits[0, -1]).item()
        top1s[prompt] = top1
        # Sample 15-token greedy generation for the report file
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=GEN_TOKENS,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        new_text = tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        sample_gens[prompt] = new_text
    return top1s, sample_gens


def evaluate(model, tokenizer, baseline_top1s, prompts):
    """Return one metrics dict over all prompts + per-prompt gen samples."""
    n = len(prompts)
    exact = 0
    top5 = 0
    bp_probs = []
    rep_rates = []
    gens = {}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        probs = F.softmax(out.logits[0, -1].float(), dim=-1)
        top5_ids = torch.topk(probs, 5).indices.tolist()
        baseline_id = baseline_top1s[prompt]
        if top5_ids[0] == baseline_id:
            exact += 1
        if baseline_id in top5_ids:
            top5 += 1
        bp_probs.append(probs[baseline_id].item())

        # Quick generation for coherence proxy
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=GEN_TOKENS,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = gen[0, inputs["input_ids"].shape[1]:].tolist()
        unique_ratio = len(set(new_ids)) / max(len(new_ids), 1)
        rep_rates.append(1.0 - unique_ratio)
        gens[prompt] = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "exact_match_pct": 100.0 * exact / n,
        "top5_pct": 100.0 * top5 / n,
        "mean_baseline_prob": float(np.mean(bp_probs)),
        "repetition_rate": float(np.mean(rep_rates)),
    }, gens


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_remove_from_end(model, tokenizer, prompts, baseline_top1s, pruner):
    """Progressively drop layers 15, 14, 13, ... down to 8 remaining layers."""
    results = []
    all_gens = {}
    pruner.reset()

    # First entry: full model (16 layers, no removal)
    print("\n  ── Experiment 1: remove from the END (progressive) ──")
    for n_remove in tqdm(range(0, 9), desc="  end-removal"):
        to_remove = list(range(16 - n_remove, 16))
        pruner.prune(to_remove)
        metrics, gens = evaluate(model, tokenizer, baseline_top1s, prompts)
        metrics["strategy"] = "end"
        metrics["layers_remaining"] = 16 - n_remove
        metrics["removed"] = ",".join(str(x) for x in to_remove)
        results.append(metrics)
        all_gens[f"end_keep{16-n_remove}"] = gens
        pruner.reset()
    return results, all_gens


def experiment_middle_single(model, tokenizer, prompts, baseline_top1s, pruner):
    """Remove one mid-stack layer at a time (L4..L11), keeping all others."""
    results = []
    all_gens = {}
    pruner.reset()

    print("\n  ── Experiment 2: remove from the MIDDLE (one at a time, L4..L11) ──")
    for L in tqdm(range(4, 12), desc="  middle-single"):
        pruner.prune([L])
        metrics, gens = evaluate(model, tokenizer, baseline_top1s, prompts)
        metrics["strategy"] = "middle_single"
        metrics["layers_remaining"] = 15
        metrics["removed"] = str(L)
        results.append(metrics)
        all_gens[f"middle_drop_L{L}"] = gens
        pruner.reset()
    return results, all_gens


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_csv(all_results, save_path: Path):
    fields = ["strategy", "layers_remaining", "removed",
              "exact_match_pct", "top5_pct",
              "mean_baseline_prob", "repetition_rate"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"  saved → {save_path}")


def write_generations(baseline_gens, all_gens, save_path: Path):
    lines = []
    lines.append("=" * 88)
    lines.append("  GENERATION SAMPLES — first 15 tokens after each prompt")
    lines.append("=" * 88)
    lines.append("")

    def block(label, gens):
        lines.append(f"\n── {label} ──")
        for prompt, text in gens.items():
            lines.append(f"  PROMPT: {prompt!r}")
            lines.append(f"  GEN   : {text!r}")
            lines.append("")

    block("BASELINE (16 layers)", baseline_gens)
    for label, gens in all_gens.items():
        block(label, gens)
    save_path.write_text("\n".join(lines))
    print(f"  saved → {save_path}")


def plot_end_curve(end_results, save_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    layers = [r["layers_remaining"] for r in end_results]
    exact = [r["exact_match_pct"] for r in end_results]
    top5 = [r["top5_pct"] for r in end_results]
    rep = [100 * r["repetition_rate"] for r in end_results]

    ax.plot(layers, exact, marker="o", linewidth=2, label="Exact match %", color="#1f77b4")
    ax.plot(layers, top5, marker="s", linewidth=2, label="Top-5 contains baseline %", color="#2ca02c")
    ax.plot(layers, rep, marker="^", linewidth=2, label="Repetition rate %", color="#d62728")
    ax.axhline(QUALITY_THRESHOLD, linestyle="--", color="gray", alpha=0.7,
               label=f"{QUALITY_THRESHOLD:.0f}% quality threshold")

    ax.set_xlabel("Layers remaining")
    ax.set_ylabel("Percent")
    ax.set_title("Exp 1: Quality vs layers remaining (remove from the end)")
    ax.set_xticks(layers)
    ax.invert_xaxis()
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def plot_middle_ablation(middle_results, save_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    layers = [int(r["removed"]) for r in middle_results]
    exact_drop = [100.0 - r["exact_match_pct"] for r in middle_results]
    rep = [100 * r["repetition_rate"] for r in middle_results]

    width = 0.4
    x = np.array(layers)
    ax.bar(x - width / 2, exact_drop, width, label="Exact-match drop (%)", color="#1f77b4")
    ax.bar(x + width / 2, rep, width, label="Repetition rate (%)", color="#d62728")
    ax.set_xlabel("Layer removed (only this one)")
    ax.set_ylabel("Percent")
    ax.set_title("Exp 2: Single-layer middle removal — quality damage per layer")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def find_75_threshold(end_results):
    for r in end_results:
        if r["exact_match_pct"] < QUALITY_THRESHOLD:
            return r["layers_remaining"] + 1, r
    return None, None


def print_table(results, title):
    print(f"\n  {title}")
    print(f"  {'Strategy':<14} {'#Layers':<9} {'Removed':<24} {'Exact%':>7} {'Top5%':>7} {'BaseProb':>9} {'RepRate':>8}")
    print(f"  {'-'*14} {'-'*9} {'-'*24} {'-'*7} {'-'*7} {'-'*9} {'-'*8}")
    for r in results:
        print(f"  {r['strategy']:<14} {r['layers_remaining']:<9} "
              f"{str(r['removed'])[:23]:<24} "
              f"{r['exact_match_pct']:>7.1f} {r['top5_pct']:>7.1f} "
              f"{r['mean_baseline_prob']:>9.3f} {r['repetition_rate']:>8.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "█" * 72)
    print("  SEQUENTIAL LAYER REMOVAL — Day 6 (Track A — Pruning, Phase 2B)")
    print("█" * 72)

    model, tokenizer = load_model()
    prompts = load_prompts()
    print(f"\n  Loaded {len(prompts)} test prompts.")

    print("\n  Computing baseline (full 16-layer model) top-1 tokens + samples...")
    baseline_top1s, baseline_gens = compute_baselines(model, tokenizer, prompts)
    print(f"  Baseline locked in for {len(baseline_top1s)} prompts.")

    pruner = LayerPruner(model)

    end_results, end_gens = experiment_remove_from_end(model, tokenizer, prompts, baseline_top1s, pruner)
    middle_results, middle_gens = experiment_middle_single(model, tokenizer, prompts, baseline_top1s, pruner)

    pruner.reset()

    # Reporting
    print_table(end_results, "EXP 1 — REMOVE FROM END (progressive)")
    print_table(middle_results, "EXP 2 — REMOVE FROM MIDDLE (single-layer)")

    threshold_layers, threshold_row = find_75_threshold(end_results)
    print("\n" + "─" * 80)
    if threshold_layers is None:
        print(f"  THRESHOLD: model maintained ≥{QUALITY_THRESHOLD:.0f}% exact-match all the way down to 8 layers.")
    else:
        print(f"  THRESHOLD: quality drops below {QUALITY_THRESHOLD:.0f}% exact-match at "
              f"{threshold_row['layers_remaining']} remaining layers. "
              f"Last 'good' config: {threshold_layers} layers.")
    print("─" * 80)

    # Save artifacts
    all_results = end_results + middle_results
    write_csv(all_results, OUTPUTS_DIR / "day6_results.csv")
    write_generations(baseline_gens,
                      {**end_gens, **middle_gens},
                      OUTPUTS_DIR / "day6_generations.txt")
    plot_end_curve(end_results, OUTPUTS_DIR / "day6_end_curve.png")
    plot_middle_ablation(middle_results, OUTPUTS_DIR / "day6_middle_ablation.png")

    print("\n" + "█" * 72)
    print("  DONE — results in outputs/pruning_results/")
    print("█" * 72 + "\n")


if __name__ == "__main__":
    main()
