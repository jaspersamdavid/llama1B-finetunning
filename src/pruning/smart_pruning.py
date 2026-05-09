"""
Day 7 — Smart Pruning + Learnable Skip Connections.

Two experiments:

  Experiment 3 — SMART PRUNING (least-important first):
    Use Day 5's combined importance ranking. Remove layers in order of
    increasing importance (L12 first, then L7, L6, ...). At each step,
    re-evaluate. This should beat Day 6's end-removal curve at every
    layer count if the Day 5 ranking is at all predictive.

  Experiment 4 — LEARNABLE SKIP WEIGHTS:
    Wrap every decoder layer with SkippableLayer:
        out = w * layer(x) + (1 - w) * x        # w = sigmoid(skip_logit)
    Freeze all 1.24B model parameters; train ONLY the 16 skip_logits via
    distillation against the frozen 16-layer baseline. Loss = KL(student,
    teacher) + λ * mean(w) so the model is pulled toward sparsity but
    must keep the layers it actually needs to match teacher predictions.

    Then compare which layers the model "voted" to skip with Day 5's manual
    importance ranking. This is a sanity check on Day 5: if the model
    agrees with our handcrafted methods, the ranking is reliable.

Both produce a unified comparison chart against Day 6's end + middle
experiments and a learned-skip-weight bar chart.

Outputs:
  - outputs/pruning_results/day7_smart_curve.csv
  - outputs/pruning_results/day7_strategies_compared.png
  - outputs/pruning_results/day7_learned_skip_weights.png
  - outputs/pruning_results/day7_skip_weights.csv

Usage:
    cd ~/projects/llama1B-finetunning
    ~/venvs/llama-xray/bin/python -m src.pruning.smart_pruning
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.inspector.model_loader import load_model
from src.pruning.eval_pruned import (
    GEN_TOKENS, compute_baselines, evaluate, load_prompts,
)
from src.pruning.layer_pruner import LayerPruner

OUTPUTS_DIR = Path("outputs/pruning_results")
DAY5_NPZ = OUTPUTS_DIR / "layer_importance_scores.npz"
DAY6_CSV = OUTPUTS_DIR / "day6_results.csv"


# ---------------------------------------------------------------------------
# Experiment 3 — Smart pruning (least-important first using Day 5 ranking)
# ---------------------------------------------------------------------------

def load_day5_ranking() -> list[int]:
    """Return list of layer indices ordered LEAST important → most important."""
    arrs = np.load(DAY5_NPZ)
    most_to_least = arrs["ranked"].astype(int).tolist()
    return list(reversed(most_to_least))   # least → most


def smart_prune_experiment(model, tokenizer, prompts, baseline_top1s, pruner,
                            least_first: list[int]) -> list[dict]:
    print("\n  ── Experiment 3: SMART pruning (least-important first, Day 5 ranking) ──")
    print(f"  Day 5 least-important → most: {least_first}")
    results = []
    pruner.reset()

    for n_remove in tqdm(range(0, 9), desc="  smart-removal"):
        to_remove = least_first[:n_remove]
        pruner.prune(to_remove)
        metrics, _ = evaluate(model, tokenizer, baseline_top1s, prompts)
        metrics["strategy"] = "smart"
        metrics["layers_remaining"] = 16 - n_remove
        metrics["removed"] = ",".join(str(int(x)) for x in to_remove)
        results.append(metrics)
        pruner.reset()
    return results


# ---------------------------------------------------------------------------
# Experiment 4 — Learnable skip weights
# ---------------------------------------------------------------------------

class SkippableLayer(nn.Module):
    """
    Wraps a LlamaDecoderLayer with a learnable scalar skip weight.

        out = w * layer(x) + (1 - w) * x       where w = sigmoid(skip_logit)

    With skip_logit init = 4 → w ≈ 0.982, behaviour ≈ baseline at start.
    During training the L1 penalty pulls w toward 0; only layers the
    student needs to match the teacher push w back up.

    `self.self_attn` is exposed so transformers' DynamicCache can find
    `decoder_layer.self_attn.layer_idx` on the wrapper.
    """

    def __init__(self, original_layer: nn.Module):
        super().__init__()
        self.layer = original_layer
        self.skip_logit = nn.Parameter(torch.tensor(4.0))

    @property
    def skip_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.skip_logit)

    @property
    def self_attn(self):
        return self.layer.self_attn

    def forward(self, hidden_states, *args, **kwargs):
        x_in = hidden_states
        out = self.layer(hidden_states, *args, **kwargs)
        w = self.skip_weight
        if isinstance(out, tuple):
            x_out = out[0]
            mixed = w * x_out + (1.0 - w) * x_in
            return (mixed,) + out[1:]
        return w * out + (1.0 - w) * x_in


def learnable_skip_experiment(model, tokenizer, prompts,
                               num_steps: int = 30,
                               lambda_l1: float = 0.05,
                               lr: float = 0.1,
                               teacher_max_new: int = 25):
    """
    1. Pre-compute teacher (full-model) logits for prompt + 25 generated tokens
    2. Wrap layers with SkippableLayer (model behavior unchanged at init)
    3. Train skip_logits to match teacher logits + sparsity penalty
    4. Read off learned skip weights; identify which layers the model "skipped"
    """

    # ---- Step 1: teacher data (frozen full model) ----
    print("\n  ── Experiment 4: LEARNABLE skip weights ──")
    print("  pre-computing teacher distributions (frozen 16-layer model)...")
    training = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=teacher_max_new,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
            teacher_out = model(input_ids=gen)
        training.append({
            "input_ids": gen,                          # [1, seq]
            "teacher_logits": teacher_out.logits.detach(),  # [1, seq, vocab]
        })

    # ---- Step 2: wrap layers ----
    print("  wrapping all 16 decoder layers with SkippableLayer...")
    original_layers = list(model.model.layers)
    wrapped = nn.ModuleList([SkippableLayer(L) for L in original_layers])
    model.model.layers = wrapped

    # Freeze EVERYTHING except skip_logits
    for name, p in model.named_parameters():
        p.requires_grad = name.endswith("skip_logit")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  trainable params: {sum(p.numel() for p in trainable)} "
          f"(should be 16 — the skip_logits)")

    optimizer = torch.optim.Adam(trainable, lr=lr)

    # ---- Step 3: training loop ----
    print(f"  training {num_steps} steps, lambda_l1 = {lambda_l1}, lr = {lr}")
    history = {"step": [], "kl": [], "l1": [], "weights": []}

    for step in tqdm(range(num_steps), desc="  skip-train"):
        optimizer.zero_grad()
        kl_total = 0.0
        for batch in training:
            student_out = model(input_ids=batch["input_ids"])
            log_p = F.log_softmax(student_out.logits.float(), dim=-1)
            log_q = F.log_softmax(batch["teacher_logits"].float(), dim=-1)
            p = log_p.exp()
            kl = (p * (log_p - log_q)).sum(dim=-1).mean()
            kl_total = kl_total + kl
        kl_avg = kl_total / len(training)

        weights = torch.stack([w.skip_weight for w in wrapped])
        l1 = weights.mean()

        loss = kl_avg + lambda_l1 * l1
        loss.backward()
        optimizer.step()

        history["step"].append(step)
        history["kl"].append(float(kl_avg.detach()))
        history["l1"].append(float(l1.detach()))
        history["weights"].append(weights.detach().cpu().numpy().copy())

    # ---- Step 4: extract final weights ----
    final_weights = np.array([float(w.skip_weight.detach()) for w in wrapped])

    # ---- Cleanup: restore original layers so any subsequent code sees a clean model ----
    for orig_idx, orig_layer in enumerate(original_layers):
        orig_layer.self_attn.layer_idx = orig_idx
    model.model.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = len(original_layers)
    for p in model.parameters():
        p.requires_grad = False

    return final_weights, history


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def load_day6_csv():
    """Load Day 6 results from CSV; partition into end-removal and middle-single."""
    end = []
    middle = []
    with open(DAY6_CSV) as f:
        for row in csv.DictReader(f):
            r = {
                "strategy": row["strategy"],
                "layers_remaining": int(row["layers_remaining"]),
                "removed": row["removed"],
                "exact_match_pct": float(row["exact_match_pct"]),
                "top5_pct": float(row["top5_pct"]),
                "mean_baseline_prob": float(row["mean_baseline_prob"]),
                "repetition_rate": float(row["repetition_rate"]),
            }
            (end if row["strategy"] == "end" else middle).append(r)
    return end, middle


def print_smart_table(smart_results):
    print("\n  EXP 3 — SMART pruning (least-important first)")
    print(f"  {'#Layers':<9} {'Removed':<32} {'Exact%':>7} {'Top5%':>7} {'BaseProb':>9} {'RepRate':>8}")
    print(f"  {'-'*9} {'-'*32} {'-'*7} {'-'*7} {'-'*9} {'-'*8}")
    for r in smart_results:
        print(f"  {r['layers_remaining']:<9} "
              f"{str(r['removed'])[:31]:<32} "
              f"{r['exact_match_pct']:>7.1f} {r['top5_pct']:>7.1f} "
              f"{r['mean_baseline_prob']:>9.3f} {r['repetition_rate']:>8.3f}")


def plot_strategies_compared(end_results, smart_results, middle_results, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))

    end_x = [r["layers_remaining"] for r in end_results]
    end_y = [r["exact_match_pct"] for r in end_results]
    smart_x = [r["layers_remaining"] for r in smart_results]
    smart_y = [r["exact_match_pct"] for r in smart_results]

    ax.plot(end_x, end_y, marker="o", linewidth=2, label="End removal (Day 6 Exp 1)", color="#d62728")
    ax.plot(smart_x, smart_y, marker="s", linewidth=2, label="Smart removal (Day 7 Exp 3)", color="#2ca02c")

    # Day 6 middle-single: 8 separate single-layer removals all at 15 remaining
    mid_y = [r["exact_match_pct"] for r in middle_results]
    ax.scatter([15] * len(mid_y), mid_y, marker="x", s=80, color="#1f77b4",
               label="Middle single-layer (Day 6 Exp 2, all at 15)")

    ax.axhline(75.0, linestyle="--", color="gray", alpha=0.7, label="75% threshold")

    ax.set_xlabel("Layers remaining")
    ax.set_ylabel("Exact match %")
    ax.set_title("Strategy comparison — exact match vs layers remaining")
    ax.set_xticks(sorted(set(end_x + smart_x), reverse=True))
    ax.invert_xaxis()
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def plot_learned_skip_weights(final_weights, day5_combined, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    layers = np.arange(16)
    axes[0].bar(layers, final_weights, color="#9467bd", alpha=0.85)
    axes[0].axhline(0.5, linestyle="--", color="gray", alpha=0.7, label="0.5 (skipped if below)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Learned skip weight (sigmoid(logit))")
    axes[0].set_title("Exp 4: Learned skip weights — model's own ranking")
    axes[0].set_xticks(layers)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Min-max normalize Day 5 combined importance for direct visual comparison
    norm_d5 = (day5_combined - day5_combined.min()) / (day5_combined.max() - day5_combined.min() + 1e-9)
    width = 0.4
    axes[1].bar(layers - width / 2, final_weights, width, label="Learned skip weight (Day 7)", color="#9467bd")
    axes[1].bar(layers + width / 2, norm_d5, width, label="Day 5 combined importance (normalized)", color="#ff7f0e")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Day 7 learned skip vs Day 5 manual importance")
    axes[1].set_xticks(layers)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def write_smart_csv(smart_results, save_path):
    fields = ["strategy", "layers_remaining", "removed",
              "exact_match_pct", "top5_pct", "mean_baseline_prob", "repetition_rate"]
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in smart_results:
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"  saved → {save_path}")


def write_skip_csv(final_weights, day5_combined, save_path):
    arrs = np.load(DAY5_NPZ)
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "learned_skip_weight",
                    "day5_combined_importance",
                    "day5_logit_lens_kl", "day5_zero_out_drop",
                    "day5_cosine_importance"])
        for L in range(16):
            w.writerow([
                L,
                f"{final_weights[L]:.4f}",
                f"{day5_combined[L]:.4f}",
                f"{arrs['logit_lens_kl'][L]:.4f}",
                f"{arrs['zero_out_drop'][L]:.4f}",
                f"{arrs['cosine_importance'][L]:.4f}",
            ])
    print(f"  saved → {save_path}")


def print_skip_summary(final_weights, day5_combined):
    print("\n  EXP 4 — Learned skip weights vs Day 5 importance")
    print(f"  {'Layer':<6} {'Learned w':>10} {'D5 combined':>12} {'Verdict':<24}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*24}")
    for L in range(16):
        w = final_weights[L]
        d5 = day5_combined[L]
        verdict = "kept" if w > 0.5 else ("partial" if w > 0.2 else "MOSTLY SKIPPED")
        print(f"  {L:<6} {w:>10.4f} {d5:>12.4f}  {verdict:<24}")

    # Rank correlation (Spearman) without scipy: rank the values, compute pearson on ranks
    def ranks(arr):
        order = np.argsort(arr)
        r = np.zeros_like(arr, dtype=float)
        r[order] = np.arange(len(arr))
        return r
    rw = ranks(final_weights)
    rd = ranks(day5_combined)
    pearson = np.corrcoef(rw, rd)[0, 1]
    print(f"\n  Spearman rank correlation (learned skip vs Day 5 combined): {pearson:.3f}")
    if pearson > 0.5:
        print("  → strong agreement: the model rediscovered Day 5's ranking.")
    elif pearson > 0.2:
        print("  → moderate agreement: partial overlap with Day 5's ranking.")
    else:
        print("  → weak/no agreement: the model's preferences differ from Day 5's manual scoring.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "█" * 72)
    print("  SMART PRUNING + LEARNABLE SKIP — Day 7 (Track A — Pruning, Phase 2C)")
    print("█" * 72)

    if not DAY5_NPZ.exists():
        raise FileNotFoundError(
            f"{DAY5_NPZ} not found. Run Day 5 (layer_importance_scorer) first.")
    if not DAY6_CSV.exists():
        raise FileNotFoundError(
            f"{DAY6_CSV} not found. Run Day 6 (eval_pruned) first.")

    model, tokenizer = load_model()
    prompts = load_prompts()
    print(f"\n  Loaded {len(prompts)} test prompts.")

    print("  Computing baseline (full 16-layer model) top-1 tokens...")
    baseline_top1s, _ = compute_baselines(model, tokenizer, prompts)

    pruner = LayerPruner(model)

    # ===== Experiment 3: Smart pruning =====
    least_first = load_day5_ranking()
    smart_results = smart_prune_experiment(model, tokenizer, prompts, baseline_top1s, pruner, least_first)
    pruner.reset()

    print_smart_table(smart_results)
    write_smart_csv(smart_results, OUTPUTS_DIR / "day7_smart_curve.csv")

    # Compare against Day 6
    end_results, middle_results = load_day6_csv()
    plot_strategies_compared(end_results, smart_results, middle_results,
                              OUTPUTS_DIR / "day7_strategies_compared.png")

    # ===== Experiment 4: Learnable skip weights =====
    final_weights, history = learnable_skip_experiment(
        model, tokenizer, prompts,
        num_steps=30, lambda_l1=0.05, lr=0.1, teacher_max_new=25)

    arrs = np.load(DAY5_NPZ)
    day5_combined = arrs["combined"]
    print_skip_summary(final_weights, day5_combined)
    write_skip_csv(final_weights, day5_combined, OUTPUTS_DIR / "day7_skip_weights.csv")
    plot_learned_skip_weights(final_weights, day5_combined,
                               OUTPUTS_DIR / "day7_learned_skip_weights.png")

    # Print smart vs end summary at every layer count
    print("\n  HEAD-TO-HEAD: end-removal vs smart-removal (exact match %)")
    print(f"  {'Layers':<8} {'End removal':>12} {'Smart removal':>14} {'Δ':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*8}")
    for sr in smart_results:
        n = sr["layers_remaining"]
        er = next((e for e in end_results if e["layers_remaining"] == n), None)
        if er is None:
            continue
        delta = sr["exact_match_pct"] - er["exact_match_pct"]
        print(f"  {n:<8} {er['exact_match_pct']:>12.1f} {sr['exact_match_pct']:>14.1f} {delta:>+8.1f}")

    print("\n" + "█" * 72)
    print("  DONE — results in outputs/pruning_results/")
    print("█" * 72 + "\n")


if __name__ == "__main__":
    main()
