"""
Layer Importance Scorer — rank Llama 3.2 1B's 16 layers by three methods.

Method 1 — Logit-lens KL delta:
    For each layer L, compute KL(P_layer_L || P_layer_{L-1}) on the
    last-token distribution. High delta = this layer changed predictions a
    lot = important. One forward pass per prompt (uses output_hidden_states).

Method 2 — Zero-out impact:
    For each layer L, zero ALL weights (attention + MLP). Run forward.
    Measure drop in the baseline top-1 token's probability.
    Higher drop = layer was carrying real signal = important.
    (num_layers + 1) forward passes per prompt.

Method 3 — Cosine similarity (1 - cos_sim) of layer input vs output:
    For the last token, cos_sim(hidden_states[L], hidden_states[L+1]).
    High sim = layer barely transformed anything = redundant.
    Reported as (1 - sim) so higher = more important. Free — uses the
    same forward pass as method 1.

Usage:
    cd ~/projects/llama1B-finetunning
    ~/venvs/llama-xray/bin/python -m src.pruning.layer_importance_scorer
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.inspector.model_loader import load_model

OUTPUTS_DIR = Path("outputs/pruning_results")
TEST_PROMPTS_FILE = Path("data/test_prompts.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_prompts() -> list[str]:
    with open(TEST_PROMPTS_FILE) as f:
        d = json.load(f)
    return [p for category in d.values() for p in category]


def project_to_logits(model, hidden_state):
    """Apply final RMSNorm + lm_head to a hidden state, return FP32 logits.
    FP32 cast is critical — KL math in FP16 underflows to NaN."""
    last = hidden_state[:, -1, :]
    normed = model.model.norm(last)
    return model.lm_head(normed).float().squeeze(0)   # [vocab], FP32


def zero_layer(model, layer_idx: int) -> None:
    """Zero attention + MLP weights for one decoder layer (in-place)."""
    layer = model.model.layers[layer_idx]
    with torch.no_grad():
        for proj in (layer.self_attn.q_proj,
                     layer.self_attn.k_proj,
                     layer.self_attn.v_proj,
                     layer.self_attn.o_proj,
                     layer.mlp.gate_proj,
                     layer.mlp.up_proj,
                     layer.mlp.down_proj):
            proj.weight.zero_()


def snapshot_layer(model, layer_idx: int) -> dict:
    """Deep-clone one layer's parameters."""
    layer = model.model.layers[layer_idx]
    return {name: p.detach().clone() for name, p in layer.state_dict().items()}


def restore_layer(model, layer_idx: int, snapshot: dict) -> None:
    """Copy snapshot tensors back into the layer in-place."""
    layer = model.model.layers[layer_idx]
    with torch.no_grad():
        sd = layer.state_dict()
        for name, original in snapshot.items():
            sd[name].copy_(original)


def normalize(arr) -> np.ndarray:
    """Min-max normalize to [0, 1] for cross-method comparison."""
    arr = np.array(arr, dtype=float)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


# ---------------------------------------------------------------------------
# Method 1 — Logit-lens KL delta + Method 3 — Cosine similarity
# (combined because both need the same single forward pass per prompt)
# ---------------------------------------------------------------------------

def score_lens_and_cosine(model, tokenizer, prompts):
    num_layers = model.config.num_hidden_layers
    deltas_per_prompt = []
    cosines_per_prompt = []

    for prompt in tqdm(prompts, desc="  lens + cosine"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        # output_hidden_states gives (num_layers + 1) tensors:
        # hidden_states[0]   = embedding output (input to layer 0)
        # hidden_states[L+1] = output of layer L

        # ----- method 1: KL between consecutive logit-lens distributions -----
        # Use log_softmax + softmax in FP32 so KL stays numerically stable.
        all_logits = [project_to_logits(model, hs) for hs in out.hidden_states]
        all_log_probs = [F.log_softmax(l, dim=-1) for l in all_logits]
        all_probs = [lp.exp() for lp in all_log_probs]
        layer_deltas = []
        for i in range(1, len(all_probs)):
            kl = (all_probs[i] * (all_log_probs[i] - all_log_probs[i - 1])).sum().item()
            layer_deltas.append(kl)

        # ----- method 3: 1 - cos_sim(last-token input, last-token output) -----
        layer_cosines = []
        for L in range(num_layers):
            inp = out.hidden_states[L][0, -1]
            outp = out.hidden_states[L + 1][0, -1]
            sim = F.cosine_similarity(inp.unsqueeze(0), outp.unsqueeze(0)).item()
            layer_cosines.append(1.0 - sim)

        deltas_per_prompt.append(layer_deltas)
        cosines_per_prompt.append(layer_cosines)

    return np.mean(deltas_per_prompt, axis=0), np.mean(cosines_per_prompt, axis=0)


# ---------------------------------------------------------------------------
# Method 2 — Zero-out impact
# ---------------------------------------------------------------------------

def score_zero_out(model, tokenizer, prompts):
    num_layers = model.config.num_hidden_layers

    # Snapshot every layer once (memory cost: ~2 GB extra for the whole model)
    print("  snapshotting all 16 layers for restore...")
    snapshots = [snapshot_layer(model, L) for L in range(num_layers)]

    drops_per_prompt = []

    for prompt in tqdm(prompts, desc="  zero-out"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model(**inputs)
        baseline_probs = F.softmax(out.logits[0, -1], dim=-1)
        baseline_top1 = torch.argmax(baseline_probs).item()
        baseline_prob = baseline_probs[baseline_top1].item()

        layer_drops = []
        for L in range(num_layers):
            zero_layer(model, L)
            with torch.no_grad():
                ablated_out = model(**inputs)
            ablated_prob = F.softmax(ablated_out.logits[0, -1], dim=-1)[baseline_top1].item()
            drop = baseline_prob - ablated_prob
            layer_drops.append(drop)
            restore_layer(model, L, snapshots[L])

        drops_per_prompt.append(layer_drops)

    return np.mean(drops_per_prompt, axis=0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def plot_overlaid_bars(scores: dict, save_path: Path) -> None:
    methods = list(scores.keys())
    num_layers = len(scores[methods[0]])
    layers = np.arange(num_layers)
    width = 0.27
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, method in enumerate(methods):
        ax.bar(layers + (i - 1) * width, normalize(scores[method]), width,
               label=method, color=colors[i], alpha=0.9)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Importance (min-max normalized to [0, 1])")
    ax.set_title("Layer importance — three independent methods (Llama 3.2 1B)")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  saved → {save_path}")


def print_table_and_ranking(scores: dict) -> tuple[np.ndarray, np.ndarray]:
    methods = list(scores.keys())
    num_layers = len(scores[methods[0]])
    normalized = {m: normalize(scores[m]) for m in methods}
    combined = np.mean([normalized[m] for m in methods], axis=0)

    print("\n" + "=" * 96)
    print("  LAYER IMPORTANCE — raw scores (high = important) + normalized combined")
    print("=" * 96)
    header = f"  {'Layer':<6}"
    for m in methods:
        header += f" {m:>22}"
    header += f" {'combined (norm)':>18}"
    print(header)
    print("  " + "-" * 94)
    for L in range(num_layers):
        line = f"  {L:<6}"
        for m in methods:
            line += f" {scores[m][L]:>22.5f}"
        line += f" {combined[L]:>18.4f}"
        print(line)
    print("=" * 96)

    ranked = np.argsort(combined)[::-1]
    print("\n  Most important → most redundant (combined ranking):")
    print(f"    {' > '.join(f'L{i}' for i in ranked)}")

    print("\n  Top-3 most important:")
    for L in ranked[:3]:
        print(f"    L{L:<3}  combined = {combined[L]:.4f}")
    print("\n  Top-3 most redundant (best pruning candidates):")
    for L in ranked[-3:][::-1]:
        print(f"    L{L:<3}  combined = {combined[L]:.4f}")
    return combined, ranked


def check_method_agreement(scores: dict) -> None:
    methods = list(scores.keys())
    rankings = {m: np.argsort(scores[m])[::-1] for m in methods}

    print("\n" + "─" * 96)
    print("  AGREEMENT CHECK — do the three methods identify the same redundant layers?")
    print("─" * 96)
    print("\n  Per-method top-3 most important:")
    for m in methods:
        print(f"    {m:<26} → {[f'L{i}' for i in rankings[m][:3]]}")
    print("\n  Per-method bottom-3 most redundant:")
    for m in methods:
        print(f"    {m:<26} → {[f'L{i}' for i in rankings[m][-3:]]}")

    # L8-specific check (Day 4 found zero_attention(L8) RAISED Paris from 48% → 69%)
    print("\n  L8 spotlight (Day 4 found zero_attention(L8) RAISED Paris probability):")
    for m in methods:
        rank_pos = list(rankings[m]).index(8) + 1
        print(f"    {m:<26} → score = {scores[m][8]:.5f}, rank = {rank_pos}/16")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "█" * 72)
    print("  LAYER IMPORTANCE SCORING — Day 5 (Track A — Pruning, Phase 2A)")
    print("█" * 72)

    model, tokenizer = load_model()
    prompts = load_test_prompts()
    print(f"\n  Loaded {len(prompts)} test prompts.")

    print("\n  ── Method 1 (logit-lens KL Δ) + Method 3 (1 - cos_sim) ──")
    deltas, cosines = score_lens_and_cosine(model, tokenizer, prompts)

    print("\n  ── Method 2 (zero-out impact) ──")
    drops = score_zero_out(model, tokenizer, prompts)

    scores = {
        "Logit-lens KL Δ": deltas,
        "Zero-out top-1 drop": drops,
        "1 - cos(in, out)": cosines,
    }

    combined, ranked = print_table_and_ranking(scores)
    check_method_agreement(scores)

    plot_overlaid_bars(scores, OUTPUTS_DIR / "layer_importance_scores.png")

    np.savez(OUTPUTS_DIR / "layer_importance_scores.npz",
             logit_lens_kl=deltas,
             zero_out_drop=drops,
             cosine_importance=cosines,
             combined=combined,
             ranked=ranked)
    print(f"  saved raw arrays → {OUTPUTS_DIR / 'layer_importance_scores.npz'}")

    print("\n" + "█" * 72)
    print("  DONE — results in outputs/pruning_results/")
    print("█" * 72 + "\n")


if __name__ == "__main__":
    main()
