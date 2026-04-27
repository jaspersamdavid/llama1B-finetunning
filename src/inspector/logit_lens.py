"""
Logit Lens — project each layer's hidden state through lm_head to see
what the model would predict if it stopped at that layer.

Produces a layer-by-layer table and a heatmap saved to outputs/logit_lens/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

OUTPUTS_DIR = Path("outputs/logit_lens")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def run_logit_lens(model, tokenizer, prompt: str) -> dict:
    """
    Run logit lens on a single prompt.

    Extracts the 17 hidden states (embedding output + 16 layer outputs),
    projects each through norm + lm_head, returns top-5 tokens per layer.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of 17 tensors, each [1, seq_len, 2048]
    # index 0 = embed_tokens output, indices 1-16 = after each decoder layer
    hidden_states = outputs.hidden_states

    norm = model.model.norm      # final RMSNorm — Llama applies this before lm_head
    lm_head = model.lm_head     # [hidden=2048 → vocab=128256]

    layers = []
    for idx, hidden in enumerate(hidden_states):
        last = hidden[0, -1, :]              # last token's hidden state: [2048]
        normed = norm(last.unsqueeze(0))     # apply final norm: [1, 2048]
        logits = lm_head(normed)             # project to vocab: [1, 128256]
        probs = torch.softmax(logits[0], dim=-1)

        top5_probs, top5_ids = torch.topk(probs, 5)
        top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids.tolist()]

        layers.append({
            "layer": idx,
            "label": "embed" if idx == 0 else f"L{idx}",
            "top5_tokens": top5_tokens,
            "top5_probs": top5_probs.tolist(),
        })

    return {"prompt": prompt, "layers": layers}


def find_emergence_layer(result: dict, target: str) -> int:
    """
    Return the first layer index where target appears in the top-5 predictions.
    Returns -1 if target never appears.

    Comparison is case-insensitive and strips leading/trailing whitespace
    (Llama tokens often have a leading space, e.g. ' Paris').
    """
    target_clean = target.strip().lower()
    for layer_info in result["layers"]:
        for tok in layer_info["top5_tokens"]:
            if tok.strip().lower() == target_clean:
                return layer_info["layer"]
    return -1


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

def print_table(result: dict) -> None:
    """Print a compact ASCII table of per-layer top-3 predictions."""
    print(f"\n{'='*72}")
    print(f"  LOGIT LENS — {repr(result['prompt'])}")
    print(f"{'='*72}")
    print(f"  {'Layer':<7} {'Top-1 (prob)':>22} {'Top-2':>18} {'Top-3':>18}")
    print(f"  {'-'*68}")

    for info in result["layers"]:
        toks = [repr(t.strip()[:12]) for t in info["top5_tokens"]]
        p = info["top5_probs"]
        col1 = f"{toks[0]} ({p[0]:.1%})"
        print(f"  {info['label']:<7} {col1:>22} {toks[1]:>18} {toks[2]:>18}")

    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _candidate_tokens(result: dict, max_tokens: int = 12) -> list[str]:
    """
    Select the most relevant candidate tokens to show on the heatmap Y axis.

    Scoring: sum of (prob × layer_weight × rank_weight) across all layers.
    Later layers and higher-ranked tokens score higher. Returns up to max_tokens,
    sorted so the highest-scoring token is at the top (last in list = bottom row).
    """
    scores: dict[str, float] = {}
    num_layers = len(result["layers"])

    for info in result["layers"]:
        layer_weight = (info["layer"] + 1) / num_layers   # 0→early, 1→late
        for rank, (tok, prob) in enumerate(zip(info["top5_tokens"], info["top5_probs"])):
            key = tok.strip() or repr(tok)
            rank_weight = (5 - rank) / 5                  # 1.0→top-1, 0.2→top-5
            scores[key] = scores.get(key, 0) + prob * layer_weight * rank_weight

    top = sorted(scores, key=lambda t: scores[t], reverse=True)[:max_tokens]
    return list(reversed(top))   # reversed so highest-score is top row in heatmap


def plot_heatmap(result: dict, save_path: Path = None) -> None:
    """
    Plot a probability heatmap:  candidates (Y) × layers (X) → probability (color).

    Each cell shows how likely that token was at that layer.
    Green-yellow = low probability, dark red = high probability.
    """
    layers = result["layers"]
    candidates = _candidate_tokens(result)

    layer_labels = [info["label"] for info in layers]

    # Build [n_candidates × n_layers] probability matrix
    matrix = np.zeros((len(candidates), len(layers)))
    for j, info in enumerate(layers):
        tok_to_prob = {
            tok.strip() or repr(tok): prob
            for tok, prob in zip(info["top5_tokens"], info["top5_probs"])
        }
        for i, cand in enumerate(candidates):
            matrix[i, j] = tok_to_prob.get(cand, 0.0)

    fig, ax = plt.subplots(figsize=(15, max(4, len(candidates) * 0.45)))

    sns.heatmap(
        matrix,
        xticklabels=layer_labels,
        yticklabels=candidates,
        cmap="YlOrRd",
        vmin=0,
        vmax=max(matrix.max(), 0.01),
        ax=ax,
        linewidths=0.3,
        linecolor="#cccccc",
        cbar_kws={"label": "Probability", "shrink": 0.8},
        annot=matrix > 0.05,          # annotate cells with prob > 5%
        fmt=".0%",
        annot_kws={"size": 7},
    )

    prompt_display = repr(result["prompt"])
    if len(prompt_display) > 55:
        prompt_display = prompt_display[:52] + "...'"
    ax.set_title(
        f"Logit Lens  ·  {prompt_display}",
        fontsize=11, pad=10, fontweight="bold",
    )
    ax.set_xlabel("Layer  (embed = raw token embeddings, L1–L16 = transformer layers)", fontsize=9)
    ax.set_ylabel("Predicted token", fontsize=9)
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    plt.close()


# ---------------------------------------------------------------------------
# Full test-suite runner
# ---------------------------------------------------------------------------

def run_all_prompts(
    model,
    tokenizer,
    prompts_path: str = "data/test_prompts.json",
    print_tables: bool = True,
) -> dict:
    """Run logit lens on every prompt in the test suite and save heatmaps."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(prompts_path) as f:
        groups = json.load(f)

    all_results = {}

    for category, prompts in groups.items():
        print(f"\n{'─'*60}")
        print(f"  Category: {category.upper()}")
        print(f"{'─'*60}")
        cat_dir = OUTPUTS_DIR / category
        cat_results = []

        for prompt in prompts:
            result = run_logit_lens(model, tokenizer, prompt)
            if print_tables:
                print_table(result)

            safe = prompt[:40].replace(" ", "_").replace("/", "_").replace("\n", "↵")
            plot_heatmap(result, save_path=cat_dir / f"{safe}.png")
            cat_results.append(result)

        all_results[category] = cat_results

    print(f"\n✓ Heatmaps saved to {OUTPUTS_DIR}/")
    return all_results


# ---------------------------------------------------------------------------
# Entry point — key emergence experiments + full suite
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.inspector.model_loader import load_model

    model, tokenizer = load_model()

    # ── Key emergence experiments ────────────────────────────────────────────
    experiments = [
        ("The capital of France is", "Paris"),
        ("2 + 2 =", "4"),
        ("The chemical formula for water is", "H"),
        ("The cat sat on the", "mat"),
        ("def hello_world():\n    print(", "Hello"),
    ]

    print("\n" + "█"*72)
    print("  KEY EMERGENCE EXPERIMENTS — at which layer does the answer appear?")
    print("█"*72)

    for prompt, target in experiments:
        result = run_logit_lens(model, tokenizer, prompt)
        print_table(result)

        layer = find_emergence_layer(result, target)
        if layer == 0:
            print(f"  ▶  '{target}' first appears at: embedding layer (layer 0)\n")
        elif layer > 0:
            print(f"  ▶  '{target}' first appears at: layer {layer} of 16\n")
        else:
            print(f"  ▶  '{target}' never reached top-5 — check alternate spellings\n")

        safe = prompt[:40].replace(" ", "_").replace("\n", "↵")
        plot_heatmap(result, save_path=OUTPUTS_DIR / "key_experiments" / f"{safe}.png")

    # ── Full test suite ──────────────────────────────────────────────────────
    print("\n" + "█"*72)
    print("  FULL TEST SUITE")
    print("█"*72)
    run_all_prompts(model, tokenizer)
