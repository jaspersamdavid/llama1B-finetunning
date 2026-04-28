"""
Attention Visualizer — extract per-head attention matrices and classify head types.

For each layer × head: shows which tokens attend to which other tokens.
Identifies head patterns: previous-token, first-token sink, semantic, positional.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import torch

OUTPUTS_DIR = Path("outputs/attention_maps")


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def get_attention_matrices(model, tokenizer, prompt: str) -> dict:
    """
    Run one forward pass and return all attention matrices.

    Returns:
        dict with:
          'tokens'   : list of decoded token strings (length = seq_len)
          'attentions': tensor [num_layers=16, num_heads=32, seq_len, seq_len]
                        each [i,j] = how much token i attends to token j
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of 16 tensors, each [1, num_heads, seq_len, seq_len]
    attn_stack = torch.stack([a[0] for a in outputs.attentions])  # [16, 32, seq, seq]

    return {
        "prompt": prompt,
        "tokens": tokens,
        "attentions": attn_stack.cpu().float(),  # move off MPS for numpy ops
    }


# ---------------------------------------------------------------------------
# Head pattern classification
# ---------------------------------------------------------------------------

def classify_head(attn_matrix: np.ndarray) -> str:
    """
    Classify a single attention head by its dominant pattern.

    attn_matrix: [seq_len, seq_len], rows = query tokens, cols = key tokens.
    Each row sums to 1.0 (softmax).

    Classification rules:
    - sink      : column 0 (BOS token) receives >40% average attention
    - prev-token: sub-diagonal (attending to token immediately before) >40% average
    - diagonal  : diagonal (self-attention) >40% average
    - uniform   : max attention weight < 0.25 (spread out evenly)
    - semantic  : everything else (content-based, variable pattern)
    """
    seq_len = attn_matrix.shape[0]
    if seq_len < 2:
        return "n/a"

    # How much attention goes to token 0 (the BOS sink)?
    sink_score = attn_matrix[:, 0].mean()

    # How much attention goes to the immediately preceding token?
    if seq_len >= 2:
        prev_token_score = np.diag(attn_matrix, k=-1).mean()
    else:
        prev_token_score = 0.0

    # How much self-attention (token attending to itself)?
    self_score = np.diag(attn_matrix).mean()

    # How uniform is the distribution?
    max_weight = attn_matrix.max()

    if sink_score > 0.40:
        return "sink"
    if prev_token_score > 0.30:
        return "prev-token"
    if self_score > 0.30:
        return "self"
    if max_weight < 0.25:
        return "uniform"
    return "semantic"


def classify_all_heads(attn_data: dict) -> dict:
    """
    Classify every head across all 16 layers.

    Returns:
        dict: layer_idx → {head_idx → pattern_type}
        Also includes summary counts per pattern type.
    """
    attentions = attn_data["attentions"].numpy()  # [16, 32, seq, seq]
    num_layers, num_heads = attentions.shape[:2]

    classification = {}
    counts = {"sink": 0, "prev-token": 0, "self": 0, "uniform": 0, "semantic": 0}

    for layer in range(num_layers):
        classification[layer] = {}
        for head in range(num_heads):
            pattern = classify_head(attentions[layer, head])
            classification[layer][head] = pattern
            counts[pattern] = counts.get(pattern, 0) + 1

    return {"per_head": classification, "counts": counts, "total": num_layers * num_heads}


def print_head_classification(classification: dict, prompt: str) -> None:
    """Print a compact grid showing head types across all layers."""
    per_head = classification["per_head"]
    counts = classification["counts"]
    total = classification["total"]

    symbols = {"sink": "S", "prev-token": "P", "self": "D", "uniform": "U", "semantic": "·"}
    colors_legend = "S=sink  P=prev-token  D=diagonal/self  U=uniform  ·=semantic"

    print(f"\n{'='*72}")
    print(f"  HEAD PATTERNS — {repr(prompt[:55])}")
    print(f"  {colors_legend}")
    print(f"{'='*72}")
    print(f"  {'Layer':<6} " + "  ".join(f"H{h:<2}" for h in range(32)))
    print(f"  {'-'*68}")

    for layer in sorted(per_head.keys()):
        row = "  ".join(symbols.get(per_head[layer][h], "?") for h in range(32))
        print(f"  L{layer:<5} {row}")

    print(f"\n  Summary: ", end="")
    for pat, count in counts.items():
        print(f"{pat}={count} ({count/total:.0%})  ", end="")
    print(f"\n{'='*72}\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_head_heatmap(
    attn_data: dict,
    layer: int,
    head: int,
    save_path: Path = None,
) -> None:
    """Plot the attention matrix for one specific layer+head."""
    attn = attn_data["attentions"][layer, head].numpy()  # [seq, seq]
    tokens = [t.replace("\n", "↵")[:10] for t in attn_data["tokens"]]

    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.6), max(5, len(tokens) * 0.5)))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Blues",
        vmin=0, vmax=1,
        ax=ax,
        linewidths=0.3,
        linecolor="#eeeeee",
        cbar_kws={"label": "Attention weight", "shrink": 0.8},
    )
    pattern = classify_head(attn)
    ax.set_title(
        f"Layer {layer}  ·  Head {head}  ·  Pattern: {pattern}\n"
        f"{repr(attn_data['prompt'][:50])}",
        fontsize=10, pad=8,
    )
    ax.set_xlabel("Key token (attended to)", fontsize=8)
    ax.set_ylabel("Query token (attending from)", fontsize=8)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


def plot_pattern_summary(classification: dict, prompt: str, save_path: Path = None) -> None:
    """
    Plot a 16×32 grid showing the head pattern type for every layer+head.
    One colored cell per head: sink=red, prev-token=blue, semantic=green, uniform=gray.
    """
    per_head = classification["per_head"]
    num_layers = len(per_head)
    num_heads = len(per_head[0])

    pattern_to_int = {"sink": 0, "prev-token": 1, "self": 2, "semantic": 3, "uniform": 4}
    cmap = mcolors.ListedColormap(["#e74c3c", "#3498db", "#9b59b6", "#2ecc71", "#95a5a6"])

    grid = np.zeros((num_layers, num_heads))
    for layer in range(num_layers):
        for head in range(num_heads):
            grid[layer, head] = pattern_to_int.get(per_head[layer][head], 3)

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=4, aspect="auto")

    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f"H{h}" for h in range(num_heads)], fontsize=7)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"L{l}" for l in range(num_layers)], fontsize=8)
    ax.set_xlabel("Head", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)
    ax.set_title(
        f"Head Pattern Map  ·  {repr(prompt[:55])}\n"
        "Red=sink  Blue=prev-token  Purple=self  Green=semantic  Gray=uniform",
        fontsize=10, pad=10,
    )

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


def plot_layer_overview(attn_data: dict, layer: int, save_path: Path = None) -> None:
    """
    Plot all 32 heads for one layer in a 4×8 grid of small heatmaps.
    Useful for seeing which heads in a layer have different patterns.
    """
    tokens = [t.replace("\n", "↵")[:6] for t in attn_data["tokens"]]
    seq_len = len(tokens)
    attentions = attn_data["attentions"][layer].numpy()  # [32, seq, seq]

    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.suptitle(
        f"All 32 heads — Layer {layer}  ·  {repr(attn_data['prompt'][:45])}",
        fontsize=11, y=1.01,
    )

    for head in range(32):
        ax = axes[head // 8][head % 8]
        pattern = classify_head(attentions[head])
        ax.imshow(attentions[head], cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"H{head} {pattern[0].upper()}", fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Multi-prompt experiment runner
# ---------------------------------------------------------------------------

def run_head_type_experiment(
    model,
    tokenizer,
    prompts: list[str] = None,
) -> dict:
    """
    Run head classification across multiple prompts and aggregate results.
    Returns per-prompt classifications + a cross-prompt consensus.
    """
    if prompts is None:
        prompts = [
            "The capital of France is",
            "The cat sat on the mat",
            "def hello_world():\n    print(",
            "Once upon a time there was a",
            "The chemical formula for water is",
        ]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    all_classifications = {}

    for prompt in prompts:
        print(f"\n  Processing: {repr(prompt[:50])}")
        attn_data = get_attention_matrices(model, tokenizer, prompt)
        classification = classify_all_heads(attn_data)
        print_head_classification(classification, prompt)

        safe = prompt[:35].replace(" ", "_").replace("\n", "↵").replace("/", "_")

        # Pattern summary grid
        plot_pattern_summary(
            classification, prompt,
            save_path=OUTPUTS_DIR / f"{safe}_pattern_map.png",
        )

        # Layer overview for layers 0, 7, 15 (early, mid, late)
        for layer in [0, 7, 15]:
            plot_layer_overview(
                attn_data, layer,
                save_path=OUTPUTS_DIR / f"{safe}_L{layer}_all_heads.png",
            )

        all_classifications[prompt] = classification

    # Cross-prompt pattern counts
    print("\n" + "="*72)
    print("  CROSS-PROMPT HEAD PATTERN SUMMARY")
    print("="*72)
    total_counts = {"sink": 0, "prev-token": 0, "self": 0, "uniform": 0, "semantic": 0}
    for prompt, cls in all_classifications.items():
        for pat, cnt in cls["counts"].items():
            total_counts[pat] = total_counts.get(pat, 0) + cnt

    grand_total = sum(total_counts.values())
    for pat, cnt in sorted(total_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / grand_total * 40)
        print(f"  {pat:<12} {cnt:>5} ({cnt/grand_total:>5.1%})  {bar}")
    print()

    return all_classifications


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.inspector.model_loader import load_model

    model, tokenizer = load_model()

    prompts = [
        "The capital of France is",
        "The cat sat on the mat",
        "def hello_world():\n    print(",
        "Once upon a time there was a",
        "The chemical formula for water is",
        "2 + 2 = 4 and 3 + 3 =",
    ]

    print("\n" + "█"*72)
    print("  ATTENTION HEAD PATTERN EXPERIMENT")
    print("█"*72)

    results = run_head_type_experiment(model, tokenizer, prompts)

    # Detailed single-head examples: show one sink head and one prev-token head
    print("\n  Plotting detailed examples for France prompt...")
    france_data = get_attention_matrices(model, tokenizer, "The capital of France is")

    # Find a sink head and a prev-token head to highlight
    france_cls = classify_all_heads(france_data)
    for layer in range(16):
        for head in range(32):
            pat = france_cls["per_head"][layer][head]
            save_name = f"france_L{layer}_H{head}_{pat}.png"
            if pat in ("sink", "prev-token", "semantic"):
                plot_head_heatmap(
                    france_data, layer, head,
                    save_path=OUTPUTS_DIR / "examples" / save_name,
                )
                # Only save first example of each type
                found = {p: False for p in ("sink", "prev-token", "semantic")}
                found[pat] = True
                if all(found.values()):
                    break

    print(f"\n✓ All visualizations saved to {OUTPUTS_DIR}/")
