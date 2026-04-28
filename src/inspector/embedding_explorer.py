"""
Embedding Explorer — explore the token embedding space via cosine similarity and 2D projections.

The embedding table is 128,256 × 2048: every token in the vocabulary has a vector.
Words trained in similar contexts have similar vectors — we can measure this with cosine similarity.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

OUTPUTS_DIR = Path("outputs/embeddings")


# ---------------------------------------------------------------------------
# Core similarity operations
# ---------------------------------------------------------------------------

def get_embedding_matrix(model) -> torch.Tensor:
    """Return the full embedding weight matrix: [vocab_size=128256, hidden=2048]."""
    return model.model.embed_tokens.weight.detach().cpu().float()


def find_similar(
    model,
    tokenizer,
    word: str,
    top_k: int = 20,
    verbose: bool = True,
) -> list[tuple[str, float]]:
    """
    Find the top_k most similar tokens to `word` by cosine similarity.

    Tokenizes `word` (uses the first token if it splits into multiple),
    looks up its embedding, then compares against the entire vocab.

    Returns list of (token_string, similarity_score) sorted descending.
    """
    embeddings = get_embedding_matrix(model)  # [128256, 2048]

    # Encode the word — try with and without leading space (Llama space-prefixed tokens)
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    token_ids_spaced = tokenizer.encode(" " + word, add_special_tokens=False)

    # Use the version that gives a single token
    if len(token_ids_spaced) == 1:
        query_id = token_ids_spaced[0]
        used_word = " " + word
    elif len(token_ids) == 1:
        query_id = token_ids[0]
        used_word = word
    else:
        # Multi-token word — use the first token and note it
        query_id = token_ids[0]
        used_word = tokenizer.decode([query_id])
        if verbose:
            print(f"  Note: '{word}' splits into {len(token_ids)} tokens, using first: {repr(used_word)}")

    query_vec = embeddings[query_id]  # [2048]

    # Cosine similarity against all embeddings
    norms = embeddings.norm(dim=1)                             # [128256]
    query_norm = query_vec.norm()
    sims = (embeddings @ query_vec) / (norms * query_norm + 1e-8)  # [128256]

    top_vals, top_ids = torch.topk(sims, top_k + 5)   # +5 to filter self

    results = []
    for val, idx in zip(top_vals.tolist(), top_ids.tolist()):
        tok_str = tokenizer.decode([idx])
        if tok_str.strip().lower() == word.strip().lower():
            continue   # skip the query word itself
        results.append((tok_str, round(val, 4)))
        if len(results) >= top_k:
            break

    if verbose:
        print(f"\n  Most similar to {repr(word)} (via token {repr(used_word)}, id={query_id}):")
        print(f"  {'Token':<20} {'Similarity':>12}")
        print(f"  {'-'*35}")
        for tok, sim in results[:15]:
            print(f"  {repr(tok):<20} {sim:>12.4f}")

    return results


def compare_words(model, tokenizer, word1: str, word2: str, verbose: bool = True) -> float:
    """Return the cosine similarity between two words' embeddings."""
    embeddings = get_embedding_matrix(model)

    def get_vec(word):
        ids_spaced = tokenizer.encode(" " + word, add_special_tokens=False)
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids_spaced) == 1:
            return embeddings[ids_spaced[0]]
        return embeddings[ids[0]]

    v1 = get_vec(word1)
    v2 = get_vec(word2)
    sim = (v1 @ v2 / (v1.norm() * v2.norm() + 1e-8)).item()

    if verbose:
        print(f"  similarity({repr(word1)}, {repr(word2)}) = {sim:.4f}")
    return sim


# ---------------------------------------------------------------------------
# 2D cluster plots
# ---------------------------------------------------------------------------

def cluster_words(
    model,
    tokenizer,
    words: list[str],
    title: str = "Embedding Cluster",
    save_path: Path = None,
    label_offset: float = 0.02,
) -> None:
    """
    Project word embeddings to 2D with PCA and plot as a scatter.

    Words that appeared in similar contexts during training will cluster together.
    """
    embeddings = get_embedding_matrix(model)

    vecs = []
    labels = []
    for word in words:
        ids_spaced = tokenizer.encode(" " + word, add_special_tokens=False)
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids_spaced) == 1:
            vec = embeddings[ids_spaced[0]].numpy()
        else:
            vec = embeddings[ids[0]].numpy()
        vecs.append(vec)
        labels.append(word)

    vecs_array = np.stack(vecs)  # [n_words, 2048]

    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs_array)  # [n_words, 2]
    variance_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=80, alpha=0.8, color="#3498db", zorder=3)

    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (coords[i, 0], coords[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(
        f"{title}\n"
        f"PCA projection  ·  PC1={variance_explained[0]:.1%} variance, "
        f"PC2={variance_explained[1]:.1%} variance",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("Principal Component 1", fontsize=9)
    ax.set_ylabel("Principal Component 2", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point — all experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.inspector.model_loader import load_model

    model, tokenizer = load_model()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "█"*72)
    print("  EMBEDDING EXPLORER — cosine similarity experiments")
    print("█"*72)

    # ── Similarity searches ──────────────────────────────────────────────────
    search_words = ["Python", "sun", "king", "France", "cat", "run"]
    for word in search_words:
        print()
        find_similar(model, tokenizer, word, top_k=15)

    # ── Word pair comparisons ────────────────────────────────────────────────
    print("\n" + "─"*50)
    print("  WORD PAIR COMPARISONS")
    print("─"*50)
    pairs = [
        ("king", "queen"),
        ("king", "castle"),
        ("Python", "Java"),
        ("Python", "snake"),
        ("France", "Paris"),
        ("France", "Germany"),
        ("cat", "dog"),
        ("cat", "table"),
        ("sun", "moon"),
        ("sun", "chair"),
    ]
    print(f"\n  {'Word 1':<12} {'Word 2':<12} {'Similarity':>12}  {'Relationship'}")
    print(f"  {'-'*55}")
    for w1, w2 in pairs:
        sim = compare_words(model, tokenizer, w1, w2, verbose=False)
        strength = "very similar" if sim > 0.6 else "similar" if sim > 0.4 else "distant" if sim < 0.2 else "moderate"
        print(f"  {repr(w1):<12} {repr(w2):<12} {sim:>12.4f}  {strength}")

    # ── Cluster plots ────────────────────────────────────────────────────────
    print("\n" + "─"*50)
    print("  CLUSTER PLOTS (PCA 2D projections)")
    print("─"*50)

    programming_words = [
        "Python", "Java", "JavaScript", "Ruby", "Go",
        "function", "class", "variable", "loop", "array",
        "database", "server", "API", "request", "error",
    ]
    cluster_words(
        model, tokenizer,
        programming_words,
        title="Programming Terms — Embedding Space",
        save_path=OUTPUTS_DIR / "cluster_programming.png",
    )
    print("  Programming cluster saved.")

    animal_words = [
        "cat", "dog", "lion", "tiger", "elephant",
        "eagle", "parrot", "penguin", "salmon", "shark",
        "ant", "bee", "spider", "snake", "frog",
    ]
    cluster_words(
        model, tokenizer,
        animal_words,
        title="Animal Terms — Embedding Space",
        save_path=OUTPUTS_DIR / "cluster_animals.png",
    )
    print("  Animal cluster saved.")

    royalty_words = [
        "king", "queen", "prince", "princess", "duke",
        "castle", "throne", "crown", "sword", "knight",
        "France", "England", "Germany", "Spain", "Rome",
    ]
    cluster_words(
        model, tokenizer,
        royalty_words,
        title="Royalty / Geography Terms — Embedding Space",
        save_path=OUTPUTS_DIR / "cluster_royalty.png",
    )
    print("  Royalty cluster saved.")

    print(f"\n✓ All outputs saved to {OUTPUTS_DIR}/")
