# CLAUDE.md — llama-xray

> **Living document. Update after every completed task.**
> Key Concepts Reference lives in `LLMXray.md` — fill it in as concepts are learned.

---

## Project Overview

**Name:** llama-xray
**Model:** Llama 3.2 1B (FP16, ~2GB)
**Machine:** Mac Mini M4 — 10-core CPU, 10-core GPU, 16GB unified memory (PyTorch MPS backend for GPU acceleration)
**Duration:** 15 working days (April 23 – May 20, 2026, excluding weekends)
**Repository:** `llama-xray/`

### Goals

1. **Understand how LLMs work — top to bottom, explained simply.** No prior ML knowledge required. Concepts are written up in `LLMXray.md` as we learn them.
2. **Track A — Layer Pruning:** find the minimum number of layers that maintains 75%+ quality, and understand which layers matter.
3. **Track B — KV Cache Optimization:** reduce token compute by 5%+ for equivalent output quality.
4. **Build a reusable inspection toolkit:** logit lens, attention maps, embedding explorer, KV cache profiler, weight tweaker.
5. **Portfolio piece:** interactive dashboard + written report, techniques that transfer to 3B/7B/70B models.

---

## Current Status

> **Last updated:** May 19, 2026 (end of Day 11)
> **Currently working on:** Day 11 complete ✅. Next session: Day 12 — Combined Optimization + Track B Results (Phase 3D, closes Track B).
>
> **Day 11 highlights (full archive in `completeness.md`):**
> - **`src/kv_optimization/advanced_eviction.py`** built — importance
>   eviction (cumulative-attention scored), quantization (INT8/INT4
>   round-trip simulation). Per-layer sizing **DEFERRED** to Day 12.
> - **Importance eviction marginally beats recency-window:**
>   b=30 → 92.9% vs sink+window's 92.0% (+0.9pp); b=20 → 67.1% vs 63.5%
>   (+3.6pp). The benefit is bounded at our short sequence lengths
>   because "middle" only has ~16 positions to choose from. Expected
>   to widen on longer sequences.
> - **INT8 cache quantization costs 28pp.** FP16 → INT8 round-trip:
>   72.2% match. INT4 → 16.9% match (catastrophic). Quantization is
>   not free at the 1B scale.
> - **Per-layer cache sizing CRASHES with between-step trim** —
>   transformers shares `cache_position` across all 16 layers, so the
>   attention mask is built for one cache length. Per-layer different
>   shapes fail with `RuntimeError: size mismatch in attn_weights +
>   attention_mask`. **First technique we've hit that genuinely needs
>   monkey-patching `LlamaAttention.forward()`** — that's Day 12's
>   work.
> - **Repetition rate is the canary again:** INT8 keeps rep low (0.41);
>   INT4 spikes to 0.58. Same pattern as Day 10 sink-loss — when the
>   model loses too much information, it loops.
>
> **Day 10 highlights (full archive in `completeness.md`):**
> - **`src/kv_optimization/token_reducer.py`** built — `trim_cache()`
>   mutates `past_kv.layers[L].keys/values` between generation steps to
>   implement window-only and sink+window eviction. No monkey-patching
>   yet — just between-step cache trimming.
> - **Attention sink confirmed universal:** across all 17 prompts ×
>   16 layers, position 0 receives 49-80% of attention. Average 63.7%.
>   Sink is structural, not prompt-specific (std 2-6% per layer).
> - **Sink+window beats window-only at every cache budget.** At budget=30:
>   92% vs 86% match. At budget=20: 64% vs 50%. Sink preservation also
>   keeps repetition rate stable (0.49-0.50) while window-only repetition
>   climbs (0.49 → 0.68) as cache shrinks.
> - **Quality cliff between budget=30 and budget=20.** With sink+window,
>   budget=30 holds 92% match (cache reduced ~25%). Budget=20 drops to
>   64%. Optimal budget for our 17-prompt × 30-gen suite is ~30.
> - **The Day 6 top-1 metric misses eviction issues** — first-token quality
>   is always 100% because eviction kicks in later. Need multi-token
>   match to surface eviction degradation. Day 10 uses 30-token greedy
>   match against no-evict baseline.
>
> **Day 9 highlights (full archive in `completeness.md`):**
> - **`src/kv_optimization/cache_profiler.py`** built — manual generation
>   loop with `output_attentions=True` at each step, accumulates per-cached-
>   position attention received across all queries.
> - **HEADLINE: 94.6% of cached tokens are "dead"** (receive <1% of their
>   layer's attention budget). Stable across all 17 prompts (std 1.1%).
>   That's *massive* eviction headroom vs Track A's pruning ceiling.
> - **Attention sinks confirmed everywhere.** BOS (position 0) receives
>   **43-81% of total attention** per layer. L2 has 81% concentrated at
>   position 0. Cannot evict position 0 without breaking the model.
> - **Per-layer dead-rate ordering (eviction headroom):** L1, L2, L3 have
>   ~99% dead → safe for aggressive eviction. L7, L8, L9 have ~87% dead
>   (lowest = densest cache, but still 87%!).
> - **Memory framing:** at our sequence lengths (~100 tokens), KV cache is
>   only 0.1% of total. At 4000 tokens it's 5%. **Cache reduction's value
>   at the 1B scale is compute-savings, not memory-savings.** Each
>   eviction also reduces attention compute per generation step.
> - **Day 5/6/7/8 layer rankings cross-validated by Day 9 cache density:**
>   L7-L9 (low dead rate = dense cache) overlap with Day 6's worst single-
>   removal layers (L7=53%, L8=65%, L11=41% exact match). The "load-bearing"
>   layers from Track A are also the ones with the densest cache here.
>
> **Day 8 highlights (full archive in `completeness.md`):**
> - **`src/pruning/benchmark.py`** built — measures params, memory, TTFT,
>   tokens/sec, and speedup for 8 representative pruning configs. Honest
>   timing via `torch.mps.synchronize()` around every timer + warm-up
>   forward to flush kernel-compile overhead.
> - **Per-layer cost:** ~116 MB and ~60.8 M params per decoder layer.
> - **Speed scales roughly linearly with layer count.** Dropping 1 layer
>   buys ~3-4% speedup. Dropping 8 layers (collapsed model) buys 41%.
> - **Pareto chart is empty in the top-right corner.** Best 15-layer
>   config (drop L4) sits at 71% quality / 1.04× speedup — 4pp below
>   the 75% threshold for 4% speed gain. No favorable trade exists.
> - **`TRACK_A_SUMMARY.md`** written to outputs/pruning_results/ as the
>   portfolio-ready Track A close-out.
>
> **Track A final verdict:** for Llama 3.2 1B, the minimum-viable layer
> count under 75% top-1 is 16. The 1B model is at the dense edge of
> what survives layer pruning. The portfolio claim: *the techniques
> (importance scoring × 3, smart pruning, learnable skip) transfer to
> bigger models that have more redundancy; the specific N<16 result is
> a property of Llama 3.2 1B.* Track B (KV cache reduction, Days 9-12)
> operates at finer granularity and should have more headroom.
>
> **Schedule shift:** original plan had Day 5 on Apr 30. Picked back up May 8 and ran Day 5 + Day 6 + Day 7 in same day. End date holds at May 20, 2026.
>
> **Day 7 highlights (full archive in `completeness.md`):**
> - **`src/pruning/smart_pruning.py`** built — Experiment 3 (smart prune by Day 5 ranking) + Experiment 4 (learnable skip weights via distillation).
> - **Smart pruning beats end-removal by avg +9pp.** At 15 layers, drop L12
>   gets 65% exact match vs end-removal's 53%. **Still never crosses 75%
>   threshold** — confirms Day 6: this 1B model is densely packed.
> - **Best 15-layer config across all 4 strategies**: drop L4 alone
>   (Day 6 mid-single, 71% exact match). Smart-prune (drop L12) is second
>   at 65%. Both still under 75%.
> - **Learnable skip experiment** wrapped 16 layers with `SkippableLayer`
>   (`out = w * layer(x) + (1−w) * x`, w trainable via sigmoid). Trained
>   only the 16 skip_logits via KL distillation against frozen 16-layer
>   teacher. **Result: all 16 weights converged in [0.96, 1.00] — no layer
>   was actually skipped.** L1 penalty (λ=0.05) was too weak to overcome
>   the KL gradient. Headline finding stands: the model's own gradient
>   signal says everything is needed.
> - **Within the narrow [0.96, 1.00] band**, the model's preferences
>   partially agree with Day 5 (Spearman ρ = 0.29). L12 has the lowest
>   learned weight (agrees with Day 5 "most redundant"). But L4 has the
>   highest (disagrees — Day 5 said L4 was mid-importance).
>
> **Day 6 highlights (full archive in `completeness.md`):**
> - **`src/pruning/layer_pruner.py`** + **`src/pruning/eval_pruned.py`** built.
>   Pruner uses a ref-only ModuleList swap (no deep-copy) + renumbers
>   `self_attn.layer_idx` so `DynamicCache` indexing survives middle removal.
> - **Headline finding: Llama 3.2 1B is much LESS prunable than Day 5 suggested.**
>   Removing even ONE layer from the end (L15) drops exact-match from 100% → 53%.
>   Below the 75% threshold immediately. There is no "75% with N<16 layers" config.
> - **Day 5 vs Day 6 metric mismatch:** Day 5 said L15 was a "sharpening"
>   layer with low zero-out drop. Day 6 confirms the prob drop is small but
>   shows the top-1 *token* changes for ~half the prompts when L15 is removed.
>   Lesson: averaging probability deltas hides per-prompt token flips.
> - **Mid-stack single removal damage** (L4-L11): exact match drops to 41-71%.
>   Best: L4 (70.6%). Worst: L11 (41.2%). No layer is a "free remove".
> - **Repetition rate** spikes to 60%+ once we drop ≥4 trailing layers — model
>   collapses into token loops. Useful collapse signal for Days 10-12.
> - **Non-monotonic damage:** dropping L13-15 (13 layers) is *worse* than
>   dropping L12-15 (12 layers). Layer interactions are non-additive.
>
> **Day 5 highlights (full archive in `completeness.md`):**
> - **`src/pruning/layer_importance_scorer.py`** built — three independent
>   methods for ranking the 16 layers: (1) logit-lens KL Δ, (2) zero-out
>   top-1 drop, (3) 1 − cos(input, output).
> - All three methods agree: **L0 and L1 are critical** (top-2 in 2/3 methods,
>   top-3 in all three). These are the input-routing layers — don't prune.
> - **L12 is the cleanest pruning candidate** — bottom-3 in 2/3 methods,
>   never in any top-3.
> - L8's Day 4 anomaly (zero_attention(L8) raised Paris) does NOT generalize:
>   L8 is mid-importance (rank 8-11 across methods). Day 4 was prompt-specific.
>
> **Inspector toolkit complete** (Days 1-4): `model_loader.py`, `logit_lens.py`,
> `attention_visualizer.py`, `embedding_explorer.py`, `kv_cache_analyzer.py`,
> `weight_tweaker.py`. All Day 1-4 detail archived in `completeness.md`.
>
> **Day 4 highlights** (full archive in `completeness.md`):
> - **`kv_cache_analyzer.py`** + **`weight_tweaker.py`** built. KV cache = 32 KB
>   per token on 1B; with-cache 2.17× faster than without at 30 tokens.
> - **Headline finding**: `zero_attention(L8)` *raises* `"capital of France is"`
>   → Paris from 48% → **69%**. Strong pre-Day-5 signal that L8 is redundant
>   or counterproductive. `zero_attention(L0)` -20pp; `zero_attention(L15)` -28pp.
> - **MLP noise collapse threshold** sits between σ=0.01 (invisible) and σ=0.1
>   (catastrophic). Sharp transition, no graceful middle ground.
> - `LLMXray.md`: **KV Cache** entry filled in; **MLP** entry expanded with
>   noise-fragility table.
> - `monkeypatching.md` parked for Days 10-12.
>
> **Next session — Day 12 (closes Track B):**
> 1. **Per-layer sizing via monkey-patching** — the deferred Strategy 4
>    from Day 11. Monkey-patch `LlamaAttention.forward()` to use
>    per-layer attention masks. First time we actually need monkey-patching.
> 2. **Combined strategy**: importance b=20 + INT8 quant — measure combined
>    quality and the speed/memory trade
> 3. Compile the Track B comparison table
> 4. Write `TRACK_B_SUMMARY.md` (matches Track A's portfolio writeup pattern)
> 5. Commit: `"Day 12: Track B complete — cache optimization results documented"`
>
> **Carry-in priors from Day 11:**
> - Best Day 11 config: importance b=30 → 92.9% match (marginally better
>   than Day 10's sink+window 92.0%)
> - Importance-vs-recency win widens at tighter budgets and (presumably)
>   on longer sequences
> - INT8 alone costs ~28pp; INT4 catastrophic. Combined eviction+INT8
>   could be even worse — needs testing
> - Per-layer sizing is the most interesting unfinished work; needs
>   monkey-patching

---

## Project Structure

```
llama-xray/
├── CLAUDE.md                    ← This file (tasks + progress)
├── LLMXray.md                   ← Key Concepts Reference (fills in as we learn)
├── README.md                    ← Portfolio-facing writeup
├── requirements.txt
├── notebooks/                   ← Jupyter exploration notebooks
│   ├── 01_model_anatomy.ipynb
│   ├── 02_logit_lens.ipynb
│   ├── 03_attention_maps.ipynb
│   ├── 04_embeddings.ipynb
│   ├── 05_kv_cache.ipynb
│   ├── 06_weight_tweaking.ipynb
│   ├── 07_layer_pruning.ipynb
│   └── 08_cache_optimization.ipynb
├── src/
│   ├── inspector/               ← Core inspection toolkit
│   │   ├── model_loader.py
│   │   ├── layer_inspector.py
│   │   ├── embedding_explorer.py
│   │   ├── attention_visualizer.py
│   │   ├── kv_cache_analyzer.py
│   │   └── logit_lens.py
│   ├── pruning/                 ← Track A
│   │   ├── layer_pruner.py
│   │   ├── layer_importance_scorer.py
│   │   └── eval_pruned.py
│   ├── kv_optimization/         ← Track B
│   │   ├── cache_profiler.py
│   │   ├── token_reducer.py
│   │   └── eval_optimized.py
│   └── eval/                    ← Shared evaluation framework
│       ├── benchmark.py
│       ├── test_prompts.py
│       └── comparison.py
├── outputs/                     ← Visualizations, charts, reports
└── data/
    └── test_prompts.json
```

---

## Dependencies

```
torch
transformers
matplotlib
seaborn
numpy
jupyter
tqdm
scikit-learn
```

Use latest stable PyTorch (2.5+, MPS built in). Python 3.11.x or 3.12.x.

---

## Environment / How to run

**venv lives at `~/venvs/llama-xray/`** (moved out of project folder April 27, 2026 so Cursor's language server doesn't index torch's 1.2GB of files).

| Action | Command |
|---|---|
| Activate venv interactively | `source ~/venvs/llama-xray/bin/activate` |
| Run a Python module directly | `~/venvs/llama-xray/bin/python -m src.inspector.logit_lens` |
| Run a notebook | `~/venvs/llama-xray/bin/jupyter notebook notebooks/02_logit_lens.ipynb` |
| Required env vars (set in `~/.zshrc`) | `PYTORCH_ENABLE_MPS_FALLBACK=1`, `HF_HOME=~/models/hf-weights` |

Inside Cursor: when running notebooks, select kernel `~/venvs/llama-xray/bin/python`.

---

## Test Prompts

Used consistently across all experiments for fair comparison.

```python
TEST_PROMPTS = {
    "factual": [
        "The capital of France is",
        "The chemical formula for water is",
        "The largest planet in our solar system is",
        "The speed of light is approximately",
        "The first president of the United States was",
    ],
    "math": [
        "2 + 2 =",
        "The square root of 144 is",
        "If x = 5, then 2x + 3 =",
    ],
    "code": [
        "def hello_world():\n    print(",
        "for i in range(10):\n    ",
        "import os\nfiles = os.listdir(",
    ],
    "pattern": [
        "The cat sat on the",
        "Once upon a time there was a",
        "Roses are red, violets are",
    ],
    "reasoning": [
        "The sun appears yellow because",
        "Water boils at 100 degrees because",
        "Birds can fly because they have",
    ],
}
```

---

## Daily Task Plan

### Day 1 — Thursday, April 23, 2026 ✅ COMPLETED (finished April 24)
**Phase 0: Mac Mini Environment Bootstrap + Project Setup + Understanding Tokenization**

> Full task checklist archived in `completeness.md`. Summary below is what future-Claude
> needs to know about the state of the project after Day 1.

**State after Day 1:**
- **Environment ready:** pyenv 2.6.27, Python 3.11.9 pinned, venv at **`~/venvs/llama-xray/`** (moved out of project folder on April 27, 2026 to keep Cursor's language server from indexing the 1.2GB venv) with torch 2.11.0 + transformers 5.6.2 + huggingface_hub 1.12.0 + full Jupyter/ML stack installed. MPS verified working. `PYTORCH_ENABLE_MPS_FALLBACK=1` in `~/.zshrc`.
- **HF auth done:** account active, Llama 3.2 1B access approved, Read token `llama-xray-mac-mini` saved. `HF_HOME=~/models/hf-weights` in `~/.zshrc` — all HF models land there (not the default `~/.cache/huggingface/`).
- **Model loaded successfully on MPS:** `~/venvs/llama-xray/bin/python -m src.inspector.model_loader` runs end-to-end. Loads in ~3s from `~/models/hf-weights/` after the initial download. `pick_device()` helper in `src/inspector/model_loader.py` auto-selects MPS → CUDA → CPU.
- **Architecture documented** (committed to memory via `LLMXray.md`): 16 `LlamaDecoderLayer`s stacked. Each layer = `self_attn` (q/k/v/o_proj) + `mlp` (gate/up/down_proj, SwiGLU) + 2 RMSNorms. Plus `embed_tokens` (128256 × 2048) at start and `lm_head` at end. Config: hidden=2048, heads=32, KV heads=8 (Grouped Query Attention), MLP intermediate=8192, vocab=128256, max_position=131072. Total ≈ 1.24B params FP16.
- **Tokenization understood** (write-up in `LLMXray.md`): Llama uses BPE subword tokenization over a 128,256-token vocab. `<|begin_of_text|>` (id 128000) is prepended automatically. Common words are one token (`"Paris"`, `"hello"`); rare words split (`"understanding"` → `["under", "standing"]`, `"pneumonia"` → `["p", "neum", "onia"]`). Leading spaces are part of tokens (` is`, ` the`).
- **Sanity check passed:** generating from `"The capital of France is"` produces coherent English about the Eiffel Tower and Louvre.

**Deliverables checked into the repo:**
- `src/inspector/model_loader.py` — MPS-aware loader with `pick_device()`
- `notebooks/01_model_anatomy.ipynb` — executed end-to-end, outputs saved in the notebook JSON
- `LLMXray.md` — **Transformer** and **Tokenization** sections filled in with concrete numbers
- Committed as `b9a5df2` on `main`

**Understanding goal for Day 1 (met):**
> Able to explain: what a token is, how text becomes numbers, what the model's architecture
> looks like (16 layers stacked), what each layer contains (attention + MLP). Don't need to
> understand HOW they work yet — just WHAT the pieces are.

---

### Day 2 — Friday, April 24, 2026 ✅ COMPLETED
**Phase 1A: Logit Lens + Understanding Attention**

- [x] **Learn: Attention mechanism from scratch**
  - What is Q (Query), K (Key), V (Value)?
  - Analogy: Q = "what am I looking for?", K = "what do I contain?", V = "what info do I give?"
  - How scores are computed: Q × K → softmax → multiply by V
  - What multi-head attention means: 32 smaller attention operations running in parallel
  - Each head can learn a different pattern (grammar, meaning, position, etc.)
  - Write this up in `LLMXray.md`
- [x] Build `logit_lens.py`:
  - Input: prompt string
  - Process: run model with `output_hidden_states=True`
  - At each of the 17 hidden states (embedding + 16 layers):
    - Take the last token's hidden state
    - Project through `model.lm_head` (and `model.model.norm` for proper normalization)
    - Get top-5 predicted words and their probabilities
  - Output: printed table showing layer-by-layer predictions
- [x] Run logit lens on ALL test prompts from the test suite
- [x] **Key observation:** For "The capital of France is" — at which layer does "Paris" first appear?
- [x] **Key observation:** For "2 + 2 =" — at which layer does "4" first appear?
- [x] **Key observation:** Are some prompts "solved" early (layer 4-5) while others need all 16 layers?
- [x] Create visualization: heatmap with layers on X axis, top-5 tokens on Y axis, probability as color
- [x] Save all visualizations to `outputs/logit_lens/`
- [x] **Concept note:** Write the "Attention (Q, K, V)", "Multi-Head Attention", and "Logit Lens" entries in `LLMXray.md`
- [x] Commit: "Day 2: Logit lens built, attention mechanism understood"

**Understanding goal for Day 2:**
> By end of day, you should be able to explain: how attention works (Q×K→scores→softmax→×V),
> why we have multiple heads, and what the logit lens shows you. You should be able to look at
> the logit lens output and say "the model figured out the answer at layer X."

---

### Day 3 — Monday, April 27, 2026 ✅ COMPLETED
**Phase 1B: Attention Visualizer + Embedding Explorer**

- [x] Build `attention_visualizer.py`:
  - Run model with `output_attentions=True`
  - For each layer × head: extract attention matrix (which words attend to which)
  - Generate heatmap: rows = tokens (from), columns = tokens (to), color = attention score
  - Save attention maps for all test prompts
- [x] **Experiment: Head pattern identification**
  - Run 5+ different prompts through the model
  - For each head, look at its attention pattern across all prompts
  - Identify heads by type:
    - "Previous token head" — each token attends to the one before it
    - "First token head" — all tokens attend to the first token
    - "Semantic head" — content words attend to related content words
    - "Position head" — fixed positional pattern regardless of content
  - Document which heads (layer X, head Y) do what
- [x] Build `embedding_explorer.py`:
  - Load embedding table: `model.model.embed_tokens.weight` (128,256 × 2,048)
  - `find_similar(word, top_k=20)` — cosine similarity against all embeddings
  - `compare(word1, word2)` — similarity score between two words
  - `cluster_words(word_list)` — 2D PCA/t-SNE projection of a word group
- [x] **Experiment: Embedding space exploration**
  - Run `find_similar("Python")` — are Java, JavaScript, code nearby?
  - Run `find_similar("sun")` — are moon, star, solar nearby?
  - Run `find_similar("king")` — is queen nearby? (classic word2vec test)
  - Cluster programming terms and plot in 2D
  - Cluster animal terms and plot in 2D
- [x] Save all visualizations to `outputs/attention_maps/` and `outputs/embeddings/`
- [x] **Concept note:** Write the "Embeddings" entry in `LLMXray.md`
- [x] Commit: "Day 3: Attention visualizer + embedding explorer built"

**Understanding goal for Day 3:**
> By end of day, you should be able to explain: what embeddings are (a lookup table where each
> word has a vector of numbers), why similar words have similar vectors (they appeared in similar
> contexts during training), and what attention maps show you (which words are "looking at" which
> other words at each layer).

---

### Day 4 — Wednesday, April 29, 2026 ✅ COMPLETED
**Phase 1C: KV Cache Analyzer + Weight Tweaker**

> Full task checklist + experiment results archived in `completeness.md`.
> Summary below is what future-Claude needs to know about the project state
> after Day 4.

**State after Day 4:**
- **`src/inspector/kv_cache_analyzer.py`** built: `cache_memory_bytes()`,
  `print_memory_breakdown()`, `profile_generation_with_cache()` (logs
  per-step cache size, memory MB, K/V norms per layer, step time),
  `profile_generation_without_cache()`, `compare_speed()`, plus three plot
  helpers. KV cache memory = **32 KB per token** for Llama 3.2 1B
  (= 2 × 16 layers × 8 kv_heads × 64 head_dim × 2 FP16 bytes).
- **`src/inspector/weight_tweaker.py`** built: `WeightTweaker` class with
  `zero_attention(L)`, `add_mlp_noise(L, σ)`, `scale_head(L, H, k)`,
  `swap_layers(a, b)`, `reset()` primitives. Two entry points: script-mode
  (runs predefined Day-4 experiment suite) and `interactive` REPL.
  Snapshot/restore done via `state_dict()` deep clone.
- **transformers 5.x KV cache API**: `past_key_values` is now a
  `DynamicCache` object — access via `pkv.layers[i].keys` /
  `pkv.layers[i].values`, NOT `pkv[i]`.

**Key findings (April 29, 2026 run, prompt = `"The capital of France is"`,
baseline final-layer prediction = `Paris` 48%):**

| Tweak | Final-layer top-1 | Δ vs baseline | Insight |
|---|---|---|---|
| `zero_attention(L0)` | `Paris` (28%) | -20pp | L0 attention does real routing |
| **`zero_attention(L8)`** | **`Paris` (69%)** | **+21pp** | **L8 redundant or harmful — prime pruning candidate** |
| `zero_attention(L15)` | `Paris` (20%) | -28pp | L15 sharpens confidence, doesn't decide |
| `mlp_noise(L5, σ=0.01)` | `Paris` (51%) | invisible | model robust to small noise |
| `mlp_noise(L5, σ=0.1)` | `being` (3%) | catastrophic | **collapse threshold between 0.01 and 0.1** |
| `mlp_noise(L5, σ=1.0)` | `genomes` (1%) | total nonsense | every layer 6-16 frozen on garbage |
| `swap_layers(L2, L14)` | `otope` (1%) | total nonsense | layers can't function out of position |
| `scale_head(L8 H0, ×10)` | `Paris` (46%) | -2pp | one head of 32 is a small lever |

**Headline:** zeroing L8 attention *improves* Paris probability — strong
pre-Day-5 signal that not all layers are equally important, and some may
be mildly counterproductive on certain prompts.

**Speed result:** with-cache vs without-cache on 30-token generation —
**2.17× speedup** (1.58s vs 3.42s). Speedup grows with sequence length
because without-cache is O(n²) per-step and with-cache is O(n).

**Deliverables checked into the repo:**
- `src/inspector/kv_cache_analyzer.py`, `src/inspector/weight_tweaker.py`
- `outputs/kv_cache/{cache_growth,k_norms_heatmap,speed_comparison}.png`
- `outputs/weight_tweaks/results.txt`
- `LLMXray.md` — **KV Cache** entry filled in with formula + memory table
  + speed numbers; **Feed-Forward Network (MLP)** entry expanded with
  noise-fragility findings
- `monkeypatching.md` — parking-lot reference doc for the
  attention-class-replacement approach (planned for Day 10-12)

**Understanding goal for Day 4 (met):**
> Able to explain: what the KV cache is and why it exists, how much memory
> it uses, and what happens when you break different parts of the model.
> Intuitive sense of "L0 routes, L15 sharpens, L8 may be redundant, MLP
> collapse threshold sits around σ=0.05."

---

### Day 5 — Friday, May 8, 2026 ✅ COMPLETED
**Phase 2A: Layer Importance Scoring**

> Full task checklist + experiment results archived in `completeness.md`.
> Summary below is what future-Claude needs to know about the project state
> after Day 5.

- [x] Build `layer_importance_scorer.py` with three scoring methods
- [x] **Method 1 — Logit lens KL Δ** between consecutive layers (FP32 cast
  required — FP16 underflows to NaN during log-prob math)
- [x] **Method 2 — Zero-out impact** — zero attention + MLP for one layer,
  measure drop in baseline top-1 probability
- [x] **Method 3 — Cosine similarity** (1 − cos_sim) of last-token hidden
  state at layer input vs output
- [x] Run all three across 16 layers × 17 test prompts
- [x] Combined importance ranking via min-max normalize then average
- [x] Bar chart with all three methods overlaid → `outputs/pruning_results/layer_importance_scores.png`
- [x] **Key question answered:** methods agree on top-important (L0, L1)
  but DISAGREE on redundancy. Cosine-sim sees L15 as important while
  zero-out sees it as the least important → L15 is a "sharpening" layer
  (transforms vector a lot but doesn't change the top-1 token).
- [x] Commit: "Day 5: Layer importance scoring complete — layers ranked"

**Understanding goal for Day 5:**
> By end of day, you should have a clear ranking of which layers matter most and which are
> redundant. You should understand three different ways to measure layer importance and why
> using multiple methods gives more confidence.

---

### Day 6 — Friday, May 8, 2026 ✅ COMPLETED
**Phase 2B: Sequential Layer Removal — Experiments**

> Full task checklist + experiment results archived in `completeness.md`.
> Summary below is what future-Claude needs to know about the project state
> after Day 6.

- [x] Build `src/pruning/layer_pruner.py` — `LayerPruner` class with `prune()`
  and `reset()`. Reference-only ModuleList swap (no deep copy). Critical fix:
  renumbers `self_attn.layer_idx` so `DynamicCache.layers[idx]` doesn't go
  out-of-range when middle layers are dropped.
- [x] Build `src/pruning/eval_pruned.py` — metrics: exact_match_pct,
  top5_pct, mean_baseline_prob, repetition_rate (1 − unique-ratio in 15
  generated tokens). Coherence proxy via repetition rate; perplexity
  replaced by mean baseline prob (simpler, equivalent for our use).
- [x] Experiment 1 — remove from end progressively (16 down to 8 layers)
- [x] Experiment 2 — single-layer middle removal (L4-L11, one at a time)
- [x] **Quality threshold finding:** model never maintains 75% exact-match
  with any layer removed. Drops below threshold at the very first removal.
- [x] Commit: "Day 6: Sequential layer removal experiments — end and middle"

---

### Day 7 — Friday, May 8, 2026 ✅ COMPLETED
**Phase 2C: Smart Pruning + Skip Connections**

> Full task checklist + experiment results archived in `completeness.md`.
> Summary below is what future-Claude needs to know about the project state
> after Day 7.

- [x] **Experiment 3 — Smart pruning (least-important first via Day 5 ranking):**
  removal order [12, 7, 6, 9, 8, 4, 5, 10, 13, 11, 15, 2, 14, 3, 1, 0].
  Beats end-removal at most layer counts (avg +9pp), peak gain +29pp at 13
  layers remaining. Never crosses 75% threshold.
- [x] **Experiment 4 — Learnable skip weights via distillation:**
  built `SkippableLayer` wrapper with `out = sigmoid(skip_logit) * layer(x)
  + (1 - sigmoid(skip_logit)) * x`. Trained only the 16 `skip_logit`
  parameters (rest frozen) for 30 steps via KL distillation against the
  frozen 16-layer teacher. λ_L1 = 0.05 was too weak — final weights all
  in [0.96, 1.00], no layer skipped. Spearman ρ vs Day 5 = 0.29.
- [x] Strategy comparison plot at `outputs/pruning_results/day7_strategies_compared.png`
  showing all 4 approaches on one axis.
- [x] Commit: "Day 7: Smart pruning + skip connections — best strategy identified"

---

### Day 8 — Friday, May 8, 2026 ✅ COMPLETED — TRACK A CLOSED
**Phase 2D: Pruning Results Analysis + Speed Benchmarks**

> Full task checklist + benchmark results archived in `completeness.md`.
> Track A summary writeup at `outputs/pruning_results/TRACK_A_SUMMARY.md`.

- [x] Build `src/pruning/benchmark.py` — measures params, memory, TTFT,
  generation time, and speedup for 8 representative pruning configs
- [x] Honest timing via `torch.mps.synchronize()` + warm-up forward
- [x] Per-layer cost measured: ~116 MB / ~60.8 M params per decoder layer
- [x] Speed scaling: ~3-4% speedup per layer dropped (linear)
- [x] Pareto-style scatter (quality × speedup) — empty in the top-right
  quadrant; no favorable Pareto-frontier trade exists for Llama 3.2 1B
- [x] **Track A reframed result:** minimum viable layer count under 75%
  exact match is 16. The 1B model is densely packed. Best 15-layer
  config (drop L4 alone) sits at 71% / 1.04× speedup.
- [x] Track A summary written to `TRACK_A_SUMMARY.md` (portfolio-ready)
- [x] Commit: "Day 8: Track A complete — pruning results analyzed and documented"

**Track A deliverable (reframed):**
> "Llama 3.2 1B has no removable layers under the 75% top-1 threshold.
> The 1B model is at the dense edge of layer-pruning tolerance. We
> characterized the redundancy structure: L0/L1/L4 are load-bearing
> (input routing + load-bearing mid layer), L12 is the most skippable
> (3 independent methods agree). Smart pruning beats naive end-removal
> by avg +9pp but never crosses the threshold. Learnable skip via
> distillation could not coerce the model to skip any layer.
> The methodology transfers directly to larger models which are known
> to have more removable redundancy."

---

### Day 9 — Tuesday, May 19, 2026 ✅ COMPLETED — Track B opened
**Phase 3A: KV Cache Profiling**

> Full task checklist + experiment results archived in `completeness.md`.
> Summary below is what future-Claude needs to know about the project state
> after Day 9.

- [x] Build `src/kv_optimization/cache_profiler.py` — manual generation
  loop with `output_attentions=True`, accumulates per-cached-position
  attention received across all queries
- [x] 17 prompts × 100 tokens generated, ~67 sec total compute on MPS
- [x] **Dead-token rate** (cached tokens receiving <1% of layer's attention):
  **94.6% mean across prompts** (std 1.1%, range 92.3-96.7%)
- [x] **Attention sink confirmed** — BOS position receives 43-81% of all
  attention per layer. L2 most concentrated (81.4% on position 0).
- [x] **Per-layer dead rate ordered** — L1 (98.9%), L2 (98.8%), L3 (98.4%)
  have nearly-empty caches. L7 (87.8%), L8 (87.3%), L9 (88.1%) have
  the densest caches — but still ≥87% dead.
- [x] Memory breakdown at 50/100/200/500/1000/2000/4000 token seq lengths
  — at 4000 tokens, KV cache is 5% of total memory (model weights dominate)
- [x] Saved canonical attention array to `outputs/cache_profiles/day9_canonical_attention.npz`
  for Day 10 to consume without re-running the profiler
- [x] Commit: "Day 9: KV cache profiled — baseline measurements established"

---

### Day 10 — Tuesday, May 19, 2026 ✅ COMPLETED
**Phase 3B: Attention Sink Analysis + Basic Eviction**

> Full task checklist + experiment results archived in `completeness.md`.
> Summary below is what future-Claude needs to know about the project state
> after Day 10.

- [x] **Attention Sink concept** — confirmed structural; BOS receives 49-80%
  of attention per layer across all 17 prompts (std 2-6%). Must preserve
  position 0 in any eviction strategy.
- [x] Build `src/kv_optimization/token_reducer.py` with `trim_cache()` —
  mutates `past_kv.layers[L].keys/values` between generation steps. No
  monkey-patching of attention class yet — between-step trimming works
  because DynamicCache stores K/V as plain tensors we can re-slice.
- [x] **Strategy 1 (window-only)** at budgets 100/30/20/10 → 100/86/50/22%
  match. Repetition rate climbs from 0.49 → 0.68 as cache shrinks.
- [x] **Strategy 2 (sink + window, sink=4)** at same budgets → 100/92/64/24%
  match. Repetition rate stays at 0.49-0.50 — sink preserves coherence.
- [x] **Sink+window beats window-only at every budget.** Sink-preservation
  is worth ≈ +6pp match at budget=30, +13pp at budget=20.
- [x] **Day 6 top-1 metric is blind to eviction** — first-token quality
  always 100% because eviction kicks in after first gen step. Day 10
  uses 30-token greedy match against no-evict baseline as the proper metric.
- [x] Commit: "Day 10: Attention sinks verified, basic eviction strategies tested"

---

### Day 11 — Tuesday, May 19, 2026 ✅ COMPLETED
**Phase 3C: Advanced Eviction + Cache Quantization**

> Full task checklist + experiment results archived in `completeness.md`.
> Summary below is what future-Claude needs to know about the project state
> after Day 11.

- [x] **Strategy 3 — Importance eviction:** `ImportanceState` class tracks
  cumulative attention received per cached position across all queries.
  When cache > budget, keep sink[:4] + top-importance middle + recent[:5].
  At b=30: 92.9% match (vs sink+window 92.0%, +0.9pp). At b=20: 67.1%
  (vs 63.5%, +3.6pp). Benefit grows at tighter budgets.
- [x] **Strategy 4 — Per-layer sizing — DEFERRED.** Crashes on
  between-step trim because transformers shares `cache_position` across
  all 16 layers; per-layer different cache lengths cause
  attention-mask shape mismatch. **First technique that genuinely
  requires monkey-patching `LlamaAttention.forward()`** — moved to Day 12.
- [x] **Strategy 5 — KV cache quantization:** symmetric per-tensor
  round-trip simulation (FP16 → INT → FP16). INT8: 72.2% match
  (significant cost). INT4: 16.9% match (catastrophic). Repetition rate
  spikes at INT4 (0.58 vs baseline 0.49) — same coherence-loss pattern as
  Day 10 sink-loss.
- [x] Commit: "Day 11: Advanced eviction + cache quantization implemented"

---

### Day 12 — Friday, May 15, 2026
**Phase 3D: Combined Optimization + Track B Results**

- [ ] Combine the best strategies from Days 10-11:
  - Best eviction strategy (probably sink + window) +
  - Per-layer cache sizing +
  - Cache quantization (INT8)
  - Measure combined effect
- [ ] Build `eval_optimized.py`:
  - Run all test prompts with combined optimizations
  - Compare against Day 9 baselines:
    - Cache memory reduction %
    - Quality score %
    - Generation speed change
    - Tokens computed saved %
- [ ] Build comparison table:
  ```
  | Strategy              | Cache Memory | Quality | Compute Saved |
  |-----------------------|-------------|---------|---------------|
  | Baseline (full cache) | 100%        | 100%    | 0%            |
  | Window only (50%)     | 50%         | ??%     | ??%           |
  | Sink + Window (50%)   | 52%         | ??%     | ??%           |
  | Importance eviction   | ??%         | ??%     | ??%           |
  | Per-layer sizing      | ??%         | ??%     | ??%           |
  | INT8 quantized cache  | 50%         | ??%     | 0%            |
  | Combined best         | ??%         | ??%     | ??%           |
  ```
- [ ] Target check: did we achieve 5%+ token compute reduction? If not, iterate.
- [ ] Write Track B summary
- [ ] Save everything to `outputs/cache_profiles/`
- [ ] Commit: "Day 12: Track B complete — cache optimization results documented"

**Track B deliverable:**
> "KV cache optimizations reduce memory by X% and token compute by Y%
> while maintaining Z% answer quality. The most effective strategy is [strategy].
> Attention sinks in the first 4 tokens are critical to preserve.
> [X]% of cached tokens receive less than 1% attention and can be safely evicted."

---

### Day 13 — Monday, May 18, 2026
**Phase 4A: Interactive Dashboard**

- [ ] Build a visualization dashboard (Streamlit or HTML/React):
  - **Tab 1: Model Anatomy** — architecture diagram, layer/head/embedding counts, click a layer to see its weights and attention patterns
  - **Tab 2: Logit Lens Explorer** — input a prompt, see predictions emerge layer by layer (green when correct appears, red when wrong, slider for specific layers)
  - **Tab 3: Attention Maps** — select layer and head, see attention heatmap, toggle between test prompts, highlight head types
  - **Tab 4: Embedding Space** — type a word, see nearest neighbors, 2D scatter plot of word clusters, compare two words
  - **Tab 5: Layer Pruning Results** — slider to remove layers, quality score updates in real time, side-by-side predictions, critical (red) vs removable (green) highlights
  - **Tab 6: KV Cache Optimizer** — visualize cache growth, toggle eviction strategies, before/after comparison
  - **Tab 7: Weight Playground** — select a layer + component (attention/MLP), slider to add noise/scale/zero, predictions change live
- [ ] Commit: "Day 13: Interactive dashboard built"

---

### Day 14 — Tuesday, May 19, 2026
**Phase 4B: Portfolio Writeup + README**

- [ ] Write comprehensive README.md:
  ```markdown
  # Llama X-Ray: LLM Interpretability & Optimization

  ## What This Is
  A hands-on research project exploring how transformer LLMs work internally,
  and whether we can make them smaller and faster without losing quality.

  ## Key Findings
  ### Layer Pruning
  - [Summary of Track A results]
  - [Which layers matter, which don't, and why]
  - [Minimum viable layer count for 75% quality]

  ### KV Cache Optimization
  - [Summary of Track B results]
  - [Best eviction strategy and why]
  - [Memory and compute savings achieved]

  ### What I Learned About Transformers
  - [How knowledge emerges layer by layer]
  - [What different attention heads specialize in]
  - [How embeddings encode word relationships]
  - [Why early, middle, and late layers have different roles]

  ## Interactive Demo
  [Link to dashboard]

  ## Techniques Used
  - Logit lens (per-layer prediction inspection)
  - Attention pattern analysis (head type identification)
  - Mechanistic interpretability (weight tweaking, layer ablation)
  - KV cache profiling and optimization
  - Layer importance scoring (3 methods)
  - Cache eviction strategies (4 approaches)

  ## Transferability
  All techniques demonstrated on Llama 3.2 1B transfer directly to
  larger models. The code is model-agnostic — swap in any HuggingFace
  transformer model and the same analysis runs.

  ## Built With
  Python, PyTorch, Hugging Face Transformers, Matplotlib, [Streamlit]
  ```
- [ ] Create summary visualizations for README:
  - One hero image: logit lens showing "Paris" emerging from "capital of France"
  - One chart: pruning curve (layers vs quality)
  - One chart: cache optimization comparison
- [ ] Review all code — clean up, add docstrings, ensure everything runs
- [ ] Commit: "Day 14: README and portfolio writeup complete"

---

### Day 15 — Wednesday, May 20, 2026
**Phase 4C: Final Review + Future Work**

- [ ] End-to-end test: clone repo fresh, install deps, run all notebooks, verify outputs
- [ ] Review CLAUDE.md and LLMXray.md — ensure all observations and concepts are documented
- [ ] Write "Future Work" section in README:
  - Scale to Llama 3.2 3B — do same layers matter?
  - Scale to Gemma 4 E2B — how does hybrid attention change things?
  - Compare pruning results across model sizes
  - Implement more advanced techniques (activation patching, probing classifiers, circuit analysis)
  - Package toolkit as a pip-installable library
- [ ] Final commit: "Day 15: Project complete — llama-xray v1.0"
- [ ] Push to GitHub
- [ ] Update portfolio / LinkedIn

---

## Observations

> This section is updated as we learn things. Every interesting finding goes here.

### Layer Behavior Observations
| Layer(s) | Observation | Date |
|----------|-------------|------|
| L0, L1 | Top-2 importance across all three Day-5 methods. Input-routing — never prune. | May 8, 2026 |
| L12 | Bottom-3 redundancy in 2/3 methods. Best pruning candidate by combined score. | May 8, 2026 |
| L15 | High cosine-sim importance (0.48 = big vector transform) but lowest zero-out drop (0.055). "Sharpening" layer — changes the vector but not the top-1 token. Confirms Day 4 (-28pp on top-1 prob, but Paris still top). | May 8, 2026 |
| L8 | Day 4 anomaly (zero_attention(L8) raised Paris) does NOT generalize. L8 is mid-importance: rank 8-11 across the 3 Day-5 methods averaged over 17 prompts. The Day 4 finding was prompt-specific. | May 8, 2026 |

### Attention Head Observations
| Layer | Head | Pattern Type | Notes | Date |
|-------|------|-------------|-------|------|
| — | — | — | *Not yet started* | — |

### Pruning Observations
| Layers Removed | Quality Impact | Notes | Date |
|----------------|---------------|-------|------|
| (Day 5 priors only) | — | Combined ranking most-important → most-redundant: L0 > L1 > L3 > L14 > L2 > L15 > L11 > L13 > L10 > L5 > L4 > L8 > L9 > L6 > L7 > L12. Day 6 will turn this ranking into actual physical removal experiments. | May 8, 2026 |
| L15 (1 from end) | 100% → 53% exact-match | Day 5 said L15 was "sharpener" with -0.055 zero-out drop; Day 6 shows that across 17 prompts the top-1 *token* changes for ~half of them. Avg-prob metric hid per-prompt token flips. | May 8, 2026 |
| L14-15 (2 from end) | 53% → 24% | Steep drop. Repetition rate climbs from 17% → 41%. | May 8, 2026 |
| L13-15 (3 from end) | 24% → 12% | Non-monotonic with L12-15 (which gives 24%). Layer interactions are non-additive. | May 8, 2026 |
| L9-15 (7 from end) | → 0% | Total collapse. Repetition rate plateaus around 60% (loops). | May 8, 2026 |
| L4 alone (mid) | 100% → 71% | Best single-mid removal. Closest to threshold but still under. | May 8, 2026 |
| L11 alone (mid) | 100% → 41% | Worst single-mid removal. | May 8, 2026 |
| L5, L7 alone (mid) | 100% → 53% | Second-worst tier — both at 53% exact-match. | May 8, 2026 |
| Smart drop L12 (15 layers) | 100% → 65% | Day 7 Exp 3. Beats end-removal (53% drop L15) by +12pp. Confirms Day 5 ranked L12 as most-redundant. Still under 75%. | May 8, 2026 |
| Smart drop L12,L7 (14 layers) | 100% → 41% | Day 7. Beats end-removal (24%) by +18pp. | May 8, 2026 |
| Smart drop L12,L7,L6 (13 layers) | 100% → 41% | Day 7. Largest gain over end-removal (+29pp). | May 8, 2026 |
| Learnable skip (Day 7 Exp 4) | All weights in [0.96, 1.00] | Distillation with λ_L1=0.05 for 30 steps. No layer skipped. L1 too weak; KL gradient pulls all weights up. **Still informative**: lowest learned weight is L12 (agrees with Day 5 "most redundant"); highest is L4 (disagrees with Day 5). Spearman ρ vs Day 5 = 0.29. | May 8, 2026 |

### KV Cache Observations
| Finding | Impact | Date |
|---------|--------|------|
| 94.6% of cached tokens are "dead" (<1% attention received) across 17 prompts × 100 generated tokens | Massive eviction headroom vs Track A's pruning ceiling | May 19, 2026 |
| BOS token receives 43-81% of attention per layer (sink); L2 most concentrated at 81% | Cannot evict position 0. Day 10 strategy must always preserve sink. | May 19, 2026 |
| Per-layer dead rate: L1=98.9%, L2=98.8%, L3=98.4% (highest) vs L7=87.8%, L8=87.3%, L9=88.1% (lowest) | Per-layer cache sizing (Day 11): tiny window for L1-L3, bigger window for L7-L9 | May 19, 2026 |
| Cross-validation with Track A: L7-L9 (densest cache) overlap with Day 6's worst single-removal layers (L7=53%, L8=65%) | "Load-bearing" layers from Track A are also the dense-cache ones in Track B — consistent signal | May 19, 2026 |
| Memory framing: KV cache is 0.1% of total memory at 100 tokens; 5% at 4000 tokens | At 1B scale, Track B's value is compute-savings per gen step, not memory-savings | May 19, 2026 |
| Attention sink universal — BOS gets 49-80% per layer across all 17 prompts (std 2-6%) | Structural finding, not prompt-specific. Sink preservation is mandatory for eviction. | May 19, 2026 |
| Sink+window @ budget=30 holds 92% match; window-only @ same budget holds 86% | Sink preservation worth ≈ +6pp at budget=30, +13pp at budget=20 | May 19, 2026 |
| Repetition rate stays at 0.49 with sink-preserved eviction; climbs to 0.68 without sink | Sink is a coherence anchor — losing it sends model into token loops | May 19, 2026 |
| Quality cliff at budget=20 — drops from 92% (b=30) → 64% (b=20) → 24% (b=10) | "Recent window" is doing real work beyond the first ~20 positions; sub-20 budgets break things | May 19, 2026 |
| Importance eviction (cumulative-attention scored) marginally beats recency-window: +0.9pp at b=30, +3.6pp at b=20 | Smarter middle-token selection has bounded win on 40-token caches; expected to widen on longer sequences | May 19, 2026 |
| Per-layer cache sizing CRASHES with between-step trim due to shared cache_position across all 16 layers | First technique that genuinely needs monkey-patching `LlamaAttention.forward()` — Day 12 work | May 19, 2026 |
| INT8 cache quantization costs 28pp on 30-token gen (72% match). INT4 catastrophic (17%) | Quantization is not free at the 1B scale; INT4 hits the coherence collapse threshold | May 19, 2026 |

### Weight Tweaking Observations
| Tweak | Effect | Date |
|-------|--------|------|
| — | *Not yet started* | — |

---

## Blockers

| Blocker | Status | Resolution | Date |
|---------|--------|------------|------|
| — | — | *None yet* | — |

---

## Key Decisions Log

| Decision | Reasoning | Date |
|----------|-----------|------|
| Use Llama 3.2 1B over Gemma 4 | Clean standard transformer, no hybrid attention tricks. Better for learning fundamentals first. | April 9, 2026 |
| Use FP16 over quantized | Need full precision to properly inspect weights, hidden states, and KV cache values. Quantized models lose detail. | April 9, 2026 |
| 15 working days, weekdays only | Sustainable pace, time for concepts to sink in between sessions. | April 9, 2026 |
| Migrate project from Intel MacBook to Mac Mini M4 | M4 (16GB unified memory, 10-core GPU via MPS) is ~3-5x faster than Intel i5 CPU — unlocks faster iteration on Day 6-8 pruning sweep and longer sequences in Day 9-10 KV profiling. | April 23, 2026 |
| Restart 15-day schedule on April 23, 2026 | Day 1 on MacBook only reached partial environment setup. Fresh start on Mac Mini aligns the day-by-day plan with actual execution. New end date: May 13, 2026. | April 23, 2026 |

---

## Completeness So Far — MacBook Pro

> **Purpose:** When this repo is pulled onto the Mac Mini server, use this section to know
> what was done on the MacBook Pro and what needs to be redone/set up on the Mac Mini.

### Code & Files Created (in repo, no action needed)
- [x] Full project directory structure (`src/`, `notebooks/`, `data/`, `outputs/`)
- [x] `requirements.txt` — all Python dependencies listed
- [x] `src/inspector/model_loader.py` — loads Llama 3.2 1B with introspection flags
- [x] `notebooks/01_model_anatomy.ipynb` — tokenization exploration + model anatomy notebook
- [x] `data/test_prompts.json` — all test prompts (factual, math, code, pattern, reasoning)
- [x] `__init__.py` files for all packages (`src/`, `inspector/`, `pruning/`, `kv_optimization/`, `eval/`)
- [x] `.gitignore` — excludes venv, __pycache__, outputs, .DS_Store, etc.

### Environment Setup — Mac Mini M4 (what actually needs to happen here)
**Do NOT copy the Intel pins — use latest PyTorch with MPS support.**
- [ ] `brew install pyenv` (verify `pyenv --version`)
- [ ] `pyenv install 3.11.9` then `pyenv local 3.11.9` (Python 3.12.x also fine; avoid 3.13+ for now)
- [ ] `mkdir -p ~/venvs && python3 -m venv ~/venvs/llama-xray && source ~/venvs/llama-xray/bin/activate` (venv lives outside project folder so Cursor's language server doesn't index torch)
- [ ] `pip install -r requirements.txt` — this picks up current stable PyTorch (2.5+) with MPS built in. Do NOT pin to 2.2.2.
- [ ] Verify MPS: `python -c "import torch; print(torch.backends.mps.is_available())"` → must print `True`
- [ ] Add `export PYTORCH_ENABLE_MPS_FALLBACK=1` to `~/.zshrc`, then `source ~/.zshrc`
- [ ] Update `src/inspector/model_loader.py` to auto-pick MPS (use a `pick_device()` helper: MPS → CUDA → CPU)

### Accounts & Access (MUST do before running)
- [ ] **Hugging Face account** — needed to download Llama 3.2 1B
  - Request access at: https://huggingface.co/meta-llama/Llama-3.2-1B
  - Meta must approve (usually quick)
- [ ] **Hugging Face CLI login** — `huggingface-cli login`
  - Create a "Read" token at: https://huggingface.co/settings/tokens
  - This token may already work if same HF account is used on Mac Mini

### Model Download (MUST do on Mac Mini)
- [ ] **Llama 3.2 1B FP16** (~2GB) — downloads automatically on first run of notebook or model_loader.py
  - Stored in `~/.cache/huggingface/` (not in repo)
  - Requires HF login + Llama access approval

### Mac Mini Transition Status (as of April 24, 2026) — ALL DONE ✅
- [x] Mac Mini environment bootstrapped (pyenv, Python 3.11.9, venv, deps)
- [x] MPS device wiring added to `model_loader.py`
- [x] HF CLI logged in on Mac Mini
- [x] Llama 3.2 1B model downloaded into `~/.cache/huggingface/`
- [x] Notebook `01_model_anatomy.ipynb` executed end-to-end with outputs saved
- [x] Transformer + Tokenization concepts filled in `LLMXray.md`
- [x] Day 1 tasks fully completed on Mac Mini

---

## How to Update This File

After completing any task:
1. Check off the task checkbox: `- [ ]` → `- [x]`
2. Update **Current Status** section at the top
3. Add any interesting findings to **Observations** tables
4. Add any problems to **Blockers** table
5. When you learn a new concept, write it up in `LLMXray.md` (not here)
6. Commit the updated CLAUDE.md with a descriptive message
7. **After every commit, append today's section to `monkeypatchknowledge.md`** — capture what we learned that's relevant to the future Day 10-12 attention monkey-patching work (tensor shapes, layer importance, cache plumbing, debug tools). The lens is: *what would the future patch-writer want to know about today's experiments?*

This file is the single source of truth for the project's progress and learnings.
