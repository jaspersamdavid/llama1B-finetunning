# CLAUDE.md — llama-xray

> **Living document. Update after every completed task.**
> Key Concepts Reference lives in `LLMXray.md` — fill it in as concepts are learned.

---

## Project Overview

**Name:** llama-xray
**Model:** Llama 3.2 1B (FP16, ~2GB)
**Machine:** Mac Mini M4 — 10-core CPU, 10-core GPU, 16GB unified memory (PyTorch MPS backend for GPU acceleration)
**Duration:** 15 working days (April 23 – May 13, 2026, excluding weekends)
**Repository:** `llama-xray/`

### Goals

1. **Understand how LLMs work — top to bottom, explained simply.** No prior ML knowledge required. Concepts are written up in `LLMXray.md` as we learn them.
2. **Track A — Layer Pruning:** find the minimum number of layers that maintains 75%+ quality, and understand which layers matter.
3. **Track B — KV Cache Optimization:** reduce token compute by 5%+ for equivalent output quality.
4. **Build a reusable inspection toolkit:** logit lens, attention maps, embedding explorer, KV cache profiler, weight tweaker.
5. **Portfolio piece:** interactive dashboard + written report, techniques that transfer to 3B/7B/70B models.

---

## Current Status

> **Last updated:** April 29, 2026 (end of Day 4)
> **Currently working on:** Day 4 complete ✅. Next session: Day 5 — Layer Importance Scoring (Track A — Pruning, Phase 2A).
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
> **Next session — Day 5 (Thursday April 30, 2026):**
> 1. Build `src/pruning/layer_importance_scorer.py` with three scoring methods:
>    (1) logit-lens delta, (2) zero-out impact, (3) cosine similarity input vs
>    output
> 2. Run all three across all 16 layers + all test prompts; produce a unified
>    ranked importance list
> 3. Visualize: bar chart of per-layer importance with all three methods overlaid
> 4. Save to `outputs/pruning_results/layer_importance_scores.png`
> 5. Key question: do all three methods agree on which layers are
>    important/redundant?
> 6. Commit: `"Day 5: Layer importance scoring complete — layers ranked"`

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

### Day 2 — Friday, April 24, 2026
**Phase 1A: Logit Lens + Understanding Attention**

- [ ] **Learn: Attention mechanism from scratch**
  - What is Q (Query), K (Key), V (Value)?
  - Analogy: Q = "what am I looking for?", K = "what do I contain?", V = "what info do I give?"
  - How scores are computed: Q × K → softmax → multiply by V
  - What multi-head attention means: 32 smaller attention operations running in parallel
  - Each head can learn a different pattern (grammar, meaning, position, etc.)
  - Write this up in `LLMXray.md`
- [ ] Build `logit_lens.py`:
  - Input: prompt string
  - Process: run model with `output_hidden_states=True`
  - At each of the 17 hidden states (embedding + 16 layers):
    - Take the last token's hidden state
    - Project through `model.lm_head` (and `model.model.norm` for proper normalization)
    - Get top-5 predicted words and their probabilities
  - Output: printed table showing layer-by-layer predictions
- [ ] Run logit lens on ALL test prompts from the test suite
- [ ] **Key observation:** For "The capital of France is" — at which layer does "Paris" first appear?
- [ ] **Key observation:** For "2 + 2 =" — at which layer does "4" first appear?
- [ ] **Key observation:** Are some prompts "solved" early (layer 4-5) while others need all 16 layers?
- [ ] Create visualization: heatmap with layers on X axis, top-5 tokens on Y axis, probability as color
- [ ] Save all visualizations to `outputs/logit_lens/`
- [ ] **Concept note:** Write the "Attention (Q, K, V)", "Multi-Head Attention", and "Logit Lens" entries in `LLMXray.md`
- [ ] Commit: "Day 2: Logit lens built, attention mechanism understood"

**Understanding goal for Day 2:**
> By end of day, you should be able to explain: how attention works (Q×K→scores→softmax→×V),
> why we have multiple heads, and what the logit lens shows you. You should be able to look at
> the logit lens output and say "the model figured out the answer at layer X."

---

### Day 3 — Monday, April 27, 2026
**Phase 1B: Attention Visualizer + Embedding Explorer**

- [ ] Build `attention_visualizer.py`:
  - Run model with `output_attentions=True`
  - For each layer × head: extract attention matrix (which words attend to which)
  - Generate heatmap: rows = tokens (from), columns = tokens (to), color = attention score
  - Save attention maps for all test prompts
- [ ] **Experiment: Head pattern identification**
  - Run 5+ different prompts through the model
  - For each head, look at its attention pattern across all prompts
  - Identify heads by type:
    - "Previous token head" — each token attends to the one before it
    - "First token head" — all tokens attend to the first token
    - "Semantic head" — content words attend to related content words
    - "Position head" — fixed positional pattern regardless of content
  - Document which heads (layer X, head Y) do what
- [ ] Build `embedding_explorer.py`:
  - Load embedding table: `model.model.embed_tokens.weight` (128,256 × 2,048)
  - `find_similar(word, top_k=20)` — cosine similarity against all embeddings
  - `compare(word1, word2)` — similarity score between two words
  - `cluster_words(word_list)` — 2D PCA/t-SNE projection of a word group
- [ ] **Experiment: Embedding space exploration**
  - Run `find_similar("Python")` — are Java, JavaScript, code nearby?
  - Run `find_similar("sun")` — are moon, star, solar nearby?
  - Run `find_similar("king")` — is queen nearby? (classic word2vec test)
  - Cluster programming terms and plot in 2D
  - Cluster animal terms and plot in 2D
- [ ] Save all visualizations to `outputs/attention_maps/` and `outputs/embeddings/`
- [ ] **Concept note:** Write the "Embeddings" entry in `LLMXray.md`
- [ ] Commit: "Day 3: Attention visualizer + embedding explorer built"

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

### Day 5 — Wednesday, April 29, 2026
**Phase 2A: Layer Importance Scoring**

- [ ] Build `layer_importance_scorer.py` with three scoring methods:
- [ ] **Method 1 — Logit lens delta:**
  - For each layer: measure KL divergence between prediction at layer N vs layer N-1
  - High delta = this layer changed the prediction a lot = important
  - Low delta = this layer barely changed anything = candidate for removal
  - Run on all test prompts, average the scores
  - Output: ranked list of layers by importance
- [ ] **Method 2 — Zero-out impact:**
  - For each layer (0-15): temporarily zero out ALL weights in that layer
  - Run all test prompts and measure quality drop (exact match, top-5 accuracy)
  - Restore weights, move to next layer
  - Output: ranked list — "zeroing out layer X causes Y% quality drop"
- [ ] **Method 3 — Cosine similarity (input vs output):**
  - For each layer: measure cosine similarity between the layer's input and output
  - If input ≈ output (cosine sim > 0.95), the layer is doing almost nothing
  - If input ≠ output (cosine sim < 0.80), the layer is transforming the representation significantly
  - Output: per-layer cosine similarity scores
- [ ] Combine all three methods into a unified importance ranking
- [ ] Visualize: bar chart showing importance score per layer (all three methods overlaid)
- [ ] Save to `outputs/pruning_results/layer_importance_scores.png`
- [ ] **Key question to answer:** Do all three methods agree on which layers are important/unimportant?
- [ ] Commit: "Day 5: Layer importance scoring complete — layers ranked"

**Understanding goal for Day 5:**
> By end of day, you should have a clear ranking of which layers matter most and which are
> redundant. You should understand three different ways to measure layer importance and why
> using multiple methods gives more confidence.

---

### Day 6 — Thursday, April 30, 2026
**Phase 2B: Sequential Layer Removal — Experiments**

- [ ] Build `layer_pruner.py`:
  - Function: `remove_layers(model, layer_indices)` — physically removes layers from the model
  - Function: `skip_layer(model, layer_index)` — adds a bypass that skips the layer
  - After removal/skip, re-wire so remaining layers connect properly
- [ ] Build `eval_pruned.py`:
  - Run all test prompts through a pruned model
  - Metrics:
    - **Exact match:** Does "capital of France is" still produce "Paris" as top-1?
    - **Top-5 accuracy:** Is the correct answer anywhere in top-5?
    - **Perplexity:** How "surprised" is the model by correct continuations?
    - **Coherence score:** Generate 50 tokens — is the output grammatical English?
    - **Code completion:** Does `def hello_world():\n    print(` still produce valid Python?
  - Output: score table per prompt category (factual, math, code, pattern, reasoning)
- [ ] **Experiment 1 — Remove from the end (one at a time):**
  - 16 layers → eval → score
  - 15 layers (remove layer 15) → eval → score
  - 14 layers (remove layers 14-15) → eval → score
  - Continue down to 8 layers
  - Record all scores
- [ ] **Experiment 2 — Remove from the middle:**
  - Keep layers 0-3 (early features) and layers 12-15 (final output)
  - Remove layers 4-11 one at a time
  - Record scores
- [ ] Document: at what point does quality drop below 75%?
- [ ] Commit: "Day 6: Sequential layer removal experiments — end and middle"

---

### Day 7 — Friday, May 1, 2026
**Phase 2C: Smart Pruning + Skip Connections**

- [ ] **Experiment 3 — Remove least important first:**
  - Use importance ranking from Day 5
  - Remove layers in order of least importance
  - After each removal, re-evaluate
  - This should give the best quality-retention curve
- [ ] **Experiment 4 — Layer skipping with learnable weights:**
  - Instead of removing layers, add a skip weight (0.0 = skip, 1.0 = use)
  - Implement `SkippableLayer` wrapper:
    ```python
    class SkippableLayer(nn.Module):
        def __init__(self, original_layer):
            self.layer = original_layer
            self.skip_weight = nn.Parameter(torch.tensor(1.0))
        def forward(self, x):
            return self.skip_weight * self.layer(x) + (1 - self.skip_weight) * x
    ```
  - Wrap all 16 layers with SkippableLayer
  - Train only the skip_weights on a small dataset (freeze everything else)
  - See which layers the model learns to skip on its own
  - Compare with manual importance scores — do they agree?
- [ ] Compare all pruning approaches:
  - Remove from end vs remove from middle vs remove least important vs learnable skip
  - Which gives best quality at each layer count?
- [ ] Commit: "Day 7: Smart pruning + skip connections — best strategy identified"

---

### Day 8 — Monday, May 4, 2026
**Phase 2D: Pruning Results Analysis + Speed Benchmarks**

- [ ] For each pruning configuration that maintains 75%+ quality:
  - Measure inference speed (tokens per second)
  - Measure model memory usage
  - Measure time-to-first-token
- [ ] Build comparison table:
  ```
  | Layers | Config | Quality % | Speed (tok/s) | Memory (MB) | Speedup |
  |--------|--------|-----------|---------------|-------------|---------|
  | 16     | Full   | 100%      | X             | Y           | 1.0x    |
  | 15     | -L15   | ??%       | X             | Y           | ?.?x    |
  | 14     | -L14,15| ??%       | X             | Y           | ?.?x    |
  | ...    | ...    | ...       | ...           | ...         | ...     |
  | 12     | Smart  | 75%       | X             | Y           | ?.?x    |
  ```
- [ ] Create visualizations:
  - Line chart: layers removed vs quality score (with 75% threshold line)
  - Line chart: layers removed vs inference speed
  - Bar chart: comparing all pruning strategies at same layer count
- [ ] Write Track A summary:
  - Which layers are critical and why?
  - Which layers are redundant and what were they "supposed" to do?
  - What's the minimum viable model?
  - How does this relate to what we learned about layer roles (early=syntax, mid=semantics, late=output)?
- [ ] Save everything to `outputs/pruning_results/`
- [ ] Commit: "Day 8: Track A complete — pruning results analyzed and documented"

**Track A deliverable:**
> "Llama 3.2 1B can be reduced from 16 to N layers while maintaining 75%+ quality.
> Layers [X, Y, Z] are critical. Removing them causes [specific failures].
> Layers [A, B, C] are redundant. The model barely uses them.
> This achieves M% speedup and P% memory reduction."

---

### Day 9 — Tuesday, May 5, 2026
**Phase 3A: KV Cache Profiling**

- [ ] Build `cache_profiler.py`:
  - For each test prompt, generate 100 tokens
  - At each generation step, record:
    - Total KV cache memory (bytes)
    - Per-layer cache memory
    - For each cached token: how much total attention does it receive from ALL generated tokens?
  - Identify "dead" cached tokens — tokens that receive < 1% of total attention
  - Calculate: what % of the cache is "dead weight"?
- [ ] **Experiment: Token importance over time**
  - Generate 100 tokens for "Explain why the sky is blue in detail."
  - At step 10, 25, 50, 75, 100: which cached tokens get the most attention?
  - Plot: heatmap — X axis = cached token position, Y axis = generation step, color = attention received
  - **Expected finding:** First token (BOS) and recent tokens get most attention. Middle tokens get ignored.
- [ ] **Experiment: Cache memory breakdown**
  - What % of total inference memory is the KV cache vs model weights vs activations?
  - How does this ratio change as sequence length grows (50, 100, 200, 500 tokens)?
  - Plot: stacked bar chart showing memory breakdown at different sequence lengths
- [ ] Document all baseline measurements — these are the numbers we're trying to beat
- [ ] Save to `outputs/cache_profiles/`
- [ ] Commit: "Day 9: KV cache profiled — baseline measurements established"

**Understanding goal for Day 9:**
> By end of day, you should know: exactly how much memory the KV cache uses, which tokens
> in the cache are actually important, and what % of the cache is wasted on tokens nobody
> attends to. This gives you the target for optimization.

---

### Day 10 — Wednesday, May 6, 2026
**Phase 3B: Attention Sink Analysis + Basic Eviction**

- [ ] **Learn: Attention Sinks**
  - Research paper: "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM)
  - Key insight: LLMs dump attention on the first token (BOS) as a "sink" — it's not because
    the first token is important, it's because attention scores must sum to 1 and the model
    needs somewhere to put "unused" attention
  - This means: the first few tokens MUST stay in cache even though they seem unimportant
- [ ] **Experiment: Verify attention sinks in Llama 3.2 1B**
  - Generate 100 tokens, at each step record attention to token 0 (BOS)
  - Does BOS consistently receive high attention across all layers?
  - Which layers show the strongest sink effect?
  - Plot: line chart — attention to BOS per layer
- [ ] Build `token_reducer.py` with Strategy 1 — Window-only cache:
  - Only keep the last N tokens in cache
  - Try window sizes: 100%, 75%, 50%, 25% of sequence length
  - For each window size: run all test prompts, generate 50 tokens, measure quality
  - Record: quality score vs cache size reduction
- [ ] Build Strategy 2 — Sink + Window:
  - Always keep first 4 tokens (attention sinks) + last N tokens
  - Evict everything in between
  - Try window sizes: 75%, 50%, 25%
  - Compare with window-only — does keeping sinks improve quality?
- [ ] Document: which strategy works better and by how much?
- [ ] **Concept note:** Write the "Attention Sinks" entry in `LLMXray.md`
- [ ] Commit: "Day 10: Attention sinks verified, basic eviction strategies tested"

---

### Day 11 — Thursday, May 7, 2026
**Phase 3C: Advanced Eviction + Cache Quantization**

- [ ] Build Strategy 3 — Importance-based eviction:
  - Track cumulative attention each cached token receives
  - Every N steps, evict the cached tokens with the lowest cumulative attention
  - Keep a minimum of sink tokens + recent window
  - Test: does this outperform static window eviction?
- [ ] Build Strategy 4 — Per-layer cache sizing:
  - Not all layers need full cache
  - Based on Day 9 profiling: which layers use their cache most/least?
  - Try: full cache for important layers, half cache for less important, quarter cache for least
  - Measure quality vs cache reduction
- [ ] **Learn: Quantization**
  - Full precision: each number stored as 16-bit float (FP16) = 2 bytes
  - INT8 quantization: compress to 8-bit integer = 1 byte (50% memory savings)
  - INT4 quantization: compress to 4-bit = 0.5 bytes (75% memory savings)
  - Tradeoff: smaller = less accurate representation of the number
- [ ] Build KV cache quantization:
  - Quantize cached K and V vectors from FP16 → INT8
  - Measure: memory savings vs quality drop
  - Try FP16 → INT4 as well — how much quality is lost?
- [ ] **Concept note:** Write the "Quantization" entry in `LLMXray.md`
- [ ] Commit: "Day 11: Advanced eviction + cache quantization implemented"

---

### Day 12 — Friday, May 8, 2026
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

### Day 13 — Monday, May 11, 2026
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

### Day 14 — Tuesday, May 12, 2026
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

### Day 15 — Wednesday, May 13, 2026
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
| — | *Not yet started* | — |

### Attention Head Observations
| Layer | Head | Pattern Type | Notes | Date |
|-------|------|-------------|-------|------|
| — | — | — | *Not yet started* | — |

### Pruning Observations
| Layers Removed | Quality Impact | Notes | Date |
|----------------|---------------|-------|------|
| — | — | *Not yet started* | — |

### KV Cache Observations
| Finding | Impact | Date |
|---------|--------|------|
| — | *Not yet started* | — |

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

This file is the single source of truth for the project's progress and learnings.
