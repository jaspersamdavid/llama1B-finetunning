# CLAUDE.md — llama-xray

> **Living document. Update after every completed task.**

---

## Project Overview

**Name:** llama-xray
**Model:** Llama 3.2 1B (FP16, ~2GB)
**Machine:** Intel i5 Mac, 8GB RAM, no GPU
**Duration:** 15 working days (April 10 – April 30, 2026, excluding weekends)
**Repository:** `llama-xray/`

---

## Project Goals

### Primary Goals

1. **Understand how LLMs work — top to bottom, like a PhD but explained simply**
   - What is a transformer? What are layers, attention, embeddings, feed-forward networks?
   - How does text become numbers (tokenization → embeddings)?
   - How do those numbers flow through layers to produce the next word?
   - What is the attention mechanism really doing — Q, K, V, multi-head, all of it?
   - What is the KV cache, why does it exist, and how does it speed up generation?
   - What are weights? What do they store? How were they trained?
   - How does the model "know" facts — where is knowledge stored in the weights?
   - What is backpropagation, loss, gradient descent — the training loop explained simply?
   - What is LoRA, quantization, fine-tuning — how do people modify these models?
   - None of this should require prior ML knowledge. Every concept is explained from scratch.

2. **Layer Pruning (Track A):**
   - Can we remove layers and still get correct answers?
   - Start at 16 layers, remove one at a time
   - Find the minimum number of layers that maintains 75%+ quality
   - Understand WHY certain layers matter and others don't

3. **KV Cache Optimization (Track B):**
   - Can we reduce the tokens the model stores during generation?
   - Target: 5% less token compute for equivalent output quality
   - Understand which cached tokens are actually useful vs wasted space
   - Build and test eviction strategies

4. **Build a reusable inspection toolkit:**
   - Logit lens (see predictions at each layer)
   - Attention maps (see which words look at which)
   - Embedding explorer (see how the model understands word relationships)
   - KV cache profiler (measure cache size and usage)
   - Weight tweaker (modify weights live, see immediate impact)

5. **Portfolio piece:**
   - Interactive dashboard showing all findings
   - Written report with clear visualizations
   - Techniques that transfer to larger models (3B, 7B, 70B+)

---

## Current Status

> **Last updated:** April 11, 2026
> **Currently working on:** Day 1 — Project Setup (in progress)
> **Last completed task:** Directory structure, model_loader.py, notebook, dependencies installed
> **Next task:** Hugging Face access + download model + run notebook

---

## Key Concepts Reference

> This section gets filled in as we learn each concept. Written in plain language, no jargon.

### Transformer
> [To be filled — Day 1-2]

### Tokenization
> [To be filled — Day 1]

### Embeddings
> [To be filled — Day 3]

### Attention (Q, K, V)
> [To be filled — Day 2]

### Multi-Head Attention
> [To be filled — Day 2]

### Feed-Forward Network (MLP)
> [To be filled — Day 2]

### KV Cache
> [To be filled — Day 3-4]

### Logit Lens
> [To be filled — Day 2]

### Backpropagation & Training
> [To be filled — as needed]

### LoRA & Fine-tuning
> [To be filled — as needed]

### Quantization
> [To be filled — Day 11]

---

## Project Structure

```
llama-xray/
├── CLAUDE.md                    ← This file (living document)
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
│   │   ├── __init__.py
│   │   ├── model_loader.py      ← Load model with full introspection flags
│   │   ├── layer_inspector.py   ← Per-layer output extraction
│   │   ├── embedding_explorer.py← Word similarity, nearest neighbors
│   │   ├── attention_visualizer.py ← Attention heatmaps per layer/head
│   │   ├── kv_cache_analyzer.py ← Cache size tracking, token importance
│   │   └── logit_lens.py        ← Predictions at each layer
│   ├── pruning/                 ← Track A: Layer pruning
│   │   ├── __init__.py
│   │   ├── layer_pruner.py      ← Remove/skip layers
│   │   ├── layer_importance_scorer.py ← Rank layers by importance
│   │   └── eval_pruned.py       ← Evaluate pruned model quality
│   ├── kv_optimization/         ← Track B: KV cache optimization
│   │   ├── __init__.py
│   │   ├── cache_profiler.py    ← Measure cache memory and usage
│   │   ├── token_reducer.py     ← Eviction strategies
│   │   └── eval_optimized.py    ← Evaluate optimized cache quality
│   └── eval/                    ← Shared evaluation framework
│       ├── __init__.py
│       ├── benchmark.py         ← Run all test prompts, score results
│       ├── test_prompts.py      ← Curated test prompts by category
│       └── comparison.py        ← Before/after comparison tables
├── outputs/                     ← Visualizations, charts, reports
│   ├── attention_maps/
│   ├── logit_lens/
│   ├── pruning_results/
│   ├── cache_profiles/
│   └── final_report/
└── data/
    └── test_prompts.json        ← All test prompts in one file
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
scikit-learn          # for PCA/t-SNE embedding visualization
```

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

### Day 1 — Friday, April 10, 2026
**Phase 0: Project Setup + Understanding Tokenization**

- [ ] Create project directory structure (as defined above)
- [ ] Create `requirements.txt` and install all dependencies
- [ ] Download Llama 3.2 1B FP16 from Hugging Face
- [ ] Write `model_loader.py` — load model with `output_hidden_states=True` and `output_attentions=True`
- [ ] Run `print(model)` — see full architecture, count layers, understand the structure
- [ ] Run `print(model.config)` — document: num_hidden_layers, hidden_size, num_attention_heads, num_key_value_heads, vocab_size
- [ ] **Learn: Tokenization** — write a notebook section showing:
  - How text becomes token IDs: `tokenizer.encode("Why is the sun yellow")`
  - How token IDs become text: `tokenizer.decode([...])`
  - What the vocabulary looks like: `tokenizer.get_vocab()`
  - Why "understanding" becomes ["under", "standing"] — subword tokenization explained
- [ ] Run one simple generation to confirm everything works: `model.generate("The capital of France is")`
- [ ] **Concept note:** Write the "Transformer" and "Tokenization" entries in Key Concepts Reference above
- [ ] Commit: "Day 1: Project setup, model loaded, tokenization understood"

**Understanding goal for Day 1:**
> By end of day, you should be able to explain: what a token is, how text becomes numbers,
> what the model's architecture looks like (16 layers stacked), and what each layer contains
> (attention + MLP). You don't need to understand HOW they work yet — just WHAT the pieces are.

---

### Day 2 — Monday, April 13, 2026
**Phase 1A: Logit Lens + Understanding Attention**

- [ ] **Learn: Attention mechanism from scratch**
  - What is Q (Query), K (Key), V (Value)?
  - Analogy: Q = "what am I looking for?", K = "what do I contain?", V = "what info do I give?"
  - How scores are computed: Q × K → softmax → multiply by V
  - What multi-head attention means: 32 smaller attention operations running in parallel
  - Each head can learn a different pattern (grammar, meaning, position, etc.)
  - Write this up in the Key Concepts Reference
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
- [ ] **Concept note:** Write the "Attention (Q, K, V)", "Multi-Head Attention", and "Logit Lens" entries
- [ ] Commit: "Day 2: Logit lens built, attention mechanism understood"

**Understanding goal for Day 2:**
> By end of day, you should be able to explain: how attention works (Q×K→scores→softmax→×V),
> why we have multiple heads, and what the logit lens shows you. You should be able to look at
> the logit lens output and say "the model figured out the answer at layer X."

---

### Day 3 — Tuesday, April 14, 2026
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
- [ ] **Concept note:** Write the "Embeddings" entry in Key Concepts Reference
- [ ] Commit: "Day 3: Attention visualizer + embedding explorer built"

**Understanding goal for Day 3:**
> By end of day, you should be able to explain: what embeddings are (a lookup table where each
> word has a vector of numbers), why similar words have similar vectors (they appeared in similar
> contexts during training), and what attention maps show you (which words are "looking at" which
> other words at each layer).

---

### Day 4 — Wednesday, April 15, 2026
**Phase 1C: KV Cache Analyzer + Weight Tweaker**

- [ ] Build `kv_cache_analyzer.py`:
  - Generate tokens one at a time with `use_cache=True`
  - At each step, log:
    - Cache size (number of tokens × layers × head_dim)
    - Total memory in MB
    - K and V vector norms per layer per token
  - Compare generation speed WITH vs WITHOUT cache (time 50 token generation both ways)
  - Visualize: cache growth chart (steps vs memory)
  - Visualize: K vector norm heatmap (layers × tokens)
- [ ] **Learn: KV Cache from scratch**
  - Why it exists: without cache, model recomputes K and V for ALL previous tokens every step
  - With cache: store K and V, only compute for the new token
  - How it grows: each new token adds one K vector + one V vector per layer per KV head
  - Calculate: for a 500 token sequence, how much memory does the KV cache use?
    - Formula: 2 (K+V) × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_value
- [ ] Build interactive `weight_tweaker.py` (CLI tool):
  - Menu-driven loop:
    1. Zero out a layer's attention (q_proj weights → all zeros)
    2. Add noise to a layer's MLP (gate_proj += random noise × strength)
    3. Scale a specific attention head (multiply head's q_proj rows by factor)
    4. Swap two layers (exchange all weights between layer A and layer B)
    5. Reset model to original weights
    6. Change input text
    7. Show logit lens before/after comparison
    8. Quit
  - After each tweak, automatically run logit lens and show side-by-side comparison
- [ ] **Experiment: Break the model and observe**
  - Zero out layer 0 attention — what happens? (early feature extraction lost)
  - Zero out layer 15 attention — what happens? (final output prep lost)
  - Zero out layer 8 attention — what happens? (mid-level reasoning lost)
  - Add noise (0.01, 0.1, 1.0) to layer 5 MLP — at what noise level does output collapse?
  - Swap layer 2 and layer 14 — does the model produce garbage? (expected: yes)
  - Scale head 0 in layer 8 by 10x — what prediction changes?
  - Document all observations
- [ ] **Concept note:** Write the "KV Cache" and "Feed-Forward Network (MLP)" entries
- [ ] Commit: "Day 4: KV cache analyzer + weight tweaker built, model broken and studied"

**Understanding goal for Day 4:**
> By end of day, you should be able to explain: what the KV cache is and why it exists,
> how much memory it uses, and what happens when you break different parts of the model.
> You should have an intuitive sense of "early layers do X, middle layers do Y, late layers do Z."

---

### Day 5 — Thursday, April 16, 2026
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

### Day 6 — Friday, April 17, 2026
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

### Day 7 — Monday, April 20, 2026
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

### Day 8 — Tuesday, April 21, 2026
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

### Day 9 — Wednesday, April 22, 2026
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

### Day 10 — Thursday, April 23, 2026
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
- [ ] Commit: "Day 10: Attention sinks verified, basic eviction strategies tested"

---

### Day 11 — Friday, April 24, 2026
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
- [ ] **Concept note:** Write the "Quantization" entry in Key Concepts Reference
- [ ] Commit: "Day 11: Advanced eviction + cache quantization implemented"

---

### Day 12 — Monday, April 27, 2026
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

### Day 13 — Tuesday, April 28, 2026
**Phase 4A: Interactive Dashboard**

- [ ] Build a visualization dashboard (Streamlit or HTML/React):
  - **Tab 1: Model Anatomy**
    - Show model architecture diagram
    - Layer count, head count, embedding size
    - Interactive: click a layer to see its weights, attention patterns
  - **Tab 2: Logit Lens Explorer**
    - Input a prompt, see predictions emerge layer by layer
    - Color-coded: green when correct answer appears, red when wrong
    - Slider to highlight specific layers
  - **Tab 3: Attention Maps**
    - Select layer and head, see attention heatmap
    - Toggle between different test prompts
    - Highlight head types (previous-token, sink, semantic)
  - **Tab 4: Embedding Space**
    - Type a word, see nearest neighbors
    - 2D scatter plot of word clusters
    - Compare two words — show similarity score
  - **Tab 5: Layer Pruning Results**
    - Slider: drag to remove layers
    - Shows quality score updating in real time
    - Side-by-side: original predictions vs pruned predictions
    - Highlights which layers are critical (red) vs removable (green)
  - **Tab 6: KV Cache Optimizer**
    - Visualize cache growth during generation
    - Toggle eviction strategies, see memory savings
    - Before/after comparison
  - **Tab 7: Weight Playground**
    - Select a layer, select a component (attention/MLP)
    - Slider to add noise, scale, or zero out
    - See predictions change live
- [ ] Commit: "Day 13: Interactive dashboard built"

---

### Day 14 — Wednesday, April 29, 2026
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

### Day 15 — Thursday, April 30, 2026
**Phase 4C: Final Review + Future Work**

- [ ] End-to-end test: clone repo fresh, install deps, run all notebooks, verify outputs
- [ ] Review CLAUDE.md — ensure all observations are documented
- [ ] Write "Future Work" section in README:
  - Scale to Llama 3.2 3B — do same layers matter?
  - Scale to Gemma 4 E2B — how does hybrid attention change things?
  - Compare pruning results across model sizes
  - Implement more advanced techniques:
    - Activation patching (swap activations between prompts)
    - Probing classifiers (train small classifiers on hidden states to find what's encoded)
    - Circuit analysis (trace specific behaviors through attention heads)
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
| 15 working days, no weekends | Sustainable pace, time for concepts to sink in between sessions. | April 9, 2026 |

---

## Resources & References

- **Logit Lens:** https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- **Attention Sinks / StreamingLLM:** https://arxiv.org/abs/2309.17453
- **Mechanistic Interpretability:** https://transformer-circuits.pub/
- **Anthropic's Interpretability Research:** https://www.anthropic.com/research#interpretability
- **Hugging Face Transformers Docs:** https://huggingface.co/docs/transformers
- **Llama 3.2 Model Card:** https://huggingface.co/meta-llama/Llama-3.2-1B
- **3Blue1Brown Neural Networks:** https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- **Andrej Karpathy — Let's build GPT:** https://www.youtube.com/watch?v=kCc8FmEb1nY

---

## How to Update This File

After completing any task:
1. Check off the task checkbox: `- [ ]` → `- [x]`
2. Update **Current Status** section at the top
3. Add any interesting findings to **Observations** tables
4. Add any problems to **Blockers** table
5. Fill in **Key Concepts Reference** when you learn a new concept
6. Commit the updated CLAUDE.md with a descriptive message

This file is the single source of truth for the project's progress and learnings.

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

### Environment Setup (MUST redo on Mac Mini)
- [x] **Python 3.11.9 installed via pyenv** — Python 3.13 does NOT work (PyTorch incompatible)
  - Command: `pyenv install 3.11.9`
  - Set local: `pyenv local 3.11.9` (creates `.python-version` file, gitignored)
- [x] **Virtual environment created** — `python3 -m venv venv`
  - The `venv/` folder is gitignored, must recreate on Mac Mini
- [x] **All dependencies installed** in venv:
  - `pip install -r requirements.txt`
  - Then fix versions: `pip install "numpy<2" "transformers>=4.40,<5"`
  - Final working versions: PyTorch 2.2.2, Transformers 4.57.6, NumPy 1.26.4
  - **Note:** PyTorch 2.4+ is NOT available for Intel Mac. Mac Mini (if Apple Silicon) may get newer PyTorch.

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

### What Has NOT Been Done Yet
- [ ] Notebook has not been run (waiting for HF access + model download)
- [ ] No model outputs or visualizations generated yet
- [ ] No concepts filled in yet in Key Concepts Reference
- [ ] Day 1 tasks not fully completed — only setup portion done
