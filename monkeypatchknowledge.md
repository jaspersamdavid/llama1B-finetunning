# monkeypatchknowledge.md — running prep for the Day 10-12 attention rewrite

> **Living document.** After every day's commit, append a section here.
> The lens is always: *what did we learn today that will be useful when
> we eventually rewrite `LlamaAttention.forward()` to run our own
> KV cache eviction / windowing / quantization logic?*
>
> See also:
> - `monkeypatching.md` — the *mechanism* (what monkey-patching is, how to swap classes in)
> - `LLMXray.md` — broader concept reference (transformer, attention, MLP, KV cache, etc.)
> - This file — the *running understanding* of the model that informs the monkey-patch we'll write

---

## Why this file exists

Monkey-patching means **copying ~80 lines of `LlamaAttention.forward()` and modifying one specific step**. To do that safely we need to know:

1. **The exact tensor shapes at every step** of forward() — wrong reshape = silent breakage
2. **Which layers are safe to mess with** vs which are critical — patching the wrong layer makes the model produce plausible-but-wrong output
3. **How the KV cache plugs in** (transformers 5.x DynamicCache API) — biggest source of subtle bugs
4. **How to verify the patch is working** — what tools we have to A/B before/after
5. **What heads do what** — so we know which to target

Each day's experiments give us pieces of this picture. This file collects them.

---

## Day 1 (Apr 23-24) — Architecture facts you need before touching forward()

### Layer structure (every decoder layer is identical)
```
LlamaDecoderLayer
├── input_layernorm        (RMSNorm)
├── self_attn              ← this is what we'll monkey-patch
│   ├── q_proj             [hidden=2048 → num_q_heads*head_dim = 2048]
│   ├── k_proj             [hidden=2048 → num_kv_heads*head_dim = 512]   ← note GQA
│   ├── v_proj             [hidden=2048 → num_kv_heads*head_dim = 512]
│   └── o_proj             [hidden=2048 → hidden=2048]
├── post_attention_layernorm (RMSNorm)
└── mlp
    ├── gate_proj
    ├── up_proj
    └── down_proj          (SwiGLU)
```

### The numbers we'll need at every step

| Symbol | Value | Where it shows up in forward() |
|---|---|---|
| `hidden_size` | 2048 | input/output of forward |
| `num_q_heads` | 32 | shape of Q after reshape |
| `num_kv_heads` | 8 | shape of K, V after reshape (smaller than Q!) |
| `head_dim` | 64 | last dim of Q, K, V tensors |
| `n_repeats` | 4 | `num_q_heads // num_kv_heads`, used in `repeat_kv` |
| `num_layers` | 16 | how many forward()s run per token |
| `vocab_size` | 128256 | comes back in lm_head, not in attention |
| `max_position_embeddings` | 131072 | hard ceiling on seq length |

**Critical for monkey-patching:** Llama uses **Grouped Query Attention (GQA)**. K and V have only 8 heads, but Q has 32. Inside `forward()` there's a `repeat_kv(k, 4)` step that broadcasts K/V to match Q's head count. If we forget this in our subclass, the matmul shapes won't line up.

### Tokenization gotcha
- BOS token `<|begin_of_text|>` (id 128000) is **auto-prepended** to every input. Position 0 is always BOS.
- This is the "attention sink" we'll explicitly preserve when doing window eviction on Day 10.

### Where the source lives (we'll read this on Day 10)
```
~/venvs/llama-xray/lib/python3.11/site-packages/transformers/models/llama/modeling_llama.py
```
The class is `LlamaAttention`. Read it end-to-end with a notebook open before writing any patch.

---

## Day 2 (Apr 24) — The attention math we're rewriting + the verification tool we'll need

### What forward() actually computes (the part we'll replace)

```
1. Project:  Q = x @ Wq    K = x @ Wk    V = x @ Wv
2. Reshape:  Q → [batch, 32, seq, 64]    K, V → [batch, 8, seq, 64]
3. RoPE:     apply rotary position embeddings to Q and K
4. GQA:      K = repeat_kv(K, 4)    V = repeat_kv(V, 4)    [now [batch, 32, seq, 64]]
5. Scores:   S = Q @ K.transpose(-2, -1) / sqrt(64)
6. Mask:     S += causal_mask                  ← blocks future tokens
7. Softmax:  A = softmax(S, dim=-1)            ← rows sum to 1
8. Weighted: out = A @ V
9. Reshape:  out → [batch, seq, 2048]
10. Project: return Wo @ out
```

**Steps 5-8 are where we'll intervene** for cache eviction (drop columns of K/V before step 5) or attention modification (zero columns of A in step 7, then renormalize).

**Steps 3 (RoPE), 4 (GQA repeat), and 6 (causal mask) are the easy-to-forget bits** — bad patches usually break by skipping or misordering these.

### The logit lens IS our debugging tool for the patch

`run_logit_lens(model, tokenizer, prompt)` runs the model and shows us per-layer top-5 predictions. When we monkey-patch:

- **Step 1: write a no-op subclass** (copies forward verbatim, changes nothing). Run logit lens. If output isn't bit-identical to baseline, *the patching mechanism is broken* — not the patch logic.
- **Step 2: write the real patch.** Run logit lens. Diff against baseline. The `diff_logit_lens()` helper in `weight_tweaker.py` does the side-by-side print.

### Knowledge emerges layer-by-layer
For "The capital of France is" → Paris emerges around layers 11-13. **Implication for patching:** a patch at L5 has different consequences than at L13. Patches deep in the stack disturb the answer formation directly; patches early in the stack disturb feature-building (and may or may not propagate).

---

## Day 3 (Apr 27) — Head types: which heads to target, which to avoid

### Heads have specializations — we identified these by visualizing 32 heads × 16 layers

| Pattern type | What it does | Relevance to monkey-patching |
|---|---|---|
| **First-token head** ("BOS sink") | All tokens attend heavily to position 0 | **This is what we remove in the Day 10 "no-sink" patch** |
| **Previous-token head** | Each token attends to the one before it | Don't touch — needed for sequence modeling |
| **Semantic head** | Content words attend to related content words | Probably don't touch — meaning depends on these |
| **Position head** | Fixed positional pattern regardless of input | Could be window-truncated safely |

The attention visualizer (`attention_visualizer.py`) is the **second debugging tool** for monkey-patches: run it before and after, eyeball whether the patch broke head specializations we wanted to keep.

### Embeddings sit BEFORE attention
- Embedding table = 128256 × 2048
- Sits in `model.model.embed_tokens` — totally upstream of `self_attn`
- **Monkey-patching attention doesn't touch embeddings.** Good — one less moving part.

---

## Day 4 (Apr 29) — KV cache plumbing + which layers are safe to mess with

### The KV cache shape we'll be slicing into on Day 10

**32 KB per token. Formula:**
```
bytes = 2 (K and V) × num_layers × num_kv_heads × head_dim × bytes_per_value
      = 2 × 16 × 8 × 64 × 2
      = 32,768 bytes per token
```

The cache stores **8-head K and V** (pre-GQA-repeat), not 32-head. Saves 4× memory. Our patch needs to keep it in this form and only do the repeat right before the matmul.

### transformers 5.x DynamicCache API — the biggest gotcha

The `past_key_values` argument to `forward()` is no longer a tuple-of-tuples. It's a `DynamicCache` object. Access pattern:

```python
# OLD (transformers 4.x — DOES NOT WORK in 5.x)
k_layer, v_layer = past_key_values[layer_idx]

# NEW (transformers 5.x)
k_layer = past_key_values.layers[layer_idx].keys     # shape [batch, num_kv_heads, seq, head_dim]
v_layer = past_key_values.layers[layer_idx].values
```

Almost every monkey-patching tutorial online uses the old API. **Our patches must use `.layers[i].keys` / `.values`.** This will be the #1 source of bugs on Day 10 if we forget.

### Layer importance — which layers can be patched safely

From `zero_attention(L)` experiments on prompt `"The capital of France is"` (baseline = Paris @ 48%):

| Layer | Effect of zeroing attention | Implication for monkey-patching |
|---|---|---|
| **L0** | Paris drops to 28% (-20pp) | **Critical.** L0 attention does real input routing. Don't patch here unless we mean to break things. |
| **L8** | Paris RISES to 69% (+21pp) | **Safe / valuable target.** Possibly the most patch-friendly layer. Cache eviction here is unlikely to hurt and may help. |
| **L15** | Paris drops to 20% (-28pp) | **Critical.** Final-layer attention sharpens the answer. Don't aggressively patch here. |

**This is a Day-5 hypothesis** — Day 5's full importance scoring (logit-lens delta, zero-out impact, cosine sim) will tell us if this generalizes across all layers. If it does, Day 10's eviction can apply *more aggressive* eviction at "redundant" layers and *gentler* eviction at critical layers (this is called per-layer cache sizing, scheduled for Day 11).

### MLP fragility threshold — informs how careful our patch math needs to be
- σ=0.01 noise on MLP gate_proj → invisible (Paris 51% vs 48%)
- σ=0.1 noise → catastrophic (model collapses to "being" 3%)
- σ=1.0 → total nonsense

**Implication:** there's no graceful middle ground. Numerical errors in our patch (e.g. wrong renormalization after dropping cache columns) won't produce "slightly worse" output — they'll either be invisible or catastrophic. We won't get warning signs.

### Speed numbers to beat
- With cache, 30 tokens generated in 1.58s
- Without cache, 30 tokens in 3.42s (2.17× slower)
- Gap grows with seq length (no-cache is O(n²) per step, with-cache is O(n))
- **Day 10-12 target:** keep speed at-or-near with-cache while reducing memory by 25-50%

### Tools available for verifying a patch
By end of Day 4 we have everything we need to verify a future monkey-patch:

1. `run_logit_lens()` → per-layer prediction A/B
2. `attention_visualizer` → eyeball head behavior change
3. `diff_logit_lens()` → side-by-side before/after table
4. `WeightTweaker(...).reset()` → snapshot/restore for clean experiment turnaround
5. `compare_speed()` → with-cache vs without-cache benchmarking
6. `print_memory_breakdown()` → cache memory at any seq length

When we write our first monkey-patch on Day 10, the workflow will be:
- Patch in → run logit lens → run attention viz → run speed compare → check memory → if anything is off, reset and debug.

---

## Day 5 (Apr 30 — pending) — Layer Importance Scoring

*To be filled in after the Day 5 commit.*
