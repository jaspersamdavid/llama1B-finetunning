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

## Day 5 (May 8) — Layer Importance Scoring → which layers can our patches afford to disrupt?

### What we built
`src/pruning/layer_importance_scorer.py` — runs three independent ranking methods and overlays them on one chart.

| Method | What it captures | Cost per prompt | Forward count |
|---|---|---|---|
| Logit-lens KL Δ | KL between layer L's predicted distribution and layer L−1's | 1 fwd with `output_hidden_states=True` | 1 |
| Zero-out top-1 drop | Drop in baseline top-1 probability when the layer's attention + MLP weights are zeroed | snapshot once, then 1 fwd per layer per prompt | 17 |
| 1 − cos(input, output) | How much the layer transforms the last-token hidden state | reuses method-1's forward, free | 0 |

### Implementation gotchas relevant to monkey-patching

- **FP16 underflow kills KL computation.** First run produced all-NaN KL values. The fix: cast logits to FP32 before `log_softmax`. The same gotcha will hit any Day 10-12 patch that tries to compute attention scores or KL on the model's native FP16 — softmax of large negative logits underflows to 0, then `log(0) = -inf`, then `0 * -inf = NaN`.
  - **Pattern for our future patches:** if your custom attention does any divergence/entropy bookkeeping, do that math in FP32. The K/V tensors and matmul outputs can stay FP16; only the post-softmax stats need the cast.
- **Snapshot-once + restore-per-iter** is way cheaper than `WeightTweaker.reset()` per iteration. We snapshot each layer's `state_dict()` once at the top of method 2, then mutate-and-restore that single layer per iteration. Zero-out for all 16 layers × 17 prompts ran in 11 seconds. Full-model `reset()` would have copied 1.24B params 272 times.
- **Zero-out as identity-skip equivalence.** In Llama's pre-norm residual (`x = x + attn(rmsnorm(x))`), zeroing all weights of the layer makes its sub-block output 0, so the residual gives `x` back unchanged. **Zeroing a layer is mathematically equivalent to skipping it.** Useful: this means `zero_layer(L)` IS our cheap "skip layer L" operation without writing any forward hook — the residual stream does the bypassing for us. Day 6's `skip_layer()` primitive can literally be `zero_layer()` under the hood.

### What we found — relevant to where to safely patch

| Finding | What it means for Day 10-12 patches |
|---|---|
| **L0, L1 are critical across all 3 methods** | Don't aggressively patch attention here. Cache eviction at L0/L1 will hit input routing. |
| **L12 is bottom-3 in 2/3 methods, never top-3 anywhere** | Best target for the "drop everything between sink and recent window" pattern. If a layer is going to tolerate aggressive eviction, this is it. |
| **L15 disagreement (high cosine, lowest zero-out drop)** | "Sharpening" layer — moves the vector around without changing top-1. **Implication:** patches at L15 that change *what gets attended to* will be visible in cosine sim but invisible in top-1 quality. We need both metrics during Day 10-12 verification, not just top-1. |
| **L8's Day 4 anomaly didn't generalize** | Reminder: per-prompt findings can mislead. Day 10-12 must test patches across all 17 test prompts, not just `"capital of France is"`. |

### Method-disagreement is itself a finding — and it changes our patch verification strategy

Cosine-sim and zero-out drop **don't measure the same thing**:
- Cosine-sim says "how much did the layer move the vector around?"
- Zero-out drop says "how much does removing the layer hurt the top-1 token?"

A layer can score high on one and low on the other. L15 is the clearest example. So when we verify a Day 10-12 monkey-patch:

- **Don't trust top-1 alone** — a patch can preserve top-1 across all prompts and still be quietly mangling the representation in ways that show up downstream (longer generations, OOD prompts).
- **Use the cosine-sim signal** — for each layer's last-token hidden state, compare patched vs unpatched. If cosine drops below ~0.95 anywhere our patch wasn't supposed to touch, we have a leak.
- **Use logit-lens KL Δ** — if the per-layer prediction trajectory diverges between patched and unpatched, even the layers we didn't directly modify are receiving altered inputs. That's a sign of bug propagation.

These three methods become our **patch verification kit** alongside the no-op subclass test from Day 2.

### Tools/infrastructure now available for Days 10-12

- `score_lens_and_cosine(model, tokenizer, prompts)` — patch verification A/B
- `score_zero_out(...)` — sanity check that our patch didn't accidentally turn a layer into nothing
- `snapshot_layer(model, L)` / `restore_layer(model, L, snapshot)` — clean primitives for selective revert (used during patch debugging)
- `outputs/pruning_results/layer_importance_scores.npz` — saved arrays. Future scripts can load this and get the layer ranking without re-running the scorer.

### Updated layer-importance hypothesis (replaces Day 4 single-prompt guess)

Day 4 hypothesis based on `"capital of France is"`:
> L0 routes (-20pp), L8 redundant (+21pp), L15 sharpens (-28pp).

Day 5 hypothesis based on 17 prompts × 3 methods:
> **L0, L1, L3 route input.** L2-L11 do mid-stack work with no single layer carrying overwhelming weight. **L12 is most redundant.** L13-L14 do answer-shaping. **L15 sharpens** — big vector change, small top-1 effect.

For Day 10's eviction strategy: aggressive at L7-L12, gentle at L0-L1 and L13-L14, "free zone" at L15 (the answer is already determined — sharpening doesn't need full cache).

---

## Day 6 (May 8) — Sequential Layer Removal → layers are NOT independently removable

### What we built
- `src/pruning/layer_pruner.py` — `LayerPruner` class that swaps
  `model.model.layers` to a smaller `nn.ModuleList` (refs only — no deep copy).
- `src/pruning/eval_pruned.py` — exact-match / top-5 / baseline-prob /
  repetition-rate metrics + two experiment drivers.

### Implementation gotchas relevant to monkey-patching

- **`self_attn.layer_idx` must be renumbered when you mutate the layer stack.**
  This was the single subtle bug that broke our first run. transformers' `DynamicCache`
  indexes K/V slots by `cache.layers[layer.self_attn.layer_idx]`. When you drop a
  middle layer, the surviving layers retain their *original* `layer_idx` values
  (so the last surviving layer still thinks it's idx 15 even though only 15 layers
  remain) — and the cache, sized to `num_hidden_layers = 15`, only has slots 0-14.
  Result: `IndexError: list index out of range`.

  **Pattern for our future patches:** any monkey-patch that wraps or replaces
  `LlamaAttention` MUST preserve `self.layer_idx` correctly, AND if the patch
  ever runs in a context where the stack has been mutated (e.g. a Day 7
  layer-skip experiment combined with a Day 10 cache patch), the indexing
  must agree with whatever `cache.layers[]` was sized for.

- **Removing layers actually saves compute; zeroing them does not.**
  Day 5 confirmed `zero_layer(L)` is mathematically equivalent to skipping in
  pre-norm. But it still runs forward through zeroed weights — same FLOPs.
  Our `LayerPruner.prune()` actually drops the layer from the ModuleList,
  which means the forward iterator skips it entirely. That's the difference
  between "behaviorally equivalent" and "actually faster". For Day 10-12,
  if our cache eviction patch wants to physically skip computation (not just
  produce identity output), we'll need similar ModuleList manipulation, not
  just clever attention math.

### What we found — sobering reality vs Day 5's hopes

**The 1B model is much less prunable than Day 5's importance scoring suggested.**

| Config | Exact match | What Day 5 predicted |
|---|---|---|
| Baseline (16 layers) | 100% | — |
| Drop L15 (1 from end) | **53%** | "L15 sharpens, ok to remove" |
| Drop L14-15 (2 from end) | 24% | partial agreement (L14 was rank 4 important) |
| Drop L13-15 (3 from end) | 12% | non-monotonic with above |
| Drop L9-15 (7 from end) | 0% | model collapses |

| Single-layer mid removal | Exact match drop |
|---|---|
| L4 alone (best) | 100% → 71% |
| L11 alone (worst) | 100% → 41% |
| L5, L7 alone | 100% → 53% |

### Why this matters for the Day 10-12 patch design

1. **Day 5's "L15 is a sharpener with low importance" was metric-specific.**
   The Day 5 zero-out method averaged the *probability drop on the baseline top-1
   token*. That's a continuous metric — a small drop (0.055) sounds tiny. But
   when you measure the *discrete* outcome ("is the top-1 still the same token?"),
   removing L15 flips the answer for ~half of all prompts. **Implication for our
   patch verification: average prob deltas hide token flips. Add per-prompt
   exact-match to the verification kit, not just average prob.**

2. **Layer interactions are non-additive.** Dropping L13-15 is *worse* than
   dropping L12-15. This means **we cannot safely combine independently-vetted
   patches.** If a Day 10 patch is "safe at L8" and another Day 11 patch is
   "safe at L11", running both together may not be safe. Each combined
   patch needs its own end-to-end evaluation.

3. **No "free zone" exists for removal.** Day 5 framed L15 as a free-zone
   candidate. Day 6 disproves this. **For Day 10's cache eviction we need
   to be more surgical than "drop layers" — we need to drop *specific cached
   tokens* from layers, not whole-layer operations.** Window-based eviction
   plus attention sinks (planned for Day 10) is fundamentally a finer-grained
   intervention than layer pruning, so it has more headroom — but the
   verification bar is the same.

4. **Repetition rate is a useful collapse signal.** When the model is
   damaged, it falls into token loops. At 4+ trailing layers removed,
   repetition rate is 60%+. **Use this as a Day 10-12 cheap canary:** if
   our cache eviction patch causes any test prompt to start looping in
   the first 15 generated tokens, we know we've over-pruned without
   needing to hand-read the output.

### New tool in the verification kit

`evaluate(model, tokenizer, baseline_top1s, prompts)` from
`src/pruning/eval_pruned.py` — runs the full 17-prompt suite and returns
the four-metric dict. Reusable as the patch-verification function for
Day 10-12: snapshot baselines once, run patch, call evaluate, compare.

### Updated layer-importance hypothesis (replaces Day 5's optimism)

Day 5 hypothesis based on 17 prompts × 3 methods:
> L0/L1 route, L12 most redundant, L15 sharpens — likely all-removable.

Day 6 actuals:
> **Llama 3.2 1B has effectively no removable layers without dropping below
> 75% top-1 accuracy.** Even the "most redundant" layer (L12) hasn't been
> tested in isolation yet (Day 7's job), but Day 6's mid-stack experiments
> all damage at least 30% of prompts. The 1B model is densely packed — the
> portfolio takeaway is "Track A reveals Llama 3.2 1B is at the lower edge
> of what dense pruning can preserve" rather than "we shrunk it 30%".
> For Day 10-12: this means cache reduction is the higher-headroom track.

---

## Day 7 (May 8) — Smart Pruning + Learnable Skip → the model votes "everything matters"

### What we built
- `src/pruning/smart_pruning.py` with two experiments:
  - **Smart prune** — uses Day 5's ranking, removes least-important first.
  - **`SkippableLayer`** wrapper class + distillation training loop that
    trains a per-layer scalar gate via KL against the frozen teacher.

### Implementation patterns relevant to monkey-patching

- **Wrapping decoder layers without breaking the model.** `SkippableLayer`
  is a clean template for any future patch that wants to wrap (rather
  than replace) a `LlamaDecoderLayer`. Two key tricks:
  1. **`*args, **kwargs` pass-through.** `LlamaDecoderLayer.forward()`
     accepts a long list of kwargs (attention_mask, position_ids,
     past_key_value, position_embeddings, ...). Don't reproduce the
     signature — pass everything through. This is the same pattern
     Day 10-12 patches will need if they wrap (rather than replace)
     attention.
  2. **`@property` for `self_attn`.** transformers internals look up
     `decoder_layer.self_attn.layer_idx` for cache indexing. Exposing
     `self_attn` as a property (not setting it as an attribute) keeps
     identity preservation: assignments to `self.layer.self_attn.x`
     correctly update through the wrapper.

- **Tuple-vs-tensor return guard.** Some transformers versions return a
  tuple from decoder layer forward, others return a tensor. The wrapper's
  isinstance check handles both. **Apply this pattern to any monkey-patch
  that wraps something whose return signature you don't fully trust.**

- **Selective freeze for surgical training.**
  ```python
  for name, p in model.named_parameters():
      p.requires_grad = name.endswith("skip_logit")
  ```
  This is the pattern for Day 11's "train just the cache eviction
  parameters" idea if we want learnable cache shortening. Freeze
  everything; mark only the parameters you're optimizing.

### What we found

**Smart pruning beats end-removal by avg +9pp.** Best gain: +29pp at 13
layers remaining. So the Day 5 ranking IS predictive — but the
improvement is bounded. Even smart-pruning never crosses 75% threshold.

**Best 15-layer config in the project:** drop L4 alone (Day 6 mid-single,
71% exact match). Drop L12 (Day 7 smart) is 65%. Both still under 75%.

**Learnable skip experiment failed to skip anything.** All 16 weights
converged in [0.96, 1.00] after 30 steps with λ_L1 = 0.05. The KL gradient
(matching teacher distributions exactly) was always strong enough to
overpower the L1 sparsity pull. **The model's own gradient signal says
every layer is needed.** This is consistent with Day 6's finding that no
single removal preserves quality — same fact viewed from a different
angle (gradient-based importance vs ablation-based importance).

**Spearman ρ between learned weights and Day 5 combined importance: 0.29.**
- L12 has the LOWEST learned weight (most "skippable") → agrees with Day 5
- L4 has the HIGHEST learned weight (most "needed") → disagrees with Day 5
  (Day 5 ranked L4 mid-importance, score 0.18)

The disagreement on L4 matters: Day 6 found L4 was the *best* single
mid-removal (71% — closest to threshold). So Day 6 says "L4 is the
hardest to remove without harm" and Day 7 learned-skip agrees ("L4 is
the layer the model wants to keep most"). Day 5's ranking missed this
specifically. **For Day 10-12: when verification methods conflict, lean
on gradient-based / ablation-based signals over manual scoring.**

### Track A reframe (relevant to the Day 14 portfolio writeup)

The original Track A goal was: "find the minimum number of layers that
maintains 75%+ quality." Days 6-7 establish that **for Llama 3.2 1B,
this minimum is 16.** No layer count below 16 holds 75%.

The reframed deliverable: **characterize the redundancy structure** —
how the layers degrade under removal, which orderings are best, what
the 1B model's "dense edge" looks like. The interesting story is the
gap between Day 5's optimistic per-layer importance scores and the
actual per-layer ablation damage.

### Implication for Days 10-12

Layer pruning gave a small win at most. **Cache reduction (Track B)
operates at finer granularity** — we're dropping individual *cached
tokens* rather than whole layers. That's a smaller intervention with
more headroom. The 5%+ compute-savings target for Track B is still
realistic; the 75% quality / N<16 target for Track A is not.

**Updated patch verification kit (after Day 7):**
- top-1 exact-match across full prompt suite (Day 6)
- avg baseline-prob delta (Day 5 method 2)
- 1 - cos(input, output) per layer (Day 5 method 3)
- repetition rate as collapse canary (Day 6)
- selective freeze pattern for any future learnable-component patches (Day 7)

