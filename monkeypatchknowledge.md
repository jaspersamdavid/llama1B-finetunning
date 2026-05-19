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

---

## Day 8 (May 8) — Speed/memory benchmarks + Track A close — what the cost of a layer actually is

### What we built
`src/pruning/benchmark.py` — measures the actual cost (params, memory, TTFT, tokens/sec) of each pruning configuration. Eight representative configs benchmarked spanning the full quality spectrum from 100% down to 0%.

### Implementation patterns relevant to monkey-patching

- **Honest wall-clock timing on MPS requires explicit sync.** MPS dispatches ops async; without `torch.mps.synchronize()` around the timer, `time.perf_counter()` reports queue-submission time, not work-completion time. Numbers will look 2-5× better than reality.

  ```python
  def sync(device):
      if device.type == "mps":
          torch.mps.synchronize()
      elif device.type == "cuda":
          torch.cuda.synchronize()

  sync(model.device)
  t0 = time.perf_counter()
  out = model.generate(...)
  sync(model.device)   # ← critical, NOT optional
  elapsed = time.perf_counter() - t0
  ```

  **For Day 10-12 patch verification:** if our cache-eviction patch claims "5% compute saved," the honest comparison requires this exact pattern around both pre- and post-patch generation. Don't trust any benchmark missing the synchronize calls.

- **Warm-up forward before timing.** First call into a code path triggers kernel compilation / Metal shader caching. Without a warm-up gen, the first measurement is 2-5× slower than steady state. The pattern:
  ```python
  # Warm-up — don't time
  model.generate(**inputs, max_new_tokens=5, ...)
  sync(model.device)

  # Now time
  for _ in range(N_RUNS):
      sync(model.device); t0 = time.perf_counter()
      ...
  ```

- **Median of N runs, not mean.** First runs are still slower than steady-state even after warm-up. Median of 3-5 runs filters this without needing more iterations.

### What we found — the dead Pareto corner

Speed scales **linearly** with layer count: ~3-4% per layer dropped. Memory drops ~116 MB per layer (= 60.8 M FP16 params per decoder layer × 2 bytes).

| Layers | Quality | Speed | Speedup |
|---|---|---|---|
| 16 (baseline) | 100% | 29.3 tok/s | 1.00× |
| 15 (drop L4 — best) | 71% | 30.4 tok/s | 1.04× |
| 15 (drop L12 — smart) | 65% | 30.1 tok/s | 1.03× |
| 14 (drop L12, L7) | 41% | 31.3 tok/s | 1.07× |
| 13 (drop L12, L7, L6) | 41% | 33.0 tok/s | 1.12× |
| 8 (drop L8-15 — collapse) | 0% | 41.4 tok/s | 1.41× |

**Pareto chart is empty in the top-right corner.** The best 15-layer config gets you 71% quality for a 4% speedup. The threshold (75% quality) is unreachable for any layer count below 16. This is the visual that closes Track A.

### Why this matters for Days 10-12 (Track B)

**The headroom for Track B is fundamentally different.** Pruning whole layers buys 3-4% per layer at huge quality cost. KV cache eviction at the *token* granularity:

- Doesn't touch model weights → quality preservation is structurally easier
- Compounds with sequence length → savings grow as context grows (whole-layer pruning is constant)
- Is reversible per-token-position → can be tuned without re-training

**The Track A speed numbers (29.3 tok/s @ full 16 layers, 30.7 ms TTFT) become the baseline that Day 10-12's cache patches must beat.** If a cache patch saves memory but generation slows, the patch failed. The Track B target is: reduce cache memory by ≥25% while keeping tok/s within 2% of baseline AND keeping quality within 90% of baseline.

### Track A close — methodology that transfers to Days 10-12

Track A built and validated these reusable pieces:

| Tool | Day | Reused for |
|---|---|---|
| `WeightTweaker.snapshot_layer / restore_layer` | 4-5 | Day 10-12 cache patch revert |
| `LayerPruner` (ref-only ModuleList swap, layer_idx renumbering) | 6 | Any layer-mutating Day 10-12 experiment |
| `evaluate()` 4-metric scorecard | 6 | Patch verification |
| `score_lens_and_cosine`, `score_zero_out` | 5 | Patch verification |
| `SkippableLayer` wrapper pattern | 7 | Layer-wrapping monkey-patches |
| `benchmark.py` timing infrastructure | 8 | Day 10-12 speed regression test |
| Test prompt suite (17 prompts × 5 categories) | 1 | Patch verification |
| Quality threshold convention (75%) + 4-metric kit | 6 | Pass/fail bar for Day 10-12 patches |

**By Day 9 (Track B opens) we have a full verification stack.** Any cache-eviction patch can be A/B-tested with the same `evaluate()` function we used for Track A. The Day 10-12 deliverables become provable, not anecdotal.

### Closing observations on the 1B model itself

The 1B model is at the lower edge of dense pruning tolerance. Three independent angles converge on this:
1. **Direct ablation (Day 6):** no single removal preserves 75%
2. **Smart-ranked ablation (Day 7 Exp 3):** even with Day 5 ranking, no config crosses 75%
3. **Gradient-based pruning (Day 7 Exp 4):** even with full gradient access, no layer is willing to be skipped

This is not a tooling story — it's a property of Llama 3.2 1B. Larger models (3B, 7B, 70B) are known in the literature to be substantially more prunable. **The methodology transfers; the specific outcome is model-size-specific.** That framing is the Day 14 portfolio claim.

---

## Day 9 (May 19) — KV Cache Profile → confirmed huge eviction headroom + strong sinks

### What we built
`src/kv_optimization/cache_profiler.py` — manual generation loop with `output_attentions=True` at each step, accumulating per-cached-position attention received across all queries.

### Implementation patterns relevant to monkey-patching

- **`output_attentions=True` + `use_cache=True` is the data source.** Each forward pass during generation returns attentions of shape `[1, num_heads, 1, kv_len]` per layer. **This is exactly the slice we need for Day 10-12 patches.** Day 9's profiler doubles as the Day 10-12 verification A/B framework.

- **Manual generation loop (not `model.generate()`) is required.** `model.generate()` doesn't surface per-step attentions cleanly. Pattern:
  ```python
  out = model(**inputs, output_attentions=True, use_cache=True)   # prompt forward
  past_kv = out.past_key_values
  for step in range(max_new_tokens):
      out = model(input_ids=next_token, past_key_values=past_kv,
                  output_attentions=True, use_cache=True)
      # attentions[L]: [1, H, 1, kv_len]
      past_kv = out.past_key_values
      next_token = out.logits[0, -1].argmax().view(1, 1)
  ```
  The manual loop gives Day 10-12 a hook point to apply eviction *between* steps.

- **Detach + numpy immediately.** Each forward produces ~860K floats of attentions across 16 layers. Across 100 steps × 17 prompts that's GBs if held. We extract `attn.sum(dim=(0, 1, 2)).cpu().numpy()` immediately and let the GPU tensor GC. **Day 10-12 patches that read attention scores must follow the same discipline.**

### What we found — the cache is mostly empty

**94.6% of cached tokens are "dead"** (receive less than 1% of their layer's attention budget). Stable across all 17 prompts (std 1.1%, range 92.3-96.7%). The cache is dominated by very few positions.

**Per-layer dead rate ordering** (sorted by eviction headroom):

| Most-dead (safest to evict) | Dead % | Least-dead (densest cache) | Dead % |
|---|---|---|---|
| L1 | 98.9% | L8 | 87.3% |
| L2 | 98.8% | L7 | 87.8% |
| L3 | 98.4% | L9 | 88.1% |
| L13 | 97.9% | L6 | 91.5% |
| L4 | 97.3% | L0 | 93.1% |

### Cross-validation with Track A — independent agreement

The "least-dead" cache layers (L7, L8, L9) overlap with **Day 6's hardest-to-remove single layers** (L7 = 53% exact match, L11 = 41%, L9 = 59%). Two completely different methodologies converging:

- Track A (ablation): "removing L8 breaks 35% of prompts"
- Track B (cache profile): "L8's cache has the densest attention pattern (lowest dead rate at 87.3%)"

Both say: **L7-L9 are doing real work and need their cache intact.**

### Attention sinks — confirmed strong, must preserve

The BOS token (position 0) receives **43-81% of attention per layer**. L2 has 81.4% concentrated at position 0; L1 has 74.5%. Even L0 has 43.1%. This is THE attention sink described in the StreamingLLM paper, and it's strong on Llama 3.2 1B.

**Implication for Day 10-12:** position 0 (and probably positions 0-3 for safety) MUST stay in cache. Evicting them collapses the model. Window-only eviction will hurt; sink + window is the safe baseline.

### Memory framing (the awkward truth at the 1B scale)

At our sequence lengths (~100 tokens), the KV cache is **0.1% of total memory**. Even at 4000 tokens it's only 5%. Model weights (2.36 GB) dominate.

**Implication:** at the 1B scale, **Track B's value is compute savings per generation step, not memory savings.** Each evicted cached token reduces one row of the attention matmul per step. For a 4000-token sequence with 95% eviction, we compute 200 K/V rows instead of 4000 — a 20× attention compute reduction per step. For 70B-class models at 100k+ context, the same technique buys huge memory savings too.

### What Day 10-12 patches will look like (Day 9 priors)

- **Day 10 (sink + window)** — keep positions 0-3 always; keep last N tokens. With 95% dead rate, N ≈ 25% of cache should work well.
- **Day 11 (per-layer sizing)** — L1, L2, L3 get tiny windows (10-20 tokens). L7-L9 get bigger windows (50-100 tokens). This is exactly the eviction-headroom signal Day 9 surfaces.
- **Day 11 (importance eviction)** — instead of keeping by recency, keep cached tokens with highest cumulative attention received. Day 9's `attn_received` array IS this signal; Day 11 just needs to track it live during generation.

### Saved data for Day 10-12 to consume
- `outputs/cache_profiles/day9_canonical_attention.npz` — per-layer × per-position attention for the canonical prompt; heatmap; prompt_len.
- `outputs/cache_profiles/day9_per_layer_attention.csv` — avg + std dead-rate per layer across 17 prompts.

### Updated patch verification kit (post-Day 9)

The kit from Days 5-8 still applies. Day 9 adds:

- **Cache attention preservation check** — after running a patch, recompute per-layer dead rate using `profile_prompt()`. If a patch breaks the model, you'll typically see either (a) dead rate flips to ~0% (attention spreads uniformly = degenerate) or (b) repetition rate spikes (model lost access to sink).
- **Sink budget check** — after eviction, verify position 0 is still in cache for every layer. Forgetting to preserve the sink is the most common Day 10-12 bug.

---

## Day 10 (May 19) — Between-step cache trimming, no monkey-patching needed (yet)

### What we built
`src/kv_optimization/token_reducer.py` — two eviction strategies + sink verification across 17 prompts. **Crucial finding for monkey-patching planning:** we got both strategies working **without** actually monkey-patching `LlamaAttention.forward()`. The trick was to mutate the DynamicCache between generation steps instead of inside the attention forward.

### Implementation pattern: between-step cache mutation

```python
def trim_cache(past_kv, keep_first: int, keep_last: int):
    """Keep first K + last N positions per layer. Drop the middle."""
    for layer in past_kv.layers:
        K, V = layer.keys, layer.values
        seq_len = K.shape[-2]
        if seq_len <= keep_first + keep_last:
            continue
        layer.keys = torch.cat([K[..., :keep_first, :], K[..., -keep_last:, :]], dim=-2)
        layer.values = torch.cat([V[..., :keep_first, :], V[..., -keep_last:, :]], dim=-2)
```

Then in the generation loop:
```python
out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
past_kv = out.past_key_values
trim_cache(past_kv, keep_first=4, keep_last=budget - 4)  # ← between-step mutation
next_token = out.logits[0, -1].argmax().view(1, 1)
```

**Why this works without re-applying RoPE:**
- Cached K vectors retain the RoPE they got when generated → consistent
- The new query's RoPE position is computed by transformers as `past_kv.get_seq_length() + step`. After trim, this gives the new query a position equal to the *trimmed* seq length, not the *original* seq length.
- This is equivalent to **StreamingLLM's positional shift**: kept tokens get re-indexed to consecutive positions from the model's POV. The math works because attention sinks absorb the position-shift weirdness.

**Implication for Days 11-12:** if our advanced strategies (importance eviction, per-layer sizing) can also be expressed as "trim between steps," we don't need to monkey-patch the attention class at all. The Day 4 `monkeypatching.md` parking lot may stay parked. Monkey-patching only becomes necessary for techniques that need to alter the math *within* the attention forward (e.g., custom softmax, per-head sparsification).

### What we found — sink is structural, sink-preservation is decisive

**Sink verification across 17 prompts (mean BOS attention share per layer):**

| Layer | Mean | Std | Layer | Mean | Std |
|---|---|---|---|---|---|
| L0 | 49.4% | 2.7 | L8 | 49.1% | 4.2 |
| L1 | 76.2% | 1.6 | L9 | 48.2% | 3.9 |
| L2 | **80.5%** | 2.8 | L10 | 60.4% | 6.1 |
| L3 | 73.5% | 2.4 | L11 | 68.9% | 3.2 |
| L4 | 65.8% | 2.4 | L12 | 69.4% | 4.3 |
| L5 | 59.8% | 4.0 | L13 | 75.5% | 3.6 |
| L6 | 53.3% | 4.2 | L14 | 70.8% | 3.2 |
| L7 | 53.1% | 3.2 | L15 | 66.0% | 2.5 |

Overall: **63.7%** of all attention goes to position 0 on average. Std is tight (2-6% per layer). **The sink is structural — it's not a property of any single prompt or any single layer; it's how this model routes attention.**

### Eviction results — sink preservation matters at every budget

| Strategy | Sink | Budget | Match% | Repetition |
|---|---|---|---|---|
| window-only | 0 | 100 (no-evict) | 100% | 0.49 |
| window-only | 0 | 30 | 86.1% | 0.53 |
| window-only | 0 | 20 | 50.2% | 0.62 |
| window-only | 0 | 10 | 21.8% | 0.68 |
| sink+window | 4 | 100 (no-evict) | 100% | 0.49 |
| **sink+window** | **4** | **30** | **92.0%** | **0.50** |
| sink+window | 4 | 20 | 63.5% | 0.49 |
| sink+window | 4 | 10 | 24.1% | 0.50 |

**Sink+window wins at every comparable budget.** Two complementary signals:
1. **Match%**: sink+window gives +5.9pp at b=30, +13.3pp at b=20.
2. **Repetition rate**: sink+window stays at 0.49-0.50; window-only climbs to 0.68. Losing the sink doesn't just degrade quality — **it sends the model into token loops.**

### A surprise — the Day 6 top-1 metric is blind to eviction

Day 10 had to introduce a new metric. Day 6's `evaluate()` computes "does the pruned model produce the same top-1 *first* generated token as baseline?" With eviction, the first generated token is unaffected because eviction kicks in only when the cache exceeds budget — which doesn't happen until later steps.

**Step0Ok% was 100% for every config tested.** The eviction damage is in steps 5+, not step 0.

**Day 10's new metric:** `match_pct = % of positions in a 30-token greedy generation that match the no-evict baseline`. This catches degradation that happens mid-sequence.

**For Day 11-12 patches:** use 30-token match against baseline, NOT top-1 of first token. The verification kit gets an addition:
- Day 5 KL Δ, Day 5 cosine sim — sensitive to representation drift
- Day 6 exact match — sensitive to top-1 flips ON THE FIRST TOKEN (use only for non-eviction patches)
- **Day 10 30-token match** — sensitive to mid-sequence degradation, the correct metric for eviction
- Day 6 repetition rate — collapse canary, especially for sink-bypass bugs
- Day 9 dead-rate / sink-attention check — sanity check that attention pattern still looks normal

### Where the quality cliff is

| Budget | Sink+window match% | Cache reduction |
|---|---|---|
| 100 | 100% | 0% (no eviction) |
| 30 | 92.0% | ~25% of 40-token max cache |
| 20 | 63.5% | ~50% |
| 10 | 24.1% | ~75% |

**The cliff is between budget=30 and budget=20.** Above 30, very good quality preservation. Below 30, things fall apart fast. At our test sequence lengths (~40 tokens max cache), budget=30 means we're saving ~25% memory and compute per step while keeping 92% match.

For Day 11 to do better, **smarter eviction within the middle** is required. The "dead" tokens we identified in Day 9 are mostly in the middle of the cache. Importance-based eviction (track cumulative attention, drop bottom-K from the middle) should push the cliff lower.

### Updated map of which approach to take for which patch

| Patch type | Use this | Monkey-patch needed? |
|---|---|---|
| Trim cache by position (window, sink+window) | between-step `trim_cache` | NO |
| Importance-based eviction by cumulative attention | between-step trim + cumulative tracking | NO (track via hooks or in the gen loop) |
| Per-layer different budgets | between-step trim, layer-by-layer | NO |
| KV cache INT8/INT4 quantization | between-step quantize on cache.keys/values | NO |
| Custom attention math (uniform softmax, top-k attn) | rewrite forward | YES (Approach #3 from `monkeypatching.md`) |
| Cache eviction *during* attention (look at scores) | rewrite forward | YES |

**Most of Days 11-12 likely stays in the "no monkey-patching" column.** Day 12's combined-best strategy may still avoid full monkey-patching if we stack between-step techniques smartly.

### Saved data for Day 11-12
- `outputs/cache_profiles/day10_per_prompt_sink.csv` — per-prompt-per-layer BOS attention share. Day 11's per-layer cache sizing could combine this with Day 9's dead-rate data.
- `outputs/cache_profiles/day10_eviction_results.csv` — strategy × budget quality, baseline for any Day 11-12 strategy comparison.

---

## Day 11 (May 19) — Importance eviction + quantization → **first technique that ACTUALLY needs monkey-patching**

### What we built
`src/kv_optimization/advanced_eviction.py` with three new techniques (one of which crashed and surfaced the first genuine monkey-patching requirement).

### THE Day 11 monkey-patching finding

**Per-layer cache sizing CRASHES with between-step trimming.** This is the first technique on this project that genuinely requires monkey-patching. The crash:

```
RuntimeError: The size of tensor a (11) must match the size of tensor b (12)
at non-singleton dimension 3
```

In `eager_attention_forward`: `attn_weights = attn_weights + attention_mask`. Different cache lengths across layers → different `attn_weights` shapes → only one `attention_mask` exists → shape mismatch on the layer whose cache doesn't match the mask.

**Why this breaks the "between-step mutation" pattern:**
- `LlamaModel.forward()` computes ONE `cache_position` and ONE `causal_mask` for the whole stack
- It assumes every layer's cache has the same length (because that's the normal case)
- Our previous Day 10 trim always took every layer to the same budget — so this never surfaced

**What it means for Day 12:** to make per-layer sizing work, we'll need to monkey-patch `LlamaAttention.forward()` (or `LlamaModel.forward()`'s mask construction) so each layer slices the attention mask to its own cache length. This is **Approach #3 from `monkeypatching.md`** — and we now have a real, motivating reason to use it, not just a hypothetical one.

### Implementation pattern for Strategy 3 — Importance eviction

`ImportanceState` class tracks cumulative attention per layer per cached position. Update after every forward, slice when we trim:

```python
class ImportanceState:
    def __init__(self, num_layers):
        self.cumulative = [np.zeros(0) for _ in range(num_layers)]

    def update(self, attentions, kv_len):
        for L, attn in enumerate(attentions):
            received = attn.sum(dim=(0, 1, 2)).cpu().numpy()
            # extend if cache grew, then accumulate
            ...
            self.cumulative[L][:kv_len] += received

    def trim(self, past_kv, sink_size, recent_min, budget):
        for L in range(self.num_layers):
            keep = list(range(sink_size)) \
                 + sorted(top_k_indices_from_middle(
                       self.cumulative[L][sink_size:seq_len-recent_min],
                       n=budget - sink_size - recent_min)) \
                 + list(range(seq_len - recent_min, seq_len))
            layer.keys = layer.keys[..., keep, :]
            layer.values = layer.values[..., keep, :]
            self.cumulative[L] = self.cumulative[L][keep]   # realign
```

**Critical detail:** when you trim by index list (not by slice), you MUST realign your tracking arrays to the new cache layout. Otherwise the cumulative scores point at the wrong positions on the next step.

**For Day 12 patches** that track *anything* over generation steps (importance, dead time, age, attention entropy, etc.), use this same realignment pattern.

### What we found — diminishing returns on smarter eviction

Comparison at same budget:

| Budget | sink+window (Day 10) | importance (Day 11) | Δ |
|---|---|---|---|
| 30 | 92.0% | 92.9% | +0.9pp |
| 20 | 63.5% | 67.1% | +3.6pp |
| 15 | (not tested) | 48.0% | — |

**The gain is small.** At our short cache lengths (~40 tokens max), there's only ~16 "middle" positions to score. Most of them have similar (low) cumulative attention because the sink absorbs everything. Recency does most of the work.

**Implication for the portfolio writeup:** importance-based eviction is a *known* technique from the literature (H2O, etc.) and should win bigger on longer sequences (1000+ tokens) where the middle pool is large enough for selection to matter. Our 1B / 30-gen-token setup is a stress test of the *floor* of the technique, not a showcase. Document this honestly.

### Quantization — significant quality cost at the 1B scale

| Precision | Match % | Repetition | Theoretical memory savings |
|---|---|---|---|
| FP16 (baseline) | 100% | 0.49 | 0% |
| INT8 | **72.2%** | 0.41 | 50% |
| INT4 | **16.9%** | 0.58 | 75% |

INT8 round-trip costs **28 percentage points of generation quality** on the 1B model. INT4 is catastrophic — same coherence-loss pattern as Day 10's sink-eviction (repetition spike, near-random output).

**Why this matters for monkey-patching plans:**
- Naive per-tensor quantization doesn't survive on this model
- To make INT8 work, we'd need **per-channel quantization** (separate scale per attention head) or **calibration-based quantization** (compute scales from a representative batch). Both involve more sophisticated cache-storage code.
- **None of this needs monkey-patching the attention forward** — quantization stays a between-step operation. But the symmetric per-tensor approach we tried is too crude.

### Verification kit update (post-Day 11)

The new piece: **per-layer cache-shape consistency check.** Before any patch that touches cache shapes, verify all 16 layers have the same `cache.keys.shape[-2]` (or accept that you're committing to a monkey-patch). The most common Day 12 bug will be a partial trim that leaves shapes inconsistent.

```python
def assert_uniform_cache_shape(past_kv):
    lens = [layer.keys.shape[-2] for layer in past_kv.layers]
    assert len(set(lens)) == 1, f"non-uniform cache lengths: {lens}"
```

### Map of monkey-patching necessity — UPDATED for Day 11

| Patch type | Approach | Monkey-patch needed? |
|---|---|---|
| Window, sink+window | between-step `trim_cache` | NO |
| Importance-based eviction | between-step trim + cumulative tracking | NO |
| **Per-layer different budgets** | **rewrite attention mask construction** | **YES** ← Day 11 confirmed |
| KV cache INT8 (per-tensor) | between-step `quantize_cache` | NO |
| KV cache INT4 (per-tensor) | between-step `quantize_cache` | NO (but quality bad) |
| Per-channel quantization (per-head scale) | between-step with calibration | NO |
| Custom attention math (uniform softmax, top-k attn) | rewrite forward | YES |
| Cache eviction *during* attention (look at scores) | rewrite forward | YES |

**Day 12 will be the first time we actually monkey-patch.** Strategy 4 (per-layer sizing) is the motivating reason. The `monkeypatching.md` parking-lot doc finally gets to be used.

### Saved data for Day 12
- `outputs/cache_profiles/day11_advanced_eviction.csv` — full results table for Day 12's comparison plot
- Day 11 code reuses Day 10 and Day 9 modules cleanly — Day 12 should follow the same modular pattern (don't re-implement what Days 9-11 built)

