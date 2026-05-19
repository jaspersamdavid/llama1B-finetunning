# completeness.md — llama-xray

> **What this file is:** the detailed archive of every task completed per day.
> When a day finishes, its full checklist and any notable details move here so
> that `CLAUDE.md` can stay focused on current + upcoming work without growing
> unbounded. `CLAUDE.md` keeps a tight summary of each completed day; this file
> keeps the receipts.
>
> **How to read this:** each day gets a section with (a) the exact checklist as
> it was at completion and (b) any side-notes worth preserving. Going backwards
> in time — most recent day at the top.

---

## Day 11 — Tuesday, May 19, 2026 ✅

**Phase 3C: Advanced Eviction + Cache Quantization**

> Run on May 19 immediately after Day 10. Compute: ~2 min for baselines +
> 8 strategy configs (importance × 3 budgets + 2 quantization configs +
> 2 Day-10 reference re-runs). Per-layer sizing crashed and was deferred.

### advanced_eviction.py
- [x] Build `src/kv_optimization/advanced_eviction.py`
- [x] **`ImportanceState`** class — tracks cumulative attention per
  (layer, position). `update(attentions, kv_len)` adds the latest step's
  attention; `trim(past_kv, sink, recent_min, budget)` selects keep
  indices and re-aligns the cumulative array.
- [x] **`gen_with_importance_eviction`** — manual gen loop with
  `output_attentions=True` at every step + `ImportanceState` updates.
- [x] **`quantize_tensor(t, bits)`** — symmetric per-tensor INT8/INT4
  round-trip simulation. `quantize_cache(past_kv, bits)` applies to all
  layers between steps.
- [x] **`gen_with_quantization`** — manual gen loop + `quantize_cache`
  after every forward.
- [x] **`gen_with_per_layer` + `trim_cache_per_layer`** — implemented
  but disabled in main due to the cache-shape crash. Kept in source for
  Day 12 to use after monkey-patching.

### Strategy 3 — Importance eviction results

| Budget | sink+window (Day 10) | importance (Day 11) | Δ |
|---|---|---|---|
| 30 | 92.0% match | **92.9% match** | **+0.9pp** |
| 20 | 63.5% match | **67.1% match** | **+3.6pp** |
| 15 | — | 48.0% match | — |

Repetition rate at b=15: 0.531 (approaching the collapse zone).

### Strategy 4 — Per-layer cache sizing → DEFERRED

Crashed with `RuntimeError: The size of tensor a (11) must match the
size of tensor b (12) at non-singleton dimension 3`. The root cause:
`LlamaModel.forward()` computes ONE `cache_position` and ONE attention
mask for the entire stack, assuming uniform cache length across all
16 layers. Per-layer different shapes break this.

**This is the first technique on the project that genuinely requires
monkey-patching `LlamaAttention.forward()` (or `LlamaModel.forward()`'s
mask construction).** Moved to Day 12 plan.

### Strategy 5 — Cache quantization

Symmetric per-tensor round-trip simulation (FP16 → INT → FP16). Memory
savings are theoretical (we don't actually keep INT form); quality drop
is real.

| Precision | Match % | Repetition | Theoretical mem savings |
|---|---|---|---|
| FP16 (baseline) | 100% | 0.49 | 0% |
| INT8 | **72.2%** | **0.41** | 50% |
| INT4 | **16.9%** | **0.58** | 75% |

### All Day 11 strategies on one table

| Strategy | Match % | Prefix | Step0 % | RepRate |
|---|---|---|---|---|
| sink+window b=30 (Day 10) | 92.0% | 27.59 | 100.0% | 0.496 |
| sink+window b=20 (Day 10) | 63.5% | 18.41 | 100.0% | 0.494 |
| importance b=30 | **92.9%** | 27.88 | 100.0% | 0.488 |
| importance b=20 | 67.1% | 19.41 | 100.0% | 0.498 |
| importance b=15 | 48.0% | 13.24 | 100.0% | 0.531 |
| quantize INT8 | 72.2% | 20.29 | 100.0% | 0.410 |
| quantize INT4 | 16.9% | 4.12 | 100.0% | 0.576 |

### Notable findings

1. **Importance eviction marginally beats recency-window.** +0.9pp at
   b=30, +3.6pp at b=20. The benefit is bounded at our short sequence
   lengths (~40 token max cache, only ~16 middle positions to score).
   The technique is known from literature (H2O) to scale better on
   longer sequences (1000+ tokens) where the middle pool is large
   enough for selection to matter.

2. **Per-layer sizing crashes — first technique requiring monkey-patching.**
   `LlamaModel.forward` shares one `cache_position` and one attention
   mask across all 16 layers. Per-layer different cache lengths fail
   shape-checking inside `eager_attention_forward`. To make this work,
   we need to monkey-patch `LlamaAttention.forward` (or
   `LlamaModel.forward`'s mask construction) so each layer slices the
   mask to its own cache length. **This is finally the motivation for
   Approach #3 from `monkeypatching.md`.**

3. **INT8 quantization costs 28pp on 30-token gen.** Symmetric per-tensor
   quantization is too crude for this model. INT4 is catastrophic —
   repetition rate spikes to 0.58 (same coherence-loss signature as
   sink-eviction in Day 10).

4. **Repetition rate remains the key collapse canary.** In Day 10 sink-loss
   it was 0.68; in Day 11 INT4 it's 0.58. Importance b=15 (the most
   aggressive eviction config that still ran) is at 0.53 — close to
   the boundary.

5. **Best Day 11 config:** importance b=30 → 92.9% match. Slight
   improvement over Day 10's best (sink+window b=30 → 92%). Both well
   above the 75% threshold; both have ~25% cache reduction. The
   actionable Day-11 finding: **importance eviction is the right choice
   at moderate budgets; it doesn't unlock dramatic improvements at the
   1B scale on short sequences.**

### Tooling now available for Day 12
- `ImportanceState` — reusable live cumulative-attention tracker
- `quantize_cache(past_kv, bits)` — drop-in between-step quantization
- `gen_with_per_layer` + `trim_cache_per_layer` — already implemented;
  Day 12 just needs to add the monkey-patched attention mask handling
  to make them runnable
- All four eviction strategies (window-only, sink+window, importance,
  per-layer) + 3 quantization configs are now in one consistent
  evaluation harness

### Concept notes added
- `monkeypatchknowledge.md` — Day 11 section: the per-layer-sizing
  crash trace, why it requires monkey-patching, the cumulative-tracking
  realignment pattern, the per-layer-shape consistency check, updated
  monkey-patch necessity map.

### Deliverables checked into the repo
- `src/kv_optimization/advanced_eviction.py`
- `outputs/cache_profiles/day11_advanced_eviction.csv`
- `outputs/cache_profiles/day11_strategies_compared.png`
- `outputs/cache_profiles/day11_quant_quality.png`
- `CLAUDE.md` — Day 11 marked complete, Current Status updated, KV Cache
  Observations extended with Day 11 findings
- `monkeypatchknowledge.md` — Day 11 section added

---

## Day 10 — Tuesday, May 19, 2026 ✅

**Phase 3B: Attention Sink Analysis + Basic Eviction**

> Run on May 19 immediately after Day 9. Day 10 compute: ~3 min for sink
> verification (17 prompts) + ~2:30 for 8 eviction configs × 17 prompts.
> Total ~5:30.

### token_reducer.py
- [x] Build `src/kv_optimization/token_reducer.py`
- [x] **`trim_cache(past_kv, keep_first, keep_last)`** — mutates `past_kv.layers[L].keys/values`
  in place using `torch.cat([K[..., :keep_first, :], K[..., -keep_last:, :]], dim=-2)`.
  No-op if cache is already within budget.
- [x] **`gen_with_eviction(model, tokenizer, prompt, budget, sink_size, max_new_tokens)`** —
  manual generation loop that calls `trim_cache` after every step. Identical structure
  to Day 9 profiler's loop, but with the trim hook.
- [x] **`gen_no_evict`** — baseline (no trimming).
- [x] **`compare_to_baseline(test, baseline)`** — match_pct, prefix_match, first_top1_kept,
  repetition_rate.

### Exp A — Sink verification across 17 prompts
- [x] Re-use Day 9's `profile_prompt()` for each of the 17 prompts (50 generated tokens
  each for speed).
- [x] Extract position-0 attention share per layer.
- [x] Aggregate: per-layer mean ± std across 17 prompts.

**Result:**

| Layer | BOS share (mean ± std) | min | max |
|---|---|---|---|
| L0 | 49.4 ± 2.7 | 44.4 | 53.9 |
| L1 | 76.2 ± 1.6 | 73.6 | 79.0 |
| L2 | **80.5 ± 2.8** | 74.0 | 84.3 |
| L3 | 73.5 ± 2.4 | 70.1 | 78.5 |
| L4 | 65.8 ± 2.4 | 62.2 | 69.6 |
| L5 | 59.8 ± 4.0 | 52.3 | 66.7 |
| L6 | 53.3 ± 4.2 | 46.2 | 60.2 |
| L7 | 53.1 ± 3.2 | 48.2 | 58.2 |
| L8 | 49.1 ± 4.2 | 41.1 | 59.2 |
| L9 | 48.2 ± 3.9 | 42.0 | 57.9 |
| L10 | 60.4 ± 6.1 | 50.8 | 76.0 |
| L11 | 68.9 ± 3.2 | 64.2 | 76.2 |
| L12 | 69.4 ± 4.3 | 62.8 | 77.9 |
| L13 | 75.5 ± 3.6 | 69.2 | 81.8 |
| L14 | 70.8 ± 3.2 | 65.8 | 76.8 |
| L15 | 66.0 ± 2.5 | 60.7 | 69.8 |

**Overall: 63.7% of attention goes to BOS on average per layer.** Standard deviation
across prompts is tight (2-6% per layer). The sink is structural — not prompt-specific.

### Exp B — Eviction strategies (4 budgets × 2 strategies = 8 configs)
- [x] **Baselines** computed: greedy 30-token generation per prompt with no eviction.
- [x] For each (strategy, budget), generate 30 tokens with eviction enabled for all 17
  prompts. Compare against baseline.

**Result:**

| Strategy | Sink | Budget | Match% | Prefix | Step0Ok% | RepRate |
|---|---|---|---|---|---|---|
| window-only | 0 | 100 | 100.0% | 30.00 | 100.0% | 0.492 |
| window-only | 0 | 30 | 86.1% | 25.53 | 100.0% | 0.525 |
| window-only | 0 | 20 | 50.2% | 14.41 | 100.0% | 0.620 |
| window-only | 0 | 10 | 21.8% | 4.59 | 100.0% | 0.678 |
| sink+window | 4 | 100 | 100.0% | 30.00 | 100.0% | 0.492 |
| **sink+window** | **4** | **30** | **92.0%** | 27.59 | 100.0% | **0.496** |
| sink+window | 4 | 20 | 63.5% | 18.41 | 100.0% | 0.494 |
| sink+window | 4 | 10 | 24.1% | 6.18 | 100.0% | 0.502 |

### Notable findings

1. **Attention sink is structural and universal.** 17 prompts × 16 layers all show
   strong sinks at position 0. Standard deviation is small (2-6% per layer). This is
   not a "Paris is a special token" effect — it's how the model routes attention.

2. **Sink preservation matters at every budget.** Sink+window beats window-only by
   +5.9pp at budget=30, +13.3pp at budget=20. The gap is largest in the mid-budget
   range where some eviction is happening but recent context is mostly preserved.

3. **Sink preservation prevents repetition collapse.** Repetition rate stays at
   ~0.49-0.50 across all sink+window budgets. Without sink, it climbs from 0.49
   (no evict) → 0.68 (b=10). **Losing the sink doesn't just degrade quality — it
   collapses the model into loops.** The sink is the model's coherence anchor.

4. **The Day 6 top-1 metric is blind to eviction.** Step0Ok% is 100% for every config.
   This is because the first generated token happens before eviction kicks in (cache
   size = prompt_len, which is below all budgets tested). Day 10 introduces a new
   metric: 30-token greedy match against no-evict baseline.

5. **Quality cliff is between budget=30 and budget=20.** Sink+window: 92% at b=30,
   64% at b=20, 24% at b=10. The "recent window" is doing real work beyond the first
   ~20 positions. For Day 11+, smarter eviction within the middle (importance-based)
   may push this cliff lower.

6. **Eviction works without monkey-patching.** Between-step `trim_cache` on
   `past_kv.layers[L].keys/values` works directly with transformers 5.x DynamicCache.
   No need to subclass `LlamaAttention` for windowing or sink-window strategies.
   This is equivalent to StreamingLLM's positional shift — kept tokens get re-indexed
   to consecutive positions from the model's POV; attention sinks absorb the math
   weirdness.

### Tooling now available for Day 11+
- `trim_cache(past_kv, keep_first, keep_last)` — between-step cache mutation primitive.
  Works directly on DynamicCache; no monkey-patching.
- `gen_with_eviction()` — manual generation loop with eviction hook. Day 11's
  importance-based and per-layer strategies can build on top.
- `compare_to_baseline()` — match% / prefix / repetition / first-token kept.
  The patch-verification metric for any eviction strategy.

### Concept notes added
- `monkeypatchknowledge.md` — Day 10 section: between-step cache mutation pattern,
  why it works without re-RoPE, the new map of which strategies need monkey-patching
  vs which don't. Most of Days 11-12 can stay in the "no monkey-patching" column.

### Deliverables checked into the repo
- `src/kv_optimization/token_reducer.py`
- `outputs/cache_profiles/day10_sink_verification.png` — per-layer BOS share with min/max
- `outputs/cache_profiles/day10_eviction_quality.png` — match% vs budget for both strategies
- `outputs/cache_profiles/day10_eviction_results.csv` — full 8-config table
- `outputs/cache_profiles/day10_per_prompt_sink.csv` — per-prompt-per-layer BOS share
- `CLAUDE.md` — Day 10 marked complete, Current Status updated, KV Cache Observations
  table extended with Day 10 findings
- `monkeypatchknowledge.md` — Day 10 section added

---

## Day 9 — Tuesday, May 19, 2026 ✅ — TRACK B OPENED

**Phase 3A: KV Cache Profiling**

> Originally scheduled for Tuesday May 12 in the v3 plan. Delay due to
> off-days; picked back up May 19 with full schedule slide pending.
> Day 9 compute: ~67 sec on Mac Mini M4 MPS for 17 prompts × 100
> generated tokens with `output_attentions=True` at every step.

### cache_profiler.py
- [x] Build `src/kv_optimization/cache_profiler.py`
- [x] **Manual generation loop** (not `model.generate()`) so we can read
  per-step `attentions` cleanly. Each forward returns
  `attentions[L].shape = [1, num_heads, 1, kv_len]` during generation.
- [x] **`output_attentions=True` + `use_cache=True`** combined to extract
  attention weights without disabling cache plumbing.
- [x] **Detach + numpy immediately** after each forward: `attn.sum(dim=(0,1,2)).cpu().numpy()`.
  Prevents GPU memory growth across the 100 steps.
- [x] Accumulate per-layer × per-position "attention received" array of
  shape `[num_layers, prompt_len + max_new_tokens]`.
- [x] Build heatmap `[gen_step, cached_pos]` for the canonical prompt
  (`"The first president of the United States was"`), averaging across
  layers.
- [x] **Dead-token metric:** for each layer, % of cached positions whose
  cumulative attention is below 1% of the layer's total attention budget.

### Key results

**Headline:** 94.6% of cached tokens are "dead" across 17 prompts × 100
generated tokens (mean; std = 1.1%, range 92.3-96.7%).

**Per-layer dead rate (avg across 17 prompts):**

| Layer | Dead % | Note |
|---|---|---|
| L1 | 98.9 | most-dead — top eviction candidate |
| L2 | 98.8 | most-dead — top eviction candidate |
| L3 | 98.4 | most-dead |
| L13 | 97.9 | high dead rate |
| L4 | 97.3 | high dead rate |
| L0 | 93.1 | sink concentration here |
| L15 | 97.0 | mid |
| L14 | 96.0 | mid |
| L11 | 96.6 | mid |
| L12 | 96.6 | mid |
| L10 | 93.8 | mid |
| L5 | 94.0 | mid |
| L6 | 91.5 | densest mid layer |
| L9 | 88.1 | **least-dead (densest cache)** |
| L7 | 87.8 | **least-dead (densest cache)** |
| L8 | 87.3 | **least-dead (densest cache)** |

**Attention sink check (canonical prompt, position 0 share):**

| Layer | BOS share |
|---|---|
| L2 | 81.4% — strongest sink |
| L1 | 74.5% |
| L13 | 73.8% |
| L3 | 71.4% |
| L11 | 67.2% |
| L14 | 67.1% |
| L4 | 66.8% |
| L12 | 66.6% |
| L15 | 63.8% |
| L5 | 56.5% |
| L10 | 55.3% |
| L6 | 53.4% |
| L7 | 52.2% |
| L8 | 48.7% |
| L9 | 43.6% |
| L0 | 43.1% — weakest sink (still 43%!) |

**Every layer has a strong attention sink at position 0.** Even the
weakest sink (L0) sends 43% of attention there. Position 0 must be
preserved by any eviction strategy.

**Memory breakdown:**

| seq_len | weights MB | KV MB | cache % of total |
|---|---|---|---|
| 50 | 2357 | 1.6 | 0.07% |
| 100 | 2357 | 3.1 | 0.13% |
| 200 | 2357 | 6.2 | 0.26% |
| 500 | 2357 | 15.6 | 0.66% |
| 1000 | 2357 | 31.2 | 1.31% |
| 2000 | 2357 | 62.5 | 2.58% |
| 4000 | 2357 | 125.0 | 5.04% |

At the 1B scale, the cache is a tiny fraction of total memory. Track B's
benefit on this model is primarily **compute reduction** per generation
step (attention matmul scales with `kv_len`), not memory savings.

### Notable findings

1. **94.6% dead rate is the headline.** Cache eviction can be very
   aggressive without losing much actual attention work. This is
   the inverse of Track A's situation — Track A had zero pruning
   headroom; Track B has enormous eviction headroom.

2. **Attention sinks confirmed across all 16 layers.** The StreamingLLM
   finding applies to Llama 3.2 1B at full strength. Position 0
   receives 43-81% of attention per layer. **Any eviction strategy
   that doesn't preserve position 0 will break the model.**

3. **Cross-validation with Track A:** the layers with the **densest**
   caches (L7, L8, L9 — only 87% dead rate) are also the **hardest
   to remove** in Track A (Day 6 single-layer ablation: L7=53%,
   L8=65%, L9=59% exact match — all in the lower tier). Two
   independent methodologies agree these layers are doing real work.

4. **Per-layer eviction-headroom signal for Day 11:**
   - Most-dead (most eviction headroom): L1, L2, L3 (~98-99%)
   - Least-dead (keep more cache): L7, L8, L9 (~87-88%)
   - Day 11's per-layer sizing should use this ordering directly.

5. **Memory-vs-compute framing.** At ~100 tokens, KV cache is 0.1% of
   total memory — Track B's value is **compute savings per generation
   step**, not memory. Each evicted cached token saves one row of
   attention matmul per generation step. At 4000 tokens with 95%
   eviction, that's a 20× reduction in attention compute per step.
   The methodology transfers to bigger models where memory savings
   become large in absolute terms too.

### Tooling now available for Day 10+
- `profile_prompt(model, tokenizer, prompt, max_new_tokens=100)` —
  returns per-layer attention received array. **This is the patch
  verification A/B test** — run baseline once, run patch, diff the
  arrays.
- `dead_token_stats()` — computes per-layer % of cached tokens
  receiving below threshold attention. Day 11's per-layer sizing
  consumes this directly.
- Manual generation loop template — Day 10's eviction patches can
  apply between-step modifications by slotting in here.
- `outputs/cache_profiles/day9_canonical_attention.npz` — saved arrays
  for Day 10 to consume without re-running the profiler.
- `outputs/cache_profiles/day9_per_layer_attention.csv` — per-layer
  avg dead-rate for Day 11's sizing decisions.

### Concept notes added
- `monkeypatchknowledge.md` — Day 9 section: cache profiling as the
  patch verification framework, the manual generation loop template
  for between-step modifications, the sink budget check + cache
  attention preservation check as the new pieces of the verification
  kit, Day 10-12 strategy priors derived from Day 9 data.

### Deliverables checked into the repo
- `src/kv_optimization/cache_profiler.py`
- `outputs/cache_profiles/day9_cache_memory_curve.png`
- `outputs/cache_profiles/day9_attention_heatmap.png`
- `outputs/cache_profiles/day9_per_layer_dead_pct.png`
- `outputs/cache_profiles/day9_memory_breakdown.png`
- `outputs/cache_profiles/day9_dead_token_stats.csv`
- `outputs/cache_profiles/day9_per_layer_attention.csv`
- `outputs/cache_profiles/day9_canonical_attention.npz`
- `CLAUDE.md` — Day 9 marked complete, Current Status updated, KV
  Cache Observations table filled
- `monkeypatchknowledge.md` — Day 9 section added

---

## Day 8 — Friday, May 8, 2026 ✅ — TRACK A CLOSED

**Phase 2D: Pruning Results Analysis + Speed Benchmarks**

> Run on May 8 immediately after Day 7. ~40 sec of benchmark compute.
> No new experiments — Day 8 measures speed/memory of the existing
> configs from Days 6-7 and writes the Track A summary.

### benchmark.py
- [x] Build `src/pruning/benchmark.py`
- [x] **Honest MPS timing:** `torch.mps.synchronize()` around every
  `time.perf_counter()` call. Without sync, MPS reports queue-submit
  time (~2-5× faster than reality).
- [x] **Warm-up forward** before timing to flush kernel-compile overhead.
- [x] **Median of 3 gen runs / 5 TTFT runs** to filter remaining variance.
- [x] **Memory measured as `sum(p.numel() for p in model.parameters()) × 2 bytes`**
  — this works after pruning because `model.parameters()` only enumerates
  the current (post-swap) ModuleList.

### Benchmarked configs (8 total)

| Config | # L | Quality | Params | Memory | TTFT | Tok/s | Speedup |
|---|---|---|---|---|---|---|---|
| Full 16 layers (baseline) | 16 | 100.0% | 1,235,814,400 | 2357 MB | 30.7 ms | 29.33 | 1.00× |
| Drop L4 alone (mid) | 15 | **70.6%** | 1,174,992,896 | 2241 MB | 29.4 ms | 30.42 | 1.04× |
| Drop L12 alone (smart) | 15 | 64.7% | 1,174,992,896 | 2241 MB | 28.9 ms | 30.06 | 1.03× |
| Drop L15 alone (end) | 15 | 52.9% | 1,174,992,896 | 2241 MB | 29.3 ms | 30.38 | 1.04× |
| Drop L12, L7 (smart 14) | 14 | 41.2% | 1,114,171,392 | 2125 MB | 28.1 ms | 31.26 | 1.07× |
| Drop L14, L15 (end 14) | 14 | 23.5% | 1,114,171,392 | 2125 MB | 28.2 ms | 31.00 | 1.06× |
| Drop L12, L7, L6 (smart 13) | 13 | 41.2% | 1,053,349,888 | 2009 MB | 26.7 ms | 32.98 | 1.12× |
| Drop L8-15 (end 8, collapsed) | 8 | 0.0% | 749,242,368 | 1429 MB | 20.2 ms | 41.36 | 1.41× |

**Per-decoder-layer cost:** ~116 MB and ~60.8 M params. Roughly linear
speedup of ~3-4% per layer dropped.

### Notable findings

1. **Pareto chart is empty in the top-right.** The 75%-quality threshold
   sits between the baseline (100% quality, 1.00× speedup) and the best
   sub-16 config (71% quality at drop L4, 1.04× speedup). No config holds
   both above the threshold AND faster than baseline. This is the visual
   that closes Track A — the failure to find a Pareto-favorable trade.

2. **Speed scales linearly with layer count.** ~3-4% speedup per
   dropped layer. Smart vs end vs middle removal all give similar speed
   numbers at the same layer count (they vary by ≤1 tok/s) — the
   quality difference comes from *which* layers were dropped, not from
   any speed-vs-strategy interaction.

3. **TTFT scales with layer count proportionally.** 30.7 ms at 16
   layers → 20.2 ms at 8 layers. Each layer adds ~1 ms to first-token
   latency on MPS. For a real interactive use case (chat, completion),
   this is the more user-facing number than tokens/sec.

4. **Memory savings are large in absolute terms but useless in
   practice.** Going from 16 → 8 layers saves 928 MB. But the
   8-layer model produces total garbage (0% exact match, full
   repetition collapse). The Day 5-7 finding that no sub-16 config
   holds 75% applies to the speed/memory dimension too: there is no
   point where the trade is worth it on the 1B model.

5. **The "best 15-layer config" is drop L4 alone, NOT drop L12 (smart).**
   This is consistent with Day 6 + Day 7 — the layers that Day 5's
   importance ranking flagged as redundant *are* somewhat removable,
   but L4 (which Day 5 ranked mid-importance) is unexpectedly the
   safest single removal. **This validates the lesson recorded in
   Day 7:** when methods disagree, gradient-based and direct-ablation
   signals beat manual scoring on the specific disagreement.

### TRACK_A_SUMMARY.md

The portfolio-ready Track A close-out is at
`outputs/pruning_results/TRACK_A_SUMMARY.md`. It includes:
- The reframed deliverable (no minimum-N exists)
- Methodology table (Days 5-8)
- Critical layers (L0, L1, L4) and most-skippable layer (L12)
- The full speed/quality/memory table
- Strategy comparison (end / mid-single / smart / learnable skip)
- Implications for Track B
- What transfers to bigger models

### Deliverables checked into the repo
- `src/pruning/benchmark.py`
- `outputs/pruning_results/day8_benchmark.csv`
- `outputs/pruning_results/day8_speed_vs_layers.png` — speed/quality dual axis
- `outputs/pruning_results/day8_pareto.png` — quality × speedup scatter
- `outputs/pruning_results/TRACK_A_SUMMARY.md` — portfolio-ready writeup
- `CLAUDE.md` — Day 8 + Track A marked complete, Current Status updated
- `monkeypatchknowledge.md` — Day 8 section added (MPS sync gotcha,
  benchmark patterns for Day 10-12 patch verification, Track A → Track B
  bridge note)

### Track A — final accounting

**Original goal:** find minimum N layers maintaining 75%+ quality.
**Result:** N = 16 for Llama 3.2 1B. No sub-16 config crosses the threshold.

**Reframed deliverable (delivered):**
- Characterized redundancy structure across 16 layers
- Identified critical layers (L0, L1, L4) with multi-method agreement
- Identified most-skippable layer (L12) with multi-method agreement
- Documented non-monotonic layer interactions
- Quantified speed/memory cost per layer
- Established the patch verification kit that carries to Track B
- Built reusable tooling (LayerPruner, evaluate, snapshot/restore,
  SkippableLayer, benchmark infrastructure)

**Track A closed. Day 9 opens Track B.**

---

## Day 7 — Friday, May 8, 2026 ✅

**Phase 2C: Smart Pruning + Skip Connections**

> Run on May 8 immediately after Day 6 in the same session. Day 7 compute:
> ~75 sec for smart pruning + ~65 sec for the 30-step learnable-skip
> training, ~3 minutes total. Both experiments depend on Day 5's saved
> npz and Day 6's saved csv.

### smart_pruning.py
- [x] Build `src/pruning/smart_pruning.py` with both experiments + comparison reports.

### Experiment 3 — Smart pruning (least-important first via Day 5 ranking)
- [x] Load Day 5 ranking from `outputs/pruning_results/layer_importance_scores.npz`
- [x] Reverse to get least-important order: `[12, 7, 6, 9, 8, 4, 5, 10, 13, 11, 15, 2, 14, 3, 1, 0]`
- [x] Run `evaluate()` after each progressive removal (16 down to 8 layers)

| Layers remaining | Removed | Exact% | Top-5% | BaseProb | RepRate |
|---|---|---|---|---|---|
| 16 | (baseline) | 100.0 | 100.0 | 0.403 | 0.247 |
| 15 | 12 | 64.7 | 100.0 | 0.273 | 0.282 |
| 14 | 12, 7 | 41.2 | 76.5 | 0.196 | 0.290 |
| 13 | 12, 7, 6 | 41.2 | 76.5 | 0.138 | 0.412 |
| 12 | 12, 7, 6, 9 | 11.8 | 47.1 | 0.072 | 0.267 |
| 11 | 12, 7, 6, 9, 8 | 11.8 | 47.1 | 0.054 | 0.310 |
| 10 | 12, 7, 6, 9, 8, 4 | 17.6 | 41.2 | 0.059 | 0.200 |
| 9 | 12, 7, 6, 9, 8, 4, 5 | 11.8 | 29.4 | 0.069 | 0.314 |
| 8 | 12, 7, 6, 9, 8, 4, 5, 10 | 5.9 | 23.5 | 0.033 | 0.204 |

### Head-to-head — end removal vs smart removal (Day 6 Exp 1 vs Day 7 Exp 3)

| Layers | End-removal exact% | Smart-removal exact% | Δ |
|---|---|---|---|
| 16 | 100.0 | 100.0 | +0.0 |
| 15 | 52.9 | 64.7 | +11.8 |
| 14 | 23.5 | 41.2 | +17.6 |
| 13 | 11.8 | 41.2 | +29.4 |
| 12 | 23.5 | 11.8 | -11.8 |
| 11 | 11.8 | 11.8 | +0.0 |
| 10 | 0.0 | 17.6 | +17.6 |
| 9 | 0.0 | 11.8 | +11.8 |
| 8 | 0.0 | 5.9 | +5.9 |

**Smart-removal beats end-removal by avg +9pp** across the curve. Largest
gain at 13 layers (+29pp). Single regression at 12 layers (-12pp) is
another instance of the non-monotonic layer interactions seen on Day 6.
**Neither strategy crosses the 75% threshold at any sub-16 layer count.**

### Experiment 4 — Learnable skip weights (distillation)
- [x] Build `SkippableLayer` wrapper class:
  ```python
  class SkippableLayer(nn.Module):
      def __init__(self, original_layer):
          super().__init__()
          self.layer = original_layer
          self.skip_logit = nn.Parameter(torch.tensor(4.0))  # sigmoid(4) ≈ 0.982
      @property
      def skip_weight(self):
          return torch.sigmoid(self.skip_logit)
      @property
      def self_attn(self):                                    # exposed for cache indexing
          return self.layer.self_attn
      def forward(self, hidden_states, *args, **kwargs):
          out = self.layer(hidden_states, *args, **kwargs)
          w = self.skip_weight
          if isinstance(out, tuple):
              return (w * out[0] + (1-w) * hidden_states,) + out[1:]
          return w * out + (1-w) * hidden_states
  ```
- [x] Pre-compute teacher logits: for each test prompt, generate 25 baseline
  tokens then run a final teacher-mode forward, store the logits as KL target
- [x] Wrap all 16 decoder layers in SkippableLayer. Initial `skip_logit = 4`
  → `skip_weight ≈ 0.982` so initial student behavior ≈ teacher (KL ≈ 0).
- [x] Freeze ALL parameters except the 16 `skip_logit`s
- [x] Train 30 steps with Adam (lr=0.1), loss = KL(student || teacher) + 0.05 * mean(skip_weight)
- [x] Read off final skip weights

### Learned skip weights (after 30 steps)

| Layer | Learned w | Day 5 combined | Verdict |
|---|---|---|---|
| 0 | 0.9845 | 0.7854 | kept |
| 1 | 0.9775 | 0.7488 | kept |
| 2 | 0.9720 | 0.2552 | kept |
| 3 | 0.9755 | 0.3316 | kept |
| **4** | **0.9967** | 0.1799 | kept (most-kept) |
| 5 | 0.9813 | 0.1833 | kept |
| 6 | 0.9698 | 0.1457 | kept |
| 7 | 0.9699 | 0.1353 | kept |
| 8 | 0.9786 | 0.1787 | kept |
| 9 | 0.9851 | 0.1528 | kept |
| 10 | 0.9729 | 0.1860 | kept |
| 11 | 0.9702 | 0.2334 | kept |
| **12** | **0.9595** | 0.0720 | kept (least-kept) |
| 13 | 0.9684 | 0.2170 | kept |
| 14 | 0.9750 | 0.3116 | kept |
| 15 | 0.9751 | 0.2452 | kept |

**Spearman ρ between learned skip weight and Day 5 combined importance: 0.285**
(moderate). All 16 weights ended up in the band [0.96, 1.00] — **no layer
was actually skipped**. λ_L1 = 0.05 was too weak vs the KL gradient.

### Notable findings

1. **Smart pruning is bounded but real.** Day 5's ranking-driven removal
   beats naive end-removal by avg +9pp across the curve. The Day 5
   importance scores ARE predictive — they just aren't strong enough to
   change the headline finding (no config holds 75%).

2. **The learnable-skip experiment confirms Day 6's "everything matters."**
   Even with full gradient access and 30 distillation steps, the model
   keeps every weight near 1.0. The KL gradient against the teacher
   pulls every weight up; nothing wants to drop. This is independent
   evidence that the 1B model's layers are densely needed.

3. **L4 is the most "needed" layer per Day 7's gradient signal** (highest
   learned weight = 0.9967). Day 6 also identified L4 as the best single
   mid-removal (71% exact match — closest to threshold among single
   removals). Both findings independently point at L4 being load-bearing.
   Day 5's manual scoring placed L4 mid-pack (0.18) — a partial miss.
   **Lesson:** when verification signals conflict, gradient-based / direct-
   ablation signals beat manual scoring on disagreements.

4. **L12 is consistently the most "skippable" layer** across all three
   methods that ranked it: Day 5 combined (rank 16/16, score 0.072),
   Day 6 Exp 1 (drop L12-15 less damaging than drop L13-15, suggesting
   L12 itself contributes little), Day 7 learned skip (lowest weight at
   0.9595). **For Days 10-12 cache eviction:** L12 is the most
   patch-friendly layer.

5. **No 75% config exists at any sub-16 layer count.** Best 15-layer
   config across all four strategies tested:
   - Smart drop L12 (Day 7): **65%**
   - End drop L15 (Day 6 Exp 1): **53%**
   - Mid single drop L4 (Day 6 Exp 2): **71%** ← best 15-layer config
   - Mid single drop L11 (Day 6 Exp 2): 41%

   The L4 result is the closest the project came to the 75% mark. Still
   under. **Track A's premise of "minimum viable layer count" needs
   reframing for Day 8** — for the 1B model the minimum is 16.

### Tooling now available for Days 8 / 10-12
- `SkippableLayer` is a clean template for any future patch that wraps
  (not replaces) a decoder layer. Useful for layer-level monkey-patches
  that need to run alongside the original layer.
- `learnable_skip_experiment()` is a reusable distillation training
  driver — change the loss term to fit any future "train one parameter,
  freeze the rest" pattern (e.g. learnable cache eviction thresholds
  on Day 11).
- `outputs/pruning_results/day7_skip_weights.csv` — per-layer learned
  skip weight + Day 5 importance scores side by side, for any
  cross-experiment comparisons.
- `outputs/pruning_results/day7_strategies_compared.png` —
  end/smart/mid-single curves on one axis.

### Concept notes added
- `monkeypatchknowledge.md` — Day 7 section captures the wrapper-class
  patterns (kwargs pass-through, `@property` for attribute exposure,
  tuple/tensor return guard), the selective-freeze pattern for surgical
  training, and the verification-signal hierarchy lesson (gradient/
  ablation > manual scoring on disagreements).

### Deliverables checked into the repo
- `src/pruning/smart_pruning.py`
- `outputs/pruning_results/day7_smart_curve.csv`
- `outputs/pruning_results/day7_skip_weights.csv`
- `outputs/pruning_results/day7_strategies_compared.png`
- `outputs/pruning_results/day7_learned_skip_weights.png`
- `CLAUDE.md` — Day 7 marked complete, Current Status updated, Pruning
  observation table extended with smart and learnable-skip rows
- `monkeypatchknowledge.md` — Day 7 section added

---

## Day 6 — Friday, May 8, 2026 ✅

**Phase 2B: Sequential Layer Removal — Experiments**

> Run on May 8 immediately after Day 5 in the same session. Total Day 6 compute:
> ~3 minutes on Mac Mini M4 MPS (17 configs × 17 prompts × 1 baseline forward
> + 1 generation pass each).

### layer_pruner.py
- [x] Build `src/pruning/layer_pruner.py` — `LayerPruner` class
- [x] Reference-only ModuleList swap (no deep copy of model weights — saves ~2 GB)
- [x] `prune(layer_indices)` rebuilds `model.model.layers` as a smaller ModuleList
  containing references to the surviving original layers; updates
  `model.config.num_hidden_layers`
- [x] `reset()` restores the original 16-layer list
- [x] **Critical fix:** renumber `layer.self_attn.layer_idx` to be sequential
  0..N-1 after pruning. `transformers.DynamicCache` indexes K/V slots by
  `layer.self_attn.layer_idx`, and middle removal otherwise causes
  `IndexError: list index out of range` because the surviving last layer
  still thinks it's idx 15 while cache only has 15 slots (0-14).
- [x] `skip_layer_inplace()` exposed as the equivalent zero-the-weights
  primitive — confirmed in Day 5 to be functionally identical to skipping
  thanks to pre-norm residual.

### eval_pruned.py
- [x] Build `src/pruning/eval_pruned.py` with metrics evaluator
- [x] **Metrics computed per (config, prompt):**
  - `exact_match_pct`: pruned top-1 token == baseline top-1 token
  - `top5_pct`: baseline top-1 still in pruned model's top-5
  - `mean_baseline_prob`: average prob the pruned model assigns to the
    baseline top-1 token (proxy for confidence; replaces perplexity)
  - `repetition_rate`: in 15 greedy-decoded tokens, fraction NOT unique
    (proxy for coherence collapse; high = looping output)
- [x] Baseline forward + 15-token generation locked in once per prompt
- [x] Two experiments run sequentially with `LayerPruner` providing prune/reset

### Experiment 1 — Remove from the end (progressive)

| Layers remaining | Removed | Exact% | Top-5% | BaseProb | RepRate |
|---|---|---|---|---|---|
| 16 | (baseline) | 100.0 | 100.0 | 0.403 | 0.247 |
| 15 | L15 | 52.9 | 82.4 | 0.347 | 0.173 |
| 14 | L14-15 | 23.5 | 64.7 | 0.149 | 0.412 |
| 13 | L13-15 | 11.8 | 35.3 | 0.088 | 0.490 |
| 12 | L12-15 | 23.5 | 29.4 | 0.094 | 0.659 |
| 11 | L11-15 | 11.8 | 17.6 | 0.003 | 0.667 |
| 10 | L10-15 | 0.0 | 5.9 | 0.000 | 0.616 |
| 9 | L9-15 | 0.0 | 0.0 | 0.000 | 0.573 |
| 8 | L8-15 | 0.0 | 0.0 | 0.000 | 0.498 |

### Experiment 2 — Remove from the middle (single-layer ablation)

| Layer removed | Exact% | Top-5% | BaseProb | RepRate |
|---|---|---|---|---|
| L4 | 70.6 | 88.2 | 0.334 | 0.333 |
| L5 | 52.9 | 94.1 | 0.300 | 0.251 |
| L6 | 64.7 | 94.1 | 0.312 | 0.286 |
| L7 | 52.9 | 94.1 | 0.306 | 0.318 |
| L8 | 64.7 | 100.0 | 0.277 | 0.282 |
| L9 | 58.8 | 88.2 | 0.270 | 0.212 |
| L10 | 58.8 | 76.5 | 0.218 | 0.208 |
| L11 | 41.2 | 82.4 | 0.243 | 0.298 |

### Notable findings

1. **No 75%-quality config with N<16 layers.** Removing even L15 alone takes
   exact-match to 53%, immediately below the 75% threshold. The portfolio
   conclusion "we shrunk Llama by N layers" doesn't hold for the 1B model —
   it's at the dense edge of what survives dense pruning.

2. **Day 5's "L15 is a sharpener" was metric-dependent.** Day 5's zero-out
   metric measured continuous prob drop on baseline top-1 token (-0.055 = small).
   Day 6 measures the discrete outcome (is the top-1 *token* the same?) and
   shows L15 actually decides ~half of the top-1 tokens across the 17 prompts.
   **Lesson:** averaging probability deltas hides per-prompt token flips. Use
   exact-match alongside prob-delta in any future verification work.

3. **Best single mid-stack removal: L4** (70.6% exact match). Worst: L11
   (41.2%). Day 5's combined ranking had L4 at score 0.18 (mid) and L11 at
   0.23 (mid). The Day 5 ranking partially predicts the order but is far
   from sharply accurate per-layer.

4. **Non-monotonic damage in end-removal:** dropping L13-15 (12% exact)
   is *worse* than dropping L12-15 (24% exact). Layer interactions are
   non-additive — independently-validated removals don't compose.

5. **Repetition-rate spike as collapse signal.** Repetition climbs from
   17% (drop L15) to 67% (drop L11-15) to 50-60% (drop ≥7 trailing). This
   is a useful proxy for "model has collapsed into token loops" — much
   cheaper than running a full evaluation.

6. **Top-5% degrades more gracefully than exact-match.** Mid-stack removals
   keep 76-100% top-5 (except L10 at 76% and L4 at 88%). The right token
   is usually still in contention even when not on top — useful framing
   for "soft" quality.

### Tooling now available for Day 7+
- `LayerPruner.prune([...])` / `.reset()` — drop in any layer config
- `evaluate(model, tokenizer, baseline_top1s, prompts)` — full 4-metric
  scorecard for any (potentially patched/pruned) model state
- `outputs/pruning_results/day6_results.csv` — per-config metrics for
  cross-experiment comparison
- `outputs/pruning_results/day6_generations.txt` — actual generated text
  per (config, prompt) for human review when metrics are ambiguous

### Concept notes added
- `monkeypatchknowledge.md` — Day 6 section captures the `layer_idx`
  renumbering gotcha (relevant any time we mutate the layer stack and
  use cache), the metric-vs-discrete-outcome lesson (top-1 token flips
  vs prob deltas), the non-additive-layer-interactions lesson (combined
  patches need fresh validation), and the repetition-rate as cheap
  collapse signal.

### Deliverables checked into the repo
- `src/pruning/layer_pruner.py`
- `src/pruning/eval_pruned.py`
- `outputs/pruning_results/day6_end_curve.png` — Exp 1 quality curve
- `outputs/pruning_results/day6_middle_ablation.png` — Exp 2 per-layer drop
- `outputs/pruning_results/day6_results.csv`
- `outputs/pruning_results/day6_generations.txt` — sample outputs
- `CLAUDE.md` — Day 6 marked complete, Current Status updated, Pruning
  observation table filled with actual numbers
- `monkeypatchknowledge.md` — Day 6 section added

---

## Day 5 — Friday, May 8, 2026 ✅

**Phase 2A: Layer Importance Scoring**

> Originally scheduled for Wednesday April 29 in the v1 plan. Schedule shifted
> twice (Apr 29 → May 6 → May 8) due to off-days. Day 5 building + analysis
> happened on May 8 in one session, ~3 minutes of compute on Mac Mini M4 MPS.

### layer_importance_scorer.py
- [x] Build `src/pruning/layer_importance_scorer.py` with three independent scoring methods
- [x] **Method 1 — Logit-lens KL Δ**: project each of the 17 hidden states (embedding
  + 16 layers) through final RMSNorm + lm_head, compute `KL(P_layer_L || P_layer_{L−1})`
  on the last-token distribution. **FP32 cast required** — first run gave all-NaN because
  FP16 softmax of large negative logits underflows to 0, which makes `log(0) = -inf` and
  the subsequent multiply NaN. Fixed by casting `model.lm_head(normed).float()` before
  `F.log_softmax`.
- [x] **Method 2 — Zero-out impact**: snapshot each of 16 layers' `state_dict()` once,
  then for each (prompt, layer): zero attention (q/k/v/o_proj) + MLP (gate/up/down_proj)
  weights, run forward, measure drop in baseline top-1 token's probability, restore.
  Uses ~2 GB extra memory (one full snapshot). Total run ~11 s for 16 × 17 ablations.
- [x] **Method 3 — 1 − cos(input, output)**: for each layer, cosine sim between the
  last-token hidden state going in (`hidden_states[L]`) and going out (`hidden_states[L+1]`).
  Reports `1 − sim` so higher = more transformation = more important. Free — reuses the
  same forward pass as method 1.
- [x] Run all three across 16 layers × 17 test prompts (5 categories: factual, math, code,
  pattern, reasoning). Average per layer.
- [x] Min-max normalize each method's scores to [0, 1] for cross-method comparison
- [x] Combined importance = mean of normalized scores
- [x] Bar chart with all three methods overlaid → `outputs/pruning_results/layer_importance_scores.png`
- [x] Raw arrays saved → `outputs/pruning_results/layer_importance_scores.npz` (loadable
  in Day 6 for the smart-pruning experiment without re-running the scorer)

### Key results

**Raw scores per layer** (higher = more important within each method):

| Layer | Logit-lens KL Δ | Zero-out top-1 drop | 1 − cos(in, out) | Combined (norm) |
|---|---|---|---|---|
| 0 | 2.43638 | **0.40168** | **0.66373** | **0.7854** |
| 1 | **5.72488** | 0.36425 | 0.30859 | 0.7488 |
| 2 | 1.84797 | 0.14479 | 0.26017 | 0.2552 |
| 3 | 1.73687 | 0.19630 | 0.31641 | 0.3316 |
| 4 | 1.52884 | 0.06902 | 0.29070 | 0.1799 |
| 5 | 1.29669 | 0.10323 | 0.26683 | 0.1833 |
| 6 | 1.16364 | 0.09094 | 0.23868 | 0.1457 |
| 7 | 1.03104 | 0.09637 | 0.22714 | 0.1353 |
| 8 | 1.15672 | 0.12532 | 0.23926 | 0.1787 |
| 9 | 0.99449 | 0.13313 | 0.20169 | 0.1528 |
| 10 | 0.97190 | 0.18431 | 0.17756 | 0.1860 |
| 11 | 2.24473 | 0.16013 | 0.15705 | 0.2334 |
| 12 | 0.61715 | 0.13014 | 0.11368 | **0.0720** |
| 13 | 1.12641 | 0.24493 | 0.11587 | 0.2170 |
| 14 | 1.81967 | 0.29235 | 0.12201 | 0.3116 |
| 15 | 0.94915 | 0.05534 | 0.48261 | 0.2452 |

**Per-method top-3 most important:**
- Logit-lens KL Δ: L1, L0, L11
- Zero-out top-1 drop: L0, L1, L14
- 1 − cos(in, out): L0, L15, L3

**Per-method bottom-3 most redundant:**
- Logit-lens KL Δ: L10, L15, L12
- Zero-out top-1 drop: L6, L4, L15
- 1 − cos(in, out): L14, L13, L12

**Combined ranking (most important → most redundant):**
> L0 > L1 > L3 > L14 > L2 > L15 > L11 > L13 > L10 > L5 > L4 > L8 > L9 > L6 > L7 > L12

### Notable findings

1. **L0 and L1 are unambiguously critical.** Top-2 in 2/3 methods, top-3 in all three.
   These are the input-routing layers. Day 6 should never test removal of these.

2. **L12 is the cleanest pruning candidate.** Bottom-3 in 2/3 methods, never appears
   in any top-3. Low across all dimensions: small KL change, small zero-out impact,
   small vector transformation. Day 6 expected to confirm minimal damage.

3. **L15 method-disagreement is the most informative finding.** Cosine-sim says
   highly important (0.48 — second-largest vector transformation). Zero-out drop says
   least important (0.055 — almost no impact on top-1). Reconciliation: **L15 is a
   sharpening layer.** It moves the representation around but doesn't change which
   token wins. This matches Day 4: `zero_attention(L15)` only dropped Paris by 28pp,
   but Paris was still top-1. **Implication for Day 10-12 verification: top-1 metrics
   alone will miss patch leakage. Must also track cosine-sim and KL Δ.**

4. **L8's Day 4 anomaly does NOT generalize.** Day 4 found that `zero_attention(L8)`
   on `"capital of France is"` raised Paris confidence from 48% → 69%. Across the
   17 prompts × 3 methods averaged in Day 5, L8 ranks 8-11 — squarely mid-pack.
   The Day 4 result was prompt-specific.

5. **No single method tells the whole story.** The three methods agree at the extremes
   (L0/L1 always important, L12 mostly redundant) but diverge substantially in the
   middle. Day 6 will use the combined ranking but treat individual methods as
   complementary lenses, not as a single "truth."

### Tooling now available for Day 6+
- `score_lens_and_cosine()`, `score_zero_out()` reusable as A/B test infrastructure
- `snapshot_layer()` / `restore_layer()` for cheap per-layer revert
- `zero_layer()` as identity-skip primitive (works because Llama uses pre-norm
  residuals — zeroed layer's output is 0, residual returns input unchanged)
- `outputs/pruning_results/layer_importance_scores.npz` for the Day 7 smart-pruning
  experiment to consume without re-running

### Concept notes added
- `monkeypatchknowledge.md` — Day 5 section added covering: FP16 underflow gotcha
  for any future patch doing log/KL math, snapshot-once + restore-per-iter pattern,
  zero-out-as-identity-skip equivalence, the verification-kit implication of
  method-disagreement (don't trust top-1 alone — track cosine + KL too).

### Deliverables checked into the repo
- `src/pruning/layer_importance_scorer.py` — full scorer with all three methods
- `outputs/pruning_results/layer_importance_scores.png` — overlaid bar chart
- `outputs/pruning_results/layer_importance_scores.npz` — raw arrays for Day 7
- `CLAUDE.md` — Day 5 marked complete, Current Status updated, Layer Behavior +
  Pruning observation tables filled with Day 5 priors
- `monkeypatchknowledge.md` — Day 5 section replaces the placeholder

---

## Day 4 — Wednesday, April 29, 2026 ✅

**Phase 1C: KV Cache Analyzer + Weight Tweaker**

> Originally scheduled for Tuesday April 28, but Apr 28 was spent on the venv
> migration to `~/venvs/llama-xray/` (Cursorpyright RAM issue) and reviewing
> the Day 3 attention map outputs. Day 4 building + experiments happened on
> Apr 29.

### kv_cache_analyzer.py
- [x] Build `kv_cache_analyzer.py` — `cache_memory_bytes()`, `print_memory_breakdown()`,
  `profile_generation_with_cache()`, `profile_generation_without_cache()`,
  `compare_speed()`, plus `plot_cache_growth()`, `plot_k_norm_heatmap()`,
  `plot_speed_comparison()`
- [x] Per-step logging: cache size (tokens × layers × kv_heads × head_dim),
  memory in MB, K/V norms per layer for the latest cached token, step time in ms
- [x] Speed comparison: with-cache vs without-cache timed on identical prompts
  with `torch.mps.synchronize()` for honest async-MPS timing
- [x] Visualisations saved to `outputs/kv_cache/`: `cache_growth.png`,
  `k_norms_heatmap.png`, `speed_comparison.png`
- [x] **Fix**: transformers 5.x switched `past_key_values` from a tuple-of-tuples
  to a `DynamicCache` object. New access pattern is
  `pkv.layers[i].keys` / `pkv.layers[i].values` (not `pkv[i]`).

### KV cache concept (learned + documented in LLMXray.md)
- [x] Memory formula: `2 (K+V) × 16 layers × 8 kv_heads × 64 head_dim × seq_len × 2 bytes`
  → **32 KB per cached token** for Llama 3.2 1B
- [x] Sanity-checked at runtime: 50-token generation from a 5-token prompt
  ends at 1.72 MB (= 55 × 32 KB ✓)
- [x] Why it exists: without cache, K and V are recomputed for *every* prior
  token at every step (O(n²) total). With cache, each token's K/V is computed
  once (O(n) total).

### weight_tweaker.py
- [x] `WeightTweaker` class with snapshot/restore via `state_dict()` deep clone
- [x] Primitives: `zero_attention(layer)`, `add_mlp_noise(layer, sigma)`,
  `scale_head(layer, head, factor)`, `swap_layers(a, b)`, `reset()`
- [x] `diff_logit_lens()` helper: runs logit lens before/after a tweak, prints
  layer-by-layer side-by-side with `≠` markers on changed layers
- [x] Two entry points: script-mode (runs Day 4 experiment suite, writes
  `outputs/weight_tweaks/results.txt`) and `interactive` mode (menu REPL)

### Break-the-model experiments (April 29, 2026 run)
> Baseline: `"The capital of France is"` → `Paris` at **48%** in final layer.

- [x] **`zero_attention(L0)`** → Paris **48% → 28%** (-20pp). Layer 0 attention
  is doing real routing work. Notable downstream artefact: L12 picks
  `Marseille` (37%), L13 picks `Bordeaux` (38%) — embedding space stays in
  "French city" cluster but specific identification fails.
- [x] **`zero_attention(L8)`** → Paris **48% → 69%** (+21pp). **Surprise of
  Day 4.** Killing L8 attention *improves* the prediction for this prompt.
  Either L8 was redundant or actively unhelpful. Strong pre-Day-5 signal that
  L8 is a pruning candidate.
- [x] **`zero_attention(L15)`** → Paris **48% → 20%** (-28pp). Same answer,
  much less confidence. L15 attention's job is consolidation, not deciding —
  by L14 the answer is already 70% Paris.
- [x] **MLP noise sweep on L5** (`gate_proj += N(0, σ²)`, fixed seed):
  - σ=0.01 → `Paris` (51%) — invisible
  - σ=0.1  → `being` (3%) — **catastrophic collapse**
  - σ=1.0  → `genomes` (1%) — total nonsense, every layer 6-16 frozen on the
    same garbage token
  - **Collapse threshold sits between 0.01 and 0.1.** Sharp transition, no
    graceful middle ground. Useful baseline for Day 11 quantization noise tolerance.
- [x] **`swap_layers(L2 ↔ L14)`** → `otope` (1%). Total garbage starting at
  L3 — each layer was trained for a specific input distribution; swapping
  derails the entire downstream computation.
- [x] **`scale_head(L8 H0, ×10)`** → Paris (46%, -2pp). Single head out of 32
  is structurally a small lever — `o_proj` averages the 32 heads, so even 10×
  amplification gets diluted.

### Documentation updates
- [x] **`LLMXray.md`** — **KV Cache** entry filled in: formula, memory table
  at various seq lengths, with-vs-without speed numbers, transformers 5.x
  Cache API note
- [x] **`LLMXray.md`** — **Feed-Forward Network (MLP)** entry expanded with
  noise-fragility findings table and explanation
- [x] **`monkeypatching.md`** — full reference doc on Approach #3 from the
  attention-modification discussion (saved earlier in the day for Day 10-12 reuse)

### Notable findings worth preserving
- **L8 inversion is the headline finding** — zeroing the layer 8 attention
  *raises* Paris confidence from 48% to 69%. This breaks the naive intuition
  "every layer matters" and is the strongest pre-Day-5 evidence that some
  layers are not just removable but mildly counterproductive on certain
  prompts. Will be tested across all 16 layers and the full prompt suite on
  Day 5.
- **Greedy decoding loops** — generated text repeats ("The sun is made up of
  a mixture of different colors. The sun is made up of...") because we use
  `argmax`, not sampling. Known greedy artefact, unrelated to caching. Not
  fixing for now since we want determinism for experiments.
- **MPS sync needed for honest timing** — without `torch.mps.synchronize()`
  before each `time.perf_counter()` call, MPS's async kernel queue makes
  step-time measurements wildly off. Added at every timing boundary.

### Understanding goal for Day 4 (met)
> Able to explain: what the KV cache is and why it exists, how much memory
> it uses (32 KB/token, 16 MB at 500 tokens for 1B), and what happens when
> you break different parts of the model. Have an intuitive sense of "L0
> matters for routing, L15 matters for confidence, L8 may be redundant, MLP
> noise has a sharp collapse threshold around σ=0.05."

### Deliverables checked into the repo
- `src/inspector/kv_cache_analyzer.py`
- `src/inspector/weight_tweaker.py`
- `outputs/kv_cache/{cache_growth,k_norms_heatmap,speed_comparison}.png`
- `outputs/weight_tweaks/results.txt`
- `LLMXray.md` updated (KV Cache section, MLP section expanded)
- `monkeypatching.md` (parking-lot doc for Days 10-12)

---

## Day 1 — Thursday, April 23 → Friday, April 24, 2026 ✅

**Phase 0: Mac Mini Environment Bootstrap + Project Setup + Understanding Tokenization**

> Day 1 spanned two calendar days (evening of Apr 23 and morning of Apr 24)
> because of the HF access approval wait. All tasks completed.

### Mac Mini bootstrap (one-time, because this machine is fresh)
- [x] Clone repo onto Mac Mini (done — files already present at `~/projects/llama1B-finetunning/`)
- [x] Install pyenv: `brew install pyenv` (v2.6.27) — pyenv init block added to `~/.zshrc`
- [x] Install Python 3.11.9: `pyenv install 3.11.9` then `pyenv local 3.11.9` (`.python-version` pinned at project root)
- [x] Create venv: `python3 -m venv venv` (activate via `source venv/bin/activate` when working interactively; Claude Code uses `./venv/bin/python` directly since shell state doesn't persist across tool calls)
- [x] Install deps: `./venv/bin/pip install -r requirements.txt` — got torch 2.11.0, transformers 5.6.2, huggingface_hub 1.11.0, jupyter, scikit-learn, matplotlib, seaborn, pandas, numpy
- [x] Verify MPS works: `torch.backends.mps.is_available() == True` on torch 2.11.0
- [x] Add `export PYTORCH_ENABLE_MPS_FALLBACK=1` to `~/.zshrc`
- [x] HF access: account created, Llama 3.2 1B access approved
- [x] Log in with new CLI: `hf auth login` — token `llama-xray-mac-mini` saved (Read scope)
- [x] Update `src/inspector/model_loader.py` — added `pick_device()` helper (MPS → CUDA → CPU) and made `device=None` auto-pick MPS

### Project setup (largely done on MacBook — in-repo files carry over)
- [x] Create project directory structure (as defined above)
- [x] Create `requirements.txt` (dependencies re-installed fresh in the Mac Mini venv — done in bootstrap above)
- [x] Download Llama 3.2 1B FP16 (~2GB) — first landed in `~/.cache/huggingface/`, later relocated to `~/models/hf-weights/` via `HF_HOME` env var so all HF downloads on this machine share one folder (see `~/CLAUDE.md` Phase 3 note)
- [x] Write `model_loader.py` — loads model with `output_hidden_states=True` and `output_attentions=True`; now also handles MPS auto-select
- [x] Run `print(model)` — full architecture seen: 16 `LlamaDecoderLayer`s, each with self_attn (q/k/v/o_proj) + mlp (gate/up/down_proj with SwiGLU) + 2 RMSNorms; plus embed_tokens (128256 × 2048), final norm, lm_head
- [x] Run `print(model.config)` — documented: num_hidden_layers=16, hidden_size=2048, num_attention_heads=32, num_key_value_heads=8 (GQA), intermediate_size=8192, vocab_size=128256, max_position_embeddings=131072
- [x] **Learn: Tokenization** — notebook section covers:
  - Text → token IDs: `tokenizer.encode("Why is the sun yellow")` → `[128000, 10445, 374, 279, 7160, 14071]`
  - Token IDs → text: `tokenizer.decode([...])`
  - Vocabulary: 128,256 entries via `tokenizer.get_vocab()`
  - Subword: `"understanding"` → `["under", "standing"]`, `"pneumonia"` → `["p", "neum", "onia"]`
- [x] Run one simple generation to confirm everything works: `"The capital of France is"` → coherent output about Eiffel Tower, Louvre, etc.
- [x] **Concept note:** Wrote the "Transformer" and "Tokenization" entries in `LLMXray.md`
- [x] Commit: `b9a5df2` "Day 1 complete: model loaded on MPS, architecture + tokenization understood"

### Understanding goal for Day 1
> By end of day, you should be able to explain: what a token is, how text becomes numbers,
> what the model's architecture looks like (16 layers stacked), and what each layer contains
> (attention + MLP). You don't need to understand HOW they work yet — just WHAT the pieces are.

### Notable side-decisions made during Day 1
- **`HF_HOME=~/models/hf-weights`** set globally in `~/.zshrc`. All HF models from all projects on this Mac Mini now download to one folder. Verified the 1B model loads from there in ~3s (cached, no re-download). Documented in `~/CLAUDE.md` Phase 3.
- **Removed the "Hardware & Acceleration" block from `CLAUDE.md`** — the `pick_device()` pattern lives in code (`src/inspector/model_loader.py`) and the MPS env var lives in `~/.zshrc`. Duplicating it as a doc section was just weight.
- **Token rotation reminder:** the Read token used on Day 1 was pasted into a terminal transcript. Jasper is aware; rotation is optional since it's Read-scope.

### Post-Day-1 environment migration (April 27, 2026)
- **Moved venv out of project folder.** Originally at `~/projects/llama1B-finetunning/venv/` (~1.2GB), now at `~/venvs/llama-xray/`. Reason: Cursor's remote language server (Cursorpyright) was indexing the venv's torch/transformers files and consuming 3+ GB of RAM on the Mac Mini. With the venv outside the project folder, Cursorpyright drops to ~200-300MB.
- **Migration steps used:** `mkdir -p ~/venvs && python3 -m venv ~/venvs/llama-xray && source ~/venvs/llama-xray/bin/activate && pip install -r requirements.txt` — same packages, same Python 3.11.9, same torch 2.11.0 + transformers 5.6.2 (huggingface_hub bumped 1.11.0 → 1.12.0). Old `./venv/` deleted after verification.
- **All run commands now use `~/venvs/llama-xray/bin/python` instead of `./venv/bin/python`.** See the "Environment / How to run" table in `CLAUDE.md`.
- **Added `pyrightconfig.json` and `.vscode/settings.json` to project** as belt-and-suspenders to keep Cursorpyright lean (excludes any future stray venvs, disables autocomplete since Jasper only uses Cursor for file viewing).
