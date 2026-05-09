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
