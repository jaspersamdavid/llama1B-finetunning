# LLMXray — Key Concepts Reference

> Living knowledge doc. Every time we learn a concept in the project, we write it up here in plain language — no jargon, no prior ML knowledge assumed. This is the "PhD explained simply" deliverable from the project goals.
>
> Companion to `CLAUDE.md` (which holds the day-by-day task plan and progress).

---

## Transformer

**What it is:** A Transformer is a neural network that reads a sequence of tokens (numbers
representing text) and predicts what comes next. It does this by passing the sequence
through a stack of identical layers, where each layer mixes information between tokens
(attention) and then refines each token individually (MLP).

**Why it exists:** Older models (RNNs) processed text one word at a time, which was slow
and forgot things from far back. Transformers process all tokens in parallel and can
attend to any previous token directly — no matter how far back it is.

**Analogy:** Think of it like a committee meeting that repeats 16 times. At each round:
1. Everyone looks around the room and decides who to pay attention to (attention).
2. Everyone thinks privately about what they just heard (MLP).
3. Their "position in the room" (hidden state) updates.

After 16 rounds, the last person in line gives their answer for what comes next.

**Llama 3.2 1B specifics (observed on April 24, 2026):**
- **16 transformer layers** stacked on top of each other
- **Hidden size: 2048** — each token is represented by a vector of 2048 numbers
- **32 attention heads, 8 KV heads** — this is Grouped Query Attention (GQA): 32 "query"
  heads share only 8 sets of keys/values, which saves memory with little quality loss
- **MLP intermediate size: 8192** — inside the MLP, each token's vector is expanded
  4× (2048 → 8192) then compressed back (8192 → 2048). Uses SwiGLU (gate_proj + up_proj
  combined, then down_proj).
- **RMSNorm** (not LayerNorm) — a simpler normalization that keeps numbers stable
- **RoPE** (Rotary Position Embedding) — how the model knows which token came first.
  Position is baked into the attention math, not added as a separate vector.
- **Total: 1,235,814,400 parameters** (~1.24B, ~2.3GB at FP16)

**The full pipeline:**
```
"The capital of France is"
  → tokenize → [128000, 791, 6864, 315, 9822, 374]         (6 integers)
  → embed    → 6 vectors of size 2048                       (6 × 2048 matrix)
  → layer 0  → 6 vectors (slightly transformed)
  → layer 1  → 6 vectors (more transformed)
  → ...
  → layer 15 → 6 vectors (fully processed)
  → lm_head  → probability distribution over 128,256 possible next tokens
  → pick top → "Paris"
```

**Where we saw it:** `notebooks/01_model_anatomy.ipynb`, Part 2 (architecture print).
Verified by running `python -m src.inspector.model_loader`.

---

## Tokenization

**What it is:** Tokenization is how text becomes numbers the model can work with.
The text is split into "tokens" — usually pieces of words, not whole words — and each
piece gets mapped to a number (its token ID).

**Why it exists:** A computer can't do math on the letter "a". It needs numbers.
You could assign a number to every possible word, but English has millions of words
(including typos, made-up names, URLs). So instead, models use **subword** tokenization:
a fixed vocabulary of common pieces that can be combined to spell anything.

**Analogy:** Like spelling with LEGO blocks. The model has a bin of ~128,000 LEGO
pieces. Common words like "the" are one piece. Longer/rarer words are built from
several pieces: "understanding" = "under" + "standing". Even typos and made-up words
can be assembled from the bin — no word is ever out of vocabulary.

**Llama 3.2 1B specifics (observed on April 24, 2026):**

- **Vocabulary size: 128,256 tokens** (subwords, full words, punctuation, special tokens)
- **BOS token:** `<|begin_of_text|>` (ID 128000) — automatically prepended to every input
- **EOS token:** `<|end_of_text|>` (ID 128001) — signals "stop generating"

**Concrete examples we verified:**
| Text | Token IDs | Pieces |
|---|---|---|
| `"hello"` | `[15339]` | `["hello"]` — common enough to be one token |
| `"Paris"` | `[60704]` | `["Paris"]` — famous enough to be one token |
| `"understanding"` | `[8154, 10276]` | `["under", "standing"]` — split into 2 pieces |
| `"transformers"` | `[4806, 388]` | `["transform", "ers"]` — "ers" suffix is its own piece |
| `"pneumonia"` | `[79, 126261, 21947]` | `["p", "neum", "onia"]` — rare, 3 pieces |

**`"Why is the sun yellow"`** → 6 tokens total (1 BOS + 5 words), each word is one token.
Notice the leading space in " is", " the", " sun", " yellow" — the space is part of the
token, which is how the tokenizer knows where words start.

**Why subword splits are fair:** The model never sees the word "understanding" as a unit
during training — it only sees the sequence `[8154, 10276]`. But because the same two
pieces appear in thousands of other words (`"misunderstanding"`, `"undergo"`, `"outstanding"`),
the model learns what each piece contributes to meaning. Rare words are built from familiar
parts.

**Where we saw it:** `notebooks/01_model_anatomy.ipynb`, Part 3. Use
`tokenizer.encode(text)` for text→IDs and `tokenizer.decode([id])` for ID→text.

## Embeddings
> [To be filled — Day 3]

## Attention (Q, K, V)

**What it is:** Attention is how each token collects relevant information from the other
tokens in the sequence before making its prediction. Every token simultaneously plays three
roles — it asks a question (Query), broadcasts what it contains (Key), and offers its
information if selected (Value).

**Why it exists:** Without attention, the model could only look at one token at a time.
With attention, the last token `"is"` in `"The capital of France is"` can directly reach
back to `"capital"` and `"France"` and pull their meaning in — no matter how far apart they are.

**The three vectors:**
- **Q (Query):** "What am I looking for?" — computed from the current token's hidden state.
  For `"is"` at the end of a geography sentence, Q says something like "I need a place name."
- **K (Key):** "What do I contain?" — each token broadcasts this label.
  `"France"` has a Key that says "I'm a country." `"capital"` has a Key that says "I'm a civic term."
- **V (Value):** "What information do I give out if selected?" — the actual content that gets
  mixed into the output.

**How the score is computed:**
```
score(token_i → token_j) = Q_i · K_j  (dot product)
```
Higher dot product = more relevant. These raw scores are scaled down (divided by √64 for
Llama's 64-dim heads) to keep them from getting too large, then softmax'd so all weights
sum to 1.0. Then:
```
output_i = sum over j of (softmax_score_ij × V_j)
```
So the output for token `i` is a weighted blend of all Values — heavily weighted toward
the most relevant tokens.

**Llama 3.2 1B specifics:**
- Q, K, V are produced by linear projections: `q_proj` (2048→2048), `k_proj` (2048→512),
  `v_proj` (2048→512)
- Note K and V are only 512-dimensional — this is Grouped Query Attention (see below)
- After attention, output is projected back: `o_proj` (2048→2048)

**Where we saw it:** The logit lens experiment on April 26, 2026. The table for
`"The capital of France is"` shows the answer `Paris` only emerging at layer 12 —
that's because 11 layers of Q×K attention are needed to connect "France" + "capital"
into a coherent "Paris" prediction.

---

## Multi-Head Attention

**What it is:** Instead of doing Q×K×V once, the model does it 32 times in parallel —
each with its own learned Q/K/V weight matrices. Each "head" can learn a different
type of relationship.

**Why it exists:** A single attention operation can only focus on one thing at a time.
Multi-head attention lets different heads specialise: one might track grammatical
structure, another tracks factual associations, another tracks position.

**How it works in practice:**
The 2048-dimensional hidden state is split 32 ways into 32 chunks of 64 dimensions.
Each head operates in 64-dimensional space (tiny, fast), then all 32 outputs are
concatenated back to 2048 dimensions. This is equivalent to doing 32 full-size attention
operations but much more efficient.

```
32 heads × 64 dimensions per head = 2048 total dimensions
```

**Grouped Query Attention (GQA) — why Llama only has 8 KV heads:**
In standard multi-head attention, every head has its own K and V matrices — that's 32 K
matrices + 32 V matrices. GQA reduces this: 32 query heads share just 8 sets of K/V.
Every group of 4 Q heads reads from the same K and V. This cuts KV cache memory by 4×
with minimal quality loss.

```
32 query heads → grouped into 8 groups of 4
8 key projections (k_proj: 2048 → 512)     ← 4 heads share each
8 value projections (v_proj: 2048 → 512)   ← 4 heads share each
```

**Head types we'll identify (Day 3):** previous-token heads, first-token sink heads,
semantic heads, positional heads. Each is a specialisation that emerges from training.

**Llama 3.2 1B specifics (verified April 24, 2026):**
- 32 attention heads, 8 KV heads
- Head dimension: 2048 / 32 = **64 per query head**
- KV head dimension: 512 / 8 = **64 per KV head** (same size, just fewer of them)

**Where we saw it:** `notebooks/01_model_anatomy.ipynb` — architecture print confirmed
`q_proj` is (2048×2048) while `k_proj` and `v_proj` are (2048×512).

---

## Feed-Forward Network (MLP)

**What it is:** After attention mixes information across tokens, the MLP refines each
token's representation individually. It's a two-step expand-then-compress transformation
applied to every token independently (no cross-token communication here).

**Why it exists:** Attention is good at routing and mixing information, but it's limited
in the transformations it can apply. The MLP adds non-linear computation — this is where
factual knowledge is thought to be stored. Research shows you can "edit" a model's facts
by changing specific MLP weights.

**How it works:**
```
hidden (2048)
  → gate_proj  (2048 → 8192)   linear, no activation
  → up_proj    (2048 → 8192)   linear, no activation
  → SwiGLU: gate_proj × sigmoid(gate_proj) × up_proj  ← element-wise
  → down_proj  (8192 → 2048)   compress back
```
The SwiGLU activation is a "gated" version — the gate_proj output acts like a filter,
deciding which parts of the up_proj expansion to let through. It outperforms plain ReLU.

**The 4× expansion:** 2048 → 8192 gives the network room to compute complex non-linear
functions. The wider the expansion, the more "compute" per layer.

**Llama 3.2 1B specifics (verified April 24, 2026):**
- `gate_proj`: 2048 → 8192
- `up_proj`: 2048 → 8192
- `down_proj`: 8192 → 2048
- Each layer's MLP has 3 × (2048 × 8192) = ~50M parameters — the majority of each layer's weight

**Where we saw it:** `notebooks/01_model_anatomy.ipynb`. Each `LlamaDecoderLayer` prints
`mlp: LlamaMLP(gate_proj, up_proj, down_proj, act_fn=silu)`.

## KV Cache
> [To be filled — Day 3-4]

## Logit Lens

**What it is:** A technique to "peek inside" the model at every layer by projecting each
layer's hidden state through the final output head (`lm_head`). This shows what the model
would predict if it stopped at that layer — revealing how the answer emerges gradually.

**Why it exists:** Without this, the model is a black box: you give it text, it gives you
a prediction. The logit lens opens the box — you can see the model going from "I have no
idea" to "I'm 57% sure it's Paris" layer by layer.

**How it works:**
```python
# For each of the 17 hidden states (embed output + 16 layer outputs):
hidden = outputs.hidden_states[layer_idx]    # shape [1, seq_len, 2048]
last_token = hidden[0, -1, :]               # last token's vector: [2048]
normed = model.model.norm(last_token)        # apply final RMSNorm
logits = model.lm_head(normed)              # project to vocab: [128256]
probs = torch.softmax(logits, dim=-1)       # convert to probabilities
top5 = torch.topk(probs, 5)                # top-5 predictions
```

The key insight: we apply the *final* norm and *final* projection to every intermediate
state. This is a slight approximation (those layers were trained for the final state, not
intermediate ones), but it works well in practice.

**Results from our run on April 26, 2026:**

| Prompt | Answer | First appears | Pattern |
|--------|--------|--------------|---------|
| `"The capital of France is"` | Paris | **Layer 12** | Jumps 4.5% → 57.9% in one layer |
| `"The chemical formula for water is"` | H | **Layer 14** | Model says "WATER" at L12-13, then shifts to "H" |
| `"The cat sat on the"` | mat | **Layer 14** (top-5) | L12: couch, L13: sofa, L15: mat takes over at 54% |
| `"def hello_world():\n    print("` | Hello | **Layer 13** | Confident at 47% by L14 |
| `"2 + 2 ="` | 4 | **Never** | 1B model predicts `?` — too small for arithmetic |

**Key observations:**
- Early layers (1–4): garbage output — the model is recognising word types, not meanings
- Middle layers (5–11): the model knows *what kind of thing* the answer is (a city, a formula)
  but not *which specific one*
- Late layers (12–16): the specific fact clicks into place
- "2 + 2 = ?" shows that factual recall and arithmetic are different — the 1B model can
  recall Paris but can't compute 4. Larger models (7B+) solve this.
- **The "WATER" → "H" shift** for the water formula is fascinating: layers 12–13 understand
  the topic is water, but layers 14+ understand the answer format is a chemical symbol (H),
  not the word.

**Where we saw it:** `src/inspector/logit_lens.py` and `notebooks/02_logit_lens.ipynb`.
Heatmaps saved to `outputs/logit_lens/`. Run with:
```
./venv/bin/python -m src.inspector.logit_lens
```

## Backpropagation & Training
> [To be filled — as needed]

## LoRA & Fine-tuning
> [To be filled — as needed]

## Quantization
> [To be filled — Day 11]

## Attention Sinks
> [To be filled — Day 10]

---

## How to update this file

As each concept is learned during the daily tasks, fill in the corresponding section above. Each entry should:
1. Start with a one-sentence "what is it" answer
2. Explain WHY it exists / what problem it solves
3. Use a concrete analogy where helpful
4. Show a small code example or number (e.g., "for a 500-token sequence, cache uses X MB") when it makes things click
5. Note where we saw it in this project (which notebook, which experiment)
