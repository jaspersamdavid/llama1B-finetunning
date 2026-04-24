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
> [To be filled — Day 2]

## Multi-Head Attention
> [To be filled — Day 2]

## Feed-Forward Network (MLP)
> [To be filled — Day 2]

## KV Cache
> [To be filled — Day 3-4]

## Logit Lens
> [To be filled — Day 2]

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
