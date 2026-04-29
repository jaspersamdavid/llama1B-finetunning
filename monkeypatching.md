# monkeypatching.md — replacing the attention algorithm

> **Status:** parking lot. We're not doing this on Day 4. Saving the full
> explanation here so we can come back to it on Days 10-12 (KV cache
> optimization) when monkey-patching becomes the right tool.
>
> **Context:** this file documents Approach #3 from the "how to influence
> attention values" discussion. Approaches #1 (weight tweaking) and #2
> (forward hooks) are simpler and covered elsewhere.

---

## 1. What "monkey-patching" means

It's a Python idiom. You take a class that already exists in some library
you can't modify (here: `LlamaAttention` inside transformers), write your
own replacement class with the *same interface* (same method names, same
input/output shapes), and then **secretly substitute your class in place
of theirs at runtime**. The model object doesn't know it's been swapped —
it keeps calling `self.attention.forward(...)` like before, but now your
code runs.

It's called "monkey-patching" because you're reaching into someone else's
code and tinkering with it from outside, like a monkey with a wrench.
There's no formal API for "please replace your attention implementation."

---

## 2. What we'd actually need to write

`LlamaAttention.forward()` does roughly this (simplified — the real version
is ~80 lines and lives at
`~/venvs/llama-xray/lib/python3.11/site-packages/transformers/models/llama/modeling_llama.py`):

```python
def forward(self, hidden_states, attention_mask, position_ids, ...):
    # 1. Project to Q, K, V
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    # 2. Reshape into (batch, num_heads, seq_len, head_dim)
    q = q.view(...).transpose(1, 2)
    k = k.view(...).transpose(1, 2)
    v = v.view(...).transpose(1, 2)

    # 3. Apply rotary position embeddings (RoPE)
    q, k = apply_rotary_pos_emb(q, k, ...)

    # 4. Repeat K/V for grouped query attention (8 KV heads → 32 Q heads)
    k = repeat_kv(k, n_repeats=4)
    v = repeat_kv(v, n_repeats=4)

    # 5. Compute attention scores
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)

    # 6. Apply causal mask (block future tokens)
    scores = scores + attention_mask

    # 7. Softmax
    attn_weights = softmax(scores, dim=-1)

    # 8. Weighted sum of V
    output = attn_weights @ v

    # 9. Reshape back and project out
    output = output.transpose(1, 2).reshape(...)
    return self.o_proj(output)
```

To monkey-patch, we write our own subclass that copies most of this and
modifies one specific step.

### Example: "remove the BOS sink"

Force column 0 of `attn_weights` to zero, then renormalize:

```python
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention

class NoSinkLlamaAttention(LlamaAttention):
    def forward(self, hidden_states, attention_mask, position_ids, **kwargs):
        # ... copy steps 1-7 from the original ...
        attn_weights = softmax(scores, dim=-1)

        # OUR CHANGE: zero out column 0 (BOS) and renormalize each row
        attn_weights[:, :, :, 0] = 0
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

        # ... continue with steps 8-9 ...
        output = attn_weights @ v
        output = output.transpose(1, 2).reshape(...)
        return self.o_proj(output)
```

Then **swap it into the loaded model**:

```python
for layer in model.model.layers:
    layer.self_attn.__class__ = NoSinkLlamaAttention
```

That last line is the actual "monkey patch" — we're rewriting the class
identity on every attention module after the model is loaded. The next
forward pass uses our `forward()` instead of theirs.

### Other changes that fit this pattern

- **Force uniform attention** — replace softmax output with `1/seq_len`
  everywhere
- **Top-k attention** — keep only the top-3 attention weights per row,
  zero the rest, renormalize
- **Sliding window** — mask out anything older than 64 tokens (this is
  what Mistral 7B does natively)
- **Cache eviction strategies** for Days 10-12 — drop K/V vectors for
  "dead" tokens before the matmul

---

## 3. Why it's "heavier" than weight tweaking

Four reasons:

### (a) You have to reproduce a lot of code you didn't write

Weight tweaking is one line: `q_proj.weight[head_idx] = 0`.
Monkey-patching means copying ~80 lines of `LlamaAttention.forward()`
from transformers, understanding what every step does (RoPE, GQA repeat,
causal mask shape, KV cache plumbing), and being careful not to break any
of them while you change the one part you care about.

### (b) Fragile across library versions

Your subclass depends on transformers' internal API. When transformers
updates (e.g. they refactor `apply_rotary_pos_emb`'s signature, or add
a new arg to `forward`), your monkey-patched version silently breaks —
or worse, runs but produces subtly wrong results. Pinning the transformers
version helps but is annoying.

### (c) Easy to break the math invisibly

Weight tweaks are loud — zero out layer 5 attention and the model produces
obvious garbage. A bad monkey-patch can produce *plausible* output that's
actually wrong. Examples of subtle bugs:

- Forget to apply the causal mask before softmax → tokens leak from the
  future
- Renormalize incorrectly after dropping a column → attention rows no
  longer sum to 1, downstream numerics drift
- Apply RoPE in the wrong order → positions get scrambled

You'll only notice when the logit lens shows weird predictions and you
can't tell whether the model is "differently behaved" or just buggy.

### (d) KV cache is annoying

The real `forward()` handles `past_key_value` (the KV cache) for fast
generation. If you copy a forward without that handling, generation works
one token at a time but breaks the moment you turn caching on. For Day 4
we don't care, but for Days 10-12 (KV cache optimization) we do — that's
exactly when monkey-patching becomes both *necessary* and *dangerous*.

---

## 4. The three approaches at a glance

| Approach | What it changes | Cost | When to use |
|---|---|---|---|
| **Weight tweaking** | The *parameters* the model uses (zero a head, add noise, swap layers) | Cheap, one-line edits | Day 4 — break the model and observe |
| **Forward hooks** | Intercept activations as they flow through; observe or rewrite outputs | Medium, no copying internals | When you want to *observe* per-layer or *patch one specific value* without rewriting the algorithm |
| **Monkey-patching** | The *algorithm* the model runs (full attention forward) | Heavy, copy ~80 lines, fragile | Days 10-12 — when the cache eviction strategy or attention kernel itself needs to change |

---

## 5. When we'll actually do this

**Day 10 — Attention sink + window eviction:** the natural place to write
our first monkey-patched attention. We'll keep the first 4 tokens (sinks)
+ last N tokens in the cache, drop everything in between, and run
generation. That requires modifying the K/V slicing inside `forward()`,
which a forward hook can't cleanly do.

**Day 11 — Importance-based eviction:** track cumulative attention per
token, evict lowest-attention tokens periodically. Same monkey-patch
pattern, more bookkeeping inside the subclass.

**Day 12 — Combined:** sink + window + importance + INT8 quantization,
all in one custom attention class.

By that point we'll have built up enough familiarity with `LlamaAttention`
to safely modify it. Doing it cold on Day 4 would be premature.

---

## 6. Pre-work for Day 10 (when we get there)

- [ ] Open `~/venvs/llama-xray/lib/python3.11/site-packages/transformers/models/llama/modeling_llama.py`
      and read `LlamaAttention.forward()` end-to-end with a notebook open
- [ ] Write down what every line does in plain English
- [ ] Note the exact shape of every tensor at every step (this is where
      most monkey-patch bugs come from)
- [ ] Pin the transformers version in `requirements.txt` (current:
      5.6.2) — any version drift will silently break the patch
- [ ] Write a "no-op" subclass first (copies forward verbatim, changes
      nothing) and verify the model produces identical output. This is
      the baseline that proves the patching mechanism works before we
      change behavior
- [ ] Then write the actual experimental subclass

---

## 7. TL;DR

- **Weight tweaking** = changing the *parameters* the model uses. Easy.
  Day 4.
- **Monkey-patching** = changing the *algorithm* the model runs. Harder,
  more code to copy, more ways to break things silently, depends on
  internal library APIs. Save it for when you specifically need to alter
  the attention math itself — which is exactly what KV cache optimization
  (Days 10-12) requires.

For Day 4, weight tweaking is the right tool. We don't need to
monkey-patch yet.
