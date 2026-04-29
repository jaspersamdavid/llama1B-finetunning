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
