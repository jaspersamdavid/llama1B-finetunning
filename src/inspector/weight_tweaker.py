"""
Weight Tweaker — break the model and observe.

Provides primitives:
  - zero_attention(layer)       set q/k/v/o_proj weights to 0
  - add_mlp_noise(layer, σ)     gate_proj += N(0, σ) noise
  - scale_head(layer, head, k)  multiply one Q head's rows by k
  - swap_layers(a, b)           exchange all weights between layers
  - reset()                     restore original state_dict

Two entry points:
  - script mode (default): runs the predefined Day-4 experiments and writes
    a results table to outputs/weight_tweaks/results.txt
  - interactive(): menu-driven REPL for ad-hoc tweaks
"""

import copy
from pathlib import Path

import torch

from src.inspector.logit_lens import run_logit_lens

OUTPUTS_DIR = Path("outputs/weight_tweaks")


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------

class WeightTweaker:
    """Holds an immutable snapshot of the model's original weights."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Deep clone every parameter so we can restore exactly.
        self._snapshot = {
            name: p.detach().clone()
            for name, p in model.state_dict().items()
        }
        self.num_layers = model.config.num_hidden_layers
        self.num_q_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_dim = model.config.hidden_size // self.num_q_heads

    # ── primitives ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Restore every parameter to its pristine value."""
        with torch.no_grad():
            sd = self.model.state_dict()
            for name, original in self._snapshot.items():
                sd[name].copy_(original)

    def zero_attention(self, layer_idx: int) -> None:
        """Zero out q/k/v/o_proj for one decoder layer."""
        with torch.no_grad():
            attn = self.model.model.layers[layer_idx].self_attn
            attn.q_proj.weight.zero_()
            attn.k_proj.weight.zero_()
            attn.v_proj.weight.zero_()
            attn.o_proj.weight.zero_()

    def add_mlp_noise(self, layer_idx: int, sigma: float, seed: int = 42) -> None:
        """Add Gaussian noise N(0, σ²) to gate_proj weights of one layer."""
        with torch.no_grad():
            gate = self.model.model.layers[layer_idx].mlp.gate_proj.weight
            gen = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(gate.shape, generator=gen, dtype=torch.float32) * sigma
            gate.add_(noise.to(gate.device, dtype=gate.dtype))

    def scale_head(self, layer_idx: int, head_idx: int, factor: float) -> None:
        """
        Multiply rows of q_proj.weight that correspond to one Q head by `factor`.
        q_proj weight shape: [num_q_heads * head_dim, hidden_size].
        Rows for head `h` are [h*head_dim : (h+1)*head_dim].
        """
        with torch.no_grad():
            q_proj = self.model.model.layers[layer_idx].self_attn.q_proj.weight
            row_start = head_idx * self.head_dim
            row_end = (head_idx + 1) * self.head_dim
            q_proj[row_start:row_end] *= factor

    def swap_layers(self, layer_a: int, layer_b: int) -> None:
        """Exchange every weight between two decoder layers."""
        with torch.no_grad():
            la = self.model.model.layers[layer_a]
            lb = self.model.model.layers[layer_b]
            for (name_a, pa), (name_b, pb) in zip(la.named_parameters(),
                                                   lb.named_parameters()):
                tmp = pa.data.clone()
                pa.data.copy_(pb.data)
                pb.data.copy_(tmp)


# ---------------------------------------------------------------------------
# Diff helper — run logit lens before/after a tweak and print side-by-side
# ---------------------------------------------------------------------------

def _top_str(layer_info: dict) -> str:
    toks = layer_info["top5_tokens"]
    probs = layer_info["top5_probs"]
    return f"{repr(toks[0].strip()[:14])} ({probs[0]:.0%})"


def diff_logit_lens(tw: WeightTweaker, prompt: str, before: dict) -> dict:
    """Run logit lens NOW (after tweak) and print before/after side-by-side."""
    after = run_logit_lens(tw.model, tw.tokenizer, prompt)

    print(f"\n  {'─'*72}")
    print(f"  Prompt: {repr(prompt)}")
    print(f"  {'Layer':<7} {'BEFORE top-1':<26} {'AFTER top-1':<26}")
    print(f"  {'-'*7} {'-'*26} {'-'*26}")
    for b, a in zip(before["layers"], after["layers"]):
        marker = "  " if _top_str(b) == _top_str(a) else "≠ "
        print(f"  {marker}{b['label']:<5} {_top_str(b):<26} {_top_str(a):<26}")
    print(f"  {'─'*72}\n")
    return after


# ---------------------------------------------------------------------------
# Script-mode experiments
# ---------------------------------------------------------------------------

EXPERIMENT_PROMPT = "The capital of France is"


def run_experiments(tw: WeightTweaker) -> list[dict]:
    """Run the Day 4 experiment suite and return a results list."""
    results = []
    prompt = EXPERIMENT_PROMPT

    def baseline():
        tw.reset()
        return run_logit_lens(tw.model, tw.tokenizer, prompt)

    base = baseline()
    base_top1 = _top_str(base["layers"][-1])

    def record(name: str, after: dict) -> None:
        after_top1 = _top_str(after["layers"][-1])
        results.append({"experiment": name, "before_top1": base_top1, "after_top1": after_top1})

    # ── Zero attention at layers 0, 8, 15 ───────────────────────────────────
    for L in [0, 8, 15]:
        tw.reset()
        tw.zero_attention(L)
        after = run_logit_lens(tw.model, tw.tokenizer, prompt)
        print(f"\n  ▶ zero_attention(layer={L})")
        diff_logit_lens(tw, prompt, base)
        record(f"zero_attention(L{L})", after)

    # ── Noise sweep on layer 5 MLP ──────────────────────────────────────────
    for sigma in [0.01, 0.1, 1.0]:
        tw.reset()
        tw.add_mlp_noise(layer_idx=5, sigma=sigma)
        after = run_logit_lens(tw.model, tw.tokenizer, prompt)
        print(f"\n  ▶ add_mlp_noise(layer=5, σ={sigma})")
        diff_logit_lens(tw, prompt, base)
        record(f"mlp_noise(L5, σ={sigma})", after)

    # ── Swap layers 2 ↔ 14 ──────────────────────────────────────────────────
    tw.reset()
    tw.swap_layers(2, 14)
    after = run_logit_lens(tw.model, tw.tokenizer, prompt)
    print(f"\n  ▶ swap_layers(2, 14)")
    diff_logit_lens(tw, prompt, base)
    record("swap_layers(2, 14)", after)

    # ── Scale head 0 in layer 8 by 10× ─────────────────────────────────────
    tw.reset()
    tw.scale_head(layer_idx=8, head_idx=0, factor=10.0)
    after = run_logit_lens(tw.model, tw.tokenizer, prompt)
    print(f"\n  ▶ scale_head(layer=8, head=0, factor=10.0)")
    diff_logit_lens(tw, prompt, base)
    record("scale_head(L8 H0, ×10)", after)

    tw.reset()
    return results


def write_results_table(results: list[dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"{'='*88}")
    lines.append(f"  WEIGHT TWEAK RESULTS — prompt = {repr(EXPERIMENT_PROMPT)}")
    lines.append(f"{'='*88}")
    lines.append(f"  {'Experiment':<32} {'Before (final layer)':<26} {'After (final layer)':<26}")
    lines.append(f"  {'-'*32} {'-'*26} {'-'*26}")
    for r in results:
        lines.append(f"  {r['experiment']:<32} {r['before_top1']:<26} {r['after_top1']:<26}")
    lines.append(f"{'='*88}\n")
    save_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

MENU = """
  ╭──────────────────────────────────────────────────╮
  │  WEIGHT TWEAKER                                  │
  │                                                  │
  │  1) Zero a layer's attention                     │
  │  2) Add Gaussian noise to a layer's MLP          │
  │  3) Scale a single attention head                │
  │  4) Swap two layers' weights                     │
  │  5) Reset to original weights                    │
  │  6) Change input prompt                          │
  │  7) Show logit lens for current prompt           │
  │  q) Quit                                         │
  ╰──────────────────────────────────────────────────╯
"""


def interactive(model, tokenizer) -> None:
    tw = WeightTweaker(model, tokenizer)
    prompt = EXPERIMENT_PROMPT
    baseline = run_logit_lens(model, tokenizer, prompt)
    print(MENU)

    while True:
        choice = input(f"\n  prompt = {prompt!r}\n  > select: ").strip().lower()
        if choice == "q":
            tw.reset()
            print("  Model reset, exiting.")
            return
        try:
            if choice == "1":
                L = int(input("  layer (0-15): "))
                tw.zero_attention(L)
                diff_logit_lens(tw, prompt, baseline)
            elif choice == "2":
                L = int(input("  layer (0-15): "))
                sigma = float(input("  noise sigma (e.g. 0.1): "))
                tw.add_mlp_noise(L, sigma)
                diff_logit_lens(tw, prompt, baseline)
            elif choice == "3":
                L = int(input("  layer (0-15): "))
                H = int(input("  head (0-31): "))
                k = float(input("  scale factor (e.g. 10.0): "))
                tw.scale_head(L, H, k)
                diff_logit_lens(tw, prompt, baseline)
            elif choice == "4":
                A = int(input("  layer A (0-15): "))
                B = int(input("  layer B (0-15): "))
                tw.swap_layers(A, B)
                diff_logit_lens(tw, prompt, baseline)
            elif choice == "5":
                tw.reset()
                baseline = run_logit_lens(model, tokenizer, prompt)
                print("  Reset done; new baseline captured.")
            elif choice == "6":
                prompt = input("  new prompt: ")
                baseline = run_logit_lens(model, tokenizer, prompt)
            elif choice == "7":
                from src.inspector.logit_lens import print_table
                print_table(run_logit_lens(model, tokenizer, prompt))
            else:
                print(MENU)
        except Exception as e:
            print(f"  ! error: {e}")
            tw.reset()


# ---------------------------------------------------------------------------
# Entry point — runs the script-mode experiments by default
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.inspector.model_loader import load_model

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model()

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive(model, tokenizer)
    else:
        print("\n" + "█"*72)
        print("  WEIGHT TWEAKER — Day 4 experiment suite")
        print("█"*72)
        tw = WeightTweaker(model, tokenizer)
        results = run_experiments(tw)
        write_results_table(results, OUTPUTS_DIR / "results.txt")
        print(f"\n✓ Results saved to {OUTPUTS_DIR}/")
