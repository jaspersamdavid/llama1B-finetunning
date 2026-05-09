"""
LayerPruner — drop or restore decoder layers in-place on a loaded Llama model.

We don't deep-copy the model (~2 GB). Instead we save references to the
16 original layers in a ModuleList, then for each pruning config we just
build a smaller ModuleList of references and reassign `model.model.layers`.
Reset is the reverse — put the original list back. No tensor copies.

Why this works:
  - `model.model.layers` is `nn.ModuleList(...)`, a thin container of refs
  - Llama's forward iterates `self.layers` and threads hidden_states through;
    it doesn't care about the count
  - We update `model.config.num_hidden_layers` to match for any code that
    reads it (rotary embeddings don't, but other transformers internals might)

Confirmed in Day 5: zeroing a layer's weights is equivalent to skipping it
because Llama uses pre-norm residuals (`x = x + attn(rmsnorm(x))`). So
`skip_layer_inplace` and `remove_layers` are functionally equivalent for
forward output — but `remove_layers` actually saves compute, while
`skip_layer_inplace` still runs the forward through zeros.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerPruner:
    """Holds references to original layers; drops/restores in-place."""

    def __init__(self, model):
        self.model = model
        # Capture original ModuleList contents + count.
        self._original_layers = list(model.model.layers)
        self._original_count = model.config.num_hidden_layers

    def prune(self, layer_indices):
        """Drop the given layer indices. Idempotent w.r.t. the original list.

        Important: each LlamaAttention has a `layer_idx` attribute used by
        DynamicCache to index per-layer K/V slots. After pruning we MUST
        renumber `self_attn.layer_idx` so they're sequential 0..N-1, or
        cache.layers[layer_idx] will index out of range when we drop a
        middle layer (the last surviving layer still thinks it's idx 15).
        """
        drop = set(layer_indices)
        keep = [layer for i, layer in enumerate(self._original_layers) if i not in drop]
        for new_idx, layer in enumerate(keep):
            layer.self_attn.layer_idx = new_idx
        self.model.model.layers = nn.ModuleList(keep)
        self.model.config.num_hidden_layers = len(keep)

    def reset(self):
        """Restore the original 16 layers AND original layer_idx values."""
        for original_idx, layer in enumerate(self._original_layers):
            layer.self_attn.layer_idx = original_idx
        self.model.model.layers = nn.ModuleList(self._original_layers)
        self.model.config.num_hidden_layers = self._original_count


def skip_layer_inplace(model, layer_index: int) -> None:
    """
    Zero attention + MLP for one layer in-place.
    Equivalent to skipping the layer thanks to pre-norm residual:
    `x = x + sublayer(rmsnorm(x))` → if sublayer ≡ 0, x_out = x_in.

    Caller is responsible for restoring weights afterward (use
    snapshot_layer / restore_layer from layer_importance_scorer).
    """
    layer = model.model.layers[layer_index]
    with torch.no_grad():
        for proj in (layer.self_attn.q_proj,
                     layer.self_attn.k_proj,
                     layer.self_attn.v_proj,
                     layer.self_attn.o_proj,
                     layer.mlp.gate_proj,
                     layer.mlp.up_proj,
                     layer.mlp.down_proj):
            proj.weight.zero_()
