"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
------------------------------------------------------------------------------

Probe heads for layer-wise analysis. A probe maps per-layer encoder features
(B, N, D) -> class logits (B, num_classes).

Selected by config via `type`:
  - "linear"    : pool tokens (N -> 1), optional pre-norm, then nn.Linear.
  - "attentive" : the stock V-JEPA AttentiveClassifier (cross-attention pooling
                  + Linear), identical to evals/video_classification_frozen.

Add new analysis heads by registering them in build_probe().
"""

import torch
import torch.nn as nn

from src.models.attentive_pooler import AttentiveClassifier


class LinearProbe(nn.Module):
    """Pool over tokens then a single linear layer (classic linear probing).

    pooling: "mean" | "max" | "meanmax"
    pre_norm: LayerNorm over the pooled feature before the linear layer.
        Recommended True for cross-layer comparison, since different encoder
        layers have very different feature scales.
    """

    def __init__(self, embed_dim, num_classes, pooling="mean", pre_norm=True):
        super().__init__()
        self.pooling = pooling
        in_dim = embed_dim * (2 if pooling == "meanmax" else 1)
        self.norm = nn.LayerNorm(in_dim) if pre_norm else nn.Identity()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)

    def _pool(self, x):  # x: (B, N, D)
        if self.pooling == "mean":
            return x.mean(dim=1)
        if self.pooling == "max":
            return x.max(dim=1).values
        if self.pooling == "meanmax":
            return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        raise ValueError(f"unknown pooling: {self.pooling}")

    def forward(self, x):
        x = self._pool(x)
        x = self.norm(x)
        return self.linear(x)


def build_probe(spec, embed_dim, num_classes, use_activation_checkpointing=False):
    """Build a probe head from a config dict.

    spec example (linear):    {type: linear, pooling: mean, pre_norm: true}
    spec example (attentive): {type: attentive, num_heads: 16, num_probe_blocks: 4}
    """
    ptype = str(spec.get("type", "attentive")).lower()

    if ptype == "linear":
        return LinearProbe(
            embed_dim=embed_dim,
            num_classes=num_classes,
            pooling=spec.get("pooling", "mean"),
            pre_norm=spec.get("pre_norm", True),
        )

    if ptype == "attentive":
        return AttentiveClassifier(
            embed_dim=embed_dim,
            num_heads=spec.get("num_heads", 16),
            depth=spec.get("num_probe_blocks", spec.get("depth", 1)),
            num_classes=num_classes,
            use_activation_checkpointing=use_activation_checkpointing,
        )

    raise ValueError(f"unknown probe type: {ptype!r} (expected 'linear' or 'attentive')")


def probe_name(spec):
    """Short stable name for logging/checkpoint columns."""
    if "name" in spec:
        return str(spec["name"])
    ptype = str(spec.get("type", "attentive")).lower()
    if ptype == "linear":
        return f"linear-{spec.get('pooling', 'mean')}"
    if ptype == "attentive":
        return f"attentive-d{spec.get('num_probe_blocks', spec.get('depth', 1))}"
    return ptype
