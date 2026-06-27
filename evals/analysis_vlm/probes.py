# -----------------------------------------------------------------------------
# analysis_vlm-specific probe: attentive classifier with an optional TEMPORAL
# positional encoding applied per frame before pooling.
#
# Why: a (depth-1) attentive pooler is permutation-invariant — it pools the token
# SET, ignoring order. When the encoder does NOT bake temporal order into token
# VALUES (notably LLaVA-Video's per-frame SigLIP), the probe can't tell "moving up"
# from "moving down". A temporal encoding restores that signal. (V-JEPA / Qwen3-VL
# bake temporal info via RoPE / temporal patches, so they usually don't need this.)
#
# mode:
#   "learnable" -> add a learnable per-frame embedding (1,T,1,D) (absolute).
#   "rope"      -> rotary: rotate each token vector by its temporal index (rotate-half).
#                  Gives RELATIVE temporal structure inside the pooler's self-attn,
#                  no learnable params, no attention-internals modification.
# ("none" is handled by the caller using the plain AttentiveClassifier.)
#
# Assumes TEMPORAL-MAJOR token order: [frame0's S tokens, frame1's S tokens, ...],
# which is how the VLM backends flatten (B, T, S, D) -> (B, T*S, D).
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn

from src.models.attentive_pooler import AttentiveClassifier
from src.utils.tensors import trunc_normal_


class TemporalAttentiveClassifier(nn.Module):
    def __init__(self, embed_dim, num_temporal, mode="learnable", num_heads=16, depth=1,
                 num_classes=4, use_activation_checkpointing=False, init_std=0.02, rope_theta=10000.0):
        super().__init__()
        assert mode in ("learnable", "rope"), f"temporal mode must be learnable|rope, got {mode!r}"
        self.mode = mode
        self.num_temporal = int(num_temporal)
        self.embed_dim = int(embed_dim)

        if mode == "learnable":
            self.temporal_pos = nn.Parameter(torch.zeros(1, self.num_temporal, 1, embed_dim))
            trunc_normal_(self.temporal_pos, std=init_std)
        else:  # rope (rotate-half along the feature dim, angle = t * inv_freq)
            assert embed_dim % 2 == 0, "rope needs an even embed_dim"
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, embed_dim, 2).float() / embed_dim))  # (D/2,)
            freqs = torch.outer(torch.arange(self.num_temporal).float(), inv_freq)                 # (T, D/2)
            self.register_buffer("rope_cos", torch.cos(freqs)[None, :, None, :], persistent=False)  # (1,T,1,D/2)
            self.register_buffer("rope_sin", torch.sin(freqs)[None, :, None, :], persistent=False)

        self.classifier = AttentiveClassifier(
            embed_dim=embed_dim, num_heads=num_heads, depth=depth,
            num_classes=num_classes, use_activation_checkpointing=use_activation_checkpointing,
        )

    def _apply_temporal(self, x):  # x: (B, T, S, D)
        if self.mode == "learnable":
            return x + self.temporal_pos
        d = self.embed_dim
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        cos, sin = self.rope_cos.to(x.dtype), self.rope_sin.to(x.dtype)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, x):  # x: (B, T*S, D), temporal-major
        b, n, d = x.shape
        t = self.num_temporal
        if n % t != 0:
            raise ValueError(
                f"TemporalAttentiveClassifier: token count {n} not divisible by num_temporal={t} "
                f"(check the encoder temporal layout / frames_per_clip)."
            )
        s = n // t
        x = self._apply_temporal(x.view(b, t, s, d)).reshape(b, n, d)
        return self.classifier(x)


class TemporalLinearProbe(nn.Module):
    """Linear probe that POOLS SPATIAL tokens WITHIN each frame but KEEPS the temporal
    axis, then concatenates per-frame vectors -> Linear. Use for temporal tasks like
    DIRECTION: a global mean over all T*S tokens (the plain LinearProbe) washes out
    'up' vs 'down', whereas concatenating per-frame features lets the linear layer read
    the temporal trajectory. Input (B, T*S, D) temporal-major -> (B, T*D) -> Linear.

    spatial_pool: 'mean' | 'max' (over the S tokens of each frame).
    """

    def __init__(self, embed_dim, num_temporal, num_classes, spatial_pool="mean", pre_norm=True):
        super().__init__()
        self.num_temporal = int(num_temporal)
        self.spatial_pool = spatial_pool
        in_dim = embed_dim * self.num_temporal           # per-frame pooled vectors concatenated
        self.norm = nn.LayerNorm(in_dim) if pre_norm else nn.Identity()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):  # x: (B, T*S, D), temporal-major
        b, n, d = x.shape
        t = self.num_temporal
        if n % t != 0:
            raise ValueError(
                f"TemporalLinearProbe: token count {n} not divisible by num_temporal={t}."
            )
        s = n // t
        x = x.view(b, t, s, d)
        if self.spatial_pool == "mean":
            x = x.mean(dim=2)                            # (B, T, D)
        elif self.spatial_pool == "max":
            x = x.max(dim=2).values
        else:
            raise ValueError(f"spatial_pool must be mean|max, got {self.spatial_pool!r}")
        return self.linear(self.norm(x.reshape(b, t * d)))
