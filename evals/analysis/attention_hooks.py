# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
# ADDITIVE attention instrumentation for the V-JEPA2 RoPE encoder.
#
# Two features, both driven ENTIRELY by config and DEFAULT-OFF, that require NO
# edits to any core model code (src/models/**). Everything is installed at
# runtime as a monkey-patch + forward-pre/-post hooks, and fully torn down on
# exit so the encoder is left byte-for-byte as it was.
#
#   (1) DISTANCE CAPTURE  -- for each block (layer) and each head, measure the
#       attention-weighted spatial (patch grid) and temporal (tubelet) distance,
#       i.e. E_{i}[ sum_j softmax_attn[h,i,j] * dist(i,j) ]. This is the standard
#       "attention distance" diagnostic, per head per layer.
#
#   (2) LOCAL-ATTENTION ABLATION -- build an additive attention bias from the
#       (T, H, W) grid that sets attention to ZERO wherever a query/key pair is
#       within a distance threshold, and inject it via `attn_mask` into
#       F.scaled_dot_product_attention. Three regimes:
#         spatial  : zero where  spatial_patch_dist <= s          (per s in s-list)
#         temporal : zero where  temporal_tubelet_dist <= t       (per t in t-list)
#         combined : zero where  (spatial<=s) AND (temporal<=t)   (per (s,t) pair)
#
# HOW IT HOOKS (no modules.py / vision_transformer.py edit):
#   * RoPEAttention.forward (src/models/utils/modules.py) already threads
#     T/H_patches/W_patches and, in its SDPA branch, calls
#     `F.scaled_dot_product_attention(q, k, v, ..., attn_mask=attn_mask)` on the
#     ALREADY ROPE-ROTATED q, k. We therefore:
#       - register a forward_pre_hook(with_kwargs=True) on every RoPEAttention
#         module that reads T/H_patches/W_patches from the call kwargs and pushes
#         a small context (layer index, grid, scale) onto a thread-local stack
#         (and a forward hook that pops it);
#       - monkey-patch torch.nn.functional.scaled_dot_product_attention with a
#         wrapper that, using the top-of-stack context, optionally (a) injects the
#         distance ablation bias and (b) captures per-head attention distance,
#         then calls the ORIGINAL sdpa for the returned value.
#   * When there is no active context (any non-RoPE attention, probe heads, etc.)
#     the wrapper is a straight pass-through -> those call sites are untouched.
#
# IDENTICAL-WHEN-OFF GUARANTEE:
#   * If config disables the feature, `attention_hooks(...)` is a nullcontext and
#     NOTHING is patched -> the encoder runs the original code path verbatim.
#   * With CAPTURE-ONLY on (no ablation), the wrapper computes the softmax weights
#     as a detached SIDE computation for the statistic and still returns the
#     ORIGINAL `F.scaled_dot_product_attention(q,k,v, attn_mask=None)` output ->
#     the encoder output is bit-identical to baseline (verified with torch.equal).
#   * ABLATION deliberately changes the output (that is the experiment).
# -----------------------------------------------------------------------------

import json
import logging
import os
import threading
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The exact symbol RoPEAttention.forward resolves at call time (module-level
# `import torch.nn.functional as F`). Patching it here is observed by modules.py.
_ORIG_SDPA = F.scaled_dot_product_attention

# thread-local stack of per-call attention contexts (one push per RoPEAttention
# forward). SDPA is called exactly once inside that forward, so top-of-stack is
# always the right layer while sdpa runs.
_TLS = threading.local()


def _stack():
    s = getattr(_TLS, "stack", None)
    if s is None:
        s = []
        _TLS.stack = s
    return s


# --------------------------------------------------------------------------- #
#  geometry: token index -> (t, h, w) and cached pairwise distance matrices
# --------------------------------------------------------------------------- #
# Token layout matches RoPEAttention.separate_positions EXACTLY:
#   t = idx // (H*W);  h = (idx - t*H*W) // W;  w = remainder
_COORD_CACHE = {}
_DIST_CACHE = {}


def _coords(T, H, W, device):
    key = (T, H, W, str(device))
    c = _COORD_CACHE.get(key)
    if c is None:
        idx = torch.arange(T * H * W, device=device)
        t = idx // (H * W)
        rem = idx - t * (H * W)
        h = rem // W
        w = rem - h * W
        c = (t.float(), h.float(), w.float())
        _COORD_CACHE[key] = c
    return c


def _dist_rows(T, H, W, device, rows=None):
    """(spatial, temporal) distance of key positions to a set of query `rows`.

    Returns two tensors of shape (len(rows), N). `rows` is a 1-D LongTensor of
    query indices (default: all N). Spatial = Euclidean patch distance, temporal
    = |dt| in tubelet units."""
    t, h, w = _coords(T, H, W, device)
    if rows is None:
        tq, hq, wq = t[:, None], h[:, None], w[:, None]
    else:
        tq, hq, wq = t[rows][:, None], h[rows][:, None], w[rows][:, None]
    ds = ((hq - h[None, :]) ** 2 + (wq - w[None, :]) ** 2).sqrt()
    dt = (tq - t[None, :]).abs()
    return ds, dt


def build_ablation_bias(T, H, W, device, dtype, spatial=None, temporal=None, mode="combined"):
    """Additive attention bias (0 keep / -inf drop), shape (1,1,N,N).

    mode:
      'spatial'  -> drop where spatial_patch_dist <= spatial
      'temporal' -> drop where temporal_tubelet_dist <= temporal
      'combined' -> drop where (spatial_dist <= spatial) AND (temporal_dist <= temporal)

    Safety: any query row that would be fully masked is left UNMASKED for that
    row (prevents an all -inf softmax row -> NaN)."""
    key = (T, H, W, str(device), str(dtype), spatial, temporal, mode)
    b = _DIST_CACHE.get(key)
    if b is not None:
        return b
    ds, dt = _dist_rows(T, H, W, device)  # (N,N)
    if mode == "spatial":
        assert spatial is not None, "spatial ablation needs a threshold s"
        drop = ds <= float(spatial)
    elif mode == "temporal":
        assert temporal is not None, "temporal ablation needs a threshold t"
        drop = dt <= float(temporal)
    elif mode == "combined":
        assert spatial is not None and temporal is not None, "combined ablation needs (s, t)"
        drop = (ds <= float(spatial)) & (dt <= float(temporal))
    else:
        raise ValueError(f"unknown ablation mode {mode!r}")
    # never fully mask a row
    full = drop.all(dim=1)
    if full.any():
        drop = drop.clone()
        drop[full] = False
    bias = torch.zeros_like(ds, dtype=dtype)
    bias.masked_fill_(drop, float("-inf"))
    bias = bias[None, None]  # (1,1,N,N)
    _DIST_CACHE[key] = bias
    return bias


# --------------------------------------------------------------------------- #
#  collector for the distance-capture statistic
# --------------------------------------------------------------------------- #
class AttentionDistanceCollector:
    """Accumulates attention-weighted distance per (layer, head).

    Streams over query chunks so the full (B,H,N,N) matrix is NEVER
    materialized (N can be a few thousand). Stores running sums -> means."""

    def __init__(self, num_layers, num_heads, query_chunk=512, max_batches=None, compute_dtype=torch.float32):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.query_chunk = query_chunk
        self.max_batches = max_batches  # cap #encoder forwards actually measured
        self.compute_dtype = compute_dtype
        self.sum_sd = torch.zeros(num_layers, num_heads, dtype=torch.float64)
        self.sum_td = torch.zeros(num_layers, num_heads, dtype=torch.float64)
        self.count = torch.zeros(num_layers, dtype=torch.float64)  # #(query,batch) rows seen
        self._forward_calls = 0  # #times layer 0 has been observed ~ #encoder forwards

    def note_forward(self, layer_idx):
        if layer_idx == 0:
            self._forward_calls += 1

    def enabled_for(self, layer_idx):
        if self.max_batches is None:
            return True
        # measure only the first `max_batches` encoder forwards (cheap subset)
        return self._forward_calls <= self.max_batches

    @torch.no_grad()
    def update(self, layer_idx, q, k, scale, T, H, W, extra_bias=None):
        """q,k: (B,heads,N,D) already RoPE-rotated. Computes softmax per query
        chunk and accumulates expected spatial/temporal distance per head."""
        B, Hh, N, D = q.shape
        device = q.device
        q = q.to(self.compute_dtype)
        k = k.to(self.compute_dtype)
        acc_sd = torch.zeros(Hh, dtype=torch.float64)
        acc_td = torch.zeros(Hh, dtype=torch.float64)
        rows_seen = 0
        for start in range(0, N, self.query_chunk):
            stop = min(start + self.query_chunk, N)
            rows = torch.arange(start, stop, device=device)
            logits = (q[:, :, start:stop] @ k.transpose(-2, -1)) * scale  # (B,H,C,N)
            if extra_bias is not None:
                logits = logits + extra_bias[..., start:stop, :].to(logits.dtype)
            attn = logits.softmax(dim=-1)  # (B,H,C,N)
            ds, dt = _dist_rows(T, H, W, device, rows=rows)  # (C,N)
            # expected distance per (B,H,C): sum_j attn * dist
            e_sd = (attn * ds[None, None]).sum(-1)  # (B,H,C)
            e_td = (attn * dt[None, None]).sum(-1)
            acc_sd += e_sd.sum(dim=(0, 2)).double().cpu()
            acc_td += e_td.sum(dim=(0, 2)).double().cpu()
            rows_seen += B * (stop - start)
        self.sum_sd[layer_idx] += acc_sd
        self.sum_td[layer_idx] += acc_td
        self.count[layer_idx] += rows_seen

    def finalize(self):
        c = self.count.clamp(min=1).unsqueeze(1)
        return {
            "spatial_distance": (self.sum_sd / c).float().tolist(),   # [layers][heads]
            "temporal_distance": (self.sum_td / c).float().tolist(),
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "rows_per_layer": self.count.tolist(),
        }

    def dump(self, path):
        out = self.finalize()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"[attention_hooks] wrote distance capture -> {path}")
        return out


# --------------------------------------------------------------------------- #
#  the monkey-patch context manager
# --------------------------------------------------------------------------- #
def _find_rope_attn(model):
    """RoPEAttention modules in block order (import here to avoid a hard dep at
    import time / circulars)."""
    from src.models.utils.modules import RoPEAttention

    mods = [m for m in model.modules() if isinstance(m, RoPEAttention)]
    return mods


@contextmanager
def attention_hooks(model, collector=None, ablation=None):
    """Install capture and/or ablation on a V-JEPA2 RoPE encoder.

    model     : the encoder (MultiLayerClipAggregation wrapper or the raw ViT).
    collector : AttentionDistanceCollector or None (no capture).
    ablation  : dict(mode=..., spatial=..., temporal=...) or None (no ablation).

    On exit everything is restored; if both args are None this is a no-op patch
    (still restores an identical sdpa, so output is unchanged)."""
    rope_mods = _find_rope_attn(model)
    if not rope_mods:
        logger.warning("[attention_hooks] no RoPEAttention modules found; nothing installed")
        yield collector
        return

    for i, m in enumerate(rope_mods):
        m._attn_layer_idx = i
        m._attn_scale = float(m.scale)

    handles = []

    def pre_hook(mod, args, kwargs):
        T = kwargs.get("T")
        H = kwargs.get("H_patches")
        W = kwargs.get("W_patches")
        tok_mask = kwargs.get("mask")
        # only instrument the dense (no token-drop) grid we can index by arange;
        # if a token mask is present or the grid is unknown, push an inert frame.
        ok = (T is not None and H is not None and W is not None and tok_mask is None)
        _stack().append(
            dict(active=ok, layer=mod._attn_layer_idx, scale=mod._attn_scale, T=T, H=H, W=W)
        )
        if ok and collector is not None:
            collector.note_forward(mod._attn_layer_idx)
        return None

    def post_hook(mod, args, kwargs, output):
        s = _stack()
        if s:
            s.pop()
        return output

    for m in rope_mods:
        handles.append(m.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(m.register_forward_hook(post_hook, with_kwargs=True))

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        s = _stack()
        ctx = s[-1] if s else None
        if ctx is None or not ctx["active"]:
            return _ORIG_SDPA(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                              is_causal=is_causal, scale=scale, **kw)
        layer, T, H, W = ctx["layer"], ctx["T"], ctx["H"], ctx["W"]
        eff_scale = scale if scale is not None else ctx["scale"]
        # (a) ablation bias
        bias = None
        if ablation is not None:
            bias = build_ablation_bias(
                T, H, W, device=query.device, dtype=query.dtype,
                spatial=ablation.get("spatial"), temporal=ablation.get("temporal"),
                mode=ablation.get("mode", "combined"),
            )
        # combine with any caller mask (encoder passes attn_mask=None here)
        if attn_mask is None:
            combined = bias
        elif bias is None:
            combined = attn_mask
        else:
            combined = attn_mask + bias  # both additive
        # (b) capture (detached side computation; does NOT affect returned value)
        if collector is not None and collector.enabled_for(layer):
            collector.update(layer, query, key, eff_scale, T, H, W, extra_bias=combined)
        return _ORIG_SDPA(query, key, value, attn_mask=combined, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, **kw)

    F.scaled_dot_product_attention = patched_sdpa
    try:
        yield collector
    finally:
        F.scaled_dot_product_attention = _ORIG_SDPA
        for h in handles:
            h.remove()
        for m in rope_mods:
            if hasattr(m, "_attn_layer_idx"):
                del m._attn_layer_idx
            if hasattr(m, "_attn_scale"):
                del m._attn_scale
        _TLS.stack = []


# --------------------------------------------------------------------------- #
#  config parsing (experiment.analysis.attention.*)
# --------------------------------------------------------------------------- #
def parse_config(att_cfg, num_layers, num_heads):
    """Turn the (optional) experiment.analysis.attention block into a plan.

    Returns None when disabled, else a dict:
      { 'capture': AttentionDistanceCollector|None,
        'ablation_settings': [ (label, {mode,spatial,temporal}), ... ] }
    ablation_settings has exactly one entry (label='baseline', dict=None) when
    ablation is off, so the caller can always iterate."""
    if not att_cfg or not att_cfg.get("enable", False):
        return None

    # -- capture --
    collector = None
    cap = att_cfg.get("capture")
    if cap:
        if cap is True:
            cap = {}
        collector = AttentionDistanceCollector(
            num_layers=num_layers, num_heads=num_heads,
            query_chunk=int(cap.get("query_chunk", 512)),
            max_batches=cap.get("max_batches"),  # None -> measure all
        )

    # -- ablation regimes -> a flat list of (label, spec) --
    settings = []
    abl = att_cfg.get("ablation")
    if abl:
        s_list = abl.get("spatial", []) or []
        t_list = abl.get("temporal", []) or []
        combos = abl.get("combined", []) or []  # list of [s, t]
        for s in s_list:
            settings.append((f"spatial_s{s}", {"mode": "spatial", "spatial": float(s)}))
        for t in t_list:
            settings.append((f"temporal_t{t}", {"mode": "temporal", "temporal": float(t)}))
        for st in combos:
            s, t = float(st[0]), float(st[1])
            settings.append((f"combined_s{s}_t{t}", {"mode": "combined", "spatial": s, "temporal": t}))
    if not settings:
        settings = [("baseline", None)]  # no ablation -> single unmodified run

    return {"capture": collector, "ablation_settings": settings}


def maybe_context(plan, model, ablation_spec):
    """Convenience: return an attention_hooks CM for one ablation setting, or a
    nullcontext when the whole feature is disabled."""
    if plan is None:
        return nullcontext()
    return attention_hooks(model, collector=plan["capture"], ablation=ablation_spec)
