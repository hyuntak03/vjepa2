"""Core predictor-surprise forward for the parabolic eval (shared-context, multi-target).

For one scene we hold 3 variant clips (possible/higher/frozen) that SHARE the context
frames (the dataset splices `possible`'s context onto every variant). We then:

  1. Encode the shared context ONCE, MASKED (leak-free):
        z_ctx = context_encoder(clip, masks=[ctx_idx])
     -> the encoder patchifies the full clip but drops the future tokens before the
        transformer, so the context representation never attends to the future.
  2. Predict the future tokens ONCE:
        z_pred = predictor(z_ctx, ctx_idx, tgt_idx, mask_index=0)
  3. For each variant, encode its full clip (target_encoder, unmasked), gather the
     target-position tokens, and measure L1 to z_pred -> the per-variant "surprise".

This mirrors `analysis/intphys2/surprise.py` (whose `_context_target_indices` and
`_distance` we reuse) but in the shared-context / multi-target pattern our controlled
dataset needs: 1 masked-context encode + 1 predictor forward + N target encodes,
instead of re-encoding the (identical) context once per variant.

The predictor is trained on MASKED context, so we always use the masked forward here
(never the leaked full-clip slice), regardless of single/dual-encoder mode.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from analysis.intphys2.surprise import _context_target_indices, _distance


@torch.inference_mode()
def scene_surprises(
    clips: dict,                      # variant -> (C, T, H, W) float tensor, normalized
    bundle,                           # analysis.intphys2.model.VJEPA2Bundle
    *,
    context_length: int,              # number of context frames (0..context_length-1)
    distance: str = "l1",
    loss_exp: float = 1.0,
    target_layer_norm: bool = True,
    mask_index: int = 0,
    context_variant: str = "possible",
    debug: bool = False,
) -> dict:
    """Return {variant: surprise_scalar} for one scene. If debug, print tensor shapes."""
    device, dtype = bundle.device, bundle.dtype
    ts = bundle.tubelet_size
    spatial = bundle.num_spatial_tokens

    N = next(iter(clips.values())).size(1)               # total frames in the clip
    ctx_frames = int(context_length)
    tgt_frames = N - ctx_frames
    assert tgt_frames > 0, f"need future frames: N={N}, context_length={ctx_frames}"
    assert ctx_frames % ts == 0 and tgt_frames % ts == 0, (
        f"context/target must be tubelet({ts})-aligned; got ctx={ctx_frames}, tgt={tgt_frames}")

    ctx_idx, tgt_idx = _context_target_indices(
        ctx_frames=ctx_frames, tgt_frames=tgt_frames,
        tubelet_size=ts, spatial_tokens=spatial, batch_size=1, device=device)

    # (1) masked context encoding (leak-free) from the shared-context variant
    ctx_clip = clips[context_variant].to(device=device, dtype=dtype).unsqueeze(0)   # (1,C,T,H,W)
    z_ctx = bundle.context_encoder(ctx_clip, masks=[ctx_idx])
    if isinstance(z_ctx, list):
        z_ctx = z_ctx[-1]

    # (2) predict future tokens once (shared across all candidate targets)
    z_pred = bundle.predictor(z_ctx, ctx_idx, tgt_idx, mask_index=mask_index)
    if isinstance(z_pred, list):
        z_pred = z_pred[-1]
    z_pred = z_pred.float()

    if debug:
        exp_ctx = (ctx_frames // ts) * spatial
        exp_tgt = (tgt_frames // ts) * spatial
        print("  [shapes] --------------------------------------------------")
        print(f"    N_frames={N}  context={ctx_frames}f  target={tgt_frames}f  tubelet={ts}  spatial_tokens={spatial}")
        print(f"    ctx_clip {tuple(ctx_clip.shape)}  (expect (1,3,{N},H,W))")
        print(f"    ctx_idx  {tuple(ctx_idx.shape)}  tgt_idx {tuple(tgt_idx.shape)}  (expect (1,{exp_ctx}) (1,{exp_tgt}))")
        print(f"    z_ctx    {tuple(z_ctx.shape)}  (expect (1,{exp_ctx},{bundle.embed_dim}))")
        print(f"    z_pred   {tuple(z_pred.shape)}  (expect (1,{exp_tgt},{bundle.embed_dim}))")
        assert tuple(ctx_idx.shape) == (1, exp_ctx) and tuple(tgt_idx.shape) == (1, exp_tgt)
        assert z_ctx.shape[1] == exp_ctx and z_pred.shape[1] == exp_tgt

    # (3) per-variant target encode (full clip) -> gather target tokens -> L1
    gather = tgt_idx.unsqueeze(-1)
    out = {}
    for variant, clip in clips.items():
        h = bundle.target_encoder(clip.to(device=device, dtype=dtype).unsqueeze(0))
        if isinstance(h, list):
            h = h[-1]
        h_tgt = torch.gather(h, dim=1, index=gather.expand(-1, -1, h.size(-1)))
        if target_layer_norm:
            h_tgt = F.layer_norm(h_tgt, (h_tgt.size(-1),))
        out[variant] = float(_distance(z_pred, h_tgt.float(), distance, loss_exp=loss_exp).item())
        if debug:
            print(f"    [{variant:9s}] h_full {tuple(h.shape)}  h_tgt {tuple(h_tgt.shape)}  surprise={out[variant]:.5f}")
    if debug:
        print("  ------------------------------------------------------------")
    return out
