"""Token extraction for the occlusion identity eval.

Token layout (PatchEmbed3D, native V-JEPA tokenization): temporal tubelets are
contiguous, spatial patches inside a tubelet. So the tokens for frames [f0, f1) are
    indices[(f0 // ts) * S : (f1 // ts) * S]
for tubelet size `ts` and S spatial tokens per tubelet. With 256px / patch 16 -> S=256,
so an 8-frame window is 4 tubelets x 256 = 1024 tokens ("4 x N, D").

We cache BASE SEQUENCES, not per-source slices, so that overlapping windows share
storage (occ24__target is a slice of the same forward as full__target):

  base "target"            : target_encoder(clip[0:N])                 -> (N/ts)*S tokens
  base "ctx_masked"        : context_encoder(clip, masks=[ctx_idx])    -> (C/ts)*S tokens
                             future tokens dropped BEFORE the transformer, so it can
                             only have seen frames 0..C-1. This is what the predictor
                             consumed at train time, and it is bit-identical to
                             context_encoder(clip[:C]).
  base "predictor"         : predictor(ctx_masked, ...)                -> ((N-C)/ts)*S
  base "isolated_target:<f0>_<f1>" : target_encoder(clip[f0:f1])       -> ((f1-f0)/ts)*S
  base "isolated_ctx:<f0>_<f1>"    : context_encoder(clip[f0:f1])      -> ((f1-f0)/ts)*S
                             the window encoded as a STANDALONE clip: no other frame is
                             in the input, so attention cannot copy the object identity
                             out of frames where it was visible.

A *source* is then (base, frame-window) and resolves to a plain token slice.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from analysis.intphys2.surprise import _context_target_indices

# encoder role -> which frozen module + what it is fed
#   target          target_encoder (EMA)     , full N-frame clip
#   ctx_masked      context_encoder (online) , full clip with the future tokens masked out
#                   BEFORE the transformer. Verified bit-identical to feeding clip[:C],
#                   so it doubles as "context encoder with the 32 context frames in".
#   predictor       predictor(ctx_masked)    , outputs exactly frames [C, N)
#   isolated_target target_encoder (EMA)     , ONLY the window's frames as a standalone clip
#   isolated_ctx    context_encoder (online) , ONLY the window's frames as a standalone clip
ENCODERS = ("target", "ctx_masked", "predictor", "isolated_target", "isolated_ctx")
_ISOLATED = {"isolated_target": "target_encoder", "isolated_ctx": "context_encoder"}


def base_name(encoder: str, window: Sequence[int]) -> str:
    """Isolated forwards are per-window (their input IS the window); the rest are shared."""
    if encoder in _ISOLATED:
        return f"{encoder}:{window[0]}_{window[1]}"
    return encoder


def resolve_sources(cfg_features: dict, n_frames: int, context_length: int,
                    tubelet: int) -> List[dict]:
    """-> [{name, encoder, window, base, offset}] with windows resolved and validated."""
    windows = {k: tuple(v) for k, v in cfg_features["windows"].items()}
    out = []
    for s in cfg_features["sources"]:
        w = s["window"]
        w = windows[w] if isinstance(w, str) else tuple(w)
        enc = s["encoder"]
        assert enc in ENCODERS, f"source {s['name']!r}: bad encoder {enc!r}; valid {ENCODERS}"
        f0, f1 = w
        assert f0 % tubelet == 0 and f1 % tubelet == 0, (
            f"source {s['name']!r}: window [{f0},{f1}) is not tubelet({tubelet})-aligned")
        assert 0 <= f0 < f1 <= n_frames, f"source {s['name']!r}: window [{f0},{f1}) out of range"
        if enc == "ctx_masked":
            assert f1 <= context_length, (
                f"source {s['name']!r}: ctx_masked only holds frames 0..{context_length}")
        if enc == "predictor":
            assert (f0, f1) == (context_length, n_frames), (
                f"source {s['name']!r}: the predictor only outputs frames "
                f"[{context_length},{n_frames}); got [{f0},{f1})")
        # frame index at which this base sequence starts
        offset = 0 if enc in ("target", "ctx_masked") else (
            context_length if enc == "predictor" else f0)
        out.append({"name": s["name"], "encoder": enc, "window": w,
                    "base": base_name(enc, w), "offset": offset})
    names = [s["name"] for s in out]
    assert len(names) == len(set(names)), f"duplicate source names: {names}"
    return out


def token_slice(src: dict, tubelet: int, spatial: int) -> Tuple[int, int]:
    f0, f1 = src["window"]
    o = src["offset"]
    return ((f0 - o) // tubelet) * spatial, ((f1 - o) // tubelet) * spatial


def base_token_counts(sources: Sequence[dict], n_frames: int, context_length: int,
                      tubelet: int, spatial: int) -> Dict[str, int]:
    """-> {base_name: number of tokens in that base sequence}."""
    out = {}
    for s in sources:
        enc, (f0, f1) = s["encoder"], s["window"]
        if enc == "target":
            n = n_frames
        elif enc == "ctx_masked":
            n = context_length
        elif enc == "predictor":
            n = n_frames - context_length
        else:                                   # isolated_*: the clip IS the window
            n = f1 - f0
        out[s["base"]] = (n // tubelet) * spatial
    return out


@torch.inference_mode()
def extract_batch(
    clips: torch.Tensor,               # (B, C, T, H, W) normalized
    bundle,                            # analysis.intphys2.model.VJEPA2Bundle
    sources: Sequence[dict],
    *,
    context_length: int = 32,
    mask_index: int = 0,
    out_dtype: torch.dtype = torch.float16,
    debug: bool = False,
) -> Dict[str, torch.Tensor]:
    """-> {base_name: (B, N_tok, D) cpu tensor of `out_dtype`}."""
    device, dtype = bundle.device, bundle.dtype
    ts, S = bundle.tubelet_size, bundle.num_spatial_tokens
    B, N = clips.size(0), clips.size(2)

    ctx_frames = int(context_length)
    tgt_frames = N - ctx_frames
    assert tgt_frames > 0 and ctx_frames % ts == 0 and tgt_frames % ts == 0

    clips = clips.to(device=device, dtype=dtype)
    ctx_idx, tgt_idx = _context_target_indices(
        ctx_frames=ctx_frames, tgt_frames=tgt_frames, tubelet_size=ts,
        spatial_tokens=S, batch_size=B, device=device,
    )
    need = {s["base"] for s in sources}
    seqs: Dict[str, torch.Tensor] = {}

    def _ln(x):
        """Affine-free LayerNorm on ENCODER outputs — train-faithful, NOT a knob.

        app/vjepa/train.py:431-432 regresses the predictor onto
            h = F.layer_norm(target_encoder(clip), (D,))
        so the predictor's output is by definition an estimate of the LN'd encoder
        feature. The encoder's own final `self.norm` is an *affine* LayerNorm
        (vision_transformer.py:109), and that affine breaks the normalization, so this
        second affine-free LN is not a no-op — it is what puts encoder features and
        predictor outputs in the SAME space. Without it a head fitted on one and applied
        to the other is comparing different scales. The prediction itself is never
        re-normalized, for the same reason.
        """
        return F.layer_norm(x, (x.size(-1),))

    if "target" in need:
        h = bundle.target_encoder(clips)
        seqs["target"] = _ln(h[-1] if isinstance(h, list) else h)
    if "ctx_masked" in need or "predictor" in need:
        z = bundle.context_encoder(clips, masks=[ctx_idx])
        z = z[-1] if isinstance(z, list) else z
        if "predictor" in need:
            zp = bundle.predictor(z, ctx_idx, tgt_idx, mask_index=mask_index)
            seqs["predictor"] = zp[-1] if isinstance(zp, list) else zp   # already in LN space
        if "ctx_masked" in need:
            seqs["ctx_masked"] = _ln(z)
    for b in need:
        enc = b.split(":")[0]
        if enc not in _ISOLATED:
            continue
        f0, f1 = (int(v) for v in b.split(":")[1].split("_"))
        module = getattr(bundle, _ISOLATED[enc])
        hi = module(clips[:, :, f0:f1])
        seqs[b] = _ln(hi[-1] if isinstance(hi, list) else hi)

    out = {k: v.to(dtype=out_dtype).cpu() for k, v in seqs.items()}
    if debug:
        print("  [shapes] --------------------------------------------------")
        print(f"    clips {tuple(clips.shape)}  N={N}f ctx={ctx_frames}f tgt={tgt_frames}f "
              f"tubelet={ts} spatial={S} embed={bundle.embed_dim}")
        for k, v in out.items():
            print(f"    base[{k:18s}] {tuple(v.shape)}")
        for s in sources:
            a, b = token_slice(s, ts, S)
            f0, f1 = s["window"]
            print(f"    {s['name']:22s} {s['encoder']:11s} frames[{f0:2d},{f1:2d}) -> "
                  f"{s['base']}[{a}:{b}] = {b-a} tokens ({(f1-f0)//ts} tubelets x {S})")
        print("  ------------------------------------------------------------")
    return out
