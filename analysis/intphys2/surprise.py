# -----------------------------------------------------------------------------
# Sliding-window "surprise" for IntPhys2 (Bordes et al., Eq. 1 / App. D).
#
#   Surprise_w = d( p(f(V_{w:w+C})),  f'(V_{w+C:w+M}) )
#
# with (for V-JEPA 2):
#   f  = context encoder  (usually target_encoder EMA at eval time)
#   f' = target encoder   (target_encoder EMA)
#   p  = latent predictor
#   d  = L1-style distance in latent space, mirroring the training loss
#        `mean(|z - h|^loss_exp) / loss_exp` (app/vjepa/train.py:447).
#
# Because V-JEPA 2 uses RoPE (no learned pos_embed) the same encoder / predictor
# state_dict is agnostic to window size, so we build the ViT with `num_frames =
# window_size` and pass real spatiotemporal indices for context / target token
# splits.
#
# Two knobs shape the split (Fig. 8 / paper Table 8):
#   window_size (M)     - # frames per surprise "window"       (V-JEPA 2 default: 48)
#   context_length (C)  - # frames used to CONDITION prediction (V-JEPA 2 sweeps 4..14)
#   -> prediction_length = M - C  (paper's "pred -1" = predict to end of window)
#   stride (S)          - # frames between successive window STARTS  (D.1 default: same S)
#
# Aggregation (Eq. 2): mean vs max over windows -> one scalar per video.
#
# All heavy compute runs under `torch.inference_mode()` in `bfloat16` (configurable).
# We deliberately do a SINGLE encoder forward per window (encode the full M-frame
# window once, slice context / target features) rather than the training-time
# "context-only" masked forward -- efficient, and semantically fine for a frozen
# encoder run zero-shot (the paper's Table 7 shows the two protocols are equivalent).
# The "faithful two-forward" mode can be reintroduced later behind a config knob.
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import contextlib

from analysis.intphys2.model import VJEPA2Bundle

logger = logging.getLogger(__name__)


# --------------------------- window scheduling -------------------------------


@dataclass
class WindowSpec:
    """One surprise window: [start, start+M) with context [start, start+C)."""

    start: int
    end: int  # start + window_size
    ctx_end: int  # start + context_length
    context_length: int

    @property
    def window_size(self) -> int:
        return self.end - self.start


def plan_growing_prefix(context_length: int, tubelet_size: int) -> List[int]:
    """Context lengths for the paper's growing-context prefix (Fig 8B, App. D.1).

        [tubelet, 2*tubelet, ..., context_length - tubelet]

    All of these run on the FIRST window (frames [0, M)) with the window size held at
    its maximum M; only the context/target split moves. Quoting the paper:

        "we propose to consider the window size M as a maximal size, and also proceed
         with all predictions using smaller ones, by progressively growing the context
         size. ... This ensures that every frame (apart from the first one) is predicted
         no matter the window size."

    This is exactly `max_context_mode` in the official release
    (IntPhys2/prediction_evals/evals/intphys2/eval.py:230-233):

        for ctxt in [2*i for i in range(1, CTXT_LEN//2)]:   # 2, 4, ..., CTXT_LEN-2
            model.nb_context_frames = ctxt
            clip_beginning = clip[:, :, :model.frames_per_clip]   # window 0, size M

    where the hardcoded 2 is the V-JEPA tubelet size. The resulting per-window losses are
    `hstack`ed BEFORE the fixed sliding-window losses, so the surprise-over-time curve is
    extended backwards to frame `tubelet` instead of starting at frame C.
    """
    return list(range(tubelet_size, context_length, tubelet_size))


def plan_windows(
    n_frames: int,
    window_size: int,
    context_length: int,
    stride: int,
    tubelet_size: int,
    protocol: str = "fixed",
) -> List[WindowSpec]:
    """Compute the list of surprise windows for a video with `n_frames` frames.

    protocol = "fixed" (paper's Fig 8A / Garrido et al. 2024):
        starts = 0, S, 2S, ..., last valid start with start + M <= n_frames
        each window has context [start, start+C) and target [start+C, start+M).

    protocol = "growing" (paper's Fig 8B — the protocol IntPhys 2 actually proposes):
        the SAME sliding windows as `fixed`, plus a prefix of growing-context
        predictions on window 0. The prefix is not part of the window LAYOUT (every
        prefix entry starts at frame 0), so it is produced separately by
        `plan_growing_prefix` and prepended in `score_video_batched` / `score_video`.
        Table 7 reports the two protocols scoring V-JEPA equivalently (52.96 vs 53.75).
    """
    if protocol not in ("fixed", "growing"):
        raise ValueError(f"protocol {protocol!r}; valid: fixed | growing")

    # Both C and M must be divisible by tubelet_size so context/target token grids are aligned.
    if window_size % tubelet_size != 0:
        raise ValueError(f"window_size={window_size} not divisible by tubelet_size={tubelet_size}")
    if context_length % tubelet_size != 0:
        raise ValueError(f"context_length={context_length} not divisible by tubelet_size={tubelet_size}")
    if not (0 < context_length < window_size):
        raise ValueError(f"need 0 < context_length < window_size; got C={context_length}, M={window_size}")

    windows: List[WindowSpec] = []
    if n_frames < window_size:
        # Short-video fallback: single window with the entire clip (context still C frames,
        # target = remaining). We DO NOT pad -- if C >= n_frames we skip.
        if n_frames <= context_length:
            logger.warning(f"video too short for surprise (n={n_frames} <= C={context_length}); skipping")
            return []
        windows.append(WindowSpec(start=0, end=n_frames, ctx_end=context_length, context_length=context_length))
        return windows

    last_start = n_frames - window_size
    start = 0
    while start <= last_start:
        windows.append(
            WindowSpec(
                start=start, end=start + window_size,
                ctx_end=start + context_length, context_length=context_length,
            )
        )
        start += stride
    return windows


# --------------------------- distance functions ------------------------------


def _distance(
    z_pred: torch.Tensor, h_tgt: torch.Tensor, kind: str, loss_exp: float = 1.0,
) -> torch.Tensor:
    """Per-window scalar distance. Both inputs (B, N_tgt, D) already dtype-aligned.

    kinds:
      l1        : mean(|z - h|^exp) / exp   -- matches V-JEPA training loss (loss_exp=1 -> L1)
      smoothl1  : F.smooth_l1_loss reduction='mean'
      l2        : mean((z - h)^2)
      cosine    : 1 - mean cosine sim over tokens
    """
    if kind == "l1":
        return torch.mean(torch.abs(z_pred - h_tgt) ** loss_exp) / loss_exp
    if kind == "smoothl1":
        return F.smooth_l1_loss(z_pred, h_tgt, reduction="mean")
    if kind == "l2":
        return torch.mean((z_pred - h_tgt) ** 2)
    if kind == "cosine":
        # 1 - mean cosine similarity across tokens & batch
        return 1.0 - F.cosine_similarity(z_pred, h_tgt, dim=-1).mean()
    raise ValueError(f"unknown distance {kind!r}; valid: l1/smoothl1/l2/cosine")


# --------------------------- token index helpers -----------------------------


def _context_target_indices(
    ctx_frames: int, tgt_frames: int, tubelet_size: int, spatial_tokens: int,
    batch_size: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mask_ctx, mask_tgt) of shape (B, N_ctx) and (B, N_tgt).

    Token layout follows PatchEmbed3D + native V-JEPA tokenization: temporal tubelets are
    contiguous in the sequence, then spatial patches within a tubelet. So the first
    (ctx_frames // tubelet_size) * spatial_tokens indices belong to the context.
    """
    ctx_tub = ctx_frames // tubelet_size
    tgt_tub = tgt_frames // tubelet_size
    n_ctx = ctx_tub * spatial_tokens
    n_tgt = tgt_tub * spatial_tokens
    n_total = n_ctx + n_tgt

    all_idx = torch.arange(n_total, device=device, dtype=torch.long)
    ctx_idx = all_idx[:n_ctx].unsqueeze(0).expand(batch_size, -1).contiguous()  # (B, N_ctx)
    tgt_idx = all_idx[n_ctx:].unsqueeze(0).expand(batch_size, -1).contiguous()  # (B, N_tgt)
    return ctx_idx, tgt_idx


# --------------------------- surprise loop -----------------------------------


@dataclass
class VideoSurprise:
    """Full surprise trace for one video."""

    window_starts: np.ndarray  # (W,)
    window_context_lengths: np.ndarray  # (W,)
    surprise: np.ndarray  # (W,) per-window scalar
    context_length: int  # scalar echoed from cfg (or -1 if the video was skipped)

    def aggregate(self, mode: str) -> float:
        if len(self.surprise) == 0:
            return float("nan")
        if mode == "avg":
            return float(self.surprise.mean())
        if mode == "max":
            return float(self.surprise.max())
        if mode == "min":
            return float(self.surprise.min())
        raise ValueError(f"unknown aggregation {mode!r}; valid: avg/max/min")


@torch.inference_mode()
def score_video(
    video: torch.Tensor,  # (C, T, H, W) float, normalized (already dtype-cast is fine)
    bundle: VJEPA2Bundle,
    *,
    window_size: int,
    context_length: int,
    stride: int,
    distance: str = "l1",
    loss_exp: float = 1.0,
    target_layer_norm: bool = True,
    protocol: str = "fixed",
    mask_index: int = 0,       # CRITICAL: only mask_tokens[0] is trained in the released
                               # V-JEPA 2 checkpoint; indices 1..9 remain at zero init and would
                               # substitute a zero vector for the "query-the-future" input.
) -> VideoSurprise:
    """Compute per-window surprise for a single video.

    We move the video to `bundle.device` and cast to `bundle.dtype` here; caller can pass
    a plain CPU float tensor. Returns a dataclass with numpy arrays so results can be
    JSON'd / cached without pulling GPU tensors around.
    """
    assert video.ndim == 4 and video.size(0) == 3, f"expected (3,T,H,W), got {tuple(video.shape)}"
    device, dtype = bundle.device, bundle.dtype
    T_total = video.size(1)

    windows = plan_windows(
        n_frames=T_total,
        window_size=window_size,
        context_length=context_length,
        stride=stride,
        tubelet_size=bundle.tubelet_size,
        protocol=protocol,
    )
    if not windows:
        return VideoSurprise(
            window_starts=np.zeros(0, dtype=np.int64),
            window_context_lengths=np.zeros(0, dtype=np.int64),
            surprise=np.zeros(0, dtype=np.float32),
            context_length=context_length,
        )

    per_window: List[float] = []
    starts: List[int] = []
    ctx_lens: List[int] = []

    spatial_tokens = bundle.num_spatial_tokens
    target_encoder = bundle.target_encoder
    context_encoder = bundle.context_encoder
    dual_forward = context_encoder is not target_encoder
    predictor = bundle.predictor

    video = video.to(device=device, dtype=dtype, non_blocking=True)

    # protocol="growing" (paper Fig 8B): prepend window 0 evaluated with every smaller
    # context. Same window span, only the context/target boundary moves. See
    # plan_growing_prefix() for the paper / official-code citation.
    jobs: List[WindowSpec] = list(windows)
    if protocol == "growing":
        w0 = windows[0]
        jobs = [
            WindowSpec(start=w0.start, end=w0.end, ctx_end=w0.start + c, context_length=c)
            for c in plan_growing_prefix(context_length, bundle.tubelet_size)
        ] + jobs

    for w in jobs:
        # slice frames [w.start, w.end) -> (3, M, H, W)
        clip = video[:, w.start:w.end]
        M_frames = clip.size(1)
        # If short-video fallback, M_frames may be < window_size -> context/target still
        # obey w.ctx_end.
        C_frames = w.ctx_end - w.start
        T_frames = M_frames - C_frames
        if T_frames <= 0:
            continue
        if C_frames % bundle.tubelet_size != 0 or T_frames % bundle.tubelet_size != 0:
            # tubelet-alignment safety guard: round C_frames DOWN, T_frames DOWN
            C_frames -= C_frames % bundle.tubelet_size
            T_frames = M_frames - C_frames
            T_frames -= T_frames % bundle.tubelet_size
            if C_frames <= 0 or T_frames <= 0:
                continue

        clip_used = clip[:, : C_frames + T_frames]  # trim to tubelet-aligned length
        clip_batch = clip_used.unsqueeze(0)  # (1, 3, M', H, W) -- the V-JEPA encoder wants NCTHW

        # Target encoder forward on the full window -> (1, N_total, D)
        h_full = target_encoder(clip_batch)
        # `out_layers` is not set at build time here so vision_transformer.forward returns the
        # final normalized token tensor directly (not a list). Defensive check:
        if isinstance(h_full, list):
            h_full = h_full[-1]

        ctx_idx, tgt_idx = _context_target_indices(
            ctx_frames=C_frames, tgt_frames=T_frames,
            tubelet_size=bundle.tubelet_size, spatial_tokens=spatial_tokens,
            batch_size=h_full.size(0), device=device,
        )

        # Target features: always sliced from the (target_encoder full-clip) forward.
        gather_tgt = tgt_idx.unsqueeze(-1).expand(-1, -1, h_full.size(-1))
        h_tgt = torch.gather(h_full, dim=1, index=gather_tgt)
        # Context features: shared forward (single-encoder EMA-only default) or a
        # second training-faithful forward through the online context encoder.
        if dual_forward:
            z_ctx = context_encoder(clip_batch, masks=[ctx_idx])  # (1, N_ctx, D)
            if isinstance(z_ctx, list):
                z_ctx = z_ctx[-1]
        else:
            gather_ctx = ctx_idx.unsqueeze(-1).expand(-1, -1, h_full.size(-1))
            z_ctx = torch.gather(h_full, dim=1, index=gather_ctx)

        if target_layer_norm:
            # Same recipe as `app/vjepa/train.py:432` -> per-token LN over D.
            h_tgt = F.layer_norm(h_tgt, (h_tgt.size(-1),))

        # Predictor: predicts target-token embeddings from context tokens + mask tokens
        # placed at the target positions.
        z_pred = predictor(z_ctx, ctx_idx, tgt_idx, mask_index=mask_index)

        # Distance -> scalar
        surprise_val = _distance(z_pred.float(), h_tgt.float(), distance, loss_exp=loss_exp)

        per_window.append(float(surprise_val.item()))
        starts.append(int(w.start))
        ctx_lens.append(int(C_frames))

    return VideoSurprise(
        window_starts=np.asarray(starts, dtype=np.int64),
        window_context_lengths=np.asarray(ctx_lens, dtype=np.int64),
        surprise=np.asarray(per_window, dtype=np.float32),
        context_length=context_length,
    )


# --------------------------- batched fast path -------------------------------


@torch.inference_mode()
def score_video_batched(
    video: torch.Tensor,  # (C, T, H, W) float, normalized
    bundle: VJEPA2Bundle,
    *,
    window_size: int,
    context_lengths: List[int],
    stride: int,
    distance: str = "l1",
    loss_exp: float = 1.0,
    target_layer_norm: bool = True,
    protocol: str = "fixed",
    max_window_batch: Optional[int] = None,  # cap the # windows encoded in one shot (VRAM)
    mask_index: int = 0,     # CRITICAL default — only mask_tokens[0] is trained.
    context_forward_mode: str = "masked",  # masked | sliced -- see fixes doc.
    autocast_dtype: Optional[torch.dtype] = None,   # None -> run in bundle.dtype as-is
) -> Dict[int, VideoSurprise]:
    """Score all sliding windows AND all context lengths with **one encoder forward per window batch**.

    The encoder output for a window `[w, w+M)` is IDENTICAL across every choice of context length
    `C` -- only the (mask_ctx, mask_tgt) split fed to the predictor changes. So the naive per-C
    outer loop wastes ~len(context_lengths)× encoder time. This function shares the encoder pass:

        1. plan_windows() with the FIRST valid C (all Cs share start positions when protocol=fixed).
        2. Stack all M-frame windows into a single (W, 3, M, H, W_spatial) batch.
        3. encoder(clip_batch) -> h_all (W, N_total, D).      # ONE forward
        4. For each C in `context_lengths`:
             a. gather z_ctx / h_tgt from the SAME h_all (no re-encode).
             b. predictor(z_ctx, mask_ctx, mask_tgt) -> z_pred.
             c. surprise[C][window] = distance(z_pred[window], h_tgt[window]).
        5. Return a dict {C -> VideoSurprise}.

    We also defer `.item()` sync -- distances accumulate on-device and hop to CPU once per video.
    `max_window_batch` caps how many windows are encoded in one go (VRAM tradeoff); with M=48 and
    the ViT-L 6144 tokens, a full 4-window batch fits comfortably in a 24 GB card in bf16.
    """
    assert video.ndim == 4 and video.size(0) == 3, f"expected (3,T,H,W), got {tuple(video.shape)}"
    if not context_lengths:
        return {}
    device, dtype = bundle.device, bundle.dtype
    T_total = video.size(1)

    # All Cs share the SAME window LAYOUT (start, end) for full-M windows; the C only affects
    # WHERE the context/target boundary sits inside a window. So window starts are C-independent
    # for the fixed protocol -- BUT the short-video fallback in plan_windows() is C-sensitive
    # (it returns [] when n_frames <= C). Plan with the SMALLEST C in the sweep so we don't
    # spuriously drop windows that other Cs would still support (review-found bug).
    C_min = min(int(c) for c in context_lengths)
    windows = plan_windows(
        n_frames=T_total,
        window_size=window_size,
        context_length=C_min,
        stride=stride,
        tubelet_size=bundle.tubelet_size,
        protocol=protocol,
    )
    if not windows:
        empty = VideoSurprise(
            window_starts=np.zeros(0, dtype=np.int64),
            window_context_lengths=np.zeros(0, dtype=np.int64),
            surprise=np.zeros(0, dtype=np.float32),
            context_length=context_lengths[0],
        )
        return {C: empty for C in context_lengths}

    # Fast path only handles full-M windows -- short-video single-window falls back to per-C loop.
    if any(w.window_size != window_size for w in windows):
        return {
            C: score_video(
                video, bundle,
                window_size=window_size, context_length=C, stride=stride,
                distance=distance, loss_exp=loss_exp,
                target_layer_norm=target_layer_norm, protocol=protocol,
                mask_index=mask_index,   # else the fallback silently ignores the config
            )
            for C in context_lengths
        }

    video = video.to(device=device, dtype=dtype, non_blocking=True)

    # Official Garrido/IntPhys2 numerics keep fp32 weights and autocast the forward
    # (evals/intuitive_physics/eval.py:437). Pass autocast_dtype=torch.float16 together
    # with model.dtype=float32 to match; leave None for the old whole-graph-cast path.
    def _ac():
        return (torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype
                else contextlib.nullcontext())

    # Stack windows as a real minibatch through the encoder.
    #   clip_batch : (W, 3, M, H, W_sp)
    def _stack(windows_chunk):
        return torch.stack([video[:, w.start : w.end] for w in windows_chunk], dim=0)

    # Two independent axes (audit findings #3, #4):
    #  (A) dual_encoder (from `bundle`): whether context and target encoders share weights
    #      (EMA-only, `dual_encoder=False` in model.py) or are two distinct modules
    #      (online for context, EMA for target -- matches training).
    #  (B) `context_forward_mode` (this arg):
    #        "masked"  -> call context_encoder with masks=[ctx_idx] so its self-attention
    #                     only covers context tokens (matches app/vjepa/train.py:435-438).
    #        "sliced"  -> gather z_ctx from a full-clip target_encoder forward (fast; old
    #                     default; NOT training-faithful because context tokens have already
    #                     self-attended over target tokens).
    # If (A) and (B) collapse (dual_forward=False AND mode=sliced), we can reuse a single
    # encoder forward for both context and target -- the original ONE-forward fast path.
    target_encoder = bundle.target_encoder
    context_encoder = bundle.context_encoder
    dual_forward = context_encoder is not target_encoder
    need_ctx_forward = dual_forward or context_forward_mode == "masked"
    predictor = bundle.predictor
    spatial_tokens = bundle.num_spatial_tokens
    embed_dim = bundle.embed_dim
    tub = bundle.tubelet_size

    # If max_window_batch is set, chunk the windows to bound VRAM. Otherwise do them all at once.
    if max_window_batch is None or max_window_batch >= len(windows):
        chunks = [windows]
    else:
        chunks = [windows[i : i + max_window_batch] for i in range(0, len(windows), max_window_batch)]

    # ---- growing-context prefix (paper Fig 8B == official `max_context_mode`) ------
    # For every C in the sweep the paper ALSO predicts window 0 with each SMALLER
    # context (tubelet, 2*tubelet, ..., C-tubelet), holding the window at its maximum
    # size M, and prepends those losses to the sliding-window ones:
    #     official evals/intphys2/eval.py:230-257
    #         for ctxt in [2*i for i in range(1, CTXT_LEN//2)]: ... losses_beginning
    #         loss = torch.hstack([loss_beginning, loss])
    # Different Cs share prefix entries (C'=2 appears for every C), so each DISTINCT
    # (window 0, C') pair is computed ONCE and assembled per C at the end.
    prefix_of: Dict[int, List[int]] = (
        {int(C): plan_growing_prefix(int(C), tub) for C in context_lengths}
        if protocol == "growing"
        else {int(C): [] for C in context_lengths}
    )
    prefix_Cs = sorted({c for v in prefix_of.values() for c in v})

    # Mask templates -- one per context length -- built ONCE and memoized. `prefix_Cs`
    # needs its own templates because those context lengths are not in the sweep.
    per_C_masks: Dict[int, Tuple[torch.Tensor, torch.Tensor, int, int]] = {}
    for C in [int(c) for c in context_lengths] + prefix_Cs:
        if C in per_C_masks:
            continue
        ctx_tub = C // tub
        tgt_tub = (window_size - C) // tub
        n_ctx = ctx_tub * spatial_tokens
        n_tgt = tgt_tub * spatial_tokens
        all_idx = torch.arange(n_ctx + n_tgt, device=device, dtype=torch.long)
        per_C_masks[C] = (all_idx[:n_ctx], all_idx[n_ctx:], n_ctx, n_tgt)

    def _window_distances(clip_batch: torch.Tensor, h_all: torch.Tensor, C: int) -> torch.Tensor:
        """Per-window surprise (Wc,) for one context length, reusing the target forward."""
        ctx_1d, tgt_1d, _, _ = per_C_masks[C]
        Wc = h_all.size(0)
        ac = _ac(); ac.__enter__()
        # Broadcast masks across the Wc-batch dim.
        ctx_idx = ctx_1d.unsqueeze(0).expand(Wc, -1).contiguous()  # (Wc, N_ctx)
        tgt_idx = tgt_1d.unsqueeze(0).expand(Wc, -1).contiguous()  # (Wc, N_tgt)

        gather_ctx = ctx_idx.unsqueeze(-1).expand(-1, -1, embed_dim)
        gather_tgt = tgt_idx.unsqueeze(-1).expand(-1, -1, embed_dim)
        h_tgt = torch.gather(h_all, dim=1, index=gather_tgt)
        if need_ctx_forward:
            # Training-faithful path: context encoder sees ONLY the ctx_idx tokens
            # (apply_masks filters BEFORE the blocks). Applies whenever dual_forward
            # is on OR context_forward_mode='masked' is requested. In the EMA-only case
            # (dual_forward=False) we still use the shared module as context_encoder --
            # semantics match training-time train.py:435-438.
            z_ctx = context_encoder(clip_batch, masks=[ctx_idx])  # (Wc, N_ctx, D)
            if isinstance(z_ctx, list):
                z_ctx = z_ctx[-1]
        else:
            # "sliced" mode: gather from the shared target-encoder full-clip forward.
            # Fast but not training-faithful; kept as an escape hatch for benchmarking.
            z_ctx = torch.gather(h_all, dim=1, index=gather_ctx)

        if target_layer_norm:
            h_tgt = F.layer_norm(h_tgt, (h_tgt.size(-1),))

        z_pred = predictor(z_ctx, ctx_idx, tgt_idx, mask_index=mask_index)  # (Wc, N_tgt, D_out)
        ac.__exit__(None, None, None)          # 거리 계산은 autocast 밖(fp32)에서
        # Per-window distance: reduce over (N_tgt, D) only, keep batch dim (Wc,).
        if distance == "l1":
            return (z_pred.float() - h_tgt.float()).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
        if distance == "smoothl1":
            return F.smooth_l1_loss(z_pred.float(), h_tgt.float(), reduction="none").mean(dim=(1, 2))
        if distance == "l2":
            return (z_pred.float() - h_tgt.float()).pow(2).mean(dim=(1, 2))
        if distance == "cosine":
            cs = F.cosine_similarity(z_pred.float(), h_tgt.float(), dim=-1)  # (Wc, N_tgt)
            return 1.0 - cs.mean(dim=1)
        raise ValueError(f"unknown distance {distance!r}")

    # Accumulate per-C, per-window distances as on-device scalars; single .cpu() at the end.
    per_C_surprise_gpu: Dict[int, List[torch.Tensor]] = {int(C): [] for C in context_lengths}

    for chunk in chunks:
        clip_batch = _stack(chunk)  # (Wc, 3, M, H, W_sp)
        # ---- 1 target-encoder forward for the entire chunk (always) ---------------
        with _ac():
            h_all = target_encoder(clip_batch)  # (Wc, N_total, D)
        if isinstance(h_all, list):
            h_all = h_all[-1]

        # ---- per-C predictor forwards ---------------------------------------------
        for C in context_lengths:
            per_C_surprise_gpu[int(C)].append(_window_distances(clip_batch, h_all, int(C)))

    # ---- growing prefix: window 0 only, one entry per distinct smaller context -----
    prefix_gpu: Dict[int, torch.Tensor] = {}
    if prefix_Cs:
        clip0 = _stack(windows[:1])                      # (1, 3, M, H, W_sp)
        with _ac():
            h0 = target_encoder(clip0)
        if isinstance(h0, list):
            h0 = h0[-1]
        for c in prefix_Cs:
            prefix_gpu[c] = _window_distances(clip0, h0, c)   # (1,)

    # ---- single sync per video --------------------------------------------------
    starts_np = np.asarray([w.start for w in windows], dtype=np.int64)
    out: Dict[int, VideoSurprise] = {}
    for C in context_lengths:
        C = int(C)
        pre = prefix_of[C]
        # Order matters: the growing entries come FIRST, matching the official
        # `torch.hstack([loss_beginning, loss])`. It is irrelevant for avg/max but keeps
        # per-window traces readable as a surprise-over-time curve.
        parts = [prefix_gpu[c] for c in pre] + per_C_surprise_gpu[C]
        surprise_np = torch.cat(parts, dim=0).detach().float().cpu().numpy()
        out[C] = VideoSurprise(
            window_starts=np.concatenate([np.zeros(len(pre), dtype=np.int64), starts_np]),
            window_context_lengths=np.concatenate([
                np.asarray(pre, dtype=np.int64),
                np.full(len(windows), C, dtype=np.int64),
            ]),
            surprise=surprise_np.astype(np.float32),
            context_length=C,
        )
    return out


# --------------------------- context sweep (legacy per-C loop) ---------------


def score_video_context_sweep(
    video: torch.Tensor,
    bundle: VJEPA2Bundle,
    *,
    window_size: int,
    context_lengths: List[int],
    stride: int,
    distance: str,
    loss_exp: float,
    target_layer_norm: bool,
    aggregation: str,
    protocol: str = "fixed",
) -> Dict[int, VideoSurprise]:
    """Slow per-C loop, retained for correctness comparisons vs the batched fast path."""
    out: Dict[int, VideoSurprise] = {}
    for C in context_lengths:
        out[C] = score_video(
            video, bundle,
            window_size=window_size, context_length=C, stride=stride,
            distance=distance, loss_exp=loss_exp,
            target_layer_norm=target_layer_norm, protocol=protocol,
        )
    return out
