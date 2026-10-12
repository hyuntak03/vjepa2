"""rollout 계열 predictor (kind = ar · ar_state · ctx_ar) 의 채점 경로. 학습 코드 `app/vjepa_frozen/ar.py` 와 같은 정의.

창 = 문맥 C 블록 + 미래 K 블록 (블록 = tubelet 2 프레임)

    h_b  = LN(target_encoder(블록 b))                  블록마다 따로 인코딩
    p    = rollout(h[0..C-1], K)                       ar: render 되먹임 · ar_state: 상태 되먹임
    S    = mean |p − h[C..C+K-1]|                      주지표
    p_tf = forward_seq(h[0..nb-2])[C-1..]              보조 (GT 되먹임 한 걸음 앞), surprise.ar_tf

⚠️ 표준 채점과 입력 · 타깃 공간이 다르다 (표준 타깃 = LN(target_encoder(창 전체))). surprise 크기 · margin 은
   다른 kind 와 비교하지 않고 쌍 정확도만 비교한다. 문맥 encoder 는 쓰지 않는다 (ctx_ar 만 예외).

config 노브
    model.predictor.kv_rollout   true (기본, 체크포인트 arch 값) = KV 캐시 rollout · false = 매 걸음 전체 재계산. 값은 같다
    surprise.ar_baseline         null | copy_last (문맥 마지막 블록을 미래 전부에) | zero — predictor 대신 기준선을 같은 공간에서 잰다
    surprise.ar_tf               보조 키 'ar_tf'
"""
from __future__ import annotations

import torch

from app.vjepa_frozen.ar import block_indices, encode_blocks

AR_BASELINES = (None, "copy_last", "zero")


def check_ar_ready(target_layer_norm: bool, baseline=None) -> None:
    if not target_layer_norm:
        raise ValueError("rollout 계열 채점은 surprise.target_layer_norm=true 가 전제다 (입력 · 타깃 · 출력이 모두 LN 공간)")
    if baseline not in AR_BASELINES:
        raise ValueError(f"surprise.ar_baseline={baseline!r}; {AR_BASELINES} 중 하나")


def rollout(predictor, x_ctx: torch.Tensor, idx_ctx: torch.Tensor, n_steps: int) -> torch.Tensor:
    m = getattr(predictor, "module", predictor)
    return (m.rollout_kv if getattr(m, "kv_rollout", False) else m.rollout)(x_ctx, idx_ctx, n_steps)


def rollout_from_blocks(predictor, h: torch.Tensor, n_ctx_blocks: int, spatial: int, with_tf: bool = False, baseline=None):
    """h (B, nb*S, D) 블록별 LN(h). return (p (B, K*S, D), h_future, p_tf | None). 인덱스는 창 기준 0 부터."""
    S, C = int(spatial), int(n_ctx_blocks)
    nb = h.size(1) // S
    K = nb - C
    if h.size(1) != nb * S or C < 1 or K < 1:
        raise ValueError(f"rollout_from_blocks: 토큰 {h.size(1)} (블록 {nb} × {S}), 문맥 {C} 블록 → 미래 {K} 블록")
    idx = block_indices(h.size(0), nb, S, h.device)
    if baseline == "copy_last":
        p = h[:, (C - 1) * S: C * S].repeat(1, K, 1)
    elif baseline == "zero":
        p = torch.zeros_like(h[:, C * S:])
    else:
        p = rollout(predictor, h[:, : C * S], idx[:, : C * S], K)
    p_tf = getattr(predictor, "module", predictor).forward_seq(h[:, :-S], idx[:, :-S])[:, (C - 1) * S:] if with_tf else None
    return p, h[:, C * S:], p_tf


def _check_window(x: torch.Tensor, ctx_frames: int, tubelet: int) -> None:
    if int(x.size(2)) % tubelet or ctx_frames % tubelet:
        raise ValueError(f"창 {x.size(2)} · 문맥 {ctx_frames} 프레임이 tubelet {tubelet} 의 배수가 아니다 "
                         "(encode_blocks 는 남는 프레임을 조용히 버린다)")


def ar_window(target_encoder, predictor, x: torch.Tensor, ctx_frames: int, tubelet: int, spatial: int,
              with_tf: bool = False, baseline=None):
    """창 (B, 3, T, H, W) 를 블록별로 인코딩하고 rollout. autocast 안에서 부른다. 미래 토큰 배치는 표준 경로 gather(ti) 와 같다."""
    _check_window(x, ctx_frames, tubelet)
    h = encode_blocks(target_encoder, x, tubelet)
    return rollout_from_blocks(predictor, h, ctx_frames // tubelet, spatial, with_tf=with_tf, baseline=baseline)


def ctx_ar_from_blocks(context_encoder, predictor, x: torch.Tensor, h_blk: torch.Tensor, n_ctx_blocks: int, spatial: int,
                       with_tf: bool = False):
    """ctx_ar: 문맥 = context_encoder(창 앞 C 블록) z, 미래 타깃 · TF 입력 = 블록별 LN(h). run_val_ctx_ar 와 같은 정의."""
    S, C = int(spatial), int(n_ctx_blocks)
    nb = h_blk.size(1) // S
    K = nb - C
    if h_blk.size(1) != nb * S or C < 1 or K < 1:
        raise ValueError(f"ctx_ar_from_blocks: 토큰 {h_blk.size(1)} (블록 {nb} × {S}), 문맥 {C} 블록 → 미래 {K} 블록")
    m = getattr(predictor, "module", predictor)
    idx = block_indices(x.size(0), nb, S, x.device)
    z = context_encoder(x, masks=[idx[:, : C * S]])
    h_fut = h_blk[:, C * S:]
    p_tf = m.forward_mixed(z, h_fut[:, :-S], idx[:, : (nb - 1) * S], C)
    p = m.rollout_ctx(z, idx[:, : C * S], K, first=p_tf[:, :S])
    return p, h_fut, (p_tf if with_tf else None)


def ctx_ar_window(context_encoder, target_encoder, predictor, x: torch.Tensor, ctx_frames: int, tubelet: int, spatial: int,
                  with_tf: bool = False):
    _check_window(x, ctx_frames, tubelet)
    h = encode_blocks(target_encoder, x, tubelet)
    return ctx_ar_from_blocks(context_encoder, predictor, x, h, ctx_frames // tubelet, spatial, with_tf=with_tf)


def assert_finite(t: torch.Tensor, where: str) -> None:
    if not torch.isfinite(t).all():
        raise FloatingPointError(f"{where}: rollout 출력에 inf/nan — fp16 넘침일 수 있다. model.autocast=bfloat16 으로 다시 돌릴 것")
