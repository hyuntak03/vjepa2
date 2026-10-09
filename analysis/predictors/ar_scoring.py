"""자기회귀 predictor (kind=ar, Ariel 팔 C) 의 채점 경로 — 2026-09-27 이식.

Ariel 학습 코드 (`z_ariel/vjepa2_train_code_20260927`, 우리 레포 `app/vjepa_frozen/ar.py`) 의
in-loop val `run_val_ar` 과 **같은 정의**를 표준 채점기 (`evals/world_model_analysis/eval.py`) 와
분석 스크립트가 쓰도록 한 곳에 모은다. 인코딩 · 인덱스 함수는 학습 코드의 것을 그대로 import 한다.

정의 (창 = 문맥 C 블록 + 미래 K 블록, 블록 = tubelet 2 프레임)

    h_b   = LN(target_encoder(블록 b 의 2 프레임))         블록마다 **따로** 인코딩 (블록 b 는 블록 b 만 본다)
    p_ar  = predictor.rollout(h[블록 0..C-1], K)           문맥에서 K 블록 자기회귀 (자기 예측을 되먹임)
    S_ar  = mean_{블록 C..C+K-1, 토큰, D} |p_ar − h|        ← 주지표 (surprise)
    보조  p_tf = predictor.forward_seq(h[블록 0..nb-2])[블록 C-1..]   한 스텝 앞 예측 (GT 되먹임)
          S_tf = mean |p_tf − h[블록 C..]|

⚠️ 표준 채점 (`surprise_c16t32` · `intphys1_sliding`) 과 **입력 · 타깃 공간이 다르다.**
   표준: 문맥 = online encoder(문맥 프레임), 타깃 = LN(target_encoder(창 전체))[미래]  (양방향 시간 attention)
   AR  : 문맥 = 타깃 = LN(target_encoder(블록 하나))                                  (블록마다 따로)
   그래서 surprise 크기 · margin · 복사 기준선은 다른 kind 와 비교하지 않는다. **쌍 정확도만** 비교한다.
⚠️ 문맥 encoder 는 쓰지 않는다 (학습 때 context_encoder_key = target_encoder). online encoder 출력을
   AR predictor 에 넣으면 분포 밖이다.
⚠️ matched pair 는 문맥 픽셀이 같으므로 p_ar 이 쌍 안에서 비트 단위로 같다 (블록별 인코딩이라 미래가 새지 않는다).
   p_tf 는 자기 영상의 미래 GT 를 되먹이므로 쌍 안에서 다르다 — 보조로만 읽는다.
⚠️ 학습 분포: 문맥 C ∈ {2,4,8,16} 블록, rollout ≤ 8 블록. surprise_c16t32 (C 8, K 8) 은 안,
   IntPhys1 skip2_w32 의 C=2 블록 (K 14) 같은 칸은 밖이다.
"""
from __future__ import annotations

import torch

from app.vjepa_frozen.ar import block_indices, encode_blocks


def check_ar_ready(target_layer_norm: bool) -> None:
    if not target_layer_norm:
        raise ValueError("kind=ar 채점은 surprise.target_layer_norm=true 가 전제다 (입력 · 타깃 · 출력이 모두 LN 공간)")


def rollout_from_blocks(predictor, h: torch.Tensor, n_ctx_blocks: int, spatial: int,
                        with_tf: bool = False):
    """블록별로 인코딩된 LN(h) (B, nb*S, D) 에서 문맥 C 블록 → 나머지 K 블록을 자기회귀로 낸다.

    return (p_ar (B, K*S, D), h_future (B, K*S, D), p_tf (B, K*S, D) | None).
    인덱스는 **창 기준 0 부터** (학습 때와 같다). 되먹임 · 출력 LN 은 predictor 안에서 한다.
    """
    S = int(spatial)
    nb = h.size(1) // S
    C = int(n_ctx_blocks)
    K = nb - C
    if h.size(1) != nb * S or C < 1 or K < 1:
        raise ValueError(f"rollout_from_blocks: 토큰 {h.size(1)} (블록 {nb} × {S}), 문맥 {C} 블록 → 미래 {K} 블록")
    idx = block_indices(h.size(0), nb, S, h.device)
    p = predictor.rollout(h[:, : C * S], idx[:, : C * S], K)
    p_tf = None
    if with_tf:
        p_tf = predictor.forward_seq(h[:, :-S], idx[:, :-S])[:, (C - 1) * S:]
    return p, h[:, C * S:], p_tf


def ar_window(target_encoder, predictor, x: torch.Tensor, ctx_frames: int, tubelet: int, spatial: int,
              with_tf: bool = False):
    """창 하나 (B, 3, T, H, W) 를 블록별로 인코딩하고 rollout 한다. autocast 안에서 부른다.

    return (p_ar, h_future, p_tf) — 모양 (B, K*S, D), K = (T − ctx_frames) / tubelet.
    미래 토큰 배치 (블록 우선, 그 안에서 공간) 는 표준 경로의 gather(ti) 결과와 같다 → 거리 · token_subset 코드를 그대로 쓴다.
    """
    T = int(x.size(2))
    if T % tubelet or ctx_frames % tubelet:
        raise ValueError(f"ar_window: 창 {T} · 문맥 {ctx_frames} 프레임이 tubelet {tubelet} 의 배수가 아니다 "
                         f"(encode_blocks 는 남는 프레임을 조용히 버린다)")
    h = encode_blocks(target_encoder, x, tubelet)          # (B, nb*S, D), 이미 LN
    return rollout_from_blocks(predictor, h, ctx_frames // tubelet, spatial, with_tf=with_tf)


def ctx_ar_from_blocks(context_encoder, predictor, x: torch.Tensor, h_blk: torch.Tensor, n_ctx_blocks: int, spatial: int,
                       with_tf: bool = False):
    """kind=ctx_ar (Ariel 팔 C', 2026-09-29): 문맥 = context encoder(online `encoder`) 가 창의 앞 C 블록을 **한 번에** 인코딩한 z
    (prefix 팔과 같은 입력, LN 안 함), 미래 타깃 · TF 입력 = 블록별 LN(h) (h_blk 는 encode_blocks 의 출력, 창 전체).
    p_ar = predictor.rollout_ctx(z, idx_ctx, K)  (자기 되먹임), p_tf = forward_mixed(z, h_fut[:-S], idx, C) (진짜 블록 되먹임, 보조).
    return (p_ar (B, K*S, D), h_fut (B, K*S, D), p_tf | None). app/vjepa_frozen/ar.py:run_val_ctx_ar 와 같은 정의."""
    S = int(spatial); nb = h_blk.size(1) // S; C = int(n_ctx_blocks); K = nb - C
    if h_blk.size(1) != nb * S or C < 1 or K < 1:
        raise ValueError(f"ctx_ar_from_blocks: 토큰 {h_blk.size(1)} (블록 {nb} × {S}), 문맥 {C} 블록 → 미래 {K} 블록")
    idx = block_indices(x.size(0), nb, S, x.device)
    z = context_encoder(x, masks=[idx[:, : C * S]])                       # 창 단위 문맥 z (미래 토큰은 encoder 첫 층 전에 제거)
    h_fut = h_blk[:, C * S:]
    p_tf = predictor.forward_mixed(z, h_fut[:, :-S], idx[:, : (nb - 1) * S], C) if (with_tf or True) else None
    p_ar = predictor.rollout_ctx(z, idx[:, : C * S], K, first=p_tf[:, :S])    # 첫 스텝 = TF 의 블록 C 예측 (같은 값, 학습 코드와 동일)
    return p_ar, h_fut, (p_tf if with_tf else None)


def ctx_ar_window(context_encoder, target_encoder, predictor, x: torch.Tensor, ctx_frames: int, tubelet: int, spatial: int,
                  with_tf: bool = False):
    """창 하나 (B, 3, T, H, W): 블록별 LN(h) 를 만들고 ctx_ar_from_blocks. autocast 안에서 부른다."""
    T = int(x.size(2))
    if T % tubelet or ctx_frames % tubelet:
        raise ValueError(f"ctx_ar_window: 창 {T} · 문맥 {ctx_frames} 프레임이 tubelet {tubelet} 의 배수가 아니다")
    h = encode_blocks(target_encoder, x, tubelet)
    return ctx_ar_from_blocks(context_encoder, predictor, x, h, ctx_frames // tubelet, spatial, with_tf=with_tf)


def assert_finite(t: torch.Tensor, where: str) -> None:
    """bf16 으로 학습한 predictor 를 fp16 autocast 로 돌릴 때 넘침을 조용히 넘기지 않는다."""
    if not torch.isfinite(t).all():
        raise FloatingPointError(f"{where}: kind=ar 출력에 inf/nan — fp16 넘침일 수 있다. "
                                 f"model.autocast=bfloat16 (학습 때와 같은 정밀도) 로 다시 돌릴 것")
