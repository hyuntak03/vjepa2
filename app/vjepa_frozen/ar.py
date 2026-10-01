"""팔 C — 자기회귀 predictor (V-JEPA 2-AC 의 레시피를 action 없이): 인코딩 · 손실 · in-loop val.

AC (`app/vjepa_droid/train.py`, `z_AC_training/README.md` §1) 와 같은 점
  * encoder 는 **블록(tubelet 2 장)마다 따로** 돈다 → 블록 b 의 토큰은 블록 b 만 본다 (시간 누수 0).
    그래서 teacher forcing 입력으로 미래 블록의 GT 를 넣어도 그 안에 더 먼 미래가 섞여 있지 않다.
    (창 전체를 한 번에 인코딩하면 h_b 가 양방향 시간 attention 으로 b 뒤 프레임을 담아 TF 가 누수다.)
  * 입력 = 타깃 = LN(encoder(블록)). predictor 출력에도 LN (AC `normalize_reps`) → 되먹임이 같은 공간.
  * jloss (teacher forcing) = mean|p_tf − h[1:]|  모든 위치.   sloss (rollout) = mean|p_ar − h[C:C+R]|.
    loss = jloss + sloss. rollout 첫 스텝은 TF 의 위치 C-1 출력 재사용 (AC 처럼 그 블록은 두 손실에 다 든다).
AC 와 다른 점
  * 블록 = 실제 연속 2 프레임 (AC 는 한 프레임을 2 번 복제). 사전학습 tubelet 과 같아 분포 안이다.
  * encoder 는 `target_encoder`(EMA) 하나 — config 가 context_encoder_key 도 target_encoder 로 둔다.
  * rollout 길이 R = min(K, loss.rollout_steps) (AC 는 1 스텝 고정).

채점 (in-loop val) 은 둘을 같이 낸다:
  rollout surprise : 문맥 C 블록 → K 블록 자기회귀, mean|p_ar − LN(h)|   ← prefix/full 팔의 채점과 같은 꼴 (주지표)
  tf surprise      : 위치 C-1..nb-2 의 다음 블록 예측 (GT 되먹임), mean|p_tf − LN(h)|  ← 한 스텝 앞 놀람 (보조)
⚠️ 정식 채점기 (analysis/intphys2/model.py, evals/world_model_analysis) 는 아직 이 모드를 모른다 (별도 작업).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from app.vjepa_frozen.val import ValSet, _score


def encode_blocks(enc, clips: torch.Tensor, tubelet: int) -> torch.Tensor:
    """(B, 3, T, H, W) → 블록마다 따로 인코딩 → (B, nb*S, D), affine-free LN. 블록 b 토큰은 블록 b 프레임만 본다."""
    B, Cc, T, H, W = clips.shape
    nb = T // tubelet
    x = clips[:, :, : nb * tubelet].reshape(B, Cc, nb, tubelet, H, W).permute(0, 2, 1, 3, 4, 5)
    x = x.reshape(B * nb, Cc, tubelet, H, W)
    h = enc(x)                                                       # (B*nb, S, D)
    h = h.reshape(B, nb * h.size(1), h.size(-1))
    return F.layer_norm(h, (h.size(-1),))


def block_indices(B: int, nb: int, S: int, device) -> torch.Tensor:
    return torch.arange(nb * S, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()


def l1(p: torch.Tensor, h: torch.Tensor, loss_exp: float) -> torch.Tensor:
    return (p.float() - h.float()).abs().pow(loss_exp).mean() / loss_exp


def ar_losses(predictor, h: torch.Tensor, S: int, C: int, K: int, loss_exp: float, rollout_steps: int):
    """h (B, nb*S, D) LN 된 블록 토큰. return (loss, jloss, sloss, R). predictor 는 DDP 래퍼여도 된다 (forward 한 번)."""
    B, N = h.shape[:2]
    nb = N // S
    if nb != C + K or nb < 2:
        raise ValueError(f"ar_losses: nb={nb} 가 C+K={C + K} 와 다르거나 2 미만")
    R = max(1, min(int(K), int(rollout_steps)))
    idx = block_indices(B, nb, S, h.device)
    p_tf, p_ar = predictor(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": R})
    jloss = l1(p_tf, h[:, S:], loss_exp)
    sloss = l1(p_ar, h[:, C * S: (C + R) * S], loss_exp)
    return jloss + sloss, jloss, sloss, R


@torch.no_grad()
def run_val_ar(vs: ValSet, tgt_enc, pred_mod, device, autocast_dtype, rank: int, ws: int,
               tubelet: int, spatial: int, context_length: int, batch_size: int, loss_exp: float) -> Dict:
    """val.run_val 의 팔 C 판. 주지표 = rollout surprise, 보조 = tf surprise (res['tf'])."""
    pred_mod.eval()
    n_frames = vs.ds.n_frames
    S = spatial
    C = context_length // tubelet
    nb = n_frames // tubelet
    K = nb - C
    if K < 1:
        raise ValueError(f"val: context_length {context_length} 가 n_frames {n_frames} 이상")
    idx_all = list(range(rank, len(vs), ws))
    out_ro, out_tf = {}, {}
    t0 = time.time()
    ac = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else torch.autocast("cuda", enabled=False)
    for s in range(0, len(idx_all), batch_size):
        chunk = idx_all[s: s + batch_size]
        x = torch.stack([vs.ds.transform(vs.ds.read_uint8(i)) for i in chunk]).to(device, non_blocking=True)
        n = len(chunk)
        with ac:
            h = encode_blocks(tgt_enc, x.to(next(tgt_enc.parameters()).dtype), tubelet)     # (n, nb*S, D)
            idx = block_indices(n, nb, S, device)
            p_ar = pred_mod.rollout(h[:, : C * S], idx[:, : C * S], K)                       # (n, K*S, D)
            p_tf = pred_mod.forward_seq(h[:, :-S], idx[:, :-S])[:, (C - 1) * S:]               # 블록 C..nb-1 예측
        tgt = h[:, C * S:].float()
        d_ro = (p_ar.float() - tgt).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
        d_tf = (p_tf.float() - tgt).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
        for k, i in enumerate(chunk):
            vid = vs.ds.rows[i]["video_id"]
            out_ro[vid], out_tf[vid] = float(d_ro[k]), float(d_tf[k])
    if dist.is_available() and dist.is_initialized() and ws > 1:
        g = [None] * ws
        dist.all_gather_object(g, (out_ro, out_tf))
        ro, tf = {}, {}
        for a, b in g:
            ro.update(a); tf.update(b)
    else:
        ro, tf = out_ro, out_tf
    pred_mod.train()
    if rank != 0:
        return {}
    el = time.time() - t0
    res = _score(vs, ro, el)
    r_tf = _score(vs, tf, el)
    res["tf"] = {k: r_tf[k] for k in ("acc", "n_ties", "surprise_pos", "surprise_neg", "margin", "by_type")}
    return res


def format_val_ar(res: Dict) -> str:
    from app.vjepa_frozen.val import format_val
    s = format_val(res)
    t = res.get("tf")
    if t:
        s += (f"\n      [tf] acc {100*t['acc']:.1f}% margin {t['margin']:+.4f} | "
              + "  ".join(f"{k}: {100*v['acc']:.1f}" for k, v in t["by_type"].items()))
    return s


# ----------------------------------------------------------------------------- ctx_ar (2026-09-27)
# 사용자 요청: "context 자체를 video 로 쫙 주고 predictor 가 미래를 생성". 문맥 = context encoder 가 C 블록 창을
# 한 번에 인코딩한 z (prefix 팔과 동일), 미래 GT 만 블록별 LN(h) (TF 입력·타깃; 블록별이라 누수 없음).
# 손실 = TF (블록 C..C+K-1, K 개 위치) + rollout (문맥에서 R 블록 되먹임).
def ctx_ar_losses(predictor, z_ctx, h_fut, S: int, C: int, K: int, loss_exp: float, rollout_steps: int):
    """z_ctx (B, C*S, D) 문맥 z (LN 안 함), h_fut (B, K*S, D) 블록별 LN(h). return (loss, jloss, sloss, R)."""
    B = z_ctx.size(0)
    if z_ctx.size(1) != C * S or h_fut.size(1) != K * S or K < 1:
        raise ValueError(f"ctx_ar_losses: z {tuple(z_ctx.shape)} h_fut {tuple(h_fut.shape)} C={C} K={K}")
    R = max(1, min(int(K), int(rollout_steps)))
    idx = block_indices(B, C + K, S, z_ctx.device)
    p_tf, p_ar = predictor(z_ctx, ar={"h_fut": h_fut, "idx": idx, "n_ctx_blocks": C, "rollout_steps": R})
    jloss = l1(p_tf, h_fut, loss_exp)
    sloss = l1(p_ar, h_fut[:, : R * S], loss_exp)
    return jloss + sloss, jloss, sloss, R


@torch.no_grad()
def run_val_ctx_ar(vs: ValSet, ctx_enc, tgt_enc, pred_mod, device, autocast_dtype, rank: int, ws: int,
                   tubelet: int, spatial: int, context_length: int, batch_size: int, loss_exp: float) -> Dict:
    """run_val_ar 의 ctx_ar 판: 문맥은 창 단위 z (ctx_masked), 미래 타깃은 블록별 LN(h). 주지표 rollout, 보조 tf."""
    pred_mod.eval()
    n_frames = vs.ds.n_frames
    S = spatial
    C = context_length // tubelet
    nb = n_frames // tubelet
    K = nb - C
    if K < 1:
        raise ValueError(f"val: context_length {context_length} 가 n_frames {n_frames} 이상")
    idx_all = list(range(rank, len(vs), ws))
    out_ro, out_tf = {}, {}
    t0 = time.time()
    ac = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else torch.autocast("cuda", enabled=False)
    for s in range(0, len(idx_all), batch_size):
        chunk = idx_all[s: s + batch_size]
        x = torch.stack([vs.ds.transform(vs.ds.read_uint8(i)) for i in chunk]).to(device, non_blocking=True)
        n = len(chunk)
        with ac:
            xe = x.to(next(tgt_enc.parameters()).dtype)
            idx = block_indices(n, nb, S, device)
            z = ctx_enc(xe, masks=[idx[:, : C * S]])                                           # 창 단위 문맥 z
            h_fut = encode_blocks(tgt_enc, xe[:, :, C * tubelet:], tubelet)                      # 블록별 LN(h), (n, K*S, D)
            p_tf = pred_mod.forward_mixed(z, h_fut[:, :-S], idx[:, : (nb - 1) * S], C)          # 블록 C..nb-1 예측
            p_ar = pred_mod.rollout_ctx(z, idx[:, : C * S], K)
        tgt = h_fut.float()
        d_ro = (p_ar.float() - tgt).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
        d_tf = (p_tf.float() - tgt).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
        for k, i in enumerate(chunk):
            vid = vs.ds.rows[i]["video_id"]
            out_ro[vid], out_tf[vid] = float(d_ro[k]), float(d_tf[k])
    if dist.is_available() and dist.is_initialized() and ws > 1:
        g = [None] * ws
        dist.all_gather_object(g, (out_ro, out_tf))
        ro, tf = {}, {}
        for a, b in g:
            ro.update(a); tf.update(b)
    else:
        ro, tf = out_ro, out_tf
    pred_mod.train()
    if rank != 0:
        return {}
    el = time.time() - t0
    res = _score(vs, ro, el)
    r_tf = _score(vs, tf, el)
    res["tf"] = {k: r_tf[k] for k in ("acc", "n_ties", "surprise_pos", "surprise_neg", "margin", "by_type")}
    return res


# ----------------------------------------------------------------------------- ctx_state (2026-09-30)
# ctx_ar 에서 **되먹임 공간**만 바꾼 팔 (사용자 결정): rollout 이 그린 출력 대신 predictor 상태(384) 를 되먹인다.
# 입력과 타깃을 나눈다 — 되먹임이 출력 공간을 안 쓰므로 가능하다:
#   TF 관측 입력 : target encoder **블록별** LN(h)   (encoder 는 양방향이라 미래 구간을 한 번에 넣으면 블록 t 가 t 뒤를 본다 = 누수)
#   손실 타깃    : target encoder **창(C+K) 전체** LN(h) 의 미래 블록  (릴리즈 사전학습 · surprise 채점 · in-loop val 과 같은 타깃)
def ctx_state_losses(predictor, z_ctx, h_obs, tgt, S: int, C: int, K: int, loss_exp: float, rollout_steps: int,
                     tf_weight: float = 1.0):
    """z_ctx (B, C*S, D), h_obs (B, K*S, D) 블록별 LN(h) 또는 None (tf_weight=0), tgt (B, K*S, D) 창 LN(h) 의 미래 블록.
    loss = tf_weight * jloss + sloss. return (loss, jloss, sloss, R) — TF 를 끄면 jloss 는 0."""
    B = z_ctx.size(0)
    use_tf = float(tf_weight) > 0
    if z_ctx.size(1) != C * S or tgt.size(1) != K * S or K < 1 or (use_tf and (h_obs is None or h_obs.size(1) != K * S)):
        raise ValueError(f"ctx_state_losses: z {tuple(z_ctx.shape)} h_obs {None if h_obs is None else tuple(h_obs.shape)} "
                         f"tgt {tuple(tgt.shape)} C={C} K={K} tf_weight={tf_weight}")
    R = max(1, min(int(K), int(rollout_steps)))
    idx = block_indices(B, C + K, S, z_ctx.device)
    p_tf, p_ar = predictor(z_ctx, ar={"h_obs": h_obs if use_tf else None, "idx": idx, "n_ctx_blocks": C, "rollout_steps": R})
    sloss = l1(p_ar, tgt[:, : R * S], loss_exp)
    if not use_tf:
        return sloss, sloss.new_zeros(()), sloss, R
    jloss = l1(p_tf, tgt, loss_exp)
    return float(tf_weight) * jloss + sloss, jloss, sloss, R


@torch.no_grad()
def run_val_ctx_state(vs: ValSet, ctx_enc, tgt_enc, pred_mod, device, autocast_dtype, rank: int, ws: int,
                      tubelet: int, spatial: int, context_length: int, batch_size: int, loss_exp: float) -> Dict:
    """ctx_state 판 in-loop val. 문맥 z → 상태 rollout K 블록 (주지표) / TF (보조). 타깃 = 창 전체 LN(h) 의 미래 블록."""
    pred_mod.eval()
    n_frames = vs.ds.n_frames
    S = spatial
    C = context_length // tubelet
    nb = n_frames // tubelet
    K = nb - C
    if K < 1:
        raise ValueError(f"val: context_length {context_length} 가 n_frames {n_frames} 이상")
    idx_all = list(range(rank, len(vs), ws))
    out_ro, out_tf = {}, {}
    t0 = time.time()
    ac = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else torch.autocast("cuda", enabled=False)
    for s in range(0, len(idx_all), batch_size):
        chunk = idx_all[s: s + batch_size]
        x = torch.stack([vs.ds.transform(vs.ds.read_uint8(i)) for i in chunk]).to(device, non_blocking=True)
        n = len(chunk)
        with ac:
            xe = x.to(next(tgt_enc.parameters()).dtype)
            idx = block_indices(n, nb, S, device)
            z = ctx_enc(xe, masks=[idx[:, : C * S]])
            h_obs = encode_blocks(tgt_enc, xe[:, :, C * tubelet:], tubelet)
            tgt = F.layer_norm(tgt_enc(xe), (z.size(-1),))[:, C * S:]
            h_tf = pred_mod.forward_state(z, pred_mod.embed_obs(h_obs[:, :-S]), idx[:, : (nb - 1) * S], C)
            p_tf = pred_mod.render(h_tf)
            p_ar = pred_mod.render(pred_mod.rollout_state_kv(z, idx[:, : C * S], K))
        t_ = tgt.float()
        d_ro = (p_ar.float() - t_).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
        d_tf = (p_tf.float() - t_).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
        for k, i in enumerate(chunk):
            vid = vs.ds.rows[i]["video_id"]
            out_ro[vid], out_tf[vid] = float(d_ro[k]), float(d_tf[k])
    if dist.is_available() and dist.is_initialized() and ws > 1:
        g = [None] * ws
        dist.all_gather_object(g, (out_ro, out_tf))
        ro, tf = {}, {}
        for a, b in g:
            ro.update(a); tf.update(b)
    else:
        ro, tf = out_ro, out_tf
    pred_mod.train()
    if rank != 0:
        return {}
    el = time.time() - t0
    res = _score(vs, ro, el)
    r_tf = _score(vs, tf, el)
    res["tf"] = {k: r_tf[k] for k in ("acc", "n_ties", "surprise_pos", "surprise_neg", "margin", "by_type")}
    return res
