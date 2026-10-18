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


def ar_losses(predictor, h: torch.Tensor, S: int, C: int, K: int, loss_exp: float, rollout_steps: int, actions=None,
              out: dict = None):
    """h (B, nb*S, D) LN 된 블록 토큰. return (loss, jloss, sloss, R). predictor 는 DDP 래퍼여도 된다 (forward 한 번).
    actions (B, nb, A): action 조건부 predictor (Push-T) 일 때만.
    out (2026-10-19): dict 를 주면 출력 p_tf (B, (nb-1)*S, D) · p_ar (B, R*S, D) 를 담아 준다 (state_head 보조 손실용)."""
    B, N = h.shape[:2]
    nb = N // S
    if nb != C + K or nb < 2:
        raise ValueError(f"ar_losses: nb={nb} 가 C+K={C + K} 와 다르거나 2 미만")
    R = max(1, min(int(K), int(rollout_steps)))
    idx = block_indices(B, nb, S, h.device)
    p_tf, p_ar = predictor(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": R, "actions": actions})
    if out is not None:
        out["p_tf"], out["p_ar"] = p_tf, p_ar
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


# ----------------------------------------------------------------------------- ctx_state (vast 이식, 2026-10-16)
# 사용자 요청: "context 를 video 로 주고 target 도 video 로". ctx_ar 에서 두 가지를 바꾼다
#   되먹임 : rollout 이 그린 출력 (LN(proj(.)), 1280) 대신 **predictor 상태** (predictor_norm, 384) 를 다음 걸음 입력으로
#   타깃   : 블록별 LN(h) 대신 **창 (C+K) 전체** LN(target_encoder) 의 미래 블록
#            = 릴리즈 사전학습 · surprise_c16t32 채점과 같은 타깃. 그래서 surprise 가 표준 채점과 같은 공간에 있다
# TF 관측 **입력**은 여전히 블록별 LN(h) 다 — encoder 가 양방향이라 미래 구간을 한 번에 넣으면 블록 t 가 t 뒤를 본다 (누수).
def ctx_state_losses(predictor, z_ctx, h_obs, tgt, S: int, C: int, K: int, loss_exp: float, rollout_steps: int,
                     tf_weight: float = 1.0):
    """z_ctx (B, C*S, D) 문맥 z (LN 안 함), h_obs (B, K*S, D) 블록별 LN(h) 또는 None (tf_weight=0),
    tgt (B, K*S, D) 창 LN(h) 의 미래 블록. loss = tf_weight * jloss + sloss. return (loss, jloss, sloss, R)."""
    B = z_ctx.size(0)
    use_tf = float(tf_weight) > 0
    if z_ctx.size(1) != C * S or tgt.size(1) != K * S or K < 1 or (use_tf and (h_obs is None or h_obs.size(1) != K * S)):
        raise ValueError(f"ctx_state_losses: z {tuple(z_ctx.shape)} h_obs {None if h_obs is None else tuple(h_obs.shape)} "
                         f"tgt {tuple(tgt.shape)} C={C} K={K} tf_weight={tf_weight}")
    R = max(1, min(int(K), int(rollout_steps)))
    idx = block_indices(B, C + K, S, z_ctx.device)
    p_tf, p_ar = predictor(z_ctx, ar={"h_obs": h_obs if use_tf else None, "idx": idx,
                                      "n_ctx_blocks": C, "rollout_steps": R})
    sloss = l1(p_ar, tgt[:, : R * S], loss_exp)
    if not use_tf:
        return sloss, sloss.new_zeros(()), sloss, R
    jloss = l1(p_tf, tgt, loss_exp)
    return float(tf_weight) * jloss + sloss, jloss, sloss, R


@torch.no_grad()
def run_val_ctx_state(vs: ValSet, ctx_enc, tgt_enc, pred_mod, device, autocast_dtype, rank: int, ws: int,
                      tubelet: int, spatial: int, context_length: int, batch_size: int, loss_exp: float,
                      ctx_target: str = "window") -> Dict:
    """ctx_state in-loop val. 문맥 z -> 상태 rollout K 블록 (주지표) / TF (보조).
    타깃 = 창 전체 LN(h) 의 미래 블록 (ctx_target=window, 기본) 또는 블록별 LN(h) (block — TF 관측과 같은 공간)."""
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
            z = ctx_enc(xe, masks=[idx[:, : C * S]])                                   # 창 단위 문맥 z
            h_obs = encode_blocks(tgt_enc, xe[:, :, C * tubelet:], tubelet)             # 블록별 LN(h) (TF 관측)
            # ctx_target=block (2026-10-16 vast ctxt_video) 이면 창 전체 forward 를 생략하고 TF 관측을 그대로 타깃으로 쓴다
            tgt = h_obs if ctx_target == "block" else F.layer_norm(tgt_enc(xe), (z.size(-1),))[:, C * S:]
            h_tf = pred_mod.forward_state(z, pred_mod.embed_obs(h_obs[:, :-S]), idx[:, : (nb - 1) * S], C)
            p_tf = pred_mod.render(h_tf)
            roll = pred_mod.ctx_rollout_state_kv if getattr(pred_mod, "kv_rollout", True) else pred_mod.ctx_rollout_state
            p_ar = pred_mod.render(roll(z, idx[:, : C * S], K))
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


# ----------------------------------------------------------------------------- Push-T val (2026-10-09)
@torch.no_grad()
def run_val_pusht(vs, tgt_enc, pred_mod, device, autocast_dtype, rank: int, ws: int, tubelet: int, spatial: int,
                  C: int, batch_size: int, loss_exp: float) -> Dict:
    """val 창 (고정 시작점) 에서 L1: tf (GT 되먹임 한 스텝 앞) · ar (문맥 C 에서 K 스텝 rollout, 스텝별) ·
    기준선 copy (이전 블록 그대로) · copy_last (문맥 마지막 블록을 미래 전부에) · zero. 모두 블록별 LN(h) 공간.
    planning 성공률은 여기서 안 잰다 (시뮬레이터 필요, z_research/action_planning/plan_pusht.py)."""
    from app.vjepa_frozen.data import normalize_clips
    pred_mod.eval()
    S = spatial
    ids = list(range(rank, len(vs), ws))
    nb = vs.nb
    K = nb - C
    sums = torch.zeros(5 + K, device=device, dtype=torch.float64)      # tf, ar, copy, copy_last, zero, ar_step_1..K
    cnt = torch.zeros((), device=device, dtype=torch.float64)
    ac = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else torch.autocast("cuda", enabled=False)
    t0 = time.time()
    for s in range(0, len(ids), batch_size):
        chunk = [vs[i] for i in ids[s: s + batch_size]]
        clips = normalize_clips(torch.stack([c["clips"] for c in chunk]).to(device, non_blocking=True))
        acts = torch.stack([c["actions"] for c in chunk]).to(device)
        n = clips.size(0)
        with ac:
            h = encode_blocks(tgt_enc, clips.to(next(tgt_enc.parameters()).dtype), tubelet)       # (n, nb*S, D)
            idx = block_indices(n, nb, S, device)
            p_tf = pred_mod.forward_seq(h[:, :-S], idx[:, :-S], acts)[:, (C - 1) * S:]           # 블록 C..nb-1
            p_ar = pred_mod.rollout(h[:, : C * S], idx[:, : C * S], K, actions=acts)
        hb = h.float().view(n, nb, S, -1)
        tgt = hb[:, C:]
        def l1(x, y):
            return (x.float() - y.float()).abs().pow(loss_exp).mean(dim=(-1, -2)) / loss_exp     # (n, K)
        e_tf = l1(p_tf.view(n, K, S, -1), tgt)
        e_ar = l1(p_ar.view(n, K, S, -1), tgt)
        e_copy = l1(hb[:, C - 1: nb - 1], tgt)
        e_last = l1(hb[:, C - 1: C].expand(-1, K, -1, -1), tgt)
        e_zero = (tgt.abs().pow(loss_exp).mean(dim=(-1, -2)) / loss_exp)
        sums[0] += e_tf.sum(); sums[1] += e_ar.sum(); sums[2] += e_copy.sum(); sums[3] += e_last.sum(); sums[4] += e_zero.sum()
        sums[5:] += e_ar.sum(0)
        cnt += n * K
    if dist.is_available() and dist.is_initialized() and ws > 1:
        dist.all_reduce(sums); dist.all_reduce(cnt)
    pred_mod.train()
    if rank != 0:
        return {}
    m = (sums / cnt.clamp(min=1)).tolist()
    nwin = int(cnt.item() // K)
    return {"tf": m[0], "ar": m[1], "copy": m[2], "copy_last": m[3], "zero": m[4],
            "ar_by_step": [sums[5 + k].item() / max(nwin, 1) for k in range(K)],
            "n_windows": nwin, "C": C, "K": K, "elapsed_s": round(time.time() - t0, 1)}


def format_val_pusht(res: Dict) -> str:
    if not res:
        return ""
    steps = " ".join(f"{x:.3f}" for x in res["ar_by_step"])
    return (f"val[pusht C{res['C']}K{res['K']} n={res['n_windows']}] tf {res['tf']:.4f} ar {res['ar']:.4f} | "
            f"copy {res['copy']:.4f} copy_last {res['copy_last']:.4f} zero {res['zero']:.4f} | ar by step {steps} | {res['elapsed_s']}s")
