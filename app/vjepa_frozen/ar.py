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


def hint_scale(progress: float, cfg: Dict) -> float:
    """힌트 일정 (2026-10-11, 사용자 결정) — 확률이 아니라 **정해진 값**. progress = step / 전체 step ∈ [0, 1].
    start 이전·end 이후 0, 그 사이 Hann 종 모양 (정점 = (start+end)/2 에서 1). 모든 rank 가 같은 값."""
    a, b = float(cfg.get("start", 0.05)), float(cfg.get("end", 0.90))
    shape = str(cfg.get("shape", "sin2"))
    if shape == "const":            # 2026-10-13: [start, end) 동안 ratio_max 고정, 그 밖 0
        return 1.0 if a <= progress < b else 0.0
    if shape != "sin2":
        raise ValueError(f"hint.shape 는 sin2 | const: {shape!r}")
    if progress <= a or progress >= b:
        return 0.0
    import math
    return math.sin(math.pi * (progress - a) / (b - a)) ** 2


def hint_count(progress: float, cfg: Dict, S: int) -> int:
    """블록당 힌트 토큰 수 = round(ratio_max × scale × S). 0 이면 cross-attn 을 부르지 않는다."""
    return int(round(float(cfg.get("ratio_max", 0.10)) * hint_scale(progress, cfg) * S))


@torch.no_grad()
def encode_hints(enc, clips: torch.Tensor, tubelet: int, S: int, n_hint: int, gen=None):
    """블록 1..nb-1 마다 무작위 patch n_hint 개를 고르고, **그 patch 만** encoder 에 넣는다 (masks=, 공간 누수 0).
    return (hx (B, (nb-1)*Nh, D) LN, hidx (B, (nb-1)*Nh) 절대 인덱스, keep (B, (nb-1)*S) bool — 힌트 자리는 False)."""
    B, Cc, T, H, W = clips.shape
    nb = T // tubelet
    m = nb - 1
    x = clips[:, :, tubelet: nb * tubelet].reshape(B, Cc, m, tubelet, H, W).permute(0, 2, 1, 3, 4, 5)
    x = x.reshape(B * m, Cc, tubelet, H, W)
    # gen: rank·step 별 생성기 (train.py). 전역 RNG 는 모든 rank 가 같은 seed 라 rank 끼리 같은 위치를 뽑는다
    loc = torch.rand(B * m, S, device=clips.device, generator=gen).argsort(dim=1)[:, :n_hint].sort(dim=1).values   # (B*m, Nh)
    hx = enc(x, masks=[loc])                                                                        # (B*m, Nh, D)
    hx = F.layer_norm(hx, (hx.size(-1),)).reshape(B, m * n_hint, hx.size(-1))
    blk = torch.arange(1, nb, device=clips.device).repeat(B).view(B * m, 1)                       # 블록 1..nb-1
    hidx = (loc + blk * S).reshape(B, m * n_hint)
    keep = torch.ones(B * m, S, dtype=torch.bool, device=clips.device)
    keep.scatter_(1, loc, False)
    return hx, hidx, keep.reshape(B, m * S)


def delta_l1(p: torch.Tensor, h: torch.Tensor, S: int, loss_exp: float):
    """차이 L1 (2026-10-13, 사용자 결정): 이웃한 두 블록 예측의 변화 (p_T − p_{T−1}) 와 실제 변화 (h_T − h_{T−1}) 의 L1.
    p, h (B, n*S, D) 같은 블록 범위 (p 의 블록 i 가 h 의 블록 i 를 예측). 블록이 2 개 미만이면 None."""
    if p.size(1) < 2 * S:
        return None
    dp = p[:, S:].float() - p[:, :-S].float()
    dh = h[:, S:].float() - h[:, :-S].float()
    return (dp - dh).abs().pow(loss_exp).mean() / loss_exp


@torch.no_grad()
def patch_embed_blocks(enc, clips: torch.Tensor, tubelet: int) -> torch.Tensor:
    """블록마다 encoder 의 patch embedding 직후 (픽셀 2 장 → 토큰, 섞이지 않은 국소 특징). return (B, nb*S, D)."""
    B, Cc, T, H, W = clips.shape
    nb = T // tubelet
    x = clips[:, :, : nb * tubelet].reshape(B, Cc, nb, tubelet, H, W).permute(0, 2, 1, 3, 4, 5).reshape(B * nb, Cc, tubelet, H, W)
    pe = enc.patch_embed(x)                                          # (B*nb, S, D)
    return pe.reshape(B, nb * pe.size(1), pe.size(-1))


def motion_target(pe: torch.Tensor, S: int) -> torch.Tensor:
    """b = ‖PE(x_{t+1}) − PE(x_t)‖₂ (토큰별) / batch 평균. (B, nb*S, D) → (B, (nb-1)*S). gradient 없음."""
    b = (pe[:, S:].float() - pe[:, :-S].float()).norm(dim=-1)
    return b / b.mean().clamp_min(1e-6)


def ar_losses_dyn(predictor, h: torch.Tensor, pe: torch.Tensor, S: int, C: int, K: int, loss_exp: float, rollout_steps: int,
                  dyn_weight: float, aux: dict = None):
    """기존 ar_losses (TF + rollout L1) + 움직임 크기 손실 (2026-10-14, 사용자 결정).
    a = Linear(|s_{t+1} − s_t|) (predictor 상태, 이웃 블록) vs b = ‖ΔPE‖ / 평균 — teacher forcing 의 이웃 위치 쌍 + rollout 의 이웃 걸음 쌍,
    모든 쌍의 평균 L1. TF 위치 b 의 상태 = 블록 b+1 예측이므로 TF 쌍 (b, b+1) ↔ 블록 (b+1, b+2), rollout 쌍 (r, r+1) ↔ 블록 (C+r, C+r+1)."""
    B, N = h.shape[:2]
    nb = N // S
    R = max(1, min(int(K), int(rollout_steps)))
    idx = block_indices(B, nb, S, h.device)
    p_tf, p_ar, a_tf, a_ar = predictor(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": R, "dyn": True})
    jloss = l1(p_tf, h[:, S:], loss_exp)
    sloss = l1(p_ar, h[:, C * S: (C + R) * S], loss_exp)
    b = motion_target(pe, S)                                         # (B, (nb-1)*S): 쌍 (블록 t, t+1), t = 0..nb-2
    terms = [(a_tf.float() - b[:, S:]).abs()]                        # TF 쌍 ↔ 블록 (1,2) .. (nb-2, nb-1) = b 의 t = 1..nb-2
    if a_ar is not None:
        terms.append((a_ar.float() - b[:, C * S: (C + R - 1) * S]).abs())   # rollout 쌍 ↔ 블록 (C, C+1) .. (C+R-2, C+R-1)
    dloss = torch.cat([t.flatten() for t in terms]).mean()
    if aux is not None:
        aux["dyn"] = float(dloss.detach())
    return jloss + sloss + float(dyn_weight) * dloss, jloss, sloss, R


def ar_losses_multistart(predictor, h: torch.Tensor, S: int, starts, rollouts, loss_exp: float):
    """여러 시작점 (2026-10-14). jloss = teacher forcing 전 위치 (기존과 같다), sloss = 시작점별 rollout L1 의 평균.
    return (loss, jloss, sloss, R 최대)."""
    B, N = h.shape[:2]
    nb = N // S
    idx = block_indices(B, nb, S, h.device)
    p_tf, p_list = predictor(h, ar={"idx": idx, "starts": list(starts), "rollouts": list(rollouts)})
    jloss = l1(p_tf, h[:, S:], loss_exp)
    sl = [l1(p, h[:, c * S: (c + R) * S], loss_exp) for p, c, R in zip(p_list, starts, rollouts)]
    sloss = sum(sl) / len(sl)
    return jloss + sloss, jloss, sloss, max(rollouts)


def encode_video_ctx_full(enc, clips: torch.Tensor, C: int, S: int):
    """ar_input=video (2026-10-18, 사용자 결정). 같은 encoder 로 **두 번**: 문맥 pass (문맥 토큰만 넣는다 = 미래를 안 본다) ·
    전체 pass (클립 전체 = 창 정답). 둘 다 affine 없는 LN. return zc (B, C*S, D), zf (B, nb*S, D)."""
    B = clips.size(0)
    zc = enc(clips, masks=[block_indices(B, C, S, clips.device)])
    zf = enc(clips)
    return F.layer_norm(zc, (zc.size(-1),)), F.layer_norm(zf, (zf.size(-1),))


def ar_losses_video(predictor, zc: torch.Tensor, zf: torch.Tensor, S: int, C: int, K: int, loss_exp: float,
                    rollout_steps: int, aux: dict = None):
    """ar_input=video · ar_target=window (2026-10-18). 입력 = 문맥 pass 영상 토큰 zc, 정답 = 전체 pass 영상 토큰 zf.
    TF 는 **문맥 자리만**: 위치 b (b = 0..C-2, block-causal) → zf 의 블록 b+1. zc[b] 는 문맥 끝까지 본 영상 토큰이라
      약간의 미래 누수가 있다 — 의도된 것 (사용자). 예측 구간은 TF 없이 상태 rollout R 걸음 → zf 의 블록 C..C+R-1.
    return (loss, jloss, sloss, R)."""
    B = zc.size(0)
    if C < 2:
        raise ValueError("ar_input=video 는 문맥 2 블록 이상 (TF 자리가 하나라도 있어야 한다)")
    if zc.size(1) != C * S or zf.size(1) != (C + K) * S:
        raise ValueError(f"ar_losses_video: zc {tuple(zc.shape)} zf {tuple(zf.shape)} C={C} K={K}")
    R = max(1, min(int(K), int(rollout_steps)))
    p_tf, p_ar = predictor(zc, ar={"idx": block_indices(B, C, S, zc.device), "n_ctx_blocks": C, "rollout_steps": R})
    jloss = l1(p_tf, zf[:, S: C * S], loss_exp)
    sloss = l1(p_ar, zf[:, C * S: (C + R) * S], loss_exp)
    return jloss + sloss, jloss, sloss, R


def ar_losses(predictor, h: torch.Tensor, S: int, C: int, K: int, loss_exp: float, rollout_steps: int, hint=None,
              delta_weight: float = 0.0, aux: dict = None, tf_span: str = "all", out: dict = None, h_in: torch.Tensor = None):
    """h (B, nb*S, D) LN 된 블록 토큰. return (loss, jloss, sloss, R). predictor 는 DDP 래퍼여도 된다 (forward 한 번).
    hint = encode_hints 의 (hx, hidx, keep) 또는 None. 주면 TF 에만 힌트를 넣고 jloss 에서 힌트 자리를 뺀다 (rollout 은 힌트 없음).
    tf_span (2026-10-15, 사용자 결정): "all" = TF 손실을 위치 0..nb-2 전부 (기존), "predict" = 예측 구간만 —
      위치 C-1..nb-2 (블록 C..nb-1 예측). 문맥 안쪽 블록을 짧은 과거로 맞히는 자리를 손실에서 뺀다. forward 는 같다 (키로는 다 쓴다)."""
    B, N = h.shape[:2]
    nb = N // S
    if nb != C + K or nb < 2:
        raise ValueError(f"ar_losses: nb={nb} 가 C+K={C + K} 와 다르거나 2 미만")
    if tf_span not in ("all", "predict"):
        raise ValueError(f"tf_span 은 all | predict: {tf_span!r}")
    if tf_span == "predict" and (hint is not None or delta_weight):
        raise ValueError("tf_span=predict 는 hint · delta_weight 와 아직 같이 쓰지 않는다")
    R = max(1, min(int(K), int(rollout_steps)))
    idx = block_indices(B, nb, S, h.device)
    arg = {"idx": idx, "n_ctx_blocks": C, "rollout_steps": R}
    if hint is not None:
        hx, hidx, keep = hint
        arg["hint"] = {"x": hx, "idx": hidx}
    if h_in is not None and (hint is not None or delta_weight or tf_span != "all"):
        raise ValueError("h_in (encoder LoRA 입력) 은 hint · delta_weight · tf_span 과 아직 같이 쓰지 않는다")
    p_tf, p_ar = predictor(h if h_in is None else h_in, ar=arg)     # 2026-10-25: h_in = LoRA 입력 경로, 타깃은 늘 h (얼린 encoder)
    if out is not None:                                       # 2026-10-19: state_head 보조 손실이 예측 출력을 쓴다 (TF 블록 1..nb-1 · rollout C..C+R-1)
        out["p_tf"], out["p_ar"] = p_tf, p_ar
    core = predictor_core(predictor)
    if aux is not None and getattr(core, "gating_residual", False) and core._gate_roll is not None:   # 2026-10-17 ar gating 모니터
        g, A = core._gate_roll
        aux.update(gate_stats(g, A, h[:, C * S: (C + R) * S], S))
    if hint is not None:
        d = (p_tf.float() - h[:, S:].float()).abs().pow(loss_exp).mean(dim=-1) / loss_exp   # (B, (nb-1)*S)
        jloss = (d * keep).sum() / keep.sum()
    elif tf_span == "predict":
        jloss = l1(p_tf[:, (C - 1) * S:], h[:, C * S:], loss_exp)          # p_tf 위치 b → 블록 b+1
    else:
        jloss = l1(p_tf, h[:, S:], loss_exp)
    sloss = l1(p_ar, h[:, C * S: (C + R) * S], loss_exp)
    loss = jloss + sloss
    if delta_weight:
        # teacher forcing: p_tf 블록 1..nb-1 의 이웃 예측끼리 / rollout: p_ar 블록 C..C+R-1 의 이웃 걸음끼리
        terms = [d for d in (delta_l1(p_tf, h[:, S:], S, loss_exp),
                             delta_l1(p_ar, h[:, C * S: (C + R) * S], S, loss_exp)) if d is not None]
        if terms:
            dloss = sum(terms)
            loss = loss + float(delta_weight) * dloss
            if aux is not None:
                aux["dloss"] = float(dloss.detach())
    return loss, jloss, sloss, R


@torch.no_grad()
def run_val_ar(vs: ValSet, tgt_enc, pred_mod, device, autocast_dtype, rank: int, ws: int,
               tubelet: int, spatial: int, context_length: int, batch_size: int, loss_exp: float,
               ar_input: str = "block", in_enc=None) -> Dict:
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
            idx = block_indices(n, nb, S, device)
            if ar_input == "video":                          # 2026-10-18: 입력 = 문맥 pass, 정답 = 전체 pass (창). 보조 tf = 문맥 자리 TF
                zc, h = encode_video_ctx_full(tgt_enc, x.to(next(tgt_enc.parameters()).dtype), C, S)
                p_ar = pred_mod.rollout_kv(zc, idx[:, : C * S], K)
                p_tf = pred_mod.forward_seq(zc[:, :-S], idx[:, : (C - 1) * S])
                tgt_tf = h[:, S: C * S].float()
            else:
                h = encode_blocks(tgt_enc, x.to(next(tgt_enc.parameters()).dtype), tubelet)     # (n, nb*S, D)
                hi = h if in_enc is None else encode_blocks(in_enc, x.to(next(tgt_enc.parameters()).dtype), tubelet)   # 2026-10-25: LoRA 입력
                p_ar = pred_mod.rollout_kv(hi[:, : C * S], idx[:, : C * S], K)                   # (n, K*S, D) KV 캐시판 (값 동일)
                p_tf = pred_mod.forward_seq(hi[:, :-S], idx[:, :-S])[:, (C - 1) * S:]              # 블록 C..nb-1 예측
                tgt_tf = None
        tgt = h[:, C * S:].float()
        tgt_tf = tgt if tgt_tf is None else tgt_tf
        d_ro = (p_ar.float() - tgt).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
        d_tf = (p_tf.float() - tgt_tf).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
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
def _hole_ratio(B, C, S, g, k, n_tok, cfg, gen):
    """문맥 끝 k 블록에 같은 자리 (tube) 로 정확히 n_tok 토큰을 가린다 (clip 마다 다른 모양, batch 안 길이 같음).
    직사각형 (면적 비율 rect_scale, 가로세로 aspect) 을 n_tok 이상 덮을 때까지 쌓고, 넘친 토큰은 무작위로 다시 보이게 한다."""
    lo, hi = cfg.get("aspect", [0.75, 1.5]); sl, sh = cfg.get("rect_scale", [0.15, 0.4])
    hole = torch.zeros(B, C, S, dtype=torch.bool)
    for i in range(B):
        sp = torch.zeros(g, g, dtype=torch.bool)
        while int(sp.sum()) < n_tok:
            ar_ = float(torch.empty(1).uniform_(float(lo), float(hi), generator=gen))
            area = float(torch.empty(1).uniform_(float(sl), float(sh), generator=gen)) * g * g
            hh = max(1, min(g, int(round((area / ar_) ** 0.5)))); ww = max(1, min(g, int(round((area * ar_) ** 0.5))))
            t = int(torch.randint(0, g - hh + 1, (1,), generator=gen)); l_ = int(torch.randint(0, g - ww + 1, (1,), generator=gen))
            sp[t:t + hh, l_:l_ + ww] = True
        f = sp.view(S)
        extra = int(f.sum()) - n_tok
        if extra > 0:
            on = f.nonzero().flatten()
            f[on[torch.randperm(on.numel(), generator=gen)[:extra]]] = False
        hole[i, C - k:] = f.view(1, S)
    hole = hole.view(B, C * S)
    keep = torch.stack([(~hole[i]).nonzero().flatten() for i in range(B)])
    return hole, keep


def make_ctx_hole(B: int, C: int, S: int, cfg: dict, gen: torch.Generator, gen_shape: torch.Generator = None):
    """문맥 구멍 (2026-10-16, 사용자 결정 — "문맥에 구멍을 내서 채우게", encoding **전에** 낸다 = V-JEPA 2 사전학습과 같다).
    return (hole (B, C*S) bool, keep (B, n_keep) long) 또는 (None, None). keep = context encoder 가 볼 토큰 (오름차순).
    cfg = {p, k_max, num_blocks, spatial_scale, aspect}. 기본 = V-JEPA 2 긴 범위 마스크 (70 % 직사각형 2 개, 가로세로 0.75–1.5).
      * **step 단위**로 확률 p 적용 (batch 안 clip 전부). 구멍 블록 수 k ~ U{1..min(k_max, C-1)} 도 step 공통
        (보이는 문맥 블록을 최소 하나 남긴다 — ZWM 의 대칭 마스킹 ablation 교훈. C < 2 면 구멍 없음)
      * 구멍 = **문맥 끝** 블록 C-k..C-1 (AR 이 마지막 1–2 블록만 보는 것 · late 가림 위치). 공간은 clip 마다 다른 직사각형,
        k 블록 내내 같은 자리 (**tube** — 블록마다 다르면 옆 블록 같은 자리에서 베낀다)
      * encoder 입력 길이를 batch 안에서 맞추려고 (V-JEPA 2 collator 와 같다) 보이는 토큰 수를 batch 최소로 자른다 —
        더 가리는 토큰은 **구멍 블록 안의 보이는 자리**에서 먼저 고른다 (앞쪽 문맥 블록은 되도록 그대로)"""
    kmax = min(int(cfg.get("k_max", 4)), C - 1)
    if kmax < 1 or float(torch.rand(1, generator=gen)) >= float(cfg.get("p", 0.75)):
        return None, None
    k = int(torch.randint(1, kmax + 1, (1,), generator=gen))
    ratio = cfg.get("ratio")                                  # 2026-10-17: [lo, hi] 이면 step 마다 r ~ U(lo, hi), 구멍 블록당 정확히 round(r·S) 토큰
    r_tok = None
    if ratio is not None:
        r_tok = int(round(float(torch.empty(1).uniform_(float(ratio[0]), float(ratio[1]), generator=gen)) * S))
    gen = gen if gen_shape is None else gen_shape            # 적용 여부 · k · 비율은 gen (rank 공통), 직사각형 · 길이 맞춤은 gen_shape (rank 별)
    g = int(round(S ** 0.5))
    if r_tok is not None:
        return _hole_ratio(B, C, S, g, k, r_tok, cfg, gen)
    nblk, sc = int(cfg.get("num_blocks", 2)), float(cfg.get("spatial_scale", 0.7))
    lo, hi = cfg.get("aspect", [0.75, 1.5])
    hole = torch.zeros(B, C, S, dtype=torch.bool)
    for i in range(B):
        sp = torch.zeros(g, g, dtype=torch.bool)
        for _ in range(nblk):
            ar_ = float(torch.empty(1).uniform_(float(lo), float(hi), generator=gen))
            area = sc * g * g
            hh = max(1, min(g, int(round((area / ar_) ** 0.5)))); ww = max(1, min(g, int(round((area * ar_) ** 0.5))))
            t = int(torch.randint(0, g - hh + 1, (1,), generator=gen)); l_ = int(torch.randint(0, g - ww + 1, (1,), generator=gen))
            sp[t:t + hh, l_:l_ + ww] = True
        hole[i, C - k:] = sp.view(1, S)
    hole = hole.view(B, C * S)
    n_keep = int((~hole).sum(1).min())
    for i in range(B):
        extra = int((~hole[i]).sum()) - n_keep
        if extra > 0:
            vis = (~hole[i]).nonzero().flatten()
            in_k = vis[vis >= (C - k) * S]                       # 구멍 블록 안의 보이는 자리부터
            pool = in_k if in_k.numel() >= extra else vis
            pick = pool[torch.randperm(pool.numel(), generator=gen)[:extra]]
            hole[i, pick] = True
    keep = torch.stack([(~hole[i]).nonzero().flatten() for i in range(B)])
    return hole, keep


def predictor_core(m):
    """DDP 로 감싼 predictor 의 안쪽 모듈."""
    return m.module if hasattr(m, "module") else m


@torch.no_grad()
def gate_stats(g, anchor, tgt, S: int) -> dict:
    """gating_residual 모니터 (rollout gate, 2026-10-17). g (B, R*S, 1), anchor (B, S, D), tgt (B, R*S, D).
    mov = 정답이 anchor 에서 많이 변한 칸 (|tgt − A| 상위 25 %, 움직이는 물체), sta = 하위 25 % (정지 배경).
    정상이면 mov < sta (움직이는 칸은 anchor 를 덜 쓴다), kR < k1 (뒤 걸음일수록 덜 쓴다)."""
    B, n, _ = g.shape
    R = n // S
    g = g.float().view(B, R, S)
    d = (tgt.float().view(B, R, S, -1) - anchor.float().unsqueeze(1)).abs().mean(-1)     # (B, R, S)
    lo, hi = torch.quantile(d.flatten(1), torch.tensor([0.25, 0.75], device=d.device), dim=1)  # (B,) 씩
    mov = g[d >= hi.view(B, 1, 1)].mean()
    sta = g[d <= lo.view(B, 1, 1)].mean()
    return {"gate": float(g.mean()), "gate_mov": float(mov), "gate_sta": float(sta),
            "gate_k1": float(g[:, 0].mean()), "gate_kR": float(g[:, -1].mean()), "gate_hi": float((g > 0.95).float().mean())}


def ctx_state_losses(predictor, z_ctx, h_obs, tgt, S: int, C: int, K: int, loss_exp: float, rollout_steps: int,
                     tf_weight: float = 1.0, hole=None, tgt_ctx=None,
                     recon_weight: float = 0.0, aux: dict = None, anchor=None):
    """z_ctx (B, C*S, D), h_obs (B, K*S, D) 블록별 LN(h) 또는 None (tf_weight=0), tgt (B, K*S, D) 창 LN(h) 의 미래 블록.
    loss = tf_weight * jloss + sloss. return (loss, jloss, sloss, R) — TF 를 끄면 jloss 는 0."""
    B = z_ctx.size(0)
    use_tf = float(tf_weight) > 0
    if z_ctx.size(1) != C * S or tgt.size(1) != K * S or K < 1 or (use_tf and (h_obs is None or h_obs.size(1) != K * S)):
        raise ValueError(f"ctx_state_losses: z {tuple(z_ctx.shape)} h_obs {None if h_obs is None else tuple(h_obs.shape)} "
                         f"tgt {tuple(tgt.shape)} C={C} K={K} tf_weight={tf_weight}")
    R = max(1, min(int(K), int(rollout_steps)))
    idx = block_indices(B, C + K, S, z_ctx.device)
    recon = hole is not None and float(recon_weight) > 0
    arg = {"h_obs": h_obs if use_tf else None, "idx": idx, "n_ctx_blocks": C, "rollout_steps": R}
    if anchor is not None:                                    # 2026-10-17 gating_residual: (B, S, D) LN(target_encoder(마지막 문맥 블록))
        arg["anchor"] = anchor
    if hole is not None:                                      # 2026-10-16 문맥 구멍 (z 는 그 자리에 0, 입력은 mask token)
        arg["ctx_hole"] = hole
        arg["ctx_recon"] = recon
    out = predictor(z_ctx, ar=arg)
    p_tf, p_ar = out[0], out[1]
    sloss = l1(p_ar, tgt[:, : R * S], loss_exp)
    if aux is not None and anchor is not None and getattr(predictor_core(predictor), "_gate_cache", None) is not None:
        aux.update(gate_stats(predictor_core(predictor)._gate_cache.detach(), anchor, tgt[:, : R * S], S))
    if not use_tf:
        return sloss, sloss.new_zeros(()), sloss, R
    jloss = l1(p_tf, tgt, loss_exp)
    loss = float(tf_weight) * jloss + sloss
    if recon:                                                 # 가려진 문맥 토큰만: |render(상태) − 타깃 (창 LN(h) 의 문맥 자리)|
        d = (out[2].float() - tgt_ctx.float()).abs().pow(loss_exp).mean(dim=-1) / loss_exp    # (B, C*S)
        rloss = (d * hole).sum() / hole.sum().clamp(min=1)
        loss = loss + float(recon_weight) * rloss
        if aux is not None:
            aux["rloss"] = float(rloss.detach())
    if aux is not None and hole is not None:
        aux["hole_frac"] = float(hole.float().mean())
    return loss, jloss, sloss, R


@torch.no_grad()
def run_val_ctx_state(vs: ValSet, ctx_enc, tgt_enc, pred_mod, device, autocast_dtype, rank: int, ws: int,
                      tubelet: int, spatial: int, context_length: int, batch_size: int, loss_exp: float,
                      ctx_target: str = "window", gating: bool = False, ctx_input: str = "video") -> Dict:
    """ctx_state 판 in-loop val. 문맥 z → 상태 rollout K 블록 (주지표) / TF (보조).
    타깃 = 창 전체 LN(h) 의 미래 블록 (ctx_target=window, 기본) 또는 블록별 LN(h) (block, 2026-10-16 — TF 관측과 같은 공간)."""
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
            shared = ctx_input == "block" and ctx_enc is tgt_enc and ctx_target == "block"
            if shared:                                       # 2026-10-17: 문맥도 블록별 LN(target_encoder) (학습과 같다)
                hb_all = encode_blocks(tgt_enc, xe[:, :, : nb * tubelet], tubelet)
                z = hb_all[:, : C * S]
            elif ctx_input == "block":
                z = encode_blocks(ctx_enc, xe[:, :, : C * tubelet], tubelet)
            else:
                z = ctx_enc(xe, masks=[idx[:, : C * S]])
            if shared:
                h_obs = hb_all[:, C * S:]
                A = z[:, -S:] if gating else None
            elif gating:                                     # 2026-10-17: anchor = 마지막 문맥 블록 LN(target_encoder) — 미래 블록과 한 번에
                hb = encode_blocks(tgt_enc, xe[:, :, (C - 1) * tubelet:], tubelet)
                A, h_obs = hb[:, :S], hb[:, S:]
            else:
                A, h_obs = None, encode_blocks(tgt_enc, xe[:, :, C * tubelet:], tubelet)
            tgt = h_obs if ctx_target == "block" else F.layer_norm(tgt_enc(xe), (z.size(-1),))[:, C * S:]
            h_tf = pred_mod.forward_state(z, pred_mod.embed_obs(h_obs[:, :-S]), idx[:, : (nb - 1) * S], C)
            p_tf = pred_mod.render(h_tf, A)
            p_ar = pred_mod.render(pred_mod.rollout_state_kv(z, idx[:, : C * S], K), A)
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
