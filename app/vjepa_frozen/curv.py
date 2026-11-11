"""predictor 은닉 곡률 손실 (2026-11-07, 사용자 "L1 post training 에 ... 올려 봐"). 진단 z_analysis/docs/HIDDEN_CURVATURE_2026-11-07.md.

  L = L_pred + λ(t) · L_curv,  L_curv = 1 − cos(v_t, v_{t+1}),  v_t = m_{t+1} − m_t,  m_t = 블록 t 의 층 L 은닉을 256 칸 평균 (배치 평균 제거).
  출력 · 정답 공간에는 안 건다. 경로별로 따로 재서 평균한다 (관측 경로 15 자리 · 굴린 경로 문맥 끝 자리부터 R 자리). 두 경로의 이음은 곡률에 안 넣는다
  (이음 곡률 112° 는 입력 꼴이 바뀌는 자리지 상태가 꺾이는 자리가 아니다).

config 최상위 `curv:` 가 있을 때 z_training/harness/launch_curv.py 가 rank 마다 install() 을 부른다.
  curv:
    weight: 0.1         # λ
    layer: 9            # 통과한 predictor 층 수 (1–12)
    warmup_steps: 200   # λ 를 0 에서 올리는 step 수
    paths: [tf, ro]     # tf = 관측 경로, ro = 굴린 경로
    mask_q: 0.0         # > 0 이면 이웃 속도 크기 하위 q 를 뺀다 (정지 구간 mask). 진단에서 크기로는 움직임을 못 가렸다
    probe_every: 100    # 이 간격마다 예측 손실 · 곡률 손실의 gradient 크기와 cos 을 잰다 (rank 0)

왜 train.py 를 안 고쳤나: slot 두 판이 도는 동안 app/vjepa_frozen/train.py · z_training/train.sh 를 건드리지 않으려고 AR.ar_losses 를 감싼다 (train.py 가 호출 시점에 찾는다).
kind=ar 기본 경로 (L1 post-training) 전용이다. negimp · multistart 등은 ar_losses 를 안 거치므로 install 이 막는다.
"""
from __future__ import annotations

import json
import logging
import os

import torch
import yaml

from app.vjepa_frozen import ar as AR

logger = logging.getLogger("app.vjepa_frozen.curv")


def curv_term(P: torch.Tensor, mask_q: float = 0.0):
    """P (B, T, D) → (1 − cos 평균, cos 평균 (detach)). 배치 평균을 자리마다 뺀다."""
    P = P.float()
    P = P - P.mean(0, keepdim=True)
    v = P[:, 1:] - P[:, :-1]
    a, b = v[:, :-1], v[:, 1:]
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    c = (a * b).sum(-1) / (na * nb).clamp_min(1e-6)
    if mask_q > 0:
        s = torch.minimum(na, nb).detach()
        c = c[s >= torch.quantile(s, float(mask_q))]
    return (1.0 - c).mean(), c.mean().detach()


class Capture:
    """층 layer 의 출력을 256 칸 평균해 모은다. 관측 경로 = forward hook, 굴린 경로 = forward_kv (module 호출이 아니라 hook 이 안 불린다)."""

    def __init__(self, core, layer: int):
        self.S = int(core.tokens_per_block)
        self.on, self.tf, self.kv = False, None, []
        blk = core.predictor_blocks[int(layer) - 1]
        blk.register_forward_hook(self._hook)
        orig = blk.forward_kv

        def forward_kv(*a, **k):
            out = orig(*a, **k)
            if self.on:
                self.kv.append(out[0].float().mean(1))
            return out
        blk.forward_kv = forward_kv

    def _hook(self, module, inp, out):
        if self.on:
            B, N, D = out.shape
            self.tf = out.float().view(B, N // self.S, self.S, D).mean(2)


def install(fname: str):
    """rank 프로세스 안에서 train.py import 전에 부른다. AR.ar_losses 를 곡률 항을 더하는 판으로 바꾼다."""
    cfg = yaml.safe_load(open(fname))
    CV = cfg.get("curv")
    if not CV:
        raise ValueError(f"launch_curv: {fname} 에 curv: 가 없다")
    if cfg.get("negimp") or (cfg.get("model", {}).get("predictor", {}) or {}).get("kind") != "ar":
        raise ValueError("curv 는 kind=ar 기본 경로 (negimp 없음) 전용이다")
    W, LAYER = float(CV["weight"]), int(CV.get("layer", 9))
    WARM, Q = int(CV.get("warmup_steps", 200)), float(CV.get("mask_q", 0.0))
    PATHS, PROBE = list(CV.get("paths", ["tf", "ro"])), int(CV.get("probe_every", 100))
    if not set(PATHS) <= {"tf", "ro"} or not PATHS:
        raise ValueError(f"curv.paths 는 tf · ro: {PATHS}")
    out_path = os.path.join(cfg["folder"], "curv.jsonl")
    orig = AR.ar_losses
    st = {"step": 0, "cap": None}

    def ar_losses(predictor, h, S, C, K, loss_exp, rollout_steps, *a, **kw):
        core = AR.predictor_core(predictor)
        if st["cap"] is None:
            st["cap"] = Capture(core, LAYER)
        cap = st["cap"]
        cap.on, cap.tf, cap.kv = True, None, []
        try:
            loss, jl, sl, R = orig(predictor, h, S, C, K, loss_exp, rollout_steps, *a, **kw)
        finally:
            cap.on = False
        terms, cos = {}, {}
        if "tf" in PATHS:
            terms["tf"], cos["tf"] = curv_term(cap.tf, Q)
        if "ro" in PATHS:
            if len(cap.kv) != C + R - 1:
                raise RuntimeError(f"curv: 굴린 경로 {len(cap.kv)} 자리 (기대 {C + R - 1})")
            terms["ro"], cos["ro"] = curv_term(torch.stack(cap.kv[C - 1:], 1), Q)
        lc = sum(terms.values()) / len(terms)
        step = st["step"]
        lam = W * min(1.0, (step + 1) / max(WARM, 1))
        rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if rank0 and (step % 10 == 0 or (PROBE and step % PROBE == 0)):
            rec = {"step": step, "lam": round(lam, 5), "l_pred": round(float(loss.detach()), 4), "l_curv": round(float(lc.detach()), 4),
                   **{f"cos_{k}": round(float(v), 4) for k, v in cos.items()}}
            if PROBE and step % PROBE == 0:
                ps = [q for q in core.parameters() if q.requires_grad]
                g = [torch.autograd.grad(x, ps, retain_graph=True, allow_unused=True) for x in (loss, lc)]
                gv = [torch.cat([(y if y is not None else torch.zeros_like(q)).reshape(-1).float() for y, q in zip(x, ps)]) for x in g]
                rec.update(g_pred=round(float(gv[0].norm()), 4), g_curv=round(float(gv[1].norm()), 4),
                           g_cos=round(float(torch.nn.functional.cosine_similarity(gv[0], gv[1], dim=0)), 4))
            logger.info("[curv] " + json.dumps(rec))
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        st["step"] = step + 1
        return loss + lam * lc, jl, sl, R

    AR.ar_losses = ar_losses
    logger.info(f"[curv] 켬: λ {W} · 층 {LAYER} · warmup {WARM} · 경로 {PATHS} · mask_q {Q}")
