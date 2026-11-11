"""짝 순위 손실 (2026-11-07, 사용자 "메소드 하나만 적용해서 학습해봐"). 음의 손실 대신 쓴다.

왜 [측정]: 음의 손실 (불가능 미래 L1 을 margin 1 위로 미는 hinge) 은 학습 끝 (2,000 step) 까지 불가능 샘플의 100 % 에서 켜져 있다 (act 1.00).
  그래서 순서가 이미 맞는 쌍에서도 예측을 계속 민다 — 가능 미래 굴린 L1 이 L1 만 판 0.469 → 0.618 (+32 %, v4bc4f 판 로그).
  또 가까운 음성 (속도만 다른 바꿔치기 미래) 의 순위를 망가뜨렸다 (형제 순위 84 → 55 %, COUNTERFACTUAL_RANK_PROPOSAL_2026-11-05.md).
무엇: 같은 묶음의 형제 Y 를 짝으로 둔다 (배치 앞 절반 X, 뒤 절반 Y, 같은 프레임 번호 · 같은 증강). 줄 i 마다
  S_own = 굴린 L1(p_i, 미래_i),  S_sib = 굴린 L1(p_i, 미래_형제)   (블록별 정답이라 S_sib = 이어 붙인 불가능 영상의 채점 surprise)
  순위 벌점 = τ · softplus((δ − (S_sib − S_own)) / τ)  — 형제 미래가 δ 이상 더 멀면 거의 0, 순서가 틀리면 (S_sib − S_own) 만큼 선형.
  손실 = TF L1 + 굴린 L1 (자기 미래) + λ · 순위 벌점 평균.
최적해: L1 만이면 칸 · 차원별 조건부 중앙값. 위치가 불확실하면 물체 칸 중앙값이 배경 → "사라짐이 싸다" 가 된다 [이론].
  순위 벌점은 순서가 이미 δ 이상 맞는 문맥에서는 기울기가 없어 L1 해를 그대로 두고, 순서가 틀린 문맥에서만 두 미래가 갈리는 좌표를 자기 미래 쪽으로 당긴다
  (|p − a| − |p − b| 의 기울기는 a · b 사이에서만 0 이 아니다 — 정답 밖으로 미는 힘이 없다).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from app.vjepa_frozen import ar as AR


def _row(a, b, loss_exp):
    return (a.float() - b.float()).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp


def ar_rank_loss(predictor, h, S: int, C: int, K: int, loss_exp: float, rollout_steps: int, codes,
                 weight: float = 1.0, tau: float = 0.005, delta: float = 0.01):
    """h (2B, nb*S, D): 앞 B = X, 뒤 B = 형제 Y. codes (B,) 1 vanish · 2 appear · 3 swap · 0 형제 없음 ·
    4 실제 영상 사건 (X = 가능 또는 흔적 가능, Y = 불가능: Y 줄은 L1 정답 · 순위에 안 쓴다 — 문맥이 X 와 같아 불가능 미래를 가르치게 된다).
    return (loss, jl, sl, R, stats)."""
    B2, N = h.shape[:2]
    nb = N // S
    B = B2 // 2
    if B2 % 2 or nb != C + K or codes.numel() != B:
        raise ValueError(f"rank_pair: 배치 {B2} · 블록 {nb} (C+K={C + K}) · 코드 {codes.numel()}")
    R = max(1, min(int(K), int(rollout_steps)))
    dev = h.device
    perm = torch.cat([torch.arange(B, B2, device=dev), torch.arange(0, B, device=dev)])
    out = predictor(h, ar={"idx": AR.block_indices(B2, nb, S, dev), "n_ctx_blocks": C, "rollout_steps": R})
    p_tf, p_ar = out[0], out[1]
    tgt = h[:, C * S: (C + R) * S]
    code2 = torch.cat([codes, codes])
    own_ok = torch.cat([torch.ones_like(codes, dtype=torch.bool), codes != 4])
    jl = _row(p_tf, h[:, S:], loss_exp)[own_ok].mean()
    s_own, s_sib = _row(p_ar, tgt, loss_exp), _row(p_ar, tgt[perm], loss_exp)
    sl = s_own[own_ok].mean()
    m = s_sib - s_own
    v = (code2 > 0) & own_ok
    pen = float(tau) * F.softplus((float(delta) - m) / float(tau))
    lrank = pen[v].mean() if bool(v.any()) else p_ar.sum() * 0.0
    loss = jl + sl + float(weight) * lrank
    with torch.no_grad():
        st = [v.sum().double(), m[v].sum().double(), (m[v] > 0).sum().double(), (m[v] < float(delta)).sum().double(), pen[v].sum().double()]
        for c in (1, 2, 3, 4):
            mc = (code2 == c) & own_ok
            st += [mc.sum().double(), (m[mc] > 0).sum().double()]
    return loss, jl.detach(), sl.detach(), R, torch.stack(st)


def summarize(st) -> dict:
    n = max(float(st[0]), 1.0)
    out = {"n": float(st[0]), "margin": float(st[1]) / n, "order_ok": float(st[2]) / n, "active": float(st[3]) / n, "pen": float(st[4]) / n}
    for i, nm in enumerate(("v", "a", "s", "r")):
        out[f"ok_{nm}"] = float(st[6 + 2 * i]) / max(float(st[5 + 2 * i]), 1.0)
    return out
