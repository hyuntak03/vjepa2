"""형제 slot 맞바꾸기 손실 (2026-11-07, 사용자 "appearance agnostic 하게 배우고 싶은 거야").

왜 [측정, z_analysis/results/slot_diag_20261107]: 맞추기 판에서 시험 때 slot 을 끄거나 맞바꿔도 시트 점수가 그대로다.
  맞바꾸기 효과 = A · B 예측 차이의 0.8–2.8 %. 몸통이 slot 의 클립별 내용을 안 읽는다. L1 은 slot 을 무시한 해도 같은 값이기 때문이다.

손실 (kind=ar, 배치 앞 절반 = 클립 X, 뒤 절반 = 같은 묶음의 형제 Y):
  L = TF L1 + 굴린 L1(p(문맥 X, slot X), 미래 X) + λ · 굴린 L1(p(문맥 X, slot Y), 미래 Y)
  둘째 항과 셋째 항을 같이 맞히려면 같은 문맥에서 slot 에 따라 미래가 달라져야 한다 → slot 을 무시한 해가 최적에서 빠진다.
  "미래 Y 를 맞혀라" 로 둔 이유: "자기 미래에서 멀어져라" (hinge) 는 slot 과 문맥이 어긋난 것을 감지해 엉터리를 내도 풀린다.
외형: 형제는 배경 · 물체 외형이 같다. 그래서 slot 이 외형을 담아도 셋째 항에 도움이 안 되고, 움직임 · 존재만 도움이 된다.
  (아무 클립과 맞바꾸면 slot 이 배경 외형을 담아 풀 수 있어 목표와 반대다.)
빈 장면 문맥 + 물체 있는 slot 줄은 뺀다 — 그 줄을 맞히려면 slot 이 물체 외형까지 담아야 한다.
  남는 줄: 물체 문맥 + 빈 장면 slot (없어진 미래) · 물체 문맥 + 다른 속도 slot. 외형은 문맥에서, 상태는 slot 에서 온다.
slot 궤적은 형제 자신의 문맥 · 굴린 예측으로만 만들어져서 미래 정답이 새지 않는다. 시험 때는 자기 slot 만 쓴다 (구조 변화 없음).
"""
from __future__ import annotations

import torch

from app.vjepa_frozen import ar as AR


def _row(a, b, loss_exp):
    return (a.float() - b.float()).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp


def ar_swap_loss(predictor, h, S: int, C: int, K: int, loss_exp: float, rollout_steps: int, codes, weight: float = 1.0):
    """h (2B, nb*S, D) 앞 B = X, 뒤 B = 형제 Y. codes (B,) 1 = X 물체 · Y 빈 장면, 2 = X 빈 장면 · Y 물체, 3 = 둘 다 물체, 0 = 형제 없음.
    return (loss, jl, sl, R, stats)."""
    B2, N = h.shape[:2]
    nb = N // S
    B = B2 // 2
    if B2 % 2 or nb != C + K or codes.numel() != B:
        raise ValueError(f"slot_swap: 배치 {B2} · 블록 {nb} (C+K={C + K}) · 코드 {codes.numel()}")
    R = max(1, min(int(K), int(rollout_steps)))
    dev = h.device
    perm = torch.cat([torch.arange(B, B2, device=dev), torch.arange(0, B, device=dev)])
    out = predictor(h, ar={"idx": AR.block_indices(B2, nb, S, dev), "n_ctx_blocks": C, "rollout_steps": R, "slot_perm": perm})
    if len(out) != 3:
        raise ValueError("slot_swap: predictor 가 (p_tf, p_ar, p_sw) 를 안 돌려줬다 (slot_n 이 켜져 있나)")
    p_tf, p_ar, p_sw = out
    tgt = h[:, C * S: (C + R) * S]
    sib = tgt[perm]
    jl = AR.l1(p_tf, h[:, S:], loss_exp)
    sl = AR.l1(p_ar, tgt, loss_exp)
    v = torch.cat([(codes == 1) | (codes == 3), (codes == 2) | (codes == 3)])       # 문맥에 물체가 있는 줄만
    d_sw = _row(p_sw, sib, loss_exp)
    lsw = d_sw[v].mean() if bool(v.any()) else p_sw.sum() * 0.0
    loss = jl + sl + float(weight) * lsw
    with torch.no_grad():
        n = v.sum().clamp_min(1).double()
        st = torch.stack([v.sum().double(), d_sw[v].sum().double(),
                          _row(p_ar, sib, loss_exp)[v].sum().double(),              # slot 을 무시하면 나올 값 (대조)
                          _row(tgt, sib, loss_exp)[v].sum().double(),               # 두 미래 사이 거리
                          _row(p_sw, p_ar, 1.0)[v].sum().double(),                  # 맞바꾸기 효과
                          _row(p_ar, p_ar[perm], 1.0)[v].sum().double()])           # A · B 예측 차이
    return loss, jl.detach(), sl.detach(), R, st


def summarize(st) -> dict:
    n = max(float(st[0]), 1.0)
    return {"n": float(st[0]), "sw": float(st[1]) / n, "ign": float(st[2]) / n, "fut": float(st[3]) / n, "eff": float(st[4]) / n, "dab": float(st[5]) / n,
            "ratio": float(st[4]) / max(float(st[5]), 1e-9)}
