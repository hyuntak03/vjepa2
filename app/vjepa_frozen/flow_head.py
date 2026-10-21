"""궤적 (trajectory) 보조 head — AR predictor 출력에 보조 손실 (2026-10-22, 사용자 결정 "traj 로").

왜 flow 인가: phys2d 의 상태 라벨 (위치 · 속도) 은 **관측만으로 얻는 정보가 아니다** (CLAUDE.md §12 Q6) —
  시뮬레이터가 알려 준 것이다. 영상 두 장에서 뽑은 운동장은 관측만으로 얻어지므로
  "action 없이 관측만으로" 주장을 깨지 않는다. 라벨은 RAFT 의 **추정치**지 진실이 아니다 —
  전후 일관성 검사를 통과한 칸에만 손실을 건다.

왜 순간 flow 가 아니라 궤적인가 [측정 2026-10-22, k400 395 편 · ssv2 130 편, 자세한 수치는 flow_labels.py]:
  순간 flow 는 0.5 s 를 넘으면 사실상 예측 불가 (cos 0.04) 이고 부호가 대칭이라 L1 의 최적 상수가 0 이다 —
  거기에 L1 을 걸면 head 는 "0 을 내라" 만 배운다. 누적 변위는 방향이 오래 남고 (cos 2 s 에서 0.39),
  크기는 등속에서 꺾인다 (등속 외삽이 4 s 에서 2.5 배 과하다). 배울 게 남아 있는 쪽은 궤적이다.

무엇을: predictor 출력 p 를 블록마다 (S = 256 토큰, D) 로 보고, **토큰 하나 → 지평선 H 개의 변위 (u, v)**.
  토큰 격자 16×16 과 라벨 격자 16×16 이 1 : 1 (pooling · attention 없음).
  앵커는 **그 블록 자신** — "이 토큰이 가리키는 내용이 h 장 뒤에 어디로 가 있나".
  head 는 LayerNorm + Linear(D → 2H) 뿐 (기본 10,244 파라미터) — 일부러 작게 뒀다.
  큰 head 를 쓰면 "predictor 토큰이 운동을 담는가" 가 아니라 "head 가 뽑아낼 수 있는가" 를 재게 된다.

단위: **패치 폭** (patch = 16 px at 256). 1 초에 화면을 가로지르는 물체면 h = 4 장 (1 s) 에 16.
  실측 평균 |d| (k400): 0.5 s 1.06 · 1 s 1.53 · 2 s 2.00 · 4 s 2.22 패치.

시간축: TF 출력 위치 b 는 블록 b+1 을 예측하므로 라벨 블록 b+1 과, rollout 걸음 r 은 블록 C+r 과 맞춘다.

⚠️ 이 보조 손실이 **가르치지 않는 것** (결론과 같은 비중으로 적는다):
  * 영속 — 가려진 물체에는 flow 가 없다. 유효 마스크가 그 자리를 지운다.
  * 카메라 — K400 · SSv2 의 운동장은 카메라 운동이 크게 섞여 있다. 궤적이 맞았다고 물체 물리를 배운 게 아니다.
  * 라벨 품질 — fps 4 (0.25 s) 는 RAFT 가 학습된 간격보다 훨씬 크다. 틀린 자리는 지워질 뿐 0 으로 채워지지 않는다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrajHead(nn.Module):
    """p (B, T*S, D) → (B, T, H, 2, g, g). hidden > 0 이면 D → hidden → 2H 작은 MLP."""

    def __init__(self, dim: int = 1280, n_h: int = 4, hidden: int = 0):
        super().__init__()
        self.n_h = int(n_h)
        self.norm = nn.LayerNorm(dim)
        out = 2 * self.n_h
        self.out = nn.Linear(dim, out) if hidden <= 0 else nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, p: torch.Tensor, S: int) -> torch.Tensor:
        B, N, D = p.shape
        T, g = N // S, int(round(S ** 0.5))
        if g * g != S:
            raise ValueError(f"flow_head: 토큰 수 {S} 가 정사각 격자가 아니다")
        y = self.out(self.norm(p.reshape(B * T, S, D).float()))                # (B*T, S, 2H)
        y = y.reshape(B, T, g, g, self.n_h, 2)
        return y.permute(0, 1, 4, 5, 2, 3).contiguous()                        # (B, T, H, 2, g, g)


def build_head(FH: dict, dim: int) -> nn.Module:
    return TrajHead(dim, n_h=len(FH.get("horizons", [1, 2, 4, 8])), hidden=int(FH.get("hidden", 0)))


def _l1_per_h(pred, tgt, ok):
    """pred · tgt (B, T, H, 2, g, g) · ok (B, T, H, g, g) → (H,) 유효 가중 L1, (H,) 유효 질량."""
    w = ok.unsqueeze(3).to(pred.dtype)
    num = ((pred - tgt.to(pred.dtype)).abs() * w).sum(dim=(0, 1, 4, 5)).sum(-1)   # (H,)
    den = w.sum(dim=(0, 1, 3, 4, 5)) * 2.0
    return num / den.clamp_min(1e-6), den


def ar_flow_loss(head, p_tf, p_ar, flow, valid, S: int, C: int, R: int, split: dict,
                 horizons=(1, 2, 4, 8), norm="auto", norm_eps: float = 1e-3, d_order: int = 0):
    """TF 출력 (블록 1..nb-1) 과 rollout 출력 (블록 C..C+R-1) 둘 다에 같은 head.
    flow (B, nb, H, 2, g, g) 패치 단위 누적 변위 · valid (B, nb, H, g, g).

    지평선마다 크기가 다르므로 (0.5 s 1.06 → 4 s 2.22 패치) **그 batch 의 평균 |d| 로 나눠** 더한다
    (norm="auto", gradient 없음). 고정값을 쓰려면 norm 에 숫자 목록을 준다.
    손실 1.0 = "0 을 예측한 만큼 틀렸다" 가 되므로, 값이 1 보다 작아야 무언가를 배운 것이다.

    로그에 파라미터 0 기준선 둘을 같이 낸다 (CLAUDE.md §12 Q5):
      z<h>  0 예측 (= 평균 |d^h| 를 자기 자신으로 나눈 1.0 이므로 생략하고, 대신 아래 둘을 본다)
      c<h>  등속 외삽 — 첫 지평선의 **진짜** 변위를 비례로 늘린 것. 이걸 못 이기면 배운 게 없다.
      l<h>  head 의 상대 L1. l < min(1, c) 여야 의미가 있다.
    """
    H = [int(h) for h in horizons]
    T_tf = p_tf.size(1) // S
    y_tf, y_ar = head(p_tf, S), head(p_ar, S)
    t_tf, v_tf = flow[:, 1:1 + T_tf], valid[:, 1:1 + T_tf]
    t_ar, v_ar = flow[:, C:C + R], valid[:, C:C + R]

    with torch.no_grad():
        if norm == "auto":
            w = v_ar.unsqueeze(3)
            sc = ((t_ar.abs() * w).sum(dim=(0, 1, 4, 5)).sum(-1) / (w.sum(dim=(0, 1, 3, 4, 5)) * 2).clamp_min(1e-6))
            sc = sc.clamp_min(norm_eps)                                        # (H,)
        else:
            sc = torch.tensor([float(x) for x in norm], device=flow.device, dtype=torch.float32)

    l_tf, _ = _l1_per_h(y_tf, t_tf, v_tf)
    l_ar, _ = _l1_per_h(y_ar, t_ar, v_ar)
    loss = (float(split.get("tf", 1.0)) * (l_tf / sc).mean()
            + float(split.get("ar", 1.0)) * (l_ar / sc).mean())

    log = {"valid": float(v_ar.mean())}
    with torch.no_grad():
        cv = torch.stack([t_ar[:, :, 0] * (h / H[0]) for h in H], 2)           # 등속 외삽 (진짜 첫 지평선 기준)
        l_cv, _ = _l1_per_h(cv, t_ar, v_ar)
        for k, h in enumerate(H):
            log[f"l{h}"] = float(l_ar[k] / sc[k])                              # head (상대)
            log[f"c{h}"] = float(l_cv[k] / sc[k])                              # 등속 (상대)
        log["tf_l"] = float((l_tf / sc).mean())
        log["ar_l"] = float((l_ar / sc).mean())
        log["ar_c"] = float((l_cv / sc).mean())
    return loss, log
