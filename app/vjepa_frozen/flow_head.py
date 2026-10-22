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
    return TrajHead(dim, n_h=len(FH["supervise"]), hidden=int(FH.get("hidden", 0)))


def _l1_per_h(pred, tgt, ok):
    """pred · tgt (B, T, H, 2, g, g) · ok (B, T, H, g, g) → (H,) 유효 가중 L1, (H,) 유효 질량."""
    w = ok.unsqueeze(3).to(pred.dtype)
    num = ((pred - tgt.to(pred.dtype)).abs() * w).sum(dim=(0, 1, 4, 5)).sum(-1)   # (H,)
    den = w.sum(dim=(0, 1, 3, 4, 5)) * 2.0
    return num / den.clamp_min(1e-6), den


def ar_flow_loss(head, p_tf, p_ar, flow, valid, S: int, C: int, R: int, split: dict,
                 horizons=(1, 2, 4, 8), supervise=(2, 4, 8), norm="auto", norm_eps: float = 1e-3):
    """TF 출력 (블록 1..nb-1) 과 rollout 출력 (블록 C..C+R-1) 둘 다에 같은 head.
    flow (B, nb, H, 2, g, g) 패치 단위 누적 변위 · valid (B, nb, H, g, g). H = len(horizons).

    horizons   라벨로 실어 온 지평선 전부. **첫 항은 등속 기준선의 재료**다 (앵커의 순간 속도).
    supervise  그 중 실제로 손실을 거는 것. horizons 의 부분집합이고 head 출력 축과 1 : 1.
               왜 나누나: 등속 기준선 d_h = (h / h0)·d_h0 을 쓰므로 h0 자리는 기준선이 정의상 0 이 된다.
               h0 를 **감독에서 빼면** 감독하는 모든 칸이 제대로 된 파라미터 0 기준선을 갖는다
               (실측 등속/0예측: 1 블록 0.58 · 2 블록 1.13 · 4 블록 1.79 — 1 블록이 가장 이기기 어렵다).

    지평선마다 크기가 다르므로 (1 블록 0.92 → 4 블록 1.78 패치) **그 batch 의 평균 |d| 로 나눠** 더한다
    (norm="auto", gradient 없음). 손실 1.0 = "0 을 예측한 만큼 틀렸다".

    로그 (CLAUDE.md §12 Q5):  l<h> head 상대 L1 · c<h> 등속 상대 L1. **l < min(1, c)** 여야 배운 것이다.
    """
    H = [int(h) for h in horizons]
    SUP = [int(h) for h in supervise]
    si = [H.index(h) for h in SUP]
    T_tf = p_tf.size(1) // S
    y_tf, y_ar = head(p_tf, S), head(p_ar, S)
    t_tf, v_tf = flow[:, 1:1 + T_tf][:, :, si], valid[:, 1:1 + T_tf][:, :, si]
    t_ar, v_ar = flow[:, C:C + R][:, :, si], valid[:, C:C + R][:, :, si]

    with torch.no_grad():
        if norm == "auto":
            w = v_ar.unsqueeze(3)
            sc = ((t_ar.abs() * w).sum(dim=(0, 1, 4, 5)).sum(-1) / (w.sum(dim=(0, 1, 3, 4, 5)) * 2).clamp_min(1e-6))
            sc = sc.clamp_min(norm_eps)
        else:
            sc = torch.tensor([float(x) for x in norm], device=flow.device, dtype=torch.float32)

    l_tf, _ = _l1_per_h(y_tf, t_tf, v_tf)
    l_ar, _ = _l1_per_h(y_ar, t_ar, v_ar)
    loss = (float(split.get("tf", 1.0)) * (l_tf / sc).mean()
            + float(split.get("ar", 1.0)) * (l_ar / sc).mean())

    log = {"valid": float(v_ar.mean())}
    with torch.no_grad():
        base = flow[:, C:C + R][:, :, 0]                                   # 앵커의 순간 변위 (h0)
        cv = torch.stack([base * (h / H[0]) for h in SUP], 2)              # 등속 외삽
        l_cv, _ = _l1_per_h(cv, t_ar, v_ar)
        for k, h in enumerate(SUP):
            log[f"l{h}"] = float(l_ar[k] / sc[k])
            log[f"c{h}"] = float(l_cv[k] / sc[k])
        log["tf_l"] = float((l_tf / sc).mean())
        log["ar_l"] = float((l_ar / sc).mean())
        log["ar_c"] = float((l_cv / sc).mean())
    return loss, log


# ---------------------------------------------------------------- 쌍(차) readout (2026-10-22, 사용자 결정)
# 왜 바꾸나 — 토큰 하나짜리 readout 은 **본 손실과 최적해가 다르다**:
#   타깃 encoder 는 블록 b 의 두 장만 인코딩하므로 "b 가 h 장 뒤 어디 가 있나" 는 **진짜 토큰 h_b 에도 없다.**
#   그래서 latent 손실 (p_b → h_b) 과 보조 손실이 서로 당긴다. 실측으로 그게 나타났다 (2026-10-22,
#   ar_vw_shortctx_traj ep 1~22): latent 가 대조군보다 꾸준히 0.007~0.012 높고 IntPhys1 val 이 12 ep 연속 아래,
#   궤적 head 는 세 지평선 모두 0.97 (= 거의 0 예측) 에서 정지.
# 쌍의 차를 읽으면 충돌이 없다 — 블록 b → b+g 의 이동량은 (h_b, h_{b+g}) 의 **함수로 실제로 존재**하므로
#   p → h 가 되면 보조 손실도 자동으로 만족된다. **두 손실의 최적해가 같다.**
# 그리고 묻는 내용이 낫다: "예측한 **수열**이 진짜 운동을 담고 있나". latent 손실만으로는 블록마다 그럴듯하되
#   시간적으로 멈춰 있는 수열에 벌점이 거의 없다 — 그게 우리가 기록한 실패 (거리 2~3 칸에서 멈춤) 와 같은 축이다.
# ⚠️ 남는 걱정: flow 추정은 본질적으로 matching 이다. 같은 자리 토큰 둘의 **차**를 선형 사상하는 걸로
#    "여기 있던 게 저기로 갔다" 를 계산할 수 있다는 보장은 없다. hidden > 0 으로 작은 MLP 를 쓸 수 있지만
#    그러면 "predictor 가 담았나" 가 아니라 "head 가 뽑아낼 수 있나" 를 재게 된다 (CLAUDE.md §7-3).
class TrajDiffHead(nn.Module):
    """diff (B, T, S, D) = p_{b+g} - p_b  →  (B, T, 2, g, g). 지평선마다 출력 선형을 따로 둔다
    (간격마다 변위 크기가 다르다). LayerNorm 은 공유."""

    def __init__(self, dim: int = 1280, n_h: int = 3, hidden: int = 0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        mk = (lambda: nn.Linear(dim, 2)) if hidden <= 0 else \
             (lambda: nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 2)))
        self.out = nn.ModuleList([mk() for _ in range(n_h)])

    def forward(self, diff: torch.Tensor, k: int) -> torch.Tensor:
        B, T, S, D = diff.shape
        g = int(round(S ** 0.5))
        y = self.out[k](self.norm(diff.reshape(B * T, S, D).float()))
        return y.reshape(B, T, g, g, 2).permute(0, 1, 4, 2, 3).contiguous()


def build_diff_head(FH: dict, dim: int) -> nn.Module:
    return TrajDiffHead(dim, n_h=len(FH["supervise"]), hidden=int(FH.get("hidden", 0)))


def _l1_one(pred, tgt, ok):
    """(B, T, 2, g, g) · ok (B, T, g, g) → 스칼라 L1, 유효 질량."""
    w = ok.unsqueeze(2).to(pred.dtype)
    den = w.sum() * 2.0
    return ((pred - tgt.to(pred.dtype)).abs() * w).sum() / den.clamp_min(1e-6), den


def ar_flow_loss_diff(head, p_tf, p_ar, flow, valid, S: int, C: int, R: int, split: dict,
                      horizons=(1, 2, 4, 8), supervise=(2, 4, 8), tubelet: int = 2, norm_eps: float = 1e-3):
    """인접(간격 g 블록) predictor 출력의 **차**를 읽어 그 구간 변위를 맞힌다.

    p_tf (B, (nb-1)*S, D)  위치 i ↔ **블록 i+1** 의 예측 토큰  → 문맥 블록도 포함한다 (사용자 지시 "context 에도 다").
    p_ar (B, R*S, D)       걸음 r ↔ 블록 C+r
    flow (B, nb, H, 2, g, g) · valid (B, nb, H, g, g)   앵커 = 각 블록의 마지막 장, 지평선 = 장 수
    지평선 h 장 ↔ 블록 간격 g = h / tubelet. supervise [2, 4, 8] 이면 간격 1 · 2 · 4 블록.

    쌍이 안 나오는 칸 (블록 수가 간격보다 적다) 은 그냥 건너뛴다 — 짧은 창에서는 긴 간격이 없다.
    상대화: 지평선마다 그 batch 의 평균 |d| 로 나눈다 (gradient 없음). 1.0 = 0 예측과 같음.
    """
    H = [int(x) for x in horizons]
    SUP = [int(x) for x in supervise]
    T = p_tf.size(1) // S
    B, _, Dm = p_tf.shape
    Ptf = p_tf.reshape(B, T, S, Dm)                      # 블록 1..nb-1
    Par = p_ar.reshape(B, R, S, Dm)                      # 블록 C..C+R-1

    l_tf_all, l_ar_all, log = [], [], {}
    for k, h in enumerate(SUP):
        g = h // tubelet
        ki = H.index(h)
        parts_tf, parts_ar, cvs = None, None, None
        if T > g:
            tgt = flow[:, 1:1 + T - g, ki]; ok = valid[:, 1:1 + T - g, ki]
            with torch.no_grad():
                sc = ((tgt.abs() * ok.unsqueeze(2)).sum() / (ok.sum() * 2).clamp_min(1e-6)).clamp_min(norm_eps)
            lv, _ = _l1_one(head(Ptf[:, g:] - Ptf[:, :-g], k), tgt, ok)
            parts_tf = lv / sc
            l_tf_all.append(parts_tf)
        if R > g:
            tgt = flow[:, C:C + R - g, ki]; ok = valid[:, C:C + R - g, ki]
            with torch.no_grad():
                sc = ((tgt.abs() * ok.unsqueeze(2)).sum() / (ok.sum() * 2).clamp_min(1e-6)).clamp_min(norm_eps)
                cv = flow[:, C:C + R - g, 0] * (h / H[0])        # 등속 외삽 (파라미터 0)
                cvs = _l1_one(cv, tgt, ok)[0] / sc
            lv, _ = _l1_one(head(Par[:, g:] - Par[:, :-g], k), tgt, ok)
            parts_ar = lv / sc
            l_ar_all.append(parts_ar)
            log[f"l{h}"] = float(parts_ar.detach()); log[f"c{h}"] = float(cvs)
            log["valid"] = float(ok.mean())
        elif parts_tf is not None:
            log[f"l{h}"] = float(parts_tf.detach()); log[f"c{h}"] = float("nan")
    zero = p_tf.sum() * 0.0
    L_tf = torch.stack(l_tf_all).mean() if l_tf_all else zero
    L_ar = torch.stack(l_ar_all).mean() if l_ar_all else zero
    log["tf_l"] = float(L_tf.detach()); log["ar_l"] = float(L_ar.detach())
    log["ar_c"] = float(sum(log.get(f"c{h}", 0.0) for h in SUP) / max(len(SUP), 1))
    log.setdefault("valid", 0.0)
    return float(split.get("tf", 1.0)) * L_tf + float(split.get("ar", 1.0)) * L_ar, log
