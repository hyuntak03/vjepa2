#!/usr/bin/env python3
"""[T] 가닥 2 의 "부드러운 섞기": 칸별 L1 최적 = 섞인 분포의 차원별 중앙값. 그 진하기 g(π) 를 실측 대비 · 잡음 크기와 모양에서 계산하고,
[P] 의 새 자리 존재감 0.4–0.5 · 마지막 자리 잔상 0.25–0.33 을 재현하는 확률 (간다 p · 남는다 q) 을 역산한다 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/mixture_median.py

모형 (차원 d 마다): 정답 = Δ_d + η (물체가 그 칸에 옴, 확률 π) 또는 η (안 옴). 배경 = 0. 예측 m_d = 섞인 분포의 중앙값.
진하기 g(π) = Σ_d m_d Δ_d / Σ_d Δ_d² (예측 편차를 물체 대비 방향에 사영, 1 = 물체를 다 그림).
Δ_d ~ N(0, τ²): 차원 평균 |Δ| = 0.73 (물체 대비 실측, 잡음 포함) 또는 0.58 (잡음 몫 뺀 역산).
η 모양: 가우시안 · 라플라스 · 실측 (3D 합성 캐시, 물체가 안 온 칸의 블록 차이 / √2 — 첨도 ≈ 5). 크기: 차원 평균 |η| = 0.45/√2 (블록 하나의 잡음) 또는 0.45 (리더 가정)."""
import glob, json, os
import numpy as np
from scipy import optimize

rng = np.random.default_rng(0)
ND = 4000
# 실측 잡음: 3D 합성 캐시에서 물체가 한 번도 안 온 칸의 이웃 블록 차이
f = sorted(glob.glob("/workspace/auto_research_cache/phys3d/h_val_target_s0.npy"))[0]
X = np.load(f, mmap_mode="r")
emp = []
for w in range(0, 40):
    h = X[w].astype(np.float32)
    d = (h[1:, :10] - h[:-1, :10]).ravel() / np.sqrt(2)                 # 왼쪽 위 10 칸 (대개 먼 배경)
    emp.append(d[rng.choice(d.size, 5000, replace=False)])
emp = np.concatenate(emp); emp = emp / np.abs(emp).mean()               # 차원 평균 |η| = 1 로 정규화
NE = 4001
shapes = {"gauss": rng.standard_normal(NE) / np.sqrt(2 / np.pi), "laplace": rng.laplace(size=NE), "empirical": rng.choice(emp, NE)}
kurt = float((emp ** 4).mean() / (emp ** 2).mean() ** 2)


def g(pi, dabs, eabs, shape):
    tau = dabs / np.sqrt(2 / np.pi)
    Dl = rng.standard_normal(ND) * tau
    eta = shapes[shape] * eabs
    srt = np.sort(eta)
    F = lambda x: np.searchsorted(srt, x) / srt.size                    # 경험 CDF
    m = np.empty(ND)
    for i, D in enumerate(Dl):
        fn = lambda x: pi * F(x - D) + (1 - pi) * F(x) - 0.5
        lo, hi = min(0, D) - 5 * eabs, max(0, D) + 5 * eabs
        m[i] = optimize.brentq(fn, lo, hi) if fn(lo) < 0 < fn(hi) else (D if pi > 0.5 else 0.0)
    return float((m * Dl).sum() / (Dl ** 2).sum())


res = {"empirical_kurtosis": kurt}
print(f"실측 잡음 첨도 {kurt:.2f} (가우시안 3, 라플라스 6)")
pis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for dabs, eabs in ((0.73, 0.45 / np.sqrt(2)), (0.58, 0.45 / np.sqrt(2)), (0.73, 0.45)):
    for shape in ("gauss", "laplace", "empirical"):
        gs = [g(p, dabs, eabs, shape) for p in pis]
        inv = lambda target: float(np.interp(target, gs, pis))
        key = f"D{dabs}_e{eabs:.3f}_{shape}"
        res[key] = {"g": gs, "p_go_for_0.40": inv(0.40), "p_go_for_0.50": inv(0.50), "q_stay_for_0.25": inv(0.25), "q_stay_for_0.33": inv(0.33)}
        print(f"|Δ| {dabs} · |η| {eabs:.3f} · {shape:9s}: g(π) " + " ".join(f"{v:.2f}" for v in gs)
              + f"  → 새 자리 0.40–0.50 ⇔ p {inv(0.40):.2f}–{inv(0.50):.2f} · 잔상 0.25–0.33 ⇔ q {inv(0.25):.2f}–{inv(0.33):.2f}")
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/mixture_median.json"), "w"), indent=1)
