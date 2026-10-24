#!/usr/bin/env python3
"""[T] 고리 1 → 2 의 다리: 블록 토큰 잡음의 **시간 구조** (변이도) 와 **분포 모양** — 이론 (§3) 의 가정을 측정으로 고정한다 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/noise_structure.py [--enc target]

phys3d val 캐시, 물체가 16 블록 동안 한 번도 안 닿은 칸 (2 칸 팽창 밖) 만.
  변이도 γ1(k) = E|h_{b+k} − h_b| (차원 평균 L1), γ2(k) = E (h_{b+k} − h_b)² / 2 (반변이도).
  모형: h_b = s + d_b + e_b, e_b 블록마다 독립 (분산 σe²), d_b 는 느린 표류 (국소 수준: 걸음마다 분산 ση² 의 증분).
        → γ2(k) = σe² + k ση² / 2 (선형) / AR(1) 표류면 γ2(k) = σe² + σd² (1 − ρ^k).
  분포 모양: 블록 차이 (h_{b+1} − h_b) 차원별 첨도 · E|x| / sqrt(E x²) (가우시안 0.798, 라플라스 0.707).
  같은 칸 예측기: copy · 최근 n 블록 평균의 L1 을 실측하고, 가우시안 + 위 공분산 모형으로 예측한 값과 대조.
  최적 선형 가중 (최소제곱, 최근 8 블록) 과 지수 평활 (칼만 정상 상태) 이득 κ."""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--enc", default="target")
ap.add_argument("--out", default=None)
a = ap.parse_args()
G, NB = 16, 16
LSW = [0.0936, 0.0688, 0.0675, 0.0645, 0.0759, 0.1017, 0.1686, 0.3594]   # 1 차 실행에서 구한 합 1 최소제곱 가중 (블록 t−7..t)


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


def dil(m, k=1):
    for _ in range(k):
        o = m.copy()
        o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
        o[1:, 1:] |= m[:-1, :-1]; o[:-1, :-1] |= m[1:, 1:]; o[1:, :-1] |= m[:-1, 1:]; o[:-1, 1:] |= m[1:, :-1]
        m = o
    return m


g1 = np.zeros(15); g2 = np.zeros(15); cnt = np.zeros(15)
kurt, shape_r = [], []
pred = {k: [] for k in ("copy", "mean_2", "mean_4", "mean_8", "median_3", "median_8", "ls_lastheavy")}
pred2 = {k: [] for k in ("copy", "mean_2", "mean_4", "mean_8", "median_3", "median_8")}
M = np.zeros((9, 9)); ncell = 0
for f in sorted(glob.glob(os.path.join(a.cache, f"h_val_{a.enc}_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(os.path.join(a.cache, f"lab_val_s{sh}.npz"))
    X = np.load(f, mmap_mode="r")
    for w in range(X.shape[0]):
        cv = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0))
        touch = ((cv[0::2] > 0) | (cv[1::2] > 0))
        never = ~dil(touch.any(0), 2).reshape(-1)
        if never.sum() < 10:
            continue
        h = X[w][:, never].astype(np.float32)                       # (16, n, D)
        for k in range(1, 16):
            d = h[k:] - h[:-k]
            g1[k - 1] += np.abs(d).mean() * d.shape[0]; g2[k - 1] += (d ** 2).mean() / 2 * d.shape[0]; cnt[k - 1] += d.shape[0]
        d1 = (h[1:] - h[:-1]).reshape(-1, h.shape[-1])
        m2 = (d1 ** 2).mean(0); m4 = (d1 ** 4).mean(0)
        kurt.append(float(np.median(m4 / np.maximum(m2 ** 2, 1e-12))))
        shape_r.append(float(np.abs(d1).mean() / np.sqrt((d1 ** 2).mean())))
        for t in range(7, NB - 1):
            y = h[t + 1]
            pred["copy"].append(np.abs(h[t] - y).mean())
            cand = {"copy": h[t], "median_3": np.median(h[t - 2:t + 1], 0), "median_8": np.median(h[t - 7:t + 1], 0)}
            for n in (2, 4, 8):
                cand[f"mean_{n}"] = h[t - n + 1:t + 1].mean(0)
            cand["ls_lastheavy"] = np.tensordot(np.array(LSW, np.float32), h[t - 7:t + 1], 1)
            for k_, v_ in cand.items():
                if k_ != "copy":
                    pred[k_].append(np.abs(v_ - y).mean())
                if k_ in pred2:
                    pred2[k_].append(((v_ - y) ** 2).mean())
            Z = h[t - 7:t + 2].reshape(9, -1).astype(np.float64)
            Z = Z - Z.mean(0, keepdims=True)                         # 칸 · 차원마다 9 블록 평균을 뺀 공분산 (정적 성분 s 제거)
            M += Z @ Z.T; ncell += 1
g1 /= cnt; g2 /= cnt
res = {"gamma_l1": g1.tolist(), "gamma_l2_semi": g2.tolist(), "kurtosis_median": float(np.median(kurt)), "abs_over_rms": float(np.median(shape_r))}
# 변이도 적합: γ2(k) = σe² + σd² (1 − ρ^k)
best = None
for rho in np.linspace(0.0, 0.999, 1000):
    A = np.stack([np.ones(15), 1 - rho ** np.arange(1, 16)], 1)
    coef, *_ = np.linalg.lstsq(A, g2, rcond=None)
    if (coef < 0).any():
        continue
    err = ((A @ coef - g2) ** 2).sum()
    if best is None or err < best[0]:
        best = (err, rho, coef)
_, rho, (se2, sd2) = best
res["fit"] = {"sigma_e2": float(se2), "sigma_d2": float(sd2), "rho": float(rho), "noise_share_lag1": float(se2 / (se2 + sd2 * (1 - rho)))}
# 모형 공분산 아래 가우시안 예측 L1 (copy 와 평균) — 비율만 대조
Cv = lambda i, j: (se2 if i == j else 0.0) + sd2 * rho ** abs(i - j)
def var_pred(wts):                                                   # 정답 = 블록 8 (t+1), 문맥 0..7
    idx = list(range(8))
    v = Cv(8, 8) - 2 * sum(wts[i] * Cv(8, i) for i in idx) + sum(wts[i] * wts[j] * Cv(i, j) for i in idx for j in idx)
    return v
wc = [0] * 7 + [1]
res["model_ratio"] = {f"mean_{n}": float(np.sqrt(var_pred([0] * (8 - n) + [1 / n] * n) / var_pred(wc))) for n in (2, 4, 8)}
res["model_ratio_iid"] = {f"mean_{n}": float(np.sqrt((1 + 1 / n) / 2)) for n in (2, 4, 8)}
res["measured_ratio"] = {k: float(np.mean(v) / np.mean(pred["copy"])) for k, v in pred.items()}
res["measured_l1"] = {k: float(np.mean(v)) for k, v in pred.items()}
res["measured_ratio_l2rms"] = {k: float(np.sqrt(np.mean(v) / np.mean(pred2["copy"]))) for k, v in pred2.items()}
# 최소제곱 가중 (정적 성분 제거 공분산 위): min E (y − Σ w h_b)² , Σ w = 1 제약
Mc = M / ncell
A = Mc[:8, :8]; bvec = Mc[:8, 8]
Ai = np.linalg.inv(A + 1e-9 * np.eye(8)); one = np.ones(8)
w_unc = Ai @ bvec
lam = (1 - one @ w_unc) / (one @ Ai @ one)
w = w_unc + lam * Ai @ one
res["ls_weights_sum1"] = w.tolist()
# 국소 수준 모형의 칼만 정상 이득: q = ση²/σe² (표류를 증분 모형으로 근사: 걸음당 증분 분산 = 2 σd² (1 − ρ))
q = 2 * sd2 * (1 - rho) / se2
kappa = (-q + np.sqrt(q * q + 4 * q)) / 2
res["kalman"] = {"q": float(q), "kappa_last_block_weight": float(kappa)}
print(json.dumps(res, indent=1))
out = a.out or os.path.join(os.path.dirname(__file__), f"../results/noise_structure_{a.enc}.json")
json.dump(res, open(out, "w"), indent=1)
