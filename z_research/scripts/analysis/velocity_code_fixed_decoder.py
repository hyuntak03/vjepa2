#!/usr/bin/env python3
"""은닉의 속도 코드가 걸음마다 줄어드나 — 고정 디코더 검사 (2026-10-13, 사용자 지적 "속도를 미래 만들 때 안 쓴다? 문제 있다").

PREDICTOR_BOTTLENECK §2-1 의 "속도 R² 0.88–0.97 평평" 은 **걸음마다 ridge 를 새로 맞춘** 값이라, 속도 코드가 같은 방향으로 줄어들어도 R² 는 높게 남는다.
여기서는 걸음 k1 (또는 참 h) 에서 맞춘 ridge 를 **고정**해 k1..k8 에 걸고, 예측 속도의 크기 (참 속도 방향 성분 / 참 속력) 가 걸음마다 어떻게 변하는지 본다.
입력 = predictor_bottleneck 산출물 (z_research/VOBench_v1/bottleneck/<name>/shard*.npz): roll_st_g{k} (은닉 블록 평균) · roll_st_l{k} (참 물체 자리 3×3) · roll_r_* (그린 출력) · h_* (참 미래, AR 공간).
  python velocity_code_fixed_decoder.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
B = REPO / "z_research/VOBench_v1/bottleneck"
K = 8


def ridge_fit(X, Y, lam):
    mu = X.mean(0); ym = Y.mean(0); Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt.T @ np.diag(s / (s ** 2 + lam)) @ U.T @ (Y - ym)
    return lambda Z: (Z - mu) @ W + ym


def main():
    res = {}
    for name in ("release", "ar40", "ar_state40"):
        fs = sorted(glob.glob(str(B / name / "shard*.npz")))
        if not fs: continue
        Z = {}
        for f in fs:
            z = np.load(f, allow_pickle=True)
            for k in z.files:
                if k in ("part", "truth") or k.startswith(("roll_st_g", "roll_st_l", "roll_r_g", "roll_r_l", "h_g", "h_l")):
                    Z.setdefault(k, []).append(z[k])
        Z = {k: np.concatenate(v) for k, v in Z.items()}
        T = Z["truth"]; v = (T[:, 15] - T[:, 0]) / 15.0                                      # px / 블록 (등속)
        sp = np.linalg.norm(v, axis=1); vh = v / sp[:, None]
        tr = Z["part"] == "train"; te = ~tr
        out = {}
        for feat in ("st_g", "st_l", "r_g", "r_l"):
            key = lambda k: f"roll_{feat}{k}"
            if key(1) not in Z: continue
            X1 = Z[key(1)].astype(np.float32)
            best = None
            for lam in (1e0, 1e1, 1e2, 1e3):                                                 # train 절반으로 λ
                h = np.zeros(len(X1), bool); idx = np.nonzero(tr)[0]; h[idx[: len(idx) // 2]] = True
                f_ = ridge_fit(X1[h], v[h], lam); e = np.abs(f_(X1[tr & ~h]) - v[tr & ~h]).mean()
                if best is None or e < best[0]: best = (e, lam)
            dec = ridge_fit(X1[tr], v[tr], best[1])
            ratio, r2 = [], []
            for k in range(1, K + 1):
                pv = dec(Z[key(k)].astype(np.float32)[te])
                ratio.append(float(np.median((pv * vh[te]).sum(1) / sp[te])))                # 참 방향 성분 / 참 속력
                r2.append(float(1 - ((pv - v[te]) ** 2).sum() / ((v[te] - v[te].mean(0)) ** 2).sum()))
            out[feat] = {"speed_ratio_fixed_k1_decoder": ratio, "r2_fixed": r2}
        # 참 미래 h 에서 같은 검사 (천장: 참 미래는 속도가 안 줄어야 한다)
        for hf in ("h_g", "h_l"):
            if f"{hf}1" not in Z: continue
            X1 = Z[f"{hf}1"].astype(np.float32); dec = ridge_fit(X1[tr], v[tr], 1e2)
            out[f"truth_{hf}"] = {"speed_ratio_fixed_k1_decoder": [float(np.median((dec(Z[f"{hf}{k}"].astype(np.float32)[te]) * vh[te]).sum(1) / sp[te])) for k in range(1, K + 1)]}
        res[name] = out
        print(f"\n== {name}  (test {te.sum()}) — k1 에서 맞춘 속도 디코더를 고정해 k1..k8 에 건 예측 속력 / 참 속력")
        for feat, d in out.items():
            print(f"  {feat:10s} " + " ".join(f"{x:5.2f}" for x in d["speed_ratio_fixed_k1_decoder"]) +
                  ("   R² " + " ".join(f"{x:4.2f}" for x in d["r2_fixed"]) if "r2_fixed" in d else ""))
    (B / "velocity_fixed").mkdir(exist_ok=True)
    json.dump(res, open(B / "velocity_fixed/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
