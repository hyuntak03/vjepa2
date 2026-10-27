"""차원마다 '무리 사이 차이 ÷ 무리 안 흔들림' — 속도 vs 색 (2026-10-27, CPU, 모델 없음).
사용자: "근데 이게 저분산이랑은 어떤 연관이 있는 거야?" → 설명 (차원 하나 안에서 두 무리가 흔들림보다 크게 떨어져 있으면 중앙값이 다수결, 아니면 평균처럼) 을 직접 잰다.
벡터 · 표본 규칙은 velocity_lowrank_median.py 와 같다 (3D 합성 물체 칸 평균 encoder 벡터 (1, 1280), train 앞 N 편 + val 350 편).
  속도: 오른쪽 (vx ≥ 6, |vy| < 4) vs 왼쪽 (vx ≤ −6, |vy| < 4) · 색: 빨강 vs 파랑.
  차원 d 마다 분리도 = |평균_A − 평균_B| / 무리 안 표준편차 (두 무리 합동). 1 보다 크면 그 차원에서 두 혹이 갈린다.
  + 판독 방향 하나로 모았을 때 (두 무리 평균 차 방향, 무리 안 공분산으로 백색화한 LDA 방향 아님 — 단순 평균 차 방향) 의 분리도.
벡터 캐시: /workspace/auto_research_cache/velocity_vectors/{train,val}.npz (X, L = [vx, vy, x, y, 모양, 색]) — 다음 분석이 다시 읽지 않게.
  /venv/vjepa/bin/python z_research/scripts/analysis/velocity_dim_snr.py --n_train 3000   → z_analysis/results/velocity_dim_snr.json
"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--n_train", type=int, default=3000); a = ap.parse_args()
SRC = "/workspace/auto_research_cache/ar_layers"; CA = "/workspace/auto_research_cache/velocity_vectors"; os.makedirs(CA, exist_ok=True)


def load(split, n):
    f = f"{CA}/{split}.npz"
    if os.path.exists(f):
        z = np.load(f); return z["X"], z["L"]
    X, L = [], []
    for m in json.load(open(f"{SRC}/{split}_meta.json"))[:n]:
        path = f"{SRC}/{split}/{m['i']:05d}.npz"; z = np.load(path); enc = z["enc"]; z.close()
        fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        for b in range(2, 16):
            bl = m["blocks"][b]
            if bl["ok"] and m["blocks"][b - 1]["ok"] and bl["cells"]:
                X.append(enc[b][bl["cells"]].astype(np.float32).mean(0)); L.append([*bl["disp"], *bl["ctr"], m["shape"], m["color"]])
    X, L = np.stack(X), np.array(L, np.float64); np.savez(f, X=X, L=L); return X, L


Xtr, Ltr = load("train", a.n_train); Xva, Lva = load("val", 350)
X = np.concatenate([Xtr, Xva]); L = np.concatenate([Ltr, Lva])
pairs = {"velocity": ((L[:, 0] >= 6) & (np.abs(L[:, 1]) < 4), (L[:, 0] <= -6) & (np.abs(L[:, 1]) < 4)), "color": (L[:, 5] == 0, L[:, 5] == 5)}
R = {"n": int(len(X))}
for nm, (ma, mb) in pairs.items():
    A, B = X[ma], X[mb]
    sd = np.sqrt(0.5 * (A.var(0) + B.var(0))); dsep = np.abs(A.mean(0) - B.mean(0)) / (sd + 1e-8)
    u = A.mean(0) - B.mean(0); u /= np.linalg.norm(u); pa, pb = A @ u, B @ u
    R[nm] = {"n_a": int(ma.sum()), "n_b": int(mb.sum()), "per_dim_separation": [round(float(v), 4) for v in dsep],
             "quantiles": {q: round(float(np.quantile(dsep, float(q))), 3) for q in ("0.5", "0.9", "0.99")}, "max": round(float(dsep.max()), 3),
             "n_dims_above_0.5": int((dsep > 0.5).sum()), "n_dims_above_1": int((dsep > 1).sum()),
             "mean_difference_direction_separation": round(float(abs(pa.mean() - pb.mean()) / np.sqrt(0.5 * (pa.var() + pb.var()))), 3)}
    print(nm, {k: v for k, v in R[nm].items() if k != "per_dim_separation"}, flush=True)
json.dump(R, open("/workspace/world_model/vjepa2/z_analysis/results/velocity_dim_snr.json", "w"), indent=1)
print("saved")
