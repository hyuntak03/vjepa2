"""개념 벡터 (클래스 평균 − 클래스 평균들의 평균) 로 encoder vs 예측을 비교해도 되나 — 같은 클립 공통 잡음 · 다른 정보 묶임 점검 (2026-10-26, 학습 없음, CPU).
사용자: "그럼 concept vector 로 봐야 되긴 하겠네. concept vector 로 봐도 됨?"
걸음 k (블록 b = 7 + k) 마다, 물체 보이는 train 표본으로:
  같은 표본 cos  : cos(c_enc(전부), c_pred(전부)) — 지금까지 쓴 값 (output_concept_compare · release_identity_concept_compare 와 같은 정의)
  엇갈린 반쪽 cos: 클립 짝수 (A) · 홀수 (B) 로 나눠 cos(c_enc^A, c_pred^B), cos(c_enc^B, c_pred^A) 평균 — 같은 클립에서 오는 공통 잡음이 없다
  재현도         : cos(c_enc^A, c_enc^B) · cos(c_pred^A, c_pred^B)
  보정 cos       : 엇갈린 반쪽 cos / sqrt(재현도_enc × 재현도_pred) (셋 다 반쪽 크기라 맞물린다)
  크기 비        : |c_pred| / |c_enc| (전부)
두 판: 원래 (클래스 평균) / 다른 정보 맞춤 (x ≈ [다른 정보, 클래스 원핫] 을 최소제곱으로 함께 맞춰 클래스 계수를 개념 벡터로 —
  다른 정보 = 장면 13 원핫 · 바닥 · 벽 RGB (1 · 2 차) · 위치와 4×4 칸 · 블록 변위 · 물리 상수, identity_readout_direction.py 와 같음).
cos 는 클래스마다 잰 뒤 평균.
  /venv/vjepa/bin/python z_research/scripts/analysis/concept_vector_validity.py   → z_analysis/results/concept_vector_validity.json
"""
import json
import numpy as np
P = "/workspace/auto_research_cache"


def design(lab, env, b):
    n = len(lab); s = np.eye(13)[lab[:, b, 3].astype(int)]
    rgb = env[:, :6]; phys = env[:, 6:]
    cx, cy = lab[:, b, 6] / 256, lab[:, b, 7] / 256; dx, dy = lab[:, b, 4] / 64, lab[:, b, 5] / 64
    grid = np.eye(16)[np.clip((cx * 4).astype(int), 0, 3) * 4 + np.clip((cy * 4).astype(int), 0, 3)]
    return np.concatenate([s, rgb, rgb ** 2, np.stack([cx, cy, dx, dy, np.hypot(dx, dy)], 1), grid, phys], 1)


def concept(X, y, K, N=None):
    if N is None:
        C = np.stack([X[y == c].mean(0) for c in range(K)])
    else:                                                    # 다른 정보와 클래스를 함께 맞춘 클래스 계수
        D = np.concatenate([N - N.mean(0), np.eye(K)[y]], 1)
        A = np.linalg.solve(D.T @ D + 1e-3 * np.eye(D.shape[1]), D.T @ X); C = A[-K:]
    return C - C.mean(0)


def cos(A, B):
    return float(np.mean((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))))


L = np.load(f"{P}/ar_layers_pooled/train.npz"); lab, ii = L["lab"], L["i"]
env = []
for c in L["clip"]:
    e = json.load(open(str(c)[:-4] + ".json"))["env"]
    env.append(list(np.array(e["floor_rgb"]) / 255) + list(np.array(e["wall_rgb"]) / 255) + [e["g"] / 9.8, e["bounce"], e["friction"]])
env = np.array(env)
R = np.load(f"{P}/release_pooled/train.npz"); o = {int(v): n for n, v in enumerate(ii)}; sel = np.array([o[int(v)] for v in R["i"]])
SRC = {"ar": (L["enc"], L["roout"], lab, env), "release": (R["gt"], R["pred"], lab[sel], env[sel])}
out = {}
for model, (GT, PR, LT, ET) in SRC.items():
    out[model] = {}
    for fam, col, K in (("shape", 1, 6), ("color", 2, 8)):
        out[model][fam] = {}
        for k in range(1, 9):
            b = 7 + k; m = LT[:, b, 0] > 0
            y = LT[m, b, col].astype(int); clip = np.nonzero(m)[0]; A_, B_ = clip % 2 == 0, clip % 2 == 1
            E, Pd, N = GT[m, b].astype(np.float64), PR[m, k - 1].astype(np.float64), design(LT[m], ET[m], b)
            row = {}
            for ver, NN in (("raw", None), ("adjusted", N)):
                f = lambda X, s: concept(X[s], y[s], K, None if NN is None else NN[s])
                ce, cp = f(E, slice(None)), f(Pd, slice(None))
                eA, eB, pA, pB = f(E, A_), f(E, B_), f(Pd, A_), f(Pd, B_)
                cross = (cos(eA, pB) + cos(eB, pA)) / 2; re, rp = cos(eA, eB), cos(pA, pB)
                row[ver] = {"same_sample_cos": round(cos(ce, cp), 3), "cross_half_cos": round(cross, 3), "rel_enc": round(re, 3), "rel_pred": round(rp, 3),
                            "corrected": round(cross / np.sqrt(max(re, 1e-6) * max(rp, 1e-6)), 3),
                            "norm_ratio": round(float(np.mean(np.linalg.norm(cp, axis=1) / np.linalg.norm(ce, axis=1))), 3)}
            out[model][fam][k] = row
        print(model, fam, flush=True)
        for ver in ("raw", "adjusted"):
            for key in out[model][fam][1][ver]:
                print(f"   {ver:8s} {key:16s}", [out[model][fam][k][ver][key] for k in range(1, 9)])
json.dump(out, open("/workspace/world_model/vjepa2/z_analysis/results/concept_vector_validity.json", "w"), indent=1)
