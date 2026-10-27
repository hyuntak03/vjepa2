"""predictor 출력 vs encoder 정답 — 같은 1280 공간에서 개념 벡터 cos 와 크기 비 (2026-10-26, 학습 없음, CPU).
사용자: "1280 차원에서는 concept vector 들 같을 거 아니야. predictor output 에서 concept vector (물체 identity, velocity, position) 이랑
encoder 1280 (GT) 이렇게 한 거 cos sim, magnitude (L2 norm) 비교해줘. 그냥 최종만".

입력 pool_objcells.py 출력 (블록마다 칸 256 개 평균 (1, 1280), 원래 좌표). 물체가 보이는 블록만. 블록 AR natural_ar_vw_state.
  encoder 정답 = 블록 b 의 encoder 출력 · teacher forcing 출력 = 블록 b 예측 (b = 1..15, 정답 블록 0..b−1 을 봄)
  굴린 예측 출력 = 걸음 k (k = 1..8) → 블록 7+k 예측 (문맥 블록 0..7 = 16 장, 예측 16 장)
개념 벡터 (블록마다 따로, 같은 블록 · 같은 표본으로 encoder 와 predictor 를 각각):
  shape (6) · color (8): c_k = m_k − (클래스 평균들의 평균)
  velocity (vx, vy) = 블록 안 변위 (px, 블록의 두 장 사이) · position (x, y) = 물체 중심 (px):
      B = Cov(특징, 라벨) Cov(라벨)⁻¹ 의 열 2 개 — 라벨 1 px 당 특징 평균 이동
지표: 같은 개념끼리 cos(c_pred, c_enc) · 크기 비 ‖c_pred‖ / ‖c_enc‖ (L2) — 클래스 (또는 열) 평균. 대조: 클래스 짝을 섞은 cos.
묶음 요약: teacher forcing 블록 1–7 · 8–15, 굴린 예측 걸음 1 … 8.

  /venv/vjepa/bin/python z_research/scripts/analysis/output_concept_compare.py
출력: z_analysis/results/output_concept_compare.json
"""
import json
import numpy as np

P = "/workspace/auto_research_cache/ar_layers_pooled"
OUTJ = "/workspace/world_model/vjepa2/z_analysis/results/output_concept_compare.json"
TR = dict(np.load(f"{P}/train.npz")); lab = TR["lab"]; ok = lab[..., 0] > 0


def concepts(X, L):
    out = {}
    for fam, col, K in (("shape", 1, 6), ("color", 2, 8)):
        y = L[:, col].astype(int); M = np.stack([X[y == k].mean(0) for k in range(K)]); out[fam] = M - M.mean(0)
    for fam, cols in (("velocity", (4, 6)), ("position", (6, 8))):
        Y = L[:, cols[0]:cols[1]].astype(np.float64); Yc = Y - Y.mean(0); Xc = X - X.mean(0)
        B = (Xc.T @ Yc / len(Y)) @ np.linalg.inv(Yc.T @ Yc / len(Y))       # (1280, 2)
        out[fam] = B.T                                                     # (2, 1280): vx · vy (또는 x · y)
    return out


def compare(Cp, Ce):
    na = Cp / np.linalg.norm(Cp, axis=1, keepdims=True); nb = Ce / np.linalg.norm(Ce, axis=1, keepdims=True)
    S = na @ nb.T
    shuf = float(S[~np.eye(len(S), dtype=bool)].mean()) if len(S) > 1 else float("nan")
    return {"cos": round(float(np.diag(S).mean()), 3), "cos_shuffled": round(shuf, 3),
            "norm_ratio": round(float((np.linalg.norm(Cp, axis=1) / np.linalg.norm(Ce, axis=1)).mean()), 3),
            "norm_pred": round(float(np.linalg.norm(Cp, axis=1).mean()), 4), "norm_enc": round(float(np.linalg.norm(Ce, axis=1).mean()), 4),
            "per_item_cos": [round(float(v), 3) for v in np.diag(S)], "per_item_ratio": [round(float(v), 3) for v in np.linalg.norm(Cp, axis=1) / np.linalg.norm(Ce, axis=1)]}


res = {"tf": {}, "ro": {}}
for b in range(1, 16):
    m = ok[:, b] & ~np.isnan(TR["tfout"][:, b]).any(1)
    Ce = concepts(TR["enc"][m, b].astype(np.float32), lab[m, b]); Cp = concepts(TR["tfout"][m, b].astype(np.float32), lab[m, b])
    res["tf"][b] = {"n": int(m.sum()), **{f: compare(Cp[f], Ce[f]) for f in Ce}}
for k in range(1, 9):
    b = 7 + k; m = ok[:, b]
    Ce = concepts(TR["enc"][m, b].astype(np.float32), lab[m, b]); Cp = concepts(TR["roout"][m, k - 1].astype(np.float32), lab[m, b])
    res["ro"][k] = {"n": int(m.sum()), **{f: compare(Cp[f], Ce[f]) for f in Ce}}
json.dump(res, open(OUTJ, "w"), indent=1)
F = ("shape", "color", "velocity", "position")
avg = lambda rows, f, key: np.mean([r[f][key] for r in rows])
print("묶음                  | " + " | ".join(f"{f:>8s} cos / 크기비" for f in F))
for name, rows in (("TF 블록 1–7", [res["tf"][b] for b in range(1, 8)]), ("TF 블록 8–15", [res["tf"][b] for b in range(8, 16)])):
    print(f"{name:20s} | " + " | ".join(f"{avg(rows, f, 'cos'):8.2f} / {avg(rows, f, 'norm_ratio'):5.2f}" for f in F))
for k in range(1, 9):
    r = res["ro"][k]
    print(f"{'굴린 예측 걸음 ' + str(k):20s} | " + " | ".join(f"{r[f]['cos']:8.2f} / {r[f]['norm_ratio']:5.2f}" for f in F))
print("대조 (클래스 짝 섞은 cos, 굴린 예측 걸음 4):", {f: res["ro"][4][f]["cos_shuffled"] for f in F})
print("속도 · 위치 열별 (vx, vy / x, y) 굴린 예측 걸음 1, 4, 8:", {k: {f: (res["ro"][k][f]["per_item_cos"], res["ro"][k][f]["per_item_ratio"]) for f in ("velocity", "position")} for k in (1, 4, 8)})
print("saved", OUTJ)
