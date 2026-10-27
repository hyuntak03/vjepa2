"""속도 개념 벡터 cos 가 teacher forcing 에서도 0.5 인 이유 — 재현도 상한 · 직전 블록 비교 (2026-10-26, 학습 없음, CPU).
사용자: "속도는 왜 처음부터 0.5 인거지?"  output_concept_compare.py 와 같은 정의의 개념 벡터.
  재현도: 클립을 반으로 나눠 같은 출처의 개념 벡터 두 개의 cos (encoder · predictor 각각) — 측정 잡음 상한
  보정 cos = cos(pred, enc) / sqrt(재현도_enc × 재현도_pred) (반쪽 추정이라 근사)
  직전 블록: cos(c_pred(블록 b), c_enc(블록 b−1)) — predictor 가 마지막으로 본 블록의 움직임을 담나
  /venv/vjepa/bin/python z_research/scripts/analysis/velocity_concept_ceiling.py   → z_analysis/results/velocity_concept_ceiling.json
"""
import json, sys
import numpy as np
sys.path.insert(0, "/workspace/world_model/vjepa2/z_research/scripts/analysis")
P = "/workspace/auto_research_cache/ar_layers_pooled"
TR = dict(np.load(f"{P}/train.npz")); lab = TR["lab"]; ok = lab[..., 0] > 0
half = np.random.default_rng(0).integers(0, 2, lab.shape[0]).astype(bool)


def concepts(X, L):
    out = {}
    for fam, col, K in (("shape", 1, 6), ("color", 2, 8)):
        y = L[:, col].astype(int); M = np.stack([X[y == k].mean(0) for k in range(K)]); out[fam] = M - M.mean(0)
    for fam, cols in (("velocity", (4, 6)), ("position", (6, 8))):
        Y = L[:, cols[0]:cols[1]].astype(np.float64); Yc = Y - Y.mean(0); Xc = X - X.mean(0)
        out[fam] = ((Xc.T @ Yc / len(Y)) @ np.linalg.inv(Yc.T @ Yc / len(Y))).T
    return out


cosm = lambda A, B: float(np.mean(np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))))
F = ("shape", "color", "velocity", "position")
rows = {}
for src in ("tf", "ro"):
    for t in (range(2, 16) if src == "tf" else range(1, 9)):
        b = t if src == "tf" else 7 + t
        m = ok[:, b] & ok[:, b - 1]
        Xp = (TR["tfout"][:, b] if src == "tf" else TR["roout"][:, t - 1]).astype(np.float32)
        Xe, Xe_prev = TR["enc"][:, b].astype(np.float32), TR["enc"][:, b - 1].astype(np.float32)
        Cp, Ce = concepts(Xp[m], lab[m, b]), concepts(Xe[m], lab[m, b])
        Cprev = concepts(Xe_prev[m], lab[m, b - 1])
        mA, mB = m & half, m & ~half
        CpA, CpB = concepts(Xp[mA], lab[mA, b]), concepts(Xp[mB], lab[mB, b]); CeA, CeB = concepts(Xe[mA], lab[mA, b]), concepts(Xe[mB], lab[mB, b])
        r = {}
        for f in F:
            re, rp = cosm(CeA[f], CeB[f]), cosm(CpA[f], CpB[f]); c = cosm(Cp[f], Ce[f])
            r[f] = {"cos": round(c, 3), "rel_enc": round(re, 3), "rel_pred": round(rp, 3), "cos_corrected": round(c / np.sqrt(max(re, 1e-6) * max(rp, 1e-6)), 3),
                    "cos_vs_prev_block_enc": round(cosm(Cp[f], Cprev[f]), 3), "enc_b_vs_enc_prev": round(cosm(Ce[f], Cprev[f]), 3)}
        rows[f"{src}{t}"] = r
json.dump(rows, open("/workspace/world_model/vjepa2/z_analysis/results/velocity_concept_ceiling.json", "w"), indent=1)
for grp, keys in (("TF 블록 2–7", [f"tf{t}" for t in range(2, 8)]), ("TF 블록 8–15", [f"tf{t}" for t in range(8, 16)]), ("굴린 1", ["ro1"]), ("굴린 2", ["ro2"]), ("굴린 4", ["ro4"]), ("굴린 8", ["ro8"])):
    print(grp, {f: tuple(round(float(np.mean([rows[k][f][q] for k in keys])), 2) for q in ("cos", "rel_enc", "rel_pred", "cos_corrected", "cos_vs_prev_block_enc", "enc_b_vs_enc_prev")) for f in F})
