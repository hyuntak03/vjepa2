"""물체 칸 정체 개념 벡터의 크기 비 (예측 / encoder) — 정지 물체 vs 이동 물체 (2026-10-26 저녁, CPU).
사용자: "concept vector 도 물체 칸에 안 놔두기 때문에 벡터 크기가 줄어드는 거 아니야? 아니면 벡터 크기가 줄어들어서 물체 자리에 안 놔두는 것처럼 보이는 거 아니야?"
정지 물체는 진짜 자리 = 마지막에 본 자리라 "제자리에 안 둠" 이 원인이 될 수 없다. 정지에서도 줄면 크기 자체가 주는 것.
입력: objcell_concept_cache.py (train 8,000 편, 블록 8–15 물체 칸 평균: encoder tgt_true · 예측 pred_true), meta. 정지 / 이동 정의는 static_vs_moving_objcell.py 와 같다
  (정지 = 블록 7 대비 4 px 안 · 블록 안 변위 < 1 px, 이동 = 32 px 이상). 개념 벡터 = 클래스 평균 − 전체 평균 (같은 표본에서 encoder · 예측 각각), 크기 비 · cos (클래스 평균).
  /venv/vjepa/bin/python z_research/scripts/analysis/concept_norm_static_vs_moving.py   → z_analysis/results/concept_norm_static_vs_moving.json
"""
import json
import numpy as np
D = "/workspace/auto_research_cache/objcell_concept"; M = "/workspace/auto_research_cache/ar_layers"
cv = lambda X, y, K: np.stack([X[y == c].mean(0) for c in range(K)]) - X.mean(0)
res = {}
for model in ("ar", "release"):
    Z = np.load(f"{D}/{model}_train.npz"); meta = {m["i"]: m for m in json.load(open(f"{M}/train_meta.json"))}; I = Z["i"]
    sh = np.array([meta[int(i)]["shape"] for i in I]); co = np.array([meta[int(i)]["color"] for i in I])
    ok = np.array([[meta[int(i)]["blocks"][b]["ok"] for b in range(16)] for i in I]); ctr = np.array([[meta[int(i)]["blocks"][b]["ctr"] for b in range(16)] for i in I])
    disp = np.array([[np.hypot(*meta[int(i)]["blocks"][b]["disp"]) for b in range(16)] for i in I])
    d = np.linalg.norm(ctr[:, 8:16] - ctr[:, 7:8], axis=-1); both = ok[:, 8:16] & ok[:, 7:8]
    G = {"static": both & (d < 4) & (disp[:, 8:16] < 1), "moving": both & (d >= 32)}
    T, P = Z["tgt_true"].astype(np.float32), Z["pred_true"].astype(np.float32); res[model] = {}
    for fam, y, K in (("shape", sh, 6), ("color", co, 8)):
        res[model][fam] = {}
        for g, mask in G.items():
            rows = []
            for k in range(1, 9):
                m = mask[:, k - 1]
                if m.sum() < 30 or len(set(y[m])) < K:
                    rows.append(None); continue
                E, Pp = cv(T[m, k - 1], y[m], K), cv(P[m, k - 1], y[m], K)
                rows.append({"norm_ratio": round(float(np.mean(np.linalg.norm(Pp, axis=1) / np.linalg.norm(E, axis=1))), 3),
                             "cos": round(float(np.mean((E * Pp).sum(1) / (np.linalg.norm(E, axis=1) * np.linalg.norm(Pp, axis=1)))), 3), "n": int(m.sum())})
            res[model][fam][g] = rows
            print(model, fam, g, "크기 비", [r["norm_ratio"] if r else None for r in rows], "cos", [r["cos"] if r else None for r in rows], "n", [r["n"] if r else None for r in rows], flush=True)
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/concept_norm_static_vs_moving.json", "w"), indent=1)
