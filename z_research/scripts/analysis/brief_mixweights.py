#!/usr/bin/env python3
"""첫 예측이 문맥 블록들의 어떤 가중합인가 + 대조군 (진짜 다음 블록) (2026-10-17).

  python z_research/scripts/analysis/brief_mixweights.py --device cuda

최소제곱 y ≈ Σ_b w_b · h_b (b = 문맥 블록 0 … 7, 같은 토큰 자리, 상수 없음). y 는 둘:
  pred = 모델의 걸음 0 예측 / true = 진짜 다음 블록 (블록 C) — 같은 방식으로 분해한 대조군
토큰 집합 둘:
  bg  = 배경 (물체 자리 3×3 제외), brief_motion.py 샘플 (primitive late 70 + visible 30)
  obj = 다음 블록 (C) 의 물체 자리 3×3 — 물체가 보이는 움직이는 clip 만 (primitive · realistic 의 visible · early · mid, 최대 120)
읽기: pred ≈ true 면 모델은 데이터가 시키는 대로 한 것. obj 에서 true 는 마지막 블록에 몰리는데 pred 가 과거로 퍼지면 "시간 평균 당김".
신뢰구간: clip bootstrap 1000 회 95 %."""
import os, sys, csv
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "brief_motion.py")).read()
src = src[: src.index("models = [Model(r) for r in a.runs]")]
__file__ = os.path.join(HERE, "brief_motion.py")
exec(src)
models = [Model(r) for r in a.runs]

def lstsq(H, y):      # H (C, n, D), y (n, D) → w (C,), R²
    A = H.permute(1, 2, 0).reshape(-1, H.size(0)); yy = y.reshape(-1, 1)
    w = torch.linalg.lstsq(A, yy).solution.flatten(); res = yy - A @ w.unsqueeze(1)
    return w.cpu().numpy(), float(1 - (res ** 2).sum() / ((yy - yy.mean()) ** 2).sum())

def run(sample, mode):
    out = {"true": [], "r2_true": [], "occ": []}
    for m in models:
        out[m.name] = []; out["r2_" + m.name] = []
    for r in sample:
        x, posf, hid = load(r); hb = blocks(x)[0]; H = hb.view(16, S, -1)
        if mode == "obj":
            if hid[C] >= 0.5:
                continue
            tok = torch.tensor(region(posf[C]), device=hb.device)
        else:
            Rg = set(region(posf[C])); tok = torch.tensor([t for t in range(S) if t not in Rg], device=hb.device)
        Hc = H[:C][:, tok]
        w, r2 = lstsq(Hc, H[C][tok]); out["true"].append(w); out["r2_true"].append(r2); out["occ"].append(r["occlusion"])
        for m in models:
            _, _, p = m.go(x, hb.unsqueeze(0)); w, r2 = lstsq(Hc, p[:S][tok]); out[m.name].append(w); out["r2_" + m.name].append(r2)
    return out

def show(out, title):
    print(f"\n=== {title} (n={len(out['true'])}) — w0 (가장 오래됨) … w7 (마지막) [95 % CI], 합, R²")
    rng = np.random.default_rng(0)
    for key in ["true"] + [m.name for m in models]:
        W = np.array(out[key]); n = len(W)
        bs = np.array([W[rng.integers(0, n, n)].mean(0) for _ in range(1000)])
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0); mu = W.mean(0)
        r2 = np.mean(out["r2_" + key]) if key != "true" else np.mean(out["r2_true"])
        name = "진짜 다음 블록" if key == "true" else key[-22:]
        print(f"  {name:>22s} " + " ".join(f"{mu[b]:+.2f}[{lo[b]:+.2f},{hi[b]:+.2f}]" for b in (0, 3, 5, 6, 7))
              + f" | 앞 0–5 합 {mu[:6].sum():.2f} | 합 {mu.sum():.2f} | R² {r2:.2f}")

bg = run(pick, "bg"); show(bg, "배경 토큰 (primitive late 70 + visible 30)")
rows2 = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv")))
         if r["variant"] == "pos_a" and r["motion"] != "flat_static" and r["occlusion"] in ("visible", "early", "mid")]
samp = list(np.random.default_rng(2).permutation(rows2))[:120]
ob = run(samp, "obj"); show(ob, "물체 자리 3×3 (움직이는 clip, 다음 블록에서 물체 보임)")
import json
json.dump({k: {kk: np.array(vv).tolist() for kk, vv in d.items()} for k, d in (("bg", bg), ("obj", ob))},
          open(os.path.join(a.out, "brief_mixweights.json"), "w"))
