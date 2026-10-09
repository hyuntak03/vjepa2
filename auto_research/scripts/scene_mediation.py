#!/usr/bin/env python3
"""I-scene 매개 분석 — predictor 사이 reach 차이가 **출처 attention 강도** 로 설명되나 (2026-09-25). python scene_mediation.py <dir>

토큰 = (clip, 슬롯, 칸), 합의 토큰 (encoder 판독이 출처를 맞힘) 중 d ≥ 1.  base 판 셋 (R release · A Ariel · P pv1).
  x_src = log(p_src),  x_cmp = log(p_src / p_copy)  (층 평균 attention, e_scene_reach 기록)
  모형 M0: hit ~ 1 + d + [A] + [P]            (predictor 차이)
       M1: hit ~ 1 + d + [A] + [P] + x_cmp     (출처 강도를 넣으면 predictor 차이가 얼마나 줄어드나)
       M2: hit ~ 1 + d + x_cmp + d·x_cmp       (한 모형으로 세 predictor 를 함께)
  보고: [A] 계수가 M0 → M1 에서 줄어든 비율 (매개 몫), x_cmp 계수, d 구간 × x_cmp 사분위 표 (구간 안에서 강도가 높은 토큰이 더 맞나).
  CI: clip bootstrap B = 200.  선형 확률 모형 (해석 쉬움; 범위 밖 예측은 보지 않는다).
"""
import json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1]); ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/scene"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); arms = meta["arms"]; recp = meta.get("rec_preds", ["R", "A", "P"]); items = meta["items"]
G = 16; rng = np.random.default_rng(0)
C = np.load(I / "arm_c.npy").astype(np.int64); E_ = np.load(I / "enc.npy"); AT = np.load(I / "att.npy").astype(np.float32)
if (I / "src.npy").exists():
    SRC = np.load(I / "src.npy").astype(np.int64); Dd = np.nan_to_num(np.load(I / "disp.npy").astype(np.float32), nan=-1)
else:
    def src_of(v, dr):
        src = np.full((8, 256), -1); sx = np.arange(256) % G; sy = np.arange(256) // G
        for i in range(8):
            xp = 16 * sx + 8 + dr * v * 2 * (i + 1); ok = (xp >= 0) & (xp < 256); src[i, ok] = sy[ok] * G + (xp[ok] // 16).astype(int)
        return src
    SRC = np.stack([src_of(it["v"], it["dir"]) for it in items])
    Dd = np.stack([np.broadcast_to((it["v"] * (np.arange(8) + 1) / 8.0)[:, None], (8, 256)) for it in items])
VAL = SRC >= 0
hitf = lambda c: (np.maximum(np.abs(c % G - SRC % G), np.abs(c // G - SRC // G)) <= 1) & VAL
AG = hitf(E_[:, 0].astype(np.int64)) & (Dd >= 1) & (C[:, 0, 0, 0] >= 0)[:, None, None]
n = len(items); CL = np.broadcast_to(np.arange(n)[:, None, None], SRC.shape)

rows = []
for pk in ("R", "A", "P"):
    k = arms.index(f"{pk}:base"); pi = recp.index(pk)
    h = hitf(C[:, k]); ps = AT[:, pi, 0, 0]; pc = AT[:, pi, 0, 2]
    m = AG & np.isfinite(ps) & (ps > 0) & (pc > 0)
    rows.append(dict(pk=pk, hit=h[m].astype(float), d=Dd[m], xs=np.log(ps[m]), xc=np.log(ps[m] / pc[m]), cl=CL[m]))
hit = np.concatenate([r["hit"] for r in rows]); d = np.concatenate([r["d"] for r in rows]); xc = np.concatenate([r["xc"] for r in rows])
xs = np.concatenate([r["xs"] for r in rows]); cl = np.concatenate([r["cl"] for r in rows])
isA = np.concatenate([np.full(len(r["hit"]), r["pk"] == "A", float) for r in rows]); isP = np.concatenate([np.full(len(r["hit"]), r["pk"] == "P", float) for r in rows])
one = np.ones_like(d)
X0 = np.stack([one, d, isA, isP], 1); X1 = np.stack([one, d, isA, isP, xc], 1); X2 = np.stack([one, d, xc, d * xc], 1); X3 = np.stack([one, d, isA, isP, xs], 1)
fit = lambda X, y: np.linalg.lstsq(X, y, rcond=None)[0]
r2 = lambda X, y, b: 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
b0, b1, b2, b3 = fit(X0, hit), fit(X1, hit), fit(X2, hit), fit(X3, hit)
uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; BS = []
for _ in range(200):
    s = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])
    a0, a1, a3 = fit(X0[s], hit[s]), fit(X1[s], hit[s]), fit(X3[s], hit[s])
    BS.append([a0[2], a1[2], 1 - a1[2] / a0[2], a1[4], a0[3], a1[3], 1 - a3[2] / a0[2]])
BS = np.array(BS); ci = lambda j: [round(float(np.percentile(BS[:, j], 2.5)), 4), round(float(np.percentile(BS[:, j], 97.5)), 4)]
res = dict(n_tokens=int(len(hit)), n_clips=int(len(uc)),
           M0=dict(coef=dict(zip(["1", "d", "A", "P"], np.round(b0, 4).tolist())), R2=round(float(r2(X0, hit, b0)), 4)),
           M1=dict(coef=dict(zip(["1", "d", "A", "P", "log(p_src/p_copy)"], np.round(b1, 4).tolist())), R2=round(float(r2(X1, hit, b1)), 4)),
           M2=dict(coef=dict(zip(["1", "d", "x", "d*x"], np.round(b2, 4).tolist())), R2=round(float(r2(X2, hit, b2)), 4)),
           M3_logpsrc=dict(coef=dict(zip(["1", "d", "A", "P", "log p_src"], np.round(b3, 4).tolist())), R2=round(float(r2(X3, hit, b3)), 4)),
           A_effect_M0=[round(float(b0[2]), 4)] + ci(0), A_effect_M1=[round(float(b1[2]), 4)] + ci(1),
           mediated_share_A=[round(float(1 - b1[2] / b0[2]), 3)] + ci(2), mediated_share_A_logpsrc=[round(float(1 - b3[2] / b0[2]), 3)] + ci(6),
           x_coef=[round(float(b1[4]), 4)] + ci(3))
# 거리 구간 안 층화 매개 (적대 검증 VERIFY_ISCENE 정정 8: pooled 는 거리 교락 → 구간 안에서 A 계수 감소 비율을 본다)
strat = []
for lo, hi in ((1, 2), (2, 3), (3, 4), (4, 6), (6, 9)):
    m = (d >= lo) & (d < hi)
    if m.sum() < 400 or isA[m].sum() < 50: continue
    Xa = np.stack([one[m], d[m], isA[m], isP[m]], 1); Xb = np.stack([one[m], d[m], isA[m], isP[m], xs[m]], 1)
    ba, bb = fit(Xa, hit[m]), fit(Xb, hit[m])
    strat.append(dict(lo=lo, hi=hi, n=int(m.sum()), A_M0=round(float(ba[2]), 4), A_with_logpsrc=round(float(bb[2]), 4),
                      mediated_share=round(float(1 - bb[2] / ba[2]), 3) if abs(ba[2]) > 1e-3 else None))
res["mediated_share_A_logpsrc_by_bin"] = strat
# d 구간 × x 사분위 (구간 안 사분위, predictor 풀링)
tab = []
for lo, hi in ((1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 13)):
    m = (d >= lo) & (d < hi)
    if m.sum() < 400: continue
    q = np.quantile(xc[m], [0.25, 0.5, 0.75]); b = np.digitize(xc[m], q)
    tab.append(dict(lo=lo, hi=hi, n=int(m.sum()), hit_by_quartile=[round(float(hit[m][b == j].mean()), 3) for j in range(4)]))
res["hit_by_d_and_strength_quartile"] = tab
json.dump(res, open(OUT / f"scene_mediation{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "hit_by_d_and_strength_quartile"}, indent=1, ensure_ascii=False))
print("층화 매개 (log p_src):", json.dumps(res["mediated_share_A_logpsrc_by_bin"]))
for t in tab: print(f"  d [{t['lo']},{t['hi']}) n {t['n']:7d}  hit by log(p_src/p_copy) 사분위 Q1..Q4: " + " ".join(f"{x:.2f}" for x in t["hit_by_quartile"]))
