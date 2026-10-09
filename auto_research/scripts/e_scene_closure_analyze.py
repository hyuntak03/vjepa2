#!/usr/bin/env python3
"""closure 분석. python e_scene_closure_analyze.py <dir>  (OUT_TAG)
hit = argmax 가 출처 1 칸 안. 유효 토큰 = 출처가 창 안. 걸음 j (= 슬롯) 별 · 변위 구간별로 팔 셋 (os · ar_z · ar_pA) 의 hit 와 짝 차이 (clip bootstrap 300).
  ar_pA − os  : 자기 예측을 붙이면 나아지나 (기대: ≤ 0)      ar_z − os : 진짜 관측을 붙이면 나아지나 (탐침 상한)
  기울기: 걸음 j 에 따른 hit 의 선형 기울기 (j 3–7) — v3 P3 의 "j3 부터 정체" 에 대응.
"""
import json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1]); TAG = os.environ.get("OUT_TAG", ""); ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/scene"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); items = meta["items"]; preds = meta["preds"]; arms = meta["arms"]; n = len(items); G = 16
AM = np.load(I / "am.npy").astype(np.int64); CT = np.load(I / "ct.npy").astype(np.float32); CC = np.load(I / "cc.npy").astype(np.float32)
done = AM[:, 0, 0, 0, 0] >= 0
sx = np.arange(256) % G; sy = np.arange(256) // G
SRC = np.full((n, 8, 256), -1, np.int64); Dd = np.zeros((n, 8, 256), np.float32)
for c, it in enumerate(items):
    for i in range(8):
        xp = 16 * sx + 8 + it["dir"] * it["v"] * 2 * (i + 1); ok = (xp >= 0) & (xp < 256)
        SRC[c, i, ok] = sy[ok] * G + (xp[ok] // 16).astype(int); Dd[c, i] = it["v"] * 2 * (i + 1) / 16.0
VAL = (SRC >= 0) & done[:, None, None]
cheb = lambda a, b: np.maximum(np.abs(a % G - b % G), np.abs(a // G - b // G))
rng = np.random.default_rng(0); CLIP = np.broadcast_to(np.arange(n)[:, None, None], SRC.shape)
def boot(v, cl, B=300):
    ok = np.isfinite(v); v, cl = v[ok], cl[ok]
    if len(v) == 0: return [np.nan] * 3
    uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}
    bs = [v[np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])].mean() for _ in range(B)]
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
res = dict(n_done=int(done.sum()), preds=preds, per_pred={})
BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 13)]
for pi, pk in enumerate(preds):
    H = {a: (cheb(AM[:, pi, ai], SRC) <= 1) & VAL for ai, a in enumerate(arms)}
    rec = dict(by_slot={a: [boot(H[a][:, j][VAL[:, j]].astype(float), CLIP[:, j][VAL[:, j]]) for j in range(8)] for a in arms},
               by_disp={a: [boot(H[a][VAL & (Dd >= lo) & (Dd < hi)].astype(float), CLIP[VAL & (Dd >= lo) & (Dd < hi)]) for lo, hi in BINS] for a in arms},
               diff_by_slot={f"{a}-os": [boot((H[a][:, j].astype(float) - H["os"][:, j].astype(float))[VAL[:, j]], CLIP[:, j][VAL[:, j]]) for j in range(8)] for a in arms[1:]})
    for a in arms:
        y = np.array([rec["by_slot"][a][j][0] for j in range(3, 8)]); rec.setdefault("slope_j3_7", {})[a] = round(float(np.polyfit(np.arange(3, 8), y, 1)[0]), 4)
    res["per_pred"][pk] = rec
json.dump(res, open(OUT / f"scene_closure{TAG}.json", "w"), indent=1, default=float)
print(f"완료 {res['n_done']}/{n}")
for pk in preds:
    r = res["per_pred"][pk]
    for a in arms: print(f"{pk} {a:5s} slot0..7 " + " ".join(f"{x[0]:.2f}" for x in r["by_slot"][a]) + f"  slope j3-7 {r['slope_j3_7'][a]:+.3f}   by disp " + " ".join(f"{x[0]:.2f}" for x in r["by_disp"][a]))
    for k, v in r["diff_by_slot"].items(): print(f"{pk} Δ{k}: " + " ".join(f"{x[0]:+.2f}[{x[1]:+.2f},{x[2]:+.2f}]" for x in v))
