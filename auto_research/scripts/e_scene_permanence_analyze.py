#!/usr/bin/env python3
"""permanence 분석. python e_scene_permanence_analyze.py <dir>   (OUT_TAG)
토큰 = (clip, 슬롯, 칸). hid = 프레임 15 에서 띠 안 (마지막 관측에 없음). vis = 띠 밖.
  hit_prev = argmax over hc[L−m] 이 u_m 1 칸 안 (가려지기 전 자리의 내용을 제자리에 그렸나).  천장 = encoder 판독 동일 조건.
  합의 토큰 = encoder 천장이 맞은 토큰. A_hid(d) vs A_vis(d), k 별. ratio = A_hid / A_vis.
  cos_prev (내용 일치) 와 cos_last (띠 = 회색 내용 일치) 도 같이: 가려진 토큰에서 predictor 가 회색 (마지막 관측) 을 그리는지, 원래 내용을 그리는지.
"""
import json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1]); TAG = os.environ.get("OUT_TAG", ""); ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/scene"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); items = meta["items"]; preds = meta["preds"]; n = len(items); G = 16
AM = np.load(I / "argmax_prev.npy").astype(np.int64); CP = np.load(I / "cos_prev.npy").astype(np.float32); CL = np.load(I / "cos_last.npy").astype(np.float32)
HID = np.load(I / "hid.npy"); EP = np.load(I / "enc_prev.npy").astype(np.float32)
done = AM[:, 0, 0, 0] >= 0
K = np.array([it["k"] for it in items]); V = np.array([it["v"] for it in items]); DR = np.array([it["dir"] for it in items])
M = (K + 1) // 2
SRC = np.full((n, 8, 256), -1, np.int64); Dd = np.zeros((n, 8, 256), np.float32)
sx = np.arange(256) % G; sy = np.arange(256) // G
for c in range(n):
    for i in range(8):
        xp = 16 * sx + 8 + DR[c] * V[c] * (2 * (i + 1) + 2 * M[c]); ok = (xp >= 0) & (xp < 256)
        SRC[c, i, ok] = sy[ok] * G + (xp[ok] // 16).astype(int); Dd[c, i] = V[c] * (2 * (i + 1)) / 16.0
VAL = (SRC >= 0) & done[:, None, None]
cheb = lambda a, b: np.maximum(np.abs(a % G - b % G), np.abs(a // G - b // G))
# encoder 천장: enc_prev 는 cos 만 저장 → 천장 판독은 cos_prev 가 그 슬롯의 상위 (여기서는 cos ≥ 0.5 를 합의로 근사하지 않고, hit 만 본다)
rng = np.random.default_rng(0); BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 9)]
res = dict(n_done=int(done.sum()), preds=preds, rows=[])
def boot(v, cl, B=300):
    ok = np.isfinite(v); v, cl = v[ok], cl[ok]
    if len(v) == 0: return [np.nan] * 3
    uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}
    bs = [v[np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])].mean() for _ in range(B)]
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
CLIP = np.broadcast_to(np.arange(n)[:, None, None], SRC.shape)
for pi, pk in enumerate(preds):
    hit = (cheb(AM[:, pi], SRC) <= 1) & VAL
    for k in meta["ks"]:
        mk = VAL & (K == k)[:, None, None]
        for lo, hi in BINS:
            m = mk & (Dd >= lo) & (Dd < hi)
            mh, mv = m & HID, m & ~HID
            if mh.sum() < 50 or mv.sum() < 50: continue
            res["rows"].append(dict(pred=pk, k=int(k), lo=lo, hi=hi, n_hid=int(mh.sum()), n_vis=int(mv.sum()),
                                    A_hid=boot(hit[mh].astype(float), CLIP[mh]), A_vis=boot(hit[mv].astype(float), CLIP[mv]),
                                    cos_prev_hid=round(float(np.nanmean(CP[:, pi][mh])), 3), cos_last_hid=round(float(np.nanmean(CL[:, pi][mh])), 3),
                                    cos_prev_vis=round(float(np.nanmean(CP[:, pi][mv])), 3), enc_prev_hid=round(float(np.nanmean(EP[mh])), 3), enc_prev_vis=round(float(np.nanmean(EP[mv])), 3)))
json.dump(res, open(OUT / f"scene_permanence{TAG}.json", "w"), indent=1, default=float)
print(f"완료 {res['n_done']}/{n}")
for pk in preds:
    for k in meta["ks"]:
        rs = [r for r in res["rows"] if r["pred"] == pk and r["k"] == k]
        if not rs: continue
        print(f"{pk} k={k}: " + "  ".join(f"[{r['lo']},{r['hi']}) hid {r['A_hid'][0]:.2f} vis {r['A_vis'][0]:.2f} | cos hid prev/last {r['cos_prev_hid']:.2f}/{r['cos_last_hid']:.2f} vis {r['cos_prev_vis']:.2f}" for r in rs))
