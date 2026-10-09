#!/usr/bin/env python3
"""분석 A — 허용폭 없는 연속 위치 판독 (cos 지도, 묶음 P). python scene_cosmap_analyze.py <dir>  (OUT_TAG)

cosmap (n, P, 8, 256_q, 256_k) fp16 = cos(p_i[s], hc_L[r]).  출처 u (팬: 기하, 자연: src.npy).
연속 판독 (토큰 s, 슬롯 i):
  soft-argmax 위치 x̂ = Σ_r softmax(cos/τ)_r · pos(r)  (τ = 0.05)  → 오차 e = ‖x̂ − pos(u)‖₂ (칸)
  top-k 기대 위치 (k = 5, cos 가중) 도 같이.  복사 기준 e_copy = ‖pos(s) − pos(u)‖ (= 변위 d).
  진행률 ρ = ((x̂ − pos(s))·(pos(u) − pos(s))) / d²  (변위 방향으로 얼마나 갔나; 1 = 출처, 0 = 복사).
합의 토큰 (encoder argmax 가 u 1 칸 안) 만. 슬롯 · 거리 구간별 평균, 회귀 ρ ~ 1 + slot + d (clip bootstrap 200) → 거리 vs 걸음을 허용폭 없이.
predictor 별 (rec_preds 순서). 메모리: 한 clip 씩 읽는다.
"""
import json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1]); TAG = os.environ.get("OUT_TAG", ""); ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/scene"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); items = meta["items"]; preds = meta.get("rec_preds", ["R", "A", "P"]); n = len(items); G = 16; TAU = 0.05; K = 5
CM = np.load(I / "cosmap.npy", mmap_mode="r"); E_ = np.load(I / "enc.npy")
if (I / "src.npy").exists():
    SRC = np.load(I / "src.npy").astype(np.int64); Dd = np.nan_to_num(np.load(I / "disp.npy").astype(np.float32), nan=-1)
else:
    sx = np.arange(256) % G; sy = np.arange(256) // G
    def src_of(v, dr):
        src = np.full((8, 256), -1)
        for i in range(8):
            xp = 16 * sx + 8 + dr * v * 2 * (i + 1); ok = (xp >= 0) & (xp < 256); src[i, ok] = sy[ok] * G + (xp[ok] // 16).astype(int)
        return src
    SRC = np.stack([src_of(it["v"], it["dir"]) for it in items]); Dd = np.stack([np.broadcast_to((it["v"] * (np.arange(8) + 1) / 8.0)[:, None], (8, 256)) for it in items])
posx = (np.arange(256) % G).astype(np.float32); posy = (np.arange(256) // G).astype(np.float32)
cheb = lambda a, b: np.maximum(np.abs(a % G - b % G), np.abs(a // G - b // G))
HE = (cheb(E_[:, 0].astype(np.int64), SRC) <= 1) & (SRC >= 0)
rows = {pk: dict(e=[], rho=[], d=[], slot=[], clip=[], ek=[]) for pk in preds}
for c in range(n):
    if not np.isfinite(CM[c, 0, 0, 0, :]).any(): continue
    m = HE[c]; if_any = m.any()
    if not if_any: continue
    ux, uy = posx[np.clip(SRC[c], 0, 255)], posy[np.clip(SRC[c], 0, 255)]
    for pi, pk in enumerate(preds):
        cs = np.asarray(CM[c, pi], dtype=np.float32)                             # (8, 256, 256)
        w = np.exp((cs - cs.max(-1, keepdims=True)) / TAU); w /= w.sum(-1, keepdims=True)
        xh, yh = w @ posx, w @ posy
        idx = np.argpartition(-cs, K, axis=-1)[..., :K]; tk = np.take_along_axis(cs, idx, -1); tw = np.maximum(tk, 0); tw /= np.maximum(tw.sum(-1, keepdims=True), 1e-9)
        xk = (tw * posx[idx]).sum(-1); yk = (tw * posy[idx]).sum(-1)
        sxx, syy = np.broadcast_to(posx[None], (8, 256)), np.broadcast_to(posy[None], (8, 256))
        e = np.sqrt((xh - ux) ** 2 + (yh - uy) ** 2); ek = np.sqrt((xk - ux) ** 2 + (yk - uy) ** 2)
        dvx, dvy = ux - sxx, uy - syy; d2 = dvx ** 2 + dvy ** 2
        rho = np.where(d2 >= 1, ((xh - sxx) * dvx + (yh - syy) * dvy) / np.maximum(d2, 1e-9), np.nan)
        r = rows[pk]; r["e"].append(e[m]); r["ek"].append(ek[m]); r["rho"].append(rho[m]); r["d"].append(Dd[c][m])
        r["slot"].append(np.broadcast_to(np.arange(8)[:, None], (8, 256))[m]); r["clip"].append(np.full(int(m.sum()), c))
rng = np.random.default_rng(0); BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 13)]
res = {}
for pk in preds:
    r = {k: np.concatenate(v) for k, v in rows[pk].items()}
    if len(r["e"]) == 0: continue
    out = dict(n=int(len(r["e"])))
    out["err_by_disp"] = [dict(lo=lo, hi=hi, n=int(((r["d"] >= lo) & (r["d"] < hi)).sum()), soft_err=round(float(r["e"][(r["d"] >= lo) & (r["d"] < hi)].mean()), 3),
                               topk_err=round(float(r["ek"][(r["d"] >= lo) & (r["d"] < hi)].mean()), 3), rho_mean=round(float(np.nanmean(r["rho"][(r["d"] >= lo) & (r["d"] < hi)])), 3),
                               rho_median=round(float(np.nanmedian(r["rho"][(r["d"] >= lo) & (r["d"] < hi)])), 3)) for lo, hi in BINS if ((r["d"] >= lo) & (r["d"] < hi)).sum() > 50]
    out["err_by_slot"] = [round(float(r["e"][r["slot"] == i].mean()), 3) for i in range(8)]
    out["rho_by_slot"] = [round(float(np.nanmedian(r["rho"][r["slot"] == i])), 3) for i in range(8)]
    ok = np.isfinite(r["rho"]) & (r["d"] >= 1); y = np.clip(r["rho"][ok], -1, 2); X = np.stack([np.ones(ok.sum()), r["slot"][ok], r["d"][ok]], 1); cl = r["clip"][ok]
    fit = lambda X_, y_: np.linalg.lstsq(X_, y_, rcond=None)[0]; b0 = fit(X, y); uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; bs = []
    for _ in range(200):
        s_ = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]); bs.append(fit(X[s_], y[s_]))
    bs = np.array(bs); out["rho_regress"] = {nm: [round(float(b0[j]), 4), round(float(np.percentile(bs[:, j], 2.5)), 4), round(float(np.percentile(bs[:, j], 97.5)), 4)] for j, nm in enumerate(("intercept", "per_slot", "per_cell"))}
    # 거리 고정 슬롯 효과 / 슬롯 고정 거리 효과 (연속 진행률)
    out["rho_fixed_d_by_slot_tercile"] = []
    for lo, hi in BINS[1:5]:
        mm = ok & (r["d"] >= lo) & (r["d"] < hi)
        if mm.sum() < 300: continue
        q = np.quantile(r["slot"][mm], [1 / 3, 2 / 3]); b = np.digitize(r["slot"][mm], q)
        out["rho_fixed_d_by_slot_tercile"].append(dict(lo=lo, hi=hi, rho=[round(float(np.nanmedian(r["rho"][mm][b == j])), 3) for j in range(3)], slot_mean=[round(float(r["slot"][mm][b == j].mean()), 1) for j in range(3)]))
    res[pk] = out
json.dump(res, open(OUT / f"scene_cosmap{TAG}.json", "w"), indent=1, default=float)
for pk, out in res.items():
    print(f"== {pk} n {out['n']}  ρ 회귀 {out['rho_regress']}")
    print("  d 구간: " + "  ".join(f"[{b['lo']},{b['hi']}) soft {b['soft_err']:.2f} topk {b['topk_err']:.2f} ρ med {b['rho_median']:+.2f}" for b in out["err_by_disp"]))
    print("  슬롯별 ρ 중앙 " + " ".join(f"{x:+.2f}" for x in out["rho_by_slot"]) + "  | 거리 고정 슬롯 삼분위 ρ: " + " ".join(f"[{t['lo']},{t['hi']}) {t['rho']}" for t in out["rho_fixed_d_by_slot_tercile"]))
