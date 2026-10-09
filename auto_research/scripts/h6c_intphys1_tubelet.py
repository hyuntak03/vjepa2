#!/usr/bin/env python3
"""H6c — IntPhys 1 튜블릿 단위: 판별을 정하는 것은 '문맥 끝에서의 거리' 인가 '사건 뒤 시간' 인가 '문맥 길이' 인가.

입력: h6b_intphys1_baselines.py 의 tubelet_l1_vith_r*.npz (창 × 미래 튜블릿 j 마다 L1: p, copyh, copyz, zero).
matched pair (pairs.csv) 에서 같은 창·같은 j 로 Δ_m(j) = L1_m(imp) − L1_m(pos),  정답 = Δ > 0.
사건 튜블릿 j_ev = floor(((F−1) − (s + k·C)) / 2k)  (F = first_div_strict 또는 magic_tick).  j_ev ≥ 0 인 창 (문맥 = pos/imp 동일) 의
사건 이후 튜블릿 j ≥ j_ev 만:  d = j (문맥 끝에서 튜블릿 수),  e = j − j_ev (사건 뒤 튜블릿 수),  C, W.
로지스틱  logit P(정답) = b0 + b_d·d + b_e·e + b_C·(C/2) + b_W·[W=32],  4 중항 단위 bootstrap.
검증: 창마다 mean_j L1_p == per_window.json surprise.

  python auto_research/scripts/h6c_intphys1_tubelet.py
"""
from __future__ import annotations
import collections, csv, json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H6 = ROOT / "auto_research/exp_results/h6"
AUX = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene"
BENCH = ROOT / "z_research/Benchmarks/exp_results"
MO = {"정지": "static", "이동": "moving"}; VI = {"눈앞": "visible", "가려짐": "occluded"}
METHODS = ("p", "copyh", "copyz", "zero")


def logit_fit(X, y, iters=100):
    X = np.c_[np.ones(len(X)), X]; w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-4 * np.eye(len(w))
        w += np.linalg.solve(H, X.T @ (y - p))
    return w


def main():
    parts = sorted(H6.glob("tubelet_l1_vith_r*.npz"))
    R = np.concatenate([np.load(f)["rows"] for f in parts]); z = np.load(parts[0])
    combos, vids = list(z["combos"]), list(z["video_ids"])
    cols = list(z["cols"]); ci = {c: i for i, c in enumerate(cols)}
    L = {}
    for r in R:
        L[(vids[int(r[0])], combos[int(r[1])], int(r[2]), int(r[3]), int(r[4]))] = r[5:9]
    # 검증
    pw = {}
    for w in (16, 32):
        for vid, rows in json.load(open(BENCH / f"intphys1_sliding__intphys1_dev_vith_w{w}/per_window.json"))["windows"].items():
            for combo, s, C, sur in rows:
                pw[(vid, combo, int(s), int(C))] = sur
    acc = collections.defaultdict(list)
    for (v, cb, s, C, j), m in L.items():
        acc[(v, cb, s, C)].append(m[0])
    dif = [abs(np.mean(x) - pw[k]) for k, x in acc.items() if k in pw]
    print(f"검증: 창 {len(dif)} 개, |mean_j L1_p − per_window| max {max(dif):.2e} median {np.median(dif):.2e}")
    pairs = list(csv.DictReader((AUX / "pairs.csv").open()))
    byv = collections.defaultdict(list)
    for k, m in L.items():
        byv[k[0]].append((k, m))
    recs = []
    for p in pairs:
        grp = f'{MO.get(p["label_motion"])}/{VI.get(p["label_vis"])}'
        for (v, cb, s, C, j), mpos in byv[p["pos"]]:
            key = (p["imp"], cb, s, C, j)
            if key not in L:
                continue
            k_, W = (int(x) for x in re.match(r"skip(\d+)_w(\d+)", cb).groups())
            e0 = s + k_ * C
            rec = dict(block=p["scene"], grp=grp, prin=p["principle"][:2], combo=cb, C=C, W=W, j=j,
                       delta={m: float(L[key][i] - mpos[i]) for i, m in enumerate(METHODS)})
            for ev, col in (("vis", "first_div_strict"), ("magic", "magic_tick")):
                rec[f"jev_{ev}"] = int(np.floor(((int(p[col]) - 1) - e0) / (2 * k_)))
            recs.append(rec)
    print("튜블릿 기록", len(recs))
    rng = np.random.default_rng(0)
    out = {"validation_maxdiff": float(max(dif))}
    for ev in ("vis", "magic"):
        for grp in sorted({r["grp"] for r in recs}):
            S = [r for r in recs if r["grp"] == grp and r["combo"] in ("skip2_w16", "skip2_w32")
                 and r[f"jev_{ev}"] >= 0 and r["j"] >= r[f"jev_{ev}"]]
            if len(S) < 100:
                continue
            d = np.array([r["j"] for r in S], float); e = d - np.array([r[f"jev_{ev}"] for r in S], float)
            Cc = np.array([r["C"] / 2 for r in S], float); w32 = np.array([r["W"] == 32 for r in S], float)
            X = np.c_[d, e, Cc, w32]
            blocks = sorted({r["block"] for r in S}); bi = collections.defaultdict(list)
            for k, r in enumerate(S):
                bi[r["block"]].append(k)
            o = {"n": len(S), "blocks": len(blocks)}
            for m in METHODS:
                y = np.array([r["delta"][m] > 0 for r in S], float)
                wv = logit_fit(X, y)
                bs = np.array([logit_fit(X[pk], y[pk], 40) for pk in
                               (np.concatenate([bi[blocks[i]] for i in rng.integers(0, len(blocks), len(blocks))]) for _ in range(300))])
                lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
                o[m] = {"acc": round(100 * y.mean(), 2),
                        "coef": {nm: [round(wv[i + 1], 3), round(lo[i + 1], 3), round(hi[i + 1], 3)] for i, nm in enumerate(("d", "e", "C/2", "W32"))}}
                # 표: d × e (칸 정확도)
                tab = collections.defaultdict(list)
                for k, r in enumerate(S):
                    tab[(min(int(d[k]), 9), min(int(e[k]), 5))].append(y[k])
                o[m]["table_d_e"] = {f"d{a}|e{b}": [round(100 * np.mean(v), 1), len(v)] for (a, b), v in sorted(tab.items())}
            out[f"{ev}|{grp}"] = o
            print(f"{ev:5s} {grp:17s} n={len(S):5d} " + "  ".join(
                f"{m}: {o[m]['acc']:5.1f} d{o[m]['coef']['d'][0]:+.2f}[{o[m]['coef']['d'][1]:+.2f},{o[m]['coef']['d'][2]:+.2f}] e{o[m]['coef']['e'][0]:+.2f}"
                for m in METHODS), flush=True)
    json.dump(out, open(H6 / "h6c_tubelet_vith.json", "w"), indent=1)


if __name__ == "__main__":
    main()
