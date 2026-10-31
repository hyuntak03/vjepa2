#!/usr/bin/env python3
r"""transfer_gap [R] R3 적대 검증 — 사후 검사 다섯 (2026-10-30). 모두 학습 없음 · GPU 없음.

⚠️ 사전 등록 없이 돌렸다 (BOARD "[R] R3 — 사전 등록 누락 기록"). 판정을 뒤집는 근거로 쓰지 않고, 해석 범위를 좁히는 데만 쓴다.

부분
  copy_ci    E36 보조: 잡음 판 IntPhys 1 Garrido 최고끼리 (릴리즈 − 복사) · (PFT − 복사) 의 장면 bootstrap 95 % CI (칸 고르기 포함)
  e33b       E33b 재계산: 물체 팔 속도 판독기 → 빈 장면 팔 전이 비 ρ 와 존재 대비 기울기를 k2 · k3 로 나누고, 가려진 t7 · t6 와
             보이는 첫 튜블릿 t0 를 같은 추정기로 비교. 빈 장면 판 (보폭 3 / 2) 구분을 t0 · t5 · t7 에서.
  offset     E34 적대 검사: 국소 판독 (그 모델 자신의 출력으로 학습한 ridge, block 5 겹) 을 참 자리에서 운동 방향 −8 ~ +4 칸 ·
             위아래 4 행 · 마지막 본 자리마다 따로 맞춘 윤곽. 행 · 판독기 · λ 고르기는 E34 와 같다 (오프셋 0 = E34 참 자리, 관문).
  e35        E35 잘라 옮기기 따라감 비의 block 민감도 (방향 미정 block 빼기 · 방향별) · 창별 오른쪽 이동 block 오차 · 원판 t13 늦음 검사.
  frames     VOBench 빈 장면 · IntPhys 1 원판 프레임의 화소 변화 (이웃 프레임 · 같은 block 두 영상) — 렌더 잡음 크기.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_review_r3.py copy_ci     # 이 셸 1–2 분
  $P z_research/scripts/analysis/transfer_gap_review_r3.py e33b        # sbatch CPU 8 · 약 5 분
  $P z_research/scripts/analysis/transfer_gap_review_r3.py offset      # sbatch CPU 12 · 약 30 분
  $P z_research/scripts/analysis/transfer_gap_review_r3.py e35
  $P z_research/scripts/analysis/transfer_gap_review_r3.py frames
출력  z_analysis/transfer_gap/results/review_r3/{E36copy_ci, E33b_recheck, offset_profile, offset_summary, E35_robust, frames}.json (+ offset_rows.npz)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
OUT = REPO / "z_analysis/transfer_gap/results/review_r3"


def _ci(x):
    return [round(float(np.percentile(x, 2.5)), 1), round(float(np.percentile(x, 97.5)), 1)]


# ------------------------------------------------------------------ copy_ci
def copy_ci():
    import transfer_gap_ip1copy as CP
    import transfer_gap_garrido_noise as GN
    import te_intphys1_score as T
    GR = GN.GR
    cfg = T.resolve_cfg(); lay = T.Layout(cfg); meta = T.index_meta(cfg["data"])

    def copy_hits(V, Cst):
        wd = T.window_dict({"video_ids": np.array(V)}, lay, Cst); per = {}
        for combo in GN.COMBOS:
            vids, starts, Cs, arr = GR.score_table(wd, combo)
            sc = {v: float(np.nanmean(np.nanmin(arr[i], 1))) for i, v in enumerate(vids)}
            pr = defaultdict(dict)
            for v, r in meta.items():
                if v in sc:
                    pr[GR.pair_key(r)]["pos" if r["plausible"] == "1" else "imp"] = v
                    pr[GR.pair_key(r)]["bt"] = r["block_type"]; pr[GR.pair_key(r)]["b"] = r["block_id"]
            per[combo] = {k: (sc[x["pos"]] < sc[x["imp"]], x["bt"], x["b"]) for k, x in pr.items() if "pos" in x and "imp" in x}
        return per
    z = np.load(CP.REF)
    H = {("copy", "orig"): copy_hits([str(v) for v in z["video_ids"]], z["copy_std"])}
    for s in range(1, 5):
        V, Cst = CP.load_seed(s); H[("copy", f"src{s}")] = copy_hits(V, Cst)
    vers = ["orig"] + [f"src{s}" for s in range(1, 5)]
    for m in GN.MODELS:
        for v in vers:
            H[(m, v)] = GN.hits(GN.dirs(m, v))[0]
    best = lambda m, v, keys: max(GN.macro(H[(m, v)][c], keys) for c in GN.COMBOS)  # noqa: E731
    keys = sorted(H[("release", "orig")]["skip2_w32"])
    assert all(set(H[("copy", v)][c]) == set(keys) for v in vers for c in GN.COMBOS)
    pt = {v: {m: round(best(m, v, keys), 2) for m in ("copy", "release", "vobpft")} for v in vers}
    kb = defaultdict(list)
    for k in keys:
        kb[H[("release", "orig")]["skip2_w32"][k][2]].append(k)
    blocks = sorted(kb); rng = np.random.default_rng(0)
    B = {v: {"rel-copy": [], "pft-copy": []} for v in vers}; sm = {"rel-copy": [], "pft-copy": []}
    for _ in range(2000):
        bs = rng.choice(blocks, len(blocks)); kk = [k for b in bs for k in kb[b]]
        acc = {v: {m: best(m, v, kk) for m in ("copy", "release", "vobpft")} for v in vers}
        for v in vers:
            B[v]["rel-copy"].append(acc[v]["release"] - acc[v]["copy"]); B[v]["pft-copy"].append(acc[v]["vobpft"] - acc[v]["copy"])
        sm["rel-copy"].append(np.mean([acc[v]["release"] - acc[v]["copy"] for v in vers[1:]]))
        sm["pft-copy"].append(np.mean([acc[v]["vobpft"] - acc[v]["copy"] for v in vers[1:]]))
    out = {"point": pt, "ci": {v: {k: _ci(x) for k, x in B[v].items()} for v in vers}, "seed_mean_ci": {k: _ci(x) for k, x in sm.items()},
           "seed_mean_point": {k: round(float(np.mean([pt[v]["release" if k == "rel-copy" else "vobpft"] - pt[v]["copy"] for v in vers[1:]])), 2) for k in sm}}
    for m in ("copy", "release"):
        for v in vers:
            g = defaultdict(list)
            for k, (h, bt, b) in H[(m, v)]["skip2_w32"].items():
                g[bt].append(h)
            out.setdefault(f"{m}_w32_by_property", {})[v] = {bt: round(100 * np.mean(x), 1) for bt, x in g.items()}
    json.dump(out, open(OUT / "E36copy_ci.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("point", "ci", "seed_mean_ci", "seed_mean_point")}, ensure_ascii=False))


# ------------------------------------------------------------------ e33b
def e33b():
    import transfer_gap_velsep as V
    items = json.load(open(V.ARR / "list.json"))
    Zs = {v: np.load(V.ARR / f"z_{v}.npy", mmap_mode="r") for v in V.VERS}
    Fe = {}
    for v in V.VERS:
        o = {t: np.zeros((len(items), 1280), np.float64) for t in (0, 5, 6, 7)}
        for s in range(0, len(items), 40):
            blk = np.asarray(Zs[v][s:s + 40], np.float32)
            for t in o:
                o[t][s:s + 40] = blk[:, t].mean(1)
        Fe[v] = o
    by_block = defaultdict(dict)
    for i, it in enumerate(items):
        by_block[it["block"]][it["variant"]] = i
    ok = lambda it: it["occ"] == "late" and it["k"] in (2, 3) and all(it["track"][v]["h"][7] >= 0.9 for v in V.VERS)  # noqa: E731
    obj = [i for i, it in enumerate(items) if it["label"] != "empty" and ok(it)]
    pairs = [(d["pos_a"], d["pos_b"]) for b, d in by_block.items() if "pos_b" in d and items[d["pos_b"]]["label"] == "empty" and ok(items[d["pos_a"]])]
    vel = lambda i, v: float(items[i]["track"][v]["c"][7][0] - items[i]["track"][v]["c"][6][0])  # noqa: E731
    res = {"n_obj": len(obj), "n_pairs": len(pairs),
           "k_of_pairs": {str(k): int(n) for k, n in zip(*np.unique([items[a]["k"] for a, _ in pairs], return_counts=True))}}

    def fit_transfer(t, seed):   # transfer_gap_velsep.transfer 와 같은 절차 (특징만 미리 계산)
        X = np.stack([Fe[v][t][i] for i in obj for v in V.VERS]); y = np.array([vel(i, v) for i in obj for v in V.VERS])
        g = np.array([items[i]["block"] for i in obj for v in V.VERS])
        ub = np.unique(g); rng = np.random.default_rng(seed); rng.shuffle(ub); folds = np.array_split(ub, 5); po, pe = {}, {}
        for f in folds:
            te = np.isin(g, f); tr = ~te; gtr = g[tr]; ubt = np.unique(gtr); rng.shuffle(ubt); inner = np.array_split(ubt, 4); best = None
            for lam in (1e-1, 1, 10, 100, 1e3, 1e4, 1e5):
                err = 0.0
                for gg in inner:
                    v_ = np.isin(gtr, gg); err += ((V._fit(X[tr][~v_], y[tr][~v_, None], lam)(X[tr][v_])[:, 0] - y[tr][v_]) ** 2).sum()
                if best is None or err < best[0]:
                    best = (err, lam)
            fn = V._fit(X[tr], y[tr][:, None], best[1])
            for q in np.where(te)[0]:
                po[q] = fn(X[q:q + 1])[0, 0]
            fs = set(f.tolist())
            for a, e in pairs:
                if items[a]["block"] in fs:
                    for v in V.VERS:
                        pe[(a, v)] = fn(Fe[v][t][e][None])[0, 0]
        qi = {(i, v): q for q, (i, v) in enumerate([(i, v) for i in obj for v in V.VERS])}
        return po, pe, qi

    def slopes(po, pe, qi, sel):
        A = [a for a, _ in pairs if sel(items[a])]
        dv = np.array([vel(a, "s3") - vel(a, "s2") for a in A]); do = np.array([po[qi[(a, "s3")]] - po[qi[(a, "s2")]] for a in A])
        de = np.array([pe[(a, "s3")] - pe[(a, "s2")] for a in A])
        bo = float((do * dv).sum() / (dv * dv).sum()); be = float((de * dv).sum() / (dv * dv).sum())
        return dict(n=len(A), b_obj=round(bo, 3), b_emp=round(be, 3), rho=round(be / bo, 3))
    for t in (7, 6, 0):
        for seed in (0, 1):
            po, pe, qi = fit_transfer(t, seed)
            res[f"transfer|t{t}|s{seed}"] = {"all": slopes(po, pe, qi, lambda it: True), "k2": slopes(po, pe, qi, lambda it: it["k"] == 2),
                                             "k3": slopes(po, pe, qi, lambda it: it["k"] == 3)}
            print(t, seed, res[f"transfer|t{t}|s{seed}"], flush=True)
    for t in (7, 0):
        X = np.stack([Fe[v][t][a] - Fe[v][t][e] for a, e in pairs for v in V.VERS]); y = np.array([vel(a, v) for a, e in pairs for v in V.VERS])
        g = np.array([items[a]["block"] for a, e in pairs for v in V.VERS]); ks = np.array([items[a]["k"] for a, e in pairs for v in V.VERS])
        for seed in (0, 1):
            pf = V.cv_pred(X, y[:, None], g, seed)[:, 0]
            dv = y[0::2] - y[1::2]; dp = pf[0::2] - pf[1::2]; kk = ks[0::2]; o = {}
            for nm, m in (("all", np.ones_like(kk, bool)), ("k2", kk == 2), ("k3", kk == 3)):
                o[nm] = dict(n=int(m.sum()), slope=round(float((dp[m] * dv[m]).sum() / (dv[m] * dv[m]).sum()), 3), corr=round(float(np.corrcoef(dp[m], dv[m])[0, 1]), 3))
            o["R2"] = round(V.r2(y, pf), 3)
            res[f"deltaz|t{t}|s{seed}"] = o; print("dz", t, seed, o, flush=True)
    emp = [e for _, e in pairs]
    for t in (0, 5, 7):
        X = np.stack([Fe[v][t][e] for e in emp for v in V.VERS]); yb = np.array([1.0 if v == "s3" else 0.0 for e in emp for v in V.VERS])
        g = np.array([items[e]["block"] for e in emp for v in V.VERS])
        pv = V.cv_pred(X, yb[:, None], g, 0)[:, 0]
        res[f"empty_stride_decode|t{t}"] = round(float(((pv > 0.5) == (yb > 0.5)).mean()), 3)
        res[f"empty_s3_s2_L1|t{t}"] = round(float(np.mean([np.abs(Fe["s3"][t][e] - Fe["s2"][t][e]).mean() for e in emp])), 4)
    json.dump(res, open(OUT / "E33b_recheck.json", "w"), indent=1, ensure_ascii=False)


# ------------------------------------------------------------------ offset
def offset():
    import transfer_gap_common as C
    import transfer_gap_arlocal as A
    OFFS = list(range(-8, 5))
    M = C.meta(); tr = np.load(C.TG / "tracks.npz"); bmap = {b: i for i, b in enumerate(tr["block_id"])}
    L = json.load(open(A.ARR / "list.json")); pos = {tp: q for q, tp in enumerate(L["tp"])}
    cell = lambda uv: (int(np.clip(uv[1] // 32.0, 0, 15)), int(np.clip(uv[0] // 32.0, 0, 15)))  # noqa: E731
    rows, spec = [], []
    for i in L["tp"]:
        m = M[i]; lab = C.label_of(m); bi = bmap[m["block_id"]]; arm = 0 if m["variant"] == "pos_a" else 1
        uv = tr["uv"][bi, arm]; hid = tr["hidden"][bi, arm]
        vis = [t for t in range(8) if not hid[t]]; tl = vis[-1] if vis else 0
        for t in range(8, 16):
            if hid[t]:
                continue
            r, c = cell(uv[t]); rl, cl = cell(uv[tl]); s = int(np.sign(uv[t][0] - uv[tl][0])) or 1
            posd = {}
            for o in OFFS:
                cc = c + o * s
                if 1 <= cc <= 14:
                    posd[f"o{o:+d}"] = A.box(r, cc)
            for dr in (-4, 4):
                rr = r + dr
                if 1 <= rr <= 14:
                    posd[f"row{dr:+d}"] = A.box(rr, c)
            posd["last"] = A.box(rl, cl)
            rows.append(dict(block=m["block_id"], objset=m["objset"], shape=lab.split("/")[0], color=lab.split("/")[-1], t=t, dist_true_last=abs(c - cl)))
            spec.append((i, t, posd))
    names = [f"o{o:+d}" for o in OFFS] + ["row-4", "row+4", "last"]
    srcs = {"rel_p": None, "h": None, "rel_pft_p": None, "ar_h_blk": A.ARR / "h_blk.npy", "vast70_p": A.ARR / "p_vast70.npy",
            "vast70_pft_p": A.ARR / "p_vast70_pft.npy", "ariel_ar38_p": A.ARR / "p_ariel_ar38.npy"}
    TP = np.load(C.TUBE / "p.npy", mmap_mode="r"); TH = np.load(C.TUBE / "h.npy", mmap_mode="r"); PF = np.load(C.TG / "p_relftg.npy", mmap_mode="r")
    g = np.array([r["block"] for r in rows]); out = {"n_rows": len(rows), "offsets": OFFS, "tasks": {}}
    COR = {}
    for src, f in srcs.items():
        t0 = time.time(); AR = np.load(f, mmap_mode="r") if f else None
        F = {nm: np.full((len(rows), 1280), np.nan, np.float32) for nm in names}
        cache_i, X = None, None
        for q, (i, t, posd) in enumerate(spec):
            if i != cache_i:
                X = (np.asarray(TP[i]) if src == "rel_p" else np.asarray(TH[i, 8:]) if src == "h" else np.asarray(PF[i]) if src == "rel_pft_p"
                     else np.asarray(AR[pos[i]]))
                cache_i = i
            Xl = A.LN(X[t - 8])
            for nm, ix in posd.items():
                F[nm][q] = Xl[ix].mean(0)
        for obj, attr in (("primitive", "shape"), ("realistic", "shape"), ("primitive", "color")):
            s = np.array([r["objset"] == obj for r in rows])
            for nm in names:
                v = s & np.isfinite(F[nm][:, 0])
                y = [r[attr] for r, ok in zip(rows, v) if ok]
                cor, _ = A.ridge_cv_pred(F[nm][v].astype(np.float64), y, g[v])
                full = np.full(len(rows), np.nan); full[v] = cor; COR[(src, obj, attr, nm)] = full
        print(src, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(OUT / "offset_rows.npz", **{"|".join(k): v for k, v in COR.items()}, block=g,
             t=np.array([r["t"] for r in rows]), objset=np.array([r["objset"] for r in rows]), dist=np.array([r["dist_true_last"] for r in rows]))
    summ = {}
    for obj, attr in (("primitive", "shape"), ("realistic", "shape"), ("primitive", "color")):
        task = {}; s = np.array([r["objset"] == obj for r in rows])
        for src in srcs:
            o = {}; c0 = COR[(src, obj, attr, "o+0")]
            for nm in names:
                c = COR[(src, obj, attr, nm)]; v = s & np.isfinite(c); d = c[v] - c0[v]
                o[nm] = dict(n=int(v.sum()), acc=round(100 * float(c[v].mean()), 1), minus_true=round(100 * float(d.mean()), 1),
                             ci=[round(100 * x, 1) for x in A.block_boot(d, g[v])])
            task[src] = o
            near = [(k, o[f"o{k:+d}"]["acc"]) for k in OFFS]; mx = max(near, key=lambda x: x[1])
            summ.setdefault(src, {})[f"{obj}|{attr}"] = dict(true=o["o+0"]["acc"], behind_far=round(float(np.mean([o[f"o{k:+d}"]["acc"] for k in range(-8, -3)])), 1),
                                                           peak_off=mx[0], peak=mx[1], o2=(o["o+2"]["minus_true"], o["o+2"]["ci"]),
                                                           rowp4=(o["row+4"]["minus_true"], o["row+4"]["ci"]), rowm4=(o["row-4"]["minus_true"], o["row-4"]["ci"]))
        out["tasks"][f"{obj}|{attr}"] = task
    json.dump(out, open(OUT / "offset_profile.json", "w"), indent=1, ensure_ascii=False)
    json.dump(summ, open(OUT / "offset_summary.json", "w"), indent=1, ensure_ascii=False)
    print("done")


# ------------------------------------------------------------------ e35
def e35():
    from transfer_gap_endprior import loc, direction, ARR
    d = json.load(open(REPO / "z_analysis/transfer_gap/results/q3/E35_endprior.json"))
    rows = d["crop_per_block_t15_col"]; pairs = [("x0y0", "x64y0"), ("x0y64", "x64y64")]

    def beta(R, model):
        dP = sum(r[b][model] - r[a][model] for r in R for a, b in pairs); dT = sum(r[b]["true"] - r[a]["true"] for r in R for a, b in pairs)
        return dP / dT
    out = {"beta": {}, "window_err_right_movers": {}, "orig_signed_err_along_motion": {}}
    for model in ("vobpft", "objho"):
        R = [r for r in rows if r["dir"] != 0]; rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            ix = rng.integers(0, len(R), len(R))
            try:
                bs.append(beta([R[i] for i in ix], model))
            except ZeroDivisionError:
                pass
        out["beta"][model] = {"all": round(beta(rows, model), 3), "no_dir0": round(beta(R, model), 3), "no_dir0_ci95": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
                              "right": round(beta([r for r in rows if r["dir"] == 1], model), 3), "left": round(beta([r for r in rows if r["dir"] == -1], model), 3),
                              "dir0_blocks": [r["block"] for r in rows if r["dir"] == 0]}
    for w in ("x0y0", "x64y0", "x0y64", "x64y64"):
        R = [r for r in rows if r["dir"] == 1]
        out["window_err_right_movers"][w] = {"true": [r[w]["true"] for r in R], "vobpft": [r[w]["vobpft"] for r in R],
                                            "mean_err": round(float(np.mean([r[w]["vobpft"] - r[w]["true"] for r in R])), 2)}
    Zo = np.load(ARR / "orig.npz"); rh, ch, _, _ = loc(Zo["Eh"]); dd = direction(ch)
    for t in ("vobpft", "objho"):
        r_, c_, _, _ = loc(Zo[f"E_{t}"])
        for q, tt in ((3, 11), (5, 13), (7, 15)):
            e = (c_[:, q] - ch[:, tt]) * dd; m = dd != 0
            out["orig_signed_err_along_motion"][f"{t}|t{tt}"] = {"mean": round(float(e[m].mean()), 2), "median": float(np.median(e[m])),
                                                                  "hist": {int(k): int(v) for k, v in zip(*np.unique(e[m], return_counts=True))}}
    json.dump(out, open(OUT / "E35_robust.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False))


# ------------------------------------------------------------------ frames
def frames():
    from PIL import Image
    im = lambda f: np.asarray(Image.open(f).convert("RGB"), np.int16)  # noqa: E731
    out = {"vobench_empty_changed_px_per_frame": {}, "ip1_consecutive_absdiff_hist": {}, "ip1_cross_video_absdiff_hist": {}}
    items = json.load(open("/data2/local_datasets/world/world_analysis/cache/transfer_gap/E33/list.json"))
    idx = {r["video_id"]: r for r in csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv"))}
    VOB = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
    for it in [x for x in items if x["label"] == "empty" and x["occ"] == "late" and x["k"] in (2, 3)][:3]:
        fs = sorted((VOB / idx[it["video_id"]]["file_name"]).glob("*.png"))
        prev = im(fs[0]); ch = []
        for k in range(1, 60):
            cur = im(fs[k]); ch.append(int((np.abs(cur - prev).max(-1) > 0).sum())); prev = cur
        out["vobench_empty_changed_px_per_frame"][it["video_id"]] = {"n_px": int(prev.shape[0] * prev.shape[1]), "frames_1_59": ch}
    root = Path("/local_datasets/world/world_analysis/IntPhys1_dev_frame_png")
    for blk in ("O1/01", "O3/10"):
        fs = [sorted((root / blk / str(i) / "scene").glob("*.png")) for i in (1, 2, 3, 4)]
        a, b = im(fs[0][20]), im(fs[0][21]); dd = np.abs(a - b).max(-1)
        out["ip1_consecutive_absdiff_hist"][blk] = {"n_px": int(dd.size), "hist_levels_cap6": {int(k): int(v) for k, v in zip(*np.unique(np.minimum(dd, 6), return_counts=True))}}
        for fi in (70, 95):
            fr = [im(f[fi]) for f in fs]
            for x, y in ((0, 1), (0, 2), (0, 3)):
                dd = np.abs(fr[x] - fr[y]).max(-1)
                if dd.max() > 0:
                    out["ip1_cross_video_absdiff_hist"][f"{blk}|f{fi}|{x + 1}v{y + 1}"] = {int(k): int(v) for k, v in zip(*np.unique(np.minimum(dd, 6), return_counts=True))}
                    break
    json.dump(out, open(OUT / "frames.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["copy_ci", "e33b", "offset", "e35", "frames"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"copy_ci": copy_ci, "e33b": e33b, "offset": offset, "e35": e35, "frames": frames}[a.part]()
