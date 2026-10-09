#!/usr/bin/env python3
"""H2 + H3 분석 — h23_extract_v3.py 산출물 → 표 (CPU).

기호 (모든 벡터는 3×3 창 평균, 창 W 의 clip 평균 ā 를 뺀 중심화 코사인 ccos(a,b) = cos(a−ā, b−b̄)):
  t = 미래 슬롯, T = 문맥 마지막 튜블릿, W_t = 슬롯 t 진실 물체 칸 3×3, W_T = 마지막으로 본 칸 3×3
  A_t^(l)  = ccos( p^(l)_t@W_t , h_t@W_t )   도착: 진실 자리에 진실 물체
  A'_t^(l) = ccos( p^(l)_t@W_t , h_T@W_t )   그 자리의 과거 (보통 배경)
  S_t^(l)  = ccos( p^(l)_t@W_T , h_T@W_T )   머묾: 옛 자리에 옛 물체
  B_t^(l)  = ccos( p^(l)_t@W_T , h_t@W_T )   옛 자리의 진실 미래 (보통 배경)
  Ash      = A 를 같은 시나리오 안에서 clip 을 섞어 잰 값 (특정 물체가 아니라 "그 자리의 무언가" 로 나오는 몫)
  읽기: A>A' · S<B = 진화 / S>B = 옛 자리에 남음 / A≈Ash = 특정 물체가 아님
  W_t 와 W_T 가 겹치지 않는 슬롯 (Chebyshev ≥ 3 칸) 만. 물체가 두 프레임 모두 화면 안 · 안 가려짐 · ignore 아님.
그 밖:
  loc   자 없는 위치 읽기 (템플릿 = h_t 진실 칸 토큰): hit_truth / hit_last / hit_cv (등속 외삽 칸) — Chebyshev ≤ 1
  l1    전 토큰 mean|x − h_t|: p^(l) vs 복사 기준선 (h_T, LN(z)_T)
  D_t   clip 사이 다양성 tr Cov_clip[p_t] / tr Cov_clip[h_t]  (가능 clip, 전 토큰)
  V_t   진행 걸음 mean|p_{t+1}−p_t| / mean|h_{t+1}−h_t|
  testA ledge·wall 짝: p 가 실제 미래 (pos) 와 이어 간 미래 (imp, 부유·통과) 중 어디에 가까운가 — frac(|p−h_imp| < |p−h_pos|)

  srun -w vll6 -c 16 --mem 128G python auto_research/scripts/h23_analyze_v3.py
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]
IN_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h23"
OUT = ROOT / "auto_research/exp_results/h23"
G, CELL, NL = 16, 18.0, 12
FREE = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc")
DBINS = [(0, 18), (18, 36), (36, 54), (54, 72), (72, 108), (108, 999)]


def arr(s):
    return np.array([float(v) for v in s.split()], dtype=np.float32)


def box3(X):
    """X[..., 256, D] → 3×3 창 평균 (가장자리는 있는 칸만) [..., 16, 16, D]."""
    sh = X.shape[:-2]; D = X.shape[-1]
    Y = X.reshape(*sh, G, G, D)
    acc = np.zeros_like(Y); cnt = np.zeros((G, G), np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys = slice(max(0, dy), G + min(0, dy)); yd = slice(max(0, -dy), G + min(0, -dy))
            xs = slice(max(0, dx), G + min(0, dx)); xd = slice(max(0, -dx), G + min(0, -dx))
            acc[..., yd, xd, :] += Y[..., ys, xs, :]
            cnt[yd, xd] += 1
    return acc / cnt[:, :, None]


def ccos(a, b):
    na = np.linalg.norm(a, axis=-1); nb = np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(na * nb, 1e-8)


def load_index(ids):
    rows = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
    R = [rows[v] for v in ids]
    X = np.stack([arr(r["px_x_by_sample"]) for r in R]); Y = np.stack([arr(r["px_y_by_sample"]) for r in R])
    ok = (np.stack([arr(r["in_frame_by_sample"]) for r in R]) > 0) & (np.stack([arr(r["visible_by_sample"]) for r in R]) > 0) \
        & (np.stack([arr(r["ignore_by_sample"]) for r in R]) == 0)
    return R, X, Y, ok


def tub_xy(X, Y, f0):
    return np.stack([(X[:, f0] + X[:, f0 + 1]) / 2, (Y[:, f0] + Y[:, f0 + 1]) / 2], -1)


def cell(xy):
    return np.clip(np.floor(xy / CELL).astype(int), 0, G - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=IN_DEFAULT)
    ap.add_argument("--windows", nargs="*", default=None)
    a = ap.parse_args()
    I = Path(a.inp); OUT.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(I / "meta.json"))
    ids, n, split = meta["video_ids"], meta["n"], meta["split"]
    scen = np.array(meta["scenario"]); plaus = np.array(meta["plausible"]) == "1"
    R, X, Y, OK = load_index(ids)
    wins = [tuple(w) for w in meta["windows"]]
    if a.windows:
        want = {tuple(int(v) for v in w.split("x")) for w in a.windows}
        wins = [w for w in wins if w in want]
    rng = np.random.default_rng(0)
    res = {}
    for C, P in wins:
        S, Tc, Stot = P // 2, C // 2, (C + P) // 2
        # ---- 평균 (가능 clip) → 3×3 창 평균 지도
        sums = [np.load(f) for f in sorted(I.glob(f"sums_C{C}_P{P}_r*.npz"))]
        N = sum(int(s["n"]) for s in sums)
        mp_ = sum(s["sum_p"] for s in sums) / N                      # (12,S,256,D)
        mh_ = sum(s["sum_h"] for s in sums) / N                      # (Stot,256,D)
        sq_p = sum(s["sumsq_p"] for s in sums) / N; sq_h = sum(s["sumsq_h"] for s in sums) / N
        trp = (sq_p - (mp_ ** 2).sum(-1)).sum(-1)                   # (12,S)
        trh = (sq_h - (mh_ ** 2).sum(-1)).sum(-1)                   # (Stot,)
        Mp, Mh = box3(mp_), box3(mh_)
        del mp_, mh_, sums
        vec = np.load(I / f"vec_C{C}_P{P}.npy", mmap_mode="r")
        l1 = np.load(I / f"l1_C{C}_P{P}.npy"); loc = np.load(I / f"loc_C{C}_P{P}.npy")
        step = np.load(I / f"step_C{C}_P{P}.npy"); conv = np.load(I / f"conv_C{C}_P{P}.npy")
        last_xy = tub_xy(X, Y, split - 2); last_c = cell(last_xy)
        prev_xy = tub_xy(X, Y, split - 4) if C >= 4 else last_xy
        v_last = last_xy - prev_xy                                   # px / 튜블릿
        okT = OK[:, split - 2] & OK[:, split - 1]
        cols = defaultdict(list)                                    # 한 줄 = (clip, slot)
        for t in range(S):
            f0 = split + 2 * t
            fxy = tub_xy(X, Y, f0); fc = cell(fxy)
            cv_c = cell(last_xy + v_last * (t + 1))
            dist = np.linalg.norm(fxy - last_xy, axis=-1)
            cheb = np.abs(fc - last_c).max(-1)
            valid = okT & OK[:, f0] & OK[:, f0 + 1]
            V = np.asarray(vec[:, t], dtype=np.float32)            # (n,29,D)
            ly, lx = fc[:, 1], fc[:, 0]; Ty, Tx = last_c[:, 1], last_c[:, 0]
            pWt = V[:, 0:12] - Mp[:, t, ly, lx].transpose(1, 0, 2)          # (n,12,D)
            pWT = V[:, 12:24] - Mp[:, t, Ty, Tx].transpose(1, 0, 2)
            htWt = V[:, 24] - Mh[Tc + t, ly, lx]; htWT = V[:, 25] - Mh[Tc + t, Ty, Tx]
            hTWt = V[:, 26] - Mh[Tc - 1, ly, lx]; hTWT = V[:, 27] - Mh[Tc - 1, Ty, Tx]
            perm = np.arange(n)                                     # 섞은 대조: 같은 시나리오 안에서 h 쪽 clip 을 섞는다
            for sc in np.unique(scen):
                ix = np.where(scen == sc)[0]; perm[ix] = rng.permutation(ix)
            hit = lambda c: (np.abs(loc[:, t, :, :2][..., ::-1] - c[:, None, :]).max(-1) <= 1)   # loc (y,x) → (x,y)
            m = valid
            cols["t"].append(np.full(m.sum(), t)); cols["b"].append(np.where(m)[0])
            cols["scen"].append(scen[m]); cols["plaus"].append(plaus[m]); cols["dist"].append(dist[m]); cols["cheb"].append(cheb[m])
            cols["A"].append(ccos(pWt, htWt[:, None])[m]); cols["A2"].append(ccos(pWt, hTWt[:, None])[m])
            cols["S"].append(ccos(pWT, hTWT[:, None])[m]); cols["B"].append(ccos(pWT, htWT[:, None])[m])
            cols["Ash"].append(ccos(pWt, htWt[perm][:, None])[m])
            cols["ht"].append(hit(fc)[m]); cols["hl"].append(hit(last_c)[m]); cols["hc"].append(hit(cv_c)[m])
            cols["contrast"].append((loc[:, t, :, 2] - loc[:, t, :, 3])[m]); cols["l1"].append(l1[:, t][m]); cols["conv"].append(conv[:, t][m])
            del V, pWt, pWT
        Q = {k: np.concatenate(v) for k, v in cols.items()}
        MET = ("A", "A2", "S", "B", "Ash", "ht", "hl", "hc", "contrast", "conv", "l1")

        def agg(mask, keys=MET):
            k_ = int(mask.sum())
            if k_ == 0:
                return None
            o = {"n": k_}
            for k in keys:
                o[k] = Q[k][mask].astype(np.float64).mean(0).round(4).tolist()
            return o
        free = np.isin(Q["scen"], FREE); moved = (Q["cheb"] >= 3) & Q["plaus"]
        W = {}
        for sc in list(np.unique(scen)) + ["FREE"]:
            ms = free if sc == "FREE" else (Q["scen"] == sc)
            for t in range(S):
                mt = ms & (Q["t"] == t)
                W[f"moved|{sc}|t{t}"] = agg(mt & moved)
                W[f"all|{sc}|t{t}"] = agg(mt & Q["plaus"], keys=("ht", "hl", "hc", "contrast", "conv", "l1"))
        for (d0, d1) in DBINS:
            md = free & (Q["dist"] >= d0) & (Q["dist"] < d1)
            for t in range(S):
                W[f"dist|{d0}-{d1}|t{t}"] = agg(md & (Q["t"] == t) & moved)
                W[f"distall|{d0}-{d1}|t{t}"] = agg(md & (Q["t"] == t) & Q["plaus"], keys=("ht", "hl", "hc", "contrast"))
        np.savez_compressed(OUT / f"rows_C{C}_P{P}.npz", **Q)
        # D_t · V_t
        W["diversity"] = {"trp": trp.tolist(), "trh_future": trh[Tc:].tolist(),
                          "D": (trp / trh[Tc:][None]).tolist()}
        stp = np.nanmean(np.where(plaus[:, None, None], step, np.nan), 0)      # (S,13)
        W["progress"] = {"V": (stp[:, :12] / stp[:, 12:13]).T.tolist(), "step_h": stp[:, 12].tolist()}
        # test A
        TA = {}
        for sc in ("ledge", "wall"):
            pid = defaultdict(dict)
            for b in np.where(scen == sc)[0]:
                pid[R[b]["pair_id"]]["pos" if plaus[b] else "imp"] = b
            pr = [(d["pos"], d["imp"]) for d in pid.values() if "pos" in d and "imp" in d]
            if not pr:
                TA[sc] = None; continue
            ps, im = np.array([x for x, _ in pr], int), np.array([y for _, y in pr], int)
            same = max(float(np.abs(np.asarray(vec[ps, :, 12:24], np.float32) - np.asarray(vec[im, :, 12:24], np.float32)).max()), 0.0) if len(pr) else None
            closer_imp = (l1[im, :, :12] < l1[ps, :, :12]).mean(0)                # (S,12)
            TA[sc] = {"n_pair": len(pr), "p_pair_maxdiff": same, "frac_closer_imp": closer_imp.round(4).tolist(),
                      "mean_l1_pos": l1[ps, :, 11].mean(0).round(4).tolist(), "mean_l1_imp": l1[im, :, 11].mean(0).round(4).tolist()}
        W["testA"] = TA
        res[f"C{C}_P{P}"] = W
        print(f"C{C}_P{P}: rows {len(Q['t'])} moved {int(moved.sum())}  testA {({k: (v or {}).get('p_pair_maxdiff') for k, v in TA.items()})}", flush=True)
    json.dump(res, open(OUT / "h23_v3.json", "w"))
    print("saved", OUT / "h23_v3.json")


if __name__ == "__main__":
    main()
