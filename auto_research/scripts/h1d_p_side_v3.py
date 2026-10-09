#!/usr/bin/env python3
"""H1d — p 쪽: 같은 (v_T, x_T) 에서 p 의 물체 변위가 법칙에 따라 얼마나 달라지나 + p 미래 토큰 probe (적대 검증 2026-09-24 대응).

왜: H1 원문은 "진실 칸 vs 등속 칸 적중" 으로 "p 는 가속을 따르지 않는다" 고 했다. 그런데 (i) p 는 flat_v 에서도 진실보다
   뒤처지고 (공통 지연), (ii) 같은 속도에서 x_T 가 법칙마다 다르며 (flat_a 가 오른쪽, flat_d 가 왼쪽), (iii) flat_v 안에서도
   p 변위가 x_T 에 따라 변한다. 그래서 **flat_v 로 맞춘 기준 모형을 (v_T, x_T) 에서 빼고** 법칙 효과를 잰다.

B. 위치 (v3_h23 loc, 자 없는 argmax — 템플릿 = h_t 의 진실 칸 토큰, 18 px 칸):
   슬롯 t (= T 뒤 k = t+1 튜블릿) 마다, 진행 방향 부호 dir 로 정렬해
     D_true = dir·(x_{T+k} − x_T),   D_cv = |v_T|·k,   r_k = D_true − D_cv   (참 가속 효과)
     D_p    = dir·(x̂_p − x_T),       x̂_p = (argmax 열 + 0.5)·18
   궤적 (law|primary|secondary, 42 개) 마다 28 복사본 평균 → flat_v 14 궤적으로 기준 모형 D_p ≈ f(|v_T|, x_T) 를 맞추고
     효과_p(궤적) = D̄_p − f(|v_T|, x_T)     (flat_a · flat_d 궤적)
     효과_true(궤적) = r_k                     (flat_v 에서 D_true = D_cv 가 정확히 성립하므로 기준이 따로 필요 없다)
     비율 = Σ효과_p / Σ효과_true  (슬롯 3–9 합침; 슬롯별도)
   f: main = [1, |v|, x_T] (선형), 민감도 = [1, |v|, x_T, |v|·x_T], 대조 = [1, |v|] (x_T 통제 없음 — 교락 크기).
   95 % CI = 궤적 bootstrap (flat_v 기준 궤적 · flat_a · flat_d 궤적을 각각 복원 추출, 2000 회).
   민감도: 물체다운 슬롯만 (contrast = max−median cos ≥ 그 창 슬롯 0–2 의 10 백분위, 평지 세 법칙 기준).
   기술: flat_v 안 x_T 기울기 ∂D_p/∂x_T, 공통 지연 (flat_v 의 D_p − D_true), 슬롯 6–9 합친 진실 칸 · 등속 칸 적중과 귀무
   (귀무 = 같은 법칙 **다른 궤적** clip 의 진실 칸에 드는 비율).

C. p 미래 토큰 probe (v3_h23 vec, 궤적 묶음 split — h1c_traj_split.evaluate 그대로):
   특징 (1 튜블릿 × 1280, 3×3 창 평균):  LN(z)_T@W_T (문맥 끝 z, 같은 창 기준),  p^(L0/L6/L11)_t@W_T (마지막으로 본 자리의 미래 토큰 —
   자리가 문맥으로만 정해져 누수 없음).  **해석하지 않는 누수 특징** (json 에만): h_T@W_T · h_T@W_t · h_t@W_t (h 는 미래 프레임까지 본
   창 전체 target encoder), p^(L11)_t@W_t (진실 미래 자리 — 자리 자체가 x_{T+k} 를 알려 준다; 깨끗한 자리 대조가 없다).
   표적 = h1c 와 같음 (r4 ↔ 슬롯 3, r8 ↔ 슬롯 7, poly2 잔차).

  auto_research/scripts/srun6.sh 16 64G python auto_research/scripts/h1d_p_side_v3.py
  → exp_results/h1/h1d_p_side.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1c_traj_split import load_flat, evaluate, tub, LAWS, OUT  # noqa: E402
import csv  # noqa: E402

I23 = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h23")
ROOT = Path(__file__).resolve().parents[2]
CELL = 18.0


def arr(s):
    return np.array([float(v) for v in s.split()], dtype=np.float64)


def ok_full(ids):
    rows = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
    R = [rows[v] for v in ids]
    return ((np.stack([arr(r["in_frame_by_sample"]) for r in R]) > 0) & (np.stack([arr(r["visible_by_sample"]) for r in R]) > 0)
            & (np.stack([arr(r["ignore_by_sample"]) for r in R]) == 0))


def design(spd, xT, kind):
    if kind == "lin":
        return np.c_[np.ones(len(spd)), spd, xT]
    if kind == "int":
        return np.c_[np.ones(len(spd)), spd, xT, spd * xT]
    if kind == "quad":                                          # 속도 포화 (거리 한계) 를 흡수
        return np.c_[np.ones(len(spd)), spd, spd ** 2, xT]
    return np.c_[np.ones(len(spd)), spd]


def part_b(D, split, windows, layers, n_boot=2000):
    meta = json.load(open(I23 / "meta.json")); OKf = ok_full(meta["video_ids"])
    idx, law, tid = D["idx"], D["law"], D["tid"]; X = D["X"]; ok = OKf[idx]
    dirn = D["sgn"]; xT = D["xT"]; spd = D["spd"]
    ut = np.arange(D["n_traj"]); lot = D["law_of_tid"]
    tl = {l: np.array([t for t in ut if lot[t] == l]) for l in LAWS}
    t_x = np.array([xT[tid == t].mean() for t in ut]); t_v = np.array([spd[tid == t].mean() for t in ut])
    res = {"traj_state": {l: {"xT": t_x[tl[l]].round(1).tolist(), "spd": t_v[tl[l]].round(2).tolist()} for l in LAWS}}
    # 같은 속도에서 x_T 가 법칙마다 얼마나 다른가 (flat_v 의 x_T ~ |v| 선형과 비교)
    wv = np.linalg.lstsq(design(t_v[tl["flat_v"]], None, "spd"), t_x[tl["flat_v"]], rcond=None)[0]
    res["xT_offset_vs_flat_v_at_same_speed"] = {l: round(float((t_x[tl[l]] - design(t_v[tl[l]], None, "spd") @ wv).mean()), 1) for l in LAWS}
    rng = np.random.default_rng(0)
    for C, P in windows:
        loc = np.load(I23 / f"loc_C{C}_P{P}.npy")[idx]            # (n,S,12,4)
        S = P // 2
        con_all = loc[:, :, :, 2] - loc[:, :, :, 3]
        W = {}
        for L in layers:
            px = (loc[:, :, L, 1] + 0.5) * CELL                    # (n,S)
            con = con_all[:, :, L]
            okT = ok[:, split - 2] & ok[:, split - 1]
            thr = np.nanpercentile(np.where(okT[:, None], con[:, 0:3], np.nan), 10)
            slots = {}
            tabs = {"all": {}, "objlike": {}}
            for t in range(S):
                f0 = split + 2 * t; k = t + 1
                valid = okT & ok[:, f0] & ok[:, f0 + 1]
                Dt = dirn * (tub(X, f0) - xT); Dp = dirn * (px[:, t] - xT); Dcv = spd * k
                for sel_name, sel in (("all", valid), ("objlike", valid & (con[:, t] >= thr))):
                    tm = {}
                    for q in ut:
                        m = sel & (tid == q)
                        if m.sum() >= 3:
                            tm[q] = (Dp[m].mean(), Dt[m].mean(), Dcv[m].mean(), int(m.sum()))
                    tabs[sel_name][t] = tm
            # ---- 효과와 비율 (슬롯 3–9 합침 + 슬롯별)
            vv, xv = t_v[tl["flat_v"]], t_x[tl["flat_v"]]
            in_sup = lambda q: (vv.min() - 1e-6 <= t_v[q] <= vv.max() + 1e-6) and (xv.min() - 1e-6 <= t_x[q] <= xv.max() + 1e-6)

            def effects(tm_by_slot, slot_list, kind, samp=None, support=False):
                """samp: {law: 궤적 배열 (복원 추출)} 또는 None. support: flat_v 의 (|v|, x_T) 범위 안 궤적만 (외삽 없음).
                반환 {law: (Σ효과_p, Σ효과_true, n_cell)}."""
                samp = samp or tl
                out = {}
                for l in ("flat_a", "flat_d"):
                    ep, et, nc = 0.0, 0.0, 0
                    for t in slot_list:
                        tm = tm_by_slot[t]
                        ref = [q for q in samp["flat_v"] if q in tm]
                        tgt_ = [q for q in samp[l] if q in tm and (in_sup(q) or not support)]
                        if len(ref) < 5 or len(tgt_) == 0:
                            continue
                        A = design(t_v[ref], t_x[ref], kind); w = np.linalg.lstsq(A, np.array([tm[q][0] for q in ref]), rcond=None)[0]
                        pred = design(t_v[tgt_], t_x[tgt_], kind) @ w
                        ep += float((np.array([tm[q][0] for q in tgt_]) - pred).sum())
                        et += float(sum(tm[q][1] - tm[q][2] for q in tgt_)); nc += len(tgt_)
                    out[l] = (ep, et, nc)
                return out

            SL = list(range(3, 10))
            for sel_name, kind, sup in (("all", "lin", False), ("all", "int", False), ("all", "quad", False), ("all", "spd", False),
                                        ("all", "lin", True), ("all", "quad", True), ("objlike", "lin", False)):
                if True:
                    base = effects(tabs[sel_name], SL, kind, support=sup)
                    B = {l: [] for l in ("flat_a", "flat_d")}
                    for _ in range(n_boot):
                        smp = {l: rng.choice(tl[l], len(tl[l])) for l in LAWS}
                        e = effects(tabs[sel_name], SL, kind, smp, support=sup)
                        for l in B:
                            ep, et, nc = e[l]
                            if nc:
                                B[l].append((ep / nc, et / nc, ep / et if et else np.nan))
                    o = {}
                    for l in ("flat_a", "flat_d"):
                        ep, et, nc = base[l]; b = np.array(B[l])
                        ci = lambda j: [round(float(np.nanpercentile(b[:, j], 2.5)), 3), round(float(np.nanpercentile(b[:, j], 97.5)), 3)]
                        o[l] = {"n_traj_slot": nc, "effect_p_px": round(ep / nc, 2), "effect_p_ci": ci(0),
                                "effect_true_px": round(et / nc, 2), "effect_true_ci": ci(1),
                                "ratio": round(ep / et, 3), "ratio_ci": ci(2)}
                    slots[f"{sel_name}|{kind}{'|support' if sup else ''}|slots3-9"] = o
            slots["support_traj"] = {l: [int(q) for q in tl[l] if in_sup(q)] for l in ("flat_a", "flat_d")}
            for sel_name in ("all", "objlike"):
                per = {}
                for t in range(S):
                    e = effects(tabs[sel_name], [t], "lin")
                    per[t] = {l: {"effect_p": round(e[l][0] / e[l][2], 2), "effect_true": round(e[l][1] / e[l][2], 2), "n": e[l][2]}
                              if e[l][2] else None for l in e}
                slots[f"{sel_name}|lin|per_slot"] = per
            # ---- 기술: flat_v 안 x_T 기울기, 공통 지연
            desc = {}
            for t in (3, 5, 7, 9):
                tm = tabs["all"][t]; ref = [q for q in tl["flat_v"] if q in tm]
                A = design(t_v[ref], t_x[ref], "lin"); w = np.linalg.lstsq(A, np.array([tm[q][0] for q in ref]), rcond=None)[0]
                lag = {l: round(float(np.mean([tm[q][0] - tm[q][1] for q in tl[l] if q in tm])), 1) for l in LAWS}
                desc[t] = {"flat_v_dDp_dxT": round(float(w[2]), 3), "flat_v_dDp_dv": round(float(w[1]), 2),
                           "lag_Dp_minus_Dtrue": lag,
                           "Dtrue_mean": {l: round(float(np.mean([tm[q][1] for q in tl[l] if q in tm])), 1) for l in LAWS},
                           "Dp_mean": {l: round(float(np.mean([tm[q][0] for q in tl[l] if q in tm])), 1) for l in LAWS}}
            slots["describe"] = desc
            # ---- 칸 적중 (슬롯 6–9 합침) + 귀무 (같은 법칙 다른 궤적)
            hits = {}
            am_x = loc[:, :, L, 1].astype(int); am_y = loc[:, :, L, 0].astype(int)
            for l in LAWS:
                h_t, h_c, h_n, nn = 0, 0, 0, 0
                for t in range(6, 10):
                    f0 = split + 2 * t; valid = okT & ok[:, f0] & ok[:, f0 + 1] & (law == l)
                    fc = np.clip(np.floor(tub(X, f0) / CELL).astype(int), 0, 15)
                    cv = np.clip(np.floor((xT + dirn * spd * (t + 1)) / CELL).astype(int), 0, 15)
                    ii = np.where(valid)[0]
                    # 귀무: 같은 법칙 다른 궤적의 진실 칸 (x 열만; 평지라 y 는 같은 행)
                    other = np.array([rng.choice(ii[tid[ii] != tid[i]]) for i in ii])
                    h_t += int((np.abs(am_x[ii, t] - fc[ii]) <= 1).sum()); h_c += int((np.abs(am_x[ii, t] - cv[ii]) <= 1).sum())
                    h_n += int((np.abs(am_x[ii, t] - fc[other]) <= 1).sum()); nn += len(ii)
                hits[l] = {"n": nn, "hit_true": round(h_t / nn, 3), "hit_cv": round(h_c / nn, 3), "hit_null_other_traj": round(h_n / nn, 3)}
            slots["cell_hits_slots6-9_xonly"] = hits
            slots["objlike_thr"] = round(float(thr), 4)
            W[f"L{L}"] = slots
            m = slots["all|lin|slots3-9"]; s_ = slots["all|spd|slots3-9"]
            print(f"C{C}_P{P} L{L}: flat_a p {m['flat_a']['effect_p_px']:+.1f} {m['flat_a']['effect_p_ci']} true {m['flat_a']['effect_true_px']:+.1f} "
                  f"ratio {m['flat_a']['ratio']:+.2f} {m['flat_a']['ratio_ci']} | flat_d p {m['flat_d']['effect_p_px']:+.1f} {m['flat_d']['effect_p_ci']} "
                  f"true {m['flat_d']['effect_true_px']:+.1f} ratio {m['flat_d']['ratio']:+.2f} {m['flat_d']['ratio_ci']} || speed-only ratio a {s_['flat_a']['ratio']:+.2f} d {s_['flat_d']['ratio']:+.2f}", flush=True)
        res[f"C{C}_P{P}"] = W
    return res


def part_c(D, windows, n_half=20, n_xfit=3):
    idx = D["idx"]; n = len(idx)
    res = {}
    for C, P in windows:
        vec = np.load(I23 / f"vec_C{C}_P{P}.npy", mmap_mode="r")
        V = np.asarray(vec[idx], np.float32)[:, [3, 7]]          # (n, 2, 29, D)
        feats = {"lnzT@WT": V[:, 0, 28], "hT@WT": V[:, 0, 27]}
        for j, t in enumerate((3, 7)):
            for nm, ch in (("pL0@WT", 12), ("pL6@WT", 18), ("pL11@WT", 23), ("pL11@Wt", 11), ("hT@Wt", 26), ("ht@Wt", 24)):
                feats[f"{nm}|t{t}"] = V[:, j, ch]
        del V
        W = {}
        for name, F in feats.items():
            r = evaluate(F.astype(np.float64), D, n_half=n_half, n_xfit=n_xfit)
            W[name] = r
            print(f"C{C}_P{P} {name:12s} half r4 {r['half']['r4']['mean']:+.3f}±{r['half']['r4']['sd']:.3f} r8 {r['half']['r8']['mean']:+.3f}±{r['half']['r8']['sd']:.3f}"
                  f" | xfit r4 {r['xfit']['r4']:+.3f} {r['xfit']['ci95']['r4']} r8 {r['xfit']['r8']:+.3f} {r['xfit']['ci95']['r8']} v {r['xfit']['v']:+.3f}", flush=True)
        res[f"C{C}_P{P}"] = W
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="bc")
    a = ap.parse_args()
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    split = json.load(open(I23 / "meta.json"))["split"]
    D = load_flat(split)
    out_p = OUT / "h1d_p_side.json"
    res = json.load(open(out_p)) if out_p.exists() else {}
    if "b" in a.part:
        res["B_position"] = part_b(D, split, [(8, 32), (16, 32), (32, 32)], layers=(11, 6))
        json.dump(res, open(out_p, "w"), indent=1)
        print(f"B done {time.time() - t0:.0f}s", flush=True)
    if "c" in a.part:
        res["C_probe"] = part_c(D, [(16, 32), (32, 32)])
        json.dump(res, open(out_p, "w"), indent=1)
    print("saved", out_p, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
