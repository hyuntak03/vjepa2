#!/usr/bin/env python3
"""H23 적대 검증 보정 (1/2) — vll6 캐시 (loc · l1 · vec) 에서 행 단위 귀무 · 공통 문턱 물체다움 · 법칙별 진행률 · 검정 A 복사 기준선 (CPU).

왜: H23 문서 (H23_PREDICTOR_LAYERS_TRAJECTORY_2026-09-24.md) 의 적대 검증에서
  (i)  시간·거리 회귀가 위치 사전확률 (귀무) 로 보정되지 않았다 → 행마다 정확한 귀무를 만들어 둔다 (회귀는 h23b_regress_tables.py).
  (ii) 물체다움 문턱이 층마다 달라 (자기 슬롯 0–2 의 10 백분위) "뒤층이 지운다" 가 문턱 인공물일 수 있다 → 층 공통 문턱으로 다시 잰다.
  (iii) 진행률 (뒤처짐) 이 법칙별·등속 외삽 대비로 확인되지 않았고 n 이 없었다 → 법칙별 n · 등속 외삽 진행률을 같이 낸다.
  (iv) 검정 A (ledge·wall) 에 '진화 없는 복사' 기준선이 없었다 → 복사 LN(z)_T · 복사 h_T 를 같은 식으로 잰다. 쌍의 위치가 처음 갈리는 프레임도 잰다.

식 (argmax·contrast 는 h23_extract_v3.py 의 loc: 템플릿 = h_t 진실 칸 토큰, p^(l)_t 256 토큰과 cos):
  적중   hit_l(b,t)  = [ cheb(argmax_l(b,t), 진실칸(b,t)) ≤ 1 ]
  귀무   null_l(b,t) = Σ_{b'∈R(s,t), b'≠b} [ cheb(argmax_l(b,t), 진실칸(b',t)) ≤ 1 ] / (|R(s,t)| − 1)
         R(s,t) = 같은 법칙 s · 슬롯 t 에서 유효한 (화면 안 · 보임 · ignore 아님) 가능 clip 전부 — 정확한 전쌍 비율.
         (옛 h23_null_hit.py 는 같은 시나리오 안 무작위 순열 1 회였다.)
  물체다움 obj_l(t; θ) = mean_b [ contrast_l(b,t) ≥ θ ],  contrast = max cos − median cos
         θ_l = 10 백분위 of contrast_l (자유 운동 가능 clip, 슬롯 0–2).  θ ∈ {θ_own(l) (옛 정의), θ_L6, θ_L11}  ← 공통 문턱 = 모든 층에 같은 θ
  진행률 prog(x) = (x − L)·(T − L) / |T − L|²   x = argmax 칸 (p) 또는 등속 외삽 칸 (cv),  L = 마지막 본 칸, T = 진실 칸  (|T − L| ≥ 3 칸만)
  검정 A frac_closer_imp(x) = mean_pair [ |x − h_imp| < |x − h_pos| ]   x = p (L11) / 복사 LN(z)_T / 복사 h_T   (l1 열 11 / 13 / 12, 전 토큰 mean|·|)

산출 (auto_research/exp_results/h23b/):
  rows_null.npz      자유 운동 · 가능 · 유효 행 (16 창): C P b t scen dist hit(12) null(12) contrast(12) am(12,2) tc lc cvc
  objectness.json    P32 4 창: 층별 θ, 물체다움 [θ 종류][층][슬롯], contrast 중앙값, L11 − L6 차이의 clip bootstrap CI
  progress_law.json  C16/P32 · C32/P32, 층 L11 · L6: 법칙별 · 슬롯별 n, 물체다운 n, 진행률 중앙값 (p · cv), 마지막 자리 근처 비율
  testA_copy.json    16 창 × ledge · wall: p · 복사 두 개의 frac_closer_imp, p − 복사 LN(z)_T 차이 CI, 상대 margin, 갈림 프레임, 짝 동일성 검사 범위

  auto_research/scripts/srun6.sh 8 64G /data/hyuntak/anaconda3/envs/vjepa2/bin/python auto_research/scripts/h23b_null_objectness_testA.py
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h23_analyze_v3 import load_index, tub_xy, cell, FREE, G  # noqa: E402

I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h23")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/h23b"
NL = 12
RNG = np.random.default_rng(0)


def box_sum(cnt):
    """cnt (16,16) → 각 칸의 3×3 이웃 합 (Chebyshev ≤ 1)."""
    out = np.zeros_like(cnt)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys = slice(max(0, dy), G + min(0, dy)); yd = slice(max(0, -dy), G + min(0, -dy))
            xs = slice(max(0, dx), G + min(0, dx)); xd = slice(max(0, -dx), G + min(0, -dx))
            out[yd, xd] += cnt[ys, xs]
    return out


def boot_ci(x, reps=1000):
    """x (n,) clip 단위 값 → 평균의 95 % bootstrap CI."""
    x = np.asarray(x, float)
    if len(x) < 5:
        return [float("nan"), float("nan")]
    ix = RNG.integers(0, len(x), (reps, len(x)))
    m = x[ix].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(I / "meta.json"))
    ids, n, split = meta["video_ids"], meta["n"], meta["split"]
    scen = np.array(meta["scenario"]); plaus = np.array(meta["plausible"]) == "1"
    R, X, Y, OK = load_index(ids)
    wins = [tuple(w) for w in meta["windows"]]
    SCI = {s: i for i, s in enumerate(FREE)}
    free = np.isin(scen, FREE) & plaus
    print(f"n={n} clips, free plausible={free.sum()}, windows={wins}", flush=True)

    # ------------------------------------------------------------------ 1. 행 단위 귀무 (16 창)
    rows = defaultdict(list)
    last_xy = tub_xy(X, Y, split - 2); last_c = cell(last_xy)
    prev_xy = tub_xy(X, Y, split - 4); v_last = last_xy - prev_xy
    okT = OK[:, split - 2] & OK[:, split - 1]
    for C, P in wins:
        S = P // 2
        loc = np.load(I / f"loc_C{C}_P{P}.npy")
        for t in range(S):
            f0 = split + 2 * t
            fxy = tub_xy(X, Y, f0); fc = cell(fxy)
            cv_c = cell(last_xy + v_last * (t + 1))
            dist = np.linalg.norm(fxy - last_xy, axis=-1)
            valid = okT & OK[:, f0] & OK[:, f0 + 1]
            m = valid & free
            am = loc[:, t, :, :2][..., ::-1].astype(int)                  # (n,12,2) (x,y)
            hit = np.abs(am - fc[:, None]).max(-1) <= 1                    # (n,12)
            null = np.full((n, NL), np.nan, np.float32)
            for s in FREE:
                ref = m & (scen == s)
                k = int(ref.sum())
                if k < 2:
                    continue
                cnt = np.zeros((G, G), np.float64)
                np.add.at(cnt, (fc[ref, 1], fc[ref, 0]), 1)
                nb = box_sum(cnt)
                ix = np.where(ref)[0]
                nbv = nb[am[ix, :, 1], am[ix, :, 0]]                          # (k,12)
                null[ix] = (nbv - hit[ix]) / (k - 1)
            con = loc[:, t, :, 2] - loc[:, t, :, 3]
            ix = np.where(m)[0]
            rows["C"].append(np.full(len(ix), C, np.int8)); rows["P"].append(np.full(len(ix), P, np.int8))
            rows["b"].append(ix.astype(np.int16)); rows["t"].append(np.full(len(ix), t, np.int8))
            rows["scen"].append(np.array([SCI[s] for s in scen[ix]], np.int8)); rows["dist"].append(dist[ix].astype(np.float32))
            rows["hit"].append(hit[ix]); rows["null"].append(null[ix]); rows["contrast"].append(con[ix].astype(np.float16))
            rows["am"].append(am[ix].astype(np.int8)); rows["tc"].append(fc[ix].astype(np.int8))
            rows["lc"].append(last_c[ix].astype(np.int8)); rows["cvc"].append(cv_c[ix].astype(np.int8))
        del loc
        print(f"  rows C{C}_P{P} done", flush=True)
    Q = {k: np.concatenate(v) for k, v in rows.items()}
    assert not np.isnan(Q["null"]).any()
    np.savez_compressed(OUT / "rows_null.npz", **Q, free_laws=np.array(FREE))
    print("saved rows_null.npz", len(Q["t"]), flush=True)

    # ------------------------------------------------------------------ 2. 공통 문턱 물체다움 (P32 창) + 3. 법칙별 진행률
    lastc = cell(tub_xy(X, Y, split - 2)).astype(float)                 # h23_progress.py 와 같은 L (튜블릿 30)
    OBJ, PROG = {}, {}
    for C, P in [w for w in wins if w[1] == 32]:
        S = P // 2
        loc = np.load(I / f"loc_C{C}_P{P}.npy")
        con = loc[..., 2] - loc[..., 3]                                   # (n,S,12)
        th = [float(np.percentile(con[free][:, :3, l], 10)) for l in range(NL)]
        kinds = {"own": th, "L6": [th[6]] * NL, "L11": [th[11]] * NL}
        tab = {k: [[None] * S for _ in range(NL)] for k in kinds}
        med = [[None] * S for _ in range(NL)]; ns = [0] * S
        diff = {k: {} for k in ("L6", "L11")}
        for t in range(S):
            f0 = split + 2 * t
            T = cell(tub_xy(X, Y, f0)).astype(float); d = T - lastc; dd = (d ** 2).sum(1)
            ok = free & OK[:, f0] & OK[:, f0 + 1] & (np.sqrt(dd) >= 3)
            ns[t] = int(ok.sum())
            if ok.sum() < 30:
                continue
            for l in range(NL):
                c = con[ok, t, l]
                med[l][t] = round(float(np.median(c)), 4)
                for k, thr in kinds.items():
                    tab[k][l][t] = round(float((c >= thr[l]).mean()), 4)
            for k in ("L6", "L11"):
                thr = kinds[k][0]
                dl = (con[ok, t, 11] >= thr).astype(float) - (con[ok, t, 6] >= thr).astype(float)   # 같은 clip 짝 차이
                diff[k][t] = dict(mean=round(float(dl.mean()), 4), ci=boot_ci(dl))
        OBJ[f"C{C}_P{P}"] = dict(theta=[round(x, 4) for x in th], n=ns, obj=tab, median_contrast=med, diff_L11_minus_L6=diff)
        # 법칙별 진행률 (C16/P32 · C32/P32)
        if C in (16, 32):
            for L in (11, 6):
                thr = th[L]; out = {}
                for s in list(FREE) + ["FREE"]:
                    ms = free if s == "FREE" else (free & (scen == s))
                    rr = []
                    for t in range(S):
                        f0 = split + 2 * t
                        T = cell(tub_xy(X, Y, f0)).astype(float); d = T - lastc; dd = (d ** 2).sum(1)
                        ok = ms & OK[:, f0] & OK[:, f0 + 1] & (np.sqrt(dd) >= 3)
                        a = loc[:, t, L, :2][:, ::-1]
                        prog = ((a - lastc) * d).sum(1) / np.maximum(dd, 1)
                        cvc = cell(last_xy + v_last * (t + 1)).astype(float)
                        progcv = ((cvc - lastc) * d).sum(1) / np.maximum(dd, 1)
                        obj = con[:, t, L] >= thr
                        sel = ok & obj
                        rr.append(dict(t=t, n_ok=int(ok.sum()), n_obj=int(sel.sum()),
                                       objlike=round(float(obj[ok].mean()), 4) if ok.sum() else None,
                                       prog_med=round(float(np.median(prog[sel])), 3) if sel.sum() >= 5 else None,
                                       prog_q=[round(float(x), 3) for x in np.percentile(prog[sel], [25, 75])] if sel.sum() >= 5 else None,
                                       cv_prog_med=round(float(np.median(progcv[ok])), 3) if ok.sum() >= 5 else None,
                                       cv_prog_med_obj=round(float(np.median(progcv[sel])), 3) if sel.sum() >= 5 else None,
                                       near_last=round(float((prog[sel] < 0.25).mean()), 4) if sel.sum() else None,
                                       lag=round(float(((prog[sel] >= 0.25) & (prog[sel] < 0.75)).mean()), 4) if sel.sum() else None,
                                       prog_lt_cv=round(float((prog[sel] < progcv[sel]).mean()), 4) if sel.sum() else None))
                    out[s] = rr
                PROG[f"C{C}_P{P}_L{L}"] = dict(theta=thr, laws=out)
        del loc
        print(f"  objectness/progress C{C}_P{P} done", flush=True)
    json.dump(OBJ, open(OUT / "objectness.json", "w"), indent=1)
    json.dump(PROG, open(OUT / "progress_law.json", "w"), indent=1)

    # ------------------------------------------------------------------ 4. 검정 A — 복사 기준선 · 갈림 프레임 · 짝 동일성 범위
    TA = {}
    rawx = {i: np.array([float(v) for v in R[i]["px_x_by_sample"].split()]) for i in range(n) if scen[i] in ("ledge", "wall")}
    rawy = {i: np.array([float(v) for v in R[i]["px_y_by_sample"].split()]) for i in range(n) if scen[i] in ("ledge", "wall")}
    for sc in ("ledge", "wall"):
        pid = defaultdict(dict)
        for b in np.where(scen == sc)[0]:
            pid[R[b]["pair_id"]]["pos" if plaus[b] else "imp"] = b
        pr = [(d["pos"], d["imp"]) for d in pid.values() if "pos" in d and "imp" in d]
        ps = np.array([x for x, _ in pr]); im = np.array([y for _, y in pr])
        div = []
        for a, b in pr:
            dd = np.hypot(rawx[a] - rawx[b], rawy[a] - rawy[b])
            k = np.where(dd > 0.5)[0]
            div.append(int(k[0]) if len(k) else 64)
        div = np.array(div)
        TA[sc] = dict(n_pair=len(pr), diverge_frame=dict(min=int(div.min()), median=float(np.median(div)), max=int(div.max()),
                                                          hist={int(k): int(v) for k, v in zip(*np.unique(div, return_counts=True))}),
                      windows={})
        for C, P in wins:
            S = P // 2
            l1 = np.load(I / f"l1_C{C}_P{P}.npy")
            vec = np.load(I / f"vec_C{C}_P{P}.npy", mmap_mode="r")
            vp = np.asarray(vec[ps], np.float32); vi = np.asarray(vec[im], np.float32)
            same_pWT = float(np.abs(vp[:, :, 12:24] - vi[:, :, 12:24]).max())
            same_zT = float(np.abs(vp[:, :, 28] - vi[:, :, 28]).max())
            del vp, vi
            o = {}
            for key, col in (("p_L11", 11), ("copy_lnz_T", 13), ("copy_h_T", 12)):
                ci = l1[im, :, col] < l1[ps, :, col]                       # (n_pair, S)
                o[key] = dict(frac=[round(float(x), 4) for x in ci.mean(0)],
                              rel_margin=[round(float(x), 4) for x in ((l1[ps, :, col] - l1[im, :, col]) / l1[ps, :, col]).mean(0)])
            g = (l1[im, :, 11] < l1[ps, :, 11]).astype(float) - (l1[im, :, 13] < l1[ps, :, 13]).astype(float)
            o["gap_p_minus_copy_lnz"] = [dict(mean=round(float(g[:, t].mean()), 4), ci=boot_ci(g[:, t])) for t in range(S)]
            o["check_scope"] = dict(p_at_WT_maxdiff=same_pWT, lnzT_at_WT_maxdiff=same_zT,
                                    note="3x3 창 평균 (W_T, 9/256 토큰) fp16, 12 층 × 전 슬롯. 전 토큰 p 동일성은 이 캐시로 확인 불가")
            TA[sc]["windows"][f"C{C}_P{P}"] = o
            del l1
        print(f"  testA {sc} done; diverge frame {TA[sc]['diverge_frame']}", flush=True)
    json.dump(TA, open(OUT / "testA_copy.json", "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
