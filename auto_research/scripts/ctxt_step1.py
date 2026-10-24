#!/usr/bin/env python3
"""[P] 영상 문맥 vs 기준 AR — 걸음 1 (예측 블록 4) 만의 존재감 · 떨어짐 · 조건화 기울기 (CPU, ctxt_compare.py 의 공유 forward 를 읽음) (auto_research, 2026-10-22 KST).

  /venv/vjepa/bin/python auto_research/scripts/ctxt_step1.py

존재감 · 떨어짐 = vob_compare 정의 (m = 블록 0..3 같은 칸 평균). 들어오는 칸 = |U[4]| ≥ 0.5, 정지 = |U[4]|, |U[3]| < 0.25.
조건화 = [T] ft_eval_reliability 회귀, 단 x1 = 문맥 4 블록 평균 (블록 0..3), x2 = 블록 3 의 내용 출처 칸, 안정도 = cos(h3(j), h2(p)).
출력: results/ctxt_compare_step1.json · results/ctxt_compare_step1_cond.json."""
import json, os
import numpy as np

D = "/workspace/auto_research_cache/ctxt_compare"; S, G = 256, 16
R = os.path.join(os.path.dirname(__file__), "../results")
EDGES = np.array([0.35, 0.449, 0.557, 0.724]); cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.where(ok, np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), np.arange(S)), ok


def flows(src, wids):
    if src == "k710_holdout":
        U = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy", mmap_mode="r")
        return [np.asarray(U[w], np.float32).reshape(8, S, 2) for w in wids]
    U = np.load(os.path.join(R, f"natural_transport_ctx5/rows_{src}.npz"))["Uwin"].astype(np.float32)
    return [U[w].reshape(8, S, 2) for w in wids]


def cosv(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


pres, cond = {}, {}
for src in ("k710_holdout", "ht100m", "phys3d"):
    wids = json.load(open(f"{D}/{src}_meta.json"))["windows"]; Us = flows(src, wids)
    gt = np.load(f"{D}/{src}_gt.npy", mmap_mode="r"); cxs = np.load(f"{D}/{src}_ctx.npy", mmap_mode="r")
    P = {n: np.load(f"{D}/{src}_pred_{n}.npy", mmap_mode="r") for n in ("base", "ctxt")}
    acc = {n: {g: [0.0] * 4 for g in ("enter", "static")} for n in P}; rows = {n: [] for n in P}
    for i, U in enumerate(Us):
        h = np.asarray(cxs[i], np.float32); y = np.asarray(gt[i, 0], np.float32); m = h.mean(0)
        u1, u0 = np.linalg.norm(U[4], axis=1), np.linalg.norm(U[3], axis=1)
        sel_g = {"enter": u1 >= 0.5, "static": (u1 < 0.25) & (u0 < 0.25)}
        js, ok1 = cell_of(P0 + U[4]); jp, ok0 = cell_of(P0[js] + U[3][js]); sel = np.where((u1 >= 0.5) & ok1 & ok0)[0]
        if sel.size:
            x1 = m[sel]; x2 = h[3][js[sel]]; yy = y[sel]; stab = cosv(x2, h[2][jp[sel]])
            cols = [(x1 * x1).sum(-1), (x1 * x2).sum(-1), (x2 * x2).sum(-1), (x1 * yy).sum(-1), (x2 * yy).sum(-1), (yy * yy).sum(-1), stab]
        for n in P:
            p = np.asarray(P[n][i, 0], np.float32)
            for g, mk in sel_g.items():
                if mk.any():
                    d, t = p[mk] - m[mk], y[mk] - m[mk]; a = acc[n][g]
                    a[0] += np.abs(d).sum(); a[1] += np.abs(t).sum(); a[2] += (d * t).sum(); a[3] += (t * t).sum()
            if sel.size:
                ym = p[sel]; rows[n].append(np.stack(cols + [(x1 * ym).sum(-1), (x2 * ym).sum(-1)], -1).astype(np.float64))
    pres[src] = {n: {g: dict(dev=round(float(v[0] / v[1]), 3), presence=round(float(v[2] / v[3]), 3)) for g, v in acc[n].items()} for n in P}
    cond[src] = {}
    for n in P:
        st = np.concatenate(rows[n]); bins = np.digitize(st[:, 6], EDGES)
        def fit(mk, yi):
            A = np.array([[st[mk, 0].sum(), st[mk, 1].sum()], [st[mk, 1].sum(), st[mk, 2].sum()]])
            return np.linalg.solve(A, [st[mk, yi[0]].sum(), st[mk, yi[1]].sum()])
        bd = [float(fit(bins == k, (3, 4))[1]) for k in range(5)]; bm = [float(fit(bins == k, (7, 8))[1]) for k in range(5)]
        allm = np.ones(len(st), bool)
        cond[src][n] = dict(b_data=[round(x, 3) for x in bd], b_model=[round(x, 3) for x in bm], slope_data=round(bd[4] - bd[0], 3), slope_model=round(bm[4] - bm[0], 3),
                            b_model_all=round(float(fit(allm, (7, 8))[1]), 3), b_data_all=round(float(fit(allm, (3, 4))[1]), 3), n=int(len(st)))
    print(src, pres[src], cond[src], flush=True)
json.dump(pres, open(os.path.join(R, "ctxt_compare_step1.json"), "w"), indent=1, ensure_ascii=False)
json.dump(cond, open(os.path.join(R, "ctxt_compare_step1_cond.json"), "w"), indent=1, ensure_ascii=False)
