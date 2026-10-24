#!/usr/bin/env python3
"""[S] 사전 등록 KST 05:02 판정 — 영상 문맥 모델의 더 큰 운반은 자연 영상 L1 에서도 이득인가 (학습 없음, CPU).
[P] 의 공유 forward (/workspace/auto_research_cache/ctxt_compare/) 를 읽는다. 걸음 1 = 예측 블록 4 (기준 AR: 블록 0..3 → 4, 영상 문맥: z(블록 0..3) → 첫 예측).

  /venv/vjepa/bin/python auto_research/scripts/ctxt_l1_shared.py

칸 무리: 움직이는 칸 = |U[4]| ≥ 0.5 칸 (블록 4 → 3 흐름), 정지 칸 = |U[4]| < 0.1 이고 |U[3]| < 0.1. (3D 합성도 같은 RAFT 칸 흐름.)
지표 (칸 합의 비, 창 bootstrap 95 %): L1 · L1 비 (영상 문맥 / 기준) · L1 / 복사 (블록 3) · L1 / 문맥 평균 (블록 0..3) ·
  정답 쪽 사영 Σ⟨p − h_3, h_4 − h_3⟩ / Σ|h_4 − h_3|²."""
import json, os
import numpy as np

CC = "/workspace/auto_research_cache/ctxt_compare"
REPO = "/workspace/world_model/vjepa2"
FLOW = {"k710_holdout": lambda: np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy").astype(np.float32),
        "ht100m": lambda: np.load(f"{REPO}/auto_research/results/natural_transport_ctx5/rows_ht100m.npz")["Uwin"].astype(np.float32),
        "phys3d": lambda: np.load(f"{REPO}/auto_research/results/natural_transport_ctx5/rows_phys3d.npz")["Uwin"].astype(np.float32)}
COLS = ("l1_base", "l1_ctxt", "l1_copy", "l1_mean", "pn_base", "pn_ctxt", "pd", "n")
out = {}
for src in ("k710_holdout", "ht100m", "phys3d"):
    if not os.path.exists(f"{CC}/{src}.done"):
        continue
    meta = json.load(open(f"{CC}/{src}_meta.json")); U = FLOW[src]()
    PB, PC = np.load(f"{CC}/{src}_pred_base.npy", mmap_mode="r"), np.load(f"{CC}/{src}_pred_ctxt.npy", mmap_mode="r")
    GT, CX = np.load(f"{CC}/{src}_gt.npy", mmap_mode="r"), np.load(f"{CC}/{src}_ctx.npy", mmap_mode="r")
    per = {g: [] for g in ("mov", "sta", "all")}
    for i, w in enumerate(meta["windows"]):
        mg = np.linalg.norm(U[w].reshape(8, 256, 2), axis=-1)
        pb, pc, y = (np.asarray(A[i, 0], np.float32) for A in (PB, PC, GT))
        cx = np.asarray(CX[i], np.float32); cp, mu = cx[3], cx.mean(0)
        l = [np.abs(p - y).mean(-1) for p in (pb, pc, cp, mu)]
        d = y - cp
        rows = l + [((pb - cp) * d).sum(-1), ((pc - cp) * d).sum(-1), (d * d).sum(-1)]
        for g, m in (("mov", mg[4] >= 0.5), ("sta", (mg[4] < 0.1) & (mg[3] < 0.1)), ("all", np.ones(256, bool))):
            per[g].append([float(r[m].sum()) for r in rows] + [float(m.sum())])
    rng = np.random.default_rng(0)
    for g, P in per.items():
        P = np.asarray(P); W = len(P)

        def st(X):
            s = X.sum(0)
            return np.array([s[0] / s[7], s[1] / s[7], s[1] / s[0], s[0] / s[2], s[1] / s[2], s[0] / s[3], s[1] / s[3], s[4] / s[6], s[5] / s[6]])
        names = ("L1_base", "L1_ctxt", "ratio_ctxt_over_base", "base_over_copy", "ctxt_over_copy", "base_over_mean", "ctxt_over_mean", "proj_base", "proj_ctxt")
        v = st(P); bs = np.array([st(P[rng.integers(0, W, W)]) for _ in range(1000)]); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        out[f"{src}|{g}"] = {nm: [round(float(v[j]), 4), round(float(lo[j]), 4), round(float(hi[j]), 4)] for j, nm in enumerate(names)}
        out[f"{src}|{g}"]["n_cells"] = int(P[:, 7].sum()); out[f"{src}|{g}"]["n_windows"] = W
        o = out[f"{src}|{g}"]
        print(f"{src:13s} {g:3s} 칸 {o['n_cells']:6d} | L1 기준 {o['L1_base'][0]:.4f} 영상 {o['L1_ctxt'][0]:.4f} 비 {o['ratio_ctxt_over_base'][0]:.4f} [{o['ratio_ctxt_over_base'][1]:.4f}, {o['ratio_ctxt_over_base'][2]:.4f}]"
              f" | /복사 {o['base_over_copy'][0]:.3f} · {o['ctxt_over_copy'][0]:.3f} | /평균 {o['base_over_mean'][0]:.3f} · {o['ctxt_over_mean'][0]:.3f}"
              f" | 사영 {o['proj_base'][0]:.3f} [{o['proj_base'][1]:.3f},{o['proj_base'][2]:.3f}] · {o['proj_ctxt'][0]:.3f} [{o['proj_ctxt'][1]:.3f},{o['proj_ctxt'][2]:.3f}]")
json.dump(out, open(f"{REPO}/auto_research/results/ctxt_l1_shared.json", "w"), indent=1)
