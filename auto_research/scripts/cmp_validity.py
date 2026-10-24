#!/usr/bin/env python3
"""[P] 1b 보조 정답 s' (흐름 없는 5×5 국소 최대 cos) 의 유효성 — 학습 없음, CPU (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/cmp_validity.py

latent 블록 토큰에서 "같은 내용은 근처 칸에 비슷한 토큰" 이 성립하는지 먼저 본다 (사용자 지적: 픽셀 가정을 그대로 들이지 말 것).
데이터: K710 보류 (natural_holdout, 칸 흐름 포함) · HT100M ([E] 캐시 + natural_transport_ctx5 흐름) · 3D 합성 ([E] 캐시 + 흐름).
움직이는 칸 c (실제 흐름 |U[t+1](c)| ≥ 0.5 칸, t = 4 · 5 · 6), 내용의 블록 t 자리 j = c + U[t+1](c), 블록 t−1 자리 p = j + U[t](j) ([T] 정의).
  s_flow = cos(h_t(j), h_{t−1}(p))
  s'     = max_{d ∈ 5×5} cos(h_t(j), h_{t−1}(j + d)),  d' = 그 위치 (화면 밖 제외)
  대조: s_rand = 5×5 안 무작위 이웃 (d ≠ 0) 의 cos · s_same = cos(h_t(j), h_{t−1}(j)) (변위 0)
(1) 상관 corr(·, s_flow) · d' 와 RAFT 문맥 흐름 반올림 (U[t](j)) 의 일치율 (정확 · 1 칸 안).
(2) 핵심: 각 자의 5 분위 구간별 데이터 최적 운반 b ([T] context_reliability 와 같은 회귀: y = h_{t+1}(c) ≈ a·같은 칸 5 블록 평균 + b·h_t(j)).
사전 등록: s' 구간 b 오름 (하위 → 상위) ≥ +0.07 이고 corr(s', s_flow) ≥ 0.5 → 유효. 미만 → 무효 (1b 접음)."""
import os
import numpy as np

EN = "/workspace/auto_research_cache/enc_noise"
RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
OFF = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
rng = np.random.RandomState(0)


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.where(ok, np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), np.arange(S)), ok


def windows(src):
    if src == "k710_holdout":
        H = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_h.npy", mmap_mode="r")
        U = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy", mmap_mode="r")
        for w in range(H.shape[0]):
            yield np.asarray(H[w], np.float32), np.asarray(U[w], np.float32).reshape(8, S, 2)
    else:
        U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32)
        w0 = 0
        for sh in (0, 1):
            T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
            for w in range(T_.shape[0]):
                yield np.asarray(T_[w, 0], np.float32), U[w0 + w].reshape(8, S, 2)
            w0 += T_.shape[0]


def bfit(st):
    A = np.array([[st[:, 0].sum(), st[:, 1].sum()], [st[:, 1].sum(), st[:, 2].sum()]])
    return float(np.linalg.solve(A, [st[:, 3].sum(), st[:, 4].sum()])[1])


ROWS = {}
for src in ("k710_holdout", "ht100m", "phys3d"):
    R = []
    for h, U in windows(src):
        hn = h / np.linalg.norm(h, axis=-1, keepdims=True)
        for t in (4, 5, 6):
            u1, u0 = U[t + 1], U[t]
            mov = np.linalg.norm(u1, axis=1) >= 0.5
            js, ok1 = cell_of(P0 + u1); jp, ok0 = cell_of(P0[js] + u0[js])
            sel = np.where(mov & ok1 & ok0)[0]
            if sel.size == 0:
                continue
            j = js[sel]; p = jp[sel]
            s_flow = (hn[t][j] * hn[t - 1][p]).sum(-1)
            jy, jx = j // G, j % G
            cs = np.full((len(j), 25), -2.0, np.float32)
            for k, (dy, dx) in enumerate(OFF):
                ty, tx = jy + dy, jx + dx; ok = (ty >= 0) & (ty < G) & (tx >= 0) & (tx < G)
                q = np.clip(ty, 0, G - 1) * G + np.clip(tx, 0, G - 1)
                cs[:, k] = np.where(ok, (hn[t][j] * hn[t - 1][q]).sum(-1), -2.0)
            kmax = cs.argmax(1); s_max = cs.max(1); dmax = np.array(OFF)[kmax]                       # (n, 2) (dy, dx)
            s_same = cs[:, 12]
            valid_rand = cs.copy(); valid_rand[:, 12] = -2.0
            rr = np.array([rng.choice(np.where(v > -1.5)[0]) for v in valid_rand])
            s_rand = cs[np.arange(len(j)), rr]
            ur = np.round(u0[j][:, ::-1])                                                          # 문맥 흐름 (dy, dx) 반올림 (칸)
            exact = (dmax == ur).all(1); near = (np.abs(dmax - ur) <= 1).all(1)
            x1 = h[t - 4:t + 1].mean(0)[sel]; x2 = h[t][j]; y = h[t + 1][sel]
            R.append(np.stack([(x1 * x1).sum(-1), (x1 * x2).sum(-1), (x2 * x2).sum(-1), (x1 * y).sum(-1), (x2 * y).sum(-1),
                               s_flow, s_max, s_rand, s_same, exact, near, np.linalg.norm(u0[j], axis=1)], -1))
    ROWS[src] = np.concatenate(R)
    print(f"{src}: 움직이는 칸 {len(ROWS[src])}", flush=True)

NAMES = {5: "s_flow (기준)", 6: "s' 5×5 최대", 7: "무작위 이웃", 8: "같은 칸 (변위 0)"}
nat = np.concatenate([ROWS["k710_holdout"], ROWS["ht100m"]])
res = {}
for grp, st in (("natural (K710 보류 + HT100M)", nat), ("phys3d", ROWS["phys3d"])):
    print(f"\n== {grp}: 칸 {len(st)}")
    e = {}
    for col in (6, 7, 8):
        e[f"corr_{col}"] = float(np.corrcoef(st[:, col], st[:, 5])[0, 1])
    mv = st[:, 11] >= 0.5
    e["argmax_exact_all"] = float(st[:, 9].mean()); e["argmax_near_all"] = float(st[:, 10].mean())
    e["argmax_exact_ctxmoving"] = float(st[mv, 9].mean()) if mv.any() else None; e["argmax_near_ctxmoving"] = float(st[mv, 10].mean()) if mv.any() else None
    print(f"  상관 (s_flow 와): s' {e['corr_6']:.3f} · 무작위 이웃 {e['corr_7']:.3f} · 같은 칸 {e['corr_8']:.3f}")
    print(f"  최대 위치 = 문맥 RAFT 흐름 반올림: 정확 {e['argmax_exact_all']:.3f} · 1 칸 안 {e['argmax_near_all']:.3f} | 문맥도 움직인 칸 (|U[t]| ≥ 0.5): {e['argmax_exact_ctxmoving']:.3f} · {e['argmax_near_ctxmoving']:.3f}")
    for col, nm in NAMES.items():
        edges = np.quantile(st[:, col], [0.2, 0.4, 0.6, 0.8]); bins = np.digitize(st[:, col], edges)
        bs = [bfit(st[bins == k]) for k in range(5)]
        e[f"bins_{col}"] = dict(edges=edges.tolist(), b=bs, rise=bs[4] - bs[0])
        print(f"  {nm:14s} 5 분위 b: " + " · ".join(f"{b:.3f}" for b in bs) + f"   → 오름 {bs[4] - bs[0]:+.3f}  (경계 {np.round(edges, 3).tolist()})")
    res[grp] = e
import json
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/cmp_validity.json"), "w"), indent=1)
