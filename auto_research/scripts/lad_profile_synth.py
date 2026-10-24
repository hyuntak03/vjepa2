#!/usr/bin/env python3
"""[P] 합성 판 (2026-10-22 KST 17:26 사전 등록) — 리더 lad_profile.py 에서 캐시 경로 · 무리 · 출력 · 스레드 · 점 수 상한만 바꿨다 (점이 3 천만보다 적으면 있는 만큼).
원문: [L] L1 최적 운반 b — 정확한 프로파일 판 (auto_research, 2026-10-22 KST 16 시).

lad_transport.py 의 IRLS 는 반쪽 표본 (0.04–0.07) 과 전체 (0.01–0.03) 가 어긋나 수렴을 믿을 수 없다.
여기서는 f(b) = min_a Σ|y − a·m − b·W| 를 정확히 푼다 (b 를 고정하면 a 는 가중 중앙값: r = y − bW, a = wmedian(r/m, |m|)).
f 는 볼록이므로 b 를 황금분할로 찾는다. 점은 창 단위로 고른 뒤 무작위 3 천만 개.
위약 뺀 b = b*(W_true) − b*(W_rot). 위약 뺀 L1 보상 = (f_rot(b*_rot) − f_true(b*_true)) / f(0)."""
import json, os
import numpy as np, torch

torch.set_num_threads(8)
CACHE = "/workspace/auto_research_cache/lad_synth"
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1)
R90 = np.array([[0.0, -1.0], [1.0, 0.0]], np.float32)


def warp(ht, u):
    s = np.floor(P0 + u).astype(int)
    ok = (s >= 0).all(1) & (s < G).all(1)
    j = np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


def wmedian(v, w):
    o = torch.argsort(v); cw = torch.cumsum(w[o], 0)
    return v[o][torch.searchsorted(cw, cw[-1] / 2)]


def prof(y, m, W, b):
    r = y - b * W
    ok = m.abs() > 1e-6
    a = wmedian(r[ok] / m[ok], m[ok].abs())
    return float((r - a * m).abs().mean()), float(a)


def golden(y, m, W, lo=-0.2, hi=0.8, it=28):
    g = (5 ** 0.5 - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = prof(y, m, W, x1)[0], prof(y, m, W, x2)[0]
    for _ in range(it):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1; x1 = hi - g * (hi - lo); f1 = prof(y, m, W, x1)[0]
        else:
            lo, x1, f1 = x1, x2, f2; x2 = lo + g * (hi - lo); f2 = prof(y, m, W, x2)[0]
    b = (lo + hi) / 2
    return b, prof(y, m, W, b)[0]


res = {}
for grp in ("phys3d", "vobench"):
    Hs = np.load(os.path.join(CACHE, f"{grp}_h.npy"), mmap_mode="r"); Us = np.load(os.path.join(CACHE, f"{grp}_U.npy")).astype(np.float32)
    D = {k: [] for k in ("y", "cp", "m", "wt", "wr", "win")}
    for w in range(Hs.shape[0]):
        h = np.asarray(Hs[w]).astype(np.float32); U = Us[w].reshape(-1, S, 2)
        for t in (4, 5, 6):
            sel = np.linalg.norm(U[t + 1], axis=1) >= 0.5
            if sel.sum() < 2:
                continue
            D["y"].append(h[t + 1][sel]); D["cp"].append(h[t][sel]); D["m"].append(h[t - 4:t + 1].mean(0)[sel])
            D["wt"].append(warp(h[t], U[t + 1])[sel]); D["wr"].append(warp(h[t], U[t + 1] @ R90.T)[sel])
            D["win"].append(np.full((sel.sum(), 1280), w, np.int32))
    A = {k: torch.from_numpy(np.concatenate(v).ravel()) for k, v in D.items()}
    rs = np.random.RandomState(0)
    wins = np.unique(A["win"].numpy()); rs.shuffle(wins); halfset = set(wins[: len(wins) // 2].tolist())
    e = {}
    for part in ("all", "half_A", "half_B"):
        if part == "all":
            idx = torch.from_numpy(rs.choice(A["y"].numel(), min(30_000_000, A["y"].numel()), replace=False))
        else:
            inA = torch.from_numpy(np.isin(A["win"].numpy(), list(halfset)))
            pool = torch.nonzero(inA if part == "half_A" else ~inA).squeeze(1)
            idx = pool[torch.from_numpy(rs.choice(pool.numel(), min(15_000_000, pool.numel()), replace=False))]
        y, cp, m, wt, wr = (A[k][idx] for k in ("y", "cp", "m", "wt", "wr"))
        for nm, yy in (("data", y), ("copy", cp)):
            f0, _ = prof(yy, m, wt, 0.0)
            bt, ft = golden(yy, m, wt); br, fr = golden(yy, m, wr)
            e[f"{part}_{nm}"] = dict(b_true=bt, b_rot=br, net=bt - br, L1_m_only=f0, reward_true=(f0 - ft) / f0, reward_rot=(f0 - fr) / f0,
                                     reward_net=(fr - ft) / f0)
            print(f"{grp} {part:6s} {nm:4s} L1 최적: b_true {bt:.4f} · b_rot {br:.4f} · 위약 뺀 {bt - br:.4f} · "
                  f"L1 보상 true {100 * (f0 - ft) / f0:.2f} % · rot {100 * (f0 - fr) / f0:.2f} % · 위약 뺀 {100 * (fr - ft) / f0:.3f} %", flush=True)
    e["n_points"] = int(A["y"].numel()); e["n_windows"] = int(len(wins))
    res[grp] = e
out = os.path.join(os.path.dirname(__file__), "../results/lad_profile_synth.json")
json.dump(res, open(out, "w"), indent=1)
print("저장:", os.path.abspath(out))
