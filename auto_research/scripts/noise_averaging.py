#!/usr/bin/env python3
"""고리 2: 실제 블록 latent 에서 L1 손실은 "문맥 평균" 을 선호하나, 그 대가를 움직이는 물체가 치르나 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/noise_averaging.py [--enc target]

캐시 phys3d val, 창마다 문맥 = 블록 t−7..t, 정답 = 블록 t+1 (t = 7..14). 학습 없는 예측기:
  copy      h_t
  mean_n    최근 n 블록 평균 (n = 2, 4, 8)
  mc_mean8  움직임 보정: 물체가 지나간 칸 (t−7..t+1 touch 합집합) 밖은 mean_8, 안은 h_t 를 정답 변위 (정수 칸) 만큼 옮겨 쓴 것
            (물체 자리 = 옮긴 h_t, 빈 옛 자리 = mean_8). "평균으로 잡음을 줄이되 물체는 따라간다" 의 손쉬운 상한
손실 = 차원 평균 |p − h_{t+1}|, 칸 무리별: obj (t 또는 t+1 에 물체가 닿은 칸) · ring (2 칸 팽창) · far.
쌍: mov (t → t+1 물체 이동 > 8 px, 두 블록 모두 보임) / sta (< 1 px).
잡음 독립성: 물체가 한 번도 안 닿은 칸의 잔차 r_b = h_b − (16 블록 평균) 의 이웃 블록 상관 (차원 · 칸 평균)."""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--enc", default="target")
ap.add_argument("--out", default=None)
a = ap.parse_args()
G, NB = 16, 16


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


def dil(m, k=1):
    for _ in range(k):
        o = m.copy()
        o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
        o[1:, 1:] |= m[:-1, :-1]; o[:-1, :-1] |= m[1:, 1:]; o[1:, :-1] |= m[:-1, 1:]; o[:-1, 1:] |= m[1:, :-1]
        m = o
    return m


PRED = ["copy", "mean_2", "mean_4", "mean_8", "mc_mean8", "oracle_mc"]
acc = {k: {p: {g: [] for g in ("obj", "ring", "far", "all")} for p in PRED} for k in ("mov", "sta")}
area = {k: [] for k in ("mov", "sta")}
ac1 = []
for f in sorted(glob.glob(os.path.join(a.cache, f"h_val_{a.enc}_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(os.path.join(a.cache, f"lab_val_s{sh}.npz"))
    X = np.load(f, mmap_mode="r")
    for w in range(X.shape[0]):
        h = X[w].astype(np.float32)                                       # (16,256,D)
        cv = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0))
        ok = lab["inframe_frac"][w] > 0.9
        touch = ((cv[0::2] > 0) | (cv[1::2] > 0))                          # (NB,G,G)
        vis_b = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & ok[0::2] & ok[1::2]
        pos = lab["pos"][w]
        never = ~dil(touch.any(0), 1).reshape(-1)
        if never.sum() > 10:
            r = h[:, never] - h[:, never].mean(0, keepdims=True)           # (NB, n, D)
            num = (r[1:] * r[:-1]).sum(0); den = np.sqrt((r[1:] ** 2).sum(0) * (r[:-1] ** 2).sum(0)) + 1e-8
            ac1.append(float((num / den).mean()))
        for t in range(7, NB - 1):
            d = pos[2 * t + 3] - pos[2 * t + 1]
            dn = np.linalg.norm(d)
            if not np.isfinite(dn):
                continue
            kind = "mov" if (dn > 8 and vis_b[t] and vis_b[t + 1]) else ("sta" if dn < 1 else None)
            if kind is None:
                continue
            y = h[t + 1]
            P = {"copy": h[t]}
            for n in (2, 4, 8):
                P[f"mean_{n}"] = h[t - n + 1:t + 1].mean(0)
            path = dil(touch[t - 7:t + 2].any(0), 1).reshape(-1)
            mc = P["mean_8"].copy()
            dx, dy = int(round(d[0] / 16)), int(round(d[1] / 16))
            src = touch[t].reshape(-1)
            ht = h[t].reshape(G, G, -1)
            sh_ = np.roll(np.roll(ht, dy, 0), dx, 1).reshape(G * G, -1)
            dst = np.roll(np.roll(touch[t], dy, 0), dx, 1).reshape(-1)
            mc[dst] = sh_[dst]
            P["mc_mean8"] = mc
            # oracle_mc: 배경 = 그 칸에 물체가 없던 문맥 블록만 평균 / 물체 (t+1 자리) = 문맥 블록마다 물체를 t+1 자리로 옮겨 평균
            orc = np.empty_like(mc)
            ctx = h[t - 7:t + 1]; tc = touch[t - 7:t + 1].reshape(8, -1)
            free = ~tc
            cnt = free.sum(0)
            bgavg = (ctx * free[..., None]).sum(0) / np.maximum(cnt, 1)[:, None]
            orc[:] = np.where((cnt > 0)[:, None], bgavg, h[t])
            dst1 = touch[t + 1].reshape(-1)
            acc_o, n_o = np.zeros((G * G, h.shape[-1]), np.float32), 0
            for j, b in enumerate(range(t - 7, t + 1)):
                if not vis_b[b]:
                    continue
                db = pos[2 * t + 3] - pos[2 * b + 1]
                if not np.isfinite(db).all():
                    continue
                ddx, ddy = int(round(db[0] / 16)), int(round(db[1] / 16))
                acc_o += np.roll(np.roll(h[b].reshape(G, G, -1), ddy, 0), ddx, 1).reshape(G * G, -1); n_o += 1
            if n_o:
                orc[dst1] = acc_o[dst1] / n_o
            P["oracle_mc"] = orc
            obj = (touch[t] | touch[t + 1]).reshape(-1)
            ring = dil(obj.reshape(G, G), 2).reshape(-1) & ~obj
            far = ~(obj | ring)
            area[kind].append((obj.mean(), ring.mean()))
            for p, v in P.items():
                e = np.abs(v - y).mean(-1)
                for gname, m in (("obj", obj), ("ring", ring), ("far", far)):
                    acc[kind][p][gname].append(float(e[m].sum() / 256))          # 화면 평균 손실에 대한 기여 (넓이 반영)
                acc[kind][p]["all"].append(float(e.mean()))
res = {"ac1_residual_static_cells": float(np.median(ac1))}
print(f"[잡음 독립성] 물체가 안 닿은 칸 잔차의 이웃 블록 상관 (중앙값) {res['ac1_residual_static_cells']:.3f}  (0 = 블록마다 새 잡음, 1 = 느린 표류)")
for kind in ("mov", "sta"):
    ar_ = np.array(area[kind]); n = len(ar_)
    print(f"\n== {kind} (n {n}, 물체 칸 넓이 {ar_[:, 0].mean():.3f}, 고리 {ar_[:, 1].mean():.3f}) — 화면 평균 L1 에 대한 기여 (obj + ring + far = all)")
    res[kind] = {}
    for p in PRED:
        m = {g: float(np.mean(acc[kind][p][g])) for g in ("obj", "ring", "far", "all")}
        res[kind][p] = m
        print(f"  {p:9s} all {m['all']:.4f} | obj {m['obj']:.4f} · ring {m['ring']:.4f} · far {m['far']:.4f}")
out = a.out or os.path.join(os.path.dirname(__file__), f"../results/noise_averaging_{a.enc}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=1)
