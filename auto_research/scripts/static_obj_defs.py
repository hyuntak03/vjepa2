#!/usr/bin/env python3
"""[E] 정지 물체 안쪽 칸의 블록 간 토큰 변화 — [E] 0.380 vs [T] 0.18 정의 맞추기 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/static_obj_defs.py

3D 합성 검증 세트 캐시 (블록 토큰 16 개/창). 한 정의 요소씩 바꿔 가며 같은 양 (물체 안쪽 칸의 |h_{t+1} − h_t|, 차원 평균, 쌍마다 평균 후 쌍 평균) 을 낸다.
  [T] 정의: t = 7..14 · 정지 = |pos[2t+3] − pos[2t+1]| < 1 (두 블록 둘째 장) · 두 블록 보임 (vis_frac > 0.7 두 장 모두, inframe > 0.9) ·
            안쪽 = 블록 덮임 (두 장 중 큰 쪽) ≥ 0.8 가 t · t+1 모두
  [E] 정의: t = 0..14 · 정지 = |블록 중심 변위| < 1 (두 장 평균 위치) · 보임 조건 없음 · 안쪽 = 블록 덮임 (두 장 중 작은 쪽) > 0.6 가 t · t+1 모두"""
import glob, json, os
import numpy as np
CACHE = "/workspace/auto_research_cache/phys3d"
G = 16


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


VARS = {
    "T (그대로)":                 dict(trange=(7, 15), disp="second", vis=True, inside="max0.8"),
    "E (그대로)":                 dict(trange=(0, 15), disp="center", vis=False, inside="min0.6"),
    "T + t 0..14":                dict(trange=(0, 15), disp="second", vis=True, inside="max0.8"),
    "T − 보임 조건":              dict(trange=(7, 15), disp="second", vis=False, inside="max0.8"),
    "T + 블록 중심 변위":         dict(trange=(7, 15), disp="center", vis=True, inside="max0.8"),
    "T + 안쪽 min>0.6":           dict(trange=(7, 15), disp="second", vis=True, inside="min0.6"),
    "E + 보임 조건":              dict(trange=(0, 15), disp="center", vis=True, inside="min0.6"),
    "E + t 7..14":                dict(trange=(7, 15), disp="center", vis=False, inside="min0.6"),
    "E + 보임 + t 7..14":         dict(trange=(7, 15), disp="center", vis=True, inside="min0.6"),
}
acc = {k: [] for k in VARS}; accw = {k: [] for k in VARS}
for f in sorted(glob.glob(os.path.join(CACHE, "h_val_target_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(os.path.join(CACHE, f"lab_val_s{sh}.npz")); X = np.load(f, mmap_mode="r")
    for w in range(X.shape[0]):
        cvf = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0))
        cmax = np.maximum(cvf[0::2], cvf[1::2]); cmin = np.minimum(cvf[0::2], cvf[1::2])
        ok = lab["inframe_frac"][w] > 0.9
        vis_b = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & ok[0::2] & ok[1::2]
        pos = lab["pos"][w]
        h = None
        for name, v in VARS.items():
            for t in range(*v["trange"]):
                d = (pos[2 * t + 3] - pos[2 * t + 1]) if v["disp"] == "second" else (pos[2 * t + 2:2 * t + 4].mean(0) - pos[2 * t:2 * t + 2].mean(0))
                if not np.isfinite(d).all() or np.linalg.norm(d) >= 1:
                    continue
                if v["vis"] and not (vis_b[t] and vis_b[t + 1]):
                    continue
                ins = ((cmax[t] >= 0.8) & (cmax[t + 1] >= 0.8)) if v["inside"] == "max0.8" else ((cmin[t] > 0.6) & (cmin[t + 1] > 0.6))
                ins = ins.reshape(-1)
                if not ins.any():
                    continue
                if h is None:
                    h = np.asarray(X[w], np.float32)
                e = np.abs(h[t + 1][ins] - h[t][ins]).mean(-1)
                acc[name].append(float(e.mean())); accw[name].append(w + 1000 * int(sh))
res = {k: {"mean": float(np.mean(v)), "n_pairs": len(v), "n_windows": len(set(accw[k]))} for k, v in acc.items() if v}
for k, r in res.items():
    print(f"{k:24s} {r['mean']:.3f}  (쌍 {r['n_pairs']}, 창 {r['n_windows']})")
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/static_obj_defs.json"), "w"), indent=1, ensure_ascii=False)
