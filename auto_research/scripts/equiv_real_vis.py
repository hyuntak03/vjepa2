#!/usr/bin/env python3
"""[E] 검사 2a (평행 이동 등변성, 실제 쌍) 를 보임 조건을 넣어 다시 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/equiv_real_vis.py

enc_equivariance.py 의 검사 2a 와 같다 (t = 0..14, 블록 중심 변위 d, 물체 안쪽 = 두 장 덮임 작은 쪽 > 0.6, round(d/16) 칸 옮겨 비교).
차이: 두 블록 모두 보임 (vis_frac > 0.7 두 장, inframe > 0.9) 인 쌍만. 보임 조건이 없으면 가려진 물체의 bbox 칸 (가림막 · 배경) 이 섞인다."""
import glob, json, os
import numpy as np
CACHE = "/workspace/auto_research_cache/phys3d"; G = 16


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


acc = {}
def put(k, v):
    acc.setdefault(k, []).append(v)


for f in sorted(glob.glob(os.path.join(CACHE, "h_val_target_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(os.path.join(CACHE, f"lab_val_s{sh}.npz")); Hc = np.load(f, mmap_mode="r")
    for w in range(Hc.shape[0]):
        cv = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0)); ins = np.minimum(cv[0::2], cv[1::2]) > 0.6
        ok = lab["inframe_frac"][w] > 0.9
        vis_b = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & ok[0::2] & ok[1::2]
        pos = lab["pos"][w]; h = None
        for t in range(15):
            if not (ins[t].any() and vis_b[t] and vis_b[t + 1]):
                continue
            d = pos[2 * t + 2:2 * t + 4].mean(0) - pos[2 * t:2 * t + 2].mean(0)
            if not np.isfinite(d).all():
                continue
            if h is None:
                h = np.asarray(Hc[w], np.float32)
            dn = np.linalg.norm(d); kk = np.round(d / 16).astype(int)
            wmov = np.linalg.norm(pos[2 * t + 1] - pos[2 * t])
            ys, xs = np.where(ins[t]); yt, xt = ys + kk[1], xs + kk[0]
            okc = (yt >= 0) & (yt < G) & (xt >= 0) & (xt < G)
            okc &= ins[t + 1][np.clip(yt, 0, G - 1), np.clip(xt, 0, G - 1)]
            if not okc.any():
                continue
            ys, xs, yt, xt = ys[okc], xs[okc], yt[okc], xt[okc]
            ht, hn = h[t].reshape(G, G, -1), h[t + 1].reshape(G, G, -1)
            eqd = float(np.abs(hn[yt, xt] - ht[ys, xs]).mean(-1).mean())
            if dn < 1:
                put("정지 물체", eqd)
            else:
                ph = np.abs(d - 16 * kk).max()
                put("움직인 물체 (칸 보정)", eqd)
                put(f"블록 안 움직임 {'<2' if wmov < 2 else ('2-5' if wmov < 5 else '>5')} px/장", eqd)
                put(f"칸 아래 위상 {'<4' if ph < 4 else ('4-8' if ph < 8.1 else '>8')} px", eqd)
res = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in acc.items()}
for k in sorted(res):
    print(f"{k:24s} {res[k]['mean']:.3f} (n {res[k]['n']})")
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/equiv_real_vis.json"), "w"), indent=1, ensure_ascii=False)
