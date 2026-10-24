#!/usr/bin/env python3
"""H-E: 고정 encoder 가 빠른 특징 (움직이는 물체) 을 약하게 담는가 — 캐시 위에서 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/encoder_slowness.py --enc target random pix

블록 b = 장 (2b, 2b+1), 칸 = 16 px. 물체 칸은 라벨 bbox 로 정한다 (두 장 모두에서 칸의 50 % 이상을 덮는 칸 = core, 조금이라도 = touch).
[E1] 물체 대비 = 물체 core 칸 토큰과 **같은 칸의 배경 기준** (같은 창에서 물체가 그 칸을 전혀 안 건드린 블록들의 평균) 사이 차원 평균 |·|.
     속도 (블록 안 두 장의 평균 px/raw장) 구간별 중앙값. 잡음 바닥 = 물체가 한 번도 안 지나간 칸에서 같은 계산.
[E2] 연속 블록 b → b+1 의 변화 Δ(c) = 차원 평균 |h_{b+1}(c) − h_b(c)|. 물체 칸 (네 장 touch 합집합) / 고리 (1 칸 팽창) / 먼 칸 의 L1 질량 몫.
     움직인 쌍 (블록 간 물체 이동 > 8 px) vs 정지 쌍 (< 1 px). 먼 칸의 (움직인 − 정지) = 움직임이 멀리 퍼뜨린 변화.
     물체 칸 + 고리 몫 = "물체를 옮기면 줄일 수 있는" 복사 손실의 몫 (복사 해의 L1 = Σ_c Δ(c) / 256).
pix 는 같은 계산을 16×16×3 픽셀 칸 (0..1) 에 한다. 단위가 달라 몫 · 비율만 비교한다."""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--enc", nargs="+", default=["target", "random", "pix"])
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/encoder_slowness.json"))
a = ap.parse_args()
G, NB = 16, 16


def load(enc):
    fs = sorted(glob.glob(os.path.join(a.cache, f"h_val_{enc}_s*.npy" if enc != "pix" else "pix_val_s*.npy")))
    return fs


def cover(bbox):                    # bbox (...,4) px → (..., G, G) 칸 덮임 비율
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


def dil(m):
    o = m.copy()
    o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= o[:, :-1].copy(); o[:, :-1] |= o[:, 1:].copy()
    return o


SP_BINS = [0, 0.5, 2, 4, 8, 100]
res = {}
for enc in a.enc:
    fs = load(enc)
    E1 = {i: [] for i in range(len(SP_BINS) - 1)}; E1n = []
    E2 = {"mov": [], "sta": []}
    for f in fs:
        sh = f.split("_s")[-1].split(".")[0]
        lab = np.load(os.path.join(a.cache, f"lab_val_s{sh}.npz"))
        X = np.load(f, mmap_mode="r")
        for w in range(X.shape[0]):
            if enc == "pix":
                px = X[w].astype(np.float32) / 255                                  # (32,256,256,3)
                h = px.reshape(NB, 2, G, 16, G, 16, 3).transpose(0, 2, 4, 1, 3, 5, 6).reshape(NB, G * G, -1)
            else:
                h = X[w].astype(np.float32)                                         # (16,256,1280)
            bb, vis, pos = lab["bbox"][w], lab["vis_frac"][w], lab["pos"][w]
            vel = np.linalg.norm(lab["vel"][w], axis=-1)
            cv = cover(bb)                                                          # (32,G,G)
            ok = np.isfinite(cv).all((1, 2)) & (lab["inframe_frac"][w] > 0.9)
            cv = np.nan_to_num(cv)
            core = (cv[0::2] >= 0.5) & (cv[1::2] >= 0.5)                            # (NB,G,G)
            touch = (cv[0::2] > 0) | (cv[1::2] > 0)
            vis_b = (vis[0::2] > 0.7) & (vis[1::2] > 0.7) & ok[0::2] & ok[1::2]
            never = ~touch.any(0)
            # E1
            for b in range(NB):
                if not vis_b[b] or not core[b].any():
                    continue
                cells = np.argwhere(core[b])
                vals = []
                for (y, x) in cells:
                    refb = [bb_ for bb_ in range(NB) if not touch[bb_, y, x]]
                    if len(refb) < 3:
                        continue
                    ref = h[refb, y * G + x].mean(0)
                    vals.append(np.abs(h[b, y * G + x] - ref).mean())
                if vals:
                    sp = 0.5 * (vel[2 * b] + vel[2 * b + 1])
                    E1[int(np.digitize(sp, SP_BINS) - 1)].append(float(np.mean(vals)))
            nv = np.argwhere(never)
            if len(nv):
                y, x = nv[len(nv) // 2]
                ref = h[:, y * G + x].mean(0)
                E1n.append(float(np.abs(h[:, y * G + x] - ref).mean()))
            # E2
            for b in range(NB - 1):
                if not (ok[2 * b:2 * b + 4].all()):
                    continue
                d = np.linalg.norm(pos[2 * b + 3] - pos[2 * b + 1])
                kind = "mov" if (d > 8 and vis_b[b] and vis_b[b + 1]) else ("sta" if d < 1 else None)
                if kind is None:
                    continue
                D_ = np.abs(h[b + 1] - h[b]).mean(-1).reshape(G, G)
                obj = touch[b] | touch[b + 1]
                ring = dil(dil(obj)) & ~obj if kind == "mov" else dil(dil(obj)) & ~obj
                far = ~(obj | ring)
                tot = D_.sum()
                E2[kind].append(dict(obj=float(D_[obj].sum() / tot), ring=float(D_[ring].sum() / tot), far=float(D_[far].sum() / tot),
                                     far_mean=float(D_[far].mean()), obj_mean=float(D_[obj].mean()) if obj.any() else np.nan,
                                     area_obj=float(obj.mean()), area_ring=float(ring.mean()), copy_l1=float(D_.mean()), disp=float(d)))
    r = {"E1_contrast_by_speed": {f"{SP_BINS[i]}-{SP_BINS[i + 1]}": (float(np.median(v)) if v else None, len(v)) for i, v in E1.items()},
         "E1_noise_floor": float(np.median(E1n)) if E1n else None}
    for k, v in E2.items():
        if not v:
            continue
        r[f"E2_{k}"] = {kk: float(np.nanmedian([x[kk] for x in v])) for kk in v[0]} | {"n": len(v)}
    res[enc] = r
    print(f"== {enc}")
    print("  E1 물체 대비 (속도 px/raw장 구간: 중앙값, n):", {k: (round(v[0], 4) if v[0] else None, v[1]) for k, v in r["E1_contrast_by_speed"].items()},
          "| 잡음 바닥", round(r["E1_noise_floor"], 4) if r["E1_noise_floor"] else None)
    for k in ("mov", "sta"):
        if f"E2_{k}" in r:
            e = r[f"E2_{k}"]
            print(f"  E2 {k}: 질량 몫 물체 {e['obj']:.3f} (넓이 {e['area_obj']:.3f}) · 고리 {e['ring']:.3f} (넓이 {e['area_ring']:.3f}) · 먼 칸 {e['far']:.3f} | "
                  f"칸당 변화 물체 {e['obj_mean']:.4f} · 먼 칸 {e['far_mean']:.4f} | 복사 L1 {e['copy_l1']:.4f} (n {e['n']})")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
