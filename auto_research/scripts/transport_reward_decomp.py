#!/usr/bin/env python3
"""[T] 고리 5: 운반 보상 0.7 % 는 왜 이렇게 작나 — 칸별로 쪼갠다 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/transport_reward_decomp.py

phys3d val 캐시, 물체가 움직인 블록 쌍 (t → t+1 변위 > 8 px, 두 블록 모두 보임), t = 7..14.
칸 무리:
  entered  : 블록 t 에 물체가 전혀 안 닿고 (덮임 0) 블록 t+1 에 거의 다 덮인 (≥ 0.8) 칸 — 운반하면 물체를 "그려야" 하는 칸
  vacated  : 블록 t 에 거의 다 덮이고 (≥ 0.8) 블록 t+1 에 전혀 안 닿는 칸 — 운반하면 물체를 "지워야" 하는 칸
  stay_obj : 두 블록 모두 거의 다 덮인 칸 (물체 안쪽)
  far      : 문맥 · 정답 어디에도 물체가 2 칸 안에 안 온 칸
칸마다 차원 평균 L1:
  bg8      = 그 칸에 물체가 없던 문맥 블록들의 평균            (= "배경" 예측)
  obj_last = 블록 t 의 토큰을 정답 변위 (정수 칸) 만큼 옮긴 것  (= 한 블록 운반)
  obj_traj = 문맥 블록마다 물체를 t+1 자리로 옮겨 평균         (= 궤적 평균 운반, 운반 오라클의 물체 칸)
  copy     = h_t (같은 칸)
  same_obj_static = 물체가 정지한 쌍에서 물체 안쪽 칸의 |h_{t+1} − h_t|  (물체 토큰 자체의 블록 잡음)
보상 = L(bg8) − L(obj_traj) (entered), L(copy 쪽) − L(bg8) (vacated). 가우시안 이동 모형 예측과 대조:
  칸당 보상 ≈ E|Δ + η| − E|η|, Δ = 깨끗한 물체 대비, η = 평균 예측의 남은 잡음."""
import glob, json, os
import numpy as np

CACHE = "/workspace/auto_research_cache/phys3d"
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


acc = {g: {k: [] for k in ("bg8", "obj_last", "obj_traj", "copy", "mean8")} for g in ("entered", "vacated", "stay_obj", "far")}
static_obj, static_far = [], []
n_cells = {g: [] for g in ("entered", "vacated")}
for f in sorted(glob.glob(os.path.join(CACHE, "h_val_target_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(os.path.join(CACHE, f"lab_val_s{sh}.npz"))
    X = np.load(f, mmap_mode="r")
    for w in range(X.shape[0]):
        h = X[w].astype(np.float32)
        cvf = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0))           # (32,G,G)
        cvb = np.maximum(cvf[0::2], cvf[1::2])                            # 블록별 덮임 (두 장 중 큰 쪽)
        ok = lab["inframe_frac"][w] > 0.9
        vis_b = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & ok[0::2] & ok[1::2]
        pos = lab["pos"][w]
        for t in range(7, NB - 1):
            d = pos[2 * t + 3] - pos[2 * t + 1]
            dn = np.linalg.norm(d)
            if not np.isfinite(dn):
                continue
            y = h[t + 1]
            if dn < 1 and vis_b[t] and vis_b[t + 1]:
                inside = ((cvb[t] >= 0.8) & (cvb[t + 1] >= 0.8)).reshape(-1)
                farm = ~dil((cvb[t - 7:t + 2] > 0).any(0), 2).reshape(-1)
                if inside.sum():
                    static_obj.append(float(np.abs(y[inside] - h[t][inside]).mean()))
                if farm.sum():
                    static_far.append(float(np.abs(y[farm] - h[t][farm]).mean()))
                continue
            if not (dn > 8 and vis_b[t] and vis_b[t + 1]):
                continue
            ctx = h[t - 7:t + 1]
            free = (cvb[t - 7:t + 1] == 0).reshape(8, -1)
            cnt = free.sum(0)
            bg8 = (ctx * free[..., None]).sum(0) / np.maximum(cnt, 1)[:, None]
            dx, dy = int(round(d[0] / 16)), int(round(d[1] / 16))
            obj_last = np.roll(np.roll(h[t].reshape(G, G, -1), dy, 0), dx, 1).reshape(G * G, -1)
            acc_o, n_o = np.zeros_like(bg8), 0
            for b in range(t - 7, t + 1):
                if not vis_b[b]:
                    continue
                db = pos[2 * t + 3] - pos[2 * b + 1]
                if not np.isfinite(db).all():
                    continue
                acc_o += np.roll(np.roll(h[b].reshape(G, G, -1), int(round(db[1] / 16)), 0), int(round(db[0] / 16)), 1).reshape(G * G, -1); n_o += 1
            obj_traj = acc_o / max(n_o, 1)
            groups = {
                "entered": ((cvb[t] == 0) & (cvb[t + 1] >= 0.8)).reshape(-1) & (cnt > 0),
                "vacated": ((cvb[t] >= 0.8) & (cvb[t + 1] == 0)).reshape(-1) & (cnt > 0),
                "stay_obj": ((cvb[t] >= 0.8) & (cvb[t + 1] >= 0.8)).reshape(-1),
                "far": ~dil((cvb[t - 7:t + 2] > 0).any(0), 2).reshape(-1),
            }
            P = {"bg8": bg8, "obj_last": obj_last, "obj_traj": obj_traj, "copy": h[t], "mean8": ctx.mean(0)}
            for g, m in groups.items():
                if m.sum() == 0:
                    continue
                if g in n_cells:
                    n_cells[g].append(int(m.sum()))
                for k, v in P.items():
                    acc[g][k].append(float(np.abs(v[m] - y[m]).mean()))
res = {g: {k: float(np.mean(v)) for k, v in d.items() if v} for g, d in acc.items()}
res["n_pairs_with"] = {g: len(acc[g]["bg8"]) for g in acc}
res["cells_per_pair"] = {g: float(np.mean(v)) for g, v in n_cells.items()}
res["static_obj_copy"] = float(np.mean(static_obj)); res["static_far_copy"] = float(np.mean(static_far))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/transport_reward_decomp.json"), "w"), indent=1)
