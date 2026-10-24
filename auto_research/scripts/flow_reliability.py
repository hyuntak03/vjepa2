#!/usr/bin/env python3
"""[P] (ii) 독립 근거 점검 — 데이터 최적 운반 신뢰 b 의 자연 vs 합성 차이 (0.205 vs 0.325) 가 RAFT 흐름 오차 때문인가 (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/flow_reliability.py

b = 움직이는 칸에서 정답 h_{t+1} ≈ a·(같은 칸 5 블록 평균 h_{t−4..t}) + b·(h_t 를 흐름으로 옮긴 것) 의 b ([T] transfer_beta 와 같은 회귀, 같은 warp).
흐름 = RAFT-large (12 회), 블록 대표 장 (둘째 장) 사이. 역방향 (b → b−1) 의 칸 평균 = 칸 흐름 u (칸 단위, 지금까지와 같다).
신뢰도 (칸마다): ① 앞-뒤 일관성 오차 = |f_bw(x) + f_fw(x + f_bw(x))| 의 칸 평균 (칸 단위) ② 칸 안 흐름 표준편차 (칸 단위, 경계 · 비강체).
데이터: 자연 (K710 · SSv2 · HT100M, [E] 캐시 창 160 개씩, 블록 8 개, t = 4 · 5 · 6) · 3D 합성 (내 캐시 358 창, 블록 16 개, t = 4..14, 라벨 위치 있음)
        · VOBench ([E] 캐시 창 160 개, 블록 8 개, A 물체 라벨 위치 있음).
라벨 흐름 (3D 합성 · VOBench): 블록 t+1 물체 칸 (3D: 상자 덮임 ≥ 0.5 / VOBench: 중심 반지름 원 덮임 ≥ 0.5) 에서 u = (물체 위치_t − 위치_{t+1}) / 16, 나머지 0.
  라벨 b 와 RAFT b 를 같은 칸 (라벨 물체 칸 · |u_라벨| ≥ 0.5) 에서 비교 → 흐름 오차가 b 를 얼마나 깎는지 직접."""
import glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F

torch.set_num_threads(8)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
dev = torch.device("cuda")
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights   # noqa: E402
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
EN = "/workspace/auto_research_cache/enc_noise"
P3 = "/workspace/auto_research_cache/phys3d"
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1)
IDX96 = list(range(0, 96, 3))[:32]


def warp(ht, u):
    s = np.floor(P0 + u).astype(int)
    ok = (s >= 0).all(1) & (s < G).all(1)
    j = np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


yy, xx = torch.meshgrid(torch.arange(256, device=dev).float(), torch.arange(256, device=dev).float(), indexing="ij")


@torch.no_grad()
def flows(rep, pairs):
    """rep (N,256,256,3) uint8 대표 장. pairs = [(b, b−1)]. return {b: (u 칸 (S,2), fb 오차 칸 (S,), 칸 안 std (S,))}."""
    x = torch.from_numpy(np.ascontiguousarray(rep)).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
    i1 = torch.stack([x[b] for b, _ in pairs]); i0 = torch.stack([x[c] for _, c in pairs])
    fbw = raft(i1, i0, num_flow_updates=12)[-1]                                       # b → b−1 (dx, dy) px
    ffw = raft(i0, i1, num_flow_updates=12)[-1]                                       # b−1 → b
    gx = (xx[None] + fbw[:, 0]) / 127.5 - 1; gy = (yy[None] + fbw[:, 1]) / 127.5 - 1
    ffs = F.grid_sample(ffw, torch.stack([gx, gy], -1), align_corners=True, padding_mode="border")
    err = (fbw + ffs).norm(dim=1)                                                     # (n,256,256) px
    u = F.avg_pool2d(fbw, 16) / 16                                                     # (n,2,16,16)
    var = (F.avg_pool2d(fbw ** 2, 16) - F.avg_pool2d(fbw, 16) ** 2).clamp_min(0).sum(1).sqrt() / 16
    e = F.avg_pool2d(err[:, None], 16)[:, 0] / 16
    out = {}
    for i, (b, _) in enumerate(pairs):
        out[b] = (u[i].permute(1, 2, 0).reshape(S, 2).cpu().numpy(), e[i].reshape(S).cpu().numpy(), var[i].reshape(S).cpu().numpy())
    return out


def cover_box(bbox):                                                                    # 3D 합성 상자 (x0,y0,x1,y1) px → (G,G)
    x0, y0, x1, y1 = bbox
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return (oy[:, None] * ox[None, :]).reshape(S)


def cover_disk(u, v, r, scale):                                                         # VOBench 원 (512 px 좌표) → 256 화면 칸 덮임 (표본 4×4)
    sub = (np.arange(4) + 0.5) / 4 * 16
    px = (cx[:, None] * 16 + sub[None, :]); py = (cy[:, None] * 16 + sub[None, :])
    inside = ((px[:, None, :] - u * scale) ** 2 + (py[:, :, None] - v * scale) ** 2) <= (r * scale) ** 2
    return inside.mean((1, 2))


ACC = {}


def add(key, X, y):
    if key not in ACC:
        ACC[key] = [np.zeros((2, 2)), np.zeros(2), 0]
    ACC[key][0] += X.T @ X; ACC[key][1] += X.T @ y; ACC[key][2] += X.shape[0] // 1280


def reg(key, h, t, u, sel):
    if sel.sum() == 0:
        return
    m = h[t - 4:t + 1].mean(0); W = warp(h[t], u)
    X = np.stack([m[sel].ravel(), W[sel].ravel()], 1).astype(np.float64)
    add(key, X, h[t + 1][sel].ravel().astype(np.float64))


def raft_bins(src, h, t, fl):
    u, e, sd = fl
    mag = np.linalg.norm(u, axis=1); mov = mag >= 0.5
    reg((src, "raft", "all"), h, t, u, mov)
    for nm, sel in (("fb<0.1", e < 0.1), ("fb0.1-0.25", (e >= 0.1) & (e < 0.25)), ("fb>=0.25", e >= 0.25),
                    ("sd<0.25", sd < 0.25), ("sd>=0.25", sd >= 0.25), ("fb<0.1&sd<0.25", (e < 0.1) & (sd < 0.25)),
                    ("mag0.5-1", mag < 1), ("mag1-2", (mag >= 1) & (mag < 2)), ("mag>=2", mag >= 2)):
        reg((src, "raft", nm), h, t, u, mov & sel)


# ---------------- 자연 영상
for src in ("k710", "ssv2", "ht100m"):
    for sh in (0, 1):
        T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r"); P_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_pix.npy"), mmap_mode="r")
        for w in range(T_.shape[0]):
            h = np.asarray(T_[w, 0]).astype(np.float32); rep = np.asarray(P_[w])[1::2]
            fl = flows(rep, [(t + 1, t) for t in (4, 5, 6)])
            for t in (4, 5, 6):
                raft_bins(src, h, t, fl[t + 1])
    print(src, "done", flush=True)

# ---------------- VOBench (A 물체 라벨)
for sh in (0, 1):
    T_ = np.load(os.path.join(EN, f"vobench_target_encoder_s{sh}_tok.npy"), mmap_mode="r"); P_ = np.load(os.path.join(EN, f"vobench_target_encoder_s{sh}_pix.npy"), mmap_mode="r")
    meta = json.load(open(os.path.join(EN, f"vobench_target_encoder_s{sh}_meta.json")))
    for w in range(T_.shape[0]):
        h = np.asarray(T_[w, 0]).astype(np.float32); rep = np.asarray(P_[w])[1::2]
        lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(meta[w]["path"]), "label.json")))["labels"]["A"]}
        fidx = [meta[w]["start"] + meta[w]["stride"] * k for k in range(16)]
        fl = flows(rep, [(t + 1, t) for t in (4, 5, 6)])
        for t in (4, 5, 6):
            q0, q1 = lab.get(fidx[2 * t + 1]), lab.get(fidx[2 * t + 3])
            if q0 is None or q1 is None or q0["hidden"] > 0.1 or q1["hidden"] > 0.1:
                continue
            sc = 256 / 512
            cov = cover_disk(q1["u"], q1["v"], q1["r_px"], sc)
            ul = np.zeros((S, 2)); d = np.array([q0["u"] - q1["u"], q0["v"] - q1["v"]]) * sc / 16
            obj = cov >= 0.5; ul[obj] = d
            if np.linalg.norm(d) < 0.5 or obj.sum() == 0:
                continue
            reg(("vobench", "label", "obj"), h, t, ul, obj)
            reg(("vobench", "raft_on_label_cells", "obj"), h, t, fl[t + 1][0], obj)
            e = fl[t + 1][1]
            reg(("vobench", "raft_on_label_cells", "obj&fb<0.1"), h, t, fl[t + 1][0], obj & (e < 0.1))
            reg(("vobench", "label", "obj&fb<0.1"), h, t, ul, obj & (e < 0.1))
            raft_bins("vobench", h, t, fl[t + 1])
print("vobench done", flush=True)

# ---------------- 3D 합성 (내 캐시, 라벨)
for f in sorted(glob.glob(os.path.join(P3, "h_val_target_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    X_ = np.load(f, mmap_mode="r"); PX = np.load(os.path.join(P3, f"pix_val_s{sh}.npy"), mmap_mode="r"); lab = np.load(os.path.join(P3, f"lab_val_s{sh}.npz"))
    for w in range(X_.shape[0]):
        h = np.asarray(X_[w]).astype(np.float32); rep = np.asarray(PX[w])[1::2]
        ts = list(range(4, 15))
        fl = flows(rep, [(t + 1, t) for t in ts])
        pos = lab["pos"][w]; vis = lab["vis_frac"][w]; bb = np.nan_to_num(lab["bbox"][w], nan=-100.0)
        for t in ts:
            raft_bins("phys3d", h, t, fl[t + 1])
            p0, p1 = pos[2 * t + 1], pos[2 * t + 3]
            if not (np.isfinite(p0).all() and np.isfinite(p1).all()) or vis[2 * t + 1] < 0.7 or vis[2 * t + 3] < 0.7:
                continue
            d = (p0 - p1) / 16
            if np.linalg.norm(d) < 0.5:
                continue
            cov = np.maximum(cover_box(bb[2 * t + 2]), cover_box(bb[2 * t + 3]))
            obj = cov >= 0.5; ul = np.zeros((S, 2)); ul[obj] = d
            if obj.sum() == 0:
                continue
            e = fl[t + 1][1]
            reg(("phys3d", "label", "obj"), h, t, ul, obj)
            reg(("phys3d", "raft_on_label_cells", "obj"), h, t, fl[t + 1][0], obj)
            reg(("phys3d", "label", "obj&fb<0.1"), h, t, ul, obj & (e < 0.1))
            reg(("phys3d", "raft_on_label_cells", "obj&fb<0.1"), h, t, fl[t + 1][0], obj & (e < 0.1))
    print("phys3d", sh, flush=True)

res = {}
print("\n== b (움직이는 칸 회귀의 '옮긴 h_t' 계수) — 소스 · 흐름 · 칸 선별 (n = 칸 수)")
for key in sorted(ACC):
    A, bv, n = ACC[key]
    if n < 30:
        continue
    b = float(np.linalg.solve(A, bv)[1])
    res["/".join(key)] = dict(b=b, n_cells=int(n))
    print(f"  {key[0]:8s} {key[1]:20s} {key[2]:16s} b {b:.3f} (n {n})")
json.dump(res, open(os.path.join(REPO, "auto_research/results/flow_reliability.json"), "w"), indent=1)
