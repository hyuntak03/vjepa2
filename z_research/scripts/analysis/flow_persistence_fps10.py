#!/usr/bin/env python3
"""0.2 초 블록 (fps 10) 에서 다음 움직임이 지금 움직임으로 맞춰지나 — 흐름 지속 (2026-10-27, 사용자 질문).

  CUDA_VISIBLE_DEVICES=7 python z_research/scripts/analysis/flow_persistence_fps10.py --n 60

흐름: RAFT, 블록 b → b+1 = 장 2b → 2b+2 (0.2 초), 16×16 칸 평균, 칸 단위. 같은 칸의 다음 흐름 u_{b+1} 을 지금 흐름 u_b 로:
  k = Σ u_{b+1}·u_b / Σ |u_b|²  (1 = 등속 지속, 0 = 멈춤),  R² = 1 − Σ|u_{b+1} − k u_b|² / Σ|u_{b+1} − ū|²
  ⚠️ 같은 칸에서 잰다 — 물체가 칸을 벗어나면 다음 흐름은 다른 내용의 흐름이다 (0.2 초 이동은 대부분 1 칸 미만이라 영향 작음).
  두 판: 그대로 (카메라 포함) · 카메라 뺌 (각 블록 쌍의 칸 흐름 중앙값을 빼서 화면 전체 움직임 제거 = 물체 상대 움직임)
움직이는 칸만: |u_b| ≥ 0.25 칸 (카메라 뺀 판은 뺀 뒤 크기 기준).
데이터: K710 · SSv2 · HowTo (fps10t 표, 16 장) · phys3d_single val · VOBench visible (30 fps 3 장 간격, 32 장)."""
import argparse, csv, glob, os, random, sys
import numpy as np, torch, torch.nn.functional as F, decord
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from app.vjepa_frozen.var_window import _Table
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); a = ap.parse_args()
dev = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(0.08)
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
jobs = []
for name, path in (("K710", "data_csv/k710_320/fps10t_table.tsv"), ("SSv2", "data_csv/ssv2/fps10t_table.tsv"), ("HowTo", "data_csv/ht100m_1pct/fps10t_table.tsv")):
    t = _Table(name, os.path.join(REPO, path), 1.0); lo = t.first_ok(16); r = random.Random(f"fp-{name}")
    jobs += [(name, t.paths[j], t.fps[j], r.random()) for j in (r.randrange(lo, len(t.paths)) for _ in range(a.n))]
ph = sorted(glob.glob("/workspace/dataset/training_data/phys3d_single_v1/val/clips/*/*.mp4"))
jobs += [("3D 합성", p, 30, None) for p in random.Random("fp-p").sample(ph, a.n)]
vb = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "visible" and r["variant"] == "pos_a"]
jobs += [("VOBench", os.path.join("/workspace/dataset/benchmarks/VOBench_v1", vb[i]["file_name"]), 30, "png") for i in np.random.default_rng(0).permutation(len(vb))[: a.n]]

def load(job):
    name, p, f, u = job
    if u == "png":
        x = torch.stack([torch.from_numpy(np.asarray(Image.open(os.path.join(p, f"{3 * k:06d}.png")).convert("RGB"))) for k in range(32)])
    else:
        vr = decord.VideoReader(p, num_threads=2); n = len(vr)
        if u is None:
            idx = [3 * k for k in range(16)]
            if idx[-1] >= n: return None
        else:
            tmax = (n - 1) / f - 1.5
            if tmax < 0: return None
            t0 = u * tmax; idx = [min(n - 1, int(round((t0 + k / 10) * f))) for k in range(16)]
        x = torch.from_numpy(vr.get_batch(idx).asnumpy())
    x = x.to(dev).permute(0, 3, 1, 2).float(); H, W = x.shape[-2:]; s = 256 / min(H, W)
    x = F.interpolate(x, size=(max(256, round(H * s)), max(256, round(W * s))), mode="bilinear", align_corners=False)
    H, W = x.shape[-2:]; return x[:, :, (H - 256) // 2:(H - 256) // 2 + 256, (W - 256) // 2:(W - 256) // 2 + 256]

P = {}
for job in jobs:
    try: x = load(job)
    except Exception: continue
    if x is None: continue
    nb = x.size(0) // 2
    a1 = x[0:2 * nb - 2:2] / 127.5 - 1; a2 = x[2:2 * nb:2] / 127.5 - 1
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        fl = torch.cat([raft(a1[i:i + 4].contiguous(), a2[i:i + 4].contiguous(), num_flow_updates=12)[-1].float() for i in range(0, len(a1), 4)])
    u = (F.avg_pool2d(fl, 16) / 16).flatten(2).transpose(1, 2).cpu().numpy()     # (nb-1, 256, 2)
    uc = u - np.median(u, axis=1, keepdims=True)                                  # 카메라 (화면 중앙값) 뺌
    for tag, U in (("그대로", u), ("카메라 뺌", uc)):
        cur, nxt = U[:-1].reshape(-1, 2), U[1:].reshape(-1, 2)
        P.setdefault((job[0], tag), []).append((cur, nxt))
print(f"{'데이터':8s} {'판':8s} {'움직이는 칸':>10s} {'k (지속)':>9s} {'R²':>7s} {'방향 cos 중앙':>12s} {'크기 비 중앙':>11s}")
for (src, tag), L in P.items():
    c = np.concatenate([l[0] for l in L]); n = np.concatenate([l[1] for l in L])
    m = np.linalg.norm(c, axis=1) >= 0.25
    c, n = c[m], n[m]
    k = (c * n).sum() / (c * c).sum()
    r2 = 1 - ((n - k * c) ** 2).sum() / ((n - n.mean(0)) ** 2).sum()
    cos = ((c * n).sum(1) / (np.linalg.norm(c, axis=1) * np.linalg.norm(n, axis=1) + 1e-9))
    ratio = np.linalg.norm(n, axis=1) / np.linalg.norm(c, axis=1)
    print(f"{src:8s} {tag:8s} {m.sum():10d} {k:9.3f} {r2:7.3f} {np.median(cos):12.3f} {np.median(ratio):11.3f}")
