#!/usr/bin/env python3
"""블록별 (2 장씩 따로) vs 영상 한 번에 (32 장) encoding — 인접 블록 토큰 변화가 어떻게 다른가 (2026-10-27, 사용자 요청, VOBench 몇 개로 빠르게).

  CUDA_VISIBLE_DEVICES=7 python z_research/scripts/analysis/video_vs_block_encoding.py --n 24

데이터: VOBench_v1 visible (가림 없음) pos_a, 30 fps PNG 3 장 간격 32 장 (블록 16 개, 블록 사이 0.2 초) — 채점과 같은 배치.
토큰: LN(target_encoder) — ① 블록마다 2 장만 넣음 (지금 학습 · 채점의 정답) ② 32 장을 한 번에 넣고 tubelet 별로 자름.
칸 흐름: RAFT (장 2b+2 → 2b) 16×16 평균, flow_block_fit.py 와 같다. 칸을 흐름 크기로 나눠
  복사 |h_{b+1} − h_b| · 흐름 옮김 · 90° 위약 → 순 이득 = (위약 − 흐름) / 복사.
추가: 같은 칸 토큰의 블록 간 cos (정지 칸) — 영상 encoding 이 블록 사이 일관성을 높이는가."""
import argparse, csv, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
from app.vjepa_frozen import ar as AR
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24); a = ap.parse_args()
dev = torch.device("cuda"); G, NB, NF = 16, 16, 32
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
sd = _load_checkpoint(CK)["target_encoder"]
def enc(nf):
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=nf, use_rope=True, uniform_power=False)
    _load_state_dict(e, sd, tag="t"); return e.to(dev, torch.bfloat16).eval()
e2, e32 = enc(2), enc(NF)
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "visible" and r["variant"] == "pos_a"]
rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[: a.n]]
def warp(hg, fl):
    yy, xx = torch.meshgrid(torch.arange(G, device=dev).float(), torch.arange(G, device=dev).float(), indexing="ij")
    g = torch.stack([(xx + fl[0]) / (G - 1) * 2 - 1, (yy + fl[1]) / (G - 1) * 2 - 1], -1)[None]
    return F.grid_sample(hg[None], g, mode="bilinear", padding_mode="border", align_corners=True)[0]
R = {"block": [], "video": []}
for r in rows:
    d = os.path.join("/workspace/dataset/benchmarks/VOBench_v1", r["file_name"])
    x = torch.stack([torch.from_numpy(np.asarray(Image.open(os.path.join(d, f"{3 * k:06d}.png")).convert("RGB"))) for k in range(NF)]).to(dev).permute(0, 3, 1, 2).float()
    if x.shape[-1] != 256: x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    xin = ((x.permute(1, 0, 2, 3)[None] / 255 - MEAN) / STD).to(torch.bfloat16)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hb = AR.encode_blocks(e2, xin, 2).float()[0].view(NB, G, G, -1)
            hv = e32(xin).float()[0]
        hv = F.layer_norm(hv, (hv.size(-1),)).view(NB, G, G, -1)
        a1 = x[0:NF - 2:2] / 127.5 - 1; a2 = x[2:NF:2] / 127.5 - 1
        with torch.autocast("cuda", dtype=torch.float16):
            fb = raft(a2.contiguous(), a1.contiguous(), num_flow_updates=12)[-1].float()
        fc = F.avg_pool2d(fb, 16) / 16; rot = torch.stack([-fc[:, 1], fc[:, 0]], 1)
        for name, h in (("block", hb), ("video", hv)):
            for b in range(NB - 1):
                p, n = h[b].permute(2, 0, 1), h[b + 1].permute(2, 0, 1)
                c = (n - p).abs().mean(0); w = (n - warp(p, fc[b])).abs().mean(0); pl = (n - warp(p, rot[b])).abs().mean(0)
                cs = F.cosine_similarity(n, p, dim=0)
                R[name].append(torch.stack([fc[b].norm(dim=0), c, w, pl, cs], -1).reshape(-1, 5).cpu().numpy())
print(f"VOBench visible pos_a {len(rows)} clip · 블록 쌍 {NB - 1} 개씩\n")
print(f"{'encoding':9s} {'칸 흐름':>10s} {'칸 수':>7s} {'복사 L1':>8s} {'흐름 L1':>8s} {'위약 L1':>8s} {'순 이득':>7s} {'같은 칸 cos':>10s}")
for name, L in R.items():
    X = np.concatenate(L)
    for lo, hi in ((0, 0.25), (0.25, 1), (1, 99)):
        m = (X[:, 0] >= lo) & (X[:, 0] < hi)
        if m.sum() < 30: continue
        c, w, pl = X[m, 1].mean(), X[m, 2].mean(), X[m, 3].mean()
        print(f"{name:9s} {f'{lo}–{hi if hi < 99 else chr(8734)}':>10s} {m.sum():7d} {c:8.4f} {w:8.4f} {pl:8.4f} {100 * (pl - w) / c:6.1f}% {X[m, 4].mean():10.3f}")
