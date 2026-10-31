#!/usr/bin/env python3
"""블록 단위 분포 손실이 "물체 지우기" 를 L1 보다 더 나쁘게 매기나 (2026-10-27, 사용자 요청: 지우는 게 손실이 더 크게).

  CUDA_VISIBLE_DEVICES=7 python z_research/scripts/analysis/set_loss_ranking.py --n 150

데이터: phys3d_single_v1 val (결정적 3D 합성, 물체 bbox 라벨), 30 fps 3 장 간격 = 블록 0.2 초.
블록 쌍 (b−1, b): 물체가 두 블록 모두 보이고 (vis > 0.3) 물체 칸 집합이 겹치지 않을 만큼 움직였을 때 (O_b ∩ O_{b−1} = ∅).
  물체 칸 O = bbox 가 덮는 16 px 칸. 배경 토큰 = 물체가 없던 쪽 블록의 같은 칸 토큰.
정답 블록 h_b 에 대해 가짜 예측 (모두 정답 토큰 공간):
  지움   — O_b 칸을 배경 (h_{b−1}[O_b]) 으로
  정지   — 지움 + O_{b−1} 칸에 옛 물체 (h_{b−1}[O_{b−1}]) 를 남김  (= 안 옮기고 제자리)
  섞음   — 정답과 지움의 칸별 평균 (L1 이 고르는 "흐린" 예측의 대용)
손실: L1 (칸·차원 평균) · 분포 손실 = 블록 토큰 256 개 집합 사이의 sliced Wasserstein (무작위 투영 256 개, 정렬 매칭, L1) — 위치 무시.
보고: 각 가짜의 손실 평균 (정답 = 0) 과 "지움 > 정지" 비율 (원하는 순서), 손실별 순위."""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F, decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
from app.vjepa_frozen import ar as AR
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150); a = ap.parse_args()
dev = torch.device("cuda"); torch.manual_seed(0); G, NB, ST = 16, 16, 3
torch.cuda.set_per_process_memory_fraction(0.08)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
PROJ = F.normalize(torch.randn(1280, 256, device=dev), dim=0)

def cells(bb, sc):
    x0, y0, x1, y1 = [v * sc for v in bb]
    c = set()
    for r in range(max(0, int(y0 // 16)), min(G, int(y1 // 16) + 1)):
        for q in range(max(0, int(x0 // 16)), min(G, int(x1 // 16) + 1)):
            c.add(r * G + q)
    return c
def sw(p, q):
    return (torch.sort(p @ PROJ, 0)[0] - torch.sort(q @ PROJ, 0)[0]).abs().mean().item()
def l1(p, q): return (p - q).abs().mean().item()

paths = sorted(p for p in glob.glob("/workspace/dataset/training_data/phys3d_single_v1/val/clips/*/*.mp4") if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= 96)
paths = [paths[i] for i in np.random.default_rng(0).permutation(len(paths))[: a.n]]
R = {k: [] for k in ("지움", "정지", "섞음")}
for p in paths:
    st = np.load(p[:-4] + "_states.npz"); vr = decord.VideoReader(p, num_threads=2)
    idx = [ST * k for k in range(2 * NB)]
    x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
    sc = 256 / x.shape[-1]
    if x.shape[-1] != 256: x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = AR.encode_blocks(tgt, ((x - MEAN) / STD).to(torch.bfloat16), 2).float()[0].view(NB, G * G, -1)
    for b in range(1, NB):
        f0, f1 = idx[2 * b - 1], idx[2 * b + 1]
        if not (st["vis_frac"][f0, 0] > 0.3 and st["vis_frac"][f1, 0] > 0.3): continue
        Oo, On = cells(st["bbox"][f0, 0], sc), cells(st["bbox"][f1, 0], sc)
        if not Oo or not On or (Oo & On): continue
        On_, Oo_ = torch.tensor(sorted(On), device=dev), torch.tensor(sorted(Oo), device=dev)
        gt = h[b]; er = gt.clone(); er[On_] = h[b - 1][On_]
        stc = er.clone(); stc[Oo_] = h[b - 1][Oo_]
        mx = 0.5 * (gt + er)
        for k, v in (("지움", er), ("정지", stc), ("섞음", mx)):
            R[k].append((l1(v, gt), sw(v, gt), len(On)))
n = len(R["지움"])
print(f"블록 쌍 {n} 개 (물체가 자기 크기 이상 움직임) · 물체 칸 평균 {np.mean([r[2] for r in R['지움']]):.1f} / 256\n")
print(f"{'가짜 예측':8s} {'L1':>10s} {'분포 손실 (SW)':>14s}")
for k, v in R.items():
    v = np.array(v); print(f"{k:8s} {v[:, 0].mean():10.5f} {v[:, 1].mean():14.5f}")
E, S, M = (np.array(R[k]) for k in ("지움", "정지", "섞음"))
print(f"\n'지움' 손실 > '정지' 손실인 비율:  L1 {100 * np.mean(E[:, 0] > S[:, 0]):.1f}% · 분포 {100 * np.mean(E[:, 1] > S[:, 1]):.1f}%")
print(f"'지움' 손실 > '섞음' 손실인 비율:  L1 {100 * np.mean(E[:, 0] > M[:, 0]):.1f}% · 분포 {100 * np.mean(E[:, 1] > M[:, 1]):.1f}%")
for w in (0.5, 1.0, 2.0, 5.0):
    c = lambda X: X[:, 0] + w * X[:, 1]
    print(f"L1 + {w} × 분포:  지움 > 정지 {100 * np.mean(c(E) > c(S)):.1f}% · 평균 지움 {c(E).mean():.5f} 정지 {c(S).mean():.5f} 섞음 {c(M).mean():.5f}")
