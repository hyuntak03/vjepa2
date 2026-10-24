#!/usr/bin/env python3
"""[물리] VOBench 끝 가림 216 block × 4 팔의 미래 블록 칸 흐름 U (역방향) · V (순방향) — 카메라 상태 흡수 검사용 (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/cam_state_vob_flow.py

블록 b = 장 IDX[2b] · IDX[2b+1] (IDX = 0, 3, …, 93), 대표 장 = 둘째 장. 가운데 정사각 → 256 bilinear. RAFT-large 12 회 (ip2_block_flow 와 같다).
출력 /workspace/auto_research_cache/cam_state/vob_UV.npz: U (216, 4, 8, 16, 16, 2) = 블록 8..15 → 직전 블록, V 같은 모양 = 직전 블록 → 블록 8..15.
팔 순서 pos_a · pos_b · imp_ab · imp_ba, block 순서 = vobench_late/meta.json."""
import json, os, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

VB = "/workspace/dataset/benchmarks/VOBench_v1"
OUTD = "/workspace/auto_research_cache/cam_state"; os.makedirs(OUTD, exist_ok=True)
ARMS = ["pos_a", "pos_b", "imp_ab", "imp_ba"]
IDX = list(range(0, 96, 3))[:32]
BL = list(range(7, 16))                                                # 대표 장이 필요한 블록 7..15
torch.set_num_threads(8)
dev = torch.device("cuda")
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
meta = json.load(open("/workspace/auto_research_cache/vobench_late/meta.json"))


def load(d):
    fr = np.stack([np.asarray(Image.open(os.path.join(VB, d, f"{IDX[2 * b + 1]:06d}.png")).convert("RGB")) for b in BL])
    x = torch.from_numpy(fr).to(dev).permute(0, 3, 1, 2).float()
    H, W = x.shape[-2:]; s = min(H, W); y0, x0 = (H - s) // 2, (W - s) // 2
    x = x[:, :, y0:y0 + s, x0:x0 + s]
    if s != 256:
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).round().clamp(0, 255)
    return x / 127.5 - 1


U = np.zeros((len(meta), 4, 8, 16, 16, 2), np.float16); V = np.zeros_like(U)
t0 = time.time()
for i, m in enumerate(meta):
    xs = torch.cat([load(m["files"][a_]) for a_ in ARMS])                 # (4·9, 3, 256, 256)
    i0 = torch.cat([xs[k * 9 + 1: k * 9 + 9] for k in range(4)]); i1 = torch.cat([xs[k * 9: k * 9 + 8] for k in range(4)])
    with torch.no_grad():
        b = torch.cat([raft(i0[j:j + 16], i1[j:j + 16], num_flow_updates=12)[-1] for j in range(0, 32, 16)])   # 블록 b → b−1
        f = torch.cat([raft(i1[j:j + 16], i0[j:j + 16], num_flow_updates=12)[-1] for j in range(0, 32, 16)])   # 블록 b−1 → b
    U[i] = (F.avg_pool2d(b, 16) / 16).permute(0, 2, 3, 1).reshape(4, 8, 16, 16, 2).cpu().numpy()
    V[i] = (F.avg_pool2d(f, 16) / 16).permute(0, 2, 3, 1).reshape(4, 8, 16, 16, 2).cpu().numpy()
    if i % 20 == 0:
        print(f"{i}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)
np.savez(os.path.join(OUTD, "vob_UV.npz"), U=U, V=V, blocks=np.array([m["block_id"] for m in meta]))
print("끝", time.time() - t0)
