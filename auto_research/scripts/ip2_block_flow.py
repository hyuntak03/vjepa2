#!/usr/bin/env python3
"""[P] IntPhys 2 블록 칸 흐름 U (역방향) · V (순방향) — [E] 공유 캐시 pix.npy 위에서 ([T] 요청, 학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/ip2_block_flow.py [--split Main]

입력: /workspace/auto_research_cache/intphys2/<split>/pix.npy (N, 64, 256, 256, 3) uint8 (채점기 전처리, 6 fps), done_s0 · done_s1 (둘 합이 전부일 때만 돈다).
블록 b = 장 2b · 2b+1 (토큰 h 의 32 블록과 같다), 대표 장 = 둘째 장 (pix[:, 1::2]). RAFT-large 12 회, 256².
출력 (같은 폴더): U.npy · V.npy (N, 32, 16, 16, 2) fp16, 행 순서 = meta.json, 마지막 축 (dx, dy), 칸 단위 (16 px 평균 / 16).
  U[:, b] = 블록 b 대표 장 → 블록 b−1 대표 장 (역방향; natural 캐시 · flow_persistence 의 U 와 같은 정의), U[:, 0] = 0.
  V[:, b] = 블록 b → 블록 b+1 (순방향), V[:, 31] = 0.
  UV_rows.npy (N,) bool = 채워진 행 (이어 하기), 다 끝나면 UV.done."""
import argparse, os, time
import numpy as np, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="Main")
ap.add_argument("--root", default="/workspace/auto_research_cache/intphys2")
a = ap.parse_args()
D = os.path.join(a.root, a.split)
done = np.load(os.path.join(D, "done_s0.npy")) | np.load(os.path.join(D, "done_s1.npy"))
assert done.all(), f"[E] 캐시가 아직 다 안 찼다: {done.sum()} / {done.size}"
torch.set_num_threads(8)
dev = torch.device("cuda")
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights   # noqa: E402
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
PIX = np.load(os.path.join(D, "pix.npy"), mmap_mode="r")
N, NF = PIX.shape[:2]; NB, G = NF // 2, 16
mk = lambda name: np.lib.format.open_memmap(os.path.join(D, name), mode="r+" if os.path.exists(os.path.join(D, name)) else "w+", dtype=np.float16, shape=(N, NB, G, G, 2))
Um, Vm = mk("U.npy"), mk("V.npy")
rp = os.path.join(D, "UV_rows.npy")
rows = np.load(rp) if os.path.exists(rp) else np.zeros(N, bool)


@torch.no_grad()
def rflow(i0, i1, bs=16):
    return torch.cat([raft(i0[k:k + bs], i1[k:k + bs], num_flow_updates=12)[-1] for k in range(0, i0.shape[0], bs)])


t0 = time.time()
for i in range(N):
    if rows[i]:
        continue
    x = torch.from_numpy(np.asarray(PIX[i, 1::2])).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1      # (NB, 3, 256, 256)
    with torch.no_grad():
        b = F.avg_pool2d(rflow(x[1:], x[:-1]), 16) / 16                                                 # (NB−1, 2, G, G): 블록 k+1 → k
        f = F.avg_pool2d(rflow(x[:-1], x[1:]), 16) / 16                                                 # 블록 k → k+1
    U = np.zeros((NB, G, G, 2), np.float16); V = np.zeros((NB, G, G, 2), np.float16)
    U[1:] = b.permute(0, 2, 3, 1).cpu().numpy(); V[:-1] = f.permute(0, 2, 3, 1).cpu().numpy()
    Um[i] = U; Vm[i] = V
    rows[i] = True
    if i % 50 == 0:
        Um.flush(); Vm.flush(); np.save(rp, rows)
        print(f"{i}/{N} ({time.time() - t0:.0f}s)", flush=True)
Um.flush(); Vm.flush(); np.save(rp, rows)
open(os.path.join(D, "UV.done"), "w").write(f"{rows.sum()} rows\n")
print("끝", rows.sum(), flush=True)
