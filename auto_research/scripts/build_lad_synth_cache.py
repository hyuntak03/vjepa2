#!/usr/bin/env python3
"""[P] 합성 세트를 ssv2_classes 와 같은 (h, U) 형식으로 — lad_profile_synth.py · direction_probe.py 입력 (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/build_lad_synth_cache.py

출력: /workspace/auto_research_cache/lad_synth/{phys3d,vobench}_h.npy (W, 8, 256, 1280) fp16 · _U.npy (W, 8, 16, 16, 2) fp32.
  phys3d : [E] enc_noise phys3d_target_encoder_s{0,1}_tok 변형 0 + 기존 칸 흐름 natural_transport_ctx5/rows_phys3d.npz Uwin (같은 창 순서).
  vobench: [E] enc_noise vobench_target_encoder_s{0,1}_tok 변형 0 + 칸 흐름을 새로 (픽셀 캐시의 블록 대표 장 = 둘째 장, RAFT-large 12 회,
           U[b] = 블록 b 대표 장 → 블록 b−1 대표 장, 16 px 칸 평균 / 16, U[0] = 0) — natural_transport · ip2_block_flow 와 같은 규약."""
import os
import numpy as np, torch, torch.nn.functional as F

torch.set_num_threads(8)
dev = torch.device("cuda")
REPO = "/workspace/world_model/vjepa2"
EN = "/workspace/auto_research_cache/enc_noise"
OUT = "/workspace/auto_research_cache/lad_synth"; os.makedirs(OUT, exist_ok=True)
S, G = 256, 16


def toks(src):
    return np.concatenate([np.asarray(np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")[:, 0]) for sh in (0, 1)])


# 3D 합성
h = toks("phys3d"); U = np.load(os.path.join(REPO, "auto_research/results/natural_transport_ctx5/rows_phys3d.npz"))["Uwin"].astype(np.float32)
assert h.shape[0] == U.shape[0], (h.shape, U.shape)
np.save(os.path.join(OUT, "phys3d_h.npy"), h.astype(np.float16)); np.save(os.path.join(OUT, "phys3d_U.npy"), U)
print("phys3d", h.shape, U.shape, flush=True)

# VOBench
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
h = toks("vobench"); Us = []
for sh in (0, 1):
    PX = np.load(os.path.join(EN, f"vobench_target_encoder_s{sh}_pix.npy"), mmap_mode="r")
    for w in range(PX.shape[0]):
        x = torch.from_numpy(np.asarray(PX[w])).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
        rep = x[1::2]                                                           # 블록 대표 장 8 개
        with torch.no_grad():
            f = raft(rep[1:], rep[:-1], num_flow_updates=12)[-1]                # 블록 b → b−1 (b = 1..7)
        u = (F.avg_pool2d(f, 16) / 16).permute(0, 2, 3, 1).cpu().numpy()        # (7, 16, 16, 2)
        Us.append(np.concatenate([np.zeros((1, G, G, 2), np.float32), u], 0))
U = np.stack(Us).astype(np.float32)
assert h.shape[0] == U.shape[0]
np.save(os.path.join(OUT, "vobench_h.npy"), h.astype(np.float16)); np.save(os.path.join(OUT, "vobench_U.npy"), U)
print("vobench", h.shape, U.shape, "움직이는 칸 몫 (|U|≥0.5)", float((np.linalg.norm(U[:, 1:], axis=-1) >= 0.5).mean()), flush=True)
