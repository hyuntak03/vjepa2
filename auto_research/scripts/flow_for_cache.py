#!/usr/bin/env python3
"""[T] 이미 만든 학습 캐시 (블록 토큰만, 흐름 끈 판) 의 같은 창을 다시 디코딩해 칸 단위 RAFT 역방향 흐름만 계산한다 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/flow_for_cache.py --src k710_320

읽기는 cache_natural.py 와 같다 (메타의 경로 · 시작 장 · 간격, 가운데 정사각 → 256 bilinear → uint8 반올림).
저장: <cache>/{src}_Uflow.npy (N, nb, 16, 16, 2) fp16, U[:, b] = 블록 b 대표 장 (둘째 장) → 블록 b−1 대표 장, 칸 단위 (U[:, 0] = 0)."""
import argparse, json, os
import numpy as np, torch, torch.nn.functional as F
import decord
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--cache", default="/workspace/auto_research_cache/natural_train")
ap.add_argument("--nb", type=int, default=8)
ap.add_argument("--fwd", type=int, default=0, help="1 = 순방향 흐름 (블록 b → b+1) 을 {src}_Vflow.npy 로")
a = ap.parse_args()
torch.set_num_threads(8)
dev = torch.device("cuda")
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
meta = json.load(open(os.path.join(a.cache, f"{a.src}_meta.json")))
NF = 2 * a.nb
OUT = []
for i, m in enumerate(meta):
    vr = decord.VideoReader(m["path"], num_threads=1)
    fr = vr.get_batch(list(range(m["start"], m["start"] + (NF - 1) * m["fstp"] + 1, m["fstp"]))).asnumpy()
    Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
    x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()
    with torch.no_grad():
        rep = x[1::2] / 127.5 - 1
        f = raft(rep[:-1], rep[1:], num_flow_updates=12)[-1] if a.fwd else raft(rep[1:], rep[:-1], num_flow_updates=12)[-1]
    U = torch.zeros(a.nb, 16, 16, 2, device=dev)
    if a.fwd:
        U[:-1] = (F.avg_pool2d(f, 16) / 16).permute(0, 2, 3, 1)          # V[b] = 블록 b → b+1 (V[-1] = 0)
    else:
        U[1:] = (F.avg_pool2d(f, 16) / 16).permute(0, 2, 3, 1)
    OUT.append(U.half().cpu().numpy())
    if (i + 1) % 50 == 0:
        print(f"{a.src}: {i + 1}/{len(meta)}", flush=True)
np.save(os.path.join(a.cache, f"{a.src}_{'Vflow' if a.fwd else 'Uflow'}.npy"), np.stack(OUT))
print("done", len(OUT), flush=True)
