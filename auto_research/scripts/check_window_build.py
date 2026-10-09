#!/usr/bin/env python3
"""창 길이 (C+P) 마다 모델을 다시 지어야 하나? — window_size 64 로 한 번 지은 bundle 과
창 총합 그대로 지은 bundle 이 같은 z / p / h 를 내는지 RollOut_v3 clip 몇 개로 비교한다.

RoPE 위치는 토큰 번호와 grid_size 로만 계산된다 (modules.py RoPEAttention.separate_positions).
window_size 가 바꾸는 것은 predictor.num_patches (mask token 을 몇 개 복제하나) 뿐이라
같아야 한다. 같으면 H2~H5 는 bundle 하나로 16 창을 다 돈다.

  srun -w vll6 ... python auto_research/scripts/check_window_build.py
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402

dev = torch.device("cuda:0")
cfg64 = arlib.v3_cfg(4, 60)                      # 모델 쪽만 쓴다
b64 = arlib.build_bundle(cfg64, dev, window=64)
for C, P in [(16, 16), (4, 4), (32, 32), (8, 32)]:
    cfg = arlib.v3_cfg(C, P)
    ds = arlib.v3_dataset(C, P)
    clips = torch.stack([ds.clip(i) for i in (0, 500, 1500, 2900)])
    bw = arlib.build_bundle(cfg, dev, window=C + P)
    a = arlib.forward(bw, clips, cfg, ctx_frames=C)
    b = arlib.forward(b64, clips, cfg, ctx_frames=C)
    d = {k: (a[k].float() - b[k].float()).abs().max().item() for k in ("z", "p", "h")}
    print(f"C{C}_P{P}  max|Δ| z {d['z']:.2e}  p {d['p']:.2e}  h {d['h']:.2e}", flush=True)
    del bw
    torch.cuda.empty_cache()
