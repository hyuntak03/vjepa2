#!/usr/bin/env python3
"""arlib 검증 — hook 을 단 층별 forward 가 표준 채점과 같은 수치를 내는가.

검사 세 가지 (전부 통과해야 층별 분석을 믿는다):
  1. hook 을 단 forward 의 p == hook 없는 forward 의 p  (max |Δ| = 0 이어야 한다)
  2. 렌즈 L11 = predictor_proj(predictor_norm(q_11[미래])) == p   (max |Δ| = 0)
  3. surprise = mean|p − LN(h)[미래]| 가 표준 채점 per_video_surprise 와 같은가.
     표준 채점은 GPUS=8, batch 16 이었다 → rank 0 의 첫 batch (records[0::8][:16]) 를 같은 구성으로 재현한다
     (fp16 autocast 는 batch 구성에 따라 값이 조금 달라진다 — CLAUDE.md §7-2).

  python auto_research/scripts/validate_arlib.py [--cfg surprise_c16t32__v11_vith] [--ref <per_block.json>]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", default="surprise_c16t32__v11_vith")
ap.add_argument("--ref", default=str(arlib.ROOT / "z_research/IntPhysGenV11/exp_results/surprise_c16t32__v11_vith/per_block.json"))
ap.add_argument("--ws", type=int, default=8)
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()

dev = torch.device("cuda:0")
cfg = arlib.load_cfg(a.cfg)
t0 = time.time()
ds, bundle = arlib.build(cfg, dev)
print(f"build {time.time()-t0:.0f}s  n={len(ds)}", flush=True)
idx = list(range(0, len(ds), a.ws))[: a.bs]
clips = torch.stack([ds.clip(i) for i in idx])
vids = [ds.records[i].video_id for i in idx]

plain = arlib.forward(bundle, clips, cfg, want_h=True)
hooked = arlib.forward(bundle, clips, cfg, enc_layers=[0, 15, 31], pred_layers=list(range(12)), want_h=True, want_lens=True)
d1 = (plain["p"].float() - hooked["p"].float()).abs().max().item()
d2 = (hooked["lens"][11].float() - hooked["p"].float()).abs().max().item()
fs = arlib.future_slice()
s = arlib.l1_surprise(plain["p"], plain["h"][:, fs]).cpu().numpy()
ref = json.load(open(a.ref))["per_video_surprise"]
r = np.array([ref[v] if not isinstance(ref[v], dict) else ref[v]["all"] for v in vids])
d3 = np.abs(s - r)
print(f"[1] hook vs plain p      max|Δ| = {d1:.3e}")
print(f"[2] lens L11 vs p        max|Δ| = {d2:.3e}")
print(f"[3] surprise vs 표준채점  max|Δ| = {d3.max():.3e}  (rel {np.max(d3/r):.2e})  n={len(vids)}")
for v, x, y in list(zip(vids, s, r))[:4]:
    print(f"    {v}  ours {x:.6f}  ref {y:.6f}")
print("shapes:", {k: tuple(v.shape) for k, v in hooked["z_layers"].items()}, tuple(hooked["q_layers"][0].shape), tuple(hooked["z"].shape))
ok = d1 == 0 and d2 == 0 and d3.max() < 1e-4
print("PASS" if ok else "FAIL")
