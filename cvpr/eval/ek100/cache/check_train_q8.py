#!/usr/bin/env python3
"""train 캐시 검사 (2026-10-16): view v · chunk c 를 같은 seed 로 다시 만들어 (1) 증강 재현 (at 일치) (2) 캐시 (int8 이면 복원) vs 새로 계산 상대 L1.
  CUDA_VISIBLE_DEVICES=<g> python cvpr/eval/ek100/cache/check_train_q8.py --config <yaml> [--view 0 --chunk 0]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[4])); sys.path.insert(0, str(Path(__file__).parent))
import extract as X

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--view", type=int, default=0); ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--out", default="/data2/local_datasets/world/world_analysis/cache/ek100_feats"); a = ap.parse_args()
cfg, data = X.load_cfg(a.config)
D = Path(a.out) / f"{data.get('anticipation_point_mode')}_{data.get('time_source')}_{Path(data['dataset_train']).stem}" / "train"
info = json.load(open(D / "info.json")); _, anos, rows, n_clip = X.clip_table(cfg, data, "train")
ds = X.ChunkSet(cfg, data, "train", anos, rows, info["views"], 0, 1, set()); k = ds.items.index((a.view, a.chunk)); it = ds[k]
meta = [json.loads(l) for f in sorted(D.glob("meta_shard*.jsonl")) for l in open(f)]
m = next(d for d in meta if d["view"] == a.view and d["chunk"] == a.chunk)
print("at 재현:", np.allclose(m["at"], it["at"]), "· 행", it["row"][:3], "…")
enc = X.build_encoder(cfg, data, torch.device("cuda:0"))
x = torch.stack([v for v, ok in zip(it["video"], it["valid"]) if ok]).cuda()
o = X.encode(enc, x, ["joint", "blocks"]); rr = [a.view * n_clip + r for r, ok in zip(it["row"], it["valid"]) if ok]
for kk in ("joint", "blocks"):
    got = torch.from_numpy(X.read_rows(D, kk, rr)).cuda(); ref = o[kk].float()
    print(f"{kk}: 캐시 vs 새로 계산 상대 L1 {float((got - ref).abs().mean() / ref.abs().mean()):.2e}")
