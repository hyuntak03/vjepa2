#!/usr/bin/env python3
"""v11 split_test 의 late 가림 · vanish 블록 (336 block × 4 variant = 1,344 clip) 을 uint8 npy 로 스테이징 (2026-09-29).
IntPhysGen_v11 PNG 가 있는 노드 (vll5 /local_datasets) 의 CPU 에서 돈다.

    python stage_v11_late.py --out auto_research/_stage/v11_late      # items.json + {i:04d}.npy (32, 256, 256, 3)

프레임 = raw 0,3,…,93 (32 장, surprise_c16t32 와 같다). 리사이즈가 필요하면 antialias=False bilinear (CLAUDE.md §1-7).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
FR = Path("/local_datasets/world/world_analysis/IntPhysGen_v11"); IDX = ROOT / "data_csv/intphysgen_v11_split/index_test.csv"
LATE = ("static_occlusion", "moving_occlusion_flat", "moving_occlusion")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IDX); df = df[df.condition.isin(LATE) & (df.violation_type == "vanish")].sort_values(["block_id", "variant"]).reset_index(drop=True)
    if a.limit: df = df[df.block_id.isin(sorted(df.block_id.unique())[: a.limit])].reset_index(drop=True)
    items = []
    for i, r in df.iterrows():
        f = out / f"{i:04d}.npy"
        it = dict(i=int(i), video_id=str(r.video_id), file_name=r.file_name, block_id=int(r.block_id), variant=r.variant, plausible=int(r.plausible), pair_id=str(r.pair_id),
                  condition=r.condition, sym_k=int(r.sym_k), role=r.role, npy=str(f))
        items.append(it)
        if f.exists() and f.stat().st_size > 0: continue
        frs = []
        for k in range(32):
            im = Image.open(FR / r.file_name / f"{3 * k:06d}.png").convert("RGB")
            if im.size != (256, 256):
                import torch, torch.nn.functional as F
                t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float()[None]
                im = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False, antialias=False)[0].round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
            frs.append(np.asarray(im, dtype=np.uint8))
        np.save(f, np.stack(frs))
        if i % 100 == 0: print(i, len(df), flush=True)
    json.dump(items, open(out / "items.json", "w")); print("done", len(items))


if __name__ == "__main__":
    main()
