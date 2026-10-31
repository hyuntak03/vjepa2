#!/usr/bin/env python3
"""BlenderTrack 학습 메타 (2026-10-31, z_analysis/release/06_blender_track) — clip 이름 + target 의 프레임별 보이는 비율.

입력: <root>/clips/s{seed}_*/labels.npz (z_research/scripts/data/blender_track/render.py)
출력: <out>.npz  clip (N,) · vis (N, T) float16 = visible_n / amodal_n (target, 화면 밖 = nan) · root
      app/vjepa_frozen/blender_track_data.py 가 이것으로 late 창 후보를 찾는다.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_training/data/build_blender_track_index.py --root /data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track/v0 --seed 0 \
     --out data_csv/blender_track_v0/train_meta.npz
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np


def _one(d):
    z = np.load(os.path.join(d, "labels.npz"))
    a, v = z["amodal_n"][:, 0].astype(np.float32), z["visible_n"][:, 0].astype(np.float32)
    return np.where(a > 0, v / np.maximum(a, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    ds = sorted(os.path.dirname(p) for p in glob.glob(f"{a.root}/clips/s{a.seed}_*/labels.npz"))
    if not ds:
        raise SystemExit(f"clip 없음: {a.root}/clips/s{a.seed}_*")
    with Pool(a.workers) as pool:
        vis = np.stack(pool.map(_one, ds, chunksize=16))
    clips = np.array([os.path.basename(d) for d in ds])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, clip=clips, vis=vis.astype(np.float16), root=np.array(os.path.join(a.root, "clips")))
    on = ~np.isnan(vis)
    print(f"{len(clips)} clips × {vis.shape[1]} 프레임 -> {a.out}")
    print(f"  화면 안 프레임 {on.mean():.3f} · 그중 가려짐 (보이는 비율 < 0.1) {(np.nan_to_num(vis, nan=1) < 0.1)[on].mean():.3f}")


if __name__ == "__main__":
    main()
