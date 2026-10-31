#!/usr/bin/env python3
"""BlenderTrack 학습 메타 (2026-10-31, z_analysis/release/06_blender_track) — clip 이름 + target 의 프레임별 보이는 비율.

입력: v0 <root>/clips/s{seed}_*/labels.npz (render.py) · v1 <root>/labels/<prefix>*.npz + <root>/clips/<id>.mp4 (gen_render.py, --prefix)
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


def _one(f):
    z = np.load(f)
    a, v = z["amodal_n"][:, 0].astype(np.float32), z["visible_n"][:, 0].astype(np.float32)
    return np.where(a > 0, v / np.maximum(a, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, default=None, help="v0 배치 (clips/s<seed>_*/labels.npz)")
    ap.add_argument("--prefix", default=None, help="v1 배치 (labels/<prefix>*.npz + clips/<id>.mp4), 예: train10_")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if a.prefix:                                   # v1 (2026-10-31): clip 하나 = mp4 하나, 라벨은 labels/
        fs = sorted(glob.glob(f"{a.root}/labels/{a.prefix}*.npz"))
        fs = [f for f in fs if not f.endswith(".tmp.npz") and os.path.exists(f"{a.root}/clips/{os.path.basename(f)[:-4]}.mp4")]
        clips = np.array([os.path.basename(f)[:-4] for f in fs])
        extra = dict(layout=np.array("mp4"), labels_root=np.array(os.path.join(a.root, "labels")))
    else:
        fs = sorted(os.path.join(os.path.dirname(p), "labels.npz") for p in glob.glob(f"{a.root}/clips/s{a.seed}_*/labels.npz"))
        clips = np.array([os.path.basename(os.path.dirname(f)) for f in fs])
        extra = {}
    if not fs:
        raise SystemExit(f"clip 없음: {a.root} (prefix={a.prefix}, seed={a.seed})")
    with Pool(a.workers) as pool:
        vis = np.stack(pool.map(_one, fs, chunksize=16))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, clip=clips, vis=vis.astype(np.float16), root=np.array(os.path.join(a.root, "clips")), **extra)
    on = ~np.isnan(vis)
    print(f"{len(clips)} clips × {vis.shape[1]} 프레임 -> {a.out}")
    print(f"  화면 안 프레임 {on.mean():.3f} · 그중 가려짐 (보이는 비율 < 0.1) {(np.nan_to_num(vis, nan=1) < 0.1)[on].mean():.3f}")


if __name__ == "__main__":
    main()
