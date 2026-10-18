#!/usr/bin/env python3
"""IntPhys 2 Main **샘플 (쌍) 단위** split + 평가와 같은 창으로 학습 데이터 만들기 (2026-10-15, 사용자: (2) 샘플 단위 · 창은 평가와 똑같이).

split: 장면 (SceneIndex) 마다 쌍 1 · 쌍 2 중 하나를 무작위로 (seed 0) train 쌍으로 —
  train = 그 쌍의 가능 영상 (253 clip) / test = 다른 쌍의 가능 + 불가능 (253 쌍, 채점기 split: MainPair).
  ⚠️ 같은 장면의 다른 쌍을 학습에서 본다 — 장면 단위 (build_intphys2_main_split.py) 보다 느슨한 판, "학습이 되는가" 확인용.

창 (analysis/intphys2 채점과 같게):
  · 디코드 = raw 0, 10, 20, … (frame_step 10 = 6 fps, analysis/intphys2/dataset.frame_step_indices)
  · 리사이즈 = 채점의 build_video_transform(256, crop_margin_ratio=1.0) 의 Resize + CenterCrop 그대로 → 256×256 PNG 로 저장
    (학습 쪽 리사이즈는 256→256 이라 항등)
  · 창 = 48 장, 시작점 0, 2, 4, …, n−48 (채점 stride 2, plan_windows 와 같은 자리)
  시작점마다 index_s{start}.csv 를 따로 쓴다 — 학습 config 가 시작점별 데이터셋 (frames_start = start) 을 크기 비례로 섞어
  모든 (영상, 시작점) 이 같은 확률이 된다. 문맥 C 의 분포는 config 의 mask.context_frames (채점 빈도로 맞춘 목록).

  python z_training/data/build_intphys2_pair_split.py            # → data_gen/IntPhys2_pair/ (프레임) + data_csv/IntPhys2/pair_split/
"""
from __future__ import annotations

import collections
import csv
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = Path("/data/dataset/world/IntPhys2/Main")
FR = ROOT / "data_gen/IntPhys2_pair"
OUT = ROOT / "data_csv/IntPhys2/pair_split"
M, STRIDE, FSTEP = 48, 2, 10
SWEEP = [4, 6, 8, 10, 12, 14, 18, 24, 30, 36, 42]


def extract(args):
    fn, vid = args
    sys.path.insert(0, str(ROOT))
    from decord import VideoReader, cpu
    from PIL import Image
    from analysis.intphys2.dataset import build_video_transform, frame_step_indices
    d = FR / vid
    vr = VideoReader(str(SRC / fn), ctx=cpu(0))
    idx = frame_step_indices(vr, FSTEP)
    if d.exists() and len(list(d.glob("*.png"))) == len(idx):
        return vid, len(idx)
    d.mkdir(parents=True, exist_ok=True)
    frames = vr.get_batch(idx).asnumpy()
    tf = build_video_transform(256, crop_margin_ratio=1.0)
    clip = tf.transforms[1](tf.transforms[0](list(frames)))     # Resize → CenterCrop (ToTensor · Normalize 앞까지)
    for i, im in enumerate(clip):
        Image.fromarray(np.asarray(im).astype(np.uint8)).save(d / f"{i:06d}.png")
    return vid, len(idx)


def main(seed=0):
    rows = list(csv.DictReader(open(SRC / "metadata.csv")))
    sc = collections.defaultdict(list)
    for r in rows: sc[r["SceneIndex"]].append(r)
    rng = random.Random(seed)
    train_pos, test_rows = [], []
    for s in sorted(sc):
        p = rng.choice(["1", "2"])
        train_pos += [r for r in sc[s] if r["type"] == f"{p}_Possible"]
        test_rows += [r for r in sc[s] if not r["type"].startswith(p + "_")]
    assert len(train_pos) == 253 and len(test_rows) == 506
    jobs = [(r["file_name"], Path(r["file_name"]).stem) for r in train_pos]
    with ProcessPoolExecutor(16) as ex:
        n = dict(ex.map(extract, jobs))
    OUT.mkdir(parents=True, exist_ok=True)
    by_start = collections.defaultdict(list)
    for r in train_pos:
        vid = Path(r["file_name"]).stem
        for s in range(0, n[vid] - M + 1, STRIDE):
            by_start[s].append({"video_id": vid, "file_name": vid, "plausible": "1", "SceneIndex": r["SceneIndex"], "type": r["type"],
                                "condition": r["condition"], "game_name": r["game_name"]})
    for s, rs in by_start.items():
        with open(OUT / f"index_s{s}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, rs[0].keys()); w.writeheader(); w.writerows(rs)
    # 채점과 같은 문맥 C 빈도: 슬라이딩 창마다 SWEEP 11 개 + 창 0 의 growing prefix (C 마다 2 … C−2)
    cnt = collections.Counter()
    for r in train_pos:
        W = len(range(0, n[Path(r["file_name"]).stem] - M + 1, STRIDE))
        for C in SWEEP:
            cnt[C] += W
            for c in range(2, C, 2): cnt[c] += 1
    # 채점기 split MainPair (test 쌍) — Videos 는 NFS 사본
    d = OUT / "root_nfs/MainPair"; d.mkdir(parents=True, exist_ok=True)
    with open(d / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(test_rows)
    if not (d / "Videos").exists(): os.symlink(SRC / "Videos", d / "Videos")
    info = {"train_clips": len(train_pos), "test_pairs": len(test_rows) // 2, "starts": sorted(by_start),
            "windows_per_start": {s: len(v) for s, v in sorted(by_start.items())}, "n_windows": sum(map(len, by_start.values())),
            "context_counts": dict(sorted(cnt.items())), "frames_per_video": [min(n.values()), max(n.values())]}
    json.dump(info, open(OUT / "info.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "windows_per_start"}))


if __name__ == "__main__":
    main()
