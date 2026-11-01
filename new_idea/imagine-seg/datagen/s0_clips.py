"""S0 — 클립 고르기: 영상 목록에서 표본 → 6 fps 로 다시 뽑기 → 가운데 정사각 → 256×256 → 창 하나.

출력: <out>/<clip_id>/clean.npy (M, 256, 256, 3) uint8 + clip.json (원본 경로 · 원본 프레임 번호 · 라벨)

    python datagen/s0_clips.py --source ssv2 --n 40 --out <dir> --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import decord
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_CSV, FPS, RES, TRAIN_DATA_ROOT, save_json, window  # noqa: E402

SSV2_DIR = TRAIN_DATA_ROOT / "SSv2" / "20bn-something-something-v2"
SSV2_LABELS = TRAIN_DATA_ROOT / "SSv2" / "labels"
K710_DIR = TRAIN_DATA_ROOT / "K710"


def list_ssv2(min_sec: float) -> list[dict]:
    """학습 split 만. lengths.tsv 의 프레임 수 (12 fps) 로 창이 안 들어가는 영상을 미리 뺀다."""
    labels = {d["id"]: d for d in json.load(open(SSV2_LABELS / "train.json"))}
    out = []
    for line in open(DATA_CSV / "ssv2" / "lengths.tsv"):
        path, n = line.rstrip("\n").split("\t")
        vid = Path(path).stem
        if vid in labels and int(n) >= 12 * min_sec + 2:
            d = labels[vid]
            out.append(dict(source="ssv2", vid=vid, path=str(SSV2_DIR / f"{vid}.webm"),
                            label=d["label"], template=d["template"]))
    return out


def list_k400() -> list[dict]:
    cats = {}
    for line in open(K710_DIR / "annotations" / "categories.txt"):
        name, idx = line.rstrip("\n").rsplit("\t", 1)
        cats[int(idx)] = name
    out = []
    for rel, lab in csv.reader(open(K710_DIR / "annotations" / "train.csv")):
        if rel.startswith("k400/"):
            out.append(dict(source="k400", vid=Path(rel).stem, path=str(K710_DIR / "videos" / rel),
                            label=cats[int(lab)], template=None))
    return out


def center_square_resize(frames: np.ndarray, res: int) -> tuple[np.ndarray, list[int]]:
    T, H, W, _ = frames.shape
    s = min(H, W)
    y0, x0 = (H - s) // 2, (W - s) // 2
    crop = frames[:, y0:y0 + s, x0:x0 + s]
    interp = cv2.INTER_AREA if s > res else cv2.INTER_LINEAR
    out = np.stack([cv2.resize(f, (res, res), interpolation=interp) for f in crop])
    return out, [x0, y0, s, s]


def decode_window(item: dict, M: int, rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    """시각 기준으로 6 fps 프레임 M 장을 뽑는다. 시작 시각은 들어가는 범위에서 균등."""
    vr = decord.VideoReader(item["path"], num_threads=2)
    n, native = len(vr), float(vr.get_avg_fps())
    need = (M - 1) / FPS
    dur = (n - 1) / native
    if dur < need:
        return None
    t0 = rng.uniform(0.0, dur - need)
    idx = np.clip(np.round((t0 + np.arange(M) / FPS) * native).astype(int), 0, n - 1)
    frames = vr.get_batch(idx.tolist()).asnumpy()
    clean, crop = center_square_resize(frames, RES)
    info = dict(native_fps=native, native_frames=n, native_hw=list(frames.shape[1:3]),
                t0_sec=t0, frame_idx=idx.tolist(), crop_xywh=crop)
    return clean, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["ssv2", "k400"], required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    M, T_ctx = window(args.source)
    rng = np.random.default_rng(args.seed)
    items = list_ssv2((M - 1) / FPS) if args.source == "ssv2" else list_k400()
    order = rng.permutation(len(items))
    args.out.mkdir(parents=True, exist_ok=True)

    done = 0
    for i in order:
        if done >= args.n:
            break
        item = items[i]
        try:
            res = decode_window(item, M, rng)
        except Exception as e:  # 깨진 영상은 건너뛴다 (기록만)
            print(f"[skip] {item['path']}: {type(e).__name__} {e}")
            continue
        if res is None:
            continue
        clean, info = res
        cid = f"{args.source}_{item['vid']}"
        d = args.out / cid
        d.mkdir(exist_ok=True)
        np.save(d / "clean.npy", clean)
        save_json(d / "clip.json", dict(clip_id=cid, **item, M=M, T_ctx=T_ctx, fps=FPS, res=RES, **info))
        done += 1
        print(f"[{done}/{args.n}] {cid}  {item['label'][:60]}")
    print(f"{done} clips -> {args.out}  (list size {len(items)})")


if __name__ == "__main__":
    main()
