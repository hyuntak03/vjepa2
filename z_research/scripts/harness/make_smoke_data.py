#!/usr/bin/env python3
"""하네스 자가검증용 **합성** 데이터셋 — 새 기계에서 실데이터 없이 채점·학습 전 구간을 돌려 본다.

  python z_research/scripts/harness/make_smoke_data.py            # 기본 8 block (32 clip x 100 장)
  python z_research/scripts/harness/make_smoke_data.py --blocks 4 --force

만드는 것 (경로는 paths.env 에서)
  ${WORLD_ROOT}/smoke/frames/<video_id>/000000..000099.png     160x160 (채점기가 256 으로 리사이즈)
  ${DATA_CSV}/smoke/index.csv                                  IntPhysGen 14 컬럼 + sym_k
  ${DATA_CSV}/smoke/context_integrity.json                     문맥일치 쌍의 픽셀 동일성 검사 결과

레지스트리에 등록된 이름
  configs/protocols/datasets.md  ## smoke            채점 (surprise_c16t32 · attn_probe)
  configs/training/datasets.md   ## smoke_possible   학습 (가능 변이만)

설계 — v11 과 같은 2x2 block
  물체 A (빨간 사각형, 왼→오) / B (파란 원, 오→왼) 두 궤적을 두고
  pos_a = A 문맥 + A 미래, pos_b = B + B, imp_ab = A 문맥 + B 미래, imp_ba = B 문맥 + A 미래.
  문맥 = raw 0..47 (surprise_c16t32 의 context 16 장 = raw 0..45 를 덮는다), 미래 = raw 48..99.
  pair_id 는 문맥이 같은 (가능, 불가능) 쌍: (pos_a, imp_ab) / (pos_b, imp_ba).
  → matched pairing 에서 p 가 비트 단위로 같아야 한다 (CLAUDE.md §1-2) — 파이프라인 정합성 점검용.

⚠️ 합성 데이터의 점수는 **아무 의미가 없다**. 배관이 도는지만 본다. 결과를 어디에도 인용하지 말 것.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_CSV, WORLD_ROOT  # noqa: E402

RAW, SIZE, SPLIT = 100, 160, 48          # 프레임 수, 한 변 픽셀, 문맥/미래 경계 (raw 인덱스)
COLS = ["video_id", "file", "file_name", "block_id", "source_block", "variant", "plausible", "pair_id",
        "condition", "motion", "has_occlusion", "violation_type", "game_name", "role", "sym_k"]


def render(track: str, t: int, bg: np.ndarray, rng_off: int) -> np.ndarray:
    """track 'A' = 빨간 사각형 왼→오, 'B' = 파란 원 오→왼. 배경은 block 마다 고정된 잡음."""
    img = bg.copy()
    r = 12
    u = t / (RAW - 1)
    if track == "A":
        cx, cy = int(20 + u * (SIZE - 40)), 60 + rng_off
        img[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r] = (220, 40, 40)
    else:
        cx, cy = int(SIZE - 20 - u * (SIZE - 40)), 100 - rng_off
        yy, xx = np.ogrid[:SIZE, :SIZE]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = (40, 80, 230)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--frames-root", default=f"{WORLD_ROOT}/smoke/frames")
    ap.add_argument("--index-root", default=f"{DATA_CSV}/smoke")
    ap.add_argument("--force", action="store_true", help="있으면 지우고 다시 만든다")
    a = ap.parse_args()

    for d in (a.frames_root, a.index_root):
        if os.path.exists(d) and os.listdir(d):
            if not a.force:
                sys.exit(f"이미 있다 -> {d}  (--force 로 다시 만든다)")
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    variants = {"pos_a": ("A", "A", "1"), "pos_b": ("B", "B", "1"),
                "imp_ab": ("A", "B", "0"), "imp_ba": ("B", "A", "0")}
    rows, digests = [], {}
    for b in range(a.blocks):
        rng = np.random.default_rng(b)
        bg = (rng.integers(90, 140, size=(SIZE, SIZE, 1)) * np.ones((1, 1, 3))).astype(np.uint8)
        off = int(rng.integers(-15, 16))
        cond = "smoke_static" if b % 2 == 0 else "smoke_moving"
        for v, (ctx, fut, plaus) in variants.items():
            vid = f"smoke_b{b:03d}_{v}"
            out = os.path.join(a.frames_root, vid)
            os.makedirs(out, exist_ok=True)
            h = hashlib.sha1()
            for t in range(RAW):
                im = render(ctx if t < SPLIT else fut, t, bg, off)
                if t < SPLIT:
                    h.update(im.tobytes())
                Image.fromarray(im).save(os.path.join(out, f"{t:06d}.png"), compress_level=1)
            digests[vid] = h.hexdigest()
            rows.append({"video_id": vid, "file": "", "file_name": vid, "block_id": str(b),
                         "source_block": str(b), "variant": v, "plausible": plaus,
                         "pair_id": f"{b}_{ctx}", "condition": cond, "motion": "linear",
                         "has_occlusion": "0", "violation_type": "swap", "game_name": "smoke",
                         "role": "smoke", "sym_k": "0"})

    with open(os.path.join(a.index_root, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    # 문맥일치 쌍의 문맥 프레임이 픽셀 단위로 같은가 (CLAUDE.md §7-4 의 byte-identical 감사와 같은 뜻)
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r["video_id"])
    mism = [p for p, vs in by_pair.items() if len({digests[v] for v in vs}) != 1]
    integ = {"n_pairs": len(by_pair), "mismatch": mism, "context_raw_frames": [0, SPLIT - 1]}
    json.dump(integ, open(os.path.join(a.index_root, "context_integrity.json"), "w"), indent=1)

    print(f"frames : {a.frames_root}  ({len(rows)} clip x {RAW} 장, {SIZE}px)")
    print(f"index  : {a.index_root}/index.csv  ({a.blocks} block, {len(by_pair)} matched pair)")
    print(f"context: 문맥일치 쌍 mismatch {len(mism)}")
    if mism:
        sys.exit(1)


if __name__ == "__main__":
    main()
