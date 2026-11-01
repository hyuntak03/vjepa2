"""S4 — 가림막 은행: COCO val2017 인스턴스 mask 로 물체 조각 (RGBA) 을 잘라 낸다.

가림막은 대상 물체를 **완전히** 덮어야 하므로 볼록한 모양만 쓴다 (solidity = 면적 / 볼록껍질 면적).
이미지 가장자리에 닿은 조각 (잘린 물체) 과 너무 작은 조각 (키우면 흐려짐) 은 뺀다.

출력: <out>/patches/<ann_id>.png (RGBA, bbox 크기) + <out>/index.json

    python datagen/s4_occluder_bank.py --out <dir> --max 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COCO_ROOT, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--min_side", type=int, default=96, help="bbox 짧은 변 최소 (px)")
    ap.add_argument("--min_solidity", type=float, default=0.85)
    ap.add_argument("--max_aspect", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coco = COCO(str(COCO_ROOT / "annotations" / "instances_val2017.json"))
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    ann_ids = np.random.default_rng(args.seed).permutation(sorted(coco.getAnnIds(iscrowd=False)))
    (args.out / "patches").mkdir(parents=True, exist_ok=True)

    kept, stats = [], dict(seen=0, small=0, border=0, aspect=0, solidity=0, multi=0)
    for aid in ann_ids:
        if len(kept) >= args.max:
            break
        ann = coco.loadAnns(int(aid))[0]
        stats["seen"] += 1
        img = coco.loadImgs(ann["image_id"])[0]
        H, W = img["height"], img["width"]
        x, y, w, h = ann["bbox"]
        if min(w, h) < args.min_side:
            stats["small"] += 1
            continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            stats["border"] += 1
            continue
        if max(w, h) / min(w, h) > args.max_aspect:
            stats["aspect"] += 1
            continue
        m = coco.annToMask(ann).astype(np.uint8)
        n_cc, cc = cv2.connectedComponents(m)
        if n_cc > 2:  # 조각이 여러 개면 가장 큰 것만 남긴다
            sizes = np.bincount(cc.ravel())[1:]
            if sizes.max() < 0.95 * sizes.sum():
                stats["multi"] += 1
                continue
            m = (cc == 1 + sizes.argmax()).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        solidity = m.sum() / max(cv2.contourArea(hull), 1.0)
        if solidity < args.min_solidity:
            stats["solidity"] += 1
            continue
        x0, y0, x1, y1 = int(x), int(y), int(np.ceil(x + w)), int(np.ceil(y + h))
        rgb = cv2.imread(str(COCO_ROOT / "val2017" / img["file_name"]))[y0:y1, x0:x1]
        rgba = np.dstack([rgb, m[y0:y1, x0:x1] * 255])
        cv2.imwrite(str(args.out / "patches" / f"{aid}.png"), rgba)
        kept.append(dict(ann_id=int(aid), category=cats[ann["category_id"]], w=x1 - x0, h=y1 - y0,
                         solidity=round(float(solidity), 3), image=img["file_name"]))
    save_json(args.out / "index.json", dict(stats=stats, patches=kept))
    print(f"kept {len(kept)}  stats {stats}")


if __name__ == "__main__":
    main()
