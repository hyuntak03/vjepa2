"""배관 점검 전용 — SAM2 없이 S5 · S7 을 돌려 보려고 실제 클립 위에 움직이는 타원 물체를 그려 넣는다.

**학습 데이터가 아니다.** 일정 탐색 · 합성 · 저장 · 시각화가 맞게 도는지만 본다.
출력 (클립 폴더마다): clean_fake.npy · object_fake.npz · object_fake.json — S1–S3 출력과 같은 꼴.

    python datagen/s_fake_objects.py --clips <dir>/ssv2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import pack, save_json  # noqa: E402

SPEED = {"ssv2": (5.0, 9.0), "k400": (2.0, 4.0)}   # px / 프레임


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for d in sorted(p for p in args.clips.iterdir() if (p / "clean.npy").exists()):
        meta = json.load(open(d / "clip.json"))
        clean = np.load(d / "clean.npy")
        M, H, W, _ = clean.shape
        ax, ay = rng.uniform(20, 34), rng.uniform(16, 30)       # 면적 ≈ 1.5–5 %
        p = np.array([rng.uniform(ax, W - ax), rng.uniform(ay, H - ay)])
        th = rng.uniform(0, 2 * np.pi)
        v = rng.uniform(*SPEED[meta["source"]]) * np.array([np.cos(th), np.sin(th)])
        color = tuple(int(c) for c in rng.integers(30, 255, 3))
        out, masks = clean.copy(), np.zeros((M, H, W), bool)
        for t in range(M):
            m = np.zeros((H, W), np.uint8)
            cv2.ellipse(m, (int(p[0]), int(p[1])), (int(ax), int(ay)), 0, 0, 360, 1, -1)
            masks[t] = m.astype(bool)
            out[t][masks[t]] = color
            p = p + v
            for k, lim, r in ((0, W, ax), (1, H, ay)):            # 벽에서 튕긴다
                if p[k] < r or p[k] > lim - r:
                    v[k] = -v[k]
                    p[k] = np.clip(p[k], r, lim - r)
        np.save(d / "clean_fake.npy", out)
        np.savez_compressed(d / "object_fake.npz", **{f"amodal_{k}": x for k, x in pack(masks).items()})
        save_json(d / "object_fake.json", dict(status="ok", fake=True, color=color, radii=[ax, ay],
                                               speed=float(np.linalg.norm(v))))
    print("done", args.clips)


if __name__ == "__main__":
    main()
