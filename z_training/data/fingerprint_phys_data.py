#!/usr/bin/env python3
"""phys2d · phys3d 생성물 지문 (2026-10-19) — 다른 기계에서 다시 만든 데이터가 같은지 확인한다.

  python z_training/data/fingerprint_phys_data.py --root <out>/val --write <tsv>          # 지문 만들기
  python z_training/data/fingerprint_phys_data.py --root <out>/val --check <tsv>          # 비교

clip 마다 라벨 지문 = sha256( _states.npz 의 배열 (키 정렬, 원시 바이트) + json 의 사건 · 행 ).
2D 는 _seg.npz (보이는 픽셀 인스턴스 세그) 도 따로 잰다 — 그림 좌표까지 같은지. mp4 바이트는 ffmpeg 버전마다 달라 재지 않는다.
경로 · 시간 · 시도 소요 같은 기계 의존 값은 넣지 않는다.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np


def arr_hash(h, z):
    for k in sorted(z.files):
        a = np.ascontiguousarray(z[k])
        h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def clip_fp(base):
    h = hashlib.sha256()
    with np.load(base + "_states.npz") as z:
        arr_hash(h, z)
    m = json.load(open(base + ".json"))
    keep = {k: m.get(k) for k in ("events", "event_rows", "rows", "n_frames", "attempt", "seed", "objects")}
    for o in keep.get("objects") or []:
        o.pop("index", None)
    h.update(json.dumps(keep, sort_keys=True, default=float).encode())
    seg = ""
    if os.path.exists(base + "_seg.npz"):
        hs = hashlib.sha256()
        with np.load(base + "_seg.npz") as z:
            arr_hash(hs, z)
        seg = hs.hexdigest()[:16]
    return h.hexdigest()[:16], seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write")
    g.add_argument("--check")
    a = ap.parse_args()
    fps = {}
    for j in sorted(glob.glob(os.path.join(a.root, "clips", "*", "*.json"))):
        base = j[:-5]
        if os.path.exists(base + "_states.npz"):
            fps[os.path.basename(base)] = clip_fp(base)
    total = hashlib.sha256("".join(f"{k}{v[0]}{v[1]}" for k, v in sorted(fps.items())).encode()).hexdigest()[:16]
    if a.write:
        with open(a.write, "w") as fh:
            fh.write(f"# clips {len(fps)} total {total}\nclip\tlabels\tseg\n")
            fh.writelines(f"{k}\t{v[0]}\t{v[1]}\n" for k, v in sorted(fps.items()))
        print(f"{len(fps):,} clip, total {total} → {a.write}")
        return 0
    ref = {}
    for line in open(a.check):
        if line.startswith(("#", "clip\t")):
            continue
        k, l, s = line.rstrip("\n").split("\t")
        ref[k] = (l, s)
    miss = sorted(set(ref) - set(fps))
    extra = sorted(set(fps) - set(ref))
    lab = sorted(k for k in set(ref) & set(fps) if ref[k][0] != fps[k][0])
    seg = sorted(k for k in set(ref) & set(fps) if ref[k][1] != fps[k][1])
    print(f"기준 {len(ref):,} · 여기 {len(fps):,} · 빠짐 {len(miss)} · 더 있음 {len(extra)} · 라벨 다름 {len(lab)} · 세그 다름 {len(seg)}")
    for name, xs in (("빠짐", miss), ("더 있음", extra), ("라벨 다름", lab), ("세그 다름", seg)):
        if xs:
            print(f"  {name}: {', '.join(xs[:10])}{' …' if len(xs) > 10 else ''}")
    return 0 if not (miss or extra or lab or seg) else 1


if __name__ == "__main__":
    sys.exit(main())
