#!/usr/bin/env python
"""phys2d_single_v1 (물체 1 개) → main 의 `type: phys2d_sup` 로더가 읽는 두 목록 (2026-10-19).

vast 는 var_window 의 `boundary` 경로로 `train_mix.tsv` (boundary impact·leave 행 + uniform 행) 를 썼다.
main 의 PFT 경로 (`phys2d_sup_vast_ar_state_pft`) 는 Phys2DSupDataset 이라 csv 두 형식만 읽는다 —
그래서 같은 두 종류를 각각 csv 로 낸다 (학습 config 에서 weight 로 섞는다).

  clip 목록    "<mp4> 0"                      → 클립 안 무작위 시작 (= vast 의 uniform 행에 해당)
  창 인덱스    path,start,stride,n_frames,…   → boundary.tsv 의 impact · leave 행, **예측 첫 장 = 사건**
                                                (vast _read_boundary 와 같은 정렬: start = boundary_frame − C)

⚠️ manifest 안 경로는 vast 의 `/workspace/...` 다 — `--src-prefix`/`--dst-prefix` 로 바꿔 쓴다.

  python z_training/data/build_phys2d_single_index.py \
      --root /local_datasets/world/training_data/phys2d_single_v1 \
      --out  data_csv/phys2d_single_v1
"""
from __future__ import annotations

import argparse, csv, os

EV = ("impact", "leave")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--src-prefix", default="/workspace/dataset/training_data/phys2d_single_v1")
    ap.add_argument("--dst-prefix", default=None, help="기본값 = --root")
    a = ap.parse_args()
    dst = a.dst_prefix or os.path.abspath(a.root)
    os.makedirs(a.out, exist_ok=True)
    T, C, s = a.n_frames, a.context, a.stride

    def fix(p):
        return dst + p[len(a.src_prefix):] if p.startswith(a.src_prefix) else p

    for split in ("train", "val"):
        # 1) clip 목록 (무작위 시작)
        src = os.path.join(a.root, split, "index.csv")
        rows = [fix(l.split()[0]) for l in open(src) if l.strip()]
        miss = [p for p in rows[:50] if not os.path.exists(p)]
        if miss:
            raise SystemExit(f"경로가 없다 (prefix 확인): {miss[0]}")
        fo = os.path.join(a.out, f"{split}_clips.csv")
        with open(fo, "w") as f:
            f.writelines(f"{p} 0\n" for p in rows)
        print(f"{fo}: clip {len(rows):,}")

        # 2) boundary 창 (예측 첫 장 = 충돌 · 이탈)
        fb = os.path.join(a.out, f"{split}_bnd_T{T}C{C}s{s}.csv")
        n, drop = 0, 0
        with open(os.path.join(a.root, split, "boundary.tsv")) as f, open(fb, "w", newline="") as g:
            w = csv.writer(g)
            w.writerow(["path", "start", "stride", "n_frames", "scenario", "category", "event", "dv"])
            for r in csv.DictReader(f, delimiter="\t"):
                if r["event"] not in EV:
                    continue
                b, nv = int(r["boundary_frame"]), int(r["video_frames"])
                start = b - C * s
                if start < 0 or start + (T - 1) * s >= nv:
                    drop += 1
                    continue
                w.writerow([fix(r["path"]), start, s, T, r["scenario"], r["category"], r["event"], r["dv"]])
                n += 1
        print(f"{fb}: 창 {n:,} (경계 밖으로 버린 행 {drop:,})")


if __name__ == "__main__":
    main()
