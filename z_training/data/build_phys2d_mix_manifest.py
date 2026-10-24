#!/usr/bin/env python
"""phys2d_* 생성 산출물 (`uniform.tsv` + `boundary.tsv`) → var_window `boundary.manifest` 용 `train_mix.tsv` (2026-10-20).

vast 는 `data.var_window.boundary.manifest` 로 `train_mix.tsv` 를 읽는다 (`app/vjepa_frozen/var_window.py:_init_boundary`).
그 파일은 생성기가 낸 두 tsv 를 합친 것이다 — 행 수로 확인했다 (vast config 주석의 511,782 : 40,117 과 일치):

  uniform.tsv   무작위 시작 행 (가림은 어디서든)            511,782
  boundary.tsv  사건 행. 그중 **impact · leave 만** 쓴다      40,117 (= 33,713 + 6,404)
                occlude · reappear 124,587 은 **뺀다** — vast 가 뺐다

`_init_boundary` 가 읽는 열은 `path · n_frames · boundary_frame` (+ 선택 `ctx_start`) 뿐이고 나머지는 무시된다.
경로는 vast 의 `/workspace/...` 라 `--src-prefix`/`--dst-prefix` 로 이 노드의 실물 경로로 바꾼다.

  python z_training/data/build_phys2d_mix_manifest.py \
      --root /local_datasets/world/training_data/phys2d_single_v1/train \
      --out  data_csv/phys2d_single_v1/train_mix.tsv
"""
from __future__ import annotations

import argparse
import csv
import os

EV = ("impact", "leave")       # vast 와 같다. occlude · reappear 는 뺀다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="생성 산출물 폴더 (uniform.tsv · boundary.tsv 가 있는 곳)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src-prefix", default="/workspace/dataset/training_data")
    ap.add_argument("--dst-prefix", default="/local_datasets/world/training_data")
    ap.add_argument("--events", default=",".join(EV))
    a = ap.parse_args()
    keep = tuple(x for x in a.events.split(",") if x)

    rows, n = [], {}
    for src, filt in (("uniform.tsv", None), ("boundary.tsv", keep)):
        p = os.path.join(a.root, src)
        with open(p, encoding="utf-8") as f:
            got = 0
            for r in csv.DictReader(f, delimiter="\t"):
                if filt is not None and r.get("event") not in filt:
                    continue
                r["path"] = r["path"].replace(a.src_prefix, a.dst_prefix, 1)
                rows.append(r)
                got += 1
        n[src] = got

    miss = [r["path"] for r in rows[::997] if not os.path.exists(r["path"])]   # 1/997 표본만 실물 확인 (50 만 행 stat 는 비싸다)
    if miss:
        raise SystemExit(f"경로 실물 없음 (표본 {len(miss)} 개): {miss[:3]}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cols = list(rows[0].keys())
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"{a.out}: {len(rows):,} 행  (uniform {n['uniform.tsv']:,} + boundary[{','.join(keep)}] {n['boundary.tsv']:,})")
    print(f"  열: {cols}")


if __name__ == "__main__":
    main()
