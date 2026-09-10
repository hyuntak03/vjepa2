#!/usr/bin/env python3
"""index.csv 여러 개를 **주어진 순서 그대로** 이어 붙인다 (block_id 는 다시 번호를 매긴다).

토큰 캐시를 이어 붙였을 때(`harness/merge_token_cache.py`) 캐시의 `video_ids` 순서는
`A 순서 + B 순서` 다. 인덱스도 **정확히 그 순서**여야 캐시가 재사용된다.
`build_intphysgen_v6_index.py` 를 다시 돌리면 metadata.csv 순서가 나와서 안 맞는다.

`block_id` 는 파일마다 0 부터 다시 시작하므로 겹친다 -> 이어 붙이면서 다시 매긴다
(split 이 block 단위라 겹치면 서로 다른 block 이 한 덩어리로 묶인다).

  python z_research/scripts/data/concat_index.py \
    --out data_csv/intphysgen_v11_full/index.csv \
    data_csv/intphysgen_v11/index.csv data_csv/intphysgen_v11_earlymid/index.csv
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out, fields, seen, remap, nxt = [], None, set(), {}, 0
    for p in a.inputs:
        rows = list(csv.DictReader(p.open()))
        if fields is None:
            fields = list(rows[0])
        elif list(rows[0]) != fields:
            raise SystemExit(f"거부: {p} 의 컬럼이 다르다")
        for r in rows:
            if r["video_id"] in seen:
                raise SystemExit(f"거부: video_id 중복 {r['video_id']}")
            seen.add(r["video_id"])
            key = (str(p), r["block_id"])
            if key not in remap:
                remap[key] = str(nxt); nxt += 1
            out.append(dict(r, block_id=remap[key]))
        print(f"  {p}  +{len(rows)} clip  (block {len({k for k in remap if k[0]==str(p)})})")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    print(f"  wrote {a.out}: {len(out)} clip, {nxt} block")


if __name__ == "__main__":
    main()
