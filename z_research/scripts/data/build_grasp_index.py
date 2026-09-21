"""GRASP level2 를 레포 인덱스 스키마로 만든다 (Garrido 쌍 비교용).

    python z_research/scripts/data/build_grasp_index.py --write

GRASP 은 원래 **단일 영상** 평가용(Video-LLM 질의응답)이라 공식 쌍 정의가 없다.
Garrido et al. A.4 가 쌍으로 쓰면서 "untrained network 가 spurious feature 로 높은 정확도를
낸다" 고 스스로 단서를 달았다. 여기서는 그 해석을 따른다 —
**같은 시나리오·같은 파일 인덱스**끼리 묶는다: `P_Gravity/037.mp4` <-> `IP_Gravity/037.mp4`.

`GravitySupport` 처럼 두 property 가 붙은 시나리오는 이름을 그대로 `block_type` 에 두고,
집계할 때 양쪽 property 에 각각 넣는다 (논문 A.4). 그 분해는 채점 후 사후에 한다.

영상은 전부 501 프레임으로 같다 (2026-09-20 실측, 표본 40) -> `raw_frames: 501`.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

LOCAL = "/data2/local_datasets/world/Benchmark/GRASP/videos/level2"
NFS = "/data/dataset/world/Benchmarks/GRASP/videos/level2"
OUT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/grasp_level2"

COLS = ["video_id", "file", "block_id", "source_block", "variant", "plausible",
        "pair_id", "condition", "block_type", "scenario", "role"]


def build(root):
    rows = []
    dirs = sorted(d for d in os.listdir(root) if d.startswith(("P_", "IP_")))
    scen = sorted({d.split("_", 1)[1] for d in dirs})
    assert len(scen) * 2 == len(dirs), f"P_/IP_ 짝이 안 맞는다: {len(dirs)} 폴더 / {len(scen)} 시나리오"
    for s in scen:
        for tag, plaus in (("P", "1"), ("IP", "0")):
            d = f"{tag}_{s}"
            for f in sorted(glob.glob(f"{root}/{d}/*.mp4")):
                idx = os.path.basename(f).split(".")[0]
                rows.append(dict(
                    video_id=f"{d}_{idx}", file=os.path.abspath(f),   # 절대경로 (Jongseo physv3 와 같은 관례)
                    block_id=f"{s}_{idx}", source_block=f"{s}_{idx}",
                    variant=("pos_a" if tag == "P" else "imp_ab"), plausible=plaus,
                    pair_id=f"{s}_{idx}", condition=s, block_type=s, scenario=s,
                    role=("pos_obj" if tag == "P" else "imp_obj")))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=LOCAL if os.path.isdir(LOCAL) else NFS)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rows = build(a.root)

    # --- 검증 -------------------------------------------------------------
    n_block = len({r["block_id"] for r in rows})
    pos = sum(r["plausible"] == "1" for r in rows)
    print(f"root      : {a.root}")
    print(f"영상      : {len(rows)}  (가능 {pos} / 불가능 {len(rows)-pos})")
    print(f"block     : {n_block}  -> block 당 {len(rows)/n_block:.1f} 영상")
    print(f"시나리오  : {len(set(r['scenario'] for r in rows))}")
    assert len(rows) == 4096, len(rows)
    assert pos * 2 == len(rows), "가능/불가능이 반반이 아니다"
    assert n_block * 2 == len(rows), "block 하나에 2 영상이 아니다"
    for r in rows[:2]:
        assert os.path.isfile(r["file"]), r["file"]

    if a.write:
        os.makedirs(OUT, exist_ok=True)
        with open(f"{OUT}/index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        print(f"-> {OUT}/index.csv")
    else:
        print("(--write 를 붙여야 저장한다)")
