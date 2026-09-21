#!/usr/bin/env python
"""InfLevel-lab 인덱스를 **공식 로더와 같은 방식으로** 만든다.

    python z_research/scripts/data/build_inflevel_index.py            # 스캔만 (분포 출력)
    python z_research/scripts/data/build_inflevel_index.py --write    # index.csv 쓰기

⚠️ 2026-09-21 전면 재작성. 이전 판은 **세 군데가 공식과 달랐다** (수치 전부 폐기):
  1. 쌍 목록을 자체 규칙으로 만들었다 -> 공식은 `auxiliary_data_loading_files/inflevel/{prop}.csv`
     한 줄이 곧 한 쌍이고 `vid1` 은 항상 possible, `vid2` 는 항상 impossible 이다.
  2. 채점을 2x2 cross 로 했다 -> 공식 `eval.py:406` 은 InfLevel/GRASP 에서 `matches=[[0,1]]`,
     즉 **한 줄에 한 비교**다. cross 를 쓰면 쌍 수가 정확히 2 배가 된다.
  3. 영상을 처음부터 읽었다 -> 공식 `inflevel_dataset.py:133-142` 는 **priming 구간을 잘라내고**
     시작한다. 컵 내부를 보여주는 구간이 거기 들어 있다.

공식 프레임 선택 (`inflevel_dataset.py`, priming=False 일 때):

    diff  = (len1 - end_priming_2_1) - (len2 - end_priming_2_2)
    start1 = end_priming_2_1 + (diff if diff > 0 else 0) + 1
    start2 = end_priming_2_2 + (-diff if diff < 0 else 0) + 1
    frames_i = arange(len_i)[start_i :: frame_step]

diff 는 **두 영상의 잔여 길이를 맞추기 위한 것**이라, 적용 뒤 남는 프레임 수는 둘이 같다.

우리 하네스는 영상마다 길이가 다른 것을 못 받으므로 (`data.n_frames` 가 전역이다)
남는 프레임 수의 **최솟값으로 잘라** 모든 쌍이 같은 수의 window 를 갖게 한다.
공식 스크립트는 대신 0 으로 패딩하는데, 그쪽은 스스로 주석에
"can lead to slightly innacurate metrics" 라고 적어 둔 경로다. 자른 값은 `raw_frames` 로 낸다.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd

AUX = ("/data/hyuntak/project/2026/2027_cvpr/jepa-intuitive-physics/evaluation_code/"
       "auxiliary_data_loading_files/inflevel")
ROOT = "/data/dataset/world/Benchmarks/InfLevel/InfLevel"
OUT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv"
PROPS = ["continuity", "gravity", "solidity"]


def official_starts(len1, e1, len2, e2):
    """공식 inflevel_dataset.py 의 시작 프레임 두 개."""
    diff = (len1 - e1) - (len2 - e2)
    s1 = e1 + (diff if diff > 0 else 0) + 1
    s2 = e2 + (-diff if diff < 0 else 0) + 1
    return s1, s2


def scan(prop: str) -> pd.DataFrame:
    import decord
    df = pd.read_csv(f"{AUX}/{prop}.csv")
    rows = []
    for i, r in df.iterrows():
        p1, p2 = os.path.join(ROOT, r["vid1_path"]), os.path.join(ROOT, r["vid2_path"])
        if not (os.path.exists(p1) and os.path.exists(p2)):
            print(f"  !! 파일 없음, 건너뜀: {r['vid1_path']}"); continue
        l1 = len(decord.VideoReader(p1, num_threads=1))
        l2 = len(decord.VideoReader(p2, num_threads=1))
        e1, e2 = int(r["vid1_end_priming_2"]), int(r["vid2_end_priming_2"])
        s1, s2 = official_starts(l1, e1, l2, e2)
        avail = min(l1 - s1, l2 - s2)
        assert r["vid1_label"] == "possible" and r["vid2_label"] == "impossible", "CSV 라벨 가정 깨짐"
        rows.append(dict(pair=i, p1=r["vid1_path"], p2=r["vid2_path"],
                         l1=l1, l2=l2, s1=s1, s2=s2, avail=avail))
        if (i + 1) % 200 == 0:
            print(f"  {prop} {i+1}/{len(df)}", flush=True)
    return pd.DataFrame(rows)


def build(prop: str, sc: pd.DataFrame, n_keep: int) -> pd.DataFrame:
    out = []
    for _, r in sc.iterrows():
        for k, (path, start, plaus) in enumerate([(r.p1, r.s1, 1), (r.p2, r.s2, 0)]):
            vid = os.path.splitext(os.path.basename(path))[0]
            out.append(dict(
                video_id=f"{vid}__p{r.pair}",       # 같은 영상이 여러 쌍에 나올 수 있다
                file=os.path.join(ROOT, path),
                block_id=f"{prop}|{r.pair}",         # 한 줄 = 한 block = 한 쌍
                variant=("pos" if plaus else "imp"),
                plausible=plaus,
                pair_id=f"{prop}|{r.pair}",
                condition=prop,
                block_type=prop,
                frame_start=int(start),              # ★ 공식 priming 절단 위치
            ))
    df = pd.DataFrame(out)
    df["n_frames"] = n_keep
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    for prop in PROPS:
        print(f"=== {prop}")
        sc = scan(prop)
        av = sc["avail"].to_numpy()
        print(f"  쌍 {len(sc)} | 잔여 프레임 min {av.min()} p1 {np.percentile(av,1):.0f} "
              f"중앙 {np.median(av):.0f} max {av.max()}")
        n_keep = int(av.min())
        print(f"  -> n_frames = {n_keep} (최솟값으로 자름)")
        if a.write:
            d = f"{OUT}/inflevel_{prop}"
            os.makedirs(d, exist_ok=True)
            df = build(prop, sc, n_keep)
            df.to_csv(f"{d}/index.csv", index=False)
            print(f"  쓰기: {d}/index.csv  ({len(df)} 행 = {len(sc)} 쌍 x 2)")


if __name__ == "__main__":
    main()
