#!/usr/bin/env python3
"""Rubric 채점 index — 원본 mp4 를 그대로 두고 **영상마다 읽을 프레임 번호 (라벨)** 만 적는다 (프레임을 뽑지 않는다, 2026-10-31).

입력: data_csv/rubric_quads_<창>/index.csv (쌍 · 변이 열) + audit.json (영상마다 실효 30 fps 프레임 번호 frames_eff · 60 fps 중복 여부)
출력: data_csv/rubric_quads_<창>/index_mp4.csv = index.csv + file (<scene>/<video>/<variant>.mp4, 원본 폴더 기준 상대 경로) + frames (원본 프레임 번호 32 개, 공백 구분)
채점기 (evals/world_model_analysis/data.py) 는 data.frames_column=frames 로 mp4 에서 그 프레임만 decord 로 읽는다 (IntPhys 2 late 와 같은 경로).
원본 프레임 번호 = frames_eff × (2 if 60 fps 중복 영상 else 1) — build_rubric_quads_frames.py 가 PNG 를 뽑을 때와 같은 번호다.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P predictor_lab/make_rubric_index.py [--window uniform]
"""
import argparse
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="uniform", choices=["uniform", "symmetric"])
    a = ap.parse_args()
    d = f"{REPO}/data_csv/rubric_quads_{a.window}"
    audit = json.load(open(f"{d}/audit.json"))
    with open(f"{d}/index.csv") as f:
        rd = csv.DictReader(f)
        fields, rows = list(rd.fieldnames), list(rd)
    for r in rows:
        au = audit[r["source_block"]]
        step = 2 if au["duplicated"] else 1
        fr = [int(x) * step for x in au["frames_eff"]]
        assert len(fr) == 32 and all(b > c for b, c in zip(fr[1:], fr)), r["video_id"]
        r["file"] = f"{r['file_name']}.mp4"
        r["frames"] = " ".join(map(str, fr))
    out = f"{d}/index_mp4.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + (["frames"] if "frames" not in fields else []))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} 행 -> {out}")


if __name__ == "__main__":
    main()
