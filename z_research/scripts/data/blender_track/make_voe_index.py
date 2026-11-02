#!/usr/bin/env python3
"""BlenderTrack v3 · v3t 시험 VoE 목록 (2026-11-01) — <split>_arms.csv 에서 렌더 타이밍 · 이음 둘 다 통과한 장면 (voe_ok = 1) 의 네 팔만.
출력 data_csv/blender_track_<이름>/voe_test/index.csv (채점기 eval/testbed · vobench_v1 형식: file = mp4 절대경로, frames_root = null)
  $P make_voe_index.py --name v3t
"""
import argparse, csv, os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
COLS = ["video_id", "file", "file_name", "block_id", "variant", "plausible", "pair_id", "condition", "objset", "place", "camera", "occ_type",
        "occ_motion", "target_mode", "occ_sub", "target_kind", "ramp_mode", "f_hide", "f_count", "f_indoor", "event_s", "event_e"]

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--split", default="test")
a = ap.parse_args()
d = f"{REPO}/data_csv/blender_track_{a.name}"
rows = [r for r in csv.DictReader(open(f"{d}/{a.split}_arms.csv")) if r["voe_ok"] == "1"]
os.makedirs(f"{d}/voe_{a.split}", exist_ok=True)
with open(f"{d}/voe_{a.split}/index.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader()
    for r in rows:
        w.writerow(dict(r, file_name=r["video_id"]))
print(f"{a.name} {a.split}: 장면 {len(rows) // 4} · 영상 {len(rows)} → {d}/voe_{a.split}/index.csv")
