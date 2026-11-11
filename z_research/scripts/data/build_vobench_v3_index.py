#!/usr/bin/env python3
"""VOBench_v3 (UE 5.4, 840 블록 × 4 팔 = 3,360 mp4, 전부 late permanence — 11-02 KST 20:15 판. 처음 받은 430 블록 판은 index_430blk.csv) → 채점 index (2026-11-02, 사용자: "VOBench_v3 로 다 채점").

블록마다 짝 둘: A = (pos_a, imp_ab) 사라짐 (문맥에 물체) · B = (pos_b, imp_ba) 나타남 (문맥에 빈 장면). 이음 = 프레임 48, 두 팔의 문맥 (0–47) 이 같은 렌더.
채점 창 = 원본 0, 3, …, 93 (32 장, 문맥 16 + 미래 16) — testbed 표준 (mp4 직독, data.frames_stride 3).
가림막 (blocks.csv occluder): panel (서 있는 판) · mover (지나가는 van / 상자, mover_dir same · opposite). 시트 Object_permanence 의
  "VOBench late" 열 = panel, "moving occluder" 열 = mover 로 나눈다 (z_research/scripts/analysis/op_cells_vob3.py).
출력: data_csv/vobench_v3/index.csv (file = 원본 폴더 기준 mp4 상대 경로, block_id = 0..429 정수, block_name = 원래 이름)
  /venv/vjepa/bin/python z_research/scripts/data/build_vobench_v3_index.py
2026-11-09 사용자 "VOBench_v3_visible_early_mid 이거 release 평가해봐": --src · --out 추가 (기본값 = 예전과 같다).
  가림막이 없는 블록 (visible, blocks.csv occluder 빈칸) 은 occluder_kind none. late 가 아닌 세트는 condition = <occlusion>_<occluder_kind>.
  /venv/vjepa/bin/python z_research/scripts/data/build_vobench_v3_index.py --src /workspace/dataset/benchmarks/VOBench_v3_visible_early_mid --out data_csv/vobench_v3_vem
"""
import argparse, csv, os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = "/workspace/dataset/benchmarks/VOBench_v3"
OUT = f"{REPO}/data_csv/vobench_v3"
PAIR = {"pos_a": "A", "imp_ab": "A", "pos_b": "B", "imp_ba": "B"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    _build(a.src, a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out))


def _build(SRC, OUT):
    blocks = {r["block_id"]: r for r in csv.DictReader(open(f"{SRC}/blocks.csv"))}
    meta = list(csv.DictReader(open(f"{SRC}/metadata.csv")))
    bid = {b: i for i, b in enumerate(sorted(blocks))}
    rows = []
    for m in meta:
        b = blocks[m["block_id"]]
        assert os.path.isfile(f"{SRC}/{m['file_name']}"), m["file_name"]
        occ = "panel" if b["occluder"] == "panel" else ("none" if not b["occluder"] else "mover")
        cond = occ if b["occlusion"].startswith("late") else f"{b['occlusion']}_{occ}"
        rows.append({"video_id": m["name"], "file": m["file_name"], "file_name": m["file_name"][:-4], "block_id": bid[m["block_id"]],
                     "block_name": m["block_id"], "variant": m["arm"], "plausible": m["is_possible"], "pair_id": PAIR[m["arm"]],
                     "condition": cond, "objset": b["objset"], "motion": b["motion"], "violation_type": b["violation"], "occlusion": b["occlusion"],
                     "k": b["k"], "background": b["background"], "occluder": b["occluder"], "occluder_kind": occ, "mover_dir": b["mover_dir"],
                     "speed": b["speed"], "replica": b["replica"], "obj_a": b["obj_a"], "samples": m["samples"]})
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} 행 · 블록 {len(bid)} -> {OUT}/index.csv")


if __name__ == "__main__":
    main()
