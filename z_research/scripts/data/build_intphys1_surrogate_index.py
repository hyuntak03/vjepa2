#!/usr/bin/env python3
"""IntPhys1_surrogate (UE 5.4 TestBed, IP1_O1_s16) → 채점 index (2026-11-03, 사용자: "IntPhys1_surrogate eval 해서 시트에 채워줘").

데이터: /workspace/dataset/benchmarks/IntPhys1_surrogate — 200 블록 × 4 팔 = 800 mp4 (96 장 · 30 fps · 512 px), 전부 primitive · permanence · late.
  motion: ip1_throw (물체 움직임, 100 블록, k 1) · flat_static (물체 정지, 100 블록, k 1–3). 배경 ip1.
  팔: pos_a (A|A) · pos_b (B|B) · imp_ab (A|B) · imp_ba (B|A), 이음 = 프레임 48. B = 빈 장면.
짝 (VOBench v3 와 같은 규칙): A = (pos_a, imp_ab) 사라짐 (문맥에 물체) · B = (pos_b, imp_ba) 나타남 (문맥이 빈 장면). 두 팔의 문맥 (0–47) 은 같은 렌더.
채점 창 = 원본 0, 3, …, 93 (32 장, 문맥 16 + 미래 16) — testbed 표준 (mp4 직독, data.frames_stride 3). README 의 "stride 3 → 32 samples" 와 같다.
condition = moving (ip1_throw) | static (flat_static) — 시트 Object_permanence "IntPhys1 Surrogate" 열 = [움직임 A, B · 정지 A, B · overall].
출력: data_csv/intphys1_surrogate/index.csv
  /venv/vjepa/bin/python z_research/scripts/data/build_intphys1_surrogate_index.py
"""
import csv, os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = "/workspace/dataset/benchmarks/IntPhys1_surrogate"
OUT = f"{REPO}/data_csv/intphys1_surrogate"
PAIR = {"pos_a": "A", "imp_ab": "A", "pos_b": "B", "imp_ba": "B"}
MOTION = {"ip1_throw": "moving", "flat_static": "static"}


def main():
    blocks = {r["block_id"]: r for r in csv.DictReader(open(f"{SRC}/blocks.csv"))}
    meta = list(csv.DictReader(open(f"{SRC}/metadata.csv")))
    bid = {b: i for i, b in enumerate(sorted(blocks))}
    rows = []
    for m in meta:
        b = blocks[m["block_id"]]
        assert os.path.isfile(f"{SRC}/{m['file_name']}"), m["file_name"]
        assert (m["arm"] in ("pos_a", "pos_b")) == (m["is_possible"] == "1"), m["name"]
        rows.append({"video_id": m["name"], "file": m["file_name"], "file_name": m["file_name"][:-4], "block_id": bid[m["block_id"]],
                     "block_name": m["block_id"], "variant": m["arm"], "plausible": m["is_possible"], "pair_id": PAIR[m["arm"]],
                     "condition": MOTION[b["motion"]], "objset": b["objset"], "motion": b["motion"], "violation_type": b["violation"],
                     "occlusion": b["occlusion"], "k": b["k"], "background": b["background"], "replica": b["replica"],
                     "obj_a": b["obj_a"], "color_a": b["color_a"], "samples": m["samples"]})
    n_arm = {}
    for r in rows:
        n_arm[r["block_id"]] = n_arm.get(r["block_id"], 0) + 1
    assert set(n_arm.values()) == {4}, "블록마다 팔 4 개여야 한다"
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} 행 · 블록 {len(bid)} -> {OUT}/index.csv")


if __name__ == "__main__":
    main()
