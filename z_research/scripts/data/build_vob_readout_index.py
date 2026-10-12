"""VOB_readout_primitive (TestBed, 2026-10) → RollOut 학습셋 index 형식 (rollout2_presence_readout · rollout2_identity_readout 가 읽는 꼴).

  python z_research/scripts/data/build_vob_readout_index.py        # → data_csv/rollout_v2_vob_readout_primitive/{index,index_probe}.csv

규칙 (데이터 README `Loader` 절):
  visible = has_object & in_frame & ~scenery_intersects  (usable) · ignore = has_object & ~usable · 빈 clip = shape_pre "none".
  좌표 = object_px_{x,y}_by_sample (3D 상자 중심, VOBench label.json 과 같은 정의) × 288/512 — 자 스크립트의 단위 (288 px, 1 칸 = 18 px = 토큰 1 개) 그대로.
  scenario = scene (block 안에서 같다 → block 단위 층화 split 이 빈 clip 과 물체 clip 을 같은 쪽에 둔다).
  balance_weight: 빈 clip 9 (block 당 1 : 48) · 물체 1. cell_weight_by_sample 은 그대로.
  프레임: Frames/<clip>/000000…000031.png (이미 10 fps 32 장) — 추출 때 PRESENCE_STRIDE=1.
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = Path("/data2/local_datasets/world/world_analysis/VOB_readout_primitive/metadata.csv")
OUT = REPO / "data_csv/rollout_v2_vob_readout_primitive"
SCALE = 288 / 512
EMPTY_WEIGHT = 9.0
COLS = ["video_id", "file", "file_name", "block_id", "variant", "plausible", "pair_id", "condition", "scenario", "probe_type",
        "shape_pre", "color_pre", "obj_apparent_px", "resolution", "px_x_by_sample", "px_y_by_sample", "in_frame_by_sample",
        "visible_by_sample", "ignore_by_sample", "scenery_intersects_by_sample", "cell_weight_by_sample", "balance_weight"]


def nums(s):
    return [float(x) for x in str(s).split()] if str(s).strip() else [0.0] * 32


def main():
    rows = list(csv.DictReader(SRC.open()))
    out = []
    for r in rows:
        obj = r["has_object"] in ("1", "True", "true")
        inf, scn = nums(r["in_frame_by_sample"]), nums(r["scenery_intersects_by_sample"])
        use = [int(obj and a > 0 and b == 0) for a, b in zip(inf, scn)]
        ign = [int(obj and not u) for u in use]
        px = " ".join(f"{v * SCALE:.2f}" for v in nums(r["object_px_x_by_sample"])) if obj else ""
        py = " ".join(f"{v * SCALE:.2f}" for v in nums(r["object_px_y_by_sample"])) if obj else ""
        rad = max(nums(r["object_r_px_by_sample"])) * SCALE if obj else 0.0
        out.append(dict(video_id=r["name"], file="", file_name=r["file_name"], block_id=r["block_id"], variant="pos", plausible=1,
                        pair_id="A", condition=r["scene"], scenario=r["scene"], probe_type="obj" if obj else "empty",
                        shape_pre=r["shape"] if obj else "none", color_pre=r["color"] if obj else "none",
                        obj_apparent_px=f"{2 * rad:.1f}", resolution=512, px_x_by_sample=px, px_y_by_sample=py,
                        in_frame_by_sample=" ".join(str(int(a > 0)) for a in inf), visible_by_sample=" ".join(map(str, use)),
                        ignore_by_sample=" ".join(map(str, ign)), scenery_intersects_by_sample=" ".join(str(int(b > 0)) for b in scn),
                        cell_weight_by_sample=r["cell_weight_by_sample"], balance_weight=1.0 if obj else EMPTY_WEIGHT))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("index.csv", "index_probe.csv"):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(out)
    n_obj = sum(o["probe_type"] == "obj" for o in out)
    n_use = sum(sum(map(int, o["visible_by_sample"].split())) for o in out)
    print(f"{len(out)} clip (물체 {n_obj} · 빈 {len(out) - n_obj}) · usable 샘플 {n_use} → {OUT}")


if __name__ == "__main__":
    main()
