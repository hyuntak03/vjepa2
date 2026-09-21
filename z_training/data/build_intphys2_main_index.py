#!/usr/bin/env python3
"""IntPhys 2 Main 의 가능 영상만 → video_csv 학습 인덱스 (2026-09-19 결정: 학습 = Main 가능, test = HeldOut 리더보드).

⚠️ 논문 A.3: "Metadata in this set [Main] should only be used for detailed error analysis, not for training."
   이 인덱스로 학습한 predictor 의 Main 점수는 논문 Table 2 와 비교할 수 없다 — 유효한 test 는 HeldOut 뿐.
출력: data_csv/IntPhys2/main_possible.csv  ("<mp4 절대경로> 0", 공백 구분, VideoDataset 형식) + main_possible_meta.csv (검사용 메타)
검사: 506 행 = 253 장면 × {1_Possible, 2_Possible}, 파일 존재, 512×512 · 60 fps · 길이 ≥ 480 (= 6 fps 48 장 창) (--check 로 전수 decord 확인)
      길이는 영상마다 다르다 (가능 506 개: 635–938 프레임, 636 인 것 389 개; 2026-09-19 전수).
  python z_training/data/build_intphys2_main_index.py [--root /local_datasets/world/IntPhys2] [--check] [--write]
"""
from __future__ import annotations
import argparse, collections, csv, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_csv/IntPhys2"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/local_datasets/world/IntPhys2")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    rows = list(csv.DictReader(open(f"{a.root}/Main/metadata.csv")))
    pos = [r for r in rows if r["type"].endswith("_Possible")]
    per_scene = collections.Counter(r["SceneIndex"] for r in pos)
    assert len(rows) == 1012 and len(pos) == 506, (len(rows), len(pos))
    assert set(per_scene.values()) == {2} and len(per_scene) == 253, "장면마다 가능 2 개여야 한다"
    assert all("Impossible" not in r["type"] for r in pos)
    paths = [os.path.join(a.root, "Main", r["file_name"]) for r in pos]
    missing = [p for p in paths if not os.path.isfile(p)]; assert not missing, missing[:3]
    if a.check:
        import decord
        bad, lens = [], []
        for p in paths:
            vr = decord.VideoReader(p, num_threads=1); h, w = vr[0].shape[:2]; lens.append(len(vr))
            if (h, w, round(vr.get_avg_fps())) != (512, 512, 60) or len(vr) < 480:
                bad.append((p, len(vr), h, w, vr.get_avg_fps()))
        print(f"decord 전수: {len(paths)} 개, 규격 밖 {len(bad)} | 길이 min {min(lens)} / max {max(lens)} / "
              f"636 인 것 {sum(l == 636 for l in lens)}"); [print("  ", b) for b in bad[:10]]
    print(f"가능 {len(pos)} / 장면 {len(per_scene)} | 난이도 {dict(collections.Counter(r['Difficulty'] for r in pos))} | "
          f"카메라 {dict(collections.Counter(r['Camera'] for r in pos))} | 조건 {dict(collections.Counter(r['condition'] for r in pos))}")
    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "main_possible.csv", "w") as f:
            for p in paths:
                f.write(f"{p} 0\n")
        with open(OUT / "main_possible_meta.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pos[0].keys()) + ["path"]); w.writeheader()
            for r, p in zip(pos, paths):
                w.writerow({**r, "path": p})
        print(f"-> {OUT / 'main_possible.csv'} ({len(paths)} 행)")


if __name__ == "__main__":
    main()
