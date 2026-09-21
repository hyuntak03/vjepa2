#!/usr/bin/env python3
"""IntPhys 2 Main 장면 단위 split — train 장면의 가능 영상으로 학습, test 장면 (가능+불가능) 으로 채점. 2026-09-19.

결정 (2026-09-19): HeldOut 은 라벨·장면 정보가 없어 (mp4 344 개뿐) 로컬 채점 불가 → test 도 Main 에서.
Main 가능 영상 전부로 학습하면 test 짝의 가능 영상을 학습에서 본 셈이라 → **장면 단위로 반반**.
층화: (condition, Difficulty, Camera) 셀마다 장면을 섞어 반으로 (홀수 셀은 번갈아 train/test 에 한 장면 더). seed 0.
⚠️ 논문 A.3: Main 을 학습에 쓰지 말 것 — 이 test 수치는 논문 Table 2 와 비교 불가 (같은 test 장면의 릴리즈 predictor 값과만 비교).

출력 (data_csv/IntPhys2/main_split/):
  train_possible.csv   "<mp4> 0" (video_csv, 학습)       train_scenes.txt / test_scenes.txt
  test_metadata.csv    test 장면의 4 유형 전부 (Main metadata.csv 와 같은 컬럼)
--link: <root>/MainTest/{metadata.csv, Videos -> ../Main/Videos} 를 만든다 → analysis/intphys2 채점기에 split: MainTest (코드 수정 없음)
  python z_training/data/build_intphys2_main_split.py [--root /local_datasets/world/IntPhys2] [--write] [--link]
"""
from __future__ import annotations
import argparse, collections, csv, os, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_csv/IntPhys2/main_split"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/local_datasets/world/IntPhys2"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true"); ap.add_argument("--link", action="store_true"); a = ap.parse_args()
    rows = list(csv.DictReader(open(f"{a.root}/Main/metadata.csv"))); fields = list(rows[0].keys())
    scenes = collections.defaultdict(list)
    for r in rows:
        scenes[r["SceneIndex"]].append(r)
    assert len(scenes) == 253 and all(len(v) == 4 for v in scenes.values())
    key = {s: (v[0]["condition"], v[0]["Difficulty"], v[0]["Camera"]) for s, v in scenes.items()}
    assert all(len({(r["condition"], r["Difficulty"], r["Camera"]) for r in v}) == 1 for v in scenes.values()), "장면 안 메타 불일치"
    cells = collections.defaultdict(list)
    for s in sorted(scenes, key=int):
        cells[key[s]].append(s)
    rng = random.Random(a.seed); train, test = [], []; extra_to_train = True
    for c in sorted(cells):
        ss = cells[c][:]; rng.shuffle(ss); h = len(ss) // 2
        if len(ss) % 2:
            h += int(extra_to_train); extra_to_train = not extra_to_train
        train += ss[:h]; test += ss[h:]
    assert not set(train) & set(test) and len(train) + len(test) == 253
    tr_pos = [r for s in train for r in scenes[s] if r["type"].endswith("_Possible")]
    te_rows = [r for s in test for r in scenes[s]]
    cnt = lambda rs, k: dict(sorted(collections.Counter(r[k] for r in rs).items()))
    print(f"장면 train {len(train)} / test {len(test)} | 학습 clip (가능만) {len(tr_pos)} | test 영상 {len(te_rows)} = matched 짝 {len(te_rows) // 2}")
    for k in ("condition", "Difficulty", "Camera"):
        print(f"  {k}: train 장면 {cnt([scenes[s][0] for s in train], k)} | test 장면 {cnt([scenes[s][0] for s in test], k)}")
    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "train_scenes.txt").write_text("\n".join(sorted(train, key=int)) + "\n")
        (OUT / "test_scenes.txt").write_text("\n".join(sorted(test, key=int)) + "\n")
        with open(OUT / "train_possible.csv", "w") as f:
            for r in tr_pos:
                f.write(f"{os.path.join(a.root, 'Main', r['file_name'])} 0\n")
        with open(OUT / "test_metadata.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(te_rows)
        print(f"-> {OUT}")
    if a.link:
        d = Path(a.root) / "MainTest"; d.mkdir(exist_ok=True)
        with open(d / "metadata.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(te_rows)
        if not (d / "Videos").exists():
            os.symlink("../Main/Videos", d / "Videos")
        print(f"-> {d} (metadata {len(te_rows)} 행, Videos -> ../Main/Videos)")


if __name__ == "__main__":
    main()
