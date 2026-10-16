#!/usr/bin/env python3
"""VOBench_v1 gravity (`VOBench_v1/gravity/metadata.csv`) → 표준 `index.csv` + 문맥 무결성 감사 `context_integrity.json` (2026-10-10).

gravity 는 VOBench 본판 (A/B 4 중항) 과 block 구조가 다르다 — **block = 과거 하나 + 미래 여럿** (데이터 README):
  ledge  : pos (낙하) · imp_float (같은 높이로 직진)                                   → 1 쌍 / block
  arc    : pos (포물선) · imp_rise · imp_float · imp_line · imp_stop · imp_stop{2,4,8,12} → 미래 9 개 / block
모든 arm 이 frame 0–47 을 공유한다 (P 의 hard link). 그래서 채점은 `scoring.pairing=cross` (block 안 가능 1 × 불가능 전부 =
문맥 일치 쌍 전부) 이고, pair_id = block 하나로 둔다. 레지스트리 `vobench_v1_gravity` 의 overrides 가 cross 를 건다.

열 (v11 index 14 열 + `sym_k` + 요인 열):
  video_id ← name · file_name ← file_name (frames_root = …/VOBench_v1/gravity) · block_id ← block 번호 (0 부터) · source_block ← block_id
  variant ← arm · plausible ← is_possible · pair_id ← block 번호 · condition ← <objset>/<law>/<cell> (채점 by_block_type 키)
  motion ← law (ledge | arc) · violation_type ← future (gravity · float · rise · line · frozen · frozen_after_<K>f)
  + objset · law · cell · background · obj · future · apex_frame · vz_splice_cm_s · land_frame (사후 분해용)

문맥 무결성 (CLAUDE.md §7-4): block 의 모든 불가능 arm 을 pos 와 frame 0–47 전부 비교한다 (어떤 창이든 문맥은 raw ≤ 47 —
sliding 창도 문맥 끝 ≤ 45). 같은 inode 면 같은 바이트, 다르면 픽셀 비교. mismatch 하나라도 있으면 죽는다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/data/build_vobench_gravity_index.py --frames-root /data2/local_datasets/world/Benchmark/VOBench_v1/gravity \
      --out data_csv/vobench_v1_gravity
"""
from __future__ import annotations
import argparse, csv, json, os, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

COLS = ["video_id", "file", "file_name", "block_id", "source_block", "variant", "plausible", "pair_id", "condition", "motion",
        "has_occlusion", "violation_type", "game_name", "role", "sym_k",
        "objset", "law", "cell", "background", "obj", "future", "apex_frame", "vz_splice_cm_s", "land_frame"]
CTX_FRAMES = list(range(0, 48))             # splice 48 전 전부
LAST_FRAME = 95
WANT = {"ledge": {"pos", "imp_float"},
        "arc": {"pos", "imp_rise", "imp_float", "imp_line", "imp_stop", "imp_stop2", "imp_stop4", "imp_stop8", "imp_stop12"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out", required=True, help="data_csv/<이름> (index.csv · context_integrity.json)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    root, out = Path(a.frames_root), Path(a.out)
    rows = list(csv.DictReader((root / "metadata.csv").open()))
    binfo = {b["block_id"]: b for b in csv.DictReader((root / "blocks.csv").open())}
    blocks = {b: i for i, b in enumerate(dict.fromkeys(r["block_id"] for r in rows))}
    idx = []
    for r in rows:
        b = binfo[r["block_id"]]
        idx.append(dict(
            video_id=r["name"], file="", file_name=r["file_name"], block_id=str(blocks[r["block_id"]]), source_block=r["block_id"],
            variant=r["arm"], plausible=r["is_possible"], pair_id=str(blocks[r["block_id"]]),
            condition=f'{r["objset"]}/{r["motion"]}/{b["cell"]}', motion=r["motion"], has_occlusion="0",
            violation_type=r["future"], game_name=r["game_name"], role=r["arm"], sym_k="0",
            objset=r["objset"], law=r["motion"], cell=b["cell"], background=r["env"], obj=r["obj_a"], future=r["future"],
            apex_frame=b["apex_frame"], vz_splice_cm_s=b["vz_splice_cm_s"], land_frame=b["land_frame_P"]))

    # block 구성 검사: 법칙별 arm 집합, 가능 정확히 1
    by = defaultdict(list)
    for d in idx:
        by[d["block_id"]].append(d)
    bad = [k for k, v in by.items() if {x["variant"] for x in v} != WANT[v[0]["law"]] or sum(x["plausible"] == "1" for x in v) != 1]
    if bad:
        raise SystemExit(f"block 구성이 맞지 않는 block {len(bad)} 개: {bad[:5]}")
    miss = [d["video_id"] for d in idx if not (root / d["file_name"] / f"{LAST_FRAME:06d}.png").exists()]
    if miss:
        raise SystemExit(f"프레임 {LAST_FRAME:06d}.png 가 없는 clip {len(miss)} 개: {miss[:3]}")

    t0 = time.time()

    def check(bid):
        v = by[bid]
        p_ = next(x for x in v if x["plausible"] == "1")
        same_inode = pixel_eq = 0; diff = {}
        for n_ in (x for x in v if x["plausible"] == "0"):
            for f in CTX_FRAMES:
                fp, fn = root / p_["file_name"] / f"{f:06d}.png", root / n_["file_name"] / f"{f:06d}.png"
                if os.stat(fp).st_ino == os.stat(fn).st_ino:
                    same_inode += 1; continue
                x, y = np.asarray(Image.open(fp)), np.asarray(Image.open(fn))
                if x.shape == y.shape and np.array_equal(x, y):
                    pixel_eq += 1
                else:
                    diff.setdefault(n_["variant"], []).append(f)
        return bid, same_inode, pixel_eq, diff

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(check, sorted(by, key=int)))
    mism = {k: d for k, _, _, d in res if d}
    n_cmp = sum(len(by[k]) - 1 for k in by)
    audit = {"n_blocks": len(res), "n_arm_comparisons": n_cmp, "n_frames_checked": n_cmp * len(CTX_FRAMES),
             "context_frames_raw": [CTX_FRAMES[0], CTX_FRAMES[-1]], "same_inode": sum(r[1] for r in res),
             "pixel_equal": sum(r[2] for r in res), "n_mismatch_blocks": len(mism), "mismatch": mism,
             "seconds": round(time.time() - t0, 1)}
    if mism:
        raise SystemExit(f"문맥 mismatch {len(mism)} block: {list(mism.items())[:3]}")

    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(idx)
    (out / "context_integrity.json").write_text(json.dumps(audit, indent=1))
    print(f"index {len(idx)} clip · {len(by)} block · 쌍 (cross) {n_cmp} -> {out / 'index.csv'}")
    print("  condition: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(d['condition'] for d in idx).items())))
    print(f"문맥 무결성: {n_cmp} arm 비교 × {len(CTX_FRAMES)} 장 — inode 같음 {audit['same_inode']} · 픽셀 같음 {audit['pixel_equal']} · "
          f"mismatch 0 ({audit['seconds']} s)")


if __name__ == "__main__":
    main()
