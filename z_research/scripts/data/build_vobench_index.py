#!/usr/bin/env python3
"""VOBench_v1 `metadata.csv` → 표준 `index.csv` + 문맥 무결성 감사 `context_integrity.json` (2026-10-10).

VOBench_v1 (UE 5.4 TestBed `tb/export.py`) 의 metadata 열 이름이 IntPhysGen 렌더와 달라 `build_intphysgen_index.py` 를 못 쓴다.
열 대응 (v11 index 와 같은 14 열 + `sym_k` + 요인 열):

  video_id ← name             file ← (빈칸, 프레임은 frames_root + file_name)        file_name ← file_name
  block_id ← block 을 0 부터 번호 (LIMIT 이 int 로 정렬한다)    source_block ← block_id
  variant ← arm (pos_a · pos_b · imp_ab · imp_ba)              plausible ← is_possible
  pair_id ← first_half (A / B = 문맥 묶음)                      condition ← <objset>/<violation>/<occlusion> (채점 by_block_type 키)
  motion ← motion             has_occlusion ← occlusion != visible     violation_type ← violation
  game_name ← game_name       role ← arm                    sym_k ← k
  + objset · violation · occlusion · k · background (← env) · obj_a · obj_b · ctx_clean_samples · samples (사후 분해용, 채점기는 안 읽는다)

matched pair = 같은 block 안에서 pair_id 가 같은 (가능 1, 불가능 1): A = (pos_a, imp_ab), B = (pos_b, imp_ba).

문맥 무결성 (CLAUDE.md §7-4): 쌍마다 문맥 16 샘플 (raw 0, 3, …, 45) 을 비교한다. arm 폴더는 A/ B/ 로의 hard link 라
같은 inode 면 같은 바이트로 세고, 다르면 PNG 를 픽셀 단위로 비교한다. mismatch 가 하나라도 있으면 죽는다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/data/build_vobench_index.py --frames-root /data2/local_datasets/world/Benchmark/VOBench_v1 \
      --out data_csv/vobench_v1
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
        "objset", "violation", "occlusion", "k", "background", "obj_a", "obj_b", "ctx_clean_samples", "samples"]
CTX_FRAMES = list(range(0, 48, 3))          # 문맥 16 샘플 (stride 3)
LAST_FRAME = 93                             # 32 번째 샘플 (stride 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out", required=True, help="data_csv/<이름> (index.csv · context_integrity.json)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    root, out = Path(a.frames_root), Path(a.out)
    rows = list(csv.DictReader((root / "metadata.csv").open()))
    blocks = {b: i for i, b in enumerate(dict.fromkeys(r["block_id"] for r in rows))}
    idx = []
    for r in rows:
        idx.append(dict(
            video_id=r["name"], file="", file_name=r["file_name"], block_id=str(blocks[r["block_id"]]), source_block=r["block_id"],
            variant=r["arm"], plausible=r["is_possible"], pair_id=r["first_half"],
            condition=f'{r["objset"]}/{r["violation"]}/{r["occlusion"]}', motion=r["motion"],
            has_occlusion=str(int(r["occlusion"] != "visible")), violation_type=r["violation"], game_name=r["game_name"],
            role=r["arm"], sym_k=r["k"], objset=r["objset"], violation=r["violation"], occlusion=r["occlusion"], k=r["k"],
            background=r["env"], obj_a=r["obj_a"], obj_b=r["obj_b"], ctx_clean_samples=r["ctx_clean_samples"], samples=r["samples"]))

    # 쌍 구성 검사: block × pair_id 마다 가능 1 · 불가능 1, 그리고 pair_id 가 arm 과 맞는가
    pairs = defaultdict(list)
    for d in idx:
        pairs[(d["block_id"], d["pair_id"])].append(d)
    want = {"A": {"pos_a", "imp_ab"}, "B": {"pos_b", "imp_ba"}}
    bad = [k for k, v in pairs.items() if {x["variant"] for x in v} != want[k[1]] or sorted(x["plausible"] for x in v) != ["0", "1"]]
    if bad:
        raise SystemExit(f"쌍 구성이 (pos, imp) 1 + 1 이 아닌 (block, pair_id) {len(bad)} 개: {bad[:5]}")
    # 프레임 실물 (stride 3 의 마지막 샘플)
    miss = [d["video_id"] for d in idx if not (root / d["file_name"] / f"{LAST_FRAME:06d}.png").exists()]
    if miss:
        raise SystemExit(f"프레임 {LAST_FRAME:06d}.png 가 없는 clip {len(miss)} 개: {miss[:3]}")

    # 문맥 무결성
    t0 = time.time()

    def check(key):
        v = pairs[key]
        p_ = next(x for x in v if x["plausible"] == "1"); n_ = next(x for x in v if x["plausible"] == "0")
        same_inode = pixel_eq = 0; diff = []
        for f in CTX_FRAMES:
            fp, fn = root / p_["file_name"] / f"{f:06d}.png", root / n_["file_name"] / f"{f:06d}.png"
            if os.stat(fp).st_ino == os.stat(fn).st_ino:
                same_inode += 1; continue
            x, y = np.asarray(Image.open(fp)), np.asarray(Image.open(fn))
            if x.shape == y.shape and np.array_equal(x, y):
                pixel_eq += 1
            else:
                diff.append(f)
        return key, same_inode, pixel_eq, diff

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(check, sorted(pairs)))
    mism = {f"{k[0]}/{k[1]}": d for k, _, _, d in res if d}
    audit = {"n_pairs": len(res), "n_frames_checked": len(res) * len(CTX_FRAMES), "context_frames_raw": CTX_FRAMES,
             "same_inode": sum(r[1] for r in res), "pixel_equal": sum(r[2] for r in res),
             "n_mismatch_pairs": len(mism), "mismatch": mism, "seconds": round(time.time() - t0, 1)}
    if mism:
        raise SystemExit(f"문맥 mismatch {len(mism)} 쌍: {list(mism.items())[:3]}")

    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(idx)
    (out / "context_integrity.json").write_text(json.dumps(audit, indent=1))
    print(f"index {len(idx)} clip · {len(blocks)} block · {len(pairs)} matched pair -> {out / 'index.csv'}")
    print(f"  condition {len(set(d['condition'] for d in idx))} 종: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(d['condition'] for d in idx).items())))
    print(f"문맥 무결성: {len(res)} 쌍 × {len(CTX_FRAMES)} 장 — inode 같음 {audit['same_inode']} · 픽셀 같음 {audit['pixel_equal']} · mismatch 0 ({audit['seconds']} s)")


if __name__ == "__main__":
    main()
