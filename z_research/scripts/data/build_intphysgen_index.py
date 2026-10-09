#!/usr/bin/env python3
"""IntPhysGen 계열 렌더의 `metadata.csv` → 표준 `index.csv` + 문맥 무결성 감사 `context_integrity.json` (2026-10-06).

기존 v11 · v13 index 와 **같은 열 · 같은 규칙**이다 (그 index 들을 만든 스크립트는 레포에 없어, v11 index 와 metadata 를 맞대 규칙을 복원했다):

  video_id ← name            file ← (빈칸, 프레임은 frames_root + file_name)   file_name ← file_name
  block_id ← block 을 0 부터 번호     source_block ← block     variant ← clip
  plausible ← is_possible    pair_id ← first_half_variant (문맥 묶음 A/B)       condition · motion · has_occlusion · game_name · role ← 그대로 · violation_type ← block 의 위반 이름
  sym_k ← sym_k (있으면, 15 번째 열 — 채점 분해용)

matched pair = 같은 block 안에서 pair_id 가 같은 (가능 1, 불가능 1). 4 중항 (v11 계열) 이면 A · B 두 쌍, 2 중항 (ledge) 이면 A 한 쌍.
문맥 하나에 미래 여럿 (gravity_realistic: 가능 1 + 불가능 3, 2026-10-07) 도 받는다 — (block, pair_id) 마다 가능 정확히 1 · 불가능 1 개 이상.
  그때는 block 전체가 문맥 하나라 채점은 `scoring.pairing: cross` (= 가능 × 불가능 전수 = 문맥 일치 쌍 전부) 로 한다.

문맥 무결성 (CLAUDE.md §7-4): 쌍마다 문맥 16 샘플 (raw 0, 3, …, 45) PNG 를 픽셀 단위로 비교한다. mismatch 가 하나라도 있으면 죽는다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/data/build_intphysgen_index.py --frames-root /local_datasets/world/world_analysis/IntPhysGen_v11_realistic \
      --out data_csv/intphysgen_v11_realistic
"""
from __future__ import annotations
import argparse, csv, json, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

COLS = ["video_id", "file", "file_name", "block_id", "source_block", "variant", "plausible", "pair_id", "condition", "motion",
        "has_occlusion", "violation_type", "game_name", "role"]
CTX_FRAMES = list(range(0, 48, 3))          # 문맥 16 샘플 (stride 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out", required=True, help="data_csv/<이름> (index.csv · context_integrity.json)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    root, out = Path(a.frames_root), Path(a.out)
    rows = list(csv.DictReader((root / "metadata.csv").open()))
    has_k = "sym_k" in rows[0]
    blocks = {b: i for i, b in enumerate(dict.fromkeys(r["block"] for r in rows))}
    # violation_type 은 **block 단위** 로 적는다 (v11 index 관례: 가능 clip 도 그 block 의 위반 이름). 새 렌더 metadata 는 가능 clip 에 'none' 을 적는다
    vio = {}
    for r in rows:
        if r["violation_type"] not in ("", "none"):
            vio[r["block"]] = r["violation_type"]
    idx = []
    for r in rows:
        d = dict(video_id=r["name"], file="", file_name=r["file_name"], block_id=str(blocks[r["block"]]), source_block=r["block"],
                 variant=r["clip"], plausible=r["is_possible"], pair_id=r["first_half_variant"], condition=r["condition"],
                 motion=r["motion"], has_occlusion=r["has_occlusion"], violation_type=vio.get(r["block"], r["violation_type"]),
                 game_name=r["game_name"], role=r["role"])
        if has_k:
            d["sym_k"] = r["sym_k"]
        idx.append(d)
    # 쌍 구성 검사: block × pair_id 마다 가능 1 · 불가능 1
    pairs = defaultdict(list)
    for d in idx:
        pairs[(d["block_id"], d["pair_id"])].append(d)
    bad = [k for k, v in pairs.items() if [x["plausible"] for x in v].count("1") != 1 or [x["plausible"] for x in v].count("0") < 1]
    if bad:
        raise SystemExit(f"쌍 구성이 가능 1 + 불가능 1 개 이상이 아닌 (block, pair_id) {len(bad)} 개: {bad[:5]}")
    multi = sum(len(v) > 2 for v in pairs.values())
    # 프레임 실물
    miss = [d["video_id"] for d in idx if not (root / d["file_name"] / "000093.png").exists()]
    if miss:
        raise SystemExit(f"프레임 000093.png 가 없는 clip {len(miss)} 개: {miss[:3]}")

    # 문맥 무결성
    t0 = time.time()

    def check(key):
        v = pairs[key]; p_ = next(x for x in v if x["plausible"] == "1")
        diff = []
        for n_ in (x for x in v if x["plausible"] != "1"):          # 불가능이 여럿이면 가능과 하나씩 비교
            for f in CTX_FRAMES:
                x = np.asarray(Image.open(root / p_["file_name"] / f"{f:06d}.png"))
                y = np.asarray(Image.open(root / n_["file_name"] / f"{f:06d}.png"))
                if x.shape != y.shape or not np.array_equal(x, y):
                    diff.append((n_["video_id"], f))
        return key, p_["video_id"], "+".join(x["video_id"] for x in v if x["plausible"] != "1"), diff
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(check, sorted(pairs)))
    mism = [dict(block_id=k[0], pair_id=k[1], possible=pv, impossible=nv, frames=df) for k, pv, nv, df in res if df]
    out.mkdir(parents=True, exist_ok=True)
    (out / "context_integrity.json").write_text(json.dumps(dict(
        frames_root=str(root), n_pairs=len(res), context_frames=CTX_FRAMES, n_mismatch=len(mism), mismatches=mism[:50],
        checked=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    print(f"[integrity] 쌍 {len(res)} · mismatch {len(mism)} · {time.time() - t0:.0f}s")
    if mism:
        raise SystemExit(f"문맥이 다른 matched pair {len(mism)} 개 — {out}/context_integrity.json")
    with (out / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS + (["sym_k"] if has_k else []))
        w.writeheader(); w.writerows(idx)
    n_cmp = sum(len(v) - 1 for v in pairs.values())
    print(f"[index] {len(idx)} clip · block {len(blocks)} · 문맥 묶음 {len(pairs)} (그중 불가능 여럿 {multi}) · 가능-불가능 비교 {n_cmp} → {out}/index.csv")


if __name__ == "__main__":
    main()
