#!/usr/bin/env python3
"""transfer_gap R4 [X2] — 전이 시험판 · v11 학습 부분집합을 NFS 에 **바이트 그대로** 복사하고 색인을 만든다 (2026-10-30).

왜: R4 진단 PFT 학습 · 채점을 vll3 (RTX 3090) 에서 돌린다. vll5 로컬 (/local_datasets, /data2) 에만 있는 프레임을
    NFS 로 옮겨야 다른 노드에서 읽힌다. 리사이즈 없이 파일을 그대로 복사하므로 (cmp 로 확인) 원판 채점과 같은 화소다
    — VOBench 256² 사본 (리사이즈한 판, transfer_gap_stage_vobench256.py) 과 달리 사본 단서가 붙지 않는다.

무엇을 (모두 이 셸 = vll5 에서 CPU 로)
  ip1dev   IntPhys 1 dev 프레임 전부 (3.3 GB) + dev 색인 csv          → <NFS>/tg4_IntPhys1_dev_frame_png · tg4_IntPhys1_dev_videos
  ip1uni   IntPhys 1 late uni (360 영상 × 32 장)                       → <NFS>/tg4_IntPhys1_late_uni
  v11      v11 split test 의 문맥 끝 가림 사라짐 (static_occlusion · moving_occlusion_flat · moving_occlusion, vanish; 1,344 clip)
           + v11 split train 부분집합 (가능 변이, block 414 개 = 828 clip, (condition × violation_type) 층화, seed 0)
                                                                        → <NFS>/tg4_IntPhysGen_v11/<file_name>/
  v11real  v11_realistic 의 가림 사라짐 3 조건 (720 clip)                → <NFS>/tg4_IntPhysGen_v11_realistic/<file_name>/
색인 (data_csv/transfer_gap_r4_*/, 기존 레지스트리 항목은 안 고친다)
  data_csv/transfer_gap_r4_v11/index_test_latevanish.csv · index_train_sub828.csv
  data_csv/transfer_gap_r4_v11real/index_occ.csv
문맥 무결성: 복사본에서 matched 쌍 (같은 block · 같은 pair_id 의 가능 · 불가능) 의 문맥 16 장 (stride 3, 0..45) 이
  바이트로 같은지 전수 검사 → data_csv/transfer_gap_r4_*/context_integrity.json. 원본과 복사본이 같은지 (cmp) 표본 검사도 같이.

재현
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/data/transfer_gap_r4_stage.py index
  $P z_research/scripts/data/transfer_gap_r4_stage.py copy --what ip1uni,v11,v11real,ip1dev --workers 6
  $P z_research/scripts/data/transfer_gap_r4_stage.py integrity
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NFS = REPO / "z_pixel_decoder/runs/_frames"
LOC = Path("/local_datasets/world/world_analysis")
V11_SRC = LOC / "IntPhysGen_v11"
V11R_SRC = LOC / "IntPhysGen_v11_realistic"
OUT_V11 = REPO / "data_csv/transfer_gap_r4_v11"
OUT_V11R = REPO / "data_csv/transfer_gap_r4_v11real"
OCC = ("static_occlusion", "moving_occlusion_flat", "moving_occlusion")


def read(p):
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write(p, cols, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  {p}  {len(rows)} 행")


def cmd_index(_):
    cols, te = read(REPO / "data_csv/intphysgen_v11_split/index_test.csv")
    late = [r for r in te if r["condition"] in OCC and r["violation_type"] == "vanish"]
    write(OUT_V11 / "index_test_latevanish.csv", cols, late)
    cols_tr, tr = read(REPO / "data_csv/intphysgen_v11_split/index_train.csv")
    byb = defaultdict(list)
    for r in tr:
        byb[r["block_id"]].append(r)
    cells = defaultdict(list)
    for b, rs in byb.items():
        cells[(rs[0]["condition"], rs[0]["violation_type"])].append(b)
    nblk = sum(len(v) for v in cells.values())
    want = 414
    rng = random.Random(0)
    quota = {c: want * len(v) / nblk for c, v in cells.items()}
    take = {c: int(q) for c, q in quota.items()}
    rest = sorted(cells, key=lambda c: -(quota[c] - take[c]))
    for c in rest[: want - sum(take.values())]:
        take[c] += 1
    chosen = []
    for c in sorted(cells):
        bl = sorted(cells[c], key=int)
        chosen += rng.sample(bl, take[c])
    rows = [r for b in sorted(chosen, key=int) for r in byb[b]]
    write(OUT_V11 / "index_train_sub828.csv", cols_tr, rows)
    cols_r, rr = read(REPO / "data_csv/intphysgen_v11_realistic/index.csv")
    occ = [r for r in rr if r["condition"] in OCC]
    write(OUT_V11R / "index_occ.csv", cols_r, occ)
    json.dump({"v11_test_latevanish": len(late), "v11_train_sub_blocks": len(chosen), "v11_train_sub_clips": len(rows),
               "v11_train_cells": {f"{a}|{b}": take[(a, b)] for a, b in sorted(take)}, "v11real_occ": len(occ),
               "seed": 0}, open(OUT_V11 / "build.json", "w"), indent=1, ensure_ascii=False)


def _copy_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(src)):
        s, d = src / fn, dst / fn
        if s.is_dir():
            n += _copy_tree(s, d)
            continue
        if d.exists() and d.stat().st_size == s.stat().st_size:
            n += 1
            continue
        tmp = d.with_name(d.name + ".tmp")
        shutil.copyfile(s, tmp)
        os.replace(tmp, d)
        n += 1
    return n


def _jobs(what):
    jobs = []
    if "ip1uni" in what:
        src = LOC / "IntPhys1_late_uni"
        jobs += [(src / v, NFS / "tg4_IntPhys1_late_uni" / v) for v in sorted(os.listdir(src))]
    if "v11" in what:
        rows = read(OUT_V11 / "index_test_latevanish.csv")[1] + read(OUT_V11 / "index_train_sub828.csv")[1]
        jobs += [(V11_SRC / r["file_name"], NFS / "tg4_IntPhysGen_v11" / r["file_name"]) for r in rows]
    if "v11real" in what:
        rows = read(OUT_V11R / "index_occ.csv")[1]
        jobs += [(V11R_SRC / r["file_name"], NFS / "tg4_IntPhysGen_v11_realistic" / r["file_name"]) for r in rows]
    if "ip1dev" in what:
        src = LOC / "IntPhys1_dev_frame_png"
        for blk in sorted(os.listdir(src)):
            for q in sorted(os.listdir(src / blk)):
                jobs.append((src / blk / q, NFS / "tg4_IntPhys1_dev_frame_png" / blk / q))
        dv = NFS / "tg4_IntPhys1_dev_videos"
        dv.mkdir(parents=True, exist_ok=True)
        for fn in ("index.csv", "labels.csv", "keystones.csv"):
            shutil.copyfile(LOC / "IntPhys1_dev_videos" / fn, dv / fn)
    return jobs


def cmd_copy(a):
    what = set(a.what.split(","))
    jobs = _jobs(what)
    print(f"{len(jobs)} 폴더 복사 → {NFS}", flush=True)
    tot = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, n in enumerate(ex.map(lambda sd: _copy_tree(*sd), jobs)):
            tot += n
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)} files {tot}", flush=True)
    # 표본 cmp (폴더마다 첫 파일 · 마지막 파일, 최대 400 폴더)
    rng = random.Random(1)
    bad = 0
    sample = rng.sample(jobs, min(400, len(jobs)))
    for s, d in sample:
        fs = sorted(p for p in s.rglob("*") if p.is_file())
        for p in (fs[0], fs[-1]):
            q = d / p.relative_to(s)
            if not filecmp.cmp(p, q, shallow=False):
                bad += 1
    rec = {"what": sorted(what), "dirs": len(jobs), "files": tot, "cmp_sample_dirs": len(sample), "cmp_mismatch": bad}
    for w in what:
        (NFS / f"_tg4_DONE_{w}.json").write_text(json.dumps(rec, ensure_ascii=False))
    print("DONE", rec, flush=True)


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def _integrity(index_csv: Path, frames_root: Path, pattern: str, ctx_frames, out: Path):
    rows = read(index_csv)[1]
    byp = defaultdict(list)
    for r in rows:
        byp[(r["block_id"], r["pair_id"])].append(r)
    n_pair = n_bad = n_missing = 0
    bad = []
    for key, rs in sorted(byp.items()):
        if len(rs) != 2 or {r["plausible"] for r in rs} != {"0", "1"}:
            continue
        n_pair += 1
        h = []
        for r in rs:
            hs = []
            for f in ctx_frames(r):
                p = frames_root / pattern.format(frame=f, **r)
                if not p.exists():
                    n_missing += 1
                    hs.append(None)
                else:
                    hs.append(_md5(p))
            h.append(hs)
        if h[0] != h[1]:
            n_bad += 1
            bad.append(list(key))
    rec = {"index": str(index_csv), "frames_root": str(frames_root), "n_pair": n_pair, "context_mismatch": n_bad,
           "missing_frames": n_missing, "bad_pairs": bad[:20]}
    json.dump(rec, open(out, "w"), indent=1, ensure_ascii=False)
    print(rec, flush=True)
    return rec


def cmd_integrity(_):
    pat = "{file_name}/{frame:06d}.png"
    ctx = lambda r: range(0, 46, 3)  # noqa: E731  stride 3 문맥 16 장 (0..45)
    _integrity(OUT_V11 / "index_test_latevanish.csv", NFS / "tg4_IntPhysGen_v11", pat, ctx,
               OUT_V11 / "context_integrity_test_latevanish.json")
    _integrity(OUT_V11R / "index_occ.csv", NFS / "tg4_IntPhysGen_v11_realistic", pat, ctx,
               OUT_V11R / "context_integrity_occ.json")
    # IntPhys 1 late uni: 영상마다 32 장을 미리 골라 둔 판 → 문맥 = 저장 프레임 0..15
    _integrity(REPO / "data_csv/intphys1_late_uni/index.csv", NFS / "tg4_IntPhys1_late_uni", "{video_id}/{frame:06d}.png",
               lambda r: range(16), OUT_V11 / "context_integrity_ip1uni_copy.json")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("index")
    c = sp.add_parser("copy")
    c.add_argument("--what", default="ip1uni,v11,v11real,ip1dev")
    c.add_argument("--workers", type=int, default=6)
    sp.add_parser("integrity")
    a = ap.parse_args()
    {"index": cmd_index, "copy": cmd_copy, "integrity": cmd_integrity}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
