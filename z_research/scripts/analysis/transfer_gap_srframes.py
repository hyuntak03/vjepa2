#!/usr/bin/env python3
"""transfer_gap Q3 · E31c — 대칭 잡음 바닥용 프레임: 확률적 반올림 (seed 별). 2026-10-30, 실험 B [X2] R2.

왜: R1 의 잡음 바닥 (NF) 은 결정적 반올림 한 판이라 뒤집힘이 PFT 쪽으로만 몰렸다 (R1 검토). 기대값이 원래 값인 평균 0 잡음을
    seed 여러 개로 걸어 (PFT − 릴리즈) 정확도 차의 분포를 내야 "IntPhys 1 하락 −5.6 이 잡음 바닥 안인가" 를 말할 수 있다.
정의
  원본 PNG (uint8) → float [0,1] → 로더와 같은 커널 (bilinear, align_corners=False, antialias 없음) 로 256 px → v (실수, 0–255 단위)
  확률적 반올림: y = floor(v + u),  u ~ U[0, 1) (화소 · 채널마다 독립)  → E[y] = v, |y − v| < 1
  ⚠️ 정정 (2026-10-30, 첫 판 뒤) — 첫 판은 seed 를 (s, 파일 경로) 로 정해, 원본이 화소로 같은 matched 쌍의 문맥 프레임에 서로 다른 잡음이
     들어갔다 (문맥 무결성 깨짐 → predictor 입력이 쌍 안에서 달라짐). 그 판은 frames_sr<s>_* 에 기록으로 남긴다.
     고친 판 (--content-key, 기본) 은 seed 를 (s, 원본 화소 내용의 md5) 로 정한다 → 같은 원본 프레임은 같은 잡음을 받는다. 출력 frames_src<s>_*.
  결정적 반올림 (R1 의 T0 · ip1q) 은 y = round(v) 다. 이 판은 같은 크기 (< 1/255) 의 잡음을 seed 마다 다르게 준다.
출력: /data2/local_datasets/world/world_analysis/cache/transfer_gap/frames_src<s>_{ip1,vob}/ (원본과 같은 상대 경로 · 이름; 첫 판은 frames_sr<s>_*)
  문맥 무결성: --check 로 matched 쌍 문맥 (VOBench 16 장 · IntPhys 1 원본이 같은 프레임) 이 화소로 같은지 검사한다

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_srframes.py --seeds 1,2,3,4 --sets ip1,vob --procs 22
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CACHE = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")
IP1 = Path("/local_datasets/world/world_analysis/IntPhys1_dev_frame_png")
VOB = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
SAMPLES = list(range(0, 96, 3))


def _resize_float(x):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float() / 255.0
    if t.shape[-1] != 256 or t.shape[-2] != 256:
        t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
    return (t[0].permute(1, 2, 0).clamp(0, 1) * 255.0).numpy()


def _sr(v, seed, key):
    h = int(hashlib.md5(key.encode()).hexdigest()[:12], 16)
    rng = np.random.default_rng([seed, h])
    return np.clip(np.floor(v + rng.random(v.shape, dtype=np.float32)), 0, 255).astype(np.uint8)


CONTENT_KEY = True


def _job(args):
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    which, seed, rel, srcs = args
    root = CACHE / (f"frames_src{seed}_{which}" if CONTENT_KEY else f"frames_sr{seed}_{which}")
    n = 0
    for src, name in srcs:
        q = root / rel / name
        if q.exists():
            continue
        o = np.ascontiguousarray(np.asarray(Image.open(src).convert("RGB")))
        v = _resize_float(o)
        key = hashlib.md5(o.tobytes()).hexdigest() if CONTENT_KEY else f"{rel}/{name}"
        q.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_sr(v, seed, key)).save(q, compress_level=1)
        n += 1
    return n


def jobs_ip1(seed):
    rows = list(csv.DictReader(open("/local_datasets/world/world_analysis/IntPhys1_dev_videos/index.csv")))
    vids = sorted({f"{r['block']}/{r['quadruplet']}/{r['run']}" for r in rows})
    out = []
    for d in vids:
        ps = sorted((IP1 / d / "scene").glob("scene_*.png"))
        out.append(("ip1", seed, f"{d}/scene", [(p, p.name) for p in ps]))
    return out


def jobs_vob(seed):
    rows = list(csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv")))
    out = []
    for fn in sorted({r["file_name"] for r in rows}):
        out.append(("vob", seed, fn, [(VOB / fn / f"{f:06d}.png", f"{f:06d}.png") for f in SAMPLES]))
    return out


def check(seed):
    """matched 쌍 문맥 무결성: 원본에서 화소가 같은 (가능, 불가능) 프레임 짝이 잡음 판에서도 같은가. VOBench 문맥 16 장 · IntPhys 1 전 프레임."""
    import json
    from collections import defaultdict
    from PIL import Image
    res = {}
    # VOBench test: block 의 (pos_a, imp_ab) · (pos_b, imp_ba), 문맥 원프레임 0..45
    by = defaultdict(dict)
    for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv")):
        by[r["block_id"]][r["variant"]] = r["file_name"]
    n = bad = orig_bad = 0
    for v in by.values():
        for p, q in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            if p in v and q in v:
                n += 1
                same_o = all(np.array_equal(np.asarray(Image.open(VOB / v[p] / f"{f:06d}.png")), np.asarray(Image.open(VOB / v[q] / f"{f:06d}.png"))) for f in SAMPLES[:16])
                same_s = all(np.array_equal(np.asarray(Image.open(CACHE / f"frames_src{seed}_vob" / v[p] / f"{f:06d}.png")),
                                            np.asarray(Image.open(CACHE / f"frames_src{seed}_vob" / v[q] / f"{f:06d}.png"))) for f in SAMPLES[:16])
                orig_bad += not same_o
                bad += same_o and not same_s
    res["vobench_test"] = {"n_pairs": n, "orig_context_mismatch": orig_bad, "noise_context_mismatch_where_orig_same": bad}
    # IntPhys 1: 쌍의 같은 번호 프레임 중 원본이 같은 것은 잡음 판에서도 같아야 한다
    by = defaultdict(dict)
    for r in csv.DictReader(open("/local_datasets/world/world_analysis/IntPhys1_dev_videos/index.csv")):
        by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = f"{r['block']}/{r['quadruplet']}/{r['run']}"
    n = same_frames = bad = 0
    for v in by.values():
        if "pos" not in v or "imp" not in v:
            continue
        n += 1
        for pth in sorted((IP1 / v["pos"] / "scene").glob("scene_*.png")):
            o1, o2 = np.asarray(Image.open(pth)), np.asarray(Image.open(IP1 / v["imp"] / "scene" / pth.name))
            if np.array_equal(o1, o2):
                same_frames += 1
                s1 = np.asarray(Image.open(CACHE / f"frames_src{seed}_ip1" / v["pos"] / "scene" / pth.name))
                s2 = np.asarray(Image.open(CACHE / f"frames_src{seed}_ip1" / v["imp"] / "scene" / pth.name))
                bad += not np.array_equal(s1, s2)
    res["intphys1"] = {"n_pairs": n, "orig_identical_frames": same_frames, "noise_mismatch_among_them": bad}
    out = REPO / "z_analysis/transfer_gap/results/q3_r2" / f"E31c_integrity_src{seed}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("integrity", seed, res, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--sets", default="ip1,vob")
    ap.add_argument("--procs", type=int, default=min(22, os.cpu_count() or 4))
    ap.add_argument("--path-key", action="store_true", help="첫 판 (잘못된) 경로 seed — 기록 재현용")
    ap.add_argument("--check", type=int, default=0, help="seed 하나의 문맥 무결성만 검사")
    a = ap.parse_args()
    global CONTENT_KEY
    CONTENT_KEY = not a.path_key
    if a.check:
        check(a.check)
        return
    for s in [int(x) for x in a.seeds.split(",")]:
        for w in a.sets.split(","):
            J = jobs_ip1(s) if w == "ip1" else jobs_vob(s)
            t0 = time.time()
            with ProcessPoolExecutor(a.procs) as ex:
                n = sum(ex.map(_job, J, chunksize=4))
            print(f"seed {s} {w}: {len(J)} 영상, {n} 장 새로, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
