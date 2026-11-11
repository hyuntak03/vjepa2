#!/usr/bin/env python3
"""BlenderTrack v2 설계 미리 계산 — 렌더 작업자보다 앞서 clip 설계 (spec) 를 CPU 노드에서 계산해 둔다 (2026-11-01).

왜: v2 는 clip 하나 설계가 CPU 로 평균 약 5 초 (다른 물체 가림 칸은 재시도가 많다) 라 렌더 (약 3 초) 보다 길다.
    렌더 작업자 안에서 설계하면 GPU 가 논다 → 남는 CPU 노드에서 미리 계산해 <out>/design/<id>.json (실패면 .fail) 로 둔다.
    gen_render.py 는 그 파일이 있으면 읽고, 없으면 스스로 설계한다 (설계는 결정적이라 어느 쪽이든 같은 장면).
순서: 렌더 작업자 (claim 모드) 가 나아가는 차례 그대로 — 작업자 w 의 시작점 w·n/N 에서 한 칸씩, 작업자를 돌아가며 (앞질러 가도록).
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P gen_design_cache.py --config configs/v2.yaml --splits train test --nworkers 39 --procs 64 [--part 0 --nparts 2]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design_v2 as G  # noqa: E402

CFG = {}


def _init(path):
    for sp in ("train", "test", "smoke"):
        CFG[sp] = G.load_cfg(path, sp)


def one(job):
    split, i = job
    cfg = CFG[split]
    out, sd = cfg["out"], cfg["splits"][split]["seed"]
    cid = f"{split}{sd}_{i:06d}"
    d = f"{out}/design"
    if os.path.exists(f"{out}/labels/{cid}.npz") or os.path.exists(f"{d}/{cid}.json") or os.path.exists(f"{d}/{cid}.fail"):
        return 0
    try:
        spec = G.sample_clip(cfg, split, i, max_tries=1500, streams=int(cfg.get("checks", {}).get("design_streams", 10)))
        tmp = f"{d}/{cid}.{os.getpid()}.tmp"
        json.dump(spec, open(tmp, "w"), default=lambda x: x.item() if hasattr(x, "item") else str(x))
        os.replace(tmp, f"{d}/{cid}.json")
        return 1
    except RuntimeError as e:
        open(f"{d}/{cid}.fail", "w").write(str(e))
        return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--nworkers", type=int, required=True, help="렌더 작업자 수 (claim 모드 시작점 계산)")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    cfg = G.load_cfg(a.config, "train")
    os.makedirs(f"{cfg['out']}/design", exist_ok=True)
    jobs = []
    for sp in a.splits:
        n = cfg["splits"][sp]["n"]
        st = [(w * n) // a.nworkers for w in range(a.nworkers)]
        seen = set()
        for k in range(n):
            for w in range(a.nworkers):
                i = (st[w] + k) % n
                if i not in seen:
                    seen.add(i); jobs.append((sp, i))
    jobs = jobs[a.part::a.nparts]
    t0, c = time.time(), [0, 0, 0]
    with ProcessPoolExecutor(a.procs, initializer=_init, initargs=(a.config,)) as ex:
        for k, r in enumerate(ex.map(one, jobs, chunksize=1)):
            c[r] += 1
            if k % 2000 == 0:
                print(f"{k}/{len(jobs)} 건너뜀 {c[0]} 설계 {c[1]} 실패 {c[2]} | {time.time() - t0:.0f}s", flush=True)
    print(f"끝 {len(jobs)} 건너뜀 {c[0]} 설계 {c[1]} 실패 {c[2]} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
