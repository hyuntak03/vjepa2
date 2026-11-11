#!/usr/bin/env python3
"""BlenderTrack v3 장면 설계 (CPU, 렌더 전) — 분할의 후보 장면을 요인표대로 설계해 <out>/design/ 에 팔 A · B spec 을 쓴다 (2026-11-01).

  장면 b → 렌더 번호 2b (A, target 있음) · 2b + 1 (B, target 만 뺌). 실패한 장면은 두 번호 모두 .fail.
  여러 노드로 나눠 돌린다 (--part k --nparts m, 장면 번호 b % m == k). 이미 있는 장면은 건너뛴다.
  끝나면 select_blocks_v3.py 가 상황마다 요인 분포가 가장 고른 blocks_per_situation 개를 고른다.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P design_blocks_v3.py --config configs/v3.yaml --split train --procs 120 --part 0 --nparts 3
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design_v3 as G  # noqa: E402

CFG = {}


def _init(path, split):
    CFG["cfg"] = G.load_cfg(path, split); CFG["split"] = split


def _dump(sp, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    json.dump(sp, open(tmp, "w"), default=lambda x: x.item() if hasattr(x, "item") else str(x))
    os.replace(tmp, path)


def one(b):
    cfg, split = CFG["cfg"], CFG["split"]
    out, sd = cfg["out"], cfg["splits"][split]["seed"]
    ca, cb = f"{out}/design/{split}{sd}_{2 * b:06d}", f"{out}/design/{split}{sd}_{2 * b + 1:06d}"
    if os.path.exists(ca + ".json") or os.path.exists(ca + ".fail"):
        return 0
    try:
        A = G.sample_clip(cfg, split, 2 * b, max_tries=1500, streams=int(cfg["checks"].get("design_streams", 3)))
    except RuntimeError as e:
        for c in (ca, cb):
            open(c + ".fail", "w").write(str(e))
        return 2
    _, _, _, _, _, F = G.block_of(cfg, split, 2 * b)
    Bs = G.arm_spec({k: v for k, v in A.items() if k not in ("arm", "block", "splice_frame", "factors")}, "B", 2 * b + 1, b, F)
    _dump(A, ca + ".json"); _dump(Bs, cb + ".json")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--procs", type=int, default=32); ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--reverse", action="store_true", help="도우미: 같은 몫을 뒤에서부터 (끝난 장면은 건너뛴다)")
    a = ap.parse_args()
    cfg = G.load_cfg(a.config, a.split)
    os.makedirs(f"{cfg['out']}/design", exist_ok=True)
    nb = len(G.block_situations(cfg)) * int(cfg["splits"][a.split]["candidates"])
    blocks = [b for b in range(nb) if b % a.nparts == a.part]
    if a.reverse:
        blocks = blocks[::-1]
    t0, c = time.time(), [0, 0, 0]
    with ProcessPoolExecutor(a.procs, initializer=_init, initargs=(a.config, a.split)) as ex:
        for k, r in enumerate(ex.map(one, blocks, chunksize=1)):
            c[r] += 1
            if k % 500 == 0:
                print(f"{a.split} part {a.part}/{a.nparts}: {k}/{len(blocks)} 건너뜀 {c[0]} 성공 {c[1]} 실패 {c[2]} | {time.time() - t0:.0f}s", flush=True)
    print(f"끝 {a.split} part {a.part}: {len(blocks)} 장면 · 성공 {c[1]} · 실패 {c[2]} · 건너뜀 {c[0]} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
