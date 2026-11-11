#!/usr/bin/env python3
"""곡률 손실 판 전용 런처 (2026-11-07). launch.py 와 같고 rank 마다 app/vjepa_frozen/curv.py 를 먼저 설치한다. 쓰는 곳 z_training/train_curv.sh."""
import os
import signal
import sys

import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch as L  # noqa: E402


def process_main(rank, fname, world_size, devices, smoke):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])
    from app.vjepa_frozen import curv
    curv.install(fname)
    L.process_main(rank, fname, world_size, devices, smoke)


if __name__ == "__main__":
    args = L.parser.parse_args()
    if args.debugmode:
        process_main(0, args.fname, 1, ["cuda:0"], False)
        sys.exit(0)
    n = len(args.devices)
    mp.set_start_method("spawn")
    procs = [mp.Process(target=process_main, args=(r, args.fname, n, args.devices, False)) for r in range(n)]
    for p in procs:
        p.start()

    def _term(signum, frame):
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=30)
            if p.exitcode is None:
                p.kill()
        sys.exit(128 + signum)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    with open(os.path.join(os.path.dirname(args.fname), "pids.txt"), "w") as f:
        f.write(" ".join(str(x) for x in [os.getpid()] + [p.pid for p in procs]) + "\n")
    bad, alive = [], list(range(n))                # 한 rank 가 죽으면 나머지를 바로 끝낸다 (launch.py 와 같다)
    while alive:
        for i in list(alive):
            procs[i].join(timeout=2)
            if procs[i].exitcode is None:
                continue
            alive.remove(i)
            if procs[i].exitcode != 0:
                bad.append((i, procs[i].exitcode))
        if bad and alive:
            print(f"[launch_curv] rank 실패 {bad} — 나머지 {alive} 종료", file=sys.stderr, flush=True)
            for i in alive:
                procs[i].terminate()
            for i in alive:
                procs[i].join(timeout=30)
                if procs[i].exitcode is None:
                    procs[i].kill()
            alive = []
    if bad:
        print(f"[launch_curv] rank 실패: {bad}", file=sys.stderr)
        sys.exit(1)
