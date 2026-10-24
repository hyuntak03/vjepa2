#!/usr/bin/env python3
"""paper_lab method 학습 런처 (진행자 2, 2026-10-21). z_training/harness/launch.py 를 고치지 않고 감싼다:
rank 프로세스마다 paper_lab/method/patch.py 를 먼저 설치하고 원래 process_main 을 부른다.

  PL_METHOD='{"input": "mnf", "target": "mnf", "mnf": {"calib": ".../calib_r0.npz", "k": 256}}' \
  python paper_lab/method/launch.py --fname <run>/config.yaml --devices cuda:0 cuda:1
  (PL_METHOD 가 없으면 원래 학습과 같다 = 기준 팔)
설정은 run 폴더의 pl_method.json 에도 남긴다.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p_ in (str(REPO), str(REPO / "paper_lab")):
    if p_ not in sys.path: sys.path.insert(0, p_)

import z_training.harness.launch as L  # noqa: E402  (parser 와 process_main 만 쓴다)


def process_main(rank, fname, world_size, devices, smoke):
    cfg = json.loads(os.environ.get("PL_METHOD", "{}") or "{}")
    if not smoke:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])   # patch 의 GPU 텐서가 맞는 장치에 올라가게 (L.process_main 도 같은 값을 쓴다)
        from method import patch
        patch.install(cfg)
        if rank == 0:
            import yaml
            folder = Path(yaml.safe_load(open(fname))["folder"]); folder.mkdir(parents=True, exist_ok=True)
            json.dump(patch.split_cfg(cfg), open(folder / "pl_method.json", "w"), indent=1)   # 평가는 "infer" 만 (patch.install_eval)
            print(f"[paper_lab method] {cfg or '기준 (패치 없음과 같음)'}", flush=True)
    L.process_main(rank, fname, world_size, devices, smoke)


if __name__ == "__main__":
    import multiprocessing as mp
    import signal
    args = L.parser.parse_args()
    if not args.smoke_ddp and not args.fname:
        L.parser.error("--fname 이 필요하다 (또는 --smoke_ddp)")
    if args.debugmode:
        process_main(0, args.fname, 1, ["cuda:0"], args.smoke_ddp); sys.exit(0)
    n = len(args.devices); mp.set_start_method("spawn"); procs = []
    for rank in range(n):
        p = mp.Process(target=process_main, args=(rank, args.fname, n, args.devices, args.smoke_ddp)); p.start(); procs.append(p)

    def _term(signum, frame):
        for p in procs:
            if p.is_alive(): p.terminate()
        for p in procs:
            p.join(timeout=30)
            if p.exitcode is None: p.kill()
        sys.exit(128 + signum)
    signal.signal(signal.SIGTERM, _term); signal.signal(signal.SIGINT, _term)
    if args.fname:
        try:
            open(os.path.join(os.path.dirname(args.fname), "pids.txt"), "w").write(" ".join(str(x) for x in [os.getpid()] + [p.pid for p in procs]) + "\n")
        except OSError:
            pass
    bad, alive = [], list(range(n))
    while alive:                                                     # 원래 런처와 같은 규칙: 하나가 죽으면 나머지를 끝낸다
        for i in list(alive):
            procs[i].join(timeout=2)
            if procs[i].exitcode is None: continue
            alive.remove(i)
            if procs[i].exitcode != 0: bad.append((i, procs[i].exitcode))
        if bad and alive:
            for i in alive: procs[i].terminate()
            for i in alive:
                procs[i].join(timeout=30)
                if procs[i].exitcode is None: procs[i].kill()
            alive = []
    if bad:
        print(f"[launch] rank 실패: {bad}", file=sys.stderr); sys.exit(1)
