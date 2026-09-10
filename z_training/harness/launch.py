#!/usr/bin/env python3
"""app/vjepa_frozen 전용 런처. app/main.py 를 **고치지 않고** 같은 흐름을 따로 둔다.

  python z_training/harness/launch.py --fname <run>/config.yaml --devices cuda:0 cuda:1 ...
  python z_training/harness/launch.py --fname <run>/config.yaml --debugmode          # 프로세스 1개, cuda:0
  python z_training/harness/launch.py --devices cuda:0 cuda:1 --smoke_ddp             # 모델 없이 spawn/포트/join 만 점검

app/main.py 와 다른 점 (그쪽은 그대로 둔다)
  * DDP 포트를 TRAIN_DDP_PORT (없으면 37129) 로, NCCL timeout 을 TRAIN_DDP_TIMEOUT_S 로 받는다.
    같은 노드에서 eval/학습이 함께 돌 때 뒤 job 의 rank0 이 bind 에 실패하고 조용히 world_size=1 로
    폴백해 두 job 이 섞이는 것을 막는다 (z_research/scripts/run.sh 가 evals 에 하는 것과 같다).
  * rank 프로세스를 join 해 exitcode 를 본다. start() 만 하면 rank 가 죽어도 부모가 exit 0 이라
    SLURM 이 COMPLETED 로 보고한다.
  * TRAIN_EXPECT_WS 가 있으면 world_size 가 다를 때 죽는다 (split-brain 가드).
  * app 은 config 의 `app:` 이 아니라 여기서 vjepa_frozen 으로 고정한다 (다른 app 은 app/main.py 로).
"""
import argparse
import datetime
import multiprocessing as mp
import os
import pprint
import sys
from pathlib import Path

import yaml

ROOT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, default=None, help="resolve_train.py 가 만든 config")
parser.add_argument("--devices", type=str, nargs="+", default=["cuda:0"])
parser.add_argument("--debugmode", action="store_true", help="프로세스를 새로 띄우지 않고 cuda:0 하나로 돈다")
parser.add_argument("--smoke_ddp", action="store_true", help="모델·데이터 없이 spawn/포트/all_reduce/join 만 점검")


def _init(rank, world_size):
    from src.utils.distributed import init_distributed
    port = int(os.environ.get("TRAIN_DDP_PORT", 37129))
    to = datetime.timedelta(seconds=int(os.environ.get("TRAIN_DDP_TIMEOUT_S", 7200)))
    ws, rk = init_distributed(port=port, timeout=to, rank_and_world_size=(rank, world_size))
    expect = os.environ.get("TRAIN_EXPECT_WS")
    if expect and int(expect) != ws:
        raise RuntimeError(f"world_size {ws} != TRAIN_EXPECT_WS {expect} — DDP 가 조용히 갈라졌다 (포트 {port} 충돌?)")
    return ws, rk


def process_main(rank, fname, world_size, devices, smoke):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])
    import logging
    from src.utils.logging import get_logger
    logger = get_logger(force=True)
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)

    if smoke:
        import torch
        ws, rk = _init(rank, world_size)
        t = torch.ones(1, device="cuda") * (rk + 1)
        if ws > 1:
            torch.distributed.all_reduce(t)
        expect = ws * (ws + 1) / 2
        ok = abs(float(t) - expect) < 1e-6
        print(f"[smoke_ddp] rank {rk}/{ws} device {devices[rank]} all_reduce={float(t):.0f} (expect {expect:.0f}) {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            sys.exit(2)
        return

    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = Path(params["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "params-train.yaml", "w") as f:
            yaml.dump(params, f)
    ws, rk = _init(rank, world_size)
    logger.info(f"Running... (rank: {rk}/{ws})")
    from app.vjepa_frozen.train import main as train_main
    train_main(args=params)


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.smoke_ddp and not args.fname:
        parser.error("--fname 이 필요하다 (또는 --smoke_ddp)")
    if args.debugmode:
        process_main(0, args.fname, 1, ["cuda:0"], args.smoke_ddp)
        sys.exit(0)
    n = len(args.devices)
    mp.set_start_method("spawn")
    procs = []
    for rank in range(n):
        p = mp.Process(target=process_main, args=(rank, args.fname, n, args.devices, args.smoke_ddp))
        p.start()
        procs.append(p)
    # rank 하나가 죽으면(nan, OOM, 예외) 나머지는 collective 에서 NCCL timeout(기본 2h)까지 기다린다.
    # 그래서 blocking join 대신 폴링하고, 실패를 보는 즉시 살아 있는 형제를 terminate 한다.
    bad = []
    alive = list(range(n))
    while alive:
        for i in list(alive):
            procs[i].join(timeout=2)
            if procs[i].exitcode is None:
                continue
            alive.remove(i)
            if procs[i].exitcode != 0:
                bad.append((i, procs[i].exitcode))
        if bad and alive:
            print(f"[launch] rank 실패 {bad} — 살아 있는 rank {alive} 를 종료한다", file=sys.stderr, flush=True)
            for i in alive:
                procs[i].terminate()
            for i in alive:
                procs[i].join(timeout=30)
                if procs[i].exitcode is None:
                    procs[i].kill()
            alive = []
    if bad:
        print(f"[launch] rank 실패: {bad} (rank, exitcode)", file=sys.stderr)
        sys.exit(1)
