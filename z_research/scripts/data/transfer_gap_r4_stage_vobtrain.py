#!/usr/bin/env python3
"""transfer_gap R4 [X2] — VOBench 256² NFS 사본 중 **학습에 쓰는 clip 만** 먼저 만든다 (2026-10-30).

왜: 전체 사본 (job 235557, transfer_gap_stage_vobench256.py) 은 gravity 부터 6,272 clip 을 차례로 만들어 학습 clip 이
    늦게 온다. 같은 함수 (do_clip: 같은 커널 · 같은 파일명 · .tmp → os.replace) 를 학습 색인의 clip 에만 먼저 건다.
    두 프로세스가 같은 파일을 쓰면 같은 바이트이고 교체는 원자적이며, 이미 있는 파일은 건너뛴다 → 충돌 없음.
입력: data_csv/vobench_v1_split_v2g/index_train.csv (828 clip) [+ --test 면 split v2 test late 432 clip]
출력: z_pixel_decoder/runs/_frames/VOBench_v1_256full/<file_name>/*.png  (+ _tg4_train_DONE.json)
재현:
  sbatch -p batch_vll -w vll5 --gres=gpu:0 -c 8 --mem=16G --time=0-02:00:00 -o z_analysis/transfer_gap/logs/%x_%j.out -J tg4_stvob \
    --wrap "/data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/data/transfer_gap_r4_stage_vobtrain.py --workers 8 --test"
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/data"))
from transfer_gap_stage_vobench256 import do_clip  # noqa: E402

SRC = "/data2/local_datasets/world/Benchmark/VOBench_v1"
DST = str(REPO / "z_pixel_decoder/runs/_frames/VOBench_v1_256full")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--test", action="store_true", help="split v2 test 의 late 도 (학습 clip 뒤에)")
    ap.add_argument("--verify", action="store_true", help="만들지 않고 모든 PNG 를 열어 256² · CRC 를 확인 (동시 쓰기 사고 뒤 검사)")
    a = ap.parse_args()
    rels = [r["file_name"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2g/index_train.csv"))]
    if a.test:
        rels += [r["file_name"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))
                 if r["occlusion"] == "late"]
    print(f"{len(rels)} clip", flush=True)
    if a.verify:
        from concurrent.futures import ThreadPoolExecutor

        def chk(rel):
            from PIL import Image
            d = os.path.join(DST, rel)
            fs = sorted(f for f in os.listdir(d) if f.endswith(".png")) if os.path.isdir(d) else []
            bad = []
            for f in fs:
                try:
                    im = Image.open(os.path.join(d, f)); im.load()
                    if im.size != (256, 256):
                        bad.append(f)
                except Exception:
                    bad.append(f)
            return rel, len(fs), bad
        res = list(ThreadPoolExecutor(a.workers).map(chk, rels))
        rec = {"clips": len(rels), "frames": sum(r[1] for r in res), "clips_not_96": [r[0] for r in res if r[1] != 96][:20],
               "n_clips_not_96": sum(r[1] != 96 for r in res), "bad": [(r[0], r[2][:3]) for r in res if r[2]][:20],
               "n_bad_files": sum(len(r[2]) for r in res)}
        json.dump(rec, open(os.path.join(DST, "_tg4_verify.json"), "w"), indent=1)
        print("verify", {k: v for k, v in rec.items() if k.startswith("n_") or k in ("clips", "frames")}, flush=True)
        return
    tot = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (rel, n) in enumerate(ex.map(do_clip, [(SRC, DST, r, 256) for r in rels], chunksize=2)):
            tot += n
            if i % 100 == 0:
                print(f"{i}/{len(rels)} {rel} {n}", flush=True)
    json.dump({"clips": len(rels), "frames": tot, "test_late": a.test}, open(os.path.join(DST, "_tg4_train_DONE.json"), "w"))
    print("done", len(rels), tot, flush=True)


if __name__ == "__main__":
    main()
