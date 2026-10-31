"""VOBench_v1 프레임을 학습 로더와 같은 커널로 256² 로 줄여 NFS 에 복사한다 (2026-10-30, transfer_gap R4).

왜: VOBench 원본 (512² PNG, 197 GB) 은 vll5 로컬 /data2 에만 있다. vll5 GPU 가 막혔을 때 다른 노드 (vll3 등) 에서
    PFT 학습 · 채점을 돌리려고 256² 사본을 NFS 에 둔다.
동일성: app/vjepa_frozen/data.py ClipTransform (out_uint8=True, 증강 없음) 과 같은 연산
    — float/255 → F.interpolate(bilinear, align_corners=False, antialias 없음) → ×255 → round → clamp → uint8.
    그래서 학습 로더가 사본에서 읽는 uint8 은 원본에서 읽는 것과 비트 단위로 같다 (256 → 256 은 리사이즈를 건너뛴다).
    ⚠️ 채점 하네스 (evals/world_model_analysis/data.py) 는 float 경로라 uint8 반올림 ≤ 0.5/255 차이가 난다
    — 같은 사본으로 잰 모델끼리만 비교한다. window (zoom) 증강은 사본에서 업샘플이 된다.

입력: 원본 프레임 폴더 (clip 폴더마다 000000.png …).  출력: <dst>/<같은 상대 경로>/<같은 파일명>.png
재현:
  sbatch -p batch_vll -w vll5 --gres=gpu:0 -c 16 --mem=32G --time=0-04:00:00 \
    --wrap "/data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/data/transfer_gap_stage_vobench256.py \
            --src /data2/local_datasets/world/Benchmark/VOBench_v1 --dst <NFS 경로> --workers 16"
끝나면 <dst>/_DONE 에 clip 수 · 프레임 수를 적는다. 다시 돌리면 이미 있는 파일은 건너뛴다.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def clip_dirs(src):
    out = []
    for dp, dns, fns in os.walk(src):
        if any(f.endswith(".png") for f in fns):
            out.append(os.path.relpath(dp, src))
    return sorted(out)


def do_clip(args):
    src, dst, rel, res = args
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(1)
    sd, dd = os.path.join(src, rel), os.path.join(dst, rel)
    os.makedirs(dd, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(sd)):
        if not fn.endswith(".png"):
            continue
        op = os.path.join(dd, fn)
        if os.path.exists(op):
            n += 1
            continue
        a = np.asarray(Image.open(os.path.join(sd, fn)).convert("RGB"))
        x = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div_(255.0)[None]
        if tuple(x.shape[-2:]) != (res, res):
            x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
        y = x[0].mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        tmp = op + ".tmp"
        Image.fromarray(y).save(tmp, format="PNG", compress_level=1)
        os.replace(tmp, op)
        n += 1
    return rel, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rels = clip_dirs(a.src)
    print(f"{len(rels)} clip 폴더", flush=True)
    os.makedirs(a.dst, exist_ok=True)
    tot = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (rel, n) in enumerate(ex.map(do_clip, [(a.src, a.dst, r, a.res) for r in rels], chunksize=4)):
            tot += n
            if i % 200 == 0:
                print(f"{i}/{len(rels)} {rel} {n}", flush=True)
    with open(os.path.join(a.dst, "_DONE"), "w") as f:
        f.write(f"clips {len(rels)} frames {tot} res {a.res} src {a.src}\n")
    print(f"done clips {len(rels)} frames {tot}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
