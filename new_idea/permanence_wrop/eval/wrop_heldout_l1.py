#!/usr/bin/env python3
"""WROP held-out 예측 L1 — 계열별 (2026-10-31, permanence_wrop).

WROP 에는 불가능 영상이 없어 짝 정확도를 못 낸다. 대신 **채점과 같은 계산의 평균 L1** (낮을수록 좋음) 을 계열별로 낸다:
  문맥 16 장 → context encoder (online `encoder`, 미래 토큰은 transformer 앞에서 버림) → predictor → 미래 16 장 토큰,
  정답 = LN(target_encoder(32 장))[미래], L1. fp32 가중치 + bf16 autocast (IntPhys 2 채점 config 와 같다).
여러 predictor 를 **같은 배치 · 같은 encoder 출력**에 건다 (encoder 는 한 번만 돈다).
공간: 568×320 의 가운데 426×320 (화면비 1.33) → 256² bilinear (antialias 없음). 학습 RRC 창의 가장 큰 쪽과 같다.

표본: data_csv/wrop_uni32/heldout_manifest.tsv 에서 과제당 --per-task 개 (이름 순 앞쪽, 학습과 서로소인 WROP heldout 표본)
출력: new_idea/permanence_wrop/results/wrop_heldout_l1.json  (모델 × 과제 × 표본 L1, 계열 평균)

  CUDA_VISIBLE_DEVICES=4 python new_idea/permanence_wrop/eval/wrop_heldout_l1.py --per-task 20 \
      --models release B=z_training/runs/permanence_wrop_B/latest.pt C=... D=...
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from analysis.intphys2.model import build_from_config  # noqa: E402

CKPT = ROOT / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def read_clip(path: str) -> torch.Tensor:
    import decord
    vr = decord.VideoReader(path, num_threads=1)
    x = torch.from_numpy(vr.get_batch(list(range(32))).asnumpy())       # (32, H, W, 3)
    T, H, W, _ = x.shape
    w = min(W, int(round(H * 4 / 3)))
    left = (W - w) // 2
    x = x[:, :, left:left + w].permute(3, 0, 1, 2).float().div_(255.0)  # (3, T, H, w)
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    return (x - MEAN) / STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data_csv/wrop_uni32/heldout_manifest.tsv"))
    ap.add_argument("--per-task", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--models", nargs="+", default=["release"], help="release | 이름=ckpt 경로")
    ap.add_argument("--out", default=str(HERE.parent / "results/wrop_heldout_l1.json"))
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest), delimiter="\t"))
    by = {}
    for r in sorted(rows, key=lambda r: r["path"]):
        by.setdefault(r["task"], []).append(r)
    rows = [r for t in sorted(by) for r in by[t][: a.per_task]]
    print(f"표본 {len(rows)} (과제 {len(by)} × ≤{a.per_task})", flush=True)

    dev = torch.device("cuda")
    mcfg = {"checkpoint": str(CKPT), "arch_name": "vit_huge", "img_size": 256, "patch_size": 16, "tubelet_size": 2,
            "window_size": 32, "use_rope": True, "uniform_power": False, "dual_encoder": True,
            "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
            "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32"}
    bundle = build_from_config(mcfg, dev)
    preds = {}
    for spec in a.models:
        name, ck = (spec.split("=", 1) + [None])[:2] if "=" in spec else (spec, None)
        if ck is None:
            preds[name] = bundle.predictor
        else:
            c = copy.deepcopy(mcfg); c["predictor_checkpoint"] = str(ROOT / ck) if not os.path.isabs(ck) else ck
            b2 = build_from_config(c, dev)
            preds[name] = b2.predictor
            del b2.encoder, b2.context_encoder, b2.target_encoder
            torch.cuda.empty_cache()
    print("모델:", list(preds), flush=True)

    S = 256
    n_ctx, n_tgt = 8 * S, 8 * S
    res = {m: [] for m in preds}
    pool = ThreadPoolExecutor(8)
    t0 = time.time()
    batches = [rows[i:i + a.bs] for i in range(0, len(rows), a.bs)]
    nxt = pool.map(lambda b: torch.stack([read_clip(r["path"]) for r in b]), batches)
    for bi, (b, x) in enumerate(zip(batches, nxt)):
        x = x.to(dev, non_blocking=True)
        B = x.size(0)
        ctx_idx = torch.arange(n_ctx, device=dev).unsqueeze(0).expand(B, -1).contiguous()
        tgt_idx = torch.arange(n_ctx, n_ctx + n_tgt, device=dev).unsqueeze(0).expand(B, -1).contiguous()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = bundle.target_encoder(x)
            h = h[-1] if isinstance(h, list) else h
            h = F.layer_norm(h[:, n_ctx:].float(), (h.size(-1),))
            z = bundle.context_encoder(x, masks=[ctx_idx])
            z = z[-1] if isinstance(z, list) else z
            for m, p in preds.items():
                y = p(z, ctx_idx, tgt_idx, mask_index=0).float()
                l1 = (y - h).abs().mean(dim=(1, 2)).cpu().numpy()
                res[m] += [float(v) for v in l1]
        if bi % 20 == 0:
            print(f"  {bi + 1}/{len(batches)} {time.time() - t0:.0f}s", flush=True)

    fam = {}
    for i, r in enumerate(rows):
        fam.setdefault(r["family"], []).append(i)
    summ = {m: {f: float(np.mean([res[m][i] for i in ix])) for f, ix in sorted(fam.items())} for m in preds}
    for m in preds:
        summ[m]["all"] = float(np.mean(res[m]))
    out = {"rows": [{"task": r["task"], "family": r["family"], "path": r["path"]} for r in rows], "l1": res, "summary": summ,
           "per_task": a.per_task, "crop": "center 4:3 -> 256", "autocast": "bf16", "models": a.models}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"))
    fams = sorted(fam)
    print("model\t" + "\t".join(fams + ["all"]))
    for m in preds:
        print(m + "\t" + "\t".join(f"{summ[m][f]:.4f}" for f in fams + ["all"]))
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
