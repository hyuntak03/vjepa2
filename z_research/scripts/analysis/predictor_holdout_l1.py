#!/usr/bin/env python3
"""Predictor_v1 held-out 예측 L1 — 가능 영상뿐이라 짝 정확도가 없는 bench 의 열 (4×4 전이 행렬, 2026-09-19).

Predictor_v1_training 은 train (14,250) · holdout (5,750) 모두 가능 영상 (plausible 1, pos_roll) 이다 → pairwise 채점 불가.
그래서 이 열은 **surprise_c16t32 와 같은 계산의 평균 L1** (낮을수록 좋음) 이다:
  evals/world_model_analysis/eval.py:run_surprise 를 그대로 부른다 (fp32 가중치 + fp16 autocast, 문맥 16 장 = encoder(online),
  target = target_encoder 32 장 → 미래 토큰만 affine-free LN, L1, mask_index 0, raw 100 장 stride 3 → 32 장, start 0).
  predictor 만 바꾼다 (model.predictor_checkpoint; 없으면 릴리즈).
표본: index_holdout.csv 에서 arm (line / occ / ramp) 층화 --n 개 (seed 0) → data_csv/predictor_v1_training/index_holdout_s<n>.csv
출력: z_training/runs/_matrix/predictor_v1_holdout_l1.{json,md}

  CUDA_VISIBLE_DEVICES=7 python z_research/scripts/analysis/predictor_holdout_l1.py [--n 1000]
"""
from __future__ import annotations
import argparse, csv, json, random, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import yaml                                                        # noqa: E402
from evals.world_model_analysis.data import WMADataset             # noqa: E402
from evals.world_model_analysis.eval import run_surprise           # noqa: E402

D = ROOT / "data_csv/predictor_v1_training"
OUT = ROOT / "z_training/runs/_matrix"
CKPT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"
PREDICTORS = [("release", None), ("v11_postft e10", "v11_postft/e10.pt"), ("predictor_v1_postft e15", "predictor_v1_postft/e15.pt"),
              ("intphys1_postft e40", "intphys1_postft/e40.pt"), ("intphys2_postft e80", "intphys2_postft/e80.pt")]


def sample(n, seed=0):
    rows = list(csv.DictReader(open(D / "index_holdout.csv"))); fields = list(rows[0].keys())
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    rng = random.Random(seed); out = []
    for arm in sorted(by):
        k = round(n * len(by[arm]) / len(rows)); out += rng.sample(by[arm], k)
    p = D / f"index_holdout_s{n}.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    return p.name, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--bs", type=int, default=16); a = ap.parse_args()
    name, rows = sample(a.n); arm = {r["video_id"]: r["arm"] for r in rows}
    proto = yaml.safe_load(open(ROOT / "configs/protocols/surprise_c16t32.yaml"))
    cfg = {"data": {**proto["data"], "root": str(D), "index_csv": name, "frames_root": "/local_datasets/world/world_analysis/Predictor_v1_training",
                    "frames_pattern": "{file_name}/{frame:06d}.png", "frames_start": 0, "frames_stride": 3, "block_column": "block_id",
                    "pair_column": "pair_id", "variant_column": "variant", "plausible_column": "plausible", "group_column": "condition"},
           "model": {**proto["model"], "checkpoint": CKPT, "arch_name": "vit_huge"},
           "surprise": {**proto["surprise"], "batch_size": a.bs, "decode_workers": 8}, "scoring": {}}
    ds = WMADataset(cfg); device = torch.device("cuda"); OUT.mkdir(parents=True, exist_ok=True); res = {}
    print(f"holdout 표본 {len(ds)} clip ({name})", flush=True)
    for lab, ck in PREDICTORS:
        c = json.loads(json.dumps(cfg))
        if ck:
            c["model"]["predictor_checkpoint"] = str(ROOT / "z_training/runs" / ck)
        t0 = time.time(); s = run_surprise(ds, c, device)
        v = {k: x["all"] for k, x in s.items()}
        res[lab] = {"ckpt": ck, "n": len(v), "mean_l1": float(np.mean(list(v.values()))),
                    "by_arm": {g: float(np.mean([x for k, x in v.items() if arm[k] == g])) for g in sorted(set(arm.values()))},
                    "per_video": v, "sec": time.time() - t0}
        print(f"{lab:26s} L1 {res[lab]['mean_l1']:.4f}  " + "  ".join(f"{g} {x:.4f}" for g, x in res[lab]["by_arm"].items()) + f"  ({res[lab]['sec']:.0f}s)", flush=True)
        json.dump({"index": name, "protocol": "surprise_c16t32 (run_surprise)", "results": res}, open(OUT / "predictor_v1_holdout_l1.json", "w"))
    base = res["release"]["mean_l1"]
    L = [f"# Predictor_v1 holdout 예측 L1 (낮을수록 좋음) — {len(ds)} clip, arm 층화, seed 0 (`{name}`)\n",
         "surprise_c16t32 와 같은 계산 (`run_surprise`), predictor 만 바꿈.\n", "| predictor | L1 | Δ vs 릴리즈 | " + " | ".join(res["release"]["by_arm"]) + " |",
         "|---|---:|---:|" + "---:|" * len(res["release"]["by_arm"])]
    for lab, r in res.items():
        L.append(f"| {lab} | {r['mean_l1']:.4f} | {r['mean_l1'] - base:+.4f} | " + " | ".join(f"{x:.4f}" for x in r["by_arm"].values()) + " |")
    L.append(f"\n## 재현\n\n```\nCUDA_VISIBLE_DEVICES=7 python z_research/scripts/analysis/predictor_holdout_l1.py --n {a.n}\n```")
    (OUT / "predictor_v1_holdout_l1.md").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
