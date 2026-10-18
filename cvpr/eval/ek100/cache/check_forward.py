#!/usr/bin/env python3
"""캐시 경로 model forward 가 원래 wrapper 와 같은가 (2026-10-16).

val 첫 영상의 clip n 개를 엔진 val 로더로 읽어 원래 wrapper (영상 → encoder → predictor) 로 feats 를 내고,
같은 clip 의 캐시 행으로 CachedModel (캐시 → predictor) feats 를 낸다. 둘의 상대 L1 을 encoder 부분 · predictor 부분으로 따로 찍는다.
  CUDA_VISIBLE_DEVICES=<g> python cvpr/eval/ek100/cache/check_forward.py --config <_resolved.yaml> --cache <…/ek100_feats/<이름>> [--pred <predictor ckpt>]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).parent))
import extract as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--base", default=None)
    ap.add_argument("--pred", default=None); ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    from evals.action_anticipation_frozen.dataloader import init_data
    from evals.action_anticipation_frozen.models import init_module
    from evals.action_anticipation_frozen.cache import CachedModel
    cfg, data = X.load_cfg(a.config)
    if a.base: data["base_path"] = a.base
    ann, anos, rows, n_clip = X.clip_table(cfg, data, "val")
    paths, an = ann["val"]; vid = lambda q: q.split("/")[-1].split(".")[0]
    _, loader, _ = init_data(dataset=data["dataset"], training=False, base_path=data["base_path"], annotations_path=(paths[:2], {vid(q): an[vid(q)] for q in paths[:2]}),
                             batch_size=a.n, frames_per_clip=int(data["frames_per_clip"]), fps=int(data["frames_per_second"]),
                             anticipation_time_sec=data["anticipation_time_sec"], anticipation_point=data["val_anticipation_point"],
                             crop_size=int(data["resolution"]), crop_width=data.get("width"), spatial_mode=data.get("spatial_mode", "center_crop"),
                             anticipation_point_mode=data.get("anticipation_point_mode", "released"), time_source=data.get("time_source", "frame"),
                             world_size=1, rank=0, num_workers=0, pin_mem=False, persistent_workers=False)
    ub = next(iter(loader)); n = len(ub[0]); dev = torch.device("cuda:0")
    mk = cfg["model_kwargs"]; pk = dict(mk["pretrain_kwargs"])
    if a.pred: pk["predictor_checkpoint"] = a.pred
    model = init_module(module_name=mk["module_name"], frames_per_clip=int(data["frames_per_clip"]), frames_per_second=int(data["frames_per_second"]),
                        resolution=int(data["resolution"]), checkpoint=mk["checkpoint"], model_kwargs=pk, wrapper_kwargs=mk["wrapper_kwargs"], device=dev)
    kind = getattr(model.predictor, "ek_kind", "default"); print("predictor kind", kind)
    at = ub[3].to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):          # train_one_epoch · validate_exact 와 같은 autocast
        live = model(ub[0].to(dev), at).float()
    blocks = kind in ("ar", "ar_state")
    D = Path(a.cache) / "val"
    J = np.load(D / "joint.npy", mmap_mode="r"); Bk = np.load(D / "blocks.npy", mmap_mode="r") if blocks else None
    rr = list(range(n))                                                           # 첫 영상 clip 0..n-1 = 행 0..n-1 (val, view 0)
    x = torch.from_numpy(np.stack([J[r] for r in rr]))
    if blocks: x = torch.cat([x, torch.from_numpy(np.stack([Bk[r] for r in rr]))], 1)
    cm = CachedModel(model, n_tok=model.tokens_per_tubelet * (int(data["frames_per_clip"]) // model.tubelet_size)).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        got = cm(x.to(dev), at).float()
    N = cm.n_tok
    rel = lambda p, q: float((p - q).abs().mean() / q.abs().mean())
    print(f"shape live {tuple(live.shape)} cached {tuple(got.shape)}")
    print(f"encoder 부분 상대 L1 {rel(got[:, :N], live[:, :N]):.2e} · predictor 부분 {rel(got[:, N:], live[:, N:]):.2e}")
    pr = live[:, N:]; print(f"(참고: predictor 부분 clip 끼리 상대 L1 {rel(pr[1:], pr[:-1]):.2e})")


if __name__ == "__main__":
    main()
