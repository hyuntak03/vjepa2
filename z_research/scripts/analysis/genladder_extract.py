#!/usr/bin/env python3
"""일반화 실패 원인 사다리 — 임의 index 의 h (미래 블록) · p (predictor 여럿) 추출 (2026-10-31, z_analysis/release/05_generalization).

정의 (릴리즈 ViT-H, probing 관례 bfloat16 · autocast none, release_v2_extract.py 와 같은 함수):
  h_fut[i] = LN(target_encoder(32 장))[블록 9..16]        (n, 8, 256, 1280) fp16 — 모든 clip
  p_<m>[i] = predictor_m(context_encoder(문맥 16 장))[블록 9..16]  (n, 8, 256, 1280) fp16 — 모든 clip (짝은 문맥이 같아 p 도 같다)
출력: <out>/{h_fut.npy, p_<m>.npy, meta.json}
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/genladder_extract.py --root data_csv/rubric_quads_uniform --frames /data/hyuntak/project/2026/2027_cvpr/nfs_cache/rubric_quads_frames/uniform \
     --stride 1 --start 0 --preds release=z_training/runs/release_vith/latest.pt,a0=...,a1=... --out /data/hyuntak/project/2026/2027_cvpr/nfs_cache/genladder/rubric_uniform
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
CFG = REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"
NF, S, D = 8, 256, 1280


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--index", default="index.csv"); ap.add_argument("--frames", required=True)
    ap.add_argument("--stride", type=int, default=3); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--preds", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from evals.world_model_analysis.data import WMADataset
    from evals.world_model_analysis.attn_probe import _ClipDS, _ac_dtype, _autocast
    from analysis.intphys2.model import build_from_config, _load_checkpoint
    from analysis.intphys2.surprise import _context_target_indices
    cfg = yaml.safe_load(open(CFG))
    cfg["data"].update(root=str(REPO / a.root) if not a.root.startswith("/") else a.root, index_csv=a.index, frames_root=a.frames,
                       frames_start=a.start, frames_stride=a.stride, n_frames=32)
    cfg["model"]["dtype"] = "bfloat16"; cfg["model"]["autocast"] = "none"; cfg["features"] = {"decode_workers": 6}
    ds = WMADataset(cfg)
    idx = list(range(len(ds.records)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    preds = dict(kv.split("=", 1) for kv in a.preds.split(","))
    dev = torch.device("cuda:0"); torch.set_num_threads(4)
    model = build_from_config(cfg["model"], dev)
    pm = {}
    for tag, path in preds.items():                                    # predictor 가중치를 tag 마다 메모리에 (같은 구조)
        sd = _load_checkpoint(str(REPO / path) if not path.startswith("/") else path)["predictor"]
        pm[tag] = {k.replace("module.", ""): v for k, v in sd.items()}
    H = np.lib.format.open_memmap(out / "h_fut.npy", "w+", np.float16, (len(idx), NF, S, D))
    P = {t: np.lib.format.open_memmap(out / f"p_{t}.npy", "w+", np.float16, (len(idx), NF, S, D)) for t in preds}
    bs = 4
    ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=bs, device=dev)
    loader = torch.utils.data.DataLoader(_ClipDS(cfg, idx), batch_size=bs, shuffle=False, num_workers=6, pin_memory=True)
    ac = _ac_dtype("none"); w, t0 = 0, time.time()
    with torch.inference_mode():
        for x in loader:
            b = x.size(0); x = x.to(dev, model.dtype, non_blocking=True)
            with _autocast(ac):
                h = model.target_encoder(x); h = h[-1] if isinstance(h, list) else h
                h = nn.functional.layer_norm(h, (h.size(-1),))
                z = model.context_encoder(x, masks=[ci[:b]]); z = z[-1] if isinstance(z, list) else z
                H[w:w + b] = h.float().cpu().numpy().reshape(b, 2 * NF, S, -1)[:, NF:].astype(np.float16)
                for t in preds:
                    model.predictor.load_state_dict(pm[t], strict=True)
                    p = model.predictor(z, ci[:b], ti[:b], mask_index=0)
                    P[t][w:w + b] = p.float().cpu().numpy().reshape(b, NF, S, -1).astype(np.float16)
            w += b
            print(f"{w}/{len(idx)} ({time.time() - t0:.0f}s)", flush=True)
    H.flush(); [v.flush() for v in P.values()]
    meta = [dict(video_id=r.video_id, **{k: r.raw.get(k, "") for k in ("block_id", "source_block", "variant", "plausible", "pair_id", "condition", "file_name")})
            for r in ds.records]
    json.dump(dict(meta=meta, preds=preds, root=a.root, frames=a.frames, stride=a.stride, start=a.start), open(out / "meta.json", "w"), indent=1)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
