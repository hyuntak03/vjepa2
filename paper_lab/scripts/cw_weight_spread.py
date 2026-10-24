#!/usr/bin/env python3
"""운반 가중 (정답 판 cw · 문맥 판 cw_ctx) 이 원래 공간 / 떨림 억제 64 공간에서 얼마나 퍼지나 (진행자 2, 2026-10-22, CPU).
진행자 1 질문: A2c (원래 공간 · cw_ctx) 가 AX 에서 매개변수 거리 0.245 만 움직였다 → 가중이 거의 1 로 고르게 나온 것인가.
VOBench 움직이는 물체 · 보임 clip 몇 개 + K400 val 몇 개를 CPU 로 인코딩해 가중 통계 (변동 계수 · 상위 10 % 칸 몫 · 최대/평균) 를 낸다.
  python paper_lab/scripts/cw_weight_spread.py --n 3
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for p_ in (str(REPO), str(REPO / "paper_lab"), str(REPO / "z_research/scripts/analysis")):
    sys.path.insert(0, p_)


def stats(w):  # w (n_blocks, S) 가중 (블록 안 평균 1)
    w = w.reshape(-1, w.shape[-1]).float()
    cv = (w.std(-1) / w.mean(-1)).mean().item()
    top = torch.topk(w, k=max(1, w.shape[-1] // 10), dim=-1).values.sum(-1) / w.sum(-1)
    return dict(cv=round(cv, 3), top10_share=round(top.mean().item(), 3), max_over_mean=round((w.max(-1).values / w.mean(-1)).mean().item(), 2))


def main(n):
    import yaml
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import encode_blocks
    from evals.world_model_analysis.data import WMADataset
    from method import patch as P
    import design_headroom as H
    torch.set_num_threads(32)
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    model = dict(ir.rp.MODEL, dtype="float32")
    bundle = build_from_config(model, torch.device("cpu"))
    cfg = yaml.safe_load(open(H.SETS["vobench"]["cfg"]))
    ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=model, features={"cache_dir": "/tmp"}, surprise={}))
    rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
    idx = [i for i, rec in enumerate(ds.records) if (r := rows.get(rec.video_id)) and r["occlusion"] == "visible"
           and r["motion"] != "flat_static" and r["variant"] == "pos_a"][:n]
    P.install({"mnf": {"calib": str(REPO / "paper_lab/_scratch/tspace/calib_r0.npz"), "k": 64}})
    S, C, R = 256, 8, 8
    out = {}
    with torch.no_grad():
        for j, i in enumerate(idx):
            x = ds.clip(i).unsqueeze(0).float()
            h = encode_blocks(bundle.target_encoder, x, 2)                         # (1, 16·S, D) 원래 공간
            mu, M = P._mnf_mats(h.device, h.size(-1))
            hm = torch.nn.functional.layer_norm(mu + (h - mu) @ M, (h.size(-1),))
            for name, X in (("raw", h), ("mnf64", hm)):
                P.CFG.clear(); P.CFG.update({"cw_ctx": {"ref": 4, "gamma": 1.0}, "cw": {"ref": 4, "gamma": 1.0}})
                wc_tf, wc_ar = P._cw_ctx_weights(X, S, C, R)
                wt_tf, wt_ar = P._cw_weights(X, S, C, R)
                for k, w in (("cw_ctx_tf", wc_tf), ("cw_ctx_ar", wc_ar), ("cw_tf", wt_tf), ("cw_ar", wt_ar)):
                    out.setdefault((name, k), []).append(stats(w.view(-1, S)))
            print(f"[{j + 1}/{len(idx)}] done", flush=True)
    print("\n공간 · 가중 판: 변동 계수 (std/mean) · 상위 10 % 칸의 가중 몫 (균등이면 0.10) · 최대/평균  [VOBench 보이는 움직임 clip 평균]")
    for (name, k), L in sorted(out.items()):
        print(f"  {name:6s} {k:10s} cv {np.mean([d['cv'] for d in L]):.3f} · top10 {np.mean([d['top10_share'] for d in L]):.3f} · max/mean {np.mean([d['max_over_mean'] for d in L]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3); a = ap.parse_args(); main(a.n)
