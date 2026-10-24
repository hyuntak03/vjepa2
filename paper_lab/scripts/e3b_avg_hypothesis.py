#!/usr/bin/env python3
"""E3b (paper_lab 진행자 2, 2026-10-21) — E3 판정의 단위를 맞춘다 (R2 반박 대응).

E3 는 마지막 관측 바꿔치기에서 '블록 6 만 조작했을 때 미래 자리 변화 Δs (칸)' 를 잰다.
R2 지적: "≤ 0.1 칸이면 마지막 관측에 묶임" 은 변위 단위, "블록 6 몫 0.06–0.15 면 평균" 은 회귀 가중치 단위라
두 가설이 같은 Δs 를 예측할 수 있다. 그래서 '최근 블록 위주 같은 칸 가중 평균' 가설이 내는 Δs 를 학습 없이 만든다:
  예측 블록 (모든 걸음 같게) = Σ_b w_b · h_b,  h_b = 조작한 문맥의 블록 토큰 (표적 encoder, LN),
  w = 다른 팀 회귀 꼴 (마지막 블록 0.46, 앞 블록 0.15 · 0.10 · 0.08 · 0.06 · 0.05 · 0.03 · 0.03, 합으로 나눔).
  그리고 '마지막 블록만' (w = 마지막 블록 1) 예측도 같이.
판독: 같은 모델 (기준 블록 AR) 용 표적 토큰 판독기 (identity_vobprim_vast70/h, 중심 오차 0.41 칸).
Δs = 조작 − 원본 (같은 판독기 차이라 고정 치우침은 상쇄). E3 의 측정 Δs 를 이 두 예측과 같은 단위로 놓는다.

  python paper_lab/scripts/e3b_avg_hypothesis.py [--max-n 600]
산출: paper_lab/_scratch/e3/avg_hypothesis.json
"""
import argparse
import json
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
spec = spec_from_file_location("pat", REPO / "z_research/scripts/analysis/predictor_anchor_test.py")
pat = module_from_spec(spec); spec.loader.exec_module(pat)
PB = pat.PB
C, S, G = 8, 256, 16
W_AVG = np.array([0.03, 0.03, 0.05, 0.06, 0.08, 0.10, 0.15, 0.46]); W_AVG = W_AVG / W_AVG.sum()
CONDS = ("true", "paste", "stay", "back", "double", "stay6", "back6")
OUT = REPO / "paper_lab/_scratch/e3"


def main(max_n, bs):
    import torch
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import encode_blocks
    from evals.world_model_analysis.data import WMADataset
    dev = torch.device("cuda", 0); torch.cuda.set_device(0)
    spec2 = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec2); spec2.loader.exec_module(ir); rp = ir.rp
    model = dict(rp.MODEL)
    bundle = build_from_config(model, dev)
    Rh = ir.IdReadout().to(dev).eval()
    Rh.load_state_dict(torch.load(REPO / "z_research/VOBench_v1/readout/identity_vobprim_vast70/h/readout.pt", map_location=dev))
    ds = WMADataset(dict(data=rp.DATA, model=model, features={"cache_dir": "/tmp"}, surprise={}))
    keep, rows = pat.select(max_n)
    w = torch.tensor(W_AVG, device=dev, dtype=torch.float32).view(1, C, 1, 1)
    out = {h: {c: [] for c in CONDS} for h in ("avg", "last")}; t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, len(keep), bs):
            ii = keep[b0:b0 + bs]; base = [ds.clip(i) for i in ii]
            for c in CONDS:
                x = torch.stack([pat.manipulate(base[j], rows[i], c) for j, i in enumerate(ii)]).to(dev, dtype=bundle.dtype); B = x.size(0)
                hb = encode_blocks(bundle.target_encoder, x[:, :, :2 * C], 2).float().view(B, C, S, -1)
                for hyp, tok in (("avg", (w * hb).sum(1)), ("last", hb[:, C - 1])):
                    xy, _, _ = Rh(tok.half())
                    out[hyp][c].append(((xy.float() + 1) * 144).cpu().numpy())
            print(f"[e3b] {b0 + len(ii)}/{len(keep)} ({time.time() - t0:.0f}s)", flush=True)
    T = np.stack([PB.truth(rows, int(i))[0] for i in keep])               # (N, 16, 2) 288 px 블록 중심
    v = T[:, C - 1] - T[:, C - 2]; step = np.linalg.norm(v, axis=1); vh = v / step[:, None]
    res = {"n": len(keep), "block_step_cells": float(np.median(step) / 18), "weights": W_AVG.tolist()}
    for hyp in ("avg", "last"):
        XY = {c: np.concatenate(out[hyp][c]) for c in CONDS}
        s = {c: ((XY[c] - T[:, C - 1]) * vh).sum(-1) / 18.0 for c in CONDS}
        res[hyp] = {c: {"s_med": float(np.median(s[c])), "ds_med": float(np.median(s[c] - s["true"])),
                        "ds_iqr": np.percentile(s[c] - s["true"], [25, 75]).tolist()} for c in CONDS}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "avg_hypothesis.json", "w"), indent=1)
    print(f"\n학습 없는 두 가설의 Δs (칸, 판독기 = 표적 토큰 판 · 같은 판독기 차이) · 블록당 참 이동 {res['block_step_cells']:.2f} 칸 · n {res['n']}")
    for c in CONDS[1:]:
        print(f"  {c:7s} 가중 평균 {res['avg'][c]['ds_med']:+5.2f}  · 마지막 블록만 {res['last'][c]['ds_med']:+5.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--max-n", type=int, default=600); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args(); main(a.max_n, a.bs)
