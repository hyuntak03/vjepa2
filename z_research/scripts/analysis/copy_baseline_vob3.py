"""복사 기준선 — 마지막 문맥 블록 (블록 8) 의 target encoder 토큰을 미래 블록 9..16 에 그대로 놓는다 (예측 없음). 2026-11-06, tak's_story 요청.

VOBench v3 late · 판 가림막 · 물체 움직임 (기본 204 장면, --cell 로 바꿈). surprise = 미래 토큰 평균 |LN(복사) − LN(정답)| (채점기와 같은 L1).
변형 (정답 공간 × 복사 원천):
  window   : 정답 = target encoder 32 장 통째 통과의 블록 9..16 (우리 채점 · R1 채점과 같은 정답).
             복사 = 문맥 16 장만 넣은 통과의 블록 8 (= 인과 통과의 블록 8 — 통째 · 시간 순 문맥이 같다).
  causal   : 정답도 시간 순 — 미래 블록 f 토큰 = 프레임 1..2f 만 넣은 통과의 마지막 블록. 복사 = 같은 인과 블록 8.
  leaky    : 참고. 복사 = 32 장 통째 통과의 블록 8 (미래 프레임을 본 토큰 — 깨끗한 기준선이 아니다). 정답 = window.
판정 = matched pair (A: imp_ab > pos_a, B: imp_ba > pos_b), 같으면 0.5.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from scene_paired_tests import boards, scene_stats, summarize  # noqa: E402
from analysis.intphys2.model import build_from_config  # noqa: E402
from evals.world_model_analysis.data import WMADataset  # noqa: E402

S, C, NB = 256, 8, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="panel_moving", choices=["panel_moving", "panel_static", "mover_moving", "mover_static"])
    ap.add_argument("--out", default=os.path.join(ROOT, "z_analysis/results/release_ladder_20261106/copy_baseline_vob3.json"))
    a = ap.parse_args()
    occ, mov = a.cell.split("_")
    B = boards()
    idx = list(csv.DictReader(open(os.path.join(ROOT, "data_csv/vobench_v3/index.csv"))))
    mv = {r["block_id"] for r in idx if r["motion"] != "flat_static"}
    bn = "VOBench v3 late · panel" if occ == "panel" else "VOBench v3 late · moving occluder"
    blk = {b: x for b, x in B[bn][1].items() if (b in mv) == (mov == "moving")}
    vids = sorted({v for x in blk.values() for pr in x.values() for v in pr})
    cfg = yaml.safe_load(open(os.path.join(ROOT, "cvpr/results/eval/testbed/vobench_v3_vith/_resolved.yaml")))
    ds = WMADataset(cfg)
    pos = {r.video_id: i for i, r in enumerate(ds.records)}
    dev = torch.device("cuda:0")
    bundle = build_from_config(cfg["model"], dev)
    enc = bundle.target_encoder
    ln = lambda h: F.layer_norm(h.float(), (h.size(-1),))  # noqa: E731
    sur = {k: {} for k in ("window", "causal", "leaky")}
    bs = 8
    for s in range(0, len(vids), bs):
        vv = vids[s:s + bs]
        x = torch.stack([ds.clip(pos[v]) for v in vv]).to(dev, bundle.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            hw = ln(enc(x))                                        # (n, 16*S, D) 32 장 통째
            hc = ln(enc(x[:, :, :2 * C].contiguous()))[:, -S:]    # 문맥 16 장 통과의 블록 8 (= 인과 블록 8)
            fut_c = torch.cat([ln(enc(x[:, :, :2 * (f + 1)].contiguous()))[:, -S:] for f in range(C, NB)], 1)   # 인과 미래 블록 9..16
        tw = hw[:, C * S:]
        cp = hc.repeat(1, NB - C, 1)
        lk = hw[:, (C - 1) * S:C * S].repeat(1, NB - C, 1)
        for k, (p, t) in {"window": (cp, tw), "causal": (cp, fut_c), "leaky": (lk, tw)}.items():
            e = (p - t).abs().mean((1, 2)).cpu().tolist()
            sur[k].update(dict(zip(vv, e)))
        print(f"{s + len(vv)}/{len(vids)}", flush=True)
    res = {}
    for k, S_ in sur.items():
        st = summarize(scene_stats(S_, blk))
        res[k] = {"score": (st["A"] + st["B"]) / 2, "A": st["A"], "B": st["B"], "n_scenes": st["n"]}
        print(f"{k}: {res[k]['score']:.1f} (A {st['A']:.1f} / B {st['B']:.1f}), 장면 {st['n']}")
    json.dump({"cell": a.cell, "results": res}, open(a.out.replace(".json", f"_{a.cell}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
