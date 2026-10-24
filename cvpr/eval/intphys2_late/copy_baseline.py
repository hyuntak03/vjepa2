#!/usr/bin/env python3
"""intphys2_late 복사 기준선 (2026-10-22, intphys1_late/copy_baseline.py 와 같은 계산; 쌍은 index.csv 의 (block_id, pair_id) 에서)

원본 설명 — intphys1_late 복사 기준선 (2026-10-17) — 예측 없이 마지막 문맥 튜블릿 latent 를 미래 8 튜블릿 전부에 복사해 같은 matched 쌍을 채점한다.

cvpr/eval/testbed/copy_baseline.sh (te_v11_score.py) 와 같은 정의인데, 그쪽은 surprise_c16t32 의 프레임 간격 (stride 3) 을 검사해서
미리 골라 둔 32 장 (stride 1) 인 이 데이터에 못 쓴다. 그래서 같은 계산을 여기서 직접 한다:
  h      = LN(target_encoder(32 장))                       (B, 16 튜블릿, 256, D)   — 표준 채점의 타깃과 같다
  copy_z = LN(context_encoder(문맥 16 장))[튜블릿 7]  → 미래 8 칸에 복사  (te 의 copy: 문맥 encoder 출력)
  copy_h = h[튜블릿 7]                                 → 미래 8 칸에 복사  (te 의 copy_target)
  surprise = mean |copy − h[미래 8 튜블릿]|,  정답 = surprise(imp) > surprise(pos)
문맥 encoder 에 문맥 16 장만 넣는 것은 표준 채점의 ctx_masked (미래 토큰을 transformer 앞에서 버림) 와 같은 입력이다 (위치는 같고 미래 토큰이 없다).

  CUDA_VISIBLE_DEVICES=<g> python cvpr/eval/intphys2_late/copy_baseline.py [--model-run cvpr/results/eval/intphys2_late/intphys2_late_uni_vith]
  → <model-run>/copy_baseline.json  (copy_z · copy_h 의 쌍 정확도 + 영상별 surprise)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model-run", default=str(REPO / "cvpr/results/eval/intphys2_late/intphys2_late_uni_vith")); a = ap.parse_args()
    from analysis.intphys2.model import build_from_config
    from evals.world_model_analysis.data import WMADataset
    run = Path(a.model_run); cfg = yaml.safe_load((run / "_resolved.yaml").read_text())
    ds = WMADataset(cfg); dev = torch.device("cuda:0"); B = build_from_config(cfg["model"], dev)
    ctx = int(cfg["surprise"]["context_length"]); ts = B.tubelet_size; S = B.num_spatial_tokens; nc = ctx // ts
    ac = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(str(cfg["model"].get("autocast", "none")).lower())
    sur = {"copy_z": {}, "copy_h": {}}
    with torch.no_grad():
        for i, r in enumerate(ds.records):
            x = ds.clip(i)[None].to(dev, B.dtype)
            with (torch.autocast("cuda", dtype=ac) if ac else contextlib.nullcontext()):
                h = B.target_encoder(x).float(); z = B.context_encoder(x[:, :, :ctx]).float()
            D = h.size(-1); h = F.layer_norm(h, (D,)).view(1, -1, S, D); z = F.layer_norm(z, (D,)).view(1, -1, S, D)
            fut = h[:, nc:]
            sur["copy_z"][r.video_id] = float((z[:, nc - 1:nc] - fut).abs().mean())
            sur["copy_h"][r.video_id] = float((h[:, nc - 1:nc] - fut).abs().mean())
            if i % 40 == 0: print(f"{i}/{len(ds.records)}", flush=True)
    by = {}
    for r in csv.DictReader(open(Path(cfg["data"]["root"]) / cfg["data"]["index_csv"])):
        by.setdefault((r["block_id"], r["pair_id"]), {})["pos" if r["plausible"] == "1" else "imp"] = r
    pairs = [{"pos": v["pos"]["video_id"], "imp": v["imp"]["video_id"], "principle": v["pos"]["condition"], "group": v["pos"]["Difficulty"],
              "occluded": v["pos"]["occluded"], "d_src": v["pos"]["d_src"]} for v in by.values() if "pos" in v and "imp" in v]
    model = json.load(open(run / "per_block.json"))["per_video_surprise"]
    out = {"n_pairs": len(pairs), "definition": __doc__.split("\n")[5:9]}
    for k, sv in (("model", model), *sur.items()):
        c = [sv[p["imp"]] > sv[p["pos"]] for p in pairs]
        out[k] = {"acc": 100 * sum(c) / len(c), "by": {}}
        for col in ("principle", "group", "occluded", "d_src"):
            for v in sorted({p[col] for p in pairs}):
                cc = [sv[p["imp"]] > sv[p["pos"]] for p in pairs if p[col] == v]; out[k]["by"][f"{col}={v}"] = 100 * sum(cc) / len(cc)
    out["per_video"] = sur
    json.dump(out, open(run / "copy_baseline.json", "w"), indent=1)
    for k in ("model", "copy_z", "copy_h"): print(k, round(out[k]["acc"], 1), {kk: round(v, 1) for kk, v in out[k]["by"].items()})


if __name__ == "__main__":
    main()
