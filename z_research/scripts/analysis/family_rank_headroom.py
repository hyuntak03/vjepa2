#!/usr/bin/env python3
"""장면 안 반사실 순위 — 학습된 predictor 가 같은 장면의 반사실 미래들 중 자기 미래를 이미 1 등으로 고르나 (2026-11-05, 학습 없음).

왜: '장면 안 반사실 순위 손실' (같은 장면 형제 영상의 미래를 음성으로, softmax) 을 제안하기 전에 그 손실이 학습 데이터에서 얼마나 살아 있나 (포화되지 않았나) 를 잰다.
무엇: 학습 목록 묶음 (families.json) 에서 묶음을 고른다. 영상마다 학습과 같은 창 (2 장 간격 32 장, 시작 0, 문맥 16 장 = 8 블록) 으로
      p_k = predictor(문맥 k), h_j = 형제 j 의 미래 정답 (채점기와 같은 공간 · LN) 을 만들고 S[k, j] = mean |p_k − h_j| (채점기 surprise).
      AR (블록별 정답) 은 미래 블록 정답이 그 블록 프레임만 보므로 S[k, j] 는 '문맥 k + 미래 j' 를 이어 붙인 불가능 영상의 surprise 와 같다.
보고: 행 k 마다 자기 미래 1 등 비율 (종류별: 단계 → 빈 장면 = 사라짐 짝, 빈 장면 → 단계 = 나타남 짝, 단계 → 다른 단계 = 바꿔치기 짝, 정지 단계),
      상대 margin (S[k, j] − S[k, k]) / S[k, k] 분포, 그리고 같은 공간의 복사 기준선 (마지막 문맥 블록 반복) — 연속성만으로 맞히는 정도.
  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_research/scripts/analysis/family_rank_headroom.py --model-cfg <채점 결과 폴더>/_resolved.yaml \
      --families data_csv/blender_track_v4a_negimp/families.json --n 120 --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
from analysis.intphys2.model import build_from_config  # noqa: E402
from analysis.intphys2.surprise import _context_target_indices  # noqa: E402
from evals.world_model_analysis.data import MEAN, STD  # noqa: E402


def read(path, start, n, stride):
    import decord
    vr = decord.VideoReader(path, num_threads=2)
    a = torch.from_numpy(vr.get_batch(list(range(start, start + n * stride, stride))).asnumpy())
    x = a.permute(3, 0, 1, 2).float() / 255.0
    return (x - MEAN) / STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cfg", required=True, help="채점 결과 폴더의 _resolved.yaml (model 절을 쓴다)")
    ap.add_argument("--families", required=True)
    ap.add_argument("--n", type=int, default=120, help="고를 묶음 수 (평면 trapdoor 를 먼저 절반, 나머지는 무작위)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.model_cfg))
    dev = torch.device("cuda")
    B = build_from_config(cfg["model"], dev)
    ac = torch.float16 if str(cfg["model"].get("autocast", "none")) in ("float16", "fp16") else None
    D = json.load(open(a.families))
    paths = [ln.strip() for ln in open(D["list"]) if ln.strip()]
    fams = D["families"]
    rng = random.Random(a.seed)
    td = [f for f in fams if f.get("occ") == "trapdoor" and f["plane"] == "flat"]
    rest = [f for f in fams if f not in td]
    pick = rng.sample(td, min(len(td), a.n // 2)) + rng.sample(rest, a.n - min(len(td), a.n // 2))
    N, stride, C = 32, 2, 16
    ts, spatial = 2, (256 // 16) ** 2
    ci, ti = _context_target_indices(ctx_frames=C, tgt_frames=N - C, tubelet_size=ts, spatial_tokens=spatial, batch_size=8, device=dev)
    rows = []
    for fi, f in enumerate(pick):
        ids = list(f["A"]) + [f["B"]]
        kinds = [("S" if (k == len(f["A"]) - 1 and f.get("occ") == "trapdoor" and f["plane"] == "flat" and len(f["A"]) == 5) else f"k{k}") for k in range(len(f["A"]))] + ["B"]
        x = torch.stack([read(paths[i], 0, N, stride) for i in ids]).to(dev)
        n = len(ids)
        with torch.no_grad(), (torch.autocast("cuda", dtype=ac) if ac else torch.no_grad()):
            z = B.context_encoder(x, masks=[ci[:n]])
            p = B.predictor(z, ci[:n], ti[:n], mask_index=0).float()
            hf = B.target_encoder(x)
            h = torch.gather(hf, 1, ti[:n].unsqueeze(-1).expand(-1, -1, hf.size(-1)))
            h = torch.nn.functional.layer_norm(h.float(), (h.size(-1),))
            hc = torch.nn.functional.layer_norm(hf[:, ci.size(1) - spatial: ci.size(1)].float(), (hf.size(-1),))
        S = (p[:, None] - h[None]).abs().mean(dim=(-1, -2)).cpu().numpy()                 # (k, j)
        pc = hc.repeat(1, h.size(1) // spatial, 1)
        Sc = (pc[:, None] - h[None]).abs().mean(dim=(-1, -2)).cpu().numpy()
        rows.append({"family": f["family"], "plane": f["plane"], "occ": f.get("occ"), "kinds": kinds, "hs": f["hs"], "S": S.tolist(), "S_copy": Sc.tolist()})
        if fi % 20 == 0:
            print(f"[{fi}/{len(pick)}] {f['plane']} {f.get('occ')} S diag {np.diag(S).round(3).tolist()}", flush=True)

    def summarize(key):
        out = {}
        for typ in ("vanish", "appear", "swap", "static_vanish", "static_swap", "all_rows"):
            win, mar = [], []
            for r in rows:
                S = np.array(r[key]); kd = r["kinds"]; n = len(kd); iB = n - 1
                for k in range(n):
                    for j in range(n):
                        if j == k:
                            continue
                        a_, b_ = kd[k], kd[j]
                        t = ("vanish" if b_ == "B" and a_.startswith("k") else "static_vanish" if b_ == "B" and a_ == "S" else
                             "appear" if a_ == "B" else "static_swap" if "S" in (a_, b_) else "swap")
                        if typ != "all_rows" and t != typ:
                            continue
                        win.append(S[k, j] > S[k, k]); mar.append((S[k, j] - S[k, k]) / S[k, k])
            if win:
                m = np.array(mar)
                out[typ] = {"n": len(win), "own_closer": round(100 * float(np.mean(win)), 1), "margin_median_pct": round(100 * float(np.median(m)), 2),
                            "margin_p10_pct": round(100 * float(np.percentile(m, 10)), 2), "margin_lt_1pct": round(100 * float(np.mean(m < 0.01)), 1)}
        top1 = [int(np.argmin(np.array(r[key])[k]) == k) for r in rows for k in range(len(r["kinds"]))]
        out["row_top1"] = round(100 * float(np.mean(top1)), 1)
        return out

    res = {"model_cfg": a.model_cfg, "predictor_checkpoint": cfg["model"].get("predictor_checkpoint"), "families": a.families, "n_families": len(rows),
           "model": summarize("S"), "copy": summarize("S_copy"), "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
