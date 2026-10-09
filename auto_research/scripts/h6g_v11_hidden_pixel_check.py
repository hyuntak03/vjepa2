#!/usr/bin/env python3
"""H6g (v11 픽셀 검사) — late 의 '가려진 미래 프레임' 이 matched pair (pos / imp) 사이에서 픽셀 단위로 같은가.

왜: IntPhys 1 에서는 사건 전 튜블릿의 pos/imp 픽셀이 같아서 (인과 표적 Δ = 0 이 정확히 나온다) 표준 표적의 판별 = 표적 번짐으로
직접 읽을 수 있었다. v11 late 의 재등장 전 슬롯 (sym_k ≥ 2 에서 튜블릿 j < ⌊k/2⌋) 에서 같은 검정을 하려면 그 슬롯의 입력이
pos/imp 에서 같아야 한다. h6g_v11_cell_ci.py 에서 인과 표적 Δ 가 한 번도 0 이 아니었으므로 (배치 수치 잡음인지 렌더 차이인지)
원본 PNG 로 직접 확인한다.

공식: 프레임 f (raw 번호) 마다  D_f = |I_pos(f) − I_imp(f)| (uint8 RGB),  보고 = 동일 프레임 비율 (max D = 0),
  픽셀 중 D > 0 비율의 평균, max D.  f ∈ {45 (문맥 마지막, 대조: 같아야 한다)} ∪ {가려진 미래 프레임 48..} ∪ {재등장 프레임}.
표본: H6f 와 같은 block (block_id % 4 == 0) 의 late · sym_k ≥ 2 matched pair 에서 무작위 300 쌍 (seed 0).

  python auto_research/scripts/h6g_v11_hidden_pixel_check.py      (CPU, /local_datasets 에 v11 원본이 있는 노드: vll5 또는 vll6)
"""
from __future__ import annotations
import collections, csv, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IDX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
FR = Path("/local_datasets/world/world_analysis/IntPhysGen_v11")
OUT = ROOT / "auto_research/exp_results/h6"

R = [r for r in csv.DictReader(IDX.open()) if r["occ_timing"] == "late" and int(r["sym_k"]) >= 2 and int(r["block_id"]) % 4 == 0]
by = collections.defaultdict(dict)
for r in R:
    by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
keys = [k for k, d in by.items() if len(d) == 2]
rng = np.random.default_rng(0); rng.shuffle(keys)
res = collections.defaultdict(list)
for key in keys[:300]:
    d = by[key]; k = int(d["pos"]["sym_k"]); rev = int(d["pos"]["reveal_frame"])
    frames = [45] + [48 + 3 * i for i in range(2 * (k // 2))] + [rev]
    for raw in frames:
        a = np.asarray(Image.open(FR / d["pos"]["file_name"] / f"{raw:06d}.png")).astype(np.int16)
        b = np.asarray(Image.open(FR / d["imp"]["file_name"] / f"{raw:06d}.png")).astype(np.int16)
        D = np.abs(a - b)
        tag = "ctx_last(45)" if raw == 45 else ("future_hidden" if raw < rev else "reveal")
        for g in (f"{tag}|all", f"{tag}|{d['pos']['motion']}", f"{tag}|{d['pos']['violation_type']}"):
            res[g].append(((D > 0).mean(), int(D.max()), float(D.mean())))
out = {"_doc": "D = |I_pos − I_imp| (uint8). identical = max D == 0 인 프레임 비율. frac_px = D>0 픽셀 비율 평균. max = 최대 D. mean = D 평균.",
       "n_pairs": min(300, len(keys))}
for g, v in sorted(res.items()):
    v = np.array(v, dtype=float)
    out[g] = {"n_frames": len(v), "identical": float((v[:, 1] == 0).mean()), "frac_px": float(v[:, 0].mean()),
              "max": int(v[:, 1].max()), "median_max": float(np.median(v[:, 1])), "mean_abs": float(v[:, 2].mean())}
    print(f"{g:32s} " + "  ".join(f"{k} {x:.4g}" for k, x in out[g].items()))
OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / "h6g_v11_hidden_pixels.json", "w"), indent=1)
