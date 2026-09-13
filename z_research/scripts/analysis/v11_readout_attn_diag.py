#!/usr/bin/env python3
"""v11 에서 p 자 (attentive readout) 의 attention 이 슬롯별로 어디에 실리는지 — "읽기가 마지막 관측에 머문다" 가 물체를 본 것인지 기본값인지 가른다.

슬롯 t 마다: 진실 물체 3×3 칸 질량 / 마지막 관측 위치 3×3 질량 / 가림막 3×3 질량 (균등이면 9/256 = 0.035),
읽기 − 진실 (px), 읽기 − 마지막 관측 (px), 읽기 − 균등 attention 읽기 Linear(토큰 평균) (px), 균등 읽기 − 마지막 관측 (px).
v11 은 물체가 경계 (샘플 15) 에 화면 가운데 오도록 설계돼 있어 **마지막 관측 위치 ≈ 화면 중심 ≈ 자의 기본 출력** 이다. 이걸 재려고 만든 진단.
출력: figures/<train>/v11_vanish/timing/attn_diag.{json,md}
  ROLLOUT2_TRAIN=v5 CUDA_VISIBLE_DEVICES=0 python z_research/scripts/analysis/v11_readout_attn_diag.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout2_test_readout as rt                       # noqa: E402
from rollout2_attn_readout import AttnReadout            # noqa: E402

ROOT = rt.ROOT; C = Path("/local_datasets/world/world_analysis/cache/v11_full_vith")
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"; META = Path("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv")
OUT = rt.FIG_ROOT / "v11_vanish/timing"
CONDS = [("flat", "k=0", "moving_visible_flat", "0"), ("flat", "early k=4", "moving_occlusion_flat_early", "4"), ("flat", "mid k=4", "moving_occlusion_flat_mid", "4"), ("flat", "late k=4", "moving_occlusion_flat", "4"),
         ("ramp", "k=0", "moving_visible", "0"), ("ramp", "early k=4", "moving_occlusion_early", "4"), ("ramp", "mid k=4", "moving_occlusion_mid", "4"), ("ramp", "late k=4", "moving_occlusion", "4")]
KEYS = ("obj_mass", "last_mass", "occ_mass", "d_truth", "d_last", "d_uniform", "uniform_vs_last")


def mass3(a, x, y):
    cx, cy = int(x // 18), int(y // 18); sel = [r * 16 + c for r in range(cy - 1, cy + 2) for c in range(cx - 1, cx + 2) if 0 <= r < 16 and 0 <= c < 16]
    return float(a[sel].sum())


def main():
    idx = list(csv.DictReader(INDEX.open())); meta = {r["name"]: r for r in csv.DictReader(META.open())}
    row = {v: i for i, v in enumerate(json.loads((C / "meta.json").read_text())["video_ids"])}; P = np.load(C / "predictor.npy", mmap_mode="r")
    model = AttnReadout().cuda(); model.load_state_dict(torch.load(rt.RES_ROOT / "attentive_pooling/p/attn.pt", map_location="cuda")); model.eval()
    res = {}
    for motion, tag, cond, K in CONDS:
        pos = [r for r in idx if r["condition"] == cond and r["sym_k"] == K and r["violation_type"] == "vanish" and r["variant"] == "pos_a"]
        M = {k: [] for k in KEYS}
        for r in pos:
            m = meta[r["video_id"]]; p = np.asarray(P[row[r["video_id"]]], np.float32).reshape(8, 256, -1)
            t32 = np.stack([rt.arr(m["object_px_x_by_sample"]), rt.arr(m["object_px_y_by_sample"])], -1); tub = t32.reshape(16, 2, 2).mean(1); last = tub[7]
            occ = t32[(int(m["hidden_start"]) // 3 + int(m["hidden_end"]) // 3) // 2] if K != "0" else None
            with torch.no_grad():
                pt = torch.from_numpy(p).cuda(); pr, a = model(pt); pr = (pr.cpu().numpy() + 1) * rt.RES; a = a.cpu().numpy()
                un = (model.out(pt.mean(1)).cpu().numpy() + 1) * rt.RES
            M["obj_mass"].append([mass3(a[t], *tub[8 + t]) for t in range(8)]); M["last_mass"].append([mass3(a[t], *last) for t in range(8)])
            M["occ_mass"].append([mass3(a[t], *occ) for t in range(8)] if occ is not None else [np.nan] * 8)
            M["d_truth"].append(np.linalg.norm(pr - tub[8:], axis=-1)); M["d_last"].append(np.linalg.norm(pr - last, axis=-1))
            M["d_uniform"].append(np.linalg.norm(pr - un, axis=-1)); M["uniform_vs_last"].append(np.linalg.norm(un - last, axis=-1))
        res[f"{motion} {tag}"] = {"n": len(pos), "cond": cond, **{k: [float(v) for v in np.nanmean(M[k], 0)] for k in KEYS}}
        print(f"{motion} {tag} ({cond}, n={len(pos)})"); [print(f"   {k:<16}", " ".join(f"{x:6.2f}" if k.endswith('mass') else f"{x:6.0f}" for x in res[f'{motion} {tag}'][k])) for k in KEYS]
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / "attn_diag.json", "w"), indent=1)
    md = ["# p 자 attention 진단 — 슬롯별 (v11, 학습셋 v5 자)", "", "3×3 질량: 균등이면 0.035. px 는 clip 평균. `uniform` = Linear(토큰 평균) = attention 이 균등일 때의 읽기.", ""]
    for k in KEYS:
        md += [f"## {k}", "", "| | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 |", "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {name} | " + " | ".join((f"{x:.2f}" if k.endswith("mass") else f"{x:.0f}") if not np.isnan(x) else "–" for x in r[k]) + " |" for name, r in res.items()]; md.append("")
    (OUT / "attn_diag.md").write_text("\n".join(md) + "\n"); print(f"→ {OUT}/attn_diag.{{json,md}}")


if __name__ == "__main__":
    main()
