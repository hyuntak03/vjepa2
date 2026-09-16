#!/usr/bin/env python3
"""v11 — 미래 8 튜블릿에서 물체 위치: 진실 vs context encoder z vs predictor p. 열 = visible (k=0) / early k=1 / mid k=1 / late k=1, 행 = flat / ramp.

y = 진행 방향 기준 물체 중심 x − 화면 중심 (px, 288 화면). 네 조건의 pos_a 궤적은 같다 (문맥 끝·시작·끝 중앙값 일치, 2026-09-14).
점선 = 가림막 출구 모서리 (clip 중앙값): 가려진 구간 한가운데 샘플의 물체 x + dir·occ_apparent_px/2. visible 은 가림막이 없어 선이 없다.
선 = clip 평균, 띠 = ± SD. 자 (학습셋 v5 attentive) 는 attention 이 퍼져도 값을 내므로 (기본값) 위치 주장은 attn_diag / emergence 표와 같이 읽는다.
출력: figures/<train>/v11_vanish/emergence/fig_position_k1.{png,pdf}, position_k1.json

  ROLLOUT2_TRAIN=v5 python z_research/scripts/figures/plot_v11_position_k1.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
import rollout2_test_readout as rt                                 # noqa: E402
from rollout2_attn_readout import AttnReadout                      # noqa: E402

META = Path("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv")
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
CACHE = Path("/local_datasets/world/world_analysis/cache")
SRC = {"p": (CACHE / "v11_full_vith", "predictor.npy", 8), "z": (CACHE / "v11_vanish_all_ctx32_vith", "isolated_ctx_0_32.npy", 16)}
OUT = rt.FIG_ROOT / "v11_vanish/emergence"
COLS = [("visible", "0", "visible (k=0)"), ("early", "1", "early occlusion, k=1"), ("mid", "1", "mid occlusion, k=1"), ("late", "1", "late occlusion, k=1")]
COND = {"flat": {"visible": "moving_visible_flat", "early": "moving_occlusion_flat_early", "mid": "moving_occlusion_flat_mid", "late": "moving_occlusion_flat"},
        "ramp": {"visible": "moving_visible", "early": "moving_occlusion_early", "mid": "moving_occlusion_mid", "late": "moving_occlusion"}}
COL = {"truth": "#222222", "z": "#2a78d6", "p": "#eb6834"}
LBL = {"truth": "truth", "z": "context encoder z (sees all frames) readout", "p": "predictor p (context only) readout"}


def main():
    if "--plot-only" in sys.argv:                                      # position_k1.json 에서 그림만 다시
        return plot(json.load(open(OUT / "position_k1.json")))
    idx = list(csv.DictReader(INDEX.open())); meta = {r["name"]: r for r in csv.DictReader(META.open())}
    models, arrays, rows = {}, {}, {}
    for rep, (c, f, _) in SRC.items():
        m = AttnReadout(); m.load_state_dict(torch.load(rt.RES_ROOT / f"attentive_pooling/{rep}/attn.pt", map_location="cpu")); m.eval(); models[rep] = m
        arrays[rep] = np.load(c / f, mmap_mode="r"); rows[rep] = {v: i for i, v in enumerate(json.loads((c / "meta.json").read_text())["video_ids"])}
    res = {}
    for motion in ("flat", "ramp"):
        for timing, K, _ in COLS:
            pos = [r for r in idx if r["condition"] == COND[motion][timing] and r["sym_k"] == K and r["violation_type"] == "vanish" and r["variant"] == "pos_a"]
            A = {"truth": [], "z": [], "p": [], "edge": []}
            for r in pos:
                mt = meta[r["video_id"]]; x = rt.arr(mt["object_px_x_by_sample"]); dirn = np.sign(x[31] - x[0])
                A["truth"].append((x.reshape(16, 2).mean(1)[8:] - 144.0) * dirn)
                if timing != "visible":
                    mid = (int(mt["hidden_start"]) // 3 + int(mt["hidden_end"]) // 3) // 2
                    A["edge"].append((x[mid] + dirn * float(mt["occ_apparent_px"]) / 2 - 144.0) * dirn)
                for rep, (_, _, n_tub) in SRC.items():
                    tok = torch.from_numpy(np.asarray(arrays[rep][rows[rep][r["video_id"]]], np.float32)).reshape(n_tub, 256, 1280)[-8:]
                    with torch.no_grad():
                        px = (models[rep](tok)[0].numpy() + 1) * 144.0
                    A[rep].append((px[:, 0] - 144.0) * dirn)
            cell = dict(n=len(pos), cond=COND[motion][timing], sym_k=K)
            for k in ("truth", "z", "p"):
                a = np.array(A[k]); cell[f"{k}_mean"] = a.mean(0).tolist(); cell[f"{k}_sd"] = a.std(0).tolist()
            if A["edge"]:
                cell["edge_median"] = float(np.median(A["edge"])); cell["edge_sd"] = float(np.std(A["edge"]))
            cell["truth_step"] = np.diff(cell["truth_mean"]).tolist()             # 튜블릿 간 이동량 (등가속이면 늘어난다)
            res[f"{motion} {timing}"] = cell
            print(f"{motion:4s} {timing:7s} k={K} n={len(pos):3d}  truth {np.round(cell['truth_mean']).astype(int).tolist()}  step {np.round(cell['truth_step'], 1).tolist()}"
                  + (f"  edge {cell['edge_median']:.0f}±{cell['edge_sd']:.0f}" if A["edge"] else ""), flush=True)
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / "position_k1.json", "w"), indent=1)
    plot(res)


def plot(res):
    plt.rcParams.update({"font.family": "Nimbus Roman", "pdf.fonttype": 42, "font.size": 8})
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9), sharex=True, sharey=True); t = np.arange(8); lab = iter("abcdefgh")
    for i, motion in enumerate(("flat", "ramp")):
        for j, (timing, K, title) in enumerate(COLS):
            ax = axes[i, j]; c = res[f"{motion} {timing}"]
            if "edge_median" in c:
                ax.axhline(c["edge_median"], color="k", lw=1.1, ls=":", label="occluder exit edge (median over clips)")
            for k in ("truth", "z", "p"):
                m, sd = np.array(c[f"{k}_mean"]), np.array(c[f"{k}_sd"])
                ax.plot(t, m, color=COL[k], lw=2.0 if k == "truth" else 1.6, label=LBL[k])
                ax.fill_between(t, m - sd, m + sd, color=COL[k], alpha=0.10, lw=0)
            ax.spines[["top", "right"]].set_visible(False); ax.set_xticks(t)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{motion}\nposition along travel (px)")
            ax.set_xlabel(("future tubelet\n" if i == 1 else "") + f"({next(lab)})")
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_position_k1.{ext}", dpi=200)
    print(f"→ {OUT}/fig_position_k1.png/.pdf, position_k1.json")


if __name__ == "__main__":
    main()
