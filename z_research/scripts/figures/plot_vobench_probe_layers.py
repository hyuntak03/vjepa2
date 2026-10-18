"""튜블릿 정체 probe — target encoder 마지막 층 vs 중간층 (16 / 32) 의 시간축 (2026-10-13).

h head (가림 없는 튜블릿으로 학습) 를 같은 층의 h 에 건 결과 (①+② 샘플) 를 가림 조건별로 t0–t15 에.
실선 = 정체 (모양/색 결합) 정답 %, 점선 = none 으로 답한 %. 회색 = 그 조건에서 실제로 가려진 튜블릿. 띠 = 시드 범위.

  python z_research/scripts/figures/plot_vobench_probe_layers.py
  → z_research/VOBench_v1/vjepa2/probing/figures/layers/fig_probe_layers.{png,pdf} · layers.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/figures"))
RES = REPO / "z_research/VOBench_v1/vjepa2/probing/results"
META = Path("/local_datasets/world/world_analysis/cache/vobench_v1_vith_tubeprobe/meta.json")
OUT = REPO / "z_research/VOBench_v1/vjepa2/probing/figures/layers"
LAYERS = {"last (32)": ["e35_s0_user", "e35_s1"], "mid (16)": ["L16_e35_s0", "L16_e35_s1"]}
CONDS = ("early", "mid", "late")
OBJ = ("primitive", "realistic")


def timeline(run, objset):
    """→ {occ: {t: [(correct, none, hidden)]}}"""
    r = json.load(open(RES / run / f"{objset}.json")); none = r["classes"].index("empty")
    M = {m["video_id"]: m for m in json.load(open(META))["meta"]}     # 샘플의 clip = video_id
    out = defaultdict(lambda: defaultdict(list))
    for cat in ("① h_vis · object", "② h_occ · object"):
        for clip, t, state, k, pr, gt in r["cats"][cat]["samples"]:
            occ = M[clip]["occlusion"]
            out[occ][int(t)].append((pr == gt, pr == none, state == "occ"))
    return out


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from plot_v11_surprise import BLUE, GREEN, MUTED, frame
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0), sharex=True, sharey=True)
    J = {}
    for ri, o in enumerate(OBJ):
        for ci, occ in enumerate(CONDS):
            ax = axes[ri, ci]; frame(ax, (-3, 104), (0, 25, 50, 75, 100))
            for layer, col in (("last (32)", GREEN), ("mid (16)", BLUE)):
                T = [timeline(run, o)[occ] for run in LAYERS[layer]]
                ts = sorted(set(T[0]))
                for mi, (met, ls) in enumerate((("acc", "-"), ("none", ":"))):
                    Y = np.array([[100 * np.mean([x[mi] for x in tl[t]]) for t in ts] for tl in T])
                    ax.fill_between(ts, Y.min(0), Y.max(0), color=col, alpha=0.15, lw=0)
                    ax.plot(ts, Y.mean(0), ls=ls, color=col, lw=1.3, label=f"{layer}: {'identity acc' if met == 'acc' else chr(39) + 'none' + chr(39) + ' chosen'}")
                    J[f"{o}/{occ}/{layer}/{met}"] = dict(t=ts, mean=Y.mean(0).round(1).tolist(), runs=Y.round(1).tolist())
                hid = [t for t in ts if np.mean([x[2] for x in T[0][t]]) > 0.5]
            for t in hid:
                ax.axvspan(t - 0.5, t + 0.5, color="0.88", lw=0, zorder=0)
            ax.axvline(7.5, color=MUTED, lw=0.8, ls="--")
            if ri == 0: ax.set_title(occ, fontsize=8.5)
            if ci == 0: ax.set_ylabel(f"{o}\naccuracy (%)", fontsize=8)
            ax.set_xticks([0, 4, 8, 12, 15]); ax.set_xticklabels(["t0", "t4", "t8", "t12", "t15"], fontsize=7)
            ax.text(0.5, -0.30 if ri else -0.14, f"({'abcdef'[ri * 3 + ci]})", transform=ax.transAxes, ha="center", fontsize=8)
    axes[0, 0].legend(fontsize=5.5, frameon=False, loc="center left")
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_probe_layers.png", dpi=200); fig.savefig(OUT / "fig_probe_layers.pdf"); plt.close(fig)
    json.dump(J, open(OUT / "layers.json", "w"), indent=1)
    print("saved →", OUT)


if __name__ == "__main__":
    main()
