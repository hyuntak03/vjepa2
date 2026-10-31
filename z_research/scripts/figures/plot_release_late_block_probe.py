#!/usr/bin/env python3
"""릴리즈 · VOBench late 블록별 물체 probing 그림 — k 마다 한 장 (2026-10-30, z_analysis/release).

입력 (기본 = v1 선형): 01_block_probe/results/v1/late_block_linear.json 의 shared (release_late_block_probe.py linear)
      = 블록 평균 토큰 · 선형 층 · 구간 (문맥 1–8 / 예측 9–16) 마다 모든 블록 행으로 학습, 블록마다 평가.
  --src attentive: 이전 판 late_block_probe_{shape,color,exist}.json (블록마다 attentive) → figures/_superseded/v1_attn_per_block/
  --src attn_v2: VOBench_v2 · attentive 구간 공유 late_block_attn_v2_<과제>.json → figures/v2_attn/ (★ 10-31 지금 판)
  --src linear_v2: VOBench_v2 (설계 칸마다 장면 3 개) late_block_linear_v2_<과제>.json → figures/v2_linear/
출력: 01_block_probe/figures/<판>/late_k{1,2,3}_block_probe.{png,pdf}
  열 (a) 문맥 블록 1–8: z (문맥 encoder = predictor 입력)
  열 (b) 예측 블록 9–16: p (predictor)
  (h 칸도 json 에 있지만 그리지 않는다 — 사용자 10-30 "z, p 만")
  행 = 과제 (shape · color · exist). 띠 = 4 중항 재표집 95 % CI (2-fold 라 모든 clip 이 test). 회색 = 가린 블록 (진함 = 두 샘플 모두, 옅음 = 한 샘플).
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/figures/plot_release_late_block_probe.py
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "z_analysis/release/01_block_probe/results"     # v1/ · v2/ (2026-10-31 폴더 정리)
FIG = REPO / "z_analysis/release/01_block_probe/figures"     # v2_attn/ · v2_linear/ · v1_linear/ · _superseded/v1_attn_per_block/
COL = {"z": "#2a78d6", "p": "#eb6834"}
NAME = {"z": "z (context encoder)", "p": "p (predictor)"}
TASKS = [("shape", "shape acc (%)"), ("color", "color acc (%)"), ("exist", "object vs empty acc (%)")]
SAMPLES = {"1": "...............H|H...............", "2": "..............HH|HH..............", "3": ".............HHH|HHH............."}


def occ_blocks(k):
    s = SAMPLES[k].replace("|", "")
    out = {}
    for b in range(1, 17):
        n = (s[2 * b - 2] == "H") + (s[2 * b - 1] == "H")
        if n:
            out[b] = n
    return out


def load(src):
    if src == "attentive":
        return {t: json.load(open(RES / "v1" / f"late_block_probe_{t}.json"))["tasks"][t] for t, _ in TASKS}, FIG / "_superseded" / "v1_attn_per_block", "attentive per block"
    if src == "attn_v2":
        L = {t: json.load(open(RES / "v2" / f"late_block_attn_v2_{t}.json"))["tasks"][t] for t, _ in TASKS}
        return {t: dict(chance=L[t]["chance"], cells=L[t]["shared"]) for t, _ in TASKS}, FIG / "v2_attn", "VOBench_v2 · attentive, shared per segment"
    if src == "linear_v2":
        L = {t: json.load(open(RES / "v2" / f"late_block_linear_v2_{t}.json"))["tasks"][t] for t, _ in TASKS}
        return {t: dict(chance=L[t]["chance"], cells=L[t]["shared"]) for t, _ in TASKS}, FIG / "v2_linear", "VOBench_v2 · mean-pool linear, shared per segment"
    L = json.load(open(RES / "v1" / "late_block_linear.json"))["tasks"]
    return {t: dict(chance=L[t]["chance"], cells=L[t]["shared"]) for t, _ in TASKS}, FIG / "v1_linear", "mean-pool linear, shared per segment"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default="linear", choices=["linear", "attentive", "linear_v2", "attn_v2"])
    src = ap.parse_args().src
    tasks, fig_dir, probe_name = load(src)
    pre = ""
    R = {"tasks": tasks}
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    for k in ("1", "2", "3"):
        fig, axes = plt.subplots(len(TASKS), 2, figsize=(7.0, 6.6), sharey="row")
        occ = occ_blocks(k)
        for r, (task, ylab) in enumerate(TASKS):
            T = R["tasks"][task]
            for c, (reps, blocks) in enumerate(((("z",), range(1, 9)), (("p",), range(9, 17)))):
                ax = axes[r, c]
                for b, n in occ.items():
                    if b in blocks:
                        ax.axvspan(b - 0.5, b + 0.5, color="0.55", alpha=0.35 if n == 2 else 0.15, lw=0)
                ax.axhline(T["chance"], color="0.5", ls="--", lw=0.8)
                for rep in reps:
                    xs = [b for b in blocks if f"{rep}{b}" in T["cells"]]
                    cell = [T["cells"][f"{rep}{b}"]["by_k"][k] for b in xs]
                    ax.plot(xs, [q["acc"] for q in cell], "-o", ms=3, lw=1.4, color=COL[rep], label=NAME[rep])
                    ax.fill_between(xs, [q["ci"][0] for q in cell], [q["ci"][1] for q in cell], color=COL[rep], alpha=0.15, lw=0)
                ax.set_xticks(list(blocks)); ax.set_xlim(blocks[0] - 0.5, blocks[-1] + 0.5); ax.set_ylim(0, 102)
                ax.grid(alpha=0.25, lw=0.5)
                if c == 0:
                    ax.set_ylabel(ylab)
                if r == 0:
                    ax.legend(loc="lower left", fontsize=7, frameon=False)
                if r == len(TASKS) - 1:
                    ax.set_xlabel("block")
        n = R["tasks"]["shape"]["cells"]["z1"]["by_k"][k]["n"]
        fig.suptitle(f"{'' if src.endswith('_v2') else 'VOBench '}late k={k} · release ViT-H · {probe_name}\nheld-out clips: shape/color {n}, exist "
                     f"{R['tasks']['exist']['cells']['z1']['by_k'][k]['n']}", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        for c, lab in enumerate(("(a) context", "(b) prediction")):
            pos = axes[-1, c].get_position()
            fig.text((pos.x0 + pos.x1) / 2, 0.012, lab, ha="center", va="bottom", fontsize=10)
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"{pre}late_k{k}_block_probe.{ext}", dpi=200)
        plt.close(fig)
        print(fig_dir / f"{pre}late_k{k}_block_probe.png")


if __name__ == "__main__":
    main()
