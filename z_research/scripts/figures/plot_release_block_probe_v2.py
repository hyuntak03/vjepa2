#!/usr/bin/env python3
"""릴리즈 · VOBench_v2 · attentive 구간 공유 블록별 물체 판독 그림 — late (k 마다) · visible (2026-10-31, z_analysis/release/01_block_probe).

입력: figures/v2_attn/<occ>/results/<occ>_block_attn_v2_{shape,color,exist}.json (release_late_block_probe.py attn_occ)
출력: figures/v2_attn/late/late_k{1,2,3}_block_probe.{png,pdf} · figures/v2_attn/visible/visible_block_probe.{png,pdf}
  (a) context  : z 자기 판독 (블록 1–8)
  (b) prediction: p 자기 판독 (블록 9–16)
  (c) transfer : 블록 9–16. 문맥 encoder z 로 학습한 판독기를 predictor 출력 LN(p) 에 건 값 (파랑)
                 (h 판독기 선은 10-31 사용자 "h 는 빼도 돼" 로 그림에서 뺐다 — json 의 transfer.h2p · shared.h 에 남아 있다)
  행 = 과제 (shape · color · exist). 띠 = 설계 칸 재표집 95 % CI. 회색 = 가린 블록 (진함 = 두 샘플 모두, 옅음 = 한 샘플).
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/figures/plot_release_block_probe_v2.py --occ late   (· visible)
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "z_analysis/release/01_block_probe/figures/v2_attn"
C_Z, C_P, C_H = "#2a78d6", "#eb6834", "#1baf7a"
TASKS = [("shape", "shape acc (%)"), ("color", "color acc (%)"), ("exist", "object vs empty acc (%)")]
SAMPLES = {"0": "................|................", "1": "...............H|H...............",
           "2": "..............HH|HH..............", "3": ".............HHH|HHH............."}


def occ_blocks(k):
    s = SAMPLES[k].replace("|", "")
    return {b: (s[2 * b - 2] == "H") + (s[2 * b - 1] == "H") for b in range(1, 17) if "H" in s[2 * b - 2:2 * b]}


def line(ax, T, key_fn, blocks, k, color, ls, label):
    cells = [T[key_fn(b)]["by_k"][k] for b in blocks]
    ax.plot(blocks, [c["acc"] for c in cells], ls, marker="o", ms=3, lw=1.4, color=color, label=label)
    ax.fill_between(blocks, [c["ci"][0] for c in cells], [c["ci"][1] for c in cells], color=color, alpha=0.13, lw=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--occ", default="late", choices=["late", "visible"]); occ = ap.parse_args().occ
    d = BASE / occ
    R = {t: json.load(open(d / "results" / f"{occ}_block_attn_v2_{t}.json"))["tasks"][t] for t, _ in TASKS}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    ctx, fut = list(range(1, 9)), list(range(9, 17))
    for k in R["shape"]["ks"]:
        fig, axes = plt.subplots(len(TASKS), 3, figsize=(10.0, 6.8), sharey="row")
        shade = occ_blocks(k)
        for r, (task, ylab) in enumerate(TASKS):
            T = R[task]
            for c, blocks in enumerate((ctx, fut, fut)):
                ax = axes[r, c]
                for b, n in shade.items():
                    if b in blocks:
                        ax.axvspan(b - 0.5, b + 0.5, color="0.55", alpha=0.35 if n == 2 else 0.15, lw=0)
                ax.axhline(T["chance"], color="0.5", ls="--", lw=0.8)
                if c == 0:
                    line(ax, T["shared"], lambda b: f"z{b}", blocks, k, C_Z, "-", "z (context encoder)")
                elif c == 1:
                    line(ax, T["shared"], lambda b: f"p{b}", blocks, k, C_P, "-", "p (predictor)")
                else:
                    line(ax, T["transfer"], lambda b: f"z2p{b}", blocks, k, C_Z, "-", "z head -> p")
                ax.set_xticks(blocks); ax.set_xlim(blocks[0] - 0.5, blocks[-1] + 0.5); ax.set_ylim(0, 102); ax.grid(alpha=0.25, lw=0.5)
                if c == 0:
                    ax.set_ylabel(ylab)
                if r == len(TASKS) - 1:                                     # 있음/없음 행 아래쪽이 비어 있다
                    ax.legend(loc="lower left", fontsize=7, frameon=False)
                if r == len(TASKS) - 1:
                    ax.set_xlabel("block")
        n_sc, n_ex = R["shape"]["shared"]["z1"]["by_k"][k]["n"], R["exist"]["shared"]["z1"]["by_k"][k]["n"]
        title = f"{occ} k={k}" if occ == "late" else "visible (no occluder)"
        fig.suptitle(f"VOBench_v2 {title} · release ViT-H · attentive probe, shared per segment\n"
                     f"held-out clips: shape/color {n_sc}, exist {n_ex}", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        for c, lab in enumerate(("(a) context", "(b) prediction", "(c) z head -> p (transfer)")):
            pos = axes[-1, c].get_position()
            fig.text((pos.x0 + pos.x1) / 2, 0.012, lab, ha="center", va="bottom", fontsize=10)
        name = f"late_k{k}_block_probe" if occ == "late" else "visible_block_probe"
        for ext in ("png", "pdf"):
            fig.savefig(d / f"{name}.{ext}", dpi=200)
        plt.close(fig)
        print(d / f"{name}.png")


if __name__ == "__main__":
    main()
