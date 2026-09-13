#!/usr/bin/env python3
"""RollOut_v2 — 학습셋 자 (attentive) 로 읽은 p / z / h 위치의 요약 그림 두 장.

  figures/<train>/summary/fig_l2.png         시나리오별 L2 위치 오차 (토큰 한 칸 18 px = 1), p / z / h + echo 기준선
  figures/summary/fig_motion_gain.png  시나리오·축별 motion gain = clip 단위 (읽은 슬롯 0→7 변위) 를 (진실 변위) 에 회귀한 기울기. 1 = 진실만큼, 0 = 안 움직임. 숫자 = 상관.
  figures/summary/fig_motion_xy.png  시나리오별 x(t), y(t) 변위 (슬롯 0 기준, clip 평균 ± SD): 진실 vs p / z / h.
                                     x 는 진실의 진행 방향으로 부호를 맞춘다 (좌/우 clip 상쇄 방지).
                                     ledge / wall 은 불가능 미래 (부유 / 통과) 의 진실도 점선으로.
입력: exp_results/attentive_pooling/{p,z,h}/preds.npz (정규화 좌표 px/144−1), index_probe.csv (문맥 위치 = echo).

  python z_research/scripts/figures/plot_rollout2_summary.py
"""
from __future__ import annotations
import csv, re, collections, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rollout2_test_readout as rt                          # noqa: E402  (ROLLOUT2_TRAIN=v1|v2)
ROOT = rt.ROOT
E = rt.RES_ROOT / "attentive_pooling"; INDEX = ROOT / "data_csv/rollout_v2/index_probe.csv"
FIG = rt.FIG_ROOT / "summary"
RES, CELL, T = 144.0, 18.0, 8
SCEN = ["flat_v", "flat_a", "ramp_a", "arc", "fall", "ledge", "wall"]
COL = {"p": "#eb6834", "z": "#2a78d6", "h": "#1baf7a", "truth": "#222222", "echo": "#999999"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", s.strip()) if v])


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    idx = list(csv.DictReader(INDEX.open())); n = len(idx)
    px = np.stack([arr(r["px_x_by_sample"]) for r in idx]); py = np.stack([arr(r["px_y_by_sample"]) for r in idx])
    Lall = np.stack([px.reshape(n, 16, 2).mean(2), py.reshape(n, 16, 2).mean(2)], -1)                   # (n, 16, 2) px
    P = {r: np.load(E / r / "preds.npz") for r in ("p", "z", "h") if (E / r / "preds.npz").is_file()}   # z 가 없는 학습셋 (v3: 프레임 삭제) 도 그린다
    sc, pl, inf = P["p"]["scenario"], P["p"]["plausible"], P["p"]["in_frame"]; truth = (P["p"]["truth"] + 1) * RES
    pred = {r: (P[r]["pred"] + 1) * RES for r in P}
    blk = collections.defaultdict(dict)
    for i, r in enumerate(idx):
        blk[r["block_id"]][int(r["plausible"])] = i

    # ── fig 1: L2 — 한 패널, 시나리오마다 p / z / h 막대 셋. 사물 유무로 묶고 한 칸 (18 px ≈ 물체 반폭) 점선 하나 (2026-09-12) ──
    L2 = {r: np.array([np.linalg.norm(pred[r][k] - truth[k], axis=-1)[inf[k]].mean() / CELL for k in [(sc == s) & (pl == 1) for s in SCEN]]) for r in pred}
    NOPROP = ["flat_v", "flat_a", "arc", "fall"]; PROP = ["ramp_a", "ledge", "wall"]; order = NOPROP + PROP
    oi = [SCEN.index(s) for s in order]; xs = np.arange(len(order)); LAB = {"flat_v": "flat\nconst. v", "flat_a": "flat\nconst. a", "arc": "arc", "fall": "fall", "ramp_a": "ramp", "ledge": "ledge", "wall": "wall"}
    plt.rcParams.update({"font.family": "Nimbus Roman", "pdf.fonttype": 42, "font.size": 8})
    fig, ax = plt.subplots(figsize=(7.0, 2.4)); w = 0.26
    reps = [r for r in ("p", "z", "h") if r in L2]; LBL = {"p": "predictor p", "z": "context encoder z", "h": "target encoder h"}
    for j, r in enumerate(reps):
        ax.bar(xs + (j - (len(reps) - 1) / 2) * w, L2[r][oi], w, color=COL[r], label=LBL[r])
    ax.axhline(1, color="k", lw=0.6, ls=":"); ax.text(len(order) - 0.45, 1.04, "1 token (18 px, half the object width)", fontsize=6.5, ha="right", color="#444")
    ax.set_xticks(xs); ax.set_xticklabels([LAB[s] for s in order], fontsize=7.5); ax.set_ylabel("L2 position error (token cells)"); ax.set_ylim(0, 2.4)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, 0.95)); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_l2.png", dpi=200); fig.savefig(FIG / "fig_l2.pdf"); plt.close(fig)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    # ── fig 2: x(t), y(t) displacement ──
    fig, axes = plt.subplots(2, len(SCEN), figsize=(7.0 * 2, 4.6), sharex=True)
    t = np.arange(T)
    for c, s in enumerate(SCEN):
        k = np.where((sc == s) & (pl == 1))[0]
        # x 는 진행 방향으로 정렬 (좌/우 clip 이 평균에서 상쇄되지 않게): 진실 x 변위 부호를 곱한다. 정지 clip 은 +1
        sgn = np.sign(truth[k][:, 7, 0] - truth[k][:, 0, 0]); sgn[sgn == 0] = 1; SG = np.stack([sgn, np.ones_like(sgn)], -1)[:, None, :]   # (n,1,2)
        for a, ax in enumerate(axes[:, c]):
            d = (truth[k][:, :, a] - truth[k][:, :1, a]) * SG[:, :, a]; ax.plot(t, d.mean(0), color=COL["truth"], lw=2, label="truth (possible)")
            ax.fill_between(t, d.mean(0) - d.std(0), d.mean(0) + d.std(0), color=COL["truth"], alpha=0.08)
            if s in ("ledge", "wall"):
                ki = np.array([blk[idx[i]["block_id"]][0] for i in k]); di = (Lall[ki][:, 8:, a] - truth[k][:, :1, a]) * SG[:, :, a]
                ax.plot(t, di.mean(0), color=COL["truth"], lw=1.5, ls="--", label="truth (impossible: float / pass)")
            for r in [r for r in ("p", "z", "h") if r in P]:
                d = (pred[r][k][:, :, a] - pred[r][k][:, :1, a]) * SG[:, :, a]; ax.plot(t, d.mean(0), color=COL[r], lw=1.6, label=r)
                ax.fill_between(t, d.mean(0) - d.std(0), d.mean(0) + d.std(0), color=COL[r], alpha=0.10)
            ax.axhline(0, color="k", lw=0.4); ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel(("x displacement (px)\nalong travel direction" if a == 0 else "y displacement (px)\n+ = down"))
            if a == 0:
                ax.set_title(s)
            else:
                ax.set_xlabel("future tubelet")
    h, l = axes[0, -1].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(FIG / "fig_motion_xy.png", dpi=170); plt.close(fig)

    # ── fig 3: motion gain — clip 단위로 슬롯 0→7 변위를 진실에 회귀한 기울기 (1 = 진실만큼 움직임, 0 = 안 움직임) ──
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True); w = 0.25
    for a, ax in enumerate(axes):
        rows = [s for s in SCEN if (truth[(sc == s) & (pl == 1)][:, 7, a] - truth[(sc == s) & (pl == 1)][:, 0, a]).std() > 2]
        xs = np.arange(len(rows))
        for j, r in enumerate([r for r in ("p", "z", "h") if r in P]):
            g, cc = [], []
            for s in rows:
                k = (sc == s) & (pl == 1); dt = truth[k][:, 7, a] - truth[k][:, 0, a]; dr = pred[r][k][:, 7, a] - pred[r][k][:, 0, a]
                g.append(np.polyfit(dt, dr, 1)[0]); cc.append(np.corrcoef(dt, dr)[0, 1])
            ax.bar(xs + (j - 1) * w, g, w, color=COL[r], label=r)
            for x_, g_, c_ in zip(xs, g, cc):
                ax.text(x_ + (j - 1) * w, max(g_, 0) + 0.03, f"{c_:.2f}", ha="center", fontsize=5.5, color=COL[r])
        ax.axhline(1, color="k", lw=0.6, ls=":"); ax.axhline(0, color="k", lw=0.4)
        ax.set_xticks(xs); ax.set_xticklabels(rows, rotation=20); ax.set_title(f"{'xy'[a]} motion gain (slot 0→7)", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("slope of read vs true displacement"); axes[0].legend(fontsize=7, frameon=False, loc="upper right"); axes[0].set_ylim(-0.3, 2.2)
    fig.text(0.5, -0.02, "bars = per-clip regression slope (1 = reproduces the true motion); number = correlation. Axes with no true motion (flat y, fall x, wall) omitted.", ha="center", fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_motion_gain.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"→ {FIG}/fig_l2.png, fig_motion_xy.png, fig_motion_gain.png")


if __name__ == "__main__":
    main()
