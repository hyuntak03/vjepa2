#!/usr/bin/env python3
"""anchor 이식 그림 (2026-10-13, v2 — 사용자 피드백 "물체 중심 crop 이라 움직임이 안 보인다 → 고정 배경 + 잔상").

행 = 조건 (original · stop · reverse · 2x speed · removed · reverse at block 6). 열:
  (a) 예시 clip 한 장 그림 — 고정 배경 (32 장 화소 중앙값 = 물체 없는 배경) 위에
      · 문맥 블록 0–6 의 물체 = 잔상 (옅게, 시간 순으로 진해짐) · 문맥 블록 중심 = 회색 점
      · 조작한 블록의 물체 = 진하게 + 빨간 원
      · 참 미래 블록 중심 = 흰 점선 · 세 predictor 의 미래를 위치 자로 읽은 자리 = 색 점 (자가 '없음' 이라 한 칸은 뺌)
  (b) 전체 165 clip, 운동 방향 위치 (칸) 중앙값 · 사분위
입력 = z_research/VOBench_v1/bottleneck/anchor/{ar40,release,ar_state40}_r*.npz (predictor_anchor_test.py). 라벨 영어.

  python z_research/scripts/figures/plot_anchor_transplant.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import predictor_anchor_test as AT  # noqa: E402
import predictor_bottleneck as PB  # noqa: E402

A = REPO / "z_research/VOBench_v1/bottleneck/anchor"
FIG = REPO / "z_research/VOBench_v1/bottleneck/figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
MODELS = [("release", "V-JEPA 2 release", "#2a78d6"), ("ar40", "Ariel AR", "#eb6834"), ("ar_state40", "Ariel AR-state", "#1baf7a")]
ROWS = [("true", "original"), ("stay", "last block:\nstop"), ("back", "last block:\nreverse"), ("double", "last block:\n2x speed"),
        ("erase", "last block:\nobject removed"), ("back6", "block 6 only:\nreverse")]
MULT = {"stay": (0, 0), "back": (-1, -2), "double": (2, 4)}
C, K, CELL, SC = 8, 8, 18.0, 256 / 288


def load(name):
    Z = {}
    for f in sorted(A.glob(f"{name}_r*.npz")):
        z = np.load(f)
        for k in z.files: Z.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in Z.items()}


def main():
    D = {m: load(m) for m, _, _ in MODELS}
    rows = PB.select(1600, 1200)[1]
    R0 = D["ar40"]["row"]
    for m in D: assert (D[m]["row"] == R0).all()
    T = np.stack([PB.truth(rows, int(i))[0] for i in R0])                                  # (N, 16, 2) 288 px 블록 중심
    v = T[:, C - 1] - T[:, C - 2]; vh = v / np.linalg.norm(v, axis=1)[:, None]
    along = lambda xy, ref: ((xy - ref[:, None]) * vh[:, None]).sum(-1) / CELL
    S = {m: {c: along(D[m][f"{c}_xy"], T[:, C - 1]) for c, _ in ROWS} for m in D}
    PN = {m: {c: D[m][f"{c}_pn"] for c, _ in ROWS} for m in D}
    fut = along(T[:, C:], T[:, C - 1])

    # 예시: 오른쪽 운동, 세 모델이 true · stay · back 에서 물체를 안 놓치고, 중앙값에 가까운 clip
    keep = (vh[:, 0] > 0.95) & np.all([PN[m][c][:, :4].max(1) < 0.5 for m in D for c in ("true", "stay", "back")], axis=0)
    cand = np.nonzero(keep)[0]
    dev = [sum(np.abs(S[m][c][i] - np.median(S[m][c], 0)).mean() for m in D for c in ("true", "stay", "back", "double")) for i in cand]
    ex = int(cand[int(np.argmin(dev))]); r = rows[int(R0[ex])]
    print("example clip", r["video_id"])

    from importlib.util import module_from_spec, spec_from_file_location
    from evals.world_model_analysis.data import MEAN, STD, WMADataset
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    ds = WMADataset(dict(data=ir.rp.DATA, model=dict(ir.rp.MODEL), features={"cache_dir": "/tmp"}, surprise={}))
    x0 = ds.clip(int(R0[ex]))
    to_img = lambda x: (x * STD + MEAN).clamp(0, 1).permute(1, 2, 3, 0).numpy()               # (32, 256, 256, 3)
    orig = to_img(x0); bg = np.median(orig, 0)                                              # 물체 없는 배경
    Ps = np.stack([PB.arr(r["px_x_by_sample"]), PB.arr(r["px_y_by_sample"])], 1)            # 288 px, 샘플 32
    Tb = T[ex] * SC                                                                          # 블록 중심, 256 px
    allp = np.concatenate([Tb] + [D[m]["true_xy"][ex] * SC for m in D])
    xa, xb = int(max(allp[:, 0].min() - 34, 0)), int(min(allp[:, 0].max() + 34, 256))
    y0c = Tb[:, 1].mean(); ya, yb = int(max(y0c - 30, 0)), int(min(y0c + 58, 256))
    xa = int(max(min(xa, Tb[:, 0].min() - 34 - 6 * CELL * SC), 0))

    def composite(frames):
        comp = 0.55 * bg + 0.45                                                             # 배경을 옅게
        for t in (0, 6, 12):                                                                # 문맥 프레임 0 · 6 · 12 잔상 (3 블록 간격, 겹치지 않게)
            f = frames[t]; m = (np.abs(f - bg).sum(-1, keepdims=True) > 0.25).astype(np.float32)
            a = 0.25 + 0.25 * t / 12
            comp = comp * (1 - a * m) + f * a * m
        for f_ in (14, 15):                                                                 # 마지막 블록 진하게
            f = frames[f_]; m = (np.abs(f - bg).sum(-1, keepdims=True) > 0.25).astype(np.float32)
            comp = comp * (1 - 0.9 * m) + f * 0.9 * m
        return comp

    fig = plt.figure(figsize=(7.2, 1.2 * len(ROWS) + 0.3))
    gs = fig.add_gridspec(len(ROWS), 2, width_ratios=[1.75, 1.0], wspace=0.12, hspace=0.42, left=0.13, right=0.99, top=0.92, bottom=0.07)
    kk = np.arange(1, K + 1)
    for ri, (c, lab) in enumerate(ROWS):
        fr = to_img(AT.manipulate(x0, r, c))
        ax = fig.add_subplot(gs[ri, 0])
        ax.imshow(composite(fr)[ya:yb, xa:xb], extent=(xa, xb, yb, ya))
        ax.plot(Tb[:C - 1, 0], Tb[:C - 1, 1], "o", color="0.45", ms=2.2, zorder=3)          # 문맥 블록 0–6 중심
        if c in MULT:
            vv = Ps[13] - Ps[12]; pos = (Ps[13] + MULT[c][0] * vv + Ps[13] + MULT[c][1] * vv) / 2 * SC
            ax.plot(*pos, "o", mfc="none", mec="#d62728", ms=10, mew=1.3, zorder=4)
        elif c == "back6":
            vv = Ps[11] - Ps[10]; pos6 = (Ps[11] - vv + Ps[11] - 2 * vv) / 2 * SC
            ax.plot(*pos6, "o", mfc="none", mec="#d62728", ms=10, mew=1.3, zorder=4)
        elif c == "erase":
            ax.plot(*Tb[C - 1], "x", color="#d62728", ms=7, mew=1.5, zorder=4)
        ax.plot(Tb[C - 1:, 0], Tb[C - 1:, 1], "--", color="w", lw=1.0, zorder=3)
        # 미래: 운동 방향 위치만, 궤적 아래 모델별 가로 줄 (참 미래 줄 + 세 predictor 줄)
        x_of = lambda sv: Tb[C - 1, 0] + sv * CELL * SC
        lanes = [("true", fut[ex], "w", "|")] + [(m, S[m][c][ex], col, "o") for m, _, col in MODELS]
        for li, (nm, sv, col, mk) in enumerate(lanes):
            yl = y0c + 22 + 9 * li
            ax.plot([x_of(-7), x_of(sv[-1] if nm == "true" else 0)], [yl, yl], color="0.85", lw=0.4, zorder=2)
            okm = np.ones(K, bool) if nm == "true" else PN[nm][c][ex] < 0.5
            ax.plot(x_of(sv[okm]), np.full(okm.sum(), yl), mk + ("-" if mk == "o" else ""), color=col, ms=2.6 if mk == "o" else 5, lw=0.8, zorder=5)
            if okm.any():
                ax.plot(x_of(sv[okm][-1]), yl, mk, color=col, ms=4.5 if mk == "o" else 7, zorder=6)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(xa, xb); ax.set_ylim(yb, ya)
        ax.set_ylabel(lab, fontsize=6.8, rotation=0, ha="right", va="center", labelpad=6)
        if ri == 0: ax.set_title("one example clip (moving right): afterimages = context frames 0, 6, 12; solid = last block (frames 14-15)\nlanes below = future position along motion, k1-k8 (true, release, AR, AR-state; big dot = k8)", fontsize=6.3)
        ax = fig.add_subplot(gs[ri, 1])
        ax.plot(kk, np.median(fut, 0), "--", color="0.35", lw=1)
        for m, mlab, col in MODELS:
            s = S[m][c]; q1, q2, q3 = np.percentile(s, [25, 50, 75], axis=0)
            ax.fill_between(kk, q1, q3, color=col, alpha=0.15, lw=0); ax.plot(kk, q2, "-", color=col, lw=1.2)
        ax.axhline(0, color="0.85", lw=0.5); ax.set_ylim(-6, 6); ax.set_xlim(0.5, 8.5)
        ax.set_ylabel("cells along motion", fontsize=6)
        if c == "erase":
            ab = np.mean([(PN[m][c] > 0.5).mean(0)[-1] for m in D])
            ax.text(8.3, -5.6, f"decoder 'absent' at k8: {ab:.0%}", ha="right", va="bottom", fontsize=5.5, color="0.3")
        if ri == len(ROWS) - 1: ax.set_xlabel("future block k")
        if ri == 0: ax.set_title(f"all {len(R0)} clips (median, IQR)", fontsize=7)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=col, marker="o", ms=3, label=f"{mlab} (decoded future, k1-k8)") for _, mlab, col in MODELS]
    h += [Line2D([], [], color="0.5", marker="|", ls="--", ms=6, label="true future (path, k1-k8)"), Line2D([], [], color="0.45", marker="o", ls="", ms=3, label="context block centers"),
          Line2D([], [], color="#d62728", marker="o", mfc="none", ls="", ms=7, label="manipulated block")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6.2, bbox_to_anchor=(0.56, 1.0))
    fig.text(0.40, 0.01, "(a)", ha="center", fontsize=8); fig.text(0.83, 0.01, "(b)", ha="center", fontsize=8)
    out = FIG / "fig_anchor_transplant"
    fig.savefig(f"{out}.png", dpi=220); fig.savefig(f"{out}.pdf"); print("saved", out)


if __name__ == "__main__":
    main()
