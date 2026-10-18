"""가림 시점 하나 (--occ late · visible · seen = visible+early+mid), 운동별 — release vs release + VOBench post-FT (split v2 test block, 2026-10-14). plot_vobench.py fig_late_motion 과 같은 꼴.
패널 = 위반 3, x = 운동 6, 막대 = 쌍 정확도 [block 부트스트랩 95 %] (연한 = release, 진한 = +FT), primitive · realistic 합침.
--ar70: AR-state ep70 (vast, post-FT 없음) 막대를 더한다 (VOBench 전체 채점에서 같은 test block 만) → fig_<occ>_motion_pft_ar70.
        이때 색은 위반이 아니라 모델로 가른다 (release 파랑 · +FT 주황 · AR-state 초록).

  python z_research/scripts/figures/plot_vobench_pft_late_motion.py
  → z_research/VOBench_v1/vjepa2_pft/surprise/figures/fig_late_motion_pft.{pdf,png}
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis")); sys.path.insert(0, str(REPO / "z_research/scripts/figures"))
import vobench_split_compare as C  # noqa: E402
import plot_vobench as PV  # noqa: E402

OUT = REPO / "z_research/VOBench_v1/vjepa2_pft/surprise/figures"
MODELS = [("release", "release", 0.62), ("release+FT", "release + VOBench FT", 0.0)]
AR70 = ("ar_state_ep70", "AR-state ep70 (vast)", 0.35)
MCOL = {"release": "#2a78d6", "release+FT": "#eb6834", "ar_state_ep70": "#1baf7a"}   # --ar70 일 때 모델별 색


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--occ", default="late", help="late | visible | seen (= visible·early·mid)")
    ap.add_argument("--ar70", action="store_true", help="AR-state ep70 (vast) 막대 추가")
    ap.add_argument("--base", default="release", choices=["release", "vast"], help="vast = vast AR-state ep70 vs + VOBench post-FT (2026-10-14)")
    a = ap.parse_args()
    global OUT
    models = MODELS + [AR70] if a.ar70 else MODELS
    if a.base == "vast":
        models = [("ar_state_ep70", "vast AR-state ep70", 0.62), ("ar_state_ep70+FT", "+ VOBench FT", 0.0)]
        OUT = REPO / "z_research/VOBench_v1/vast_ar_state_ep70/postft_vobench/figures"
    OCCS = {"seen": ("visible", "early", "mid")}.get(a.occ, (a.occ,))
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    C.set_split("v2")
    rows = list(csv.DictReader(C.TEST.open())); by = defaultdict(dict)
    for r in rows: by[r["block_id"]][r["variant"]] = r
    C.RUNS["ar_state_ep70"] = C.EX / "surprise_c16t32__vobench_v1_vith_vast_ar_state_ep70"
    C.RUNS["ar_state_ep70+FT"] = REPO / "z_training/runs/vobench_vast_ar_state_postft/eval/surprise_c16t32__vobench_v1_split2_test_latest"
    S = {m: C.surprise(C.RUNS[m]) for m, _, _ in models}
    H = defaultdict(list)                                             # (model, viol, motion) → [(block, hit)]
    for b, v in by.items():
        r = v["pos_a"]
        if r["occlusion"] not in OCCS: continue
        for p, q in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            for m in S:
                sp, si = S[m][v[p]["video_id"]], S[m][v[q]["video_id"]]
                H[(m, r["violation"], r["motion"])].append((b, 1.0 if si > sp else 0.5 if si == sp else 0.0))
    rng = np.random.default_rng(0)

    def stat(pairs):
        g = defaultdict(list)
        for b, h in pairs: g[b].append(h)
        g = [np.array(x) for x in g.values()]
        bs = [np.mean(np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])) for _ in range(2000)]
        return dict(acc=100 * np.mean([h for _, h in pairs]), lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5), n=len(pairs))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
    W = 0.38 if len(models) == 2 else 0.27; js = {}
    for pi, (ax, vi) in enumerate(zip(axes, PV.VIOL)):
        PV.frame(ax, (-3, 112), (0, 25, 50, 75, 100))
        col = PV.VCOL[vi]; js[vi] = {}
        for j, (m, lab, f) in enumerate(models):
            st = [stat(H[(m, vi, mo)]) for mo in PV.MOT]; js[vi][m] = dict(zip(PV.MOT, st))
            xs = np.arange(len(PV.MOT)) + (j - (len(models) - 1) / 2) * W
            if a.ar70: c = ec = MCOL[m]
            else: c, ec = (PV.tint(col, f) if f else col), PV.tint(col, 0.25)
            ax.bar(xs, [s["acc"] for s in st], width=W * 0.92, color=c, edgecolor=ec, lw=0.6, zorder=3)
            ax.errorbar(xs, [s["acc"] for s in st], yerr=[[s["acc"] - s["lo"] for s in st], [s["hi"] - s["acc"] for s in st]],
                        fmt="none", ecolor=PV.INK2, elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=4)
            for x, s in zip(xs, st):
                ax.annotate(f"{s['acc']:.0f}", (x, 4.0), ha="center", va="bottom", fontsize=5.6 if len(models) == 2 else 5.0, color=PV.INK, zorder=7,
                            rotation=0 if len(models) == 2 else 90,
                            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.8) if len(models) > 2 else None)
        ax.axhline(50, color=PV.MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=2)
        ax.set_xticks(range(len(PV.MOT))); ax.set_xticklabels([PV.MLAB[x] for x in PV.MOT], fontsize=6.8, linespacing=0.95)
        ax.set_xlim(-0.6, len(PV.MOT) - 0.4); ax.tick_params(axis="x", length=0, pad=3)
        ax.set_title(PV.VLAB[vi], fontsize=9.0, color=PV.INK, pad=4); n_cell = st[0]["n"]
        if pi: ax.set_yticklabels([])
        else: ax.set_ylabel("pairwise acc. (%)", fontsize=9.0)
        ax.text(0.5, -0.30, PV.PANEL[pi], transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=PV.INK)
    if a.ar70:
        h = [Patch(facecolor=MCOL[m], edgecolor="none", label=l) for m, l in
             (("release", "release V-JEPA 2"), ("release+FT", "release + VOBench post-FT (possible clips only)"), ("ar_state_ep70", "AR-state ep70 (no post-FT)"))]
    else:
        h = [Patch(facecolor=PV.tint(PV.INK2, 0.62), edgecolor=PV.tint(PV.INK2, 0.25), lw=0.6, label="vast AR-state ep70" if a.base == "vast" else "release V-JEPA 2"),
             Patch(facecolor=PV.INK2, edgecolor=PV.tint(PV.INK2, 0.25), lw=0.6, label="+ VOBench post-FT (possible clips only)")]
    occ_lab = {"seen": "visible · early · mid"}.get(a.occ, a.occ)
    h.append(Patch(facecolor="none", edgecolor="none", label=f"occlusion: {occ_lab}   ({n_cell} pairs / motion, split v2 test)"))
    fig.legend(handles=h, loc="upper center", ncol=len(h) if len(h) == 3 else 2, frameon=False, fontsize=7.4, bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.85 if len(h) == 3 else 0.80, bottom=0.25, wspace=0.09)
    OUT.mkdir(parents=True, exist_ok=True); name = f"fig_{a.occ}_motion_pft" + ("_ar70" if a.ar70 else "")
    for ext in (".pdf", ".png"):
        fig.savefig((OUT / name).with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    json.dump(js, open(OUT / f"{name}.json", "w"), indent=1)
    print("saved →", OUT / name)


if __name__ == "__main__":
    main()
