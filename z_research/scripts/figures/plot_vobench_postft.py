"""VOBench split test block — ar_state vs ar_state + VOBench post-FT (2026-10-12).
행 = objset, 열 = 위반, x = 가림 시점. 막대 = 쌍 정확도 [block 부트스트랩 95 %], 표식 = release (점) · AR 공간 복사 (가로선).

  python z_research/scripts/figures/plot_vobench_postft.py
  → z_research/VOBench_v1/ariel_ar_state_ep40/postft_vobench/figures/fig_postft_compare.{png,pdf}
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis")); sys.path.insert(0, str(REPO / "z_research/scripts/figures"))
import vobench_split_compare as C  # noqa: E402

VIOL, OCC = ("permanence", "shape", "color"), ("visible", "early", "mid", "late")


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="v1", choices=["v1", "v2"])
    ap.add_argument("--base", default="ar_state", help="FT 전 모델 (release · ar_state) — 막대 = <base> vs <base>+FT")
    a = ap.parse_args()
    C.set_split(a.split)
    base, ft = a.base, a.base + "+FT"
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from plot_v11_surprise import BLUE, ORANGE, MUTED, INK, frame
    rows = list(csv.DictReader(C.TEST.open())); by = defaultdict(dict)
    for r in rows: by[r["block_id"]][r["variant"]] = r
    S = {m: C.surprise(C.RUNS[m]) for m in dict.fromkeys(("release", base, ft))}
    if base == "release":                                            # 릴리즈 공간 복사 (LN(z) 마지막 문맥 튜블릿) — AR 공간 copy_blk 가 아니라
        import vobench_tables as VT
        S["copy"] = VT.copy_scores(C.EX / "copy__vobench_v1_vith")["copy"]
    else:
        S["copy"] = C.copy_blk()
    H = defaultdict(lambda: defaultdict(list))                       # (objset, viol, occ) → model → [(block, hit)]
    for b, v in by.items():
        r = v["pos_a"]
        for p, q in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            for m, s in S.items():
                sp, si = s[v[p]["video_id"]], s[v[q]["video_id"]]
                H[(r["objset"], r["violation"], r["occlusion"])][m].append((b, 1.0 if si > sp else 0.5 if si == sp else 0.0))
    rng = np.random.default_rng(0)

    def stat(pairs):
        blk = defaultdict(list)
        for b, h in pairs: blk[b].append(h)
        g = [np.array(x) for x in blk.values()]
        bs = [np.mean(np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])) for _ in range(2000)]
        return 100 * np.mean([h for _, h in pairs]), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0), sharey=True)
    x = np.arange(len(OCC)); w = 0.36
    for ri, o in enumerate(("primitive", "realistic")):
        for ci, vi in enumerate(VIOL):
            ax = axes[ri, ci]; frame(ax, (35, 104), (50, 75, 100))
            for j, (m, col, lab) in enumerate(((base, "#9ec5f0", base), (ft, BLUE, f"{base} + VOBench FT"))):
                st = np.array([stat(H[(o, vi, oc)][m]) for oc in OCC])
                xs = x + (j - 0.5) * w
                ax.bar(xs, st[:, 0], w * 0.92, color=col, label=lab)
                ax.errorbar(xs, st[:, 0], yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]], fmt="none", ecolor=INK, lw=0.7, capsize=1.5)
            rel = [stat(H[(o, vi, oc)]["release"])[0] for oc in OCC] if base != "release" else None; cp = [stat(H[(o, vi, oc)]["copy"])[0] for oc in OCC]
            if rel is not None: ax.scatter(x, rel, marker="o", s=14, color=ORANGE, zorder=4, label="release V-JEPA 2")
            ax.scatter(x, cp, marker="_", s=180, color=MUTED, linewidths=2, zorder=4, label="copy (release space)" if base == "release" else "copy (AR space)")
            ax.axhline(50, color=MUTED, lw=0.7, ls=":")
            ax.set_xticks(x); ax.set_xticklabels(OCC, fontsize=7)
            ax.set_title(f"{o} · {vi}", fontsize=8)
            if ci == 0: ax.set_ylabel("pair accuracy (%)", fontsize=8)
            ax.text(0.5, -0.24, f"({'abcdef'[ri * 3 + ci]})", transform=ax.transAxes, ha="center", fontsize=8)
    axes[0, 0].legend(fontsize=5.8, frameon=False, loc="lower left")
    fig.tight_layout()
    out = C.OUT / "figures"; out.mkdir(parents=True, exist_ok=True)
    tag = "" if a.split == "v1" else f"_{base}"
    fig.savefig(out / f"fig_postft_compare{tag}.png", dpi=200); fig.savefig(out / f"fig_postft_compare{tag}.pdf"); plt.close(fig)
    print("saved →", out)


if __name__ == "__main__":
    main()
