#!/usr/bin/env python3
"""AR predictor rollout attention 그림 (2026-10-13, 사용자 요청 "과거 context 를 잘 attend 안 하는지 attention map 으로").

입력 = ar_attention_history.py 산출 (z_research/VOBench_v1/analysis/attention_map/<model>/<set>_r*.npz).
  fig_ar_attention_blocks : (a) vast AR-state 70 — 걸음 k (예측하는 미래 블록) × key 블록 질량 지도, 세트 셋
                            (b) 세 AR 모델 + 릴리즈 (2026-10-14, 검정) — 질량을 [문맥 앞 7 블록 · 문맥 마지막 블록 · 자기 생성 블록] 으로 묶어 걸음별
                            (c) 물체 query — key 블록 안 물체 자리 몫 (균등 9/256) : 마지막으로 보인 블록 vs 거울 대조
  fig_ar_attention_maps   : late 예시 clip 하나 — 물체 query (예측 대상 블록의 참 물체 자리) 의 attention 을 key 블록 프레임 위에
라벨 영어.

  python z_research/scripts/figures/plot_ar_attention.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import ar_attention_history as AH  # noqa: E402

AMD = REPO / "z_research/VOBench_v1/analysis/attention_map"; FIG = AMD / "compare"                    # 세 모델 비교 그림 → compare/, 한 모델 예시 → <model>/figures/
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
MODELS = [("vast_ar_state_ep70", "AR-state (vast, ep70)", "#1baf7a"), ("ariel_ar_ep40", "Ariel AR (ep40)", "#eb6834"), ("ariel_ar_state_ep40", "Ariel AR-state (ep40)", "#2a78d6"),
          ("vjepa2", "V-JEPA 2 release (one pass)", "0.15")]   # 릴리즈: key 16 블록 (미래 mask 포함) — 균등선이 다르다
SETS = [("vob_visible", "VOBench visible"), ("vob_late", "VOBench late occlusion"), ("ssv2", "SSv2 val")]
C, K, L, NBMAX = AH.C, AH.K, AH.L_, AH.NBMAX
MAIN = "vast_ar_state_ep70"


def load(s, m):
    try:
        return AH.load_all(s, m)
    except Exception:
        return None


def mass(Z):                                                       # (K, NBMAX) 층·head·clip 평균
    return np.nanmean(Z["bm_all"].astype(np.float32), axis=(0, 2, 3))


def groups(P):
    early = np.array([P[r, :C - 1].sum() for r in range(K)]); last = P[:, C - 1]
    own = np.array([P[r, C:C + r].sum() for r in range(K)])
    return early, last, own


def fig_blocks():
    D = {(s, m): load(s, m) for s, _ in SETS for m, _, _ in MODELS}
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.55, wspace=0.32, left=0.07, right=0.98, top=0.93, bottom=0.13)
    for j, (s, slab) in enumerate(SETS):
        ax = fig.add_subplot(gs[0, j]); Z = D[(s, MAIN)]
        if Z is None: ax.set_axis_off(); continue
        P = mass(Z); M = np.full_like(P, np.nan)
        for r in range(K): M[r, :C + r] = P[r, :C + r] * (C + r)          # 균등 대비 배수
        im = ax.imshow(np.log2(M), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        ax.axvline(C - 0.5, color="k", lw=0.8)
        ax.set_xticks(range(0, NBMAX, 2)); ax.set_yticks(range(K)); ax.set_yticklabels([f"k={r + 1}" for r in range(K)])
        ax.set_xlabel("key block (0-7 context, 8-14 own rollout)"); ax.set_title(f"{slab} (n={Z['bm_all'].shape[0]})", fontsize=7)
        if j == 0: ax.set_ylabel("predicting future block")
        for r in range(K):
            for b in range(C + r):
                if b in (0, C - 1) or b == C + r - 1:
                    ax.text(b, r, f"{P[r, b]:.2f}", ha="center", va="center", fontsize=4.2, color="k")
    cax = fig.add_axes([0.985, 0.57, 0.006, 0.33]); cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-3, -2, -1, 0, 1, 2, 3]); cb.set_ticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"]); cb.ax.tick_params(labelsize=5)
    cb.set_label("attention mass / uniform", fontsize=5.5)
    fig.text(0.5, 0.505, "(a) AR-state (vast, ep70): attention mass per key block, all queries, mean over 12 layers x 12 heads (numbers = raw mass)", ha="center", fontsize=6.5)

    # (b) 세 모델 · 묶은 질량
    ax = fig.add_subplot(gs[1, 0:2]); kk = np.arange(1, K + 1)
    for m, mlab, col in MODELS:
        for s, ls in (("vob_late", "-"), ("ssv2", ":")):
            Z = D[(s, m)]
            if Z is None: continue
            e, l_, o = groups(mass(Z))
            ax.plot(kk, e, ls, color=col, marker="o", ms=2.5, lw=1.1, label=f"{mlab}" if s == "vob_late" else None)
            ax.plot(kk, l_, ls, color=col, marker="s", ms=2.5, lw=0.8, alpha=0.7)
    ax.plot(kk, (C - 1) / (C + kk - 1), "--", color="0.5", lw=0.8)
    ax.text(K + 0.1, (C - 1) / (C + K - 1) + 0.01, "uniform, AR\n(blocks 0-6)", fontsize=5, color="0.4", va="bottom")
    ax.axhline((C - 1) / AH.NB, ls="--", color="0.15", lw=0.6)
    ax.text(K + 0.1, (C - 1) / AH.NB, "uniform, release\n(7 of 16 blocks)", fontsize=5, color="0.15", va="top")
    ax.set_xlabel("predicting future block k"); ax.set_ylabel("attention mass"); ax.set_xlim(0.7, K + 1.2); ax.set_ylim(0, 1)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=col, lw=1.1, label=mlab) for _, mlab, col in MODELS]
    h += [Line2D([], [], color="0.3", marker="o", ms=3, label="context blocks 0-6 (history)"), Line2D([], [], color="0.3", marker="s", ms=3, alpha=0.7, label="context block 7 (last observed)"),
          Line2D([], [], color="0.3", ls="-", label="VOBench late"), Line2D([], [], color="0.3", ls=":", label="SSv2")]
    ax.legend(handles=h, fontsize=5.2, ncol=2, frameon=False, loc="upper right")
    ax.text(0.5, -0.3, "(b) mass on history vs last observed block", transform=ax.transAxes, ha="center", fontsize=6.5)

    # (c) 같은 물체 · 같은 거리: 물체 query → 블록 6 의 물체 자리. visible (블록 7 에도 보임) vs late (블록 6 이 마지막으로 보인 블록)
    ax = fig.add_subplot(gs[1, 2]); w = 0.1; xs = np.arange(3)
    for mi, (m, mlab, col) in enumerate(MODELS):
        V, Lt = D[("vob_visible", m)], D[("vob_late", m)]
        if V is None or Lt is None: continue
        o_v, m_v = b6(V, None); o_l, m_l = b6(Lt, 6)
        x0 = xs + (mi - (len(MODELS) - 1) / 2) * 2.2 * w
        ax.bar(x0 - w / 2, o_v[:3] * 100, w, color=col, label=mlab); ax.bar(x0 + w / 2, o_l[:3] * 100, w, color=col, alpha=0.45, hatch="////", ec="w", lw=0)
        ax.plot(x0 - w / 2, m_v[:3] * 100, "_", color="k", ms=5); ax.plot(x0 + w / 2, m_l[:3] * 100, "_", color="k", ms=5)
    ax.set_xticks(xs); ax.set_xticklabels([f"k={r + 1}" for r in range(3)]); ax.set_ylabel("attention on object region (%)")
    from matplotlib.patches import Patch
    h = [Patch(color="0.4", label="visible (object also in block 7)"), Patch(fc="0.4", alpha=0.45, hatch="////", ec="w", label="late (block 6 = last seen)"),
         Line2D([], [], color="k", marker="_", ls="", ms=6, label="mirror region (same distance)")]
    ax.legend(handles=h, fontsize=5, frameon=False, loc="upper right")
    ax.text(0.5, -0.3, "(c) object query -> object in context block 6", transform=ax.transAxes, ha="center", fontsize=6.5)
    FIG.mkdir(parents=True, exist_ok=True); out = FIG / "fig_ar_attention_blocks"
    fig.savefig(f"{out}.png", dpi=230); fig.savefig(f"{out}.pdf"); print("saved", out)


def b6(Z, want_lv):
    """물체 query (블록 C+r 의 참 물체 자리) 가 문맥 블록 6 의 물체 자리 / 거울 자리에 준 질량, 걸음별. want_lv = 6 이면 late 중 블록 6 이 마지막으로 보인 clip 만."""
    sel = np.ones(len(Z["hid"]), bool)
    if want_lv is not None:
        lv = np.array([max([b for b in range(C) if not h[b]], default=-1) for h in Z["hid"]]); sel = lv == want_lv
    oo = np.nanmean(Z["om_obj"].astype(np.float32)[sel][:, :, :, :, 6], axis=(2, 3)); om = np.nanmean(Z["om_mir"].astype(np.float32)[sel][:, :, :, :, 6], axis=(2, 3))
    return np.nanmean(oo, 0), np.nanmean(om, 0)


def fig_maps(setname="vob_late", ex_i=0, steps=(0, 3, 7), keys=(4, 5, 6, 7)):
    import csv

    import yaml
    import design_headroom as Hm
    from evals.world_model_analysis.data import MEAN, STD, WMADataset
    Z = load(setname, MAIN)
    item = int(Z["ex_item"][ex_i]); A = Z["ex_A"][ex_i].astype(np.float32)          # (K, L, 2, NBMAX*S)
    yx = Z["ex_yx"][ex_i]; hid = Z["ex_hid"][ex_i]
    cfg = yaml.safe_load(open(Hm.SETS["vobench"]["cfg"]))
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=dict(ir.rp.MODEL), features={"cache_dir": "/tmp"}, surprise={}))
    x = ds.clip(item); fr = (x * STD + MEAN).clamp(0, 1).permute(1, 2, 3, 0).numpy()
    print("example", ds.records[item].video_id)
    cols = list(keys) + ["own"]
    fig, axs = plt.subplots(len(steps), len(cols) + 1, figsize=(7.2, 1.35 * len(steps) + 0.4), gridspec_kw=dict(wspace=0.05, hspace=0.25))
    for i, r in enumerate(steps):
        am = A[r, :, 1].mean(0)                                                       # 층 평균, 물체 query
        nb = C + r
        blk = am[:nb * 256].reshape(nb, 16, 16)
        vmax = np.percentile(blk[list(keys) + [nb - 1]], 99.5)
        bg = np.median(fr, 0)
        for j, b in enumerate(cols):
            ax = axs[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(("own rollout block\n(newest)" if b == "own" else f"context block {b}"), fontsize=6)
            if j == 0: ax.set_ylabel(f"predicting k={r + 1}", fontsize=6)
            if b == "own" and r == 0:
                ax.text(0.5, 0.5, "no own block yet", ha="center", va="center", fontsize=6, transform=ax.transAxes); ax.set_facecolor("0.92"); continue
            bb = nb - 1 if b == "own" else b
            f = bg if b == "own" else fr[2 * bb + 1]                                  # 자기 블록은 픽셀이 없다 → 물체 없는 배경 위에
            rgba = plt.get_cmap("plasma")(np.clip(blk[bb] / vmax, 0, 1)); rgba[..., 3] = 0.85 * np.clip(blk[bb] / vmax, 0, 1)
            ax.imshow(0.55 * f + 0.1, extent=(0, 16, 16, 0)); ax.imshow(rgba, extent=(0, 16, 16, 0), interpolation="nearest")
            if b != "own":
                y, x_ = yx[bb]; ax.add_patch(plt.Rectangle((x_ - 1, y - 1), 3, 3, fill=False, ec="c" if not hid[bb] else "w", lw=0.9, ls="-" if not hid[bb] else ":"))
            yq, xq = yx[C + r]; ax.add_patch(plt.Rectangle((xq - 1, yq - 1), 3, 3, fill=False, ec="lime", lw=0.7, ls="--"))
            ax.text(0.3, 15.5, f"{blk[bb].sum():.2f}", color="w", fontsize=5.5)
        ax = axs[i, -1]; ax.bar(range(nb), blk.sum((1, 2)), color=["0.6"] * (C - 1) + ["0.3"] + ["#1baf7a"] * (nb - C)); ax.set_ylim(0, 1)
        ax.set_xticks([0, C - 1, nb - 1]); ax.tick_params(labelsize=5); ax.yaxis.tick_right()
        if i == 0: ax.set_title("mass per key block", fontsize=6)
    d_ = AMD / MAIN / "figures/_superseded"; d_.mkdir(parents=True, exist_ok=True); out = d_ / "fig_ar_attention_maps"   # 옛 예시판 (rows 그림이 대체)
    fig.savefig(f"{out}.png", dpi=230, bbox_inches="tight"); fig.savefig(f"{out}.pdf", bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    fig_blocks()
    fig_maps()
