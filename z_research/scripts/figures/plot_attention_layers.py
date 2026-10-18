#!/usr/bin/env python3
"""층별 attention 지도 (2026-10-14, 사용자 "지도 같은 attention map 의 layer 별 변화를 그림으로").

(A) clip 하나 — 행 = 층 L0–L11, 열 = 보는 블록. query = 미래 블록 t(9+step) 의 참 물체 자리 3×3 (head 평균).
    색은 **행 (층) 마다 한 범위** — 그 층 안에서 어디를 보나. 칸 숫자 = 그 블록에 간 %. 오른쪽 = 같은 값의 층 × 블록 행렬.
    AR: 열 t1 … t(8+step) (마지막 열 = query 자기 블록, step 0 이면 t8). 릴리즈 (one-shot · 양방향): 열 t1 … t16, 자기 블록 = t(9+step).
    → attention_map/<model>/figures/fig_attention_layers_{visible,late}_k{step+1}
(B) 전체 clip 평균 — ar_attention_history 의 objmap (물체 query → key 블록 b, **그 블록의 물체 자리** 중심 7×7 crop 합 / clip 수).
    행 = 층, 열 = 블록 5 · 6 · 7 × (visible | late). 색 = 칸 질량 / 블록 안 균등 (1/256), 모델마다 한 범위 (log). k1.
    → attention_map/compare/fig_attention_layers_objcentered
(C) 기준 vs 그 post-FT — 릴리즈 vs vobench_release_postft · vast AR-state ep70 vs vobench_vast_ar_state_postft (2026-10-14) — 층 × 블록 질량 · 차 · 마지막으로 보인 블록의 물체 vs 거울 (층별).
    VOBench 는 post-FT 의 split_v2 TEST clip 만 (train 은 점선). → attention_map/compare/_superseded/fig_{release,vast_ar_state}_vs_postft_k{step+1}
(D) 같은 비교를 걸음 = 행 (making t9 … t16) 으로 — 블록 % (12 층 평균 · 가장 바뀐 층) · 물체 자리 % (가장 바뀐 층), 기준 · post-FT · 차.
    → attention_map/compare/fig_{release,vast_ar_state}_vs_postft_rows_{visible,late}  (정본 비교 그림)
(E) 미래 query → 물체가 보이던 과거 프레임의 물체 자리, post-FT 로 가장 많이 바뀐 층 하나 (전체 test late 에서 고름) —
    예시 clip 지도 (기준 | post-FT, 행 = 걸음, 열 = t5–t8) + 전체 clip 걸음별 곡선 (물체 자리 vs 같은 좌표 vs 균등). → compare/fig_{vast,release}_spotlight_L{l}
라벨 영어.

  python z_research/scripts/figures/plot_attention_layers.py --model vast_ar_state_ep70     # (A), GPU 1 장
  python z_research/scripts/figures/plot_attention_layers.py --model vjepa2
  python z_research/scripts/figures/plot_attention_layers.py --objcentered                  # (B), CPU
  python z_research/scripts/figures/plot_attention_layers.py --pair release [--step 7]       # (C) 릴리즈 vs post-FT, CPU
  python z_research/scripts/figures/plot_attention_layers.py --pair vast [--step 7]          # (C) vast AR-state ep70 vs 그 post-FT, CPU
  python z_research/scripts/figures/plot_attention_layers.py --rows vast                     # (D) 걸음 = 행, CPU (--rows release 도)
  python z_research/scripts/figures/plot_attention_layers.py --spotlight vast                # (E) 가장 바뀐 층 하나, 예시 + 전체 곡선, GPU 1 장 (--spotlight release 도)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis")); sys.path.insert(0, str(Path(__file__).parent))
import ar_attention_history as AH  # noqa: E402
import plot_ar_attention_rows as RW  # noqa: E402

AMD = RW.AMD
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
C, K, G, NB, L = 8, 8, 16, 16, 12
MODELS = [("vast_ar_state_ep70", "AR-state (vast, ep70)"), ("ariel_ar_ep40", "Ariel AR (ep40)"),
          ("ariel_ar_state_ep40", "Ariel AR-state (ep40)"), ("vjepa2", "V-JEPA 2 release (one-shot, bidirectional)")]


def draw_clip(d, tag, title, mname, step):
    ml, yx, hs, fr, rel = d["maps_l"][:, step], d["yx"], d["hid_s"], d["frames"], d["release"]
    tgt = C + step; ncol = NB if rel else tgt                                          # 보이는 key 블록 수
    selfb = tgt if rel else tgt - 1
    fig = plt.figure(figsize=(max(1.0 + 0.62 * ncol + 4.2, 10.5), 8.4))
    gs = fig.add_gridspec(L, ncol + 2, width_ratios=[1] * ncol + [0.6, 0.38 * ncol], wspace=0.04, hspace=0.06,
                          left=0.05, right=0.985, top=0.87, bottom=0.06)
    for l in range(L):
        vmax = ml[l, :ncol].max()
        for b in range(ncol):
            ax = fig.add_subplot(gs[l, b]); ax.set_xticks([]); ax.set_yticks([])
            if l == 0: ax.set_title(f"t{b + 1}" + ("\n(self)" if b == selfb else ""), fontsize=6.5, color="k" if b < C else "0.45")
            if b == 0: ax.set_ylabel(f"L{l}", fontsize=7, rotation=0, ha="right", va="center")
            if b < C:
                f_ = 2 * b + 1; ax.imshow(0.6 * fr[f_] + 0.1, extent=(0, G, G, 0))
                if not hs[f_]: ax.add_patch(plt.Circle((yx[b][1] + 0.5, yx[b][0] + 0.5), 1.5, fill=False, ec="cyan", lw=0.7))
            else:
                ax.imshow(np.full((G, G, 3), 0.8 if b == tgt else 0.62), extent=(0, G, G, 0))
            if b == tgt:
                ax.add_patch(plt.Rectangle((yx[b][1] - 1, yx[b][0] - 1), 3, 3, fill=False, ec="lime", lw=1.0, ls="--"))
            if b == selfb:
                for sp in ax.spines.values(): sp.set_edgecolor("lime"); sp.set_linewidth(1.2)
            v = np.clip(ml[l, b] / vmax, 0, 1); rgba = plt.get_cmap("autumn_r")(v); rgba[..., 3] = 0.95 * v
            ax.imshow(rgba, extent=(0, G, G, 0), interpolation="nearest")
            ax.text(0.4, 15.4, f"{ml[l, b].sum():.0%}", color="w", fontsize=5, va="bottom", weight="bold")
    ax = fig.add_subplot(gs[:, ncol + 1])
    M = ml[:, :ncol].sum((2, 3)) * 100
    im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=70, aspect="auto")
    for l in range(L):
        for b in range(ncol):
            ax.text(b, l, f"{M[l, b]:.0f}", ha="center", va="center", fontsize=5.5, color="w" if M[l, b] > 40 else "k")
    ax.add_patch(plt.Rectangle((selfb - 0.5, -0.5), 1, L, fill=False, ec="lime", lw=1.2))
    ax.axvline(C - 0.5, color="k", lw=1)
    ax.set_xticks(range(ncol)); ax.set_xticklabels([f"t{b + 1}" for b in range(ncol)], fontsize=5.5)
    ax.set_yticks(range(L)); ax.set_yticklabels([f"L{l}" for l in range(L)], fontsize=6)
    ax.set_xlabel("attended block"); ax.set_title("% of attention per block, per layer", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.5).set_label("% attention", fontsize=6)
    how = ("One-shot, bidirectional: query = mask token of the future block (all 8 future blocks in one pass; it can also attend to later future blocks)."
           if rel else f"Autoregressive: block t{tgt + 1} is produced by the tokens of block t{tgt} (the query's own block = 'self').")
    fig.suptitle(f"{title}  |  {RW.LABEL[mname]}, predicting t{tgt + 1}, head-averaged, one row per layer\n{how}\n"
                 f"Query = the 3x3 tokens where the object really is in t{tgt + 1} (green dashed box). Cyan circle = object in that input frame.\n"
                 "Color scale per row (layer); number = % of that layer's attention on the block.", fontsize=7.2, x=0.05, ha="left", y=0.995)
    out = AMD / mname / "figures"; out.mkdir(parents=True, exist_ok=True); out = out / f"fig_attention_layers_{tag}_k{step + 1}"
    fig.savefig(f"{out}.png", dpi=180); fig.savefig(f"{out}.pdf"); plt.close(fig); print("saved", out)


def objcentered(step=0, blocks=(5, 6, 7)):
    sets = [("vob_visible", "visible"), ("vob_late", "late")]
    fig = plt.figure(figsize=(7.4, 7.6))
    outer = fig.add_gridspec(1, len(MODELS), wspace=0.18, left=0.05, right=0.93, top=0.89, bottom=0.03)
    norm = LogNorm(vmin=0.5, vmax=64)
    for mi, (m, mlab) in enumerate(MODELS):
        Z = {s: AH.load_all(s, m) for s, _ in sets}
        inner = outer[mi].subgridspec(L, len(blocks) * 2, wspace=0.05, hspace=0.08)
        for si, (s, slab) in enumerate(sets):
            om = Z[s]["objmap"][step] / Z[s]["n_map"] * (G * G)                       # (L, NBK, 7, 7) 블록 안 균등 대비
            for bi, b in enumerate(blocks):
                for l in range(L):
                    ax = fig.add_subplot(inner[l, si * len(blocks) + bi]); ax.set_xticks([]); ax.set_yticks([])
                    im = ax.imshow(np.clip(om[l, b], 0.5, 64), cmap="magma", norm=norm, interpolation="nearest")
                    ax.plot(3, 3, "+", color="cyan", ms=3, mew=0.6)
                    for sp in ax.spines.values(): sp.set_linewidth(0.3)
                    if l == 0: ax.set_title(f"{slab}\nt{b + 1}" if bi == 1 else f"\nt{b + 1}", fontsize=5.5)
                    if mi == 0 and bi == 0 and si == 0: ax.set_ylabel(f"L{l}", fontsize=6, rotation=0, ha="right", va="center")
        fig.text(0.05 + (0.88 / len(MODELS)) * (mi + 0.5), 0.925, mlab.replace(" (", "\n("), ha="center", fontsize=6.5, weight="bold")
    cax = fig.add_axes([0.945, 0.3, 0.012, 0.4]); cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.5, 1, 2, 4, 8, 16, 32, 64]); cb.set_ticklabels(["<=0.5", "1", "2", "4", "8", "16", "32", "64"]); cb.ax.tick_params(labelsize=5.5)
    cb.set_label("attention per token / uniform within block", fontsize=6)
    fig.suptitle(f"Object query (true object position in t{C + step + 1}) -> 7x7 window centred on the object in key block t6-t8 (cyan +),\n"
                 "mean over all clips, per layer (head-averaged). VOBench visible (n=100) vs late occlusion (n=300; object hidden in t8)",
                 fontsize=7, x=0.05, ha="left", y=0.995)
    out = AMD / "compare" / "fig_attention_layers_objcentered"
    fig.savefig(f"{out}.png", dpi=220); fig.savefig(f"{out}.pdf"); print("saved", out)


def _load(m, s, split=None):
    """ar_attention_history 산출을 split_v2 train / test 로 거른다 (summarize 와 같은 규칙)."""
    Z = AH.load_all(s, m)
    if split:
        import csv
        ids = {r["video_id"] for r in csv.DictReader(open(AH.SPLIT / f"index_{split}.csv"))}
        keep = np.array([v in ids for v in Z["vid"]]); Z = {k: (v[keep] if k in AH.PER_CLIP else v) for k, v in Z.items()}
    return Z


def _lastvis(Z, step, fixed=None, dmin=3):
    """물체 query → 마지막으로 보인 문맥 블록 (fixed 를 주면 그 블록) 의 물체 자리 3×3 질량과 대조 (query 와 같은 좌표 3×3), 층별 (L,), (L,).
    물체가 그 블록과 query 블록 사이에 dmin 토큰 이상 움직인 clip 만 (아니면 두 자리가 겹친다). clip · head 평균.
    visible 은 마지막으로 보인 블록이 t8 (query 와 붙어 있다) 이라 fixed = 6 (t7) 을 쓴다.
    (거울 대조 om_mir 는 먼 걸음에서 화면 밖이라 비어서 10-14 에 om_ql 로 바꿨다)"""
    oo = np.nanmean(Z["om_obj"][:, step].astype(np.float32), 2); ql = np.nanmean(Z["om_ql"][:, step].astype(np.float32), 2)   # (N, L, NB)
    lv = np.array([max([b for b in range(C) if not h[b]], default=-1) if fixed is None else fixed for h in Z["hid"]])
    yx = Z["yx"]; ok = [i for i in range(len(lv)) if lv[i] >= 0 and np.abs(yx[i, C + step] - yx[i, lv[i]]).max() >= dmin]
    if not ok: return np.full(L, np.nan), np.full(L, np.nan), 0                         # visible k1: 물체가 t7→t9 에 3 토큰 못 간다
    o = np.stack([oo[i, :, lv[i]] for i in ok]); q = np.stack([ql[i, :, lv[i]] for i in ok])
    return o.mean(0), q.mean(0), len(ok)


PAIRS = {"release": ("vjepa2", "vobench_release_postft_ep30", "release", "fig_release_vs_postft"),                  # 기준, post-FT, 기준 이름, 그림 이름
         "vast": ("vast_ar_state_ep70", "vobench_vast_ar_state_postft_ep30", "vast AR-state ep70", "fig_vast_ar_state_vs_postft")}


def pair(which="release", step=0):
    """(C) 같은 구조의 두 predictor — 기준 vs 그것을 VOBench 로 post-FT 한 것 (PAIRS). 층 × 블록 질량 (물체 query, k=step+1) 과 그 차,
    마지막으로 보인 블록의 물체 자리 vs 같은 좌표 대조 (층별). VOBench 는 post-FT 의 split_v2 TEST clip 만 (train 은 점선으로 따로)."""
    a, b, alab, fname = PAIRS[which]
    lab = {a: alab, b: "post-FT (VOBench, ep30)"}
    col = {a: "#2a78d6", b: "#eb6834"}
    Zk = AH.load_all("vob_late", a); rel = "kind" in Zk and str(Zk["kind"][0]) == "release"                           # 릴리즈: 자기 블록 = C+r · AR: 자기 블록 = C+r−1 (query = 앞 블록 토큰)
    selfb = C + step if rel else C + step - 1
    rows = [("vob_visible", "VOBench visible", "test"), ("vob_late", "VOBench late occl.", "test"), ("ssv2", "SSv2 val (all queries)", None)]
    fig = plt.figure(figsize=(10.5, 8.6))
    gs = fig.add_gridspec(len(rows), 4, width_ratios=[1, 1, 1, 1.05], wspace=0.32, hspace=0.62, left=0.06, right=0.985, top=0.9, bottom=0.07)
    pl = iter("abcdefghijkl")
    for ri, (s, slab, sp) in enumerate(rows):
        key = "bm_obj" if s != "ssv2" else "bm_all"
        Za, Zb = _load(a, s, sp), _load(b, s, sp)
        Ma = np.nanmean(Za[key][:, step].astype(np.float32), (0, 2)) * 100; Mb = np.nanmean(Zb[key][:, step].astype(np.float32), (0, 2)) * 100   # (L, NB) %
        nbk = Ma.shape[1]
        for ci, (M, t, cm, vr) in enumerate([(Ma, lab[a], "Oranges", (0, 70 if not rel else 50)), (Mb, lab[b], "Oranges", (0, 70 if not rel else 50)),
                                             (Mb - Ma, f"post-FT minus {alab}", "RdBu_r", (-20, 20))]):
            ax = fig.add_subplot(gs[ri, ci]); im = ax.imshow(M, cmap=cm, vmin=vr[0], vmax=vr[1], aspect="auto")
            for l in range(L):
                for bb in range(nbk):
                    if np.isnan(M[l, bb]): continue                                         # AR: 아직 없는 블록
                    ax.text(bb, l, f"{M[l, bb]:+.0f}" if ci == 2 else f"{M[l, bb]:.0f}", ha="center", va="center", fontsize=4.2,
                            color="w" if (abs(M[l, bb]) > 12 if ci == 2 else M[l, bb] > 30) else "k")
            ax.add_patch(plt.Rectangle((selfb - 0.5, -0.5), 1, L, fill=False, ec="lime", lw=1.0)); ax.axvline(C - 0.5, color="k", lw=0.8)
            ax.set_xticks(range(nbk)); ax.set_xticklabels([f"t{x + 1}" for x in range(nbk)], fontsize=4.8, rotation=90)
            ax.set_yticks(range(L)); ax.set_yticklabels([f"L{l}" for l in range(L)], fontsize=5)
            n = Za[key].shape[0] if ci != 1 else Zb[key].shape[0]
            ax.set_title(f"{slab}{' (' + sp + ' split)' if sp else ''}\n{t}" + (f", n={n}" if ci < 2 else ""), fontsize=6.5)
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(labelsize=5)
            ax.text(0.5, -0.27, f"({next(pl)})", transform=ax.transAxes, ha="center", va="top", fontsize=7)
        ax = fig.add_subplot(gs[ri, 3])
        if s != "ssv2":
            for m in (a, b):
                for spl, ls, lw in ([(sp, "-", 1.3)] + ([("train", "--", 0.9)] if m == b else [])):
                    o, mr, n = _lastvis(_load(m, s, spl), step, 6 if s == "vob_visible" else None)
                    ax.plot(range(L), o * 100, color=col[m], ls=ls, lw=lw, marker="o", ms=2.2, label=f"{lab[m]}, {spl} (n={n}): object spot")
                    if spl == sp: ax.plot(range(L), mr * 100, color=col[m], ls=":", lw=1.0, label=f"{lab[m]}, {spl}: same xy as query (control)")
            ax.set_ylabel("% attention to 3x3 spot", fontsize=6); ax.set_xlabel("layer", fontsize=6)
            ax.set_title("object spot vs. same-xy control in " + ("t7" if s == "vob_visible" else "last visible blk (t6/t7)") + "\n(clips where the object moved >= 3 tokens)", fontsize=6)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.55); ax.legend(fontsize=4.6, frameon=False, loc="upper left", ncol=1)   # 범례 자리
        else:                                                                            # ssv2: k 걸음 전체, 블록 묶음 질량 (층 · head 평균)
            sb = (lambda r: C + r) if rel else (lambda r: C + r - 1)                       # 걸음 r 의 자기 블록
            grp = [("t1-t7", lambda P, r: np.nansum(P[:C - 1])), ("t8\n(not self)", lambda P, r: P[C - 1] if sb(r) != C - 1 else 0.0),
                   ("earlier\nfuture", lambda P, r: np.nansum(P[C:sb(r)])), ("self", lambda P, r: P[sb(r)]), ("later\nfuture", lambda P, r: np.nansum(P[sb(r) + 1:]))]
            w = 0.38
            for mi, (m, Z) in enumerate([(a, Za), (b, Zb)]):
                P = np.nanmean(Z["bm_all"].astype(np.float32), (0, 2, 3))                 # (K, NB 또는 AR 15)
                v = [np.mean([f(P[r], r) for r in range(K)]) for _, f in grp]
                ax.bar(np.arange(len(grp)) + (mi - 0.5) * w, np.array(v) * 100, w, color=col[m], label=lab[m])
            ax.set_xticks(range(len(grp))); ax.set_xticklabels([g for g, _ in grp], fontsize=5.5)
            ax.set_ylabel("% attention (mean over 8 query blocks)", fontsize=6); ax.legend(fontsize=5, frameon=False)
            ax.set_title("SSv2: attention by block group", fontsize=6.5)
        ax.tick_params(labelsize=5.5); ax.text(0.5, -0.27, f"({next(pl)})", transform=ax.transAxes, ha="center", va="top", fontsize=7)
    how = (f"One-shot, bidirectional; query = mask tokens of t{C + step + 1}" if rel else
           f"Autoregressive, block-causal; query = tokens of block t{C + step} that produce t{C + step + 1}")
    fig.suptitle(f"{alab} vs. the same predictor post-FT on VOBench (split_v2 train, 30 ep). {how}"
                 f" (green column = self).\nVOBench rows: query = 3x3 tokens at the true object position, clips from the post-FT TEST split; SSv2 row: all query tokens."
                 " Heat maps: % attention per key block, per layer (head-averaged).", fontsize=7.2, x=0.06, ha="left", y=0.995)
    out = AMD / "compare" / "_superseded" / f"{fname}_k{step + 1}"                         # 10-14 행 그림 (--rows) 이 대체, 층별 대조용으로만 남김
    fig.savefig(f"{out}.png", dpi=200); fig.savefig(f"{out}.pdf"); plt.close(fig); print("saved", out)


BAND = {"release": (5, 12), "vast": (3, 5)}                                            # post-FT 로 가장 많이 바뀐 층 (fig_*_vs_postft_k* 에서 읽음) — 표시용


def pair_rows(which="vast", setname="vob_late", sp="test"):
    """(D) 걸음 = 행 (making t9 … t16) 으로 기준 vs post-FT 를 한눈에 (2026-10-14, 사용자 "t9 생성할 때, t10 생성할 때 이런 게 한눈에").
    물체 query (그 걸음이 만드는 블록의 참 물체 자리 3×3). 행 1 = 블록에 간 % (12 층 평균) · 행 2 = 같은 값, 가장 많이 바뀐 층만 ·
    행 3 = 그 블록의 **물체 자리 3×3** 에 간 % (가장 많이 바뀐 층). 열 = 기준 · post-FT · 차. clip · head 평균, split_v2 test."""
    a, b, alab, fname = PAIRS[which]
    Za, Zb = _load(a, setname, sp), _load(b, setname, sp)
    rel = "kind" in Za and str(Za["kind"][0]) == "release"
    nbk = Za["bm_obj"].shape[-1]; l0, l1 = BAND[which]
    hidf = Za["hid"].mean(0)                                                              # 블록마다 물체가 가려진 clip 비율
    def mats(Z):
        bo = np.nanmean(Z["bm_obj"].astype(np.float32), (0, 3)) * 100; oo = np.nanmean(Z["om_obj"].astype(np.float32), (0, 3)) * 100   # (K, L, nbk)
        return [bo.mean(1), bo[:, l0:l1].mean(1), oo[:, l0:l1].mean(1)]
    A, B = mats(Za), mats(Zb)
    rows = [("% of attention per block, mean of 12 layers", 60), (f"% of attention per block, layers L{l0}-L{l1 - 1} only", 60),
            (f"% of attention on the OBJECT's 3x3 spot in that block, layers L{l0}-L{l1 - 1}", 20)]
    fig = plt.figure(figsize=(14.5, 10.2))
    gs = fig.add_gridspec(3, 3, wspace=0.1, hspace=0.42, left=0.07, right=0.985, top=0.885, bottom=0.06)
    pl = iter("abcdefghi")
    for ri, (rt, vmax) in enumerate(rows):
        for ci, (M, t, cm, vr) in enumerate([(A[ri], alab, "Oranges", (0, vmax)), (B[ri], "post-FT (VOBench, ep30)", "Oranges", (0, vmax)),
                                             (B[ri] - A[ri], "post-FT minus " + alab, "RdBu_r", (-vmax / 3, vmax / 3))]):
            ax = fig.add_subplot(gs[ri, ci]); M = M.copy()
            for r in range(K):                                                            # AR: 아직 없는 블록 · 만들 블록은 비운다
                if not rel: M[r, C + r - 1 + 1:] = np.nan
            im = ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=vr[0], vmax=vr[1], aspect="auto")
            for r in range(K):
                tgt = C + r
                for bb in range(nbk):
                    v = M[r, bb]
                    if bb == tgt and rel:
                        ax.text(bb, r, f"?{v:+.0f}" if ci == 2 else f"?{v:.0f}", ha="center", va="center", fontsize=5.2, color="0.35"); continue
                    if np.isnan(v):
                        if bb == tgt: ax.text(bb, r, "?", ha="center", va="center", fontsize=7, color="0.35")
                        continue
                    big = abs(v) > vr[1] * 0.55
                    ax.text(bb, r, (f"{v:+.0f}" if ci == 2 else f"{v:.0f}") if (ri < 2 or abs(v) >= 1) else (f"{v:+.1f}" if ci == 2 else f"{v:.1f}"),
                            ha="center", va="center", fontsize=5.2, color="w" if big else "k")
                ax.add_patch(plt.Rectangle((tgt - 0.5, r - 0.5), 1, 1, fill=False, ec="0.4", lw=0.8, ls="--"))
            ax.axvline(C - 0.5, color="k", lw=1)
            ax.set_xticks(range(nbk)); ax.set_xticklabels([f"t{x + 1}" + ("*" if x < C and hidf[x] > 0.5 else "") for x in range(nbk)], fontsize=6)
            ax.set_yticks(range(K)); ax.set_yticklabels([(f"t{C + r + 1} mask query" if rel else f"making t{C + r + 1}") for r in range(K)] if ci == 0 else [], fontsize=6.5)
            ax.set_title(f"{t}\n{rt}", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015).ax.tick_params(labelsize=5.5)
            ax.text(0.5, -0.17, f"({next(pl)})", transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
    hid_txt = ", ".join(f"t{x + 1} {hidf[x]:.0%}" for x in range(C) if hidf[x] > 0.05)
    fig.suptitle(f"{alab} vs. the same predictor post-FT on VOBench (split_v2 train, 30 ep)  |  VOBench {setname.split('_')[1]} clips, post-FT {sp.upper()} split, n={len(Za['vid'])}\n"
                 + ("One-shot, bidirectional: each row = mask-token query of one future block (one pass, not time steps)."
                    if rel else "Autoregressive: each row = one step, '?' = the block being made (query = tokens of the block to its left).")
                 + " Query = 3x3 tokens at the true object position in the '?' block. Mean over clips and heads.\n"
                 + (f"* = object hidden in most clips ({hid_txt}). " if hid_txt else "") + f"Layers L{l0}-L{l1 - 1} = where post-FT changed attention the most.",
                 fontsize=7.6, x=0.07, ha="left", y=0.995)
    out = AMD / "compare" / f"{fname}_rows_{setname.split('_')[1]}"
    fig.savefig(f"{out}.png", dpi=180); fig.savefig(f"{out}.pdf"); plt.close(fig); print("saved", out)


SPOT_CLIP = "prim_assets_flat_vel_late_k1_shap_cone_pos_a"                              # post-FT TEST split, 물체 t1–t7 보임 · t8–t9 가림, x 2→13 토큰


def _pop(m, setname, sp="test", dmin=3):
    """걸음 r 마다: 물체 query → **물체가 보이던 과거 문맥 블록들** (가림 아님, query 블록 물체 자리에서 dmin 토큰 이상 떨어진 것) 의
    물체 자리 3×3 질량 합 O · 같은 블록들의 query 좌표 3×3 합 Q · 균등 3×3 합 U. 층별 (K, L) — clip 평균 (해당 블록이 있는 clip 만), clip 수 (K,)."""
    Z = _load(m, setname, sp)
    oo = np.nanmean(Z["om_obj"].astype(np.float32), 3); ql = np.nanmean(Z["om_ql"].astype(np.float32), 3); bo = np.nanmean(Z["bm_obj"].astype(np.float32), 3)
    yx, hid = Z["yx"], Z["hid"]; N = len(yx)
    O = np.zeros((K, L)); Q = np.zeros((K, L)); U = np.zeros((K, L)); n = np.zeros(K, int)
    for r in range(K):
        for i in range(N):
            bs = [bb for bb in range(C) if not hid[i, bb] and np.abs(yx[i, C + r] - yx[i, bb]).max() >= dmin]
            if not bs: continue
            O[r] += oo[i, r][:, bs].sum(1); Q[r] += ql[i, r][:, bs].sum(1); U[r] += bo[i, r][:, bs].sum(1) * 9 / 256; n[r] += 1
        O[r] /= max(n[r], 1); Q[r] /= max(n[r], 1); U[r] /= max(n[r], 1)
    return O * 100, Q * 100, U * 100, n


def spotlight(which="vast", vid=SPOT_CLIP, steps=(0, 2, 4, 7), blocks=(4, 5, 6, 7)):
    """(E) 미래 query 가 **물체가 보이던 과거 프레임의 물체 자리**를 얼마나 보나 — post-FT 로 가장 많이 바뀐 층 하나에서 (2026-10-14, 사용자 시각화 요청).
    층 = 전체 test late clip 에서 (post-FT − 기준) 물체 자리 질량이 걸음 평균으로 가장 큰 층.
    왼쪽: 예시 clip (post-FT test split) 의 그 층 attention — 행 = 걸음, 열 = 과거 블록 t5–t8, 기준 | post-FT. 색 범위는 행마다 두 모델 공통.
    오른쪽: 전체 clip, 걸음별 물체 자리 합 (실선) vs 같은 좌표 합 (점선) vs 균등 (회색) — late · visible test."""
    a, b, alab, fname = PAIRS[which]
    pops = {(m, st): _pop(m, st) for m in (a, b) for st in ("vob_late", "vob_visible")}
    d = (pops[(b, "vob_late")][0] - pops[(a, "vob_late")][0]).mean(0)                    # (L,) 걸음 평균 차
    lay = int(np.argmax(d)); print(f"[{which}] post-FT − base, object-spot sum (late test), per layer:", np.round(d, 2).tolist(), "→ L", lay)
    ex = {m: RW.compute(m, [vid])[0] for m in (a, b)}
    rel = ex[a]["release"]; yx, hs, fr = ex[a]["yx"], ex[a]["hid_s"], ex[a]["frames"]
    col = {a: "#2a78d6", b: "#eb6834"}; lab = {a: alab, b: "post-FT (VOBench)"}
    fig = plt.figure(figsize=(13.6, 7.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.15, 1], wspace=0.13, left=0.075, right=0.985, top=0.81, bottom=0.08)
    left = outer[0].subgridspec(len(steps), 2 * len(blocks) + 1, width_ratios=[1] * len(blocks) + [0.25] + [1] * len(blocks), wspace=0.04, hspace=0.1)
    for ri, r in enumerate(steps):
        A = {m: ex[m]["maps_l"][lay, r] for m in (a, b)}                                 # (NB, G, G)
        vmax = max(A[m][list(blocks)].max() for m in (a, b))
        for mi, m in enumerate((a, b)):
            for bi, bb in enumerate(blocks):
                ax = fig.add_subplot(left[ri, bi + mi * (len(blocks) + 1)]); ax.set_xticks([]); ax.set_yticks([])
                ax.imshow(0.55 * fr[2 * bb + 1] + 0.1, extent=(0, G, G, 0))
                v = np.clip(A[m][bb] / vmax, 0, 1); rgba = plt.get_cmap("autumn_r")(v); rgba[..., 3] = 0.95 * v
                ax.imshow(rgba, extent=(0, G, G, 0), interpolation="nearest")
                qy, qx = yx[C + r]
                ax.add_patch(plt.Rectangle((qx - 1, qy - 1), 3, 3, fill=False, ec="lime", lw=1.0, ls="--"))   # query 와 같은 좌표
                oy, ox = yx[bb]
                if not hs[2 * bb + 1]:
                    ax.add_patch(plt.Circle((ox + 0.5, oy + 0.5), 1.6, fill=False, ec="cyan", lw=1.1))     # 그 프레임의 물체
                    pobj = A[m][bb][max(oy - 1, 0):oy + 2, max(ox - 1, 0):ox + 2].sum() * 100
                    ax.text(0.4, 15.4, f"obj {pobj:.1f}%", color="w", fontsize=5.6, va="bottom", weight="bold")
                else:
                    ax.text(0.4, 15.4, "hidden", color="w", fontsize=5.6, va="bottom", weight="bold")
                for sp_ in ax.spines.values(): sp_.set_edgecolor(col[m]); sp_.set_linewidth(1.6)
                if ri == 0: ax.set_title(f"t{bb + 1}", fontsize=7)
                if bi == 0 and mi == 0:
                    ax.set_ylabel(f"t{C + r + 1} mask query" if rel else f"making t{C + r + 1}", fontsize=7.5, rotation=0, ha="right", va="center")
        if ri == 0:
            for mi, m in enumerate((a, b)):
                x0 = left[0, mi * (len(blocks) + 1)].get_position(fig).x0; x1 = left[0, mi * (len(blocks) + 1) + len(blocks) - 1].get_position(fig).x1
                fig.text((x0 + x1) / 2, 0.85, lab[m], ha="center", fontsize=8.5, weight="bold", color=col[m])
    right = outer[1].subgridspec(2, 1, hspace=0.75)
    xs = np.arange(K)
    for pi, (st, stl) in enumerate([("vob_late", "late occlusion (object hidden at t8)"), ("vob_visible", "visible")]):
        ax = fig.add_subplot(right[pi])
        for m in (a, b):
            O, Q, U, n = pops[(m, st)]
            ax.plot(xs, O[:, lay], color=col[m], lw=1.8, marker="o", ms=3, label=f"{lab[m]}: object spot")
            ax.plot(xs, Q[:, lay], color=col[m], lw=1.2, ls=":", label=f"{lab[m]}: same xy as query")
        ax.plot(xs, pops[(a, st)][2][:, lay], color="0.55", lw=1.0, ls="--", label="uniform (3x3 share)")
        nn = pops[(a, st)][3]
        ax.set_xticks(xs); ax.set_xticklabels([f"t{C + r + 1}\nn={nn[r]}" for r in range(K)], fontsize=5.8)
        ax.set_xlabel("future block being predicted", fontsize=6.5); ax.set_ylabel(f"% attention, layer L{lay}", fontsize=6.5)
        ax.set_title(f"all test clips, {stl}\nsum over past blocks where the object was visible (moved >= 3 tokens)", fontsize=6.8)
        ax.tick_params(labelsize=6); ax.set_ylim(bottom=0)
        if pi == 0: ax.legend(fontsize=5.6, frameon=False, loc="upper right")
        ax.text(1.0, 1.02, f"({'bc'[pi]})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)   # 아래 두면 잘린다
    fig.text(0.075 + 0.5 * (0.985 - 0.075) * 2.15 / 3.15, 0.025, "(a)", ha="center", fontsize=8)
    how = ("One-shot: row = mask-token query of that future block" if rel else
           "Autoregressive: row = one step; the query tokens sit in the block before it and their output becomes that block")
    fig.suptitle(f"Do future queries look at the object in past frames?  {alab} vs. post-FT on VOBench  |  layer L{lay} (largest post-FT change), mean of 12 heads\n"
                 f"(a) one test clip ({vid}). {how}.\nQuery = 3x3 tokens at the object's true position in that future block. "
                 "Cyan circle = object in that frame, green dashed box = same xy as the query. Color scale shared by both models within a row. "
                 "(b, c) all test clips.", fontsize=7.6, x=0.075, ha="left", y=0.995)
    out = AMD / "compare" / f"fig_{which}_spotlight_L{lay}"
    fig.savefig(f"{out}.png", dpi=190); fig.savefig(f"{out}.pdf"); plt.close(fig); print("saved", out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=list(RW.CKPTS)); ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--objcentered", action="store_true")
    ap.add_argument("--pair", choices=list(PAIRS), help="(C) 기준 vs 그 post-FT (CPU): release | vast")
    ap.add_argument("--rows", choices=list(PAIRS), help="(D) 걸음 = 행으로 기준 vs post-FT (CPU), visible · late 둘 다")
    ap.add_argument("--spotlight", choices=list(PAIRS), help="(E) 가장 많이 바뀐 층 하나 — 예시 clip 지도 + 전체 clip 곡선 (GPU 1 장)"); a = ap.parse_args()
    if a.spotlight: spotlight(a.spotlight)
    if a.rows:
        for st in ("vob_visible", "vob_late"): pair_rows(a.rows, st)
    if a.objcentered: objcentered(a.step)
    if a.pair: pair(a.pair, step=a.step)
    if a.model:
        for d, (_, tag, title) in zip(RW.compute(a.model), RW.CLIPS):
            draw_clip(d, tag, title.split(" (")[0], a.model, a.step)


if __name__ == "__main__":
    main()
