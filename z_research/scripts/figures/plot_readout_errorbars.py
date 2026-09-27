#!/usr/bin/env python3
"""자 (attn) 의 **존재 판정**과 **위치 오차**를 p·z·h 나란히, 95 % 신뢰구간과 함께.

같은 자, 두 데이터:
  **학습셋 test** 자를 배운 곳의 held-out. **음성이 있는 유일한 세트**라 precision·ROC 를 여기서만 잰다
  **v3**      test 전용. 물체가 f16~f63 내내 화면 안 → **음성이 0 개**다. recall 만 정의된다

용어 (이 자의 두 번째 head 가 내는 것이 presence 다):
  recall       물체가 보이는 칸을 "있다" 고 맞힌 비율 (1 − miss)
  precision    "있다" 고 답한 칸 중 실제로 있던 비율
  specificity  물체가 없는 칸을 "없다" 고 맞힌 비율 (1 − 오탐률)
  **occluded** 화면 안인데 판자 뒤에 가려진 칸. 학습에서 **제외**돼 **정답 라벨이 없다**
               → 오류율이 아니라 "있다고 답한 비율" 만 적는다 (측정값)

⚠️ 위치 오차는 전부 **L2 (유클리드) 거리** `‖pred − truth‖₂` 다. 평균·중앙·p90 은 같은 거리의
   요약 통계일 뿐 노름이 다른 게 아니다. (2026-09-24 정정 — 예전 그림의 `mean |err|` 는 L1 이 아니었다)
⚠️ 위치는 **"있다" 고 답한 칸에서만** 잰다 (두 오류는 경로가 다르다).
⚠️ 좌표는 학습셋 test 에서 잰 **attn 치우침을 뺀 값** (`attn_bias_px.json`).
⚠️ **신뢰구간은 clip 을 재표집해 만든다** (2,000 회). 한 clip 의 8~16 튜블릿은 서로 독립이 아니다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_readout_errorbars.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
import sys; sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))   # noqa: E702
from rollout3_paths import PRES, WIN, fig, check   # noqa: E402  경로·자 지문은 한 곳에서
def train_set():
    """자를 배운 세트 이름 — `summary.json` 의 `train_set` (없으면 옛 training_v6)."""
    return json.loads((PRES / "summary.json").read_text()).get("train_set", "training_v6")


def train_index():
    return ROOT / f"data_csv/rollout_v2_{train_set()}/index_probe.csv"
FIG = fig("readout")
RESN, CELL, SPLIT = 144.0, 18.0, 32
REPS = ("p", "z", "h")
COL = {"p": "#eb6834", "z": "#2a78d6", "h": "#1baf7a"}
NB = 2000
rng = np.random.default_rng(0)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def ci(num, den, clip_of, f=lambda r: r):
    """clip 을 재표집한 (값, -폭, +폭). num/den 은 칸마다의 분자·분모."""
    if len(clip_of) == 0:
        return (np.nan, np.nan, np.nan)
    nc = int(clip_of.max()) + 1
    sn = np.bincount(clip_of, num, minlength=nc)
    sd = np.bincount(clip_of, den, minlength=nc)
    pt = f(sn.sum() / max(sd.sum(), 1e-9))
    idx = rng.integers(0, nc, size=(NB, nc))
    b = f(sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1e-9))
    lo, hi = np.percentile(b, [2.5, 97.5])
    return pt, pt - lo, hi - pt


def _groups(clip_of):
    """clip 단위 재표집을 **벡터로** 하기 위한 색인.

    ⚠️ 루프로 clip 마다 슬라이스를 붙이면 2,000 회에 몇 분이 걸린다 (2026-09-24 실측).
       `np.repeat` 두 번으로 한 번에 색인을 만든다.
    """
    order = np.argsort(clip_of, kind="stable")
    cs = clip_of[order]
    nc = int(clip_of.max()) + 1
    return order, np.searchsorted(cs, np.arange(nc)), np.bincount(cs, minlength=nc)


def _pick(start, cnt, pick):
    c = cnt[pick]
    off = np.repeat(np.cumsum(c) - c, c)
    return np.repeat(start[pick], c) + (np.arange(c.sum()) - off)


def ci_quant(v, clip_of, q, nb=600):
    """분위수의 clip 재표집 구간. 비율이 아니라 순서 통계라 `ci` 와 따로 둔다."""
    order, start, cnt = _groups(clip_of)
    vs = v[order]; nc = len(cnt)
    pt = float(np.percentile(v, q))
    b = np.array([np.percentile(vs[_pick(start, cnt, rng.integers(0, nc, nc))], q) for _ in range(nb)])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return pt, pt - lo, hi - pt


def roc(score, lab):
    """(fpr, tpr, auc). sklearn 이 없어 직접 만든다."""
    o = np.argsort(-score, kind="stable")
    y = lab[o]
    tpr = np.concatenate([[0], np.cumsum(y) / max(y.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(~y) / max((~y).sum(), 1)])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def _auc(s, y):
    """Mann-Whitney 순위 공식 = P(양성 점수 > 음성 점수)."""
    r = np.empty(len(s)); r[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    a, b = y.sum(), (~y).sum()
    return (r[y].sum() - a * (a + 1) / 2) / max(a * b, 1)


def auc_ci(score, lab, clip_of, nb=600):
    order, start, cnt = _groups(clip_of)
    ss, ll = score[order], lab[order]; nc = len(cnt)
    pt = _auc(score, lab)
    b = np.empty(nb)
    for i in range(nb):
        k = _pick(start, cnt, rng.integers(0, nc, nc))
        b[i] = _auc(ss[k], ll[k])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return pt, pt - lo, hi - pt


def train_domain():
    rows = list(csv.DictReader(train_index().open()))
    n = len(rows)
    arr = lambda s: np.array([float(v) for v in str(s).split()], np.float32)
    inf_ = (np.stack([arr(r["in_frame_by_sample"]) for r in rows]) > 0).reshape(n, 16, 2).all(2)[:, 8:]
    emp = np.array([r["scenario"] == "empty" for r in rows])[:, None].repeat(8, 1)
    S = json.loads((PRES / "summary.json").read_text())
    bias = json.loads((PRES / "attn_bias_px.json").read_text())
    cl = np.repeat(np.arange(n)[:, None], 8, 1)
    out = {}
    for rep in REPS:
        d = np.load(PRES / rep / "preds_attn.npz", allow_pickle=True)
        thr = S["reps"][rep]["attn"]["thr_val_fpr5"]
        pos, ign, T = d["pos"], d["ign"], d["test"][:, None]
        neg = (~pos) & (~ign)
        say = d["presence"] > thr
        MP, MN = T & pos, T & neg
        r = {}
        r["recall"] = ci(say[MP].astype(float), np.ones(int(MP.sum())), cl[MP])
        r["specificity"] = ci((~say)[MN].astype(float), np.ones(int(MN.sum())), cl[MN])
        # precision: 분자 = TP, 분모 = "있다" 고 답한 칸 전부 (양성 + 음성 양쪽에서)
        S_ = T & (pos | neg) & say
        r["precision"] = ci(pos[S_].astype(float), np.ones(int(S_.sum())), cl[S_])
        OC = T & ign & inf_
        r["occluded"] = ci(say[OC].astype(float), np.ones(int(OC.sum())), cl[OC])
        # training_v8 부터: 제외가 두 종류다 — 판과 겹침 / **구조물 속**. 따로 적는다 (둘 다 라벨 없음)
        if "scenery_intersects_by_sample" in rows[0] and any(x["scenery_intersects_by_sample"].strip() for x in rows):
            sc_ = (np.stack([arr(x["scenery_intersects_by_sample"]) if x["scenery_intersects_by_sample"].strip()
                             else np.zeros(32, np.float32) for x in rows]) > 0).reshape(n, 16, 2).any(2)[:, 8:]
            OP, OS = T & ign & ~sc_, T & ign & sc_
            r["excl_prop"] = ci(say[OP].astype(float), np.ones(int(OP.sum())), cl[OP])
            r["excl_scenery"] = ci(say[OS].astype(float), np.ones(int(OS.sum())), cl[OS])
        r["cm"] = np.array([[int((say & MP).sum()), int((~say & MP).sum())],      # 실제 있음 → 있다 / 없다
                            [int((say & MN).sum()), int((~say & MN).sum())]])     # 실제 없음 → 있다 / 없다
        lab = pos[T & (pos | neg)]
        sco = d["presence"][T & (pos | neg)]
        r["roc"] = roc(sco, lab)
        r["auc"] = auc_ci(sco, lab, cl[T & (pos | neg)])
        e = np.linalg.norm((d["pred"] - d["truth"]) * RESN - np.array(bias[rep], np.float32), axis=-1)
        m = MP & say
        r["L2 mean"] = ci(e[m], np.ones(int(m.sum())), cl[m])
        r["L2 median"] = ci_quant(e[m], cl[m], 50)
        r["L2 p90"] = ci_quant(e[m], cl[m], 90)
        r["thr"] = thr
        out[rep] = r
    return out


def v3():
    z = np.load(WIN / "readings.npz", allow_pickle=True)
    check(z, PRES)                                  # 이 readings 가 지금 자로 읽혔는가
    bias = json.loads((PRES / "attn_bias_px.json").read_text())
    S = json.loads((PRES / "summary.json").read_text())
    L, inf, POS = z["truth"], z["in_frame"], z["role"] == "roll"
    nclip = int(POS.sum())
    out = {}
    for rep in REPS:
        thr = S["reps"][rep]["attn"]["thr_val_fpr5"]
        for p in (4, 8, 16, 32):
            tp = p // 2
            gt = L[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2, 2).mean(2)[POS] * RESN
            vis = inf[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2).all(2)[POS]
            rn, rc, el, ec, al, ac = [], [], [], [], [], []
            st = {c: [] for c in (4, 8, 16, 32)}                           # 튜블릿별 recall, C 마다
            for c in (4, 8, 16, 32):
                v = z[f"C{c}_P{p}_{rep}"][POS]
                say = v[..., 2] > thr
                xy = v[..., 0:2] * RESN - np.array(bias[rep], np.float32)
                cl = np.repeat(np.arange(nclip)[:, None], tp, 1)
                rn.append(say[vis].astype(float)); rc.append(cl[vis])
                st[c] = [ci(say[vis[:, t], t].astype(float), np.ones(int(vis[:, t].sum())),
                             cl[vis[:, t], t]) for t in range(tp)]
                e_ = np.linalg.norm(xy - gt, axis=-1)
                k = vis & say
                el.append(e_[k]); ec.append(cl[k])                       # 있다고 한 칸만
                al.append(e_[vis]); ac.append(cl[vis])                   # 진실이 화면 안인 칸 전부
            rn, rc = np.concatenate(rn), np.concatenate(rc)
            el, ec = np.concatenate(el), np.concatenate(ec)
            al, ac = np.concatenate(al), np.concatenate(ac)
            out[(rep, p)] = dict(recall_t=st,
                                 recall=ci(rn, np.ones(len(rn)), rc),
                                 all=ci(al, np.ones(len(al)), ac),
                                 gated=ci(el, np.ones(len(el)), ec))
    return out


def bars(ax, keys, get, ylab, pct=False, short=None):
    w, x = 0.26, np.arange(len(keys))
    for i, rep in enumerate(REPS):
        v = np.array([get(rep, k) for k in keys]); s = 100 if pct else 1
        ax.bar(x + (i - 1) * w, v[:, 0] * s, w, color=COL[rep], label=rep, edgecolor="white", lw=0.6)
        ax.errorbar(x + (i - 1) * w, v[:, 0] * s, yerr=v[:, 1:].T * s, fmt="none",
                    ecolor="#333333", elinewidth=1.0, capsize=2.5)
    ax.set_xticks(x); ax.set_xticklabels([(short or {}).get(k, k) for k in keys], fontsize=8.5)
    ax.set_ylabel(ylab); ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=0); ax.grid(axis="y", lw=0.3, alpha=0.4); ax.set_axisbelow(True)


def conf(ax, cm, rep):
    """2 x 2 혼동 행렬. 행 = 실제, 열 = 자의 답. 칸에 개수와 **행 비율**을 같이 쓴다."""
    rs = cm.sum(1, keepdims=True)
    frac = cm / np.maximum(rs, 1)
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=COL[rep],
                                       alpha=0.10 + 0.55 * frac[i, j], edgecolor="white", lw=1.5))
            ax.text(j + 0.5, 1.5 - i, f"{cm[i, j]:,}\n{100*frac[i, j]:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold" if i == j else "normal")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["says\npresent", "says\nabsent"], fontsize=8.5)
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(["truly\npresent", "truly\nabsent"], fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    ax.set_title(f"{rep}\nrecall {100*tp/(tp+fn):.1f}%  ·  precision {100*tp/max(tp+fp,1):.1f}%",
                 fontsize=9.5, fontweight="bold", color=COL[rep])


def fig_train(A):
    fig = plt.figure(figsize=(11.0, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.42)

    a = fig.add_subplot(gs[0, 0:2])                                   # (a) ROC
    for rep in REPS:
        f_, t_, _ = A[rep]["roc"]; p_, lo, hi = A[rep]["auc"]
        a.plot(f_, t_, color=COL[rep], lw=1.8, label=f"{rep}   AUROC {p_:.3f} (−{lo:.3f}/+{hi:.3f})")
    a.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls=":")
    a.set_xlim(0, 0.25); a.set_ylim(0.75, 1.002)
    a.set_xlabel("false positive rate  (1 − specificity)"); a.set_ylabel("recall  (true positive rate)")
    a.legend(frameon=False, fontsize=9, loc="lower right")
    a.spines[["top", "right"]].set_visible(False)
    a.set_title("(a)  presence — ROC (zoomed to the left corner)", fontsize=10, fontweight="bold")

    c = fig.add_subplot(gs[0, 2])                                     # (b) 위치 오차
    bars(c, ["L2 mean", "L2 median"], lambda r, k: A[r][k], "position L2 error (px)",
         short={"L2 mean": "mean", "L2 median": "median"})
    c.set_ylim(0, 20.5)                                   # 안내선 (한 칸 = 18 px) 이 축 안에 들어오게
    c.axhline(CELL, color="#666666", lw=0.8, ls="--")
    c.text(-0.45, CELL + 0.4, "1 cell = 18 px", fontsize=7.5, color="#666666", ha="left")
    c.set_title("(b)  position error", fontsize=10, fontweight="bold")

    for j, rep in enumerate(REPS):                                    # (c) 혼동 행렬
        conf(fig.add_subplot(gs[1, j]), A[rep]["cm"], rep)
    fig.text(0.5, 0.072, "(c)   confusion at the operating threshold  "
             f"(p {A['p']['thr']:+.2f} · z {A['z']['thr']:+.2f} · h {A['h']['thr']:+.2f}, "
             f"set for 5% false alarms on the {train_set()} val split)", ha="center", fontsize=10, fontweight="bold")
    if "excl_scenery" in A["p"]:          # training_v8 부터: 제외가 두 종류 (판과 겹침 / 구조물 속)
        tp_ = "  ".join(f"{r} {100*A[r]['excl_prop'][0]:.0f}%" for r in REPS)
        ts_ = "  ".join(f"{r} {100*A[r]['excl_scenery'][0]:.0f}%" for r in REPS)
        cap = ("Excluded tubelets carry NO label and appear in no cell above.  The readout calls them 'present' —\n"
               f"overlapping a prop: {tp_}   ·   inside a structure (ledge / wedge / wall): {ts_}.")
    else:
        txt = "   ·   ".join(f"{r} {100*A[r]['occluded'][0]:.0f}%" for r in REPS)
        cap = ("Occluded tubelets — object in frame but hidden behind a prop — carry NO label and were excluded from "
               f"training,\nso they appear in no cell above. The readout calls them 'present':  {txt}.")
    fig.text(0.5, 0.028, cap, ha="center", fontsize=8.5)
    fig.suptitle(f"attn readout on {train_set()} test — the readout's OWN held-out set (training domain)\n"
                 "error bars = 95% bootstrap CI over clips   ·   position error = L2 distance ‖pred − truth‖, "
                 "measured only where the readout says 'present'", fontsize=9.5)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.87, bottom=0.145)
    # ⚠️ 파일 이름은 세트와 무관하게 고정한다 (자를 다른 세트로 배워도 문서 링크가 안 깨지게). 제목에 세트를 쓴다
    fig.savefig(FIG / "fig_readout_train.png", dpi=175); plt.close(fig)


def fig_v3(B):
    P4 = [4, 8, 16, 32]
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 4.0))
    bars(ax[0], P4, lambda r, k: B[(r, k)]["recall"], "recall (%)", pct=True)
    ax[0].set_ylim(40, 101)
    ax[0].set_title("(a)  recall", fontsize=10, fontweight="bold")
    bars(ax[1], P4, lambda r, k: B[(r, k)]["all"], "position L2 error (px)")
    ax[1].set_title("(b)  position L2, every in-frame tubelet", fontsize=10, fontweight="bold")
    bars(ax[2], P4, lambda r, k: B[(r, k)]["gated"], "position L2 error (px)")
    ax[2].set_title("(c)  position L2, only where it says 'present'", fontsize=10, fontweight="bold")
    for a_ in ax[1:]:
        a_.axhline(CELL, color="#666666", lw=0.8, ls="--")
        a_.text(-0.45, CELL + 0.8, "1 cell = 18 px", fontsize=7.5, color="#666666", ha="left")
    lo = min(a_.get_ylim()[0] for a_ in ax[1:]); hi = max(a_.get_ylim()[1] for a_ in ax[1:])
    for a_ in ax[1:]:
        a_.set_ylim(lo, hi)
    for a_ in ax:
        a_.set_xlabel("prediction length P (frames)")
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("attn readout on v3 — a set the readout never saw (test domain).  context lengths C pooled\n"
                 "the object is in frame in every tubelet, so recall is the only presence measure that exists here",
                 fontsize=9.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.235, wspace=0.28)
    fig.savefig(FIG / "fig_readout_v3.png", dpi=175); plt.close(fig)


def fig_recall_t(B):
    """튜블릿마다의 recall 을 **창 하나에 칸 하나씩** (C x P 16 칸).

    ⚠️ C 를 합치거나 한 칸에 여러 C 를 겹쳐 그리면 안 된다 — `p` 는 C 마다 무너지는 자리가 다르고
       매끄럽지 않게 깜빡여서, 합치면 **어느 창에도 없는 봉우리**가 생긴다 (2026-09-24 에 t12 에서 봤다).
    """
    CS, PS_ = [4, 8, 16, 32], [4, 8, 16, 32]
    fig, ax = plt.subplots(len(CS), len(PS_), figsize=(10.4, 8.6), sharey=True)
    for r_, c in enumerate(CS):
        for j, p in enumerate(PS_):
            a_ = ax[r_, j]; tp = p // 2; t = np.arange(tp)
            for rep in REPS:
                v = np.array(B[(rep, p)]["recall_t"][c]) * 100
                a_.plot(t, v[:, 0], color=COL[rep], lw=1.7, marker="o", ms=3.4,
                        label=rep if (r_ == 0 and j == 0) else None)
                a_.fill_between(t, v[:, 0] - v[:, 1], v[:, 0] + v[:, 2],
                                color=COL[rep], alpha=0.20, lw=0)
            a_.axhline(50, color="#bbbbbb", lw=0.7, ls=":")
            a_.set_ylim(0, 104); a_.set_xlim(-0.5, tp - 0.5)
            a_.set_xticks(t if tp <= 8 else t[::3])
            a_.set_yticks([0, 50, 100])
            a_.tick_params(labelsize=8.5)
            a_.spines[["top", "right"]].set_visible(False)
            a_.grid(axis="y", lw=0.3, alpha=0.35); a_.set_axisbelow(True)
            if r_ == 0:
                a_.set_title(f"prediction P = {p}f", fontsize=10, fontweight="bold")
            if j == 0:
                a_.set_ylabel(f"context C = {c}f\nrecall (%)", fontsize=9.5)
            if r_ == len(CS) - 1:
                a_.set_xlabel("future tubelet", fontsize=9)
            if r_ == 0 and j == 0:
                a_.legend(frameon=False, fontsize=9, loc="lower left", handlelength=1.4)
    fig.suptitle("v3 — recall tubelet by tubelet, one window per cell (16 windows)\n"
                 "does the readout still find the object at the end of the window?", fontsize=10.5)
    fig.text(0.5, 0.028,
             "Bands are 95% bootstrap CI over clips.  z and h stay flat in every window, so a fall is a property of p, "
             "not of the readout.\nWindows are NOT averaged together: p collapses at a different tubelet for each C "
             "and flickers on and off, so an average would invent a shape no window has.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.105, hspace=0.30, wspace=0.13)
    fig.savefig(FIG / "fig_recall_by_tubelet.png", dpi=175); plt.close(fig)


def main():
    A, B = train_domain(), v3()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_train(A); fig_v3(B); fig_recall_t(B)

    L = ["# attn 자 — 존재 판정과 위치 오차 (95 % bootstrap CI, clip 재표집 2,000 회)", "",
         "위치 오차는 전부 **L2 거리** `‖pred − truth‖₂` 의 요약 통계다 (평균·중앙·p90).", "",
         f"## {train_set()} test — 자를 배운 세트의 held-out (음성이 있는 유일한 세트)", "",
         "| 표현 | 문턱 | AUROC | recall % | precision % | specificity % | L2 평균 px | L2 중앙 px | *제외 칸: '있다' %* |",
         "|---|---|---|---|---|---|---|---|---|"]
    split_ = "excl_scenery" in A["p"]                   # training_v8 부터 제외가 두 종류 — 판과 겹침 / 구조물 속
    g = lambda t, s=1: f"{t[0]*s:.2f} (−{t[1]*s:.2f}/+{t[2]*s:.2f})"
    for r in REPS:
        au = A[r]["auc"]
        L.append(f"| `{r}` | {A[r]['thr']:+.2f} | {au[0]:.4f} (−{au[1]:.4f}/+{au[2]:.4f}) | {g(A[r]['recall'],100)} | "
                 f"{g(A[r]['precision'],100)} | {g(A[r]['specificity'],100)} | {g(A[r]['L2 mean'])} | "
                 f"{g(A[r]['L2 median'])} | *" + (f"판과 겹침 {g(A[r]['excl_prop'],100)} · 구조물 속 {g(A[r]['excl_scenery'],100)}"
                                                    if split_ else g(A[r]['occluded'], 100)) + "* |")
    L += ["", "⚠️ **제외 칸 열은 오류율이 아니다.** 그 칸은 학습에서 제외돼 정답 라벨이 없다 — "
          "자가 뭐라고 답하는지만 적은 측정값이다. 판과 겹친 물체는 보이므로 '있다' 가 자연스럽고, "
          "구조물 속 물체는 안 보이므로 '없다' 가 자연스럽다.", "",
          "## v3 — 음성이 0 개라 recall 만 정의된다. C 넷을 합쳤다", "",
          "| 표현 | P | recall % | L2 평균 px (전부) | L2 평균 px ('있다' 만) |", "|---|---|---|---|---|"]
    for r in REPS:
        for p in (4, 8, 16, 32):
            b = B[(r, p)]
            L.append(f"| `{r}` | {p} | {g(b['recall'],100)} | {g(b['all'])} | {g(b['gated'])} |")
    (FIG / "ERRORS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n→ {FIG}/fig_readout_train.png · fig_readout_v3.png · fig_recall_by_tubelet.png")


if __name__ == "__main__":
    main()
