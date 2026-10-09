#!/usr/bin/env python3
"""**정체 자 (위치 + 57-way) 를 자기 학습셋 held-out test 에서** — 존재 · 정체 · 위치 오차, p · z · h 나란히 (2026-09-26).

v3 · v11 그림보다 **먼저** 보는 그림이다. 자가 배운 곳에서 무엇을 얼마나 틀리는지를 보인 뒤에야, 그 자로 predictor 의 세계를 말한다.
(presence 자 시기의 `plot_readout_errorbars.py` → `fig_readout_train.png` 와 같은 자리. 그 스크립트는 presence 자 전용이다)

입력 — `exp_results/<R3_DECODER>/{p,z,h}/preds.npz` (학습 스크립트 `rollout2_identity_readout.py` 가 가중치와 함께 쓴 예측).
       열: xy · truth (정규화 좌표, px = v·144 + 144) · prob (57) · cls (0..55 조합 · 56 없음 · −1 제외) · train / val / test · scenario.
       학습셋 index (`data_csv/rollout_v2_<summary.train_set>/index_probe.csv`) 는 제외 칸을 나누는 데만 쓴다.

정의 (test 분할, 미래 튜블릿 8 개, 칸 = clip × 튜블릿):
  점수      s = log max_{c<56} P(c) − log P(없음).  **'있음' ⇔ s > 0 ⇔ 57-way argmax ≠ 없음** (v3 · v11 그림과 같은 규칙, 튜닝한 문턱 없음)
  양성      cls ∈ 0..55 (물체 전체가 화면 안, 판 · 구조물과 안 겹침).  음성 = cls 56 (빈 장면 또는 통째로 화면 밖).  제외 = −1
  recall    P(s > 0 | 양성)             false alarm  P(s > 0 | 음성)             precision  P(양성 | s > 0), 라벨 있는 칸 중
  조합 정답  P(argmax57 = 참 조합 | 양성)   — '없음' 이라고 하면 틀린 것으로 센다 (summary.json 의 acc57 과 같다)
  모양 · 색  조합 확률을 모양 / 색으로 더한 주변 확률의 argmax (양성 칸, '있다' 조건부 — summary 의 shape_acc · color_acc)
  위치 오차  e = ‖(xy − truth)·144 − b‖₂ px, **양성이고 s > 0 인 칸만**. b = `attn_bias_px.json` (test × 양성 평균, v3 · v11 그림과 같은 값)
            ⚠️ b 를 같은 test 에서 쟀으므로 평균 부호 오차는 정의상 0 이다. L2 는 b 를 빼도 거의 같다
  구간      clip 재표집 95 % (비율 · 평균 2,000 회, 분위수 · AUROC 600 회). 한 clip 의 8 튜블릿은 독립이 아니다

자기 검사: 조합 정답 · 모양 · 색 · b 없는 위치 오차 평균 (양성 전부) 을 summary.json 과 대조해 다르면 죽는다.

그림 (`figures/train_readout/`, 도장 `_decoder.json`):
  fig_train_overview.png     (a) 없음 vs 물체 ROC  (b) 정체 정확도  (c) 위치 오차  (d) 있음/없음 혼동 행렬
  fig_train_by_tubelet.png   미래 튜블릿 t0–t7 마다 recall · false alarm · 조합 정답 · 위치 오차
  fig_train_by_family.png    무대 가족 (plain · prop · panel · ramp · ledge · wall · edge) 마다 같은 넷
  fig_train_error_map.png    진실 위치 (18 px 칸) 마다 평균 위치 오차 — 가장자리 실패 구역이 어디인가
  fig_train_confusion.png    모양 7 × 7 · 색 8 × 8 혼동 (행 정규화)
  values.json                그림의 모든 수치

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  R3_DECODER=identity_r8 $P z_research/scripts/figures/plot_train_readout.py      # CPU 1–2 분
  (new_archive_redraw.py --steps figs 가 자와 함께 다시 그린다)
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
sys.path.insert(0, str(ROOT / "z_research/scripts/figures"))
from rollout3_paths import PRES, ARCHIVE, is_identity, stamp   # noqa: E402
from plot_readout_errorbars import ci, ci_quant, roc, auc_ci   # noqa: E402  clip 재표집 도구 (같은 규칙)

OUT = ARCHIVE / "train_readout"
REPS = tuple(r for r in ("p", "z", "h") if (PRES / r / "preds.npz").exists())   # 2026-09-28: ar 자는 p · h 뿐
COL = {"p": "#eb6834", "z": "#2a78d6", "h": "#1baf7a"}
RES, CELL, T, W = 144.0, 18.0, 8, 288
FAMS = ["plain", "prop", "panel", "ramp", "ledge", "wall", "edge"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def family(sc):
    """시나리오 → 무대 가족. `rollout2_presence_readout.family` 와 같다 (torch 를 안 불러오려고 옮겼다)."""
    for k in ("ledge", "wedge", "ramp", "wall", "edge", "prop", "panel", "empty"):
        if sc.startswith(k):
            return k
    return "plain"


def excluded_kind(S, vids):
    """제외 칸 (cls −1) 을 셋으로: 구조물 속 / 소품 · 판과 겹침 / 그 밖 (화면 가장자리에 걸림 등). (n, 8) 문자열."""
    idx = ROOT / f"data_csv/rollout_v2_{S['train_set']}/index_probe.csv"
    rows = {r["video_id"]: r for r in csv.DictReader(idx.open())}
    arr = lambda s: np.array([float(v) for v in s.split()], np.float32) if s.strip() else np.zeros(32, np.float32)
    sc = np.stack([arr(rows[v]["scenery_intersects_by_sample"]) for v in vids]) > 0
    pr = np.stack([arr(rows[v]["prop_intersects_by_sample"]) for v in vids]) > 0
    n = len(vids)
    sc = sc.reshape(n, 16, 2).any(2)[:, T:]; pr = pr.reshape(n, 16, 2).any(2)[:, T:]
    return np.where(sc, "scenery", np.where(pr, "prop", "other"))


def load(S, bias):
    none = S["none_index"]; ns, nc = len(S["shapes"]), len(S["colors"])
    out = {}
    for rep in REPS:
        d = np.load(PRES / rep / "preds.npz", allow_pickle=True)
        te = d["test"]; cls = d["cls"][te]; P = d["prob"][te].astype(np.float32)
        lp = np.log(np.clip(P, 1e-12, None))
        s = lp[..., :none].max(-1) - lp[..., none]
        p57 = P.argmax(-1)
        pm = P[..., :none].reshape(*P.shape[:-1], ns, nc)
        e_raw = np.linalg.norm((d["xy"][te] - d["truth"][te]) * RES, axis=-1)
        e = np.linalg.norm((d["xy"][te] - d["truth"][te]) * RES - np.array(bias[rep], np.float32), axis=-1)
        out[rep] = dict(cls=cls, s=s, say=s > 0, p57=p57, ps=pm.sum(-1).argmax(-1), pc=pm.sum(-2).argmax(-1),
                        e=e, e_raw=e_raw, truth=d["truth"][te] * RES + RES, vid=d["video_id"][te], scen=d["scenario"][te])
    v0 = out["p"]["vid"]
    assert all((out[r]["vid"] == v0).all() for r in REPS), "p · z · h 의 test clip 순서가 다르다"
    return out


def selfcheck(R, S):
    """summary.json 의 같은 정의 수치와 대조 (float16 확률 저장이라 1e-3 까지 허용)."""
    none, nc = S["none_index"], len(S["colors"])
    for rep in REPS:
        r, s = R[rep], S["reps"][rep]
        o = (r["cls"] >= 0) & (r["cls"] < none)
        got = dict(acc57=(r["p57"][o] == r["cls"][o]).mean(), shape_acc=(r["ps"][o] == r["cls"][o] // nc).mean(),
                   color_acc=(r["pc"][o] == r["cls"][o] % nc).mean(), center_mean=r["e_raw"][o].mean() / CELL)
        for k, v in got.items():
            if abs(v - s[k]) > 1e-3:
                raise SystemExit(f"자기 검사 실패: {rep} {k} {v:.5f} vs summary {s[k]:.5f}")
    print("자기 검사: 조합 · 모양 · 색 · 위치 평균이 summary.json 과 일치 ✓")


def metrics(r, none, mask=None):
    """한 묶음 (mask 로 고른 칸) 의 recall · false alarm · 조합 · 위치 오차, clip 재표집 구간."""
    n = len(r["cls"]); cl = np.repeat(np.arange(n)[:, None], T, 1)
    M = np.ones_like(r["say"]) if mask is None else mask
    pos = M & (r["cls"] >= 0) & (r["cls"] < none); neg = M & (r["cls"] == none); g = pos & r["say"]
    one = lambda m: np.ones(int(m.sum()))
    nan3 = (np.nan, np.nan, np.nan)
    return dict(
        n_pos=int(pos.sum()), n_neg=int(neg.sum()),
        recall=ci(r["say"][pos].astype(float), one(pos), cl[pos]) if pos.any() else nan3,
        false_alarm=ci(r["say"][neg].astype(float), one(neg), cl[neg]) if neg.any() else nan3,
        combo=ci((r["p57"][pos] == r["cls"][pos]).astype(float), one(pos), cl[pos]) if pos.any() else nan3,
        L2=ci(r["e"][g], one(g), cl[g]) if g.any() else nan3)


def overall(R, S, excl):
    none, nc = S["none_index"], len(S["colors"])
    A = {}
    for rep in REPS:
        r = R[rep]; n = len(r["cls"]); cl = np.repeat(np.arange(n)[:, None], T, 1)
        pos = (r["cls"] >= 0) & (r["cls"] < none); neg = r["cls"] == none; g = pos & r["say"]
        one = lambda m: np.ones(int(m.sum()))
        a = metrics(r, none)
        lab = pos | neg; sl = lab & r["say"]
        a["precision"] = ci(pos[sl].astype(float), one(sl), cl[sl])
        a["shape"] = ci((r["ps"][pos] == r["cls"][pos] // nc).astype(float), one(pos), cl[pos])
        a["color"] = ci((r["pc"][pos] == r["cls"][pos] % nc).astype(float), one(pos), cl[pos])
        a["L2 median"] = ci_quant(r["e"][g], cl[g], 50)
        a["L2 p90"] = ci_quant(r["e"][g], cl[g], 90)
        a["roc"] = roc(r["s"][lab], pos[lab])
        a["auc"] = auc_ci(r["s"][lab], pos[lab], cl[lab])
        a["op"] = (float(r["say"][neg].mean()), float(r["say"][pos].mean()))          # 작동점 (fpr, tpr) at s > 0
        a["cm"] = np.array([[int((r["say"] & pos).sum()), int((~r["say"] & pos).sum())],
                            [int((r["say"] & neg).sum()), int((~r["say"] & neg).sum())]])
        ex = r["cls"] < 0
        a["excluded"] = {k: (int((ex & (excl == k)).sum()), float(r["say"][ex & (excl == k)].mean()) if (ex & (excl == k)).any() else None)
                         for k in ("prop", "scenery", "other")}
        A[rep] = a
    return A


# ---------------------------------------------------------------- 그림

def bars(ax, keys, get, ylab, pct=False, short=None):
    w, x = 0.26, np.arange(len(keys))
    for i, rep in enumerate(REPS):
        v = np.array([get(rep, k) for k in keys], float); s = 100 if pct else 1
        ax.bar(x + (i - 1) * w, v[:, 0] * s, w, color=COL[rep], label=rep, edgecolor="white", lw=0.6)
        ax.errorbar(x + (i - 1) * w, v[:, 0] * s, yerr=np.nan_to_num(v[:, 1:].T) * s, fmt="none",
                    ecolor="#333333", elinewidth=1.0, capsize=2.5)
    ax.set_xticks(x); ax.set_xticklabels([(short or {}).get(k, k) for k in keys], fontsize=8.5)
    ax.set_ylabel(ylab); ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=0); ax.grid(axis="y", lw=0.3, alpha=0.4); ax.set_axisbelow(True)


def cell_line(ax, x0=-0.45):
    ax.axhline(CELL, color="#666666", lw=0.8, ls="--")
    ax.text(x0, CELL + 0.4, "1 cell = 18 px", fontsize=7.5, color="#666666", ha="left", va="bottom")


def conf(ax, cm, rep):
    rs = cm.sum(1, keepdims=True); frac = cm / np.maximum(rs, 1)
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=COL[rep], alpha=0.10 + 0.55 * frac[i, j], edgecolor="white", lw=1.5))
            ax.text(j + 0.5, 1.5 - i, f"{cm[i, j]:,}\n{100*frac[i, j]:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold" if i == j else "normal")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["says\npresent", "says\nabsent"], fontsize=8.5)
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(["truly\npresent", "truly\nabsent"], fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    tp, fn, fp, tn = cm.ravel()
    ax.set_title(f"{rep}\nrecall {100*tp/(tp+fn):.1f}%  ·  precision {100*tp/max(tp+fp,1):.1f}%",
                 fontsize=9.5, fontweight="bold", color=COL[rep])


def fig_overview(A, S, dec):
    f = plt.figure(figsize=(12.0, 8.0))
    gs = f.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.62, wspace=0.36)
    a = f.add_subplot(gs[0, 0])
    for rep in REPS:
        fp_, tp_, _ = A[rep]["roc"]; p_, lo, hi = A[rep]["auc"]
        a.plot(fp_, tp_, color=COL[rep], lw=1.8, label=f"{rep}  AUROC {p_:.4f} (−{lo:.4f}/+{hi:.4f})")
        a.scatter(*A[rep]["op"], s=40, color=COL[rep], edgecolor="black", lw=0.7, zorder=5)
    a.set_xlim(0, 0.05); a.set_ylim(0.95, 1.001)
    a.set_xlabel("false alarm rate (truly absent, says present)"); a.set_ylabel("recall")
    a.legend(frameon=False, fontsize=7.5, loc="lower right")
    a.spines[["top", "right"]].set_visible(False)
    a.set_title("(a)  object vs none — ROC (zoomed)\ndot = operating point (argmax ≠ none)", fontsize=9.5, fontweight="bold")

    b = f.add_subplot(gs[0, 1])
    keys = ["combo", "shape", "color"]
    bars(b, keys, lambda r, k: A[r][k], "accuracy on truly-present tubelets (%)", pct=True,
         short={"combo": "combo (56)", "shape": "shape (7)", "color": "colour (8)"})
    ch = S["chance"]
    for i, k in enumerate(keys):
        b.plot([i - 0.42, i + 0.42], [100 * ch[k]] * 2, color="#555555", lw=1.0, ls=":")
    b.set_ylim(0, 102)
    b.set_title("(b)  identity  (dotted = chance)\ncombo counts 'none' as wrong", fontsize=9.5, fontweight="bold")

    c = f.add_subplot(gs[0, 2])
    bars(c, ["L2", "L2 median", "L2 p90"], lambda r, k: A[r][k], "position L2 error (px)",
         short={"L2": "mean", "L2 median": "median", "L2 p90": "90th pct"})
    cell_line(c)
    c.set_title("(c)  position error\n(truly present and says present)", fontsize=9.5, fontweight="bold")

    for j, rep in enumerate(REPS):
        conf(f.add_subplot(gs[1, j]), A[rep]["cm"], rep)
    f.text(0.5, 0.075, "(d)   present / absent at the operating rule  (57-way argmax ≠ none — no tuned threshold)",
           ha="center", fontsize=10, fontweight="bold")
    name = {"prop": "overlapping a prop or panel", "scenery": "inside a structure (ledge / ramp / wall)",
            "other": "crossing the frame edge (partly visible)"}
    parts = [f"{name[k]} (n={A['p']['excluded'][k][0]:,}): " + "  ".join(f"{r} {100*A[r]['excluded'][k][1]:.0f}%" for r in REPS)
             for k in ("prop", "scenery", "other") if A["p"]["excluded"][k][0] > 0]
    f.text(0.5, 0.022, "Excluded tubelets carry NO label and appear in no cell above.  Share the readout calls 'present':\n"
           + "   ·   ".join(parts), ha="center", fontsize=8.3)
    f.suptitle(f"{dec} on {S['train_set']} test — the readout's own held-out split (block level, appearance held out, trajectories shared)\n"
               "error bars = 95% bootstrap CI over clips   ·   position error = ‖pred − truth‖ minus the readout's mean offset", fontsize=9.5)
    f.subplots_adjust(left=0.065, right=0.98, top=0.86, bottom=0.14)
    f.savefig(OUT / "fig_train_overview.png", dpi=170); plt.close(f)


def four_panel(fname, xs, get, xlabel, title, xt=None, bar=False):
    """recall · false alarm · 조합 · 위치 오차 네 칸. get(rep, x) → metrics dict. 막대 (가족) 는 칸이 좁아 2 × 2 로 놓는다."""
    f, ax = plt.subplots(2, 2, figsize=(13.0, 8.2)) if bar else plt.subplots(1, 4, figsize=(15.0, 3.9))
    ax = np.ravel(ax)
    spec = [("recall", "recall (%)", True, "(a)  recall  (truly present → says present)"),
            ("false_alarm", "false alarm (%)", True, "(b)  false alarm  (truly absent → says present)"),
            ("combo", "combo accuracy (%)", True, "(c)  combo correct  (truly present)"),
            ("L2", "position L2 error (px)", False, "(d)  position error  (present, says present)")]
    for a, (k, yl, pct, tt) in zip(ax, spec):
        if bar:
            bars(a, xs, lambda r, x: get(r, x)[k], yl, pct=pct)
            a.set_xticklabels(xt or xs, fontsize=7.8)
        else:
            t = np.arange(len(xs))
            for rep in REPS:
                v = np.array([get(rep, x)[k] for x in xs], float) * (100 if pct else 1)
                a.plot(t, v[:, 0], color=COL[rep], lw=1.8, marker="o", ms=3.5, label=rep)
                a.fill_between(t, v[:, 0] - v[:, 1], v[:, 0] + v[:, 2], color=COL[rep], alpha=0.2, lw=0)
            a.set_xticks(t); a.set_xticklabels(xt or xs)
            a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", lw=0.3, alpha=0.4)
            a.set_ylabel(yl)
        a.set_xlabel(xlabel); a.set_title(tt, fontsize=9.3, fontweight="bold")
        if k == "L2":
            cell_line(a, x0=-0.4); a.set_ylim(0, max(a.get_ylim()[1], CELL + 3))
        elif k == "false_alarm":
            a.set_ylim(0, max(a.get_ylim()[1], 2))
        else:
            a.set_ylim(min(a.get_ylim()[0], 50) if not bar else 0, 101)
    h_, l_ = ax[0].get_legend_handles_labels()
    f.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    f.suptitle(title, fontsize=10)
    if bar:
        f.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.12, wspace=0.18, hspace=0.55)
    else:
        f.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.25, wspace=0.28)
    f.savefig(OUT / fname, dpi=170); plt.close(f)


def fig_error_map(R, S, dec):
    """진실 위치 18 px 칸 (16 × 16) 마다 평균 위치 오차. 칸당 20 칸 미만은 비운다."""
    none = S["none_index"]; nb = W // int(CELL)
    f, ax = plt.subplots(1, 4, figsize=(15.0, 4.2))
    maps = {}
    for j, rep in enumerate(REPS):
        r = R[rep]; g = (r["cls"] >= 0) & (r["cls"] < none) & r["say"]
        ij = np.clip((r["truth"][g] // CELL).astype(int), 0, nb - 1)
        cnt = np.zeros((nb, nb)); sm = np.zeros((nb, nb))
        np.add.at(cnt, (ij[:, 1], ij[:, 0]), 1); np.add.at(sm, (ij[:, 1], ij[:, 0]), r["e"][g])
        m = np.where(cnt >= 20, sm / np.maximum(cnt, 1), np.nan); maps[rep] = (m, cnt)
    vmax = np.nanpercentile(np.concatenate([maps[r][0].ravel() for r in REPS]), 98)
    cnt = maps["p"][1]
    im0 = ax[0].imshow(np.where(cnt > 0, cnt, np.nan), extent=(0, W, W, 0), cmap="Greys")
    ax[0].set_title("tubelets per 18 px cell\n(truly present and p says present)", fontsize=9.3, fontweight="bold")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.03)
    for j, rep in enumerate(REPS):
        a = ax[j + 1]
        im = a.imshow(maps[rep][0], extent=(0, W, W, 0), cmap="viridis", vmin=0, vmax=vmax)
        a.set_title(f"{rep}: mean position L2 error (px)\nby true position", fontsize=9.3, fontweight="bold", color=COL[rep])
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    for a in ax:
        a.set_xlim(0, W); a.set_ylim(W, 0); a.set_xlabel("true x (px)"); a.set_ylabel("true y (px)")
        for v in (36, W - 36):
            a.axvline(v, color="white", lw=0.8, ls="--"); a.axhline(v, color="white", lw=0.8, ls="--")
        a.axhline(152, color="#eb6834", lw=1.0, ls=":")
    ax[0].text(4, 170, "y = 152: floor line (v11 static / flat objects,\nv3 rolling objects) = lowest training row",
               fontsize=7, color="#eb6834", va="top")
    f.suptitle(f"{dec} on {S['train_set']} test — where on screen the position readout errs  "
               "(dashed = 36 px from the frame edge; cells with < 20 tubelets left blank)", fontsize=10)
    f.subplots_adjust(left=0.04, right=0.98, top=0.80, bottom=0.12, wspace=0.35)
    f.savefig(OUT / "fig_train_error_map.png", dpi=170); plt.close(f)
    return {rep: dict(edge36=float(np.nanmean(np.r_[maps[rep][0][:2].ravel(), maps[rep][0][-2:].ravel(),
                                                   maps[rep][0][2:-2, :2].ravel(), maps[rep][0][2:-2, -2:].ravel()])),
                      inner=float(np.nanmean(maps[rep][0][2:-2, 2:-2]))) for rep in REPS}


def fig_confusion(R, S, dec):
    none, ns, nc = S["none_index"], len(S["shapes"]), len(S["colors"])
    f, ax = plt.subplots(2, 3, figsize=(13.0, 8.6))
    out = {}
    for j, rep in enumerate(REPS):
        r = R[rep]; o = (r["cls"] >= 0) & (r["cls"] < none)
        for i, (name, true, pred, lab) in enumerate((("shape", r["cls"][o] // nc, r["ps"][o], S["shapes"]),
                                                     ("colour", r["cls"][o] % nc, r["pc"][o], S["colors"]))):
            k = len(lab); cm = np.bincount(true * k + pred, minlength=k * k).reshape(k, k)
            fr = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            a = ax[i, j]; a.imshow(fr, cmap="Blues", vmin=0, vmax=1)
            for y in range(k):
                for x in range(k):
                    if fr[y, x] >= 0.005:
                        a.text(x, y, f"{100*fr[y, x]:.0f}", ha="center", va="center", fontsize=7,
                               color="white" if fr[y, x] > 0.6 else "black")
            a.set_xticks(range(k)); a.set_xticklabels(lab, rotation=45, ha="right", fontsize=7.5)
            a.set_yticks(range(k)); a.set_yticklabels(lab, fontsize=7.5)
            a.set_xlabel("read as"); a.set_ylabel("true")
            a.set_title(f"{rep} — {name}: {100*np.trace(cm)/cm.sum():.1f}%", fontsize=9.5, fontweight="bold", color=COL[rep])
            out[f"{rep}|{name}"] = dict(acc=float(np.trace(cm) / cm.sum()), recall_by_class=dict(zip(lab, np.round(np.diag(fr), 3).tolist())))
    f.suptitle(f"{dec} on {S['train_set']} test — shape and colour confusion (truly-present tubelets, row-normalised %)\n"
               "shape / colour = argmax of the 56 combo probabilities summed over the other attribute", fontsize=10)
    f.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08, hspace=0.45, wspace=0.35)
    f.savefig(OUT / "fig_train_confusion.png", dpi=170); plt.close(f)
    return out


def fig_summary(R, S, A):
    """논문 판 한 장 (2026-09-28 사용자 지시): (a) 위치 L2 오차 평균 · (b) 57-class 정확도, p · z · h.
    (a) = '있음' 이면서 실제로 있는 칸, 치우침 뺀 L2 (overview (c) 의 mean 과 같은 값).
    (b) = 라벨 있는 칸 전부 (56 조합 + 없음) 에서 57-way argmax 가 라벨과 같은 비율. 점선 = 늘 '없음' 이라고 답하는 기준선 (최다 클래스 비율)."""
    none = S["none_index"]
    paper = {"font.family": "Nimbus Roman", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 7,
             "pdf.fonttype": 42, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}
    acc = {}
    for rep in REPS:
        r = R[rep]; n = len(r["cls"]); cl = np.repeat(np.arange(n)[:, None], T, 1); lab = r["cls"] >= 0
        acc[rep] = ci((r["p57"][lab] == r["cls"][lab]).astype(float), np.ones(int(lab.sum())), cl[lab])
    maj = float((R["p"]["cls"][R["p"]["cls"] >= 0] == none).mean())
    x = np.arange(len(REPS))
    with plt.rc_context(paper):
        f, ax = plt.subplots(1, 2, figsize=(3.5, 2.3))
        for a, vals, ylab, fmt in ((ax[0], {r_: A[r_]["L2"] for r_ in REPS}, "Position L2 error (px)", "{:.1f}"),
                                   (ax[1], {r_: tuple(100 * np.array(acc[r_])) for r_ in REPS}, "57-class accuracy (%)", "{:.1f}")):
            v = np.array([vals[r_] for r_ in REPS], float)
            a.bar(x, v[:, 0], 0.62, color=[COL[r_] for r_ in REPS], edgecolor="white", lw=0.6)
            a.errorbar(x, v[:, 0], yerr=v[:, 1:].T, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
            for xi, (m_, _, hi) in zip(x, v):
                a.text(xi, m_ + hi + (0.4 if a is ax[0] else 1.5), fmt.format(m_), ha="center", va="bottom", fontsize=7)
            a.set_xticks(x); a.set_xticklabels(REPS); a.set_ylabel(ylab)
        ax[0].axhline(CELL, color="#777777", lw=0.7, ls="--"); ax[0].text(2.45, CELL + 0.3, "1 cell", fontsize=6.5, color="#666666", ha="right", va="bottom")
        ax[0].set_ylim(0, 20)
        ax[1].axhline(100 * maj, color="#777777", lw=0.7, ls=":", label=f"always 'none' ({100 * maj:.1f}%)")
        ax[1].set_ylim(0, 118); ax[1].set_yticks([0, 20, 40, 60, 80, 100])
        ax[1].legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), fontsize=6.5, handlelength=1.6, borderaxespad=0.0)
        for k, a in enumerate(ax):
            a.text(0.5, -0.2, f"({'ab'[k]})", transform=a.transAxes, ha="center", va="top", fontsize=9)
        f.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.2, wspace=0.45)
        f.savefig(OUT / "fig_train_summary.png", dpi=300); f.savefig(OUT / "fig_train_summary.pdf"); plt.close(f)
    return {rep: dict(L2_mean=[round(float(v), 3) for v in A[rep]["L2"]], acc57_all=[round(float(v), 4) for v in acc[rep]]) for rep in REPS} | {"majority_none": round(maj, 4)}


def main():
    if not is_identity(PRES):
        raise SystemExit(f"{PRES.name} 는 정체 자가 아니다. presence 자는 plot_readout_errorbars.py 를 쓴다")
    S = json.loads((PRES / "summary.json").read_text()); none = S["none_index"]
    S["chance"] = S["chance"] if "chance" in S else S["reps"]["p"]["chance"]
    bias = json.loads((PRES / "attn_bias_px.json").read_text())
    dec = f"readout '{PRES.name}' (position + 57-way identity)"
    R = load(S, bias); selfcheck(R, S)
    excl = excluded_kind(S, R["p"]["vid"])
    OUT.mkdir(parents=True, exist_ok=True)

    A = overall(R, S, excl); fig_overview(A, S, dec)
    summ = fig_summary(R, S, A)
    TT = {rep: [metrics(R[rep], none, np.eye(T, dtype=bool)[t][None, :].repeat(len(R[rep]["cls"]), 0)) for t in range(T)] for rep in REPS}
    four_panel("fig_train_by_tubelet.png", list(range(T)), lambda r, t: TT[r][t], "future tubelet",
               f"{dec} on {S['train_set']} test — by future tubelet (context 16 samples, future 8 tubelets)", xt=[f"t{t}" for t in range(T)])
    fam = np.array([family(s) for s in R["p"]["scen"]])
    fams = [x for x in FAMS if (fam == x).any()]
    FF = {rep: {x: metrics(R[rep], none, (fam == x)[:, None].repeat(T, 1)) for x in fams} for rep in REPS}
    nfam = {x: (FF["p"][x]["n_pos"], FF["p"][x]["n_neg"]) for x in fams}
    four_panel("fig_train_by_family.png", fams, lambda r, x: FF[r][x], "stage family",
               f"{dec} on {S['train_set']} test — by stage family  (x labels: family, present / absent tubelets)",
               xt=[f"{x}\n{nfam[x][0]:,} / {nfam[x][1]:,}" for x in fams], bar=True)
    emap = fig_error_map(R, S, dec)
    conf_ = fig_confusion(R, S, dec)

    rnd = lambda t: [round(float(v), 4) for v in t]
    vals = dict(decoder=PRES.name, train_set=S["train_set"], n_test_clip=int(len(R["p"]["cls"])),
                overall={rep: {k: (rnd(v) if isinstance(v, tuple) and len(v) == 3 else v) for k, v in A[rep].items()
                               if k not in ("roc", "cm", "op")} | dict(cm=A[rep]["cm"].tolist(), op=A[rep]["op"])
                         for rep in REPS},
                by_tubelet={rep: [{k: (rnd(v) if isinstance(v, tuple) else v) for k, v in m.items()} for m in TT[rep]] for rep in REPS},
                by_family={rep: {x: {k: (rnd(v) if isinstance(v, tuple) else v) for k, v in m.items()} for x, m in FF[rep].items()} for rep in REPS},
                error_map_edge_vs_inner=emap, confusion=conf_, bias_px=bias, summary_fig=summ)
    (OUT / "values.json").write_text(json.dumps(vals, indent=1, ensure_ascii=False))
    stamp(OUT, PRES, [PRES / r / "preds.npz" for r in REPS])
    for rep in REPS:
        a = A[rep]
        print(f"{rep}: recall {100*a['recall'][0]:.2f} · false alarm {100*a['false_alarm'][0]:.2f} · precision {100*a['precision'][0]:.2f} · "
              f"combo {100*a['combo'][0]:.1f} · shape {100*a['shape'][0]:.1f} · colour {100*a['color'][0]:.1f} · "
              f"L2 mean {a['L2'][0]:.1f} / med {a['L2 median'][0]:.1f} / p90 {a['L2 p90'][0]:.1f} px · AUROC {a['auc'][0]:.4f} · "
              f"edge36 {emap[rep]['edge36']:.1f} vs inner {emap[rep]['inner']:.1f} px")
    print(OUT)


if __name__ == "__main__":
    main()
