#!/usr/bin/env python3
"""IntPhysGen v11 — 가림 타이밍별로 p 의 미래에 **물체가 있나 (있음/없음) · 어디 있나 (위치 L2)**, 미래 튜블릿 0~7 전부.

  열 = 가림 조건 (visible · occluded mid · occluded last), 선 = 운동 (static · flat · ramp) (사용자 지시 2026-09-24)
  한 줄 = 그 튜블릿에서 p 를 '있다' 로 읽은 **clip 비율 (%)**. 50 % 점선은 참고선이다
  (2026-09-27 사용자 지적: 이전 판은 과반이면 있음 · 아니면 없음의 **이진**이라 50 % 근처 비율이 깜빡이는 것처럼 보였다.
   이진 판은 figures/v11_presence/_superseded/binary_majority_2026-09-27/)
  조건 = visible (k=0) · 문맥 중간 가림 mid (k=1~4) · **문맥 끝 가림 last (k=1~4)**
  (2026-09-25 사용자 지시: 위치 L2 줄은 뺀다 · k=1 도 같이 그린다. 이전 판은 k ≥ 2 + L2 줄)
  튜블릿을 빼는 규칙은 없다 — 가려진 칸 · 화면 가장자리 칸 · 화면 밖 칸 (움직이는 물체의 t7) 도 전부 센다 (사용자 지시).
  자 = R3_DECODER (p 자기 자). '있다' 규칙은 rollout3_paths.present_thr (정체 자 = 57-way argmax ≠ 없음). 가능 변이 · 물체 있는 clip 만.

⚠️ 자마다 가림막 오탐이 다르다. presence 자 · 옛 identity 자는 판 빈 장면에서도 '있다' 를 냈다 (7–99 %).
   지금 자 identity_r8 은 0 % 다 (audit/training_v8/panel_false_alarm_identity_r8.json).
⚠️ 화면 가장자리 · 화면 밖 칸 (이동 물체의 t7) 은 자의 정의 밖이다. 그래도 사용자 지시로 빼지 않고 그린다.

  $P z_research/scripts/figures/plot_v11_presence.py   → z_research/RollOutV3/figures/v11_presence/fig_v11_presence.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, EXP, TAG, V11, V11_MID, ARCHIVE, check, present_thr, is_identity, stamp       # noqa: E402
DEC = "decoder: position + identity (56 shape x colour + none)" if is_identity(PRES) else "decoder: position + presence"
SFX = f"_{TAG}" if TAG else ""                     # R3_OUT=identity → exp_results/v11_identity · v11_mid_identity

OUT = ARCHIVE / "v11_presence"                   # R3_FIGROOT 로 바꾼다 (기본 RollOutV3/figures)
RES, TC, TP, CELL, KMIN = 144.0, 8, 8, 18.0, 1
COLS = {"static": "#2a78d6", "flat": "#1baf7a", "ramp": "#eb6834"}
LAB = {"visible": "visible (k=0)", "mid": "occluded mid-context (k=1-4)", "last": "occluded at context end (k=1-4)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def motion(cond):
    return "static" if cond.startswith("static") else ("flat" if "flat" in cond else "ramp")


def groups():
    """→ {(timing, motion): (xy (n,8,2) 절대 px, say (n,8), 진실 (n,8,2) 절대 px)}"""
    thr = present_thr("p", PRES)                                          # 자 종류마다 뜻이 다르다 (rollout3_paths)
    bias = np.array(json.loads((PRES / "attn_bias_px.json").read_text())["p"])
    out = {}
    for name in (V11.name, V11_MID.name):
        z = np.load(EXP / name / "readings.npz", allow_pickle=True); check(z, PRES)
        n = len(z["video_id"]); k = z["sym_k"].astype(int)
        T = (z["truth"].reshape(n, 16, 2, 2).mean(2) * RES + RES)[:, TC:]
        tim = np.where(k == 0, "visible", "mid" if name.startswith("v11_mid") else "last")   # 이름에 꼬리표가 붙을 수 있다 (v11_identity)
        mot = np.array([motion(c) for c in z["condition"]])
        v = z["p"][:, -TP:]; xy = v[..., 0:2] * RES + RES - bias; say = v[..., 2] > thr
        for tg in np.unique(tim):
            for mo in ("static", "flat", "ramp"):
                m = z["obj"] & (tim == tg) & (mot == mo) & ((k == 0) | (k >= KMIN))
                if m.any():
                    out[(tg, mo)] = (xy[m], say[m], T[m])
    return out


def main():
    G = groups(); OUT.mkdir(parents=True, exist_ok=True)
    t = np.arange(TP); vals = {}
    f_, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True, squeeze=False)
    for j, tg in enumerate(("visible", "mid", "last")):
        for mo in ("static", "flat", "ramp"):
            xy, say, T = G[(tg, mo)]; col = COLS[mo]
            frac = say.mean(0)
            err = np.array([np.linalg.norm(xy[say[:, i], i] - T[say[:, i], i], axis=-1).mean() if say[:, i].any() else np.nan for i in t])
            ax[0, j].plot(t, 100 * frac, color=col, lw=2, marker="o", ms=4)
            vals[f"{mo}|{tg}"] = dict(n=int(len(say)), present_frac=[round(float(x), 3) for x in frac],
                                      l2=[None if not np.isfinite(x) else round(float(x), 1) for x in err])
        ax[0, j].set_title(LAB[tg], fontsize=11, fontweight="bold")
        ax[0, j].set_ylim(-3, 103); ax[0, j].set_yticks([0, 25, 50, 75, 100]); ax[0, j].axhline(50, color="#bbbbbb", lw=0.8, ls=":")
        ax[0, j].set_xlabel("future tubelet (6 raw frames each)")
        for a in ax[:, j]:
            a.spines[["top", "right"]].set_visible(False); a.set_xticks(t); a.set_xlim(-0.4, TP - 0.6)
    ax[0, 0].set_ylabel("clips where p's future\nhas the object (%)")
    hd = [Line2D([], [], color=COLS[m], lw=2, marker="o", ms=4, label=m) for m in ("static", "flat", "ramp")]
    f_.legend(handles=hd, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    f_.suptitle(f"IntPhysGen v11 · context 16 / prediction 16 samples · is the object in predictor p's future? (p's own readout) · {DEC}", fontsize=10.5)
    f_.tight_layout(rect=(0, 0.1, 1, 0.92))
    f_.savefig(OUT / "fig_v11_presence.png", dpi=150); plt.close(f_)
    (OUT / "values.json").write_text(json.dumps(vals, indent=1))
    stamp(OUT, PRES, [V11 / "readings.npz", V11_MID / "readings.npz"])   # 어느 자로 그렸나 (_decoder.json)
    print(OUT)


def by_k():
    """k 별 판 — **visible · k=1 · k=2 · k=3 · k=4 를 한 그림에** (사용자 지시 2026-09-25). 행 = last 가림 · mid 가림, 선 = 운동.
    각 행 첫 칸은 같은 visible (k=0) 을 비교용으로 다시 그린다.
    회색 칸 = 그 k 에서 물체가 **실제로 가려진** 미래 튜블릿 (진회색 = 두 샘플 모두, 연회색 = 한 샘플).
    last k 는 미래 샘플 16..16+k−1 이 가려진다 → k=1 t0 절반 · k=2 t0 · k=3 t0 + t1 절반 · k=4 t0~t1. mid 는 미래에 가림이 없다."""
    from matplotlib.patches import Patch
    thr = present_thr("p", PRES)                                          # 자 종류마다 뜻이 다르다 (rollout3_paths)
    D = {}
    for timing, name in (("last", V11.name), ("mid", V11_MID.name)):
        z = np.load(EXP / name / "readings.npz", allow_pickle=True); check(z, PRES)
        n = len(z["video_id"])
        D[timing] = dict(k=z["sym_k"].astype(int), obj=z["obj"], mot=np.array([motion(c) for c in z["condition"]]),
                         say=z["p"][:, -TP:, 2] > thr, hid=z["hidden"].reshape(n, 16, 2)[:, TC:].sum(-1))
    t = np.arange(TP); vals = {}
    f_, ax = plt.subplots(2, 5, figsize=(19, 6.6), sharex=True, sharey=True)
    for r, timing in enumerate(("last", "mid")):
        for j, kk in enumerate((0, 1, 2, 3, 4)):
            d = D["last" if kk == 0 else timing]                                 # visible 은 v11 (late 판) 의 k=0
            a = ax[r, j]
            for mo in ("static", "flat", "ramp"):
                m = d["obj"] & (d["k"] == kk) & (d["mot"] == mo)
                frac = d["say"][m].mean(0)
                a.plot(t, 100 * frac, color=COLS[mo], lw=2, marker="o", ms=4)
                vals[f"{'visible' if kk == 0 else timing}|k{kk}|{mo}"] = dict(n=int(m.sum()), present_frac=[round(float(x), 3) for x in frac])
            hk = d["hid"][d["obj"] & (d["k"] == kk)].max(0)
            for i in t:
                if hk[i] > 0:
                    a.axvspan(i - 0.5, i + 0.5, color="#bbbbbb" if hk[i] == 2 else "#e3e3e3", lw=0, zorder=0)
            a.spines[["top", "right"]].set_visible(False); a.set_xticks(t); a.set_xlim(-0.5, TP - 0.5)
            a.set_ylim(-3, 103); a.set_yticks([0, 25, 50, 75, 100]); a.axhline(50, color="#bbbbbb", lw=0.8, ls=":")
            if kk == 0:
                a.set_title("visible (k=0)", fontsize=11, fontweight="bold")
            else:
                a.set_title(f"{'occluded at context end' if timing == 'last' else 'occluded mid-context'}, k={kk}", fontsize=10.5, fontweight="bold")
            if r == 1: a.set_xlabel("future tubelet (6 raw frames each)")
        ax[r, 0].set_ylabel("clips where p's future\nhas the object (%)")
    hd = [Line2D([], [], color=COLS[m], lw=2, marker="o", ms=4, label=m) for m in ("static", "flat", "ramp")]
    hd += [Patch(color="#bbbbbb", label="object hidden (both samples)"), Patch(color="#e3e3e3", label="object hidden (one sample)")]
    f_.legend(handles=hd, loc="lower center", ncol=5, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    f_.suptitle(f"IntPhysGen v11 · is the object in predictor p's future? by occlusion length k (top: occluded at context end, bottom: mid-context) · p's own readout · {DEC}", fontsize=11)
    f_.tight_layout(rect=(0, 0.06, 1, 0.95))
    f_.savefig(OUT / "fig_v11_presence_by_k.png", dpi=150); plt.close(f_)
    (OUT / "values_by_k.json").write_text(json.dumps(vals, indent=1))
    stamp(OUT, PRES, [V11 / "readings.npz", V11_MID / "readings.npz"])
    return vals


def identity_grid():
    """3 × 3 (2026-09-27 사용자 지시). 열 = '있음' 이면서 (a) 모양 유지 · (b) 색 유지 · (c) 모양과 색 유지, 행 = 가림 없음 · 문맥 중간 가림 · 문맥 끝 가림.
    '있음' = 57-way argmax ≠ 없음. 유지 = 그 argmax 조합의 모양 (색) 이 문맥 물체의 모양 (색) 과 같다 → (c) ⊆ (a), (b) ⊆ '있음'.
    분모 = 그 행 · 운동의 **물체 clip 전부** (1,568). 회색 칸 = 과반 clip 에서 물체가 실제로 가려진 튜블릿.
    (2026-09-27 사용자 지시로 '있음' 전체 점선은 뺐다 — 그 값은 fig_v11_presence 에 있다)
    정체 자만. 논문 판 (Nimbus Roman · 7.0 in · PDF + PNG)."""
    if not is_identity(PRES):
        raise SystemExit("정체 자 (identity) 에서만 그린다")
    S = json.loads((PRES / "summary.json").read_text()); none = S["none_index"]; SH, CO = S["shapes"], S["colors"]
    Z = {"last": np.load(EXP / V11.name / "readings.npz", allow_pickle=True), "mid": np.load(EXP / V11_MID.name / "readings.npz", allow_pickle=True)}
    for z in Z.values():
        check(z, PRES)
    rows = (("visible", "last", [0], "Never occluded"), ("mid", "mid", [1, 2, 3, 4], "Occluded mid-context"),
            ("last", "last", [1, 2, 3, 4], "Occluded at context end"))
    cols = (("shape", "Present, shape kept"), ("color", "Present, colour kept"), ("both", "Present, shape and colour kept"))
    paper = {"font.family": "Nimbus Roman", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
             "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6, "legend.frameon": False,
             "axes.spines.top": False, "axes.spines.right": False}
    t = np.arange(TP); vals = {}
    with plt.rc_context(paper):
        f, ax = plt.subplots(3, 3, figsize=(7.0, 6.3), sharex=True, sharey=True)
        for i, (tg, src, ks, rlab) in enumerate(rows):
            z = Z[src]; n = len(z["video_id"]); k = z["sym_k"].astype(int)
            mot = np.array([motion(c) for c in z["condition"]])
            top = z["p_prob"][:, -TP:].argmax(-1); say = top != none
            s_true = np.array([SH.index(x) if x in SH else -1 for x in z["shape_pre"]])[:, None]
            c_true = np.array([CO.index(x) if x in CO else -1 for x in z["color_pre"]])[:, None]
            keep = {"shape": say & (top // 8 == s_true), "color": say & (top % 8 == c_true), "both": say & (top // 8 == s_true) & (top % 8 == c_true)}
            hid = z["hidden"].reshape(n, 16, 2)[:, TC:].any(-1)
            for j, (key, clab) in enumerate(cols):
                a = ax[i, j]
                g_all = z["obj"] & np.isin(k, ks)
                for tt_ in t:
                    if hid[g_all, tt_].mean() > 0.5:
                        a.axvspan(tt_ - 0.5, tt_ + 0.5, color="#e6e6e6", lw=0, zorder=0)
                for mo in ("static", "flat", "ramp"):
                    g = g_all & (mot == mo)
                    y = 100 * keep[key][g].mean(0)
                    a.plot(t, y, color=COLS[mo], lw=1.6, marker="o", ms=2.6, zorder=3)
                    vals[f"{tg}|{mo}|{key}"] = dict(n=int(g.sum()), rate=[round(float(v), 3) for v in keep[key][g].mean(0)],
                                                    present=[round(float(v), 3) for v in say[g].mean(0)])
                a.set_ylim(-3, 103); a.set_yticks([0, 25, 50, 75, 100]); a.set_xticks(t); a.set_xlim(-0.5, TP - 0.5)
                if i == 0:
                    a.set_title(clab)
                if j == 0:
                    a.set_ylabel(f"{rlab}\n(% of clips)")
                if i == 2:
                    a.set_xlabel("Future tubelet")
                a.text(0.5, -0.17 if i < 2 else -0.33, f"({'abcdefghi'[3 * i + j]})", transform=a.transAxes, ha="center", va="top", fontsize=9)
        hs = [Line2D([], [], color=COLS[m], lw=1.6, marker="o", ms=2.6, label=m) for m in ("static", "flat", "ramp")]
        hs += [
               plt.Rectangle((0, 0), 1, 1, color="#e6e6e6", label="object hidden in most clips"),
               Line2D([], [], color="none", label="occluded rows pool k = 1–4")]
        f.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
        f.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.09, hspace=0.32, wspace=0.08)
        OUT.mkdir(parents=True, exist_ok=True)
        f.savefig(OUT / "fig_v11_presence_identity.png", dpi=300); f.savefig(OUT / "fig_v11_presence_identity.pdf"); plt.close(f)
    (OUT / "values_identity.json").write_text(json.dumps(vals, indent=1))
    stamp(OUT, PRES, [EXP / V11.name / "readings.npz", EXP / V11_MID.name / "readings.npz"])
    return vals


def acc57_grid():
    """3 × 2 (2026-09-28 사용자 지시, context_to_future/5_acc57_v3 의 v11 판).
    행 = 가림 없음 · 문맥 중간 가림 · 문맥 끝 가림 (가림은 k=1–4 합침), 열 = (a) 57-class 정확도 · (b) '없음' 을 고른 비율.
    57-class 정확도 = 57-way argmax 가 문맥 물체의 (모양, 색) 조합 ('없음' 이면 틀림). 분모 = 그 행 · 운동의 물체 clip 전부 (1,568).
    predictor p = 주황 (운동은 선 모양 · 표식으로), target encoder h (실제 프레임) = 파랑 파선 (운동 합침). 회색 칸 = 과반 clip 에서 물체가 가려진 튜블릿.
    정체 자만. 논문 판."""
    if not is_identity(PRES):
        raise SystemExit("정체 자 (identity) 에서만 그린다")
    S = json.loads((PRES / "summary.json").read_text()); none = S["none_index"]; SH, CO = S["shapes"], S["colors"]
    Z = {"last": np.load(EXP / V11.name / "readings.npz", allow_pickle=True), "mid": np.load(EXP / V11_MID.name / "readings.npz", allow_pickle=True)}
    for z in Z.values():
        check(z, PRES)
    rows = (("visible", "last", [0], "Never occluded"), ("mid", "mid", [1, 2, 3, 4], "Occluded mid-context"),
            ("last", "last", [1, 2, 3, 4], "Occluded at context end"))
    STY = {"static": dict(ls="-", marker="o"), "flat": dict(ls="--", marker="s"), "ramp": dict(ls="-.", marker="^")}
    ORANGE, GREY, BLUE = "#eb6834", "#8c8c8c", "#2a78d6"   # 2026-09-28: h 기준선 = 파랑 (사용자 지시)
    paper = {"font.family": "Nimbus Roman", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 8.5,
             "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6, "legend.frameon": False,
             "axes.spines.top": False, "axes.spines.right": False}
    t = np.arange(TP); vals = {}
    with plt.rc_context(paper):
        f, ax = plt.subplots(3, 2, figsize=(7.0, 6.2), sharex=True, sharey=True)
        for i, (tg, src, ks, rlab) in enumerate(rows):
            z = Z[src]; n = len(z["video_id"]); k = z["sym_k"].astype(int)
            mot = np.array([motion(c) for c in z["condition"]])
            combo = np.array([SH.index(a) * 8 + CO.index(b) if a in SH else -1 for a, b in zip(z["shape_pre"], z["color_pre"])])[:, None]
            tp_, th_ = z["p_prob"][:, -TP:].argmax(-1), z["h_prob"][:, -TP:].argmax(-1)
            hid = z["hidden"].reshape(n, 16, 2)[:, TC:].any(-1)
            g_all = z["obj"] & np.isin(k, ks)
            for j, (key, clab) in enumerate((("acc", "57-class accuracy"), ("none", "'None' chosen"))):
                a = ax[i, j]
                for tt_ in t:
                    if hid[g_all, tt_].mean() > 0.5:
                        a.axvspan(tt_ - 0.5, tt_ + 0.5, color="#e6e6e6", lw=0, zorder=0)
                for mo in ("static", "flat", "ramp"):
                    g = g_all & (mot == mo)
                    y = (tp_[g] == combo[g]) if key == "acc" else (tp_[g] == none)
                    a.plot(t, 100 * y.mean(0), color=ORANGE, lw=1.4, ms=3, mfc="white", mew=0.9, zorder=3, **STY[mo])
                    vals[f"{tg}|{mo}|{key}"] = [round(float(v), 4) for v in y.mean(0)]
                yh = (th_[g_all] == combo[g_all]) if key == "acc" else (th_[g_all] == none)
                a.plot(t, 100 * yh.mean(0), color=BLUE, ls=(0, (3.5, 2.0)), lw=1.4, zorder=2)
                vals[f"{tg}|h|{key}"] = [round(float(v), 4) for v in yh.mean(0)]
                if key == "acc":
                    a.axhline(100 / 57, color=GREY, ls=":", lw=0.8)
                a.set_ylim(-3, 103); a.set_yticks([0, 25, 50, 75, 100]); a.set_xticks(t); a.set_xlim(-0.5, TP - 0.5)
                if i == 0:
                    a.set_title(clab)
                if j == 0:
                    a.set_ylabel(f"{rlab}\n(% of clips)")
                if i == 2:
                    a.set_xlabel("Future tubelet")
                a.text(0.5, -0.17 if i < 2 else -0.33, f"({'abcdef'[2 * i + j]})", transform=a.transAxes, ha="center", va="top", fontsize=9)
        hs = [Line2D([], [], color=ORANGE, lw=1.4, ms=3, mfc="white", mew=0.9, label=f"predictor p, {m}", **STY[m]) for m in ("static", "flat", "ramp")]
        hs += [Line2D([], [], color=BLUE, ls=(0, (3.5, 2.0)), lw=1.4, label="target encoder h (all motions)"),
               Line2D([], [], color=GREY, ls=":", lw=0.8, label="chance (1/57)"),
               plt.Rectangle((0, 0), 1, 1, color="#e6e6e6", label="object hidden in most clips")]
        f.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
        f.subplots_adjust(left=0.1, right=0.99, top=0.86, bottom=0.09, hspace=0.32, wspace=0.08)
        OUT.mkdir(parents=True, exist_ok=True)
        f.savefig(OUT / "fig_v11_acc57.png", dpi=300); f.savefig(OUT / "fig_v11_acc57.pdf"); plt.close(f)
    (OUT / "values_acc57.json").write_text(json.dumps(vals, indent=1))
    stamp(OUT, PRES, [EXP / V11.name / "readings.npz", EXP / V11_MID.name / "readings.npz"])
    return vals


if __name__ == "__main__":
    if "--acc57" in sys.argv:
        acc57_grid(); print(OUT)
    elif "--identity" in sys.argv:
        identity_grid(); print(OUT)
    elif "--by-k" in sys.argv:
        by_k(); print(OUT)
    else:
        main()
