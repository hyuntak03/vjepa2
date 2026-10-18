#!/usr/bin/env python3
"""predictor 병목 진단 그림 (2026-10-13). 입력 = z_research/VOBench_v1/bottleneck/{analysis.json, natural_*.json, late_identity/late_identity.json, steer/*.json}.
출력 = z_research/VOBench_v1/bottleneck/figures/*.{png,pdf}. 라벨은 영어, 서술 문장은 캡션 (문서) 으로.

  python z_research/scripts/figures/plot_predictor_bottleneck.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
B = REPO / "z_research/VOBench_v1/bottleneck"
FIG = B / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
COL = {"release": "#2a78d6", "ar40": "#eb6834", "ar_state40": "#1baf7a"}
LAB = {"release": "V-JEPA 2 release", "ar40": "Ariel AR", "ar_state40": "Ariel AR-state"}
K = range(1, 9)


def panel_label(ax, s):
    ax.text(0.5, -0.30, s, transform=ax.transAxes, ha="center", va="top", fontsize=9)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
    print("saved", FIG / f"{name}.png")


def fig_belief_behavior(R):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for n in ("release", "ar40", "ar_state40"):
        if n not in R: continue
        pr = R[n]["probes"]
        axs[0].plot(list(K), [pr[f"roll_st/k{k}"]["vel_r2"] for k in K], "-o", ms=3, color=COL[n], label=LAB[n])
        v = R[n]["step_gain"]["roll_velocity"]
        axs[1].plot(list(range(2, 9)), [v[str(k)] for k in range(2, 9)], "-o", ms=3, color=COL[n], label=LAB[n])
    n0 = next(n for n in ("ar40", "ar_state40", "release") if n in R)
    axs[0].plot(list(K), [R[n0]["probes"][f"truth_h/k{k}"]["vel_r2"] for k in K], "--", color="0.5", label="true future h (AR space)")
    axs[0].set_ylim(0, 1.02); axs[0].set_xlabel("prediction block k"); axs[0].set_ylabel("velocity R² (linear probe)")
    axs[1].axhline(1, color="0.6", lw=0.8, ls=":"); axs[1].axhline(0, color="0.8", lw=0.6)
    axs[1].set_xlabel("prediction block k"); axs[1].set_ylabel("object advance per block\n(fraction of true step)")
    axs[0].legend(frameon=False, fontsize=7, loc="lower left")
    panel_label(axs[0], "(a)"); panel_label(axs[1], "(b)")
    save(fig, "fig_belief_vs_behavior")


def fig_interventions(R):
    names = [n for n in ("ar40", "ar_state40") if n in R]
    fig, axs = plt.subplots(1, len(names), figsize=(3.5 * len(names), 2.4), squeeze=False)
    sty = {"roll": ("-", "o", "own rollout"), "tf": ("--", "s", "true history"), "injL": (":", "^", "own history + true last block"), "injO": ("-.", "v", "true history + own last block")}
    for ax, n in zip(axs[0], names):
        sg = R[n]["step_gain"]
        for c, (ls, mk, lab) in sty.items():
            if c not in sg: continue
            ks = sorted(int(k) for k in sg[c]); ax.plot(ks, [sg[c][str(k)]["lag"] for k in ks], ls=ls, marker=mk, ms=3, color=COL[n] if c == "roll" else "0.35", label=lab)
        ax.axhline(0, color="0.8", lw=0.6); ax.set_xlabel("prediction block k"); ax.set_ylabel("lag along motion (steps)"); ax.set_title(LAB[n], fontsize=8)
    axs[0][0].legend(frameon=False, fontsize=6.5, loc="lower left")
    for i, ax in enumerate(axs[0]): panel_label(ax, f"({'ab'[i]})")
    save(fig, "fig_interventions")


def fig_loss(R, NAT):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    if "ar40" in R:
        er = R["ar40"]["errors"]
        for c, ls in (("roll", "-"), ("tf", "--")):
            axs[0].plot(list(K), [er[c][str(k)]["obj"] for k in K], ls, color=COL["ar40"], label=f"object 3x3, {c}")
            axs[0].plot(list(K), [er[c][str(k)]["bg"] for k in K], ls, color="0.4", label=f"background, {c}")
        axs[0].plot(list(K), [er["roll"][str(k)]["copy_obj"] for k in K], ":", color=COL["ar40"], label="object, copy")
        axs[0].plot(list(K), [er["roll"][str(k)]["copy_bg"] for k in K], ":", color="0.4", label="background, copy")
        axs[0].set_xlabel("prediction block k"); axs[0].set_ylabel("token L1 error (Ariel AR space)"); axs[0].legend(frameon=False, fontsize=6, ncol=2)
    for n, d in NAT.items():
        if "roll" not in d: continue
        axs[1].plot(list(K), [r["frac_gain_from_changed"] for r in d["roll"]], "-o", ms=3, color=COL[n], label=f"{LAB[n]}: share of gain over copy")
    if NAT:
        d0 = next(iter(NAT.values()))
        axs[1].plot(list(K), d0["copy_error_share_in_changed"], "--", color="0.4", label="share of copy error in changed tokens")
        axs[1].axhline(0.1, color="0.8", lw=0.6, ls=":")
        axs[1].set_ylim(0, 0.4); axs[1].set_xlabel("prediction block k"); axs[1].set_ylabel("fraction (SSv2 val)")
        axs[1].legend(frameon=False, fontsize=6)
    panel_label(axs[0], "(a)"); panel_label(axs[1], "(b)")
    save(fig, "fig_loss_allocation")


def fig_late(L):
    names = [n for n in ("release", "ar40", "ar_state40") if n in L]
    if not names: return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, o in zip(axs, ("primitive", "realistic")):
        key = f"{o}/late"
        for n in names:
            if key not in L[n]: continue
            ax.plot(list(K), L[n][key]["st_g"], "-", color=COL[n], label=f"{LAB[n]}: whole-frame mean")
            ax.plot(list(K), L[n][key]["st_l"], ":", color=COL[n], lw=1.6, label=f"{LAB[n]}: at true location")
        if key in L[names[0]]:
            ax.plot(list(K), L[names[0]][key]["h_l"], "--", color="0.4", label="true future h: at true location")
            ax.axhline(L[names[0]][key]["chance_shape"], color="0.8", lw=0.6)
        ax.set_ylim(0, 1.02); ax.set_xlabel("prediction block k"); ax.set_ylabel("shape accuracy (late, moving)"); ax.set_title(o, fontsize=8)
    axs[0].legend(frameon=False, fontsize=5.8, ncol=1, loc="lower right")
    panel_label(axs[0], "(a)"); panel_label(axs[1], "(b)")
    save(fig, "fig_late_identity")


def fig_steer(S):
    names = [n for n in ("ar40", "ar_state40") if n in S]
    if not names: return
    fig, axs = plt.subplots(1, len(names), figsize=(3.5 * len(names), 2.4), squeeze=False)
    for ax, n in zip(axs[0], names):
        for a, ls in (("-2.0", ":"), ("-1.0", "--"), ("1.0", "--"), ("2.0", "-")):
            ax.plot(list(K), S[n]["u"][a], ls, color=COL[n], label=f"velocity code, alpha={float(a):+.0f}")
            ax.plot(list(K), S[n]["rand"][a], ls, color="0.6", lw=0.8, label=f"random code, alpha={float(a):+.0f}" if a in ("-2.0", "2.0") else None)
        ax.axhline(0, color="0.8", lw=0.6); ax.axvline(2, color="0.85", lw=0.6)
        ax.set_xlabel("prediction block k"); ax.set_ylabel("screen x shift vs alpha=0 (cells)"); ax.set_title(LAB[n], fontsize=8)
    axs[0][0].legend(frameon=False, fontsize=6)
    for i, ax in enumerate(axs[0]): panel_label(ax, f"({'ab'[i]})")
    save(fig, "fig_velocity_steering")


def main():
    R = json.load(open(B / "analysis.json")) if (B / "analysis.json").exists() else {}
    NAT = {n: json.load(open(B / f"natural_{n}.json")) for n in ("release", "ar40", "ar_state40") if (B / f"natural_{n}.json").exists()}
    L = json.load(open(B / "late_identity/late_identity.json")) if (B / "late_identity/late_identity.json").exists() else {}
    S = {n: json.load(open(B / f"steer/{n}.json")) for n in ("ar40", "ar_state40") if (B / f"steer/{n}.json").exists()}
    fig_belief_behavior(R); fig_interventions(R); fig_loss(R, NAT); fig_late(L); fig_steer(S)


if __name__ == "__main__":
    main()
