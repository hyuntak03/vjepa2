#!/usr/bin/env python3
"""v11_realistic · v11_realistic_ledge surprise 그림 (2026-10-06). v11 surprise 그림 (`plot_v11_surprise.py`) 과 같은 관례.

  fig_rl_occlusion   운동 팔 3 × (기하 도형 v11 / 실물 메시) × (가림 없음 / 가림) — 영속성 정확도
  fig_rl_direction   k × 방향 (물체→빈 / 빈→물체), 실물 = 진한 선 + 95 % CI, 기하 도형 v11 = 회색 참조선
  fig_rl_ledge       (a) 예시 프레임 (낙하 vs 공중 직진) (b) 쌍별 surprise 산점도 (c) 상대 차 (불가능 − 가능) / 가능

입력: z_research/v11_realistic/exp_results/tables.json (v11_realistic_tables.py 가 summary.json 과 대조 검증해 만든다)
     + ledge per_block.json (산점도) + ledge 프레임 (예시)

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_v11_realistic.py          # → z_research/v11_realistic/figures/
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_v11_surprise import BLUE, ORANGE, GREEN, INK, INK2, MUTED, BAND, PANEL, frame, tint, stack_labels  # noqa: E402  (rcParams 도 같이 들어온다)

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "z_research/v11_realistic/exp_results"
OUT = REPO / "z_research/v11_realistic/figures"
LEDGE_RUN = EXP / "surprise_c16t32__v11_realistic_ledge_vith"
LEDGE_IDX = REPO / "data_csv/intphysgen_v11_realistic_ledge/index.csv"
LEDGE_FR = Path("/local_datasets/world/world_analysis/IntPhysGen_v11_realistic_ledge")
ARMS = [("static_visible", "static_occlusion", "Static"),
        ("moving_visible_flat", "moving_occlusion_flat", "Moving (flat)"),
        ("moving_visible", "moving_occlusion", "Moving (ramp)")]
GREY = "#8c8c8c"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig((OUT / name).with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {OUT / name}.pdf  + .png")


def fig_occlusion(T):
    C = T["by_condition"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9))
    W = 0.32
    for pi, (ax, (vis, occ, title)) in enumerate(zip(axes, ARMS)):
        frame(ax, (0, 124), (0, 25, 50, 75, 100))
        for gi, (key, base) in enumerate([("v11_all", GREY), ("all", BLUE)]):
            for bi, (cond, lab, f) in enumerate([(vis, "visible", 0.46), (occ, "occluded", 0.10)]):
                s = C[cond][key]; y = s["acc"]
                x = gi + (bi - 0.5) * W
                ax.bar(x, y, width=W * 0.88, color=tint(base, f), edgecolor=tint(base, max(0, f - 0.28)), lw=0.7, zorder=3)
                ax.errorbar(x, y, yerr=[[y - s["lo"]], [s["hi"] - y]], fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.6, zorder=4)
                ax.annotate(f"{y:.0f}", (x, s["hi"] + 1.4), ha="center", va="bottom", fontsize=6.8, color=INK, zorder=5)
                ax.annotate(lab, (x, 2.5), ha="center", va="bottom", fontsize=6.2,
                            color="white" if y > 20 else INK2, rotation=90, zorder=6)
            d = C[occ][key]["acc"] - C[vis][key]["acc"]
            ax.annotate(f"{d:+.1f}", (gi, 113), ha="center", va="bottom", fontsize=7.6,
                        color=INK if abs(d) > 3 else MUTED, fontweight="bold" if abs(d) > 3 else "normal", zorder=5)
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Geometric (v11)\nn={C[vis]['v11_all']['n']}+{C[occ]['v11_all']['n']}",
                            f"Realistic\nn={C[vis]['all']['n']}+{C[occ]['all']['n']}"], fontsize=7.6)
        ax.set_xlim(-0.5 - W, 1.5 + W); ax.tick_params(axis="x", length=0, pad=3)
        ax.set_title(title, fontsize=9.5, color=INK, pad=4)
        if pi:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("pairwise acc. (%)", fontsize=9.5)
        ax.text(0.5, -0.20, PANEL[pi], transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    fig.text(0.5, 1.00, "Object permanence (vanish). Numbers above each pair: occluded $-$ visible (pp); bars: 95% block-bootstrap CI",
             ha="center", va="top", fontsize=7.8, color=INK2, style="italic")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.885, bottom=0.20, wspace=0.09)
    save(fig, "fig_rl_occlusion")


def fig_direction(T):
    C, K = T["by_condition"], T["by_k"]
    ks = [0, 1, 2, 3, 4]
    xs = {k: (0 if k == 0 else 1.6 + k - 1) for k in ks}
    occx = [xs[k] for k in ks if k]
    DIRS = [("object $\\to$ empty", "A", tint(BLUE, 0.00), "o"), ("empty $\\to$ object", "B", tint(BLUE, 0.52), "^")]
    # 기하 도형 v11 방향별 k 값은 tables.json 에 없다 (전체만) → 같은 실행에서 다시 센다
    v11 = v11_direction_by_k()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    for pi, (ax, (vis, occ, title)) in enumerate(zip(axes, ARMS)):
        frame(ax, (-6, 112), (0, 25, 50, 75, 100))
        ax.axvspan(min(occx) - 0.5, max(occx) + 0.5, color=BAND, lw=0, zorder=0)
        x = [xs[k] for k in ks]
        Y = {}
        for lab, d, col, mk in DIRS:
            # 기하 도형 v11 참조선 (회색, 속 빈 마커)
            g = [v11[(vis if k == 0 else occ, k, d)] for k in ks]
            ax.plot(x[1:], g[1:], "-", marker=mk, color=GREY, ms=4.6, lw=0.9, mfc="white", mec=GREY, mew=0.9, alpha=0.8, zorder=2)
            ax.plot(x[:2], g[:2], ":", color=GREY, lw=0.8, alpha=0.6, zorder=1)
            ss = [C[vis][d] if k == 0 else K[f"{occ}/k{k}"][d] for k in ks]
            y = [s["acc"] for s in ss]; Y[d] = y
            ax.errorbar(x, y, yerr=[[a - s["lo"] for a, s in zip(y, ss)], [s["hi"] - a for a, s in zip(y, ss)]],
                        fmt="none", ecolor=tint(col, 0.35), elinewidth=0.8, capsize=1.8, zorder=3)
            ax.plot(x[1:], y[1:], "-", marker=mk, color=col, ms=6.0, lw=1.4, mec="white", mew=0.9, zorder=4)
            ax.plot(x[:1], y[:1], marker=mk, color=col, ms=6.8, ls="none", mec="white", mew=0.9, zorder=4)
            ax.plot(x[:2], y[:2], ":", color=tint(col, 0.45), lw=1.0, zorder=2)
        at = [stack_labels([Y[d][i] for _, d, _, _ in DIRS], pad=2.1, ymax=108.0) for i in range(len(ks))]
        for di, (lab, d, col, mk) in enumerate(DIRS):
            for i, xx in enumerate(x):
                ly, lva = at[i][di]
                ax.annotate(f"{Y[d][i]:.0f}", (xx + 0.22, ly), ha="left", va=lva, fontsize=6.4, color=col, zorder=5)
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=1)
        ax.set_xticks(list(xs.values())); ax.set_xticklabels([str(k) for k in ks], fontsize=8.4, color=INK)
        ax.set_xlim(-0.55, max(xs.values()) + 0.75)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.annotate("no occl.\nn=60 / dir.", (xs[0], -0.115), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=6.8, color=INK2, style="italic")
        ax.annotate("occluded — hidden frames (k)\nn=15 / dir. / k", (float(np.mean(occx)), -0.115),
                    xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=6.8, color=INK2, style="italic")
        ax.set_title(title, fontsize=9.5, color=INK, pad=3.5)
        if pi == 0:
            ax.set_ylabel("pairwise acc. (%)", fontsize=9.5)
        ax.text(0.5, -0.27, PANEL[pi], transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    h = [Line2D([], [], color=c, marker=mk, ms=6.0, lw=1.4, mec="white", mew=0.9, label=f"Realistic: {l}") for l, _, c, mk in DIRS]
    h += [Line2D([], [], color=GREY, marker=mk, ms=4.6, lw=0.9, mfc="white", mec=GREY, mew=0.9, label=f"Geometric (v11): {l}")
          for l, _, _, mk in DIRS]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7.8, handlelength=1.5, handletextpad=0.4,
               columnspacing=1.4, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.905, bottom=0.25, wspace=0.20)
    save(fig, "fig_rl_direction")


def v11_direction_by_k():
    """기하 도형 v11 의 vanish 방향별 정확도 (조건, k, A/B) — v11_realistic_tables.pairs 로 같은 실행에서 센다."""
    sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
    import v11_realistic_tables as TT
    V = [x for x in TT.pairs("v11") if x["vio"] == "vanish"]
    out = {}
    for x in V:            # 원본 v11 은 가림 없는 clip 이름에도 _k1_…_k4_ 가 붙어 있다 (짝 설계 번호) → 가림 없음 = k 0
        k = 0 if "visible" in x["cond"] else x["k"]
        out.setdefault((x["cond"], k, x["dir"]), []).append(x["ok"])
    return {k: 100 * float(np.mean(v)) for k, v in out.items()}


def ledge_pairs():
    S = json.loads((LEDGE_RUN / "per_block.json").read_text())["per_video_surprise"]
    rows = list(csv.DictReader(LEDGE_IDX.open()))
    by = {}
    for r in rows:
        by.setdefault(r["block_id"], {})[r["variant"]] = r
    P = []
    for b, v in by.items():
        f, fl = v["pos_fall"], v["imp_float"]
        c = f["condition"]                                   # grav_v140_z300
        P.append(dict(cond=c, v=int(c.split("_")[1][1:]), z=int(c.split("_")[2][1:]), fall=S[f["video_id"]], float_=S[fl["video_id"]],
                      name=f["video_id"], fn_fall=f["file_name"], fn_float=fl["file_name"]))
    return P


def fig_ledge():
    P = ledge_pairs()
    acc = 100 * np.mean([p["float_"] > p["fall"] for p in P])
    VCOL = {140: BLUE, 180: ORANGE, 220: GREEN}
    ZMK = {300: "o", 400: "s"}
    fig = plt.figure(figsize=(7.0, 4.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.0], hspace=0.42, wspace=0.34, left=0.10, right=0.98, top=0.95, bottom=0.185)
    # (a) 예시 프레임 — 한 줄 전체
    ax = fig.add_subplot(gs[0, :])
    ex = P[0]
    cols = [0, 45, 57, 69, 81, 93]
    rows = []
    for fn in (ex["fn_fall"], ex["fn_float"]):
        ims = [np.asarray(Image.open(LEDGE_FR / fn / f"{f:06d}.png").convert("RGB"))[40:200] for f in cols]
        gap = np.full((ims[0].shape[0], 8, 3), 255, np.uint8)
        rows.append(np.concatenate(sum([[im, gap] for im in ims], [])[:-1], 1))
    hg = np.full((10, rows[0].shape[1], 3), 255, np.uint8)
    ax.imshow(np.concatenate([rows[0], hg, rows[1]], 0)); ax.set_axis_off()
    wpx = (rows[0].shape[1] + 8) / len(cols); hpx = rows[0].shape[0]
    lab = {0: "context start", 45: "context end"}
    for i, f in enumerate(cols):
        ax.text((i + 0.5) * wpx - 4, -8, lab.get(f, f"future +{(f - 45) // 3}"), ha="center", va="bottom", fontsize=7.4,
                color=INK if f > 45 else INK2)
    ax.text(-12, hpx / 2, "possible\n(falls)", ha="right", va="center", fontsize=8.0, color=INK)
    ax.text(-12, hpx * 1.5 + 10, "impossible\n(floats on)", ha="right", va="center", fontsize=8.0, color=INK)
    ax.text(0.5, -0.06, "(a)", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    # (b) 쌍별 산점도
    ax = fig.add_subplot(gs[1, 0])
    for p in P:
        ax.plot(p["fall"], p["float_"], ZMK[p["z"]], color=VCOL[p["v"]], ms=3.4, mec="white", mew=0.4, alpha=0.9, zorder=3)
    lo = min(min(p["fall"], p["float_"]) for p in P); hi = max(max(p["fall"], p["float_"]) for p in P)
    pad = (hi - lo) * 0.06; lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.text(0.04, 0.96, "above diagonal = correct\n(impossible more surprising)", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.8, color=MUTED, style="italic")
    ax.set_title(f"below diagonal: {sum(p['float_'] < p['fall'] for p in P)}/{len(P)} pairs (pairwise acc. {acc:.0f}%)",
                 fontsize=8.0, color=INK, pad=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.2, colors=INK2, pad=1.5)
    ax.set_xlabel("surprise, possible (falls)", fontsize=8.0); ax.set_ylabel("surprise, impossible (floats)", fontsize=8.0)
    ax.text(0.5, -0.27, "(b)", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    # (c) 상대 차
    ax = fig.add_subplot(gs[1, 1])
    order = [(v, z) for v in (140, 180, 220) for z in (300, 400)]
    rng = np.random.default_rng(0)
    for i, (v, z) in enumerate(order):
        r = np.array([100 * (p["float_"] - p["fall"]) / p["fall"] for p in P if p["v"] == v and p["z"] == z])
        ax.plot(i + rng.uniform(-0.18, 0.18, len(r)), r, ZMK[z], color=VCOL[v], ms=3.0, mec="white", mew=0.3, alpha=0.85, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(r)] * 2, color=INK, lw=1.2, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{v}\nz{z}" for v, z in order], fontsize=6.6)
    ax.set_xlim(-0.6, len(order) - 0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4); ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.2, colors=INK2, pad=1.5); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("relative surprise gap (%)\n(impossible $-$ possible) / possible", fontsize=7.6)
    ax.set_xlabel("speed (cm/s), depth (cm)", fontsize=7.6)
    ax.set_title("all 120 pairs below 0", fontsize=8.0, color=INK, pad=4)
    ax.text(0.5, -0.27, "(c)", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    h = [Line2D([], [], color=VCOL[v], marker="o", ls="none", ms=4.0, label=f"{v} cm/s") for v in (140, 180, 220)]
    h += [Line2D([], [], color=INK2, marker=ZMK[z], ls="none", mfc="white", ms=4.0, label=f"depth z={z}") for z in (300, 400)]
    h += [Line2D([], [], color=INK, lw=1.2, label="median")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=7.6, handletextpad=0.3, columnspacing=1.4,
               bbox_to_anchor=(0.54, -0.005))
    save(fig, "fig_rl_ledge")
    print(f"  ledge acc {acc:.1f}%  (n={len(P)})")


def fig_summary(T):
    """사용자 지시 (2026-10-06): `IntPhysGenV11_occlusion_timing_ablation` 의 fig_timing_step 처럼 단순하게.
    패널 = (실물 · 기하 v11 · ledge), 선 = 운동 팔, x = k (0 = 가림막 없음, 1–4 = 문맥 끝 가림). 방향 분해는 tables.md 와 _superseded/fig_rl_direction."""
    import plot_v11_timing as TM
    C, K = T["by_condition"], T["by_k"]
    MOT = [("static_visible", "static_occlusion", "static"), ("moving_visible_flat", "moving_occlusion_flat", "flat"),
           ("moving_visible", "moving_occlusion", "ramp")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharey=True)
    for ci, (ax, key, title) in enumerate(zip(axes[:2], [("all", "all"), ("v11_all", "v11")], ["Realistic objects", "Geometric objects (v11)"])):
        ax.axvspan(0.5, 4.5, color=TM.BAND, zorder=0)
        TM.frame(ax, ylim=(-6, 124), yt=(0, 25, 50, 75, 100), chance=50)
        ys = [[C[vis][key[0]]["acc"]] + [K[f"{occ}/k{k}"][key[1]]["acc"] for k in range(1, 5)] for vis, occ, _ in MOT]
        TM.lines(ax, ys, [TM.MCOL[m] for _, _, m in MOT], [TM.MLAB[m] if ci == 0 else None for _, _, m in MOT], gap=7.5)
        ax.set_xlim(-0.45, 4.45); ax.set_xticks(range(5)); ax.set_xticklabels(["0", "1", "2", "3", "4"], fontsize=7.2)
        ax.set_xlabel("k   (0 = no occluder)", fontsize=8.2, color=TM.INK)
        ax.set_title(f"{title}: object permanence", fontsize=9.0, color=TM.INK, pad=4)
        if ci == 0:
            ax.set_ylabel("pairwise accuracy (%)", fontsize=8.6, color=TM.INK)
    ax = axes[2]
    TM.frame(ax, ylim=(-6, 124), yt=(0, 25, 50, 75, 100), chance=50)
    L = T["ledge_by_condition"]
    # 깊이 두 개 (z 300 · 400, 각 20 쌍) 를 합친다 — 둘 다 0 이라 따로 그리면 선이 완전히 겹친다
    y = [(L[f"grav_v{v}_z300"]["acc"] * L[f"grav_v{v}_z300"]["n"] + L[f"grav_v{v}_z400"]["acc"] * L[f"grav_v{v}_z400"]["n"])
         / (L[f"grav_v{v}_z300"]["n"] + L[f"grav_v{v}_z400"]["n"]) for v in (140, 180, 220)]
    n = [L[f"grav_v{v}_z300"]["n"] + L[f"grav_v{v}_z400"]["n"] for v in (140, 180, 220)]
    TM.lines(ax, [y], [TM.INK2])
    print(f"  ledge 속도별 n = {n} (깊이 합침)")       # 서술 문장은 그림이 아니라 캡션으로 (CLAUDE.md §8-4)
    ax.set_xlim(-0.45, 2.45); ax.set_xticks(range(3)); ax.set_xticklabels(["140", "180", "220"], fontsize=7.2)
    ax.set_xlabel("horizontal speed (cm/s)", fontsize=8.2, color=TM.INK)
    ax.set_title("Realistic objects: falling off a ledge", fontsize=9.0, color=TM.INK, pad=4)
    axes[0].legend(frameon=False, fontsize=TM.LGF, ncol=3, loc="lower center", bbox_to_anchor=(1.70, 1.13),
                   handlelength=1.3, columnspacing=1.8)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.845, bottom=0.275, wspace=0.10)
    TM.panel_labels(fig, axes)
    save(fig, "fig_rl_summary")


def fig_gravity_choice(key="pick", name="fig_gravity_choice", ylab="chosen future (%)\n(lowest L1 surprise)"):
    """gravity_realistic — 문맥 (오르는 중 · 꼭대기 · 내려가기 시작) 마다 p 가 고른 미래 비율 (argmin L1 surprise), 사용자 지시 2026-10-07.
    x = 미래 넷 (가능 1 · 불가능 3), 막대 = block 부트스트랩 95 % CI, 점선 = chance 25 %.
    막대 아래 작은 그림 = 그 미래 (회색 점 = 본 문맥 16 샘플, 색 점 = 그 미래 16 샘플, 세로 점선 = 문맥 끝).
      metadata 궤적 (렌더 프레임과 겹쳐 대조함), **왼→오 block 하나** (시간이 왼쪽에서 오른쪽).
      ⚠️ 첫 판은 오→왼 block · 선이라 문맥이 점 '뒤' 로 보이고 세 조건이 같아 보였다 (사용자 지적). 큰 도식 한 장 판은 사용자가 작은 칸 판을 골라 물렸다."""
    import plot_v11_timing as TM
    J = json.loads((EXP / "gravity_retrieval.json").read_text())
    meta = {r["name"]: r for r in csv.DictReader(open("/local_datasets/world/world_analysis/IntPhysGen_gravity_realistic/metadata.csv"))}
    NF = int(J.get("n_future", 4))                      # 2026-10-07 재렌더판은 미래 다섯 (stop 추가)
    FUT = ["pos_arc", "imp_rise", "imp_float", "imp_line", "imp_stop"][:NF]
    XL = {"pos_arc": "arc", "imp_rise": "rise", "imp_float": "float", "imp_line": "line", "imp_stop": "stop"}
    COL = {"pos_arc": BLUE, "imp_rise": ORANGE, "imp_float": ORANGE, "imp_line": ORANGE, "imp_stop": ORANGE}
    CONDS = [("arc_pre", "Context: rising"), ("arc_apex", "Context ends at the apex"), ("arc_post", "Context: descending")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.5), sharey=True)
    for ci, (ax, (c, title)) in enumerate(zip(axes, CONDS)):
        TM.frame(ax, ylim=(0, 112), yt=(0, 25, 50, 75, 100), chance=100 / NF)
        for i, k in enumerate(FUT):
            m, lo, hi = [100 * v for v in J[key][c][k]]
            ax.bar(i, m, width=0.62, color=tint(COL[k], 0.15), edgecolor=COL[k], lw=0.8, zorder=3)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.8, zorder=4)
            ax.annotate(f"{m:.0f}", (i, hi + 1.5), ha="center", va="bottom", fontsize=7.4, color=INK, zorder=5)
        ax.set_xlim(-0.55, NF - 0.45); ax.set_xticks(range(NF)); ax.set_xticklabels([XL[k] for k in FUT], fontsize=7.6)
        for tl, k in zip(ax.get_xticklabels(), FUT):
            tl.set_color(COL[k]); tl.set_fontweight("bold" if k == "pos_arc" else "normal")
        ax.tick_params(axis="x", length=0, pad=38)
        ax.set_title(title, fontsize=8.8, color=INK, pad=4)
        if ci == 0:
            ax.set_ylabel(ylab, fontsize=8.2, color=INK)
        blk = sorted({r["block"] for r in meta.values() if r["condition"] == c and r["travel_direction"] == "l2r"})[0]
        for i, k in enumerate(FUT):
            r = meta[f"{blk}_{k}"]
            x = np.array(r["object_px_x_by_sample"].split(), float); y = np.array(r["object_px_y_by_sample"].split(), float)
            w = 0.92 / NF
            ia = ax.inset_axes([(i + 0.5) / NF - w / 2, -0.235, w, 0.19], transform=ax.transAxes)
            ia.axvline(x[15], color="#9a9a9a", lw=0.5, ls=(0, (1.5, 1.5)), zorder=1)
            ia.plot(x[:16], y[:16], "o", color="#a8a8a8", ms=1.3, zorder=2)
            ia.plot(x[16:], y[16:], "o", color=COL[k], ms=1.5 if k != "imp_stop" else 2.4, zorder=3)
            ia.set_xlim(25, 270); ia.set_ylim(288, 0); ia.set_xticks([]); ia.set_yticks([])
            for sp in ia.spines.values():
                sp.set_color("#d0d0d0"); sp.set_linewidth(0.5)
    # 작은 그림 읽는 법 (범례 한 줄)
    from matplotlib.lines import Line2D as L2
    h = [L2([], [], color="#a8a8a8", marker="o", ls="none", ms=3, label="context (seen)"),
         L2([], [], color=BLUE, marker="o", ls="none", ms=3, label="possible future"),
         L2([], [], color=ORANGE, marker="o", ls="none", ms=3, label="impossible future"),
         L2([], [], color="#9a9a9a", lw=0.8, ls=(0, (1.5, 1.5)), label="end of context")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7.4, handletextpad=0.3, columnspacing=1.6,
               bbox_to_anchor=(0.55, -0.01))
    fig.subplots_adjust(left=0.10, right=0.995, top=0.91, bottom=0.31, wspace=0.10)
    TM.panel_labels(fig, axes, pad=0.008)
    save(fig, name)


def main():
    T = json.loads((EXP / "tables.json").read_text())
    fig_summary(T)                  # 본 그림 (2026-10-06 사용자 지시로 단순화)
    fig_gravity_choice()            # gravity_realistic 미래 다섯 — 표준 채점 (화면 전체 토큰)
    fig_gravity_choice("pick_ball", "fig_gravity_choice_ball", "chosen future (%)\n(lowest L1, ball tokens only)")   # 공 칸만 (2026-10-08)
    if "--all" in sys.argv:         # 옛 세 장 — figures/_superseded/ 로 내렸다
        fig_occlusion(T); fig_direction(T); fig_ledge()


if __name__ == "__main__":
    main()
