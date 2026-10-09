#!/usr/bin/env python3
"""IntPhysGen v11 — p 가 '있음' 이라고 할 때 **물체가 어디로 읽히나**, 프레임 위에 겹쳐 (2026-09-26).

행 = 정지 · flat · ramp, 열 = 가림 없음 (k=0) · 문맥 끝 가림 k=1–4. 한 칸 = 그 조건의 **모든 물체 clip** (이동은 왼→오 clip 만 —
방향마다 장면 · 판 자리가 달라서) 의 미래 튜블릿 8 개.
  배경   같은 조건 · 같은 k · 같은 방향 · concrete_brick 의 **빈 장면** (판은 있고 물체는 없다) 프레임, 미래 t3 시점 (샘플 22)
  점     p 가 '있다' 고 한 튜블릿의 읽힌 위치 (자 치우침 뺌, 288 px), 색 = 튜블릿 t0 … t7
  흰 선  진실 궤적 (튜블릿마다 clip 평균), 흰 점 = t0 … t7
  ×      가려지기 직전 마지막으로 보인 자리 (샘플 16 − k − 1, 가림 없음은 15), clip 평균
  제목   그 칸의 '있음' 비율 (clip × 튜블릿)
가림막 빈 장면 오탐이 없는 자 (예: identity_r8) 에서만 뜻이 있다.

  R3_DECODER=identity_r8 R3_OUT=identity_r8 $P z_research/scripts/figures/plot_v11_readout_overlay.py → figures/v11_readout_overlay/
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, V11, ARCHIVE, check, present_thr, is_identity, stamp   # noqa: E402
DEC = "decoder: position + identity (56 shape x colour + none)" if is_identity(PRES) else "decoder: position + presence"

FR = Path("/local_datasets/world/world_analysis/IntPhysGen_v11")
OUT = ARCHIVE / "v11_readout_overlay"
RES, CS, TP, W = 144.0, 16, 8, 288
MOTIONS = ["static", "flat", "ramp"]
KS = [0, 1, 2, 3, 4]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})


def motion(c):
    return "static" if c.startswith("static") else ("flat" if "flat" in c else "ramp")


def main():
    z = np.load(V11 / "readings.npz", allow_pickle=True); check(z, PRES)
    meta = {m["name"]: m for m in csv.DictReader((FR / "metadata.csv").open())}
    thr = present_thr("p", PRES); bp = np.array(json.loads((PRES / "attn_bias_px.json").read_text())["p"])
    n = len(z["video_id"]); k = z["sym_k"].astype(int); mo = np.array([motion(c) for c in z["condition"]])
    env = np.array([meta[v]["env"] for v in z["video_id"]]); td = z["travel_dir"]
    L = z["truth"] * RES + RES
    X = L[:, CS:].reshape(n, TP, 2, 2).mean(2)
    p = z["p"][:, -TP:]; say = p[..., 2] > thr; xy = p[..., :2] * RES + RES - bp
    cmap = plt.get_cmap("viridis"); tcol = [cmap(t / (TP - 1)) for t in range(TP)]
    f_, ax = plt.subplots(len(MOTIONS), len(KS), figsize=(3.1 * len(KS), 3.3 * len(MOTIONS)), squeeze=False)
    vals = {}
    for i, m_ in enumerate(MOTIONS):
        for j, kk in enumerate(KS):
            a = ax[i, j]
            dirm = (td == 1) if m_ != "static" else np.ones(n, bool)
            g = z["obj"] & (mo == m_) & (k == kk) & dirm
            # 배경: 같은 조건 · k · 방향 · concrete_brick 의 빈 장면
            cond = z["condition"][g][0]
            e = np.where(~z["obj"] & (z["condition"] == cond) & (k == kk) & dirm & (env == "concrete_brick"))[0]
            if len(e) == 0:
                e = np.where(~z["obj"] & (z["condition"] == cond) & (k == kk) & dirm)[0]
            bg = Image.open(FR / meta[z["video_id"][e[0]]]["file_name"] / f"{3 * (CS + 6):06d}.png").convert("RGB")
            a.imshow(np.asarray(bg), extent=(0, W, W, 0))
            for t in range(TP):
                pts = xy[g, t][say[g, t]]
                a.scatter(pts[:, 0], pts[:, 1], s=7, color=tcol[t], alpha=0.55, lw=0, zorder=3)
            gt = X[g].mean(0)
            a.plot(gt[:, 0], gt[:, 1], color="white", lw=1.4, zorder=4)
            a.scatter(gt[:, 0], gt[:, 1], s=16, color="white", edgecolor="black", lw=0.5, zorder=5)
            s_last = CS - 1 if kk == 0 else CS - kk - 1
            ls = L[g, s_last].mean(0)
            a.scatter([ls[0]], [ls[1]], marker="x", s=70, color="black", lw=2.4, zorder=6)
            a.scatter([ls[0]], [ls[1]], marker="x", s=55, color="white", lw=1.2, zorder=7)
            a.set_xlim(0, W); a.set_ylim(W, 0); a.set_xticks([]); a.set_yticks([])
            pr = float(say[g].mean())
            a.set_title(f"{'visible (k=0)' if kk == 0 else f'occluded at context end, k={kk}'}\npresent {100 * pr:.0f}% of clip-tubelets (n={int(g.sum())} clips)",
                        fontsize=8.5)
            if j == 0:
                a.set_ylabel(m_, fontsize=11, fontweight="bold")
            vals[f"{m_}|k{kk}"] = dict(n_clip=int(g.sum()), present=[round(float(x), 3) for x in say[g].mean(0)],
                                       mean_xy_present=[[round(float(v), 1) for v in xy[g, t][say[g, t]].mean(0)] if say[g, t].any() else None
                                                        for t in range(TP)],
                                       truth=[[round(float(v), 1) for v in gt[t]] for t in range(TP)],
                                       last_seen=[round(float(v), 1) for v in ls])
    hd = [Line2D([], [], ls="", marker="o", ms=5, color=tcol[t], label=f"p read as present, t{t}") for t in range(TP)]
    hd += [Line2D([], [], color="white", marker="o", markeredgecolor="black", ms=5, lw=1.4, label="true path (mean)"),
           Line2D([], [], ls="", marker="x", ms=7, color="black", label="last seen position (before occlusion)")]
    f_.legend(handles=hd, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    f_.suptitle("IntPhysGen v11 · where predictor p's own readout puts the object when it reads 'present' · moving clips: left-to-right only\n"
                f"background = empty scene of the same condition at future t3 · {DEC}", fontsize=10)
    f_.tight_layout(rect=(0, 0.07, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    f_.savefig(OUT / "fig_v11_readout_overlay.png", dpi=150); plt.close(f_)
    (OUT / "values.json").write_text(json.dumps(vals, indent=1))
    stamp(OUT, PRES, [V11 / "readings.npz"])
    print(OUT)


if __name__ == "__main__":
    main()
