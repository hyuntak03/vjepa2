#!/usr/bin/env python3
"""IntPhysGen v11 — p 의 **앞섬 (+) / 뒤처짐 (−) 위치 오차**, 미래 튜블릿마다 (2026-09-26, v3_signed_error 의 v11 판).

그림 하나 = 조건 하나 (가림 없음 · 문맥 끝 가림 k=1–4 · 문맥 중간 가림 k=1–4 = 9 장), 열 = 정지 · flat · ramp, 행 = x · y.
**가림막 장면의 '있음' 을 믿을 수 있는 자** (판 빈 장면 오탐 0 %, 예: identity_r8) 에서만 뜻이 있다 — 옛 자는 판을 물체로 읽었다.

  s_a(t) = 그 튜블릿에서 GT 가 축 a 로 움직이는 방향 (±1, clip 마다). 1 px/튜블릿 미만이면 마지막으로 움직이던 방향 (문맥 끝에서 시작).
           한 번도 안 움직인 축은 x → + = 오른쪽, y → + = 위 (정지 물체는 두 축 다, flat 은 y)
  오차_a = (p̂_a(t) − GT_a(t)) · s_a(t)      + = 그 축의 GT 운동 방향으로 앞섬, − = 뒤처짐. p 가 '있다' 고 한 clip 의 평균 (px, 자 치우침 뺌)
  기준선 (GT 만으로, p 와 같은 clip 으로 평균, 같은 부호):
    진회색 파선  "본 속도 그대로 연장했다면"   x0 + v0 · (튜블릿 중심 − s_last) − GT
    밝은 회색 쇄선 "마지막으로 본 자리에 멈췄다면"  x0 − GT
    x0 = **마지막으로 보인 샘플** s_last 의 위치 (가림 없음 · 문맥 중간 가림: 15, 문맥 끝 가림 k: 16 − k − 1 = 가려지기 직전),
    v0 = 그 샘플 직전 두 샘플의 속도 (샘플당). 문맥 끝 가림에서 쇄선 = "가려지기 전 자리에 멈춤", 파선 = "가려진 동안에도 계속 감"
  초록 선 = 같은 계산을 **h (target encoder, 실제 미래 프레임)** 에 — 자 자체의 오차 바닥.
            v3 그림은 z 를 쓴다. v11 다시 읽기는 z 를 뽑지 않는다 (new_archive_redraw --v11-reps 기본 p h)
  빨간 점 = 그 튜블릿에서 clip 의 절반 넘게 p 가 '없다' → 0 으로 둔다
  회색 띠 = 물체가 실제로 가려진 튜블릿 (진회색 = 두 샘플 모두, 연회색 = 한 샘플)
  1 칸 = 18 px (점선). 세로 범위 = 줄마다 값의 95 % (최소 1 칸 남짓), 밖으로 나간 점은 가장자리 ▲▼
  물체 있는 가능 clip 만.

  R3_DECODER=identity_r8 R3_OUT=identity_r8 $P z_research/scripts/figures/plot_v11_signed_error.py   → figures/v11_signed_error/
  (new_archive_redraw.py 가 자를 바꿀 때 같이 그린다)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, V11, V11_MID, ARCHIVE, check, present_thr, is_identity, stamp   # noqa: E402
DEC = "decoder: position + identity (56 shape x colour + none)" if is_identity(PRES) else "decoder: position + presence"

OUT = ARCHIVE / "v11_signed_error"
RES, CS, TP, CELL, STILL = 144.0, 16, 8, 18.0, 1.0
MOTIONS = ["static", "flat", "ramp"]
COL = ("#2a78d6", "#eb6834")                    # x · y
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})


def motion(c):
    return "static" if c.startswith("static") else ("flat" if "flat" in c else "ramp")


def axis_dir(L, X):
    """(n, 8, 2) 축마다 GT 운동 방향 부호 (화면 좌표). v3 판 (plot_v3_l2_error.axis_dir) 과 같은 규칙, 문맥 끝 = 샘플 16."""
    prev = L[:, CS - 1] - L[:, CS - 3]
    Xp = np.concatenate([L[:, CS - 2:CS].mean(1)[:, None], X], 1)
    V = Xp[:, 1:] - Xp[:, :-1]
    last = np.where(np.abs(prev) >= STILL / 2, np.sign(prev), 0.0)
    s = np.zeros_like(V)
    for t in range(V.shape[1]):
        cur = np.where(np.abs(V[:, t]) >= STILL, np.sign(V[:, t]), last)
        s[:, t] = cur; last = cur
    never = s == 0
    s[..., 0][never[..., 0]] = 1.0; s[..., 1][never[..., 1]] = -1.0
    return s, never


def load(path):
    z = np.load(path, allow_pickle=True); check(z, PRES)
    return z


def main():
    zl, zm = load(V11 / "readings.npz"), load(V11_MID / "readings.npz")
    thr, thr_h = present_thr("p", PRES), present_thr("h", PRES)
    B = json.loads((PRES / "attn_bias_px.json").read_text()); bp, bh = np.array(B["p"]), np.array(B["h"])
    conds = [("visible", "visible (k=0)", zl, 0)] + \
            [(f"last_k{k}", f"occluded at context end, k={k}", zl, k) for k in (1, 2, 3, 4)] + \
            [(f"mid_k{k}", f"occluded mid-context, k={k}", zm, k) for k in (1, 2, 3, 4)]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, title, z, kk in conds:
        n = len(z["video_id"]); k = z["sym_k"].astype(int); mo = np.array([motion(c) for c in z["condition"]])
        L = z["truth"] * RES + RES                                            # (n, 32, 2) 샘플 진실 px
        X = L[:, CS:].reshape(n, TP, 2, 2).mean(2)                           # 미래 튜블릿 진실
        p = z["p"][:, -TP:]; h = z["h"][:, -TP:]
        say, hsay = p[..., 2] > thr, h[..., 2] > thr_h
        d = p[..., 0:2] * RES + RES - bp - X; dh = h[..., 0:2] * RES + RES - bh - X
        sgn, never = axis_dir(L, X)
        err, herr = d * sgn, dh * sgn
        s_last = CS - 1 if (tag == "visible" or tag.startswith("mid")) else CS - kk - 1
        x0 = L[:, s_last]; v0 = (L[:, s_last] - L[:, s_last - 2]) / 2.0     # 샘플당 속도
        tc = CS + 0.5 + 2 * np.arange(TP)                                    # 튜블릿 중심 샘플
        ext = (x0[:, None] + v0[:, None] * (tc - s_last)[None, :, None] - X) * sgn
        stop = (x0[:, None] - X) * sgn
        hid = z["hidden"].reshape(n, 16, 2)[:, CS // 2:].sum(-1)
        T = np.arange(TP)
        f_, ax = plt.subplots(2, 3, figsize=(10.5, 5.4), sharex=True, sharey="row", squeeze=False)
        allv = [[], []]; lines = []
        for j, m_ in enumerate(MOTIONS):
            g = z["obj"] & (mo == m_) & (k == kk)
            absent = say[g].mean(0) <= 0.5
            hk = hid[g].max(0)
            for a_ in range(2):
                a = ax[a_, j]
                for t in T:
                    if hk[t] > 0:
                        a.axvspan(t - 0.5, t + 0.5, color="#bbbbbb" if hk[t] == 2 else "#e3e3e3", lw=0, zorder=0)
                e = np.array([err[g & say[:, t], t, a_].mean() if (g & say[:, t]).any() else 0.0 for t in T]); e[absent] = 0.0
                eh = np.array([herr[g & hsay[:, t], t, a_].mean() if (g & hsay[:, t]).any() else np.nan for t in T])
                a.plot(T, eh, color="#1baf7a", lw=1.2, zorder=1)
                # 기준선은 GT 만으로 정해지므로 **조건의 모든 clip** 으로 평균해 끊김 없이 그린다 (2026-09-26 — v11 은 '있음' 칸이
                # 띄엄띄엄이라 v3 처럼 '있음' clip 으로만 평균하고 없음 칸을 비우면 선이 안 이어져 안 보였다)
                for ref, st in ((ext, dict(color="#333333", lw=1.1, ls="--")), (stop, dict(color="#aaaaaa", lw=1.1, ls="-."))):
                    a.plot(T, ref[g][..., a_].mean(0), zorder=1, **st)
                a.scatter(T[absent], np.zeros(absent.sum()), s=22, color="red", zorder=3)
                a.axhline(CELL, color="#aaaaaa", lw=0.7, ls=":"); a.axhline(-CELL, color="#aaaaaa", lw=0.7, ls=":")
                a.axhline(0, color="#777777", lw=0.6)
                a.spines[["top", "right"]].set_visible(False)
                allv[a_] += list(np.abs(e[~absent])) + list(np.abs(eh[np.isfinite(eh)]))
                lines.append((a_, j, e))
                nev = never[g][..., a_].mean() > 0.5
                if nev:
                    a.text(0.03, 0.93, f"no GT {'xy'[a_]}-motion: + = {'right' if a_ == 0 else 'above'}", transform=a.transAxes,
                           fontsize=6.5, color="#555555", va="top")
                rows.append(dict(cond=tag, motion=m_, axis="xy"[a_], n_clip=int(g.sum()),
                                 present=[round(float(x), 3) for x in say[g].mean(0)],
                                 err=[None if ab else round(float(x), 1) for x, ab in zip(e, absent)],
                                 h_err=[None if not np.isfinite(x) else round(float(x), 1) for x in eh],
                                 stop=[None if ab else round(float(np.nanmean(stop[g & say[:, t], t, a_])), 1) if (g & say[:, t]).any() else None
                                       for t, ab in zip(T, absent)]))
            ax[0, j].set_title(m_, fontsize=10, fontweight="bold")
            ax[-1, j].set_xlabel("future tubelet (6 raw frames each)")
        for a_ in range(2):
            ax[a_, 0].set_ylabel(f"{'xy'[a_]} error (px)\n+ = ahead, − = behind\nalong GT {'xy'[a_]}-motion")
            m = max(CELL * 1.25, np.percentile(allv[a_], 95) * 1.15 if allv[a_] else CELL * 1.25); ax[a_, 0].set_ylim(-m, m)
            for a2, j2, e2 in lines:
                if a2 != a_:
                    continue
                ec = np.clip(e2, -m * 0.97, m * 0.97); a = ax[a2, j2]
                a.plot(T, ec, color=COL[a2], lw=1.8, marker="o", ms=3, zorder=2)
                hi, lo = e2 > m * 0.97, e2 < -m * 0.97
                a.scatter(T[hi], np.full(hi.sum(), m * 0.97), marker="^", s=30, color=COL[a2], zorder=4)
                a.scatter(T[lo], np.full(lo.sum(), -m * 0.97), marker="v", s=30, color=COL[a2], zorder=4)
        ax[0, 0].set_xticks(T); ax[0, 0].set_xlim(-0.5, TP - 0.5)
        hd = [Line2D([], [], color=COL[0], marker="o", ms=3, label="x: ahead (+) / behind (−)"),
              Line2D([], [], color=COL[1], marker="o", ms=3, label="y: ahead (+) / behind (−)"),
              Line2D([], [], color="#333333", lw=1.1, ls="--", label="if p continued the last seen velocity"),
              Line2D([], [], color="#aaaaaa", lw=1.1, ls="-.", label="if p stopped at the last seen position"),
              Line2D([], [], ls="", marker="o", ms=5, color="red", label="p reads absent (>50% of clips) → 0"),
              Line2D([], [], color="#1baf7a", lw=1.2, label="h (sees the actual frames): readout's own error"),
              Line2D([], [], color="#aaaaaa", ls=":", label="±1 cell = 18 px"),
              Line2D([], [], ls="", marker="^", ms=6, color="#555555", label="beyond the axis range"),
              Patch(color="#bbbbbb", label="object hidden (both samples)"), Patch(color="#e3e3e3", label="object hidden (one sample)")]
        f_.legend(handles=hd, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
        f_.suptitle(f"IntPhysGen v11 · {title} · predictor p ahead/behind error, own decoder (mean over clips read as present)\n{DEC}",
                    fontsize=9.5)
        f_.tight_layout(rect=(0, 0.17, 1, 0.92))
        f_.savefig(OUT / f"fig_{tag}.png", dpi=150); plt.close(f_)
    (OUT / "values.json").write_text(json.dumps(rows, indent=1))
    stamp(OUT, PRES, [V11 / "readings.npz", V11_MID / "readings.npz"])
    print(OUT)


if __name__ == "__main__":
    main()
