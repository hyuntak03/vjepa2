#!/usr/bin/env python3
"""RollOut v3 — p 의 위치 오차, 미래 튜블릿마다 (창 16 개 × 시나리오 8 개). 그림 하나 = 창 하나, 열 = 시나리오.

기본 (`--out <폴더>/v3_l2_error[_hhead]/`, 2026-09-25 부터 new_archive 에는 안 쓴다 — 거기서 뺀 판이다) — 행 = x · y
  오차   |x̂ − x|, |ŷ − y|  (px, 좌표 치우침 `attn_bias_px.json` 을 뺀 값). 시나리오 안에서 p 가 '있다' 고 한 clip 의 평균

--signed (`v3_signed_error[_hhead]/`) — 행 = x · y, **축마다 앞섬 (+) / 뒤처짐 (−)** (사용자 정의 2026-09-24)
  s_a(t) = 그 튜블릿에서 GT 가 축 a 로 움직이는 방향 (±1, clip 마다). 그 축으로 1 px/튜블릿 미만이면 **마지막으로 움직이던 방향**
           (wall 의 x = 문맥 방향). 그 축으로 **한 번도 안 움직였으면** (flat·wall 의 y) y 는 위 = + 로 둔다 (패널에 표시)
  오차_a = (p̂_a(t) − GT_a(t)) · s_a(t)     + = 그 축의 GT 운동 방향으로 앞섬, − = 뒤처짐
  운동 방향을 부호에 녹였으므로 방향별로 쪼개지 않는다. 떨어지는 ledge·arc 의 y 는 덜 떨어지면 − 다.
  세로 범위 = 줄마다 값의 95 % 가 들어가는 범위 (최소 1 칸 남짓). 밖으로 나간 점은 가장자리에 ▲▼ 로 찍는다.
  기준선 두 개 (GT 만으로 계산, p 와 같은 clip 으로 평균, 같은 부호 규칙):
    진회색 파선  = "본 속도 그대로 연장했다면" — (x0 + v0·(t+1) − GT(t)) · s   x0 · v0 = 문맥 끝 튜블릿 위치 · 속도
    밝은 회색 쇄선 = "마지막 자리에 멈춰 있었다면" — (x0 − GT(t)) · s
    p 가 파선을 따라가면 연장, 쇄선 쪽이면 멈춤, 0 과 쇄선 사이면 움직이되 덜 감.
  (버린 정의: "x 에만 GT 수평 부호" · "화면 기준 + 운동 화살표" · "운동 방향 벡터에 투영한 한 줄" — `_superseded/`)

공통
  빨간 점 = 그 튜블릿에서 clip 의 **절반 넘게** p 가 '없다' 고 한 곳 → 오차 0 으로 둔다
  초록 선 = 같은 계산을 z (context encoder, 창 전체 = 미래 프레임도 본다) 에 — **자 자체의 오차 바닥**
  1 칸 = 18 px = 물체 반폭 (점선)
  자가 물체를 못 읽는 칸 (물체가 화면에 있는데 z 조차 clip 의 10 % 넘게 '없다') 은 그림에 안 칠하고 `values.json` 에만 표시한다
  뺀 칸 없음 — 예전 일괄 규칙 ("48f 이상 마지막 칸 · C4 첫 칸 빼기") 은 버렸다 (C4/P4·P8 의 t0 는 z·h '없다' 0~7 %, 위치 3~9 px)
  가능 clip 만 (`role == roll`).

  --head p (기본) p 자기 자로 p 를 읽는다
  --head h        **h 로 학습한 자**로 p 를 읽는다 (readings 의 `p@h`). 문턱 · 좌표 치우침은 h 의 것 (p 기준 재보정 안 함)

  $P z_research/scripts/figures/plot_v3_l2_error.py --signed [--head h]      # figures/v3_signed_error[_hhead]/
  $P z_research/scripts/figures/plot_v3_l2_error.py --out <new_archive 밖> [--head h]   # 부호 없는 판 (플래그 없이 돌리면 멈춘다)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, WIN, ARCHIVE, check, present_thr, is_identity, stamp       # noqa: E402
DEC = "decoder: position + identity (56 shape x colour + none)" if is_identity(PRES) else "decoder: position + presence"

ARCH = ARCHIVE                                   # R3_FIGROOT 로 바꾼다 (기본 RollOutV3/figures)
RESN, SPLIT, CELL, HFAIL, STILL = 144.0, 32, 18.0, 0.10, 1.0
CS = PS = (4, 8, 16, 32)
LAWS = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "ledge", "wall"]
COL = ("#2a78d6", "#eb6834")                       # x · y (부호 판은 COL[0])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})


def axis_dir(L, X):
    """(n, tp, 2) 축마다 GT 운동 방향 부호 (화면 좌표). 멈춘 칸은 마지막으로 움직이던 부호 (문맥 끝에서 시작).
    한 번도 안 움직인 축은 x → +1 (오른쪽), y → −1 (화면 위 = 앞). 두 번째 반환값 = 그렇게 채운 칸 (never moved)."""
    prev = L[:, SPLIT - 1] - L[:, SPLIT - 3]                                    # 문맥 끝 속도 (2 프레임)
    Xp = np.concatenate([L[:, SPLIT - 2:SPLIT].mean(1)[:, None], X], 1)
    V = Xp[:, 1:] - Xp[:, :-1]
    last = np.where(np.abs(prev) >= STILL / 2, np.sign(prev), 0.0)
    s = np.zeros_like(V)
    for t in range(V.shape[1]):
        cur = np.where(np.abs(V[:, t]) >= STILL, np.sign(V[:, t]), last)
        s[:, t] = cur; last = cur
    never = s == 0
    s[..., 0][never[..., 0]] = 1.0; s[..., 1][never[..., 1]] = -1.0
    return s, never


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--head", default="p", choices=["p", "h"])
    ap.add_argument("--signed", action="store_true")
    ap.add_argument("--out", default=None, help="부호 없는 판 (|오차|) 의 출력 폴더. new_archive 밖이어야 한다")
    args = ap.parse_args()
    key = "p" if args.head == "p" else f"p@{args.head}"
    if args.signed:
        OUT = ARCH / ("v3_signed_error" + ("" if args.head == "p" else f"_{args.head}head"))
    else:
        # 2026-09-25: new_archive 의 v3_l2_error{,_hhead}/ 는 사용자가 지웠다 (README "뺀 것"). 플래그 없는 실행이
        # 그 폴더를 new_archive 안에 되살리지 않도록, 부호 없는 판은 --out 을 명시할 때만 그 자리에 쓴다.
        if args.out is None or ARCH in Path(args.out).resolve().parents or Path(args.out).resolve() == ARCH:
            sys.exit("부호 없는 판 (v3_l2_error) 은 new_archive 에서 뺐다. 본문 그림은 --signed [--head h] 로 만든다.\n"
                     "부호 없는 판이 꼭 필요하면 new_archive 밖 폴더를 --out 으로 준다 (예: 스크래치).")
        OUT = Path(args.out) / ("v3_l2_error" + ("" if args.head == "p" else f"_{args.head}head"))
    z = np.load(WIN / "readings.npz", allow_pickle=True); check(z, PRES)
    thr = present_thr(args.head, PRES); thr_r = present_thr("z", PRES)          # 자 종류마다 뜻이 다르다 (rollout3_paths)
    B = json.loads((PRES / "attn_bias_px.json").read_text())
    bias, bias_r = np.array(B[args.head], np.float32), np.array(B["z"], np.float32)
    POS = z["role"] == "roll"; L = z["truth"][POS] * RESN; sc = z["scenario"][POS]
    OUT.mkdir(parents=True, exist_ok=True)
    nrow = 2
    rows = []
    for c in CS:
        for p in PS:
            tp = p // 2
            X = L[:, SPLIT:SPLIT + p].reshape(-1, tp, 2, 2).mean(2)                  # 진실 (튜블릿)
            v = z[f"C{c}_P{p}_{key}"][POS]; say = v[..., 2] > thr
            rv = z[f"C{c}_P{p}_z"][POS]; rsay = rv[..., 2] > thr_r                  # 자 검사 = z
            d, dr = v[..., 0:2] * RESN - bias - X, rv[..., 0:2] * RESN - bias_r - X  # 예측 − GT (px)
            if args.signed:
                sgn, never = axis_dir(L, X)
                err, rerr = d * sgn, dr * sgn                                        # (n, tp, 2) 축마다 앞섬 +
                x0 = L[:, SPLIT - 2:SPLIT].mean(1); v0 = x0 - L[:, SPLIT - 4:SPLIT - 2].mean(1)   # 문맥 끝 위치 · 속도 (튜블릿당)
                ext = (x0[:, None] + v0[:, None] * (np.arange(tp) + 1)[None, :, None] - X) * sgn    # 본 속도 그대로
                stop = (x0[:, None] - X) * sgn                                                   # 멈춰 있었다면
            else:
                err, rerr = np.abs(d), np.abs(dr)
            vis = z["in_frame"][POS][:, SPLIT:SPLIT + p].reshape(-1, tp, 2).all(2)
            T = np.arange(tp)
            f_, ax = plt.subplots(nrow, 8, figsize=(17, 5.2 if args.signed else 4.6), sharex=True, sharey="row", squeeze=False)
            ymax, ymin = [1.0] * nrow, [0.0] * nrow
            allv = [[] for _ in range(nrow)]; lines = []                          # 부호 판: 줄마다 95 % 범위로 자른다
            for j, law in enumerate(LAWS):
                k = sc == law
                absent = say[k][:, T].mean(0) <= 0.5
                hfail = np.array([1 - rsay[k & vis[:, t], t].mean() > HFAIL if (k & vis[:, t]).any() else True for t in T])
                for a_ in range(nrow):
                    e = np.array([err[k & say[:, t], t, a_].mean() if (k & say[:, t]).any() else 0.0 for t in T])
                    e[absent] = 0.0
                    ymax[a_] = max(ymax[a_], np.nanmax(e)); ymin[a_] = min(ymin[a_], np.nanmin(e))
                    a = ax[a_, j]
                    er = np.array([rerr[k & rsay[:, t], t, a_].mean() if (k & rsay[:, t]).any() else np.nan for t in T])
                    a.plot(T, er, color="#1baf7a", lw=1.2, zorder=1)
                    if args.signed:
                        allv[a_] += list(np.abs(e[~absent])) + list(np.abs(er[np.isfinite(er)]))
                        lines.append((a_, j, e))
                        for ref, st in ((ext, dict(color="#333333", lw=1.1, ls="--")), (stop, dict(color="#aaaaaa", lw=1.1, ls="-."))):
                            rv_ = np.array([ref[k & say[:, t], t, a_].mean() if (k & say[:, t]).any() else np.nan for t in T])
                            rv_[absent] = np.nan
                            a.plot(T, rv_, zorder=1, **st)
                        if a_ == 1 and never[k][..., 1].mean() > 0.5:
                            a.text(0.03, 0.93, "no GT y-motion: + = above", transform=a.transAxes, fontsize=6.5, color="#555555", va="top")
                    else:
                        a.plot(T, e, color=COL[a_], lw=1.8, marker="o", ms=3, zorder=2)
                    a.scatter(T[absent], np.zeros(absent.sum()), s=22, color="red", zorder=3)
                    a.axhline(CELL, color="#aaaaaa", lw=0.7, ls=":")
                    if args.signed:
                        a.axhline(-CELL, color="#aaaaaa", lw=0.7, ls=":"); a.axhline(0, color="#777777", lw=0.6)
                    a.spines[["top", "right"]].set_visible(False)
                    rows.append((c, p, law, "xy"[a_],
                                 [(int(t), round(float(x), 1), bool(ab), bool(hf)) for t, x, ab, hf in zip(T, e, absent, hfail)]))
                ax[0, j].set_title(law, fontsize=9.5, fontweight="bold")
                ax[-1, j].set_xlabel("future tubelet")
            for a_ in range(nrow):
                if args.signed:
                    ax[a_, 0].set_ylabel(f"{'xy'[a_]} error (px)\n+ = ahead, − = behind\nalong GT {'xy'[a_]}-motion")
                    m = max(CELL * 1.25, np.percentile(allv[a_], 95) * 1.15 if allv[a_] else CELL * 1.25); ax[a_, 0].set_ylim(-m, m)
                    for a2, j2, e2 in lines:                                          # 범위 밖은 가장자리에 ▲▼
                        if a2 != a_: continue
                        ec = np.clip(e2, -m * 0.97, m * 0.97); a = ax[a2, j2]
                        a.plot(T, ec, color=COL[a2], lw=1.8, marker="o", ms=3, zorder=2)
                        hi, lo = e2 > m * 0.97, e2 < -m * 0.97
                        a.scatter(T[hi], np.full(hi.sum(), m * 0.97), marker="^", s=30, color=COL[a2], zorder=4)
                        a.scatter(T[lo], np.full(lo.sum(), -m * 0.97), marker="v", s=30, color=COL[a2], zorder=4)
                else:
                    ax[a_, 0].set_ylabel(f"|{'xy'[a_]} error| (px)")
                    ax[a_, 0].set_ylim(-2, ymax[a_] * 1.1 + 2)
            ax[0, 0].set_xticks(range(0, tp, 2 if tp > 4 else 1)); ax[0, 0].set_xlim(-0.5, tp - 0.5)
            if args.signed:
                hd = [Line2D([], [], color=COL[0], marker="o", ms=3, label="x: ahead (+) / behind (−)"),
                      Line2D([], [], color=COL[1], marker="o", ms=3, label="y: ahead (+) / behind (−)"),
                      Line2D([], [], ls="", marker="^", ms=6, color="#555555", label="beyond the axis range"),
                      Line2D([], [], color="#333333", lw=1.1, ls="--", label="if p continued the seen velocity"),
                      Line2D([], [], color="#aaaaaa", lw=1.1, ls="-.", label="if p stopped at the last seen position")]
            else:
                hd = [Line2D([], [], color=COL[0], marker="o", ms=3, label="|x error|"), Line2D([], [], color=COL[1], marker="o", ms=3, label="|y error|")]
            hd += [Line2D([], [], ls="", marker="o", ms=5, color="red", label="p reads absent (>50% of clips) → 0"),
                   Line2D([], [], color="#1baf7a", lw=1.2, label="z (sees the actual frames): readout's own error"),
                   Line2D([], [], color="#aaaaaa", ls=":", label="±1 cell = 18 px" if args.signed else "1 cell = 18 px")]
            f_.legend(handles=hd, loc="lower center", ncol=min(len(hd), 5), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
            who = "own decoder" if args.head == "p" else f"decoder trained on {args.head}"
            kind = "ahead/behind error" if args.signed else "position error"
            f_.suptitle(f"RollOut v3 · context {c}f / prediction {p}f · predictor p {kind}, read with the {who} (mean over clips read as present) · {DEC}", fontsize=10)
            f_.tight_layout(rect=(0, 0.1 if args.signed else 0.06, 1, 0.94))
            f_.savefig(OUT / f"fig_C{c}_P{p}.png", dpi=150); plt.close(f_)
    (OUT / "values.json").write_text(json.dumps(rows))
    stamp(OUT, PRES, [WIN / "readings.npz"])                 # 어느 자로 그렸나 (_decoder.json)
    print(OUT)


if __name__ == "__main__":
    main()
