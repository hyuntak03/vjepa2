#!/usr/bin/env python3
"""RollOut v3 16 창 — **자가 그 창에 전이됐는가** 와 p 의 위치·존재 요약.

창이 바뀌면 RoPE 격자가 바뀐다. 자는 `window_size=32` 짜리 학습셋에서 배웠으므로
**다른 창에서도 통한다는 보장이 없다.** 그 검증이 `z`·`h` 다 — 둘은 창 전체를 보므로
위치를 못 맞히면 그건 표현이 아니라 **자가 그 창에서 깨진 것**이고, 그 창은 판정 불가다.

v3 에는 **음성이 0 개다** (물체가 f16~f63 내내 화면 안). 그래서 문턱은 못 고치지만
**라벨 없는 검증**은 된다 — `z` 가 "없다" 고 말하는 비율이 곧 자의 오작동률이다.

  ⚠️ 좌표는 학습셋 held-out 에서 잰 **attn 편향을 뺀 값**으로 잰다 (`attn_bias_px.json`).
  ⚠️ 위치 오차는 **"있다" 고 답한 칸에서만** 잰다 (사용자 지시 2026-09-23).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/rollout3_window_summary.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
import sys; sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))   # noqa: E702
from rollout3_paths import PRES, WIN as RES, check   # noqa: E402  경로·자 지문은 한 곳에서
RESN, CELL, SPLIT = 144.0, 18.0, 32
CTX, PRD = (4, 8, 16, 32), (4, 8, 16, 32)
SCEN = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "ledge", "wall"]


def thresholds(d):
    """문턱을 **학습 산출물에서 읽는다.**

    ⚠️ 하드코딩하면 자를 다시 학습했을 때 **조용히 옛 문턱으로** 그려진다 (2026-09-24 에 이걸 막았다).
       값은 `summary.json` 의 `thr_val_fpr5` — 학습셋 val 음성 5 % 오탐 지점이다.
    """
    S = json.loads((d / "summary.json").read_text())
    return {r: S["reps"][r]["attn"]["thr_val_fpr5"] for r in S["reps"]}


THR = None   # thresholds(PRES) 로 채운다


def main():
    global THR
    THR = thresholds(PRES)
    z = np.load(RES / "readings.npz", allow_pickle=True)
    check(z, PRES)                                  # 이 readings 가 지금 자로 읽혔는가
    bias = json.loads((PRES / "attn_bias_px.json").read_text())
    L, inf, sc, role = z["truth"], z["in_frame"], z["scenario"], z["role"]
    POS = role == "roll"
    # 이식 키 (p@h 등) 는 뺀다 — 표현 이름 셋만
    reps = sorted({k.split("_")[-1] for k in z.files if k.startswith("C")} & {"p", "z", "h"}, key="pzh".index)

    out = ["# RollOut v3 — 16 창 자 전이 검증", "",
           f"자 = attn (attention pooling + head 2 개), 학습셋에서 frozen. 표현 {reps}. "
           f"1 칸 = {CELL:.0f} px. 가능 clip {int(POS.sum()):,} 개.", "",
           "**읽는 법** — `z`·`h` 는 창 전체를 보므로 **자의 천장**이다. 그 줄이 1 칸을 넘거나 "
           "'없다' 비율이 크면 **그 창은 자가 안 통하는 것**이라 `p` 도 못 읽는다.", ""]

    hdr = "| 창 (C/P) | " + " | ".join(
        f"{r} 없다% | {r} 오차(칸)" for r in reps) + " |"
    out += [hdr, "|" + "---|" * (1 + 2 * len(reps))]
    grid = {}
    for c in CTX:
        for p in PRD:
            tp = p // 2
            gt = L[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2, 2).mean(2) * RESN
            vis = inf[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2).all(2)
            cells = []
            for r in reps:
                k = f"C{c}_P{p}_{r}"
                if k not in z.files:
                    cells += ["–", "–"]; continue
                v = z[k][POS]
                xy = v[..., 0:2] * RESN - np.array(bias[r], np.float32)
                say = v[..., 2] > THR[r]
                m = vis[POS]
                absent = 100 * (1 - say[m].mean())
                e = np.linalg.norm(xy - gt[POS], axis=-1)[m & say] / CELL
                grid[(c, p, r)] = (absent, float(e.mean()))
                cells += [f"{absent:.1f}", f"{e.mean():.2f}"]
            out.append(f"| C{c}/P{p} | " + " | ".join(cells) + " |")

    # 창을 가로지르는 폭 — 자가 창에 흔들리는가
    out += ["", "## 창에 따른 흔들림 (자 자신)", ""]
    for r in reps:
        a = [grid[(c, p, r)][0] for c in CTX for p in PRD if (c, p, r) in grid]
        e = [grid[(c, p, r)][1] for c in CTX for p in PRD if (c, p, r) in grid]
        if a:
            out.append(f"- `{r}` — '없다' {min(a):.1f}~{max(a):.1f} % · 오차 {min(e):.2f}~{max(e):.2f} 칸")

    # 시나리오별 (기본 창)
    out += ["", "## 시나리오별 (C16/P32)", "", "| 시나리오 | " +
            " | ".join(f"{r} 없다% | {r} 오차(칸)" for r in reps) + " |",
            "|" + "---|" * (1 + 2 * len(reps))]
    c, p = 16, 32; tp = p // 2
    gt = L[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2, 2).mean(2) * RESN
    vis = inf[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2).all(2)
    for s in SCEN:
        k_ = POS & (sc == s)
        cells = []
        for r in reps:
            key = f"C{c}_P{p}_{r}"
            if key not in z.files:
                cells += ["–", "–"]; continue
            v = z[key][k_]
            xy = v[..., 0:2] * RESN - np.array(bias[r], np.float32)
            say = v[..., 2] > THR[r]; m = vis[k_]
            e = np.linalg.norm(xy - gt[k_], axis=-1)[m & say] / CELL
            cells += [f"{100*(1-say[m].mean()):.1f}", f"{e.mean():.2f}"]
        out.append(f"| {s} | " + " | ".join(cells) + " |")

    # ── 창 독립성: **같은 튜블릿**을 다른 창에서 재면 같은 값이 나오는가 ────────────────
    #    v3 를 만든 이유가 이것이다 — v2 의 dynamics 해석이 C16/P16 이라는 창의 산물인지 가른다.
    #    t0·t1 은 P=4 에도 있으므로 네 P 를 전부 비교할 수 있다.
    out += ["", "## 창 독립성 — **같은 튜블릿**을 다른 창에서 재면", "",
            "미래 튜블릿 t0·t1 은 네 P 에 모두 있다. 같은 문맥 C 에서 P 만 바꿔 그 두 칸을 다시 재면, "
            "창이 판독을 바꾸지 않는 한 값이 같아야 한다.", ""]
    for r in reps:
        out += [f"### `{r}`", "", "| C \\ P | 4 | 8 | 16 | 32 | 폭 |", "|---|---|---|---|---|---|"]
        for c in CTX:
            row, vals = [], []
            for p in PRD:
                tp = p // 2
                gt = L[:, SPLIT:SPLIT + 4].reshape(len(L), 2, 2, 2).mean(2) * RESN   # t0, t1 만
                vis = inf[:, SPLIT:SPLIT + 4].reshape(len(L), 2, 2).all(2)
                v = z[f"C{c}_P{p}_{r}"][POS][:, :2]
                xy = v[..., 0:2] * RESN - np.array(bias[r], np.float32)
                say = v[..., 2] > THR[r]; m = vis[POS]
                e = float(np.linalg.norm(xy - gt[POS], axis=-1)[m & say].mean() / CELL)
                a_ = 100 * (1 - say[m].mean())
                vals.append(e); row.append(f"{e:.2f} ({a_:.0f}%)")
            out.append(f"| C{c} | " + " | ".join(row) + f" | **{max(vals)-min(vals):.2f} 칸** |")
        out.append("")
    # 더 긴 겹침: t0..t7 은 P=16 과 P=32 에 모두 있다 → 궤적 모양 자체를 비교한다
    out += ["### 겹치는 8 칸 전부 (C16, P=16 vs P=32)", "", "| 표현 | P | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in reps:
        for p in (16, 32):
            gt8 = L[:, SPLIT:SPLIT + 16].reshape(len(L), 8, 2, 2).mean(2) * RESN
            vis8 = inf[:, SPLIT:SPLIT + 16].reshape(len(L), 8, 2).all(2)[POS]
            v = z[f"C16_P{p}_{r}"][POS][:, :8]
            xy = v[..., 0:2] * RESN - np.array(bias[r], np.float32)
            say = v[..., 2] > THR[r]
            e = np.linalg.norm(xy - gt8[POS], axis=-1) / CELL
            out.append(f"| `{r}` | {p} | " + " | ".join(
                f"{e[:, t][vis8[:, t] & say[:, t]].mean():.2f}" for t in range(8)) + " |")
    out += ["", "**오차가 튜블릿을 따라 커지는 모양이 P 와 무관하게 같다.** 예측을 두 배로 늘려도 "
            "앞의 8 칸에서 predictor 가 하는 일이 달라지지 않는다는 뜻이다.", ""]

    out += ["칸 값은 **t0·t1 두 칸의 위치 오차 (칸)**, 괄호는 그 두 칸의 '없다' 비율. "
            "폭이 자의 재현성 (~1 px = 0.06 칸) 수준이면 **창은 판독을 바꾸지 않는다.**", ""]

    # ── 창 가장자리: 첫/마지막 튜블릿에서 자가 흔들린다 (z·h 가 같이 흔들리므로 자 쪽 성질) ──
    out += ["", "## ⚠️ 창 **가장자리** 튜블릿은 빼고 읽는다", "",
            "`z` 의 튜블릿별 '없다' 비율 (%, 진실이 화면 안인 칸만):", "", "```"]
    for c, p in ((4, 16), (4, 32), (16, 32), (32, 32)):
        tp = p // 2
        vis = inf[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2).all(2)[POS]
        line = []
        for r in ("z", "h"):
            say = z[f"C{c}_P{p}_{r}"][POS][..., 2] > THR[r]
            line.append(f"C{c}/P{p} {r}: " + " ".join(
                f"{100*(1-say[vis[:, t], t].mean()):3.0f}" for t in range(tp)))
        out += line
    out += ["```", ""]
    # ⚠️ 이 표는 **읽은 값에서 계산한다** (2026-09-24 전에는 첫 자의 수치가 글자로 박혀 있어서
    #    자를 바꿔도 옛 수치가 찍혔다). 시나리오별 `z` '없다' % — 가장자리 칸 vs 가운데 칸.
    sc_all = z["scenario"][POS]

    def absent(c, p, t, s_=None):
        tp = p // 2
        vis = inf[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2).all(2)[POS]
        say = z[f"C{c}_P{p}_z"][POS][..., 2] > THR["z"]
        k = vis[:, t] & (sc_all == s_ if s_ else True)
        return 100 * (1 - say[k, t].mean()) if k.any() else float("nan")

    mid = [absent(c, p, t) for c, p in ((16, 32), (32, 32)) for t in range(1, p // 2 - 1)]
    out += [f"가운데 칸 (첫·마지막 제외, C16·C32 / P32) 의 `z` '없다' 는 {min(mid):.0f}~{max(mid):.0f} % 다. "
            "**양 끝**을 시나리오별로 보면:", "",
            "| 시나리오 | 마지막 칸 C16/P32 | 마지막 칸 C32/P32 | 첫 칸 C4/P32 | 첫 칸 C8/P32 (대조) |",
            "|---|---|---|---|---|"]
    for s_ in SCEN:
        out.append(f"| `{s_}` | {absent(16, 32, 15, s_):.1f} | {absent(32, 32, 15, s_):.1f} | "
                   f"{absent(4, 32, 0, s_):.1f} | {absent(8, 32, 0, s_):.1f} |")
    out += ["", "원인은 둘이다 — **마지막 칸**은 32f 를 달린 물체가 화면 가장자리에 붙는 것 (멀리 가는 법칙에서 크다), "
            "**C4 의 첫 칸**은 문맥이 2 튜블릿뿐이라 경계 바로 뒤가 불안정한 것 (C8 이상에서 사라진다).", "",
            "**규칙** — 총 창이 48f 이상이면 **마지막 튜블릿을 뺀다.** 문맥이 C4 면 **첫 미래 튜블릿을 뺀다.** "
            "둘 다 `z`·`h` 가 같이 흔들리므로 predictor 의 성질이 아니다.", ""]

    out += ["", "⚠️ `p` 의 오차는 **자의 오차가 아니라 측정값이다** — predictor 가 문맥만 보고 만든 "
            "미래라 진실과 벌어지는 것이 결과다. 자의 상태는 `z`·`h` 줄로만 판단한다.", ""]
    f = RES / "WINDOW_SUMMARY.md"
    f.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"→ {f}")


if __name__ == "__main__":
    main()
