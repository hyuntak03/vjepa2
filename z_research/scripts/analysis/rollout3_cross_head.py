#!/usr/bin/env python3
"""**남의 자를 p 에 걸어 본다** — "predictor 가 물체를 없앴나, 자가 못 읽나" 를 가르는 시도.

`p` 는 `LN(target encoder)` = `h` 를 맞추도록 학습됐다. **그래서 `h` 에서 배운 자를 `p` 에 거는 것이
원리적으로 맞다** (같은 공간). `z` 자는 공간이 달라 대조군이고, **실패해도 "정보가 없다" 의 증거가 아니다.**

읽는 법:
  `p@h` 가 `p` 보다 훨씬 오래 물체를 찾는다  → 정보는 있었고 **p 전용 자가 무너진 것**
  `p@h` 도 같은 자리에서 무너진다            → 서로 다른 자 두 개가 일치. **표현 쪽 근거가 세진다**

⚠️ **어느 쪽이든 완전히 가르지는 못한다.** 자 셋 다 **미래 튜블릿 8 개까지만** 학습했다
   (학습셋이 그렇다). v3 의 절벽이 t8~t10 이라 **세 자 모두 그 너머가 외삽**이다.
   완전히 가르려면 (a) 자 없는 잠재 비교 (`|p−h_pos|` vs `|p−h_imp|`) 또는
   (b) **미래 16 튜블릿까지 덮는 학습셋**이 필요하다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/rollout3_cross_head.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
import sys; sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))   # noqa: E702
from rollout3_paths import PRES, WIN, check   # noqa: E402  경로·자 지문은 한 곳에서
RESN, CELL, SPLIT = 144.0, 18.0, 32


def main():
    z = np.load(WIN / "readings.npz", allow_pickle=True)
    check(z, PRES)
    S = json.loads((PRES / "summary.json").read_text())
    bias = json.loads((PRES / "attn_bias_px.json").read_text())
    thr = {r: S["reps"][r]["attn"]["thr_val_fpr5"] for r in S["reps"]}
    POS = z["role"] == "roll"
    L, inf = z["truth"] * RESN, z["in_frame"]
    keys = [k for k in z.files if k.startswith("C") and "@" in k]
    if not keys:
        raise SystemExit("이식 결과가 없다 — rollout3_window_readout.py --reps p --cross h z 를 먼저 돌릴 것")
    out = ["# 남의 자를 `p` 에 걸면 — 이식 (2026-09-24)", "",
           "`p` 자체 자 vs `h` 자 이식 vs `z` 자 이식. 문턱은 **자를 배운 표현의 것**을 쓴다.", ""]
    for c, p in sorted({(int(k.split("_")[0][1:]), int(k.split("_")[1][1:])) for k in keys}):
        tp = p // 2
        vis = inf[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2).all(2)[POS]
        gt = L[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2, 2).mean(2)[POS]
        out += [f"## C{c} / P{p}", "", "| 읽은 값 | 자 | " +
                " | ".join(f"t{t}" for t in range(tp)) + " | 전체 recall | L2 평균 px |",
                "|---|---|" + "---|" * (tp + 2)]
        for nm in ("p", f"p@h", f"p@z"):
            k = f"C{c}_P{p}_{nm}"
            if k not in z.files:
                continue
            head = nm.split("@")[-1]
            v = z[k][POS]
            say = v[..., 2] > thr[head]
            xy = v[..., 0:2] * RESN - np.array(bias[head], np.float32)
            r = [100 * say[vis[:, t], t].mean() for t in range(tp)]
            m = vis & say
            e = np.linalg.norm(xy - gt, axis=-1)[m].mean()
            out.append(f"| `{nm}` | {head} 자 | " + " | ".join(f"{x:.0f}" for x in r) +
                       f" | **{100*say[vis].mean():.1f} %** | {e:.1f} |")
        out.append("")
    # ── 두 자가 **같은 판정**을 내는가. 한계를 재는 것은 이쪽이다 ──────────────────────
    out += ["## 두 자가 같은 판정을 내는가 (`p` 자 vs `h` 자, 같은 `p` 토큰)", ""]
    for c, p in sorted({(int(k.split("_")[0][1:]), int(k.split("_")[1][1:])) for k in keys}):
        tp = p // 2
        vis = inf[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2).all(2)[POS]
        if f"C{c}_P{p}_p@h" not in z.files:
            continue
        a_ = z[f"C{c}_P{p}_p"][POS][..., 2] > thr["p"]
        b_ = z[f"C{c}_P{p}_p@h"][POS][..., 2] > thr["h"]
        hh = z[f"C{c}_P{p}_h"][POS][..., 2] > thr["h"] if f"C{c}_P{p}_h" in z.files else None
        rows = [("일치", lambda t: (a_[vis[:, t], t] == b_[vis[:, t], t]).mean()),
                ("둘 다 '있다'", lambda t: (a_ & b_)[vis[:, t], t].mean()),
                ("둘 다 '없다'", lambda t: (~a_ & ~b_)[vis[:, t], t].mean())]
        if hh is not None:
            rows.append(("**대조: `h` 자를 `h` 에**", lambda t: hh[vis[:, t], t].mean()))
        out += [f"### C{c} / P{p}", "", "| | " + " | ".join(f"t{t}" for t in range(tp)) + " |",
                "|---|" + "---|" * tp]
        for nm, fn in rows:
            out.append(f"| {nm} | " + " | ".join(f"{100*fn(t):.0f}" for t in range(tp)) + " |")
        out.append("")

    out += ["## 무엇을 말할 수 있나", "",
            "**`h` 자는 튜블릿 번호를 모른다** — 토큰 (256, D) 만 받아 읽는다. 그런데 같은 후반 튜블릿에서",
            "`h` 토큰은 읽어 내고 (94~100 %) `p` 토큰은 못 읽는다 (4~39 %).",
            "`p` 는 `LN(target encoder)` = `h` 를 맞추도록 학습된 것이므로, 이는",
            "**`p` 의 후반 토큰이 '물체가 든 encoder 스러운 것' 이 아니라는 뜻**이다.", "",
            "→ 가능한 해석이 둘이고 **둘 다 predictor 쪽 변화다.**",
            "  1. predictor 가 물체를 놓았다",
            "  2. 물체를 들고 있되 **encoder 공간을 벗어난 형태**라 어떤 encoder 자로도 못 찾는다", "",
            "**'자가 고장 났다' 는 이 대조로 배제된다** — 같은 자, 같은 튜블릿, 다른 표현에서 멀쩡하다.", "",
            "## 그래도 남는 단서", "",
            "1. ⚠️ **자 셋 다 미래 8 튜블릿까지만 학습했다** (학습셋이 그렇다). `p` 에게 t8 이후는",
            "   **학습 때 존재한 적 없는 종류의 입력**이다 (그만큼 굴려 본 적이 없다). `h` 대조가",
            "   '창 후반이라 자가 깨진다' 는 배제해 주지만, **`p` 고유의 외삽**까지 배제하지는 못한다.",
            "2. ⚠️ **두 자가 중간 구간에서 서로 다르다** — C16 에서 t5~t12 일치율이 30~63 % 다.",
            "   **절벽의 정확한 위치와 깜빡임은 자에 따라 달라진다.** 위치를 숫자로 인용하지 말 것.",
            "3. `p@z` 는 공간이 달라 대조군이다. 낮게 나오는 것이 정보 부재의 증거가 아니다.", ""]
    f = WIN / "CROSS_HEAD.md"
    f.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"→ {f}")


if __name__ == "__main__":
    main()
