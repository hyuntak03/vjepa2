#!/usr/bin/env python3
"""RollOut v3 에서 **옛 자 vs 새 자** — 같은 test 셋에서 나란히.

각 자는 **자기** 문턱 (`summary.json` 의 thr_val_fpr5) 과 **자기** 치우침 (`attn_bias_px.json`) 으로 채점하고,
readings 가 그 자로 읽혔는지 지문으로 확인한다 (`rollout3_paths.check`).

이번 비교의 요점은 **구조물 시나리오 (ledge · wall · ramp) 에서 존재 판정이 평면과 같아졌는가** 다.
옛 학습셋은 구조물이 물체 없는 clip 에만 있어서 자가 "구조물 = 없음" 을 배울 수 있었다 (데이터 README §1).
v3 에는 음성이 0 개라 **recall 만** 정의된다 — 오탐률은 학습셋 test 에서만 잰다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/rollout3_decoder_compare.py \
      --old <옛자폴더>:<옛readings폴더> --new presence:windows
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import EXP, FIGROOT, check   # noqa: E402

RESN, CELL, SPLIT = 144.0, 18.0, 32
SCEN = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "ledge", "wall"]
WINDOWS = [(c, p) for c in (4, 8, 16, 32) for p in (4, 8, 16, 32)]


def load(spec):
    """'자폴더:readings폴더' → (이름, readings, 문턱, 치우침)."""
    dec, win = spec.split(":")
    pres, w = EXP / dec, EXP / win
    z = np.load(w / "readings.npz", allow_pickle=True)
    check(z, pres)
    S = json.loads((pres / "summary.json").read_text())
    thr = {r: S["reps"][r]["attn"]["thr_val_fpr5"] for r in S["reps"]}
    bias = json.loads((pres / "attn_bias_px.json").read_text())
    return dict(name=f"{dec} ({S.get('train_set', 'training_v6')})", z=z, thr=thr, bias=bias)


def stats(D, rep, sel_scen=None, windows=WINDOWS):
    """(recall %, L2 평균 px — '있다' 칸만). 창들을 합친다. 끝 튜블릿 가장자리 효과는 빼지 않는다 (두 자 공통)."""
    z = D["z"]; POS = z["role"] == "roll"; L = z["truth"] * RESN; inf = z["in_frame"]; sc = z["scenario"]
    keep = POS & (np.isin(sc, sel_scen) if sel_scen is not None else True)
    n_say = n_vis = 0; e_sum = e_n = 0.0
    for c, p in windows:
        k = f"C{c}_P{p}_{rep}"
        if k not in z.files:
            continue
        tp = p // 2
        vis = inf[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2).all(2)[keep]
        gt = L[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2, 2).mean(2)[keep]
        v = z[k][keep]
        say = v[..., 2] > D["thr"][rep]
        xy = v[..., 0:2] * RESN - np.array(D["bias"][rep], np.float32)
        n_say += int((say & vis).sum()); n_vis += int(vis.sum())
        m = say & vis
        e = np.linalg.norm(xy - gt, axis=-1)[m]
        e_sum += float(e.sum()); e_n += len(e)
    return 100 * n_say / max(n_vis, 1), e_sum / max(e_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="옛 자 — 자폴더:readings폴더")
    ap.add_argument("--new", default="presence:windows")
    ap.add_argument("--out", default=None,
                    help="출력 md. 기본은 **새 readings 폴더**/DECODER_COMPARE.md (그림 폴더가 아니라 기록 쪽에 둔다)")
    a = ap.parse_args()
    O, N = load(a.old), load(a.new)
    out = [f"# v3 에서 옛 자 vs 새 자 (2026-09-24)", "",
           f"- 옛 자: `{O['name']}`  ·  새 자: `{N['name']}`",
           "- 각 자는 **자기** 문턱·치우침으로 채점. readings 는 지문으로 짝을 확인했다.",
           "- v3 에는 음성이 0 개라 **recall 만** 정의된다. L2 는 '있다' 고 한 칸만, px (1 칸 = 18 px).", "",
           "## 16 창을 합쳐서", "", "| 표현 | recall 옛 → 새 | L2 평균 옛 → 새 (px) |", "|---|---|---|"]
    for rep in ("z", "h", "p"):
        (ro, eo), (rn, en) = stats(O, rep), stats(N, rep)
        out.append(f"| `{rep}` | {ro:.1f} → **{rn:.1f} %** | {eo:.1f} → **{en:.1f}** |")
    for title, wins in (("C16 / P32 창", [(16, 32)]), ("16 창을 합쳐서", WINDOWS)):
        out += ["", f"## 시나리오별 recall — {title}", "",
                "| 시나리오 | " + " | ".join(f"`{r}` 옛 → 새" for r in ("z", "h", "p")) + " |",
                "|---|---|---|---|"]
        for s in SCEN:
            cells = []
            for rep in ("z", "h", "p"):
                ro, _ = stats(O, rep, [s], wins); rn, _ = stats(N, rep, [s], wins)
                cells.append(f"{ro:.1f} → **{rn:.1f}**")
            out.append(f"| `{s}` | " + " | ".join(cells) + " |")
    out += ["", "⚠️ `p` 의 recall 은 **자의 성질이 아니라 predictor 의 성질**이 섞인 값이다 (후반 튜블릿에서 물체를 놓는다).",
            "자 비교는 `z`·`h` 줄로 한다 — 둘은 창 전체를 보므로 자의 천장이다.",
            "⚠️ 화면 가장자리 칸 (마지막 튜블릿, C=4 의 첫 튜블릿) 은 빼지 않았다 — 두 자에 똑같이 들어 있다.", "",
            "## 재현", "", "```bash", "P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python",
            f"$P z_research/scripts/analysis/rollout3_decoder_compare.py --old {a.old} --new {a.new}", "```"]
    f = Path(a.out) if a.out else EXP / a.new.split(":")[1] / "DECODER_COMPARE.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(out) + "\n")
    print("\n".join(out)); print(f"→ {f}")


if __name__ == "__main__":
    main()
