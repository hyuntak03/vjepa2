#!/usr/bin/env python3
"""v11 빈 장면의 '있음' 오탐은 **가림막을 가리키는가** — 자가 읽은 위치 vs 가림막 픽셀 (CPU, 1 분 안팎).

빈 장면 (vanish `pos_b`, 물체가 처음부터 끝까지 없다) 에서 자가 '있음' 이라고 한 미래 튜블릿마다:
  판 자리 = |가림 조건 빈 장면 프레임 − 같은 배경 · 같은 운동의 visible 빈 장면 프레임| 의 최대 채널 차 > 25 (판과 그 그림자)
            → 반 칸 (9 px) 팽창. 두 장면은 카메라 · 배경 · 쐐기가 같고 판만 다르다
  적중    = 읽은 위치 (xy − 자 치우침, 288 px) 가 판 자리 안
  귀무    = 판 자리가 물체가 갈 수 있는 띠 (v 20..152) 에서 차지하는 넓이 비율 — 아무 데나 찍었을 때 적중할 확률
튜블릿의 두 번째 샘플 프레임을 쓴다 (정지 조건의 판은 움직이므로 튜블릿 안에서도 바뀐다 — 근사).

  R3_DECODER=identity R3_OUT=identity $P z_research/scripts/analysis/v11_panel_false_alarm.py
"""
from __future__ import annotations
import csv, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, EXP, TAG, check, present_thr   # noqa: E402

FR = Path("/local_datasets/world/world_analysis/IntPhysGen_v11")
SFX = f"_{TAG}" if TAG else ""
RES, W, DIL = 144.0, 288, 9


def mot(c):
    return "static" if c.startswith("static") else ("flat" if "flat" in c else "ramp")


def frame(fn, s):
    return np.asarray(Image.open(FR / fn / f"{3*s:06d}.png").convert("RGB"), np.int16)


def dilate(m, r):
    """정사각 팽창 (scipy 없이): 누적합으로 (2r+1)^2 창 안에 참이 하나라도 있으면 참."""
    c = np.pad(m.astype(np.int32), r).cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) > 0


def main():
    meta = {m["name"]: m for m in csv.DictReader((FR / "metadata.csv").open())}
    bias = json.loads((PRES / "attn_bias_px.json").read_text())
    out = {}
    refs = {}
    for name, tim in ((f"v11{SFX}", "last"), (f"v11_mid{SFX}", "mid")):
        z = np.load(EXP / name / "readings.npz", allow_pickle=True); check(z, PRES)
        vid, cond, obj, k = z["video_id"], z["condition"], z["obj"], z["sym_k"].astype(int)
        if tim == "last":                                           # visible 빈 장면 = 기준 프레임 (배경 · 운동별 하나)
            for i in np.where(~obj & (k == 0))[0]:
                m = meta[vid[i]]; key = (mot(cond[i]), m["env"])
                refs.setdefault(key, m["file_name"])
        sel = np.where(~obj & (k > 0))[0]

        def work(i):
            m = meta[vid[i]]; ref = refs[(mot(cond[i]), m["env"])]
            res = []
            for t in range(8):
                s = 16 + 2 * t + 1
                mask = dilate(np.abs(frame(m["file_name"], s) - frame(ref, s)).max(-1) > 25, DIL)
                band = mask[20:153].mean()
                row = [band]
                for r in ("p", "h"):
                    v = z[r][i, -8:][t]
                    x, y = v[0] * RES + RES - bias[r][0], v[1] * RES + RES - bias[r][1]
                    xi, yi = int(np.clip(round(x), 0, W - 1)), int(np.clip(round(y), 0, W - 1))
                    row += [float(v[2] > present_thr(r, PRES)), float(mask[yi, xi])]
                res.append(row)
            return i, np.array(res)
        with ThreadPoolExecutor(32) as ex:
            R = dict(ex.map(work, sel))
        for mo in ("static", "flat", "ramp"):
            ii = [i for i in sel if mot(cond[i]) == mo]
            A = np.stack([R[i] for i in ii])                          # (n, 8, 5): band, p_say, p_hit, h_say, h_hit
            rec = dict(n_clip=len(ii), null_area=round(float(A[..., 0].mean()), 3))
            for j, r in ((1, "p"), (3, "h")):
                say = A[..., j] > 0
                rec[r] = dict(say=round(float(say.mean()), 3),
                              on_panel_given_say=round(float(A[..., j + 1][say].mean()), 3) if say.any() else None)
            out[f"{tim}|{mo}"] = rec
            print(f"{tim:4s} {mo:6s} n={len(ii):3d}  귀무 (판 넓이) {rec['null_area']:.2f}  "
                  + "  ".join(f"{r}: '있음' {100*rec[r]['say']:.0f}% → 그중 판 위 {100*(rec[r]['on_panel_given_say'] or 0):.0f}%" for r in ("p", "h")),
                  flush=True)
    o = ROOT / f"z_research/RollOutV3/audit/training_v8/panel_false_alarm_{PRES.name}.json"
    o.write_text(json.dumps(dict(decoder=str(PRES), rule="present_thr", dilate_px=DIL, results=out), indent=1))
    print(f"→ {o}")


if __name__ == "__main__":
    main()
