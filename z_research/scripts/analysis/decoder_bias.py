#!/usr/bin/env python3
"""자 폴더 (`exp_results/<R3_DECODER>`) 의 **좌표 치우침** `attn_bias_px.json` — presence 자 · 정체 자 공통.

정의 (presence_readout_compare.bias 와 같다): 학습셋 **test 분할 × 양성 칸**의 평균 (pred − truth) px, 표현마다 (x, y).
  presence 자  `<rep>/preds_attn.npz` 의 pred · truth · test · pos
  정체 자      `<rep>/preds.npz` 의 xy · truth · test · cls (0..55 = 조합 = 양성)
쓰는 쪽 (그림 스크립트) 은 읽은 좌표에서 이 값을 뺀다.

  R3_DECODER=identity $P z_research/scripts/analysis/decoder_bias.py            # → exp_results/identity/attn_bias_px.json
  $P z_research/scripts/analysis/decoder_bias.py --check                         # 쓰지 않고 기존 파일과 비교만
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout3_paths import PRES, is_identity   # noqa: E402

RES = 144.0


def compute(pres: Path = PRES) -> dict:
    ident = is_identity(pres); out = {}
    for rep in ("p", "z", "h"):
        f = Path(pres) / rep / ("preds.npz" if ident else "preds_attn.npz")
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        xy = z["xy"] if ident else z["pred"]
        pos = ((z["cls"] >= 0) & (z["cls"] < 56)) if ident else z["pos"]
        m = z["test"][:, None] & pos
        e = (xy - z["truth"]) * RES
        out[rep] = [round(float(e[m][:, 0].mean()), 2), round(float(e[m][:, 1].mean()), 2)]
    return out


if __name__ == "__main__":
    b = compute()
    f = PRES / "attn_bias_px.json"
    if "--check" in sys.argv:
        old = json.loads(f.read_text()) if f.exists() else None
        print(f"{PRES.name}: 계산 {b}\n{'':{len(PRES.name)}}  파일 {old}\n  같음: {old == b}")
    else:
        f.write_text(json.dumps(b, indent=1)); print(f"→ {f}  {b}")
