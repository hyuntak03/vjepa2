#!/usr/bin/env python3
"""RollOutV3 의 **경로와 자(decoder) 지문을 한 곳에서** 정한다. 스크립트마다 경로를 하드코딩하지 않는다.

환경변수 두 개로 고른다 (안 주면 옛 동작 그대로):
  R3_DECODER  자 폴더 (exp_results/<이름>).                                  기본 presence
  R3_OUT      결과 꼬리표. 주면 exp_results/windows_<꼬리표>/ 와
              figures/<꼬리표>/{windows,examples,readout,windows_gif}/ 로 쓴다.   기본 없음 (windows/, figures/<하위>/)

  예) R3_OUT=training_v8 python z_research/scripts/analysis/rollout3_window_readout.py --reps p z h --gpus 8

⚠️ **readings 에는 자가 구워져 있다.** `readings.npz` 는 특징이 아니라 자가 읽은 좌표·presence 다.
   자를 다시 학습하면 문턱·치우침 (`summary.json`, `attn_bias_px.json`) 은 새 자 것이 되는데 readings 는
   옛 자 것이라, 섞으면 **옛 읽은 값에 새 문턱을 거는 조용히 틀린 그림**이 나온다 (2026-09-24 에 막았다).
   → 쓰는 쪽은 `fingerprint()` 를 싣고, 읽는 쪽은 `check()` 를 부른다.

  $P z_research/scripts/analysis/rollout3_paths.py     # 지금 경로·자 지문·readings 지문
"""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
EXP = ROOT / "z_research/RollOutV3/exp_results"
FIGROOT = ROOT / "z_research/RollOutV3/figures"
DECODER = os.environ.get("R3_DECODER", "presence")
TAG = os.environ.get("R3_OUT", "")
PRES = EXP / DECODER                                           # 자 (readout_attn.pt · summary.json · attn_bias_px.json)
WIN = EXP / (f"windows_{TAG}" if TAG else "windows")           # v3 16 창에서 읽은 값


def fig(sub: str) -> Path:
    """그림 폴더. 꼬리표가 있으면 figures/<꼬리표>/<sub>."""
    return FIGROOT / TAG / sub if TAG else FIGROOT / sub


def fingerprint(pres: Path = PRES, reps=("p", "z", "h")) -> str:
    """p·z·h 자 가중치 파일의 sha1 앞 12 자리 (없는 파일은 'missing' 으로 섞는다)."""
    h = hashlib.sha1()
    for r in reps:
        f = Path(pres) / r / "readout_attn.pt"
        h.update(r.encode()); h.update(f.read_bytes() if f.exists() else b"missing")
    return h.hexdigest()[:12]


def check(readings, pres: Path = PRES):
    """`readings` (np.load 결과) 가 **지금 자**로 읽힌 것인지. 아니면 죽는다."""
    got = str(readings["decoder_fp"]) if "decoder_fp" in readings.files else None
    want = fingerprint(pres)
    if got != want:
        raise SystemExit(
            f"\n⚠️ {WIN}/readings.npz 는 **다른 자**로 읽혔다 (readings {got} vs 자 {pres.name} {want}).\n"
            f"   새 문턱·치우침을 옛 읽은 값에 걸면 조용히 틀린다. 16 창을 이 자로 다시 읽을 것 (RERUN.md §3),\n"
            f"   또는 R3_DECODER 로 그 readings 를 읽은 자를 고를 것.\n")


if __name__ == "__main__":
    import numpy as np
    print(f"자       {PRES}   지문 {fingerprint()}")
    print(f"readings {WIN}")
    f = WIN / "readings.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        print(f"         지문 {str(z['decoder_fp']) if 'decoder_fp' in z.files else '(없음)'}")
    print(f"그림     {fig('<하위>')}")
