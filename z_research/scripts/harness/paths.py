#!/usr/bin/env python3
"""절대경로의 python 쪽 입구. `paths.env` 하나만 읽는다.

    from paths import ROOT, CKPT_ROOT, expand        # harness 안 (sys.path[0])
    sys.path.insert(0, ".../z_research/scripts/harness"); from paths import ...

왜 있나 — 레포를 옮길 때마다 100 개 넘는 파일의 `ROOT = "/data/..."` 를 고치는 일을
없애려고 둔다. 기계별 값은 `paths.env` 한 곳이고, 레포 위치는 **이 파일 위치에서** 잡는다.

`expand()` 는 레지스트리(md)·config(yaml) 값 안의 `${VAR}` / `~` 를 푼다.
그래서 `configs/protocols/datasets.md` 가 `frames_root: ${BENCH_ROOT}/IntPhys1` 처럼
기계와 무관하게 쓰일 수 있다.

⚠️ import 만으로 os.environ 에 **기본값을 채운다** (이미 있는 값은 안 건드린다).
   자식 프로세스(DDP rank)도 그대로 물려받는다.
"""
from __future__ import annotations

import os
import pathlib
import re

_HERE = pathlib.Path(__file__).resolve().parent
# harness -> scripts -> z_research -> <repo>
ROOT = str(_HERE.parents[2])
ENV_FILE = str(_HERE / "paths.env")

# paths.env 한 줄: KEY=${KEY:-기본값}   또는   KEY=값
_LINE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*$")
_SELF_DEFAULT = re.compile(r"^\$\{([A-Z][A-Z0-9_]*):-(.*)\}$")


def _load_env_file(path: str, env: dict) -> None:
    """paths.env 를 읽어 env 에 **없는 키만** 채운다 (환경이 이긴다)."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.lstrip().startswith("#"):
                continue
            m = _LINE.match(ln)
            if not m:
                continue
            key, raw = m.group(1), m.group(2)
            sd = _SELF_DEFAULT.match(raw)
            if sd:
                if sd.group(1) != key:              # KEY=${OTHER:-...} 는 형식 위반
                    raise ValueError(f"{path}: {key} 줄의 self-default 가 {sd.group(1)} 이다")
                if key in env:
                    continue                        # 이미 있는 값이 이긴다
                raw = sd.group(2)
            elif key in env:
                continue
            env[key] = os.path.expandvars(raw.strip('"').strip("'"))


# VJEPA2_ROOT 는 파일 위치에서 (paths.env 에 없다)
os.environ.setdefault("VJEPA2_ROOT", ROOT)
_load_env_file(ENV_FILE, os.environ)
os.environ.setdefault("CKPT_ROOT", os.path.join(ROOT, "checkpoints"))
os.environ.setdefault("DATA_CSV", os.path.join(ROOT, "data_csv"))

VJEPA2_PY = os.environ["VJEPA2_PY"]
WORLD_ROOT = os.environ["WORLD_ROOT"]
BENCH_ROOT = os.environ["BENCH_ROOT"]
TRAIN_DATA_ROOT = os.environ["TRAIN_DATA_ROOT"]
WMA_CACHE_DIR = os.environ["WMA_CACHE_DIR"]
BENCH_CODE_ROOT = os.environ["BENCH_CODE_ROOT"]
CKPT_ROOT = os.environ["CKPT_ROOT"]
DATA_CSV = os.environ["DATA_CSV"]


def expand(v):
    """문자열 안의 `${VAR}` · `$VAR` · `~` 를 푼다. dict/list 는 재귀."""
    if isinstance(v, str):
        return os.path.expanduser(os.path.expandvars(v))
    if isinstance(v, dict):
        return {k: expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [expand(x) for x in v]
    return v


if __name__ == "__main__":
    for k in ("VJEPA2_ROOT", "VJEPA2_PY", "WORLD_ROOT", "BENCH_ROOT",
              "TRAIN_DATA_ROOT", "WMA_CACHE_DIR", "BENCH_CODE_ROOT", "CKPT_ROOT", "DATA_CSV",
              "WM_SLURM_PARTITION", "WM_SLURM_NODE"):
        print(f"{k:<18} {os.environ.get(k)}")
