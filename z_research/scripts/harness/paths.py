#!/usr/bin/env python3
"""경로 정본(paths.env) 파이썬 입구.

  import sys; sys.path.insert(0, "<레포>/z_research/scripts/harness")
  from paths import ROOT, expand, BENCH_ROOT, DATA_CSV, TRAIN_DATA_ROOT, ...

  python z_research/scripts/harness/paths.py      # 표 + 실재 여부

규칙 (env.sh 가 bash 로 읽는 것과 **같은 뜻**이어야 한다)
  * VJEPA2_ROOT(=ROOT) 는 이 파일 위치에서 파생한다 (paths.env 에 없다)
  * paths.env 는 KEY=VALUE 한 줄에 하나. 값 안의 ${KEY} 는 위에서 정의된 키로 푼다.
    모르는 키를 참조하면 죽는다 (bash 는 조용히 빈 문자열로 만들어 경로가 틀어진다)
  * 환경변수로 덮지 않는다 — 파일이 이긴다. 경로를 바꾸려면 paths.env 를 고친다

expand(obj) — md/yaml 에서 읽은 값의 ${KEY} 를 푼다
  * dict(값만)·list·tuple 을 재귀로 돈다. 문자열이 아닌 값은 그대로
  * ${KEY} 꼴만 푼다 ($KEY 는 안 건드린다 — 포맷 문자열·셸 조각을 망가뜨리지 않게)
  * 찾는 순서: paths.env 의 키 → os.environ. 둘 다 없으면 **그대로 둔다**
    (뒤의 실물 검사가 "... 가 없다 -> ${FOO}/x" 로 원인을 그대로 보여 준다)
"""
from __future__ import annotations

import os
import pathlib
import re

HARNESS = pathlib.Path(__file__).resolve().parent
ENV_FILE = HARNESS / "paths.env"
ROOT = str(HARNESS.parents[2])            # z_research/scripts/harness -> 레포
VJEPA2_ROOT = ROOT

_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_LINE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _load(path: pathlib.Path) -> dict:
    vals = {"VJEPA2_ROOT": ROOT}
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        m = _LINE.match(ln)
        if not m:
            raise ValueError(f"{path}:{n}: KEY=VALUE 가 아니다 -> {raw!r}")
        key, val = m.group(1), m.group(2).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if "$(" in val or "`" in val:
            raise ValueError(f"{path}:{n}: 명령 치환은 쓸 수 없다 -> {raw!r}")

        def sub(mm, n=n):
            k = mm.group(1)
            if k not in vals:
                raise KeyError(f"{path}:{n}: ${{{k}}} 가 위에서 정의되지 않았다")
            return vals[k]
        vals[key] = _REF.sub(sub, val)
    return vals


PATHS: dict = _load(ENV_FILE)
globals().update(PATHS)                     # BENCH_ROOT, DATA_CSV, TRAIN_DATA_ROOT, ... 를 모듈 상수로


def expand(obj):
    """문자열 안의 ${KEY} 를 paths.env → os.environ 순으로 푼다 (dict/list/tuple 재귀)."""
    if isinstance(obj, str):
        return _REF.sub(lambda m: PATHS.get(m.group(1), os.environ.get(m.group(1), m.group(0))), obj)
    if isinstance(obj, dict):
        return {k: expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(expand(v) for v in obj)
    return obj


if __name__ == "__main__":
    print(f"paths.env : {ENV_FILE}")
    w = max(len(k) for k in PATHS)
    for k, v in PATHS.items():
        tag = ""
        if v.startswith("/"):
            tag = "  ok" if os.path.exists(v) else "  (없음)"
        print(f"  {k:<{w}}  {v or '(비어 있음)'}{tag}")
