#!/usr/bin/env python3
"""BlenderTrack 설정 사본 — yaml 의 out · data_root 를 이 머신 자리로 바꿔 <새 out>/<이름> 에 쓴다 (2026-11-02, Vast 이식).

설계 · 고르기 · 렌더 · 묶기 스크립트는 전부 --config 로 받은 파일의 out · data_root 만 본다. 그래서 드라이버가 처음에 이 사본을 만들고
그 뒤 모든 단계에 사본을 넘긴다 (파이썬 스크립트는 안 고친다). 바꾸는 규칙 (환경변수, env.sh 가 기본값을 준다):
  BT_OUT / BT_DATA_ROOT   주면 그 값으로 통째로
  BT_ROOT                 주면 값의 앞부분 BT_CLUSTER_ROOT (/data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track) 를 BT_ROOT 로
                          (비우면 yaml 그대로 — 클러스터에서는 원본과 바이트까지 같은 사본)
  BT_CFG_ONLY="a b"       그 분할만 남긴다 (스모크) · BT_CFG_SET="splits.s_speed.candidates=52 …" 점 경로로 덮는다 (값은 yaml 로 읽음)
out · data_root 두 줄만 바뀔 때는 줄 단위로 고쳐 주석을 그대로 둔다. ONLY · SET 이 있으면 yaml 을 다시 쓴다 (주석 빠짐, 맨 위에 출처 한 줄).

  $BT_PY localize_cfg.py configs/v5mc.yaml --write config_v5mc.yaml     → <새 out>/config_v5mc.yaml 경로를 찍는다
  $BT_PY localize_cfg.py configs/v4vc.yaml --write 'config_{name}.yaml'
  $BT_PY localize_cfg.py configs/v5mc.yaml --show                          → 바뀐 out · data_root 만 찍는다 (아무것도 안 씀)
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import yaml


def _remap(val, key, env):
    if val is None:
        return None
    full = env.get("BT_OUT" if key == "out" else "BT_DATA_ROOT")
    if full:
        return full
    root, croot = env.get("BT_ROOT", ""), env.get("BT_CLUSTER_ROOT", "/data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track")
    if not root:
        return val
    croot = croot.rstrip("/")
    if val == croot or val.startswith(croot + "/"):
        return root.rstrip("/") + val[len(croot):]
    sys.exit(f"!! {key}: {val} 가 {croot} 아래가 아니라 BT_ROOT 로 바꿀 수 없다 — BT_OUT / BT_DATA_ROOT 로 직접 준다")


def _set(cfg, path, raw):
    cur = cfg
    ks = path.split(".")
    for k in ks[:-1]:
        cur = cur[k]
    cur[ks[-1]] = yaml.safe_load(raw)


def localize(src, env):
    text = open(src).read()
    cfg = yaml.safe_load(text)
    new = {k: _remap(cfg.get(k), k, env) for k in ("out", "data_root")}
    only = (env.get("BT_CFG_ONLY") or "").replace(",", " ").split()
    sets = (env.get("BT_CFG_SET") or "").split()
    if not only and not sets:                                  # 줄 단위 (주석 유지)
        lines = text.splitlines(keepends=True)
        for k, v in new.items():
            if v is None or v == cfg.get(k):
                continue
            pat = re.compile(rf"^({k}:\s*)(\S+)(.*)$", re.S)
            hit = [i for i, ln in enumerate(lines) if pat.match(ln)]
            assert len(hit) == 1, f"{src}: 맨 앞 '{k}:' 줄이 {len(hit)} 개"
            i = hit[0]
            lines[i] = pat.sub(lambda m: m.group(1) + v + m.group(3), lines[i])
        out_text = "".join(lines)
    else:                                                      # yaml 다시 쓰기
        for k, v in new.items():
            if v is not None:
                cfg[k] = v
        if only:
            miss = [s for s in only if s not in cfg["splits"]]
            assert not miss, f"없는 분할 {miss}"
            cfg["splits"] = {s: cfg["splits"][s] for s in cfg["splits"] if s in only}
        for kv in sets:
            k, v = kv.split("=", 1)
            _set(cfg, k, v)
        out_text = (f"# localize_cfg.py 사본: {os.path.abspath(src)} (BT_CFG_ONLY={' '.join(only)!r} BT_CFG_SET={' '.join(sets)!r})\n"
                    + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    chk = yaml.safe_load(out_text)                             # 검산: 바꾼 것만 바뀌었나
    for k, v in new.items():
        assert chk.get(k) == v, (k, chk.get(k), v)
    return out_text, chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--write", help="<새 out>/<이 이름> 에 쓰고 그 경로를 찍는다 ({name} = 설정의 name)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    text, cfg = localize(a.src, os.environ)
    if a.show or not a.write:
        print(f"out       {cfg.get('out')}\ndata_root {cfg.get('data_root')}\nsplits    {' '.join(cfg['splits'])}")
        return
    dst = os.path.join(cfg["out"], a.write.format(name=cfg.get("name", "")))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.abspath(dst) != os.path.abspath(a.src):
        tmp = f"{dst}.{os.getpid()}.tmp"
        open(tmp, "w").write(text)
        os.replace(tmp, dst)
    print(dst)


if __name__ == "__main__":
    main()
