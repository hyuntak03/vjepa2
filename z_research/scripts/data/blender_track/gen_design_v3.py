#!/usr/bin/env python3
"""BlenderTrack v3 설계 — 물체 영속 학습셋, 물리적으로 자연스러운 장면만 (2026-11-01, 사용자 설계).

원칙 (사용자)
  · 가르치려는 것 = 물체 영속. 가림 타이밍은 전부 late — 문맥 끝 (샘플 15) 에서 target 이 완전히 가려지고 미래에서 다시 보인다.
  · 물리적으로 자연스럽게: 실제 축척 (중력 9.81 · 물체 15–40 cm · 카메라 1–4 m) · 그림자 · 원인 없는 출발 · 공중 정지 · 점프 없음.
    움직이는 물체는 굴러 (공) 또는 미끄러져 (상자 · 통) 마찰로 천천히 느려진다. 움직이는 판은 바닥 레일 위를 일정 속도로.
    빗면: 내려감 = 꼭대기 근처에서 놓아 구름 (가속) · 올라감 = 아래에서 굴려 올려 느려지다 되돌아옴.
  · 빈 장면: target 이 처음부터 없으면 끝까지 없다 (같은 장면에서 target 만 뺀다).
  · 다른 물체가 가리면 가려지는 물체는 너무 작지 않게 (화면 지름 ≥ 22 px), 가리는 물체는 너무 크지 않게 (화면 크기 1.1–1.8 배).
  · 그 밖 물체는 잘 보이게 (≥ 28 px · 창의 70 % 이상 보임), target 과 가리는 물체 · 그 밖 물체는 생김새가 확실히 다르게.
축 (configs/v3.yaml cells)
  objset  primitive (단색 기본 도형) · realistic (무늬 있는 실제 물건: 농구공 · 축구공 · 비치볼 · 종이 상자 · 나무 상자 · 페인트 통 · 쓰레기통 · 교통 고깔)
  place   flat · ramp (빗면은 target 이 움직이는 상황만)
  가림 종류 barrier (판 · 낮은 벽 · 탁자 / 레일 위 판) · object (상자 · 통 · 큰 공 · 키 큰 물체 / 굴러가는 공)
  가림 움직임 static · moving   ·   target moving · static · none (빈 장면)   ·   camera front (눈높이) · side (눈높이, 30–60° 비껴) · bird (위)
  제외: 정지 가림 × 정지 target (가림이 안 생김) · bird × object (내려다보면 1.8 배 안에서 덮을 수 없음, 4 배로도 불가 — 11-01 실측)
  + 물체 수 = 장면의 움직일 수 있는 물체 총수 (target · 가리는 물체 · 그 밖 물체) 0–3, 상황마다 가능한 값만.
카메라는 장면 가운데 (target 이 가려지는 자리) 를 본다 — 위치 (x, y, h) · pitch · yaw (design.Cam 이 yaw 를 받는다).
운동 · 가림 판정 · 물리 검사는 gen_design_v2 를 그대로 쓴다 (flat_static · flat_decel · ramp_accel · ramp_reverse · 상자 가림 · 쐐기 · physics_check).
  python gen_design_v3.py --config configs/v3.yaml --preview      # 상황마다 하나 → <out>/design/preview90_*.json + preview_index.json
"""
from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D0  # noqa: E402
import gen_design_v2 as G2  # noqa: E402

DIMS = D0.DIMS
STRIDE = 2
load_cfg = G2.load_cfg
track = G2.track
center_path = G2.center_path
tracks = G2.tracks
clip_id = G2.clip_id
_u = G2._u


# ----------------------------------------------------------------------------- 상황 · 물체 수
def situations(cfg):
    C = cfg["cells"]
    out = []
    for objset in C["objset"]:
        for place in C["place"]:
            for cam in C["camera"]:
                for ot in C["occ_type"]:
                    for om in C["occ_motion"]:
                        for tg in C["target"]:
                            if om == "static" and tg == "static":
                                continue
                            if cam == "bird" and (ot == "object" or om == "trapdoor"):   # 11-01 v3t: 위에서는 솟는 판이 못 가린다
                                continue
                            if place == "ramp" and tg != "moving":
                                continue
                            out.append(dict(objset=objset, place=place, camera=cam, occ_type=ot, occ_motion=om, target=tg))
    return out


def count_levels(sit):
    base = int(sit["target"] != "none") + int(sit["occ_type"] == "object")
    return [c for c in (0, 1, 2, 3) if c >= base]


def cell_of(cfg, idx):
    S = situations(cfg)
    s = S[idx % len(S)]
    lv = count_levels(s)
    return s, lv[(idx // len(S)) % len(lv)]


def cells(cfg):
    return situations(cfg)


# ----------------------------------------------------------------------------- 겉모습 (primitive · realistic) · 환경
WOOD = [(0.42, 0.26, 0.13), (0.55, 0.36, 0.2), (0.3, 0.18, 0.09), (0.62, 0.47, 0.3)]
# realistic 물건: 모양 → [(이름, 크기 범위 m, 무늬, 색, 색 2, 거칠기, 금속, 크기 자유)]. 가리는 물체는 크기가 정해지므로 크기 자유인 물건만
REAL = {
    "sphere": [("basketball", (0.23, 0.25), "lines", (0.80, 0.35, 0.10), (0.06, 0.04, 0.03), 0.75, 0.0, False),
               ("soccer_ball", (0.21, 0.23), "cells", (0.95, 0.95, 0.95), (0.04, 0.04, 0.04), 0.45, 0.0, False),
               ("beach_ball", (0.25, 0.60), "segments", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.3, 0.0, True)],
    "cube": [("cardboard_box", (0.18, 0.60), "tape", (0.62, 0.46, 0.28), (0.80, 0.74, 0.62), 0.9, 0.0, True),
             ("wooden_crate", (0.22, 0.60), "grain", None, None, 0.7, 0.0, True)],
    "cylinder": [("paint_bucket", (0.18, 0.34), "band", (0.75, 0.75, 0.78), None, 0.35, 0.7, False),
                 ("plastic_bin", (0.25, 0.60), "band", None, None, 0.5, 0.0, True)],
    "cone": [("traffic_cone", (0.28, 0.45), "band", (0.95, 0.38, 0.05), (0.95, 0.95, 0.95), 0.5, 0.0, False)],
}
PRIM_SHAPES = ["sphere", "cube", "cylinder", "cone", "torus", "capsule", "pyramid"]


def _jit(rng, c, a=0.08):
    return tuple(float(np.clip(x + rng.uniform(-a, a), 0.02, 0.98)) for x in c)


def _bright(rng):
    return tuple(float(x) for x in colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.5, 0.9), rng.uniform(0.5, 0.9)))


def look(rng, objset, shape, size=None, flex_only=False):
    """겉모습 dict (color · rough · metal · pattern · kind · [color2 · hz · band]) 와 크기. size 가 주어지면 그 크기에 맞는 물건만 (가리는 물체)."""
    if objset == "primitive" or shape not in REAL:
        return {"color": _bright(rng), "rough": float(rng.uniform(0.25, 0.7)), "metal": 0.0, "pattern": "solid", "kind": f"{shape}"}, size
    cands = [k for k in REAL[shape] if (not flex_only or k[7]) and (size is None or k[1][0] * 0.95 <= size <= k[1][1] * 1.05)]
    if not cands:
        return None, None
    name, rng_s, pat, c1, c2, rough, metal, _ = cands[int(rng.integers(len(cands)))]
    if c1 is None:
        c1 = _jit(rng, WOOD[rng.integers(len(WOOD))]) if name == "wooden_crate" else _bright(rng)
    if c2 is None:
        c2 = tuple(0.55 * x for x in c1) if name == "wooden_crate" else (_bright(rng) if name == "paint_bucket" else tuple(0.7 * x for x in c1))
    lk = {"color": tuple(float(x) for x in c1), "color2": tuple(float(x) for x in c2), "rough": rough, "metal": metal, "pattern": pat, "kind": name,
          "hz": float(DIMS[shape][2])}
    if name == "paint_bucket":
        lk["band"] = (0.22, 0.8)
    if name == "plastic_bin":
        lk["band"] = (0.86, 0.96)
    if name == "traffic_cone":
        lk["band"] = (0.42, 0.6)
    s = size if size is not None else float(rng.uniform(*rng_s))
    return lk, s


# ----------------------------------------------------------------------------- 요인 균형 배정 (11-01 사용자: probing 용 testbed 로도 쓰니 bias 없이)
KIND_SHAPE = {k[0]: sh for sh, L in REAL.items() for k in L}
REAL_BY = {k[0]: k for L in REAL.values() for k in L}
HUE_FREE = {"paint_bucket", "plastic_bin"}                              # realistic 중 색이 자유로운 물건 (색 칸을 따른다)


def target_kinds(objset, place, tmode):
    if place == "ramp":
        return ["sphere"] if objset == "primitive" else ["basketball", "soccer_ball", "beach_ball"]
    if tmode == "moving":
        return ["sphere", "cube", "cylinder"] if objset == "primitive" else ["basketball", "soccer_ball", "beach_ball", "cardboard_box", "wooden_crate",
                                                                            "paint_bucket", "plastic_bin"]
    return list(PRIM_SHAPES) if objset == "primitive" else list(REAL_BY)


def look_fixed(rng, cfg, objset, F):
    """요인 F 의 kind · size (3 칸) · hue (8 칸) 로 겉모습과 크기."""
    kind = F["kind"]
    if objset == "primitive":
        shape = kind
        hsv = ((F["hue"] + rng.uniform()) / 8.0, rng.uniform(0.5, 0.9), rng.uniform(0.5, 0.9))
        lk = {"color": tuple(float(x) for x in colorsys.hsv_to_rgb(*hsv)), "rough": float(rng.uniform(0.25, 0.7)), "metal": 0.0, "pattern": "solid", "kind": shape}
        return lk, _bin(rng, cfg["objects"]["size_m"], F["size"], 3), shape
    shape = KIND_SHAPE[kind]
    name, rng_s, pat, c1, c2, rough, metal, _ = REAL_BY[kind]
    s = _bin(rng, (rng_s[0], min(rng_s[1], 0.4)), F["size"], 3)          # target 은 실제 물건 크기 (최대 40 cm)
    if kind in HUE_FREE:
        c1 = tuple(float(x) for x in colorsys.hsv_to_rgb((F["hue"] + rng.uniform()) / 8.0, rng.uniform(0.5, 0.9), rng.uniform(0.5, 0.9)))
    if c1 is None:
        c1 = _jit(rng, WOOD[rng.integers(len(WOOD))])
    if c2 is None:
        c2 = tuple(0.55 * x for x in c1) if name == "wooden_crate" else (_bright(rng) if name == "paint_bucket" else tuple(0.7 * x for x in c1))
    lk = {"color": tuple(float(x) for x in c1), "color2": tuple(float(x) for x in c2), "rough": rough, "metal": metal, "pattern": pat, "kind": name,
          "hz": float(DIMS[shape][2])}
    lk.update({"paint_bucket": {"band": (0.22, 0.8)}, "plastic_bin": {"band": (0.86, 0.96)}, "traffic_cone": {"band": (0.42, 0.6)}}.get(name, {}))
    return lk, s, shape


def factor_levels(sit, cfg):
    """장면 요인과 수준 (상황마다). 각 요인은 상황 안에서 수준별로 같은 수, 요인끼리는 따로 섞는다."""
    tmode = sit["target"]
    F = {"kind": target_kinds(sit["objset"], sit["place"], tmode), "hue": list(range(8)), "size": [0, 1, 2], "dir": [-1, 1],
         "hide": ["short", "long"] if sit["occ_type"] == "barrier" else ["short"],   # 11-01 실측: 다른 물체 (≤ 1.8 배) 는 긴 가림 0 / 77 — 짧은 가림만 (VOBench late 길이)
         "count": count_levels(sit), "indoor": [True, False], "floor": [0, 1, 2], "side_sign": [-1, 1]}
    if sit["place"] == "ramp":
        F["ramp_mode"] = ["up", "down"]
    if tmode == "moving" and sit["place"] == "flat":
        F["speed"] = [0, 1, 2]
    if sit["occ_type"] == "barrier" and sit["occ_motion"] == "static":
        F["occ_sub"] = ["board", "wall"] + (["table"] if sit["camera"] != "front" else [])
    if sit["occ_type"] == "object" and sit["occ_motion"] == "static":
        F["occ_sub"] = ["cube", "cylinder", "sphere"] + (["capsule"] if sit["objset"] == "primitive" else [])
    return F


_SCHED = {}


def schedule(cfg, split, si, sit, N):
    """상황 si 의 장면 N 개 요인표 (층화 뒤섞기: 각 요인 수준 개수가 ±1 안에서 같다)."""
    key = (split, si, N)
    if key not in _SCHED:
        rng = np.random.default_rng([int(cfg["splits"][split]["seed"]), 7777, si])
        rows = [dict() for _ in range(N)]
        for fname, levels in factor_levels(sit, cfg).items():
            reps = (N + len(levels) - 1) // len(levels)
            order = rng.permutation(reps * len(levels))[:N] % len(levels)
            for k in range(N):
                v = levels[int(order[k])]
                rows[k][fname] = v.item() if hasattr(v, "item") else v
        _SCHED[key] = rows
    return _SCHED[key]


def block_situations(cfg):
    """본 데이터 상황: target 있음 (빈 장면은 팔 B) · 빗면 × 다른 물체 가림 제외 (11-01 실측 성공 0–12 %: 빗면 위 공을 바닥 물체 ≤ 1.8 배로 못 덮음)."""
    return [s for s in situations(cfg) if s["target"] != "none" and not (s["place"] == "ramp" and s["occ_type"] == "object")
            and situation_key(s) not in set(cfg.get("infeasible", []))]


def situation_key(s):
    return "|".join([s["objset"], s["place"], s["camera"], s["occ_type"], s["occ_motion"], s["target"]])


def block_of(cfg, split, i):
    """렌더 번호 i → (장면 b, 팔 A/B, 상황, 상황 번호, 장면 안 번호 k, 요인 F). 짝수 = A (target 있음), 홀수 = B (target 없음)."""
    S = block_situations(cfg)
    b, arm = i // 2, "AB"[i % 2]
    si = b % len(S)
    k = b // len(S)
    N = int(cfg["splits"][split].get("candidates", cfg["splits"][split]["blocks_per_situation"]))
    return b, arm, S[si], si, k, schedule(cfg, split, si, S[si], N)[k]


def _hue(c):
    return colorsys.rgb_to_hsv(*c[:3])


def distinct(a, b, min_dh):
    """두 물체 생김새가 확실히 다른가: 모양이 다르면 됨. 같은 모양이면 realistic = 다른 물건 (또는 색상 차), primitive = 색상 차이 ≥ min_dh 또는 밝기 차 > 0.35."""
    if a["shape"] != b["shape"]:
        return True
    ha, hb = _hue(a["color"]), _hue(b["color"])
    dh = abs(ha[0] - hb[0]); dh = min(dh, 1 - dh)
    if a.get("pattern", "solid") != "solid" or b.get("pattern", "solid") != "solid":
        return a.get("kind") != b.get("kind") or dh >= min_dh
    return dh >= min_dh or abs(ha[2] - hb[2]) > 0.35


def panel_look(rng, kind):
    if kind == "wall":                                                  # 낮은 벽: 칠한 회벽 · 콘크리트
        c = _jit(rng, [(0.8, 0.78, 0.72), (0.6, 0.6, 0.6), (0.72, 0.62, 0.5)][rng.integers(3)], 0.06)
        return {"color": c, "tex": "noise", "tex_scale": float(rng.uniform(3, 8))}
    if rng.uniform() < 0.6:                                             # 나무 판 (결 = wave)
        return {"color": _jit(rng, WOOD[rng.integers(len(WOOD))]), "tex": "wave", "tex_scale": float(rng.uniform(4, 10))}
    c = _jit(rng, [(0.85, 0.85, 0.82), (0.25, 0.35, 0.55), (0.55, 0.15, 0.12), (0.2, 0.4, 0.25)][rng.integers(4)], 0.05)
    return {"color": c, "tex": "plain", "tex_scale": 1.0}               # 칠한 판


def sample_env(rng, cfg, F=None):
    E = cfg["env"]
    indoor = bool(F["indoor"]) if F is not None else bool(rng.uniform() < E["indoor_p"])
    fl = E["indoor_floor"] if indoor else E["outdoor_floor"]
    floor = str(fl[F["floor"] % len(fl)]) if F is not None else str(rng.choice(fl))
    if floor == "wood":
        c1 = _jit(rng, WOOD[rng.integers(len(WOOD))]); c2 = _jit(rng, c1, 0.06); sc = rng.uniform(1.6, 2.6)
    elif floor == "tile":
        base = [(0.8, 0.78, 0.74), (0.6, 0.6, 0.62), (0.55, 0.45, 0.38), (0.3, 0.32, 0.35)][rng.integers(4)]
        c1 = _jit(rng, base, 0.05); c2 = _jit(rng, base, 0.05); sc = rng.uniform(0.7, 1.6)
    elif floor == "carpet":
        base = [(0.45, 0.4, 0.35), (0.3, 0.35, 0.45), (0.5, 0.3, 0.3), (0.35, 0.45, 0.3)][rng.integers(4)]
        c1 = _jit(rng, base, 0.04); c2 = tuple(0.8 * x for x in c1); sc = rng.uniform(0.8, 1.5)
    else:                                                               # concrete
        g = rng.uniform(0.35, 0.65); c1 = (g, g, g * 0.98); c2 = tuple(0.8 * x for x in c1); sc = rng.uniform(0.5, 2.0)
    wall = indoor or bool(rng.uniform() < 0.35)
    wc = _jit(rng, [(0.88, 0.86, 0.8), (0.75, 0.8, 0.85), (0.85, 0.78, 0.68), (0.6, 0.35, 0.28)][rng.integers(4)], 0.05)
    return {"floor": floor, "floor_c1": tuple(float(x) for x in c1), "floor_c2": tuple(float(x) for x in c2), "floor_scale": float(sc),
            "hdri": str(rng.choice(E["indoor_hdri"] if indoor else E["outdoor_hdri"])), "hdri_rot": float(rng.uniform(0, 360)),
            "hdri_strength": float(rng.uniform(0.3, 0.6) if indoor else rng.uniform(0.4, 0.8)), "indoor": indoor,    # 11-01: 1 차 확인에서 바닥이 하얗게 날아감 → 낮춤
            "sky_hor": (0.8, 0.8, 0.8), "sky_top": (0.3, 0.45, 0.7), "sky_strength": 1.0,
            "sun_energy": float(rng.uniform(0.4, 1.2) if indoor else rng.uniform(1.2, 2.8)),
            "sun_elev": float(rng.uniform(30, 70)), "sun_azim": float(rng.uniform(0, 360)), "exposure": float(rng.uniform(-0.6, -0.1)),
            "wall": wall, "wall_y": float(rng.uniform(1.2, 3.0)), "wall_h": float(rng.uniform(2.4, 3.2) if indoor else rng.uniform(2.5, 6.0)),
            "wall_tex": str(rng.choice(["plain", "noise"] if indoor else ["brick", "plain"])), "wall_c1": wc, "wall_c2": tuple(0.85 * x for x in wc)}


# ----------------------------------------------------------------------------- 물체 · 카메라
def _wchoice(rng, w):
    ks = list(w); p = np.array([w[k] for k in ks], float)
    return str(rng.choice(ks, p=p / p.sum()))


def s_eff(o):
    h = G2.obj_half(o)
    return 2.0 * math.sqrt(max(h[0], h[1]) * h[2])


def _bin(rng, lohi, k, n):
    """범위 lohi 를 n 칸으로 나눈 k 번째 칸 안에서 고른다 (요인 균형 배정)."""
    lo, hi = lohi
    return float(lo + (hi - lo) * (k + rng.uniform()) / n)


def new_object(rng, cfg, objset, shapes, role, moving, F=None):
    """물체 하나 (겉모습 · 크기 · 운동). shapes: 무게 dict 또는 목록. 움직임 = 굴러가는 공 · 미끄러지는 상자 · 통 (flat_decel).
    F (장면 요인, target 만): kind · size (3 칸) · hue (8 칸) · speed (3 칸) · dir (±1) 를 정해 둔 값으로."""
    O = cfg["objects"]
    if F is not None:
        lk, s, shape = look_fixed(rng, cfg, objset, F)
    else:
        shape = _wchoice(rng, shapes) if isinstance(shapes, dict) else str(rng.choice(shapes))
        lk, s = look(rng, objset, shape)
        if s is None:
            s = _u(rng, O["size_m"])
    o = {"shape": shape, "s": float(s), "role": role, **lk}
    if moving:
        roll = shape == "sphere"
        rr = O["roll_speed"] if roll else O["slide_speed"]
        sp = _bin(rng, rr, F["speed"], 3) if F is not None else _u(rng, rr)
        mu = _u(rng, O["roll_mu"] if roll else O["slide_mu"])
        hd = math.radians(_u(rng, O["heading_deg"]))
        sg = float(F["dir"]) if F is not None else float(rng.choice([-1.0, 1.0]))
        o["motion"] = {"kind": "flat_decel", "p0": [0.0, 0.0], "v": [sg * sp * math.cos(hd), sp * math.sin(hd)], "mu": float(mu),
                       "yaw0": float(math.atan2(math.sin(hd), sg * math.cos(hd))) if not roll else float(rng.uniform(0, 2 * np.pi)), "omega": 0.0}
    else:
        o["motion"] = {"kind": "flat_static", "p0": [0.0, 0.0], "yaw0": float(rng.uniform(0, 2 * np.pi))}
    return o


def new_ramp_target(rng, cfg, objset, F=None):
    """빗면 위 공: 내려감 (ramp_accel, 꼭대기 근처에서 놓음) · 올라감 (ramp_reverse, 아래에서 굴려 올려 되돌아옴). 쐐기 소품 포함 (o['props'])."""
    R, g = cfg["ramp"], float(cfg["g"])
    if F is not None:
        lk, s, _ = look_fixed(rng, cfg, objset, F)
    else:
        lk, s = look(rng, objset, "sphere")
        if s is None:
            s = _u(rng, cfg["objects"]["size_m"])
    th = _u(rng, R["deg"]); thr = math.radians(th)
    L = float(min(_u(rng, R["len"]), R["max_h"] / math.sin(thr)))
    sx = float(F["dir"]) if F is not None else float(rng.choice([-1.0, 1.0]))
    a = 5.0 / 7.0 * g * math.sin(thr)
    r = 0.5 * s
    s_c = r * math.tan(thr / 2)
    up = (F["ramp_mode"] == "up") if F is not None else bool(rng.uniform() < R["up_p"])
    if up:
        t_ap = _u(rng, R["apex_s"])
        v0 = a * t_ap
        rise = 0.5 * a * t_ap * t_ap
        s0 = max(s_c + 0.05, _u(rng, (0.03, 0.2)) * L)
        if s0 + rise > 0.92 * L:
            s0 = 0.92 * L - rise
        if s0 < s_c + 0.03:
            return None
        kind = "ramp_reverse"
    else:
        s0, v0, kind = _u(rng, (0.75, 0.95)) * L, -_u(rng, (0.0, 0.25)), "ramp_accel"
    m = {"kind": kind, "p0": [0.0, 0.0], "theta_deg": float(th), "sx": sx, "L": L, "x_low": 0.0, "y": 0.0, "s0": float(s0), "v0": float(v0),
         "mu_g": float(_u(rng, (0.01, 0.03))), "yaw0": 0.0, "omega": 0.0}
    Wd, thick = max(0.35, 3.0 * r), 0.12
    phi = -sx * thr
    top_mid = np.array([sx * (L / 2) * math.cos(thr), 0.0, (L / 2) * math.sin(thr)])
    loc = top_mid - np.array([thick * math.sin(phi), 0.0, thick * math.cos(phi)])
    col = _jit(rng, WOOD[rng.integers(len(WOOD))]) if rng.uniform() < 0.6 else _jit(rng, [(0.85, 0.85, 0.82), (0.35, 0.45, 0.6)][rng.integers(2)], 0.05)
    return {"shape": "sphere", "s": float(s), "role": "target", **lk, "motion": m,
            "props": [{"role": "ramp", "solid": "wedge", "loc": loc.tolist(), "rot": [0.0, phi, 0.0], "scale": [L, Wd, thick], "color": col}],
            "ramp_mode": "up" if up else "down"}


def shift(o, dx, dy):
    m = o["motion"]
    if m["kind"].startswith("ramp_"):
        m["x_low"] += dx; m["y"] += dy
        for p in o.get("props", []):
            p["loc"][0] += dx; p["loc"][1] += dy
    else:
        m["p0"] = [m["p0"][0] + dx, m["p0"][1] + dy]


def sample_camera(rng, cfg, view, tgt, z_c, F=None):
    """장면 가운데 (원점, 높이 z_c) 를 보는 카메라. 거리는 target 화면 지름 (target_px) 에서 정한다."""
    C = cfg["camera"][view]
    lens = _u(rng, C["lens"])
    f = 256.0 * lens / D0.SENSOR
    D3 = f * s_eff(tgt) / _u(rng, cfg["objects"]["target_px"])
    az = _u(rng, C["azim"]) * ((float(F["side_sign"]) if F is not None else float(rng.choice([-1.0, 1.0]))) if view == "side" else 1.0)
    if view == "bird":
        p = abs(_u(rng, C["pitch"]))
        dz, d = D3 * math.sin(math.radians(p)), D3 * math.cos(math.radians(p))
        h = z_c + dz
    else:
        h = _u(rng, C["h"])
        dz = h - z_c
        if D3 <= abs(dz) + 0.25:
            return None
        d = math.sqrt(D3 * D3 - dz * dz)
    pitch = -math.degrees(math.atan2(dz, d))
    j = cfg["camera"]["jitter_deg"]
    a = math.radians(az)
    return {"h": float(h), "pitch": float(pitch + rng.uniform(-j, j)), "lens": float(lens), "yaw": float(-az + rng.uniform(-j, j)),
            "x": float(-d * math.sin(a)), "y": float(-d * math.cos(a)), "level": view, "dist": float(D3)}


def cam_obj(c):
    cam = D0.Cam(c["h"], c["pitch"], c["lens"], yaw_deg=c["yaw"], x=c["x"], y=c["y"])
    cam.level = c["level"]
    return cam


def cam_yaw(cam):
    return math.degrees(math.atan2(cam.R[1, 0], cam.R[0, 0]))


def want_interval(rng, cfg, F=None):
    L = cfg["timing"]["late"]
    if (F["hide"] == "short") if F is not None else (rng.uniform() < L["short_p"]):                                    # 짧은 가림 (VOBench late 꼴): 문맥 끝 앞뒤로 0–2 샘플씩 더
        a, b = int(rng.integers(L["short_side"][0], L["short_side"][1] + 1)), int(rng.integers(L["short_side"][0], L["short_side"][1] + 1))
    else:
        a, b = int(rng.integers(L["long_before"][0], L["long_before"][1] + 1)), int(rng.integers(L["long_after"][0], L["long_after"][1] + 1))
    return 15 - a, 16 + b


# ----------------------------------------------------------------------------- 가리는 것
def _front_y(ctr, tgt, props):
    """target 경로 (와 빗면 쐐기) 의 카메라 쪽 끝 y."""
    y = float(ctr[:, 1].min()) - 0.5 * tgt["s"] * max(DIMS[tgt["shape"]][0], DIMS[tgt["shape"]][1])
    for w in G2._wedges(props):
        y = min(y, w["y0"])
    return y


def make_board(rng, cfg, cam, tgt, ctr, E, fs, fe, kind, props):
    P = cfg["occluders"]
    th = _u(rng, P["board_th"] if kind == "board" else P["wall_th"])
    y_b = _front_y(ctr, tgt, props) - _u(rng, P["gap"]) - th / 2
    if y_b - cam.C[1] < 0.3:
        return None
    X, Z = G2._plane_proj(cam.C, E[fs:fe + 1], y_b - th / 2)
    mx = 0.03 * tgt["s"]
    x0, x1 = float(X.min()) - mx, float(X.max()) + mx
    w, h = x1 - x0, float(Z.max()) + mx
    if w < 0.15 or h > 2.0 * w or h > 1.2:                              # 11-01: 좁고 긴 기둥 꼴 금지 (높이 ≤ 폭 × 2). 최소 폭 30 → 15 cm (짧은 가림 × 작은 물체가 막혔다) (위 카메라에서 높이만 커지던 것) — 판 · 벽다운 비율만
        return None
    pan = {"role": "event", "x": 0.5 * (x0 + x1), "y": float(y_b), "w": w, "h": h, "th": float(th),
           **panel_look(rng, kind), "mode": "static", "kind": kind}
    return [pan], [], kind


def make_trapdoor(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, props):
    """11-01 v3t (VOBench late 가림막과 같은 꼴): 제자리 판이 바닥에 카메라 쪽으로 누워 있다가 가림 구간 직전 rise 초 동안 일어서고,
    가림 구간 [fs, fe] 동안 서 있고, 끝나면 다시 눕는다 (각도 = design.panel_angle). 크기 · 자리는 고정 판 (make_board) 과 같은 덮기 규칙."""
    r = make_board(rng, cfg, cam, tgt, ctr, E, fs, fe, "board", props)
    if r is None:
        return None
    pan = r[0][0]
    pan.update(mode="trapdoor", rise=float(_u(rng, cfg["occluders"].get("trapdoor_rise", (0.08, 0.2)))),
               t_up=float(fs / fps), t_down=float(min((T - 1) / fps, fe / fps)), kind="trapdoor")
    return [pan], [], "trapdoor"


def make_table(rng, cfg, cam, tgt, ctr, E, fs, fe):
    top = float(E[:, :, 2].max())
    z_r = top + _u(rng, (0.1, 0.4))
    X, Y = G2._hplane_proj(cam.C, E[fs:fe + 1].reshape(-1, 3), z_r)
    mx = 0.04 * tgt["s"]
    x0, x1, y0, y1 = X.min() - mx, X.max() + mx, Y.min() - mx, Y.max() + mx
    if x1 - x0 > 2.2 or y1 - y0 > 1.6:
        return None
    col = _jit(rng, WOOD[rng.integers(len(WOOD))]) if rng.uniform() < 0.7 else _jit(rng, (0.85, 0.85, 0.82), 0.05)
    th, lg = 0.035, 0.04
    roof = {"role": "occ_roof", "loc": [float((x0 + x1) / 2), float((y0 + y1) / 2), float(z_r)], "rot": [0, 0, 0],
            "scale": [float(x1 - x0), float(y1 - y0), th], "color": col}
    legs = [{"role": "occ_post", "loc": [float(x), float(y), 0.0], "rot": [0, 0, 0], "scale": [lg, lg, float(z_r)], "color": col}
            for x in (x0 + lg, x1 - lg) for y in (y0 + lg, y1 - lg)]
    return [], [roof] + legs, "table"


def make_slider(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, props):
    """바닥 레일 위를 일정 속도로 미끄러지는 판 (가림막 움직임). gen_design_v2.make_moving 과 같은 덮기 규칙, 속도는 m/s."""
    P = cfg["occluders"]
    th = _u(rng, P["board_th"])
    y_b = _front_y(ctr, tgt, props) - _u(rng, P["gap"]) - th / 2
    if y_b - cam.C[1] < 0.3:
        return None
    X, Z = G2._plane_proj(cam.C, E, y_b - th / 2)
    xmin, xmax = X.min(1), X.max(1)
    t = np.arange(T) / fps
    xc = 0.5 * (xmin + xmax)
    vt = float(np.gradient(xc)[(fs + fe) // 2] * fps)
    sp = _u(rng, P["slide_speed"])
    if abs(vt) < 0.05:
        vx, rel = float(rng.choice([-1.0, 1.0]) * sp), "pass"
    elif rng.uniform() < 0.5:
        vx, rel = float(np.sign(vt) * max(sp, 1.6 * abs(vt)) if rng.uniform() < 0.5 else np.sign(vt) * min(sp, 0.5 * abs(vt))), "same"
    else:
        vx, rel = float(-np.sign(vt) * sp), "opposite"
    rel_disp = xc[fs:fe + 1] - vx * t[fs:fe + 1]
    ext = float((xmax - xmin)[fs:fe + 1].max())
    w = ext + float(np.ptp(rel_disp)) + 0.06 * tgt["s"]
    hh = float(Z[fs:fe + 1].max()) + 0.04 * tgt["s"]
    if w > 2.0 or w < 0.15 or hh > 2.0 * w or hh > 1.2:                 # 판다운 비율만 (좁고 긴 기둥 금지)
        return None
    lo = (xmax - vx * t - w / 2)[fs:fe + 1].max()
    hi = (xmin - vx * t + w / 2)[fs:fe + 1].min()
    if lo > hi:
        return None
    x0 = float(0.5 * (lo + hi))
    pan = {"role": "event", "x": float(x0 + vx * t[(fs + fe) // 2]), "y": float(y_b), "w": float(w), "h": hh,
           "th": float(th), **panel_look(rng, "board"), "mode": "slide", "vx": vx, "x0": x0, "rel": rel, "kind": "rail_board"}
    xa, xb = min(x0, x0 + vx * t[-1]) - w / 2 - 0.1, max(x0, x0 + vx * t[-1]) + w / 2 + 0.1
    rail = {"role": "rail", "loc": [float((xa + xb) / 2), float(y_b), 0.0], "rot": [0, 0, 0], "scale": [float(xb - xa), float(th + 0.05), 0.012],
            "color": (0.18, 0.18, 0.2)}
    return [pan], [rail], f"rail_{rel}"


def _ratio_ok(cfg, cam, tgt, ctr, oc, fs, fe, T, fps, g):
    pc, _, Rc = track(oc, T, fps, g)
    r = G2.occluder_screen_ratio(cam, tgt, ctr, oc, center_path(oc, pc, Rc), fs, fe)
    lo, hi = cfg["occluders"]["object_screen_ratio"]
    return (0.9 * lo <= r <= 1.08 * hi), r


def make_static_object(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, g, objset, props, shape=None):
    """서 있는 다른 물체 (상자 · 통 · 큰 공 · 키 큰 물체) 가 앞에 있고 target 이 그 뒤로 지나간다. 크기 = 가림 구간의 target 투영을 덮는 만큼."""
    P = cfg["occluders"]
    w = dict(P["object_static_shapes"])
    if objset == "realistic":
        w.pop("capsule", None)
    shape = shape or _wchoice(rng, w)
    gap = _u(rng, P["object_gap"])
    dx, dz = min(DIMS[shape][0], DIMS[shape][1]), DIMS[shape][2]
    s_o = tgt["s"]
    for _ in range(4):
        r_o = 0.5 * s_o * max(DIMS[shape][0], DIMS[shape][1])
        y_o = _front_y(ctr, tgt, props) - gap - r_o
        X, Z = G2._plane_proj(cam.C, E[fs:fe + 1], y_o)
        W, H = float(np.ptp(X)) + 0.04 * tgt["s"], float(Z.max()) + 0.03 * tgt["s"]
        s_o = max(W / (dx * (0.92 if shape == "sphere" else 1.0)), H / dz) * 1.03
    if y_o - cam.C[1] < 0.3:
        return None
    lk, _ = look(rng, objset, shape, size=s_o, flex_only=True)
    if lk is None:
        return None
    oc = {"shape": shape, "s": float(s_o), "role": "occluder", **lk,
          "motion": {"kind": "flat_static", "p0": [float(0.5 * (X.min() + X.max())), float(y_o)], "yaw0": float(rng.uniform(0, 2 * np.pi))}}
    if shape == "cube":
        oc["motion"]["yaw0"] = float(math.radians(cam_yaw(cam)) + rng.uniform(-0.15, 0.15))        # 상자 앞면이 카메라를 향하게 (가장 좁은 폭으로 덮게)
    ok, r = _ratio_ok(cfg, cam, tgt, ctr, oc, fs, fe, T, fps, g)
    if not ok:
        return None
    oc["mode"], oc["screen_ratio"] = "static", float(r)
    return oc


def make_moving_object(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, g, objset, props):
    """굴러가는 공이 target 앞을 지나간다 (가림 구간 동안 덮을 수 있는 상대 속도 안에서). 크기 = 화면 크기 비 1.1–1.8."""
    P = cfg["occluders"]
    rs = _u(rng, P["object_screen_ratio"])
    gap = _u(rng, P["object_gap"])
    d_t = float(cam.to_px(ctr[(fs + fe) // 2][None])[2][0])
    k_t = math.sqrt(max(DIMS[tgt["shape"]][0], DIMS[tgt["shape"]][1]) * DIMS[tgt["shape"]][2])
    s_o = rs * tgt["s"] * k_t
    fy = _front_y(ctr, tgt, props)
    for _ in range(3):
        y_o = fy - gap - 0.5 * s_o
        d_o = float(cam.to_px(np.array([[ctr[(fs + fe) // 2, 0], y_o, 0.5 * s_o]]))[2][0])
        s_o = rs * tgt["s"] * k_t * d_o / d_t
    y_o = fy - gap - 0.5 * s_o
    if y_o - cam.C[1] < 0.3:
        return None
    X, Z = G2._plane_proj(cam.C, E[fs:fe + 1], y_o)
    if float(Z.max()) > 0.95 * s_o:
        return None
    D = max((fe - fs) / fps, 1e-3)
    vmax = (0.9 * s_o - float((X.max(1) - X.min(1)).max())) / D
    if vmax <= 0.03:
        return None
    xc = 0.5 * (X.min(1) + X.max(1))
    vt = float(xc[-1] - xc[0]) / D
    vx = vt + float(rng.choice([-1.0, 1.0]) * _u(rng, (0.3, 0.95)) * vmax)
    mu = _u(rng, P["object_moving_mu"])
    a = mu * g
    tm = (fs + fe) / 2 / fps
    if abs(vx) < 0.05 or abs(vx) / a < fe / fps + 0.2:
        return None
    x_at = float(xc[len(xc) // 2])
    x0 = x_at - vx * tm + np.sign(vx) * 0.5 * a * tm * tm
    lk, _ = look(rng, objset, "sphere", size=s_o, flex_only=True)
    if lk is None:
        return None
    oc = {"shape": "sphere", "s": float(s_o), "role": "occluder", **lk,
          "motion": {"kind": "flat_decel", "p0": [float(x0), float(y_o)], "v": [float(vx), 0.0], "mu": float(mu), "yaw0": 0.0, "omega": 0.0}}
    ok, r = _ratio_ok(cfg, cam, tgt, ctr, oc, fs, fe, T, fps, g)
    if not ok:
        return None
    oc["mode"], oc["screen_ratio"] = "rolling_ball", float(r)
    return oc


# ----------------------------------------------------------------------------- 그 밖 물체 (물체 수 채우기)
def sample_extra(rng, cfg, cam, T, fps, g, objset):
    O = cfg["objects"]
    moving = rng.uniform() < 0.5
    u, v = rng.uniform(30, 226), rng.uniform(110, 230)
    d = cam.ray(np.array(u), np.array(v))
    if d[2] > -1e-3:
        return None
    p = cam.C + (-cam.C[2] / d[2]) * d
    if np.hypot(*(p[:2] - cam.C[:2])) > 5.0:
        return None
    if moving:
        o = new_object(rng, cfg, objset, O["moving_shapes"], "distractor", True)
        pos, _, _ = track(o, T, fps, g)
        k = int(rng.integers(0, T))                                     # 그 시각에 화면 안 그 자리를 지나도록
        shift(o, p[0] - pos[k, 0], p[1] - pos[k, 1])
    else:
        shapes = PRIM_SHAPES if objset == "primitive" else list(REAL)
        o = new_object(rng, cfg, objset, shapes, "distractor", False)
        shift(o, p[0], p[1])
    return o


# ----------------------------------------------------------------------------- 한 clip
def sample_clip(cfg, split, idx, max_tries=1500, streams=4):
    """확인용 (preview): 번호 = 상황. 본 데이터 (train · test): 번호 i → 장면 i // 2 의 팔 A (짝수, target 있음) · B (홀수, 같은 장면에서 target 만 뺌).
    장면 요인 (물건 · 크기 · 색 · 속도 · 방향 · 가림 길이 · 물체 수 · 실내외 · 바닥 · 가림 하위 종류 · 측면 부호) 은 요인표에서 정해지고 다시 시도해도 그대로다."""
    if split != "preview":
        b, arm, sit, si, k, F = block_of(cfg, split, idx)
        for st in range(streams):
            try:
                A = _sample_clip(cfg, split, b, max_tries, st, sit=sit, n_total=int(F["count"]), F=F)
                A["stream"] = st
                return arm_spec(A, arm, idx, b, F)
            except RuntimeError:
                continue
        raise RuntimeError(f"장면 {split} {b} 설계 실패 (줄기 {streams} × {max_tries}, {situation_key(sit)}, {F})")
    for k in range(streams):
        try:
            sp = _sample_clip(cfg, split, idx, max_tries, k)
            sp["stream"] = k
            return sp
        except RuntimeError:
            continue
    raise RuntimeError(f"clip {split} {idx} 설계 실패 (줄기 {streams} × {max_tries})")


def arm_spec(A, arm, idx, b, F):
    """장면 설계 (A = target 있음) → 팔 spec. B = 같은 장면에서 target 만 뺌 (그림자도 없다). 네 팔은 렌더 뒤 문맥 끝 (프레임 32) 에서 이어 붙인다."""
    import copy
    sp = copy.deepcopy(A)
    sp.update(idx=int(idx), block=int(b), arm=arm, splice_frame=32, factors=F)
    if arm == "B":
        sp["ghost_target"] = sp["objects"][0]
        sp["objects"] = sp["objects"][1:]
        sp["cell"]["count_b"] = int(sp["cell"]["count"]) - 1
    return sp


def _sample_clip(cfg, split, idx, max_tries, stream, sit=None, n_total=None, F=None):
    sp = cfg["splits"][split]
    rng = np.random.default_rng([sp["seed"], idx, stream] if F is None else [sp["seed"], 100000 + idx, stream])
    if sit is None:
        sit, n_total = cell_of(cfg, idx)
    if split == "preview":                                              # 확인용: 상황마다 물체 수를 섞어 보여 준다
        lv = count_levels(sit); n_total = lv[idx % len(lv)]
    T, fps, g = int(cfg["frames"]), float(cfg["fps"]), float(cfg["g"])
    W = cfg["timing"]
    O = cfg["objects"]
    objset = sit["objset"]
    for attempt in range(max_tries):
        # 1 target (빈 장면도 가림 배치를 위해 보이지 않는 target 으로 설계한 뒤 뺀다)
        tmode = "static" if sit["target"] == "static" else ("moving" if sit["target"] == "moving" else
                                                              ("static" if (sit["occ_motion"] == "moving" and rng.uniform() < 0.5) else "moving"))
        if sit["place"] == "ramp":
            tgt = new_ramp_target(rng, cfg, objset, F)
            if tgt is None:
                continue
        elif tmode == "moving":
            tgt = new_object(rng, cfg, objset, O["moving_shapes"], "target", True, F)
        else:
            tgt = new_object(rng, cfg, objset, O["static_shapes"] if objset == "primitive" else list(REAL), "target", False, F)
        s_w, e_w = want_interval(rng, cfg, F)
        fs, fe = STRIDE * s_w, STRIDE * e_w
        fm = (fs + fe) // 2
        pos, _, R = track(tgt, T, fps, g)
        shift(tgt, -pos[fm, 0], -pos[fm, 1])                            # 가림 구간 가운데에 target 이 장면 가운데
        props = list(tgt.pop("props", []))
        pos, _, R = track(tgt, T, fps, g)
        ctr = center_path(tgt, pos, R)
        c = sample_camera(rng, cfg, sit["camera"], tgt, float(ctr[fm, 2]), F)
        if c is None:
            continue
        cam = cam_obj(c)
        E = G2.extent_pts(tgt, ctr, R)
        u, v, dep = cam.to_px(ctr[::STRIDE])
        px = cam.f * s_eff(tgt) / np.maximum(dep, 1e-3)
        if px.min() < O["min_px"] or (dep < 0.3).any():
            continue
        # 2 가리는 것
        panels, occ_objs = [], []
        if sit["occ_type"] == "barrier" and sit["occ_motion"] == "trapdoor":
            r = make_trapdoor(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, props)
        elif sit["occ_type"] == "barrier" and sit["occ_motion"] == "static":
            w = dict(cfg["occluders"]["barrier_static_w"])
            if sit["camera"] == "front":
                w.pop("table", None)
            sub = F["occ_sub"] if F is not None else _wchoice(rng, w)
            r = make_table(rng, cfg, cam, tgt, ctr, E, fs, fe) if sub == "table" else make_board(rng, cfg, cam, tgt, ctr, E, fs, fe, sub, props)
        elif sit["occ_type"] == "barrier":
            r = make_slider(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, props)
        else:
            if sit["occ_motion"] == "static":
                oc = make_static_object(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, g, objset, props, shape=F["occ_sub"] if F is not None else None)
            else:
                oc = make_moving_object(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, g, objset, props)
            r = None if oc is None else ([], [], oc["kind"] if sit["occ_motion"] == "static" else "rolling_ball")
            if oc is not None:
                if not distinct(tgt, oc, O["min_hue_diff"]):
                    continue
                occ_objs.append(oc)
        if r is None:
            continue
        pa, pr, sub = r
        panels += pa; props += pr
        if G2._hits_props(ctr, G2.foot_r(tgt), [p for p in props if p["role"] != "ramp"], panels, T, fps, clear=0.04):
            continue
        if G2.occ_hits_wedge(props, panels, T, fps):                     # 가림막이 빗면 쐐기 속에 서면 안 된다
            continue
        # 3 가림 타이밍 (두 추정이 모두 late 정의를 지켜야 — 렌더와 어긋남을 줄인다)
        des = G2.panel_boxes(panels, T, fps) + G2.prop_boxes([p for p in props if p["role"].startswith("occ_")]) + G2.obj_boxes(occ_objs, T, fps, g, 1.0)
        off = G2.off_frame(cam, E)
        ok, ev = G2.timing_ok("late", G2.target_hidden(cam, des, E)[::STRIDE], off[::STRIDE], W)
        if not ok:
            continue
        E_in = ctr[:, None] + 0.55 * (E - ctr[:, None])
        if not G2.timing_ok("late", G2.target_hidden(cam, des, E_in)[::STRIDE], off[::STRIDE], W)[0]:
            continue
        others = G2.prop_boxes([p for p in props if p["role"] != "ramp" and not p["role"].startswith("occ_")]) + G2.wedge_boxes(props)
        if others and G2.target_hidden(cam, others, E).max() > 0.05:    # 레일 · 빗면이 target 을 가리면 안 된다 (낮은 카메라)
            continue
        world = [(tgt, ctr)]
        bad = False
        for oc in occ_objs:
            pc, _, Rc = track(oc, T, fps, g); coc = center_path(oc, pc, Rc)
            if G2.collide(tgt, ctr, oc, coc, 0.04) or G2._hits_props(coc, G2.foot_r(oc), props, panels, T, fps) or G2.obj_hits_wedge(oc, coc, props):
                bad = True
            world.append((oc, coc))
        if bad:
            continue
        # 4 그 밖 물체 (총수 n_total) — 잘 보이고 (≥ 28 px · 창의 70 % 이상 보임) · target 을 가리지 않고 · 부딪히지 않고 · target 과 달라 보이게
        n_extra = n_total - int(sit["target"] != "none") - int(sit["occ_type"] == "object")
        extras, fail = [], False
        for _ in range(n_extra):
            for _try in range(80):
                d = sample_extra(rng, cfg, cam, T, fps, g, objset)
                if d is None or not distinct(tgt, d, O["min_hue_diff"]):
                    continue
                pd, _, Rd = track(d, T, fps, g); cd = center_path(d, pd, Rd)
                if any(G2.collide(d, cd, ow, cw, 0.12) for ow, cw in world) or G2._hits_props(cd, G2.foot_r(d), props, panels, T, fps) \
                        or G2.obj_hits_wedge(d, cd, props):
                    continue
                if G2.target_hidden(cam, G2.obj_boxes([d], T, fps, g, 1.05), E).max() > 0.02:
                    continue
                ud, vd, dd = cam.to_px(cd[::STRIDE])
                inn = (ud > 0) & (ud < 256) & (vd > 0) & (vd < 256) & (dd > 0.3)
                if inn.mean() < O["extra_visible_min"] or np.median(cam.f * s_eff(d) / np.maximum(dd, 1e-3)) < O["extra_min_px"]:
                    continue
                blockers = [w_[0] for w_ in world if (sit["target"] != "none" or w_[0] is not tgt)]
                vis_occ = des + G2.wedge_boxes(props) + G2.obj_boxes(blockers, T, fps, g, 1.0)
                hid_d = G2.target_hidden(cam, vis_occ, G2.extent_pts(d, cd, Rd))[::STRIDE]
                if hid_d[inn].max(initial=0.0) > O["extra_max_hidden"]:       # 11-01 (사용자): 그 밖 물체가 반쯤 가려지면 (판 위로 튀어나옴) 가림 사건이 섞인다 → 어느 순간도 10 % 넘게 가려지지 않게
                    continue
                extras.append(d); world.append((d, cd)); break
            else:
                fail = True
                break
        if fail:
            continue
        objs = ([tgt] if sit["target"] != "none" else []) + extras + occ_objs
        tv = np.gradient(ctr[::STRIDE], STRIDE / fps, axis=0)
        spec = {"cam": c, "objects": objs, "panels": panels, "props": props, "env": sample_env(rng, cfg, F), "shadows": bool(O["shadows"]),
                "cell": dict(sit, count=int(n_total), occ_sub=sub, target_mode=tmode, target_shape=tgt["shape"], target_kind=tgt.get("kind"),
                             ramp_mode=tgt.get("ramp_mode")),
                "event_samples": list(ev), "seed": int(sp["seed"]), "idx": int(idx), "attempt": attempt, "T": T, "fps": fps, "g": g,
                "size": int(cfg["size"]), "version": cfg["name"], "split": split, "design": "gen_design_v3",
                "target_px_median": float(np.median(px)), "target_speed_at_hide": float(np.linalg.norm(tv[ev[0]])),
                "target_speed_max": float(np.linalg.norm(tv, axis=1).max())}
        if sit["target"] == "none":
            spec["ghost_target"] = tgt                                  # 기록만 (렌더 안 함): 가림 배치에 쓴 보이지 않는 target
        if G2.physics_check(spec):
            continue
        return spec
    raise RuntimeError(f"clip {split} {idx} 설계 실패 ({sit}, 물체 {n_total})")


def physics_check(spec, tol=0.02):
    return G2.physics_check(spec, tol)


# ----------------------------------------------------------------------------- 확인용 (상황마다 하나)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    cfg = load_cfg(a.config, "preview")
    out = cfg["out"]
    os.makedirs(f"{out}/design", exist_ok=True)
    S = situations(cfg)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(a.workers, len(S))) as ex:
        specs = list(ex.map(_one_preview, [(a.config, i) for i in range(len(S))]))
    idx = []
    for i, sp in enumerate(specs):
        cid = f"preview{cfg['splits']['preview']['seed']}_{i:06d}"
        if isinstance(sp, str):
            print(i, S[i], "실패", sp, flush=True)
            open(f"{out}/design/{cid}.fail", "w").write(sp)
            idx.append({"i": i, "cell": S[i], "fail": sp}); continue
        json.dump(sp, open(f"{out}/design/{cid}.json", "w"), default=lambda x: x.item() if hasattr(x, "item") else str(x))
        c = sp["cell"]
        idx.append({"i": i, "cid": cid, "cell": c, "event": sp["event_samples"], "px": round(sp["target_px_median"], 1), "attempt": sp["attempt"], "stream": sp["stream"]})
        print(i, cid, c["objset"], c["place"], c["camera"], c["occ_type"], c["occ_motion"], c["target"], "물체", c["count"], c["occ_sub"],
              c.get("target_kind"), c.get("ramp_mode") or "", "가림", sp["event_samples"], "px", round(sp["target_px_median"], 1),
              "속력", round(sp["target_speed_at_hide"], 2), "시도", sp["attempt"] + 1500 * sp["stream"], flush=True)
    json.dump(idx, open(f"{out}/preview_index.json", "w"), indent=1, ensure_ascii=False, default=str)


def _one_preview(arg):
    path, i = arg
    cfg = load_cfg(path, "preview")
    try:
        return sample_clip(cfg, "preview", i, max_tries=1500, streams=int(cfg["checks"].get("design_streams", 4)))
    except RuntimeError as e:
        return str(e)


if __name__ == "__main__":
    main()
