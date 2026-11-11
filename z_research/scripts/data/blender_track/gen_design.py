#!/usr/bin/env python3
"""BlenderTrack v1 설계 — 설정 파일 (configs/<판>.yaml) 이 모든 분포를 정한다 (2026-10-31). numpy 만, bpy 없음.

v0 (design.py, 데이터 v0 를 낸 정본 — 고치지 않는다) 와 다른 점 (데이터 설계 원칙, z_analysis/release/06_blender_track/README.md):
  ① 물리 운동만: slide (등속 미끄러짐) · roll (구 굴러감, 회전 포함) · friction (마찰 감속 → 정지) · ramp (경사로 굴러 오르내림,
     a = 5/7 g sinθ, 되돌아옴 포함) · toss (포물선 + 튕김) · wall (벽에 부딪혀 반사, 반발 계수 e) · static. 원인 없는 가속 · 휨 없음.
  ② 누수 차단: 물체 그림자 끔 · 가림막 폭 · 높이 · 가로 위치를 숨는 구간에 맞추지 않고 무작위 (가림은 기하로 생긴다) ·
     trapdoor 가 일어서는 · 눕는 시점을 숨는 구간 앞뒤로 흔든다.
  ③ 시간 · 길이: 기본 20 fps · 8 초 (보폭 2 = 10 fps = VOBench).
  ⑤ (10-31 재렌더) 옆으로 움직이는 판 (mode slide): target 과 같은 방향 (판이 따라잡음 · target 이 따라잡음) · 반대 방향 (마주 옴) ·
     target 이 서 있으면 판이 지나감. target 과 무관하게 지나가는 판도 있다 (extra). VOBench PFT 가 안 옮겨진 장면이
     "가림막과 물체가 서로 움직이는" 경우였다 (Summary 탭, 10-31).
  정정 (10-31): 첫 v1 렌더는 구르는 공이 바닥 점 둘레로 돌아 바닥을 뚫었다 (train 40 %) · 빗면 공이 면에 파묻히고 내려갈 때 거꾸로 돌았다.
     tracks() 가 원점을 옮겨 자기 중심 둘레로 돌게 하고, 빗면 중심 = 면 + r · 법선, 굴림 각 = 부호 있는 거리 / r 로 고쳤다.
     옛 렌더는 nfs_cache/blender_track/v1_rollbug_20261031 (쓰지 말 것).
  ④ 분할: train · val (같은 분포) · ood (범위 밖: 더 빠름 · 처음 보는 모양 · 더 가파른 경사) — 설정의 splits.<이름>.overrides.
좌표 · 카메라는 v0 와 같다 (design.Cam, 바닥면 중심 원점, 미터 · 초).
"""
from __future__ import annotations

import copy
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D0  # noqa: E402  (Cam · DIMS · 메시 비율 · 판 각도 · 가림 판정 · 색 — v0 와 공유)

G = 9.81
DIMS = D0.DIMS


# ----------------------------------------------------------------------------- 설정
def load_cfg(path: str, split: str | None = None) -> dict:
    """yaml (또는 json — blender env 에는 yaml 모듈이 없어 pipeline.sh 가 json 으로 바꿔 넘긴다) → dict.
    split 을 주면 splits.<split>.overrides (점 경로) 를 덮는다."""
    if path.endswith(".json"):
        import json
        cfg = json.load(open(path))
    else:
        import yaml
        cfg = yaml.safe_load(open(path))
    if split:
        for k, v in (cfg["splits"][split].get("overrides") or {}).items():
            cur = cfg
            ks = k.split(".")
            for kk in ks[:-1]:
                cur = cur[kk]
            cur[ks[-1]] = v
        cfg["_split"] = split
    return cfg


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def _choice_p(rng, ps):
    return int(rng.choice(len(ps), p=np.asarray(ps, float) / np.sum(ps)))


# ----------------------------------------------------------------------------- 회전
def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _axis_angle(axis, ang):
    """축 (3,) · 각 (T,) → (T, 3, 3) (로드리게스)."""
    a = np.asarray(axis, float); a = a / max(np.linalg.norm(a), 1e-9)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    ang = np.asarray(ang, float)[:, None, None]
    return np.eye(3)[None] + np.sin(ang) * K[None] + (1 - np.cos(ang)) * (K @ K)[None]


# ----------------------------------------------------------------------------- 운동 → 궤적
def track(o: dict, T: int, fps: float):
    """물체 하나 → pos (T,3) 바닥면 중심 · yaw (T,) · R (T,3,3)."""
    m, t = o["motion"], np.arange(T) / fps
    k = m["kind"]
    p0, v = np.array(m["p0"], float), np.array(m.get("v", [0.0, 0.0]), float)
    r = 0.5 * o["s"] * DIMS[o["shape"]][0]
    yaw = m.get("yaw0", 0.0) + m.get("omega", 0.0) * t
    Ry = np.stack([_rz(a) for a in yaw])
    z = np.zeros(T)
    if k == "static":
        xy = np.repeat(p0[None], T, 0)
    elif k in ("slide", "roll", "toss"):
        xy = p0 + t[:, None] * v
        if k == "toss":
            z = D0._toss_z(t, m["z0"], m["vz0"], m["e"], m["t_launch"])
    elif k == "friction":
        sp = np.linalg.norm(v); dec = m["mu"] * G
        ts = sp / dec if sp > 0 else 0.0
        tt = np.minimum(t, ts)[:, None]
        xy = p0 + v * tt - 0.5 * dec * (v / max(sp, 1e-9)) * tt ** 2
    elif k == "wall":
        xy = p0 + t[:, None] * v
        tc = m["t_contact"]
        aft = t > tc
        xc = p0[0] + v[0] * tc
        xy[aft, 0] = xc - m["e"] * v[0] * (t[aft] - tc)
    elif k == "ramp":
        # 경사로 위: s(t) = s0 + v0 t + ½ a t² (s = 아래 끝에서 경사를 따라 잰 거리, a < 0). 아래 끝 (s = 0) 에 닿으면
        # 그 속력으로 바닥을 따라 같은 방향으로 계속 굴러간다 (마찰 μ_g 로 감속).
        # 2026-10-31 정정 (v1 첫 렌더의 빗면 버그): ① 공 중심 = 면 위 점 + r · 면 법선 (전에는 + r ẑ 라 면에 파묻혔다),
        # ② 굴림 각 = 경사를 따라 움직인 부호 있는 거리 / r (전에는 거리의 절댓값이라 내려갈 때 거꾸로 돌았다).
        th, sx = math.radians(m["theta_deg"]), m["sx"]
        s = m["s0"] + m["v0"] * t + 0.5 * m["a"] * t ** 2
        disc = m["v0"] ** 2 - 2 * m["a"] * m["s0"]                 # s(t) = 0 의 근 (내려와 닿는 시각)
        tb = (-m["v0"] - math.sqrt(max(disc, 0.0))) / m["a"]
        sc = np.clip(s, 0, None)
        x = m["x_low"] + sx * sc * math.cos(th) - sx * r * math.sin(th)           # 중심의 x (법선 방향 r 만큼 비켜남)
        z = np.maximum(sc * math.sin(th) + r * math.cos(th), r) - r                # 바닥면 중심 = 중심 − r ẑ (바닥 아래로 안 감)
        ang = sx * (s - m["s0"]) / max(r, 1e-6)                                    # +x 로 d 가면 +y 축으로 d / r
        if tb < t[-1]:
            vb = abs(m["v0"] + m["a"] * tb)                          # 닿을 때 속력 (경사를 따라)
            aft = t > tb
            dec = m.get("mu_g", 0.02) * G
            tt = np.minimum(t[aft] - tb, vb / dec)
            g = vb * tt - 0.5 * dec * tt ** 2                       # 바닥 위 이동 거리
            x[aft] = m["x_low"] - sx * r * math.sin(th) - sx * g    # 아래 끝에서 경사 반대쪽 (내려온 방향) 으로, 위치 연속
            z[aft] = 0.0
            ang[aft] = -sx * (m["s0"] + g) / max(r, 1e-6)
        R = _axis_angle([0, 1, 0], ang) @ Ry
        return np.c_[x, np.full(T, m["y"]), z], yaw, R
    else:
        raise ValueError(k)
    pos = np.c_[xy, z]
    if k in ("roll", "friction") and o["shape"] == "sphere":   # 구는 굴러간다: 진행 방향에 수직인 수평축, 각 = 이동 거리 / 반지름
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        hv = v / max(np.linalg.norm(v), 1e-9)
        R = _axis_angle([-hv[1], hv[0], 0.0], d / max(r, 1e-6)) @ Ry
    else:
        R = Ry
    return pos, yaw, R


def tracks(spec: dict):
    """render.Scene.track_fn: 물체마다 (메시 원점 위치, yaw, R) · 판마다 각도 · 판마다 가로 위치.

    메시 원점은 바닥면 중심이다. 2026-10-31 정정: 구르기 회전 R 를 원점 (바닥 점) 둘레로 걸면 공이 바닥을 뚫고 돌았다
    (v1 첫 렌더, train 의 40 %). 그래서 원점 위치 = pos + h ẑ − R (h ẑ) 로 두어 물체가 자기 중심 (pos + h ẑ) 둘레로 돌게 한다.
    yaw 만 도는 물체는 R (h ẑ) = h ẑ 라 원점 = pos (바뀌지 않음). 라벨의 pos 는 이 원점이다.
    """
    T, fps = spec["T"], spec["fps"]
    out = []
    for o in spec["objects"]:
        pos, yaw, R = track(o, T, fps)
        hz = np.array([0.0, 0.0, 0.5 * o["s"] * DIMS[o["shape"]][2]])
        out.append((pos + hz - np.einsum("tij,j->ti", R, hz), yaw, R))
    return out, [panel_ang(p, T, fps) for p in spec["panels"]], [panel_x(p, T, fps) for p in spec["panels"]]


# ----------------------------------------------------------------------------- 판 (v1: 옆으로 미끄러지는 판 slide 추가)
def panel_x(p: dict, T: int, fps: float) -> np.ndarray:
    """판 가로 위치 (T,). slide = x0 + vx t (서 있는 채로 옆으로 움직임), 나머지 = 고정 x."""
    if p.get("mode") == "slide":
        return p["x0"] + p["vx"] * np.arange(T) / fps
    return np.full(T, float(p["x"]))


def panel_ang(p: dict, T: int, fps: float) -> np.ndarray:
    return np.zeros(T) if p.get("mode") == "slide" else D0.panel_angle(p, T, fps)


def panel_hides(cam, p: dict, T: int, fps: float, P: np.ndarray) -> np.ndarray:
    """점 P (T, n, 3) 가 판 p 뒤에 가려지나 (T, n). v0 판정을 프레임별 가로 위치로."""
    return D0.panel_hides(cam, dict(p, x=panel_x(p, T, fps)[:, None]), panel_ang(p, T, fps), P)


def panel_hits_path(p: dict, pos: np.ndarray, r: float, T: int, fps: float, pad: float = 0.15) -> bool:
    """판 바닥 자국과 물체 바닥 원이 어느 프레임에서든 겹치나 (slide 는 프레임별 위치로)."""
    if p.get("mode") != "slide":
        return D0._panel_hits_path(p, panel_ang(p, T, fps), pos, r, pad)
    xs = panel_x(p, T, fps)
    inx = np.abs(pos[:, 0] - xs) <= p["w"] / 2 + r + pad
    iny = np.abs(pos[:, 1] - p["y"]) <= p["th"] / 2 + r + pad
    return bool((inx & iny).any())


def panels_clash(p: dict, q: dict, T: int, fps: float) -> bool:
    """같은 깊이 (0.3 m 안) 의 두 판이 어느 프레임에서든 가로로 겹치나."""
    return abs(q["y"] - p["y"]) < 0.3 and bool((np.abs(panel_x(p, T, fps) - panel_x(q, T, fps)) < (p["w"] + q["w"]) / 2).any())


def panel_on_props(p: dict, props: list, T: int, fps: float) -> bool:
    xy = np.c_[panel_x(p, T, fps), np.full(T, p["y"])]
    return any(_rect_hit(pr["rect"], xy, p["w"] / 2) for pr in props)


# ----------------------------------------------------------------------------- 표본 뽑기
def _motion(rng, cfg, cam, shape, kind, y0, T, fps):
    M = cfg["motion"]
    t_end = (T - 1) / fps
    hw = cam.half_width(y0)
    v = _u(rng, M["speed_frac"]) * 2 * hw
    psi = math.radians(_u(rng, M["heading_wide_deg"]) if rng.uniform() < M["heading_wide_p"] else _u(rng, M["heading_deg"]))
    sgn = float(rng.choice([-1.0, 1.0]))
    vv = np.array([sgn * v * math.cos(psi), v * math.sin(psi)])
    m = {"kind": kind, "p0": [0.0, y0], "v": vv.tolist(), "yaw0": float(rng.uniform(0, 2 * np.pi)),
         "omega": float(math.radians(_u(rng, M["spin_deg_s"]))) if (rng.uniform() < M["spin_p"] and kind in ("slide", "toss", "wall", "static")) else 0.0}
    if kind == "static":
        m["v"] = [0.0, 0.0]
    elif kind == "friction":
        m["mu"] = _u(rng, M["friction_mu"])
    elif kind == "toss":
        m["v"] = (vv * rng.uniform(0.5, 1.0)).tolist()
        if rng.uniform() < 0.6:
            m["z0"], m["vz0"] = 0.0, _u(rng, M["toss_vz"])
        else:
            m["z0"], m["vz0"] = float(rng.uniform(0.5, 2.5)), float(rng.uniform(-1.0, 4.0))
        m["e"] = _u(rng, M["restitution"])
        m["t_launch"] = float(rng.uniform(0, 0.5) * t_end) if rng.uniform() < 0.5 else 0.0
    elif kind == "wall":
        m["v"] = [sgn * v, float(v * math.sin(psi) * 0.3)]       # 벽 (x 에 수직) 쪽으로 주로 가로
        m["e"] = _u(rng, M["restitution"])
        m["t_contact"] = float(rng.uniform(0.25, 0.7) * t_end)
    elif kind == "ramp":
        th = _u(rng, M["ramp_deg"])
        a = -5.0 / 7.0 * G * math.sin(math.radians(th))          # 구가 굴러 내려가는 가속 (경사 위쪽 = +s)
        L = float(min(rng.uniform(4.0, 12.0), M["ramp_max_height"] / math.sin(math.radians(th))))
        if rng.uniform() < 0.5:                                    # 위에서 굴러 내려옴 (다 내려오면 바닥으로 이어서)
            s0, v0 = float(rng.uniform(0.3, 0.95) * L), float(-rng.uniform(0.0, 0.5) * v)
        else:                                                      # 아래에서 밀려 올라가 정점 (경사 길이의 40–90 %) 에서 되돌아옴
            s0 = float(rng.uniform(0.02, 0.3) * L)
            apex = float(rng.uniform(0.4, 0.9) * L)
            v0 = float(math.sqrt(max(2 * abs(a) * (apex - s0), 0.0)))
        m.update(theta_deg=th, sx=sgn, a=a, L=L, s0=s0, v0=v0, x_low=0.0, y=y0, mu_g=float(rng.uniform(0.01, 0.08)))
        m["v"] = [0.0, 0.0]
    return m


def _shift_x(m, dx):
    if m["kind"] == "ramp":
        m["x_low"] += dx
    else:
        m["p0"][0] += dx


def _props(o, m, rng):
    """ramp · wall 의 소품 (render.Scene 이 바닥면 중심 원점 상자로 그린다)."""
    r = 0.5 * o["s"] * DIMS[o["shape"]][0]
    col = D0._color(rng, (0.0, 0.5), (0.3, 0.85))
    if m["kind"] == "ramp":
        th, sx, L = math.radians(m["theta_deg"]), m["sx"], m["L"]
        Wd, thick = max(1.2, 3.0 * r), 0.12
        phi = -sx * th
        top_mid = np.array([m["x_low"] + sx * (L / 2) * math.cos(th), m["y"], (L / 2) * math.sin(th)])
        loc = top_mid - np.array([thick * math.sin(phi), 0.0, thick * math.cos(phi)])
        xs = sorted([m["x_low"], m["x_low"] + sx * L * math.cos(th)])
        return [{"role": "ramp", "loc": loc.tolist(), "rot": [0.0, phi, 0.0], "scale": [L, Wd, thick], "color": col,
                 "rect": [xs[0], xs[1], m["y"] - Wd / 2, m["y"] + Wd / 2]}]
    if m["kind"] == "wall":
        thick, h, Wd = 0.15, float(max(0.6, rng.uniform(1.2, 2.0) * o["s"] * DIMS[o["shape"]][2])), max(1.5, 4.0 * r)
        sg = 1.0 if m["v"][0] > 0 else -1.0
        xc = m["p0"][0] + m["v"][0] * m["t_contact"]
        xw = xc + sg * (r + thick / 2)
        yc = m["p0"][1] + m["v"][1] * m["t_contact"]
        return [{"role": "wall", "loc": [xw, yc, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [thick, Wd, h], "color": col,
                 "rect": [xw - thick / 2, xw + thick / 2, yc - Wd / 2, yc + Wd / 2]}]
    return []


def _rect_hit(rect, pos, r, pad=0.1):
    x0, x1, y0, y1 = rect
    return bool(((pos[:, 0] > x0 - r - pad) & (pos[:, 0] < x1 + r + pad) & (pos[:, 1] > y0 - r - pad) & (pos[:, 1] < y1 + r + pad)).any())


def _sample_obj(rng, cfg, cam, T, fps, kinds=None, px=None, y_rng=(4.0, 14.0), size_check=True):
    O, M = cfg["objects"], cfg["motion"]
    kinds = kinds or M["kinds"]
    names, ps = list(kinds), [kinds[k] for k in kinds]
    kind = names[_choice_p(rng, ps)]                               # 종류는 한 번만 (재시도마다 바꾸면 거절이 잦은 종류가 줄어든다)
    for _ in range(80):
        shape = "sphere" if kind in ("roll", "ramp") else str(rng.choice(O["shapes"]))
        y0 = rng.uniform(*y_rng)
        s = float(np.clip(_u(rng, px or O["px"]) * y0 / cam.f, 0.2, 2.5))
        o = {"shape": shape, "s": s}
        m = _motion(rng, cfg, cam, shape, kind, y0, T, fps)
        o["motion"] = m
        pos, _, _ = track(o, T, fps)
        if kind != "static":                                       # 가운데 시각에 화면 가로 30–70 % 에 오게 옮긴다
            tm = T // 2
            x_m = rng.uniform(-0.4, 0.4) * cam.half_width(max(pos[tm, 1], 1.0))
            _shift_x(m, x_m - pos[tm, 0])
        else:
            _shift_x(m, rng.uniform(-0.85, 0.85) * cam.half_width(y0))
        pos, _, _ = track(o, T, fps)
        if pos[:, 1].min() < 3.0 or pos[:, 1].max() > 30.0:
            continue
        if kind == "ramp":                                         # 경사로 꼭대기를 넘지 않게 (바닥면 중심 z → 경사 거리)
            th_ = math.radians(m["theta_deg"]); r_ = 0.5 * s * DIMS[shape][0]
            if ((pos[:, 2] + r_ - r_ * math.cos(th_)) > m["L"] * math.sin(th_) * 0.98).any():
                continue
        fr, _, _ = D0.in_frame_frac(cam, pos, shape, s)
        if fr < M["in_frame_min"]:
            continue
        if size_check:
            dx, dy, dz = (s * d for d in DIMS[shape])
            cu, cv, dep = cam.to_px(pos + np.array([0.0, 0.0, 0.5 * dz]))
            sz = cam.f * math.sqrt(max(dx, dy) * dz) / np.maximum(dep, 1e-3)
            inn = (cu >= 0) & (cu < cam.size) & (cv >= 0) & (cv < cam.size)
            if not inn.any() or np.median(sz[inn]) < O["min_px_median"] or sz[inn].min() < O["min_px"]:
                continue
        o.update(color=D0._color(rng), rough=float(rng.uniform(0.3, 0.9)),
                 metal=float(rng.uniform(0.0, 0.6)) if rng.uniform() < 0.2 else 0.0)
        o["props"] = _props(o, m, rng)
        return o
    return None


def _hidden_runs(hid_frac, thr=0.9):
    runs, st = [], None
    for i, h in enumerate(list(hid_frac >= thr) + [False]):
        if h and st is None:
            st = i
        elif not h and st is not None:
            runs.append((st, i - 1)); st = None
    return runs


def _event_panel(rng, cfg, cam, tgt, pos_t, E, T, fps):
    """경로 위 가림막 하나. 폭 · 높이 · 가로 위치는 무작위, 가림은 기하로 생긴다 (숨는 구간에 맞추지 않음)."""
    P = cfg["occluders"]
    t_end = (T - 1) / fps
    r_t = D0.footprint_r(tgt["shape"], tgt["s"])
    for _ in range(40):
        tc = rng.uniform(0.3, 0.7) * t_end if rng.uniform() < 0.7 else rng.uniform(0.12, 0.88) * t_end
        f = int(round(tc * fps))
        near = slice(max(0, f - int(1.5 * fps)), min(T, f + int(1.5 * fps)))
        yp = float(pos_t[near, 1].min() - r_t - _u(rng, P["gap"]))
        if yp < 1.8:
            continue
        C = cam.C
        top = E[f][E[f][:, 2].argmax()]
        ctr = pos_t[f] + np.array([0, 0, 0.5 * tgt["s"] * DIMS[tgt["shape"]][2]])
        tp = (yp - C[1]) / (ctr[1] - C[1])
        Xc = C[0] + tp * (ctr[0] - C[0])
        Zt = C[2] + (yp - C[1]) / (top[1] - C[1]) * (top[2] - C[2])
        w = _u(rng, P["width"])
        p = {"role": "event", "x": float(Xc + rng.uniform(-0.35, 0.35) * w), "y": yp, "w": w,
             "h": float(max(0.3, Zt * _u(rng, P["height_over_obj"]))), **D0._panel_style(rng), "mode": "static"}
        slide = rng.uniform() < P.get("slide_p", 0.0)
        if slide:                                                # v1 (10-31): 서 있는 채로 옆으로 움직이는 판 — 같은 방향 (따라잡음 / 따라잡힘) · 반대 방향 (마주 옴)
            # target 중심이 판 평면에 투영되는 x 의 속도 (판 깊이에서 본 target 의 가로 속도)
            ctr_all = pos_t + np.array([0, 0, 0.5 * tgt["s"] * DIMS[tgt["shape"]][2]])
            tp_all = (yp - C[1]) / (ctr_all[:, 1] - C[1])
            Xall = C[0] + tp_all * (ctr_all[:, 0] - C[0])
            vt = float(np.gradient(Xall)[f] * fps)
            hw = cam.half_width(yp)
            vmin, vmax = P["slide_speed_frac"][0] * 2 * hw, P["slide_speed_frac"][1] * 2 * hw
            if abs(vt) < P["slide_pass_below_frac"] * 2 * hw:   # target 이 (거의) 서 있다 → 판이 지나간다
                if rng.uniform() > P["slide_pass_keep_p"]:        # 멈춘 구간에 몰리지 않게 (마찰로 선 뒤 등) 다른 시각을 다시 뽑는다
                    continue
                vx = float(rng.choice([-1.0, 1.0]) * rng.uniform(vmin, vmax)); rel = "pass"
            elif rng.uniform() < P["slide_same_p"]:
                ratio = _u(rng, P["slide_ratio_fast"]) if rng.uniform() < 0.5 else _u(rng, P["slide_ratio_slow"])
                vx = float(np.sign(vt) * float(np.clip(ratio * abs(vt), vmin, vmax))); rel = "same"
            else:
                vx = float(-np.sign(vt) * float(np.clip(_u(rng, P["slide_ratio_opp"]) * abs(vt), vmin, vmax))); rel = "opposite"
            p.update(mode="slide", vx=vx, x0=float(p["x"] - vx * tc), rel=rel)
        if panel_hits_path(p, pos_t, r_t, T, fps):
            continue
        hid = panel_hides(cam, p, T, fps, E).mean(1)
        runs = [ru for ru in _hidden_runs(hid) if (ru[1] - ru[0] + 1) / fps >= cfg["target"]["min_hidden_run_s"]]
        if not runs:
            continue
        if not slide and rng.uniform() < P["trapdoor_p"]:        # 숨는 구간 앞뒤로 시점을 흔든다 (누수 차단)
            f0, f1 = runs[0]
            p.update(mode="trapdoor", rise=float(rng.uniform(0.1, 0.25)),
                     t_up=float(max(0.0, f0 / fps - _u(rng, P["trapdoor_lead_s"]))),
                     t_down=float(min(t_end, f1 / fps + _u(rng, P["trapdoor_lag_s"]))))
            if panel_hits_path(p, pos_t, r_t, T, fps):
                continue
        return p
    return None


def _target_visibility(cam, o, pos, panels, T, fps):
    E = D0.extent_pts(o["shape"], o["s"], pos)
    hid = np.zeros(E.shape[:2], bool)
    for p in panels:
        hid |= panel_hides(cam, p, T, fps, E)
    u, v, _ = cam.to_px(E)
    off = ((u < 0) | (u >= cam.size) | (v < 0) | (v >= cam.size)).mean(1)
    h = hid.mean(1)
    vis = float(((h + off) < 0.3).mean())
    run = best = 0
    for x in (h >= 0.9) & (off < 0.5):
        run = run + 1 if x else 0
        best = max(best, run)
    return vis, best / fps


def _try_clip(rng, cfg, T, fps):
    C = cfg["camera"]
    cam_p = {"h": _u(rng, C["h"]), "pitch": _u(rng, C["pitch"]), "lens": _u(rng, C["lens"])}
    cam = D0.Cam(cam_p["h"], cam_p["pitch"], cam_p["lens"], size=cfg["size"])
    tgt = _sample_obj(rng, cfg, cam, T, fps)
    if tgt is None:
        return None
    tgt["role"] = "target"
    pos_t, _, _ = track(tgt, T, fps)
    r_t = D0.footprint_r(tgt["shape"], tgt["s"])
    tracks_ = [(pos_t, r_t)]
    props = list(tgt.pop("props"))
    E = D0.extent_pts(tgt["shape"], tgt["s"], pos_t)
    panels = []
    for _ in range(_choice_p(rng, cfg["occluders"]["n_on_path_p"])):
        p = _event_panel(rng, cfg, cam, tgt, pos_t, E, T, fps)
        if p is None or any(panels_clash(p, q, T, fps) for q in panels):
            continue
        if panel_on_props(p, props, T, fps):
            continue
        panels.append(p)
    objs = [tgt]
    dk = {k: v for k, v in cfg["motion"]["kinds"].items() if k in ("slide", "roll", "friction", "toss", "static")}
    for _ in range(_choice_p(rng, cfg["objects"]["n_distractor_p"])):
        o = _sample_obj(rng, cfg, cam, T, fps, kinds=dk)
        if o is None:
            continue
        o.pop("props", None)
        pos_o, _, _ = track(o, T, fps)
        r_o = D0.footprint_r(o["shape"], o["s"])
        if any(D0._collide(pos_o, r_o, pa, ra) for pa, ra in tracks_) or any(_rect_hit(pr["rect"], pos_o, r_o) for pr in props):
            continue
        if any(panel_hits_path(p, pos_o, r_o, T, fps) for p in panels):
            continue
        o["role"] = "distractor"; objs.append(o); tracks_.append((pos_o, r_o))
    P = cfg["occluders"]
    t_end = (T - 1) / fps
    for _ in range(_choice_p(rng, P["n_extra_p"])):
        yp = float(rng.uniform(3.0, 12.0)); hw = cam.half_width(yp)
        p = {"role": "extra", "x": float(rng.uniform(-0.8, 0.8) * hw), "y": yp, "w": _u(rng, P["width"]),
             "h": float(rng.uniform(0.6, 2.5)), **D0._panel_style(rng)}
        u_ = rng.uniform()
        if u_ < P.get("extra_slide_p", 0.0):                    # target 과 무관하게 옆으로 지나가는 판 (움직이는 판 = 늘 target 을 가린다는 단서 차단)
            vx = float(rng.choice([-1.0, 1.0]) * _u(rng, P["slide_speed_frac"]) * 2 * hw)
            p.update(mode="slide", vx=vx, x0=float(p["x"] - vx * rng.uniform(0.2, 0.8) * t_end), rel="extra")
        elif u_ < P.get("extra_slide_p", 0.0) + P["trapdoor_p"]:
            ts = rng.uniform(0.05, 0.8) * t_end
            p.update(mode="trapdoor", t_up=float(ts), t_down=float(ts + rng.uniform(0.2, 1.5)), rise=float(rng.uniform(0.1, 0.25)))
        else:
            p.update(mode="static")
        if any(panel_hits_path(p, pa, ra, T, fps) for pa, ra in tracks_):
            continue
        if any(panels_clash(p, q, T, fps) for q in panels):
            continue
        if panel_on_props(p, props, T, fps):
            continue
        panels.append(p)
    for _ in range(_choice_p(rng, cfg["objects"]["n_clutter_p"])):
        o = _sample_obj(rng, cfg, cam, T, fps, kinds={"static": 1.0}, px=(10, 40), y_rng=(4.0, 20.0), size_check=False)
        if o is None:
            continue
        o.pop("props", None)
        pos_o, _, _ = track(o, T, fps)
        r_o = D0.footprint_r(o["shape"], o["s"])
        if any(D0._collide(pos_o, r_o, pa, ra) for pa, ra in tracks_) or any(_rect_hit(pr["rect"], pos_o, r_o) for pr in props):
            continue
        if any(panel_hits_path(p, pos_o, r_o, T, fps) for p in panels):
            continue
        o["role"] = "clutter"; objs.append(o); tracks_.append((pos_o, r_o))
    vis, run = _target_visibility(cam, tgt, pos_t, panels, T, fps)
    if vis < cfg["target"]["min_visible_frac"] or run > cfg["target"]["max_hidden_run_s"]:
        return None
    rng_env = rng
    env = {"floor": str(rng_env.choice(D0.FLOORS)), "floor_c1": D0._color(rng_env, (0.0, 0.5), (0.15, 0.6)),
           "floor_c2": D0._color(rng_env, (0.0, 0.5), (0.15, 0.6)), "floor_scale": float(rng_env.uniform(0.3, 1.5)),
           "sky_hor": D0._color(rng_env, (0.0, 0.25), (0.7, 1.0)), "sky_top": D0._color(rng_env, (0.2, 0.8), (0.3, 0.9)),
           "sky_strength": float(rng_env.uniform(0.6, 1.3)), "sun_elev": float(rng_env.uniform(25, 70)),
           "sun_azim": float(rng_env.uniform(0, 360)), "sun_energy": float(rng_env.uniform(2.0, 5.0)),
           "wall": bool(rng_env.uniform() < 0.45), "wall_y": float(rng_env.uniform(15, 30)), "wall_h": float(rng_env.uniform(3, 10)),
           "wall_tex": str(rng_env.choice(["brick", "noise", "plain"])), "wall_c1": D0._color(rng_env, (0.0, 0.5), (0.25, 0.8)),
           "wall_c2": D0._color(rng_env, (0.0, 0.5), (0.25, 0.8)), "exposure": float(rng_env.uniform(-0.3, 0.3))}
    return {"cam": cam_p, "objects": objs, "panels": panels, "props": props, "env": env, "shadows": bool(cfg["objects"]["shadows"])}


def sample_clip(cfg: dict, seed: int, idx: int, max_tries: int = 300) -> dict:
    """clip 하나의 설계. 같은 (설정, seed, idx) 는 같은 장면."""
    T, fps = int(round(cfg["seconds"] * cfg["fps"])), float(cfg["fps"])
    rng = np.random.default_rng([seed, idx])
    for attempt in range(max_tries):
        spec = _try_clip(rng, cfg, T, fps)
        if spec is not None:
            spec.update(seed=int(seed), idx=int(idx), attempt=attempt, T=T, fps=fps, size=int(cfg["size"]),
                        version=cfg["name"], split=cfg.get("_split"))
            return spec
    raise RuntimeError(f"clip {idx}: 설계 실패")


def target_state(spec: dict):
    """설계 근사: target 중심 픽셀 (T,2) · 가려진 비율 (T,) (판 · 화면 밖; 물체끼리 · 소품 가림은 안 셈)."""
    cam = D0.Cam(spec["cam"]["h"], spec["cam"]["pitch"], spec["cam"]["lens"], size=spec["size"])
    o, T, fps = spec["objects"][0], spec["T"], spec["fps"]
    pos, _, R = track(o, T, fps)
    E = D0.extent_pts(o["shape"], o["s"], pos)
    hid = np.zeros(E.shape[:2], bool)
    for p in spec["panels"]:
        hid |= panel_hides(cam, p, T, fps, E)
    u, v, _ = cam.to_px(E)
    off = (u < 0) | (u >= cam.size) | (v < 0) | (v >= cam.size)
    ctr = pos + np.array([0, 0, 0.5 * o["s"] * DIMS[o["shape"]][2]])     # 중심 = 바닥면 중심 + h ẑ (물체는 자기 중심 둘레로 돈다)
    cu, cv, _ = cam.to_px(ctr)
    return np.c_[cu, cv], (hid | off).mean(1)


def clip_id(spec: dict) -> str:
    return f"{spec['split'] or 's'}{spec['seed']}_{spec['idx']:06d}"
