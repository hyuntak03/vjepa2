#!/usr/bin/env python3
"""BlenderTrack v2 설계 — 축 다섯 개를 교차한 칸 (324) 마다 같은 수의 clip (2026-11-01, 사용자 설계).

창 = 20 fps × 64 장, 간격 2 로 32 샘플 (문맥 16 + 미래 16 = 블록 8 + 8). 샘플 i = 프레임 2i, 시각 i/10 초.
축 (configs/v2.yaml 의 cells)
  1 timing  early · mid · late — 물체가 가려지는 때 (샘플 기준, timing_ok 참고)
  2 form    fixed (제자리 판 · 상자 · 가로대 · 지붕 · 여닫는 판) · moving (옆으로 미끄러지는 판) · object (움직이는 다른 물체가 가림, 크기 ≤ 2 배)
  3×5 motion (운동 × 장소, 물리적으로 맞는 12 칸)
        air_accel (자유낙하) · air_reverse (위로 던짐 → 꼭대기 → 낙하) · air_bounce (바닥에 튕김)
        flat_static · flat_const (등속) · flat_decel (마찰 감속) · flat_wall (벽에 튕김)
        ramp_static (경사에 정지) · ramp_decel (밀려 올라가며 마찰로 멈춤) · ramp_accel (굴러 내려감) ·
        ramp_reverse (굴러 올라갔다 내려옴) · ramp_wall (굴러 내려가 벽에 튕김)
  4 camera  front (눈높이, pitch −2…−14°) · oblique (−20…−32°) · high (−40…−55°)
  + count   방해 물체 (움직이는 다른 물체) 0 · 1 · 2–3 — 칸 안에서 균등. 가리는 물체 (form object) 는 세지 않는다.
세계: g = cfg.g (축소 세계, 기본 3.5 m/s², VOBench gravity 3.75 와 비슷) · 미터 · 초 · 카메라 (0, 0, h) 에서 +y.
물체 위치 pos = "회전 안 했을 때의 바닥면 중심" (= 중심 − h ẑ), 렌더는 tracks() 가 자기 중심 둘레 회전으로 바꾼다 (v1 과 같은 약속).
가림 판정은 카메라 → 물체 점 선분이 상자 (판 · 상자 · 가로대 · 지붕 · 기둥 · 물체 근사) 를 지나는지 (어느 카메라 각도에서도 같은 식).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D0  # noqa: E402  (Cam · DIMS · 색 · 판 모양)
import gen_design as G1  # noqa: E402  (load_cfg · _axis_angle · _rz · 판 가로 위치)

DIMS = D0.DIMS
load_cfg = G1.load_cfg
NS = 32                      # 샘플
STRIDE = 2


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


# ----------------------------------------------------------------------------- 칸
def cells(cfg):
    C = cfg["cells"]
    return [dict(timing=a, form=b, motion=c, camera=d) for a in C["timing"] for b in C["form"] for c in C["motion"] for d in C["camera"]]


def cell_of(cfg, idx):
    """clip 번호 → (칸, 방해 물체 수준). 번호 순서대로 칸을 돌아 칸마다 정확히 같은 수, 칸 안에서 수준도 균등."""
    cs = cells(cfg)
    lv = cfg["cells"]["count"]
    return cs[idx % len(cs)], lv[(idx // len(cs)) % len(lv)]


# ----------------------------------------------------------------------------- 상자 가림 (선분 · 상자 교차)
def box_hides(C, P, ctr, half, yaw=0.0):
    """카메라 C (3,) → 점 P (T, n, 3) 선분이 상자 (중심 ctr (T, 3) 또는 (3,) · 반치수 half (3,) · z 축 회전 yaw) 를 지나나 → (T, n) bool."""
    ctr = np.asarray(ctr, float)
    if ctr.ndim == 1:
        ctr = np.broadcast_to(ctr, (P.shape[0], 3))
    c, s = math.cos(yaw), math.sin(yaw)
    Rt = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])                 # 세계 → 상자 (z 축 −yaw)
    o = (C[None] - ctr) @ Rt.T                                           # (T, 3)
    q = (P - ctr[:, None]) @ Rt.T                                        # (T, n, 3)
    d = q - o[:, None]
    h = np.asarray(half, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-h - o[:, None]) / d
        t2 = (h - o[:, None]) / d
    inside = np.abs(o[:, None]) <= h                                     # d = 0 인 축: 카메라가 그 판 사이면 늘 통과
    lo = np.where(d == 0, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    hi = np.where(d == 0, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    tmin, tmax = lo.max(-1), hi.min(-1)
    return (tmax >= np.maximum(tmin, 0.0)) & (tmin < 1.0 - 1e-6) & (tmax > 1e-6)


# ----------------------------------------------------------------------------- 물체 모양 · 점
def obj_half(o, k=1.0):
    return 0.5 * o["s"] * np.array(DIMS[o["shape"]], float) * k


def extent_pts(o, center, R):
    """물체 표면 점 (T, n, 3): 상자 꼭짓점 8 · 면 중심 6 · 중심 1 (구는 축 · 대각 방향 14 + 중심). R (T, 3, 3) 로 돌린다."""
    h = obj_half(o)
    if o["shape"] == "sphere":
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        dirs += [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
        loc = np.array([np.array(v, float) / np.linalg.norm(v) * h[0] for v in dirs] + [(0, 0, 0)])
    else:
        cor = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
        fac = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        loc = np.array([np.array(v, float) * h for v in cor + fac] + [(0, 0, 0)])
    return center[:, None] + np.einsum("tij,nj->tni", R, loc)


# ----------------------------------------------------------------------------- 운동 (12 칸)
def _toss(t, z0, vz0, e, g, v_stop=0.4):
    """수직: z0 에서 vz0 로 출발, 바닥 (z = 0) 에 닿으면 반발 e 로 튕김 (e = 0 이면 그 자리에 멈춤). → z (T,), 마지막 착지 시각."""
    z = np.zeros_like(t)
    ts, zs, vs = 0.0, z0, vz0
    segs = []
    for _ in range(40):
        disc = vs * vs + 2 * g * zs
        tau = (vs + math.sqrt(max(disc, 0.0))) / g
        segs.append((ts, zs, vs, ts + tau))
        vh = math.sqrt(max(disc, 0.0))
        ts, zs, vs = ts + tau, 0.0, e * vh
        if vs < v_stop or e <= 0:
            break
    for a, za, va, b in segs:
        m = (t >= a) & (t < b)
        tt = t[m] - a
        z[m] = np.maximum(za + va * tt - 0.5 * g * tt * tt, 0.0)
    z[t >= ts] = 0.0
    return z, ts, segs


def _roll_R(path_xy, r, Ry):
    """평면 위 구 굴림 회전: 진행 방향에 수직인 수평축, 각 = 이동 거리 / r (방향 바뀌면 그 구간 축도 바뀜)."""
    T = len(path_xy)
    R = np.repeat(np.eye(3)[None], T, 0)
    acc = np.eye(3)
    for i in range(1, T):
        dv = path_xy[i] - path_xy[i - 1]
        dd = float(np.linalg.norm(dv))
        if dd > 1e-9:
            ax = np.array([-dv[1], dv[0], 0.0]) / dd
            acc = G1._axis_angle(ax, np.array([dd / r]))[0] @ acc
        R[i] = acc
    return R @ Ry


def track(o, T, fps, g):
    """물체 하나 → pos (T, 3) 바닥면 중심 (회전 전) · yaw (T,) · R (T, 3, 3). 운동 종류는 o['motion']['kind']."""
    m, t = o["motion"], np.arange(T) / fps
    k = m["kind"]
    r = 0.5 * o["s"] * DIMS[o["shape"]][2]
    yaw = m.get("yaw0", 0.0) + m.get("omega", 0.0) * t
    Ry = np.stack([G1._rz(a) for a in yaw])
    x0, y0 = m["p0"]
    sph = o["shape"] == "sphere"
    if k in ("static", "flat_static"):
        pos = np.tile([x0, y0, 0.0], (T, 1)); R = Ry
    elif k in ("slide", "flat_const", "flat_decel", "flat_wall"):
        vx, vy = m["v"]; mu = m.get("mu", 0.0)
        sp = math.hypot(vx, vy)
        if k == "flat_decel" and sp > 0:
            dec = mu * g; ts = sp / dec
            tt = np.minimum(t, ts)
            dist = sp * tt - 0.5 * dec * tt * tt
            xy = np.c_[x0 + vx / sp * dist, y0 + vy / sp * dist]
        else:
            xy = np.c_[x0 + vx * t, y0 + vy * t]
        if k == "flat_wall":                                       # 벽 x = xw 에서 반사 (가로 성분만, 반발 e)
            tc = m["t_contact"]; aft = t > tc
            xc = x0 + vx * tc
            xy[aft, 0] = xc - m["e"] * vx * (t[aft] - tc)
            xy[aft, 1] = y0 + vy * tc + m["e"] * vy * (t[aft] - tc)
        pos = np.c_[xy, np.zeros(T)]
        R = _roll_R(xy, r, Ry) if sph else Ry
    elif k.startswith("air_"):
        z, t_land, _ = _toss(t, m["z0"], m["vz0"], m["e"], g)
        vx, vy = m["v"]
        tl = min(t_land, t[-1])
        xy = np.c_[x0 + vx * np.minimum(t, tl), y0 + vy * np.minimum(t, tl)]
        if t_land < t[-1]:                                         # 마지막 착지 뒤: 수평으로 미끄러지며 마찰로 멈춤
            sp = math.hypot(vx, vy)
            if sp > 1e-9:
                dec = m.get("mu", 0.3) * g
                tt = np.clip(t - t_land, 0, sp / dec)
                d = sp * tt - 0.5 * dec * tt * tt
                xy = xy + np.c_[vx / sp * d, vy / sp * d] * (t > t_land)[:, None]
        pos = np.c_[xy, z]
        R = Ry
    elif k.startswith("ramp_"):
        th, sx = math.radians(m["theta_deg"]), m["sx"]
        hz = r if sph else 0.5 * o["s"] * DIMS[o["shape"]][2]
        if k in ("ramp_static", "ramp_decel"):                     # 상자 (구 아님): 경사에 붙어 미끄러짐 · 정지, 경사만큼 기울임
            s = np.full(T, m["s0"])
            if k == "ramp_decel":
                dec = g * (math.sin(th) + m["mu"] * math.cos(th)); ts = m["v0"] / dec
                tt = np.minimum(t, ts)
                s = m["s0"] + m["v0"] * tt - 0.5 * dec * tt * tt
            n = np.array([-sx * math.sin(th), 0.0, math.cos(th)])
            surf = np.c_[m["x_low"] + sx * s * math.cos(th), np.full(T, m["y"]), s * math.sin(th)]
            ctr = surf + hz * n
            Rt = G1._axis_angle([0, 1, 0], np.full(T, -sx * th))
            R = Rt @ Ry
            pos = ctr - np.array([0, 0, hz])
        else:                                                      # 구: 굴러 오르내림 (a = 5/7 g sinθ). 아래 끝 오목 모서리에서 바닥에 닿는다
            a = -5.0 / 7.0 * g * math.sin(th)
            s = m["s0"] + m["v0"] * t + 0.5 * a * t * t
            s_c = r * math.tan(th / 2)                             # 중심 높이가 r 이 되는 경사 거리 = 바닥에 닿는 자리
            disc = m["v0"] ** 2 - 2 * a * (m["s0"] - s_c)
            tb = (-m["v0"] - math.sqrt(max(disc, 0.0))) / a
            sc = np.maximum(s, s_c)
            x = m["x_low"] + sx * sc * math.cos(th) - sx * r * math.sin(th)
            z = sc * math.sin(th) + r * math.cos(th) - r
            ang = sx * (s - m["s0"]) / r
            if tb < t[-1]:
                vb = abs(m["v0"] + a * tb) * math.cos(th)          # 모서리 충돌: 세로 성분이 사라지고 가로만 남는다
                aft = t > tb
                xb = m["x_low"] + sx * s_c * math.cos(th) - sx * r * math.sin(th)
                ang_b = sx * (s_c - m["s0"]) / r
                dec = m.get("mu_g", 0.03) * g
                tt = t[aft] - tb
                if k == "ramp_wall":                                # 바닥에서 벽 (중심 x = x_wall_c) 에 닿으면 반발 e 로 되돌아옴
                    dist_w = abs(m["x_wall_c"] - xb)
                    disc2 = vb * vb - 2 * dec * dist_w
                    if disc2 <= 0:
                        g1 = np.minimum(tt, vb / dec); gd = vb * g1 - 0.5 * dec * g1 * g1
                        x[aft] = xb - sx * gd; angd = gd
                    else:
                        tw = (vb - math.sqrt(disc2)) / dec; vw = math.sqrt(disc2) * m["e"]
                        pre = tt <= tw
                        g1 = vb * tt - 0.5 * dec * tt * tt
                        t2 = np.clip(tt - tw, 0, vw / dec)
                        back = vw * t2 - 0.5 * dec * t2 * t2
                        x[aft] = np.where(pre, xb - sx * g1, xb - sx * dist_w + sx * back)
                        angd = np.where(pre, g1, dist_w - back)
                    z[aft] = 0.0
                    ang[aft] = ang_b - sx * angd / r
                else:
                    g1 = np.minimum(tt, vb / dec)
                    gd = vb * g1 - 0.5 * dec * g1 * g1
                    x[aft] = xb - sx * gd; z[aft] = 0.0
                    ang[aft] = ang_b - sx * gd / r
            pos = np.c_[x, np.full(T, m["y"]), z]
            R = G1._axis_angle([0, 1, 0], ang) @ Ry
    else:
        raise ValueError(k)
    return pos, yaw, R


def center_path(o, pos, R):
    """회전 반영 중심 (T, 3): 자기 중심 둘레로 돈다 → 중심 = pos + h ẑ (경사 위 상자는 pos 가 이미 중심 − h ẑ)."""
    return pos + np.array([0.0, 0.0, 0.5 * o["s"] * DIMS[o["shape"]][2]])


def tracks(spec):
    """render.Scene.track_fn: 물체마다 (메시 원점 위치, yaw, R) · 판마다 각도 · 판마다 가로 위치 (v1 tracks 와 같은 약속)."""
    T, fps, g = spec["T"], spec["fps"], spec["g"]
    out = []
    for o in spec["objects"]:
        pos, yaw, R = track(o, T, fps, g)
        hz = np.array([0.0, 0.0, 0.5 * o["s"] * DIMS[o["shape"]][2]])
        out.append((pos + hz - np.einsum("tij,j->ti", R, hz), yaw, R))
    return out, [G1.panel_ang(p, T, fps) for p in spec["panels"]], [G1.panel_x(p, T, fps) for p in spec["panels"]]


# ----------------------------------------------------------------------------- 카메라 · 자리
def sample_cam(rng, cfg, level):
    c = cfg["camera"][level]
    return {"h": _u(rng, c["h"]), "pitch": _u(rng, c["pitch"]), "lens": _u(rng, c["lens"]), "level": level}


def ground_center(cam):
    """화면 가운데 광선이 바닥과 만나는 깊이 y (pitch 가 얕으면 None)."""
    d = cam.ray(np.array(cam.c), np.array(cam.c))
    if d[2] > -1e-3:
        return None
    return float(cam.C[1] + (-cam.C[2] / d[2]) * d[1])


def depth_range(cam, level, cfg):
    if level == "front":
        return tuple(cfg["camera"]["front"]["depth"])
    gc = ground_center(cam)
    return (max(2.5, 0.65 * gc), 1.35 * gc)


# ----------------------------------------------------------------------------- 운동 뽑기
def _speed(rng, cfg, cam, y):
    return _u(rng, cfg["motion"]["speed_frac"]) * 2 * cam.half_width(max(y - cam.C[1], 1.0))


def sample_motion(rng, cfg, cam, level, kind, T, fps, g):
    """target 운동 · 모양 · 크기 · 소품 (경사로 · 벽). 화면 안 · 크기 · 경사로 보임 검사는 sample_target 이 한다."""
    M, O = cfg["motion"], cfg["objects"]
    lo, hi = depth_range(cam, level, cfg)
    y = _u(rng, (lo, hi))
    if kind in ("ramp_accel", "ramp_reverse", "ramp_wall"):
        shape = "sphere"
    elif kind in ("ramp_static", "ramp_decel"):
        shape = str(rng.choice([s for s in O["shapes"] if s in ("cube", "cylinder", "pyramid", "cone")]))
    elif kind in ("flat_const", "flat_decel", "flat_wall"):
        shape = str(rng.choice(O["shapes"]))
    else:
        shape = str(rng.choice(O["shapes"]))
    s = float(np.clip(_u(rng, O["px"]) * (y - cam.C[1]) / cam.f, 0.2, 2.5))
    o = {"shape": shape, "s": s}
    sgn = float(rng.choice([-1.0, 1.0]))
    m = {"kind": kind, "p0": [0.0, y], "yaw0": float(rng.uniform(0, 2 * np.pi))}
    spin = (shape != "sphere") and rng.uniform() < M["spin_p"] and kind in ("flat_const", "flat_decel", "flat_wall", "air_accel", "air_reverse", "air_bounce")
    m["omega"] = float(math.radians(_u(rng, M["spin_deg_s"]))) if spin else 0.0
    props = []
    t_end = (T - 1) / fps
    if kind == "flat_static":
        pass
    elif kind in ("flat_const", "flat_decel", "flat_wall"):
        v = _speed(rng, cfg, cam, y)
        psi = math.radians(_u(rng, M["heading_deg"]))
        m["v"] = [sgn * v * math.cos(psi), v * math.sin(psi)]
        if kind == "flat_decel":
            m["mu"] = v / (g * _u(rng, M["decel_stop_s"]))          # 멈추는 시각을 창 안 · 밖에 고르게
        if kind == "flat_wall":
            m["v"][1] *= 0.3
            m["e"] = _u(rng, M["restitution"]); m["t_contact"] = _u(rng, M["wall_t"]) * t_end
    elif kind.startswith("air_"):
        hv = _u(rng, M["air_hspeed_frac"]) * 2 * cam.half_width(max(y - cam.C[1], 1.0))
        m["v"] = [sgn * hv, 0.0]; m["mu"] = 0.3
        if kind == "air_accel":
            m["z0"], m["vz0"], m["e"] = _u(rng, M["air_z0"]), -_u(rng, (0.0, 1.0)), 0.0
        elif kind == "air_reverse":
            m["z0"], m["vz0"], m["e"] = _u(rng, (0.3, 1.2)), _u(rng, M["toss_vz"]), 0.0   # 11-01: 첫 장부터 공중 (바닥에서 튀어 오르지 않는다)
        else:
            m["z0"], m["vz0"], m["e"] = _u(rng, (0.5, 3.0)), _u(rng, (-1.0, 2.0)), _u(rng, M["restitution"])
    else:                                                          # 경사로: 경사는 x 방향으로 sx 쪽이 높다
        th = _u(rng, M["ramp_deg"]); thr = math.radians(th)
        L = float(min(_u(rng, M["ramp_len"]), M["ramp_max_height"] / math.sin(thr)))
        m.update(theta_deg=th, sx=sgn, L=L, x_low=0.0, y=y)
        if kind == "ramp_static":
            m["s0"] = _u(rng, (0.25, 0.8)) * L
        elif kind == "ramp_decel":
            m["mu"] = math.tan(thr) + _u(rng, (0.05, 0.4))
            m["s0"] = _u(rng, (0.05, 0.3)) * L
            dec = g * (math.sin(thr) + m["mu"] * math.cos(thr))
            stop = _u(rng, (0.5, 0.85)) * L
            m["v0"] = math.sqrt(2 * dec * max(stop - m["s0"], 0.05))
        else:
            a = 5.0 / 7.0 * g * math.sin(thr)
            if kind in ("ramp_accel", "ramp_wall"):
                m["s0"] = _u(rng, (0.5, 0.95)) * L; m["v0"] = -_u(rng, (0.0, 0.4)) * math.sqrt(2 * a * L)
            else:                                                  # 꼭대기 시각을 창 안 (0.15–0.75 × 창 길이) 에서 고른다
                t_ap = _u(rng, (0.15, 0.75)) * t_end
                m["v0"] = a * t_ap
                rise = 0.5 * a * t_ap * t_ap
                m["s0"] = max(_u(rng, (0.02, 0.25)) * L, 0.5 * s * math.tan(thr / 2) * 2 + 0.05)
                if m["s0"] + rise > 0.92 * L:
                    m["s0"] = max(0.02 * L, 0.92 * L - rise)
            m["mu_g"] = _u(rng, (0.01, 0.06))
            if kind == "ramp_wall":
                m["e"] = _u(rng, M["restitution"])
                r = 0.5 * s
                xb = -sgn * r * math.sin(thr)
                m["x_wall_c"] = xb - sgn * _u(rng, M["ramp_wall_dist"])      # 벽에 닿는 중심 x (아래 끝에서 내려온 쪽)
        col = D0._color(rng, (0.0, 0.5), (0.3, 0.85))
        r_o = 0.5 * s * DIMS[shape][0]
        Wd, thick = max(0.8, 2.6 * r_o), 0.12
        phi = -sgn * thr
        top_mid = np.array([sgn * (L / 2) * math.cos(thr), y, (L / 2) * math.sin(thr)])
        loc = top_mid - np.array([thick * math.sin(phi), 0.0, thick * math.cos(phi)])
        # 11-01 (사용자): 판 하나는 떠 있는 것처럼 보인다 → 렌더는 직각삼각형 쐐기 (빗변 = 이 판의 윗면, 밑면 = 바닥, 높은 끝 = 세운 면).
        #   설계 쪽 표현은 판 그대로 (loc · rot · scale) 두고 ramp_wedge() 가 쐐기를 계산한다 — 운동이 닿는 면은 같다
        props.append({"role": "ramp", "solid": "wedge", "loc": loc.tolist(), "rot": [0.0, phi, 0.0], "scale": [L, Wd, thick], "color": col})
        if kind == "ramp_wall":
            r = 0.5 * s
            xw = m["x_wall_c"] - sgn * (r + 0.075)
            props.append({"role": "wall", "loc": [xw, y, 0.0], "rot": [0, 0, 0], "scale": [0.15, max(1.5, 4 * r), max(0.6, 2.5 * r)], "color": col})
    if kind == "flat_wall":
        r = 0.5 * s * DIMS[shape][0]
        xc = m["p0"][0] + m["v"][0] * m["t_contact"]
        sg = 1.0 if m["v"][0] > 0 else -1.0
        props.append({"role": "wall", "loc": [xc + sg * (r + 0.075), y + m["v"][1] * m["t_contact"], 0.0], "rot": [0, 0, 0],
                      "scale": [0.15, max(1.5, 4 * r), max(0.6, 2.0 * o["s"] * DIMS[shape][2])], "color": D0._color(rng, (0.0, 0.5), (0.3, 0.85))})
    o["motion"] = m
    o["props"] = props
    return o


def shift_x(o, dx):
    m = o["motion"]
    if m["kind"].startswith("ramp_"):
        m["x_low"] += dx
        if "x_wall_c" in m:
            m["x_wall_c"] += dx
    else:
        m["p0"][0] += dx
    for p in o["props"]:
        p["loc"][0] += dx


def sample_target(rng, cfg, cam, level, kind, T, fps, g, tries=60):
    """화면 안 (샘플의 90 % 이상) · 크기 · 경사로가 화면 안 · 사건 (벽 · 꼭대기 · 튕김) 이 창 안인 target."""
    O = cfg["objects"]
    for _ in range(tries):
        o = sample_motion(rng, cfg, cam, level, kind, T, fps, g)
        pos, _, R = track(o, T, fps, g)
        ctr = center_path(o, pos, R)
        mid = ctr[T // 2]
        x_m = rng.uniform(-0.45, 0.45) * cam.half_width(max(mid[1] - cam.C[1], 1.0))
        shift_x(o, x_m - mid[0])
        pos, _, R = track(o, T, fps, g)
        ctr = center_path(o, pos, R)
        samp = ctr[::STRIDE]
        u, v, dep = cam.to_px(samp)
        inn = (u > 4) & (u < cam.size - 4) & (v > 4) & (v < cam.size - 4) & (dep > 0.5)
        if inn.mean() < 0.9:
            continue
        hs = obj_half(o)
        sz = cam.f * 2 * math.sqrt(max(hs[0], hs[1]) * hs[2]) / np.maximum(dep, 1e-3)
        if np.median(sz[inn]) < O["min_px_median"] or sz[inn].min() < O["min_px"] or np.median(sz[inn]) > O["max_px_median"]:
            continue
        if not event_in_window(o, pos, ctr, T, fps):
            continue
        ok = True
        for p in o["props"]:                                       # 경사로 · 벽이 화면에 보여야 운동이 읽힌다
            L = np.array(p["scale"]); c0 = np.array(p["loc"])
            pts = c0[None] + np.array([[a * L[0] / 2, 0, b * L[2]] for a in (-0.8, 0, 0.8) for b in (0, 1)])
            if p["role"] == "ramp":
                th = p["rot"][1]
                pts = c0[None] + np.array([[a * L[0] / 2 * math.cos(th), 0, -a * L[0] / 2 * math.sin(th)] for a in (-0.8, 0, 0.8)])
            uu, vv, dd = cam.to_px(pts)
            if (((uu > 0) & (uu < cam.size) & (vv > 0) & (vv < cam.size) & (dd > 0.3)).mean()) < 0.6:
                ok = False
        if not ok:
            continue
        o["role"] = "target"
        o.update(color=D0._color(rng), rough=float(rng.uniform(0.3, 0.9)), metal=float(rng.uniform(0.0, 0.6)) if rng.uniform() < 0.2 else 0.0)
        return o, pos, R, ctr
    return None


def event_in_window(o, pos, ctr, T, fps, lo=0.12, hi=0.88):
    """사건이 있는 운동은 그 사건이 창 안 (lo–hi × 창 길이) 에서 일어나야 그 칸의 운동이 보인다.
    벽 (ramp_wall · flat_wall) = 가로 속도 부호가 바뀜 · 튕김 (air_bounce) = 바닥에서 세로 속도 − → + · 반전 = 꼭대기 · 가속 = 창 안에서 속력이 늘어남."""
    k = o["motion"]["kind"]
    v = np.diff(ctr, axis=0) * fps
    a_, b_ = int(lo * (T - 1)), int(hi * (T - 1))
    if k in ("ramp_wall", "flat_wall"):
        sx = np.sign(v[:, 0]); ch = np.nonzero((sx[1:] * sx[:-1]) < 0)[0]
        if not (len(ch) >= 1 and a_ <= ch[0] <= b_):
            return False
        if k == "ramp_wall":                                       # 튕긴 뒤 빗면 모서리까지 되돌아오면 (빗면을 다시 오르는 운동은 이 칸 밖) 버린다
            m = o["motion"]; th = math.radians(m["theta_deg"]); r = 0.5 * o["s"]
            xb = m["x_low"] + m["sx"] * r * math.tan(th / 2) * math.cos(th) - m["sx"] * r * math.sin(th)
            after = ctr[ch[0] + 1:, 0]
            if len(after) and (m["sx"] * (after - xb) > -0.05).any():
                return False
        return True
    if k == "air_bounce":
        vz = v[:, 2]; ch = np.nonzero((vz[:-1] < -1e-3) & (vz[1:] > 1e-3))[0]
        return len(ch) >= 1 and a_ <= ch[0] <= b_
    if k in ("air_reverse", "ramp_reverse"):
        if k == "air_reverse":
            comp = v[:, 2]
        else:
            th, sx = math.radians(o["motion"]["theta_deg"]), o["motion"]["sx"]
            comp = v @ np.array([sx * math.cos(th), 0, math.sin(th)])
        ch = np.nonzero((comp[:-1] > 1e-3) & (comp[1:] < -1e-3))[0]
        return len(ch) >= 1 and a_ <= ch[0] <= b_
    if k in ("air_accel", "ramp_accel"):
        sp = np.linalg.norm(v, axis=1)
        return sp[b_] > 1.3 * sp[a_] + 1e-3 or (sp[: b_].max() > 1.3 * sp[a_] + 1e-3)
    return True


# ----------------------------------------------------------------------------- 가림 타이밍
def target_hidden(cam, occ, P):
    """occ: 가림 상자 목록 [(ctr (T,3)|(3,), half (3,), yaw, alive (T,) bool|None)] → 가려진 점 비율 (T,)."""
    hid = np.zeros(P.shape[:2], bool)
    for ctr, half, yaw, alive in occ:
        h = box_hides(cam.C, P, ctr, half, yaw)
        if alive is not None:
            h &= alive[:, None]
        hid |= h
    return hid.mean(1)


def off_frame(cam, P):
    u, v, d = cam.to_px(P)
    return ((u < 0) | (u >= cam.size) | (v < 0) | (v >= cam.size) | (d < 0.2)).mean(1)


def timing_ok(timing, hid_s, off_s, W):
    """샘플 (32) 의 가려진 비율 hid_s · 화면 밖 비율 off_s → (맞나, [s, e]).
    가림 사건 = 완전히 가려진 (hid ≥ 0.95) 샘플이 이어진 구간 하나 [s, e]. 그 앞뒤로 반쯤 가려진 샘플 (들어가고 나오는 중) 은 허용하되
    사건 앞 · 뒤에 보이는 샘플 (hid < 0.15 · 화면 밖 < 0.3) 이 있어야 하고, 사건 밖에서는 다시 완전히 가려지지 않는다 (가림 사건 하나).
    가리기 전에는 반 이상 보인 샘플이 있으면 된다 (판 뒤로 들어가는 동안 반쯤 가려지는 것은 물리).
    early: s ∈ early.start, e ≤ early.end_max, vis_after (문맥 끝) 완전히 보임 · half_after 반 이상 보임 · mid: s ∈ mid.start, e ≤ mid.end_max, vis_after 보임 ·
    late: 15 가 완전히 가려짐, s ∈ late.start, e ∈ late.end."""
    full = hid_s >= 0.95
    if not full.any():
        return False, None
    idx = np.nonzero(full)[0]
    s, e = int(idx[0]), int(idx[-1])
    if not full[s:e + 1].all():
        return False, (s, e)
    vis = (hid_s < 0.15) & (off_s < 0.3)                       # 완전히 보임
    half = (hid_s < 0.5) & (off_s < 0.3)                       # 반 이상 보임
    if not half[:s].any() or not vis[e + 1:].any():            # 가리기 전에 반 이상 본 적이 있고, 뒤에 다시 완전히 보인다
        return False, (s, e)
    # 사건과 이어진 반쯤 가려진 구간 밖에서는 반 이상 가려지면 안 된다 (두 번째 가림 사건 금지)
    a = s
    while a > 0 and not vis[a - 1]:
        a -= 1
    b = e
    while b < len(vis) - 1 and not vis[b + 1]:
        b += 1
    out = np.ones(len(vis), bool); out[a:b + 1] = False
    if (hid_s[out] >= 0.5).any():
        return False, (s, e)
    w = W[timing]
    if not (w["start"][0] <= s <= w["start"][1]):
        return False, (s, e)
    if timing == "late":
        if not full[15] or not (w["end"][0] <= e <= w["end"][1]):
            return False, (s, e)
    else:                                                      # 문맥 끝 쪽: vis_after 는 완전히 보임, half_after 는 반 이상 보임
        if e > w["end_max"] or not vis[w["vis_after"][0]:w["vis_after"][1] + 1].all():
            return False, (s, e)
        if "half_after" in w and not half[w["half_after"][0]:w["half_after"][1] + 1].all():
            return False, (s, e)
    if (off_s >= 0.3).mean() > 0.15:
        return False, (s, e)
    return True, (s, e)


def want_interval(rng, timing, W):
    w = W[timing]
    s = int(rng.integers(w["aim_start"][0], w["aim_start"][1] + 1))
    if timing == "late":
        e = int(rng.integers(max(w["aim_end"][0], s + 2), w["aim_end"][1] + 1))
    else:
        e = int(min(s + rng.integers(w["aim_len"][0], w["aim_len"][1] + 1) - 1, w["end_max"]))
    return s, e


# ----------------------------------------------------------------------------- 가림막 만들기 (form 별)
def _plane_proj(C, P, y_o):
    """점 P (..., 3) 를 카메라에서 평면 y = y_o 로 투영 → (x, z)."""
    tpar = (y_o - C[1]) / (P[..., 1] - C[1])
    return C[0] + tpar * (P[..., 0] - C[0]), C[2] + tpar * (P[..., 2] - C[2])


def _hplane_proj(C, P, z_r):
    tpar = (z_r - C[2]) / (P[..., 2] - C[2])
    return C[0] + tpar * (P[..., 0] - C[0]), C[1] + tpar * (P[..., 1] - C[1])


def pick_fixed_sub(rng, cfg, kind, level, exclude=()):
    """고정 가림막 하위 종류를 clip 마다 먼저 정한다 (시도마다 다시 뽑으면 잘 되는 여닫는 판으로 쏠린다).
    서 있는 물체 (flat_static · ramp_static) 는 여닫는 판만 가렸다 보여 줄 수 있다. 가로대 = 공중 운동만 · 지붕 = 비스듬 · 위 카메라만."""
    if kind in ("flat_static", "ramp_static"):
        return "trapdoor"
    W = dict(cfg["occluders"]["fixed_sub_w"])
    if not kind.startswith("air_"):
        W.pop("beam", None)
    if level == "front":
        W.pop("roof", None)
    for k in exclude:
        W.pop(k, None)
    if not W:
        return "trapdoor"
    ks = list(W); ps = np.array([W[k] for k in ks], float)
    return str(rng.choice(ks, p=ps / ps.sum()))


def make_fixed(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, sub=None):
    """제자리 가림막: 판 · 상자 · 가로대 (공중, 기둥 둘) · 지붕 (위에서 보는 카메라, 기둥 넷) · 여닫는 판.
    가림 구간 프레임 [fs, fe] 의 target 실루엣을 덮도록 크기 · 자리를 정한다 (가림은 기하로 생긴다). → (판 목록, 소품 목록, 하위 종류) 또는 None."""
    P = cfg["occluders"]
    seg = E[fs:fe + 1]
    move = np.ptp(ctr[fs:fe + 1, 0]) + np.ptp(ctr[fs:fe + 1, 2])
    size = tgt["s"]
    air = tgt["motion"]["kind"].startswith("air_")
    col = D0._color(rng, (0.0, 0.7), (0.25, 0.9))
    if sub is None:
        sub = pick_fixed_sub(rng, cfg, tgt["motion"]["kind"], cam_level(cam))
    still = float(np.ptp(ctr[:, 0]) + np.ptp(ctr[:, 2])) < 0.6 * size            # 창 전체에서 거의 안 움직임 → 제자리 가림막으로는 못 보여 준다
    if still and sub != "trapdoor":
        return None
    if sub == "beam" and np.ptp(ctr[:, 2]) <= np.ptp(ctr[:, 0]):                  # 가로대는 세로로 지나가는 물체에만
        return None
    yt = float(ctr[fs:fe + 1, 1].min())
    r_t = 0.5 * size * max(DIMS[tgt["shape"]][0], DIMS[tgt["shape"]][1])
    if sub == "roof":
        top = float(E[:, :, 2].max())
        z_r = top + _u(rng, (0.4, 1.2))
        X, Y = _hplane_proj(cam.C, seg.reshape(-1, 3), z_r)
        mx = 0.05 * size
        x0, x1, y0, y1 = X.min() - mx, X.max() + mx, Y.min() - mx, Y.max() + mx
        th = 0.08
        roof = {"role": "occ_roof", "loc": [(x0 + x1) / 2, (y0 + y1) / 2, z_r], "rot": [0, 0, 0], "scale": [x1 - x0, y1 - y0, th], "color": col}
        legs = [{"role": "occ_post", "loc": [x, yy, 0.0], "rot": [0, 0, 0], "scale": [0.08, 0.08, z_r], "color": col}
                for x in (x0 + 0.05, x1 - 0.05) for yy in (y0 + 0.05, y1 - 0.05)]
        return [], [roof] + legs, sub
    gap = _u(rng, P["gap"])
    y_o = yt - r_t - gap
    if y_o - cam.C[1] < 1.0:
        return None
    X, Z = _plane_proj(cam.C, seg.reshape(-1, 3), y_o)
    mx = 0.04 * size
    x0, x1 = X.min() - mx, X.max() + mx
    if sub == "beam":
        z0, z1 = max(Z.min() - mx, 0.05), Z.max() + mx
        th = _u(rng, (0.06, 0.15))
        beam = {"role": "occ_beam", "loc": [(x0 + x1) / 2, y_o, z0], "rot": [0, 0, 0], "scale": [x1 - x0, th, z1 - z0], "color": col}
        legs = [{"role": "occ_post", "loc": [x, y_o, 0.0], "rot": [0, 0, 0], "scale": [0.08, th, z0], "color": col} for x in (x0 - 0.05, x1 + 0.05)]
        return [], [beam] + legs, sub
    hgt = float(Z.max()) + mx
    if sub == "box":
        dep = _u(rng, (0.3, 1.0)) * size
        box = {"role": "occ_box", "loc": [(x0 + x1) / 2, y_o - dep / 2, 0.0], "rot": [0, 0, 0], "scale": [x1 - x0, dep, hgt], "color": col}
        return [], [box], sub
    pan = {"role": "event", "x": float((x0 + x1) / 2), "y": float(y_o), "w": float(x1 - x0), "h": float(hgt), **D0._panel_style(rng), "mode": "static"}
    if sub == "trapdoor":
        t_end = (T - 1) / fps
        pan.update(mode="trapdoor", rise=float(_u(rng, (0.08, 0.2))), t_up=float(fs / fps), t_down=float(min(t_end, fe / fps)))
    return [pan], [], sub


def cam_level(cam):
    return getattr(cam, "level", "front")


def make_moving(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps):
    """옆으로 미끄러지는 판 (서 있는 채로 가로 이동): target 과 같은 방향 · 반대 · 서 있는 target 을 지나감. 가림 구간에만 덮게 x0 를 고른다."""
    P = cfg["occluders"]
    size = tgt["s"]
    yt = float(ctr[:, 1].min())
    r_t = 0.5 * size * max(DIMS[tgt["shape"]][0], DIMS[tgt["shape"]][1])
    y_o = yt - r_t - _u(rng, P["gap"])
    if y_o - cam.C[1] < 1.0:
        return None
    X, Z = _plane_proj(cam.C, E, y_o)                                # (T, n)
    xmin, xmax = X.min(1), X.max(1)
    t = np.arange(T) / fps
    xc = 0.5 * (xmin + xmax)
    vt = float(np.gradient(xc)[(fs + fe) // 2] * fps)
    hw = cam.half_width(y_o - cam.C[1])
    vmin, vmax = P["slide_speed_frac"][0] * 2 * hw, P["slide_speed_frac"][1] * 2 * hw
    if abs(vt) < P["slide_pass_below_frac"] * 2 * hw:
        vx = float(rng.choice([-1.0, 1.0]) * rng.uniform(vmin, vmax)); rel = "pass"
    elif rng.uniform() < P["slide_same_p"]:
        ratio = _u(rng, P["slide_ratio_fast"]) if rng.uniform() < 0.5 else _u(rng, P["slide_ratio_slow"])
        vx = float(np.sign(vt) * np.clip(ratio * abs(vt), vmin, vmax)); rel = "same"
    else:
        vx = float(-np.sign(vt) * np.clip(_u(rng, P["slide_ratio_opp"]) * abs(vt), vmin, vmax)); rel = "opposite"
    rel_disp = (xc[fs:fe + 1] - vx * t[fs:fe + 1])
    ext = float((xmax - xmin)[fs:fe + 1].max())
    w = ext + float(np.ptp(rel_disp)) + 0.06 * size
    lo = (xmax - vx * t - w / 2)[fs:fe + 1].max()
    hi = (xmin - vx * t + w / 2)[fs:fe + 1].min()
    if lo > hi:
        return None
    x0 = float(0.5 * (lo + hi))
    hgt = float(Z[fs:fe + 1].max()) + 0.05 * size
    pan = {"role": "event", "x": float(x0 + vx * t[(fs + fe) // 2]), "y": float(y_o), "w": float(w), "h": hgt, **D0._panel_style(rng),
           "mode": "slide", "vx": vx, "x0": x0, "rel": rel}
    return [pan], [], f"slide_{rel}"


def make_object_occluder(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, g, props=None):
    """움직이는 다른 물체가 가림 (화면 크기 비 ∈ occluders.object_screen_ratio, 11-01). 바닥에서 미끄러지는 물체 (앞쪽 깊이) ·
    공중을 나는 물체 (첫 장부터 이미 날고 있음, 위에서 보는 카메라) · 경사로 target 이면 같은 쐐기 앞 차선에서 굴러 오르내리는 공.
    가림 구간 가운데에 시선을 지나도록 놓는다. → 물체 dict (role occluder) 또는 None."""
    P = cfg["occluders"]
    size = tgt["s"]
    yt = float(ctr[:, 1].min())
    r_t = 0.5 * size * max(DIMS[tgt["shape"]][0], DIMS[tgt["shape"]][1])
    shape = str(rng.choice(P["object_shapes"]))
    t = np.arange(T) / fps
    fm = (fs + fe) // 2
    # 11-01 (사용자: 가리는 물체가 너무 크다): 크기는 화면 기준으로 정한다. 가리는 물체가 카메라에 더 가까워 실제 크기 비 1–2 가
    #   화면에서 2.3 배 (중앙값) · 3 배 넘음 18 % 로 커졌다 → 화면 크기 비 rs ∈ object_screen_ratio, 실제 크기 = rs × target × (깊이 비) × (모양 보정)
    rs = _u(rng, P["object_screen_ratio"])
    k_t = math.sqrt(max(DIMS[tgt["shape"]][0], DIMS[tgt["shape"]][1]) * DIMS[tgt["shape"]][2])
    k_o = math.sqrt(max(DIMS[shape][0], DIMS[shape][1]) * DIMS[shape][2])
    d_t = float(cam.to_px(ctr[fm:fm + 1])[2][0])
    s_for = lambda d_o: rs * size * (d_o / d_t) * k_t / k_o          # noqa: E731  깊이 d_o 에서 화면 크기 비가 rs 가 되는 실제 크기
    if (props is not None and tgt["motion"]["kind"].startswith("ramp_") and ctr[fm, 2] > 1.3 * r_t
            and rng.uniform() < P.get("object_lane_p", 0.5)):
        return _lane_occluder(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, g, props, rs, k_t, d_t)
    air_mode = rng.uniform() < P["object_air_p"][cam_level(cam)]
    if not air_mode:                                               # 땅 위 물체로는 높이가 모자라는 target (공중 · 경사 위) 이면 공중으로
        s_hi = size * P["object_screen_ratio"][1] * k_t / k_o
        y_try = yt - r_t - P["object_gap"][1] - 0.5 * s_hi * DIMS[shape][1]
        if y_try - cam.C[1] > 1.0:
            _, Zt = _plane_proj(cam.C, E[fs:fe + 1], y_try)
            if float(Zt.max()) > s_hi * DIMS[shape][2] * 0.95:
                air_mode = True
    if air_mode:
        # 공중: 시선 (카메라 → target 중심) 위의 한 점을 꼭대기 근처에서 지나가는 포물선
        lam = _u(rng, (0.35, 0.85))
        pt = cam.C + lam * (ctr[fm] - cam.C)
        s_o = s_for(lam * d_t)
        hz = 0.5 * s_o * DIMS[shape][2]
        if pt[2] - hz < 0.3:
            return None
        # 11-01: 가로 속도는 가림 구간 동안 덮을 수 있는 상대 속도 안에서 (땅 위 물체와 같은 규칙). 전에는 화면 폭 기준으로 빨리 던져
        #   작은 물체 (화면 크기 비 ≤ 1.8) 로는 정지 · 느린 target 을 오래 못 덮었다 (late × 경사 정지 · 감속 칸 설계 실패)
        Xp, Zp = _plane_proj(cam.C, E[fs:fe + 1], pt[1])
        D = max((fe - fs) / fps, 1e-3)
        w_o = s_o * min(DIMS[shape][0], DIMS[shape][1]) * 0.9
        vmax_rel = (w_o - float((Xp.max(1) - Xp.min(1)).max())) / D
        if vmax_rel <= 0:
            return None
        xc, zc = 0.5 * (Xp.min(1) + Xp.max(1)), 0.5 * (Zp.min(1) + Zp.max(1))
        vt, vzt = float(xc[-1] - xc[0]) / D, float(zc[-1] - zc[0]) / D
        vx = vt + float(rng.choice([-1.0, 1.0]) * _u(rng, (0.1, 0.95)) * vmax_rel)
        pt = np.array([float(xc[len(xc) // 2]), pt[1], pt[2]])
        vz_m = vzt + _u(rng, (-0.3, 0.3))
        tm = t[fm]
        z0 = pt[2] - hz - vz_m * tm - 0.5 * g * tm * tm               # z(tm) = 꼭짓점 근처 높이 (바닥면 기준)
        vz0 = vz_m + g * tm
        if z0 < P.get("object_air_z0_min", 0.3):                      # 11-01 (사용자: 갑자기 점프해서 가린다): 첫 장부터 이미 날고 있어야 한다 — 창 안에서 바닥을 박차고 뜨지 않는다
            return None
        t_apex = vz0 / g                                               # 창 안에서 꼭대기를 지나면 '반전', 아니면 내려가기만 ('가속') — 물리 검사의 운동 꼴 이름
        kind_o = "air_reverse" if 0.1 < t_apex < (T - 1) / fps - 0.1 else "air_accel"
        m = {"kind": kind_o, "p0": [float(pt[0] - vx * tm), float(pt[1])], "v": [vx, 0.0], "z0": float(z0), "vz0": float(vz0), "e": 0.0, "mu": 0.3,
             "yaw0": float(rng.uniform(0, 2 * np.pi)), "omega": 0.0}
        o = {"shape": shape, "s": float(s_o), "motion": m, "role": "occluder", "mode": "air"}
    else:
        # 바닥에서 미끄러지는 물체: target 바로 앞 (간격 작게) 에 두고, 가림 구간 동안 덮을 수 있는 상대 속도 안에서 속도를 고른다
        gap_o = _u(rng, P["object_gap"])
        s_o = s_for(d_t)
        for _ in range(3):                                          # 깊이가 크기에 달려 있어 몇 번 되풀이 (수렴 빠름)
            y_o = yt - r_t - gap_o - 0.5 * s_o * DIMS[shape][1]
            d_o = float(cam.to_px(np.array([[ctr[fm, 0], y_o, 0.5 * s_o * DIMS[shape][2]]]))[2][0])
            s_o = s_for(max(d_o, 0.3))
        y_o = yt - r_t - gap_o - 0.5 * s_o * DIMS[shape][1]
        if y_o - cam.C[1] < 1.0:
            return None
        X, Z = _plane_proj(cam.C, E, y_o + 0.5 * s_o * DIMS[shape][1])      # 가리는 물체의 뒷면 평면 (target 쪽)
        xmin, xmax = X.min(1), X.max(1)
        xc = 0.5 * (xmin + xmax)
        if float(Z[fs:fe + 1].max()) > s_o * DIMS[shape][2] * 0.95:       # 높이가 모자라면 못 가린다
            return None
        w_o = s_o * DIMS[shape][0] * 0.9
        ext = float((xmax - xmin)[fs:fe + 1].max())
        D = (fe - fs) / fps
        vmax_rel = (w_o - ext) / max(D, 1e-3)
        if vmax_rel <= 0:
            return None
        vt = float(np.gradient(xc)[fm] * fps)
        hw = cam.half_width(y_o - cam.C[1])
        vmin_abs = P["slide_speed_frac"][0] * 2 * hw
        if abs(vt) < P["slide_pass_below_frac"] * 2 * hw:
            rel = "pass"; vx = float(rng.choice([-1.0, 1.0]) * _u(rng, (min(vmin_abs, 0.95 * vmax_rel), 0.95 * vmax_rel)))
        else:
            vr = _u(rng, (0.3, 0.95)) * vmax_rel * float(rng.choice([-1.0, 1.0]))
            vx = vt + vr
            rel = "same" if np.sign(vx) == np.sign(vt) else "opposite"
        x_at = float(xc[fm])
        m = {"kind": "flat_const", "p0": [x_at - vx * t[fm], float(y_o)], "v": [float(vx), 0.0], "yaw0": 0.0, "omega": 0.0}
        o = {"shape": shape, "s": float(s_o), "motion": m, "role": "occluder", "mode": f"ground_{rel}"}
    # 가림 구간에서 잰 화면 크기 비가 범위 안인가 (공중 물체는 움직이며 깊이가 바뀐다)
    pc, _, Rc = track(o, T, fps, g)
    ratio = occluder_screen_ratio(cam, tgt, ctr, o, center_path(o, pc, Rc), fs, fe)
    lo_r, hi_r = P["object_screen_ratio"]
    if not (0.9 * lo_r <= ratio <= hi_r * 1.08):
        return None
    o.update(color=D0._color(rng), rough=float(rng.uniform(0.3, 0.9)), metal=0.0, screen_ratio=float(ratio))
    return o


def _lane_occluder(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, g, props, rs, k_t, d_t):
    """경사로 target 의 가림: 같은 쐐기의 앞 차선 (카메라 쪽) 에서 공이 굴러 올라가 꼭대기에서 target 앞을 지나고 다시 내려온다 (11-01).
    꼭대기 시각 = 가림 구간 가운데, 꼭대기 자리 = 그 시각 target 의 (앞 차선 평면) 투영 가로 중심. 쐐기는 앞 차선까지 넓힌다 (o['_ramp_prop'])."""
    P = cfg["occluders"]
    m_t = tgt["motion"]
    th, sx, L = math.radians(m_t["theta_deg"]), m_t["sx"], m_t["L"]
    size = tgt["s"]
    r_t = 0.5 * size * max(DIMS[tgt["shape"]][0], DIMS[tgt["shape"]][1])
    t = np.arange(T) / fps
    fm = (fs + fe) // 2
    gap = _u(rng, (0.1, 0.35))
    s_o = rs * size * k_t                                              # 같은 깊이 근처 — 아래에서 깊이 비로 한 번 고친다
    for _ in range(2):
        r = 0.5 * s_o
        y_o = m_t["y"] - r_t - gap - r
        d_o = float(cam.to_px(np.array([[ctr[fm, 0], y_o, ctr[fm, 2]]]))[2][0])
        s_o = rs * size * (d_o / d_t) * k_t
    r = 0.5 * s_o
    y_o = m_t["y"] - r_t - gap - r
    if y_o - cam.C[1] < 1.0:
        return None
    Xp, _ = _plane_proj(cam.C, E[fm], y_o)
    X = 0.5 * (float(Xp.min()) + float(Xp.max()))
    s_ap = ((X - m_t["x_low"]) * sx + r * math.sin(th)) / math.cos(th)   # 공 중심 x = x_low + sx s cosθ − sx r sinθ = X
    a = -5.0 / 7.0 * g * math.sin(th)
    t_ap = float(t[fm])
    s0 = s_ap + 0.5 * a * t_ap * t_ap
    v0 = -a * t_ap
    s_c = r * math.tan(th / 2)
    if s0 < s_c + 0.02 or s_ap > 0.95 * L:
        return None
    m = {"kind": "ramp_reverse", "p0": [float(m_t["x_low"]), float(y_o)], "theta_deg": float(m_t["theta_deg"]), "sx": sx, "L": L,
         "x_low": float(m_t["x_low"]), "y": float(y_o), "s0": float(s0), "v0": float(v0), "mu_g": float(_u(rng, (0.01, 0.06))),
         "yaw0": 0.0, "omega": 0.0}
    o = {"shape": "sphere", "s": float(s_o), "motion": m, "role": "occluder", "mode": "lane"}
    pc, _, Rc = track(o, T, fps, g)
    ratio = occluder_screen_ratio(cam, tgt, ctr, o, center_path(o, pc, Rc), fs, fe)
    lo_r, hi_r = P["object_screen_ratio"]
    if not (0.9 * lo_r <= ratio <= hi_r * 1.08):
        return None
    ramp = next(p for p in props if p["role"] == "ramp")
    w = ramp_wedge(ramp)
    y0 = min(w["y0"], y_o - r - 0.15)
    nr = dict(ramp, loc=[ramp["loc"][0], 0.5 * (y0 + w["y1"]), ramp["loc"][2]], scale=[ramp["scale"][0], w["y1"] - y0, ramp["scale"][2]])
    o.update(color=D0._color(rng), rough=float(rng.uniform(0.3, 0.9)), metal=0.0, screen_ratio=float(ratio), _ramp_prop=nr)
    return o


def occluder_screen_ratio(cam, tgt, ctr_t, oc, ctr_o, fs, fe):
    """가림 구간 [fs, fe] 프레임에서 (가리는 물체 화면 크기) / (target 화면 크기) 의 중앙값. 화면 크기 = f · 2 √(max(가로 반치수) · 세로 반치수) / 깊이."""
    def px(o, c):
        h = obj_half(o)
        return cam.f * 2 * math.sqrt(max(h[0], h[1]) * h[2]) / np.maximum(cam.to_px(c[fs:fe + 1])[2], 1e-3)
    return float(np.median(px(oc, ctr_o) / px(tgt, ctr_t)))


# ----------------------------------------------------------------------------- 방해 물체 · 잡동사니
def sample_distractor(rng, cfg, cam, level, T, fps, g):
    kind = str(rng.choice(["flat_const", "flat_decel", "air_bounce"]))
    for _ in range(30):
        o = sample_motion(rng, cfg, cam, level, kind if kind != "air_bounce" else "air_bounce", T, fps, g)
        if o["props"]:
            continue
        pos, _, R = track(o, T, fps, g)
        ctr = center_path(o, pos, R)
        x_m = rng.uniform(-0.6, 0.6) * cam.half_width(max(ctr[T // 2, 1] - cam.C[1], 1.0))
        shift_x(o, x_m - ctr[T // 2, 0])
        o["role"] = "distractor"
        o.update(color=D0._color(rng), rough=float(rng.uniform(0.3, 0.9)), metal=0.0)
        return o
    return None


def foot_r(o):
    """수평 발자국 반경: 구 = 반지름, 도는 물체 · 경사 위 물체 = 대각 반경, 나머지 = 긴 변 반."""
    h = obj_half(o)
    if o["shape"] == "sphere":
        return float(h[0])
    if o["motion"].get("omega") or o["motion"]["kind"].startswith("ramp") or o["motion"]["kind"].startswith("air"):
        return float(math.hypot(h[0], h[1]))
    return float(max(h[0], h[1]))


def collide(o1, c1, o2, c2, pad=0.05):
    """두 물체 (중심 경로 c (T, 3)) 가 어느 프레임에서든 겹치나: 수평 거리 < 발자국 반경 합 + pad 이고 높이 구간이 겹침."""
    dh = np.linalg.norm(c1[:, :2] - c2[:, :2], axis=1)
    h1, h2 = obj_half(o1)[2], obj_half(o2)[2]
    vz = (np.minimum(c1[:, 2] + h1, c2[:, 2] + h2) - np.maximum(c1[:, 2] - h1, c2[:, 2] - h2)) > 0
    return bool(((dh < foot_r(o1) + foot_r(o2) + pad) & vz).any())


def obj_boxes(objs, T, fps, g, k=1.0):
    """물체 → 가림 상자 근사 [(중심 (T,3), 반치수, yaw 0, None)] (구 · 회전 물체는 회전해도 덮는 반경으로)."""
    out = []
    for o in objs:
        pos, _, R = track(o, T, fps, g)
        c = center_path(o, pos, R)
        h = obj_half(o, k)
        if o["shape"] != "sphere":
            rxy = math.hypot(h[0], h[1]) if (o["motion"].get("omega") or o["motion"]["kind"].startswith("ramp")) else None
            if rxy is not None:
                h = np.array([rxy, rxy, max(h[2], rxy if o["motion"]["kind"].startswith("ramp") else h[2])])
        out.append((c, h, 0.0, None))
    return out


def ramp_wedge(p):
    """경사로 소품 (판 표현: 바닥면 중심 원점 상자 loc · y 축 회전 rot[1] · scale [L, 폭, 두께]) → 쐐기 (직각삼각형 기둥).
    빗변 = 판의 윗면 (낮은 끝 xl 이 바닥, 높은 끝 xh 높이 zh), 밑면 = 바닥, 높은 끝에 세운 면. y 폭 [y0, y1]. render.py 도 같은 식."""
    L, Wd, th = (float(x) for x in p["scale"])
    phi = float(p["rot"][1])
    top = np.array(p["loc"], float) + th * np.array([math.sin(phi), 0.0, math.cos(phi)])
    d = np.array([math.cos(phi), 0.0, -math.sin(phi)])
    e1, e2 = top - 0.5 * L * d, top + 0.5 * L * d
    lo, hi = (e1, e2) if e1[2] < e2[2] else (e2, e1)
    return {"xl": float(lo[0]), "xh": float(hi[0]), "zh": float(hi[2]), "y0": float(top[1] - Wd / 2), "y1": float(top[1] + Wd / 2)}


def _wedges(props):
    return [ramp_wedge(p) for p in props if p["role"] == "ramp" and p.get("solid") == "wedge"]


def _wedge_hit(lo, hi, w, clear=0.0):
    """상자 [lo, hi] ((T,3) 또는 (3,)) 가 쐐기 w 와 겹치나 (프레임별 bool). 높이 = 상자 x 구간 안에서 쐐기가 가장 높은 곳."""
    lo, hi = np.atleast_2d(lo) - clear, np.atleast_2d(hi) + clear
    a, b = min(w["xl"], w["xh"]), max(w["xl"], w["xh"])
    ov = (hi[:, 0] > a) & (lo[:, 0] < b) & (hi[:, 1] > w["y0"]) & (lo[:, 1] < w["y1"])
    xn = np.clip(np.clip(w["xh"], lo[:, 0], hi[:, 0]), a, b)                      # 상자 안에서 높은 끝에 가장 가까운 x
    hgt = w["zh"] * np.clip((xn - w["xl"]) / (w["xh"] - w["xl"]), 0.0, 1.0)
    return ov & (lo[:, 2] < hgt)


def obj_hits_wedge(o, c, props, clear=0.05):
    """물체 (중심 경로 c (T,3)) 가 경사로 쐐기 속으로 들어가나 (받침인 target 은 부르지 않는다).
    경사로 운동 물체 (앞 차선 공) 는 쐐기가 받침이라 뺀다 — 그 물체가 경사면을 뚫는지는 physics_check 의 경사로 검사가 본다."""
    if o["motion"]["kind"].startswith("ramp_"):
        return False
    r, hz = foot_r(o), obj_half(o)[2]
    lo, hi = c - np.array([r, r, hz]), c + np.array([r, r, hz])
    return any(_wedge_hit(lo, hi, w, clear).any() for w in _wedges(props))


def occ_hits_wedge(props, panels, T, fps, clear=0.02):
    """가림막 (판 · occ_ 상자 · 기둥 · 지붕 · 벽) 이 쐐기와 겹치나."""
    W = _wedges(props)
    if not W:
        return False
    boxes = []
    for p in props:
        if p["role"] == "ramp":
            continue
        L = np.array(p["scale"], float); lo = np.array(p["loc"], float) - np.array([L[0] / 2, L[1] / 2, 0]); boxes.append((lo, lo + L))
    for p in panels:
        boxes.append(_panel_lohi(p, T, fps))
    return any(_wedge_hit(lo, hi, w, clear).any() for w in W for lo, hi in boxes)


def wedge_boxes(props, n=12):
    """가림 계산용: 쐐기를 x 로 n 조각 낸 안쪽 상자들 (조각마다 낮은 쪽 높이 — 쐐기보다 작게, 가림을 부풀리지 않는다)."""
    out = []
    for w in _wedges(props):
        xs = np.linspace(w["xl"], w["xh"], n + 1)
        for i in range(n):
            h = w["zh"] * i / n
            if h <= 1e-3:
                continue
            out.append((np.array([0.5 * (xs[i] + xs[i + 1]), 0.5 * (w["y0"] + w["y1"]), h / 2]),
                        np.array([0.5 * abs(xs[i + 1] - xs[i]), 0.5 * (w["y1"] - w["y0"]), h / 2]), 0.0, None))
    return out


def prop_boxes(props):
    out = []
    for p in props:
        if p["role"] == "ramp":
            continue
        L = np.array(p["scale"], float); c = np.array(p["loc"], float) + np.array([0, 0, L[2] / 2])
        out.append((c, L / 2, float(p["rot"][2]) if len(p["rot"]) > 2 else 0.0, None))
    return out


def panel_boxes(panels, T, fps):
    out = []
    for p in panels:
        xs = G1.panel_x(p, T, fps)
        ang = G1.panel_ang(p, T, fps)
        c = np.c_[xs, np.full(T, p["y"]), np.full(T, p["h"] / 2)]
        out.append((c, np.array([p["w"] / 2, p["th"] / 2, p["h"] / 2]), 0.0, ang > -0.15))
    return out


# ----------------------------------------------------------------------------- 한 clip
def sample_clip(cfg, split, idx, max_tries=1500, streams=4):
    """clip 하나의 설계 (같은 설정 · 분할 · 번호면 같은 장면). 한 난수 줄기에서 max_tries 안에 안 되면 다음 줄기로 (결정적)."""
    for k in range(streams):
        try:
            sp = _sample_clip(cfg, split, idx, max_tries, k)
            sp["stream"] = k
            return sp
        except RuntimeError:
            continue
    raise RuntimeError(f"clip {split} {idx} 설계 실패 (줄기 {streams} 개 × {max_tries})")


def _sample_clip(cfg, split, idx, max_tries, stream):
    sp = cfg["splits"][split]
    rng = np.random.default_rng([int(sp["seed"]), int(idx)] + ([int(stream)] if stream else []))
    T, fps, g = int(cfg["frames"]), float(cfg["fps"]), float(cfg["g"])
    cell, nd = cell_of(cfg, idx)
    W = cfg["timing"]
    fixed_sub = pick_fixed_sub(rng, cfg, cell["motion"], cell["camera"]) if cell["form"] == "fixed" else None
    tried = set()
    for attempt in range(max_tries):
        if fixed_sub is not None and attempt and attempt % 300 == 0:   # 이 하위 종류로 300 번 안 되면 그것을 빼고 다시 뽑는다 (운동과 안 맞는 조합)
            tried.add(fixed_sub)
            fixed_sub = pick_fixed_sub(rng, cfg, cell["motion"], cell["camera"], tried)
        cam_p = sample_cam(rng, cfg, cell["camera"])
        cam = D0.Cam(cam_p["h"], cam_p["pitch"], cam_p["lens"], size=cfg["size"])
        cam.level = cell["camera"]
        got = sample_target(rng, cfg, cam, cell["camera"], cell["motion"], T, fps, g)
        if got is None:
            continue
        tgt, pos, R, ctr = got
        E = extent_pts(tgt, ctr, R)
        s_w, e_w = want_interval(rng, cell["timing"], W)
        fs, fe = STRIDE * s_w, STRIDE * e_w
        panels, props, occ_objs, sub = [], list(tgt.pop("props")), [], None
        if cell["form"] == "fixed":
            r = make_fixed(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, fixed_sub)
            if r is None:
                continue
            pa, pr, sub = r; panels += pa; props += pr
        elif cell["form"] == "moving":
            r = make_moving(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps)
            if r is None:
                continue
            pa, pr, sub = r; panels += pa
        else:
            oc = make_object_occluder(rng, cfg, cam, tgt, ctr, E, fs, fe, T, fps, g, props)
            if oc is None:
                continue
            if "_ramp_prop" in oc:                                       # 앞 차선 공: 쐐기를 앞 차선까지 넓힌 것으로 바꾼다
                ir = next(i for i, p in enumerate(props) if p["role"] == "ramp")
                props = props[:ir] + [oc.pop("_ramp_prop")] + props[ir + 1:]
            occ_objs.append(oc); sub = oc["mode"]
        if _hits_props(ctr, foot_r(tgt), [p for p in props if p["role"].startswith("occ_")], panels, T, fps, clear=0.05):
            continue
        if occ_hits_wedge(props, panels, T, fps):                   # 가림막이 경사로 쐐기 속에 서면 안 된다
            continue
        # 지정 가림막으로 타이밍 확인 (샘플 단위)
        # 11-01: 렌더로 다시 재면 설계보다 일찍 · 오래 가려진다 (상자 꼭짓점이 실제 실루엣보다 바깥이라). 스모크 320 개 실측:
        #   설계 1 개 추정 (꼭짓점 그대로, 가리는 물체 0.9 배) 만 통과 → 렌더 통과 89 %. 두 추정 (꼭짓점 그대로 · 0.55 배로 줄인 점) 이
        #   둘 다 칸 정의를 지키면 → 렌더 통과 96 % (표본 안 추정), 설계 통과는 0.83 배 (재시도만 는다)
        des = panel_boxes(panels, T, fps) + prop_boxes([p for p in props if p["role"].startswith("occ_")]) + obj_boxes(occ_objs, T, fps, g, 1.0)
        off = off_frame(cam, E)
        ok, ev = timing_ok(cell["timing"], target_hidden(cam, des, E)[::STRIDE], off[::STRIDE], W)
        if not ok:
            continue
        E_in = ctr[:, None] + 0.55 * (E - ctr[:, None])
        if not timing_ok(cell["timing"], target_hidden(cam, des, E_in)[::STRIDE], off[::STRIDE], W)[0]:
            continue
        # 방해 물체 (정확히 nd 개, 2 수준이면 2–3) · 잡동사니
        n_d = int(nd) if nd < 2 else int(rng.integers(2, 4))
        objs = [tgt]
        world = [(tgt, ctr)]
        bad_oc = False
        for oc in occ_objs:
            pc, _, Rc = track(oc, T, fps, g); coc = center_path(oc, pc, Rc)
            if collide(tgt, ctr, oc, coc, 0.05) or _hits_props(coc, foot_r(oc), props, panels, T, fps) or obj_hits_wedge(oc, coc, props):
                bad_oc = True
            world.append((oc, coc))
        if bad_oc:
            continue
        fail = False
        for _ in range(n_d):
            for _try in range(40):
                d = sample_distractor(rng, cfg, cam, cell["camera"], T, fps, g)
                if d is None:
                    continue
                pd, _, Rd = track(d, T, fps, g); cd = center_path(d, pd, Rd)
                rd = foot_r(d)
                if any(collide(d, cd, ow, cw, 0.15) for ow, cw in world):
                    continue
                if _hits_props(cd, rd, props, panels, T, fps) or obj_hits_wedge(d, cd, props):
                    continue
                # 방해 물체가 target 을 가리면 안 된다 (가림 형태가 섞임)
                if target_hidden(cam, obj_boxes([d], T, fps, g, 1.05), E).max() > 0.05:
                    continue
                objs.append(d); world.append((d, cd)); break
            else:
                fail = True
                break
        if fail:
            continue
        # 가림막 아닌 것 (벽 · 방해 물체 · 잡동사니) 이 target 을 가리면 안 된다
        others = prop_boxes([p for p in props if not p["role"].startswith("occ_")]) + wedge_boxes(props)   # 쐐기 앞면도 (낮은 카메라)
        if others and target_hidden(cam, others, E).max() > 0.05:
            continue
        objs += occ_objs
        clutter = []
        for _ in range(int(rng.choice(cfg["objects"]["n_clutter"]))):
            c = sample_motion(rng, cfg, cam, cell["camera"], "flat_static", T, fps, g)
            depth = cfg["objects"]["clutter_depth"]
            c["motion"]["p0"] = [float(rng.uniform(-1, 1) * cam.half_width(5.0)), float(rng.uniform(*depth))]
            c["s"] = float(c["s"] * rng.uniform(0.5, 1.0))
            pc, _, Rc = track(c, T, fps, g); cc = center_path(c, pc, Rc); rc = foot_r(c)
            if any(collide(c, cc, ow, cw, 0.2) for ow, cw in world) or _hits_props(cc, rc, props, panels, T, fps) or obj_hits_wedge(c, cc, props):
                continue
            if target_hidden(cam, obj_boxes([c], T, fps, g, 1.05), E).max() > 0.05:
                continue
            c["role"] = "clutter"; c.update(color=D0._color(rng), rough=0.7, metal=0.0)
            clutter.append(c); world.append((c, cc))
        objs += clutter
        env = _env(rng)
        return {"cam": cam_p, "objects": objs, "panels": panels, "props": props, "env": env, "shadows": bool(cfg["objects"]["shadows"]),
                "cell": dict(cell, count=int(nd), n_distractor=n_d, form_sub=sub), "event_samples": list(ev),
                "seed": int(sp["seed"]), "idx": int(idx), "attempt": attempt, "T": T, "fps": fps, "g": g, "size": int(cfg["size"]),
                "version": cfg["name"], "split": split, "design": "gen_design_v2"}
    raise RuntimeError(f"clip {split} {idx} 설계 실패 ({cell}, 방해 {nd})")


def _hits_props(c, r, props, panels, T, fps, clear=0.05, skip_roles=("ramp",)):
    """중심 경로 c (T, 3) · 수평 반경 r 인 물체가 소품 상자 (벽 · 가림막 · 기둥) 또는 판에 r + clear 보다 가까워지나 (경사로는 받침이라 뺀다).
    판은 서 있을 때 세운 상자, 누운 여닫는 판은 카메라 쪽 바닥에 깔린 얇은 상자."""
    for p in props:
        if p["role"] in skip_roles:
            continue
        L = np.array(p["scale"], float); lo = np.array(p["loc"], float) - np.array([L[0] / 2, L[1] / 2, 0]); hi = lo + L
        q = np.clip(c, lo, hi)
        if (np.linalg.norm(c - q, axis=1) < r + clear).any():
            return True
    for p in panels:
        lo, hi = _panel_lohi(p, T, fps)
        q = np.clip(c, lo, hi)
        if (np.linalg.norm(c - q, axis=1) < r + clear).any():
            return True
    return False


def _panel_lohi(p, T, fps):
    """판의 프레임별 상자 (lo, hi) (T, 3): 서 있으면 세운 상자, 누운 여닫는 판은 카메라 쪽 바닥에 깔린 얇은 상자."""
    xs = G1.panel_x(p, T, fps)
    ang = G1.panel_ang(p, T, fps)
    lo = np.c_[xs - p["w"] / 2, np.full(T, p["y"] - p["th"] / 2), np.zeros(T)]
    hi = np.c_[xs + p["w"] / 2, np.full(T, p["y"] + p["th"] / 2), np.full(T, p["h"])]
    lying = ang < -0.15
    lo[lying, 1] = p["y"] - p["h"]; hi[lying, 1] = p["y"]; hi[lying, 2] = p["th"]
    return lo, hi


def _env(rng):
    return {"floor": str(rng.choice(D0.FLOORS)), "floor_c1": D0._color(rng, (0.0, 0.5), (0.15, 0.6)),
            "floor_c2": D0._color(rng, (0.0, 0.5), (0.15, 0.6)), "floor_scale": float(rng.uniform(0.3, 1.5)),
            "sky_hor": D0._color(rng, (0.0, 0.25), (0.7, 1.0)), "sky_top": D0._color(rng, (0.2, 0.8), (0.3, 0.9)),
            "sky_strength": float(rng.uniform(0.6, 1.3)), "sun_elev": float(rng.uniform(25, 70)),
            "sun_azim": float(rng.uniform(0, 360)), "sun_energy": float(rng.uniform(2.0, 5.0)),
            "wall": bool(rng.uniform() < 0.45), "wall_y": float(rng.uniform(15, 30)), "wall_h": float(rng.uniform(3, 10)),
            "wall_tex": str(rng.choice(["brick", "noise", "plain"])), "wall_c1": D0._color(rng, (0.0, 0.5), (0.25, 0.8)),
            "wall_c2": D0._color(rng, (0.0, 0.5), (0.25, 0.8)), "exposure": float(rng.uniform(-0.3, 0.3))}


def clip_id(spec):
    return f"{spec['split']}{spec['seed']}_{spec['idx']:06d}"


# ----------------------------------------------------------------------------- 물리 검사 (설계 · 렌더 뒤 둘 다)
def physics_check(spec, tol=0.02):
    """운동이 물리에 맞나 — 어긴 항목 목록 (빈 목록 = 통과).
    ① 바닥 · 경사로를 뚫지 않음 ② 물체끼리 · 소품 · 판과 겹치지 않음 (받침 · 설계된 벽 접촉 제외)
    ③ 운동 종류별 꼴: 정지 = 안 움직임 · 등속 = 가속도 0 · 감속 = 속력 줄어듦 · 가속 = 속력 늘어남 · 반전 = 방향이 한 번 바뀜 · 충돌 = 튕김에서 반발 e < 1
    ④ 에너지가 늘지 않음 (운동 + 위치, 구는 회전 7/5) ⑤ 구는 미끄러지지 않고 구름 (바닥 위)."""
    T, fps, g = spec["T"], spec["fps"], spec["g"]
    bad = []
    trs, _, _ = tracks(spec)
    cs = []
    for k, (o, (loc, _, R)) in enumerate(zip(spec["objects"], trs)):
        hz = 0.5 * o["s"] * DIMS[o["shape"]][2]
        c = loc + np.einsum("tij,j->ti", R, np.array([0, 0, hz]))
        pos, _, _ = track(o, T, fps, g)
        if np.abs(c - center_path(o, pos, R)).max() > 1e-6:
            bad.append(f"obj{k} 회전 중심 어긋남")
        cs.append(c)
        m = o["motion"]; kind = m["kind"]
        r = hz
        if o["shape"] == "sphere" or not kind.startswith("ramp"):
            if (c[:, 2] - r < -tol * 2 * r).any():
                bad.append(f"obj{k} 바닥 아래 {float((r - c[:, 2]).max()):.3f}")
        if kind.startswith("ramp_"):
            th, sx = math.radians(m["theta_deg"]), m["sx"]
            p0 = np.array([m["x_low"], m["y"], 0.0])
            uu = (c - p0) @ np.array([sx * math.cos(th), 0.0, math.sin(th)])
            dd = (c - p0) @ np.array([-sx * math.sin(th), 0.0, math.cos(th)])
            on = (uu >= 0) & (uu <= m["L"])
            if on.any() and (dd[on] < r * (1 - tol)).any():
                bad.append(f"obj{k} 경사로 속 {float((r - dd[on]).max()):.3f}")
        v = np.gradient(c, 1 / fps, axis=0)
        sp = np.linalg.norm(v, axis=1)
        a = np.gradient(v, 1 / fps, axis=0)
        if kind in ("flat_static", "ramp_static") and np.ptp(c, axis=0).max() > 1e-6:
            bad.append(f"obj{k} 정지인데 움직임")
        if kind == "flat_const" and np.linalg.norm(a[2:-2], axis=1).max() > 1e-6:
            bad.append(f"obj{k} 등속인데 가속")
        if kind in ("flat_decel", "ramp_decel") and (np.diff(sp[1:-1]) > 1e-6).any():
            bad.append(f"obj{k} 감속인데 속력 늘어남")
        if kind == "air_accel":
            fl = c[:, 2] > r + 1e-3
            if fl.sum() > 4 and np.abs(a[fl][2:-2, 2] + g).max() > 0.05 * g:
                bad.append(f"obj{k} 자유낙하 가속도 ≠ −g")
        if kind == "ramp_accel":
            s_c = r * math.tan(math.radians(m["theta_deg"]) / 2)
            onr = (uu > s_c + 0.25) & (uu < m["L"])
            if onr.sum() > 4 and (np.diff(sp[onr][1:-1]) < -1e-6).any():
                bad.append(f"obj{k} 내려가는데 느려짐")
        if kind in ("ramp_reverse", "air_reverse"):
            comp = (v @ np.array([m.get("sx", 0) * math.cos(math.radians(m.get("theta_deg", 0))), 0, math.sin(math.radians(m.get("theta_deg", 0)))])
                    if kind == "ramp_reverse" else v[:, 2])
            sgn = np.sign(comp[np.abs(comp) > 1e-3])
            if len(sgn) < 2 or (np.diff(sgn) != 0).sum() < 1:
                bad.append(f"obj{k} 반전 없음")
        # ④ 에너지 (마찰 · 반발로만 줄어야 한다). 앞차분 속도 (구간 가운데), 충돌 · 방향 급변 구간과 그 이웃은 뺀다
        kf = 1.4 if o["shape"] == "sphere" and (kind in ("ramp_accel", "ramp_reverse", "ramp_wall") or (kind.startswith("flat_") and kind != "flat_static")) else 1.0
        vf = np.diff(c, axis=0) * fps
        spf = np.linalg.norm(vf, axis=1)
        zm = 0.5 * (c[1:, 2] + c[:-1, 2])
        en = 0.5 * kf * spf ** 2 + g * zm
        cosv = np.sum(vf[1:] * vf[:-1], 1) / np.maximum(spf[1:] * spf[:-1], 1e-9)
        ev = np.zeros(len(vf), bool)
        ev[1:] |= (cosv < 0.9) & (spf[1:] > 1e-3) & (spf[:-1] > 1e-3)
        ev[1:] |= ((vf[1:] * vf[:-1]) < -1e-6).any(1)                 # 성분 부호가 바뀜 (낮은 튕김 · 벽)
        ev[:-1] |= ev[1:].copy(); ev[1:] |= ev[:-1].copy()
        dE = np.diff(en)
        okm = ~(ev[1:] | ev[:-1])
        if (dE[okm] > 0.01 * max(en.max(), 1e-6) + 1e-4).any():
            bad.append(f"obj{k} 에너지 늘어남 {float(dE[okm].max()):.4f}")
    # ② 물체끼리 (받침 위 · 공중 모두, 3 차원 거리)
    rr = [foot_r(o) for o in spec["objects"]]
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            if collide(spec["objects"][i], cs[i], spec["objects"][j], cs[j], 0.0):
                bad.append(f"obj{i}·obj{j} 겹침")
    for k, c in enumerate(cs):
        o = spec["objects"][k]
        if _hits_props(c, rr[k], spec["props"], spec["panels"], T, fps, clear=0.0, skip_roles=("ramp", "wall")):
            bad.append(f"obj{k} 가림막 · 기둥과 겹침")
        # 벽: target 은 설계된 접촉 (거리 = 반지름) 까지 허용, 나머지 물체는 닿으면 안 된다
        wall_r = (0.5 * o["s"] * DIMS[o["shape"]][0] - 0.02) if o["role"] == "target" else rr[k]
        if _hits_props(c, wall_r, [p for p in spec["props"] if p["role"] == "wall"], [], T, fps, clear=0.0, skip_roles=()):
            bad.append(f"obj{k} 벽 속")
        if o["role"] != "target" and obj_hits_wedge(o, c, spec["props"], clear=0.0):     # 경사로 쐐기는 target 의 받침 — 다른 물체는 못 들어간다
            bad.append(f"obj{k} 경사로 쐐기 속")
    if occ_hits_wedge(spec["props"], spec["panels"], T, fps, clear=0.0):
        bad.append("가림막이 경사로 쐐기 속")
    return bad
