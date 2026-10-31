#!/usr/bin/env python3
"""BlenderTrack v0 — 장면 설계 (numpy 만, bpy 없음). render.py 와 check_design.py 가 같은 함수를 부른다 (2026-10-31).

목적: predictor 가 "학습에서 본 궤적을 다시 꺼내기" 가 아니라 "문맥의 운동으로 자리를 계산하기" 를 배우게 하는 학습셋.
VOBench 는 설계 격자라 미래 경로가 21 가지뿐이었다 → 여기서는 모든 요인을 연속 무작위로 뽑는다.

장면 (눈높이 — 위에서 내려다보는 시점 금지):
  카메라   (0, 0, h) 에서 +y 를 본다. h 1.2–2.6 m, pitch −14…−2°, 렌즈 18–32 mm (센서 36 mm → 수평 화각 90…59°)
           VOBench = h 2.1 m, pitch −3°, 22 mm.
  바닥     z = 0 평면. 질감 checker · noise · brick · wave 중 하나, 색 · 축척 무작위. 뒷벽 45 %.
  물체     target 1 + 방해 물체 0–2 + 정지 잡동사니 0–3. 모양 8 종, 크기 = 출발 깊이에서 화면 지름 18–56 px 가 되게.
  운동     static · cv (등속) · acc (빨라짐 / 느려짐 · 되돌아옴) · stop (감속 정지) · curve (옆 가속, 지면 포물선) ·
           toss (수직 포물선 + 튕김) — 해석식, 매개변수 연속 무작위.
  가림막   target 가림 사건 0–2 개: 사건 시각 t_c · 가려지는 시간 D 를 먼저 뽑고, 그 구간의 target 실루엣을
           덮도록 판 (바닥에 선 판) 을 카메라와 target 사이에 세운다. 판은 서 있는 판 (static) 또는
           VOBench 처럼 누웠다 일어나는 판 (trapdoor). + target 과 무관한 판 0–2 개 (판 있음 ≠ 가림).
시간: 30 fps, 180 프레임 (6 s). 화면 속도 0.10–0.50 화면폭/초 (VOBench ≈ 0.24), 가운데 시각에 화면 가로 30–70 %, 화면 안 50 % 이상, 화면 크기 (√가로×세로, 실제 치수) 중앙값 ≥ 18 px · 최소 ≥ 11 px.
target 은 보이는 프레임 ≥ 45 %, 판 뒤에 숨은 구간 ≤ 2 s.
가림 사건 시각: 75 % 는 clip 가운데 (30–70 %) — 문맥 16 + 미래 16 창 (보폭 3 → 3.1 s) 이 사건을 문맥 끝에 둘 수 있는 자리. 학습 창 (보폭 2–4, 문맥 · 미래 길이 가변) 은 학습 때 뽑는다.

좌표: 미터 · 초. 물체 원점 = 바닥면 중심 (z = 바닥 높이). 카메라 행렬은 Blender 카메라
(rotation_euler = (90° + pitch, 0, 0), sensor_fit AUTO, 정사각 해상도) 와 같다 — render.py 가 Blender 투영과 대조한다.
"""
from __future__ import annotations

import math

import numpy as np

FPS = 30
T_FRAMES = 180
SIZE = 256
SENSOR = 36.0
G = 9.81

SHAPES = ["sphere", "cube", "cone", "cylinder", "torus", "pyramid", "capsule", "diamond"]
# 크기 s = 가장 긴 변. 모양별 (x, y, z) 비율 — render.py 가 메시를 이 상자에 정확히 맞춘다
DIMS = {"sphere": (1, 1, 1), "cube": (1, 1, 1), "cone": (1, 1, 1), "cylinder": (0.84, 0.84, 1),
        "torus": (1, 0.3, 1), "pyramid": (1, 1, 0.8), "capsule": (0.52, 0.52, 1), "diamond": (0.55, 0.55, 1)}
FLOORS = ["checker", "noise", "brick", "wave"]
KINDS = ["static", "cv", "acc", "stop", "curve", "toss"]
KIND_P = [0.04, 0.24, 0.18, 0.12, 0.16, 0.26]


# ----------------------------------------------------------------------------- camera
class Cam:
    def __init__(self, h, pitch_deg, lens, size=SIZE, sensor=SENSOR):
        th = math.radians(90.0 + pitch_deg)
        self.R = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
        self.C = np.array([0.0, 0.0, h])
        self.f = size * lens / sensor
        self.c = size / 2.0
        self.size = size

    def to_px(self, P):
        """P (..., 3) 세계 좌표 → (u, v, depth). u 오른쪽, v 아래쪽, 픽셀 단위."""
        P = np.asarray(P, dtype=np.float64)
        pc = (P - self.C) @ self.R            # = R^T (P - C)
        d = -pc[..., 2]
        dd = np.where(d > 1e-6, d, 1e-6)
        return self.c + self.f * pc[..., 0] / dd, self.c - self.f * pc[..., 1] / dd, d

    def ray(self, u, v):
        """픽셀 (u, v) 를 지나는 세계 방향 (정규화 안 함)."""
        dc = np.stack([(np.asarray(u) - self.c) / self.f, -(np.asarray(v) - self.c) / self.f,
                       -np.ones_like(np.asarray(u, dtype=np.float64))], -1)
        return dc @ self.R.T

    def half_width(self, depth):
        return depth * self.c / self.f       # 그 깊이에서 화면 반폭 (m)


# ----------------------------------------------------------------------------- motion
def _toss_z(t, z0, vz0, e, t_launch):
    """수직 포물선 + 튕김. t (n,) 초 → z (n,). t_launch 전에는 z0 에 머문다 (z0 = 0 이면 바닥에서 미끄러짐)."""
    z = np.full_like(t, z0, dtype=np.float64)
    # 사건 목록: (시작 시각, 시작 높이, 시작 속도)
    segs, ts, zs, vs = [], t_launch, z0, vz0
    for _ in range(30):
        # zs + vs τ − g τ²/2 = 0 의 양의 근
        disc = vs * vs + 2 * G * zs
        tau = (vs + math.sqrt(max(disc, 0.0))) / G
        segs.append((ts, zs, vs, ts + tau))
        v_hit = math.sqrt(max(disc, 0.0))
        ts, zs, vs = ts + tau, 0.0, e * v_hit
        if vs < 0.8:
            break
    t_rest = ts
    for (a, z_a, v_a, b) in segs:
        m = (t >= a) & (t < b)
        tt = t[m] - a
        z[m] = np.maximum(z_a + v_a * tt - 0.5 * G * tt * tt, 0.0)
    z[t >= t_rest] = 0.0
    return z


def track(m, T=T_FRAMES, fps=FPS):
    """운동 매개변수 → (T, 3) 위치 (바닥면 중심) · (T,) yaw (rad)."""
    t = np.arange(T) / fps
    p0, v, a = np.array(m["p0"], float), np.array(m["v"], float), np.array(m.get("a", [0.0, 0.0]), float)
    k = m["kind"]
    if k == "static":
        xy = np.repeat(p0[None], T, 0)
    elif k in ("cv", "toss"):
        xy = p0 + t[:, None] * v
    elif k in ("acc", "curve"):
        xy = p0 + t[:, None] * v + 0.5 * t[:, None] ** 2 * a
    elif k == "stop":
        ts = m["t_stop"]
        tt = np.minimum(t, ts)[:, None]
        xy = p0 + tt * v + 0.5 * tt ** 2 * a
    else:
        raise ValueError(k)
    if k == "toss":
        z = _toss_z(t, m["z0"], m["vz0"], m["e"], m["t_launch"])
        # 공중에 뜨기 전 (t < t_launch) 에도 수평 운동은 이어진다
    else:
        z = np.zeros(T)
    yaw = m.get("yaw0", 0.0) + m.get("omega", 0.0) * t
    return np.c_[xy, z], yaw


def extent_pts(shape, s, pos, n_ang=8):
    """물체 바깥 점 (T, 2 n_ang + 2, 3): 바닥 · 꼭대기 원 (반지름 = 회전해도 덮는 xy 반경) + 중심 두 점."""
    dx, dy, dz = (s * d for d in DIMS[shape])
    r = 0.5 * math.hypot(dx, dy)
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    ring = np.c_[r * np.cos(ang), r * np.sin(ang)]
    pts = np.concatenate([np.c_[ring, np.zeros(n_ang)], np.c_[ring, np.full(n_ang, dz)],
                          [[0, 0, 0.02 * dz], [0, 0, dz * 0.98]]], 0)
    return pos[:, None, :] + pts[None]


def footprint_r(shape, s):
    dx, dy, _ = (s * d for d in DIMS[shape])
    return 0.5 * math.hypot(dx, dy)


# ----------------------------------------------------------------------------- panel
def panel_angle(p, T=T_FRAMES, fps=FPS):
    """판 각도 (T,) 라디안: 0 = 서 있음, −π/2 = 카메라 쪽으로 누움."""
    if p["mode"] == "static":
        return np.zeros(T)
    t = np.arange(T) / fps
    ts, te, r = p["t_up"], p["t_down"], p["rise"]
    up = np.clip((t - (ts - r)) / r, 0, 1) * np.clip(((te + r) - t) / r, 0, 1)
    up = 0.5 - 0.5 * np.cos(np.pi * up)      # 부드럽게
    return -0.5 * np.pi * (1 - up)


def panel_hides(cam, p, ang, P):
    """점 P (T, n, 3) 가 판 p 뒤에 가려지나 (T, n) bool. 판이 '거의 서 있을 때' (각 > −0.15 rad) 만 센다 (설계 검사용 근사)."""
    C = cam.C
    dy = P[..., 1] - C[1]
    tpar = (p["y"] - C[1]) / np.where(np.abs(dy) < 1e-9, 1e-9, dy)
    X = C[0] + tpar * (P[..., 0] - C[0])
    Z = C[2] + tpar * (P[..., 2] - C[2])
    behind = P[..., 1] > p["y"] + 1e-3
    inside = (np.abs(X - p["x"]) <= p["w"] / 2) & (Z >= 0) & (Z <= p["h"]) & (tpar > 0) & (tpar < 1)
    standing = (ang > -0.15)[:, None]
    return behind & inside & standing


# ----------------------------------------------------------------------------- sampling
def _hsv2rgb(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def _srgb2lin(c):
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


def _color(rng, s_rng=(0.45, 1.0), v_rng=(0.35, 1.0)):
    return _srgb2lin(_hsv2rgb(rng.uniform(), rng.uniform(*s_rng), rng.uniform(*v_rng)))


def _loguni(rng, a, b):
    return float(math.exp(rng.uniform(math.log(a), math.log(b))))


def sample_motion(rng, cam, shape, s, T, kind=None, y_rng=(4.0, 14.0)):
    t_end = (T - 1) / FPS
    kind = kind or rng.choice(KINDS, p=KIND_P)
    y0 = rng.uniform(*y_rng)
    hw = cam.half_width(y0)
    x0 = rng.uniform(-0.95, 0.95) * hw
    w = _loguni(rng, 0.10, 0.50)                     # 화면 폭 / 초 (VOBench ≈ 0.24)
    v = w * 2 * hw
    psi = math.radians(rng.uniform(-35, 35) if rng.uniform() < 0.8 else rng.uniform(-70, 70))
    sgn = rng.choice([-1.0, 1.0])
    vv = np.array([sgn * v * math.cos(psi), v * math.sin(psi)])
    m = {"kind": str(kind), "p0": [x0, y0], "v": vv.tolist(), "a": [0.0, 0.0],
         "yaw0": float(rng.uniform(0, 2 * np.pi)),
         "omega": float(math.radians(rng.uniform(-120, 120))) if rng.uniform() < 0.4 else 0.0}
    vhat = vv / max(np.linalg.norm(vv), 1e-9)
    if kind == "static":
        m["v"] = [0.0, 0.0]
    elif kind == "acc":
        rho = rng.uniform(0.3, 1.5)
        m["a"] = (rng.choice([-1.0, 1.0]) * rho * v / 3.0 * vhat).tolist()
    elif kind == "stop":
        ts = rng.uniform(0.15, 0.85) * t_end
        m["t_stop"] = float(ts)
        m["a"] = (-vv / ts).tolist()
    elif kind == "curve":
        perp = np.array([-vhat[1], vhat[0]]) * rng.choice([-1.0, 1.0])
        m["a"] = (rng.uniform(0.3, 1.0) * v / 3.0 * perp).tolist()
    elif kind == "toss":
        m["v"] = (vv * rng.uniform(0.5, 1.0)).tolist()
        if rng.uniform() < 0.6:
            m["z0"], m["vz0"] = 0.0, float(rng.uniform(2.5, 6.5))
        else:
            m["z0"], m["vz0"] = float(rng.uniform(0.5, 2.5)), float(rng.uniform(-1.0, 4.0))
        m["e"] = float(rng.uniform(0.35, 0.7))
        m["t_launch"] = float(rng.uniform(0, 0.5) * t_end) if rng.uniform() < 0.5 else 0.0
    if kind != "static":
        # 가운데 시각에 화면 안 (가로 30–70 %) 에 오게 x 를 옮긴다 — 빠른 물체를 거절로 걸러 느린 쪽으로 치우치지 않게
        pos, _ = track(m, T)
        tm = T // 2
        x_m = rng.uniform(-0.4, 0.4) * cam.half_width(max(pos[tm, 1], 1.0))
        m["p0"] = [float(m["p0"][0] + x_m - pos[tm, 0]), float(m["p0"][1])]
    return m


def in_frame_frac(cam, pos, shape, s, margin=0.0):
    c = pos.copy()
    c[:, 2] += 0.5 * s * DIMS[shape][2]
    u, v, d = cam.to_px(c)
    ok = (u >= -margin) & (u < cam.size + margin) & (v >= -margin) & (v < cam.size + margin) & (d > 0.5)
    return float(ok.mean()), u, v


def _collide(posA, rA, posB, rB, pad=0.15):
    d = np.linalg.norm(posA[:, :2] - posB[:, :2], axis=1)
    return bool((d < rA + rB + pad).any())


def _panel_hits_path(p, ang, pos, r, pad=0.15):
    """판 바닥 자국과 물체 바닥 원이 겹치나. 서 있을 때 y ∈ y±th/2, 누웠을 때 y−h…y (카메라 쪽)."""
    x0, x1 = p["x"] - p["w"] / 2 - r - pad, p["x"] + p["w"] / 2 + r + pad
    lying = ang < -0.15
    ylo = np.where(lying, p["y"] - p["h"], p["y"] - p["th"] / 2) - r - pad
    yhi = p["y"] + p["th"] / 2 + r + pad
    inx = (pos[:, 0] >= x0) & (pos[:, 0] <= x1)
    iny = (pos[:, 1] >= ylo) & (pos[:, 1] <= yhi)
    return bool((inx & iny).any())


def _panel_style(rng):
    return {"color": _color(rng, (0.0, 0.7), (0.25, 0.9)), "tex": str(rng.choice(["plain", "noise", "wave"])),
            "tex_scale": float(rng.uniform(2, 12)), "th": float(rng.uniform(0.04, 0.10))}


def sample_clip(seed, idx, T=T_FRAMES, max_tries=200):
    """clip 하나의 설계 (JSON 으로 저장 가능한 dict). 같은 (seed, idx) 는 같은 장면."""
    rng = np.random.default_rng([seed, idx])
    for attempt in range(max_tries):
        spec = _try_clip(rng, T)
        if spec is not None:
            spec.update(seed=int(seed), idx=int(idx), attempt=attempt, T=T, fps=FPS, size=SIZE)
            return spec
    raise RuntimeError(f"clip {idx}: 설계 실패")


def _sample_obj(rng, cam, T, kind=None, px_rng=(18, 56), y_rng=(4.0, 14.0)):
    shape = str(rng.choice(SHAPES))
    for _ in range(50):
        m = sample_motion(rng, cam, shape, 1.0, T, kind=kind, y_rng=y_rng)
        y0 = m["p0"][1]
        s = float(np.clip(rng.uniform(*px_rng) * y0 / cam.f, 0.2, 2.5))
        pos, _ = track(m, T)
        if pos[:, 1].min() < 3.0 or pos[:, 1].max() > 30.0:
            continue
        fr, _, _ = in_frame_frac(cam, pos, shape, s)
        if fr < 0.5:
            continue
        # 화면 크기 = √(가로 × 세로) 를 실제 치수로 (감싸는 원은 40 % 까지 크게 친다): 화면 안 프레임 중앙값 ≥ 18 px, 최소 ≥ 11 px
        if px_rng[0] >= 14:
            dx, dy, dz = (s * d for d in DIMS[shape])
            ctr = pos + np.array([0.0, 0.0, 0.5 * dz])
            cu, cv, dep = cam.to_px(ctr)
            sz = cam.f * math.sqrt(max(dx, dy) * dz) / np.maximum(dep, 1e-3)
            inn = (cu >= 0) & (cu < cam.size) & (cv >= 0) & (cv < cam.size)
            if not inn.any() or np.median(sz[inn]) < 18 or sz[inn].min() < 11:
                continue
        return {"shape": shape, "s": s, "motion": m, "color": _color(rng),
                "rough": float(rng.uniform(0.3, 0.9)), "metal": float(rng.uniform(0.0, 0.6)) if rng.uniform() < 0.2 else 0.0}
    return None


def _try_clip(rng, T):
    cam_p = {"h": float(rng.uniform(1.2, 2.6)), "pitch": float(rng.uniform(-14, -2)), "lens": float(rng.uniform(18, 32))}
    cam = Cam(cam_p["h"], cam_p["pitch"], cam_p["lens"])
    tgt = _sample_obj(rng, cam, T)
    if tgt is None:
        return None
    tgt["role"] = "target"
    pos_t, _ = track(tgt["motion"], T)
    r_t = footprint_r(tgt["shape"], tgt["s"])
    tracks = [(pos_t, r_t)]

    # target 가림 사건
    panels = []
    n_ev = int(rng.choice([0, 1, 2, 3], p=[0.10, 0.35, 0.35, 0.20]))
    t_end = (T - 1) / FPS
    E = extent_pts(tgt["shape"], tgt["s"], pos_t)
    for _ in range(n_ev):
        for _try in range(30):
            tc = rng.uniform(0.3, 0.7) * t_end if rng.uniform() < 0.75 else rng.uniform(0.12, 0.88) * t_end
            D = _loguni(rng, 0.15, 1.2)
            f0, f1 = max(0, int(round((tc - D / 2) * FPS))), min(T - 1, int(round((tc + D / 2) * FPS)))
            if f1 <= f0:
                continue
            seg = E[f0:f1 + 1]
            gap = rng.uniform(0.6, 2.5)
            yp = float(pos_t[f0:f1 + 1, 1].min() - r_t - gap)
            if yp < 1.8:
                continue
            C = cam.C
            tpar = (yp - C[1]) / (seg[..., 1] - C[1])
            X = C[0] + tpar * (seg[..., 0] - C[0])
            Z = C[2] + tpar * (seg[..., 2] - C[2])
            x0, x1, top = X.min() - 0.04, X.max() + 0.04, Z.max() + 0.04
            if x1 - x0 > 6.0 or top > 4.5:
                continue
            p = {"role": "event", "x": float((x0 + x1) / 2), "y": yp, "w": float(x1 - x0), "h": float(top),
                 "t_c": float(tc), "D": float(D), "f_hide": [f0, f1], **_panel_style(rng)}
            if rng.uniform() < 0.45:
                rise = float(rng.uniform(0.1, 0.25))
                p.update(mode="trapdoor", t_up=f0 / FPS, t_down=f1 / FPS, rise=rise)
            else:
                p.update(mode="static")
            ang = panel_angle(p, T)
            if _panel_hits_path(p, ang, pos_t, r_t):
                continue
            if any(abs(q["y"] - p["y"]) < 0.3 and abs(q["x"] - p["x"]) < (q["w"] + p["w"]) / 2 for q in panels):
                continue
            panels.append(p)
            break

    # 방해 물체
    objs = [tgt]
    for _ in range(int(rng.choice([0, 1, 2], p=[0.45, 0.40, 0.15]))):
        o = _sample_obj(rng, cam, T)
        if o is None:
            continue
        pos_o, _ = track(o["motion"], T)
        r_o = footprint_r(o["shape"], o["s"])
        if any(_collide(pos_o, r_o, pa, ra) for pa, ra in tracks):
            continue
        if any(_panel_hits_path(p, panel_angle(p, T), pos_o, r_o) for p in panels):
            continue
        o["role"] = "distractor"
        objs.append(o)
        tracks.append((pos_o, r_o))
    # 이벤트 판이 방해 물체와 부딪히면 이미 걸렀다. 이제 무관한 판
    for _ in range(int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]))):
        yp = float(rng.uniform(3.0, 12.0))
        hw = cam.half_width(yp)
        p = {"role": "extra", "x": float(rng.uniform(-0.8, 0.8) * hw), "y": yp, "w": float(rng.uniform(0.5, 3.0)),
             "h": float(rng.uniform(0.6, 2.5)), **_panel_style(rng)}
        if rng.uniform() < 0.4:
            ts = rng.uniform(0.05, 0.8) * t_end
            p.update(mode="trapdoor", t_up=float(ts), t_down=float(ts + rng.uniform(0.2, 1.5)), rise=float(rng.uniform(0.1, 0.25)))
        else:
            p.update(mode="static")
        ang = panel_angle(p, T)
        if any(_panel_hits_path(p, ang, pa, ra) for pa, ra in tracks):
            continue
        if any(abs(q["y"] - p["y"]) < 0.3 and abs(q["x"] - p["x"]) < (q["w"] + p["w"]) / 2 for q in panels):
            continue
        panels.append(p)
    # 정지 잡동사니
    for _ in range(int(rng.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1]))):
        o = _sample_obj(rng, cam, T, kind="static", px_rng=(10, 40), y_rng=(4.0, 20.0))
        if o is None:
            continue
        pos_o, _ = track(o["motion"], T)
        r_o = footprint_r(o["shape"], o["s"])
        if any(_collide(pos_o, r_o, pa, ra) for pa, ra in tracks):
            continue
        if any(_panel_hits_path(p, panel_angle(p, T), pos_o, r_o) for p in panels):
            continue
        o["role"] = "clutter"
        objs.append(o)
        tracks.append((pos_o, r_o))

    # target 이 충분히 보여야 한다: 보이는 프레임 ≥ 45 %, 화면 안에서 판 뒤에 숨은 구간 ≤ 2 s (정지 물체가 끝까지 숨는 clip 제거)
    vis, run = _target_visibility(cam, tgt, pos_t, panels, T)
    if vis < 0.45 or run > 2.0:
        return None

    env = {"floor": str(rng.choice(FLOORS)), "floor_c1": _color(rng, (0.0, 0.5), (0.15, 0.6)),
           "floor_c2": _color(rng, (0.0, 0.5), (0.15, 0.6)), "floor_scale": float(rng.uniform(0.3, 1.5)),
           "sky_hor": _color(rng, (0.0, 0.25), (0.7, 1.0)), "sky_top": _color(rng, (0.2, 0.8), (0.3, 0.9)),
           "sky_strength": float(rng.uniform(0.6, 1.3)),
           "sun_elev": float(rng.uniform(25, 70)), "sun_azim": float(rng.uniform(0, 360)), "sun_energy": float(rng.uniform(2.0, 5.0)),
           "wall": bool(rng.uniform() < 0.45), "wall_y": float(rng.uniform(15, 30)), "wall_h": float(rng.uniform(3, 10)),
           "wall_tex": str(rng.choice(["brick", "noise", "plain"])), "wall_c1": _color(rng, (0.0, 0.5), (0.25, 0.8)),
           "wall_c2": _color(rng, (0.0, 0.5), (0.25, 0.8)), "exposure": float(rng.uniform(-0.3, 0.3))}
    return {"cam": cam_p, "objects": objs, "panels": panels, "env": env}


def _target_visibility(cam, o, pos, panels, T):
    E = extent_pts(o["shape"], o["s"], pos)
    hid = np.zeros(E.shape[:2], bool)
    for p in panels:
        hid |= panel_hides(cam, p, panel_angle(p, T), E)
    u, v, _ = cam.to_px(E)
    off = ((u < 0) | (u >= cam.size) | (v < 0) | (v >= cam.size)).mean(1)
    h = hid.mean(1)
    vis = float(((h + off) < 0.3).mean())
    run = best = 0
    for x in (h >= 0.9) & (off < 0.5):
        run = run + 1 if x else 0
        best = max(best, run)
    return vis, best / FPS


def all_tracks(spec):
    """spec → 물체마다 (pos (T,3), yaw (T,)) · 판마다 angle (T,)."""
    T = spec["T"]
    obj = [track(o["motion"], T) for o in spec["objects"]]
    pan = [panel_angle(p, T) for p in spec["panels"]]
    return obj, pan


def target_state(spec):
    """설계 근사: target 중심 픽셀 (T,2) · 가려진 비율 (T,) (판 · 화면 밖, 다른 물체는 안 셈)."""
    cam = Cam(spec["cam"]["h"], spec["cam"]["pitch"], spec["cam"]["lens"])
    o = spec["objects"][0]
    (pos, _), pans = track(o["motion"], spec["T"]), [panel_angle(p, spec["T"]) for p in spec["panels"]]
    E = extent_pts(o["shape"], o["s"], pos)
    hid = np.zeros(E.shape[:2], bool)
    for p, ang in zip(spec["panels"], pans):
        hid |= panel_hides(cam, p, ang, E)
    u, v, d = cam.to_px(E)
    off = (u < 0) | (u >= cam.size) | (v < 0) | (v >= cam.size)
    _, cu, cv = in_frame_frac(cam, pos, o["shape"], o["s"])
    return np.c_[cu, cv], (hid | off).mean(1)
