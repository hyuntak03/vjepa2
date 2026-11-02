#!/usr/bin/env python3
"""BlenderTrack v5_motion_control (v5mc) 설계 — 같은 장면에서 운동만 바꾼 묶음, 속도 · 가속도 · 격자 (2026-11-02, 사용자 요청).

왜 (predictor_lab/v4vc/README.md · PLAN_v5_motion_control_2026-11-02.md)
  v4 (같은 장면 · 속도 4 단계) 로 처음 VOBench late 움직이는 물체 사라짐이 옮겨 갔다 (평면 89 % · 빗면 25 %, release · v3 는 0 %) [측정].
  v4 는 평면 가속을 본 적이 없고 빗면 가속이 VOBench 의 절반쯤이다 → 속도 · 가속도를 넓게, 가속도도 "같은 장면에서 그것만 바뀌는" 묶음으로.
묶음 종류 (분할 설정의 ftype)
  speed : 속력 K 단계 (가장 느린 : 가장 빠른 = 묶음마다 3 → 2.5 → 2 중 설계가 통과하는 가장 넓은 값). 평면은 묶음마다 가속도 하나 (−0.6 ~ +0.6 m/s²),
          빗면은 묶음마다 각도 하나. 평면 trapdoor 묶음은 정지 단계 (속력 0) 를 하나 더.
  accel : 가려질 때 속력은 같고 가속도만 4 단계. 평면 = 보이지 않는 일정한 힘 (−A, −A/3, +A/3, +A), 빗면 = 경사 각도 4 단계 (쐐기가 바뀐다 → 빈 장면도 각도마다).
  grid  : 속력 3 × 가속도 3 (평면 −A, 0, +A / 빗면 각도 3).
원칙 (v4 와 같다): 모든 단계가 같은 자리를 같은 시각 t_in 에 지나 완전히 가려지기 시작한다 (late). 가리는 것 · 그 밖 물체 · 카메라 · 환경은 묶음 안에서 같다.
  다시 보임 기준은 v4 그대로 (창 끝 전에 완전히 보임 · 사용자 11-02). 화면 밖 상한만 0.35.
평면 가속: 새 운동 꼴 flat_accel (위치 = p0 + 방향 · (v0 t + ½ a t²), 감속은 0 에서 멈춤). gen_design_v2.track 을 감싸 이 꼴만 따로 계산한다
  (모듈 전역 G2.track 을 바꿔 G2 의 tracks · physics_check · obj_boxes 도 같은 계산을 쓴다; 기존 꼴의 계산은 그대로).
  물리 검사에서 flat_accel 물체의 "에너지 늘어남" 만 뺀다 (바깥 힘).
렌더 번호: 묶음 f → f · SMAX + j (SMAX = 16). j 순서 = 운동 단계 (k0 …) 다음 빈 장면 (B 또는 B0, B1 …). 쓰지 않는 번호는 .fail.
"""
from __future__ import annotations

import copy
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D0  # noqa: E402
import gen_design_v2 as G2  # noqa: E402
import gen_design_v3 as G3  # noqa: E402

DIMS = D0.DIMS
STRIDE = 2
SMAX = 16
load_cfg = G2.load_cfg
center_path = G2.center_path
clip_id = G2.clip_id
_u = G2._u

# ----------------------------------------------------------------------------- 평면 가속 운동
_track0 = getattr(G2, "_track_orig", G2.track)
G2._track_orig = _track0


def track(o, T, fps, g):
    m = o["motion"]
    if m["kind"] != "flat_accel":
        return _track0(o, T, fps, g)
    t = np.arange(T) / fps
    r = 0.5 * o["s"] * DIMS[o["shape"]][2]
    yaw = m.get("yaw0", 0.0) + m.get("omega", 0.0) * t
    Ry = np.stack([G2.G1._rz(a) for a in yaw])
    v0, a = float(m["v0"]), float(m["a"])
    ux, uy = m["u"]
    tt = np.minimum(t, v0 / -a) if a < 0 else t
    d = v0 * tt + 0.5 * a * tt * tt
    xy = np.c_[m["p0"][0] + ux * d, m["p0"][1] + uy * d]
    pos = np.c_[xy, np.zeros(T)]
    R = G2._roll_R(xy, r, Ry) if o["shape"] == "sphere" else Ry
    return pos, yaw, R


G2.track = track
tracks = G2.tracks


def physics_check(spec, tol=0.02):
    bad = G2.physics_check(spec, tol)
    acc = [k for k, o in enumerate(spec["objects"]) if o["motion"]["kind"] == "flat_accel"]
    return [b for b in bad if not any(b.startswith(f"obj{k} 에너지") for k in acc)]


# ----------------------------------------------------------------------------- 상황 · 요인 · 번호
def situations(cfg):
    C = cfg["cells"]
    ex = set(C.get("exclude", []))
    out = []
    for pl in C["place"]:
        for o in C["objset"]:
            for c in C["camera"]:
                for oc in C["occ"]:
                    if _excluded(f"{pl}|{o}|{c}|{oc}", ex):
                        continue
                    out.append(dict(place=pl, objset=o, camera=c, occ=oc))
    return out


def _excluded(key, ex):
    pl, o, c, oc = key.split("|")
    for e in ex:
        parts = dict(p.split("=") for p in e.split("&"))
        if all({"place": pl, "objset": o, "camera": c, "occ": oc}[k] == v for k, v in parts.items()):
            return True
    return False


def sit_key(s):
    return f"{s['place']}|{s['objset']}|{s['camera']}|{s['occ']}"


def factor_levels(sit, cfg):
    V = cfg["velocity"]
    if sit["place"] == "ramp":
        kinds = ["sphere"] if sit["objset"] == "primitive" else ["basketball", "soccer_ball", "beach_ball"]
    else:
        kinds = list(G3.PRIM_SHAPES) if sit["objset"] == "primitive" else G3.target_kinds("realistic", "flat", "moving")
    F = {"kind": kinds, "hue": list(range(8)), "size": [0, 1, 2], "dir": [-1, 1], "hide": list(V["hide"]), "ubin": [0, 1, 2],
         "abin": [0, 1, 2], "count": [2, 3] if sit["occ"] == "object" else [1, 2, 3], "indoor": [True, False], "floor": [0, 1, 2], "side_sign": [-1, 1]}
    if sit["place"] == "ramp":
        F["ramp_mode"] = ["down", "up"]
    return F


_SCHED = {}


def schedule(cfg, split, si, sit, N):
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


def family_of(cfg, split, f):
    S = situations(cfg)
    si, j = f % len(S), f // len(S)
    per = (int(cfg["splits"][split]["candidates"]) + len(S) - 1) // len(S)
    return S[si], si, j, schedule(cfg, split, si, S[si], per)[j]


def ftype_of(cfg, split):
    return cfg["splits"][split]["ftype"]


# ----------------------------------------------------------------------------- late 검사 (v4 와 같은 정의)
def timing_ok(hid_s, off_s, W, max_off):
    full = hid_s >= 0.95
    if not full.any():
        return False, None
    idx = np.nonzero(full)[0]
    s, e = int(idx[0]), int(idx[-1])
    if not full[s:e + 1].all():
        return False, (s, e)
    vis = (hid_s < 0.15) & (off_s < 0.3)
    half = (hid_s < 0.5) & (off_s < 0.3)
    if not half[:s].any() or not vis[e + 1:].any():
        return False, (s, e)
    a = s
    while a > 0 and not vis[a - 1]:
        a -= 1
    b = e
    while b < len(vis) - 1 and not vis[b + 1]:
        b += 1
    out = np.ones(len(vis), bool); out[a:b + 1] = False
    if (hid_s[out] >= 0.5).any():
        return False, (s, e)
    w = W["late"]
    if not (w["start"][0] <= s <= w["start"][1]) or not full[15] or not (w["end"][0] <= e <= w["end"][1]):
        return False, (s, e)
    if (off_s >= 0.3).mean() > max_off:
        return False, (s, e)
    return True, (s, e)


# ----------------------------------------------------------------------------- 운동 단계 계획
def level_plan(ftype, sit, F, rng, V, attempt, max_tries):
    """단계 목록 [{'key', 'urel', 'alpha' (평면) | 'tq' (빗면 각도 분위), 'static'}] · 묶음 비율 R.
    urel = 가려질 때 화면 속력 / target 화면 지름. 평면 가속도 a = alpha · v_ref / t_in (v_ref = 묶음에서 가장 느린 움직이는 단계의 속력):
      alpha 가 그 단계가 문맥 동안 바뀌는 속력의 비율이라 (프레임 0 속력 = v (1 − alpha)), 원인 없는 출발 · 뒤로 감 없이 넓게 고를 수 있다.
    빗면 각도 = 설계가 가능한 범위 [lo, hi_eff] 안의 분위 tq (hi_eff 는 속력을 안 뒤 정한다: 내려감은 문맥 내내 가려지는 자리 위에 있어야)."""
    ramp = sit["place"] == "ramp"
    ub = int(F["ubin"]); ab = int(F["abin"])
    lv = []
    if ftype == "speed":
        Rs = V["ratio_try"]
        R = Rs[min(len(Rs) - 1, attempt * len(Rs) // max_tries)]
        K = int(V["speed_levels"])
        rho = R ** (1.0 / (K - 1))
        u0 = G3._bin(rng, V["urel_slow"], ub, 3)
        if ramp:
            tq = (ab + rng.uniform()) / 3.0
            lv = [{"key": f"k{k}", "urel": u0 * rho ** k, "tq": tq} for k in range(K)]
        else:
            al = G3._bin(rng, V["flat_alpha"], ab, 3)
            al = 0.0 if abs(al) < V["flat_alpha_zero"] else al
            lv = [{"key": f"k{k}", "urel": u0 * rho ** k, "alpha": al} for k in range(K)]
            if sit["occ"] == "trapdoor":
                lv.insert(0, {"key": "k_static", "urel": 0.0, "alpha": 0.0, "static": True})
        return lv, R
    uh = G3._bin(rng, V["urel_mid"], ub, 3)
    if ftype == "accel":
        if ramp:
            tqs = np.sort(np.clip(np.array([0.02, 0.35, 0.65, 0.98]) + rng.uniform(-0.04, 0.04, 4), 0, 1))
            lv = [{"key": f"k{k}", "urel": uh, "tq": float(t)} for k, t in enumerate(tqs)]
        else:
            A = G3._bin(rng, V["flat_alpha_A"], ab, 3)
            lv = [{"key": f"k{k}", "urel": uh, "alpha": float(a)} for k, a in enumerate([-A, -A / 3, A / 3, A])]
        return lv, None
    sp = [uh / math.sqrt(2), uh, uh * math.sqrt(2)]
    if ramp:
        acc = [("tq", float(t)) for t in np.clip(np.array([0.05, 0.5, 0.95]) + rng.uniform(-0.04, 0.04, 3), 0, 1)]
    else:
        A = G3._bin(rng, V["flat_alpha_A"], ab, 3)
        acc = [("alpha", -A), ("alpha", 0.0), ("alpha", A)]
    for i, s_ in enumerate(sp):
        for j, (kk, av) in enumerate(acc):
            lv.append({"key": f"k{i}{j}", "urel": s_, kk: av, "si": i, "ai": j})
    return lv, None


# ----------------------------------------------------------------------------- 빗면 구 중심
def _ramp_ctr(m, r, s):
    th, sx = math.radians(m["theta_deg"]), m["sx"]
    return np.array([m["x_low"] + sx * (s * math.cos(th) - r * math.sin(th)), m["y"], s * math.sin(th) + r * math.cos(th)])


def _ramp_kind(v0, A, T, fps):
    t_end = (T - 1) / fps
    if v0 > 0 and v0 / A < t_end:
        return "ramp_reverse"
    return "ramp_accel" if v0 <= 0 else "ramp_up"


def _wedge(rng_col, m, r, Lr):
    thr = math.radians(m["theta_deg"]); sx = m["sx"]
    Wd, thick = max(0.35, 3.0 * r), 0.12
    ph = -sx * thr
    top_mid = np.array([sx * (Lr / 2) * math.cos(thr) + m["x_low"], 0.0, (Lr / 2) * math.sin(thr)])
    loc = top_mid - np.array([thick * math.sin(ph), 0.0, thick * math.cos(ph)])
    return {"role": "ramp", "solid": "wedge", "loc": loc.tolist(), "rot": [0.0, ph, 0.0], "scale": [Lr, Wd, thick], "color": rng_col}


# ----------------------------------------------------------------------------- 묶음 하나
def design_family(cfg, split, f, max_tries=500, streams=3):
    last = None
    for st in range(streams):
        try:
            return _design_family(cfg, split, f, max_tries, st)
        except RuntimeError as e:
            last = e
    raise RuntimeError(str(last))


def _bump(why, k):
    why[k] = why.get(k, 0) + 1


def _design_family(cfg, split, f, max_tries, stream):
    sp = cfg["splits"][split]
    ftype = ftype_of(cfg, split)
    sit, si, j, F = family_of(cfg, split, f)
    rng = np.random.default_rng([int(sp["seed"]), 300000 + f, stream])
    T, fps, g = int(cfg["frames"]), float(cfg["fps"]), float(cfg["g"])
    O, V, W = cfg["objects"], cfg["velocity"], cfg["timing"]
    dt = STRIDE / fps
    s_in = int(F["hide"])
    objset, occ, ramp = sit["objset"], sit["occ"], sit["place"] == "ramp"
    tt = np.arange(T) / fps
    why = {}
    for attempt in range(max_tries):
        # 1 target 생김새 · 방향
        lk, s_, shape = G3.look_fixed(rng, cfg, objset, F)
        tgt = {"shape": shape, "s": float(s_), "role": "target", **lk}
        roll = shape == "sphere"
        sg = float(F["dir"])
        hd = 0.0 if ramp else math.radians(_u(rng, O["heading_deg"]))
        ud = np.array([sg * math.cos(hd), math.sin(hd)])
        yaw0 = float(rng.uniform(0, 2 * np.pi)) if roll else float(math.atan2(ud[1], ud[0]))
        plan, R = level_plan(ftype, sit, F, rng, V, attempt, max_tries)
        r = 0.5 * tgt["s"] * DIMS[shape][2]
        # 2 빗면: 묶음 공통 길이 · 바탕 (각도는 단계마다 다를 수 있다)
        if ramp:
            mu_g = float(_u(rng, (0.01, 0.03)))
            wcol = G3._jit(rng, G3.WOOD[rng.integers(len(G3.WOOD))]) if rng.uniform() < 0.6 else G3._jit(rng, [(0.85, 0.85, 0.82), (0.35, 0.45, 0.6)][rng.integers(2)], 0.05)
            lo_d, hi_d = V["deg_" + F["ramp_mode"]]
            Lr0 = float(_u(rng, cfg["ramp"]["len"]))
            z_c = 0.45 * min(Lr0 * math.sin(math.radians(0.5 * (lo_d + hi_d))), cfg["ramp"]["max_h"]) + r     # 임시 (카메라 높이 기준)
        else:
            z_c = 0.5 * tgt["s"] * DIMS[shape][2]
        # 3 카메라 · 화면 척도 (가려지는 자리 = 원점 근처)
        c = G3.sample_camera(rng, cfg, sit["camera"], tgt, z_c, F)
        if c is None:
            _bump(why, "cam"); continue
        cam = G3.cam_obj(c)
        dirv = np.array([ud[0], ud[1], 0.0]) if not ramp else np.array([sg * math.cos(math.radians(lo_d)), 0.0, math.sin(math.radians(lo_d))])
        P = np.stack([np.array([0, 0, z_c]) - 0.05 * dirv, np.array([0, 0, z_c]) + 0.05 * dirv])
        uu, vv, dd = cam.to_px(P)
        if (dd < 0.3).any():
            continue
        ppm = math.hypot(uu[1] - uu[0], vv[1] - vv[0]) / 0.1
        if ppm < 25:
            _bump(why, "ppm"); continue
        px_h = cam.f * G3.s_eff(tgt) / float(cam.to_px(np.array([[0.0, 0.0, z_c]]))[2][0])
        phi = float(rng.uniform(0.25, 0.55))
        t_in = (s_in - phi) * dt
        need_fast = (16.3 - s_in + phi)
        # 4 단계마다 운동 (가려지기 시작하는 자리 = 원점 근처, 시각 t_in)
        lv, bad, props_lv = [], False, []
        if ramp:
            # 가려지는 경사 거리 s_h 는 단계마다 (각도가 다르면) 다를 수 있다. 모든 단계의 구 중심이 t_in 에 같은 자리 (x = 0, z = z_h) 를 지나게 x_low 를 맞춘다
            mode = F["ramp_mode"]; sgn = -1.0 if mode == "down" else 1.0
            vs_ = [p["urel"] * px_h / (ppm * dt) for p in plan]
            hi_eff = hi_d
            if mode == "down":                                          # 문맥 내내 가려지는 자리 위: v t_in − ½ A t_in² ≥ 0.03 (가장 느린 단계)
                Amax = 2.0 * (min(vs_) * t_in - 0.03) / (t_in * t_in)
                hi_eff = min(hi_d, math.degrees(math.asin(max(0.0, min(1.0, Amax / (5.0 / 7.0 * g))))))
            if hi_eff - lo_d < (3 * V["deg_gap"] if ftype != "speed" else 0.3):
                _bump(why, "deg_range"); continue
            for p in plan:
                p["theta"] = float(lo_d + p["tq"] * (hi_eff - lo_d))
            th_max = max(p["theta"] for p in plan)
            Lr = float(min(Lr0, cfg["ramp"]["max_h"] / math.sin(math.radians(th_max))))
            offs_all = []
            for p, v in zip(plan, vs_):
                A = 5.0 / 7.0 * g * math.sin(math.radians(p["theta"]))
                off = sgn * v * (tt - t_in) - 0.5 * A * (tt - t_in) ** 2
                offs_all.append((A, v, off))
            ctx = tt <= t_in - 0.2
            if mode == "down" and min(o_[ctx].min() for _, _, o_ in offs_all) < 0.02:
                _bump(why, "ctx_below"); continue
            q = float(rng.uniform(0.15, 0.85))                          # 단계마다 가능한 s_h 구간에서 같은 분위 (각도가 다르면 높이도 조금 다르다)
            shs = []
            for p, (A, v, off) in zip(plan, offs_all):
                thr = math.radians(p["theta"])
                s_c = r * math.tan(thr / 2)
                lo_s = s_c + 0.03 - off[tt <= t_in].min()
                hi_s = Lr - 1.2 * r - off.max()
                if lo_s > hi_s:
                    shs = None; break
                shs.append(lo_s + q * (hi_s - lo_s))
            if shs is None:
                _bump(why, "ramp_len"); continue
            zs = []
            for p, (A, v, off), s_h in zip(plan, offs_all, shs):
                thr = math.radians(p["theta"])
                m = {"kind": "ramp_accel", "p0": [0.0, 0.0], "theta_deg": float(p["theta"]), "sx": sg, "L": float(Lr), "x_low": 0.0, "y": 0.0,
                     "s0": 0.0, "v0": 0.0, "mu_g": mu_g, "yaw0": 0.0, "omega": 0.0}
                m["x_low"] = float(-_ramp_ctr(m, r, s_h)[0])
                Vk = sgn * v
                v0 = Vk + A * t_in
                s0 = s_h - v0 * t_in + 0.5 * A * t_in * t_in
                m.update(s0=float(s0), v0=float(v0), kind=_ramp_kind(v0, A, T, fps))
                lv.append({**copy.deepcopy(tgt), "motion": m})
                props_lv.append([_wedge(wcol, m, r, Lr)])
                zs.append(s_h * math.sin(thr) + r * math.cos(thr))
            z_h = float(np.mean(zs))
            if max(abs(z - z_c) for z in zs) > 0.4:
                _bump(why, "zc"); continue
        else:
            for p in plan:
                o = copy.deepcopy(tgt)
                if p.get("static"):
                    o["motion"] = {"kind": "flat_static", "p0": [0.0, 0.0], "yaw0": yaw0}
                else:
                    v = p["urel"] * px_h / (ppm * dt)
                    v_ref = min(q_["urel"] for q_ in plan if not q_.get("static")) * px_h / (ppm * dt)
                    a = p["alpha"] * v_ref / t_in
                    p["a"] = float(a)
                    v0 = v - a * t_in
                    if v0 < 0.15 * v:                                   # 프레임 0 에 이미 같은 방향으로 움직이고 있어야 (원인 없는 출발 · 뒤로 감 없음)
                        bad = True; break
                    if a < 0 and v0 / -a < t_in + 0.3:                  # 가려지기 전에 멈추면 안 된다
                        bad = True; break
                    d_in = v0 * t_in + 0.5 * a * t_in * t_in
                    p0 = -ud * d_in
                    o["motion"] = {"kind": "flat_accel", "p0": [float(p0[0]), float(p0[1])], "u": [float(ud[0]), float(ud[1])], "v0": float(v0), "a": float(a),
                                   "yaw0": yaw0, "omega": 0.0}
                lv.append(o); props_lv.append([])
            if bad:
                _bump(why, "flat_v0"); continue
        trk = []
        for o in lv:
            pos, _, Rm = G2.track(o, T, fps, g)
            ctr = center_path(o, pos, Rm)
            E = G2.extent_pts(o, ctr, Rm)
            u_, v_, d_ = cam.to_px(ctr[::STRIDE])
            px = cam.f * G3.s_eff(o) / np.maximum(d_, 1e-3)
            inn = (u_ > 0) & (u_ < 256) & (v_ > 0) & (v_ < 256)
            if (d_ < 0.3).any() or px[inn].min(initial=1e9) < O["min_px"]:
                bad = True; break
            trk.append((ctr, E, px))
        if bad:
            _bump(why, "px"); continue
        # 5 가려지는 구간 (모든 단계의 경로 중 가려지기 시작한 자리에서 L_cov 안쪽) → 가리는 것의 크기
        i_in = int(math.ceil(t_in * fps))
        c_in = [trk[k][0][max(i_in - 1, 0)] for k in range(len(lv))]
        disp = [np.linalg.norm(trk[k][0] - trk[k][0][max(i_in - 1, 0)], axis=1) for k in range(len(lv))]
        if occ == "trapdoor":
            hold = max(_u(rng, V["trap_hold"]), need_fast)
            t_dn = t_in + hold * dt
            H = max(hold * float(rng.uniform(0.8, 1.25)), need_fast)
        else:
            H = max(_u(rng, V["fast_hide"]), need_fast)
        iH = min(T - 1, int(round((t_in + H * dt) * fps)))
        L_cov = max(float(d_[iH]) for d_ in disp)
        if L_cov < 0.02:
            _bump(why, "Lcov"); continue
        segE = np.concatenate([trk[k][1][max(i_in - 1, 0):][disp[k][max(i_in - 1, 0):] <= L_cov] for k in range(len(lv))], 0)
        segC = np.concatenate([trk[k][0][max(i_in - 1, 0):][disp[k][max(i_in - 1, 0):] <= L_cov] for k in range(len(lv))], 0)
        if len(segC) < 2:
            _bump(why, "seg"); continue
        if len(segC) > 60:                                              # 기존 가림 크기 함수는 프레임 수 (64) 안의 번호를 쓴다 → 60 개로 고르게 줄임
            ii = np.unique(np.linspace(0, len(segC) - 1, 60).round().astype(int))
            segC, segE = segC[ii], segE[ii]
        props_all = [p for pl in props_lv for p in pl]
        panels, occ_objs = [], []
        if occ in ("board", "wall", "trapdoor"):
            rr = G3.make_board(rng, cfg, cam, tgt, segC, segE, 0, len(segC) - 1, "wall" if occ == "wall" else "board", props_all)
            if rr is None:
                _bump(why, "occ"); continue
            pan = rr[0][0]
            if occ == "trapdoor":
                pan.update(mode="trapdoor", rise=float(_u(rng, cfg["occluders"]["trapdoor_rise"])), t_up=float(t_in - float(rng.uniform(0.0, 0.05))),
                           t_down=float(t_dn), kind="trapdoor")
            panels.append(pan)
            sub = occ
        else:
            w = dict(cfg["occluders"]["object_static_shapes"])
            if objset == "realistic":
                w.pop("capsule", None)
            oc = G3.make_static_object(rng, cfg, cam, tgt, segC, segE, 0, len(segC) - 1, T, fps, g, objset, [], shape=G3._wchoice(rng, w))
            if oc is None or not G3.distinct(tgt, oc, O["min_hue_diff"]):
                _bump(why, "occ"); continue
            occ_objs.append(oc); sub = oc["kind"]
        if ramp and G2.occ_hits_wedge(props_all, panels, T, fps):
            _bump(why, "occ_wedge"); continue
        # 6 단계마다 검사
        des = G2.panel_boxes(panels, T, fps) + G2.obj_boxes(occ_objs, T, fps, g, 1.0)
        evs = []
        for k, (ctr, E, px) in enumerate(trk):
            o = lv[k]
            if G2._hits_props(ctr, G2.foot_r(o), [], panels, T, fps, clear=0.04):
                bad = True; _bump(why, "hit"); break
            hit = False
            for oc in occ_objs:
                pc, _, Rc = G2.track(oc, T, fps, g)
                if G2.collide(o, ctr, oc, center_path(oc, pc, Rc), 0.04):
                    hit = True
            if hit:
                bad = True; _bump(why, "hit"); break
            if ramp and G2.target_hidden(cam, G2.wedge_boxes(props_lv[k]), E).max() > 0.05:
                bad = True; _bump(why, "wedge_hides"); break
            off = G2.off_frame(cam, E)[::STRIDE]
            ok, ev = timing_ok(G2.target_hidden(cam, des, E)[::STRIDE], off, W, float(V["max_off"]))
            if not ok:
                bad = True; _bump(why, f"late_{plan[k]['key']}"); break
            E_in = ctr[:, None] + 0.55 * (E - ctr[:, None])
            if not timing_ok(G2.target_hidden(cam, des, E_in)[::STRIDE], off, W, float(V["max_off"]))[0]:
                bad = True; _bump(why, "late_in"); break
            evs.append(list(ev))
        if bad:
            continue
        # 7 그 밖 물체 — 모든 단계 경로 · 모든 쐐기와 부딪히지 않고 target 을 가리지 않는다
        world = [(lv[k], trk[k][0]) for k in range(len(lv))]
        for oc in occ_objs:
            pc, _, Rc = G2.track(oc, T, fps, g); world.append((oc, center_path(oc, pc, Rc)))
        n_extra = int(F["count"]) - 1 - int(occ == "object")
        extras, fail = [], False
        wb_all = G2.wedge_boxes(props_all) if ramp else []
        for _ in range(n_extra):
            for _try in range(80):
                d = G3.sample_extra(rng, cfg, cam, T, fps, g, objset)
                if d is None or not G3.distinct(tgt, d, O["min_hue_diff"]):
                    continue
                pd, _, Rd = G2.track(d, T, fps, g); cd = center_path(d, pd, Rd)
                if any(G2.collide(d, cd, ow, cw, 0.12) for ow, cw in world) or G2._hits_props(cd, G2.foot_r(d), props_all, panels, T, fps) \
                        or (ramp and G2.obj_hits_wedge(d, cd, props_all)):
                    continue
                dbox = G2.obj_boxes([d], T, fps, g, 1.05)
                if max(G2.target_hidden(cam, dbox, trk[k][1]).max() for k in range(len(lv))) > 0.02:
                    continue
                ud_, vd_, dd_ = cam.to_px(cd[::STRIDE])
                inn = (ud_ > 0) & (ud_ < 256) & (vd_ > 0) & (vd_ < 256) & (dd_ > 0.3)
                if inn.mean() < O["extra_visible_min"] or np.median(cam.f * G3.s_eff(d) / np.maximum(dd_, 1e-3)) < O["extra_min_px"]:
                    continue
                vis_occ = des + wb_all + G2.obj_boxes([w_[0] for w_ in world], T, fps, g, 1.0)
                hid_d = G2.target_hidden(cam, vis_occ, G2.extent_pts(d, cd, Rd))[::STRIDE]
                if hid_d[inn].max(initial=0.0) > O["extra_max_hidden"]:
                    continue
                extras.append(d); world.append((d, cd)); break
            else:
                fail = True
                break
        if fail:
            _bump(why, "extra"); continue
        # 8 spec — 운동 단계마다 + 빈 장면 (빗면 각도가 단계마다 다르면 각도마다)
        env = G3.sample_env(rng, cfg, F)
        base = {"cam": c, "panels": panels, "env": env, "shadows": bool(O["shadows"]), "T": T, "fps": fps, "g": g, "size": int(cfg["size"]),
                "version": cfg["name"], "split": split, "design": "gen_design_v5mc", "seed": int(sp["seed"]), "family": int(f), "ftype": ftype,
                "attempt": attempt, "stream": stream, "factors": F,
                "motion_family": {"ratio": R, "px_per_m": float(ppm), "target_px_hide": float(px_h), "t_in": float(t_in), "s_in": s_in, "phi": phi,
                                  "fast_hide": float(H), "L_cov_m": float(L_cov), "t_down": float(t_dn) if occ == "trapdoor" else None,
                                  "levels": [{k_: (float(v_) if isinstance(v_, (int, float, np.floating)) and not isinstance(v_, bool) else v_) for k_, v_ in p.items()} for p in plan],
                                  "ramp_mode": F.get("ramp_mode")},
                "cell": {"place": sit["place"], "objset": objset, "camera": sit["camera"], "occ": occ, "occ_sub": sub, "count": int(F["count"]),
                         "target_shape": shape, "target_kind": tgt.get("kind"), "ftype": ftype, "ramp_mode": F.get("ramp_mode") if ramp else None}}
        out = {}
        for k, p in enumerate(plan):
            s2 = copy.deepcopy(base)
            mot = lv[k]["motion"]
            vk = p["urel"] * px_h / (ppm * dt)
            s2.update(objects=[lv[k]] + extras + occ_objs, props=props_lv[k], level=p["key"], level_info=s2["motion_family"]["levels"][k],
                      event_samples=evs[k], target_px_median=float(np.median(trk[k][2])), v_hide_mps=float(vk), motion_kind=mot["kind"])
            if physics_check(s2):
                bad = True; _bump(why, "phys"); break
            out[p["key"]] = s2
        if bad:
            continue
        thetas = sorted({round(p["theta"], 4) for p in plan}) if ramp else [None]
        for bi, th in enumerate(thetas):
            sB = copy.deepcopy(base)
            kk = [k for k, p in enumerate(plan) if (not ramp) or round(p["theta"], 4) == th][0]
            sB.update(objects=list(extras) + list(occ_objs), props=props_lv[kk], level="B" if len(thetas) == 1 else f"B{bi}", ghost_target=lv[kk],
                      event_samples=None, b_for=[plan[k]["key"] for k, p in enumerate(plan) if (not ramp) or round(p["theta"], 4) == th])
            out[sB["level"]] = sB
        return out
    raise RuntimeError(f"묶음 {split} {f} 설계 실패 ({sit_key(sit)}, {ftype}, {F}, 줄기 {stream}) 이유 {why}")


def sample_clip(cfg, split, idx, max_tries=1500, streams=4):
    raise RuntimeError(f"v5mc: {split} {idx} 설계 파일 없음 (design_families_v5mc.py 가 쓰지 않은 번호)")
