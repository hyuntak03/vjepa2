#!/usr/bin/env python3
"""BlenderTrack v4_velocity_control (v4vc) 설계 — 같은 장면, 다른 속도 (2026-11-02, 사용자 요청).

왜
  v3 · v3t 로 학습한 predictor 는 같은 장면에서 속도만 바꿔도 예측 자리가 따라가지 않았다 (속도 따라감 0.00, Blender 안에서도) [측정].
  v3 에서는 장면 배치가 다시 나올 자리를 속도보다 잘 알려 준다 (같은 꼴 장면 평균 경로 오차 1.06 토큰 < 등속 외삽 2.27 토큰) [측정].
  → 장면 (카메라 · 환경 · 가림막 · target 생김새 · 방향 · 그 밖 물체) 은 고정하고 속도만 K 단계로 바꾼 묶음 (family) 을 만든다.
    같은 묶음 안에서는 장면이 답을 정하지 못하고, 문맥에서 읽은 속도만이 다시 나올 때 · 자리를 정한다 [이론].

설계 (configs/v4vc.yaml)
  · late 가림: 모든 단계에서 target 은 같은 자리 (d = 0) 를 같은 시각 t_in = (s_in − φ) · 0.1 s 에 지나 완전히 가려지기 시작한다.
    그래서 문맥 끝 직전 보이는 자리는 단계마다 거의 같고, 다른 것은 그 전까지의 움직임 (속도) 뿐이다.
  · 가려지는 길이 L 은 묶음에 하나. 판 · 낮은 벽 · 서 있는 물체: 빠른 단계는 일찍 · 멀리, 느린 단계는 늦게 · 가까이 옆으로 빠져나온다.
    trapdoor (VOBench late 꼴): 판이 정해진 시각에 누우며 드러난다 — 다시 보이는 시각은 거의 같고 자리만 속도에 따라 다르다.
  · 속도 단계 (K 4, 가장 느린 : 가장 빠른 = 1 : 2) = 화면 속력 u_k = u0 · ρ^k, u0 = target 화면 지름 × 0.105–0.135 (가려지기 시작하는 자리에서, px / 샘플, 샘플 = 0.1 s), k = 0..K−1. 세계 속도 v_k = u_k / (px/m · 0.1 s).
  · 운동: 공은 구르고 (구름 저항 작게) 상자 · 통은 미끄러진다 (매끈한 바닥). 가장 느린 단계도 창 끝까지 속력 50 % 이상.
  · 그 밖 물체 · 가림막은 모든 단계에서 같고, 모든 단계의 target 경로와 부딪히거나 target 을 가리면 안 된다.
  · 빈 장면 (팔 B) = 같은 장면에서 target 만 뺌 (묶음마다 하나).
렌더 번호 (gen_render.py 는 design/<id>.json 을 읽는다. 쓰지 않는 번호는 .fail)
  fam 분할 (train_fam · test · probe): 묶음 f → 번호 f·(K+1) + k (k < K 속도 단계, k = K 빈 장면)
  ind 분할 (train_ind): 장면 s → 번호 2s (단계 j % K 하나만, j = 상황 안 번호) · 2s+1 (빈 장면, j % K == 0 인 장면만 — 빈 장면 수를 fam 과 맞춘다)
검사: 단계마다 late 정의 (v3 timing_ok, 화면 밖 상한만 max_off) · 안쪽 extent 로도 late · 최소 화면 크기 · 물리 검사 (gen_design_v2.physics_check).
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
load_cfg = G2.load_cfg
track = G2.track
center_path = G2.center_path
tracks = G2.tracks
clip_id = G2.clip_id
_u = G2._u
physics_check = G2.physics_check


# ----------------------------------------------------------------------------- 상황 · 요인
def situations(cfg):
    C = cfg["cells"]
    ex = set(C.get("exclude", []))
    pl = C.get("place", ["flat"])[0]                                    # 11-02 2 차: 빗면 설정 (configs/v4vc_ramp.yaml) 은 place: [ramp]
    return [dict(objset=o, camera=c, occ=oc, place=pl) for o in C["objset"] for c in C["camera"] for oc in C["occ"] if f"{o}|{c}|{oc}" not in ex]


def sit_key(s):
    return f"{s['objset']}|{s['camera']}|{s['occ']}" + ("|ramp" if s.get("place") == "ramp" else "")


def factor_levels(sit, cfg):
    V = cfg["velocity"]
    F = {"kind": G3.target_kinds(sit["objset"], sit.get("place", "flat"), "moving"), "hue": list(range(8)), "size": [0, 1, 2], "dir": [-1, 1],
         "hide": list(V["hide"]), "ubin": [0, 1, 2], "count": [2, 3] if sit["occ"] == "object" else [1, 2, 3],
         "indoor": [True, False], "floor": [0, 1, 2], "side_sign": [-1, 1]}
    if sit.get("place") == "ramp":
        # 내려감 (가속) · 올라감 (감속, 되돌아오기도 함). 11-02 시험: 올라감은 판 · 벽 뒤에서 되돌아 나오는 것이 창 안에 안 맞아 0/19 → trapdoor 에만 (2/3)
        F["ramp_mode"] = ["down", "up", "up"] if sit["occ"] == "trapdoor" else ["down"]
    return F


_SCHED = {}


def schedule(cfg, split, si, sit, N):
    """상황 si 의 장면 N 개 요인표 (층화 뒤섞기: 각 요인 수준 개수가 ±1 안에서 같다) — gen_design_v3.schedule 과 같은 방식."""
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
    """묶음 (장면) 번호 f → (상황, 상황 번호, 상황 안 번호, 요인)."""
    S = situations(cfg)
    si, j = f % len(S), f // len(S)
    per = (int(cfg["splits"][split]["candidates"]) + len(S) - 1) // len(S)
    return S[si], si, j, schedule(cfg, split, si, S[si], per)[j]


def is_ind(split):
    return split.endswith("_ind")


def render_slots(cfg, split, f):
    """묶음 f 에서 렌더할 (번호, 단계) 목록. 단계 = 0..K−1 또는 'B'."""
    K = int(cfg["velocity"]["levels"])
    if is_ind(split):                                                   # 단계 = 상황 안 번호 j 의 나머지 (f % K 로 하면 상황 (f % 16) 과 단계가 묶인다)
        j = f // len(situations(cfg))
        out = [(2 * f, j % K)]
        if j % K == 0:
            out.append((2 * f + 1, "B"))
        return out
    return [(f * (K + 1) + k, k) for k in range(K)] + [(f * (K + 1) + K, "B")]


def all_slots(cfg, split, f):
    K = int(cfg["velocity"]["levels"])
    return [2 * f, 2 * f + 1] if is_ind(split) else [f * (K + 1) + k for k in range(K + 1)]


# ----------------------------------------------------------------------------- late 검사 (v3 timing_ok, 화면 밖 상한만 바꿈)
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


# ----------------------------------------------------------------------------- 묶음 하나
def design_family(cfg, split, f, max_tries=400, streams=3):
    last = None
    for st in range(streams):
        try:
            if family_of(cfg, split, f)[0].get("place") == "ramp":
                return _design_ramp_family(cfg, split, f, max_tries, st)
            return _design_family(cfg, split, f, max_tries, st)
        except RuntimeError as e:
            last = e
    raise RuntimeError(str(last))


def _design_family(cfg, split, f, max_tries, stream):
    sp = cfg["splits"][split]
    sit, si, j, F = family_of(cfg, split, f)
    rng = np.random.default_rng([int(sp["seed"]), 200000 + f, stream])
    T, fps, g = int(cfg["frames"]), float(cfg["fps"]), float(cfg["g"])
    O, V, W = cfg["objects"], cfg["velocity"], cfg["timing"]
    K, rho = int(V["levels"]), float(V["ratio_step"])
    dt = STRIDE / fps                                                   # 샘플 간격 (s)
    s_in = int(F["hide"])
    objset = sit["objset"]
    why = {}
    for attempt in range(max_tries):
        # 1 target 생김새 (요인표의 kind · size · hue) · 방향
        tgt = G3.new_object(rng, cfg, objset, O["moving_shapes"], "target", True, dict(F, speed=1))
        roll = tgt["shape"] == "sphere"
        hd = math.radians(_u(rng, O["heading_deg"])); sg = float(F["dir"])
        ud = np.array([sg * math.cos(hd), math.sin(hd)])
        yaw0 = float(rng.uniform(0, 2 * np.pi)) if roll else float(math.atan2(ud[1], ud[0]))
        z_c = 0.5 * tgt["s"] * DIMS[tgt["shape"]][2]
        # 2 카메라 (원점 = 가려지는 구간 가운데)
        c = G3.sample_camera(rng, cfg, sit["camera"], tgt, z_c, F)
        if c is None:
            why["cam"] = why.get("cam", 0) + 1; continue
        cam = G3.cam_obj(c)
        P = np.array([[-0.05 * ud[0], -0.05 * ud[1], z_c], [0.05 * ud[0], 0.05 * ud[1], z_c]])
        uu, vv, dd = cam.to_px(P)
        if (dd < 0.3).any():
            continue
        ppm = math.hypot(uu[1] - uu[0], vv[1] - vv[0]) / 0.1           # 화면 px / 경로 m (가려지는 자리)
        if ppm < 25:
            why["ppm"] = why.get("ppm", 0) + 1; continue
        # 3 속도 단계 · 가려지는 길이 · 감속
        px_h = cam.f * G3.s_eff(tgt) / float(cam.to_px(np.array([[0.0, 0.0, z_c]]))[2][0])   # 가려지는 자리의 target 화면 지름
        u0 = px_h * G3._bin(rng, V["u0_rel"], int(F["ubin"]), 3)
        us = [u0 * rho ** k for k in range(K)]
        vs = [u / (ppm * dt) for u in us]
        phi = float(rng.uniform(0.25, 0.55))
        t_in = (s_in - phi) * dt
        need_fast = (16.3 - s_in + phi)                                 # 가장 빠른 단계도 샘플 16 까지 완전히 가려짐 (late 끝 ≥ 16)
        occ = sit["occ"]
        if occ == "trapdoor":
            hold = max(_u(rng, V["trap_hold"]), need_fast)
            hf = hold
            t_dn = t_in + hold * dt
            L = vs[-1] * max(hold * float(rng.uniform(0.8, 1.25)), need_fast) * dt
        else:
            hf = max(_u(rng, V["fast_hide"]), need_fast)
            L = vs[-1] * hf * dt
        a = _u(rng, V["mu_roll"] if roll else V["mu_slide"]) * g
        t_rest = (T - 1) / fps - t_in
        a = max(min(a, (1 - float(V["keep_speed"])) * vs[0] / t_rest), 1e-4)
        entry = -0.5 * L * ud

        def level_obj(k):
            o = copy.deepcopy(tgt)
            v0 = vs[k] + a * t_in
            d_in = v0 * t_in - 0.5 * a * t_in * t_in
            p0 = entry - ud * d_in
            o["motion"] = {"kind": "flat_decel", "p0": [float(p0[0]), float(p0[1])], "v": [float(ud[0] * v0), float(ud[1] * v0)],
                           "mu": float(a / g), "yaw0": yaw0, "omega": 0.0}
            return o

        lv = [level_obj(k) for k in range(K)]
        trk = []
        bad = False
        for o in lv:
            pos, _, R = track(o, T, fps, g)
            ctr = center_path(o, pos, R)
            E = G2.extent_pts(o, ctr, R)
            u_, v_, d_ = cam.to_px(ctr[::STRIDE])
            px = cam.f * G3.s_eff(o) / np.maximum(d_, 1e-3)
            inn = (u_ > 0) & (u_ < 256) & (v_ > 0) & (v_ < 256)
            if (d_ < 0.3).any() or px[inn].min(initial=1e9) < O["min_px"]:
                bad = True; break
            trk.append((ctr, E, px))
        if bad:
            why["px"] = why.get("px", 0) + 1; continue
        # 4 가려지는 구간 (경로 위 d ∈ [0, L]) 의 target extent — 가리는 것의 크기를 정한다 (모든 단계가 같은 길을 지난다)
        n = 25
        seg = copy.deepcopy(tgt)
        seg["motion"] = {"kind": "flat_const", "p0": [float(entry[0]), float(entry[1])], "v": [float(ud[0] * L / (n - 1)), float(ud[1] * L / (n - 1))],
                         "yaw0": yaw0, "omega": 0.0}
        ps, _, Rs = track(seg, n, 1.0, g)
        cs = center_path(seg, ps, Rs)
        Es = G2.extent_pts(seg, cs, Rs)
        # 5 가리는 것
        panels, props, occ_objs = [], [], []
        if occ in ("board", "wall", "trapdoor"):
            r = G3.make_board(rng, cfg, cam, tgt, cs, Es, 0, n - 1, "wall" if occ == "wall" else "board", [])
            if r is None:
                why["occ"] = why.get("occ", 0) + 1; continue
            pan = r[0][0]
            if occ == "trapdoor":
                rise = _u(rng, cfg["occluders"]["trapdoor_rise"])
                pan.update(mode="trapdoor", rise=float(rise), t_up=float(t_in - float(rng.uniform(0.0, 0.05))), t_down=float(t_dn), kind="trapdoor")
            panels.append(pan)
            sub = occ
        else:
            w = dict(cfg["occluders"]["object_static_shapes"])
            if objset == "realistic":
                w.pop("capsule", None)
            oc = G3.make_static_object(rng, cfg, cam, tgt, cs, Es, 0, n - 1, T, fps, g, objset, [], shape=G3._wchoice(rng, w))
            if oc is None or not G3.distinct(tgt, oc, O["min_hue_diff"]):
                why["occ"] = why.get("occ", 0) + 1; continue
            occ_objs.append(oc)
            sub = oc["kind"]
        # 6 단계마다: 부딪힘 · late 타이밍 (바깥 · 안쪽 extent)
        des = G2.panel_boxes(panels, T, fps) + G2.obj_boxes(occ_objs, T, fps, g, 1.0)
        evs = []
        for k, (ctr, E, px) in enumerate(trk):
            o = lv[k]
            if G2._hits_props(ctr, G2.foot_r(o), [], panels, T, fps, clear=0.04):
                bad = True; why["hit"] = why.get("hit", 0) + 1; break
            for oc in occ_objs:
                pc, _, Rc = track(oc, T, fps, g)
                if G2.collide(o, ctr, oc, center_path(oc, pc, Rc), 0.04):
                    bad = True
            if bad:
                why["hit"] = why.get("hit", 0) + 1; break
            off = G2.off_frame(cam, E)[::STRIDE]
            ok, ev = timing_ok(G2.target_hidden(cam, des, E)[::STRIDE], off, W, float(V["max_off"]))
            if not ok:
                bad = True; why[f"late{k}"] = why.get(f"late{k}", 0) + 1; break
            E_in = ctr[:, None] + 0.55 * (E - ctr[:, None])
            if not timing_ok(G2.target_hidden(cam, des, E_in)[::STRIDE], off, W, float(V["max_off"]))[0]:
                bad = True; why[f"late_in{k}"] = why.get(f"late_in{k}", 0) + 1; break
            evs.append(list(ev))
        if bad:
            continue
        world = [(lv[k], trk[k][0]) for k in range(K)]
        for oc in occ_objs:
            pc, _, Rc = track(oc, T, fps, g); world.append((oc, center_path(oc, pc, Rc)))
        # 7 그 밖 물체 (총수 = target 1 + 가리는 물체 + 그 밖)
        n_extra = int(F["count"]) - 1 - int(occ == "object")
        extras, fail = [], False
        for _ in range(n_extra):
            for _try in range(80):
                d = G3.sample_extra(rng, cfg, cam, T, fps, g, objset)
                if d is None or not G3.distinct(tgt, d, O["min_hue_diff"]):
                    continue
                pd, _, Rd = track(d, T, fps, g); cd = center_path(d, pd, Rd)
                if any(G2.collide(d, cd, ow, cw, 0.12) for ow, cw in world) or G2._hits_props(cd, G2.foot_r(d), [], panels, T, fps):
                    continue
                dbox = G2.obj_boxes([d], T, fps, g, 1.05)
                if max(G2.target_hidden(cam, dbox, trk[k][1]).max() for k in range(K)) > 0.02:
                    continue
                ud_, vd_, dd_ = cam.to_px(cd[::STRIDE])
                inn = (ud_ > 0) & (ud_ < 256) & (vd_ > 0) & (vd_ < 256) & (dd_ > 0.3)
                if inn.mean() < O["extra_visible_min"] or np.median(cam.f * G3.s_eff(d) / np.maximum(dd_, 1e-3)) < O["extra_min_px"]:
                    continue
                vis_occ = des + G2.obj_boxes([w_[0] for w_ in world], T, fps, g, 1.0)
                hid_d = G2.target_hidden(cam, vis_occ, G2.extent_pts(d, cd, Rd))[::STRIDE]
                if hid_d[inn].max(initial=0.0) > O["extra_max_hidden"]:
                    continue
                extras.append(d); world.append((d, cd)); break
            else:
                fail = True
                break
        if fail:
            why["extra"] = why.get("extra", 0) + 1; continue
        # 8 spec (단계마다 + 빈 장면)
        env = G3.sample_env(rng, cfg, F)
        base = {"cam": c, "panels": panels, "props": props, "env": env, "shadows": bool(O["shadows"]), "T": T, "fps": fps, "g": g,
                "size": int(cfg["size"]), "version": cfg["name"], "split": split, "design": "gen_design_v4vc", "seed": int(sp["seed"]),
                "family": int(f), "attempt": attempt, "stream": stream, "factors": F,
                "velocity": {"u_px_per_sample": [float(x) for x in us], "v_mps": [float(x) for x in vs], "px_per_m": float(ppm), "target_px_hide": float(px_h), "decel": float(a),
                             "t_in": float(t_in), "s_in": s_in, "fast_hide": float(hf), "L_m": float(L), "phi": phi,
                             "t_down": float(t_dn) if occ == "trapdoor" else None},
                "cell": {"objset": objset, "camera": sit["camera"], "occ": occ, "occ_sub": sub, "count": int(F["count"]),
                         "target_shape": tgt["shape"], "target_kind": tgt.get("kind")}}
        specs = []
        for k in range(K):
            s_ = copy.deepcopy(base)
            s_.update(objects=[lv[k]] + extras + occ_objs, level=k, event_samples=evs[k], target_px_median=float(np.median(trk[k][2])))
            if G2.physics_check(s_):
                bad = True; break
            specs.append(s_)
        if bad:
            why["phys"] = why.get("phys", 0) + 1; continue
        sB = copy.deepcopy(base)
        sB.update(objects=extras + occ_objs, level="B", ghost_target=lv[0], event_samples=None)
        specs.append(sB)
        return specs
    raise RuntimeError(f"묶음 {split} {f} 설계 실패 ({sit_key(sit)}, {F}, 줄기 {stream}) 이유 {why}")


# ----------------------------------------------------------------------------- 빗면 묶음 (11-02 2 차, 사용자: 빗면 · trapdoor 도 넣어서)
def _ramp_ctr(m, r, s):
    """경사 거리 s 에서 구 중심 (gen_design_v2.track 의 구 식과 같다)."""
    th, sx = math.radians(m["theta_deg"]), m["sx"]
    return np.array([m["x_low"] + sx * (s * math.cos(th) - r * math.sin(th)), m["y"], s * math.sin(th) + r * math.cos(th)])


def _ramp_kind(v0, A, T, fps):
    """구간 [0, T) 에서 경사 속도 v(t) = v0 − A t 의 부호가 바뀌면 되돌아옴, 아니면 내려감 (가속) · 올라감 (감속)."""
    t_end = (T - 1) / fps
    if v0 > 0 and v0 / A < t_end:
        return "ramp_reverse"
    return "ramp_accel" if v0 <= 0 else "ramp_up"


def _design_ramp_family(cfg, split, f, max_tries, stream):
    """같은 빗면 · 같은 장면에서 공의 속도만 K 단계. 내려감 = 가려지는 자리에서 아래로 굴러가는 중 (가속), 올라감 = 위로 굴러가는 중 (감속, 가려진 뒤 되돌아오기도 함).
    모든 단계가 같은 경사 거리 s_h 를 같은 시각 t_in 에 지난다 (late). 문맥 시작 (프레임 0) 의 자리 · 속도는 단계마다 물리로 거꾸로 푼다
    (s(t) = s_h + V (t − t_in) − ½ A (t − t_in)², A = 5/7 g sinθ) — 느린 내려감 단계는 문맥 안에서 위로 굴러 올라갔다 되돌아오는 꼴이 된다 (원인 없는 출발 없음).
    가리는 것은 판 · 낮은 벽 · trapdoor (빗면 앞 바닥에 선다)."""
    sp = cfg["splits"][split]
    sit, si, j, F = family_of(cfg, split, f)
    rng = np.random.default_rng([int(sp["seed"]), 200000 + f, stream])
    T, fps, g = int(cfg["frames"]), float(cfg["fps"]), float(cfg["g"])
    O, V, W, RP = cfg["objects"], cfg["velocity"], cfg["timing"], cfg["ramp"]
    K, rho = int(V["levels"]), float(V["ratio_step"])
    dt = STRIDE / fps
    s_in = int(F["hide"])
    objset, mode, occ = sit["objset"], F["ramp_mode"], sit["occ"]
    why = {}
    for attempt in range(max_tries):
        lk, s_, shape = G3.look_fixed(rng, cfg, objset, F)
        tgt = {"shape": "sphere", "s": float(s_), "role": "target", **lk}
        r = 0.5 * tgt["s"]
        th = _u(rng, RP["deg_" + F["ramp_mode"]]); thr = math.radians(th)
        Lr = float(min(_u(rng, RP["len"]), RP["max_h"] / math.sin(thr)))
        sx = float(F["dir"])
        A = 5.0 / 7.0 * g * math.sin(thr)
        s_c = r * math.tan(thr / 2)
        s_h = 0.5 * Lr                                                  # 임시 (카메라 · 화면 척도용) — 속도가 정해진 뒤 가능한 구간에서 다시 고른다
        m0 = {"kind": "ramp_accel", "p0": [0.0, 0.0], "theta_deg": float(th), "sx": sx, "L": Lr, "x_low": 0.0, "y": 0.0,
              "s0": float(s_h), "v0": 0.0, "mu_g": float(_u(rng, (0.01, 0.03))), "yaw0": 0.0, "omega": 0.0}
        m0["x_low"] = float(-_ramp_ctr(m0, r, s_h)[0])                  # 가려지는 자리 (s_h 의 구 중심) 가 x = 0
        z_c = float(_ramp_ctr(m0, r, s_h)[2])
        c = G3.sample_camera(rng, cfg, sit["camera"], tgt, z_c, F)
        if c is None:
            why["cam"] = why.get("cam", 0) + 1; continue
        cam = G3.cam_obj(c)
        P = np.stack([_ramp_ctr(m0, r, s_h - 0.05), _ramp_ctr(m0, r, s_h + 0.05)])
        uu, vv, dd = cam.to_px(P)
        if (dd < 0.3).any():
            continue
        ppm = math.hypot(uu[1] - uu[0], vv[1] - vv[0]) / 0.1
        if ppm < 25:
            why["ppm"] = why.get("ppm", 0) + 1; continue
        px_h = cam.f * G3.s_eff(tgt) / float(cam.to_px(np.array([[0.0, 0.0, z_c]]))[2][0])
        u0 = px_h * G3._bin(rng, V["u0_rel"], int(F["ubin"]), 3)
        us = [u0 * rho ** k for k in range(K)]
        vs = [u / (ppm * dt) for u in us]
        phi = float(rng.uniform(0.25, 0.55))
        t_in = (s_in - phi) * dt
        need_fast = (16.3 - s_in + phi)
        sg = -1.0 if mode == "down" else 1.0
        # s_h 고르기: 단계마다 s(t) − s_h = V (t − t_in) − ½ A (t − t_in)² 는 s_h 와 무관하다 → 모든 단계가
        #   (가) 창 안에서 꼭대기를 넘지 않고 (나) 문맥 동안 아래 끝 위에 있고 (다) 내려감은 문맥 동안 가려지는 자리 위쪽에 머무는 s_h 구간
        tt = np.arange(T) / fps
        ctx = tt <= t_in - 0.2
        offs = [sg * vs[k] * (tt - t_in) - 0.5 * A * (tt - t_in) ** 2 for k in range(K)]
        if mode == "down" and min(o_[ctx].min() for o_ in offs) < 0.02:     # 문맥에서 이미 가려지는 쪽 (아래) 으로 들어간 적이 있으면 가림이 두 번
            why["ctx_below"] = why.get("ctx_below", 0) + 1; continue
        lo = s_c + 0.03 - min(o_[tt <= t_in].min() for o_ in offs)
        hi = Lr - 1.2 * r - max(o_.max() for o_ in offs)
        if lo > hi:
            why["ramp_len"] = why.get("ramp_len", 0) + 1; continue
        s_h = float(rng.uniform(lo, hi))
        m0["x_low"] = 0.0
        m0["x_low"] = float(-_ramp_ctr(m0, r, s_h)[0])
        if abs(float(_ramp_ctr(m0, r, s_h)[2]) - z_c) > 0.35:           # 카메라는 임시 높이 z_c 를 본다 — 너무 멀면 다시
            why["zc"] = why.get("zc", 0) + 1; continue
        # 단계마다 운동 (프레임 0 의 자리 · 속도를 거꾸로 푼다)
        lv, bad = [], False
        for k in range(K):
            Vk = sg * vs[k]
            v0 = Vk + A * t_in
            s0 = s_h - v0 * t_in + 0.5 * A * t_in * t_in
            st = s0 + v0 * tt - 0.5 * A * tt * tt
            ctxs = st[: int(round(t_in * fps)) + 1]
            if st.max() > Lr - 1.2 * r or ctxs.min() < s_c + 0.03:
                bad = True; break
            m = dict(m0, s0=float(s0), v0=float(v0), kind=_ramp_kind(v0, A, T, fps))
            lv.append({**copy.deepcopy(tgt), "motion": m})
        if bad:
            why["ramp_len"] = why.get("ramp_len", 0) + 1; continue
        trk = []
        for o in lv:
            pos, _, R = track(o, T, fps, g)
            ctr = center_path(o, pos, R)
            E = G2.extent_pts(o, ctr, R)
            u_, v_, d_ = cam.to_px(ctr[::STRIDE])
            px = cam.f * G3.s_eff(o) / np.maximum(d_, 1e-3)
            inn = (u_ > 0) & (u_ < 256) & (v_ > 0) & (v_ < 256)
            if (d_ < 0.3).any() or px[inn].min(initial=1e9) < O["min_px"]:
                bad = True; break
            trk.append((ctr, E, px))
        if bad:
            why["px"] = why.get("px", 0) + 1; continue
        # 쐐기 (gen_design_v3.new_ramp_target 과 같은 꼴, x_low 만큼 옮김)
        Wd, thick = max(0.35, 3.0 * r), 0.12
        ph = -sx * thr
        top_mid = np.array([sx * (Lr / 2) * math.cos(thr) + m0["x_low"], 0.0, (Lr / 2) * math.sin(thr)])
        loc = top_mid - np.array([thick * math.sin(ph), 0.0, thick * math.cos(ph)])
        col = G3._jit(rng, G3.WOOD[rng.integers(len(G3.WOOD))]) if rng.uniform() < 0.6 else G3._jit(rng, [(0.85, 0.85, 0.82), (0.35, 0.45, 0.6)][rng.integers(2)], 0.05)
        props = [{"role": "ramp", "solid": "wedge", "loc": loc.tolist(), "rot": [0.0, ph, 0.0], "scale": [Lr, Wd, thick], "color": col}]
        # 가려지는 구간 = 가장 빠른 단계가 t_in 부터 D 동안 지나는 길 (모든 단계가 같은 길 위에 있다; 느린 단계는 그 안에서 머물거나 되돌아온다)
        if occ == "trapdoor":
            hold = max(_u(rng, V["trap_hold"]), need_fast)
            t_dn = t_in + hold * dt
            D = max(hold * float(rng.uniform(0.8, 1.25)), need_fast) * dt
            hf = hold
        else:
            hf = max(_u(rng, V["fast_hide"]), need_fast)
            D = hf * dt
        n = 25
        fast = lv[-1]
        seg = copy.deepcopy(fast)
        sf = fast["motion"]["s0"] + fast["motion"]["v0"] * t_in - 0.5 * A * t_in * t_in
        seg["motion"] = dict(fast["motion"], s0=float(sf), v0=float(fast["motion"]["v0"] - A * t_in))
        ps, _, Rs = track(seg, n, (n - 1) / D, g)
        cs = center_path(seg, ps, Rs)
        Es = G2.extent_pts(seg, cs, Rs)
        r_ = G3.make_board(rng, cfg, cam, tgt, cs, Es, 0, n - 1, "wall" if occ == "wall" else "board", props)
        if r_ is None:
            why["occ"] = why.get("occ", 0) + 1; continue
        pan = r_[0][0]
        if occ == "trapdoor":
            pan.update(mode="trapdoor", rise=float(_u(rng, cfg["occluders"]["trapdoor_rise"])), t_up=float(t_in - float(rng.uniform(0.0, 0.05))),
                       t_down=float(t_dn), kind="trapdoor")
        panels = [pan]
        if G2.occ_hits_wedge(props, panels, T, fps):
            why["occ_wedge"] = why.get("occ_wedge", 0) + 1; continue
        des = G2.panel_boxes(panels, T, fps)
        others = G2.wedge_boxes(props)
        evs = []
        for k, (ctr, E, px) in enumerate(trk):
            o = lv[k]
            if G2._hits_props(ctr, G2.foot_r(o), [], panels, T, fps, clear=0.04):
                bad = True; why["hit"] = why.get("hit", 0) + 1; break
            if G2.target_hidden(cam, others, E).max() > 0.05:          # 쐐기가 공을 가리면 안 된다 (낮은 카메라)
                bad = True; why["wedge_hides"] = why.get("wedge_hides", 0) + 1; break
            off = G2.off_frame(cam, E)[::STRIDE]
            ok, ev = timing_ok(G2.target_hidden(cam, des, E)[::STRIDE], off, W, float(V["max_off"]))
            if not ok:
                bad = True; why[f"late{k}"] = why.get(f"late{k}", 0) + 1; break
            E_in = ctr[:, None] + 0.55 * (E - ctr[:, None])
            if not timing_ok(G2.target_hidden(cam, des, E_in)[::STRIDE], off, W, float(V["max_off"]))[0]:
                bad = True; why[f"late_in{k}"] = why.get(f"late_in{k}", 0) + 1; break
            evs.append(list(ev))
        if bad:
            continue
        world = [(lv[k], trk[k][0]) for k in range(K)]
        n_extra = int(F["count"]) - 1
        extras, fail = [], False
        for _ in range(n_extra):
            for _try in range(80):
                d = G3.sample_extra(rng, cfg, cam, T, fps, g, objset)
                if d is None or not G3.distinct(tgt, d, O["min_hue_diff"]):
                    continue
                pd, _, Rd = track(d, T, fps, g); cd = center_path(d, pd, Rd)
                if any(G2.collide(d, cd, ow, cw, 0.12) for ow, cw in world) or G2._hits_props(cd, G2.foot_r(d), props, panels, T, fps) \
                        or G2.obj_hits_wedge(d, cd, props):
                    continue
                dbox = G2.obj_boxes([d], T, fps, g, 1.05)
                if max(G2.target_hidden(cam, dbox, trk[k][1]).max() for k in range(K)) > 0.02:
                    continue
                ud_, vd_, dd_ = cam.to_px(cd[::STRIDE])
                inn = (ud_ > 0) & (ud_ < 256) & (vd_ > 0) & (vd_ < 256) & (dd_ > 0.3)
                if inn.mean() < O["extra_visible_min"] or np.median(cam.f * G3.s_eff(d) / np.maximum(dd_, 1e-3)) < O["extra_min_px"]:
                    continue
                vis_occ = des + others + G2.obj_boxes([w_[0] for w_ in world], T, fps, g, 1.0)
                hid_d = G2.target_hidden(cam, vis_occ, G2.extent_pts(d, cd, Rd))[::STRIDE]
                if hid_d[inn].max(initial=0.0) > O["extra_max_hidden"]:
                    continue
                extras.append(d); world.append((d, cd)); break
            else:
                fail = True
                break
        if fail:
            why["extra"] = why.get("extra", 0) + 1; continue
        env = G3.sample_env(rng, cfg, F)
        base = {"cam": c, "panels": panels, "props": props, "env": env, "shadows": bool(O["shadows"]), "T": T, "fps": fps, "g": g,
                "size": int(cfg["size"]), "version": cfg["name"], "split": split, "design": "gen_design_v4vc", "seed": int(sp["seed"]),
                "family": int(f), "attempt": attempt, "stream": stream, "factors": F,
                "velocity": {"u_px_per_sample": [float(x) for x in us], "v_mps": [float(x) for x in vs], "px_per_m": float(ppm), "target_px_hide": float(px_h),
                             "accel_mps2": float(A), "theta_deg": float(th), "ramp_mode": mode, "s_h": float(s_h), "t_in": float(t_in), "s_in": s_in,
                             "fast_hide": float(hf), "phi": phi, "t_down": float(t_dn) if occ == "trapdoor" else None,
                             "kinds": [o["motion"]["kind"] for o in lv]},
                "cell": {"objset": objset, "camera": sit["camera"], "occ": occ, "occ_sub": occ, "place": "ramp", "ramp_mode": mode, "count": int(F["count"]),
                         "target_shape": "sphere", "target_kind": tgt.get("kind")}}
        specs = []
        for k in range(K):
            s2 = copy.deepcopy(base)
            s2.update(objects=[lv[k]] + extras, level=k, event_samples=evs[k], target_px_median=float(np.median(trk[k][2])))
            if G2.physics_check(s2):
                bad = True; why["phys"] = why.get("phys", 0) + 1; break
            specs.append(s2)
        if bad:
            continue
        sB = copy.deepcopy(base)
        sB.update(objects=list(extras), level="B", ghost_target=lv[0], event_samples=None)
        specs.append(sB)
        return specs
    raise RuntimeError(f"빗면 묶음 {split} {f} 설계 실패 ({sit_key(sit)}, {F}, 줄기 {stream}) 이유 {why}")


def sample_clip(cfg, split, idx, max_tries=1500, streams=4):
    """gen_render.py 호환: 설계는 design_families_v4vc.py 가 미리 쓴다. 미리 쓴 설계가 없으면 이 번호는 렌더하지 않는다."""
    raise RuntimeError(f"v4vc: {split} {idx} 설계 파일 없음 (design_families_v4vc.py 가 쓰지 않은 번호 = 쓰지 않는 번호)")
