#!/usr/bin/env python3
"""BlenderTrack v5id 설계 (2026-11-09) — v4b 묶음 (긴 가림, 속도 단계 K) 에 target 정체 (모양 × 색) 변형을 더한다.

사용자 (11-09): "general data (0.2M 정도), 내용 구성은 v4bc4 랑 비슷한데 shape * color 에 대해서도 다양하게 compositional 하게.
                 우린 지금 속도만 그렇게 했으니까. 위반도 color * shape 위반 만들어봐."
지금 v4b 묶음의 A 영상 K 편은 모양 · 색 · 크기가 같고 속도만 다르다 [측정, 500 묶음]. 그래서 묶음 안 짝은 속도 바뀜뿐이었다.

묶음 하나 = 렌더 번호 S = K + 3 + 1 칸:
  0..K−1   A_k        속도 단계 k, 원래 정체 (모양 m0, 색 c0)                       ← v4b 와 같다
  K        V_C        속도 단계 k* 와 같은 궤적, 색만 c1                               (정체 변형)
  K+1      V_S        같은 궤적, 모양 (평면: 기하 모양 · 실물: 다른 물건) 만 m1, 화면 크기는 같게
  K+2      V_CS       같은 궤적, 모양 m1 + 색 c1  → (m0, m1) × (c0, c1) 2 × 2 격자
  K+3      B          빈 장면                                                        ← v4b 와 같다
  k* 는 묶음마다 무작위 (속도 단계와 정체 변형이 얽히지 않게). 빗면 primitive 는 구만 구를 수 있어 모양 변형 대신 색 3 개 (V_C, V_C2, V_C3).
  설계 json 의 "identity" 에 무엇이 바뀌었나를 적는다 (목록 · 위반 짝은 이 기록으로 만든다).
위반 (렌더 뒤 목록 단계): 같은 궤적 두 영상을 프레임 32 (target 이 가려진 동안) 에서 잇는다 — A_k*[0:32] + V[32:64] = 가림막 뒤 정체 바뀜.
변형마다 v4b 와 같은 검사를 다시 한다: 화면 크기 · 판과 부딪힘 · 가리는 물체와 충돌 · late 타이밍 (바깥 · 안쪽 extent) · 그 밖 물체에 가려짐 ·
  그 밖 물체와 생김새 구분 · 물리 검사. 하나라도 실패하면 크기를 0.9 배씩 (2 번) 줄여 다시, 그래도 실패하면 다른 모양 · 색으로 (최대 12 번), 끝내 실패하면 줄기 (stream) 를 바꿔 묶음을 다시 설계한다.
설계기 선택: 설정의 design: gen_design_v5id (렌더기) · BT_V4_DESIGN=gen_design_v5id (design_families_v4vc.py).
"""
from __future__ import annotations

import colorsys
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design_v2 as G2  # noqa: E402
import gen_design_v3 as G3  # noqa: E402
import gen_design_v4vc as V4  # noqa: E402
from gen_design_v4vc import (DIMS, STRIDE, center_path, clip_id, family_of, is_ind, load_cfg, sit_key, situations,  # noqa: E402,F401
                             timing_ok, track, tracks)

NV = 3                                                                  # 정체 변형 칸 수
VAR_NAMES = ("C", "S", "CS")


def n_slots(cfg):
    return int(cfg["velocity"]["levels"]) + NV + 1


def render_slots(cfg, split, f):
    """묶음 f 의 (렌더 번호, specs 번호). specs = [A_0 .. A_{K−1}, V_C, V_S, V_CS, B] 라서 단계 칸은 번호, 빈 장면은 'B' (select 가 specs[-1])."""
    if is_ind(split):
        raise ValueError("v5id 는 묶음 분할만 (ind 없음)")
    S = n_slots(cfg)
    return [(f * S + j, j) for j in range(S - 1)] + [(f * S + S - 1, "B")]


def all_slots(cfg, split, f):
    S = n_slots(cfg)
    return [f * S + j for j in range(S)]


# ----------------------------------------------------------------------------- 정체 변형
def _hsv_color(rng, h):
    return tuple(float(x) for x in colorsys.hsv_to_rgb(h % 1.0, rng.uniform(0.5, 0.9), rng.uniform(0.5, 0.9)))


def _new_color(rng, o, min_dh):
    """원래 색과 색상 거리 ≥ max(0.25, min_dh) 인 새 색 (색 원 위 연속)."""
    h0 = G3._hue(o["color"])[0]
    dh = float(rng.uniform(max(0.25, min_dh), 1.0 - max(0.25, min_dh)))
    return _hsv_color(rng, h0 + dh)


def _apply_color(o, c):
    o = copy.deepcopy(o)
    o["color"] = tuple(c)
    if "color2" in o:
        k = o.get("kind")
        if k == "wooden_crate":                                         # 칠한 상자: 결 색은 본 색의 어두운 쪽
            o["color2"] = tuple(0.55 * x for x in c)
        elif k in ("cardboard_box",):                                   # 테이프는 그대로
            pass
        elif k not in ("soccer_ball", "beach_ball", "basketball"):
            o["color2"] = tuple(0.7 * x for x in c)
    return o


def _shape_choices(objset, place, o):
    """같은 궤적을 따를 수 있는 다른 모양. 평면: primitive 7 종 중 다른 것 / 실물: 기하 모양이 다른 물건. 빗면: 구르는 것끼리만."""
    if place == "ramp":
        if objset == "primitive":
            return []                                                   # 빗면 primitive = 구뿐 (다른 모양은 굴림 운동을 못 따른다)
        return [k for k in G3.target_kinds(objset, "ramp", "moving") if k != o.get("kind")]
    if objset == "primitive":
        return [s for s in G3.PRIM_SHAPES if s != o["shape"]]
    base_shape = G3.KIND_SHAPE.get(o.get("kind"), o["shape"])
    return [k for k in G3.target_kinds(objset, "flat", "moving") if G3.KIND_SHAPE[k] != base_shape]


def _apply_shape(rng, o, objset, new):
    """모양 (primitive) · 물건 (실물) 을 바꾸고 화면 지름 (s_eff) 을 원래와 같게. 색은 그대로 (실물 고정 색 물건은 원래 색을 입힌다)."""
    v = copy.deepcopy(o)
    if objset == "primitive":
        v["shape"] = new; v["kind"] = new
    else:
        name, _, pat, c1, c2, rough, metal, _ = G3.REAL_BY[new]
        shape = G3.KIND_SHAPE[new]
        v.update(shape=shape, kind=new, pattern=pat, rough=rough, metal=metal, hz=float(G3.DIMS[shape][2]))
        for k_ in ("band",):
            v.pop(k_, None)
        v.update({"paint_bucket": {"band": (0.22, 0.8)}, "plastic_bin": {"band": (0.86, 0.96)}, "traffic_cone": {"band": (0.42, 0.6)}}.get(new, {}))
        if "color2" not in v:
            v["color2"] = tuple(0.7 * x for x in v["color"])
        v = _apply_color(v, o["color"])                                 # 정체 = (모양, 색) 이라 색은 원래 색으로 맞춘다
    v["s"] = 1.0                                                        # s_eff 는 s 에 비례 → 화면 지름이 원래와 같아지는 s
    v["s"] = float(G3.s_eff(o) / max(G3.s_eff(v), 1e-6))
    return v


def _check_variant(cfg, spec, o):
    """v4b 설계와 같은 검사 (변형 target o 를 spec 장면에 넣었을 때). 통과하면 (event_samples, target_px_median), 아니면 이유 문자열."""
    T, fps, g = spec["T"], spec["fps"], spec["g"]
    O, V, W = cfg["objects"], cfg["velocity"], cfg["timing"]
    cam = G3.cam_obj(spec["cam"])
    others = spec["objects"][1:]
    occ_objs = [x for x in others if x.get("role") == "occluder"]
    extras = [x for x in others if x.get("role") != "occluder"]
    pos, _, R = track(o, T, fps, g)
    ctr = center_path(o, pos, R)
    E = G2.extent_pts(o, ctr, R)
    u_, v_, d_ = cam.to_px(ctr[::STRIDE])
    px = cam.f * G3.s_eff(o) / np.maximum(d_, 1e-3)
    inn = (u_ > 0) & (u_ < 256) & (v_ > 0) & (v_ < 256)
    if (d_ < 0.3).any() or px[inn].min(initial=1e9) < O["min_px"]:
        return "px"
    if G2._hits_props(ctr, G2.foot_r(o), [], spec["panels"], T, fps, clear=0.04):
        return "hit"
    for x in occ_objs + extras:
        px_, _, Rx = track(x, T, fps, g)
        if G2.collide(o, ctr, x, center_path(x, px_, Rx), 0.04 if x in occ_objs else 0.08):
            return "collide"
    des = G2.panel_boxes(spec["panels"], T, fps) + G2.obj_boxes(occ_objs, T, fps, g, 1.0)
    if any(p.get("role") == "ramp" for p in spec.get("props", [])):
        if G2.target_hidden(cam, G2.wedge_boxes(spec["props"]), E).max() > 0.05:
            return "wedge_hides"
    off = G2.off_frame(cam, E)[::STRIDE]
    ok, ev = timing_ok(G2.target_hidden(cam, des, E)[::STRIDE], off, W, float(V["max_off"]))
    if not ok:
        return "late"
    E_in = ctr[:, None] + 0.55 * (E - ctr[:, None])
    if not timing_ok(G2.target_hidden(cam, des, E_in)[::STRIDE], off, W, float(V["max_off"]))[0]:
        return "late_in"
    for x in extras:
        if G2.target_hidden(cam, G2.obj_boxes([x], T, fps, g, 1.05), E).max() > 0.02:
            return "extra_hides"
    for x in extras + occ_objs:
        if not G3.distinct(o, x, O["min_hue_diff"]):
            return "not_distinct"
    s_ = copy.deepcopy(spec)
    s_["objects"][0] = o
    if G2.physics_check(s_):
        return "phys"
    return list(ev), float(np.median(px[inn])) if inn.any() else float(np.median(px))


def _fit(cfg, spec, o):
    """크기를 1 · 0.9 · 0.81 배로 줄여 가며 검사. (변형, 결과) 또는 (None, 마지막 이유)."""
    why = None
    for sc in (1.0, 0.9, 0.81):
        v = dict(copy.deepcopy(o), s=float(o["s"] * sc))
        r = _check_variant(cfg, spec, v)
        if not isinstance(r, str):
            return v, r, sc
        why = r
    return None, why, None


def add_identity(cfg, split, f, specs, stream, why):
    """specs = [A_0..A_{K−1}, B] → [A_0..A_{K−1}, V_1, V_2, V_3, B] 또는 None (변형 실패)."""
    K = int(cfg["velocity"]["levels"])
    sp = cfg["splits"][split]
    rng = np.random.default_rng([int(sp["seed"]), 300000 + f, stream])
    sit = family_of(cfg, split, f)[0]
    objset, place = sit["objset"], sit.get("place", "flat")
    O = cfg["objects"]
    kstar = int(rng.integers(K))
    base = specs[kstar]
    o0 = base["objects"][0]
    shapes = _shape_choices(objset, place, o0)
    for _try in range(12):
        c1 = _new_color(rng, o0, float(O["min_hue_diff"]))
        if shapes:
            m1 = shapes[int(rng.integers(len(shapes)))]
            cands = [("C", _apply_color(o0, c1), {"color": True, "shape": False}),
                     ("S", _apply_shape(rng, o0, objset, m1), {"color": False, "shape": True}),
                     ("CS", _apply_color(_apply_shape(rng, o0, objset, m1), c1), {"color": True, "shape": True})]
        else:                                                          # 빗면 primitive: 색 3 개 (서로도 구분되게 색상 1/3 씩 돌림)
            h0 = G3._hue(o0["color"])[0]
            off = float(rng.uniform(0.25, 0.42))
            cands = [(f"C{j + 1}", _apply_color(o0, _hsv_color(rng, h0 + off + j / 3.0 * (1 - 2 * 0.25))), {"color": True, "shape": False})
                     for j in range(NV)]
            m1 = None
        out, bad = [], None
        for name, o, chg in cands:
            v, r, sc = _fit(cfg, base, o)
            if v is None:
                bad = r; break
            s_ = copy.deepcopy(base)
            s_["objects"][0] = v
            s_.update(level=f"V{name}", event_samples=r[0], target_px_median=r[1],
                      identity={"var": name, "base_level": kstar, **chg, "shape0": o0["shape"], "kind0": o0.get("kind"), "color0": list(o0["color"]),
                                "shape1": v["shape"], "kind1": v.get("kind"), "color1": list(v["color"]), "size_scale": sc})
            out.append(s_)
        if bad is None:
            for k in range(K):
                specs[k]["identity"] = {"var": "base", "base_level": kstar, "color": False, "shape": False,
                                        "shape0": o0["shape"], "kind0": o0.get("kind"), "color0": list(o0["color"])}
            return specs[:K] + out + [specs[-1]]
        why[f"id_{bad}"] = why.get(f"id_{bad}", 0) + 1
    return None


def design_family(cfg, split, f, max_tries=400, streams=3):
    """v4b 묶음 설계 (줄기마다) → 정체 변형. 변형이 실패하면 다음 줄기 (줄기 수 = streams × 2)."""
    why, last = {}, None
    for st in range(2 * streams):
        try:
            if family_of(cfg, split, f)[0].get("place") == "ramp":
                specs = V4._design_ramp_family(cfg, split, f, max_tries, st)
            else:
                specs = V4._design_family(cfg, split, f, max_tries, st)
        except RuntimeError as e:
            last = e
            continue
        out = add_identity(cfg, split, f, specs, st, why)
        if out is not None:
            for s_ in out:
                s_["design"] = "gen_design_v5id"; s_["version"] = cfg["name"]
            return out
        last = RuntimeError(f"묶음 {split} {f} 정체 변형 실패 이유 {why}")
    raise RuntimeError(str(last))


def sample_clip(cfg, seed, idx):                                       # 렌더기는 미리 만든 design/*.json 만 읽는다
    raise RuntimeError("v5id: 설계 파일 없음 (design_families_v4vc.py --select 먼저)")
