"""phys3d_single v3 (2026-10-23) — v2 에서 두 가지만 바꾼다. 문서 z_training/docs/PHYS3D_SINGLE_V3_DATASET_2026-10-23.md

1. **놓기 전에 물체가 공중에 떠 있지 않다.** v2 는 놓는 시각을 0~120 장 늦추려고 물체를 그 자리에 붙잡아 두었는데 공중이면 떠 있었다
   (사용자 2026-10-23 "처음에 왜 공중에 떠 있음…? 이상하잖아." — train 의 drop_* · shelf_land · ramp_roll · ledge_fall 86~91 % 가 12 장 넘게 떠 있음).
   늦출 때 (delay > 0) 붙잡는 자리를 셋 중 하나로 바꾼다 (env["hold"]):
     rest  — 아래 면 (바닥 · 선반 · 경사면) 과 SNAP 안이면 그 면에 내려놓는다. 놓는 순간 초속도를 받는다 (차인다).
     throw — 떠 있고 초속도가 THROW_V 보다 크면: 같은 포물선을 거꾸로 따라가 아래 면에 닿는 자리에서 출발 (바닥에서 던져 올린 것).
             거꾸로 간 자리가 방 밖이거나 가는 길이 막히면 바로 아래 면에서 같은 수평 속도 · 같은 에너지로 (포물선을 옆으로 옮긴 것).
     rod   — 떠 있고 느리면: 위 (화면 밖) 에서 내려온 막대가 물체를 쥐고 있다가 놓는 순간 위로 빠진다 (충돌 없음, 2 장 안에 화면 밖).
   delay = 0 이면 v2 와 같다 (붙잡는 장이 렌더 앞 준비 장 안이다).
2. **렌더 · 저장은 3 장마다 1 장 (4 fps).** 시뮬 · 라벨 · 사건 · 행은 12 fps 로 만들고 저장할 때 옮긴다 (S.subsample).
   mp4 · states.npz · 행이 전부 4 fps 장 단위 → 학습 config 는 stride 1. 렌더 장이 1/3.
  /workspace/tools/blender-4.2.3-linux-x64/blender -t 1 -b -noaudio -P z_training/data/gen_phys3d_single_v3_blender.py -- \
      --scenario drop_behind --seed 0 --frames 288 --out /path/to/out
대량 생성은 gen_phys3d.py --single-v3 --stride 1 --frames 288.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_phys3d_single_v2_blender as V2                       # noqa: E402  (v2 값 · 갈고리를 그대로 깐다)
from gen_phys3d_single_v2_blender import S, B                   # noqa: E402

S.SAVE_STRIDE = V2.STRIDE
SNAP = 0.15                                   # m — 아래 면과 이만큼 안이면 내려놓는다
THROW_V = 1.5                                 # m/s — 이보다 빠르면 던짐 (throw), 아니면 막대 (rod)
ROD_R = 0.03
STATE = {"hold": "none"}


def _world_verts(o):
    bpy.context.view_layer.update()
    mw = o.matrix_world
    return np.array([(mw @ v.co)[:] for v in o.data.vertices])


def _surface_below(sc, x, y, z, skip):
    """(x, y, z) 아래 첫 면의 높이 (없으면 0 = 바닥 윗면)."""
    dg = bpy.context.evaluated_depsgraph_get()
    org = Vector((x, y, z))
    for _ in range(4):
        ok, loc, _n, _i, ob, _m = sc.ray_cast(dg, org, Vector((0, 0, -1)))
        if not ok:
            return 0.0
        if ob is not None and ob.name == skip:
            org = loc - Vector((0, 0, 1e-3))
            continue
        return float(loc.z)
    return 0.0


def _gap(sc, o, q):
    """물체 중심을 q 에 둘 때 아래 면까지 틈 (m). 아래쪽 꼭짓점 (높이 하위 30 %, 최대 48 개) 마다 잰 것의 최소."""
    o.location = q
    V = _world_verts(o)
    lo, hi = V[:, 2].min(), V[:, 2].max()
    low = V[V[:, 2] <= lo + 0.3 * (hi - lo)]
    if len(low) > 48:
        low = low[np.linspace(0, len(low) - 1, 48).astype(int)]
    return min(v[2] - _surface_below(sc, v[0], v[1], v[2] - 1e-3, o.name) for v in low), lo, hi


def _blocked(sc, a, b):
    dg = bpy.context.evaluated_depsgraph_get()
    d = b - a
    if d.length < 1e-6:
        return False
    ok, *_ = sc.ray_cast(dg, a, d.normalized(), distance=d.length)
    return ok


def _rod(o, at):
    """물체 위에서 화면 밖 위로 뻗은 막대 + 집게. at − 1 까지 그 자리, at + 2 에 8 m 위 (화면 밖)."""
    V = _world_verts(o)
    x, y, top = float(o.location.x), float(o.location.y), float(V[:, 2].max())
    L = 9.0
    m = B.material("rod", (48, 48, 52), 0.5)
    bpy.ops.mesh.primitive_cylinder_add(radius=ROD_R, depth=L, vertices=12, location=(x, y, top + L / 2))
    rod = bpy.context.object
    rod.name = "hold_rod"
    rod.data.materials.append(m)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, top + 0.04))
    clamp = bpy.context.object
    clamp.name = "hold_clamp"
    clamp.scale = (0.16, 0.16, 0.08)
    clamp.data.materials.append(m)
    for ob in (rod, clamp):
        z0 = ob.location.z
        for f, dz in ((1, 0.0), (at - 1, 0.0), (at + 2, 8.0)):
            ob.location.z = z0 + dz
            ob.keyframe_insert("location", index=2, frame=f)
        for fc in ob.animation_data.action.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = "LINEAR"
        ob.location.z = z0


def support(o, v, at):
    """붙잡는 자리 · 초속도를 고친다. 반환 (p, v): launch 는 at − 1 까지 p − v·dt 에 붙잡고 at 에 p 로 옮겨 놓는다."""
    sc = bpy.context.scene
    dt = 1.0 / B.FPS
    v = Vector(v)
    p = o.location.copy()
    gap, _, _ = _gap(sc, o, p - v * dt)
    if gap < SNAP:                                              # 거의 닿아 있다 — 내려놓는다
        STATE["hold"] = "rest"
        p.z -= max(gap, 0.0)
        o.location = p
        return p, v
    if v.length > THROW_V:                                      # 던짐 — 같은 포물선을 거꾸로 따라 아래 면까지
        G = B.G
        r = float(max(o.dimensions)) / 2
        best = None
        prev = p.copy()
        for i in range(1, 361):                                 # 1/120 초씩 최대 3 초
            tau = i / 120.0
            q = Vector((p.x - v.x * tau, p.y, p.z - v.z * tau - 0.5 * G * tau * tau))
            if abs(q.x) > B.ROOM_X - r - 0.05 or _blocked(sc, prev, q):
                break
            g, _, _ = _gap(sc, o, q)
            if g < 0.01:
                best = (q, Vector((v.x, 0.0, v.z + G * tau)))
                break
            prev = q
        if best is None:                                        # 방 밖 · 막힘 — 바로 아래 면에서 같은 수평 속도 · 같은 에너지
            g, _, _ = _gap(sc, o, p)
            q = Vector((p.x, p.y, p.z - g))
            vz = math.sqrt(max(v.z, 0.0) ** 2 + 2 * G * g) if v.z >= 0 else math.sqrt(v.z ** 2 + 2 * G * g)
            best = (q, Vector((v.x, 0.0, vz)))
            STATE["hold"] = "throw_shift"
        else:
            STATE["hold"] = "throw"
        q, vn = best
        g, _, _ = _gap(sc, o, q)
        q.z -= g                                                # 면에 딱 붙인다
        p = q + vn * dt                                         # 붙잡는 자리 (p − v·dt) 가 q
        o.location = p
        return p, vn
    STATE["hold"] = "rod"                                       # 느린 놓기 — 위에서 쥐고 있다 놓는다
    o.location = p - v * dt
    _rod(o, at)
    o.location = p
    return p, v


_launch_v2 = V2.launch


def launch(o, v, at=B.PRE - 2):
    if V2.STATE["delay"] > 0:
        p, v = support(o, v, at + V2.STATE["delay"])
        o.location = p
        v = tuple(v)
    return _launch_v2(o, v, at)


B.launch = S.launch = V2.launch = launch       # 원본 · v1 · v2 시나리오 함수가 각 모듈 전역 launch 를 부른다
_build_v2 = S.build


def build(name, seed, attempt, ident, n_frames, res, engine, frames_dir):
    STATE["hold"] = "none"
    out = _build_v2(name, seed, attempt, ident, n_frames, res, engine, frames_dir)
    out[2]["hold"] = STATE["hold"]
    return out


S.build = build

if __name__ == "__main__":
    B.main()
