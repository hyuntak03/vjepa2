"""phys3d_multi (2026-10-25) — synthetic 만으로 predictor 를 학습할 3D 물리 영상. 가림막 없음 · 물체 1~5 개 · 모양 12 · 색 연속.
문서 z_training/docs/PHYS3D_MULTI_DATASET_2026-10-25.md

사용자 (2026-10-25): "synthetic 으로만 predictor 학습 … 물체 모양이나 색상도 다양하게 … 경사면 중간에 점프대 … 떨어지다가 선반에서 멈추거나
물체 2 개 또는 다중 물체가 충돌 … ledge, gravity, arc, 평면 등속 · 가속 · 감속, 경사면, 평면에서 점프대, 경사로에서 점프대,
경사로 내려오고 평면의 벽과 충돌, 물체 2 개 경사로에서 충돌 … 일단 가림막 없는 버전으로 샘플".

방 · 빛 · 카메라 · rigid body · 라벨 (`_states.npz`, 물체별) · 사건 (impact · leave) · 렌더는 phys3d (gen_phys3d_blender.py) 것을 쓴다.
바꾼 것: 시나리오 표 (아래 SCENARIOS), 물체 (모양 12 · HSV 연속 색 · 크기 · 밀도 · 마찰 · 반발 무작위), 가림막 0, 받침 있는 선반.
라벨 추가: rgb · rgb2 (N, 3) uint8 (무늬 둘째 색, 없으면 rgb) · pattern (N,) 0 한 색 / 1 체크 / 2 줄무늬 · mass (N,). identity = shape_id × 8 + 가장 가까운 이름 색 (phys3d 의 8 색) — 기존 head 와 호환용.

  /workspace/tools/blender-4.2.3-linux-x64/blender -t 8 -b -noaudio -P z_training/data/gen_phys3d_multi_blender.py -- \
      --scenario ramp_kicker --seed 0 --frames 144 --out /path/to/out [--engine cpu|cycles]
"""
import colorsys
import json
import math
import os
import shutil
import subprocess
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_phys3d_blender as B                                   # noqa: E402
from gen_phys3d_blender import PRE, ROOM_X, SIZE_SCALE, box, material   # noqa: E402

B.P_EXTRA_SCREEN = 0.0
SHAPES = ("sphere", "cube", "cuboid", "cylinder", "disc", "cone", "pyramid", "tetra", "hex_prism", "icosa", "ellipsoid", "torus",
          "tri_prism", "pent_prism", "octa", "capsule", "rounded_cube", "bicone")     # 뒤 6 개 2026-10-25 추가 ("모양 · 색은 더 많이 해도")
G_RANGE = (2.0, 16.0)                         # m/s² — 영상마다 로그 균등 (천천히 · 빨리 떨어지는 것, 사용자 2026-10-25). phys3d 는 3.5~9.81 균등
P_PATTERN = 0.3                               # 두 색 무늬 (체크 · 줄무늬) 확률
P_ACHROMATIC = 0.08                           # 흰 · 회색 · 검정 쪽 확률
NAMED = [colorsys.rgb_to_hsv(*(c / 255 for c in rgb))[0] for _, rgb in B.COLORS]   # 이름 색 8 개의 hue (identity 호환용)
NC = len(NAMED)
G = 9.81
SIZE_RANGE = (0.27, 0.8)                      # 물체 크기 (m, SIZE_SCALE 곱하기 전) — 로그 균등. 샘플 v0 의 최소 0.18 의 1.5 배 (사용자 2026-10-25). 256 px 에서 약 15 ~ 45 px
SPEED_RANGE = (0.6, 2.2)                      # 영상마다 모든 초속도에 곱하는 배율 — 로그 균등 (사용자 "빠른 것도"). 경사 · 낙하는 중력이 정한다
STATE = {"speed": 1.0}
STILL_V = 0.05                                # m/s — 모든 물체가 이보다 느리면 장면 정지
TAIL = 12                                     # 장면이 멈춘 뒤 남기는 장 (1 초) — 사용자 2026-10-25 "정지해 있는 프레임 비율이 너무 높아선 안 돼"
MIN_T = 48                                    # 자른 뒤 이보다 짧으면 다시 (4 초)


def loguni(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def launch(o, v, at=PRE - 2):
    return B.launch(o, tuple(STATE["speed"] * x for x in v), at)


# ----------------------------------------------------------------------------- 물체
def rand_color(rng):
    """HSV 연속 (s 0.3~1, v 0.35~1). P_ACHROMATIC 은 채도 0~0.08 (흰 · 회색 · 검정 쪽, v 0.12~0.95)."""
    if rng.random() < P_ACHROMATIC:
        h, s, v = rng.uniform(0, 1), rng.uniform(0, 0.08), rng.uniform(0.12, 0.95)
    else:
        h, s, v = rng.uniform(0, 1), rng.uniform(0.3, 1.0), rng.uniform(0.35, 1.0)
    return tuple(int(255 * x) for x in colorsys.hsv_to_rgb(h, s, v)), h


def obj_material(name, rgb, rgb2, pattern, rough, scale):
    """pattern 0 = 한 색, 1 = 체크, 2 = 줄무늬 (wave bands, 두 색 경계가 또렷하게)."""
    m = material(name, rgb, rough)
    if pattern == 0:
        return m
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    c1 = [B.srgb2lin(c) for c in rgb] + [1.0]
    c2 = [B.srgb2lin(c) for c in rgb2] + [1.0]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    if pattern == 1:
        t = nt.nodes.new("ShaderNodeTexChecker")
        t.inputs["Scale"].default_value = scale
        t.inputs["Color1"].default_value, t.inputs["Color2"].default_value = c1, c2
        nt.links.new(tc.outputs["Object"], t.inputs["Vector"])
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        t = nt.nodes.new("ShaderNodeTexWave")
        t.inputs["Scale"].default_value = scale
        nt.links.new(tc.outputs["Object"], t.inputs["Vector"])
        g = nt.nodes.new("ShaderNodeMath")
        g.operation = "GREATER_THAN"
        g.inputs[1].default_value = 0.5
        nt.links.new(t.outputs["Fac"], g.inputs[0])
        mx = nt.nodes.new("ShaderNodeMixRGB")
        mx.inputs["Color1"].default_value, mx.inputs["Color2"].default_value = c1, c2
        nt.links.new(g.outputs[0], mx.inputs["Fac"])
        nt.links.new(mx.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def mesh_obj(name, V, F, loc):
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], F)
    me.validate()
    me.update()
    o = bpy.data.objects.new(name, me)
    o.location = loc
    bpy.context.collection.objects.link(o)
    for ob in bpy.context.selected_objects:
        ob.select_set(False)
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    return o


def color_bin(h):
    d = [min(abs(h - c), 1 - abs(h - c)) for c in NAMED]
    return int(np.argmin(d))


def obj(rng, loc, size=None, shape=None, rot=None, friction=None, bounce=None, density=None, name="obj"):
    """무작위 물체 하나 (ACTIVE rigid body). size = 지름 비슷한 값 (m, SIZE_SCALE 곱하기 전). None = SIZE_RANGE 로그 균등, 음수 −c = 그 분포를 c 로 자름 (여럿을 붙여 놓을 때)."""
    sid = int(rng.integers(len(SHAPES))) if shape is None else SHAPES.index(shape)
    shp = SHAPES[sid]
    size = (loguni(rng, *SIZE_RANGE) if size is None else min(-size, loguni(rng, *SIZE_RANGE)) if size < 0 else size) * SIZE_SCALE
    r = size / 2
    ops = bpy.ops.mesh
    if shp == "sphere":
        ops.primitive_uv_sphere_add(radius=r, segments=32, ring_count=16, location=loc)
    elif shp == "cube":
        ops.primitive_cube_add(size=size * 0.8, location=loc)
    elif shp == "cuboid":
        ops.primitive_cube_add(size=1, location=loc, scale=(size * 1.1, size * 0.7, size * 0.55))
    elif shp == "cylinder":
        ops.primitive_cylinder_add(radius=r * 0.85, depth=size, vertices=32, location=loc)
    elif shp == "disc":
        ops.primitive_cylinder_add(radius=r * 1.05, depth=size * 0.35, vertices=32, location=loc, rotation=(math.pi / 2, 0, 0))
    elif shp == "cone":
        ops.primitive_cone_add(radius1=r, depth=size, vertices=32, location=loc)
    elif shp == "pyramid":
        ops.primitive_cone_add(radius1=r * 1.15, depth=size, vertices=4, location=loc)
    elif shp == "tetra":
        ops.primitive_cone_add(radius1=r * 1.2, depth=size * 0.9, vertices=3, location=loc)
    elif shp == "hex_prism":
        ops.primitive_cylinder_add(radius=r * 0.9, depth=size * 0.8, vertices=6, location=loc)
    elif shp == "icosa":
        ops.primitive_ico_sphere_add(radius=r, subdivisions=1, location=loc)
    elif shp == "ellipsoid":
        ops.primitive_uv_sphere_add(radius=r, segments=32, ring_count=16, location=loc, scale=(1.3, 0.9, 0.75))
    elif shp == "torus":
        ops.primitive_torus_add(major_radius=r * 0.72, minor_radius=r * 0.28, location=loc, rotation=(math.pi / 2, 0, 0))
    elif shp == "tri_prism":
        ops.primitive_cylinder_add(radius=r * 1.0, depth=size * 0.8, vertices=3, location=loc, rotation=(math.pi / 2, 0, 0))
    elif shp == "pent_prism":
        ops.primitive_cylinder_add(radius=r * 0.9, depth=size * 0.7, vertices=5, location=loc)
    elif shp == "octa":
        mesh_obj("octa", [(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)],
                 [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)], loc)
    elif shp == "capsule":
        ops.primitive_uv_sphere_add(radius=r * 0.55, segments=32, ring_count=16, location=loc)
        for v in bpy.context.object.data.vertices:          # 위 · 아래 반구를 벌려 원통 몸통
            v.co.z += math.copysign(r * 0.45, v.co.z) if abs(v.co.z) > 1e-6 else 0.0
    elif shp == "rounded_cube":
        ops.primitive_cube_add(size=size * 0.8, location=loc)
        bv = bpy.context.object.modifiers.new("bevel", "BEVEL")
        bv.width, bv.segments = size * 0.12, 3
        bpy.ops.object.modifier_apply(modifier="bevel")
    elif shp == "bicone":
        k = 24
        ring = [(r * 0.8 * math.cos(2 * math.pi * i / k), r * 0.8 * math.sin(2 * math.pi * i / k), 0.0) for i in range(k)]
        mesh_obj("bicone", ring + [(0, 0, r), (0, 0, -r)], [(i, (i + 1) % k, k) for i in range(k)] + [((i + 1) % k, i, k + 1) for i in range(k)], loc)
    o = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    if rot is not None:
        o.rotation_euler = rot
    if shp in ("sphere", "torus", "ellipsoid", "cylinder", "disc", "capsule", "rounded_cube"):
        bpy.ops.object.shade_smooth()
    rgb, hue = rand_color(rng)
    pattern = int(rng.choice(3, p=[1 - P_PATTERN, P_PATTERN / 2, P_PATTERN / 2]))
    rgb2 = rand_color(rng)[0]
    pscale = float(rng.uniform(2.0, 6.0))
    o.name = f"{name}_{shp}"
    o.data.materials.append(obj_material(o.name, rgb, rgb2, pattern, float(rng.uniform(0.2, 0.75)), pscale))
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    rb = o.rigid_body
    rb.collision_shape = {"sphere": "SPHERE", "cube": "BOX", "cuboid": "BOX", "cone": "CONE"}.get(shp, "CONVEX_HULL")
    dens = float(rng.uniform(0.5, 2.0)) if density is None else density
    rb.mass = float(size ** 3 * 8 * dens)
    rb.friction = float(rng.uniform(0.05, 0.35)) if friction is None else friction
    rb.restitution = float(rng.uniform(0.3, 0.7)) if bounce is None else bounce
    rb.linear_damping = 0.01
    rb.angular_damping = 0.05
    rb.use_margin = True
    rb.collision_margin = 0.002
    o["shape_id"], o["rgb"], o["ident"] = sid, list(rgb), sid * NC + color_bin(hue)
    o["rgb2"], o["pattern"] = list(rgb2) if pattern else list(rgb), pattern
    return o


def rest_z(o, z_surface):
    """물체 아래면이 z_surface 에 닿는 중심 높이."""
    bpy.context.view_layer.update()
    lo = min((o.matrix_world @ v.co).z for v in o.data.vertices)
    return o.location.z - lo + z_surface + 0.003


def place_on(o, z_surface):
    o.location.z = rest_z(o, z_surface)
    bpy.context.view_layer.update()


def slab(name, x0, z0, d, L, a, rng, t=0.25, mat=None, friction=None, bounce=None, back=False):
    """윗면이 (x0, z0) 에서 (x0 + d·L·cos a, z0 + L·sin a) 로 가는 판 (d = ±1, a 라디안, 음수면 내려간다). 윗면 중심 · 법선 반환.
    back = True 면 판이 뒷벽까지 이어진다 (y −0.8 ~ ROOM_Y) — 다리 없이 벽에 붙어 받쳐진다."""
    dvec = Vector((d * math.cos(a), 0, math.sin(a)))
    n = Vector((-d * math.sin(a), 0, math.cos(a)))
    mid = Vector((x0, 0, z0)) + dvec * (L / 2)
    m = mat or material(name, B.rand_rgb(rng, 90, 200), 0.7)
    yc, dy = ((B.ROOM_Y - 0.8) / 2, B.ROOM_Y + 0.8) if back else (0.0, 1.6)
    c = mid - n * (t / 2)
    box(name, (c.x, yc, c.z), (L, dy, t), m, rot=(0, -d * a, 0),
        friction=float(rng.uniform(0.2, 0.8)) if friction is None else friction,
        bounce=float(rng.uniform(0.2, 0.5)) if bounce is None else bounce)
    return mid, n, m


def block(name, x, w, h, rng, z0=0.0, bounce=None, mat=None):
    """바닥 (또는 z0) 에 선 상자."""
    return box(name, (x, 0, z0 + h / 2), (w, 1.6, h), mat or material(name, B.rand_rgb(rng, 80, 200), 0.7),
               bounce=float(rng.uniform(0.3, 0.8)) if bounce is None else bounce)


def shelf(name, x, z, w, rng, tilt=0.0):
    """받침 다리 (뒤쪽 y = 0.65, 물체 길 밖) 가 있는 선반."""
    m = material(name, B.rand_rgb(rng, 80, 200), 0.7)
    box(name, (x, 0, z), (w, 1.6, 0.12), m, rot=(0, tilt, 0), bounce=float(rng.uniform(0.15, 0.5)))
    box(name + "_leg", (x, 0.68, (z - 0.06) / 2), (0.12, 0.12, z - 0.06), m)
    return z + 0.06


def prism(name, pts, rng, mat=None, friction=0.05, bounce=0.2):
    """x-z 평면 볼록 다각형 pts [(x, z), …] 를 y ±0.8 로 뽑은 기둥 (passive, convex hull) — 쐐기 점프대."""
    V = [(x, y, z) for y in (-0.8, 0.8) for x, z in pts]
    n = len(pts)
    F = [tuple(range(n))[::-1], tuple(range(n, 2 * n))] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], F)
    me.validate()
    me.update()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    o.data.materials.append(mat or material(name, B.rand_rgb(rng, 40, 120), 0.6))
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    o.rigid_body.collision_shape = "CONVEX_HULL"
    o.rigid_body.friction = friction
    o.rigid_body.restitution = bounce
    o.select_set(False)
    return o


def ramp_from_top(name, rng, d, H, L, a, p_ground=0.5):
    """위 (벽 쪽) 에서 d 방향으로 내려가는 경사 + 받침. p_ground 확률로 끝이 바닥에 닿게 (L = H / sin a) — 사용자 2026-10-25
    "경사면 끝이 지면이랑 붙어 있는 경우도". 아니면 끝이 떠 있고 아래 받침. 윗면 꼭대기 · 끝 반환."""
    xt = -d * (ROOM_X - 0.15)
    if rng.random() < p_ground:
        H = min(H, 5.2 * math.tan(a))             # 수평 길이 ≤ 5.2 m (방 폭 6.8)
        L = H / math.sin(a) + 0.02
    mid, n, m = slab(name, xt, H, d, L, -a, rng)
    xe, ze = xt + d * L * math.cos(a), H - L * math.sin(a)
    if ze > 0.05:
        block(name + "_base", xe - d * 0.2, 0.3, ze, rng, mat=m)
    block(name + "_top", xt + d * 0.15, 0.3, H, rng, mat=m)
    return Vector((xt, 0, H)), Vector((xe, 0, ze)), m


def on_slope(o, top, d, a, s):
    """경사 윗면 위 꼭대기에서 s m 아래 자리에 물체를 올린다."""
    p = top + Vector((d * math.cos(a), 0, -math.sin(a))) * s
    o.location = (p.x, 0, p.z + 0.6)
    bpy.context.view_layer.update()
    # 경사 윗면까지 수직으로 내리기 — 아래쪽 꼭짓점과 면 사이 최소 틈
    dg = bpy.context.evaluated_depsgraph_get()
    sc = bpy.context.scene
    gaps = []
    for v in o.data.vertices:
        w = o.matrix_world @ v.co
        ok, loc, *_r = sc.ray_cast(dg, w - Vector((0, 0, 1e-3)), Vector((0, 0, -1)))
        if ok and _r[2] is not None and _r[2].name != o.name:
            gaps.append(w.z - loc.z)
    if gaps:
        o.location.z -= min(gaps) - 0.003
    bpy.context.view_layer.update()


def toward(o, target, speed_xy, rng):
    """수평 d 로 speed_xy · 지면 위 미끄러짐."""
    return (speed_xy * math.copysign(1, target - o.location.x), 0, 0)


# ----------------------------------------------------------------------------- 시나리오
# (rng, n_frames) → (objs, info). objs[0] = 주 물체. info: require_impact (물체끼리 충돌이 있어야), unpred
SCENARIOS = {}


def scenario(name, category, desc):
    def deco(fn):
        SCENARIOS[name] = (category, desc, fn)
        return fn
    return deco


@scenario("ledge_fall", "gravity", "탁자 위를 미끄러져 · 굴러 가다 끝에서 떨어짐 — 탁자 높이 · 속도 무작위, 30 % 는 물체 2 개")
def s_ledge(rng, n):
    d = rng.choice([-1, 1])
    h, w = float(rng.uniform(0.9, 2.2)), float(rng.uniform(2.0, 3.2))
    cx = -d * (ROOM_X - w / 2)
    block("table", cx, w, h, rng)
    objs = []
    for i in range(1 if rng.random() < 0.7 else 2):
        o = obj(rng, (cx - d * (w / 2 - 0.5 - 0.9 * i), 0, h + 1), size=-0.5, friction=float(rng.uniform(0.02, 0.15)))
        place_on(o, h)
        launch(o, (d * rng.uniform(0.8, 2.5) * (1 - 0.4 * i), 0, 0))
        objs.append(o)
    return objs, {}


@scenario("free_fall", "gravity", "1~3 물체를 2~4.5 m 에서 놓음 (살짝 던짐) — 자유 낙하 · 바닥에서 튕김 · 서로 부딪힘")
def s_fall(rng, n):
    objs = []
    xs = rng.permutation(np.linspace(-2.4, 2.4, 6))
    for i in range(int(rng.integers(1, 4))):
        o = obj(rng, (float(xs[i]), 0, float(rng.uniform(2.0, 4.5))), size=-0.55, rot=tuple(rng.uniform(0, 3.14, 3)))
        launch(o, (rng.uniform(-1, 1), 0, rng.uniform(-1, 1.5)))
        objs.append(o)
    return objs, {}


@scenario("arc", "gravity", "바닥에서 비스듬히 던진 물체의 포물선 — 꼭대기 높이 · 수평 속도 무작위, 반대 벽 · 바닥에서 튕김")
def s_arc(rng, n):
    d = rng.choice([-1, 1])
    o = obj(rng, (float(-d * rng.uniform(2.2, 2.9)), 0, 1))
    place_on(o, 0)
    # 꼭대기 높이 (오르는 높이) 를 직접 고르게 뽑는다 — 낮은 깡충 0.2 m 부터 화면 위 근처 (꼭대기 APEX_MAX) 까지 (사용자 2026-10-25
    # "arc 의 세로 초속도도 다양하게 … 최고점 높은 거랑 낮은 거 … 범위 넓게"). 세로 초속도 = sqrt(2 g h) 라 속도 배율 (STATE speed) 은 수평에만 건다.
    rise = float(rng.uniform(0.2, B.APEX_MAX - o.location.z))
    vx = d * rng.uniform(1.0, 4.5) * STATE["speed"]
    B.launch(o, (vx, 0, math.sqrt(2 * G * rise)))
    return [o], {"rise": rise}


@scenario("flat_const", "motion", "바닥 위 등속 — 마찰 0 으로 미끄러짐 (끝까지 같은 속도, 벽에 닿으면 튕김)")
def s_const(rng, n):
    d = rng.choice([-1, 1])
    o = obj(rng, (float(-d * rng.uniform(1.5, 2.8)), 0, 1), friction=0.0)
    o.rigid_body.linear_damping = 0.0
    o.rigid_body.angular_damping = 0.0
    place_on(o, 0)
    launch(o, (d * rng.uniform(0.3, 0.9), 0, 0))
    return [o], {}


@scenario("flat_decel", "motion", "바닥 위 감속 — 마찰로 미끄러지다 멈춤 (마찰 · 초속도 무작위)")
def s_decel(rng, n):
    d = rng.choice([-1, 1])
    o = obj(rng, (float(-d * rng.uniform(2.0, 2.8)), 0, 1), shape=str(rng.choice(["cube", "cuboid", "pyramid", "hex_prism", "tetra", "cone"])),
            friction=float(rng.uniform(0.12, 0.45)))        # 0.3~0.9 는 대부분 4 초 안에 멈춰 MIN_T 에서 다시 만들었다 (시험 평균 10 번)
    place_on(o, 0)
    launch(o, (d * rng.uniform(2.0, 4.0), 0, 0))
    return [o], {}


@scenario("ramp_down", "gravity", "경사면을 굴러 · 미끄러져 내려와 바닥으로 (가속) — 경사 각 · 높이 무작위")
def s_ramp_down(rng, n):
    d = rng.choice([-1, 1])
    a = math.radians(rng.uniform(15, 35))
    top, end, _ = ramp_from_top("ramp", rng, d, float(rng.uniform(1.6, 2.8)), 2.8, a)
    o = obj(rng, (0, 0, 3), friction=float(rng.uniform(0.02, 0.4)))
    on_slope(o, top, d, a, float(rng.uniform(0.3, 0.8)))
    launch(o, (d * rng.uniform(0.0, 0.8) * math.cos(a), 0, -rng.uniform(0.0, 0.8) * math.sin(a)))
    return [o], {}


@scenario("ramp_up", "gravity", "바닥에서 오르막으로 밀려 올라가며 감속 → 멈췄다 되돌아 내려옴 (또는 넘어 떨어짐)")
def s_ramp_up(rng, n):
    d = rng.choice([-1, 1])                      # 물체가 가는 방향
    a = math.radians(rng.uniform(12, 28))
    L = 2.6
    x0 = d * float(rng.uniform(0.0, 0.6))
    slab("ramp", x0, 0.0, d, L, a, rng, t=0.6)
    block("ramp_base", x0 + d * (L * math.cos(a) - 0.2), 0.4, L * math.sin(a) - 0.02, rng)
    o = obj(rng, (x0 - d * rng.uniform(1.5, 2.5), 0, 1), friction=float(rng.uniform(0.02, 0.3)))
    place_on(o, 0)
    launch(o, (d * rng.uniform(2.5, 4.5), 0, 0))
    return [o], {}


@scenario("flat_kicker", "gravity", "바닥 위를 미끄러지다 점프대 (작은 쐐기) 를 타고 날아올라 착지")
def s_flat_kicker(rng, n):
    d = rng.choice([-1, 1])
    a = math.radians(rng.uniform(15, 32))
    L = float(rng.uniform(0.9, 1.6))
    x0 = -d * float(rng.uniform(0.6, 1.4))
    h = L * math.tan(a)
    prism("kicker", [(x0, -0.05), (x0 + d * L, -0.05), (x0 + d * L, h), (x0 - d * 0.02, 0.0)], rng)   # 쐐기: 바닥에서 올라가 뒤는 수직
    o = obj(rng, (x0 - d * rng.uniform(1.0, 1.8), 0, 1), friction=float(rng.uniform(0.0, 0.1)))
    place_on(o, 0)
    launch(o, (d * rng.uniform(2.5, 4.5), 0, 0))
    return [o], {}


@scenario("ramp_kicker", "gravity", "경사로를 내려오다 경사 중간의 점프대 (쐐기) 에서 날아올라 아래 경사 · 바닥에 착지")
def s_ramp_kicker(rng, n):
    d = rng.choice([-1, 1])
    a = math.radians(rng.uniform(18, 32))
    top, end, m = ramp_from_top("ramp", rng, d, float(rng.uniform(2.4, 3.4)), float(rng.uniform(3.0, 4.0)), a, p_ground=0.6)
    u = Vector((d * math.cos(a), 0, -math.sin(a)))                    # 경사 아래 방향
    nv = Vector((d * math.sin(a), 0, math.cos(a)))                    # 윗면 법선
    Ls = (end - top).length
    s0, Lw = float(rng.uniform(0.35, 0.5)) * Ls, float(rng.uniform(0.5, 0.8))
    b = math.radians(rng.uniform(22, 40))                             # 경사에 대한 점프대 각
    P0, P1 = top + u * s0, top + u * (s0 + Lw)
    P2 = P1 + nv * (Lw * math.tan(b))
    q = [P0 - nv * 0.06, P1 - nv * 0.06, P2, P0]
    prism("kicker", [(p.x, p.z) for p in q], rng)
    o = obj(rng, (0, 0, 4), friction=float(rng.uniform(0.0, 0.1)))
    on_slope(o, top, d, a, 0.4)
    launch(o, (d * rng.uniform(0.3, 1.2) * math.cos(a), 0, -rng.uniform(0.3, 1.2) * math.sin(a)))
    return [o], {}


@scenario("ramp_wall", "solidity", "경사로를 내려와 바닥을 지나 벽에 부딪혀 되튕김 · 멈춤 — 벽 위치 · 높이 · 반발 무작위")
def s_ramp_wall(rng, n):
    d = rng.choice([-1, 1])
    a = math.radians(rng.uniform(18, 35))
    top, end, _ = ramp_from_top("ramp", rng, d, float(rng.uniform(1.4, 2.4)), 2.4, a)
    wx = d * float(rng.uniform(1.0, 2.6))
    block("wall_mid", wx, 0.3, float(rng.uniform(0.8, 2.0)), rng)
    o = obj(rng, (0, 0, 3), friction=float(rng.uniform(0.02, 0.3)))
    on_slope(o, top, d, a, 0.4)
    launch(o, (d * rng.uniform(0.2, 1.0) * math.cos(a), 0, -rng.uniform(0.2, 1.0) * math.sin(a)))
    return [o], {"require_impact": "wall"}


@scenario("ramp_collide", "collision", "경사로를 내려온 물체가 다른 물체와 충돌 — 아래 바닥에 놓인 것 · 경사로를 올라오는 것 · 마주 오는 것")
def s_ramp_collide(rng, n):
    d = rng.choice([-1, 1])
    a = math.radians(rng.uniform(15, 30))
    top, end, _ = ramp_from_top("ramp", rng, d, float(rng.uniform(1.6, 2.6)), 2.8, a)
    A = obj(rng, (0, 0, 3), friction=float(rng.uniform(0.02, 0.3)))
    on_slope(A, top, d, a, 0.4)
    launch(A, (d * rng.uniform(0.2, 0.8) * math.cos(a), 0, -rng.uniform(0.2, 0.8) * math.sin(a)))
    mode = rng.choice(["rest", "up", "oncoming"], p=[0.4, 0.3, 0.3])
    if mode == "rest":
        Bo = obj(rng, (end.x + d * rng.uniform(0.6, 2.0), 0, 1))
        place_on(Bo, 0)
    elif mode == "up":                            # 바닥에서 경사로로 밀려 올라와 내려오는 것과 경사 위에서 부딪힘
        Bo = obj(rng, (end.x + d * rng.uniform(0.8, 1.6), 0, 1), friction=float(rng.uniform(0.02, 0.2)))
        place_on(Bo, 0)
        launch(Bo, (-d * rng.uniform(2.5, 4.0), 0, 0))
    else:                                         # 바닥에서 마주 와서 경사 밑에서
        Bo = obj(rng, (d * rng.uniform(2.2, 2.9), 0, 1), friction=float(rng.uniform(0.02, 0.2)))
        place_on(Bo, 0)
        launch(Bo, (-d * rng.uniform(0.8, 2.0), 0, 0))
    return [A, Bo], {"require_impact": "objects", "mode": str(mode)}


@scenario("shelf_catch", "solidity", "떨어지던 물체가 선반 (다리 있음) 위에 떨어져 거기서 멈춤 · 튕겨 내려감 — 바닥이 아니라 선반에서 선다")
def s_shelf_catch(rng, n):
    xs = rng.permutation([-1.8, 0.0, 1.8])[:int(rng.integers(1, 3))]
    tops = []
    for k, x in enumerate(xs):
        w = float(rng.uniform(1.2, 2.0))
        tops.append((float(x), shelf(f"shelf{k}", float(x), float(rng.uniform(0.8, 2.4)), w, rng), w))
    objs = []
    for i in range(1 if rng.random() < 0.7 else 2):
        x, z, w = tops[i % len(tops)]
        o = obj(rng, (x + float(rng.uniform(-0.3, 0.3)) * w, 0, float(rng.uniform(z + 1.0, 4.6))),
                bounce=float(rng.uniform(0.1, 0.6)), rot=tuple(rng.uniform(0, 3.14, 3)))
        launch(o, (rng.uniform(-0.5, 0.5), 0, rng.uniform(-1, 0.5)))
        objs.append(o)
    return objs, {"require_rest_above": 0.6}


@scenario("collide_2", "collision", "물체 2 개가 바닥 위에서 (30 % 는 공중 포물선으로) 마주 와 충돌 — 크기 · 밀도 따라 운동량 교환")
def s_collide2(rng, n):
    air = rng.random() < 0.3
    tt = float(rng.uniform(0.8, 1.6))
    xm = float(rng.uniform(-0.8, 0.8))
    objs = []
    for i, d in enumerate((1, -1)):
        x0 = xm - d * float(rng.uniform(1.6, 2.6))
        o = obj(rng, (x0, 0, 1), friction=float(rng.uniform(0.0, 0.15)))
        place_on(o, 0)
        if air:
            launch(o, ((xm - x0) / tt, 0, (1.2 - o.location.z + 0.5 * G * tt ** 2) / tt))
        else:
            launch(o, ((xm - x0) / tt, 0, 0))
        objs.append(o)
    return objs, {"require_impact": "objects"}


@scenario("collide_multi", "collision", "굴러온 물체가 바닥의 2~4 개 무리를 침 (당구 깨기) · 또는 3 개가 사방에서 모임")
def s_collide_multi(rng, n):
    d = rng.choice([-1, 1])
    objs = []
    if rng.random() < 0.6:
        s = obj(rng, (-d * 2.7, 0, 1), friction=float(rng.uniform(0.0, 0.1)), density=float(rng.uniform(1.0, 2.5)))
        place_on(s, 0)
        launch(s, (d * rng.uniform(3.0, 5.0), 0, 0))
        objs.append(s)
        x = float(rng.uniform(-0.8, 0.4)) * d
        for k in range(int(rng.integers(2, 5))):
            o = obj(rng, (x, 0, 1), size=-0.55, friction=float(rng.uniform(0.05, 0.3)))
            bpy.context.view_layer.update()
            o.location.x = x + d * o.dimensions.x / 2
            place_on(o, 0)
            x += d * (o.dimensions.x + float(rng.uniform(0.02, 0.3)))
            objs.append(o)
    else:
        tt = float(rng.uniform(1.0, 1.6))
        xm = float(rng.uniform(-0.6, 0.6))
        for k, (x0, z) in enumerate([(xm - 2.4, 0), (xm + 2.4, 0), (xm + float(rng.uniform(-0.4, 0.4)), 3.8)]):
            o = obj(rng, (x0, 0, max(z, 1)), size=-0.6, friction=float(rng.uniform(0.0, 0.15)))
            if z == 0:
                place_on(o, 0)
                launch(o, ((xm - x0) / tt, 0, 0))
            else:
                launch(o, (0, 0, -(z - 0.6 - 0.5 * G * tt ** 2) / tt))
            objs.append(o)
    return objs, {"require_impact": "objects"}


@scenario("stairs", "gravity", "계단 3~4 단을 튕기며 내려옴 — 단 높이 · 폭 무작위")
def s_stairs(rng, n):
    d = rng.choice([-1, 1])
    k = int(rng.integers(3, 5))
    h, w = float(rng.uniform(0.35, 0.6)), float(rng.uniform(0.7, 1.0))
    m = material("stairs", B.rand_rgb(rng, 90, 200), 0.7)
    x0 = -d * (ROOM_X - 0.1)
    for i in range(k):
        block(f"step{i}", x0 + d * (i + 0.5) * w, w, (k - i) * h, rng, mat=m)
    o = obj(rng, (x0 + d * 0.4 * w, 0, k * h + 1), bounce=float(rng.uniform(0.3, 0.6)))
    place_on(o, k * h)
    launch(o, (d * rng.uniform(1.0, 2.2), 0, rng.uniform(0, 1.5)))
    return [o], {}


@scenario("ledge_to_shelf", "solidity", "탁자 끝에서 떨어진 물체가 아래 선반에 걸려 멈춤 · 선반에서 다시 굴러 떨어짐")
def s_ledge_shelf(rng, n):
    d = rng.choice([-1, 1])
    h, w = float(rng.uniform(2.4, 3.2)), float(rng.uniform(1.6, 2.4))
    cx = -d * (ROOM_X - w / 2)
    block("table", cx, w, h, rng)
    edge = cx + d * w / 2
    sw = float(rng.uniform(1.2, 2.0))
    z = shelf("shelf", edge + d * (sw / 2 + float(rng.uniform(0.0, 0.5))), float(rng.uniform(0.9, h - 1.0)), sw, rng,
              tilt=float(rng.uniform(-0.15, 0.15)))
    o = obj(rng, (edge - d * 0.6, 0, h + 1), friction=float(rng.uniform(0.02, 0.2)), bounce=float(rng.uniform(0.1, 0.5)))
    place_on(o, h)
    launch(o, (d * rng.uniform(0.6, 1.8), 0, 0))
    return [o], {}


# --- phys3d_single · phys3d v1 에서 옮긴 상황 + 진자 · 도미노 (2026-10-25, 사용자 "진자 운동 이런 것도 … phys3d_single 그것도 포함해서 … 상황 엄청 많이")
def _wall_arm(name, px, pz, rng, m, beam=True):
    """양옆 벽 사이를 가로지르는 보 (보이기만, 충돌 없음) + 축에서 보까지 짧은 걸이 — 진자 축이 공중에 뜨지 않게.
    ⚠️ 정정 2026-10-25: 처음엔 뒷벽에서 나온 팔이었는데 카메라 쪽을 향해 거의 안 보여 축이 떠 보였다."""
    by = 0.25
    if beam:
        box(name, (0, by, pz + 0.12), (2 * B.ROOM_X, 0.16, 0.16), m, passive=None)
    return box(name + "_hook", (px, by / 2, pz + 0.06), (0.06, by, 0.06), m, passive=None)


def _pendulum(rng, px, pz, L, th, name="pend", size=None, beam=True):
    dark = material(f"{name}_pivot", (40, 40, 44), 0.5)
    _wall_arm(f"{name}_arm", px, pz, rng, dark, beam=beam)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=(px, 0, pz))
    pivot = bpy.context.object
    pivot.name = f"{name}_pivot"
    pivot.data.materials.append(dark)
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    pivot.rigid_body.collision_shape = "SPHERE"
    pivot.rigid_body.collision_collections[0] = False
    pivot.rigid_body.collision_collections[1] = True
    bob_c = Vector((px + L * math.sin(th), 0, pz - L * math.cos(th)))
    mid = (Vector((px, 0, pz)) + bob_c) / 2
    bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth=L, location=mid, rotation=(0, -th, 0))
    rod = bpy.context.object
    rod.name = f"{name}_rod"
    rod.data.materials.append(dark)
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    rod.rigid_body.mass = 0.05
    rod.rigid_body.collision_collections[0] = False
    rod.rigid_body.collision_collections[1] = True
    rod.rigid_body.linear_damping = rod.rigid_body.angular_damping = 0.0
    bob = obj(rng, tuple(bob_c), size=size if size is not None else -0.5, name=name)
    bob.rigid_body.mass = float(rng.uniform(1.0, 3.0))
    bob.rigid_body.angular_damping = bob.rigid_body.linear_damping = 0.0
    for a, b, kind, loc in ((pivot, rod, "POINT", Vector((px, 0, pz))), (rod, bob, "FIXED", bob_c)):
        bpy.ops.object.empty_add(location=loc)
        e = bpy.context.object
        bpy.ops.rigidbody.constraint_add(type=kind)
        e.rigid_body_constraint.object1, e.rigid_body_constraint.object2 = a, b
    return bob


@scenario("pendulum", "gravity", "진자 (뒷벽 팔에 매단 막대 + 추) — 길이 · 진폭 무작위. 자유 진동 · 스윙 길의 상자에 부딪혀 튕김 · 받침 위 물체를 쳐서 떨어뜨림")
def s_pendulum(rng, n):
    L = float(rng.uniform(1.2, 2.8))
    pz = float(rng.uniform(max(3.3, L + 0.5), 4.6))
    px = float(rng.uniform(-1.0, 1.0))
    th = math.radians(rng.uniform(25, 80)) * float(rng.choice([-1, 1]))
    mode = str(rng.choice(["free", "block", "hit"], p=[0.35, 0.3, 0.35]))
    bob = _pendulum(rng, px, pz, L, th)
    objs, info = [bob], {"mode": mode}
    zb = pz - L                                   # 가장 낮은 점의 추 중심
    s = -math.copysign(1, th)                     # 가장 낮은 점을 지날 때 가는 방향
    if mode == "block":
        bx = px + s * L * float(rng.uniform(0.45, 0.8))
        bh = float(np.clip(pz - L * rng.uniform(0.55, 0.9), 0.4, 3.5))
        block("block", bx, 0.35, bh, rng)
        info["require_impact"] = "any"
    elif mode == "hit":
        tx = px + s * float(rng.uniform(0.25, 0.6))
        t = obj(rng, (tx, 0, 1), size=-0.5)
        hz = zb - 0.05 - (t.dimensions.z / 2) * 0.6
        if hz > 0.25:
            block("stand", tx, 0.5, hz, rng)
        place_on(t, max(hz, 0.0))
        objs.append(t)
        info["require_impact"] = "objects"
    return objs, info


@scenario("container", "solidity", "통 (U 자) 안으로 떨어지거나 던져진 물체 — 벽 안에서 튕기고 갇힌다")
def s_container(rng, n):
    cx = float(rng.uniform(-1.5, 1.5))
    w, h = float(rng.uniform(1.6, 2.6)), float(rng.uniform(0.6, 1.3))
    m = material("bin", B.rand_rgb(rng, 80, 200), 0.7)
    bb = float(rng.uniform(0.2, 0.6))
    box("bin_floor", (cx, 0, 0.06), (w, 1.6, 0.12), m, bounce=bb)
    box("bin_l", (cx - w / 2, 0, h / 2), (0.15, 1.6, h), m, bounce=bb)
    box("bin_r", (cx + w / 2, 0, h / 2), (0.15, 1.6, h), m, bounce=bb)
    objs = []
    for i in range(1 if rng.random() < 0.65 else 2):
        if rng.random() < 0.5:
            o = obj(rng, (cx + float(rng.uniform(-0.3, 0.3)) * w, 0, float(rng.uniform(2.3, 4.2))), size=-0.55)
            launch(o, (rng.uniform(-1, 1), 0, rng.uniform(-1, 1)))
        else:
            d = float(rng.choice([-1, 1]))
            x0 = float(np.clip(cx - d * rng.uniform(1.8, 3.0), -2.9, 2.9))
            o = obj(rng, (x0, 0, 1), size=-0.55)
            place_on(o, 0)
            tt = float(rng.uniform(0.8, 1.5))
            B.launch(o, ((cx - x0) / tt, 0, (h + 0.6 - o.location.z + 0.5 * G * tt ** 2) / tt))
        objs.append(o)
    return objs, {"final_in": [cx - w / 2, cx + w / 2, h + 0.3]}


@scenario("obstacle_bounce", "solidity", "장애물 (바닥에 선 벽 · 다리 있는 블록) 1~3 개 사이를 빠르게 튕겨 다님 — 반발 큼")
def s_obstacle(rng, n):
    rects = []
    for k in range(int(rng.integers(1, 4))):
        for _ in range(30):
            if rng.random() < 0.5:
                w, h = float(rng.uniform(0.2, 0.45)), float(rng.uniform(0.9, 2.8))
                x, z = float(rng.uniform(-2.6, 2.6)), h / 2
            else:
                w, h = float(rng.uniform(0.5, 1.4)), float(rng.uniform(0.25, 0.5))
                x, z = float(rng.uniform(-2.4, 2.4)), float(rng.uniform(1.2, 3.2))
            r = [x - w / 2, x + w / 2, z - h / 2, z + h / 2]
            if any(not (r[1] + 0.6 < q[0] or r[0] - 0.6 > q[1] or r[3] + 0.6 < q[2] or r[2] - 0.6 > q[3]) for q in rects):
                continue
            mt = material(f"obstacle{k}", B.rand_rgb(rng, 80, 200), 0.7)
            box(f"obstacle{k}", (x, 0, z), (w, 1.6, h), mt, bounce=float(rng.uniform(0.6, 0.95)))
            if z - h / 2 > 0.05:                  # 떠 있는 블록은 뒤쪽 다리
                box(f"obstacle{k}_leg", (x, 0.68, (z - h / 2) / 2), (0.12, 0.12, z - h / 2), mt)
            rects.append(r)
            break
    for _ in range(30):
        p = (float(rng.uniform(-2.6, 2.6)), float(rng.uniform(0.8, 3.8)))
        if all(not (q[0] - 0.6 < p[0] < q[1] + 0.6 and q[2] - 0.6 < p[1] < q[3] + 0.6) for q in rects):
            break
    o = obj(rng, (p[0], 0, p[1]), size=-0.55, friction=0.1, bounce=float(rng.uniform(0.8, 0.95)))
    sp, ang = float(rng.uniform(2.5, 5.0)), float(rng.uniform(0, 2 * math.pi))
    launch(o, (sp * math.cos(ang), 0, sp * math.sin(ang)))
    return [o], {"require_impact": "wall"}


@scenario("toss_up", "gravity", "바닥에서 위로 던진 물체가 꼭대기에서 멈췄다 되돌아 떨어짐 — 오르는 높이 0.2 m ~ 화면 위 근처 고르게")
def s_toss(rng, n):
    o = obj(rng, (float(rng.uniform(-2.2, 2.2)), 0, 1))
    place_on(o, 0)
    rise = float(rng.uniform(0.2, B.APEX_MAX - o.location.z))
    B.launch(o, (rng.uniform(-0.8, 0.8) * STATE["speed"], 0, math.sqrt(2 * G * rise)))
    return [o], {"rise": rise}


@scenario("wall_bounce", "solidity", "던지거나 미끄러진 물체가 방 가운데 벽에 부딪혀 되튕김 — 벽 높이 · 위치 · 반발 무작위")
def s_wall(rng, n):
    d = float(rng.choice([-1, 1]))
    wx, wh = float(d * rng.uniform(0.3, 1.5)), float(rng.uniform(1.0, 3.0))
    block("wall_mid", wx, 0.3, wh, rng, bounce=float(rng.uniform(0.3, 0.85)))
    x0 = float(-d * 2.6)
    o = obj(rng, (x0, 0, 1), friction=float(rng.uniform(0.01, 0.1)))
    place_on(o, 0)
    vx = float(rng.uniform(2.0, 4.0)) * STATE["speed"]
    if rng.random() < 0.5:                        # 던짐 — 벽 높이 안에서 맞게
        tt = abs(wx - x0) / vx
        vz = float(np.clip((rng.uniform(0.3, 0.8) * wh - o.location.z + 0.5 * G * tt ** 2) / tt, 0, 8))
    else:
        vz = 0.0
    B.launch(o, (d * vx, 0, vz))
    return [o], {"require_impact": "wall"}


STACK_SHAPES = ("cube", "cuboid", "cylinder", "hex_prism", "rounded_cube", "pent_prism")


@scenario("stack_topple", "collision", "쌓인 블록 2~4 개 — 날아온 물체가 치거나 위에 떨어져 무너짐")
def s_stack(rng, n):
    d = float(rng.choice([-1, 1]))
    sx = float(d * rng.uniform(0.2, 1.4))
    objs, z = [], 0.0
    blocks = []
    for k in range(int(rng.integers(2, 5))):
        o = obj(rng, (sx + float(rng.uniform(-0.04, 0.04)), 0, z + 1), size=-0.5, shape=str(rng.choice(STACK_SHAPES)),
                friction=float(rng.uniform(0.4, 0.8)), bounce=0.1)
        place_on(o, z)
        bpy.context.view_layer.update()
        z = max((o.matrix_world @ v.co).z for v in o.data.vertices)
        blocks.append(o)
    zt = float(rng.uniform(0.3, 0.9)) * z
    if rng.random() < 0.7:                        # 옆에서 던져 침
        sh = obj(rng, (float(-d * 2.7), 0, 1), size=-0.6, density=float(rng.uniform(1.0, 3.0)))
        place_on(sh, 0)
        tt = float(rng.uniform(0.7, 1.4))
        B.launch(sh, ((sx - sh.location.x) / tt, 0, (zt - sh.location.z + 0.5 * G * tt ** 2) / tt))
    else:                                         # 위에서 떨어뜨림 (살짝 어긋나게)
        sh = obj(rng, (sx + float(rng.uniform(-0.3, 0.3)), 0, float(min(z + rng.uniform(1.0, 2.5), 4.6))), size=-0.6,
                 density=float(rng.uniform(1.0, 3.0)))
        launch(sh, (rng.uniform(-0.5, 0.5), 0, 0))
    return [sh] + blocks, {"require_impact": "objects"}


@scenario("domino", "collision", "도미노 4~8 개 — 첫 판이 기울어 쓰러지거나 굴러온 물체가 쳐서 차례로 넘어짐")
def s_domino(rng, n):
    d = float(rng.choice([-1, 1]))
    k = int(rng.integers(4, 9))
    hgt = float(rng.uniform(0.6, 1.1))
    gap = hgt * float(rng.uniform(0.45, 0.7))
    x0 = -d * float(rng.uniform(1.0, 1.8))
    tiles = []
    m_rgb = rng.random() < 0.5                    # 판 색을 같게 (50 %) · 제각각
    rgb0 = None
    for i in range(k):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x0 + d * i * gap, 0, hgt / 2 + 0.003), scale=(hgt * 0.16, 0.5, hgt))
        o = bpy.context.object
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        rgb, hue = rand_color(rng)
        if m_rgb:
            rgb0 = rgb0 or (rgb, hue)
            rgb, hue = rgb0
        o.name = f"domino{i}_cuboid"
        o.data.materials.append(obj_material(o.name, rgb, rgb, 0, float(rng.uniform(0.3, 0.7)), 1.0))
        bpy.ops.rigidbody.object_add(type="ACTIVE")
        rb = o.rigid_body
        rb.collision_shape = "BOX"
        rb.mass, rb.friction, rb.restitution = float(hgt ** 3 * 0.8), float(rng.uniform(0.3, 0.6)), 0.05
        rb.use_margin, rb.collision_margin = True, 0.002
        sid = SHAPES.index("cuboid")
        o["shape_id"], o["rgb"], o["ident"], o["rgb2"], o["pattern"] = sid, list(rgb), sid * NC + color_bin(hue), list(rgb), 0
        tiles.append(o)
    if rng.random() < 0.5:                        # 첫 판이 기울어 있다 (놓자마자 쓰러짐)
        tiles[0].rotation_euler.y = d * float(rng.uniform(0.15, 0.3))
        return tiles, {"require_impact": "objects"}
    st = obj(rng, (x0 - d * float(rng.uniform(1.2, 2.0)), 0, 1), size=-0.5, density=float(rng.uniform(1.0, 3.0)), friction=0.05)
    place_on(st, 0)
    tt = float(rng.uniform(0.5, 1.0))
    zt = hgt * float(rng.uniform(0.55, 0.85))
    B.launch(st, ((x0 - st.location.x) / tt, 0, (zt - st.location.z + 0.5 * G * tt ** 2) / tt))
    return [st] + tiles, {"require_impact": "objects"}


# --- 여러 단계 코스 (2026-10-25, 사용자 "여러 단계 코스도 괜찮고 … 최대한 fair 하고 make sense 하게") — 긴 운동 (C 8 · 16 · 32 의 문맥 끝에 물체가 아직 움직이게)
ROLLERS = ("sphere", "cylinder", "capsule", "disc", "ellipsoid", "torus", "icosa")


@scenario("marble_run", "gravity", "지그재그 경사 3~4 단 (뒷벽에 붙은 판) 을 굴러 · 미끄러져 내려와 단마다 떨어짐 — 긴 운동, 단마다 충돌")
def s_marble_run(rng, n):
    d = float(rng.choice([-1, 1]))
    k = int(rng.integers(3, 5))
    top = float(rng.uniform(3.6, 4.4))
    gapz = (top - 0.5) / k
    m = material("run", B.rand_rgb(rng, 80, 200), 0.65)
    L = float(rng.uniform(3.4, 4.4))
    z = top
    xs = -d * (ROOM_X - 0.1)
    first = None
    for i in range(k):
        di = d if i % 2 == 0 else -d
        a = math.radians(rng.uniform(6, 14))
        x0 = xs if i == 0 else xe + di * 0.15
        slab(f"run{i}", x0, z, di, L, -a, rng, t=0.15, mat=m, friction=float(rng.uniform(0.2, 0.6)), bounce=0.25, back=True)
        if first is None:
            first = (Vector((x0, 0, z)), di, a)
        xe, ze = x0 + di * L * math.cos(a), z - L * math.sin(a)
        z = ze - gapz * float(rng.uniform(0.55, 0.8))
        if z < 0.35:
            break
    shp = str(rng.choice(ROLLERS)) if rng.random() < 0.75 else None
    o = obj(rng, (0, 0, 5), size=-0.5, shape=shp, friction=float(rng.uniform(0.05, 0.4)) if shp else float(rng.uniform(0.0, 0.08)))
    t0, di, a = first
    on_slope(o, t0, di, a, 0.45)
    launch(o, (di * rng.uniform(0.2, 1.0) * math.cos(a), 0, -rng.uniform(0.2, 1.0) * math.sin(a)))
    return [o], {}


@scenario("course", "gravity", "코스 — 경사를 내려와 평면의 점프대 · 둔덕 · 낮은 턱을 지나 반대쪽 오르막을 올랐다 되돌아옴 (여러 번 오감)")
def s_course(rng, n):
    d = float(rng.choice([-1, 1]))
    a = math.radians(rng.uniform(15, 30))
    top, end, m = ramp_from_top("ramp_in", rng, d, float(rng.uniform(1.4, 2.6)), 2.6, a, p_ground=1.0)
    b = math.radians(rng.uniform(15, 30))           # 반대쪽 오르막 (끝이 바닥에 닿은 판, 벽 쪽이 높다)
    Lb = float(rng.uniform(2.0, 2.8))
    xb = d * (ROOM_X - 0.15) - d * Lb * math.cos(b)
    slab("ramp_out", xb, 0.0, d, Lb, b, rng, t=0.6, mat=m, friction=0.3)
    block("ramp_out_back", d * (ROOM_X - 0.3), 0.3, Lb * math.sin(b), rng, mat=m)
    x_lo, x_hi = sorted((end.x + d * 0.4, xb - d * 0.4))
    feats = []
    nf = int(rng.integers(1, 3)) if x_hi - x_lo > 1.4 else 0      # 평면이 짧으면 장애물 없이 (오르막만)
    for j in range(nf):
        kind = str(rng.choice(["kicker", "bump", "step"]))
        for _ in range(20):
            x = float(rng.uniform(x_lo + 0.4, x_hi - 0.4))
            if all(abs(x - f) > 1.0 for f in feats):
                break
        feats.append(x)
        mk = material(f"feat{j}", B.rand_rgb(rng, 40, 140), 0.6)
        if kind == "kicker":
            Lk = float(rng.uniform(0.5, 0.9))
            h = Lk * math.tan(math.radians(rng.uniform(12, 25)))
            prism(f"feat{j}", [(x, -0.05), (x + d * Lk, -0.05), (x + d * Lk, h), (x, 0.0)], rng, mat=mk)
        elif kind == "bump":
            Lk = float(rng.uniform(0.6, 1.2))
            h = float(rng.uniform(0.08, 0.25))
            prism(f"feat{j}", [(x - Lk / 2, -0.02), (x, h), (x + Lk / 2, -0.02)], rng, mat=mk)
        else:
            block(f"feat{j}", x, float(rng.uniform(0.15, 0.4)), float(rng.uniform(0.05, 0.15)), rng, mat=mk)
    shp = str(rng.choice(ROLLERS)) if rng.random() < 0.75 else None
    o = obj(rng, (0, 0, 4), size=-0.5, shape=shp, friction=float(rng.uniform(0.02, 0.3)) if shp else float(rng.uniform(0.0, 0.06)))
    o.rigid_body.linear_damping = 0.0
    on_slope(o, top, d, a, 0.4)
    launch(o, (d * rng.uniform(0.2, 1.0) * math.cos(a), 0, -rng.uniform(0.2, 1.0) * math.sin(a)))
    return [o], {}


# --- 블록 쌓기 · 시소 · 구덩이 · 진자 둘 · 움직이는 받침 (2026-10-25, 사용자 "block 쌓기도 좋다. 이 시나리오도 추가해서")
@scenario("block_stack", "collision", "블록 3~6 개가 화면 위 밖에서 차례로 떨어져 쌓임 — 어긋나면 미끄러지거나 무너짐")
def s_block_stack(rng, n):
    x = float(rng.uniform(-1.5, 1.5))
    objs = []
    z0 = 5.6
    for i in range(int(rng.integers(3, 7))):
        o = obj(rng, (x + float(rng.normal(0, 0.12)), 0, z0), size=-0.5, shape=str(rng.choice(STACK_SHAPES)),
                friction=float(rng.uniform(0.4, 0.9)), bounce=float(rng.uniform(0.0, 0.2)), rot=(0, 0, float(rng.normal(0, 0.15))))
        bpy.context.view_layer.update()
        z0 += o.dimensions.z + float(rng.uniform(0.6, 1.4))
        objs.append(o)
    return objs, {"enter_from_top": True, "require_impact": "objects"}


@scenario("tower_collapse", "collision", "층마다 조금씩 어긋나게 쌓인 블록 탑 3~6 층 — 무게 중심이 받침 밖이면 스스로 무너지고, 안이면 서 있다가 위에서 떨어진 물체에 무너짐")
def s_tower(rng, n):
    d = float(rng.choice([-1, 1]))
    x = float(rng.uniform(-1.2, 1.2))
    z, objs = 0.0, []
    over = float(rng.uniform(0.1, 0.45))          # 층마다 어긋남 (블록 폭 비율)
    for i in range(int(rng.integers(3, 7))):
        o = obj(rng, (x, 0, z + 1), size=-0.5, shape=str(rng.choice(("cube", "cuboid", "rounded_cube"))),
                friction=float(rng.uniform(0.5, 0.9)), bounce=0.05)
        place_on(o, z)
        bpy.context.view_layer.update()
        z = max((o.matrix_world @ v.co).z for v in o.data.vertices)
        x += d * over * o.dimensions.x
        objs.append(o)
    objs = objs[::-1]                              # 주 물체 = 맨 위
    if over < 0.3 and rng.random() < 0.7:          # 버티는 탑 — 위에서 하나 떨어뜨림
        sh = obj(rng, (x + float(rng.uniform(-0.3, 0.3)), 0, float(min(z + rng.uniform(1.0, 2.0), 4.6))), size=-0.55,
                 density=float(rng.uniform(1.5, 3.0)))
        launch(sh, (rng.uniform(-0.5, 0.5), 0, 0))
        objs.append(sh)
    return objs, {"overhang": over}


@scenario("seesaw", "collision", "시소 (받침점 쐐기 + 판, 경첩) — 높은 쪽에 떨어진 물체가 낮은 쪽 물체를 튕겨 올림 · 또는 무거운 물체가 굴러와 기울임")
def s_seesaw(rng, n):
    cx = float(rng.uniform(-1.0, 1.0))
    hf = float(rng.uniform(0.35, 0.6))
    L = float(rng.uniform(2.4, 3.4))
    m = material("seesaw", B.rand_rgb(rng, 80, 200), 0.6)
    prism("fulcrum", [(cx - 0.3, 0.0), (cx + 0.3, 0.0), (cx, hf)], rng, mat=m, friction=0.6)
    tilt = math.asin(min(0.95, hf / (L / 2)))
    d = float(rng.choice([-1, 1]))                 # 낮은 쪽 방향
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, 0, hf + 0.05), rotation=(0, d * tilt, 0), scale=(L, 0.9, 0.08))
    pl = bpy.context.object
    pl.name = "plank"
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    pl.data.materials.append(m)
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    pl.rigid_body.mass = float(rng.uniform(0.5, 2.0))
    pl.rigid_body.friction = 0.7
    bpy.ops.object.empty_add(location=(cx, 0, hf + 0.05), rotation=(math.pi / 2, 0, 0))
    e = bpy.context.object
    bpy.ops.rigidbody.constraint_add(type="HINGE")
    e.rigid_body_constraint.object1 = bpy.data.objects["fulcrum"]
    e.rigid_body_constraint.object2 = pl
    lo = Vector((cx + d * (L / 2 - 0.35) * math.cos(tilt), 0, hf + 0.05 - (L / 2 - 0.35) * math.sin(tilt)))
    hi = Vector((cx - d * (L / 2 - 0.35) * math.cos(tilt), 0, hf + 0.05 + (L / 2 - 0.35) * math.sin(tilt)))
    A = obj(rng, (lo.x, 0, lo.z + 1), size=-0.5, density=float(rng.uniform(0.3, 1.0)), friction=0.6)
    on_slope(A, lo, d, 0, 0)
    Bo = obj(rng, (hi.x + float(rng.uniform(-0.15, 0.15)), 0, float(min(hi.z + rng.uniform(1.2, 3.0), 4.6))), size=-0.6,
             density=float(rng.uniform(1.5, 4.0)))
    launch(Bo, (0, 0, -rng.uniform(0, 1.0)))
    return [A, Bo], {"require_rise": 0.3}           # 둘은 판을 거쳐 주고받는다 (직접 닿지 않는다) — 낮은 쪽 물체가 떠올랐나로 본다


def _pit_floor(rng, xg, wg, depth):
    """방 바닥을 구덩이 [xg, xg + wg) 를 남기고 다시 깐다 (구덩이 바닥 −depth)."""
    old = bpy.data.objects["floor"]
    m, rb = old.data.materials[0], old.rigid_body
    fr, bo = rb.friction, rb.restitution
    bpy.data.objects.remove(old, do_unlink=True)
    X0, X1 = -B.ROOM_X - 1, B.ROOM_X + 1
    Y = 2 * B.ROOM_Y + 6
    for name, a, b in (("floor", X0, xg), ("floor_r", xg + wg, X1)):
        box(name, ((a + b) / 2, 0, -0.25), (b - a, Y, 0.5), m, friction=fr, bounce=bo)
    box("pit_bottom", (xg + wg / 2, 0, -depth - 0.25), (wg, Y, 0.5), m, friction=fr, bounce=bo)


@scenario("pit_fall", "gravity", "바닥의 구덩이로 굴러 · 미끄러져 빠짐 — 빠르면 건너뜀, 구덩이 폭 · 깊이 · 속도 무작위")
def s_pit(rng, n):
    d = float(rng.choice([-1, 1]))
    wg = float(rng.uniform(0.8, 1.6))
    xg = float(rng.uniform(-0.8, 0.6)) - wg / 2 + d * 0.3
    _pit_floor(rng, xg, wg, float(rng.uniform(0.5, 1.0)))
    o = obj(rng, (xg + wg / 2 - d * float(rng.uniform(1.6, 2.4)), 0, 1), friction=float(rng.uniform(0.02, 0.3)))
    place_on(o, 0)
    launch(o, (d * rng.uniform(0.8, 3.0), 0, 0))
    return [o], {"pit": [xg, wg]}


@scenario("pendulum_pair", "collision", "진자 둘이 나란히 — 하나를 들어 놓아 다른 하나를 침 (운동량 전달) · 또는 둘을 마주 놓아 가운데서 부딪힘")
def s_pend_pair(rng, n):
    L = float(rng.uniform(1.4, 2.6))
    pz = float(rng.uniform(max(3.4, L + 0.6), 4.6))
    size = float(min(0.5, loguni(rng, *SIZE_RANGE)))
    gap = size * SIZE_SCALE * 1.02
    px = float(rng.uniform(-0.6, 0.6))
    both = rng.random() < 0.4
    th = math.radians(rng.uniform(25, 70))
    A = _pendulum(rng, px - gap / 2, pz, L, -th, name="pendA", size=size)
    Bo = _pendulum(rng, px + gap / 2, pz, L, th if both else 0.0, name="pendB", size=size, beam=False)
    return [A, Bo], {"require_impact": "objects", "both": bool(both)}


@scenario("moving_platform", "motion", "움직이는 받침 (수레) 위의 물체 — 받침이 갑자기 서면 관성으로 미끄러지거나 넘어져 떨어짐")
def s_platform(rng, n):
    d = float(rng.choice([-1, 1]))
    w, h = float(rng.uniform(1.0, 1.8)), float(rng.uniform(0.15, 0.45))
    x0 = -d * float(rng.uniform(1.4, 2.2))
    m = material("cart", B.rand_rgb(rng, 60, 160), 0.6)
    cart = box("cart", (x0, 0, h / 2), (w, 1.2, h), m, friction=float(rng.uniform(0.6, 1.0)))
    cart.rigid_body.kinematic = True
    v = float(rng.uniform(1.0, 3.0)) * STATE["speed"]
    dist = float(rng.uniform(1.5, 3.2))
    run = max(int(dist / v * B.FPS), 3)
    stop = int(rng.integers(1, 4))                 # 1~3 장 안에 섬 (급정거)
    t0 = PRE + int(rng.integers(0, 6))             # 출발 전 정지를 짧게 (0~24 장은 정지 17~65 %, 원칙 §2)
    keys = [(1, x0), (t0, x0), (t0 + run, x0 + d * (dist - v * stop / B.FPS / 2)), (t0 + run + stop, x0 + d * dist)]
    for f, xx in keys:
        cart.location.x = xx
        cart.keyframe_insert("location", index=0, frame=f)
    for fc in cart.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    cart.location.x = x0
    o = obj(rng, (x0 + float(rng.uniform(-0.2, 0.2)) * w, 0, h + 1), shape=str(rng.choice(("cube", "cuboid", "cylinder", "cone", "pyramid", "sphere", "hex_prism", "capsule"))),
            friction=float(rng.uniform(0.3, 0.9)))
    place_on(o, h)
    return [o], {"cart": dict(v=v, dist=dist, start=t0 - PRE)}


# ----------------------------------------------------------------------------- 한 편
def sample_points(o):
    """면적 비례 표본 200 점 (rng 0) — 다각형 순서를 정규화 (gen_phys3d_single_blender.sample_points 와 같다, 프로세스마다 같은 점)."""
    me = o.data
    V = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
    polys = []
    for pl in me.polygons:
        vs = list(pl.vertices)
        i = vs.index(min(vs))
        polys.append(tuple(vs[i:] + vs[:i]))
    tri = np.array([(q[0], q[j], q[j + 1]) for q in sorted(polys) for j in range(1, len(q) - 1)])
    a, b, c = V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    r = np.random.default_rng(0)
    k = r.choice(len(tri), size=200, p=area / area.sum())
    u, w = r.random(200), r.random(200)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    return a[k] + u[:, None] * (b[k] - a[k]) + w[:, None] * (c[k] - a[k])


B.sample_points = sample_points


def check(lab, ev, info, objs):
    why = []
    p = lab["pos"]
    if np.linalg.norm(p[:, 0].max(0) - p[:, 0].min(0)) < 25:
        why.append("main barely moves")
    out = lab["inframe_frac"] < 0.5
    if info.get("enter_from_top"):                # 화면 위 밖에서 떨어져 들어오는 것 — 처음 들어온 뒤만 센다
        first = np.where((~out).any(0), np.argmax(~out, axis=0), len(out))   # 창 안에 안 들어오는 물체 (늦게 놓인 것) 는 세지 않는다
        out = out & (np.arange(len(out))[:, None] >= first[None, :])
    keep = [i for i, o in enumerate(objs) if o.get("role") != "front"]   # 앞 물체 (가림) 는 화면 밖에서 들어와 나간다 · 방 밖 길이다 — 검사에서 뺀다
    if out[:, keep].mean(0).max() > 0.1:          # ⚠️ 정정 2026-10-25: 빼지 않아 front_object 가림이 train 19,800 편에서 한 번도 안 남았다
        why.append("object out of frame")
    q = lab["pos3d"][:, keep]
    if (np.abs(q[..., 0]) > ROOM_X + 0.3).any() or (q[..., 2] < (-1.3 if "pit" in info else -0.3)).any():
        why.append("escaped room")
    req = info.get("require_impact")
    if req == "objects":                          # 물체끼리 닿았나 — 두 중심 거리가 크기 합 반 안
        r = np.array([objs[i].dimensions.length / 2 for i in keep]) * 0.75
        hit = any((np.linalg.norm(q[:, i] - q[:, j], axis=-1) < (r[i] + r[j])).any()
                  for i in range(len(keep)) for j in range(i + 1, len(keep)))
        if not hit:
            why.append("no object contact")
    elif req in ("wall", "any") and not any(e["kind"] == "impact" and e["obj"] == 0 and e["frame"] > 6 for e in ev):
        why.append("no impact")
    if "final_in" in info:
        x0, x1, h = info["final_in"]
        if not (x0 < q[-1, 0, 0] < x1 and q[-1, 0, 2] < h):
            why.append("not in container")
    if "require_rise" in info and q[:, 0, 2].max() - q[0, 0, 2] < info["require_rise"]:
        why.append("main did not rise")
    if "require_rest_above" in info and q[-1, 0, 2] < info["require_rest_above"]:
        why.append("main not on shelf")
    return why


# ----------------------------------------------------------------------------- 가림 (2026-10-25)
# 사용자: "고정된 가림막을 물체가 통과 · 더 큰 물체가 작은 물체 앞으로 지나가서 가려짐 · 가림막이 올라갔다가 내려옴 · 가림막이 움직여서 가려짐
# … 가림막도 다양하게". 가림막은 충돌이 없어 물리가 그대로다 — 가림 없이 시뮬한 궤적에서 가릴 시각 t* 를 고르고 판을 놓은 뒤 보임만 다시 잰다.
# 앞 물체 (front_object) 는 카메라 쪽 길 (y = FRONT_Y) 을 따로 지나가 주 물체와 닿지 않는다. 판은 바닥에 서 있거나 바닥 밑에서 올라온다 (공중에 뜬 판 없음).
OCC_MODES = ("fixed", "front_object", "rise_lower", "sweep")
OCC_Y = B.OCC_Y                               # −1.3 (물체 길 y ≈ 0, 앞 유리 −0.9)
FRONT_Y = -2.3
MAX_BLIND = 36                                # 장 (12 fps, 3 초) — 학습이 보는 3 장마다 1 장에서 잰다 (원칙 §2)
OCC_TRIES = 10
FRONT_SHAPES = ("sphere", "cube", "cuboid", "cylinder", "hex_prism", "icosa", "ellipsoid", "rounded_cube", "capsule", "pent_prism", "octa")   # 위가 뾰족한 cone · tetra · pyramid 빼고
MIN_HIDDEN = 3                                # 장 — 빨리 지나가는 앞 물체가 한 장만 가리면 가림이 아니다


def occ_material(rng, name):
    """가림막 겉: 한 색 · 체크 · 줄무늬 · 어두운 판 (채도 낮게 — 물체와 헷갈리지 않게, 그래도 다양하게)."""
    style = int(rng.choice(4, p=[0.35, 0.25, 0.2, 0.2]))
    h = rng.uniform(0, 1)
    c1 = tuple(int(255 * x) for x in colorsys.hsv_to_rgb(h, rng.uniform(0.0, 0.35), rng.uniform(0.3, 0.85)))
    c2 = tuple(int(255 * x) for x in colorsys.hsv_to_rgb(h + rng.uniform(-0.1, 0.1), rng.uniform(0.0, 0.35), rng.uniform(0.2, 0.75)))
    if style == 3:
        c1 = tuple(int(x * 0.35) for x in c1)
    return obj_material(name, c1, c2, {0: 0, 1: 1, 2: 2, 3: 0}[style], float(rng.uniform(0.5, 0.95)), float(rng.uniform(1.5, 5.0))), style


def panel(name, x, w, h, rng, y=OCC_Y, feet=True):
    """바닥에 선 판 (충돌 없음). feet = 발 두 개 (판과 한 덩어리)."""
    t = float(rng.uniform(0.04, 0.16))
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, h / 2))
    o = bpy.context.object
    o.name = name
    o.scale = (w, t, h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if feet:
        parts = [o]
        for sx in (-1, 1):
            bpy.ops.mesh.primitive_cube_add(size=1, location=(x + sx * w * 0.38, y, 0.04))
            f = bpy.context.object
            f.scale = (min(0.25, w * 0.2), 0.45, 0.08)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            parts.append(f)
        for ob in bpy.context.selected_objects:
            ob.select_set(False)
        for ob in parts:
            ob.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.join()
        o = bpy.context.object
        o.name = name
        # 원점을 판 바닥 가운데로 (애니메이션이 판 전체를 옮긴다)
    m, style = occ_material(rng, name)
    o.data.materials.append(m)
    return o, style


def _proj(cam, p, y):
    """카메라에서 p 로 가는 선이 평면 y 와 만나는 점 · 축척 (평면에서 본 크기 / p 에서의 크기)."""
    c = cam.matrix_world.translation
    k = (y - c.y) / (p.y - c.y)
    return c + (p - c) * k, k


def _key_x(o, keys, interp="LINEAR"):
    x0 = o.location.x
    for f, x in keys:
        o.location.x = x
        o.keyframe_insert("location", index=0, frame=int(f))
    for fc in o.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = interp
    o.location.x = x0


def _key_z(o, keys):
    for f, z in keys:
        o.location.z = z
        o.keyframe_insert("location", index=2, frame=int(f))
    for fc in o.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def place_occlusion(mode, rng, sc, objs, lab, T, t0=None, lo=8, need_move=None):
    """mode 의 가림 하나를 놓는다 (t0 = 물체 앞을 덮을 장, None 이면 고른다). 반환 (만든 것들, 앞 물체 [] 또는 [o], 설명) — 실패면 None."""
    cam = sc.camera
    p3, inf = lab["pos3d"][:, 0], lab["inframe_frac"][:, 0]
    sp = np.linalg.norm(lab["vel3d"][:, 0], axis=-1) * B.FPS
    r = float(max(objs[0].dimensions)) / 2
    if t0 is None:
        mv_ok = (lambda t: sp[t] > 0.3) if (need_move or mode == "fixed") else (lambda t: sp[t] < 0.05) if need_move is False else (lambda t: True)
        ok_t = [t for t in range(lo, T - 10) if inf[t] > 0.9 and mv_ok(t)]
        if not ok_t:
            return None
        t0 = int(rng.choice(ok_t))
    elif inf[t0] < 0.6:
        return None
    p = Vector(p3[t0].tolist())
    made, front, desc = [], [], dict(mode=mode, t_star=t0)
    if mode == "front_object":
        # 카메라 쪽 길 (y = FRONT_Y) 을 등속으로 지나가는 큰 물체. 시뮬하지 않고 키프레임으로 움직인다 (rigid body 없음) —
        # dynamic 으로 두니 옆벽에서 되튕기거나 멈췄다 (2026-10-25 시험). 둥근 모양은 굴러가는 각도도 맞춘다. 화면 밖 (|x| > 2.2 + 크기) 에서 들어와 나간다.
        q, k = _proj(cam, p, FRONT_Y)
        d = float(rng.choice([-1, 1]))
        # 바닥에 놓인 앞 물체가 주 물체를 다 덮으려면 카메라 → 주 물체 윗면 선이 앞 길을 지나는 높이보다 높아야 한다
        # (⚠️ 정정 2026-10-25: 크기 0.6~1.0 · 모양 무작위로는 아랫부분만 가려 보임 0.37~0.6 에 그쳐 train 19,800 편에서 한 번도 통과 못 했다)
        r_main = float(max(objs[0].dimensions)) / 2
        top_q, _ = _proj(cam, p + Vector((0, 0, r_main)), FRONT_Y)
        need = top_q.z + 0.15
        if need > 2.6:                                 # 높이 떠 있는 물체는 바닥 물체로 못 가린다
            return None
        shp = str(rng.choice(FRONT_SHAPES))
        big = obj(rng, (float(q.x), FRONT_Y, 1), size=1.0, shape=shp, name="front")
        bpy.context.view_layer.update()
        sc_f = max(need / max(big.dimensions.z, 1e-3), float(rng.uniform(0.9, 1.3))) * float(rng.uniform(1.0, 1.2))
        big.scale = (sc_f, sc_f, sc_f)
        bpy.context.view_layer.objects.active = big
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        size = float(max(big.dimensions)) / SIZE_SCALE
        place_on(big, 0)
        bpy.context.view_layer.objects.active = big
        bpy.ops.rigidbody.object_remove()
        v = float(loguni(rng, 0.8, 4.0))
        park = 2.2 + size * SIZE_SCALE
        fa = PRE + t0 + (-d * park - q.x) / (d * v) * B.FPS
        fb = PRE + t0 + (d * park - q.x) / (d * v) * B.FPS
        _key_x(big, [(fa, -d * park), (fb, d * park)])
        shp = SHAPES[big["shape_id"]]
        if shp in ("sphere", "cylinder", "capsule", "ellipsoid", "disc", "torus", "bicone"):
            rr = max(big.dimensions.z / 2, 0.05)
            ang = 2 * park / rr
            for f, a in ((fa, 0.0), (fb, d * ang)):
                big.rotation_euler.y = a
                big.keyframe_insert("rotation_euler", index=1, frame=int(f))
            for fc in big.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
            big.rotation_euler.y = 0.0
        front.append(big)
        desc.update(front_size=size * SIZE_SCALE, front_speed=v, front_shape=shp)
        return made, front, desc
    top, k = _proj(cam, p + Vector((0, 0, r)), OCC_Y)
    q, _ = _proj(cam, p, OCC_Y)
    w = float(np.clip(2 * k * r * rng.uniform(1.3, 3.2), 0.35, 2.8))
    h = float(np.clip(top.z + rng.uniform(0.1, 0.8), 0.6, 4.9))
    if rng.random() < 0.2 and mode == "fixed":   # 좁고 높은 기둥
        w, h = float(np.clip(2 * k * r * rng.uniform(1.1, 1.6), 0.3, 0.8)), float(rng.uniform(max(h, 2.5), 4.9))
    o, style = panel(f"occ_{mode}", float(q.x), w, h, rng, feet=(mode != "rise_lower"))
    made.append(o)
    desc.update(w=w, h=h, style=style)
    if mode == "rise_lower":                      # 바닥 밑에서 올라와 hold 장 서 있다 내려감
        up = int(rng.integers(4, 9))
        hold = int(rng.integers(8, 31))
        s = max(1, PRE + t0 - up - int(rng.integers(0, hold // 2 + 1)))
        lo, hi = -h / 2 - 0.05, h / 2             # 원점 = 판 가운데. lo 면 윗면이 바닥 밑
        _key_z(o, [(1, lo), (s, lo), (s + up, hi), (s + up + hold, hi), (s + 2 * up + hold, lo)])
        desc.update(rise_frames=up, hold=hold)
    elif mode == "sweep":                         # 화면 한쪽 밖에서 다른 쪽 밖으로 지나감, t0 에 물체 앞
        d = float(rng.choice([-1, 1]))
        vs = float(loguni(rng, 0.6, 3.0))
        park = 3.0 + w / 2
        fa = PRE + t0 - (q.x - (-d * park)) / (d * vs) * B.FPS
        fb = PRE + t0 + (d * park - q.x) / (d * vs) * B.FPS
        _key_x(o, [(fa, -d * park), (fb, d * park)])
        desc.update(sweep_speed=vs)
    return made, front, desc


def blind_ok(vis, T):
    """안 보임 (vis ≤ SEEN) 연속 ≤ MAX_BLIND (3 장마다 1 장에서 잼) · 가려졌다가 끝 전에 다시 보임."""
    b = vis <= B.SEEN
    g = b[::3]
    best = cur = 0
    for x in g:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    if best * 3 > MAX_BLIND:
        return False, f"blind {best * 3}"
    hid = np.nonzero(b)[0]
    if len(hid) < MIN_HIDDEN:
        return False, f"hidden {len(hid)}"
    if hid[-1] >= T - 2:
        return False, "hidden at end"
    return True, ""


def generate(name, k, seed, main_ident, out, n_frames=144, res=256, engine="cpu", max_attempts=40, keep_frames=False):
    global G
    stem = os.path.join(out, name, f"{name}_{k:05d}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    frames_dir = stem + "_frames"
    why = []
    for attempt in range(max_attempts):
        rng = np.random.default_rng([seed, list(SCENARIOS).index(name), attempt])
        sc = B.reset()
        G = B.G = loguni(np.random.default_rng([seed, 5, attempt]), *G_RANGE)
        STATE["speed"] = loguni(np.random.default_rng([seed, 6, attempt]), *SPEED_RANGE)
        sc.gravity = (0, 0, -G)
        env = B.room(rng)
        env["g"] = G
        env["speed_scale"] = STATE["speed"]
        cat, desc, fn = SCENARIOS[name]
        objs, info = fn(rng, n_frames)
        sc.frame_end = PRE + n_frames - 1
        rw = sc.rigidbody_world
        rw.point_cache.frame_start = 1
        rw.point_cache.frame_end = sc.frame_end
        B.setup_render(sc, engine, res, os.path.join(frames_dir, "f"))
        lab = B.simulate_and_label(sc, sc.camera, objs, n_frames, res)
        mv = np.nonzero(~(np.linalg.norm(lab["vel3d"], axis=-1) * B.FPS < STILL_V).all(1))[0]
        T = min(n_frames, (int(mv[-1]) + 1 if len(mv) else 0) + TAIL)
        if T < MIN_T:
            why = [f"too short {T}"]
            continue
        lab = {k2: (v[:T] if v.shape[:1] == (n_frames,) else v) for k2, v in lab.items()}
        ev = [e for e in B.events(lab) if e["kind"] in ("impact", "leave")]
        why = check(lab, ev, info, objs)
        if os.environ.get("PHYS3D_DEBUG") and why:
            print("WHY", attempt, why, flush=True)
        if why:
            continue
        mode = os.environ.get("PHYS3D_OCC", "none")
        if mode == "random":
            mode = str(np.random.default_rng([seed, 7, attempt]).choice(OCC_MODES))
        occ_desc = None
        if mode != "none":
            ro = np.random.default_rng([seed, attempt, 8])
            for _ in range(OCC_TRIES):
                res_o = place_occlusion(mode, ro, sc, objs, lab, T)
                if res_o is None:
                    why = ["no occlusion time"]
                    continue
                made, front, occ_desc = res_o
                lab2 = B.simulate_and_label(sc, sc.camera, objs + front, n_frames, res)
                lab2 = {k2: (v[:T] if v.shape[:1] == (n_frames,) else v) for k2, v in lab2.items()}
                okb, wb = blind_ok(lab2["vis_frac"][:, 0], T)
                if okb:
                    objs, lab = objs + front, lab2
                    for f in front:
                        f["role"] = "front"
                    break
                why = [wb]
                for ob in made + front:
                    bpy.data.objects.remove(ob, do_unlink=True)
                occ_desc = None
            if occ_desc is None:
                continue
            ev = B.events(lab)
        unpred = sorted(info.get("unpred", []))
        rows = B.event_rows(ev, lab, unpred, np.random.default_rng([seed, attempt, 3]), T)
        N = len(objs)
        impact = np.zeros((T, N), np.float32)
        for e in ev:
            if e["kind"] == "impact":
                impact[e["frame"], e["obj"]] = max(impact[e["frame"], e["obj"]], e["dv"])
        bb = lab["bbox"]
        idents = np.array([o["ident"] for o in objs])
        ratio = lab["vis_frac"] / np.maximum(lab["inframe_frac"], 1e-6)
        lab.update(present=np.ones((T, N), bool), hidden=(lab["inframe_frac"] > 0.25) & (ratio < B.HIDDEN),
                   seen=np.maximum.accumulate(lab["vis_frac"] > B.SEEN, axis=0), impact=impact,
                   identity=idents.astype(np.int16), shape_id=np.array([o["shape_id"] for o in objs], np.int16),
                   color_id=(idents % NC).astype(np.int16), rgb=np.array([list(o["rgb"]) for o in objs], np.uint8),
                   rgb2=np.array([list(o["rgb2"]) for o in objs], np.uint8), pattern=np.array([o["pattern"] for o in objs], np.int8),
                   mass=np.array([o.rigid_body.mass if o.rigid_body else 0.0 for o in objs], np.float32),
                   size=np.array([o.dimensions.length / math.sqrt(3) for o in objs], np.float32),
                   area=((bb[0, :, 2] - bb[0, :, 0]) * (bb[0, :, 3] - bb[0, :, 1])).astype(np.float32))
        np.savez_compressed(stem + "_states.npz", **lab)
        if os.environ.get("PHYS3D_NORENDER") != "1":
            os.makedirs(frames_dir, exist_ok=True)
            sc.frame_start = PRE
            sc.frame_end = PRE + T - 1
            bpy.ops.render.render(animation=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(B.FPS), "-start_number", str(PRE),
                            "-i", os.path.join(frames_dir, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                            stem + ".mp4"], check=True)
            if not keep_frames:
                shutil.rmtree(frames_dir)
        meta = dict(scenario=name, category=cat, desc=desc, k=k, seed=seed, attempt=attempt, fps=B.FPS, n_frames=T, res=res,
                    engine=engine, env=env, unpred=unpred, info={k2: v for k2, v in info.items() if k2 != "unpred"}, occlusion=occ_desc,
                    objects=[dict(index=i, role="main" if i == 0 else o.get("role", "other"), shape=SHAPES[o["shape_id"]], rgb=list(o["rgb"]),
                                  rgb2=list(o["rgb2"]), pattern=int(o["pattern"]),
                                  identity=int(o["ident"]), mass=o.rigid_body.mass if o.rigid_body else 0.0,
                                  friction=o.rigid_body.friction if o.rigid_body else 0.0, restitution=o.rigid_body.restitution if o.rigid_body else 0.0, size=float(o.dimensions.length / math.sqrt(3)))
                             for i, o in enumerate(objs)],
                    events=ev, rows=rows)
        with open(stem + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        return dict(ok=True, mp4=stem + ".mp4", attempt=attempt, n_rows=len(rows), n_frames=T)
    return dict(ok=False, mp4=stem + ".mp4", why=why)


# ----------------------------------------------------------------------------- 배경 (2026-10-25, 사용자 "배경도 다양하게 … 자연 경관도 괜찮고 … 최대한 다양하게")
# room 55 % — 방 (벽 · 바닥 절차 재질). open 45 % — HDRI (Poly Haven CC0 1k, 200 장) 를 배경 · 빛으로, 뒷벽 숨김, 바닥을 지평선까지 넓힘,
# 옆벽은 물체가 튕기므로 남기되 담 (돌 · 나무 · 벽돌) 재질. 물리는 그대로 (같은 바닥 · 벽 · 유리).
HDRI_DIR = os.environ.get("PHYS3D_HDRI_DIR", "/workspace/dataset/assets/hdri_polyhaven_1k")
P_OPEN = 0.45
WALL_KINDS = ("plain", "brick", "plaster", "stripes", "stone", "magic", "gradient")
FLOOR_KINDS = ("checker", "tiles", "wood", "concrete", "sand", "grass", "stone", "plain")


def _hsv(rng, s=(0.0, 0.5), v=(0.3, 0.9), h=None):
    h = rng.uniform(0, 1) if h is None else h
    return tuple(int(255 * x) for x in colorsys.hsv_to_rgb(h, rng.uniform(*s), rng.uniform(*v)))


def pattern_material(name, kind, rng, c1, c2, scale):
    """절차 재질 — 바닥 · 벽 · 담. kind 별 텍스처 → 두 색 섞기 → Principled."""
    m = material(name, c1, float(rng.uniform(0.45, 0.95)))
    if kind == "plain":
        return m
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    l1 = [B.srgb2lin(c) for c in c1] + [1.0]
    l2 = [B.srgb2lin(c) for c in c2] + [1.0]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    if kind in ("brick", "tiles"):
        t = nt.nodes.new("ShaderNodeTexBrick")
        t.inputs["Color1"].default_value, t.inputs["Color2"].default_value = l1, [0.85 * x for x in l1[:3]] + [1.0]
        t.inputs["Mortar"].default_value = l2
        t.inputs["Mortar Size"].default_value = float(rng.uniform(0.005, 0.03))
        if kind == "tiles":
            t.offset = float(rng.choice([0.0, 0.5]))
            t.inputs["Brick Width"].default_value = t.inputs["Row Height"].default_value = float(rng.uniform(0.4, 0.8))
        nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
        return m
    if kind == "checker":
        t = nt.nodes.new("ShaderNodeTexChecker")
        t.inputs["Color1"].default_value, t.inputs["Color2"].default_value = l1, l2
        nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
        return m
    if kind in ("plaster", "concrete", "sand", "grass"):
        t = nt.nodes.new("ShaderNodeTexNoise")
        t.inputs["Detail"].default_value = float(rng.uniform(2, 12))
        t.inputs["Roughness"].default_value = float(rng.uniform(0.4, 0.75))
        out = t.outputs["Fac"]
    elif kind in ("stripes", "wood"):
        t = nt.nodes.new("ShaderNodeTexWave")
        t.wave_type = "BANDS" if kind == "stripes" else str(rng.choice(["BANDS", "RINGS"]))
        t.inputs["Distortion"].default_value = 0.0 if kind == "stripes" else float(rng.uniform(2, 8))
        t.inputs["Detail"].default_value = float(rng.uniform(0, 4))
        out = t.outputs["Fac"]
    elif kind == "stone":
        t = nt.nodes.new("ShaderNodeTexVoronoi")
        out = t.outputs["Distance"]
    elif kind == "magic":
        t = nt.nodes.new("ShaderNodeTexMagic")
        t.inputs["Distortion"].default_value = float(rng.uniform(0.5, 2.5))
        out = t.outputs["Fac"]
    else:                                         # gradient
        t = nt.nodes.new("ShaderNodeTexGradient")
        mp.inputs["Scale"].default_value = (0.15, 0.15, 0.15)
        out = t.outputs["Fac"]
    nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
    mx = nt.nodes.new("ShaderNodeMixRGB")
    mx.inputs["Color1"].default_value, mx.inputs["Color2"].default_value = l1, l2
    nt.links.new(out, mx.inputs["Fac"])
    nt.links.new(mx.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def _floor_colors(rng, kind):
    if kind == "grass":
        return _hsv(rng, (0.4, 0.75), (0.25, 0.55), h=rng.uniform(0.2, 0.35)), _hsv(rng, (0.4, 0.8), (0.15, 0.45), h=rng.uniform(0.18, 0.33))
    if kind == "sand":
        return _hsv(rng, (0.2, 0.45), (0.55, 0.9), h=rng.uniform(0.07, 0.13)), _hsv(rng, (0.2, 0.45), (0.45, 0.8), h=rng.uniform(0.06, 0.12))
    if kind == "wood":
        return _hsv(rng, (0.35, 0.7), (0.3, 0.7), h=rng.uniform(0.05, 0.1)), _hsv(rng, (0.35, 0.7), (0.2, 0.55), h=rng.uniform(0.04, 0.09))
    h = rng.uniform(0, 1)
    return _hsv(rng, (0.0, 0.35), (0.35, 0.85), h=h), _hsv(rng, (0.0, 0.35), (0.25, 0.75), h=h + rng.uniform(-0.1, 0.1))


def background(rng, sc, env):
    """방 (B.room 이 만든 floor · back · left · right) 의 겉과 world 를 바꾼다. env 에 기록."""
    ob = bpy.data.objects
    mode = "open" if rng.random() < P_OPEN else "room"
    if "pit_bottom" in ob:                         # 구덩이는 먼 바닥 판이 덮어 버린다 → 방으로
        mode = "room"
    fk = str(rng.choice(FLOOR_KINDS))
    f1, f2 = _floor_colors(rng, fk)
    fm = pattern_material("floor_bg", fk, rng, f1, f2, float(rng.uniform(0.3, 2.0)))
    for name in ("floor", "floor_r", "pit_bottom"):
        if name in ob:
            ob[name].data.materials.clear()
            ob[name].data.materials.append(fm)
    wk = str(rng.choice(WALL_KINDS if mode == "room" else ("brick", "stone", "wood", "plaster", "plain")))
    w1, w2 = _hsv(rng, (0.0, 0.4), (0.4, 0.92)), _hsv(rng, (0.0, 0.4), (0.3, 0.85))
    if wk == "wood":
        w1, w2 = _floor_colors(rng, "wood")
    wm = pattern_material("wall_bg", wk, rng, w1, w2, float(rng.uniform(0.3, 1.5)))
    for name in ("back", "left", "right"):
        ob[name].data.materials.clear()
        ob[name].data.materials.append(wm)
    world = sc.world
    nt = world.node_tree
    bg = nt.nodes["Background"]
    rec = dict(bg_mode=mode, floor_kind=fk, wall_kind=wk)
    if mode == "open":
        ob["back"].hide_render = True
        bpy.ops.mesh.primitive_plane_add(size=120, location=(0, 0, -0.002))   # 지평선까지 이어진 바닥 (보이기만)
        gp = bpy.context.object
        gp.name = "ground_far"
        gp.data.materials.append(fm)
        files = sorted(f for f in os.listdir(HDRI_DIR) if f.endswith(".hdr"))
        hf = files[int(rng.integers(len(files)))]
        env_t = nt.nodes.new("ShaderNodeTexEnvironment")
        env_t.image = bpy.data.images.load(os.path.join(HDRI_DIR, hf), check_existing=True)
        mpg = nt.nodes.new("ShaderNodeMapping")
        tcg = nt.nodes.new("ShaderNodeTexCoord")
        rot = float(rng.uniform(0, 2 * math.pi))
        mpg.inputs["Rotation"].default_value = (0, 0, rot)
        nt.links.new(tcg.outputs["Generated"], mpg.inputs["Vector"])
        nt.links.new(mpg.outputs["Vector"], env_t.inputs["Vector"])
        nt.links.new(env_t.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = float(rng.uniform(0.6, 1.4))
        for l in [o for o in ob if o.type == "LIGHT"]:
            l.data.energy *= float(rng.uniform(0.3, 0.8))
        rec.update(hdri=hf[:-4], hdri_rot=round(rot, 3))
    else:
        bg.inputs["Color"].default_value = [B.srgb2lin(c) for c in _hsv(rng, (0.0, 0.25), (0.5, 0.9))] + [1.0]
        for l in [o for o in ob if o.type == "LIGHT"]:
            l.data.color = colorsys.hsv_to_rgb(rng.uniform(0.05, 0.6), rng.uniform(0.0, 0.25), 1.0)
            l.data.energy *= float(rng.uniform(0.7, 1.3))
    env.update(rec)
    return rec


# ----------------------------------------------------------------------------- 문맥 끝 고정 (2026-10-25)
# 사용자: "3 stride 로 샘플링 … 문맥 C · 예측 K ∈ {2, 4, 8, 16, 32} … 문맥 C 마지막에 가려지도록 … 가려지는 타이밍이 2 · 4 · 8 · 16 · 32 에 … 분포 균등 …
# 사건의 시작 (ledge 등) 도 문맥 끝에 … 문맥 8 이면 실제로 24 프레임이니 충돌은 25 프레임부터" · "애초에 4 fps 로 64 프레임" · "C 가 2 나 4 면 그냥 넘어가야지.
# 대신 문맥 진행 중에도 가려지는 거" · "절반은 사건 근처로 … 이후 사건이 어떻게 진행됐는지는 보이도록".
# 영상 = 저장 64 장 (4 fps, 렌더 192 장을 3 장마다). 영상마다 C* 를 C_SET 에서 고르게 뽑고 앵커 종류를 고른다:
#   event     — 사건 (충돌 dv ≥ DV_ROW · 받침 이탈) 이 저장 장 C* (= 예측 첫 장) 에 온다. 문맥 C* 장 동안은 사건 전
#   occ_event — 문맥 마지막 장 (C* − 1) 에 가려지기 시작하고, 가려진 동안 사건이 일어나고, 사건 뒤 다시 보인다 (결과가 예측 안에 보임)
#   occ_any   — 문맥 마지막 장에 가려지기 시작 (사건과 무관한 움직이는 때)
# 앞을 넉넉히 시뮬한 뒤 영상 시작 s 를 (앵커 렌더 장 − 3 C*) 로 잘라 맞춘다 — 영상 첫 장부터 물체가 이미 움직인다 (공중 붙잡기 없음).
# 그다음 C* ≥ 8 이면 P_MID_OCC 확률로 문맥 안에 가림 하나 더 (문맥 진행 중 가려졌다 다시 보임, 앵커 전에 끝남).
C_SET = (2, 4, 8, 16, 32)
T_SAVE, ST = 64, 3
W_RAW = T_SAVE * ST                           # 192
N_SIM = 330                                   # 렌더 장 (27.5 초) — 앵커가 3 C* ~ N_SIM − (W_RAW − 3 C*) 사이에 와야 한다
ANCHORS = ("event", "occ_event", "occ_any")    # 가림 둘 중 하나는 사건 근처 (사용자 "절반은 사건 근처")
P_MID_OCC = 0.5


P_ANCHOR_MOVING = 0.8                         # 앵커 때 움직이는 물체 80 % · 멈춘 물체 20 % (모든 C 같게)
MAX_CTX_STILL = 0.5                           # 문맥 안 장면 정지 장 비율 상한 (모든 C 같게)
P_ARRIVE = (0.4, 0.3, 0.2, 0.1)               # 나중에 들어오는 물체 0 · 1 · 2 · 3 개


def arrivals(rng, objs, info):
    """나중에 들어오는 물체 — 화면 위 밖 (z 6.5) 에서 기다렸다가 아무 시각에 놓여 떨어져 들어온다 (화면 안에서 뜨지 않는다).
    모든 영상에 같은 분포 (C 와 무관) — 장면이 16 초 창 끝까지 살아 있게 (사용자 2026-10-25 "조금 더 늘리면"). 놓는 장은 unpred."""
    m = int(rng.choice(len(P_ARRIVE), p=P_ARRIVE))
    for i in range(m):
        o = obj(rng, (float(rng.uniform(-2.6, 2.6)), 0, 6.5), size=-0.55, rot=tuple(rng.uniform(0, 3.14, 3)), name="arrive")
        delay = int(rng.integers(12, N_SIM - 60))
        B.launch(o, (float(rng.uniform(-1.0, 1.0)), 0, 0), at=PRE - 2 + delay)
        o["role"] = "arrival"
        objs.append(o)
        info.setdefault("unpred", []).append(delay - 2)
    if m:
        info["enter_from_top"] = True
        info["n_arrivals"] = m


@scenario("half_pipe", "gravity", "반원통 계곡 (두 경사 + 바닥, 뒷벽에 붙음) 에서 물체가 여러 번 왕복하며 서서히 멈춤 — 마찰 낮음")
def s_half_pipe(rng, n):
    a = math.radians(rng.uniform(15, 35))
    w = float(rng.uniform(0.6, 2.0))               # 가운데 평평한 폭
    L = float(rng.uniform(2.0, 2.8))
    cx = float(rng.uniform(-0.5, 0.5))
    m = material("pipe", B.rand_rgb(rng, 80, 200), 0.5)
    fr = float(rng.uniform(0.02, 0.2))
    slab("pipe_l", cx - w / 2, 0.0, -1, L, a, rng, t=0.4, mat=m, friction=fr, bounce=0.1, back=True)
    slab("pipe_r", cx + w / 2, 0.0, 1, L, a, rng, t=0.4, mat=m, friction=fr, bounce=0.1, back=True)
    d = float(rng.choice([-1, 1]))
    top = Vector((cx + d * (w / 2 + L * math.cos(a)), 0, L * math.sin(a)))
    shp = str(rng.choice(ROLLERS)) if rng.random() < 0.7 else None
    o = obj(rng, (0, 0, 4), size=-0.5, shape=shp, friction=float(rng.uniform(0.02, 0.2)) if shp else 0.0)
    o.rigid_body.linear_damping = 0.0
    on_slope(o, top, -d, a, float(rng.uniform(0.3, 0.9)))
    objs = [o]
    if rng.random() < 0.35:                        # 둘째 물체 — 반대쪽에서 출발해 가운데서 부딪힘
        top2 = Vector((cx - d * (w / 2 + L * math.cos(a)), 0, L * math.sin(a)))
        o2 = obj(rng, (0, 0, 4), size=-0.5, shape=shp, friction=float(rng.uniform(0.02, 0.2)) if shp else 0.0)
        on_slope(o2, top2, d, a, float(rng.uniform(0.3, 0.9)))
        objs.append(o2)
    return objs, {}


@scenario("billiards", "collision", "마찰 거의 0 · 반발 큰 물체 3~5 개가 방 안에서 오래 서로 · 벽과 부딪히며 돌아다님")
def s_billiards(rng, n):
    objs, xs = [], rng.permutation(np.linspace(-2.6, 2.6, 7))
    for i in range(int(rng.integers(3, 6))):
        o = obj(rng, (float(xs[i]), 0, 1), size=-0.5, shape=str(rng.choice(("sphere", "cylinder", "disc", "cube", "hex_prism", "rounded_cube"))),
                friction=float(rng.uniform(0.0, 0.03)), bounce=float(rng.uniform(0.8, 0.95)))
        o.rigid_body.linear_damping = o.rigid_body.angular_damping = 0.0
        place_on(o, 0)
        launch(o, (float(rng.uniform(-3, 3)), 0, 0))
        objs.append(o)
    for nm in ("left", "right"):
        bpy.data.objects[nm].rigid_body.restitution = float(rng.uniform(0.8, 0.95))
    return objs, {"require_impact": "objects"}



def _runs(m):
    """bool 배열의 True 연속 구간 [(a, b), …] (b 미포함)."""
    out, a = [], None
    for i, x in enumerate(m):
        if x and a is None:
            a = i
        elif not x and a is not None:
            out.append((a, i)); a = None
    if a is not None:
        out.append((a, len(m)))
    return out


def _save_sub(lab, s):
    """렌더 장 [s, s + W_RAW) 를 3 장마다 → 저장 64 장. 속도 · 가속도 · 각속도는 남긴 장에서 다시 미분 (단위 = 저장 장당)."""
    keep = np.arange(s, s + W_RAW, ST)
    out = {k: (v[keep] if v.shape[:1] == (N_SIM,) else v) for k, v in lab.items()}
    g = lambda a: np.gradient(a.astype(np.float64), axis=0).astype(np.float32)
    out["vel"], out["vel3d"] = g(out["pos"]), g(out["pos3d"])
    out["acc"] = g(out["vel"])
    out["omega"] = g(np.unwrap(out["angle"].astype(np.float64), axis=0))
    return out


def fast_label(sc, objs, res, frames=(), which=()):
    """simulate_and_label 과 같은 키 · 단위 (N_SIM 장). 보임 (ray 200 점) 은 frames × which 칸만 재고, 나머지는 중심이 화면 안이면 1 (보임) —
    라벨 계산이 병목이라 (2026-10-25 시험: 1 분에 13.6 편) 앵커를 고를 때는 위치만, 가림을 맞춰 볼 때는 주 물체 · 앵커 주변만, 마지막에 창 192 장만 전부 잰다."""
    cam, T, N = sc.camera, N_SIM, len(objs)
    fr = np.zeros(T, bool)
    fr[[t for t in frames if 0 <= t < T]] = True
    wh = set(which)
    pts = {i: B.sample_points(objs[i]) for i in wh}
    pos3d = np.zeros((T, N, 3)); pos = np.zeros((T, N, 2)); inframe = np.zeros((T, N)); vis = np.zeros((T, N))
    ang = np.zeros((T, N)); bbox = np.zeros((T, N, 4))
    for f in range(1, PRE + T):
        sc.frame_set(f)
        t = f - PRE
        if t < 0:
            continue
        for i, o in enumerate(objs):
            w = o.matrix_world.translation
            pos3d[t, i] = w[:]
            ang[t, i] = -o.matrix_world.to_euler()[1]
            c = B.world_to_camera_view(sc, cam, w)
            pos[t, i] = (c.x * res, (1 - c.y) * res)
            if fr[t] and i in wh:
                inframe[t, i], vis[t, i], bbox[t, i] = B.visibility(sc, cam, o, pts[i], res)
            else:
                inside = float(0 <= c.x <= 1 and 0 <= c.y <= 1 and c.z > 0)
                inframe[t, i] = vis[t, i] = inside
                bbox[t, i] = (pos[t, i, 0], pos[t, i, 1], pos[t, i, 0], pos[t, i, 1])
    grad = lambda a: np.gradient(a, axis=0)
    vel = grad(pos)
    return dict(pos=pos.astype(np.float32), vel=vel.astype(np.float32), acc=grad(vel).astype(np.float32),
                angle=ang.astype(np.float32), omega=grad(np.unwrap(ang, axis=0)).astype(np.float32), bbox=bbox.astype(np.float32),
                pos3d=pos3d.astype(np.float32), vel3d=grad(pos3d).astype(np.float32),
                inframe_frac=inframe.astype(np.float32), vis_frac=vis.astype(np.float32))


def save_for_render(sc, stem, s):
    """라벨 · 렌더 분리 (2026-10-25): 물리를 키프레임으로 구워 `<stem>_r.blend` 로 저장 — 렌더 워커 (render_queue_blender.py, GPU 에 오래 사는 프로세스) 가 연다.
    GPU 워커 수십 개가 렌더마다 CUDA 를 새로 잡으면 드라이버 잠금 (os_acquire_rwlock) 에서 줄을 서 멈췄다 (128 개 중 84 개 D 상태)."""
    act = [o for o in bpy.data.objects if o.rigid_body is not None and o.rigid_body.type == "ACTIVE"]
    frames = list(range(PRE + s, PRE + s + W_RAW, ST))            # 렌더할 장만
    mats = {o.name: [] for o in act}
    for f in range(1, frames[-1] + 1):                            # 시뮬은 처음부터 차례로 (캐시)
        sc.frame_set(f)
        if f in frames:
            for o in act:
                mats[o.name].append(o.matrix_world.copy())
    # bpy.ops.rigidbody.bake_to_keyframes 는 background 에서 context 가 안 맞아 죽는다 → 직접: 애니메이션을 비우고 kinematic (키프레임을 따름) 으로, 렌더 장마다 위치 · 회전 키
    for o in act:
        o.animation_data_clear()
        o.rigid_body.kinematic = True
        o.rotation_mode = "QUATERNION"
        for f, m in zip(frames, mats[o.name]):
            loc, rot, _sc = m.decompose()
            o.location, o.rotation_quaternion = loc, rot
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("rotation_quaternion", frame=f)
    sc["clip_start_render"] = int(s)
    tmp = stem + "_tmp.blend"
    bpy.ops.wm.save_as_mainfile(filepath=tmp, compress=True, check_existing=False)
    os.replace(tmp, stem + "_r.blend")


def generate2(name, k, seed, main_ident, out, n_frames=None, res=256, engine="cpu", max_attempts=40, keep_frames=False):
    global G
    stem = os.path.join(out, name, f"{name}_{k:05d}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    frames_dir = stem + "_frames"
    rk = np.random.default_rng([seed, list(SCENARIOS).index(name), 11])
    Cs = int(C_SET[k % len(C_SET)])               # 작업 k 로 층화 — 시나리오마다 C 가 같은 수 (C 별 시나리오 구성이 같다, 공정성)
    want_move = bool(rk.random() < P_ANCHOR_MOVING)   # 앵커 때 주 물체가 움직이는가 — 모든 C 에 같은 비율
    # C* ≤ 4: 사건 · 사건 가림 · 그냥 가림 1/3 씩. C* ≥ 8: 문맥 끝은 가림 (사건 가림 · 그냥 가림 반반) — 충돌 같은 사건은 문맥 안에서 먼저 일어나도 된다
    # (사용자 2026-10-25 "충돌 같은 건 이전에 일어나도 되는데 context 8, 16, 32 에선 가려지는 걸 넣던가"). 사건 가림을 못 하면 그냥 가림으로 (기록)
    anchor = str(os.environ.get("PHYS3D_ANCHOR") or (rk.choice(ANCHORS) if Cs <= 4 else rk.choice(ANCHORS[1:])))
    anchor_asked = anchor
    why = []
    for attempt in range(max_attempts):
        anchor = anchor_asked
        rng = np.random.default_rng([seed, list(SCENARIOS).index(name), attempt])
        sc = B.reset()
        G = B.G = loguni(np.random.default_rng([seed, 5, attempt]), *G_RANGE)
        STATE["speed"] = loguni(np.random.default_rng([seed, 6, attempt]), *SPEED_RANGE)
        sc.gravity = (0, 0, -G)
        env = B.room(rng)
        env.update(g=G, speed_scale=STATE["speed"])
        cat, desc, fn = SCENARIOS[name]
        objs, info = fn(rng, N_SIM)
        arrivals(np.random.default_rng([seed, list(SCENARIOS).index(name), attempt, 14]), objs, info)
        if os.environ.get("PHYS3D_BG", "1") == "1":
            background(np.random.default_rng([seed, list(SCENARIOS).index(name), attempt, 13]), sc, env)
        sc.frame_end = PRE + N_SIM - 1
        rw = sc.rigidbody_world
        rw.point_cache.frame_start, rw.point_cache.frame_end = 1, sc.frame_end
        # 투영 (world_to_camera_view) 은 해상도 · 종횡비만 쓴다. 렌더 설정 (setup_render) 은 통과한 뒤 렌더 직전에 한 번 —
        # Cycles 디노이저 장치 목록 조회가 시도마다 2.8 초씩, 전체의 64 % 였다 (2026-10-25 cProfile)
        sc.render.resolution_x = sc.render.resolution_y = res
        sc.render.resolution_percentage = 100
        lab = fast_label(sc, objs, res)              # 위치만 (보임은 중심 근사)
        ra = np.random.default_rng([seed, attempt, 12])
        ev = [e for e in B.events(lab) if e["kind"] == "leave" or (e["kind"] == "impact" and e["dv"] >= B.DV_ROW)]
        lo_t, hi_t = ST * Cs, N_SIM - (W_RAW - ST * Cs)                 # 앵커 렌더 장이 들어갈 수 있는 범위
        occ_desc, s, anchor_row = [], None, None
        if anchor == "event":
            cand = [e for e in ev if lo_t <= e["frame"] <= hi_t]
            if not cand:
                why = [f"no event in [{lo_t}, {hi_t}]"]
                continue
            e = cand[int(ra.integers(len(cand)))]
            s = e["frame"] - ST * Cs
            vis = fast_label(sc, objs, res, range(e["frame"] - ST, e["frame"] + ST + 1), [e["obj"]])["vis_frac"][:, e["obj"]]
            if (vis[max(0, e["frame"] - ST):e["frame"] + ST + 1] <= B.SEEN).any():
                why = ["event not visible"]
                continue
            anchor_row = dict(event=e["kind"], event_frame=Cs, obj=int(e["obj"]), dv=round(float(e["dv"]), 3), pred_min=1)
        else:
            ok = False
            for _ in range(OCC_TRIES):
                mode = str(ra.choice(OCC_MODES if want_move or anchor == "occ_event" else OCC_MODES[1:]))   # 멈춘 물체는 고정 판으로 못 가린다
                if anchor == "occ_event":
                    cand = [e for e in ev if e["obj"] == 0 and lo_t + 6 <= e["frame"] <= hi_t]
                    if not cand:
                        anchor = "occ_any"                  # 사건 가림을 못 함 → 그냥 가림 (meta anchor_asked 에 남는다)
                        e, t0 = None, None
                    else:
                        e = cand[int(ra.integers(len(cand)))]
                        t0 = e["frame"]
                else:
                    e, t0 = None, None
                res_o = place_occlusion(mode, ra, sc, objs, lab, hi_t + 10, t0=t0, lo=lo_t, need_move=None if e else want_move)
                if res_o is None:
                    why = ["no occlusion time"]
                    continue
                made, front, d0 = res_o
                tc = e["frame"] if e else d0["t_star"]
                sv0 = (np.linalg.norm(lab["vel3d"][max(0, tc - ST * Cs):tc], axis=-1) * B.FPS < STILL_V).all(1)
                if sv0.mean() > MAX_CTX_STILL:              # 문맥이 정지면 보임을 재기 전에 버린다 (싸게)
                    why = [f"context still {sv0.mean():.2f}"]
                    for ob in made + front:
                        bpy.data.objects.remove(ob, do_unlink=True)
                    continue
                r0, r1 = max(0, tc - MAX_BLIND - 3 * ST), min(N_SIM, tc + MAX_BLIND + 3 * ST)
                lab2 = fast_label(sc, objs + front, res, range(r0, r1), [0])
                runs = [(a, b) for a, b in _runs(lab2["vis_frac"][:, 0] <= B.SEEN) if b - a >= ST and a > r0 and b < r1]
                hit = [(a, b) for a, b in runs if a <= tc < b]
                good = False
                if hit:
                    h0, h1 = hit[0]
                    s0 = h0 - ST * (Cs - 1)                         # 가려지기 시작하는 장 = 저장 장 C* − 1 (문맥 마지막 장)
                    good = (s0 >= 0 and s0 + W_RAW <= N_SIM and h1 - h0 <= MAX_BLIND and h1 < s0 + W_RAW - 2 * ST
                            and (e is None or (e["frame"] - h0 >= ST and h1 - e["frame"] >= ST)))
                    if good:                                         # 3 장마다 1 장에서도: C* − 2 는 보이고 C* − 1 은 안 보임
                        g = lab2["vis_frac"][s0:s0 + W_RAW:ST, 0] <= B.SEEN
                        good = bool(g[Cs - 1]) and (Cs < 2 or not g[Cs - 2])
                if good:
                    objs, lab, s = objs + front, lab2, s0
                    for f in front:
                        f["role"] = "front"
                    d0.update(kind="anchor", hidden=[int(h0 - s0), int(h1 - s0)])
                    occ_desc.append(d0)
                    reap = Cs - 1 + -(-(h1 - h0) // ST)          # 다시 보이는 첫 저장 장
                    anchor_row = dict(event="occlude" if e is None else f"occlude_{e['kind']}", event_frame=Cs - 1, obj=0,
                                      dv=round(float(e["dv"]), 3) if e else 0.0, pred_min=int(reap - Cs + 1))
                    ok = True
                    break
                why = [f"anchor occ {mode} bad"]
                for ob in made + front:
                    bpy.data.objects.remove(ob, do_unlink=True)
            if not ok:
                continue
        # 문맥 진행 중 가림 (앵커 전에 끝남)
        if Cs >= 8 and ra.random() < P_MID_OCC:
            g_old = fast_label(sc, objs, res, range(s, s + W_RAW), [0])["vis_frac"][s:s + W_RAW:ST, 0] <= B.SEEN
            for _ in range(4):
                mode = str(ra.choice(OCC_MODES))
                lo1, hi1 = s + ST * 2, s + ST * (Cs - 4)
                okt = [t for t in range(lo1, hi1) if lab["inframe_frac"][t, 0] > 0.9]
                if not okt:
                    break
                res_o = place_occlusion(mode, ra, sc, objs, lab, N_SIM, t0=int(ra.choice(okt)))
                if res_o is None:
                    continue
                made, front, d1 = res_o
                lab2 = fast_label(sc, objs + front, res, range(s, s + W_RAW), [0])
                g_new = lab2["vis_frac"][s:s + W_RAW:ST, 0] <= B.SEEN
                extra = g_new & ~g_old
                ex_idx = np.nonzero(extra)[0]
                ok_mid = (len(ex_idx) >= 1 and ex_idx.max() <= Cs - 3 and bool((g_new[Cs - 2:] == g_old[Cs - 2:]).all())
                          and blind_ok(lab2["vis_frac"][s:s + W_RAW, 0], W_RAW)[0])
                if ok_mid:
                    objs, lab = objs + front, lab2
                    for f in front:
                        f["role"] = "front"
                    d1.update(kind="mid_context", hidden_saved=[int(ex_idx.min()), int(ex_idx.max()) + 1])
                    occ_desc.append(d1)
                    break
                for ob in made + front:
                    bpy.data.objects.remove(ob, do_unlink=True)
        sv = (np.linalg.norm(lab["vel3d"][s:s + ST * Cs], axis=-1) * B.FPS < STILL_V).all(1)
        if sv.mean() > MAX_CTX_STILL:
            why = [f"context still {sv.mean():.2f}"]
            continue
        lab = fast_label(sc, objs, res, range(s, s + W_RAW, ST), range(len(objs)))   # 마지막: 저장할 64 장 · 모든 물체 (보임 ray 는 저장 장만)
        lw = {k2: (v[s:s + W_RAW] if v.shape[:1] == (N_SIM,) else v) for k2, v in lab.items()}
        evw = [dict(e2, frame=e2["frame"] - s) for e2 in B.events(lab) if s <= e2["frame"] < s + W_RAW and e2["kind"] in ("impact", "leave")]
        L0 = {k2: (v[s:s + W_RAW:ST] if v.shape[:1] == (N_SIM,) else v) for k2, v in lab.items()}   # 가려짐 · 다시 보임은 저장 장에서
        evw += [dict(e2, frame=e2["frame"] * ST) for e2 in B.events(L0) if e2["kind"] in ("occlude", "reappear")]
        evw.sort(key=lambda e2: (e2["frame"], e2["obj"]))
        why = check(lw, [e2 for e2 in evw if e2["kind"] in ("impact", "leave")], info, objs)
        if why:
            continue
        L = _save_sub(lab, s)
        T, N = T_SAVE, len(objs)
        ev4 = [dict(e2, frame=int(round(e2["frame"] / ST))) for e2 in evw if round(e2["frame"] / ST) < T]
        unpred = sorted({-(-(u - s) // ST) for u in info.get("unpred", []) if s <= u < s + W_RAW})
        end_a = min([u for u in unpred if u >= Cs] + [T])           # 들어오는 물체 (unpred) 를 넘는 예측은 만들지 않는다
        rows = [dict(boundary=Cs, end=end_a, ctx_start=0, ctx_gate=0, anchor=anchor, **anchor_row)]
        for r2 in B.event_rows(ev4, L, unpred, np.random.default_rng([seed, attempt, 3]), T, 2, 2):
            if r2["boundary"] != Cs:
                rows.append(dict(r2, ctx_gate=r2["ctx_start"], ctx_start=0, anchor="", pred_min=1))
        impact = np.zeros((T, N), np.float32)
        for e2 in ev4:
            if e2["kind"] == "impact":
                impact[e2["frame"], e2["obj"]] = max(impact[e2["frame"], e2["obj"]], e2["dv"])
        bb = L["bbox"]
        idents = np.array([o["ident"] for o in objs])
        ratio = L["vis_frac"] / np.maximum(L["inframe_frac"], 1e-6)
        L.update(present=np.ones((T, N), bool), hidden=(L["inframe_frac"] > 0.25) & (ratio < B.HIDDEN),
                 seen=np.maximum.accumulate(L["vis_frac"] > B.SEEN, axis=0), impact=impact,
                 identity=idents.astype(np.int16), shape_id=np.array([o["shape_id"] for o in objs], np.int16),
                 color_id=(idents % NC).astype(np.int16), rgb=np.array([list(o["rgb"]) for o in objs], np.uint8),
                 rgb2=np.array([list(o["rgb2"]) for o in objs], np.uint8), pattern=np.array([o["pattern"] for o in objs], np.int8),
                 mass=np.array([o.rigid_body.mass if o.rigid_body else 0.0 for o in objs], np.float32),
                 size=np.array([o.dimensions.length / math.sqrt(3) for o in objs], np.float32),
                 area=((bb[0, :, 2] - bb[0, :, 0]) * (bb[0, :, 3] - bb[0, :, 1])).astype(np.float32))
        np.savez_compressed(stem + "_states.npz", **L)
        if os.environ.get("PHYS3D_DEFER_RENDER") == "1":
            save_for_render(sc, stem, s)
        elif os.environ.get("PHYS3D_NORENDER") != "1":
            B.setup_render(sc, engine, res, os.path.join(frames_dir, "f"))
            sc.render.use_persistent_data = False # 64 장만 렌더 — GPU 워커 수십 개가 장면을 붙잡고 있으면 GPU 메모리가 찬다 (2026-10-25: 31 개에 91.9 GB)
            if engine != "cpu":                   # OptiX 디노이저가 프로세스마다 GPU 메모리를 크게 잡아 워커 201 / 248 이 "out of GPU memory" 로 죽었다 (2026-10-25)
                sc.cycles.denoiser = "OPENIMAGEDENOISE"
                sc.cycles.denoising_use_gpu = False
            os.makedirs(frames_dir, exist_ok=True)
            sc.frame_start, sc.frame_end, sc.frame_step = PRE + s, PRE + s + W_RAW - 1, ST
            try:
                bpy.ops.render.render(animation=True)
            except RuntimeError as err:           # GPU 가 모자라면 이 영상만 CPU 로 (워커는 살린다)
                print("GPU render failed, CPU fallback:", err, flush=True)
                B.setup_render(sc, "cpu", res, os.path.join(frames_dir, "f"))
                sc.frame_start, sc.frame_end, sc.frame_step = PRE + s, PRE + s + W_RAW - 1, ST
                bpy.ops.render.render(animation=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{B.FPS / ST:g}", "-pattern_type", "glob",
                            "-i", os.path.join(frames_dir, "f*.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                            stem + ".mp4"], check=True)
            if not keep_frames:
                shutil.rmtree(frames_dir)
        meta = dict(scenario=name, category=cat, desc=desc, k=k, seed=seed, attempt=attempt, fps=B.FPS / ST, render_fps=B.FPS,
                    save_stride=ST, n_frames=T, clip_start_render=int(s), C_star=Cs, anchor=anchor, anchor_asked=anchor_asked, want_move=want_move, res=res, engine=engine, env=env,
                    unpred=unpred, info={k2: v for k2, v in info.items() if k2 != "unpred"}, occlusion=occ_desc,
                    objects=[dict(index=i, role="main" if i == 0 else o.get("role", "other"), shape=SHAPES[o["shape_id"]], rgb=list(o["rgb"]),
                                  rgb2=list(o["rgb2"]), pattern=int(o["pattern"]), identity=int(o["ident"]),
                                  mass=o.rigid_body.mass if o.rigid_body else 0.0, size=float(o.dimensions.length / math.sqrt(3)))
                             for i, o in enumerate(objs)],
                    events=ev4, rows=rows)
        with open(stem + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        return dict(ok=True, mp4=stem + ".mp4", attempt=attempt, n_rows=len(rows), n_frames=T, C_star=Cs, anchor=anchor)
    return dict(ok=False, mp4=stem + ".mp4", why=why, C_star=Cs, anchor=anchor)


B.SCENARIOS = SCENARIOS
B.generate = generate2 if os.environ.get("PHYS3D_LEGACY") != "1" else generate   # 기본 = 문맥 끝 고정 64 장 (4 fps)

if __name__ == "__main__":
    B.main()
