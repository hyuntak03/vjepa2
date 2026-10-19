"""phys3d — phys2d_sup 의 3D 판 (Blender 4.2 rigid body + 렌더) (2026-10-19). 문서 z_training/docs/PHYS3D_DATASET_2026-10-19.md

blender 안에서 돈다:
  /workspace/tools/blender-4.2.3-linux-x64/blender -b -noaudio -P z_training/data/gen_phys3d_blender.py -- \
      --scenario drop_bounce --seed 0 --out /path/to/out [--frames 96] [--engine cpu|cycles] [--res 256]
대량 생성은 gen_phys3d.py (워커 모드 --jobs --claims --results). ⚠️ --engine eevee 는 대량 생성에 쓰지 말 것 (EGL 이 GPU 0 에만 붙는다)

2D 판과 맞춘 것
  - 정체 48 = 모양 6 × 색 8, id = shape_id * 8 + color_id (색 8 은 2D 와 같은 이름 · 같은 RGB)
  - 장마다 물체별 라벨 `_states.npz`: phys2d_sup 과 같은 키 · 단위 (pos px · vel px/장 · acc · angle · omega · bbox · present ·
    vis_frac · inframe_frac · hidden · seen · impact · identity · shape_id · color_id · size · area) + pos3d · vel3d (m, m/장)
  - 사건 (json `events`): impact (dv) · leave (받침 이탈 = 자유 낙하 개시) · occlude · reappear. post = 전환 뒤 첫 장
  - 방 (바닥 · 뒷벽 · 양옆 벽) 안에서만 움직여 화면 밖 이탈이 없다

2D 와 다른 것
  - 모양은 3D 에서 구별되는 것으로 골랐다: sphere · cube · cylinder · cone · pyramid · torus
    (torus 충돌 모양은 convex hull = 원판. 구멍으로 무엇이 지나가지는 않는다)
  - 보임 비율은 표면 표본 200 점 (면적 가중) 마다 카메라에서 ray 를 쏴 처음 맞는 것이 그 물체인 비율 (자기 가림은 보임으로 센다, 유리벽은 건너뛴다)
  - 초속도는 Blender 관례대로 처음 몇 장을 kinematic 으로 움직였다 놓아서 준다. 그 장들은 렌더 앞 (PRE) 이다
"""
import argparse, json, math, os, shutil, subprocess, sys, time

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

FPS = 12
PRE = 6                   # 렌더 앞 준비 장 (발사 · 놓기)
SUBSTEPS = 10             # 120 Hz
G = 9.81                  # build() 가 클립마다 G_RANGE 에서 다시 정한다 (시나리오의 조준 계산이 이 값을 쓴다)
G_RANGE = (3.5, 9.81)
DV_IMPACT = 0.8           # m/s — 이 이상 속도 변화 (중력 몫 빼고) = 충돌
HIDDEN, SEEN = 0.02, 0.3
DV_ROW = 1.5              # m/s — boundary 행 · 문맥 제한 (ctx_start) 으로 쓰는 충돌. 2D DV_ROW 60 px/s ≈ 1.6 m/s (화면 256 px ≈ 7 m)
MAX_IMPACT_ROWS = 3       # 클립당 충돌 행 상한
P_EXTRA_SCREEN = 0.4      # 영속 시나리오가 아닌 곳에 가림막을 하나 더 둘 확률
MAX_ATTEMPTS = 60          # 버리는 시도는 싸다 (시뮬 + 라벨 1~2 초, 렌더는 통과한 것만)
APEX_MAX = 4.6            # m — 던진 물체 꼭대기 상한 (화면 위 ≈ 5.3 m)
CPU_THREADS = 1           # --engine cpu: 워커당 렌더 스레드
CPU_SAMPLES = int(os.environ.get("PHYS3D_SAMPLES", 4))    # Cycles 표본 수 + denoise. 4 와 16 이 눈으로 같고 장당 1.06 vs 2.4 초 (CPU 스레드 1)

SHAPES = ("sphere", "cube", "cylinder", "cone", "pyramid", "torus")
COLORS = (("red", (226, 40, 40)), ("orange", (244, 132, 22)), ("yellow", (236, 212, 32)), ("green", (40, 168, 62)),
          ("cyan", (32, 188, 204)), ("blue", (46, 86, 226)), ("purple", (136, 62, 200)), ("magenta", (224, 56, 166)))
NC = len(COLORS)

ROOM_X, ROOM_Y = 3.4, 2.5           # 양옆 벽 x = ±ROOM_X, 뒷벽 y = ROOM_Y
OCC_Y = -1.3                         # 가림막 평면 (물체는 y≈0 에서 움직인다)
SIZE_SCALE = 1.6                     # 시나리오가 주는 크기 (m) 에 곱한다 — 256 px 에서 물체 ≈ 30 px


def srgb2lin(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# ----------------------------------------------------------------------------- 장면 기본
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.frame_start = 1
    bpy.ops.rigidbody.world_add()
    rw = sc.rigidbody_world
    rw.substeps_per_frame = SUBSTEPS
    rw.solver_iterations = 30
    return sc


def material(name, rgb, rough=0.5, checker=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    lin = [srgb2lin(c) for c in rgb] + [1.0]
    bsdf.inputs["Base Color"].default_value = lin
    bsdf.inputs["Roughness"].default_value = rough
    if checker is not None:                       # 바닥 · 벽 무늬: 움직임 단서
        nt = m.node_tree
        ck = nt.nodes.new("ShaderNodeTexChecker")
        ck.inputs["Scale"].default_value = checker
        ck.inputs["Color1"].default_value = lin
        ck.inputs["Color2"].default_value = [x * 0.75 for x in lin[:3]] + [1.0]
        nt.links.new(ck.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def box(name, loc, size, mat, passive=True, rot=(0, 0, 0), friction=0.6, bounce=0.3):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    if passive is not None:
        bpy.ops.rigidbody.object_add(type="PASSIVE")
        o.rigid_body.collision_shape = "BOX"
        o.rigid_body.friction = friction
        o.rigid_body.restitution = bounce
    return o


def room(rng):
    sc = bpy.context.scene
    hue = rng.uniform(0.0, 1.0)
    import colorsys
    def muted(h, s, v):
        return tuple(int(255 * x) for x in colorsys.hsv_to_rgb(h % 1, s, v))
    floor_c = muted(hue, rng.uniform(0.05, 0.2), rng.uniform(0.55, 0.8))
    wall_c = muted(hue + rng.uniform(0.3, 0.7), rng.uniform(0.03, 0.15), rng.uniform(0.7, 0.9))
    bounce = float(rng.uniform(0.2, 0.6))
    fric = float(rng.uniform(0.3, 0.8))
    fm = material("floor", floor_c, 0.8, checker=float(rng.choice([4, 6, 8])))
    wm = material("wall", wall_c, 0.9)
    box("floor", (0, 0, -0.25), (2 * ROOM_X + 2, 2 * ROOM_Y + 6, 0.5), fm, friction=fric, bounce=bounce)
    box("back", (0, ROOM_Y + 0.25, 3), (2 * ROOM_X + 2, 0.5, 6), wm, bounce=bounce)
    box("left", (-ROOM_X - 0.25, 0, 3), (0.5, 2 * ROOM_Y + 6, 6), wm, bounce=bounce)
    box("right", (ROOM_X + 0.25, 0, 3), (0.5, 2 * ROOM_Y + 6, 6), wm, bounce=bounce)
    box("glass_front", (0, -0.9, 3), (2 * ROOM_X, 0.2, 6), wm, passive=True).hide_render = True   # 물체를 y≈0 근처에 묶는다
    box("glass_back", (0, 0.9, 3), (2 * ROOM_X, 0.2, 6), wm, passive=True).hide_render = True
    # 빛 · 카메라
    bpy.ops.object.light_add(type="SUN", rotation=(math.radians(rng.uniform(20, 40)), 0, math.radians(rng.uniform(-15, 15))))
    bpy.context.object.data.energy = float(rng.uniform(2.5, 4.0))
    bpy.context.object.data.angle = math.radians(8)
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.6, 0.6, 0.62, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9
    sc.world = w
    cam_x = float(rng.uniform(-0.6, 0.6))
    bpy.ops.object.camera_add(location=(cam_x, -6.0, float(rng.uniform(1.6, 2.4))))
    cam = bpy.context.object
    cam.data.lens = 29
    d = Vector((0, 0, 1.7)) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    return dict(floor_rgb=floor_c, wall_rgb=wall_c, bounce=bounce, friction=fric)


def make_object(ident, loc, size, rng, rot=None, mass=None, friction=None, bounce=None):
    shape = SHAPES[ident // NC]
    size = size * SIZE_SCALE
    r = size / 2
    if shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=32, ring_count=16, location=loc)
    elif shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=size * 0.85, location=loc)
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=size, vertices=32, location=loc)
    elif shape == "cone":
        bpy.ops.mesh.primitive_cone_add(radius1=r, depth=size, vertices=32, location=loc)
    elif shape == "pyramid":
        bpy.ops.mesh.primitive_cone_add(radius1=r * 1.15, depth=size, vertices=4, location=loc)
    elif shape == "torus":
        bpy.ops.mesh.primitive_torus_add(major_radius=r * 0.72, minor_radius=r * 0.28, location=loc,
                                         rotation=(math.pi / 2, 0, 0))
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    o = bpy.context.object
    if rot is not None:
        o.rotation_euler = rot
    bpy.ops.object.shade_smooth() if shape in ("sphere", "torus") else None
    name, rgb = COLORS[ident % NC]
    o.name = f"obj_{name}_{shape}"
    o.data.materials.append(material(o.name, rgb, 0.45))
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    rb = o.rigid_body
    rb.collision_shape = {"sphere": "SPHERE", "cube": "BOX", "cylinder": "CYLINDER", "cone": "CONE"}.get(shape, "CONVEX_HULL")
    rb.mass = mass if mass is not None else float(size ** 3 * 8)
    rb.friction = friction if friction is not None else float(rng.uniform(0.05, 0.35))   # Bullet 은 두 면 마찰을 곱한다
    rb.restitution = bounce if bounce is not None else float(rng.uniform(0.3, 0.75))
    rb.linear_damping = 0.01
    rb.angular_damping = 0.05
    rb.use_margin = True
    rb.collision_margin = 0.002
    o["ident"] = ident
    return o


def launch(o, v, at=PRE - 2):
    """초속도 v (m/s) — at−1 → at 사이를 kinematic 으로 움직였다 at+1 부터 dynamic 으로 놓는다."""
    p = o.location.copy()
    v = Vector(v)
    dt = 1.0 / FPS
    o.rigid_body.kinematic = True
    o.location = p - v * dt
    for f in (1, at - 1):
        o.keyframe_insert("location", frame=f)
    o.location = p
    o.keyframe_insert("location", frame=at)
    o.rigid_body.keyframe_insert("kinematic", frame=1)
    o.rigid_body.keyframe_insert("kinematic", frame=at)
    o.rigid_body.kinematic = False
    o.rigid_body.keyframe_insert("kinematic", frame=at + 1)
    for fc in o.animation_data.action.fcurves:
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR" if fc.data_path == "location" else "CONSTANT"


def occluder(name, x, width, height, rng, z0=0.0, anim=None):
    """카메라와 물체 사이 판 (충돌 없음). anim = [(frame, x), ...] 이면 오간다."""
    import colorsys
    c = tuple(int(255 * t) for t in colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.0, 0.12), rng.uniform(0.35, 0.6)))
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, OCC_Y, z0 + height / 2))
    o = bpy.context.object
    o.name = name
    o.scale = (width, 0.08, height)
    o.data.materials.append(material(name, c, 0.85, checker=3.0))
    if anim:
        for f, xx in anim:
            o.location.x = xx
            o.keyframe_insert("location", index=0, frame=f)
        for fc in o.animation_data.action.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = "SINE"
    return o


def pick_idents(rng, n):
    return [int(i) for i in rng.choice(len(SHAPES) * NC, size=n, replace=False)]


# ----------------------------------------------------------------------------- 시나리오
# 시나리오 함수 (rng, n_frames, main_ident) → (objs, occluders, info). objs[0] = 주 물체 (정체를 층화한다).
# info["unpred"] = 문맥으로 예측 못 하는 시작 장 (클립 장 번호) — 예측 구간은 그 전에 끊는다.
SCENARIOS = {}


def scenario(name, category, desc):
    def deco(fn):
        SCENARIOS[name] = (category, desc, fn)
        return fn
    return deco


def others(rng, main_ident, n):
    pool = [i for i in range(len(SHAPES) * NC) if i != main_ident]
    return [int(i) for i in rng.choice(pool, size=n, replace=False)]


def rand_rgb(rng, lo=70, hi=190):
    return tuple(int(x) for x in rng.integers(lo, hi, 3))


def screen_on_path(rng, xs, name="screen", k=0):
    """경로 위 x 들 중 하나 앞에 고정 가림막 — 바닥에 서거나 공중에 뜬 판."""
    x = float(rng.choice(xs))
    w = float(rng.uniform(0.8, 1.4))
    if rng.random() < 0.7:
        return occluder(f"{name}{k}", x, w, float(rng.uniform(1.4, 3.8)), rng)
    z0 = float(rng.uniform(0.6, 2.0))
    return occluder(f"{name}{k}", x, w, float(rng.uniform(1.0, 2.0)), rng, z0=z0)


@scenario("drop_bounce", "gravity", "1~3 물체를 여러 높이에서 놓거나 살짝 던짐 — 자유 낙하 · 튕김 · 서로 부딪힘")
def s_drop(rng, n_frames, main_ident):
    objs = []
    n = int(rng.integers(1, 4))
    xs = rng.permutation(np.linspace(-2.2, 2.2, 5))[:n]
    for i, ident in enumerate([main_ident] + others(rng, main_ident, n - 1)):
        o = make_object(ident, (float(xs[i]), 0, float(rng.uniform(1.8, 4.5))), float(rng.uniform(0.4, 0.7)), rng,
                        rot=tuple(rng.uniform(0, 3.14, 3)))
        launch(o, (rng.uniform(-1.5, 1.5), 0, rng.uniform(-1, 2.5)))
        objs.append(o)
    return objs, [], {}


@scenario("projectile", "gravity", "옆에서 던진 물체의 포물선 — 바닥 · 옆벽에서 튕김")
def s_proj(rng, n_frames, main_ident):
    side = rng.choice([-1, 1])
    z0 = float(rng.uniform(0.6, 2.5))
    o = make_object(main_ident, (float(-side * 2.7), 0, z0), float(rng.uniform(0.4, 0.6)), rng)
    apex = float(rng.uniform(max(z0, 2.2) + 0.3, APEX_MAX))
    launch(o, (side * rng.uniform(1.5, 4.0), 0, math.sqrt(2 * G * (apex - z0))))
    return [o], [], {}


@scenario("toss_up", "gravity", "위로 던진 물체가 꼭대기에서 멈췄다 되돌아 떨어짐 — 높이 · 수평 속도 무작위")
def s_toss(rng, n_frames, main_ident):
    z0 = float(rng.uniform(0.4, 1.2))
    o = make_object(main_ident, (float(rng.uniform(-2.0, 2.0)), 0, z0), float(rng.uniform(0.4, 0.6)), rng)
    launch(o, (rng.uniform(-1.2, 1.2), 0, math.sqrt(2 * G * (rng.uniform(2.8, APEX_MAX) - z0))))
    return [o], [], {}


@scenario("ramp_roll", "gravity", "경사면을 굴러 · 미끄러져 내려와 바닥으로 — 경사 각 · 높이 무작위")
def s_ramp(rng, n_frames, main_ident):
    side = rng.choice([-1, 1])
    ang = math.radians(rng.uniform(18, 34))
    L = 3.0
    rm = material("ramp", rand_rgb(rng, 90, 200), 0.7)
    cx = -side * 1.6
    h0 = float(rng.uniform(0.0, 0.8))
    box("ramp", (cx, 0, h0 + L / 2 * math.sin(ang)), (L, 1.6, 0.12), rm, rot=(0, side * ang, 0),
        friction=float(rng.uniform(0.2, 0.9)))
    if h0 > 0.05:
        box("ramp_base", (cx - side * (L / 2 * math.cos(ang) - 0.2), 0, h0 / 2), (0.3, 1.2, h0), rm)
    top = Vector((cx - side * (L / 2 - 0.3) * math.cos(ang), 0, h0 + (L - 0.6) * math.sin(ang) + 0.5))
    rot = (math.pi / 2, 0, 0) if SHAPES[main_ident // NC] == "cylinder" and rng.random() < 0.7 else None
    o = make_object(main_ident, tuple(top), 0.5, rng, rot=rot, friction=float(rng.uniform(0.02, 0.4)))
    launch(o, (side * rng.uniform(0.0, 1.0), 0, 0))
    return [o], [], {}


@scenario("ledge_fall", "gravity", "선반 위를 미끄러져 · 굴러 가다 끝에서 떨어짐 (중력 개시) — 선반 높이 · 속도 무작위")
def s_ledge(rng, n_frames, main_ident):
    side = rng.choice([-1, 1])
    h = float(rng.uniform(0.9, 2.4))
    w = float(rng.uniform(1.8, 3.2))
    tm = material("table", rand_rgb(rng, 80, 190), 0.7)
    cx = -side * (ROOM_X - w / 2)
    box("table", (cx, 0, h / 2), (w, 1.6, h), tm, friction=float(rng.uniform(0.2, 0.6)))
    o = make_object(main_ident, (cx - side * (w / 2 - 0.6), 0, h + 0.45), 0.5, rng, friction=float(rng.uniform(0.02, 0.1)))
    launch(o, (side * rng.uniform(0.8, 2.5), 0, 0))
    return [o], [], {}


@scenario("pendulum", "gravity", "진자 (막대 + 추) — 길이 · 진폭 무작위, 앞에 가림막 (영속) 또는 옆에 막는 상자 (고체성)")
def s_pend(rng, n_frames, main_ident):
    L = float(rng.uniform(1.4, 2.6))
    pz = float(rng.uniform(3.3, 4.2))
    px = float(rng.uniform(-1.0, 1.0))
    dark = material("pivot", (40, 40, 40), 0.5)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=(px, 0, pz))
    pivot = bpy.context.object
    pivot.data.materials.append(dark)
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    pivot.rigid_body.collision_shape = "SPHERE"
    pivot.rigid_body.collision_collections[0] = False
    pivot.rigid_body.collision_collections[1] = True
    th = math.radians(rng.uniform(30, 75)) * rng.choice([-1, 1])
    bob_c = Vector((px + L * math.sin(th), 0, pz - L * math.cos(th)))
    mid = (Vector((px, 0, pz)) + bob_c) / 2
    bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth=L, location=mid, rotation=(0, -th, 0))
    rod = bpy.context.object
    rod.data.materials.append(dark)
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    rod.rigid_body.mass = 0.05
    rod.rigid_body.collision_collections[0] = False
    rod.rigid_body.collision_collections[1] = True
    bob = make_object(main_ident, tuple(bob_c), float(rng.uniform(0.4, 0.55)), rng, mass=2.0)
    bob.rigid_body.angular_damping = 0.0
    bob.rigid_body.linear_damping = 0.0
    rod.rigid_body.linear_damping = rod.rigid_body.angular_damping = 0.0
    for a, b, kind, loc in ((pivot, rod, "POINT", Vector((px, 0, pz))), (rod, bob, "FIXED", bob_c)):
        bpy.ops.object.empty_add(location=loc)
        e = bpy.context.object
        bpy.ops.rigidbody.constraint_add(type=kind)
        e.rigid_body_constraint.object1, e.rigid_body_constraint.object2 = a, b
    occ = []
    r = rng.random()
    if r < 0.55:
        occ.append(occluder("screen", float(px + rng.uniform(-0.8, 0.8)), float(rng.uniform(0.8, 1.4)), float(pz - 0.3), rng))
    elif r < 0.9:                                      # 반대편 스윙 길에 상자 — 추가 부딪혀 튕긴다
        bx = px - math.copysign(L * rng.uniform(0.45, 0.8), th)
        bh = pz - L * rng.uniform(0.55, 0.9)
        box("block", (float(bx), 0, bh / 2), (0.35, 1.2, float(bh)), material("block", rand_rgb(rng), 0.7),
            bounce=float(rng.uniform(0.3, 0.8)))
    return [bob], occ, {}


@scenario("occ_pass", "permanence", "미끄러지는 · 던진 물체가 고정 가림막 1~2 개 뒤를 지나 다시 나옴 (가려진 동안에도 튕김)")
def s_occ_pass(rng, n_frames, main_ident):
    side = rng.choice([-1, 1])
    throw = rng.random() < 0.5
    z = float(rng.uniform(1.0, 2.5)) if throw else 0.45
    o = make_object(main_ident, (float(-side * 2.8), 0, z), 0.5, rng, friction=None if throw else float(rng.uniform(0.01, 0.06)))
    launch(o, (side * rng.uniform(2.0, 4.0), 0, rng.uniform(1.0, 4.0) if throw else 0))
    xs = [x * side for x in (-1.2, 0.0, 1.2)]
    occ = [screen_on_path(rng, xs[k::2] if int(rng.integers(1, 3)) == 2 else xs, k=k) for k in range(int(rng.integers(1, 3)))]
    return [o], occ, {}


@scenario("moving_screen", "permanence", "가림막이 오가며 물체를 여러 번 가림 — 물체는 정지 · 미끄러짐 · 튕김")
def s_moving_screen(rng, n_frames, main_ident):
    objs = []
    for i, ident in enumerate([main_ident] + others(rng, main_ident, int(rng.integers(0, 2)))):
        air = rng.random() < 0.4
        o = make_object(ident, (float(rng.uniform(-2, 2)), 0, float(rng.uniform(1, 3)) if air else 0.45), 0.5, rng,
                        friction=float(rng.uniform(0.01, 0.3)))
        launch(o, (rng.uniform(-2.5, 2.5), 0, rng.uniform(0, 3) if air else 0))
        objs.append(o)
    n_pass = int(rng.integers(2, 5))
    x0 = float(rng.choice([-1, 1]) * 3.0)
    keys, f = [], 1
    step = (n_frames + PRE) / n_pass
    for k in range(n_pass + 1):
        keys.append((int(f), x0 if k % 2 == 0 else -x0))
        f += step
    occ = [occluder("moving_screen", x0, float(rng.uniform(1.0, 1.6)), float(rng.uniform(2.5, 4.0)), rng, anim=keys)]
    return objs, occ, {}


@scenario("drop_behind", "permanence", "물체가 가림막 뒤로 떨어져 (튕기고) 멈춤, 그다음 가림막이 비켜나며 드러남")
def s_drop_behind(rng, n_frames, main_ident):
    x = float(rng.uniform(-1.8, 1.8))
    o = make_object(main_ident, (x + float(rng.uniform(-0.3, 0.3)), 0, float(rng.uniform(3.0, 4.5))), 0.5, rng,
                    rot=tuple(rng.uniform(0, 3.14, 3)))
    launch(o, (rng.uniform(-0.4, 0.4), 0, rng.uniform(-1, 1)))
    w = float(rng.uniform(1.5, 2.2))
    t_move = int(rng.integers(36, 60))                   # 클립 장 — 가림막 출발은 예측 못 한다
    dx = float(rng.choice([-1, 1]) * rng.uniform(2.5, 3.5))
    keys = [(1, x), (PRE + t_move, x), (PRE + t_move + int(rng.integers(8, 16)), x + dx)]
    occ = [occluder("screen", x, w, float(rng.uniform(2.0, 2.8)), rng, anim=keys)]
    return [o], occ, {"unpred": [t_move]}


@scenario("shelf_land", "solidity", "선반 · 판 여러 개 위로 떨어져 그 위에 서거나 튕겨 내려감 — 통과하지 않는다")
def s_shelf(rng, n_frames, main_ident):
    objs = []
    n_sh = int(rng.integers(1, 3))
    xs = rng.permutation([-1.6, 0.0, 1.6])[:n_sh]
    for k, x in enumerate(xs):
        box(f"shelf{k}", (float(x), 0, float(rng.uniform(0.8, 2.2))), (float(rng.uniform(1.2, 2.0)), 1.6, 0.12),
            material(f"shelf{k}", rand_rgb(rng), 0.7), rot=(0, float(rng.uniform(-0.25, 0.25)), 0), bounce=float(rng.uniform(0.2, 0.7)))
    n = int(rng.integers(1, 3))
    for i, ident in enumerate([main_ident] + others(rng, main_ident, n - 1)):
        o = make_object(ident, (float(xs[i % n_sh] + rng.uniform(-0.5, 0.5)), 0, float(rng.uniform(3.2, 4.6))), 0.45, rng)
        launch(o, (rng.uniform(-1, 1), 0, rng.uniform(-1, 1.5)))
        objs.append(o)
    return objs, [], {}


@scenario("wall_bounce", "solidity", "던지거나 미끄러진 물체가 방 가운데 벽에 부딪혀 되튕김 — 벽 높이 · 위치 · 반발 무작위")
def s_wall(rng, n_frames, main_ident):
    side = rng.choice([-1, 1])
    wx = float(side * rng.uniform(0.3, 1.5))
    wh = float(rng.uniform(1.0, 3.0))
    box("wall_mid", (wx, 0, wh / 2), (0.3, 1.6, wh), material("wallmid", rand_rgb(rng), 0.7), bounce=float(rng.uniform(0.3, 0.8)))
    throw = rng.random() < 0.5
    x0 = float(-side * 2.6)
    o = make_object(main_ident, (x0, 0, float(rng.uniform(0.8, 1.5)) if throw else 0.45), 0.5, rng,
                    friction=None if throw else float(rng.uniform(0.01, 0.06)))
    vx = float(rng.uniform(2.0, 4.0))
    tt = abs(wx - x0) / vx                          # 벽까지 걸리는 시간 — 던지면 벽 높이 안에서 맞도록 vz 를 맞춘다
    vz = float(np.clip((rng.uniform(0.3, 0.8) * wh - o.location.z + 0.5 * G * tt ** 2) / tt, 0, 6)) if throw else 0
    launch(o, (side * vx, 0, vz))
    return [o], [], {}


@scenario("collide", "solidity", "2~3 물체가 바닥 위 / 공중에서 서로 마주 와 충돌 — 크기 (질량) 따라 운동량 교환")
def s_collide(rng, n_frames, main_ident):
    objs = []
    air = rng.random() < 0.35
    z = float(rng.uniform(1.0, 2.0)) if air else 0.45
    xm = float(rng.uniform(-0.8, 0.8))               # 만나는 자리
    tt = float(rng.uniform(1.3, 2.2))                # 만나는 시각 (초) — 문맥 16 장 (1.33 초) 이 충돌 앞에 들어가도록
    if air:
        tt = min(tt, math.sqrt(8 * (APEX_MAX - z) / G))   # 꼭대기 z + G tt² / 8 ≤ APEX_MAX
    vz = float(0.5 * G * tt) if air else 0.0         # 공중이면 같은 높이로 돌아왔을 때 만나도록
    ids = [main_ident] + others(rng, main_ident, int(rng.integers(1, 3)))
    for i, ident in enumerate(ids):
        side = -1 if i % 2 == 0 else 1
        x0 = float(np.clip(xm + side * float(rng.uniform(1.6, 2.6)), -2.9, 2.9))
        if i == 2:                                     # 셋째는 가운데 근처에 서 있다가 맞는다
            x0 = xm + float(rng.uniform(-0.3, 0.3))
        o = make_object(ident, (x0, 0, z if i < 2 else 0.45), float(rng.uniform(0.4, 0.75)), rng,
                        friction=float(rng.uniform(0.01, 0.08)))
        v = (xm - x0) / tt if i < 2 else 0.0
        launch(o, (v, 0, vz if i < 2 else 0))
        objs.append(o)
    return objs, [], {}


@scenario("stack_topple", "solidity", "쌓인 블록 (cube · cylinder) — 물체가 날아와 치거나 위에 떨어져 무너짐")
def s_stack(rng, n_frames, main_ident):
    side = rng.choice([-1, 1])
    sx = float(side * rng.uniform(0.3, 1.4))
    n = int(rng.integers(2, 4))
    blocks = [i for i in range(len(SHAPES) * NC) if SHAPES[i // NC] in ("cube", "cylinder") and i != main_ident]
    objs, z = [], 0.0
    shooter = make_object(main_ident, (float(-side * 2.7), 0, float(rng.uniform(0.6, 1.8))), 0.5, rng)
    objs.append(shooter)
    for k in range(n):
        ident = int(rng.choice(blocks))
        sz = float(rng.uniform(0.38, 0.5))
        h = sz * SIZE_SCALE * (0.85 if SHAPES[ident // NC] == "cube" else 1.0)
        o = make_object(ident, (sx + float(rng.uniform(-0.05, 0.05)), 0, z + h / 2 + 0.002), sz, rng,
                        friction=float(rng.uniform(0.4, 0.8)), bounce=0.1)
        z += h
        objs.append(o)
    zt = float(rng.uniform(0.3, 0.9)) * z
    tt = float(rng.uniform(0.9, 1.6))
    tt = min(tt, math.sqrt(2 * (APEX_MAX - max(zt, shooter.location.z)) / G) * 1.9)
    vz = (zt - shooter.location.z + 0.5 * G * tt ** 2) / tt
    launch(shooter, ((sx - shooter.location.x) / tt, 0, vz))
    return objs, [], {}


@scenario("container", "solidity", "통 (U 자) 안으로 떨어지거나 던져진 물체 — 벽 안에서 튕기고 갇힌다")
def s_container(rng, n_frames, main_ident):
    cx = float(rng.uniform(-1.5, 1.5))
    w = float(rng.uniform(1.6, 2.4))
    h = float(rng.uniform(0.7, 1.3))
    m = material("bin", rand_rgb(rng), 0.7)
    b = float(rng.uniform(0.2, 0.6))
    box("bin_floor", (cx, 0, 0.06), (w, 1.6, 0.12), m, bounce=b)
    box("bin_l", (cx - w / 2, 0, h / 2), (0.15, 1.6, h), m, bounce=b)
    box("bin_r", (cx + w / 2, 0, h / 2), (0.15, 1.6, h), m, bounce=b)
    objs = []
    for i, ident in enumerate([main_ident] + others(rng, main_ident, int(rng.integers(0, 2)))):
        if rng.random() < 0.5:                                         # 위에서 떨어뜨림
            o = make_object(ident, (cx + float(rng.uniform(-0.4, 0.4) * w), 0, float(rng.uniform(2.5, 4.2))), 0.45, rng)
            launch(o, (rng.uniform(-1, 1), 0, rng.uniform(-1, 1)))
        else:                                                          # 옆에서 던져 넣음
            side = rng.choice([-1, 1])
            x0 = float(np.clip(cx - side * rng.uniform(2.0, 3.0), -2.9, 2.9))
            o = make_object(ident, (x0, 0, float(rng.uniform(0.6, 1.4))), 0.45, rng)
            tt = float(rng.uniform(1.0, 1.6))
            tt = min(tt, 2 * math.sqrt(2 * (APEX_MAX - h - 0.6) / G))
            launch(o, ((cx - x0) / tt, 0, (h + 0.6 - o.location.z + 0.5 * G * tt ** 2) / tt))
        objs.append(o)
    return objs, [], {}


# ----------------------------------------------------------------------------- 라벨
def sample_points(o):
    """표면 위 고정 표본 점 200 개 (면적 비례, 물체 좌표). 꼭짓점만 쓰면 cube 는 8 점이라 보임 비율이 장마다 튄다."""
    dg = bpy.context.evaluated_depsgraph_get()
    me = o.evaluated_get(dg).to_mesh()
    me.calc_loop_triangles()
    V = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
    tri = np.array([t.vertices[:] for t in me.loop_triangles])
    o.evaluated_get(dg).to_mesh_clear()
    a, b, c = V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    r = np.random.default_rng(0)
    k = r.choice(len(tri), size=200, p=area / area.sum())
    u, w = r.random(200), r.random(200)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    return a[k] + u[:, None] * (b[k] - a[k]) + w[:, None] * (c[k] - a[k])


def visibility(sc, cam, o, pts_local, res):
    """(화면 안 비율, 보이는 비율, amodal bbox px) — 표본 점마다 카메라에서 ray, 처음 맞는 것이 그 물체면 보임."""
    dg = bpy.context.evaluated_depsgraph_get()
    M = o.matrix_world
    origin = cam.matrix_world.translation
    n_in = n_vis = 0
    xs, ys = [], []
    for p in pts_local:
        w = M @ Vector(p)
        c = world_to_camera_view(sc, cam, w)
        xs.append(c.x * res); ys.append((1 - c.y) * res)
        if not (0 <= c.x <= 1 and 0 <= c.y <= 1 and c.z > 0):
            continue
        n_in += 1
        d = w - origin
        u, start, left = d.normalized(), origin.copy(), d.length + 1e-3
        for _ in range(4):                          # 안 보이는 유리 벽 (렌더 제외) 은 건너뛴다
            hit, loc, _, _, ob, _ = sc.ray_cast(dg, start, u, distance=left)
            if not (hit and ob is not None and ob.name.startswith("glass")):
                break
            left -= (loc - start).length + 1e-4
            start = loc + u * 1e-4
        if hit and ob is not None and ob.name == o.name:
            n_vis += 1
    n = len(pts_local)
    return n_in / n, n_vis / n, (min(xs), min(ys), max(xs), max(ys))


def simulate_and_label(sc, cam, objs, n_frames, res):
    """라벨 단위는 phys2d_sup 과 같다: pos px · vel px/장 · acc px/장² · angle rad · omega rad/장 · bbox px.
    3D 추가: pos3d m · vel3d m/장 (카메라 축 y 는 유리벽으로 ±0.8 m 안)."""
    T = n_frames
    N = len(objs)
    pos3d = np.zeros((T, N, 3)); pos = np.zeros((T, N, 2)); inframe = np.zeros((T, N)); vis = np.zeros((T, N))
    ang = np.zeros((T, N)); bbox = np.zeros((T, N, 4))
    pts = [sample_points(o) for o in objs]
    for f in range(1, PRE + T):
        sc.frame_set(f)
        t = f - PRE
        if t < 0:
            continue
        for i, o in enumerate(objs):
            w = o.matrix_world.translation
            pos3d[t, i] = w[:]
            ang[t, i] = -o.matrix_world.to_euler()[1]           # 세계 y (≈ 카메라 축) 둘레 회전 = 화면 안 회전
            c = world_to_camera_view(sc, cam, w)
            pos[t, i] = (c.x * res, (1 - c.y) * res)
            inframe[t, i], vis[t, i], bbox[t, i] = visibility(sc, cam, o, pts[i], res)
    grad = lambda a: np.gradient(a, axis=0) if T > 2 else np.zeros_like(a)
    vel = grad(pos)
    om = grad(np.unwrap(ang, axis=0))
    return dict(pos=pos.astype(np.float32), vel=vel.astype(np.float32), acc=grad(vel).astype(np.float32),
                angle=ang.astype(np.float32), omega=om.astype(np.float32), bbox=bbox.astype(np.float32),
                pos3d=pos3d.astype(np.float32), vel3d=grad(pos3d).astype(np.float32),
                inframe_frac=inframe.astype(np.float32), vis_frac=vis.astype(np.float32))


def events(lab):
    """impact: 앞뒤 장 속도 변화가 이웃 장 (±2) 속도 변화 평균과 DV_IMPACT 이상 다르다 (post = 바뀐 뒤 첫 장).
    leave: 3 장 이상 받쳐져 있던 물체가 받침 높이에서 내려가 떨어진다.
    occlude / reappear: 화면 안 부분 중 보이는 비율 기준 (2D 와 같은 문턱)."""
    p, dt = lab["pos3d"], 1.0 / FPS
    v = np.diff(p, axis=0) / dt                     # v[t] = t → t+1 사이 속도
    T, N = p.shape[:2]
    out = []
    for i in range(N):
        dvv = np.diff(v[:, i], axis=0)              # dvv[t] = v[t+1] − v[t], 장 t+1 근처 변화
        # 매끄러운 힘 (중력 · 진자 장력 · 굴림) 은 이웃 장의 속도 변화와 거의 같다 → 이웃 평균과의 차이만 충격으로 센다
        dv = np.zeros(len(dvv))                     # 양 끝 2 장은 이웃이 없어 재지 않는다
        dv[2:-2] = np.linalg.norm(dvv[2:-2] - 0.5 * (dvv[:-4] + dvv[4:]), axis=1)
        last = -10
        for t in np.where(dv >= DV_IMPACT)[0]:
            if t - last > 2:
                out.append(dict(kind="impact", obj=i, frame=int(t + 2), dv=float(dv[t])))
            last = t
        # 모서리에서 기울며 넘어가면 처음 몇 장은 −g 보다 덜 가속하므로 가속도 대신 높이로 잰다
        z, vz = p[:, i, 2], np.r_[v[:, i, 2], v[-1, i, 2]]
        held, z_hold = 0, None
        for t in range(T):
            if held >= 3 and z_hold - z[t] > 0.03:  # 받침 높이에서 내려갔다
                if (vz[t:t + 4] < -1.0).any():
                    out.append(dict(kind="leave", obj=i, frame=int(t), dv=0.0))
                held = 0
            elif abs(vz[t]) < 0.2 and z[t] > 0.6:   # 바닥 위 받침 (선반 · 경사 끝 · 다른 물체)
                held, z_hold = held + 1, z[t]
            elif abs(vz[t]) >= 0.2 and held < 3:
                held = 0
        ratio = lab["vis_frac"][:, i] / np.maximum(lab["inframe_frac"][:, i], 1e-6)
        state = "hid" if ratio[0] < HIDDEN else "vis"
        for t in range(1, T):
            if state == "vis" and ratio[t] < HIDDEN:
                out.append(dict(kind="occlude", obj=i, frame=t, dv=0.0)); state = "hid"
            elif state == "hid" and ratio[t] > SEEN:
                out.append(dict(kind="reappear", obj=i, frame=t, dv=0.0)); state = "vis"
    return sorted(out, key=lambda e: (e["frame"], e["obj"]))


def event_rows(ev, lab, unpred, rng, T, min_ctx=4, min_pred=8):
    """boundary 행 — phys2d_sup `event_rows` 와 같은 규칙 (2026-10-18 사용자 결정: 전환은 예측 구간에).
    b = post + j, j ∈ {−2, −1, 0, +1}. ctx_start = 이 행보다 앞선 충돌 (dv ≥ DV_ROW) · 받침 이탈의 post (가림 · 다시 보임은 문맥에 있어도 된다).
    end = 예측 못 하는 시작 (info unpred) 직전, 없으면 클립 끝."""
    gate = sorted({e["frame"] for e in ev if e["kind"] == "leave" or (e["kind"] == "impact" and e["dv"] >= DV_ROW)})
    imp = [e for e in ev if e["kind"] == "impact" and e["dv"] >= DV_ROW]
    if len(imp) > MAX_IMPACT_ROWS:
        imp = [imp[j] for j in sorted(rng.choice(len(imp), MAX_IMPACT_ROWS, replace=False))]
    cand = imp + [e for e in ev if e["kind"] in ("leave", "occlude", "reappear")]
    pos = lab["pos"]
    rows, used = [], set()
    for e in sorted(cand, key=lambda d: d["frame"]):
        b = e["frame"] + int(rng.integers(-2, 2))
        end = min([u for u in unpred if u >= b] + [T])
        ctx_start = max([p for p in gate if p < b and p < e["frame"]] + [0])
        if b < 1 or b - ctx_start < min_ctx or end - b < min_pred or b in used:
            continue
        q = min(end, b + 8) - 1
        if e["kind"] not in ("occlude", "reappear") and np.linalg.norm(pos[q] - pos[b], axis=-1).max() < 6:
            continue                                   # 예측 구간 앞 8 장에 움직임이 없다 (가림 · 다시 보임은 가림막이 움직여도 된다)
        used.add(b)
        rows.append(dict(boundary=int(b), end=int(end), ctx_start=int(ctx_start), event=e["kind"], event_frame=int(e["frame"]),
                         obj=int(e["obj"]), dv=round(float(e["dv"]), 3)))
    return rows


# ----------------------------------------------------------------------------- 렌더
def setup_render(sc, engine, res, out_png):
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.compression = 0
    sc.render.filepath = out_png
    if engine == "eevee":
        sc.render.engine = "BLENDER_EEVEE_NEXT"
        sc.eevee.taa_render_samples = 16
        sc.eevee.use_shadows = True
    elif engine == "cpu":
        # Cycles CPU, 워커당 스레드 1 — GPU 를 안 쓴다. ⚠️ EEVEE (EGL) 는 CUDA_VISIBLE_DEVICES 를 무시하고 전부 GPU 0 에 붙는다
        # (2026-10-19 시험: 워커 56 개가 GPU 0 에 몰려 학습 중인 GPU 메모리 96.9/97.9 GB). 대량 생성은 이쪽으로
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.render.threads_mode = "FIXED"
        sc.render.threads = CPU_THREADS
        sc.cycles.samples = CPU_SAMPLES
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
        sc.cycles.max_bounces = 3
        sc.cycles.diffuse_bounces = 2
        sc.cycles.glossy_bounces = 1
        sc.cycles.transmission_bounces = 0
        sc.cycles.transparent_max_bounces = 0
        sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
        sc.render.use_persistent_data = True
    else:
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "OPTIX"
        sc.cycles.device = "GPU"                       # CUDA_VISIBLE_DEVICES 를 따른다 (EEVEE 와 다르다)
        sc.cycles.samples = CPU_SAMPLES
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = "OPTIX"
        sc.cycles.max_bounces = 3
        sc.cycles.diffuse_bounces = 2
        sc.cycles.glossy_bounces = 1
        sc.cycles.transmission_bounces = 0
        sc.cycles.transparent_max_bounces = 0
        sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
        sc.render.use_persistent_data = True


# ----------------------------------------------------------------------------- 한 편
def build(name, seed, attempt, main_ident, n_frames, res, engine, frames_dir):
    global G
    rng = np.random.default_rng([seed, list(SCENARIOS).index(name), attempt])
    sc = reset()
    # 중력 세기도 클립마다 (2D 의 g 350~900 px/s² 와 같은 생각). 12 fps 에서 9.81 이면 낙하가 1 초 안에 끝나 C16 문맥이 안 나온다
    G = float(np.random.default_rng([seed, 5]).uniform(*G_RANGE))
    sc.gravity = (0, 0, -G)
    env = room(rng)
    env["g"] = G
    cat, desc, fn = SCENARIOS[name]
    objs, occ, info = fn(rng, n_frames, main_ident)
    if cat != "permanence" and rng.random() < P_EXTRA_SCREEN:   # 어느 시나리오에나 가림 — 영속 사건이 자주 나오게
        occ.append(screen_on_path(rng, [-2.0, -1.0, 0.0, 1.0, 2.0], name="extra"))
    sc.frame_end = PRE + n_frames - 1
    rw = sc.rigidbody_world
    rw.point_cache.frame_start = 1
    rw.point_cache.frame_end = sc.frame_end
    setup_render(sc, engine, res, os.path.join(frames_dir, "f"))  # 투영 (world_to_camera_view) 이 렌더 해상도 · 종횡비를 쓴다 — 라벨 전에
    return rng, sc, env, objs, occ, info


def check(lab, rows, T):
    """버리는 이유 (없으면 빈 목록). 주 물체 (0) 가 움직여야 하고, C16 · K16 이 들어가는 행이 하나는 있어야 한다."""
    why = []
    p = lab["pos"][:, 0]
    if np.linalg.norm(p.max(0) - p.min(0)) < 25:
        why.append("main barely moves")
    if not any(r["boundary"] - r["ctx_start"] >= 16 and r["end"] - r["boundary"] >= 16 for r in rows):
        why.append("no event row C16K16")
    if (lab["inframe_frac"][:, 0] < 0.5).mean() > 0.1:
        why.append("main out of frame")
    return why


def generate(name, k, seed, main_ident, out, n_frames=96, res=256, engine="cpu", max_attempts=MAX_ATTEMPTS, keep_frames=False):
    stem = os.path.join(out, name, f"{name}_{k:05d}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    frames_dir = stem + "_frames"
    why = []
    for attempt in range(max_attempts):
        rng, sc, env, objs, occ, info = build(name, seed, attempt, main_ident, n_frames, res, engine, frames_dir)
        lab = simulate_and_label(sc, sc.camera, objs, n_frames, res)
        ev = events(lab)
        unpred = sorted(info.get("unpred", []))
        rows = event_rows(ev, lab, unpred, np.random.default_rng([seed, attempt, 3]), n_frames)
        why = check(lab, rows, n_frames)
        if why:
            continue
        T, N = n_frames, len(objs)
        idents = np.array([o["ident"] for o in objs])
        ratio = lab["vis_frac"] / np.maximum(lab["inframe_frac"], 1e-6)
        impact = np.zeros((T, N), np.float32)
        for e in ev:
            if e["kind"] == "impact":
                impact[e["frame"], e["obj"]] = max(impact[e["frame"], e["obj"]], e["dv"])
        present = np.ones((T, N), bool)
        bb = lab["bbox"]
        lab.update(present=present, hidden=present & (lab["inframe_frac"] > 0.25) & (ratio < HIDDEN),
                   seen=np.maximum.accumulate(lab["vis_frac"] > SEEN, axis=0), impact=impact,
                   identity=idents.astype(np.int16), shape_id=(idents // NC).astype(np.int16), color_id=(idents % NC).astype(np.int16),
                   size=np.array([o.dimensions.length / math.sqrt(3) for o in objs], np.float32),
                   area=((bb[0, :, 2] - bb[0, :, 0]) * (bb[0, :, 3] - bb[0, :, 1])).astype(np.float32))
        np.savez_compressed(stem + "_states.npz", **lab)
        os.makedirs(frames_dir, exist_ok=True)
        sc.frame_start = PRE
        bpy.ops.render.render(animation=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-start_number", str(PRE),
                        "-i", os.path.join(frames_dir, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        stem + ".mp4"], check=True)
        if not keep_frames:
            shutil.rmtree(frames_dir)
        cat, desc, _ = SCENARIOS[name]
        meta = dict(scenario=name, category=cat, desc=desc, k=k, seed=seed, attempt=attempt, fps=FPS, n_frames=n_frames, res=res,
                    engine=engine, env=env, unpred=unpred,
                    objects=[dict(index=i, role="main" if i == 0 else "other", identity=int(o["ident"]),
                                  shape=SHAPES[o["ident"] // NC], color=COLORS[o["ident"] % NC][0], mass=o.rigid_body.mass,
                                  friction=o.rigid_body.friction, restitution=o.rigid_body.restitution)
                             for i, o in enumerate(objs)],
                    occluders=[o.name for o in occ], events=ev, rows=rows)
        with open(stem + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        return dict(ok=True, mp4=stem + ".mp4", attempt=attempt, n_rows=len(rows))
    return dict(ok=False, mp4=stem + ".mp4", why=why)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", help="한 편만 (예시)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", help="작업 목록 tsv (name k seed main_ident) — gen_phys3d.py 가 만든다")
    ap.add_argument("--results", help="--jobs 의 결과 jsonl (이어 쓴다)")
    ap.add_argument("--claims", help="--jobs 의 작업 가져가기 표시 폴더 (워커 공용)")
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--engine", default="cpu", choices=["cpu", "eevee", "cycles"])
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.scenario:
        r = generate(a.scenario, a.seed, a.seed, a.seed % (len(SHAPES) * NC), a.out, a.frames, a.res, a.engine, keep_frames=a.keep_frames)
        print("DONE", r)
        return
    # 작업 목록은 워커 전부가 같은 것을 읽고, claims 폴더에 O_EXCL 로 파일을 만든 쪽이 가져간다 (빠른 GPU 워커가 더 많이 가져간다)
    os.makedirs(a.claims, exist_ok=True)
    for line in open(a.jobs):
        name, k, seed, ident = line.split()
        job = f"{name}/{k}"
        try:
            os.close(os.open(os.path.join(a.claims, f"{name}_{k}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        t0 = time.time()
        r = generate(name, int(k), int(seed), int(ident), a.out, a.frames, a.res, a.engine)
        r.update(job=job, name=name, k=int(k), seed=int(seed), main_identity=int(ident), sec=round(time.time() - t0, 1), engine=a.engine)
        with open(a.results, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
