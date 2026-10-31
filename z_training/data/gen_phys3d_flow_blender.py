"""phys3d_flow (2026-10-27) — 오래 이어지는 운동이 기본인 3D 합성 학습 세트. 문서 z_training/docs/PHYS3D_FLOW_DATASET_2026-10-27.md
원칙: z_training/docs/DATA_DESIGN_PRINCIPLES_2026-10-26.md (정본, 원칙 A: 물리적인 흐름이 그 상황의 흔한 결과) · SYNTHETIC_TRAINSET_PRINCIPLES_2026-10-25.md

사용자 (2026-10-27): "데이터셋 새로 설계 … 지금 데이터셋 보니까 정지해 있는 경우도 있고" · "오래 이어지는 운동이 기본이 되도록. 15 % 괜찮네 정지 …
시나리오 최대한 많이 그리고 물체 종류도 최대한 많이 물체 texture 도 다양하게 배경도 다양하게" · "배경에도 살짝 노이즈 시간에 따라 계속 변하게" ·
"물체가 좀 시원시원하게 움직였으면".

phys3d_multi (gen_phys3d_multi_blender.py, v1) 을 import 해 갈고리로만 바꾼다 (v1 은 그대로 재현된다):
  * 운동을 길게 · 시원하게: 속도 배율 1.0~3.0 (v1 0.6~2.2), 바닥 · 물체 마찰 낮게, 벽 · 바닥 반발 높게, 들어오는 물체 더 자주, 시뮬 420 장.
  * 창 검사 (WINDOW_HOOK): 영상의 85 % 는 주 물체가 창의 ≥ 85 % 장에서 움직이고 장면 정지 ≤ 5 %, 움직이는 동안 화면 속력 중앙 ≥ MIN_PX px / 저장 장.
    15 % 는 "멈춤 허용" (사용자 "15 % 괜찮네 정지") — 장면 정지 ≤ 35 % 이고 멈춤은 문맥 뒤 (예측 안) 에서만.
  * 모양 18 → 30 (EXTRA_SHAPES), 물체 질감 3 → 10 종 (절차 7 · Poly Haven CC0 사진 질감 · 금속 · 광택), 배경: HDRI 400 · 벽 · 바닥에 사진 질감,
    시간에 따라 천천히 변하는 노이즈 (벽 · 바닥 무늬 흐름 · 빛 세기 · 색 흔들림 · 하늘 회전).
  * 시나리오 33 → 45 (오래 가는 것 위주): plinko · bowling · curling · bounce_ball · newton_cradle · double_pendulum · rolling_hills · tumble_slope ·
    pendulum_ball · ball_rain · sweeper · cart_shuttle.
  /workspace/tools/blender-4.2.3-linux-x64/blender -t 1 -b -noaudio -P z_training/data/gen_phys3d_flow_blender.py -- --scenario plinko --seed 0 --out /tmp/f
대량 생성: gen_phys3d.py --multi --flow --defer-render.
"""
import colorsys
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_phys3d_multi_blender as M            # noqa: E402  (B.generate = M.generate2 가 깔린다)
from gen_phys3d_multi_blender import B, obj, place_on, launch, material, box, block, slab, prism, loguni, PRE, ROOM_X, SIZE_SCALE   # noqa: E402

TEX_DIR = os.environ.get("PHYS3D_TEX_DIR", "/workspace/dataset/assets/tex_polyhaven_1k")
# 10 fps · 문맥 최대 16 블록 (32 장) · 예측 최대 8 블록 (16 장) — 사용자 2026-10-27 "fps 10 으로 뽑을 거고 context 최대 16 block, prediction 최대 8 block".
# 시뮬도 10 fps (부분 단계 12 = 120 Hz 그대로), 렌더 장 = 저장 장 (stride 1), 영상 48 장 (4.8 초) = 문맥 32 + 예측 16.
B.FPS, B.SUBSTEPS = 10, 12
M.ST, M.T_SAVE = 1, 48
M.W_RAW = M.T_SAVE * M.ST
M.MAX_BLIND = 30                              # 장 (3 초)
# ⚠️ 정정 2026-10-27 (사용자 "ledge 가 평면에서 좀 운동하다가 떨어지는 게 아니라 그냥 바로 떨어지고 좀 어색한 게 많은 거 같은데 … 물리적으로 좀 생각해봐"):
#   속도 1~3 배 · 바닥 마찰 0.05~0.5 · 물체 마찰 × 0.3~1 · 벽 반발 0.8~0.97 · 중력 2~16 이 겹쳐 얼음 위 고무공처럼 보였고, 영상마다 C* 에 앵커를 둬 C* 가 작으면
#   사건 0.2 초 전에 영상이 시작됐다 (train ledge_fall 300 편: 57 % 는 영상 시작 전에 이미 떨어짐, 나머지 중앙 0.4 초). → 아래 값으로 되돌리고 앵커는 늘 32 번째 장.
M.SPEED_RANGE = (1.0, 2.2)                    # 사용자 "시원시원하게" — 재질은 보통으로 두고 속도만 조금 높게 (시간을 맞추는 시나리오는 이 배율을 나눠 둔다)
M.G_RANGE = (4.0, 12.0)
M.C_SET = (32,)                               # 앵커 = 저장 장 32 (문맥 최대 32 장 = 16 블록). 로더가 C ∈ {2..32} 를 고르면 문맥은 늘 앵커 바로 앞에서 끝난다
M.N_SIM = 360                                 # 36 초
M.P_ARRIVE = (0.25, 0.3, 0.25, 0.2)
P_ALLOW_STOP = 0.15
MIN_PX_S = 12.0                               # 움직이는 동안 화면 속력 중앙 ≥ 12 px / 초 — "시원시원하게"
STILL_V = 0.05


# ----------------------------------------------------------------------------- 모양 (+12)
def _mesh(name, V, F, loc):
    return M.mesh_obj(name, V, F, loc)


def _ring(n, r, z):
    return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]


def _prism_mesh(name, pts2, depth, loc):
    n = len(pts2)
    V = [(x, -depth / 2, z) for x, z in pts2] + [(x, depth / 2, z) for x, z in pts2]
    F = [tuple(range(n))[::-1], tuple(range(n, 2 * n))] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return _mesh(name, V, F, loc)


def _frustum(r, size, loc):
    bpy.ops.mesh.primitive_cone_add(radius1=r, radius2=r * 0.5, depth=size * 0.8, vertices=32, location=loc)


def _barrel(r, size, loc):
    bpy.ops.mesh.primitive_cylinder_add(radius=r * 0.8, depth=size * 0.9, vertices=32, location=loc)
    for v in bpy.context.object.data.vertices:
        v.co.x *= 1.0 + 0.18 * (1 - (2 * v.co.z / (size * 0.9)) ** 2)
        v.co.y *= 1.0 + 0.18 * (1 - (2 * v.co.z / (size * 0.9)) ** 2)


def _egg(r, size, loc):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r * 0.8, segments=32, ring_count=16, location=loc)
    for v in bpy.context.object.data.vertices:
        v.co.z *= 1.35 if v.co.z > 0 else 1.0


def _diamond(r, size, loc):
    ring = _ring(8, r * 0.85, size * 0.12)
    _mesh("diamond", ring + [(0, 0, r * 0.55), (0, 0, -r * 0.95)],
          [(i, (i + 1) % 8, 8) for i in range(8)] + [((i + 1) % 8, i, 9) for i in range(8)], loc)


def _half_sphere(r, size, loc):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=32, ring_count=16, location=loc)
    o = bpy.context.object
    for v in o.data.vertices:
        v.co.z = max(v.co.z, 0.0)


def _star(r, size, loc):
    pts = []
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.45
        a = math.pi / 2 + i * math.pi / 5
        pts.append((rr * math.cos(a), rr * math.sin(a)))
    _prism_mesh("star", pts, size * 0.35, loc)


def _wedge(r, size, loc):
    _prism_mesh("wedge", [(-r, -r * 0.6), (r, -r * 0.6), (-r, r * 0.6)], size * 0.7, loc)


def _gem(r, size, loc):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=r, subdivisions=2, location=loc)
    for v in bpy.context.object.data.vertices:
        v.co.z *= 0.7


def _shape(fn_name, **kw):
    def f(r, size, loc):
        getattr(bpy.ops.mesh, fn_name)(location=loc, **{k: (v(r, size) if callable(v) else v) for k, v in kw.items()})
    return f


NEW_SHAPES = {
    "rod": _shape("primitive_cylinder_add", radius=lambda r, s: r * 0.25, depth=lambda r, s: s * 1.6, vertices=16, rotation=(0, math.pi / 2, 0)),
    "plank": _shape("primitive_cube_add", size=1, scale=lambda r, s: (s * 1.6, s * 0.5, s * 0.18)),
    "plate": _shape("primitive_cylinder_add", radius=lambda r, s: r * 1.2, depth=lambda r, s: s * 0.12, vertices=40),
    "can": _shape("primitive_cylinder_add", radius=lambda r, s: r * 0.6, depth=lambda r, s: s * 1.2, vertices=32),
    "frustum": _frustum, "barrel": _barrel, "egg": _egg, "diamond": _diamond, "half_sphere": _half_sphere,
    "star": _star, "wedge": _wedge, "gem": _gem,
}
M.EXTRA_SHAPES.update(NEW_SHAPES)
M.EXTRA_SMOOTH.update({"rod", "plate", "can", "barrel", "egg", "half_sphere", "frustum"})
M.SHAPES = M.SHAPES + tuple(NEW_SHAPES)
_obj_v1 = M.obj
LIE = {"cylinder", "can", "barrel", "capsule", "frustum", "rod", "plate", "disc"}   # 축 대칭 — 눕혀야 구른다


def obj_lie(rng, loc, size=None, shape=None, rot=None, **kw):
    """원기둥 · 캔 · 통 · 캡슐 같은 축 대칭 모양을 60 % 확률로 눕혀 (축 = y, 카메라 축) 굴러가게 한다.
    ⚠️ 2026-10-27 시험: 세워진 채 놓여 미끄러지다 마찰로 1~2 초 만에 멈췄다 (rolling_hills 1 / 4 통과)."""
    o = _obj_v1(rng, loc, size=size, shape=shape, rot=rot, **kw)
    if rot is None and M.SHAPES[o["shape_id"]] in LIE and rng.random() < 0.6:
        o.rotation_euler = (math.pi / 2, 0, float(rng.uniform(-0.2, 0.2))) if M.SHAPES[o["shape_id"]] != "disc" else (0, 0, 0)
        bpy.context.view_layer.update()
    return o


M.obj = obj_lie
obj = obj_lie
ROLLERS = ("sphere", "cylinder", "capsule", "disc", "ellipsoid", "torus", "icosa", "can", "barrel", "egg", "gem", "plate")


# ----------------------------------------------------------------------------- 질감 (10 종)
TEX_FILES = sorted(f for f in os.listdir(TEX_DIR) if f.endswith(".jpg")) if os.path.isdir(TEX_DIR) else []
OBJ_PATTERNS = ("plain", "checker", "stripes", "marble", "spots", "rings", "magic", "brick", "photo", "metal")
OBJ_P = (0.16, 0.1, 0.1, 0.1, 0.1, 0.08, 0.08, 0.06, 0.14, 0.08)


def _image_node(nt, rng, scale):
    f = TEX_FILES[int(rng.integers(len(TEX_FILES)))]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(os.path.join(TEX_DIR, f), check_existing=True)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
    return t, f


def obj_mat(rng, name, rgb, rgb2):
    """물체 질감 10 종. 라벨 rgb 는 바탕색 (photo 는 바탕색으로 물들인 사진)."""
    k = int(rng.choice(len(OBJ_PATTERNS), p=OBJ_P)) if TEX_FILES else int(rng.choice(8, p=np.array(OBJ_P[:8]) / sum(OBJ_P[:8])))
    kind = OBJ_PATTERNS[k]
    if kind in ("plain", "checker", "stripes"):
        return M.obj_material(name, rgb, rgb2, k, float(rng.uniform(0.05, 0.85)), float(rng.uniform(2.0, 7.0))), k
    m = material(name, rgb, float(rng.uniform(0.05, 0.85)))
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    l1 = [B.srgb2lin(c) for c in rgb] + [1.0]
    l2 = [B.srgb2lin(c) for c in rgb2] + [1.0]
    if kind == "metal":
        bsdf.inputs["Metallic"].default_value = float(rng.uniform(0.7, 1.0))
        bsdf.inputs["Roughness"].default_value = float(rng.uniform(0.1, 0.5))
        return m, k
    if kind == "photo":
        t, _f = _image_node(nt, rng, float(rng.uniform(0.8, 3.0)))
        mx = nt.nodes.new("ShaderNodeMixRGB")
        mx.blend_type = "MULTIPLY"
        mx.inputs["Fac"].default_value = float(rng.uniform(0.5, 1.0))
        mx.inputs["Color2"].default_value = [min(1.0, x * 1.6) for x in l1[:3]] + [1.0]
        nt.links.new(t.outputs["Color"], mx.inputs["Color1"])
        nt.links.new(mx.outputs["Color"], bsdf.inputs["Base Color"])
        return m, k
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    sc_ = float(rng.uniform(1.5, 6.0))
    mp.inputs["Scale"].default_value = (sc_, sc_, sc_)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    if kind == "marble":
        t = nt.nodes.new("ShaderNodeTexNoise")
        t.inputs["Detail"].default_value = float(rng.uniform(4, 15))
        t.inputs["Distortion"].default_value = float(rng.uniform(0.5, 4))
        out = t.outputs["Fac"]
    elif kind == "spots":
        t = nt.nodes.new("ShaderNodeTexVoronoi")
        out = t.outputs["Distance"]
    elif kind == "rings":
        t = nt.nodes.new("ShaderNodeTexWave")
        t.wave_type = "RINGS"
        out = t.outputs["Fac"]
    elif kind == "magic":
        t = nt.nodes.new("ShaderNodeTexMagic")
        out = t.outputs["Fac"]
    else:
        t = nt.nodes.new("ShaderNodeTexBrick")
        t.inputs["Color1"].default_value, t.inputs["Color2"].default_value = l1, l2
        t.inputs["Mortar"].default_value = [0.8 * x for x in l2[:3]] + [1.0]
        nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
        return m, k
    nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
    if kind in ("spots", "marble"):
        g = nt.nodes.new("ShaderNodeMath")
        g.operation = "GREATER_THAN"
        g.inputs[1].default_value = float(rng.uniform(0.35, 0.6))
        nt.links.new(out, g.inputs[0])
        out = g.outputs[0]
    mx = nt.nodes.new("ShaderNodeMixRGB")
    mx.inputs["Color1"].default_value, mx.inputs["Color2"].default_value = l1, l2
    nt.links.new(out, mx.inputs["Fac"])
    nt.links.new(mx.outputs["Color"], bsdf.inputs["Base Color"])
    return m, k


M.OBJ_MAT_HOOK = obj_mat


# ----------------------------------------------------------------------------- 배경 (+ 사진 질감 · 시간에 따라 변하는 노이즈)
_bg_v1 = M.background


def _photo_material(name, rng, scale):
    m = material(name, (200, 200, 200), float(rng.uniform(0.5, 0.95)))
    nt = m.node_tree
    t, f = _image_node(nt, rng, scale)
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Hue"].default_value = float(rng.uniform(0.45, 0.55))
    hsv.inputs["Saturation"].default_value = float(rng.uniform(0.6, 1.3))
    hsv.inputs["Value"].default_value = float(rng.uniform(0.7, 1.2))
    nt.links.new(t.outputs["Color"], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return m, f


def _animate_noise(m, rng, amp):
    """재질의 바탕색 위에 4D 노이즈를 얹고 W 를 시간에 따라 움직인다 — 배경 무늬가 천천히 흐른다 (물체 라벨에는 영향 없음)."""
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    link = bsdf.inputs["Base Color"].links
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.noise_dimensions = "4D"
    n.inputs["Scale"].default_value = float(rng.uniform(0.5, 4.0))
    n.inputs["Detail"].default_value = float(rng.uniform(1, 6))
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
    mx = nt.nodes.new("ShaderNodeMixRGB")
    mx.blend_type = "OVERLAY"
    mx.inputs["Fac"].default_value = amp
    if link:
        nt.links.new(link[0].from_socket, mx.inputs["Color1"])
    else:
        mx.inputs["Color1"].default_value = bsdf.inputs["Base Color"].default_value
    nt.links.new(n.outputs["Color"], mx.inputs["Color2"])
    nt.links.new(mx.outputs["Color"], bsdf.inputs["Base Color"])
    w = n.inputs["W"]
    w0, rate = float(rng.uniform(0, 100)), float(rng.uniform(0.002, 0.012))   # 장당
    for f in (1, PRE + M.N_SIM):
        w.default_value = w0 + rate * f
        w.keyframe_insert("default_value", frame=f)
    for fc in nt.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def background(rng, sc, env):
    rec = _bg_v1(rng, sc, env)
    ob = bpy.data.objects
    if TEX_FILES and rng.random() < 0.4:          # 바닥 사진 질감
        fm, f = _photo_material("floor_photo", rng, float(rng.uniform(0.15, 0.8)))
        for nm in ("floor", "floor_r", "pit_bottom", "ground_far"):
            if nm in ob:
                ob[nm].data.materials.clear()
                ob[nm].data.materials.append(fm)
        rec["floor_kind"] = "photo"
        rec["floor_tex"] = f[:-4]
    if TEX_FILES and rng.random() < 0.4:          # 벽 사진 질감
        wm, f = _photo_material("wall_photo", rng, float(rng.uniform(0.15, 0.8)))
        for nm in ("back", "left", "right"):
            ob[nm].data.materials.clear()
            ob[nm].data.materials.append(wm)
        rec["wall_kind"] = "photo"
        rec["wall_tex"] = f[:-4]
    amp = float(rng.uniform(0.03, 0.12))         # 시간에 따라 흐르는 노이즈 (사용자 "배경에도 살짝 노이즈 시간에 따라 계속 변하게")
    done = set()
    for nm in ("floor", "back", "left", "right"):
        m = ob[nm].data.materials[0]
        if m.name not in done:
            _animate_noise(m, rng, amp)
            done.add(m.name)
    for l in [o for o in ob if o.type == "LIGHT"]:   # 빛 세기 · 색 흔들림 (느린 무작위 걸음)
        e0 = l.data.energy
        c0 = colorsys.rgb_to_hsv(*l.data.color)
        fr = list(range(1, PRE + M.N_SIM + 1, 24))
        walk = np.cumsum(rng.normal(0, 0.03, len(fr)))
        hue = np.cumsum(rng.normal(0, 0.004, len(fr)))
        for f, w_, h_ in zip(fr, walk, hue):
            l.data.energy = float(e0 * np.clip(1 + w_, 0.7, 1.3))
            l.data.color = colorsys.hsv_to_rgb((c0[0] + h_) % 1, c0[1], c0[2])
            l.data.keyframe_insert("energy", frame=f)
            l.data.keyframe_insert("color", frame=f)
    if rec.get("bg_mode") == "open":              # 하늘이 천천히 돈다
        nt = sc.world.node_tree
        mpg = [n for n in nt.nodes if n.type == "MAPPING"]
        if mpg:
            r0 = mpg[0].inputs["Rotation"].default_value[2]
            rate = float(rng.uniform(-1, 1)) * math.radians(0.15)
            for f in (1, PRE + M.N_SIM):
                mpg[0].inputs["Rotation"].default_value[2] = r0 + rate * f
                mpg[0].inputs["Rotation"].keyframe_insert("default_value", index=2, frame=f)
    rec["bg_noise_amp"] = round(amp, 3)
    env.update(rec)
    return rec


M.background = background


# ----------------------------------------------------------------------------- 운동을 길게 · 시원하게
def scene_hook(sc, objs, info, rng, env):
    """재질다운 물리 — 바닥 · 벽 · 물체 마찰 · 반발은 방 (B.room) 과 시나리오가 정한 보통 값 그대로. 얼음 바닥은 그런 시나리오 (info["ice"]) 만.
    오래 가는 운동은 미끄럼이 아니라 구조로 (구르는 물체 · 경사 · 둔덕 · 진자 · 반원통 · 계속 들어오는 물체)."""
    ob = bpy.data.objects
    if info.get("ice"):
        for nm in ("floor", "floor_r"):
            if nm in ob and ob[nm].rigid_body:
                ob[nm].rigid_body.friction = float(rng.uniform(0.0, 0.03))
    info["allow_stop"] = bool(rng.random() < P_ALLOW_STOP)
    env.update(floor_friction=round(ob["floor"].rigid_body.friction, 3), allow_stop=info["allow_stop"])


def window_hook(lw, objs, info, Cs, rng):
    keep = [i for i, o in enumerate(objs) if o.get("role") != "front"]
    st_, fps_s = M.ST, B.FPS / M.ST
    p3 = lw["pos3d"][::st_][:, keep]
    v = np.linalg.norm(np.diff(p3, axis=0), axis=-1) * fps_s      # m/s, 저장 장
    still = (v < STILL_V).all(1)
    main_mv = v[:, 0] >= STILL_V
    px = np.linalg.norm(np.diff(lw["pos"][::st_][:, 0], axis=0), axis=-1) * fps_s   # px / 초
    why = []
    if info.get("active_any"):                    # 계속 들어오는 · 쌓이는 장면 — 주 물체 대신 장면 안 아무 물체나 움직이면 된다
        main_mv = ~still
    if info.get("allow_stop") or info.get("stop_scene"):
        if still.mean() > 0.35:
            why.append(f"stop clip still {still.mean():.2f}")
        if still[:max(Cs - 1, 0)].any():
            why.append("stop inside context")
    else:
        if main_mv.mean() < 0.85:
            why.append(f"main moving {main_mv.mean():.2f}")
        if still.mean() > 0.05:
            why.append(f"scene still {still.mean():.2f}")
    if main_mv.any() and np.median(px[main_mv]) < MIN_PX_S:
        why.append(f"slow {np.median(px[main_mv]):.1f}px/s")
    return why


# ----------------------------------------------------------------------------- 깊이 (사용자 2026-10-27 "물체의 움직임 궤적 depth 도 다양하게")
Y_MIN, Y_MAX = -1.0, 2.3                      # 물체 길 (y). 가림막 −1.3 · 앞 물체 길 −2.3 보다 늘 뒤, 뒷벽 2.5 앞
P_VY = 0.7                                    # 초속도에 앞뒤 (y) 성분을 줄 확률
_launch_flow = M.launch


def launch_depth(o, v, at=PRE - 2):
    """초속도에 앞뒤 성분 — 수평 속력의 ±35 % (P_VY 확률). 진자처럼 묶인 물체는 launch 를 안 부른다."""
    rng = STATE_D["rng"]
    v = list(v)
    if rng is not None and rng.random() < P_VY:
        vh = math.hypot(v[0], v[2]) if abs(v[1]) < 1e-6 else abs(v[1])
        v[1] += float(rng.uniform(-0.35, 0.35)) * max(vh, 0.5)
    return _launch_flow(o, tuple(v), at)


STATE_D = {"rng": None}
M.launch = launch_depth
launch = launch_depth
_gen2 = M.generate2


def generate_depth(name, k, seed, *a, **kw):
    STATE_D["rng"] = np.random.default_rng([seed, 17])
    return _gen2(name, k, seed, *a, **kw)


B.generate = generate_depth


def _constrained():
    out = set()
    for o in bpy.data.objects:
        c = o.rigid_body_constraint
        if c is not None:
            out.update(x.name for x in (c.object1, c.object2) if x is not None)
    return out


def depth_hook(sc, objs, info, rng, env):
    """유리벽을 넓히고 (y −1.0 ~ 2.3), 구조물 (passive, 깊이 ≤ 1.7 m, y 가운데 ≈ 0) 을 그 깊이로 늘리고, 묶이지 않은 물체의 출발 깊이를 바꾼다."""
    ob = bpy.data.objects
    ob["glass_front"].location.y = Y_MIN - 0.1
    ob["glass_back"].location.y = Y_MAX + 0.1
    skip = {"floor", "floor_r", "pit_bottom", "back", "left", "right", "glass_front", "glass_back"}
    for o in list(ob):
        if o.name in skip or o.type != "MESH" or o.rigid_body is None or o.rigid_body.type != "PASSIVE":
            continue
        ys = [v.co.y for v in o.data.vertices]
        lo, hi = min(ys), max(ys)
        wy = o.matrix_world.translation.y
        if not (1.2 <= hi - lo <= 1.7 and abs(wy + (lo + hi) / 2) < 0.3) or o.rotation_euler.x != 0:
            continue                                   # 다리 · 못 · 보 같은 얇은 것 · 돌린 원기둥은 그대로
        for v in o.data.vertices:                      # 로컬 y [lo, hi] → 세계 y [Y_MIN, Y_MAX]
            v.co.y = (Y_MIN - wy) + (v.co.y - lo) / (hi - lo) * (Y_MAX - Y_MIN)
        o.data.update()
    cons = _constrained()
    same = info.get("require_impact") == "objects"     # 서로 부딪혀야 하는 장면은 같은 깊이로 옮긴다 (따로 옮기면 비껴간다)
    dy0 = float(rng.uniform(-0.6, 1.4))
    for o in objs:
        if o.rigid_body is None or o.rigid_body.type != "ACTIVE" or o.name in cons:
            continue
        dy = dy0 if same else float(rng.uniform(-0.6, 1.4))
        fc = o.animation_data.action.fcurves if o.animation_data and o.animation_data.action else []
        for c in fc:                                   # launch 의 키프레임 (location y) 도 같이 옮긴다
            if c.data_path == "location" and c.array_index == 1:
                for kp in c.keyframe_points:
                    kp.co.y += dy
                    kp.handle_left.y += dy
                    kp.handle_right.y += dy
        o.location.y += dy
    env["depth_range"] = [Y_MIN, Y_MAX]


def scene_and_depth(sc, objs, info, rng, env):
    scene_hook(sc, objs, info, rng, env)
    depth_hook(sc, objs, info, rng, env)


M.SCENE_HOOK = scene_and_depth


def anchor_policy(rk, Cs):
    """앵커 늘 32 장 — 사건 (보임) · 사건 가림 · 그냥 가림 1/3 씩 (못 하면 그냥 가림, meta anchor_asked 에 원래 뜻)."""
    return str(rk.choice(M.ANCHORS))


M.ANCHOR_POLICY = anchor_policy


def _ledge(rng, h_rng, with_shelf):
    """탁자 위를 3.4~6 초 굴러 · 미끄러져 간 뒤 끝에서 떨어진다 (속도를 거리 ÷ 시간으로 맞춘다 — 앵커 32 장 앞이 탁자 위 운동이 되게)."""
    d = float(rng.choice([-1, 1]))
    h, w = float(rng.uniform(*h_rng)), float(rng.uniform(3.2, 4.6))
    cx = -d * (ROOM_X - w / 2)
    block("table", cx, w, h, rng, bounce=0.3)
    edge = cx + d * w / 2
    roll = rng.random() < 0.6
    o = obj(rng, (cx - d * (w / 2 - 0.5), 0, h + 1), size=-0.55, shape=str(rng.choice(ROLLERS)) if roll else None,
            friction=float(rng.uniform(0.05, 0.4)) if roll else float(rng.uniform(0.0, 0.02)), bounce=float(rng.uniform(0.2, 0.5)))
    place_on(o, h)
    dist = abs(edge - o.location.x)
    t_edge = float(rng.uniform(3.4, 6.0))
    v = dist / t_edge / M.STATE["speed"]          # launch 가 속도 배율을 곱한다 — 시간을 맞추려고 나눠 둔다
    launch(o, (d * v, 0, 0))
    objs = [o]
    info = {"t_edge": round(t_edge, 2)}
    if with_shelf:
        sw = float(rng.uniform(1.2, 2.0))
        M.shelf("shelf", edge + d * (sw / 2 + float(rng.uniform(0.0, 0.4))), float(rng.uniform(0.8, max(0.9, h - 1.0))), sw, rng,
                tilt=float(rng.uniform(-0.15, 0.15)))
    return objs, info


M.SCENARIOS["ledge_fall"] = ("gravity", "탁자 위를 3.4~6 초 굴러 · 미끄러져 가다 끝에서 떨어짐 (속도를 시간에 맞춤) — 탁자 높이 · 길이 무작위",
                             lambda rng, n: _ledge(rng, (0.9, 2.2), False))
M.SCENARIOS["ledge_to_shelf"] = ("solidity", "탁자 위를 3.4~6 초 가다 끝에서 떨어져 아래 선반에 걸림 · 다시 떨어짐",
                                 lambda rng, n: _ledge(rng, (2.4, 3.2), True))
M.WINDOW_HOOK = window_hook
scenario = M.scenario


# ----------------------------------------------------------------------------- 새 시나리오 12 (오래 가는 운동)
def _pegs(rng, rows, x0, x1, z0, dz, r, m):
    """못 판: 바닥에 선 판 (y 1.0) + 판에서 앞으로 나온 짧은 못. ⚠️ 정정 2026-10-27 — 못을 뒷벽까지 길게 뽑으니 원근으로 방사형 막대처럼 보였다."""
    box("peg_board", (0, 1.05, (z0 + 0.4) / 2), (2 * ROOM_X, 0.1, z0 + 0.4), m)
    k = 0
    for j in range(rows):
        xs = np.arange(x0 + (j % 2) * 0.45, x1, 0.9)
        for x in xs:
            bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=1.9, vertices=16,
                                                location=(float(x), 0.05, z0 - j * dz), rotation=(math.pi / 2, 0, 0))
            o = bpy.context.object
            o.name = f"peg{k}"
            o.data.materials.append(m)
            bpy.ops.rigidbody.object_add(type="PASSIVE")
            o.rigid_body.collision_shape = "CYLINDER"
            o.rigid_body.restitution = float(rng.uniform(0.4, 0.8))
            o.rigid_body.friction = 0.2
            k += 1


@scenario("plinko", "collision", "뒷벽에 박힌 못 (원기둥) 판 사이로 물체 1~3 개가 튕기며 떨어짐 — 바닥에서 굴러 계속 감")
def s_plinko(rng, n):
    m = material("pegs", B.rand_rgb(rng, 60, 200), 0.4)
    rows = int(rng.integers(3, 6))
    _pegs(rng, rows, -2.6, 2.7, float(rng.uniform(3.4, 4.0)), float(rng.uniform(0.55, 0.75)), 0.07, m)
    objs = []
    for i in range(int(rng.integers(1, 4))):
        o = obj(rng, (float(rng.uniform(-2.0, 2.0)), 0, 6.0 + i * 1.5), size=-0.4, shape=str(rng.choice(ROLLERS)),
                bounce=float(rng.uniform(0.4, 0.8)))
        launch(o, (rng.uniform(-0.6, 0.6), 0, 0))
        objs.append(o)
    return objs, {"enter_from_top": True}


@scenario("bowling", "collision", "긴 바닥을 굴러간 공이 핀 4~10 개 (세모 · 줄) 를 쳐 넘어뜨림 — 핀은 굴러 흩어짐")
def s_bowling(rng, n):
    d = float(rng.choice([-1, 1]))
    ball = obj(rng, (-d * 2.9, 0, 1), size=float(rng.uniform(0.35, 0.55)), shape="sphere", friction=0.05, density=float(rng.uniform(1.5, 3.0)))
    place_on(ball, 0)
    launch(ball, (d * rng.uniform(3.0, 6.0), 0, 0))
    pins, x = [ball], d * float(rng.uniform(0.6, 1.6))
    rows = int(rng.integers(2, 5))
    for j in range(rows):
        for i in range(j + 1):
            y = (i - j / 2) * 0.32
            if abs(y) > 0.7:
                continue
            p = obj(rng, (x + d * j * 0.36, y, 1), size=float(rng.uniform(0.22, 0.3)), shape=str(rng.choice(("can", "cylinder", "rod", "barrel"))),
                    friction=0.3, bounce=0.4, density=0.6)
            if M.SHAPES[p["shape_id"]] == "rod":
                p.rotation_euler = (0, 0, 0)
            p.rotation_euler = (0, 0, 0)
            bpy.context.view_layer.update()
            place_on(p, 0)
            pins.append(p)
    return pins, {"require_impact": "objects"}


@scenario("curling", "collision", "얼음 같은 바닥 (마찰 거의 0) 을 미끄러진 돌 (원판 · 둥근 상자) 이 다른 돌들을 침 — 오래 미끄러짐")
def s_curling(rng, n):
    d = float(rng.choice([-1, 1]))
    shp = str(rng.choice(("disc", "plate", "rounded_cube", "can", "hex_prism")))
    st = obj(rng, (-d * 2.8, 0, 1), size=-0.5, shape=shp, friction=0.0, bounce=float(rng.uniform(0.6, 0.9)))
    place_on(st, 0)
    launch(st, (d * rng.uniform(2.0, 4.5), 0, 0))
    objs = [st]
    for i in range(int(rng.integers(2, 5))):
        o = obj(rng, (float(d * rng.uniform(-0.5, 2.4)), float(rng.uniform(-0.5, 0.5)), 1), size=-0.5, shape=shp, friction=0.0,
                bounce=float(rng.uniform(0.6, 0.9)))
        place_on(o, 0)
        objs.append(o)
    return objs, {"require_impact": "objects", "ice": True}


@scenario("bounce_ball", "gravity", "반발 큰 공 1~3 개가 바닥 · 벽에서 계속 튕김 — 높이가 천천히 준다")
def s_bounce(rng, n):
    objs = []
    for i in range(int(rng.integers(1, 4))):
        o = obj(rng, (float(rng.uniform(-2.3, 2.3)), 0, float(rng.uniform(1.0, 4.0))), size=-0.55,
                shape=str(rng.choice(("sphere", "icosa", "gem", "egg", "rounded_cube"))), bounce=float(rng.uniform(0.85, 0.97)), friction=0.1)
        o.rigid_body.linear_damping = o.rigid_body.angular_damping = 0.0
        launch(o, (rng.uniform(-4, 4), 0, rng.uniform(-1, 4)))
        objs.append(o)
    return objs, {}


@scenario("newton_cradle", "collision", "진자 3~5 개가 한 줄로 닿아 있음 — 끝 하나 (또는 둘) 를 들어 놓아 반대 끝이 튕겨 나감, 오래 주고받음")
def s_cradle(rng, n):
    k = int(rng.integers(3, 6))
    L = float(rng.uniform(1.4, 2.4))
    pz = float(rng.uniform(max(3.4, L + 0.6), 4.6))
    size = float(min(0.42, loguni(rng, *M.SIZE_RANGE)))
    gap = size * SIZE_SCALE * 1.005
    x0 = -gap * (k - 1) / 2 + float(rng.uniform(-0.5, 0.5))
    th = math.radians(rng.uniform(25, 60))
    lift = int(rng.integers(1, 3))
    objs = []
    for i in range(k):
        a = -th if i < lift else 0.0
        objs.append(M._pendulum(rng, x0 + i * gap, pz, L, a, name=f"cr{i}", size=size, beam=(i == 0)))
    for o in objs:
        o.rigid_body.restitution = float(rng.uniform(0.9, 0.98))
    return objs, {"require_impact": "objects"}


@scenario("double_pendulum", "gravity", "이중 진자 (막대 둘 · 추 둘) — 혼돈스럽게 오래 흔들림")
def s_double(rng, n):
    L1, L2 = float(rng.uniform(0.9, 1.5)), float(rng.uniform(0.7, 1.3))
    pz = float(rng.uniform(max(3.6, L1 + L2 + 0.5), 4.7))
    px = float(rng.uniform(-0.8, 0.8))
    th = math.radians(rng.uniform(60, 150)) * float(rng.choice([-1, 1]))
    b1 = M._pendulum(rng, px, pz, L1, th, name="dp1", size=-0.35)
    c1 = b1.location.copy()
    dark = material("dp2_rod", (40, 40, 44), 0.5)
    th2 = th + math.radians(rng.uniform(-60, 60))
    c2 = Vector((c1.x + L2 * math.sin(th2), 0, c1.z - L2 * math.cos(th2)))
    bpy.ops.mesh.primitive_cylinder_add(radius=0.022, depth=L2, location=(c1 + c2) / 2, rotation=(0, -th2, 0))
    rod2 = bpy.context.object
    rod2.data.materials.append(dark)
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    rod2.rigid_body.mass = 0.05
    rod2.rigid_body.collision_collections[0] = False
    rod2.rigid_body.collision_collections[1] = True
    b2 = obj(rng, tuple(c2), size=-0.4, name="dp2")
    b2.rigid_body.mass = float(rng.uniform(0.8, 2.5))
    for o in (b1, b2, rod2):
        o.rigid_body.linear_damping = o.rigid_body.angular_damping = 0.0
    for a, b, kind, loc in ((b1, rod2, "POINT", c1), (rod2, b2, "FIXED", c2)):
        bpy.ops.object.empty_add(location=loc)
        e = bpy.context.object
        bpy.ops.rigidbody.constraint_add(type=kind)
        e.rigid_body_constraint.object1, e.rigid_body_constraint.object2 = a, b
    return [b2, b1], {}


@scenario("rolling_hills", "gravity", "바닥의 둔덕 3~5 개 (높이 · 폭 무작위) 위를 굴러 넘고 골짜기에서 오가며 감")
def s_hills(rng, n):
    xs = np.sort(rng.uniform(-2.6, 2.6, int(rng.integers(3, 6))))
    m = material("hills", B.rand_rgb(rng, 60, 180), 0.6)
    last = -9
    for j, x in enumerate(xs):
        if x - last < 1.0:
            continue
        L, h = float(rng.uniform(1.0, 1.8)), float(rng.uniform(0.08, 0.35))
        arc = [(x - L / 2 + L * i / 12, h * math.sin(math.pi * i / 12) - 0.02 * (i in (0, 12))) for i in range(13)]   # 매끄러운 둔덕 (각진 쐐기는 넘을 때 에너지를 크게 잃었다)
        prism(f"hill{j}", arc, rng, mat=m, friction=0.3)
        last = x
    d = float(rng.choice([-1, 1]))
    objs = []
    for i in range(int(rng.integers(1, 3))):
        o = obj(rng, (-d * (2.9 - 0.8 * i), 0, 1), size=-0.5, shape=str(rng.choice(ROLLERS)), friction=float(rng.uniform(0.05, 0.4)))
        place_on(o, 0)
        launch(o, (d * rng.uniform(2.0, 5.0), 0, 0))
        objs.append(o)
    return objs, {}


@scenario("tumble_slope", "gravity", "긴 완만한 경사 (끝이 바닥에 닿음) 를 다면체가 구르듯 넘어지며 내려와 바닥에서 미끄러짐")
def s_tumble(rng, n):
    d = float(rng.choice([-1, 1]))
    a = math.radians(rng.uniform(12, 26))
    top, end, _ = M.ramp_from_top("slope", rng, d, float(rng.uniform(1.6, 2.8)), 4.0, a, p_ground=1.0)
    o = obj(rng, (0, 0, 4), size=-0.6, shape=str(rng.choice(("cube", "rounded_cube", "pent_prism", "hex_prism", "octa", "diamond", "gem", "star", "wedge", "tri_prism"))),
            friction=float(rng.uniform(0.3, 0.8)), bounce=float(rng.uniform(0.2, 0.5)))
    M.on_slope(o, top, d, a, 0.5)
    launch(o, (d * rng.uniform(0.5, 2.0) * math.cos(a), 0, -rng.uniform(0.5, 2.0) * math.sin(a)))
    return [o], {}


@scenario("pendulum_ball", "collision", "진자가 바닥의 공을 쳐서 공이 멀리 굴러감 · 벽에서 튕겨 돌아옴")
def s_pend_ball(rng, n):
    L = float(rng.uniform(1.8, 3.0))
    pz = float(rng.uniform(max(3.4, L + 0.4), 4.6))
    px = float(rng.uniform(-0.6, 0.6))
    th = math.radians(rng.uniform(40, 80)) * float(rng.choice([-1, 1]))
    bob = M._pendulum(rng, px, pz, L, th, size=-0.5)
    s = -math.copysign(1, th)
    ball = obj(rng, (px + s * 0.25, 0, 1), size=-0.45, shape=str(rng.choice(ROLLERS)), friction=0.05)
    place_on(ball, 0)
    zl = pz - L
    bpy.context.view_layer.update()
    if zl - bob.dimensions.z / 2 > ball.dimensions.z * 0.8:     # 추가 공 높이까지 안 내려오면 받침대에 올린다
        hz = zl - ball.dimensions.z * 0.4
        block("stand", px + s * 0.25, 0.4, hz, rng)
        place_on(ball, hz)
    return [ball, bob], {"require_impact": "objects"}


@scenario("ball_rain", "gravity", "물체 4~8 개가 시간차로 화면 위에서 떨어져 경사 · 바닥에서 튕기고 굴러감 (계속 들어옴)")
def s_rain(rng, n):
    d = float(rng.choice([-1, 1]))
    if rng.random() < 0.7:
        a = math.radians(rng.uniform(12, 25))
        M.ramp_from_top("rain_ramp", rng, d, float(rng.uniform(1.2, 2.2)), 3.2, a, p_ground=0.6)
    objs = []
    for i in range(int(rng.integers(4, 9))):
        o = obj(rng, (float(rng.uniform(-2.5, 2.5)), 0, 6.5), size=-0.45, rot=tuple(rng.uniform(0, 3.14, 3)), name="rain")
        delay = 0 if i == 0 else int(rng.integers(6, M.N_SIM - 80))
        B.launch(o, (float(rng.uniform(-1.5, 1.5)), 0, 0), at=PRE - 2 + delay)
        objs.append(o)
    return objs, {"enter_from_top": True}


@scenario("sweeper", "collision", "기둥에 달린 회전 막대 (풍차) 가 일정하게 돌며 바닥 · 공중의 물체를 쳐 날림")
def s_sweeper(rng, n):
    px = float(rng.uniform(-1.0, 1.0))
    hz = float(rng.uniform(0.9, 1.6))
    L = float(rng.uniform(1.4, 2.2))
    m = material("sweeper", B.rand_rgb(rng, 40, 140), 0.5)
    box("sweep_post", (px, 0.75, hz / 2), (0.15, 0.15, hz), m)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(px, 0.2, hz), scale=(L, 0.25, 0.12))
    arm = bpy.context.object
    arm.name = "sweep_arm"
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    arm.data.materials.append(m)
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    arm.rigid_body.kinematic = True
    arm.rigid_body.restitution = 0.6
    w = float(rng.uniform(0.6, 2.0)) * float(rng.choice([-1, 1]))   # rad/s
    a0 = float(rng.uniform(0, math.pi))
    for f in (1, PRE + M.N_SIM):
        arm.rotation_euler.y = a0 + w * f / B.FPS
        arm.keyframe_insert("rotation_euler", index=1, frame=f)
    for fc in arm.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    objs = []
    for i in range(int(rng.integers(1, 4))):
        o = obj(rng, (float(rng.uniform(-2.6, 2.6)), 0, 1), size=-0.45, friction=float(rng.uniform(0.05, 0.3)))
        place_on(o, 0)
        launch(o, (float(rng.uniform(-2.5, 2.5)), 0, 0))
        objs.append(o)
    return objs, {}


@scenario("cart_shuttle", "motion", "수레가 왕복 (2~4 번) 하며 위의 물체가 관성으로 미끄러지고 · 굴러 떨어짐")
def s_shuttle(rng, n):
    w, h = float(rng.uniform(1.2, 2.0)), float(rng.uniform(0.15, 0.4))
    m = material("cart", B.rand_rgb(rng, 60, 160), 0.6)
    cart = box("cart", (0, 0, h / 2), (w, 1.2, h), m, friction=float(rng.uniform(0.3, 0.8)))
    cart.rigid_body.kinematic = True
    amp = float(rng.uniform(1.0, 2.2))
    per = float(rng.uniform(2.0, 4.5)) * B.FPS                    # 장
    f = 1
    x = -amp
    while f < PRE + M.N_SIM:
        cart.location.x = x
        cart.keyframe_insert("location", index=0, frame=int(f))
        f += per / 2
        x = -x
    for fc in cart.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = str(rng.choice(["LINEAR", "SINE"]))
    cart.location.x = -amp
    objs = []
    for i in range(int(rng.integers(1, 3))):
        o = obj(rng, (-amp + float(rng.uniform(-0.3, 0.3)) * w, 0, h + 1), size=-0.45, friction=float(rng.uniform(0.1, 0.5)))
        place_on(o, h)
        objs.append(o)
    return objs, {}


# ----------------------------------------------------------------------------- 금방 멈추던 시나리오 고치기 (2026-10-27 시험: tower_collapse 0 / 10 · block_stack 2 · shelf_catch 3 · container 3 · seesaw 5)
def _stream(rng, objs, info, xs, n, dt, size=-0.5, shape=None, t0=None):
    """화면 위 밖 (z 6.5) 에서 기다리던 물체가 dt 간격으로 차례로 놓여 떨어진다 — 장면이 끝까지 살아 있게. 놓는 장은 unpred."""
    t = int(rng.integers(0, 10)) if t0 is None else t0
    for i in range(n):
        x = float(rng.uniform(*xs))
        o = obj(rng, (x, 0, 6.5 + 1.1 * i), size=size,   # 기다리는 높이를 물체마다 달리 (같은 자리에 겹쳐 기다리다 놓이면 튕겨 나갔다) shape=shape if shape is None or isinstance(shape, str) else str(rng.choice(shape)),
                rot=tuple(rng.uniform(0, 3.14, 3)), name="drop")
        if t > 0:
            B.launch(o, (float(rng.uniform(-0.5, 0.5)), 0, 0), at=PRE - 2 + t)
            info.setdefault("unpred", []).append(t - 2)
        objs.append(o)
        t += int(rng.uniform(*dt) * B.FPS)
        if t >= M.N_SIM - 20:
            break
    info["active_any"] = True
    info["enter_from_top"] = True


def _wrap(name, extra):
    cat, desc, fn = M.SCENARIOS[name]
    def f(rng, n):
        objs, info = fn(rng, n)
        extra(rng, objs, info)
        return objs, info
    M.SCENARIOS[name] = (cat, desc + " · (flow) 계속 들어오는 물체", f)


def _tower_aimed(rng, n):
    """버티는 탑 (어긋남 작게) 을 바닥에서 굴러온 물체가 3.4~5 초 뒤에 친다 — 그 전 문맥 = 굴러오는 운동."""
    d = float(rng.choice([-1, 1]))
    x = float(rng.uniform(-0.6, 0.6)) * d
    z, objs = 0.0, []
    for i in range(int(rng.integers(3, 7))):
        o = obj(rng, (x, 0, z + 1), size=-0.5, shape=str(rng.choice(("cube", "cuboid", "rounded_cube", "plank", "can"))),
                friction=float(rng.uniform(0.5, 0.9)), bounce=0.05)
        o.rotation_euler = (0, 0, 0)
        place_on(o, z)
        bpy.context.view_layer.update()
        z = max((o.matrix_world @ v.co).z for v in o.data.vertices)
        x += d * float(rng.uniform(0.0, 0.12)) * o.dimensions.x
        objs.append(o)
    sh = obj(rng, (-d * 2.9, 0, 1), size=-0.6, shape="sphere", friction=0.05, density=float(rng.uniform(2.0, 4.0)))
    place_on(sh, 0)
    t_hit = float(rng.uniform(3.4, 5.0))
    launch(sh, (d * abs(objs[0].location.x - sh.location.x) / t_hit / M.STATE["speed"] * 1.05, 0, 0))
    return [sh] + objs[::-1], {"require_impact": "objects", "t_hit": round(t_hit, 2)}


M.SCENARIOS["tower_collapse"] = ("collision", "블록 탑 3~6 층을 바닥에서 굴러온 무거운 공이 3.4~5 초 뒤에 쳐서 무너뜨림", _tower_aimed)
def _block_stack_stream(rng, n):
    """블록 6~10 개가 0.6~1.4 초 간격으로 화면 위에서 한 자리 (± 0.15 m) 로 떨어져 쌓인다 — 어긋나면 미끄러지거나 무너짐, 끝까지 들어온다.
    (원래 판은 5.6~12 m 에 세워 둔 블록이 한꺼번에 떨어져 기다리던 물체와 부딪혀 튕겨 나갔다 — 1 / 10 통과)"""
    x = float(rng.uniform(-1.2, 1.2))
    objs, info = [], {"require_impact": "objects"}
    _stream(rng, objs, info, (x - 0.15, x + 0.15), int(rng.integers(6, 11)), (0.6, 1.4), shape=M.STACK_SHAPES, t0=0)
    for o in objs:
        o.rigid_body.friction, o.rigid_body.restitution = float(rng.uniform(0.4, 0.9)), float(rng.uniform(0.0, 0.2))
    return objs, info


M.SCENARIOS["block_stack"] = ("collision", "블록 6~10 개가 0.6~1.4 초 간격으로 한 자리에 떨어져 쌓임 — 어긋나면 미끄러지거나 무너짐", _block_stack_stream)
_wrap("shelf_catch", lambda rng, objs, info: (_stream(rng, objs, info, (-2.2, 2.2), 7, (0.8, 1.6), t0=int(rng.integers(8, 20))), info.pop("require_rest_above", None)))
_wrap("container", lambda rng, objs, info: _stream(rng, objs, info, (-1.5, 1.5), 7, (0.7, 1.5), t0=int(rng.integers(8, 20))))
_wrap("seesaw", lambda rng, objs, info: _stream(rng, objs, info, (-1.0, 1.0), 6, (1.0, 2.0), size=-0.55, t0=int(rng.integers(15, 30))))
M.SCENARIOS["moving_platform"] = ("motion", "수레 왕복 (cart_shuttle 과 같은 장치, 한 번 서고 끝나면 장면이 멈춰 flow 판에서는 왕복으로)", s_shuttle)
_wrap("stack_topple", lambda rng, objs, info: _stream(rng, objs, info, (-2.0, 2.0), 5, (1.0, 2.0), t0=int(rng.integers(20, 40))))
_wrap("flat_decel", lambda rng, objs, info: info.update(stop_scene=True))   # 멈춤이 이 장면의 결과 (멈춤 허용 판정)


if __name__ == "__main__":
    B.main()
