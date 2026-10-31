"""phys3d_perm (2026-10-30) — 물체 영속 (object permanence) 학습 세트, 4 fps. 문서 z_training/docs/PHYS3D_PERM_DATASET_2026-10-30.md
원칙: z_training/docs/DATA_DESIGN_PRINCIPLES_2026-10-26.md · SYNTHETIC_TRAINSET_PRINCIPLES_2026-10-25.md

사용자 (2026-10-30): "release predictor 한테 permanence 가르칠 거야. 4 fps 로 다양한 obj 나와도 돼. 가림막 움직이거나 가림막 고정이거나.
아니면 다른 물체가 물체 가리거나 … 물체가 나무한테 가려지는 것도 ㄱㅊ 다양하게" — 릴리즈 predictor post-training 용.

phys3d_flow (gen_phys3d_flow_blender.py — 시나리오 45 · 모양 30 · 질감 10 · 배경 노이즈 · 깊이) 를 import 하고 갈고리로만 바꾼다:
  * 4 fps: 시뮬 12 fps (부분 단계 10 = 120 Hz) 에서 3 장마다 저장 (phys3d_multi_v1 과 같은 시간 척도). 영상 32 장 = 8 초
    (릴리즈 사전학습 창 16 장 · 채점 창 32 장 둘 다 들어간다).
  * 앵커 = 가림 (사건 가림 40 % · 그냥 가림 60 %, 보임 사건 앵커 없음). 가려지기 시작하는 장 C* − 1, C* ∈ {12, 14, 16, 18, 20} (영상 번호로 층화) —
    16 장 문맥 (temporal_prefix 16) 으로 자르면 가림이 문맥 끝 4 장 전 ~ 미래 4 장 안에서 시작한다. 안 보임 최대 4 초, 끝 2 장 전에는 다시 보인다.
  * 가림 5 종: 고정 (판 · 기둥 · 나무 · 덤불 · 장 · 벽 · 상자 더미 · 천 가림막) · 앞 물체 (다른 물체가 카메라 쪽으로 지나감) ·
    올라왔다 내려감 · 지나감 (판 · 상자 · 천) · 미끄러져 와 멈췄다 떠남 (slide_hold, 새로).
  * 멈춰 있는 물체도 가린다 (still_life 시나리오 — 물체 1~4 개가 바닥 · 탁자 위에 놓여 있고 가림막 · 앞 물체가 지나감).
  * 창 검사: 움직이는 영상은 장면 정지 ≤ 30 % · 주 물체 움직임 ≥ 50 % (flow 판의 85 % 보다 느슨 — 영속이 주인공). still_life 는 검사 없음.
  /workspace/tools/blender-4.2.3-linux-x64/blender -t 1 -b -noaudio -P z_training/data/gen_phys3d_perm_blender.py -- --scenario still_life --seed 0 --out /tmp/p
대량 생성: gen_phys3d.py --multi --perm --defer-render.
"""
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_phys3d_flow_blender as F             # noqa: E402  (flow 갈고리가 깔린다)
from gen_phys3d_flow_blender import M, B, obj, place_on, material, PRE, ROOM_X, scenario   # noqa: E402

# ----------------------------------------------------------------------------- 4 fps · 32 장
B.FPS, B.SUBSTEPS = 12, 10
M.ST, M.T_SAVE = 3, 32
M.W_RAW = M.T_SAVE * M.ST                     # 96 렌더 장
M.MAX_BLIND = 48                              # 렌더 장 (4 초)
M.C_SET = (12, 14, 16, 18, 20)
M.N_SIM = 360                                 # 30 초
M.P_MID_OCC = 0.6                             # 문맥 중 가림 (앵커 전에 끝남) — 가림 사건을 하나 더
M.OCC_MODES = ("fixed", "fixed", "fixed", "front_object", "rise_lower", "sweep", "slide_hold")
# 고정 (나무 · 기둥 · 장 …) 을 3 / 7 로 — 시험 (2026-10-30, 균등) 에서 고정이 가림의 9 % 뿐이었다 (움직이는 물체만 가릴 수 있어 자주 실패).
# 멈춘 물체는 generate2 가 OCC_MODES[1:] 를 쓴다 → 거기 남은 fixed 는 place_occlusion 이 다른 모드로 바꾼다.
M.FRONT_SHAPES = M.FRONT_SHAPES + ("barrel", "can", "egg", "gem", "frustum")
P_ANCHOR_MOVING = 0.9                         # still_life 는 0 (generate 에서 바꾼다)
MIN_PX_S = 8.0
STILL_V = 0.05


# ----------------------------------------------------------------------------- 가림막 모양
def _tex_files(keys):
    return [f for f in F.TEX_FILES if any(k in f for k in keys)]


TEX_BARK = _tex_files(("bark",))
TEX_LEAF = _tex_files(("leaves", "leaf", "moss", "grass"))
TEX_WOOD = _tex_files(("wood", "plank"))
TEX_STONE = _tex_files(("brick", "stone", "rock", "concrete", "plaster", "wall"))
TEX_FABRIC = _tex_files(("fabric", "denim", "leather", "carpet", "cloth"))


def _photo(name, rng, files, scale, tint=None):
    """사진 질감 재질 (files 중 하나). 없으면 한 색."""
    if not files:
        return material(name, tint or B.rand_rgb(rng, 60, 180), float(rng.uniform(0.5, 0.95)))
    m = material(name, (200, 200, 200), float(rng.uniform(0.5, 0.95)))
    nt = m.node_tree
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(os.path.join(F.TEX_DIR, files[int(rng.integers(len(files)))]), check_existing=True)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Hue"].default_value = float(rng.uniform(0.46, 0.54))
    hsv.inputs["Value"].default_value = float(rng.uniform(0.75, 1.15))
    nt.links.new(t.outputs["Color"], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return m


def _leaf_mat(name, rng):
    if TEX_LEAF and rng.random() < 0.5:
        return _photo(name, rng, TEX_LEAF, float(rng.uniform(1.0, 3.0)))
    import colorsys
    h = float(rng.uniform(0.18, 0.38)) if rng.random() < 0.85 else float(rng.uniform(0.02, 0.12))   # 초록 · 가끔 단풍
    c1 = tuple(int(255 * x) for x in colorsys.hsv_to_rgb(h, rng.uniform(0.4, 0.8), rng.uniform(0.25, 0.6)))
    c2 = tuple(int(255 * x) for x in colorsys.hsv_to_rgb(h + rng.uniform(-0.04, 0.04), rng.uniform(0.4, 0.8), rng.uniform(0.15, 0.45)))
    m = material(name, c1, 0.8)
    nt = m.node_tree
    n = nt.nodes.new("ShaderNodeTexVoronoi")
    n.inputs["Scale"].default_value = float(rng.uniform(6, 18))
    mx = nt.nodes.new("ShaderNodeMixRGB")
    mx.inputs["Color1"].default_value = [B.srgb2lin(c) for c in c1] + [1.0]
    mx.inputs["Color2"].default_value = [B.srgb2lin(c) for c in c2] + [1.0]
    nt.links.new(n.outputs["Distance"], mx.inputs["Fac"])
    nt.links.new(mx.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return m


def _join(parts, name):
    for ob in parts:                              # 크기 · 회전을 메시에 굳힌다 (합친 뒤 원점을 옮길 때 메시 좌표 = 세계 좌표)
        for o2 in bpy.context.selected_objects:
            o2.select_set(False)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    for ob in bpy.context.selected_objects:
        ob.select_set(False)
    for ob in parts:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    if len(parts) > 1:
        bpy.ops.object.join()
    o = bpy.context.object
    o.name = name
    return o


def _cube(loc, scale, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object
    o.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    return o


def _set_origin(o, x, y, z):
    """원점을 (x, y, z) 로 (메시는 제자리) — rise_lower 가 원점 = 판 가운데로 키를 넣는다."""
    d = o.location.copy()
    d.x, d.y, d.z = o.location.x - x, o.location.y - y, o.location.z - z
    for v in o.data.vertices:
        v.co += d
    o.location = (x, y, z)


def prop(kind, name, x, w, h, rng, y):
    """충돌 없는 가림 물체 (판과 같이 원점 = (x, y, h/2)). 반환 (o, kind)."""
    mats = []
    if kind == "pillar":
        r = float(np.clip(w / 2, 0.15, 0.8))
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, vertices=32, location=(x, y, h / 2))
        o = bpy.context.object
        o.data.materials.append(_photo(name, rng, TEX_STONE, float(rng.uniform(0.6, 2.0))))
        bpy.ops.object.shade_smooth()
        parts = [o]
    elif kind == "tree":
        r = float(np.clip(w / 2 * rng.uniform(0.9, 1.15), 0.12, 0.6))
        th = float(max(h, 1.2) * rng.uniform(0.9, 1.2))
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=th + 0.2, vertices=24, location=(x, y, (th + 0.2) / 2))
        trunk = bpy.context.object
        for v in trunk.data.vertices:              # 위로 가며 조금 가늘게
            v.co.x *= 1.0 - 0.25 * (v.co.z / (th + 0.2) + 0.5)
            v.co.y *= 1.0 - 0.25 * (v.co.z / (th + 0.2) + 0.5)
        trunk.data.materials.append(_photo(name + "_bark", rng, TEX_BARK, float(rng.uniform(1.0, 3.0)), tint=(90, 65, 45)))
        bpy.ops.object.shade_smooth()
        leaf = _leaf_mat(name + "_leaf", rng)
        parts = [trunk]
        R = float(rng.uniform(0.8, 1.6))
        for i in range(int(rng.integers(3, 7))):
            bpy.ops.mesh.primitive_ico_sphere_add(radius=R * rng.uniform(0.55, 1.0), subdivisions=2,
                                                  location=(x + rng.uniform(-0.6, 0.6) * R, y + rng.uniform(-0.2, 0.3), th + rng.uniform(-0.1, 0.7) * R))
            c = bpy.context.object
            c.scale = (1.0, 0.7, float(rng.uniform(0.7, 1.0)))
            c.data.materials.append(leaf)
            parts.append(c)
    elif kind == "bush":
        leaf = _leaf_mat(name, rng)
        parts = []
        n = int(rng.integers(3, 7))
        for i in range(n):
            rr = float(max(h, w / n * 1.4) * rng.uniform(0.45, 0.6))
            cx = x + (i / max(n - 1, 1) - 0.5) * max(w - rr, 0.0)
            bpy.ops.mesh.primitive_ico_sphere_add(radius=rr, subdivisions=2, location=(cx, y, rr * rng.uniform(0.6, 0.9)))
            c = bpy.context.object
            c.scale = (1.0, 0.6, float(np.clip(h / (1.6 * rr), 0.6, 1.3)))
            c.data.materials.append(leaf)
            parts.append(c)
    elif kind in ("cabinet", "wall"):
        d = float(rng.uniform(0.3, 0.6)) if kind == "cabinet" else float(rng.uniform(0.15, 0.3))
        m = _photo(name, rng, TEX_WOOD if kind == "cabinet" else TEX_STONE, float(rng.uniform(0.4, 1.5)))
        parts = [_cube((x, y, h / 2), (w, d, h), m)]
        if kind == "cabinet" and rng.random() < 0.6:   # 문 두 짝 경계선 · 손잡이
            dark = material(name + "_trim", (40, 35, 30), 0.6)
            parts.append(_cube((x, y - d / 2 - 0.01, h / 2), (0.02, 0.02, h * 0.92), dark))
            for sx in (-1, 1):
                parts.append(_cube((x + sx * 0.08, y - d / 2 - 0.03, h * 0.55), (0.03, 0.04, 0.18), dark))
    elif kind == "crates":
        m = _photo(name, rng, TEX_WOOD, float(rng.uniform(0.8, 2.5)), tint=(150, 110, 70))
        s = float(np.clip(w, 0.4, 1.4))
        nx = max(1, int(round(w / s)))
        nz = max(1, int(math.ceil(h / s)))
        parts = [_cube((x + (i - (nx - 1) / 2) * s, y + rng.uniform(-0.05, 0.05), s / 2 + j * s), (s * 0.98, s * 0.9, s * 0.98), m)
                 for j in range(nz) for i in range(nx)]
    else:                                          # "screen" — 천 · 사진 질감 판 (발 있음)
        o, _st = M._panel_v1(name, x, w, h, rng, y=y, feet=True)
        o.data.materials.clear()
        o.data.materials.append(_photo(name + "_cloth", rng, TEX_FABRIC or TEX_WOOD, float(rng.uniform(0.6, 2.0))))
        return o, kind
    o = _join(parts, name)
    _set_origin(o, x, y, h / 2)
    return o, kind


FIXED_KINDS = ("panel", "pillar", "tree", "bush", "cabinet", "wall", "crates", "screen")
FIXED_P = (0.2, 0.1, 0.18, 0.1, 0.12, 0.1, 0.1, 0.1)
MOVING_KINDS = ("panel", "screen", "crates")    # 장 · 벽이 미끄러져 가는 건 어색해서 뺐다 (2026-10-30 시험)
MOVING_P = (0.5, 0.3, 0.2)
M._panel_v1 = M.panel


def panel_any(name, x, w, h, rng, y=M.OCC_Y, feet=True):
    """place_occlusion 이 부르는 판 → 가림 모양 고르기. 고정은 8 종, 지나가는 · 멈추는 것은 4 종, 올라오는 것은 판."""
    if name == "occ_fixed":
        kind = str(rng.choice(FIXED_KINDS, p=FIXED_P))
        if kind == "tree" and w > 1.3:            # 굵기를 못 맞추는 나무는 덤불 · 장으로
            kind = "bush" if h < 1.8 else "cabinet"
        if kind == "bush" and h > 2.0:
            kind = "tree" if w <= 1.3 else "wall"
    elif name == "occ_sweep":
        kind = str(rng.choice(MOVING_KINDS, p=MOVING_P))
    else:
        kind = "panel"
    if kind == "panel":
        return M._panel_v1(name, x, w, h, rng, y=y, feet=feet)
    return prop(kind, name, x, w, h, rng, y)


M.panel = panel_any
_place_v1 = M.place_occlusion


def place_occlusion(mode, rng, sc, objs, lab, T, t0=None, lo=8, need_move=None):
    """slide_hold (새로): 가림막이 화면 밖에서 미끄러져 와 물체 앞에서 0.8~3 초 멈췄다가 (40 % 는 왔던 쪽으로) 떠난다. 나머지는 v1 그대로."""
    if mode == "fixed" and need_move is False:
        mode = str(rng.choice(("front_object", "rise_lower", "sweep", "slide_hold")))
    if mode != "slide_hold":
        return _place_v1(mode, rng, sc, objs, lab, T, t0=t0, lo=lo, need_move=need_move)
    r = _place_v1("sweep", rng, sc, objs, lab, T, t0=t0, lo=lo, need_move=need_move)
    if r is None:
        return None
    made, front, desc = r
    o = made[0]
    fc = [c for c in o.animation_data.action.fcurves if c.data_path == "location" and c.array_index == 0][0]
    (fa, xa), (fb, xb) = [(k.co.x, k.co.y) for k in fc.keyframe_points][:2]
    vs = (xb - xa) / (fb - fa)                    # x / 렌더 장
    tc = PRE + desc["t_star"]
    qx = xa + vs * (tc - fa)
    ta = tc - int(rng.integers(0, max(2, int(0.6 * B.FPS))))
    hold = int(rng.uniform(0.8, 3.0) * B.FPS)
    back = bool(rng.random() < 0.4)
    xe = xa if back else xb
    keys = [(ta - (qx - xa) / vs, xa), (ta, qx), (ta + hold, qx), (ta + hold + abs((xe - qx) / vs), xe)]
    o.animation_data_clear()
    M._key_x(o, keys, interp="BEZIER")            # 들어올 때 늦추고 떠날 때 붙는다
    desc.update(mode="slide_hold", hold=hold, back=back)
    return made, front, desc


M.place_occlusion = place_occlusion


# ----------------------------------------------------------------------------- 앵커 · 창 검사
def anchor_policy(rk, Cs):
    return "occ_event" if rk.random() < 0.4 else "occ_any"


M.ANCHOR_POLICY = anchor_policy


MIN_HID_SAVED = 2                             # 앵커 가림이 저장 장 2 장 (0.5 초) 이상 — 1 장 스치는 가림은 영속을 가르치지 않는다


def _size(rng):
    """영속 시나리오의 물체 크기 — 화면 30~65 px (flow 기본 15~45 px 보다 크게, 가려짐 · 다시 나타남이 잘 보이게)."""
    return float(M.loguni(rng, 0.4, 0.85))


PERM_SHAPES = tuple(sh for sh in M.SHAPES if sh not in ("plate", "rod", "plank"))   # 납작 · 가는 것은 눕으면 안 보인다


def window_hook(lw, objs, info, Cs, rng):
    hid = lw["vis_frac"][::M.ST, 0] <= B.SEEN
    if max([b - a for a, b in M._runs(hid)] + [0]) < MIN_HID_SAVED:
        return [f"hidden < {MIN_HID_SAVED} saved"]
    if info.get("static"):
        return []
    keep = [i for i, o in enumerate(objs) if o.get("role") != "front"]
    fps_s = B.FPS / M.ST
    v = np.linalg.norm(np.diff(lw["pos3d"][::M.ST][:, keep], axis=0), axis=-1) * fps_s
    still = (v < STILL_V).all(1)
    main_mv = ~still if info.get("active_any") else v[:, 0] >= STILL_V
    px = np.linalg.norm(np.diff(lw["pos"][::M.ST][:, 0], axis=0), axis=-1) * fps_s
    why = []
    if still.mean() > 0.3:
        why.append(f"scene still {still.mean():.2f}")
    if main_mv.mean() < 0.5:
        why.append(f"main moving {main_mv.mean():.2f}")
    if main_mv.any() and np.median(px[main_mv]) < MIN_PX_S:
        why.append(f"slow {np.median(px[main_mv]):.1f}px/s")
    return why


M.WINDOW_HOOK = window_hook
_gen_flow = B.generate


def generate_perm(name, k, seed, *a, **kw):
    M.P_ANCHOR_MOVING = 0.0 if name == "still_life" else P_ANCHOR_MOVING
    return _gen_flow(name, k, seed, *a, **kw)


B.generate = generate_perm


# ----------------------------------------------------------------------------- 새 시나리오
@scenario("still_life", "permanence", "물체 1~4 개가 바닥 · 탁자 위에 놓여 있고 (멈춤) 가림막 · 앞 물체가 지나가며 가렸다 드러냄")
def s_still_life(rng, n):
    objs = []
    xs = rng.permutation(np.linspace(-2.4, 2.4, 6))
    table = rng.random() < 0.3
    if table:
        tw, th = float(rng.uniform(1.6, 3.0)), float(rng.uniform(0.6, 1.2))
        tx = float(rng.uniform(-1.5, 1.5))
        M.block("table", tx, tw, th, rng, bounce=0.2)
    for i in range(int(rng.integers(1, 5))):
        if table and i == 0:
            x, z = tx + float(rng.uniform(-0.3, 0.3)) * tw, th
        else:
            x, z = float(xs[i]), 0.0
        o = obj(rng, (x, 0, z + 1), size=_size(rng) * (1.0 if i == 0 else 0.7), shape=str(rng.choice(PERM_SHAPES)), friction=0.8)
        place_on(o, z)
        o.rigid_body.linear_damping = o.rigid_body.angular_damping = 0.9
        objs.append(o)
    return objs, {"static": True, "active_any": True}


@scenario("roll_behind", "permanence", "물체 1~2 개가 바닥을 천천히 · 빠르게 굴러 · 미끄러져 가림막 뒤를 지나감 (벽에서 튕겨 돌아오기도)")
def s_roll_behind(rng, n):
    objs = []
    for i in range(int(rng.integers(1, 3))):
        d = float(rng.choice([-1, 1]))
        o = obj(rng, (-d * float(rng.uniform(1.8, 2.8)), 0, 1), size=_size(rng), shape=str(rng.choice(F.ROLLERS[:-1] if rng.random() < 0.7 else PERM_SHAPES)),
                friction=float(rng.uniform(0.05, 0.3)))
        place_on(o, 0)
        F.launch(o, (d * float(rng.uniform(0.6, 1.6)), 0, 0))
        objs.append(o)
    return objs, {}


@scenario("toss_behind", "permanence", "물체를 포물선으로 던져 높은 가림막 · 나무 뒤로 날아가 떨어짐 · 튕김")
def s_toss_behind(rng, n):
    d = float(rng.choice([-1, 1]))
    o = obj(rng, (-d * 2.6, 0, float(rng.uniform(0.5, 1.5))), size=_size(rng) * 0.8, shape=str(rng.choice(PERM_SHAPES)))
    F.launch(o, (d * float(rng.uniform(1.0, 2.2)), 0, float(rng.uniform(3.0, 5.5))))
    return [o], {}


# ----------------------------------------------------------------------------- v2 (2026-10-31) — 영속 · 이식 원칙 (DATA_DESIGN_PRINCIPLES 1.5 부)
# 사용자 (2026-10-31): "데이터 이렇게 만들어야될 듯 … 그릇 덮기 사건은 안해도 될 듯". PHYS3D_PERM_V2=1 일 때만 켠다 (기본 = v1 그대로).
#   원칙 6 ① 그림자 누수: 움직이는 물체 (ACTIVE rigid body) 는 그림자를 만들지 않는다 — 가려진 동안 그림자로 자리가 보이면 계산 대신 그림자를 쓴다.
#   원칙 6 ② 가림막 누수: 가림막은 물리가 끝난 뒤 물체 경로 위 (t* 의 투영 자리) 에만 놓여, 가림막이 있으면 곧 물체가 거기 있다는 단서다.
#       → 미끼 가림막 1~2 개 (같은 종류 · 같은 움직임) 를 경로와 무관한 자리 · 시각에 둔다. 주 물체의 보임 (저장 장) 을 바꾸면 버린다.
#   원칙 7 late 다양성: 앵커 가림의 P_STAY 는 창 끝까지 다시 안 나타나는 것 (multi 의 P_STAY).
#   물체 크기: 카메라를 당긴다 (PHYS3D_LENS, run 스크립트가 넣는다 — 라벨 투영도 같은 카메라라 맞다).
V2 = os.environ.get("PHYS3D_PERM_V2") == "1"
P_DECOY = 0.7
if V2:
    M.P_STAY = float(os.environ.get("PHYS3D_P_STAY", 0.6))
    _save_v1 = M.save_for_render

    def save_for_render_v2(sc, stem, s):
        for ob in bpy.data.objects:
            if ob.rigid_body is not None and ob.rigid_body.type == "ACTIVE":
                ob.visible_shadow = False
        return _save_v1(sc, stem, s)

    M.save_for_render = save_for_render_v2

    def decoys(sc, objs, lab, s, Cs, rng, res, info):
        info["n_decoy"] = 0
        if rng.random() >= P_DECOY:
            return objs
        cam = sc.camera
        g0 = M.fast_label(sc, objs, res, range(s, s + M.W_RAW, M.ST), [0])["vis_frac"][s:s + M.W_RAW:M.ST, 0] <= B.SEEN
        for _ in range(int(rng.integers(1, 3))):
            for _try in range(6):
                t = int(rng.integers(s, s + M.W_RAW))
                y, z = (float(v) for v in lab["pos3d"][t, 0, 1:])
                x = float(rng.uniform(-2.0, 2.0))
                c = B.world_to_camera_view(sc, cam, M.Vector((x, y, z)))
                if not (0.1 < c.x < 0.9 and 0.1 < c.y < 0.9 and c.z > 0):
                    continue
                fake = dict(pos3d=np.tile(np.array([x, y, z])[None, None], (M.N_SIM, 1, 1)), inframe_frac=np.ones((M.N_SIM, 1)),
                            vel3d=np.zeros((M.N_SIM, 1, 3)))
                mode = str(rng.choice(M.OCC_MODES))
                r = M.place_occlusion(mode, rng, sc, objs, fake, M.N_SIM, t0=t, need_move=None)
                if r is None:
                    continue
                made, front, desc = r
                g1 = M.fast_label(sc, objs + front, res, range(s, s + M.W_RAW, M.ST), [0])["vis_frac"][s:s + M.W_RAW:M.ST, 0] <= B.SEEN
                if (g1 != g0).any():                          # 주 물체 보임이 바뀌면 미끼가 아니다
                    for ob in made + front:
                        bpy.data.objects.remove(ob, do_unlink=True)
                    continue
                for f in front:
                    f["role"] = "front"
                objs = objs + front
                for ob in made:
                    ob["decoy"] = 1
                info["n_decoy"] += 1
                info.setdefault("decoy_modes", []).append(desc.get("mode", mode))
                break
        return objs

    M.PRE_FINAL_HOOK = decoys


if __name__ == "__main__":
    B.main()
