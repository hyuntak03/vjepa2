"""phys3d_single — 물체 1 개 3D 판 (Blender 4.2 rigid body + 렌더) (2026-10-19). 문서 z_training/docs/PHYS3D_SINGLE_DATASET_2026-10-19.md

gen_phys3d_blender.py (phys3d_v1) 를 import 해서 쓰고 바뀐 것만 둔다 — 2D 의 gen_phys2d_single.py 가 gen_phys2d_sup.py 에 하는 것과 같은 관계.
blender 안에서 돈다:
  /workspace/tools/blender-4.2.3-linux-x64/blender -b -noaudio -P z_training/data/gen_phys3d_single_blender.py -- \
      --scenario wall_bounce --seed 0 --out /path/to/out [--engine cpu]
대량 생성은 gen_phys3d.py --single (워커 모드 --jobs --claims --results 는 v1 과 같다).

v1 과 다른 것 (2D single 과 같은 생각)
  * **물체는 언제나 1 개.** 방해 물체 · 물체끼리 충돌 (collide) · 쌓기 (stack_topple) 를 뺐다.
    고체성은 정적 지형 (가운데 벽 · 떠 있는 블록 · 선반 · 통 · 진자 앞 상자) 과의 충돌로만 배운다.
  * **가림 시점 · 속력에 편향 없음** — 1 차 시뮬 궤적을 본 뒤 목표 가림 구간 n 개 (0~8, clip 마다 seed 로 고정) 마다 시작 장 a ~ U[−L+1, T−1] 를
    한 번 뽑고 그 구간 궤적 (카메라에서 가림막 평면으로 투영한 물체 외접 상자) 을 덮는 고정 판을 놓는다. 가림막은 충돌이 없어 물리는 그대로다.
    판은 실제로 덮는 장 수에 반비례해 받는다 (NCOV) — 고정 판이 느린 · 멈춘 물체를 오래 가리는 몫을 되돌린다. 실패한 구간은 버린다
    (a 를 다시 뽑으면 잘 통과하는 첫머리로 몰린다). 시험 312 편: 상대 시간 10 칸 가림 0.28~0.38, 속력 5 칸 0.32~0.37.
  * **길이를 자른다** — 물체가 멈추면 TAIL 장만 남긴다 (MIN_T 이상). 렌더도 그만큼만 한다.
  * C16K16 사건 행 요구를 뺐다. 경계 행은 두 벌: rows (사건, v1 규칙) + uniform (4 장 간격 · ctx_start 0, 드라이버가 만든다).
  * 라벨 키 · 단위는 v1 과 같다 (= phys2d_sup 과 같다).
"""
import json, math, os, shutil, subprocess, sys, time

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_phys3d_blender as B                                 # noqa: E402
from gen_phys3d_blender import (PRE, NC, SHAPES, COLORS, HIDDEN, SEEN, ROOM_X, OCC_Y, APEX_MAX,   # noqa: E402
                                box, make_object, launch, occluder, material, rand_rgb, screen_on_path)

N_FRAMES = 96
P_EPISODES = (0.06, 0.06, 0.10, 0.12, 0.14, 0.14, 0.14, 0.12, 0.12)   # 목표 가림 구간 0~8 개 (clip 마다 seed 로 고정). 실제로 놓이는 것은
                                               # 그 37 % 쯤 — 덮는 장 수로 받기 (NCOV) · 겹침 때문에 (시험 2026-10-19: 목표 2.21 → 0.81, 가림 22 %)
HIDE_LEN = (2, 16)
SCREEN_MAX = (2.2, 2.4)                        # 가림막 최대 폭 · 높이 (m, 가림막 평면 — 화면 폭 ≈ 5 m)
P_COVER_REST = float(os.environ.get("PHYS3D_COVER_REST", 1.0))   # 구간마다: 멈춘 자리를 덮는 가림막을 허락할 확률. 1 = 규칙 없음 —
                                               # NCOV 가 멈춘 자리 (오래 덮는 판) 를 이미 덜 받는다. 0.3 이면 늦은 구간이 거의 다 떨어져 가림이
                                               # 영상 첫머리로 몰렸다 (시험 2026-10-19: 목표 구간 P t=0 0.21 · 중간 0.07)
MAX_HIDDEN_FRAC = 0.7
SCREEN_TRIES = 12
NCOV = os.environ.get("PHYS3D_NCOV", "1") == "1"
MIN_T, TAIL = 48, 16                           # 12 fps 96 장에서 물체가 멈추면 장면이 멈춘다. 창 C16+K16 = 32 장
STILL_PX = 0.4                                 # px/장 — 이하면 멈춤
MOTION_MIN = 0.5                               # 자른 영상에서 움직이는 장 비율 하한
MAX_ATTEMPTS = 40
MAX_BLIND_RUN = None                           # 안 보임 (vis_frac ≤ SEEN, 화면 밖 포함) 이 이어지는 장 수 상한. None = 없음 (v1).
                                               # v2 (gen_phys3d_single_v2_blender.py) 가 정한다 — v1 은 상한이 없어 가림막이 겹쳐 59 장 내리 가려진 영상이 나왔다
TIMED_SLIDE = None                             # 장 수 D 면 시간 고른 가림막이 화면 밖에서 D 장 동안 미끄러져 들어와 목표 구간 [s, e) 만 서 있다 나간다 (v2).
                                               # None = 고정 판 (v1). 고정 판은 멈춘 물체를 멈춰 있는 내내 가려 상한 (MAX_BLIND_RUN) 과 함께 쓸 수 없다
                                               # (v2 시험 2026-10-21: 멈춘 장 가림 3.6 % · 움직이는 장 20~29 %, 영상 뒤쪽 3 %)
EP_REF_T = None                                # 장 수 R 이면 목표 가림 구간 수를 (자른 길이 T / R) 배 — 길이가 다양해도 가림 비율이 같게 (v2). None = 그대로 (v1)
ROW_MIN_CTX, ROW_MIN_PRED = 4, 8               # 사건 행 (B.event_rows) 의 최소 문맥 · 예측 장 수 (v1 값)
SAVE_STRIDE = 1                                # s > 1 이면 시뮬 · 라벨 · 사건은 12 fps 로 하고 렌더 · 저장은 s 장마다 1 장 (v3: 3 → 4 fps). 1 = v1

SCENARIOS = {}


def scenario(name, category, desc, fn=None):
    def deco(f):
        SCENARIOS[name] = (category, desc, f)
        return f
    return deco(fn) if fn is not None else deco


def wall_info(o):
    """정적 지형 상자 하나의 (x0, x1, z0, z1) — 충돌 판정용 (월드 AABB)."""
    cs = [o.matrix_world @ Vector(c) for c in o.bound_box]
    return [min(c.x for c in cs), max(c.x for c in cs), min(c.z for c in cs), max(c.z for c in cs)]


# --- 중력 -------------------------------------------------------------------------
@scenario("drop_bounce", "gravity", "물체 하나를 여러 높이에서 놓거나 살짝 던짐 — 자유 낙하 · 튕김 (바닥 · 옆벽)")
def s_drop(rng, n_frames, ident):
    o = make_object(ident, (float(rng.uniform(-2.4, 2.4)), 0, float(rng.uniform(1.8, 4.5))), float(rng.uniform(0.4, 0.7)), rng,
                    rot=tuple(rng.uniform(0, 3.14, 3)))
    launch(o, (rng.uniform(-2.0, 2.0), 0, rng.uniform(-1, 2.5)))
    return [o], [], {}


scenario("projectile", "gravity", "옆에서 던진 물체의 포물선 — 바닥 · 옆벽에서 튕김", B.s_proj)
scenario("toss_up", "gravity", "위로 던진 물체가 꼭대기에서 멈췄다 되돌아 떨어짐", B.s_toss)
scenario("ramp_roll", "gravity", "경사면을 굴러 · 미끄러져 내려와 바닥으로 — 경사 각 · 높이 무작위", B.s_ramp)
scenario("ledge_fall", "gravity", "선반 위를 가다 끝에서 떨어짐 (중력 개시)", B.s_ledge)


@scenario("pendulum", "gravity", "진자 (막대 + 추) — 길이 · 진폭 무작위. 35 % 는 스윙 길에 상자 (부딪혀 튕김)")
def s_pend(rng, n_frames, ident):
    objs, occ, info = B.s_pend(rng, n_frames, ident)
    for o in occ:                                  # v1 의 진자 앞 가림막은 뺀다 — 시간 고른 가림막이 대신한다
        bpy.data.objects.remove(o, do_unlink=True)
    blk = bpy.data.objects.get("block")
    if blk is not None:
        info["walls"] = [wall_info(blk)]
    return objs, [], info


# --- 영속 (시나리오 고유 가림막. 시간 고른 가림막은 모든 시나리오에 따로 붙는다) -----------------
scenario("occ_pass", "permanence", "미끄러지는 · 던진 물체가 고정 가림막 1~2 개 뒤를 지나 다시 나옴", B.s_occ_pass)


@scenario("moving_screen", "permanence", "가림막이 오가며 물체를 여러 번 가림 — 물체는 정지 · 미끄러짐 · 튕김")
def s_moving_screen(rng, n_frames, ident):
    air = rng.random() < 0.4
    o = make_object(ident, (float(rng.uniform(-2, 2)), 0, float(rng.uniform(1, 3)) if air else 0.45), 0.5, rng,
                    friction=float(rng.uniform(0.01, 0.3)))
    launch(o, (rng.uniform(-2.5, 2.5), 0, rng.uniform(0, 3) if air else 0))
    n_pass = int(rng.integers(2, 5))
    x0 = float(rng.choice([-1, 1]) * 3.0)
    keys, f = [], 1
    step = (n_frames + PRE) / n_pass
    for k in range(n_pass + 1):
        keys.append((int(f), x0 if k % 2 == 0 else -x0))
        f += step
    return [o], [occluder("moving_screen", x0, float(rng.uniform(1.0, 1.6)), float(rng.uniform(2.5, 4.0)), rng, anim=keys)], \
        {"no_trim": True, "skip_motion": True}


@scenario("drop_behind", "permanence", "물체가 가림막 뒤로 떨어져 멈춤, 그다음 가림막이 비켜나며 드러남")
def s_drop_behind(rng, n_frames, ident):
    objs, occ, info = B.s_drop_behind(rng, n_frames, ident)
    return objs, occ, dict(info, min_T=info["unpred"][0] + 16 + TAIL, skip_motion=True)   # 가림막이 비켜난 뒤까지는 남긴다


# --- 고체성 (정적 지형과의 충돌만) ------------------------------------------------------
@scenario("shelf_land", "solidity", "선반 · 판 1~3 개 위로 떨어져 그 위에 서거나 튕겨 내려감 — 통과하지 않는다")
def s_shelf(rng, n_frames, ident):
    n_sh = int(rng.integers(1, 4))
    xs = rng.permutation([-1.8, 0.0, 1.8])[:n_sh]
    walls = []
    for k, x in enumerate(xs):
        s = box(f"shelf{k}", (float(x), 0, float(rng.uniform(0.8, 2.4))), (float(rng.uniform(1.2, 2.0)), 1.6, 0.12),
                material(f"shelf{k}", rand_rgb(rng), 0.7), rot=(0, float(rng.uniform(-0.25, 0.25)), 0), bounce=float(rng.uniform(0.2, 0.7)))
        walls.append(wall_info(s))
    o = make_object(ident, (float(rng.choice(xs) + rng.uniform(-0.5, 0.5)), 0, float(rng.uniform(3.2, 4.6))), 0.45, rng)
    launch(o, (rng.uniform(-1, 1), 0, rng.uniform(-1, 1.5)))
    return [o], [], {"walls": walls, "require_wall": True}


@scenario("wall_bounce", "solidity", "던지거나 미끄러진 물체가 방 가운데 벽에 부딪혀 되튕김 — 벽 높이 · 위치 · 반발 무작위")
def s_wall(rng, n_frames, ident):
    objs, occ, info = B.s_wall(rng, n_frames, ident)
    return objs, occ, dict(info, walls=[wall_info(bpy.data.objects["wall_mid"])], require_wall=True)


@scenario("obstacle_bounce", "solidity", "방 안 장애물 (바닥에 선 벽 · 떠 있는 블록) 1~3 개 사이를 잘 튀는 물체가 돌아다님 — 부딪히면 되튕긴다")
def s_obstacle(rng, n_frames, ident):
    walls = []
    for k in range(int(rng.integers(1, 4))):
        for _ in range(30):
            if rng.random() < 0.5:                     # 바닥에서 선 벽
                w, h = float(rng.uniform(0.2, 0.45)), float(rng.uniform(0.9, 2.8))
                x, z = float(rng.uniform(-2.6, 2.6)), h / 2
            else:                                      # 떠 있는 블록
                w, h = float(rng.uniform(0.5, 1.4)), float(rng.uniform(0.25, 0.6))
                x, z = float(rng.uniform(-2.4, 2.4)), float(rng.uniform(1.2, 3.4))
            r = [x - w / 2, x + w / 2, z - h / 2, z + h / 2]
            if any(not (r[1] + 0.6 < q[0] or r[0] - 0.6 > q[1] or r[3] + 0.6 < q[2] or r[2] - 0.6 > q[3]) for q in walls):
                continue
            box(f"obstacle{k}", (x, 0, z), (w, 1.6, h), material(f"obstacle{k}", rand_rgb(rng), 0.7), bounce=float(rng.uniform(0.6, 0.95)))
            walls.append(r)
            break
    for _ in range(30):                                # 시작 자리는 장애물 밖
        p = (float(rng.uniform(-2.6, 2.6)), float(rng.uniform(0.8, 3.8)))
        if all(not (q[0] - 0.5 < p[0] < q[1] + 0.5 and q[2] - 0.5 < p[1] < q[3] + 0.5) for q in walls):
            break
    o = make_object(ident, (p[0], 0, p[1]), float(rng.uniform(0.4, 0.55)), rng, friction=0.1, bounce=float(rng.uniform(0.8, 0.95)))
    sp, ang = float(rng.uniform(3.0, 6.0)), float(rng.uniform(0, 2 * math.pi))
    launch(o, (sp * math.cos(ang), 0, sp * math.sin(ang)))
    return [o], [], {"walls": walls, "require_wall": True}


@scenario("container", "solidity", "통 (U 자) 안으로 떨어지거나 던져진 물체 — 벽 안에서 튕기고 갇힌다")
def s_container(rng, n_frames, ident):
    cx = float(rng.uniform(-1.5, 1.5))
    w = float(rng.uniform(1.6, 2.4))
    h = float(rng.uniform(0.7, 1.3))
    m = material("bin", rand_rgb(rng), 0.7)
    b = float(rng.uniform(0.2, 0.6))
    box("bin_floor", (cx, 0, 0.06), (w, 1.6, 0.12), m, bounce=b)
    walls = [wall_info(box("bin_l", (cx - w / 2, 0, h / 2), (0.15, 1.6, h), m, bounce=b)),
             wall_info(box("bin_r", (cx + w / 2, 0, h / 2), (0.15, 1.6, h), m, bounce=b))]
    if rng.random() < 0.5:                                         # 위에서 떨어뜨림
        o = make_object(ident, (cx + float(rng.uniform(-0.4, 0.4) * w), 0, float(rng.uniform(2.5, 4.2))), 0.45, rng)
        launch(o, (rng.uniform(-1, 1), 0, rng.uniform(-1, 1)))
    else:                                                          # 옆에서 던져 넣음
        side = rng.choice([-1, 1])
        x0 = float(np.clip(cx - side * rng.uniform(2.0, 3.0), -2.9, 2.9))
        o = make_object(ident, (x0, 0, float(rng.uniform(0.6, 1.4))), 0.45, rng)
        tt = float(rng.uniform(1.0, 1.6))
        tt = min(tt, 2 * math.sqrt(2 * (APEX_MAX - h - 0.6) / B.G))
        launch(o, ((cx - x0) / tt, 0, (h + 0.6 - o.location.z + 0.5 * B.G * tt ** 2) / tt))
    return [o], [], {"walls": walls, "final_in": (cx - w / 2 + 0.075, cx + w / 2 - 0.075, h)}


# ----------------------------------------------------------------------------- 시간 고른 가림막
def project_box(cam, p, r):
    """물체 중심 p (m) 반경 r 의 외접 상자 8 꼭짓점을 카메라에서 가림막 평면 y = OCC_Y 로 투영 → (x, z) 배열."""
    c = np.array(cam.location[:])
    out = []
    for dx in (-r, r):
        for dy in (-r, r):
            for dz in (-r, r):
                q = p + np.array([dx, dy, dz])
                s = (OCC_Y - c[1]) / (q[1] - c[1])
                out.append(c + s * (q - c))
    out = np.array(out)
    return out[:, [0, 2]]


import collections
FAIL = collections.Counter()                   # 가림막 후보가 떨어진 이유 (시험용 집계, json metrics 에 남긴다)


def inside(pts, r):
    return bool((pts[:, 0] >= r[0]).all() and (pts[:, 0] <= r[1]).all() and (pts[:, 1] >= r[2]).all() and (pts[:, 1] <= r[3]).all())


def longest_run(m):
    best = cur = 0
    for x in m:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def blind_run(m):
    """상한 검사용 — 저장하는 장 (0, s, 2s, …) 에서 잰 연속 길이 × s (렌더 장 단위). s = 1 이면 longest_run.
    12 fps 로 재면 판 가장자리에서 vis_frac 가 0.3 을 오르내려 한 장씩 끊기는데, 3 장마다 남기면 그 끊김이 빠져 연속이 길어진다
    (2026-10-23 v3 시험: 12 fps 42 장 → 4 fps 26 장 = 렌더 78 장)."""
    return longest_run(m[::SAVE_STRIDE]) * SAVE_STRIDE


def place_timed_screens(rng, cam, lab, radius, T, n_ep):
    """가림 구간 n_ep 개. [a, a+L) 의 a ~ U[−L+1, T−1] → 장마다 구간에 들 확률이 같다 (2D single 과 같다).
    구간 동안 물체가 화면 안이어야 하고, 그 궤적의 투영을 모두 덮는 고정 판 (x0, x1, z0, z1) 을 고른다."""
    p3, inf = lab["pos3d"][:T, 0], lab["inframe_frac"][:T, 0]
    rest = project_box(cam, p3[T - 1], radius)
    rects = []
    if MAX_BLIND_RUN is not None:                  # 이미 안 보이는 장 (시나리오 가림막 · 화면 밖) + 받은 판이 완전히 덮는 장
        blind = lab["vis_frac"][:T, 0] <= SEEN
        boxes = [project_box(cam, p3[t], radius) for t in range(T)]
    for _ in range(n_ep):
        # 구간 시작 a 는 구간마다 한 번만 뽑고, 실패하면 같은 a 에서 길이 · 여백만 다시 뽑는다 (8 번 넘게 실패하면 그 구간은 없다).
        # a 까지 다시 뽑으면 잘 통과하는 자리 (영상 첫머리: 잘린 짧은 구간 = 작은 판) 로 몰려 시험에서 목표 구간 P 가 t=0 0.57 · 중간 0.16~0.20 였다
        # (2026-10-19, 311 편). 멈춘 자리를 덮어도 되는지도 구간마다 한 번 (후보마다 정하면 결국 덮는 후보가 뽑혀 멈춘 장 가림 70 %)
        allow_rest = rng.random() < P_COVER_REST
        L = int(rng.integers(HIDE_LEN[0], HIDE_LEN[1] + 1))
        a = int(rng.integers(-L + 1, T))
        for j in range(8):
            if j:
                L = int(rng.integers(max(HIDE_LEN[0], 1 - a), HIDE_LEN[1] + 1)) if a + HIDE_LEN[1] > 0 else L
            s, e = max(0, a), min(T, a + L)
            if e - s < 1 or (inf[s:e] < 0.9).any():
                FAIL["frame"] += 1
                continue
            m0, m1 = rng.uniform(0.05, 0.2, 2), rng.uniform(0.05, 0.2, 2)
            # 궤적이 크기 상한을 넘으면 버리지 않고 구간을 줄인다 — 버리면 빠른 구간이 덜 가려진다.
            # 줄이는 쪽은 장마다 앞 · 뒤 무작위. 끝만 당기면 구간 시작이 살아남아 t = 0 (잘린 구간 a ≤ 0 의 시작) 이 더 가려졌다
            # (val 1,040 편 2026-10-19: 첫 8 장 43 % · 중간 35 %, uniform 창 블록 자리 36.4 → 32.0)
            while True:
                pts = np.concatenate([project_box(cam, p3[t], radius) for t in range(s, e)])
                r = [pts[:, 0].min() - m0[0], pts[:, 0].max() + m1[0], pts[:, 1].min() - m0[1], pts[:, 1].max() + m1[1]]
                if (r[1] - r[0] <= SCREEN_MAX[0] and r[3] - r[2] <= SCREEN_MAX[1]) or e - s == 1:
                    break
                if rng.random() < 0.5:
                    e -= 1
                else:
                    s += 1
            if r[1] - r[0] > SCREEN_MAX[0] or r[3] - r[2] > SCREEN_MAX[1]:
                FAIL["size"] += 1
                continue
            if r[2] < 0.5:                             # 바닥 가까우면 바닥에 세운다
                r[2] = 0.0
            touch_rest = not (r[1] < rest[:, 0].min() or r[0] > rest[:, 0].max() or r[3] < rest[:, 1].min() or r[2] > rest[:, 1].max())
            if touch_rest and not allow_rest:
                FAIL["rest"] += 1
                continue
            if NCOV and not TIMED_SLIDE:             # 미끄러지는 판은 [s, e) 에만 서 있어 덮는 장 수가 목표 길이 그대로다
                # 고정 판은 느린 물체 · 멈춘 물체를 목표 구간보다 오래 가린다 → 실제로 덮는 장 수 n_cov 에 반비례해 받고, 떨어지면 그 구간은 없다.
                # 없으면 시험에서 멈춘 장 52 % · 움직이는 장 34~38 % 가림 (2026-10-19, 156 편). 있으면 속력별 19~25 %
                n_cov = int(sum(inside(project_box(cam, p3[t], radius), r) for t in range(T)))
                if rng.random() > (e - s) / max(n_cov, e - s):
                    FAIL["ncov"] += 1
                    break
            if MAX_BLIND_RUN is not None:
                # 이어지는 안 보임이 상한을 넘으면 그 구간은 없다 (NCOV 와 같은 처리 — a 를 다시 뽑지 않는다)
                cov = np.zeros(T, bool)
                for t in (range(s, e) if TIMED_SLIDE else range(T)):
                    cov[t] = inside(boxes[t], r)
                if blind_run(blind | cov) > MAX_BLIND_RUN:
                    FAIL["run"] += 1
                    break
                blind = blind | cov
            FAIL["ok"] += 1                            # 겹쳐도 된다 — 판마다 깊이를 조금씩 달리한다 (add_screens). 겹침을 막으면 목표 구간의 1/4 만 놓였다
            rects.append([float(v) for v in r] + [s, e])     # 끝 둘 = 목표 구간 [s, e) (기록용)
            break
    # 미끼 0~2 개 — 궤적과 무관한 자리 ("가림막이 있다 = 물체가 뒤에 있다" 지름길 막기). 멈춘 자리 규칙은 같다
    n_dec = int(rng.choice(3, p=(0.6, 0.3, 0.1)))
    decoys = []
    if TIMED_SLIDE:
        # 미끼도 들어왔다 나간다 ("판이 들어오면 물체가 뒤에 있다" 지름길 막기) — 구간은 가림 구간과 같은 분포, 그 구간 동안 물체 투영과 안 닿는 자리
        for _ in range(n_dec):
            for _ in range(40):
                L = int(rng.integers(HIDE_LEN[0], HIDE_LEN[1] + 1))
                a = int(rng.integers(-L + 1, T))
                s, e = max(0, a), min(T, a + L)
                x, w = float(rng.uniform(-2.4, 2.4)), float(rng.uniform(0.6, 1.3))
                if rng.random() < 0.6:
                    z0, h = 0.0, float(rng.uniform(1.0, 2.6))
                else:
                    z0, h = float(rng.uniform(0.6, 2.4)), float(rng.uniform(0.6, 1.4))
                r = [x - w / 2, x + w / 2, z0, z0 + h]
                if any(not (r[1] < bx[:, 0].min() or r[0] > bx[:, 0].max() or r[3] < bx[:, 1].min() or r[2] > bx[:, 1].max())
                       for bx in boxes[s:e]):
                    continue
                decoys.append(r + [s, e])
                break
        return rects, decoys
    for _ in range(n_dec):
        allow_rest = rng.random() < P_COVER_REST
        for _ in range(40):
            x, w = float(rng.uniform(-2.4, 2.4)), float(rng.uniform(0.6, 1.3))
            if rng.random() < 0.6:
                z0, h = 0.0, float(rng.uniform(1.0, 2.6))
            else:
                z0, h = float(rng.uniform(0.6, 2.4)), float(rng.uniform(0.6, 1.4))
            r = [x - w / 2, x + w / 2, z0, z0 + h]
            touch_rest = not (r[1] < rest[:, 0].min() or r[0] > rest[:, 0].max() or r[3] < rest[:, 1].min() or r[2] > rest[:, 1].max())
            if touch_rest and not allow_rest:
                continue
            if any(not (r[1] + 0.1 < q[0] or r[0] - 0.1 > q[1] or r[3] + 0.1 < q[2] or r[2] - 0.1 > q[3]) for q in rects + decoys):
                continue
            if MAX_BLIND_RUN is not None:
                cov = np.array([inside(bx, r) for bx in boxes])
                if blind_run(blind | cov) > MAX_BLIND_RUN:
                    continue
                blind = blind | cov
            decoys.append(r)
            break
    return rects, decoys


def add_screens(rng, rects, name="timed", dy0=0.0):     # rect = (x0, x1, z0, z1[, s, e])
    out = []
    for k, r in enumerate(rects):
        anim = None
        if TIMED_SLIDE and len(r) == 6:              # 화면 밖 (가까운 쪽) 에서 D 장 동안 들어와 [s, e) 동안 서 있다 D 장 동안 나간다
            xc, w, s, e = (r[0] + r[1]) / 2, r[1] - r[0], int(r[4]), int(r[5])
            park = math.copysign(3.6 + w / 2, xc if xc != 0 else 1.0)   # 가림막 평면에서 화면 반폭 ≈ 2.9 m + 카메라 x ±0.6
            anim = [(PRE + s - TIMED_SLIDE, park), (PRE + s, xc), (PRE + e - 1, xc), (PRE + e - 1 + TIMED_SLIDE, park)]
        o = occluder(f"{name}{k}", (r[0] + r[1]) / 2, r[1] - r[0], r[3] - r[2], rng, z0=r[2], anim=anim)
        o.location.y = OCC_Y - dy0 - 0.03 * k        # 겹친 판이 같은 면에서 깜빡이지 않게 (판 두께 0.08)
        out.append(o)
    return out


def trim_len(lab, T, info):
    """마지막으로 움직인 장 + TAIL (화면 속도 px/장). 시나리오가 그 뒤 사건을 두면 (info min_T) 거기까지는 남긴다."""
    if info.get("no_trim"):
        return T
    sp = np.linalg.norm(lab["vel"][:T, 0], axis=-1)
    mv = np.where(sp > STILL_PX)[0]
    t = T if not len(mv) else int(mv[-1] + 1 + TAIL)
    return int(min(T, max(t, info.get("min_T", 0))))


def wall_hits(ev, lab, walls, radius):
    """정적 지형에 닿은 충돌 (impact 사건 중 물체 외접 상자가 지형 상자에 맞닿은 것) 의 장 목록."""
    out = []
    for e in ev:
        if e["kind"] != "impact":
            continue
        x, _, z = lab["pos3d"][min(e["frame"], len(lab["pos3d"]) - 1), 0]
        for q in walls:
            if q[0] - radius - 0.12 < x < q[1] + radius + 0.12 and q[2] - radius - 0.12 < z < q[3] + radius + 0.12:
                out.append(e["frame"])
                break
    return out


def check(lab, T, info, ev, radius):
    why = []
    p = lab["pos"][:T, 0]
    if np.linalg.norm(p.max(0) - p.min(0)) < 25:
        why.append("main barely moves")
    if (lab["inframe_frac"][:T, 0] < 0.5).mean() > 0.1:
        why.append("main out of frame")
    sp = np.linalg.norm(lab["vel"][:T, 0], axis=-1)
    if not info.get("skip_motion") and (sp > STILL_PX).mean() < MOTION_MIN:
        why.append(f"motion_frac {(sp > STILL_PX).mean():.2f}")
    if info.get("require_wall") and not wall_hits(ev, lab, info.get("walls", []), radius):
        why.append("no wall hit")
    if "final_in" in info:
        x0, x1, h = info["final_in"]
        x, _, z = lab["pos3d"][T - 1, 0]
        if not (x0 < x < x1 and z < h):
            why.append("not in container")
    return why


# ----------------------------------------------------------------------------- 한 편
def build(name, seed, attempt, ident, n_frames, res, engine, frames_dir):
    """v1 build 와 같고 시나리오 표만 이 모듈 것. v1 의 덤 가림막 (40 %) 은 없다 (미끼 가림막은 place_timed_screens 가 놓는다)."""
    rng = np.random.default_rng([seed, list(SCENARIOS).index(name), attempt])
    sc = B.reset()
    B.G = float(np.random.default_rng([seed, 5]).uniform(*B.G_RANGE))
    sc.gravity = (0, 0, -B.G)
    env = B.room(rng)
    env["g"] = B.G
    cat, desc, fn = SCENARIOS[name]
    objs, occ, info = fn(rng, n_frames, ident)
    assert len(objs) == 1, (name, len(objs))
    sc.frame_end = PRE + n_frames - 1
    rw = sc.rigidbody_world
    rw.point_cache.frame_start = 1
    rw.point_cache.frame_end = sc.frame_end
    B.setup_render(sc, engine, res, os.path.join(frames_dir, "f"))
    return rng, sc, env, objs, occ, info


def cut(lab, T):
    return {k: v[:T] for k, v in lab.items()}


def generate(name, k, seed, ident, out, n_frames=N_FRAMES, res=256, engine="cpu", max_attempts=MAX_ATTEMPTS, keep_frames=False):
    """흐름: 시뮬 + 라벨 (가림막 없이) → 자르기 → 시간 고른 가림막 (구간 수는 seed 로 고정) → 보임만 다시 재기 → 검사 → 렌더."""
    stem = os.path.join(out, name, f"{name}_{k:05d}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    frames_dir = stem + "_frames"
    n_ep = int(np.random.default_rng([seed, 8]).choice(len(P_EPISODES), p=P_EPISODES))
    FAIL.clear()
    why = []
    for attempt in range(max_attempts):
        rng, sc, env, objs, occ, info = build(name, seed, attempt, ident, n_frames, res, engine, frames_dir)
        o = objs[0]
        radius = float(max(o.dimensions)) / 2
        lab = B.simulate_and_label(sc, sc.camera, objs, n_frames, res)
        T = trim_len(lab, n_frames, info)
        unpred = [u for u in sorted(info.get("unpred", [])) if u < T]
        if T < MIN_T:
            why = [f"too short {T}"]
            continue
        lab = cut(lab, T)
        ev0 = B.events(lab)
        why = check(lab, T, info, ev0, radius)
        if why:
            continue
        rr = np.random.default_rng([seed, attempt, 4])
        ok = False
        timed = []
        for _ in range(SCREEN_TRIES):
            for s in timed:
                bpy.data.objects.remove(s, do_unlink=True)
            rects, decoys = place_timed_screens(rr, sc.camera, lab, radius, T, n_ep if EP_REF_T is None else int(round(n_ep * T / EP_REF_T)))
            timed = add_screens(rr, rects) + add_screens(rr, decoys, "decoy", dy0=0.03 * len(rects) + 0.015)
            if timed or occ:
                lab2 = cut(B.simulate_and_label(sc, sc.camera, objs, T, res), T)    # 가림막은 충돌이 없다 — 물리는 같고 보임만 바뀐다
            else:
                lab2 = lab
            ratio = lab2["vis_frac"][:, 0] / np.maximum(lab2["inframe_frac"][:, 0], 1e-6)
            hidden = (lab2["inframe_frac"][:, 0] > 0.25) & (ratio < HIDDEN)
            if hidden.mean() > MAX_HIDDEN_FRAC:
                why = [f"hidden_frac {hidden.mean():.2f}"]
                continue
            if not (lab2["vis_frac"][:, 0] > SEEN).any():
                why = ["never seen"]
                continue
            if MAX_BLIND_RUN is not None and blind_run(lab2["vis_frac"][:, 0] <= SEEN) > MAX_BLIND_RUN:
                why = [f"blind run {blind_run(lab2['vis_frac'][:, 0] <= SEEN)}"]     # 판이 일부만 덮어 생긴 안 보임 (기하 검사가 못 잡는 것)
                continue
            ok = True
            break
        if not ok:
            continue
        lab = lab2
        ev = B.events(lab)
        rows = B.event_rows(ev, lab, unpred, np.random.default_rng([seed, attempt, 3]), T, ROW_MIN_CTX, ROW_MIN_PRED)
        idents = np.array([ob["ident"] for ob in objs])
        impact = np.zeros((T, 1), np.float32)
        for e in ev:
            if e["kind"] == "impact":
                impact[e["frame"], 0] = max(impact[e["frame"], 0], e["dv"])
        present = np.ones((T, 1), bool)
        ratio2 = lab["vis_frac"] / np.maximum(lab["inframe_frac"], 1e-6)
        bb = lab["bbox"]
        lab.update(present=present, hidden=present & (lab["inframe_frac"] > 0.25) & (ratio2 < HIDDEN),
                   seen=np.maximum.accumulate(lab["vis_frac"] > SEEN, axis=0), impact=impact,
                   identity=idents.astype(np.int16), shape_id=(idents // NC).astype(np.int16), color_id=(idents % NC).astype(np.int16),
                   size=np.array([ob.dimensions.length / math.sqrt(3) for ob in objs], np.float32),
                   area=((bb[0, :, 2] - bb[0, :, 0]) * (bb[0, :, 3] - bb[0, :, 1])).astype(np.float32))
        T_render = T
        if SAVE_STRIDE > 1:
            lab, T, rows, ev, unpred = subsample(lab, T, rows, ev, unpred, SAVE_STRIDE)
        np.savez_compressed(stem + "_states.npz", **lab)
        os.makedirs(frames_dir, exist_ok=True)
        sc.frame_start = PRE
        sc.frame_end = PRE + T_render - 1
        sc.frame_step = SAVE_STRIDE                           # 렌더 장 PRE, PRE + s, … — 라벨의 남긴 장과 같다
        if os.environ.get("PHYS3D_NORENDER") != "1":          # 라벨 · 가림 분포만 볼 때 (렌더 생략)
            bpy.ops.render.render(animation=True)
            src = (["-framerate", str(B.FPS), "-start_number", str(PRE), "-i", os.path.join(frames_dir, "f%04d.png")] if SAVE_STRIDE == 1 else
                   ["-framerate", f"{B.FPS / SAVE_STRIDE:g}", "-pattern_type", "glob", "-i", os.path.join(frames_dir, "f*.png")])
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + src + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                            stem + ".mp4"], check=True)
        if not keep_frames:
            shutil.rmtree(frames_dir)
        h = lab["hidden"][:, 0].astype(int)
        metrics = dict(n_ep_target=n_ep, n_timed_screens=len(rects), hidden_frac=float(h.mean()), n_hidden_frames=int(h.sum()),
                       n_hidden_spans=int(((h[1:] - h[:-1]) == 1).sum() + h[0]), hidden_at_start=bool(h[0]), hidden_at_end=bool(h[-1]),
                       n_wall_hits=len(wall_hits(ev, lab, info.get("walls", []), radius)), screen_fail=dict(FAIL),
                       max_blind_run=longest_run(lab["vis_frac"][:, 0] <= SEEN))
        cat, desc, _ = SCENARIOS[name]
        meta = dict(scenario=name, category=cat, desc=desc, k=k, seed=seed, attempt=attempt, fps=B.FPS / SAVE_STRIDE, n_frames=T, res=res,
                    render_fps=B.FPS, save_stride=SAVE_STRIDE,
                    engine=engine, env=env, unpred=unpred, metrics=metrics,
                    objects=[dict(index=0, role="main", identity=int(o["ident"]), shape=SHAPES[o["ident"] // NC],
                                  color=COLORS[o["ident"] % NC][0], mass=o.rigid_body.mass, friction=o.rigid_body.friction,
                                  restitution=o.rigid_body.restitution)],
                    occluders=[s.name for s in occ + timed], timed_screens=rects, decoy_screens=decoys, walls=info.get("walls", []), events=ev, rows=rows)
        with open(stem + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        return dict(ok=True, mp4=stem + ".mp4", attempt=attempt, n_rows=len(rows), n_frames=T)
    return dict(ok=False, mp4=stem + ".mp4", why=why)


def subsample(lab, T, rows, ev, unpred, s):
    """12 fps 로 만든 라벨 · 행 · 사건을 s 장마다 1 장 (렌더 장 0, s, 2s, …) 으로 옮긴다 (SAVE_STRIDE).
    속도 · 가속도 · 각속도는 남긴 장에서 다시 미분한다 (단위 = 저장 장당). 충돌 세기는 가장 가까운 남긴 장으로.
    행: 경계 b → round(b / s) · end → ceil(end / s) (예측이 예측 못 하는 장을 넘지 않게) · 가려짐 행의 pred_min 은 다시 보이는 렌더 장을
    덮는 가장 작은 K. 옮긴 뒤 문맥 · 예측이 2 장 안 되거나 다시 보임이 경계 앞이 되거나 경계가 겹치면 뺀다."""
    keep = np.arange(0, T, s)
    Tn = len(keep)
    imp = lab["impact"]
    out = {k: (v[keep] if v.shape[:1] == (T,) else v) for k, v in lab.items()}
    g = lambda a: np.gradient(a, axis=0).astype(np.float32)
    out["vel"] = g(out["pos"].astype(np.float64))
    out["acc"] = g(out["vel"].astype(np.float64))
    out["omega"] = g(np.unwrap(out["angle"].astype(np.float64), axis=0))
    out["vel3d"] = g(out["pos3d"].astype(np.float64))
    ni = np.zeros((Tn,) + imp.shape[1:], np.float32)
    for t in np.nonzero(imp.max(axis=tuple(range(1, imp.ndim))))[0]:
        i = min(Tn - 1, int(round(t / s)))
        ni[i] = np.maximum(ni[i], imp[t])
    out["impact"] = ni
    up = lambda x: -(-int(x) // s)
    nrows, used = [], set()
    for r in rows:
        b = int(round(r["boundary"] / s))
        end = min(up(r["end"]), Tn)
        rr = dict(r, boundary=b, end=end, ctx_start=up(r["ctx_start"]), event_frame=int(round(r["event_frame"] / s)))
        if "ctx_gate" in r:
            rr["ctx_gate"] = up(r["ctx_gate"])
        if "pred_min" in r and r["event"] == "occlude":
            reap = r["boundary"] + r["pred_min"] - 1              # 렌더 장: 예측이 덮어야 하는 마지막 장 (가려짐 행은 다시 보이는 장)
            if reap < b * s:
                continue
            rr["pred_min"] = up(reap - b * s) + 1
            if rr["pred_min"] < 2 or b + rr["pred_min"] > end:      # 1 이면 경계에서 이미 보인다 — 가려진 채 끝나는 문맥이 아니다
                continue
        if b - rr["ctx_start"] < 2 or end - b < 2 or b in used:
            continue
        used.add(b)
        nrows.append(rr)
    nev = [dict(e, frame=int(round(e["frame"] / s))) for e in ev]
    return out, Tn, nrows, nev, sorted({up(u) for u in unpred})


SEED_RETRIES = 6                               # 실패하면 같은 정체로 seed 만 바꿔 다시 (정체 층화가 실패율과 얽히지 않게, 2D single 과 같다)


def generate_retry(name, k, seed, ident, out, *a, **kw):
    for r in range(SEED_RETRIES):
        res = generate(name, k, seed + r * 7_919_007, ident, out, *a, **kw)
        if res["ok"]:
            res["seed_retry"] = r
            return res
    res["seed_retry"] = SEED_RETRIES
    return res


def sample_points(o):
    """v1 sample_points 와 같은 표본 (면적 비례 200 점, rng 0) 인데 다각형을 정규화한 뒤 부채꼴로 직접 나눈다.
    v1 은 calc_loop_triangles 를 써서 sphere (UV 구) 의 삼각형 순서가 프로세스마다 달랐다 — 꼭짓점은 같은데 표본 점이 달라져
    vis_frac · bbox 가 실행마다 바뀌었다 (2026-10-19 확인: 같은 seed 두 번, sphere 만 다름. 나머지 다섯 모양은 같음)."""
    me = o.data
    V = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
    polys = []
    for pl in me.polygons:                         # 다각형 순서 · 시작 꼭짓점도 프로세스마다 달라서 (꼭짓점 번호는 같다) 둘 다 정규화한다
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


# 워커 루프 · 인자 처리 · 라벨은 v1 것을 그대로 쓴다 (simulate_and_label 이 B.sample_points 를 부르므로 이것으로 바뀐다)
B.sample_points = sample_points
B.SCENARIOS = SCENARIOS
B.generate = generate_retry

if __name__ == "__main__":
    B.main()
