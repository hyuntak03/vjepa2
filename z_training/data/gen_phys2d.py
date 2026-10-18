#!/usr/bin/env python3
"""2D 직관물리 학습셋 생성기 — 중력 · 영속 · 고체성 (2026-10-18, CPU 전용).

  python z_training/data/gen_phys2d.py --list                                   # 시나리오 목록
  python z_training/data/gen_phys2d.py --per-scenario 2 --out /tmp/phys2d_smp   # 시나리오마다 2 편 + 미리보기
  python z_training/data/gen_phys2d.py --per-scenario 500 --workers 64 --no-preview   # 본 생성

물리는 pymunk (chipmunk2d) 가 푼다. 렌더는 cv2 (2 배 슈퍼샘플 → INTER_AREA).
출력: <out>/clips/<scenario>/<clip>.mp4 · <clip>.json (변수 · 검사 지표) · <clip>_states.npz (물체별 위치 · 각도 · 보임 비율)
      <out>/index.csv   "<abs mp4> 0"   (configs/training/datasets.md 의 type: video_csv 형식)
      <out>/manifest.tsv  scenario · category · 지표
      <out>/_preview/<scenario>__<clip>.png (8 장 띠) · .gif

**학습셋이므로 가능(plausible) 영상만 만든다.** 위반 영상은 만들지 않는다.

누수 · 복사 지름길을 막는 장치
  * 운동 검사 — 프레임의 85 % 이상에서 무언가 (동적 물체 · 움직이는 가림막/컵) 가 0.6 px/frame 넘게 움직여야 통과.
    끝에 멈춘 구간은 15 % 까지만 허용. 미달이면 변수를 다시 뽑는다 (rejection)
  * 복사 기준선 지표 — 프레임 Δ (기본 8) 뒤와의 평균 |차이| 를 기록, 너무 작으면 버린다
  * 등속 외삽 잔차 — 주 물체에 대해 `pos_t + v_t·Δ` 의 오차 (px). 복사뿐 아니라 직선 외삽으로도 못 푼다는 것을 클립마다 남긴다
  * 시작 위상 무작위 — 시뮬을 앞에서 무작위 길이만큼 돌린 뒤 녹화 (첫 프레임이 늘 "놓는 순간" 이 아니다)
  * g · 크기 · 반발 · 마찰 · 색 · 배경 · 지면 높이 · 좌우 반전을 전부 클립마다 뽑는다
    — 특히 **g 를 클립마다 바꾼다** (문맥에서 가속도를 읽어야 미래를 맞힌다; 고정 g 이면 prior 로 외운다)
  * 영속 시나리오는 "완전히 가려진 프레임 수 ≥ 4" 와 "가려지기 전 · 다시 보인 뒤 둘 다 보임" 을 검사한다.
    다시 나오는 자리 · 시각이 가려진 동안의 동역학 (중력 포함) 으로만 정해진다
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import cv2
import numpy as np
import pymunk

W = H = 256          # 출력 해상도
SS = 2               # 슈퍼샘플
SHIFT = 3            # cv2 고정소수점 비트
FPS = 12             # 시뮬 시간 1 장 = 1/12 초 (낙하 한 번이 8~13 장)
VIDEO_FPS = 4        # mp4 에 적는 fps — 학습 로더 (fps 4, fstp 1) 가 **렌더한 장을 하나도 안 건너뛰고** 읽게 한다
N_FRAMES = 128       # 최대 128 장 — 마지막 운동 장에서 자른다 (가변 길이)
MIN_FRAMES = 64      # 창 64 장 (문맥 32 + 예측 32) 이 들어가야 한다
DV_THR = 30.0        # px/s (접촉 기록)
SUPPORT_TAGS = ("ledge", "shelf", "ramp")   # 떠나는 순간 = 받침 이탈 사건 (중력 낙하 시작)
DV_ROW = 60.0        # px/s — boundary 행으로 쓰는 충돌의 최소 속도 변화
MAX_IMPACT_ROWS = 2  # 영상당 충돌 행 상한 (가림 · 이탈 행은 전부 남긴다) — v1 은 6 이라 충돌이 행의 85 % 였다
IMPACT_PRIORITY = 3.0  # 뜻있는 충돌 (물체끼리 · 벽 · 선반 · 통 · 경사) 을 바닥 튕김보다 이만큼 더 뽑는다
P_EXTRA_SCREEN = 0.4   # 모든 시나리오에 궤적과 무관한 고정 가림막 (가림 사건을 늘리고, 가림막 위치 → 궤적 상관을 줄인다)
P_EXTRA_SHELF = 0.35   # 선반이 없는 시나리오에 기운 선반 (위에서 떨어진 물체가 착지 → 굴러 떨어짐 = 이탈 사건) — 이보다 작은 접촉 (구르기 · 미세 튕김) 은 사건으로 안 친다
SUBSTEPS = 10
MOVE_THR = 14.4      # px/s
P_DECOY = 0.35        # occ_pass 의 가림막을 궤적 밖에 두는 비율 (가림막 위치 → 미래 궤적 누수 차단)
P_SPAWN = 0.8        # 화면 밖에서 물체가 들어오는 클립 비율 (진자 · 용수철은 0.3)
P_DISTRACTOR = 0.25   # 시나리오 밖에서 붙이는 방해 물체 확률 (cup_cover 는 항상)


# ----------------------------------------------------------------------------- 색 · 배경
def _hsv(h, s, v):
    c = cv2.cvtColor(np.uint8([[[h * 179, s * 255, v * 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(x) for x in c)


def rand_color(rng, avoid=(), min_dist=90):
    for _ in range(200):
        c = _hsv(rng.random(), rng.uniform(0.45, 1.0), rng.uniform(0.35, 1.0))
        if all(np.linalg.norm(np.subtract(c, a)) > min_dist for a in avoid):
            return c
    return c


def make_background(rng):
    """정적 배경 (그라디언트 + 무늬 하나). 반환: (img SS 배율, 평균색)"""
    c1 = np.array(_hsv(rng.random(), rng.uniform(0, 0.35), rng.uniform(0.55, 0.95)), float)
    c2 = np.clip(c1 + rng.normal(0, 25, 3), 0, 255)
    t = np.linspace(0, 1, H * SS)[:, None, None]
    if rng.random() < 0.5:
        img = c1 * (1 - t) + c2 * t
        img = np.broadcast_to(img, (H * SS, W * SS, 3)).copy()
    else:
        img = c1 * (1 - t.transpose(1, 0, 2)) + c2 * t.transpose(1, 0, 2)
        img = np.broadcast_to(img, (H * SS, W * SS, 3)).copy()
    kind = rng.choice(["plain", "noise", "stripes", "grid", "dots"])
    if kind == "noise":
        n = rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32)
        n = cv2.resize(n, (W * SS, H * SS), interpolation=cv2.INTER_CUBIC)
        img += n[..., None] * rng.uniform(4, 12)
    elif kind == "stripes":
        p = rng.integers(10, 40) * SS
        ang = rng.uniform(0, math.pi)
        yy, xx = np.mgrid[0:H * SS, 0:W * SS]
        s = ((xx * math.cos(ang) + yy * math.sin(ang)) // p) % 2
        img += (s[..., None] - 0.5) * rng.uniform(8, 22)
    elif kind == "grid":
        p = rng.integers(16, 48) * SS
        yy, xx = np.mgrid[0:H * SS, 0:W * SS]
        g = ((xx % p) < SS) | ((yy % p) < SS)
        img -= g[..., None] * rng.uniform(10, 30)
    elif kind == "dots":
        for _ in range(int(rng.integers(20, 60))):
            x, y = rng.integers(0, W * SS, 2)
            cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 6) * SS),
                       tuple(float(v) for v in np.clip(c1 + rng.normal(0, 20, 3), 0, 255)), -1, cv2.LINE_AA)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, tuple(int(v) for v in img.reshape(-1, 3).mean(0)), str(kind)


def _pts(pts):
    return (np.asarray(pts, float) * SS * (1 << SHIFT)).round().astype(np.int32).reshape(-1, 1, 2)


def _p(x, y):
    return (int(round(x * SS * (1 << SHIFT))), int(round(y * SS * (1 << SHIFT))))


def shade(c, f):
    return tuple(int(np.clip(v * f, 0, 255)) for v in c)


def draw_poly(img, pts, color, edge=True):
    cv2.fillPoly(img, [_pts(pts)], color, cv2.LINE_AA, SHIFT)
    if edge:
        cv2.polylines(img, [_pts(pts)], True, shade(color, 0.6), SS, cv2.LINE_AA, SHIFT)


def draw_pattern_rect(img, x0, y0, x1, y1, color, pattern, rng_seed):
    """가림막 (불투명). 무늬로 배경과 구별된다."""
    draw_poly(img, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color)
    if pattern == "stripes":
        c2 = shade(color, 0.75)
        step = 10
        x = x0 + 4
        while x < x1 - 2:
            draw_poly(img, [(x, y0 + 2), (min(x + 4, x1 - 2), y0 + 2), (min(x + 4, x1 - 2), y1 - 2), (x, y1 - 2)], c2, edge=False)
            x += step
    elif pattern == "frame":
        draw_poly(img, [(x0 + 4, y0 + 4), (x1 - 4, y0 + 4), (x1 - 4, y1 - 4), (x0 + 4, y1 - 4)], shade(color, 1.15), edge=True)


# ----------------------------------------------------------------------------- 장면 객체
@dataclass
class Obj:
    body: pymunk.Body
    shape: pymunk.Shape
    kind: str
    size: float
    color: tuple
    role: str = "main"          # main | distractor | bob


@dataclass
class Mover:
    """경로를 따라 움직이는 것 — 가림막 (충돌 없음) 또는 kinematic 강체 (컵 · 판)."""
    path: list                  # [(t, x, y), ...] piecewise linear, x/y = 중심
    w: float
    h: float
    color: tuple
    kind: str = "screen"        # screen (가림막, 맨 앞) | cup (충돌, 앞면 불투명)
    pattern: str = "plain"
    body: pymunk.Body | None = None
    ease: bool = True           # False = 구간 안 등속 (가림막 왕복: 속도가 문맥에서 읽힌다)

    def pos(self, t):
        ts = [p[0] for p in self.path]
        if t <= ts[0]:
            return self.path[0][1:]
        if t >= ts[-1]:
            return self.path[-1][1:]
        i = int(np.searchsorted(ts, t)) - 1
        t0, x0, y0 = self.path[i]
        t1, x1, y1 = self.path[i + 1]
        a = (t - t0) / max(t1 - t0, 1e-9)
        if self.ease:
            a = 0.5 - 0.5 * math.cos(math.pi * a)  # 부드러운 출발 · 정지
        return (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a)


@dataclass
class Scene:
    space: pymunk.Space
    g: float
    floor_y: float
    statics: list = field(default_factory=list)     # [(pts, color)]
    objs: list = field(default_factory=list)
    movers: list = field(default_factory=list)
    strings: list = field(default_factory=list)     # [(anchor (x,y) | Obj, Obj, kind)] kind: rod | spring
    preroll: float = 0.0
    need_occlusion: list = field(default_factory=list)   # 영속 검사 대상 Obj
    params: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)             # 정적 shape → 이름 (wall · shelf · ramp …)
    require: list = field(default_factory=list)          # 사건 검사 — 시나리오의 사건이 실제로 일어났는가
    contacts: set = field(default_factory=set)           # 시뮬 중 기록 (obj_idx, tag) · (obj_i, obj_j)
    spawns: list = field(default_factory=list)           # [(t, Obj, pos, vel)] — 화면 밖에서 들어오는 물체 (그 전엔 공간 밖에 둔다)
    spawn_frames: list = field(default_factory=list)     # 들어온 장 — 예측 불가 시작 (예측 구간에 넣지 않는다)
    impacts: list = field(default_factory=list)          # [(frame, obj_idx, other, dv)] — frame 장 뒤 substep 에서 일어남 → 첫 사후 장 = frame + 1
    cur_frame: int = -1
    touching: dict = field(default_factory=dict)         # obj_idx → 지금 닿아 있는 받침 tag 집합
    leaves: list = field(default_factory=list)           # [(frame, obj_idx, tag)] — 받침 (ledge · shelf · ramp) 을 떠난 순간


def new_scene(rng, g_range=(350, 900)):
    sp = pymunk.Space()
    g = float(rng.uniform(*g_range))
    sp.gravity = (0, g)
    sp.iterations = 30
    floor_y = float(H - rng.uniform(14, 46))
    sc = Scene(space=sp, g=g, floor_y=floor_y)
    sc.floor_color = None
    return sc


def add_static_poly(sc, pts, color, friction=0.7, elasticity=0.6, tag=None):
    s = pymunk.Poly(sc.space.static_body, pts)
    s.friction = friction
    s.elasticity = elasticity
    sc.space.add(s)
    if tag:
        sc.tags[s] = tag
    sc.statics.append((pts, color))
    return s


def add_floor(sc, rng, color, elasticity=None):
    e = float(rng.uniform(0.3, 0.85)) if elasticity is None else elasticity
    add_static_poly(sc, [(-200, sc.floor_y), (W + 200, sc.floor_y), (W + 200, H + 200), (-200, H + 200)], color,
                    friction=float(rng.uniform(0.3, 1.0)), elasticity=e)


def add_bounds(sc, visible_walls=False, color=None, rng=None):
    """화면 밖 좌우 벽. 화면 밖이라 안 보이는 벽에 튕기는 장면은 생기지 않는다 (검사에서 화면 이탈 클립을 버린다)."""
    for x0, x1 in [(-300, -60), (W + 60, W + 300)]:
        s = pymunk.Poly(sc.space.static_body, [(x0, -400), (x1, -400), (x1, H + 100), (x0, H + 100)])
        s.elasticity = 0.5
        s.friction = 0.5
        sc.space.add(s)


def add_obj(sc, rng, kind, size, pos, vel=(0, 0), color=None, role="main", elasticity=None, friction=None,
            avoid=(), density=1.0, angvel=None, add=True):
    if color is None:
        color = rand_color(rng, avoid=avoid)
    if kind == "circle":
        m = density * math.pi * size ** 2 / 100
        b = pymunk.Body(m, pymunk.moment_for_circle(m, 0, size))
        s = pymunk.Circle(b, size)
    else:
        if kind == "box":
            a = size * rng.uniform(0.8, 1.25)
            bb = size * rng.uniform(0.8, 1.25)
            verts = [(-a, -bb), (a, -bb), (a, bb), (-a, bb)]
        elif kind == "tri":
            verts = [(size * math.cos(t), size * math.sin(t)) for t in np.linspace(-math.pi / 2, 1.5 * math.pi, 4)[:3]]
        else:  # pent
            verts = [(size * math.cos(t), size * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 6)[:5]]
        m = density * size ** 2 * 3 / 100
        b = pymunk.Body(m, pymunk.moment_for_poly(m, verts))
        s = pymunk.Poly(b, verts)
    b.position = pos
    b.velocity = vel
    b.angle = float(rng.uniform(0, 2 * math.pi))
    b.angular_velocity = float(rng.normal(0, 2)) if angvel is None else angvel
    s.elasticity = float(rng.uniform(0.3, 0.85)) if elasticity is None else elasticity
    s.friction = float(rng.uniform(0.3, 0.9)) if friction is None else friction
    if add:
        sc.space.add(b, s)
    o = Obj(b, s, kind, size, color, role)
    sc.objs.append(o)
    return o


def add_distractor(sc, rng, avoid, away_x=None):
    """방해 물체 — 잘 튀는 공. 주 물체와 다른 쪽에서 튄다 (CLAUDE.md §11: 물체 하나 = 화면 움직임 전부 상태를 깬다)."""
    side_left = (away_x is not None and away_x > W / 2) or (away_x is None and rng.random() < 0.5)
    x = float(rng.uniform(25, 70)) if side_left else float(rng.uniform(W - 70, W - 25))
    size = float(rng.uniform(5, 9))
    return add_obj(sc, rng, "circle", size, (x, rng.uniform(20, sc.floor_y - 80)), (rng.normal(0, 60), rng.uniform(-200, 0)),
                   avoid=avoid, role="distractor", elasticity=float(rng.uniform(0.85, 0.97)), friction=0.2)


def _overlaps(sc, x0, y0, x1, y1, pad=6):
    """직사각형이 기존 정적 지형 · 물체 시작 자리와 겹치는가 (바닥 · 옆벽 제외)"""
    for pts, _ in sc.statics:
        p = np.asarray(pts)
        if p[:, 1].min() >= sc.floor_y - 1 or p[:, 0].max() <= 20 or p[:, 0].min() >= W - 20:
            continue                                                  # 바닥 · 옆벽
        if not (p[:, 0].max() < x0 - pad or p[:, 0].min() > x1 + pad or p[:, 1].max() < y0 - pad or p[:, 1].min() > y1 + pad):
            return True
    for o in sc.objs:
        x, y = o.body.position
        if x0 - o.size - pad < x < x1 + o.size + pad and y0 - o.size - pad < y < y1 + o.size + pad:
            return True
    return False


def add_extras(sc, rng, bgc):
    """시나리오와 무관하게 붙는 장치 — 고정 가림막 · 기운 선반. 둘 다 자리를 무작위로 고른다 (궤적을 보지 않는다)."""
    x0w, x1w = getattr(sc, "wall_x", (0.0, float(W)))
    if rng.random() < P_EXTRA_SHELF and not any(t in ("shelf", "ledge") for t in sc.tags.values()):
        c = rand_color(rng, avoid=[bgc] + [col for _, col in sc.statics], min_dist=60)
        for _ in range(20):
            w = float(rng.uniform(50, 100))
            x = float(rng.uniform(x0w + 15, x1w - 15 - w))
            y = float(rng.uniform(70, sc.floor_y - 55))
            ang = float(rng.uniform(0.08, 0.3)) * (1 if rng.random() < 0.5 else -1)   # 기울어 있어야 굴러 떨어진다
            dx, dy = w * math.cos(ang), w * math.sin(ang)
            if _overlaps(sc, x, min(y, y + dy) - 10, x + dx, max(y, y + dy) + 18):
                continue
            add_static_poly(sc, [(x, y), (x + dx, y + dy), (x + dx, y + dy + 8), (x, y + 8)], c,
                            friction=float(rng.uniform(0.3, 0.8)), elasticity=float(rng.uniform(0.2, 0.6)), tag="shelf")
            sc.params["extra_shelf"] = True
            break
    if rng.random() < P_EXTRA_SCREEN:
        avoid = [bgc] + [o.color for o in sc.objs] + [m.color for m in sc.movers]
        for _ in range(20):
            w, h = float(rng.uniform(40, 90)), float(rng.uniform(40, 120))
            x = float(rng.uniform(x0w + 5, x1w - 5 - w))
            y = float(sc.floor_y + 2 - h) if rng.random() < 0.6 else float(rng.uniform(20, sc.floor_y - h - 20))
            if _overlaps(sc, x, y, x + w, y + h, pad=4):
                continue
            sc.movers.append(_occluder(rng, avoid, x + w / 2, y + h / 2, w, h))
            sc.params["extra_screen"] = True
            break


def add_spawns(sc, rng, avoid, n, t_range=(1.0, 9.5), x_range=None):
    """화면 밖에서 새 물체가 들어온다 (위에서 떨어지거나, 옆벽이 없으면 옆에서 던져짐). 장면이 길게 살아 있고 뒤쪽에 새 사건이 생긴다.
    들어오는 순간은 문맥으로 예측할 수 없으므로 spawn_frames 로 표시해 boundary 행의 예측 구간에서 뺀다."""
    x0, x1 = getattr(sc, "wall_x", (0.0, float(W)))
    walls = sc.params.get("walls", True)
    for t in sorted(rng.uniform(*t_range, n)):
        size = float(rng.uniform(6, 14))
        o = add_obj(sc, rng, rand_kind(rng), size, (-1000.0, -1000.0), avoid=avoid, role="spawn", add=False,
                    elasticity=float(rng.uniform(0.4, 0.9)))
        if walls or rng.random() < 0.5:
            lo, hi = x_range if x_range else (x0 + size + 4, x1 - size - 4)
            p, v = (float(rng.uniform(lo, hi)), -size - 2.0), (float(rng.normal(0, 50)), float(rng.uniform(0, 60)))
        else:
            left = rng.random() < 0.5
            p = (-size - 2.0 if left else W + size + 2.0, float(rng.uniform(40, sc.floor_y - 80)))
            v = (float(rng.uniform(120, 260)) * (1 if left else -1), -float(rng.uniform(80, 250)))
        sc.spawns.append((float(t), o, p, v))
        avoid = list(avoid) + [o.color]


def rand_kind(rng, roll=False):
    if roll:
        return "circle"
    return str(rng.choice(["circle", "circle", "box", "tri", "pent"]))


def add_cup_body(sc, mv, wall=4.0):
    """컵 (아래가 열린 U) — kinematic. 충돌은 세 벽, 그림은 불투명 앞면."""
    b = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    x, y = mv.pos(0)
    b.position = (x, y)
    sc.space.add(b)
    w, h = mv.w / 2, mv.h / 2
    for verts in ([(-w, -h), (w, -h), (w, -h + wall), (-w, -h + wall)],
                  [(-w, -h), (-w + wall, -h), (-w + wall, h), (-w, h)],
                  [(w - wall, -h), (w, -h), (w, h), (w - wall, h)]):
        s = pymunk.Poly(b, verts)
        s.friction = 0.6
        s.elasticity = 0.3
        sc.space.add(s)
    mv.body = b


# ----------------------------------------------------------------------------- 시나리오
SCENARIOS = {}


def scenario(name, category, desc):
    def deco(fn):
        SCENARIOS[name] = (category, desc, fn)
        return fn
    return deco


def _base(rng, bgc, walls=None):
    """walls: None = 확률 p_walls 로 보이는 옆벽 (방). 옆벽이 있으면 물체가 화면에 남아 운동이 길게 이어진다."""
    sc = new_scene(rng)
    fc = rand_color(rng, avoid=[bgc], min_dist=70)
    add_floor(sc, rng, fc)
    add_bounds(sc)
    if walls is None:
        walls = rng.random() < 0.9
    sc.wall_x = (0.0, float(W))
    if walls:
        wc = rand_color(rng, avoid=[bgc], min_dist=60) if rng.random() < 0.5 else fc
        e = float(rng.uniform(0.5, 0.9))
        t0, t1 = float(rng.uniform(4, 16)), float(rng.uniform(4, 16))
        add_static_poly(sc, [(-100, -300), (t0, -300), (t0, sc.floor_y + 1), (-100, sc.floor_y + 1)], wc, elasticity=e)
        add_static_poly(sc, [(W - t1, -300), (W + 100, -300), (W + 100, sc.floor_y + 1), (W - t1, sc.floor_y + 1)], wc, elasticity=e)
        sc.wall_x = (t0, W - t1)
    sc.params["walls"] = bool(walls)
    sc.floor_color = fc
    return sc, fc


# --- 중력 -------------------------------------------------------------------------
@scenario("drop_bounce", "gravity", "1~3 물체를 여러 높이에서 놓거나 던짐 — 낙하 · 튕김 (반발 계수 무작위)")
def sc_drop(rng, bgc):
    sc, fc = _base(rng, bgc)
    n = int(rng.integers(1, 4))
    used = [bgc, fc]
    for i in range(n):
        size = float(rng.uniform(8, 20))
        x = float(rng.uniform(30, W - 30))
        y = float(rng.uniform(15, sc.floor_y - 90))
        o = add_obj(sc, rng, rand_kind(rng), size, (x, y), (rng.normal(0, 40), rng.uniform(-150, 30)), avoid=used,
                    elasticity=float(rng.uniform(0.55, 0.9)))
        used.append(o.color)
    sc.preroll = float(rng.uniform(0, 0.3))
    return sc


@scenario("projectile", "gravity", "옆에서 던진 물체의 포물선 — 속도 · 각도 무작위, 바닥 · 벽에서 튕김")
def sc_proj(rng, bgc):
    sc, fc = _base(rng, bgc)
    if rng.random() < 0.5:   # 보이는 벽 하나
        wx = float(rng.uniform(W - 40, W - 15))
        add_static_poly(sc, [(wx, sc.floor_y - rng.uniform(80, 200)), (W + 60, sc.floor_y - 200), (W + 60, sc.floor_y), (wx, sc.floor_y)],
                        rand_color(rng, avoid=[bgc, fc]), elasticity=0.7)
    size = float(rng.uniform(7, 16))
    sp = float(rng.uniform(180, 420))
    ang = float(rng.uniform(math.radians(20), math.radians(70)))
    add_obj(sc, rng, rand_kind(rng), size, (rng.uniform(15, 50), sc.floor_y - size - rng.uniform(0, 60)),
            (sp * math.cos(ang), -sp * math.sin(ang)), avoid=[bgc, fc])
    sc.preroll = float(rng.uniform(0, 0.25))
    return sc


@scenario("pendulum", "gravity", "진자 1~2 개 (단진자 / 이중 진자) — 줄 길이 · 진폭 · 위상 무작위")
def sc_pend(rng, bgc):
    sc, fc = _base(rng, bgc)
    n = int(rng.integers(1, 3))
    used = [bgc, fc]
    for i in range(n):
        px = float(rng.uniform(50, W - 50)) if n == 1 else float((i + 0.5) * W / n + rng.uniform(-20, 20))
        py = float(rng.uniform(10, 50))
        L = float(rng.uniform(60, min(150, sc.floor_y - py - 30)))
        th = float(rng.uniform(0.5, 1.3)) * (1 if rng.random() < 0.5 else -1)
        size = float(rng.uniform(8, 15))
        bob = add_obj(sc, rng, "circle", size, (px + L * math.sin(th), py + L * math.cos(th)), avoid=used, role="bob", angvel=0)
        used.append(bob.color)
        bob.shape.filter = pymunk.ShapeFilter(group=1)
        j = pymunk.PinJoint(sc.space.static_body, bob.body, (px, py), (0, 0))
        sc.space.add(j)
        sc.strings.append(((px, py), bob, "rod"))
        if rng.random() < 0.35 and L > 90:   # 이중 진자 (혼돈)
            L2 = float(rng.uniform(30, 55))
            th2 = th + float(rng.uniform(-1.5, 1.5))
            x0, y0 = bob.body.position
            bob2 = add_obj(sc, rng, "circle", size * 0.8, (x0 + L2 * math.sin(th2), y0 + L2 * math.cos(th2)), avoid=used,
                           role="bob", angvel=0)
            bob2.shape.filter = pymunk.ShapeFilter(group=1)
            sc.space.add(pymunk.PinJoint(bob.body, bob2.body, (0, 0), (0, 0)))
            sc.strings.append((bob, bob2, "rod"))
    sc.preroll = float(rng.uniform(0, 2.0))
    return sc


@scenario("spring", "gravity", "용수철에 매단 물체 — 수직 진동 또는 탄성 진자 (흔들리며 늘었다 줄었다)")
def sc_spring(rng, bgc):
    sc, fc = _base(rng, bgc)
    n = int(rng.integers(1, 3))
    used = [bgc, fc]
    for i in range(n):
        px = float((i + 0.5) * W / n + rng.uniform(-25, 25))
        py = float(rng.uniform(8, 30))
        size = float(rng.uniform(9, 16))
        o = add_obj(sc, rng, rand_kind(rng), size, (px + rng.uniform(-40, 40), py + rng.uniform(60, 140)), avoid=used, role="bob",
                    angvel=0)
        used.append(o.color)
        k = float(rng.uniform(4, 15)) * o.body.mass
        rest = float(rng.uniform(40, 90))
        sp = pymunk.DampedSpring(sc.space.static_body, o.body, (px, py), (0, 0), rest, k, float(rng.uniform(0, 0.15)) * o.body.mass)
        sc.space.add(sp)
        if rng.random() < 0.5:   # 수직 홈에 묶기
            gj = pymunk.GrooveJoint(sc.space.static_body, o.body, (px, py), (px, H), (0, 0))
            sc.space.add(gj)
            o.body.position = (px, o.body.position.y)
            o.body.moment = float("inf")
        sc.strings.append(((px, py), o, "spring"))
    sc.preroll = float(rng.uniform(0, 1.5))
    return sc


@scenario("ramp_roll", "gravity", "경사면에서 굴러 내려옴 (등가속) — 경사 각 · 길이 · 방향 무작위, 끝에서 바닥 · 벽")
def sc_ramp(rng, bgc):
    sc, fc = _base(rng, bgc)
    rc = rand_color(rng, avoid=[bgc, fc])
    top = float(rng.uniform(70, 150))
    run = float(rng.uniform(110, 190))
    add_static_poly(sc, [(0, sc.floor_y - top), (run, sc.floor_y), (0, sc.floor_y)], rc, friction=0.9, tag="ramp")
    sc.require.append(("tag", "ramp"))
    if rng.random() < 0.5:
        add_static_poly(sc, [(W - rng.uniform(12, 30), sc.floor_y - rng.uniform(40, 100)), (W + 60, sc.floor_y - 100),
                             (W + 60, sc.floor_y), (W - 12, sc.floor_y)], rc, elasticity=0.6)
    size = float(rng.uniform(8, 16))
    f = float(rng.uniform(0.0, 0.5))      # 경사 위 출발 위치 비율
    x = f * run
    y = sc.floor_y - top + top * (x / run) - size * 1.1
    add_obj(sc, rng, "circle" if rng.random() < 0.8 else "box", size, (x + size, y), (rng.uniform(0, 60), 0),
            avoid=[bgc, fc, rc], friction=0.9)
    sc.preroll = float(rng.uniform(0, 0.2))
    return sc


@scenario("ledge_fall", "gravity", "선반 위를 굴러가다 끝에서 떨어짐 (중력 개시) — 선반 높이 · 속도 무작위")
def sc_ledge(rng, bgc):
    sc, fc = _base(rng, bgc)
    rc = rand_color(rng, avoid=[bgc, fc])
    sh_y = float(sc.floor_y - rng.uniform(60, 140))
    sh_x1 = float(rng.uniform(90, 170))
    add_static_poly(sc, [(-100, sh_y), (sh_x1, sh_y), (sh_x1, sh_y + 10), (-100, sh_y + 10)], rc, friction=0.3, tag="ledge")
    sc.require.append(("below", sh_y + 25))
    add_static_poly(sc, [(sh_x1 - 10, sh_y + 10), (sh_x1 - 4, sh_y + 10), (sh_x1 - 4, sc.floor_y), (sh_x1 - 10, sc.floor_y)],
                    shade(rc, 0.8))
    size = float(rng.uniform(7, 14))
    v = float(rng.uniform(60, 160))
    add_obj(sc, rng, "circle", size, (rng.uniform(10, sh_x1 - 60), sh_y - size - 0.5), (v, 0), avoid=[bgc, fc, rc],
            friction=0.3, angvel=v / size)
    sc.preroll = 0.0
    return sc


# --- 영속 -------------------------------------------------------------------------
def _occluder(rng, avoid, x, y, w, h, path=None):
    c = rand_color(rng, avoid=avoid, min_dist=80)
    return Mover(path=path or [(0, x, y)], w=w, h=h, color=c, kind="screen",
                 pattern=str(rng.choice(["plain", "stripes", "frame"])))


@scenario("occ_pass", "permanence", "움직이는 물체 (굴러감 · 포물선 · 낙하 · 튕김) 가 고정 가림막 뒤를 지나 다시 나옴 — 가려진 동안에도 중력 · 충돌이 작용. 가림막은 1 차 시뮬 궤적 위에 놓는다")
def sc_occ_pass(rng, bgc):
    sc, fc = _base(rng, bgc, walls=bool(rng.random() < 0.8))
    mode = str(rng.choice(["roll", "throw", "fall"]))
    size = float(rng.uniform(7, 13))
    if mode == "roll":
        v = float(rng.uniform(80, 170))
        o = add_obj(sc, rng, "circle", size, (rng.uniform(20, 60), sc.floor_y - size - 0.5), (v, 0), avoid=[bgc, fc],
                    friction=0.8, elasticity=0.5, angvel=v / size)
    elif mode == "throw":
        sp = float(rng.uniform(220, 360))
        ang = float(rng.uniform(math.radians(35), math.radians(65)))
        o = add_obj(sc, rng, rand_kind(rng), size, (rng.uniform(20, 50), sc.floor_y - size - 5),
                    (sp * math.cos(ang), -sp * math.sin(ang)), avoid=[bgc, fc])
    else:
        o = add_obj(sc, rng, rand_kind(rng), size, (rng.uniform(60, W - 60), rng.uniform(10, 40)),
                    (rng.normal(0, 40), rng.uniform(-60, 20)), avoid=[bgc, fc], elasticity=float(rng.uniform(0.6, 0.9)))
    sc.need_occlusion.append(o)
    sc.auto_occ = dict(hide_frames=(int(rng.integers(5, 16))), avoid=[bgc, fc, o.color], decoy=bool(rng.random() < P_DECOY))
    sc.params["decoy_occluder"] = sc.auto_occ["decoy"]
    sc.params["mode"] = mode
    sc.preroll = float(rng.uniform(0, 0.2))
    return sc


def place_auto_occluder(sc, rng, pos, n_valid):
    """1 차 궤적 pos (T, n, 2) 를 보고, 대상 물체가 hide_frames 동안 완전히 가려지도록 가림막을 놓는다."""
    i = sc.objs.index(sc.need_occlusion[0])
    size = sc.objs[i].size
    L = sc.auto_occ["hide_frames"]
    T = n_valid
    inside = (pos[:T, i, 0] > size) & (pos[:T, i, 0] < W - size) & (pos[:T, i, 1] > size)
    cands = [a for a in range(6, T - L - 8) if inside[a - 4:a + L + 6].all()]
    if not cands:
        return None
    for _ in range(20):
        a = int(rng.choice(cands))
        seg = pos[a:a + L, i]
        x0, y0 = seg.min(0) - size - 3
        x1, y1 = seg.max(0) + size + 3
        if x1 - x0 > 150 or y1 - y0 > 150:
            continue
        if y1 > sc.floor_y - 2 * size - 4:   # 바닥 근처면 바닥까지 내린다 (바닥에 선 가림막)
            y1 = sc.floor_y + 1
        rect = (x0, y0, x1, y1)
        # 앞뒤로 보이는지: 시작 전 · 끝난 뒤에 물체가 가림막과 안 겹치는 프레임이 있어야 한다
        def overlap(p):
            return (p[0] + size > x0) and (p[0] - size < x1) and (p[1] + size > y0) and (p[1] - size < y1)
        if overlap(pos[0, i]) or all(overlap(p) for p in pos[a + L:T, i]):
            continue
        if sc.auto_occ.get("decoy"):
            # 미끼 — 같은 크기 가림막을 궤적과 겹치지 않는 자리로 옮긴다. 가림막 위치가 미래 궤적을 알려 주는 상관을 끊는다
            w, h = x1 - x0, y1 - y0
            for _ in range(50):
                nx0 = float(rng.uniform(0, W - w))
                ny0 = float(sc.floor_y + 1 - h) if y1 >= sc.floor_y else float(rng.uniform(10, sc.floor_y - h - 5))
                r2 = (nx0, ny0, nx0 + w, ny0 + h)
                x0, y0, x1, y1 = r2
                if not any(overlap(p) for p in pos[:T, i]):
                    return r2
            return None
        return rect
    return None


@scenario("moving_screen", "permanence", "가림막이 옮겨 와 물체를 가렸다가 지나감 — 물체는 정지 · 진자 · 굴러감 중 하나")
def sc_screen(rng, bgc):
    sc, fc = _base(rng, bgc)
    mode = str(rng.choice(["rest", "pendulum", "roll"]))
    size = float(rng.uniform(8, 15))
    if mode == "rest":
        o = add_obj(sc, rng, rand_kind(rng), size, (rng.uniform(80, 180), sc.floor_y - size - 1), avoid=[bgc, fc],
                    angvel=0, elasticity=0.1)
        oy = sc.floor_y - 45 + 2       # 아래 끝이 바닥 속 2 px (틈 없이 가린다)
        oh = 90
    elif mode == "pendulum":
        px, py = float(rng.uniform(90, 170)), float(rng.uniform(15, 40))
        L = float(rng.uniform(70, 120))
        th = float(rng.uniform(0.4, 0.9))
        o = add_obj(sc, rng, "circle", size, (px + L * math.sin(th), py + L * math.cos(th)), avoid=[bgc, fc], role="bob", angvel=0)
        sc.space.add(pymunk.PinJoint(sc.space.static_body, o.body, (px, py), (0, 0)))
        sc.strings.append(((px, py), o, "rod"))
        oy = py + L
        oh = float(rng.uniform(60, 90))
    else:
        v = float(rng.uniform(40, 90))
        o = add_obj(sc, rng, "circle", size, (rng.uniform(40, 110), sc.floor_y - size - 0.5), (v, 0), avoid=[bgc, fc],
                    friction=0.8, elasticity=0.2, angvel=v / size)
        oy = sc.floor_y - 45 + 2       # 아래 끝이 바닥 속 2 px (틈 없이 가린다)
        oh = 90
    ow = float(rng.uniform(65, 110))
    # 가림막 왕복 1~3 회 — 구간 안 등속이라 속도가 문맥에서 읽힌다. 끝점은 화면 밖, 끝점 쉼은 짧게
    if mode == "rest":
        o.body.velocity = (0, 0)
    vert = rng.random() < 0.35            # 위에서 내려왔다 올라가는 가림막
    sp = float(rng.uniform(55, 130))
    t = float(rng.uniform(0.0, 0.6))
    if vert:
        cx = float(o.body.position.x) if mode != "pendulum" else px
        if mode == "roll":                # 굴러가는 물체가 도착할 자리를 겨냥한다
            cx = float(np.clip(cx + float(o.body.velocity.x) * (t + (oy + oh) / sp), 40, W - 40))
        a, b = (cx, -oh), (cx, oy)
    else:
        a, b = (-ow, oy), (W + ow, oy)
    path = [(0, *a), (t, *a)]
    cur, other = a, b
    for _ in range(int(rng.integers(2, 5))):
        t += float(np.hypot(other[0] - cur[0], other[1] - cur[1]) / sp)
        path.append((t, *other))
        t += float(rng.uniform(0.05, 0.25))
        path.append((t, *other))
        cur, other = other, cur
    mv = _occluder(rng, [bgc, fc, o.color], 0, oy, ow, oh, path=path)
    mv.ease = bool(vert)
    sc.movers.append(mv)
    sc.need_occlusion.append(o)
    sc.params["mode"] = mode
    sc.params["vertical_screen"] = bool(vert)
    return sc


@scenario("cup_cover", "permanence", "컵이 내려와 물체를 덮고 (옆으로 밀고 갈 때도 있음) 다시 들림 — 물체는 컵 안에 남는다")
def sc_cup(rng, bgc):
    sc, fc = _base(rng, bgc)
    size = float(rng.uniform(7, 12))
    x = float(rng.uniform(70, 180))
    o = add_obj(sc, rng, str(rng.choice(["circle", "box"])), size, (x, sc.floor_y - size - 0.2), (0, 0), avoid=[bgc, fc],
                angvel=0, elasticity=0.05, friction=0.8)
    o.body.angle = 0.0
    if o.kind == "box":
        o.body.position = (x, sc.floor_y - max(abs(v.y) for v in o.shape.get_vertices()) - 0.2)
    cw = float(rng.uniform(3.2, 4.5) * size)
    ch = float(rng.uniform(3.0, 4.0) * size)
    cy_dn = sc.floor_y - ch / 2 + 1.0       # 앞면 아래 끝이 바닥 윗면보다 1 px 아래 (틈 없이 덮는다)
    cy_up = float(rng.uniform(ch / 2 + 2, cy_dn - ch - 2 * size))
    x0 = x + float(rng.uniform(-60, 60))
    t1 = float(rng.uniform(0.1, 0.4))
    path = [(0, x0, cy_up), (t1 + 0.5, x, cy_up), (t1 + 0.9, x, cy_dn)]
    t = t1 + 0.9
    if rng.random() < 0.5:     # 덮은 채로 옆으로 밀고 감
        x2 = float(np.clip(x + rng.choice([-1, 1]) * rng.uniform(40, 80), 40, W - 40))
        path += [(t + 0.6, x2, cy_dn)]
        t += 0.6
        x = x2
    tp = t + float(rng.uniform(0.05, 0.15))
    path += [(tp, x, cy_dn), (tp + 0.4, x, cy_up)]
    t = tp + 0.4
    for _ in range(int(rng.integers(0, 2))):      # 한 번 더 덮었다 들기 (같은 자리 또는 옆으로 밀기)
        path += [(t + float(rng.uniform(0.1, 0.3)), x, cy_up), ]
        t = path[-1][0]
        path += [(t + 0.4, x, cy_dn)]
        t += 0.4
        if rng.random() < 0.5:
            x = float(np.clip(x + rng.choice([-1, 1]) * rng.uniform(30, 70), 40, W - 40))
            path += [(t + 0.6, x, cy_dn)]
            t += 0.6
        path += [(t + float(rng.uniform(0.05, 0.15)), x, cy_dn), ]
        t = path[-1][0]
        path += [(t + 0.4, x, cy_up)]
        t += 0.4
    path += [(t + 0.6, x + rng.uniform(-60, 60), cy_up)]
    mv = Mover(path=path, w=cw, h=ch, color=rand_color(rng, avoid=[bgc, fc, o.color]), kind="cup",
               pattern=str(rng.choice(["plain", "stripes", "frame"])))
    add_cup_body(sc, mv)
    sc.movers.append(mv)
    sc.need_occlusion.append(o)
    add_distractor(sc, rng, [bgc, fc, o.color, mv.color], away_x=x)
    return sc


@scenario("drop_behind", "permanence", "물체가 가림막 뒤로 떨어져 (튕기고) 멈춤, 그다음 가림막이 비켜나며 드러남")
def sc_drop_behind(rng, bgc):
    sc, fc = _base(rng, bgc)
    size = float(rng.uniform(7, 13))
    ox = float(rng.uniform(80, 180))
    ow = float(rng.uniform(5, 8) * size)
    oh = float(rng.uniform(50, 90))
    oy = sc.floor_y - oh / 2 + 1
    o = add_obj(sc, rng, rand_kind(rng), size, (ox + rng.uniform(-ow / 5, ow / 5), rng.uniform(15, 60)),
                (rng.normal(0, 15), rng.uniform(-40, 30)), avoid=[bgc, fc], elasticity=float(rng.uniform(0.3, 0.7)))
    t_go = float(rng.uniform(0.9, 1.4))
    dx = float(rng.choice([-1, 1]) * rng.uniform(1.0, 1.6) * ow)
    dy = float(-rng.uniform(0, 1) * oh * 1.4)
    path = [(0, ox, oy), (t_go, ox, oy), (t_go + 1.0, ox + dx, oy + dy)]
    mv = _occluder(rng, [bgc, fc, o.color], 0, 0, ow, oh, path=path)
    sc.movers.append(mv)
    sc.need_occlusion.append(o)
    sc.preroll = float(rng.uniform(0, 0.15))
    sc.require.append(("hidden_at", t_go))       # 가림막이 비키기 직전에 물체가 그 뒤에 있어야 한다
    return sc


@scenario("tunnel", "permanence", "굴러가는 물체가 터널 (바닥의 지붕 있는 가림) 로 들어가 반대편으로 나옴 — 경사 터널이면 안에서 가속")
def sc_tunnel(rng, bgc):
    sc, fc = _base(rng, bgc)
    size = float(rng.uniform(6, 11))
    tx0 = float(rng.uniform(70, 120))
    tw = float(rng.uniform(50, 100))
    th = float(size * 2 + rng.uniform(10, 30))
    slope = rng.random() < 0.4
    if slope:   # 터널 바닥이 경사 → 가려진 동안 가속
        rc = rand_color(rng, avoid=[bgc, fc])
        top = float(rng.uniform(20, 45))
        add_static_poly(sc, [(-50, sc.floor_y - top), (tx0, sc.floor_y - top), (tx0 + tw, sc.floor_y), (-50, sc.floor_y)], rc,
                        friction=0.9)
        y0 = sc.floor_y - top - size - 0.5
    else:
        y0 = sc.floor_y - size - 0.5
    v = float(rng.uniform(60, 140))
    o = add_obj(sc, rng, "circle", size, (rng.uniform(5, tx0 - 30), y0), (v, 0), avoid=[bgc, fc], friction=0.9,
                elasticity=0.2, angvel=v / size)
    mv = _occluder(rng, [bgc, fc, o.color], tx0 + tw / 2, sc.floor_y - th / 2 + 1 - (0 if not slope else 10), tw, th + (20 if slope else 0))
    sc.movers.append(mv)
    sc.need_occlusion.append(o)
    sc.params["slope"] = bool(slope)
    return sc


# --- 고체성 -----------------------------------------------------------------------
@scenario("shelf_land", "solidity", "선반 · 판 여러 개 위로 떨어져 그 위에 서거나 튕겨 내려감 — 통과하지 않는다")
def sc_shelf(rng, bgc):
    sc, fc = _base(rng, bgc)
    used = [bgc, fc]
    for i in range(int(rng.integers(1, 4))):
        c = rand_color(rng, avoid=used)
        used.append(c)
        x0 = float(rng.uniform(0, W - 70))
        w = float(rng.uniform(50, 110))
        y = float(rng.uniform(80, sc.floor_y - 30))
        ang = float(rng.uniform(-0.35, 0.35)) if rng.random() < 0.5 else 0.0
        dx, dy = w * math.cos(ang), w * math.sin(ang)
        add_static_poly(sc, [(x0, y), (x0 + dx, y + dy), (x0 + dx, y + dy + 8), (x0, y + 8)], c,
                        elasticity=float(rng.uniform(0.2, 0.7)), tag="shelf")
    sc.require.append(("tag", "shelf"))
    for i in range(int(rng.integers(1, 4))):
        o = add_obj(sc, rng, rand_kind(rng), float(rng.uniform(7, 14)), (rng.uniform(20, W - 20), rng.uniform(5, 50)),
                    (rng.normal(0, 30), rng.uniform(-80, 20)), avoid=used)
        used.append(o.color)
    sc.preroll = float(rng.uniform(0, 0.2))
    return sc


@scenario("wall_bounce", "solidity", "던지거나 굴린 물체가 벽에 부딪혀 되튕김 — 벽 높이 · 위치 · 반발 무작위")
def sc_wall(rng, bgc):
    sc, fc = _base(rng, bgc)
    wc = rand_color(rng, avoid=[bgc, fc])
    wx = float(rng.uniform(130, 210))
    wh = float(rng.uniform(70, 170))
    ww = float(rng.uniform(10, 25))
    add_static_poly(sc, [(wx, sc.floor_y - wh), (wx + ww, sc.floor_y - wh), (wx + ww, sc.floor_y), (wx, sc.floor_y)], wc,
                    elasticity=float(rng.uniform(0.4, 0.9)), tag="wall")
    sc.require.append(("tag", "wall"))
    size = float(rng.uniform(7, 14))
    if rng.random() < 0.5:
        v = float(rng.uniform(120, 260))
        add_obj(sc, rng, "circle", size, (rng.uniform(10, 60), sc.floor_y - size - 0.5), (v, 0), avoid=[bgc, fc, wc],
                angvel=v / size, elasticity=float(rng.uniform(0.5, 0.9)))
    else:
        sp = float(rng.uniform(200, 330))
        ang = float(rng.uniform(math.radians(15), math.radians(55)))
        add_obj(sc, rng, rand_kind(rng), size, (rng.uniform(10, 50), sc.floor_y - size - rng.uniform(5, 60)),
                (sp * math.cos(ang), -sp * math.sin(ang)), avoid=[bgc, fc, wc])
    sc.preroll = float(rng.uniform(0, 0.15))
    return sc


@scenario("collide", "solidity", "2~3 물체가 바닥 위 / 공중에서 서로 충돌 — 질량 (크기) 따라 운동량 교환")
def sc_collide(rng, bgc):
    sc, fc = _base(rng, bgc)
    used = [bgc, fc]
    n = int(rng.integers(2, 4))
    for i in range(n):
        size = float(rng.uniform(7, 18))
        left = i % 2 == 0
        x = float(rng.uniform(15, 80)) if left else float(rng.uniform(W - 80, W - 15))
        v = float(rng.uniform(70, 200)) * (1 if left else -1)
        air = rng.random() < 0.4
        y = sc.floor_y - size - (rng.uniform(40, 120) if air else 0.5)
        o = add_obj(sc, rng, "circle" if not air else rand_kind(rng), size, (x, y), (v, -rng.uniform(50, 200) if air else 0),
                    avoid=used, angvel=v / size, elasticity=float(rng.uniform(0.6, 0.95)))
        used.append(o.color)
    sc.preroll = float(rng.uniform(0, 0.2))
    sc.require.append(("objobj",))
    return sc


@scenario("stack_topple", "solidity", "쌓인 상자들 — 공이 날아와 치거나 불안정하게 쌓여 무너짐")
def sc_stack(rng, bgc):
    sc, fc = _base(rng, bgc)
    used = [bgc, fc]
    n = int(rng.integers(2, 5))
    s = float(rng.uniform(9, 14))
    x = float(rng.uniform(120, 190))
    y = sc.floor_y
    unstable = rng.random() < 0.5
    for i in range(n):
        c = rand_color(rng, avoid=used[-2:] + [bgc, fc], min_dist=60)
        used.append(c)
        m = s * s * 4 / 100
        b = pymunk.Body(m, pymunk.moment_for_box(m, (2 * s, 2 * s)))
        off = float(rng.uniform(0.35, 0.6) * s) if unstable else float(rng.uniform(-2, 2))
        b.position = (x + i * off, y - s - 0.1)
        sh = pymunk.Poly.create_box(b, (2 * s, 2 * s))
        sh.friction = 0.7
        sh.elasticity = 0.2
        sc.space.add(b, sh)
        sc.objs.append(Obj(b, sh, "box", s, c, "main"))
        y -= 2 * s + 0.2
    if not unstable or rng.random() < 0.5:
        size = float(rng.uniform(7, 12))
        sp = float(rng.uniform(200, 330))
        ang = float(rng.uniform(math.radians(10), math.radians(45)))
        add_obj(sc, rng, "circle", size, (rng.uniform(10, 40), sc.floor_y - size - rng.uniform(0, 40)),
                (sp * math.cos(ang), -sp * math.sin(ang)), avoid=used, density=3.0)
    sc.params["unstable"] = bool(unstable)
    sc.require.append(("moved", list(range(n)), s))     # 상자가 실제로 무너지거나 밀려야 한다
    return sc


@scenario("container", "solidity", "통 (U 자) 안으로 떨어지거나 던져진 물체 — 벽 안에서 튕기고 갇힌다")
def sc_container(rng, bgc):
    sc, fc = _base(rng, bgc)
    cc = rand_color(rng, avoid=[bgc, fc])
    cx = float(rng.uniform(90, 170))
    cw = float(rng.uniform(60, 110))
    ch = float(rng.uniform(40, 80))
    t = 6.0
    yb = sc.floor_y
    for pts in ([(cx - cw / 2, yb - ch), (cx - cw / 2 + t, yb - ch), (cx - cw / 2 + t, yb), (cx - cw / 2, yb)],
                [(cx + cw / 2 - t, yb - ch), (cx + cw / 2, yb - ch), (cx + cw / 2, yb), (cx + cw / 2 - t, yb)]):
        add_static_poly(sc, pts, cc, elasticity=0.6, tag="bin")
    sc.require.append(("final_in", cx - cw / 2 + t, cx + cw / 2 - t, yb - ch))
    sc.params["bin_x"] = (cx - cw / 2 + t + 8, cx + cw / 2 - t - 8)
    used = [bgc, fc, cc]
    for i in range(int(rng.integers(1, 3))):
        size = float(rng.uniform(6, 11))
        if rng.random() < 0.5:
            pos = (cx + rng.uniform(-cw / 3, cw / 3), rng.uniform(10, 60))
            vel = (rng.normal(0, 60), rng.uniform(-60, 20))
        else:
            pos = (rng.uniform(10, 40), sc.floor_y - rng.uniform(40, 100))
            vel = (rng.uniform(120, 200), -rng.uniform(150, 300))
        o = add_obj(sc, rng, rand_kind(rng), size, pos, vel, avoid=used, elasticity=float(rng.uniform(0.6, 0.9)))
        used.append(o.color)
    return sc


@scenario("occ_wall", "solidity", "고체성 × 영속 — 벽 아랫부분이 가림막에 가려 있고 (윗부분은 보임), 굴러간 물체가 가림막 뒤에서 벽에 튕겨 되돌아 나옴")
def sc_occ_wall(rng, bgc):
    sc, fc = _base(rng, bgc)
    wc = rand_color(rng, avoid=[bgc, fc])
    wx = float(rng.uniform(150, 200))
    wh = float(rng.uniform(80, 150))
    add_static_poly(sc, [(wx, sc.floor_y - wh), (wx + 12, sc.floor_y - wh), (wx + 12, sc.floor_y), (wx, sc.floor_y)], wc,
                    elasticity=float(rng.uniform(0.6, 0.9)), tag="wall")
    sc.require.append(("tag", "wall"))
    size = float(rng.uniform(7, 11))
    v = float(rng.uniform(110, 200))
    o = add_obj(sc, rng, "circle", size, (rng.uniform(10, 50), sc.floor_y - size - 0.5), (v, 0), avoid=[bgc, fc, wc],
                angvel=v / size, elasticity=float(rng.uniform(0.6, 0.9)), friction=0.5)
    ow = float(rng.uniform(55, 80))
    oh = float(rng.uniform(2.6, 4) * size)
    mv = _occluder(rng, [bgc, fc, wc, o.color], wx - ow / 2 + 14, sc.floor_y - oh / 2 + 1, ow, oh)
    sc.movers.append(mv)
    sc.need_occlusion.append(o)
    return sc


# ----------------------------------------------------------------------------- 렌더
def draw_obj(img, o: Obj):
    b = o.body
    if o.kind == "circle":
        x, y = b.position
        r = o.size
        cv2.circle(img, _p(x, y), int(round(r * SS * (1 << SHIFT))), o.color, -1, cv2.LINE_AA, SHIFT)
        # 반쪽 음영 — 회전이 보이게
        ang = math.degrees(b.angle)
        cv2.ellipse(img, _p(x, y), (int(round(r * SS * (1 << SHIFT))),) * 2, ang, 0, 180, shade(o.color, 0.62), -1,
                    cv2.LINE_AA, SHIFT)
        cv2.circle(img, _p(x, y), int(round(r * SS * (1 << SHIFT))), shade(o.color, 0.5), SS, cv2.LINE_AA, SHIFT)
    else:
        vs = [b.local_to_world(v) for v in o.shape.get_vertices()]
        draw_poly(img, [(v.x, v.y) for v in vs], o.color)
        c = b.local_to_world(o.shape.get_vertices()[0] * 0.55)
        cv2.circle(img, _p(c.x, c.y), int(o.size * 0.28 * SS * (1 << SHIFT)), shade(o.color, 0.55), -1, cv2.LINE_AA, SHIFT)


def obj_mask(o: Obj):
    m = np.zeros((H, W), np.uint8)
    b = o.body
    if o.kind == "circle":
        cv2.circle(m, (int(round(b.position.x)), int(round(b.position.y))), int(round(o.size)), 1, -1)
    else:
        vs = np.array([(v.x, v.y) for v in (b.local_to_world(v) for v in o.shape.get_vertices())]).round().astype(np.int32)
        cv2.fillPoly(m, [vs], 1)
    return m


def mover_rect(mv, t):
    x, y = (mv.body.position if mv.body is not None else mv.pos(t))
    return x - mv.w / 2, y - mv.h / 2, x + mv.w / 2, y + mv.h / 2


def draw_spring(img, a, b, color, coils=9, amp=5):
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    L = np.linalg.norm(d) + 1e-9
    u, n = d / L, np.array([-d[1], d[0]]) / L
    pts = [a, a + u * 6]
    for i in range(coils * 2):
        f = (6 + (L - 12) * (i + 0.5) / (coils * 2)) / L
        pts.append(a + d * f + n * amp * (1 if i % 2 == 0 else -1))
    pts += [b - u * 6, b]
    cv2.polylines(img, [_pts(pts)], False, color, SS, cv2.LINE_AA, SHIFT)


def render(sc: Scene, bg, t, string_color):
    img = bg.copy()
    for s in sc.strings:
        a, o, kind = s
        pa = a if isinstance(a, tuple) else tuple(a.body.position)
        pb = tuple(o.body.position)
        if kind == "rod":
            cv2.line(img, _p(*pa), _p(*pb), string_color, SS, cv2.LINE_AA, SHIFT)
        else:
            draw_spring(img, pa, pb, string_color)
        if isinstance(a, tuple):
            cv2.circle(img, _p(*pa), 3 * SS * (1 << SHIFT), string_color, -1, cv2.LINE_AA, SHIFT)
    for o in sc.objs:
        draw_obj(img, o)
    front = np.zeros((H, W), np.uint8)
    for mv in sc.movers:
        x0, y0, x1, y1 = mover_rect(mv, t)
        draw_pattern_rect(img, x0, y0, x1, y1, mv.color, mv.pattern, 0)
        cv2.rectangle(front, (int(round(x0)), int(round(y0))), (int(round(x1)) - 1, int(round(y1)) - 1), 1, -1)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), front


# ----------------------------------------------------------------------------- 실행 · 검사
def step_movers(sc, t, dt):
    for mv in sc.movers:
        if mv.body is not None:
            x, y = mv.pos(t + dt)
            px, py = mv.body.position
            mv.body.velocity = ((x - px) / dt, (y - py) / dt)


def record_contacts(sc):
    bidx = {o.body: i for i, o in enumerate(sc.objs)}
    for i, o in enumerate(sc.objs):
        now = set()

        def f(arb, i=i, o=o, now=now):
            for sh in arb.shapes:
                if sc.tags.get(sh) in SUPPORT_TAGS:
                    now.add(sc.tags[sh])
            if arb.is_first_contact:
                dv = float(arb.total_impulse.length / o.body.mass)
                if dv > DV_THR:
                    other = next((bidx.get(sh.body, sc.tags.get(sh, "static")) for sh in arb.shapes if sh.body is not o.body), "static")
                    sc.impacts.append((sc.cur_frame, i, other, dv))
            if o.role == "distractor":
                return
            for sh in arb.shapes:
                if sh in sc.tags:
                    sc.contacts.add((i, sc.tags[sh]))
                j = bidx.get(sh.body)
                if j is not None and j != i and sc.objs[j].role != "distractor":
                    sc.contacts.add((min(i, j), max(i, j)))
        o.body.each_arbiter(f)
        for tag in sc.touching.get(i, set()) - now:
            sc.leaves.append((sc.cur_frame, i, tag))
        sc.touching[i] = now


def check_require(sc, pos, vis):
    """시나리오 사건이 실제로 일어났는가. 반환: 실패 사유 목록"""
    why = []
    main = [i for i, o in enumerate(sc.objs) if o.role != "distractor"]
    for r in sc.require:
        if r[0] == "tag" and not any((i, r[1]) in sc.contacts for i in main):
            why.append(f"no_contact {r[1]}")
        elif r[0] == "objobj" and not any(isinstance(c[1], int) for c in sc.contacts):
            why.append("no_objobj")
        elif r[0] == "below" and not (pos[:, main, 1] > r[1]).any():
            why.append("no_fall")
        elif r[0] == "moved":
            d = np.linalg.norm(pos[:, r[1]] - pos[:1, r[1]], axis=-1).max()
            if d < r[2]:
                why.append("no_topple")
        elif r[0] == "final_in":
            x, y = pos[-1, main, 0], pos[-1, main, 1]
            if not ((x > r[1]) & (x < r[2]) & (y > r[3])).any():
                why.append("not_in_bin")
        elif r[0] == "hidden_at":
            f = int(round((r[1] - sc.preroll) * FPS)) - 1
            i = sc.objs.index(sc.need_occlusion[0])
            if f >= len(vis) or vis[max(f, 0), i] > 0.02:
                why.append("not_hidden_at")
    return why


def simulate(sc: Scene, bg, string_color, n_frames=N_FRAMES, render_frames=True):
    dt = 1.0 / (FPS * SUBSTEPS)
    t = 0.0
    for _ in range(int(sc.preroll / dt)):
        step_movers(sc, t, dt)
        sc.space.step(dt)
        t += dt
    t0 = t
    frames, pos, ang, vis, mover_speed = [], [], [], [], []
    for f in range(n_frames):
        sc.cur_frame = f
        if render_frames:
            img, front = render(sc, bg, t, string_color)
            frames.append(img)
        else:
            front = np.zeros((H, W), np.uint8)
        pos.append([tuple(o.body.position) for o in sc.objs])
        ang.append([o.body.angle for o in sc.objs])
        v = []
        for o in sc.objs:
            m = obj_mask(o)
            tot = m.sum()
            v.append(float((m & (1 - front)).sum() / tot) if tot > 0 else 0.0)
        vis.append(v)
        ms = 0.0
        for mv in sc.movers:
            a, b = np.array(mv.pos(t)), np.array(mv.pos(t + 1.0 / FPS))
            ms = max(ms, float(np.linalg.norm(b - a) * FPS))
        mover_speed.append(ms)
        for _ in range(SUBSTEPS):
            step_movers(sc, t, dt)
            while sc.spawns and sc.spawns[0][0] <= t:
                _, o, p, v = sc.spawns.pop(0)
                o.body.position, o.body.velocity = p, v
                sc.space.add(o.body, o.shape)
                sc.spawn_frames.append(f + 1)
            sc.space.step(dt)
            t += dt
            record_contacts(sc)
    return (np.stack(frames) if frames else None, np.array(pos, np.float32), np.array(ang, np.float32), np.array(vis, np.float32),
            np.array(mover_speed, np.float32), t0)


def check(sc, frames, pos, vis, mover_speed, delta=8):
    """통과 여부와 지표. 복사 · 직선 외삽으로 미래가 풀리지 않는지를 잰다."""
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * FPS
    vel[0] = vel[1]
    spd = np.linalg.norm(vel, axis=-1)                          # (T, n_obj)
    inframe = ((pos[..., 0] > -5) & (pos[..., 0] < W + 5) & (pos[..., 1] > -5) & (pos[..., 1] < H + 5))
    moving = ((spd * inframe).max(1) > MOVE_THR) | (mover_speed > MOVE_THR)
    motion_frac = float(moving.mean())
    # 끝에서 연속으로 멈춘 비율
    still_tail = 0
    for m in moving[::-1]:
        if m:
            break
        still_tail += 1
    still_tail_frac = still_tail / len(moving)
    f = frames.astype(np.float32)
    copy_l1 = float(np.abs(f[delta:] - f[:-delta]).mean())
    copy_l1_1 = float(np.abs(f[1:] - f[:-1]).mean())
    # 등속 외삽 잔차 — 화면 안 · 보이는 프레임만
    lin = pos[:-delta] + vel[:-delta] * (delta / FPS)
    err = np.linalg.norm(lin - pos[delta:], axis=-1)
    copy_err = np.linalg.norm(pos[:-delta] - pos[delta:], axis=-1)
    both = inframe[:-delta] & inframe[delta:]
    lin_err = float(np.nanmean(np.where(both, err, np.nan).max(1, initial=0)))
    pos_copy_err = float(np.nanmean(np.where(both, copy_err, 0).max(1)))
    acc = np.where(inframe[1:] & inframe[:-1], np.linalg.norm(np.diff(vel, axis=0), axis=-1) * FPS, np.nan)
    ok, why = True, []
    if motion_frac < 0.7:
        ok, why = False, why + [f"motion_frac {motion_frac:.2f}"]
    main_idx = [i for i, o in enumerate(sc.objs) if o.role in ("main", "bob")]
    occ_idx = [sc.objs.index(o) for o in sc.need_occlusion]
    for i in main_idx:
        if i in occ_idx:
            continue
        if inframe[:, i].mean() < 0.8:
            ok, why = False, why + [f"obj{i} offscreen {1 - inframe[:, i].mean():.2f}"]
    n_hidden = 0
    for i in occ_idx:
        hid = (vis[:, i] < 0.02) & inframe[:, i]
        n_hidden = int(hid.sum())
        idx = np.where(hid)[0]
        if n_hidden < 4:
            ok, why = False, why + [f"hidden {n_hidden}"]
            continue
        before = (vis[: idx[0], i] > 0.7).any()
        after = (vis[idx[-1]:, i] > 0.7).any()
        if not (before and after):
            ok, why = False, why + [f"vis before {before} after {after}"]
        if inframe[:, i].mean() < 0.9:
            ok, why = False, why + [f"occ obj offscreen"]
    # 복사 지름길 검사 (물체 중심) — Δ 프레임 동안 어떤 주 물체가 자기 크기의 절반 이상 움직인 창의 비율.
    # 화면 전체 L1 은 작은 물체에선 배경이 지배해 늘 작으므로 기준으로 쓰지 않고 기록만 한다.
    sizes = np.array([o.size for o in sc.objs], np.float32)
    big = (copy_err * both > 0.5 * sizes[None]).any(1)     # 방해 물체 포함 (화면 안의 무엇이든)
    mv_disp = np.array([max([np.linalg.norm(np.subtract(m.pos((j + delta) / FPS + sc.preroll), m.pos(j / FPS + sc.preroll)))
                             for m in sc.movers] + [0.0]) for j in range(len(copy_err))])
    big |= mv_disp > 8
    copy_frac = float(big.mean())
    if copy_frac < 0.6:
        ok, why = False, why + [f"copy_frac {copy_frac:.2f}"]
    req = check_require(sc, pos, vis)
    if req:
        ok, why = False, why + req
    metrics = dict(motion_frac=motion_frac, displaced_frac=copy_frac, still_tail_frac=still_tail_frac, copy_l1_delta=copy_l1, copy_l1_1=copy_l1_1,
                   delta=delta, pos_copy_err_px=pos_copy_err, linear_extrap_err_px=lin_err,
                   mean_accel=float(np.nanmean(acc)), n_hidden_frames=n_hidden)
    return ok, why, metrics


def flip_lr(frames, pos, ang):
    frames = frames[:, :, ::-1].copy()
    pos = pos.copy()
    pos[..., 0] = W - pos[..., 0]
    return frames, pos, -ang


def write_mp4(path, frames, crf=18):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(VIDEO_FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                          "-pix_fmt", "yuv420p", "-g", "16", path], stdin=subprocess.PIPE)
    p.stdin.write(frames.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def write_preview(path_png, path_gif, frames, title):
    idx = np.linspace(0, len(frames) - 1, 8).round().astype(int)
    strip = np.concatenate([np.pad(frames[i], ((0, 0), (0, 2), (0, 0)), constant_values=255) for i in idx], 1)
    top = np.full((18, strip.shape[1], 3), 255, np.uint8)
    cv2.putText(top, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    for k, i in enumerate(idx):
        cv2.putText(top, f"t={i}", (k * (W + 2) + W - 40, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(path_png, cv2.cvtColor(np.concatenate([top, strip], 0), cv2.COLOR_RGB2BGR))
    small = frames
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", "8", "-i", "-", "-vf", "split[a][b];[a]palettegen[p];[b][p]paletteuse", path_gif],
                         stdin=subprocess.PIPE)
    p.stdin.write(small.tobytes())
    p.stdin.close()
    p.wait()


def event_rows(sc, pos, vis, mover_speed, rng, min_ctx=4, min_pred=8):
    """사건 (충돌 · 가림) 마다 boundary 행 하나. 전환 구간 (사건 직전 장 + 직후 장) 이 문맥 마지막 몇 장에 들어가도록
    b = 사건 직후 장 + 1 + j (j ∈ {0,1,2}) — 가장 짧은 문맥 4 장에도 전환이 들어간다.
    예측 구간이 깨끗해야 한다: 예측 못 하는 일 (멈춰 있던 가림막 · 컵의 출발) 직전에서 끊는다 (end). 로더는 n_frames = end 로 K 를 거른다."""
    n = len(pos)
    inframe = (pos[..., 0] > -5) & (pos[..., 0] < W + 5) & (pos[..., 1] > -5) & (pos[..., 1] < H + 5)
    ev = []
    imp = [(f + 1, "impact", i, str(other), dv) for f, i, other, dv in sc.impacts
           if dv >= DV_ROW and 1 <= f + 1 < n and inframe[f + 1, i] and sc.objs[i].role != "distractor"]
    if len(imp) > MAX_IMPACT_ROWS:
        w = np.array([1.0 if e[3] == "static" else IMPACT_PRIORITY for e in imp])
        imp = [imp[j] for j in sorted(rng.choice(len(imp), MAX_IMPACT_ROWS, replace=False, p=w / w.sum()))]
    ev += imp
    # 받침 이탈 — 직후 4 장 안에 반폭 이상 실제로 떨어질 때만 (튕김으로 잠깐 뜨는 것 제외), 같은 물체 · 받침은 8 장 안 중복 제거
    last = {}
    for f, i, tag in sc.leaves:
        post = f + 1
        if not (1 <= post < n - 4 and inframe[post, i]):
            continue
        if pos[post + 4, i, 1] - pos[post, i, 1] < 0.5 * sc.objs[i].size:
            continue
        if post - last.get((i, tag), -99) < 8:
            continue
        last[(i, tag)] = post
        ev.append((post, "leave", i, tag, 0.0))
    for i in range(len(sc.objs)):
        for h in range(1, n):
            if vis[h, i] < 0.02 <= vis[h - 1, i] and inframe[h, i] and (vis[max(0, h - 8):h, i] > 0.7).any():
                ev.append((h, "occlude", i, "", 0.0))
    # 예측 못 하는 시작: 3 장 이상 멈춰 있던 가림막 · 컵이 움직이기 시작
    unpred = [u for u in range(3, n) if mover_speed[u] > MOVE_THR and (mover_speed[u - 3:u] <= MOVE_THR).all()] + list(sc.spawn_frames)
    sizes = np.array([o.size for o in sc.objs], np.float32)
    rows, used = [], set()
    for post, kind, i, other, dv in sorted(ev):
        b = post + 1 + int(rng.integers(0, 3))
        end = min([u for u in unpred if u >= b] + [n])
        if b < min_ctx or end - b < min_pred or b in used:
            continue
        # 예측 구간 안에서 실제로 움직여야 한다 (복사 정답 금지): 앞 8 장 사이 물체가 반폭 이상 또는 가림막이 8 px 이상
        e = min(end, b + 8) - 1
        d = np.linalg.norm(pos[e] - pos[b], axis=-1) * (inframe[b] & inframe[e])
        moving = (d > 0.5 * sizes).any() or abs(mover_speed[b:e + 1]).max() * (e - b) / FPS > 8
        if not moving:
            continue
        used.add(b)
        rows.append(dict(boundary=int(b), end=int(end), event=kind, event_frame=int(post), obj=int(i),
                         obj_role=sc.objs[i].role, other=other, dv=round(float(dv), 1)))
    return rows


def trim_len(sc, pos, mover_speed, tail=4):
    """마지막으로 화면 안의 무언가가 움직인 프레임 + tail 에서 자른다 (정지 화면 = 복사 정답이라 버린다)."""
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * FPS
    inframe = (pos[..., 0] > -5) & (pos[..., 0] < W + 5) & (pos[..., 1] > -5) & (pos[..., 1] < H + 5)
    mov = ((np.linalg.norm(vel, axis=-1) * inframe).max(1) > MOVE_THR) | (mover_speed > MOVE_THR)
    idx = np.where(mov)[0]
    return int(min(len(pos), (idx[-1] + 1 + tail) if len(idx) else 0))


def build(name, seed, attempt):
    _, _, fn = SCENARIOS[name]
    rng = np.random.default_rng([seed, attempt])
    bg, bgc, bg_kind = make_background(rng)
    sc = fn(rng, bgc)
    if name != "cup_cover" and rng.random() < P_DISTRACTOR:
        add_distractor(sc, rng, [bgc] + [o.color for o in sc.objs])
    add_extras(sc, rng, bgc)
    if rng.random() < (0.3 if name in ("pendulum", "spring") else P_SPAWN):
        xr = sc.params.get("bin_x")
        add_spawns(sc, rng, [bgc] + [o.color for o in sc.objs], int(rng.integers(1, 3)), x_range=xr)
    sc.params["n_spawn"] = len(sc.spawns)
    sc.params["n_distractor"] = sum(o.role == "distractor" for o in sc.objs)
    for pts, color in sc.statics:          # 정적 지형 (바닥 · 벽 · 선반 · 경사 · 통) 은 배경에 한 번 그린다
        draw_poly(bg, pts, color)
    return rng, sc, bg, bgc, bg_kind


def generate(name, seed, max_attempts=80):
    """검사를 통과할 때까지 변수를 다시 뽑는다. 반환: (ok, why, 결과 dict)"""
    why = []
    for attempt in range(max_attempts):
        rng, sc, bg, bgc, bg_kind = build(name, seed, attempt)
        string_color = shade(bgc, 0.35)
        if getattr(sc, "auto_occ", None):
            _, pos1, _, _, ms1, _ = simulate(sc, bg, string_color, render_frames=False)
            n1 = trim_len(sc, pos1, ms1)
            rect = place_auto_occluder(sc, np.random.default_rng([seed, attempt, 1]), pos1, n1)
            if rect is None:
                why = ["no occluder slot"]
                continue
            rng, sc, bg, bgc, bg_kind = build(name, seed, attempt)      # 같은 seed → 같은 물리
            x0, y0, x1, y1 = rect
            r2 = np.random.default_rng([seed, attempt, 2])
            sc.movers.append(_occluder(r2, sc.auto_occ["avoid"], (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
            if sc.auto_occ["decoy"]:
                sc.need_occlusion = []
        frames, pos, ang, vis, ms, t0 = simulate(sc, bg, string_color)
        n = trim_len(sc, pos, ms)
        if n < MIN_FRAMES:
            why = [f"too short {n}"]
            continue
        frames, pos, ang, vis, ms = frames[:n], pos[:n], ang[:n], vis[:n], ms[:n]
        ok, why, metrics = check(sc, frames, pos, vis, ms)
        rows = event_rows(sc, pos, vis, ms, np.random.default_rng([seed, attempt, 3]))
        if ok and not any(r["boundary"] >= 16 and r["end"] - r["boundary"] >= 16 for r in rows):
            ok, why = False, ["no event row C16K16"]
        if ok:
            metrics["n_rows"] = len(rows)
            return True, [], dict(rng=rng, sc=sc, bg_kind=bg_kind, frames=frames, pos=pos, ang=ang, vis=vis, t0=t0,
                                  attempt=attempt, metrics=metrics, rows=rows)
    return False, why, None


def make_clip(job):
    name, k, seed, out, preview = job
    category = SCENARIOS[name][0]
    ok, why, R = generate(name, seed)
    if not ok:
        return dict(name=name, k=k, ok=False, why=why)
    rng, sc, bg_kind, frames, pos, ang, vis, t0, attempt, metrics, rows = (R[x] for x in
        ["rng", "sc", "bg_kind", "frames", "pos", "ang", "vis", "t0", "attempt", "metrics", "rows"])
    flipped = bool(rng.random() < 0.5)
    if flipped:
        frames, pos, ang = flip_lr(frames, pos, ang)
    clip = f"{name}_{k:05d}"
    d = os.path.join(out, "clips", name)
    os.makedirs(d, exist_ok=True)
    mp4 = os.path.abspath(os.path.join(d, clip + ".mp4"))
    write_mp4(mp4, frames)
    meta = dict(clip=clip, scenario=name, category=category, seed=seed, attempt=attempt, sim_fps=FPS, video_fps=VIDEO_FPS,
                n_frames=len(frames), event_rows=rows,
                size=[W, H], g_px_s2=sc.g, floor_y=sc.floor_y, background=bg_kind, flipped_lr=flipped, preroll_s=t0,
                objects=[dict(kind=o.kind, size=o.size, role=o.role, color=o.color,
                              elasticity=o.shape.elasticity, friction=o.shape.friction) for o in sc.objs],
                occluders=[dict(kind=m.kind, w=m.w, h=m.h, n_waypoints=len(m.path)) for m in sc.movers],
                params=sc.params, metrics=metrics)
    with open(os.path.join(d, clip + ".json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(d, clip + "_states.npz"), pos=pos, angle=ang, visible=vis)
    if preview:
        pd = os.path.join(out, "_preview")
        os.makedirs(pd, exist_ok=True)
        title = (f"{name} [{category}] g={sc.g:.0f} copy_L1(d8)={metrics['copy_l1_delta']:.1f} "
                 f"lin_err={metrics['linear_extrap_err_px']:.1f}px hidden={metrics['n_hidden_frames']}")
        write_preview(os.path.join(pd, f"{clip}.png"), os.path.join(pd, f"{clip}.gif"), frames, title)
    return dict(name=name, k=k, ok=True, mp4=mp4, category=category, attempt=attempt, n_frames=len(frames), rows=rows, **metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/phys2d_samples")
    ap.add_argument("--per-scenario", type=int, default=2)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 32))
    ap.add_argument("--preview-n", type=int, default=2, help="시나리오마다 미리보기를 만들 편 수")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for n, (c, d, _) in SCENARIOS.items():
            print(f"{c:11s} {n:14s} {d}")
        return
    names = list(SCENARIOS) if a.scenarios == "all" else a.scenarios.split(",")
    jobs = [(n, k, a.seed * 1_000_003 + i * 10_007 + k, a.out, (not a.no_preview) and k < a.preview_n)
            for i, n in enumerate(names) for k in range(a.per_scenario)]
    os.makedirs(a.out, exist_ok=True)
    res = []
    import time
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(make_clip, j) for j in jobs]):
            r = fut.result()
            res.append(r)
            if not r["ok"]:
                print(f"FAIL {r['name']}_{r['k']:05d} {r['why']}", flush=True)
            if len(res) % 500 == 0 or len(res) == len(jobs):
                el = time.time() - t0
                print(f"[{len(res):,}/{len(jobs):,}] {el / 60:.1f} 분, 남은 {el / len(res) * (len(jobs) - len(res)) / 60:.1f} 분", flush=True)
    res.sort(key=lambda r: (r["name"], r["k"]))
    good = [r for r in res if r["ok"]]
    with open(os.path.join(a.out, "index.csv"), "w") as fh:
        fh.writelines(f"{r['mp4']} 0\n" for r in good)
    # var_window 표 (build_fps4_table.py 형식): path fps n_frames fstp n4 — mp4 가 4 fps 라 fstp 1, n4 = 장 수
    with open(os.path.join(a.out, "fps4_table.tsv"), "w") as fh:
        fh.writelines(f"{r['mp4']}\t{VIDEO_FPS}\t{r['n_frames']}\t1\t{r['n_frames']}\n" for r in good)
    # var_window boundary manifest: 예측 첫 장 = boundary_frame, n_frames = 깨끗한 예측 구간 끝 (로더가 n − b ≥ K 로 거른다)
    bcols = ["path", "n_frames", "boundary_frame", "video_frames", "event", "event_frame", "obj_role", "other", "dv", "scenario", "category"]
    nb = 0
    with open(os.path.join(a.out, "boundary.tsv"), "w") as fh:
        fh.write("\t".join(bcols) + "\n")
        for r in good:
            for w in r["rows"]:
                fh.write("\t".join(str(v) for v in [r["mp4"], w["end"], w["boundary"], r["n_frames"], w["event"], w["event_frame"],
                                                    w["obj_role"], w["other"], w["dv"], r["name"], r["category"]]) + "\n")
                nb += 1
    print(f"boundary 행 {nb:,} (영상당 {nb / max(1, len(good)):.1f})")
    cols = ["category", "name", "k", "n_frames", "attempt", "motion_frac", "displaced_frac", "still_tail_frac", "copy_l1_1", "copy_l1_delta",
            "pos_copy_err_px", "linear_extrap_err_px", "mean_accel", "n_hidden_frames", "n_rows"]
    with open(os.path.join(a.out, "manifest.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in good:
            fh.write("\t".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"\n{len(good)}/{len(res)} 통과 → {a.out}/index.csv")


if __name__ == "__main__":
    sys.exit(main())
