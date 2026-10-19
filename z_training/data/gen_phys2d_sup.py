#!/usr/bin/env python3
"""2D 직관물리 학습셋 + 프레임별 상태 라벨 — 중력 · 영속 · 고체성 (2026-10-18, CPU 전용).

vast 브랜치 `z_training/data/gen_phys2d.py` (174bf30, phys2d_v2) 를 포크했다. 바뀐 것:
  * **정체 라벨** — 물체 = 모양 6 × 색 8 = 48 정체 (이산). 한 영상 안 물체는 정체가 모두 다르다.
    시나리오마다 주 물체 정체를 48 칸에 층화 배정한다 (rejection 과 무관한 별도 RNG) → 모양 · 색이 시나리오 · 운동과 상관하지 않는다.
    모든 시나리오에서 모든 모양을 쓴다 (vast 판은 굴림 시나리오가 원을 강제해 주 물체 원 72 %).
    물체 색은 채도 높은 8 색 팔레트, 지형 · 가림막 · 배경은 채도 낮은 색 (팔레트와 거리 검사) — 색 칸이 늘 읽히게.
  * **프레임마다 모든 물체의 상태** (`<clip>_states.npz`) — 위치 · 속도 · 가속도 · 각도 · 각속도 · 보임 비율 · 화면 안 비율 ·
    존재 · 이미 본 적 있음 · amodal bbox · 충돌 크기. 가려진 동안에도 참값이 있다 (영속 supervision).
    인스턴스 세그 (보이는 픽셀, 가림막 = 255) 는 `<clip>_seg.npz`.
  * **여러 번 가림** — occ_pass 는 궤적 위에 가림막 1~3 개, osc_behind 는 진동 · 왕복하는 물체 앞 가림막 (2 번 이상 가려짐),
    모든 시나리오에 궤적과 무관한 고정 가림막 0~3 개 + 떠다니는 가림막 (12 %).
  * 시나리오 추가 — toss_up (위로 던져 꼭대기에서 되돌아옴) · osc_behind · sandbox (지형 · 장치 · 물체를 무작위 조합).
  * 물리를 한 번만 돌린다 (가림막은 충돌이 없어 물리에 영향이 없다 → 궤적을 보고 가림막을 놓은 뒤 다시 돌리지 않는다).
    검사 통과한 시도만 렌더한다.

  python z_training/data/gen_phys2d_sup.py --list
  python z_training/data/gen_phys2d_sup.py --per-scenario 2 --out /tmp/p2s --debug-overlay   # 미리보기 + 라벨 겹친 그림
  python z_training/data/gen_phys2d_sup.py --per-scenario 1000 --seed 1 --workers 40 --out <root>/train

좌표 (라벨 전부): 화면 256×256 px, 원점 왼쪽 위, y 아래로. 픽셀 j 는 [j, j+1) — 중심 j+0.5.
시간: 1 장 = 시뮬 1/12 초. 속도 px/장, 가속도 px/장², 각속도 rad/장 (중력 g px/s² = g/144 px/장²).
mp4 는 4 fps 로 적는다 (학습 로더 fps 4 · fstp 1 이 렌더한 장을 하나도 안 건너뛴다) — vast phys2d 와 같은 관례.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import cv2
import numpy as np
import pymunk

W = H = 256          # 출력 해상도
SS = 2               # 슈퍼샘플
SHIFT = 3            # cv2 고정소수점 비트
FPS = 12             # 시뮬 시간 1 장 = 1/12 초
VIDEO_FPS = 4        # mp4 에 적는 fps (vast phys2d 와 같은 관례)
N_FRAMES = 128
MIN_FRAMES = 64
SUBSTEPS = 10
MOVE_THR = 14.4      # px/s
DV_THR = 30.0        # px/s — 충돌 기록
DV_ROW = 60.0        # px/s — boundary 행으로 쓰는 충돌
MAX_IMPACT_ROWS = 2
IMPACT_PRIORITY = 3.0
SUPPORT_TAGS = ("ledge", "shelf", "ramp")
LOW_TAGS = ("floor", "side", "static")       # 바닥 · 방 옆벽 — 뜻있는 충돌이 아니다

P_WALLS = 0.75        # 보이는 방 옆벽
P_DECOY = 0.25        # occ_pass 가림막을 궤적 밖에 두는 비율
P_SPAWN = 0.5         # 화면 밖에서 물체가 들어오는 클립 (진동 시나리오는 0.3)
SPAWN_FIXED_ATTEMPTS = 100   # 이 시도 수까지는 spawn 여부를 clip 마다 고정
P_DISTRACTOR = 0.25
P_EXTRA_SHELF = 0.3
EXTRA_SCREENS_P = (0.45, 0.33, 0.16, 0.06)   # 궤적과 무관한 고정 가림막 0 · 1 · 2 · 3 개
P_DRIFT_SCREEN = 0.12                        # 처음부터 화면 안에 있다가 등속으로 떠다니는 가림막
HIDDEN = 0.02         # 화면 안 부분 중 보이는 비율 < 이것 = 완전히 가려짐
SEEN = 0.3            # 전체 면적 중 보이는 비율 > 이것 = 봤다

SHAPES = ("circle", "square", "rectangle", "triangle", "pentagon", "hexagon")
COLORS = (("red", (226, 40, 40)), ("orange", (244, 132, 22)), ("yellow", (236, 212, 32)), ("green", (40, 168, 62)),
          ("cyan", (32, 188, 204)), ("blue", (46, 86, 226)), ("purple", (136, 62, 200)), ("magenta", (224, 56, 166)))
NC = len(COLORS)
N_ID = len(SHAPES) * NC      # 정체 id = shape_id * 8 + color_id


# ----------------------------------------------------------------------------- 색 · 배경
def _hsv(h, s, v):
    c = cv2.cvtColor(np.uint8([[[h * 179, s * 255, v * 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(x) for x in c)


def _dist(a, b):
    return float(np.linalg.norm(np.subtract(a, b, dtype=float)))


def muted_color(rng, avoid=(), min_dist=55):
    """지형 · 가림막 색 — 채도 낮음, 물체 팔레트와 멀다 (물체 색 칸이 늘 읽히게)."""
    for _ in range(300):
        c = _hsv(rng.random(), rng.uniform(0.0, 0.32), rng.uniform(0.2, 0.85))
        if all(_dist(c, a) > min_dist for a in avoid) and all(_dist(c, p) > 75 for _, p in COLORS):
            return c
    return c


def make_background(rng):
    """정적 배경 (그라디언트 + 무늬). 반환: (img SS 배율, 평균색, 무늬 이름)"""
    for _ in range(100):
        c1 = np.array(_hsv(rng.random(), rng.uniform(0, 0.22), rng.uniform(0.3, 0.95)), float)
        if all(_dist(c1, p) > 85 for _, p in COLORS):
            break
    c2 = np.clip(c1 + rng.normal(0, 22, 3), 0, 255)
    t = np.linspace(0, 1, H * SS)[:, None, None]
    tt = t if rng.random() < 0.5 else t.transpose(1, 0, 2)
    img = np.broadcast_to(c1 * (1 - tt) + c2 * tt, (H * SS, W * SS, 3)).copy()
    kind = str(rng.choice(["plain", "noise", "stripes", "grid", "dots"]))
    if kind == "noise":
        n = rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32)
        n = cv2.resize(n, (W * SS, H * SS), interpolation=cv2.INTER_CUBIC)
        img += n[..., None] * rng.uniform(4, 12)
    elif kind == "stripes":
        p = rng.integers(10, 40) * SS
        ang = rng.uniform(0, math.pi)
        yy, xx = np.mgrid[0:H * SS, 0:W * SS]
        s = ((xx * math.cos(ang) + yy * math.sin(ang)) // p) % 2
        img += (s[..., None] - 0.5) * rng.uniform(8, 20)
    elif kind == "grid":
        p = rng.integers(16, 48) * SS
        yy, xx = np.mgrid[0:H * SS, 0:W * SS]
        g = ((xx % p) < SS) | ((yy % p) < SS)
        img -= g[..., None] * rng.uniform(10, 26)
    elif kind == "dots":
        for _ in range(int(rng.integers(20, 60))):
            x, y = rng.integers(0, W * SS, 2)
            cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 6) * SS),
                       tuple(float(v) for v in np.clip(c1 + rng.normal(0, 18, 3), 0, 255)), -1, cv2.LINE_AA)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, tuple(int(v) for v in img.reshape(-1, 3).mean(0)), kind


def shade(c, f):
    return tuple(int(np.clip(v * f, 0, 255)) for v in c)


# 렌더 (SS 배율, AA) · 라벨 (1 배율, AA 없음) 좌표 — 연속 좌표 x 의 픽셀 중심 규약을 맞춘다
def _pts(pts, s=SS):
    a = np.asarray(pts, float) * s - 0.5
    return (a * (1 << SHIFT)).round().astype(np.int32).reshape(-1, 1, 2)


def _p(x, y, s=SS):
    return (int(round((x * s - 0.5) * (1 << SHIFT))), int(round((y * s - 0.5) * (1 << SHIFT))))


def _r(r, s=SS):
    return int(round(r * s * (1 << SHIFT)))


def draw_poly(img, pts, color, edge=True):
    cv2.fillPoly(img, [_pts(pts)], color, cv2.LINE_AA, SHIFT)
    if edge:
        cv2.polylines(img, [_pts(pts)], True, shade(color, 0.6), SS, cv2.LINE_AA, SHIFT)


def draw_pattern_rect(img, x0, y0, x1, y1, color, pattern):
    """가림막 (불투명). 무늬로 배경과 구별된다."""
    draw_poly(img, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color)
    if pattern == "stripes":
        c2 = shade(color, 0.75)
        x = x0 + 4
        while x < x1 - 2:
            draw_poly(img, [(x, y0 + 2), (min(x + 4, x1 - 2), y0 + 2), (min(x + 4, x1 - 2), y1 - 2), (x, y1 - 2)], c2, edge=False)
            x += 10
    elif pattern == "frame":
        draw_poly(img, [(x0 + 4, y0 + 4), (x1 - 4, y0 + 4), (x1 - 4, y1 - 4), (x0 + 4, y1 - 4)], shade(color, 1.15))
    elif pattern == "checker":
        c2 = shade(color, 0.8)
        s = 12
        for j, y in enumerate(np.arange(y0 + 2, y1 - 2, s)):
            for i, x in enumerate(np.arange(x0 + 2, x1 - 2, s)):
                if (i + j) % 2 == 0:
                    draw_poly(img, [(x, y), (min(x + s, x1 - 2), y), (min(x + s, x1 - 2), min(y + s, y1 - 2)), (x, min(y + s, y1 - 2))],
                              c2, edge=False)


# ----------------------------------------------------------------------------- 정체 · 모양
def shape_verts(kind, size):
    """모양 6 종의 면적을 원 (π r²) 의 ±8 % 안으로 맞춘다 — 크기가 모양과 얽히지 않게 (CLAUDE.md §11)."""
    if kind == "square":
        a = 0.86 * size
        return [(-a, -a), (a, -a), (a, a), (-a, a)]
    if kind == "rectangle":
        a, b = 1.24 * size, 0.62 * size
        return [(-a, -b), (a, -b), (a, b), (-a, b)]
    n, R, off = {"triangle": (3, 1.45, -math.pi / 2), "pentagon": (5, 1.12, -math.pi / 2), "hexagon": (6, 1.05, 0.0)}[kind]
    return [(R * size * math.cos(off + 2 * math.pi * k / n), R * size * math.sin(off + 2 * math.pi * k / n)) for k in range(n)]
    # 각도 0 이면 여섯 모양 모두 아랫변이 평평하다 (받침 위에 놓을 때 각도 0 을 쓴다)


def poly_area(v):
    v = np.asarray(v)
    return float(0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], 1)) - np.dot(v[:, 1], np.roll(v[:, 0], 1))))


class IdPool:
    """영상 안 정체 배정. first = 층화된 주 물체 정체 — 처음 '모양 제약 없는' 요청에 준다. 그 밖은 남은 정체에서 무작위."""

    def __init__(self, rng, first=None, g=None):
        self.rng, self.first, self.used = rng, first, set()
        self.g = g          # 클립의 중력 — 시도 번호와 무관하게 정한다 (rejection 이 낮은 g 를 고르지 않게)
        if first is not None:
            self.used.add(first)

    def take(self, kind=None):
        if kind is None and self.first is not None:
            ident, self.first = self.first, None
            return ident
        sid = None if kind is None else SHAPES.index(kind)
        cands = [i for i in range(N_ID) if i not in self.used and (sid is None or i // NC == sid)]
        ident = int(self.rng.choice(cands))
        self.used.add(ident)
        return ident


# ----------------------------------------------------------------------------- 장면 객체
@dataclass
class Obj:
    body: pymunk.Body
    shape: pymunk.Shape
    kind: str
    size: float
    ident: int
    verts: list            # 몸체 좌표 꼭짓점 (원은 빈 목록)
    area: float
    bound: float           # 중심에서 가장 먼 점
    role: str = "main"     # main | bob | distractor | spawn | stack

    @property
    def color(self):
        return COLORS[self.ident % NC][1]


@dataclass
class Mover:
    """경로를 따라 움직이는 것 — 가림막 (충돌 없음) 또는 kinematic 강체 (컵)."""
    path: list                  # [(t, x, y), ...] piecewise linear, x/y = 중심
    w: float
    h: float
    color: tuple
    kind: str = "screen"        # screen | cup
    pattern: str = "plain"
    body: pymunk.Body | None = None
    ease: bool = True
    origin: str = "scenario"    # scenario | auto | extra | drift
    track: list = field(default_factory=list)   # 컵: 장마다 몸체 중심 (시뮬 중 기록)

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
            a = 0.5 - 0.5 * math.cos(math.pi * a)
        return (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a)


@dataclass
class Scene:
    space: pymunk.Space
    g: float
    floor_y: float
    ids: IdPool
    statics: list = field(default_factory=list)     # [(pts, color, tag)]
    objs: list = field(default_factory=list)
    movers: list = field(default_factory=list)
    strings: list = field(default_factory=list)     # [(anchor (x,y) | obj_idx, obj_idx, kind)]
    preroll: float = 0.0
    need_occlusion: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)
    require: list = field(default_factory=list)
    contacts: set = field(default_factory=set)
    spawns: list = field(default_factory=list)
    spawn_frames: list = field(default_factory=list)
    impacts: list = field(default_factory=list)     # [(frame, obj_idx, other, dv)] → 첫 사후 장 = frame + 1
    cur_frame: int = -1
    touching: dict = field(default_factory=dict)
    leaves: list = field(default_factory=list)
    auto_occ: dict | None = None
    wall_x: tuple = (0.0, float(W))
    floor_color: tuple = (0, 0, 0)


def new_scene(rng, ids, g_range=(350, 900)):
    sp = pymunk.Space()
    g = float(rng.uniform(*g_range))
    if ids.g is not None:
        g = float(ids.g)
    sp.gravity = (0, g)
    sp.iterations = 30
    return Scene(space=sp, g=g, floor_y=float(H - rng.uniform(14, 46)), ids=ids)


def add_static_poly(sc, pts, color, friction=0.7, elasticity=0.6, tag="static", draw=True):
    s = pymunk.Poly(sc.space.static_body, pts)
    s.friction = friction
    s.elasticity = elasticity
    sc.space.add(s)
    sc.tags[s] = tag
    if draw:
        sc.statics.append((pts, color, tag))
    return s


def add_obj(sc, rng, size, pos, vel=(0, 0), kind=None, role="main", elasticity=None, friction=None, density=1.0,
            angvel=None, angle=None, add=True, slide=False):
    """kind=None 이면 정체 풀이 모양을 정한다 (층화). slide=True: 원이 아니면 마찰을 낮춰 미끄러지게 (굴림 시나리오)."""
    ident = sc.ids.take(kind)
    kind = SHAPES[ident // NC]
    if kind == "circle":
        area = math.pi * size ** 2
        m = density * area / 100
        b = pymunk.Body(m, pymunk.moment_for_circle(m, 0, size))
        s = pymunk.Circle(b, size)
        verts, bound = [], size
    else:
        verts = shape_verts(kind, size)
        area = poly_area(verts)
        m = density * area / 100
        b = pymunk.Body(m, pymunk.moment_for_poly(m, verts))
        s = pymunk.Poly(b, verts)
        bound = max(math.hypot(*v) for v in verts)
    b.position = pos
    b.velocity = vel
    b.angle = float(rng.uniform(0, 2 * math.pi)) if angle is None else float(angle)
    b.angular_velocity = float(rng.normal(0, 2)) if angvel is None else float(angvel)
    s.elasticity = float(rng.uniform(0.3, 0.85)) if elasticity is None else float(elasticity)
    if friction is None:
        friction = float(rng.uniform(0.05, 0.3)) if (slide and kind != "circle") else float(rng.uniform(0.3, 0.9))
    elif slide and kind != "circle":
        friction = min(friction, float(rng.uniform(0.05, 0.3)))
    s.friction = float(friction)
    if add:
        sc.space.add(b, s)
    o = Obj(b, s, kind, float(size), ident, verts, area, bound, role)
    sc.objs.append(o)
    return o


def low_offset(o, angle=None):
    """중심에서 가장 아래 점까지 (y 아래 +)."""
    if o.kind == "circle":
        return o.size
    a = o.body.angle if angle is None else angle
    return max(pymunk.Vec2d(*v).rotated(a).y for v in o.verts)


def place_on(o, x, surface_y, gap=0.3):
    o.body.position = (float(x), float(surface_y - low_offset(o) - gap))


def roll_angvel(o, v):
    return v / o.size if o.kind == "circle" else 0.0


def add_distractor(sc, rng, away_x=None):
    """방해 물체 — 잘 튀는 물체. 주 물체와 다른 쪽에서 튄다 (CLAUDE.md §11)."""
    side_left = (away_x is not None and away_x > W / 2) or (away_x is None and rng.random() < 0.5)
    x = float(rng.uniform(25, 70)) if side_left else float(rng.uniform(W - 70, W - 25))
    return add_obj(sc, rng, float(rng.uniform(6, 10)), (x, rng.uniform(20, sc.floor_y - 80)), (rng.normal(0, 60), rng.uniform(-200, 0)),
                   role="distractor", elasticity=float(rng.uniform(0.85, 0.97)), friction=0.2)


def _overlaps(sc, x0, y0, x1, y1, pad=6, movers=False):
    """직사각형이 기존 정적 지형 · 물체 시작 자리 (· 가림막) 와 겹치는가 (바닥 · 방 옆벽 제외)"""
    for pts, _, tag in sc.statics:
        if tag in ("floor", "side"):
            continue
        p = np.asarray(pts)
        if not (p[:, 0].max() < x0 - pad or p[:, 0].min() > x1 + pad or p[:, 1].max() < y0 - pad or p[:, 1].min() > y1 + pad):
            return True
    for o in sc.objs:
        if o.body.space is None:
            continue
        x, y = o.body.position
        if x0 - o.bound - pad < x < x1 + o.bound + pad and y0 - o.bound - pad < y < y1 + o.bound + pad:
            return True
    if movers:
        for mv in sc.movers:
            mx, my = mv.pos(0)
            if not (mx + mv.w / 2 < x0 - pad or mx - mv.w / 2 > x1 + pad or my + mv.h / 2 < y0 - pad or my - mv.h / 2 > y1 + pad):
                return True
    return False


def _screen(rng, sc, x, y, w, h, path=None, origin="scenario"):
    avoid = [sc.params["bg_color"], sc.floor_color] + [c for _, c, _ in sc.statics[-4:]] + [m.color for m in sc.movers]
    return Mover(path=path or [(0, x, y)], w=w, h=h, color=muted_color(rng, avoid=avoid, min_dist=50), kind="screen",
                 pattern=str(rng.choice(["plain", "stripes", "frame", "checker"])), origin=origin)


def add_extras(sc, rng):
    """시나리오와 무관하게 붙는 장치 — 기운 선반 · 고정 가림막 0~3 개 · 떠다니는 가림막. 자리는 궤적을 보지 않고 고른다."""
    x0w, x1w = sc.wall_x
    if rng.random() < P_EXTRA_SHELF and not any(t in ("shelf", "ledge") for t in sc.tags.values()):
        for _ in range(20):
            w = float(rng.uniform(50, 100))
            x = float(rng.uniform(x0w + 15, x1w - 15 - w))
            y = float(rng.uniform(70, sc.floor_y - 55))
            ang = float(rng.uniform(0.08, 0.3)) * (1 if rng.random() < 0.5 else -1)
            dx, dy = w * math.cos(ang), w * math.sin(ang)
            if _overlaps(sc, x, min(y, y + dy) - 10, x + dx, max(y, y + dy) + 18):
                continue
            add_static_poly(sc, [(x, y), (x + dx, y + dy), (x + dx, y + dy + 8), (x, y + 8)], muted_color(rng, [sc.params["bg_color"]]),
                            friction=float(rng.uniform(0.3, 0.8)), elasticity=float(rng.uniform(0.2, 0.6)), tag="shelf")
            sc.params["extra_shelf"] = True
            break
    n_extra = int(rng.choice(4, p=EXTRA_SCREENS_P))
    placed = 0
    for _ in range(n_extra):
        for _ in range(25):
            w, h = float(rng.uniform(30, 85)), float(rng.uniform(35, 120))
            x = float(rng.uniform(x0w + 5, x1w - 5 - w))
            y = float(sc.floor_y + 2 - h) if rng.random() < 0.55 else float(rng.uniform(15, sc.floor_y - h - 15))
            if _overlaps(sc, x, y, x + w, y + h, pad=4, movers=True):
                continue
            sc.movers.append(_screen(rng, sc, x + w / 2, y + h / 2, w, h, origin="extra"))
            placed += 1
            break
    sc.params["n_extra_screens"] = placed
    if rng.random() < P_DRIFT_SCREEN:
        # 처음부터 화면 안에 있는 가림막이 등속으로 떠다닌다 (예측 못 하는 등장이 없다)
        w, h = float(rng.uniform(30, 70)), float(rng.uniform(40, 110))
        y = float(sc.floor_y + 2 - h / 2) if rng.random() < 0.5 else float(rng.uniform(15 + h / 2, sc.floor_y - h / 2 - 10))
        x = float(rng.uniform(w, W - w))
        v = float(rng.uniform(15, 50)) * (1 if rng.random() < 0.5 else -1)
        mv = _screen(rng, sc, x, y, w, h, path=[(0, x, y), (40.0, x + v * 40.0, y)], origin="drift")
        mv.ease = False
        sc.movers.append(mv)
        sc.params["drift_screen"] = True


def add_spawns(sc, rng, n, t_range=(1.0, 9.5), x_range=None):
    """화면 밖에서 새 물체가 들어온다. 들어오는 순간은 문맥으로 예측할 수 없으므로 spawn_frames 로 표시한다."""
    x0, x1 = sc.wall_x
    walls = sc.params.get("walls", True)
    for t in sorted(rng.uniform(*t_range, n)):
        size = float(rng.uniform(8, 14))
        o = add_obj(sc, rng, size, (-1000.0, -1000.0), role="spawn", add=False, elasticity=float(rng.uniform(0.4, 0.9)))
        if walls or rng.random() < 0.5:
            lo, hi = x_range if x_range else (x0 + o.bound + 4, x1 - o.bound - 4)
            p, v = (float(rng.uniform(lo, hi)), -o.bound - 2.0), (float(rng.normal(0, 50)), float(rng.uniform(0, 60)))
        else:
            left = rng.random() < 0.5
            p = (-o.bound - 2.0 if left else W + o.bound + 2.0, float(rng.uniform(40, sc.floor_y - 80)))
            v = (float(rng.uniform(120, 260)) * (1 if left else -1), -float(rng.uniform(80, 250)))
        sc.spawns.append((float(t), o, p, v))


def add_cup_body(sc, mv, wall=4.0):
    """컵 (아래가 열린 U) — kinematic. 충돌은 세 벽, 그림은 불투명 앞면."""
    b = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    b.position = mv.pos(0)
    sc.space.add(b)
    w, h = mv.w / 2, mv.h / 2
    for verts in ([(-w, -h), (w, -h), (w, -h + wall), (-w, -h + wall)],
                  [(-w, -h), (-w + wall, -h), (-w + wall, h), (-w, h)],
                  [(w - wall, -h), (w, -h), (w, h), (w - wall, h)]):
        s = pymunk.Poly(b, verts)
        s.friction = 0.6
        s.elasticity = 0.3
        sc.space.add(s)
        sc.tags[s] = "cup"
    mv.body = b


def add_pendulum(sc, rng, px, py, L, th, size, double=False):
    bob = add_obj(sc, rng, size, (px + L * math.sin(th), py + L * math.cos(th)), role="bob", angvel=0)
    bob.shape.filter = pymunk.ShapeFilter(group=1)
    sc.space.add(pymunk.PinJoint(sc.space.static_body, bob.body, (px, py), (0, 0)))
    sc.strings.append(((px, py), len(sc.objs) - 1, "rod"))
    if double:
        L2 = float(rng.uniform(30, 55))
        th2 = th + float(rng.uniform(-1.5, 1.5))
        x0, y0 = bob.body.position
        bob2 = add_obj(sc, rng, size * 0.85, (x0 + L2 * math.sin(th2), y0 + L2 * math.cos(th2)), role="bob", angvel=0)
        bob2.shape.filter = pymunk.ShapeFilter(group=1)
        sc.space.add(pymunk.PinJoint(bob.body, bob2.body, (0, 0), (0, 0)))
        sc.strings.append((len(sc.objs) - 2, len(sc.objs) - 1, "rod"))
    return bob


def add_spring(sc, rng, px, py, size, groove=None):
    o = add_obj(sc, rng, size, (px + rng.uniform(-40, 40), py + rng.uniform(60, 140)), role="bob", angvel=0)
    k = float(rng.uniform(4, 15)) * o.body.mass
    rest = float(rng.uniform(40, 90))
    sc.space.add(pymunk.DampedSpring(sc.space.static_body, o.body, (px, py), (0, 0), rest, k, float(rng.uniform(0, 0.15)) * o.body.mass))
    if groove if groove is not None else rng.random() < 0.5:   # 수직 홈에 묶기
        sc.space.add(pymunk.GrooveJoint(sc.space.static_body, o.body, (px, py), (px, H), (0, 0)))
        o.body.position = (px, o.body.position.y)
        o.body.moment = float("inf")
    sc.strings.append(((px, py), len(sc.objs) - 1, "spring"))
    return o


# ----------------------------------------------------------------------------- 시나리오
SCENARIOS = {}


def scenario(name, category, desc):
    def deco(fn):
        SCENARIOS[name] = (category, desc, fn)
        return fn
    return deco


def _base(rng, bgc, ids, walls=None):
    sc = new_scene(rng, ids)
    sc.params["bg_color"] = bgc
    fc = muted_color(rng, avoid=[bgc], min_dist=60)
    sc.floor_color = fc
    add_static_poly(sc, [(-200, sc.floor_y), (W + 200, sc.floor_y), (W + 200, H + 200), (-200, H + 200)], fc,
                    friction=float(rng.uniform(0.3, 1.0)), elasticity=float(rng.uniform(0.3, 0.85)), tag="floor")
    for x0, x1 in [(-300, -60), (W + 60, W + 300)]:          # 화면 밖 벽 (안 보인다)
        add_static_poly(sc, [(x0, -400), (x1, -400), (x1, H + 100), (x0, H + 100)], None, 0.5, 0.5, tag="side", draw=False)
    if walls is None:
        walls = rng.random() < P_WALLS
    if walls:
        wc = muted_color(rng, avoid=[bgc], min_dist=55) if rng.random() < 0.5 else fc
        e = float(rng.uniform(0.5, 0.9))
        t0, t1 = float(rng.uniform(4, 16)), float(rng.uniform(4, 16))
        add_static_poly(sc, [(-100, -300), (t0, -300), (t0, sc.floor_y + 1), (-100, sc.floor_y + 1)], wc, elasticity=e, tag="side")
        add_static_poly(sc, [(W - t1, -300), (W + 100, -300), (W + 100, sc.floor_y + 1), (W - t1, sc.floor_y + 1)], wc, elasticity=e,
                        tag="side")
        sc.wall_x = (t0, W - t1)
    sc.params["walls"] = bool(walls)
    return sc


def _mc(sc, rng, **kw):
    return muted_color(rng, avoid=[sc.params["bg_color"], sc.floor_color] + [c for _, c, _ in sc.statics if c is not None][-3:], **kw)


# --- 중력 -------------------------------------------------------------------------
@scenario("drop_bounce", "gravity", "1~3 물체를 여러 높이에서 놓거나 던짐 — 자유 낙하 · 튕김")
def sc_drop(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    for _ in range(int(rng.integers(1, 4))):
        add_obj(sc, rng, float(rng.uniform(8, 16)), (rng.uniform(30, W - 30), rng.uniform(15, sc.floor_y - 90)),
                (rng.normal(0, 40), rng.uniform(-150, 30)), elasticity=float(rng.uniform(0.55, 0.9)))
    sc.preroll = float(rng.uniform(0, 0.3))
    return sc


@scenario("projectile", "gravity", "옆에서 던진 물체의 포물선 — 속도 · 각도 무작위, 바닥 · 벽에서 튕김")
def sc_proj(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    if rng.random() < 0.5:
        wx = float(rng.uniform(W - 40, W - 15))
        add_static_poly(sc, [(wx, sc.floor_y - rng.uniform(80, 200)), (W + 60, sc.floor_y - 200), (W + 60, sc.floor_y), (wx, sc.floor_y)],
                        _mc(sc, rng), elasticity=0.7, tag="wall")
    sp = float(rng.uniform(180, 420))
    ang = float(rng.uniform(math.radians(20), math.radians(70)))
    size = float(rng.uniform(8, 15))
    add_obj(sc, rng, size, (rng.uniform(15, 50), sc.floor_y - 1.5 * size - rng.uniform(0, 60)), (sp * math.cos(ang), -sp * math.sin(ang)))
    sc.preroll = float(rng.uniform(0, 0.25))
    return sc


@scenario("toss_up", "gravity", "위로 던진 물체가 꼭대기에서 멈췄다 되돌아 떨어짐 — 꼭대기 높이 · 수평 속도 무작위")
def sc_toss(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    n = int(rng.integers(1, 3))
    for i in range(n):
        size = float(rng.uniform(8, 15))
        x = float(rng.uniform(40, W - 40)) if n == 1 else float((i + 0.5) * W / n + rng.uniform(-25, 25))
        y = float(sc.floor_y - 1.5 * size - rng.uniform(0, 50))
        h = float(rng.uniform(50, max(60.0, y - 2 * size - 15)))     # 꼭대기까지 오를 높이 (화면 안)
        vy = -math.sqrt(2 * sc.g * h)
        vx = float(rng.normal(0, 45))
        o = add_obj(sc, rng, size, (x, y), (vx, vy), elasticity=float(rng.uniform(0.3, 0.8)))
        if i == 0:
            sc.require.append(("apex", len(sc.objs) - 1))
    sc.preroll = float(rng.uniform(0, 0.15))
    return sc


@scenario("pendulum", "gravity", "진자 1~2 개 (단진자 / 이중 진자) — 줄 길이 · 진폭 · 위상 무작위")
def sc_pend(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    n = int(rng.integers(1, 3))
    for i in range(n):
        px = float(rng.uniform(50, W - 50)) if n == 1 else float((i + 0.5) * W / n + rng.uniform(-20, 20))
        py = float(rng.uniform(10, 50))
        L = float(rng.uniform(60, min(150, sc.floor_y - py - 30)))
        th = float(rng.uniform(0.5, 1.3)) * (1 if rng.random() < 0.5 else -1)
        add_pendulum(sc, rng, px, py, L, th, float(rng.uniform(8, 14)), double=bool(rng.random() < 0.35 and L > 90))
    sc.preroll = float(rng.uniform(0, 2.0))
    return sc


@scenario("spring", "gravity", "용수철에 매단 물체 — 수직 진동 또는 탄성 진자")
def sc_spring(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    n = int(rng.integers(1, 3))
    for i in range(n):
        add_spring(sc, rng, float((i + 0.5) * W / n + rng.uniform(-25, 25)), float(rng.uniform(8, 30)), float(rng.uniform(9, 15)))
    sc.preroll = float(rng.uniform(0, 1.5))
    return sc


@scenario("ramp_roll", "gravity", "경사면을 굴러 · 미끄러져 내려옴 (등가속) — 경사 각 · 길이 무작위, 끝에서 바닥 · 벽")
def sc_ramp(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    rc = _mc(sc, rng)
    top = float(rng.uniform(70, 150))
    run = float(rng.uniform(110, 190))
    add_static_poly(sc, [(0, sc.floor_y - top), (run, sc.floor_y), (0, sc.floor_y)], rc, friction=0.9, tag="ramp")
    sc.require.append(("tag", "ramp"))
    if rng.random() < 0.5:
        add_static_poly(sc, [(W - rng.uniform(12, 30), sc.floor_y - rng.uniform(40, 100)), (W + 60, sc.floor_y - 100),
                             (W + 60, sc.floor_y), (W - 12, sc.floor_y)], rc, elasticity=0.6, tag="wall")
    size = float(rng.uniform(8, 14))
    slope = math.atan2(top, run)
    o = add_obj(sc, rng, size, (0, 0), (0, 0), slide=True, angle=slope, angvel=0)
    x = float(rng.uniform(0.0, 0.45)) * run + o.bound
    ys = sc.floor_y - top + top * (x / run)
    # 경사면 법선 방향으로 띄워 놓는다
    off = o.size if o.kind == "circle" else max(-(pymunk.Vec2d(*v).rotated(slope)).dot((math.sin(slope), -math.cos(slope)))
                                               for v in o.verts)
    o.body.position = (x + math.sin(slope) * (off + 0.5), ys - math.cos(slope) * (off + 0.5))
    v0 = float(rng.uniform(0, 60))
    o.body.velocity = (v0 * math.cos(slope), v0 * math.sin(slope))
    o.body.angular_velocity = roll_angvel(o, v0)
    sc.preroll = float(rng.uniform(0, 0.2))
    return sc


@scenario("ledge_fall", "gravity", "선반 위를 굴러 · 미끄러져 가다 끝에서 떨어짐 (중력 개시) — 선반 높이 · 속도 무작위")
def sc_ledge(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    rc = _mc(sc, rng)
    sh_y = float(sc.floor_y - rng.uniform(60, 140))
    sh_x1 = float(rng.uniform(90, 170))
    add_static_poly(sc, [(-100, sh_y), (sh_x1, sh_y), (sh_x1, sh_y + 10), (-100, sh_y + 10)], rc, friction=0.3, tag="ledge")
    sc.require.append(("below", sh_y + 25))
    add_static_poly(sc, [(sh_x1 - 10, sh_y + 10), (sh_x1 - 4, sh_y + 10), (sh_x1 - 4, sc.floor_y), (sh_x1 - 10, sc.floor_y)],
                    shade(rc, 0.8), tag="ledge")
    v = float(rng.uniform(60, 160))
    o = add_obj(sc, rng, float(rng.uniform(8, 13)), (0, 0), (v, 0), friction=0.3, slide=True, angle=0.0, angvel=0)
    o.body.angular_velocity = roll_angvel(o, v)
    place_on(o, rng.uniform(10 + o.bound, sh_x1 - 60), sh_y)
    return sc


# --- 영속 -------------------------------------------------------------------------
@scenario("occ_pass", "permanence", "움직이는 물체 (굴러감 · 포물선 · 낙하 · 튕김) 가 궤적 위 고정 가림막 1~3 개 뒤를 지나 다시 나옴 — 가려진 동안에도 중력 · 충돌")
def sc_occ_pass(rng, bgc, ids):
    sc = _base(rng, bgc, ids, walls=bool(rng.random() < 0.8))
    mode = str(rng.choice(["roll", "throw", "fall", "bounce"]))
    size = float(rng.uniform(8, 13))
    if mode == "roll":
        v = float(rng.uniform(80, 170))
        o = add_obj(sc, rng, size, (0, 0), (v, 0), friction=0.8, elasticity=0.5, slide=True, angle=0.0, angvel=0)
        o.body.angular_velocity = roll_angvel(o, v)
        place_on(o, rng.uniform(20, 60), sc.floor_y)
    elif mode == "throw":
        sp = float(rng.uniform(220, 360))
        ang = float(rng.uniform(math.radians(35), math.radians(65)))
        o = add_obj(sc, rng, size, (rng.uniform(20, 50), sc.floor_y - 1.5 * size - 5), (sp * math.cos(ang), -sp * math.sin(ang)))
    elif mode == "fall":
        o = add_obj(sc, rng, size, (rng.uniform(60, W - 60), rng.uniform(10, 40)), (rng.normal(0, 40), rng.uniform(-60, 20)),
                    elasticity=float(rng.uniform(0.6, 0.9)))
    else:   # 방 안에서 계속 튕기며 돌아다님 → 같은 가림막 뒤를 여러 번 지날 수 있다
        o = add_obj(sc, rng, size, (rng.uniform(40, W - 40), rng.uniform(30, sc.floor_y - 60)),
                    (rng.uniform(100, 220) * rng.choice([-1, 1]), rng.uniform(-200, 50)), elasticity=float(rng.uniform(0.85, 0.97)),
                    friction=0.2)
    sc.need_occlusion.append(o)
    k = int(rng.choice([1, 2, 3], p=[0.45, 0.35, 0.2]))
    sc.auto_occ = dict(obj=o, n=k, hide=(3, 14), min_episodes=1, decoy=bool(rng.random() < P_DECOY))
    sc.params.update(mode=mode, n_auto_screens=k, decoy_occluder=sc.auto_occ["decoy"])
    sc.preroll = float(rng.uniform(0, 0.2))
    return sc


@scenario("osc_behind", "permanence", "진자 · 용수철 · 왕복 공이 가림막 뒤를 여러 번 드나듦 — 가려질 때마다 같은 동역학으로 다시 나온다")
def sc_osc(rng, bgc, ids):
    sc = _base(rng, bgc, ids, walls=True)
    mode = str(rng.choice(["pendulum", "spring", "bounce"]))
    size = float(rng.uniform(8, 13))
    if mode == "pendulum":
        px, py = float(rng.uniform(70, W - 70)), float(rng.uniform(10, 40))
        L = float(rng.uniform(70, min(150, sc.floor_y - py - 30)))
        o = add_pendulum(sc, rng, px, py, L, float(rng.uniform(0.5, 1.2)) * rng.choice([-1, 1]), size)
    elif mode == "spring":
        o = add_spring(sc, rng, float(rng.uniform(60, W - 60)), float(rng.uniform(8, 30)), size)
    else:
        o = add_obj(sc, rng, size, (0, 0), (rng.uniform(90, 200) * rng.choice([-1, 1]), 0), elasticity=float(rng.uniform(0.9, 0.98)),
                    friction=0.1, angle=0.0, angvel=0)
        if rng.random() < 0.5:
            place_on(o, rng.uniform(40, W - 40), sc.floor_y)                       # 바닥에서 좌우 왕복
        else:
            o.body.position = (float(rng.uniform(40, W - 40)), float(rng.uniform(30, sc.floor_y - 80)))   # 튕기며 왕복
    sc.need_occlusion.append(o)
    sc.auto_occ = dict(obj=o, n=1, hide=(2, 14), min_episodes=2, decoy=False, multi=True)
    sc.require.append(("episodes", 2))
    sc.params["mode"] = mode
    sc.preroll = float(rng.uniform(0, 1.0))
    return sc


@scenario("moving_screen", "permanence", "가림막이 오가며 물체를 여러 번 가림 — 물체는 정지 · 진자 · 굴러감 중 하나")
def sc_screen(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    mode = str(rng.choice(["rest", "pendulum", "roll"]))
    size = float(rng.uniform(8, 14))
    if mode == "rest":
        o = add_obj(sc, rng, size, (0, 0), angvel=0, angle=0.0, elasticity=0.1)
        place_on(o, rng.uniform(80, 180), sc.floor_y)
        oy, oh = sc.floor_y - 45 + 2, 90
        px = float(o.body.position.x)
    elif mode == "pendulum":
        px, py = float(rng.uniform(90, 170)), float(rng.uniform(15, 40))
        L = float(rng.uniform(70, 120))
        o = add_pendulum(sc, rng, px, py, L, float(rng.uniform(0.4, 0.9)), size)
        oy, oh = py + L, float(rng.uniform(60, 90))
    else:
        v = float(rng.uniform(40, 90))
        o = add_obj(sc, rng, size, (0, 0), (v, 0), friction=0.8, elasticity=0.2, slide=True, angle=0.0, angvel=0)
        o.body.angular_velocity = roll_angvel(o, v)
        place_on(o, rng.uniform(40, 110), sc.floor_y)
        oy, oh = sc.floor_y - 45 + 2, 90
        px = float(o.body.position.x)
    ow = float(rng.uniform(65, 110))
    vert = rng.random() < 0.35
    sp = float(rng.uniform(55, 130))
    t = float(rng.uniform(0.0, 0.6))
    if vert:
        cx = px
        if mode == "roll":
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
    mv = _screen(rng, sc, 0, oy, ow, oh, path=path)
    mv.ease = bool(vert)
    sc.movers.append(mv)
    sc.need_occlusion.append(o)
    sc.params.update(mode=mode, vertical_screen=bool(vert))
    return sc


@scenario("cup_cover", "permanence", "컵이 내려와 물체를 덮고 (옆으로 밀고 갈 때도 있음) 다시 들림 — 1~2 번")
def sc_cup(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    size = float(rng.uniform(8, 12))
    x = float(rng.uniform(70, 180))
    o = add_obj(sc, rng, size, (0, 0), angvel=0, angle=0.0, elasticity=0.05, friction=0.8)
    place_on(o, x, sc.floor_y, gap=0.2)
    cw = float(rng.uniform(3.0, 4.2) * o.bound)
    ch = float(rng.uniform(2.8, 3.6) * o.bound)
    cy_dn = sc.floor_y - ch / 2 + 1.0
    cy_up = float(rng.uniform(ch / 2 + 2, cy_dn - ch - 2 * o.bound))
    x0 = x + float(rng.uniform(-60, 60))
    t1 = float(rng.uniform(0.1, 0.4))
    path = [(0, x0, cy_up), (t1 + 0.5, x, cy_up), (t1 + 0.9, x, cy_dn)]
    t = t1 + 0.9
    if rng.random() < 0.5:
        x2 = float(np.clip(x + rng.choice([-1, 1]) * rng.uniform(40, 80), 40, W - 40))
        path += [(t + 0.6, x2, cy_dn)]
        t += 0.6
        x = x2
    tp = t + float(rng.uniform(0.05, 0.15))
    path += [(tp, x, cy_dn), (tp + 0.4, x, cy_up)]
    t = tp + 0.4
    for _ in range(int(rng.integers(0, 2))):
        t += float(rng.uniform(0.1, 0.3))
        path += [(t, x, cy_up), (t + 0.4, x, cy_dn)]
        t += 0.4
        if rng.random() < 0.5:
            x = float(np.clip(x + rng.choice([-1, 1]) * rng.uniform(30, 70), 40, W - 40))
            path += [(t + 0.6, x, cy_dn)]
            t += 0.6
        t += float(rng.uniform(0.05, 0.15))
        path += [(t, x, cy_dn), (t + 0.4, x, cy_up)]
        t += 0.4
    path += [(t + 0.6, x + rng.uniform(-60, 60), cy_up)]
    mv = Mover(path=path, w=cw, h=ch, color=_mc(sc, rng), kind="cup", pattern=str(rng.choice(["plain", "stripes", "frame", "checker"])))
    add_cup_body(sc, mv)
    sc.movers.append(mv)
    sc.need_occlusion.append(o)
    add_distractor(sc, rng, away_x=x)
    return sc


@scenario("drop_behind", "permanence", "물체가 가림막 뒤로 떨어져 (튕기고) 멈춤, 그다음 가림막이 비켜나며 드러남")
def sc_drop_behind(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    size = float(rng.uniform(8, 13))
    ox = float(rng.uniform(80, 180))
    o = add_obj(sc, rng, size, (0, 0), (rng.normal(0, 15), rng.uniform(-40, 30)), elasticity=float(rng.uniform(0.3, 0.7)))
    ow = float(rng.uniform(4.5, 7) * o.bound)
    oh = float(rng.uniform(50, 90))
    oy = sc.floor_y - oh / 2 + 1
    o.body.position = (ox + rng.uniform(-ow / 5, ow / 5), rng.uniform(15, 60))
    t_go = float(rng.uniform(0.9, 1.4))
    dx = float(rng.choice([-1, 1]) * rng.uniform(1.0, 1.6) * ow)
    dy = float(-rng.uniform(0, 1) * oh * 1.4)
    sc.movers.append(_screen(rng, sc, 0, 0, ow, oh, path=[(0, ox, oy), (t_go, ox, oy), (t_go + 1.0, ox + dx, oy + dy)]))
    sc.need_occlusion.append(o)
    sc.preroll = float(rng.uniform(0, 0.15))
    sc.require.append(("hidden_at", t_go))
    return sc


@scenario("tunnel", "permanence", "굴러 · 미끄러져 가는 물체가 터널로 들어가 반대편으로 나옴 — 경사 터널이면 안에서 가속")
def sc_tunnel(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    size = float(rng.uniform(7, 11))
    tx0 = float(rng.uniform(70, 120))
    tw = float(rng.uniform(50, 100))
    slope = rng.random() < 0.4
    v = float(rng.uniform(60, 140))
    o = add_obj(sc, rng, size, (0, 0), (v, 0), friction=0.9, elasticity=0.2, slide=True, angle=0.0, angvel=0)
    o.body.angular_velocity = roll_angvel(o, v)
    th = float(o.bound * 2 + rng.uniform(10, 30))
    if slope:
        top = float(rng.uniform(20, 45))
        add_static_poly(sc, [(-50, sc.floor_y - top), (tx0, sc.floor_y - top), (tx0 + tw, sc.floor_y), (-50, sc.floor_y)], _mc(sc, rng),
                        friction=0.9, tag="ramp")
        place_on(o, rng.uniform(5 + o.bound, tx0 - 30), sc.floor_y - top)
    else:
        place_on(o, rng.uniform(5 + o.bound, tx0 - 30), sc.floor_y)
    sc.movers.append(_screen(rng, sc, tx0 + tw / 2, sc.floor_y - th / 2 + 1 - (10 if slope else 0), tw, th + (20 if slope else 0)))
    sc.need_occlusion.append(o)
    sc.params["slope"] = bool(slope)
    return sc


# --- 고체성 -----------------------------------------------------------------------
@scenario("shelf_land", "solidity", "선반 · 판 여러 개 위로 떨어져 그 위에 서거나 튕겨 내려감 — 통과하지 않는다")
def sc_shelf(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    for _ in range(int(rng.integers(1, 4))):
        x0 = float(rng.uniform(0, W - 70))
        w = float(rng.uniform(50, 110))
        y = float(rng.uniform(80, sc.floor_y - 30))
        ang = float(rng.uniform(-0.35, 0.35)) if rng.random() < 0.5 else 0.0
        dx, dy = w * math.cos(ang), w * math.sin(ang)
        add_static_poly(sc, [(x0, y), (x0 + dx, y + dy), (x0 + dx, y + dy + 8), (x0, y + 8)], _mc(sc, rng),
                        elasticity=float(rng.uniform(0.2, 0.7)), tag="shelf")
    sc.require.append(("tag", "shelf"))
    for _ in range(int(rng.integers(1, 4))):
        add_obj(sc, rng, float(rng.uniform(8, 14)), (rng.uniform(20, W - 20), rng.uniform(5, 50)), (rng.normal(0, 30), rng.uniform(-80, 20)))
    sc.preroll = float(rng.uniform(0, 0.2))
    return sc


@scenario("wall_bounce", "solidity", "던지거나 굴린 물체가 벽에 부딪혀 되튕김 — 벽 높이 · 위치 · 반발 무작위")
def sc_wall(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    wx = float(rng.uniform(130, 210))
    wh = float(rng.uniform(70, 170))
    ww = float(rng.uniform(10, 25))
    add_static_poly(sc, [(wx, sc.floor_y - wh), (wx + ww, sc.floor_y - wh), (wx + ww, sc.floor_y), (wx, sc.floor_y)], _mc(sc, rng),
                    elasticity=float(rng.uniform(0.4, 0.9)), tag="wall")
    sc.require.append(("tag", "wall"))
    size = float(rng.uniform(8, 14))
    if rng.random() < 0.5:
        v = float(rng.uniform(120, 260))
        o = add_obj(sc, rng, size, (0, 0), (v, 0), elasticity=float(rng.uniform(0.5, 0.9)), slide=True, angle=0.0, angvel=0)
        o.body.angular_velocity = roll_angvel(o, v)
        place_on(o, rng.uniform(10 + o.bound, 60), sc.floor_y)
    else:
        sp = float(rng.uniform(200, 330))
        ang = float(rng.uniform(math.radians(15), math.radians(55)))
        add_obj(sc, rng, size, (rng.uniform(10, 50), sc.floor_y - 1.5 * size - rng.uniform(5, 60)), (sp * math.cos(ang), -sp * math.sin(ang)))
    sc.preroll = float(rng.uniform(0, 0.15))
    return sc


@scenario("collide", "solidity", "2~3 물체가 바닥 위 / 공중에서 서로 충돌 — 질량 (크기) 따라 운동량 교환")
def sc_collide(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    for i in range(int(rng.integers(2, 4))):
        size = float(rng.uniform(8, 16))
        left = i % 2 == 0
        x = float(rng.uniform(15, 80)) if left else float(rng.uniform(W - 80, W - 15))
        v = float(rng.uniform(70, 200)) * (1 if left else -1)
        if rng.random() < 0.4:   # 공중
            add_obj(sc, rng, size, (x, sc.floor_y - size - rng.uniform(40, 120)), (v, -rng.uniform(50, 200)),
                    elasticity=float(rng.uniform(0.6, 0.95)))
        else:
            o = add_obj(sc, rng, size, (0, 0), (v, 0), elasticity=float(rng.uniform(0.6, 0.95)), slide=True, angle=0.0, angvel=0)
            o.body.angular_velocity = roll_angvel(o, v)
            place_on(o, x, sc.floor_y)
    sc.preroll = float(rng.uniform(0, 0.2))
    sc.require.append(("objobj",))
    return sc


@scenario("stack_topple", "solidity", "쌓인 상자들 — 물체가 날아와 치거나 불안정하게 쌓여 무너짐")
def sc_stack(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    n = int(rng.integers(2, 5))
    s = float(rng.uniform(9, 13))
    x = float(rng.uniform(120, 190))
    y = sc.floor_y
    unstable = rng.random() < 0.5
    first = len(sc.objs)
    for i in range(n):
        o = add_obj(sc, rng, s, (0, 0), kind=str(rng.choice(["square", "rectangle"])), role="stack", angle=0.0, angvel=0,
                    friction=0.7, elasticity=0.2)
        off = float(rng.uniform(0.35, 0.6) * s) if unstable else float(rng.uniform(-2, 2))
        place_on(o, x + i * off, y, gap=0.1)
        y = float(o.body.position.y - low_offset(o) - 0.1)       # 다음 상자는 이 상자 윗면 위에
    if not unstable or rng.random() < 0.5:
        sp = float(rng.uniform(200, 330))
        ang = float(rng.uniform(math.radians(10), math.radians(45)))
        size = float(rng.uniform(8, 12))
        add_obj(sc, rng, size, (rng.uniform(10, 40), sc.floor_y - 1.5 * size - rng.uniform(0, 40)), (sp * math.cos(ang), -sp * math.sin(ang)),
                density=3.0)
    sc.params["unstable"] = bool(unstable)
    sc.require.append(("moved", list(range(first, first + n)), s))
    return sc


@scenario("container", "solidity", "통 (U 자) 안으로 떨어지거나 던져진 물체 — 벽 안에서 튕기고 갇힌다")
def sc_container(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    cc = _mc(sc, rng)
    cx = float(rng.uniform(90, 170))
    cw = float(rng.uniform(60, 110))
    ch = float(rng.uniform(40, 80))
    t, yb = 6.0, sc.floor_y
    for pts in ([(cx - cw / 2, yb - ch), (cx - cw / 2 + t, yb - ch), (cx - cw / 2 + t, yb), (cx - cw / 2, yb)],
                [(cx + cw / 2 - t, yb - ch), (cx + cw / 2, yb - ch), (cx + cw / 2, yb), (cx + cw / 2 - t, yb)]):
        add_static_poly(sc, pts, cc, elasticity=0.6, tag="bin")
    sc.require.append(("final_in", cx - cw / 2 + t, cx + cw / 2 - t, yb - ch))
    sc.params["bin_x"] = (cx - cw / 2 + t + 8, cx + cw / 2 - t - 8)
    for _ in range(int(rng.integers(1, 3))):
        size = float(rng.uniform(7, 11))
        if rng.random() < 0.5:
            pos, vel = (cx + rng.uniform(-cw / 3, cw / 3), rng.uniform(10, 60)), (rng.normal(0, 60), rng.uniform(-60, 20))
        else:
            pos, vel = (rng.uniform(10, 40), sc.floor_y - rng.uniform(40, 100)), (rng.uniform(120, 200), -rng.uniform(150, 300))
        add_obj(sc, rng, size, pos, vel, elasticity=float(rng.uniform(0.6, 0.9)))
    return sc


@scenario("occ_wall", "solidity", "고체성 × 영속 — 벽 아랫부분이 가림막에 가려 있고, 굴러간 물체가 가림막 뒤에서 벽에 튕겨 되돌아 나옴")
def sc_occ_wall(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    wx = float(rng.uniform(150, 200))
    wh = float(rng.uniform(80, 150))
    add_static_poly(sc, [(wx, sc.floor_y - wh), (wx + 12, sc.floor_y - wh), (wx + 12, sc.floor_y), (wx, sc.floor_y)], _mc(sc, rng),
                    elasticity=float(rng.uniform(0.6, 0.9)), tag="wall")
    sc.require.append(("tag", "wall"))
    v = float(rng.uniform(110, 200))
    o = add_obj(sc, rng, float(rng.uniform(7, 11)), (0, 0), (v, 0), elasticity=float(rng.uniform(0.6, 0.9)), friction=0.5, slide=True,
                angle=0.0, angvel=0)
    o.body.angular_velocity = roll_angvel(o, v)
    place_on(o, rng.uniform(10 + o.bound, 50), sc.floor_y)
    ow = float(rng.uniform(55, 80))
    oh = float(rng.uniform(2.6, 3.6) * o.bound)
    sc.movers.append(_screen(rng, sc, wx - ow / 2 + 14, sc.floor_y - oh / 2 + 1, ow, oh))
    sc.need_occlusion.append(o)
    return sc


# --- 섞음 -------------------------------------------------------------------------
@scenario("sandbox", "mixed", "지형 (경사 · 선반 · 벽 · 통) · 장치 (진자 · 용수철 · 가림막 · 움직이는 가림막) 2~4 개와 물체 2~4 개를 무작위 조합 — 사건 종류 2 가지 이상")
def sc_sandbox(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    elems = list(rng.choice(["ramp", "shelf", "wall", "bin", "pendulum", "spring", "screen", "moving_screen"],
                            size=int(rng.integers(2, 5)), replace=False))
    for e in elems:
        for _ in range(20):
            if e == "ramp":
                top, run = float(rng.uniform(50, 120)), float(rng.uniform(80, 150))
                left = rng.random() < 0.5
                x0 = 0.0 if left else W - run
                pts = ([(x0, sc.floor_y - top), (x0 + run, sc.floor_y), (x0, sc.floor_y)] if left
                       else [(x0, sc.floor_y), (W, sc.floor_y - top), (W, sc.floor_y)])
                if _overlaps(sc, x0, sc.floor_y - top, x0 + run, sc.floor_y - 1):
                    continue
                add_static_poly(sc, pts, _mc(sc, rng), friction=0.8, tag="ramp")
            elif e == "shelf":
                w, y = float(rng.uniform(50, 110)), float(rng.uniform(70, sc.floor_y - 40))
                x0 = float(rng.uniform(0, W - w))
                ang = float(rng.uniform(-0.3, 0.3)) if rng.random() < 0.6 else 0.0
                dx, dy = w * math.cos(ang), w * math.sin(ang)
                if _overlaps(sc, x0, min(y, y + dy) - 10, x0 + dx, max(y, y + dy) + 18):
                    continue
                add_static_poly(sc, [(x0, y), (x0 + dx, y + dy), (x0 + dx, y + dy + 8), (x0, y + 8)], _mc(sc, rng),
                                elasticity=float(rng.uniform(0.2, 0.7)), tag="shelf")
            elif e == "wall":
                wx, wh, ww = float(rng.uniform(60, W - 70)), float(rng.uniform(50, 150)), float(rng.uniform(10, 22))
                if _overlaps(sc, wx, sc.floor_y - wh, wx + ww, sc.floor_y - 1):
                    continue
                add_static_poly(sc, [(wx, sc.floor_y - wh), (wx + ww, sc.floor_y - wh), (wx + ww, sc.floor_y), (wx, sc.floor_y)],
                                _mc(sc, rng), elasticity=float(rng.uniform(0.4, 0.9)), tag="wall")
            elif e == "bin":
                cx, cw, ch = float(rng.uniform(60, W - 60)), float(rng.uniform(55, 95)), float(rng.uniform(35, 70))
                if _overlaps(sc, cx - cw / 2, sc.floor_y - ch, cx + cw / 2, sc.floor_y - 1):
                    continue
                c, yb = _mc(sc, rng), sc.floor_y
                for pts in ([(cx - cw / 2, yb - ch), (cx - cw / 2 + 6, yb - ch), (cx - cw / 2 + 6, yb), (cx - cw / 2, yb)],
                            [(cx + cw / 2 - 6, yb - ch), (cx + cw / 2, yb - ch), (cx + cw / 2, yb), (cx + cw / 2 - 6, yb)]):
                    add_static_poly(sc, pts, c, elasticity=0.6, tag="bin")
            elif e == "pendulum":
                px, py = float(rng.uniform(50, W - 50)), float(rng.uniform(10, 40))
                L = float(rng.uniform(60, min(130, sc.floor_y - py - 40)))
                if _overlaps(sc, px - L, py, px + L, py + L + 15):
                    continue
                add_pendulum(sc, rng, px, py, L, float(rng.uniform(0.5, 1.2)) * rng.choice([-1, 1]), float(rng.uniform(8, 13)))
            elif e == "spring":
                px = float(rng.uniform(50, W - 50))
                if _overlaps(sc, px - 45, 10, px + 45, 170):
                    continue
                add_spring(sc, rng, px, float(rng.uniform(8, 30)), float(rng.uniform(8, 13)))
            elif e == "screen":
                w, h = float(rng.uniform(35, 85)), float(rng.uniform(40, 110))
                x = float(rng.uniform(5, W - 5 - w))
                y = float(sc.floor_y + 2 - h) if rng.random() < 0.6 else float(rng.uniform(15, sc.floor_y - h - 15))
                if _overlaps(sc, x, y, x + w, y + h, pad=4, movers=True):
                    continue
                sc.movers.append(_screen(rng, sc, x + w / 2, y + h / 2, w, h))
            else:   # moving_screen — 화면 안에서 왕복 (구간 등속, 끝점 쉼 짧게)
                w, h = float(rng.uniform(40, 80)), float(rng.uniform(50, 100))
                y = float(sc.floor_y + 2 - h / 2) if rng.random() < 0.5 else float(rng.uniform(20 + h / 2, sc.floor_y - h / 2 - 10))
                xa, xb = float(rng.uniform(w / 2, W / 2)), float(rng.uniform(W / 2, W - w / 2))
                sp = float(rng.uniform(40, 110))
                t = 0.0
                path, cur, other = [(0.0, xa, y)], xa, xb
                for _ in range(8):
                    t += abs(other - cur) / sp
                    path.append((t, other, y))
                    t += float(rng.uniform(0.05, 0.2))
                    path.append((t, other, y))
                    cur, other = other, cur
                mv = _screen(rng, sc, 0, 0, w, h, path=path)
                mv.ease = False
                sc.movers.append(mv)
            break
    for _ in range(int(rng.integers(2, 5))):
        size = float(rng.uniform(8, 14))
        mode = rng.choice(["drop", "throw", "roll"])
        if mode == "drop":
            add_obj(sc, rng, size, (rng.uniform(25, W - 25), rng.uniform(10, 70)), (rng.normal(0, 50), rng.uniform(-100, 20)))
        elif mode == "throw":
            left = rng.random() < 0.5
            sp, ang = float(rng.uniform(180, 380)), float(rng.uniform(math.radians(20), math.radians(70)))
            add_obj(sc, rng, size, (rng.uniform(15, 50) if left else rng.uniform(W - 50, W - 15), sc.floor_y - rng.uniform(30, 100)),
                    (sp * math.cos(ang) * (1 if left else -1), -sp * math.sin(ang)))
        else:
            v = float(rng.uniform(70, 180)) * rng.choice([-1, 1])
            o = add_obj(sc, rng, size, (0, 0), (v, 0), slide=True, angle=0.0, angvel=0)
            o.body.angular_velocity = roll_angvel(o, v)
            place_on(o, rng.uniform(20, W - 20), sc.floor_y)
    sc.params["elements"] = [str(e) for e in elems]
    sc.require.append(("events_min", 2))
    sc.preroll = float(rng.uniform(0, 0.3))
    return sc


# ----------------------------------------------------------------------------- 시뮬
def step_movers(sc, t, dt):
    for mv in sc.movers:
        if mv.body is not None:
            x, y = mv.pos(t + dt)
            px, py = mv.body.position
            mv.body.velocity = ((x - px) / dt, (y - py) / dt)


def record_contacts(sc):
    bidx = {o.body: i for i, o in enumerate(sc.objs)}
    for i, o in enumerate(sc.objs):
        if o.body.space is None:
            continue
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
        for tag in sorted(sc.touching.get(i, set()) - now):     # 정렬 — 문자열 집합 순서는 PYTHONHASHSEED 를 따른다 (재현성, 2026-10-19)
            sc.leaves.append((sc.cur_frame, i, tag))
        sc.touching[i] = now


def simulate(sc, n_frames=N_FRAMES):
    """물리만 돌리고 장마다 상태를 기록한다 (렌더 없음). 장 f 의 상태 = 그 장을 그리는 순간."""
    dt = 1.0 / (FPS * SUBSTEPS)
    t = 0.0
    for _ in range(int(sc.preroll / dt)):
        step_movers(sc, t, dt)
        sc.space.step(dt)
        t += dt
    t0 = t
    n = len(sc.objs)
    pos = np.full((n_frames, n, 2), -1000.0, np.float32)
    vel = np.zeros((n_frames, n, 2), np.float32)
    ang = np.zeros((n_frames, n), np.float32)
    om = np.zeros((n_frames, n), np.float32)
    present = np.zeros((n_frames, n), bool)
    ts = np.zeros(n_frames, np.float64)
    for f in range(n_frames):
        sc.cur_frame = f
        ts[f] = t
        for i, o in enumerate(sc.objs):
            if o.body.space is not None:
                present[f, i] = True
                pos[f, i] = tuple(o.body.position)
                vel[f, i] = tuple(o.body.velocity)
                ang[f, i] = o.body.angle
                om[f, i] = o.body.angular_velocity
        for mv in sc.movers:
            if mv.body is not None:
                mv.track.append(tuple(mv.body.position))
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
    return dict(pos=pos, vel=vel / FPS, ang=ang, om=om / FPS, present=present, ts=ts, t0=t0)


def mover_rects(sc, S, n):
    """(n, m, 4) x0 y0 x1 y1 — 컵은 시뮬 중 기록한 몸체 자리, 가림막은 경로."""
    m = len(sc.movers)
    r = np.zeros((n, m, 4), np.float32)
    for j, mv in enumerate(sc.movers):
        for f in range(n):
            x, y = mv.track[f] if mv.body is not None else mv.pos(S["ts"][f])
            r[f, j] = (x - mv.w / 2, y - mv.h / 2, x + mv.w / 2, y + mv.h / 2)
    return r


def mover_speed(rects, active=None):
    """장마다 화면 안에 있는 가장 빠른 가림막 · 컵의 속도 (px/s). 화면 밖으로 나간 가림막의 움직임은 세지 않는다
    (안 세면 떠다니는 가림막이 나간 뒤의 정지 화면이 운동 검사 · 끝 자르기를 통과한다).
    active: 셀 가림막 (기본 전부). 생성기는 시나리오 가림막만 센다 — 무관한 extra · drift 가림막의 움직임이 운동 검사를 대신 통과시키면
    물체가 가만있는 클립이 남고, rejection 이 drift 가림막 있는 클립을 골라 그 비율이 설계값 (12 %) 의 2.5 배가 됐다 (2026-10-18 중간 점검)."""
    if rects.shape[1] == 0:
        return np.zeros(len(rects), np.float32)
    if active is not None:
        rects = rects[:, np.asarray(active, bool)]
        if rects.shape[1] == 0:
            return np.zeros(len(rects), np.float32)
    c = (rects[..., :2] + rects[..., 2:]) / 2
    on = (rects[..., 2] > 0) & (rects[..., 0] < W) & (rects[..., 3] > 0) & (rects[..., 1] < H)
    d = np.zeros(len(rects), np.float32)
    d[:-1] = (np.linalg.norm(c[1:] - c[:-1], axis=-1) * (on[1:] | on[:-1])).max(1) * FPS
    d[-1] = d[-2] if len(d) > 1 else 0
    return d


# ----------------------------------------------------------------------------- 가림막 자동 배치
def _inframe(p, b):
    return (p[..., 0] > b) & (p[..., 0] < W - b) & (p[..., 1] > b) & (p[..., 1] < H - b)


def _covered(p, b, r):
    """물체 (중심 p, 외접 반경 b) 가 직사각형 r 안에 완전히 들어가는가"""
    return (p[..., 0] - b >= r[0]) & (p[..., 0] + b <= r[2]) & (p[..., 1] - b >= r[1]) & (p[..., 1] + b <= r[3])


def _touch(p, b, r, pad=0.0):
    return (p[..., 0] + b > r[0] - pad) & (p[..., 0] - b < r[2] + pad) & (p[..., 1] + b > r[1] - pad) & (p[..., 1] - b < r[3] + pad)


def _episodes(mask):
    """True 구간 [(시작, 끝 포함)]"""
    out, s = [], None
    for f, v in enumerate(mask):
        if v and s is None:
            s = f
        elif not v and s is not None:
            out.append((s, f - 1))
            s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


def place_auto_screens(sc, rng, pos, T):
    """1 차 궤적을 보고 가림막을 놓는다.
    occ_pass: 서로 다른 구간 n 개를 각각 hide 장 동안 완전히 덮는 가림막 n 개 (decoy 면 궤적과 안 겹치는 자리로 옮긴다).
    osc_behind (multi): 가림막 하나가 물체를 min_episodes 번 이상 따로 덮는다."""
    A = sc.auto_occ
    i = sc.objs.index(A["obj"])
    b = sc.objs[i].bound
    p = pos[:T, i]
    inside = _inframe(p, b)
    rects = []
    if T < 24:
        return None
    if A.get("multi"):
        for _ in range(80):
            f = int(rng.integers(4, T - 4))
            if not inside[f]:
                continue
            w, h = float(rng.uniform(2 * b + 14, 2 * b + 70)), float(rng.uniform(2 * b + 14, 2 * b + 80))
            cx, cy = p[f, 0] + rng.uniform(-0.3, 0.3) * (w - 2 * b), p[f, 1] + rng.uniform(-0.3, 0.3) * (h - 2 * b)
            r = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            if r[3] > sc.floor_y - 2 * b - 4:
                r[3] = sc.floor_y + 1
            if r[0] < 0 or r[2] > W or r[1] < 0:
                continue
            ep = [e for e in _episodes(_covered(p, b, r)) if e[1] - e[0] + 1 >= A["hide"][0]]
            if len(ep) < A["min_episodes"] or _touch(p[0], b, r) or not (~_touch(p[ep[-1][1]:], b, r)).any():
                continue
            rects.append(r)
            break
        return rects or None
    used = []
    for _ in range(A["n"]):
        for _ in range(30):
            L = int(rng.integers(A["hide"][0], A["hide"][1] + 1))
            cands = [a for a in range(4, T - L - 6) if inside[a - 4:a + L + 6].all() and all(abs(a - u) > L + 8 for u in used)]
            if not cands:
                break
            a = int(rng.choice(cands))
            seg = p[a:a + L]
            r = [*(seg.min(0) - b - 3), *(seg.max(0) + b + 3)]
            if r[2] - r[0] > 150 or r[3] - r[1] > 150:
                continue
            if r[3] > sc.floor_y - 2 * b - 4:
                r[3] = sc.floor_y + 1
            if _touch(p[0], b, r) or _touch(p[a + L:], b, r).all():
                continue
            if any(not (r[2] + 6 < q[0] or r[0] - 6 > q[2] or r[3] + 6 < q[1] or r[1] - 6 > q[3]) for q in rects):
                continue
            rects.append(r)
            used.append(a)
            break
    if not rects:
        return None
    if A["decoy"]:
        # 미끼 — 같은 크기 가림막을 궤적과 겹치지 않는 자리로 옮긴다 (가림막 위치 → 미래 궤적 상관을 끊는다)
        moved = []
        for r in rects:
            w, h = r[2] - r[0], r[3] - r[1]
            for _ in range(50):
                nx0 = float(rng.uniform(0, W - w))
                ny0 = float(sc.floor_y + 1 - h) if r[3] >= sc.floor_y else float(rng.uniform(10, sc.floor_y - h - 5))
                q = [nx0, ny0, nx0 + w, ny0 + h]
                if not _touch(p, b, q).any():
                    moved.append(q)
                    break
        if not moved:
            return None
        rects = moved
    return rects


# ----------------------------------------------------------------------------- 라벨 (래스터)
def world_verts(o, p, a):
    c, s = math.cos(a), math.sin(a)
    return [(p[0] + vx * c - vy * s, p[1] + vx * s + vy * c) for vx, vy in o.verts]


def _fill_obj(lab, o, p, a, val):
    if o.kind == "circle":
        cv2.circle(lab, _p(*p, s=1), _r(o.size, s=1), val, -1, cv2.LINE_8, SHIFT)
    else:
        cv2.fillPoly(lab, [_pts(world_verts(o, p, a), s=1)], val, cv2.LINE_8, SHIFT)


def rasterize(sc, S, rects, T):
    """보이는 픽셀 인스턴스 세그 (0 = 배경 · 지형, i+1 = 물체 i, 255 = 가림막 · 컵) 와 물체별 픽셀 수.
    그리는 순서 = 렌더 순서 (물체 → 가림막)."""
    n = len(sc.objs)
    seg = np.zeros((T, H, W), np.uint8)
    vis_px = np.zeros((T, n), np.float32)
    amo_px = np.zeros((T, n), np.float32)
    scratch = np.zeros((H, W), np.uint8)
    for f in range(T):
        lab = seg[f]
        for i, o in enumerate(sc.objs):
            if not S["present"][f, i]:
                continue
            p, a = S["pos"][f, i], float(S["ang"][f, i])
            if not _touch(p, o.bound, (0, 0, W, H)):
                continue
            x0, y0 = max(0, int(p[0] - o.bound) - 2), max(0, int(p[1] - o.bound) - 2)
            x1, y1 = min(W, int(p[0] + o.bound) + 3), min(H, int(p[1] + o.bound) + 3)
            scratch[y0:y1, x0:x1] = 0
            _fill_obj(scratch, o, p, a, 1)
            amo_px[f, i] = scratch[y0:y1, x0:x1].sum()
            _fill_obj(lab, o, p, a, i + 1)
        for r in rects[f]:
            xa, ya = max(0, int(math.ceil(r[0] - 0.5))), max(0, int(math.ceil(r[1] - 0.5)))
            xb, yb = min(W - 1, int(math.floor(r[2] - 0.5))), min(H - 1, int(math.floor(r[3] - 0.5)))
            if xa <= xb and ya <= yb:
                lab[ya:yb + 1, xa:xb + 1] = 255
        bc = np.bincount(lab.ravel(), minlength=256)
        vis_px[f] = bc[1:n + 1]
    return seg, vis_px, amo_px


# ----------------------------------------------------------------------------- 검사
def check_require(sc, S, ratio, T):
    why = []
    pos = S["pos"][:T]
    main = [i for i, o in enumerate(sc.objs) if o.role in ("main", "bob", "stack")]
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
            f = int(np.searchsorted(S["ts"][:T], r[1])) - 1
            i = sc.objs.index(sc.need_occlusion[0])
            if f < 0 or f >= T or ratio[f, i] > HIDDEN:
                why.append("not_hidden_at")
        elif r[0] == "apex":
            i = r[1]
            a = int(np.argmin(pos[:, i, 1]))
            vy = S["vel"][:T, i, 1]
            if not (2 <= a <= T - 3 and pos[a, i, 1] > sc.objs[i].bound and (vy[:a] < -2).any() and (vy[a:] > 2).any()):
                why.append("no_apex")
        elif r[0] == "episodes":
            i = sc.objs.index(sc.need_occlusion[0])
            if n_episodes(ratio[:, i]) < r[1]:
                why.append(f"episodes<{r[1]}")
        elif r[0] == "events_min":
            kinds = set()
            if any(not (isinstance(o, str) and o in LOW_TAGS) and f + 1 < T for f, i, o, dv in sc.impacts if dv >= DV_ROW):
                kinds.add("impact")
            if any(f + 1 < T for f, i, tag in sc.leaves):
                kinds.add("leave")
            if any(n_episodes(ratio[:, i]) > 0 for i in range(len(sc.objs))):
                kinds.add("occlude")
            if len(kinds) < r[1]:
                why.append(f"events {sorted(kinds)}")
    return why


def n_episodes(ratio_i, min_len=2):
    """완전히 가려졌다가 (min_len 장 이상) 다시 보인 횟수 — 앞뒤로 절반 넘게 보인 장이 있어야 센다."""
    hid = ratio_i < HIDDEN
    k = 0
    for s, e in _episodes(hid):
        if e - s + 1 >= min_len and s > 0 and (ratio_i[:s] > 0.5).any() and e + 1 < len(ratio_i) and (ratio_i[e + 1:] > 0.5).any():
            k += 1
    return k


def check(sc, S, ratio, ms, T, delta=8):
    """통과 여부와 지표 (vast 판과 같은 기준). 복사 · 직선 외삽으로 미래가 풀리지 않는지를 잰다."""
    pos, vel = S["pos"][:T], S["vel"][:T] * FPS
    spd = np.linalg.norm(vel, axis=-1)
    inframe = _inframe(pos, -5)
    moving = ((spd * inframe).max(1) > MOVE_THR) | (ms > MOVE_THR)
    motion_frac = float(moving.mean())
    still_tail = 0
    for m in moving[::-1]:
        if m:
            break
        still_tail += 1
    lin = pos[:-delta] + vel[:-delta] * (delta / FPS)
    err = np.linalg.norm(lin - pos[delta:], axis=-1)
    copy_err = np.linalg.norm(pos[:-delta] - pos[delta:], axis=-1)
    both = inframe[:-delta] & inframe[delta:]
    lin_err = float(np.nanmean(np.where(both, err, np.nan).max(1, initial=0)))
    pos_copy_err = float(np.nanmean(np.where(both, copy_err, 0).max(1)))
    ok, why = True, []
    if motion_frac < 0.7:
        ok, why = False, why + [f"motion_frac {motion_frac:.2f}"]
    main_idx = [i for i, o in enumerate(sc.objs) if o.role in ("main", "bob")]
    occ_idx = [sc.objs.index(o) for o in sc.need_occlusion]
    for i in main_idx:
        if i not in occ_idx and inframe[:, i].mean() < 0.8:
            ok, why = False, why + [f"obj{i} offscreen {1 - inframe[:, i].mean():.2f}"]
    n_hidden = 0
    for i in occ_idx:
        hid = (ratio[:, i] < HIDDEN) & inframe[:, i]
        n_hidden = int(hid.sum())
        idx = np.where(hid)[0]
        if n_hidden < 3:
            ok, why = False, why + [f"hidden {n_hidden}"]
            continue
        if not ((ratio[:idx[0], i] > 0.7).any() and (ratio[idx[-1]:, i] > 0.7).any()):
            ok, why = False, why + ["vis before/after"]
        if inframe[:, i].mean() < 0.9:
            ok, why = False, why + ["occ obj offscreen"]
    sizes = np.array([o.bound for o in sc.objs], np.float32)
    big = (copy_err * both > 0.5 * sizes[None]).any(1)
    big |= ms[:-delta] * delta / FPS > 8
    displaced = float(big.mean())
    if displaced < 0.6:
        ok, why = False, why + [f"displaced {displaced:.2f}"]
    req = check_require(sc, S, ratio, T)
    if req:
        ok, why = False, why + req
    eps = [n_episodes(ratio[:, i]) for i in range(len(sc.objs))]
    metrics = dict(motion_frac=motion_frac, displaced_frac=displaced, still_tail_frac=still_tail / T, delta=delta,
                   pos_copy_err_px=pos_copy_err, linear_extrap_err_px=lin_err, n_hidden_frames=n_hidden,
                   n_occ_episodes=int(sum(eps)), max_obj_episodes=int(max(eps) if eps else 0))
    return ok, why, metrics


def trim_len(S, ms, tail=4):
    """마지막으로 화면 안의 무언가가 움직인 장 + tail 에서 자른다 (정지 화면 = 복사 정답)."""
    mov = ((np.linalg.norm(S["vel"] * FPS, axis=-1) * _inframe(S["pos"], -5)).max(1) > MOVE_THR) | (ms > MOVE_THR)
    idx = np.where(mov)[0]
    return int(min(len(mov), (idx[-1] + 1 + tail) if len(idx) else 0))


# ----------------------------------------------------------------------------- 사건
def find_events(sc, ratio, T):
    """사건 목록 — 첫 사후 장 기준. impact (vast 와 같은 정의) · leave · occlude (완전히 가려진 첫 장) · reappear (다시 보이기 시작한 첫 장)."""
    ev = []
    for f, i, other, dv in sc.impacts:
        if f + 1 < T:
            ev.append(dict(frame=f + 1, kind="impact", obj=int(i), other=other if isinstance(other, str) else int(other), dv=round(dv, 1)))
    for f, i, tag in sc.leaves:
        if f + 1 < T:
            ev.append(dict(frame=f + 1, kind="leave", obj=int(i), other=tag, dv=0.0))
    for i in range(len(sc.objs)):
        hid = ratio[:, i] < HIDDEN
        for s, e in _episodes(hid):
            if s > 0 and ratio[s - 1, i] >= HIDDEN and (ratio[max(0, s - 8):s, i] > 0.7).any():
                ev.append(dict(frame=s, kind="occlude", obj=i, other="", dv=0.0))
                if e + 1 < T:
                    ev.append(dict(frame=e + 1, kind="reappear", obj=i, other="", dv=0.0))
    return sorted(ev, key=lambda d: (d["frame"], d["kind"], d["obj"]))


def event_rows(sc, S, ratio, ms, events, rng, T, unpred_extra=(), min_ctx=4, min_pred=8):
    """boundary 행. **전환은 예측 구간에 둔다** (2026-10-18 사용자 결정: "충돌 직전까지만 context 에 보여주고 그 다음 충돌하고
    멈추거나 그런 걸 예측 / 아니면 딱 충돌 시점까지만 context"). e["frame"] = 전환 뒤 첫 장 (post). b = 예측 첫 장 = post + j,
    j ∈ {−2, −1, 0} → 문맥이 전환 2 · 1 · 0 장 전에서 끝남, j = +1 → 문맥 마지막 장이 딱 전환 순간.
    ctx_start = 문맥이 시작해도 되는 가장 이른 장 = 이 행보다 앞선 **충돌 (dv ≥ DV_ROW, 방해 물체 제외) · 받침 이탈** 의 post.
    가림 · 다시 보임은 문맥에 있어도 된다 (사용자: "permanence 는 context 에서 가려져도 되고 여러 번 가려져도 돼").
    로더 (var_window boundary) 가 C ≤ (b − ctx_start) / s 로 거른다.
    end = 예측 못 하는 시작 (멈춰 있던 가림막 · 컵의 출발, 화면 밖 물체가 처음 보이는 장, 떠다니는 가림막 진입) 직전."""
    pos = S["pos"][:T]
    inframe = _inframe(pos, -5)
    gate = sorted({e["frame"] for e in events if (e["kind"] == "leave" or (e["kind"] == "impact" and e["dv"] >= DV_ROW))
                   and sc.objs[e["obj"]].role != "distractor"})
    cand = []
    imp = [e for e in events if e["kind"] == "impact" and e["dv"] >= DV_ROW and inframe[e["frame"], e["obj"]]
           and sc.objs[e["obj"]].role != "distractor"]
    if len(imp) > MAX_IMPACT_ROWS:
        w = np.array([1.0 if (isinstance(e["other"], str) and e["other"] in LOW_TAGS) else IMPACT_PRIORITY for e in imp])
        imp = [imp[j] for j in sorted(rng.choice(len(imp), MAX_IMPACT_ROWS, replace=False, p=w / w.sum()))]
    cand += imp
    last = {}
    for e in events:
        if e["kind"] != "leave":
            continue
        post, i = e["frame"], e["obj"]
        if not (1 <= post < T - 4 and inframe[post, i]):
            continue
        if pos[post + 4, i, 1] - pos[post, i, 1] < 0.5 * sc.objs[i].bound:
            continue
        if post - last.get((i, e["other"]), -99) < 8:
            continue
        last[(i, e["other"])] = post
        cand.append(e)
    cand += [e for e in events if e["kind"] in ("occlude", "reappear")]
    unpred = [u for u in range(3, T) if ms[u] > MOVE_THR and (ms[u - 3:u] <= MOVE_THR).all()] + list(sc.spawn_frames) + list(unpred_extra)
    for i, o in enumerate(sc.objs):            # 화면 밖에서 생긴 물체가 처음 보이는 장 — 생성이 문맥 안이어도 문맥엔 안 보인다
        if o.role == "spawn":
            seen = np.where(ratio[:T, i] > HIDDEN)[0]
            if len(seen):
                unpred.append(int(seen[0]))
    sizes = np.array([o.bound for o in sc.objs], np.float32)
    rows, used = [], set()
    for e in sorted(cand, key=lambda d: d["frame"]):
        b = e["frame"] + int(rng.integers(-2, 2))
        end = min([u for u in unpred if u >= b] + [T])
        ctx_start = max([p for p in gate if p < b and p < e["frame"]] + [0])
        if b - ctx_start < min_ctx or end - b < min_pred or b in used:
            continue
        q = min(end, b + 8) - 1
        d = np.linalg.norm(pos[q] - pos[b], axis=-1) * (inframe[b] & inframe[q])
        if not ((d > 0.5 * sizes).any() or ms[b:q + 1].max() * (q - b) / FPS > 8):
            continue
        used.add(b)
        rows.append(dict(boundary=int(b), end=int(end), ctx_start=int(ctx_start), event=e["kind"], event_frame=int(e["frame"]), obj=int(e["obj"]),
                         obj_role=sc.objs[e["obj"]].role, other=str(e["other"]), dv=float(e["dv"])))
    return rows


# ----------------------------------------------------------------------------- 렌더
def draw_obj(img, o, p, a):
    if o.kind == "circle":
        R = _r(o.size)
        cv2.circle(img, _p(*p), R, o.color, -1, cv2.LINE_AA, SHIFT)
        cv2.ellipse(img, _p(*p), (R, R), math.degrees(a), 0, 180, shade(o.color, 0.62), -1, cv2.LINE_AA, SHIFT)   # 회전이 보이게
        cv2.circle(img, _p(*p), R, shade(o.color, 0.5), SS, cv2.LINE_AA, SHIFT)
    else:
        vs = world_verts(o, p, a)
        draw_poly(img, vs, o.color)
        c = (p[0] + (vs[0][0] - p[0]) * 0.55, p[1] + (vs[0][1] - p[1]) * 0.55)
        cv2.circle(img, _p(*c), _r(o.size * 0.26), shade(o.color, 0.55), -1, cv2.LINE_AA, SHIFT)


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


def render_frames(sc, bg, S, rects, T, string_color):
    frames = np.empty((T, H, W, 3), np.uint8)
    pos, ang, present = S["pos"], S["ang"], S["present"]
    for f in range(T):
        img = bg.copy()
        for a, j, kind in sc.strings:
            pa = tuple(a) if isinstance(a, tuple) else tuple(pos[f, a])
            pb = tuple(pos[f, j])
            if kind == "rod":
                cv2.line(img, _p(*pa), _p(*pb), string_color, SS, cv2.LINE_AA, SHIFT)
            else:
                draw_spring(img, pa, pb, string_color)
            if isinstance(a, tuple):
                cv2.circle(img, _p(*pa), _r(3), string_color, -1, cv2.LINE_AA, SHIFT)
        for i, o in enumerate(sc.objs):
            if present[f, i] and _touch(pos[f, i], o.bound, (0, 0, W, H), pad=2):
                draw_obj(img, o, pos[f, i], float(ang[f, i]))
        for j, mv in enumerate(sc.movers):
            draw_pattern_rect(img, *rects[f, j], mv.color, mv.pattern)
        frames[f] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return frames


def write_mp4(path, frames, crf=18):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(VIDEO_FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-threads", "1",
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
    if path_gif:
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "8", "-i", "-",
                              "-vf", "split[a][b];[a]palettegen[p];[b][p]paletteuse", "-threads", "1", path_gif], stdin=subprocess.PIPE)
        p.stdin.write(frames.tobytes())
        p.stdin.close()
        p.wait()


def overlay_labels(frames, L, names):
    """라벨 검증용 — 라벨 위치 (가려져 있으면 빈 원) · 속도 화살표 (×4 장) · 정체를 그린다."""
    out = frames.copy()
    for f in range(len(out)):
        img = out[f]
        for i in range(L["pos"].shape[1]):
            if not L["present"][f, i] or not np.isfinite(L["pos"][f, i]).all():
                continue
            x, y = L["pos"][f, i]
            vx, vy = L["vel"][f, i]
            col = (255, 255, 255) if L["vis_frac"][f, i] > 0.3 else (255, 0, 0)
            cv2.circle(img, (int(x), int(y)), 3, col, -1 if L["vis_frac"][f, i] > 0.3 else 1)
            cv2.line(img, (int(x), int(y)), (int(x + 4 * vx), int(y + 4 * vy)), col, 1)
            if f % 8 == 0:
                cv2.putText(img, names[i], (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)
    return out


# ----------------------------------------------------------------------------- 한 편
def ident_name(ident):
    return f"{COLORS[ident % NC][0]}_{SHAPES[ident // NC]}"


def build(name, seed, attempt, first_ident):
    _, _, fn = SCENARIOS[name]
    rng = np.random.default_rng([seed, attempt])
    ids = IdPool(np.random.default_rng([seed, attempt, 9]), first_ident, g=np.random.default_rng([seed, 5]).uniform(350, 900))
    bg, bgc, bg_kind = make_background(rng)
    sc = fn(rng, bgc, ids)
    if name != "cup_cover" and rng.random() < P_DISTRACTOR:
        add_distractor(sc, rng)
    add_extras(sc, rng)
    # spawn 여부는 재시도와 무관하게 정한다 (원판 점검: 설계 50 % → rejection 뒤 76.5 %). 다만 그러면 한 번 사건 시나리오의 15 % 가
    # 200 회 안에 통과 못 하고, 실패가 모양과 얽혀 (원이 굴러서 잘 통과) 주 물체 층화가 깨진다 → 100 회째부터는 시도마다 다시 뽑는다
    p_sp = 0.3 if name in ("pendulum", "spring", "osc_behind") else P_SPAWN
    if (np.random.default_rng([seed, 6]).random() if attempt < SPAWN_FIXED_ATTEMPTS else rng.random()) < p_sp:
        add_spawns(sc, rng, int(rng.integers(1, 3)), x_range=sc.params.get("bin_x"))
    sc.params["n_spawn"] = len(sc.spawns)
    sc.params["n_distractor"] = sum(o.role == "distractor" for o in sc.objs)
    for pts, color, _ in sc.statics:          # 정적 지형은 배경에 한 번 그린다
        draw_poly(bg, pts, color)
    return rng, sc, bg, bgc, bg_kind


def generate(name, seed, first_ident, max_attempts=200, log=None):
    why = []
    for attempt in range(max_attempts):
        if log is not None and attempt:
            log.append(list(why))            # 진단용: 앞 시도가 떨어진 이유
        rng, sc, bg, bgc, bg_kind = build(name, seed, attempt, first_ident)
        S = simulate(sc)
        if sc.auto_occ:
            ms0 = mover_speed(mover_rects(sc, S, N_FRAMES), [mv.origin == "scenario" for mv in sc.movers])
            rects = place_auto_screens(sc, np.random.default_rng([seed, attempt, 1]), S["pos"], trim_len(S, ms0))
            if rects is None:
                why = ["no occluder slot"]
                continue
            r2 = np.random.default_rng([seed, attempt, 2])
            for r in rects:
                sc.movers.append(_screen(r2, sc, (r[0] + r[2]) / 2, (r[1] + r[3]) / 2, r[2] - r[0], r[3] - r[1], origin="auto"))
            if sc.auto_occ["decoy"]:
                sc.need_occlusion = []
        rects = mover_rects(sc, S, N_FRAMES)
        ms = mover_speed(rects, [mv.origin in ("scenario", "auto") for mv in sc.movers])
        T = trim_len(S, ms)
        if T < MIN_FRAMES:
            why = [f"too short {T}"]
            continue
        seg, vis_px, amo_px = rasterize(sc, S, rects, T)
        ratio = np.where(amo_px > 0, vis_px / np.maximum(amo_px, 1), 0.0)       # 화면 안 부분 중 보이는 비율
        ok, why, metrics = check(sc, S, ratio, ms[:T], T)
        if not ok:
            continue
        events = find_events(sc, ratio, T)
        drift_entry = []
        rows = event_rows(sc, S, ratio, ms, events, np.random.default_rng([seed, attempt, 3]), T, drift_entry)
        if not any(r["boundary"] - r["ctx_start"] >= 16 and r["end"] - r["boundary"] >= 16 for r in rows):
            why = ["no event row C16K16"]
            continue
        metrics["n_rows"] = len(rows)
        return True, [], dict(rng=rng, sc=sc, bg=bg, bgc=bgc, bg_kind=bg_kind, S=S, rects=rects[:T], T=T, seg=seg, vis_px=vis_px,
                              amo_px=amo_px, ratio=ratio, events=events, rows=rows, attempt=attempt, metrics=metrics)
    return False, why, None


def build_labels(sc, S, T, vis_px, amo_px, ratio, events):
    n = len(sc.objs)
    present = S["present"][:T].copy()
    nan = lambda a: np.where(present[..., None] if a.ndim == 3 else present, a, np.nan).astype(np.float32)
    pos, vel = S["pos"][:T].astype(np.float32), S["vel"][:T].astype(np.float32)
    acc = np.zeros_like(vel)                                   # 중앙 차분 (끝 장은 한쪽 차분), 충돌 장에서는 크다
    if T > 2:
        acc[1:-1] = (vel[2:] - vel[:-2]) / 2
        acc[0], acc[-1] = vel[1] - vel[0], vel[-1] - vel[-2]
    area = np.array([o.area for o in sc.objs], np.float32)
    vis_frac = np.clip(vis_px / area, 0, 1)
    inframe_frac = np.clip(amo_px / area, 0, 1)
    hidden = present & (inframe_frac > 0.25) & (ratio < HIDDEN)
    seen = np.maximum.accumulate(vis_frac > SEEN, axis=0)
    bbox = np.zeros((T, n, 4), np.float32)
    for i, o in enumerate(sc.objs):
        if o.kind == "circle":
            bbox[:, i] = np.concatenate([pos[:, i] - o.size, pos[:, i] + o.size], -1)
        else:
            v = np.asarray(o.verts, np.float32)
            a = S["ang"][:T, i]
            c, s = np.cos(a)[:, None], np.sin(a)[:, None]
            wx = pos[:, i, 0:1] + v[None, :, 0] * c - v[None, :, 1] * s
            wy = pos[:, i, 1:2] + v[None, :, 0] * s + v[None, :, 1] * c
            bbox[:, i] = np.stack([wx.min(1), wy.min(1), wx.max(1), wy.max(1)], -1)
    impact = np.zeros((T, n), np.float32)
    for e in events:
        if e["kind"] == "impact":
            impact[e["frame"], e["obj"]] = max(impact[e["frame"], e["obj"]], e["dv"] / FPS)
    acc_n = acc
    return dict(pos=nan(pos), vel=nan(vel), acc=nan(acc_n), angle=nan(S["ang"][:T]), omega=nan(S["om"][:T]), bbox=nan(bbox),
                present=present, vis_frac=vis_frac.astype(np.float32), inframe_frac=inframe_frac.astype(np.float32),
                hidden=hidden, seen=seen, impact=impact,
                identity=np.array([o.ident for o in sc.objs], np.int16), shape_id=np.array([o.ident // NC for o in sc.objs], np.int16),
                color_id=np.array([o.ident % NC for o in sc.objs], np.int16), size=np.array([o.size for o in sc.objs], np.float32),
                area=area)


def flip_all(frames, seg, L, rects, statics_pts, events):
    frames, seg = frames[:, :, ::-1].copy(), seg[:, :, ::-1].copy()
    L = dict(L)
    for k in ("pos", "vel", "acc"):
        L[k] = L[k].copy()
        L[k][..., 0] = (W - L[k][..., 0]) if k == "pos" else -L[k][..., 0]
    L["angle"], L["omega"] = -L["angle"], -L["omega"]
    b = L["bbox"].copy()
    L["bbox"] = np.stack([W - b[..., 2], b[..., 1], W - b[..., 0], b[..., 3]], -1)
    r = rects.copy()
    rects = np.stack([W - r[..., 2], r[..., 1], W - r[..., 0], r[..., 3]], -1) if r.size else r
    statics_pts = [[(W - x, y) for x, y in pts] for pts in statics_pts]
    return frames, seg, L, rects, statics_pts


def make_clip(job):
    name, k, seed, first_ident, out, preview, overlay = job
    clip = f"{name}_{k:05d}"
    d = os.path.join(out, "clips", name)
    jpath = os.path.join(d, clip + ".json")
    if os.path.exists(jpath) and os.path.exists(os.path.join(d, clip + "_seg.npz")):      # 이어 하기
        with open(jpath) as fh:
            return json.load(fh)["result"]
    category = SCENARIOS[name][0]
    try:
        ok, why, R = generate(name, seed, first_ident)
    except Exception as e:          # 한 편의 예외로 전체가 멈추지 않게 — failed.tsv 에 남긴다
        ok, why, R = False, [f"exception {type(e).__name__}: {e}"], None
    if not ok:
        return dict(name=name, k=k, ok=False, why=why, first_ident=int(first_ident))
    sc, S, T, rng = R["sc"], R["S"], R["T"], R["rng"]
    string_color = shade(R["bgc"], 0.35) if np.mean(R["bgc"]) > 110 else (225, 225, 225)
    frames = render_frames(sc, R["bg"], S, R["rects"], T, string_color)
    L = build_labels(sc, S, T, R["vis_px"], R["amo_px"], R["ratio"], R["events"])
    statics_pts = [pts for pts, _, _ in sc.statics]
    rects, seg = R["rects"], R["seg"]
    flipped = bool(rng.random() < 0.5)
    if flipped:
        frames, seg, L, rects, statics_pts = flip_all(frames, seg, L, rects, statics_pts, R["events"])
    os.makedirs(d, exist_ok=True)
    mp4 = os.path.abspath(os.path.join(d, clip + ".mp4"))
    write_mp4(mp4, frames)
    np.savez_compressed(os.path.join(d, clip + "_states.npz"), occluder_rect=rects.astype(np.float32), **L)
    m = R["metrics"]
    m["copy_l1_1"] = float(np.abs(frames[1:].astype(np.int16) - frames[:-1]).mean())
    m["copy_l1_delta"] = float(np.abs(frames[8:].astype(np.int16) - frames[:-8]).mean())
    main = next((i for i, o in enumerate(sc.objs) if o.ident == first_ident), 0)
    result = dict(name=name, k=k, ok=True, mp4=mp4, category=category, attempt=R["attempt"], n_frames=int(T), rows=R["rows"],
                  n_objects=len(sc.objs), main_obj=int(main), main_identity=int(sc.objs[main].ident),
                  main_shape=sc.objs[main].kind, main_color=COLORS[sc.objs[main].ident % NC][0], first_ident=int(first_ident),
                  n_occluders=len(sc.movers), flipped=flipped, g=float(sc.g), **m)
    meta = dict(clip=clip, scenario=name, category=category, seed=int(seed), attempt=R["attempt"], sim_fps=FPS, video_fps=VIDEO_FPS,
                n_frames=int(T), size=[W, H], g_px_s2=float(sc.g), g_px_frame2=float(sc.g / FPS ** 2), floor_y=float(sc.floor_y),
                background=R["bg_kind"], flipped_lr=flipped, preroll_s=float(S["t0"]),
                objects=[dict(index=i, identity=o.ident, name=ident_name(o.ident), shape=o.kind, color=COLORS[o.ident % NC][0],
                              color_rgb=list(o.color), size=o.size, bound=o.bound, area=o.area, role=o.role, mass=float(o.body.mass),
                              elasticity=float(o.shape.elasticity), friction=float(o.shape.friction)) for i, o in enumerate(sc.objs)],
                occluders=[dict(index=j, kind=mv.kind, origin=mv.origin, w=mv.w, h=mv.h, pattern=mv.pattern, color_rgb=list(mv.color))
                           for j, mv in enumerate(sc.movers)],
                statics=[dict(tag=tag, pts=[[float(x), float(y)] for x, y in pts2]) for (pts, _, tag), pts2 in zip(sc.statics, statics_pts)],
                strings=[dict(anchor=list(a) if isinstance(a, tuple) else int(a), obj=int(j), kind=kind) for a, j, kind in sc.strings],
                events=R["events"], event_rows=R["rows"], params={k2: v for k2, v in sc.params.items() if k2 != "bg_color"}, metrics=m,
                result=result)
    if flipped:
        for s in meta["strings"]:
            if isinstance(s["anchor"], list):
                s["anchor"] = [W - s["anchor"][0], s["anchor"][1]]
    np.savez_compressed(os.path.join(d, clip + "_seg.npz"), seg=seg)
    with open(jpath, "w") as fh:
        json.dump(meta, fh, indent=1, default=float)
    if preview:
        pd = os.path.join(out, "_preview")
        os.makedirs(pd, exist_ok=True)
        title = (f"{name} [{category}] g={sc.g:.0f} main={ident_name(sc.objs[main].ident)} occ_ep={m['n_occ_episodes']} "
                 f"lin_err={m['linear_extrap_err_px']:.1f}px")
        write_preview(os.path.join(pd, f"{clip}.png"), os.path.join(pd, f"{clip}.gif"), frames, title)
        if overlay:
            ov = overlay_labels(frames, L, [ident_name(o.ident) for o in sc.objs])
            write_preview(os.path.join(pd, f"{clip}_labels.png"), os.path.join(pd, f"{clip}_labels.gif"), ov, title + " [labels]")
    return result


def _init_worker():
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/phys2d_sup_samples")
    ap.add_argument("--per-scenario", type=int, default=2)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, min(44, (os.cpu_count() or 8) - 4)))
    ap.add_argument("--preview-n", type=int, default=2, help="시나리오마다 미리보기를 만들 편 수")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--debug-overlay", action="store_true", help="미리보기에 라벨 (위치 · 속도 · 정체) 을 겹친 그림도")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for n, (c, d, _) in SCENARIOS.items():
            print(f"{c:11s} {n:14s} {d}")
        return
    names = list(SCENARIOS) if a.scenarios == "all" else a.scenarios.split(",")
    jobs = []
    for i, n in enumerate(names):
        si = list(SCENARIOS).index(n)
        perm = np.random.default_rng([a.seed, si, 4242]).permutation(N_ID)     # 주 물체 정체 층화 (48 칸을 돌아가며)
        for k in range(a.per_scenario):
            jobs.append((n, k, a.seed * 1_000_003 + si * 10_007 + k, int(perm[k % N_ID]), a.out,
                         (not a.no_preview) and k < a.preview_n, a.debug_overlay))
    jobs.sort(key=lambda j: (j[1], j[0]))     # 시나리오를 섞어 돌린다 (무거운 시나리오가 끝에 몰리지 않게)
    os.makedirs(a.out, exist_ok=True)
    res = []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init_worker) as ex:
        for fut in as_completed([ex.submit(make_clip, j) for j in jobs]):
            r = fut.result()
            res.append(r)
            if not r["ok"]:
                print(f"FAIL {r['name']}_{r['k']:05d} {ident_name(r['first_ident'])} {r['why']}", flush=True)
            if len(res) % 500 == 0 or len(res) == len(jobs):
                el = time.time() - t0
                print(f"[{len(res):,}/{len(jobs):,}] {el / 60:.1f} 분, 남은 {el / len(res) * (len(jobs) - len(res)) / 60:.1f} 분", flush=True)
    res.sort(key=lambda r: (r["name"], r["k"]))
    good = [r for r in res if r["ok"]]
    with open(os.path.join(a.out, "failed.tsv"), "w") as fh:
        fh.write("name\tk\tfirst_identity\twhy\n")
        fh.writelines(f"{r['name']}\t{r['k']}\t{ident_name(r['first_ident'])}\t{'; '.join(r['why'])}\n" for r in res if not r["ok"])
    with open(os.path.join(a.out, "index.csv"), "w") as fh:
        fh.writelines(f"{r['mp4']} 0\n" for r in good)
    with open(os.path.join(a.out, "fps4_table.tsv"), "w") as fh:
        fh.writelines(f"{r['mp4']}\t{VIDEO_FPS}\t{r['n_frames']}\t1\t{r['n_frames']}\n" for r in good)
    bcols = ["path", "n_frames", "boundary_frame", "ctx_start", "video_frames", "event", "event_frame", "obj_role", "other", "dv", "scenario", "category"]
    nb = 0
    with open(os.path.join(a.out, "boundary.tsv"), "w") as fh:
        fh.write("\t".join(bcols) + "\n")
        for r in good:
            for w in r["rows"]:
                fh.write("\t".join(str(v) for v in [r["mp4"], w["end"], w["boundary"], w["ctx_start"], r["n_frames"], w["event"], w["event_frame"],
                                                    w["obj_role"], w["other"], w["dv"], r["name"], r["category"]]) + "\n")
                nb += 1
    print(f"boundary 행 {nb:,} (영상당 {nb / max(1, len(good)):.1f})")
    cols = ["category", "name", "k", "n_frames", "attempt", "n_objects", "main_shape", "main_color", "main_identity", "n_occluders", "g",
            "flipped", "motion_frac", "displaced_frac", "still_tail_frac", "copy_l1_1", "copy_l1_delta", "pos_copy_err_px",
            "linear_extrap_err_px", "n_hidden_frames", "n_occ_episodes", "max_obj_episodes", "n_rows"]
    with open(os.path.join(a.out, "manifest.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in good:
            fh.write("\t".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"\n{len(good)}/{len(res)} 통과 → {a.out}/index.csv  ({(time.time() - t0) / 60:.1f} 분)")


if __name__ == "__main__":
    sys.exit(main())
