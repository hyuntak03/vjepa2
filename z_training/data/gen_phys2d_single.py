#!/usr/bin/env python3
"""2D 물체 1 개 학습셋 + 프레임별 상태 라벨 — 중력 · 영속 · 고체성 (2026-10-19, CPU 전용).

`gen_phys2d_sup.py` (phys2d_sup_v2) 를 import 해서 쓰고, 바뀐 것만 여기 둔다 (렌더 · 라벨 · 검사 · 사건 행은 그대로):
  * **물체는 언제나 1 개.** 방해 물체 · 화면 밖에서 들어오는 물체 · 이중 진자 · 물체끼리 충돌 · 쌓기 시나리오를 뺐다.
    고체성은 정적 지형 (벽 · 장애물 · 선반 · 통 · 받침) 과의 충돌로만 배운다.
  * **정체 편향 없음** — 주 물체 정체는 시나리오마다 48 칸 (모양 6 × 색 8) 에 층화 배정 (gen_phys2d_sup 과 같다).
    크기 · 속도 · 위치는 정체와 무관하게 뽑고, 모든 시나리오에서 모든 모양을 쓴다.
  * **가림 시점에 편향 없음** (사용자 2026-10-19: "초기에 가려져도 되고 나중에 가려져도 되고 둘 다 … 시간에 대한 편향도 없도록").
    1 차 궤적을 본 뒤 가림 구간 n 개 (0~4, clip 마다 고정) 를 **시작 장 a ~ U[−L+1, T−1]** 로 뽑아 그 구간 궤적을 덮는 고정 가림막을 놓는다
    → 장 t 가 가림 구간에 들어갈 확률이 t 와 무관하다 (가장자리 포함). 첫 장부터 가려진 채 시작하는 영상도 있다.
    시나리오 고유 가림막 (moving_screen · cup · drop_behind · tunnel · osc_behind) 은 그 위에 더해진다.
  * **경계 행 두 벌** — `boundary.tsv` (사건 경계, phys2d_sup 과 같은 규칙) + `uniform.tsv` (경계 = 영상 안 고른 자리, ctx_start 0).
    상태 head 를 블록 단위로 걸 때 창 안 시점 편향이 없게 하려면 `uniform.tsv` 를 쓴다 (사건 행은 가림 · 충돌이 창의 정해진 자리에 온다).
  * C16K16 사건 행 요구를 뺐다 — 사건 행이 없어도 상태 라벨은 모든 장에 있다 (그 요구가 가림 없는 진자 영상을 떨어뜨려 가림 수와 시나리오를 얽는다).

  /venv/vjepa/bin/python z_training/data/gen_phys2d_single.py --list
  /venv/vjepa/bin/python z_training/data/gen_phys2d_single.py --per-scenario 4 --out /tmp/p2one --debug-overlay
  /venv/vjepa/bin/python z_training/data/gen_phys2d_single.py --per-scenario 2200 --seed 1 --workers 40 --out <root>/train

좌표 · 시간 · 라벨 키는 gen_phys2d_sup 과 같다 (px, px/장, 12 장/초 시뮬, mp4 는 4 fps 로 적는다). 물체 축 길이는 늘 1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_phys2d_sup as G                                   # noqa: E402
from gen_phys2d_sup import (H, W, N_FRAMES, HIDDEN, SEEN, N_ID, NC, SHAPES, COLORS,   # noqa: E402
                            _base, _mc, _overlaps, _inframe, _screen, add_obj, add_static_poly, add_pendulum, add_spring, place_on,
                            roll_angvel, shade)

P_EPISODES = (0.08, 0.27, 0.30, 0.22, 0.13)   # 가림 구간 0 · 1 · 2 · 3 · 4 개 (clip 마다 고정 — 시도 번호와 무관)
HIDE_LEN = (2, 16)                             # 구간 길이 (장)
SCREEN_MAX = (160.0, 170.0)                    # 가림막 최대 폭 · 높이 (px)
P_COVER_REST = 0.3                             # 마지막 자리 (= 멈춘 자리) 를 덮는 가림막을 받을 확률. 1 이면 끝 장 가림이 70~88 % (시험 2026-10-19) → "가려짐 = 멈춤" 지름길
MAX_HIDDEN_FRAC = 0.7                          # 영상에서 완전히 가려진 장 비율 상한
SCREEN_TRIES = 12
MIN_T = 40                                     # 물체 하나가 멈추면 장면이 멈춘다 — 64 장 (sup 판) 이면 절반이 떨어졌다 (2026-10-19 시험 51/108). 창 C16+K16 = 32 장
TAIL = 16                                      # 멈춘 뒤 남기는 장 (sup 판 4). 가려진 채 멈춰 있는 물체도 영속 감독 재료다
MOTION_MIN, DISPLACED_MIN = 0.6, 0.5           # sup 판 0.7 / 0.6 — 꼬리가 길어진 만큼 낮춘다
UNIFORM_STEP = 4                               # uniform.tsv 경계 간격 (장)
UNIFORM_MIN_CTX, UNIFORM_K = 4, 16

SCENARIOS = {}


def scenario(name, category, desc, fn=None):
    def deco(f):
        SCENARIOS[name] = (category, desc, f)
        return f
    return deco(fn) if fn is not None else deco


def _one_drop(sc, rng, size_range=(8, 16)):
    return add_obj(sc, rng, float(rng.uniform(*size_range)), (rng.uniform(30, W - 30), rng.uniform(15, sc.floor_y - 90)),
                   (rng.normal(0, 60), rng.uniform(-150, 30)), elasticity=float(rng.uniform(0.55, 0.9)))


# --- 중력 -------------------------------------------------------------------------
@scenario("drop_bounce", "gravity", "물체 하나를 여러 높이에서 놓거나 던짐 — 자유 낙하 · 튕김")
def sc_drop(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    _one_drop(sc, rng)
    sc.preroll = float(rng.uniform(0, 0.3))
    return sc


scenario("projectile", "gravity", "옆에서 던진 물체의 포물선 — 속도 · 각도 무작위, 바닥 · 벽에서 튕김", G.sc_proj)


@scenario("toss_up", "gravity", "위로 던진 물체가 꼭대기에서 멈췄다 되돌아 떨어짐")
def sc_toss(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    size = float(rng.uniform(8, 15))
    x, y = float(rng.uniform(40, W - 40)), float(sc.floor_y - 1.5 * size - rng.uniform(0, 50))
    h = float(rng.uniform(50, max(60.0, y - 2 * size - 15)))
    add_obj(sc, rng, size, (x, y), (float(rng.normal(0, 45)), -math.sqrt(2 * sc.g * h)), elasticity=float(rng.uniform(0.3, 0.8)))
    sc.require.append(("apex", 0))
    sc.preroll = float(rng.uniform(0, 0.15))
    return sc


scenario("ramp_roll", "gravity", "경사면을 굴러 · 미끄러져 내려옴 (등가속) — 끝에서 바닥 · 벽", G.sc_ramp)
scenario("ledge_fall", "gravity", "받침 위를 가다 끝에서 떨어짐 (중력 개시)", G.sc_ledge)


@scenario("pendulum", "gravity", "단진자 하나 — 줄 길이 · 진폭 · 위상 무작위")
def sc_pend(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    px, py = float(rng.uniform(50, W - 50)), float(rng.uniform(10, 50))
    L = float(rng.uniform(60, min(150, sc.floor_y - py - 30)))
    add_pendulum(sc, rng, px, py, L, float(rng.uniform(0.5, 1.3)) * rng.choice([-1, 1]), float(rng.uniform(8, 14)))
    sc.preroll = float(rng.uniform(0, 2.0))
    return sc


@scenario("spring", "gravity", "용수철에 매단 물체 하나 — 수직 진동 또는 탄성 진자")
def sc_spring(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    add_spring(sc, rng, float(rng.uniform(60, W - 60)), float(rng.uniform(8, 30)), float(rng.uniform(9, 15)))
    sc.preroll = float(rng.uniform(0, 1.5))
    return sc


# --- 영속 (시나리오 고유 가림막. 시간 고른 가림막은 모든 시나리오에 따로 붙는다) -----------------
scenario("osc_behind", "permanence", "진자 · 용수철 · 왕복 물체가 가림막 뒤를 여러 번 드나듦", G.sc_osc)
scenario("moving_screen", "permanence", "가림막이 오가며 물체를 여러 번 가림 — 물체는 정지 · 진자 · 굴러감", G.sc_screen)
scenario("cup_cover", "permanence", "컵이 내려와 물체를 덮고 (밀고 갈 때도 있음) 다시 들림", G.sc_cup)   # 방해 물체는 아래에서 끈다
scenario("drop_behind", "permanence", "물체가 가림막 뒤로 떨어져 멈춤, 그다음 가림막이 비켜나며 드러남", G.sc_drop_behind)
scenario("tunnel", "permanence", "굴러가는 물체가 터널로 들어가 반대편으로 나옴", G.sc_tunnel)


# --- 고체성 (정적 지형과의 충돌만) ------------------------------------------------------
@scenario("shelf_land", "solidity", "선반 · 판 1~3 개 위로 떨어져 그 위에 서거나 튕겨 내려감 — 통과하지 않는다")
def sc_shelf(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    for _ in range(int(rng.integers(1, 4))):
        x0, w, y = float(rng.uniform(0, W - 70)), float(rng.uniform(50, 110)), float(rng.uniform(80, sc.floor_y - 30))
        ang = float(rng.uniform(-0.35, 0.35)) if rng.random() < 0.5 else 0.0
        dx, dy = w * math.cos(ang), w * math.sin(ang)
        add_static_poly(sc, [(x0, y), (x0 + dx, y + dy), (x0 + dx, y + dy + 8), (x0, y + 8)], _mc(sc, rng),
                        elasticity=float(rng.uniform(0.2, 0.7)), tag="shelf")
    sc.require.append(("tag", "shelf"))
    add_obj(sc, rng, float(rng.uniform(8, 14)), (rng.uniform(20, W - 20), rng.uniform(5, 50)), (rng.normal(0, 30), rng.uniform(-80, 20)))
    sc.preroll = float(rng.uniform(0, 0.2))
    return sc


scenario("wall_bounce", "solidity", "던지거나 굴린 물체가 벽에 부딪혀 되튕김 — 벽 높이 · 위치 · 반발 무작위", G.sc_wall)
scenario("occ_wall", "solidity", "벽 아랫부분이 가림막에 가려 있고, 물체가 가림막 뒤에서 벽에 튕겨 되돌아 나옴", G.sc_occ_wall)


@scenario("container", "solidity", "통 (U 자) 안으로 떨어지거나 던져진 물체 — 벽 안에서 튕기고 갇힌다")
def sc_container(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    cc = _mc(sc, rng)
    cx, cw, ch = float(rng.uniform(90, 170)), float(rng.uniform(60, 110)), float(rng.uniform(40, 80))
    t, yb = 6.0, sc.floor_y
    for pts in ([(cx - cw / 2, yb - ch), (cx - cw / 2 + t, yb - ch), (cx - cw / 2 + t, yb), (cx - cw / 2, yb)],
                [(cx + cw / 2 - t, yb - ch), (cx + cw / 2, yb - ch), (cx + cw / 2, yb), (cx + cw / 2 - t, yb)]):
        add_static_poly(sc, pts, cc, elasticity=0.6, tag="bin")
    sc.require.append(("final_in", cx - cw / 2 + t, cx + cw / 2 - t, yb - ch))
    size = float(rng.uniform(7, 11))
    if rng.random() < 0.5:
        pos, vel = (cx + rng.uniform(-cw / 3, cw / 3), rng.uniform(10, 60)), (rng.normal(0, 60), rng.uniform(-60, 20))
    else:
        pos, vel = (rng.uniform(10, 40), sc.floor_y - rng.uniform(40, 100)), (rng.uniform(120, 200), -rng.uniform(150, 300))
    add_obj(sc, rng, size, pos, vel, elasticity=float(rng.uniform(0.6, 0.9)))
    return sc


@scenario("obstacle_bounce", "solidity", "방 안 장애물 (바닥 벽 · 떠 있는 블록) 1~3 개 사이를 잘 튀는 물체가 돌아다님 — 부딪히면 되튕긴다")
def sc_obstacle(rng, bgc, ids):
    sc = _base(rng, bgc, ids, walls=True)
    for _ in range(int(rng.integers(1, 4))):
        for _ in range(20):
            if rng.random() < 0.5:     # 바닥에서 선 벽
                w, h = float(rng.uniform(10, 24)), float(rng.uniform(40, 130))
                x0, y0 = float(rng.uniform(50, W - 50 - w)), sc.floor_y - h
                y1 = sc.floor_y
            else:                      # 떠 있는 블록
                w, h = float(rng.uniform(20, 60)), float(rng.uniform(12, 30))
                x0, y0 = float(rng.uniform(30, W - 30 - w)), float(rng.uniform(50, sc.floor_y - 70))
                y1 = y0 + h
            if _overlaps(sc, x0, y0, x0 + w, y1, pad=24):
                continue
            add_static_poly(sc, [(x0, y0), (x0 + w, y0), (x0 + w, y1), (x0, y1)], _mc(sc, rng),
                            elasticity=float(rng.uniform(0.6, 0.95)), tag="wall")
            break
    sc.require.append(("tag", "wall"))
    sp, ang = float(rng.uniform(150, 320)), float(rng.uniform(0, 2 * math.pi))
    add_obj(sc, rng, float(rng.uniform(8, 14)), (rng.uniform(25, W - 25), rng.uniform(20, sc.floor_y - 40)),
            (sp * math.cos(ang), sp * math.sin(ang)), elasticity=float(rng.uniform(0.8, 0.95)), friction=0.2)
    sc.preroll = float(rng.uniform(0, 0.2))
    return sc


@scenario("sandbox", "mixed", "지형 (경사 · 선반 · 벽 · 통) 2~4 개를 무작위로 놓고 물체 하나를 떨어뜨리기 · 던지기 · 굴리기")
def sc_sandbox(rng, bgc, ids):
    sc = _base(rng, bgc, ids)
    elems = list(rng.choice(["ramp", "shelf", "wall", "bin"], size=int(rng.integers(2, 5)), replace=False))
    for e in elems:
        for _ in range(20):
            if e == "ramp":
                top, run = float(rng.uniform(50, 120)), float(rng.uniform(80, 150))
                left = rng.random() < 0.5
                x0 = 0.0 if left else W - run
                if _overlaps(sc, x0, sc.floor_y - top, x0 + run, sc.floor_y - 1):
                    continue
                pts = ([(x0, sc.floor_y - top), (x0 + run, sc.floor_y), (x0, sc.floor_y)] if left
                       else [(x0, sc.floor_y), (W, sc.floor_y - top), (W, sc.floor_y)])
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
            else:
                cx, cw, ch = float(rng.uniform(60, W - 60)), float(rng.uniform(55, 95)), float(rng.uniform(35, 70))
                if _overlaps(sc, cx - cw / 2, sc.floor_y - ch, cx + cw / 2, sc.floor_y - 1):
                    continue
                c, yb = _mc(sc, rng), sc.floor_y
                for pts in ([(cx - cw / 2, yb - ch), (cx - cw / 2 + 6, yb - ch), (cx - cw / 2 + 6, yb), (cx - cw / 2, yb)],
                            [(cx + cw / 2 - 6, yb - ch), (cx + cw / 2, yb - ch), (cx + cw / 2, yb), (cx + cw / 2 - 6, yb)]):
                    add_static_poly(sc, pts, c, elasticity=0.6, tag="bin")
            break
    size = float(rng.uniform(8, 14))
    mode = str(rng.choice(["drop", "throw", "roll"]))
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
    sc.params.update(elements=[str(e) for e in elems], mode=mode)
    sc.require.append(("events_min", 1))
    sc.preroll = float(rng.uniform(0, 0.3))
    return sc


# ----------------------------------------------------------------------------- 시간 고른 가림막
def place_timed_screens(sc, rng, pos, T, n_ep):
    """가림 구간 n_ep 개. 구간 [a, a+L) 의 a ~ U[−L+1, T−1] (영상 밖은 잘라냄) → 장마다 구간에 들 확률이 같다.
    구간 동안 물체 (외접원) 가 화면 안에 있어야 하고 (화면 밖 = 가림이 아니다), 그 궤적을 모두 덮는 고정 가림막을 놓는다.
    가림막이 정적이라 물체가 그 자리를 다시 지나면 또 가려진다 — 라벨은 래스터로 다시 재므로 맞다."""
    b = sc.objs[0].bound
    p = pos[:T, 0]
    rects = []
    for _ in range(n_ep):
        for _ in range(40):
            L = int(rng.integers(HIDE_LEN[0], HIDE_LEN[1] + 1))
            a = int(rng.integers(-L + 1, T))
            s, e = max(0, a), min(T, a + L)
            seg = p[s:e]
            if e - s < 1 or not _inframe(seg, b).all():
                continue
            m0, m1 = rng.uniform(2, 10, 2), rng.uniform(2, 10, 2)
            r = [*(seg.min(0) - b - m0), *(seg.max(0) + b + m1)]
            if r[2] - r[0] > SCREEN_MAX[0] or r[3] - r[1] > SCREEN_MAX[1]:
                continue
            if r[3] > sc.floor_y - 2 * b - 4:
                r[3] = sc.floor_y + 1
            if G._touch(p[T - 1], b, r) and rng.random() > P_COVER_REST:
                continue
            if any(not (r[2] + 4 < q[0] or r[0] - 4 > q[2] or r[3] + 4 < q[1] or r[1] - 4 > q[3]) for q in rects):
                continue
            rects.append([float(v) for v in r])
            break
    return rects


# ----------------------------------------------------------------------------- 한 편
def build(name, seed, attempt, first_ident):
    _, _, fn = SCENARIOS[name]
    rng = np.random.default_rng([seed, attempt])
    ids = G.IdPool(np.random.default_rng([seed, attempt, 9]), first_ident, g=np.random.default_rng([seed, 5]).uniform(350, 900))
    bg, bgc, bg_kind = G.make_background(rng)
    sc = fn(rng, bgc, ids)
    assert len(sc.objs) == 1, (name, len(sc.objs))
    G.add_extras(sc, rng)                  # 기운 선반 · 궤적과 무관한 고정 가림막 · 떠다니는 가림막 (방해 물체 · spawn 은 없다)
    sc.params.update(n_spawn=0, n_distractor=0)
    for pts, color, _ in sc.statics:
        G.draw_poly(bg, pts, color)
    return rng, sc, bg, bgc, bg_kind


def _measure(sc, S, T):
    rects = G.mover_rects(sc, S, N_FRAMES)
    seg, vis_px, amo_px = G.rasterize(sc, S, rects, T)
    ratio = np.where(amo_px > 0, vis_px / np.maximum(amo_px, 1), 0.0)
    return rects, seg, vis_px, amo_px, ratio


def generate(name, seed, first_ident, max_attempts=200, log=None):
    """gen_phys2d_sup.generate 와 같은 흐름 + 시간 고른 가림막. 가림 구간 수는 clip 마다 고정 (rejection 이 가림 수를 고르지 않게).
    가림막만 바꿔 다시 재는 것은 같은 시도 안에서 (물리는 다시 안 돌린다)."""
    n_ep = int(np.random.default_rng([seed, 8]).choice(len(P_EPISODES), p=P_EPISODES))
    why = []
    for attempt in range(max_attempts):
        if log is not None and attempt:
            log.append(list(why))
        rng, sc, bg, bgc, bg_kind = build(name, seed, attempt, first_ident)
        S = G.simulate(sc)
        T0 = G.trim_len(S, G.mover_speed(G.mover_rects(sc, S, N_FRAMES), [mv.origin == "scenario" for mv in sc.movers]), tail=TAIL)
        if T0 < MIN_T:
            why = [f"too short {T0}"]
            continue
        if sc.auto_occ:                                     # osc_behind — 한 가림막이 여러 번 덮는다 (시나리오 정의)
            ar = G.place_auto_screens(sc, np.random.default_rng([seed, attempt, 1]), S["pos"], T0)
            if ar is None:
                why = ["no occluder slot"]
                continue
            r2 = np.random.default_rng([seed, attempt, 2])
            for r in ar:
                sc.movers.append(_screen(r2, sc, (r[0] + r[2]) / 2, (r[1] + r[3]) / 2, r[2] - r[0], r[3] - r[1], origin="auto"))
        base = list(sc.movers)
        rr = np.random.default_rng([seed, attempt, 4])
        ok = False
        for _ in range(SCREEN_TRIES):
            sc.movers = list(base)
            timed = place_timed_screens(sc, rr, S["pos"], T0, n_ep)
            if len(timed) < n_ep:
                why = [f"timed screens {len(timed)}/{n_ep}"]
                continue
            for r in timed:
                sc.movers.append(_screen(rr, sc, (r[0] + r[2]) / 2, (r[1] + r[3]) / 2, r[2] - r[0], r[3] - r[1], origin="timed"))
            ms = G.mover_speed(G.mover_rects(sc, S, N_FRAMES), [mv.origin in ("scenario", "auto") for mv in sc.movers])
            T = G.trim_len(S, ms, tail=TAIL)
            if T < MIN_T:
                why = [f"too short {T}"]
                break
            rects, seg, vis_px, amo_px, ratio = _measure(sc, S, T)
            area = sc.objs[0].area
            hidden = (amo_px[:, 0] / area > 0.25) & (ratio[:, 0] < HIDDEN)
            if hidden.mean() > MAX_HIDDEN_FRAC:
                why = [f"hidden_frac {hidden.mean():.2f}"]
                continue
            if not (vis_px[:, 0] / area > SEEN).any():
                why = ["never seen"]
                continue
            ok, why, metrics = check(sc, S, ratio, ms[:T], T)
            if ok:
                break
        if not ok:
            continue
        events = G.find_events(sc, ratio, T)
        rows = G.event_rows(sc, S, ratio, ms, events, np.random.default_rng([seed, attempt, 3]), T)
        hid_i = hidden.astype(int)
        metrics.update(n_rows=len(rows), n_ep_target=n_ep, n_timed_screens=len(timed), hidden_frac=float(hidden.mean()),
                       n_hidden_frames=int(hidden.sum()), n_hidden_spans=int(((hid_i[1:] - hid_i[:-1]) == 1).sum() + hid_i[0]),
                       hidden_at_start=bool(hidden[0]), hidden_at_end=bool(hidden[T - 1]))
        return True, [], dict(rng=rng, sc=sc, bg=bg, bgc=bgc, bg_kind=bg_kind, S=S, rects=rects[:T], T=T, seg=seg, vis_px=vis_px,
                              amo_px=amo_px, ratio=ratio, events=events, rows=rows, attempt=attempt, metrics=metrics)
    return False, why, None


def check(sc, S, ratio, ms, T):
    """G.check 에서 운동 · 변위 문턱만 이 판 값으로 바꾼다 (나머지 검사는 그대로)."""
    _, why, m = G.check(sc, S, ratio, ms, T)
    why = [w for w in why if not w.startswith(("motion_frac", "displaced"))]
    if m["motion_frac"] < MOTION_MIN:
        why.append(f"motion_frac {m['motion_frac']:.2f}")
    if m["displaced_frac"] < DISPLACED_MIN:
        why.append(f"displaced {m['displaced_frac']:.2f}")
    return not why, why, m


# gen_phys2d_sup 의 make_clip 이 이 모듈의 시나리오 · 생성을 쓰게 한다 (워커는 이 모듈을 import 하므로 패치가 따라간다)
G.SCENARIOS = SCENARIOS
G.generate = generate
G.add_distractor = lambda *a, **k: None        # cup_cover 가 부르는 방해 물체를 끈다
G.P_DRIFT_SCREEN = 0.2


SEED_RETRIES = 12                              # 4 이면 cup_cover 의 14 % 가 떨어지고 그 실패가 원 아닌 모양에 몰렸다 (v1 첫 실행 2026-10-19)


def make_clip(job):
    """실패하면 같은 정체로 seed 만 바꿔 다시 (정체 층화가 실패율과 얽히지 않게). 결과의 seed_retry 에 몇 번째인지 남긴다."""
    name, k, seed, first_ident, *rest = job
    for r in range(SEED_RETRIES):
        res = G.make_clip((name, k, seed + r * 7_919_007, first_ident, *rest))
        if res["ok"]:
            break
    if res["ok"]:       # 이어 하기로 읽은 편도 맞게 — 실제 쓴 seed 는 json 에 있다
        with open(res["mp4"][:-4] + ".json") as fh:
            r = (json.load(fh)["seed"] - seed) // 7_919_007
    res["seed_retry"] = int(r)
    return res


def uniform_rows(T, rng):
    """경계 = UNIFORM_STEP 간격 (clip 마다 시작 위치 무작위) 의 모든 자리, 문맥 4 장 이상 · 예측 16 장."""
    lo, hi = UNIFORM_MIN_CTX, T - UNIFORM_K
    return list(range(lo + int(rng.integers(0, UNIFORM_STEP)), hi + 1, UNIFORM_STEP))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/phys2d_single_samples")
    ap.add_argument("--per-scenario", type=int, default=2)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, min(44, (os.cpu_count() or 8) - 4)))
    ap.add_argument("--preview-n", type=int, default=2)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--debug-overlay", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for n, (c, d, _) in SCENARIOS.items():
            print(f"{c:11s} {n:16s} {d}")
        return
    names = list(SCENARIOS) if a.scenarios == "all" else a.scenarios.split(",")
    jobs = []
    for n in names:
        si = list(SCENARIOS).index(n)
        perm = np.random.default_rng([a.seed, si, 4242]).permutation(N_ID)        # 주 물체 정체 층화
        for k in range(a.per_scenario):
            jobs.append((n, k, a.seed * 1_000_003 + si * 10_007 + k, int(perm[k % N_ID]), a.out,
                         (not a.no_preview) and k < a.preview_n, a.debug_overlay))
    jobs.sort(key=lambda j: (j[1], j[0]))
    os.makedirs(a.out, exist_ok=True)
    res, t0 = [], time.time()
    with ProcessPoolExecutor(a.workers, initializer=G._init_worker) as ex:
        for fut in as_completed([ex.submit(make_clip, j) for j in jobs]):
            r = fut.result()
            res.append(r)
            if not r["ok"]:
                print(f"FAIL {r['name']}_{r['k']:05d} {G.ident_name(r['first_ident'])} {r['why']}", flush=True)
            if len(res) % 500 == 0 or len(res) == len(jobs):
                el = time.time() - t0
                print(f"[{len(res):,}/{len(jobs):,}] {el / 60:.1f} 분, 남은 {el / len(res) * (len(jobs) - len(res)) / 60:.1f} 분", flush=True)
    res.sort(key=lambda r: (r["name"], r["k"]))
    good = [r for r in res if r["ok"]]
    with open(os.path.join(a.out, "failed.tsv"), "w") as fh:
        fh.write("name\tk\tfirst_identity\twhy\n")
        fh.writelines(f"{r['name']}\t{r['k']}\t{G.ident_name(r['first_ident'])}\t{'; '.join(r['why'])}\n" for r in res if not r["ok"])
    with open(os.path.join(a.out, "index.csv"), "w") as fh:
        fh.writelines(f"{r['mp4']} 0\n" for r in good)
    with open(os.path.join(a.out, "fps4_table.tsv"), "w") as fh:
        fh.writelines(f"{r['mp4']}\t{G.VIDEO_FPS}\t{r['n_frames']}\t1\t{r['n_frames']}\n" for r in good)
    bcols = ["path", "n_frames", "boundary_frame", "ctx_start", "video_frames", "event", "event_frame", "obj_role", "other", "dv", "scenario", "category"]
    nb = nu = 0
    with open(os.path.join(a.out, "boundary.tsv"), "w") as fb, open(os.path.join(a.out, "uniform.tsv"), "w") as fu:
        fb.write("\t".join(bcols) + "\n")
        fu.write("\t".join(bcols) + "\n")
        for r in good:
            for w in r["rows"]:
                fb.write("\t".join(str(v) for v in [r["mp4"], w["end"], w["boundary"], w["ctx_start"], r["n_frames"], w["event"], w["event_frame"],
                                                    w["obj_role"], w["other"], w["dv"], r["name"], r["category"]]) + "\n")
                nb += 1
            for b in uniform_rows(r["n_frames"], np.random.default_rng([a.seed, r["k"], list(SCENARIOS).index(r["name"]), 77])):
                fu.write("\t".join(str(v) for v in [r["mp4"], r["n_frames"], b, 0, r["n_frames"], "uniform", b, "main", "", 0.0,
                                                    r["name"], r["category"]]) + "\n")
                nu += 1
    print(f"boundary 행 {nb:,} (영상당 {nb / max(1, len(good)):.1f}) · uniform 행 {nu:,} (영상당 {nu / max(1, len(good)):.1f})")
    cols = ["category", "name", "k", "n_frames", "attempt", "n_objects", "main_shape", "main_color", "main_identity", "n_occluders", "g",
            "flipped", "motion_frac", "displaced_frac", "linear_extrap_err_px", "n_ep_target", "n_timed_screens", "hidden_frac",
            "n_hidden_frames", "n_hidden_spans", "hidden_at_start", "hidden_at_end", "n_occ_episodes", "n_rows", "seed_retry"]
    with open(os.path.join(a.out, "manifest.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in good:
            fh.write("\t".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"\n{len(good)}/{len(res)} 통과 → {a.out}/index.csv  ({(time.time() - t0) / 60:.1f} 분)")


if __name__ == "__main__":
    sys.exit(main())
