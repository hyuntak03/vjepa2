"""phys3d_single v2 (2026-10-21) — 사건 창 (C · K ∈ {2, 4, 8, 16, 32}, stride 3) 이 영상 곳곳에서 나오게 · 이어지는 안 보임 상한.
문서 z_training/docs/PHYS3D_SINGLE_V2_DATASET_2026-10-21.md

v1 (gen_phys3d_single_blender.py) 을 import 해서 값 · 몇 함수만 바꾼다. v1 은 그대로 재현된다 — v1 모듈의 갈고리
(MAX_BLIND_RUN · TIMED_SLIDE · EP_REF_T · ROW_MIN_*) 는 v1 기본값이면 v1 동작과 같다 (시험 2026-10-21: v1 val 26 편 라벨 지문 26/26 일치).
  /workspace/tools/blender-4.2.3-linux-x64/blender -t 1 -b -noaudio -P z_training/data/gen_phys3d_single_v2_blender.py -- \
      --scenario wall_bounce --seed 0 --frames 288 --out /path/to/out
대량 생성은 gen_phys3d.py --single-v2 --stride 3 --frames 288.

사용자 (2026-10-21): "문맥 C 와 예측 K 를 각각 2 · 4 · 8 · 16 · 32 장 … 가림은 문맥 끝이 가려지는 걸로 끝나고 예측에서 언젠가 다시 나와야 …
event transition 을 prediction start 로 … 64 × 3 = 192 frame 최대 그 이상으로". 장 수는 모두 렌더 장 (12 fps), 학습은 3 장마다 1 장.

v1 과 다른 것
  * 길이: 최대 288 장 (24 초 = 학습 96 장), 최소 192 장 (학습 64 장 = C32 + K32), 멈춘 뒤 96 장 (학습 32 장 = K32) 까지 남긴다.
    v1 은 96 장에서 멈춘 뒤 16 장으로 잘랐고 짧으면 버렸다.
  * **놓는 시각을 늦춘다**: 물체를 0~120 장 붙잡아 두었다가 놓는다 (launch 의 kinematic 구간을 늘림 — 공중이면 떠 있다 떨어진다).
    v1 은 사건 (충돌 · 이탈) 이 영상 첫머리에 몰려 stride 3 에서 C16 사건 창이 110 개 이하 · K32 0 개였다 (natural_ar_vw_state_phys3d_event.yaml).
    놓는 장은 예측 못 하는 사건이라 info unpred 에 넣는다 — 그 장을 넘는 예측은 만들지 않는다 (end).
  * 사건 행 (boundary.tsv): 충돌 (dv ≥ 1.5 m/s, 영상당 최대 6) · 받침 이탈 · **가려짐** 마다 하나. 예측 첫 장 b = 사건 장 + {−2..1}.
    ctx_start = 0 — 문맥 안에 앞선 사건이 있어도 된다 (C32 = 8 초 문맥). 앞선 충돌 · 이탈은 ctx_gate 열에 (v1 의 ctx_start).
    가려짐 행은 다시 보이는 장이 같은 영상 (예측 못 하는 사건 전) 에 있어야 하고, 그 장까지의 렌더 장 수를 pred_min 열에 둔다
    (예측이 그만큼 길어야 "다시 나옴" 이 예측에 들어간다 — 로더가 K ≥ pred_min / stride 인 K 만 고르게 하려면 로더에 이 열을 읽혀야 한다).
  * 안 보임 (vis_frac ≤ 0.3, 화면 밖 포함) 이 이어지는 길이 ≤ 45 장 (학습 15 장). 학습 문맥 16 장이 통째로 안 보이는 창이 없다.
    v1 은 상한이 없어 가림막이 겹쳐 이어 붙고 59 장 내리 가려진 영상이 나왔다 (train 의 22.7 % 가 32 장 이상).
    상한과 고정 판은 같이 못 쓴다 (멈춘 물체를 멈춰 있는 내내 가린다, 시험: 멈춘 장 가림 3.6 % · 움직이는 장 20~29 %) →
    **가림막이 화면 밖에서 6 장 동안 미끄러져 들어와 [s, e) 만 서 있다 나간다** (TIMED_SLIDE). 미끼 판도 같은 분포의 구간 동안 들어왔다 나간다.
  * 가림 구간 길이 L ~ U[6, 45] 장. 목표 구간 수는 (자른 길이 / 140) 배 (EP_REF_T) — 길이가 달라도 가림 비율이 같게.
  * drop_behind: 가림막 출발 = 놓는 시각 + 20~43 장 (v1 36~59) — 뒤로 떨어진 물체가 상한 안에서 드러나게.
  * moving_screen: 오가는 횟수 = 길이 / 48 ~ 길이 / 24 (v1 2~4 / 96 장) — 길어진 영상에서 같은 빠르기.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_phys3d_single_blender as S                           # noqa: E402  (import 하면서 v1 이 B 를 갈아 끼운다)
from gen_phys3d_single_blender import B, make_object, occluder, PRE   # noqa: E402

STRIDE = 3
S.N_FRAMES = 288
S.MIN_T, S.TAIL = 64 * STRIDE, 32 * STRIDE
S.HIDE_LEN = (2 * STRIDE, 15 * STRIDE)
S.MAX_BLIND_RUN = 15 * STRIDE
S.ROW_MIN_CTX, S.ROW_MIN_PRED = 2 * STRIDE, 2 * STRIDE
S.P_EPISODES = (0.08, 0.10, 0.14, 0.16, 0.16, 0.14, 0.10, 0.07, 0.05)   # v1 분포 (평균 4.5) 로는 가림 43.8 % (시험 2026-10-21, 140 장) → 평균 3.7
S.EP_REF_T = 140
S.SCREEN_MAX = (2.8, 2.8)                     # 빠른 구간이 판 크기 상한에 줄어 덜 가려졌다 (시험: 멈춘 장 47.5 % · 6 px/장 이상 36~38 %)
S.MOTION_MIN = 0.1                            # 붙잡아 둔 장 · 멈춘 뒤 장을 남기므로 움직이는 장 비율이 낮다 (v1 0.5)
S.TIMED_SLIDE = 2 * STRIDE                    # 가림막이 6 장 (학습 2 장) 동안 들어온다 · 나간다
DELAY_MAX = 120                               # 놓는 시각 0~120 장 (0~10 초)
MAX_IMPACT_ROWS = 6

_trim_v1 = S.trim_len


def trim_len(lab, T, info):
    """멈춘 뒤 TAIL 장에서 자르되 MIN_T 보다 짧으면 MIN_T 까지 남긴다 (v1 은 버렸다 — 일찍 멈추는 장면이 seed 재시도로 빠져 장면이 편향된다)."""
    return max(_trim_v1(lab, T, info), min(T, S.MIN_T))


S.trim_len = trim_len

# --- 놓는 시각 늦추기 -------------------------------------------------------------------
_launch0 = B.launch
STATE = {"delay": 0, "launched": False}


def launch(o, v, at=PRE - 2):
    STATE["launched"] = True
    return _launch0(o, v, at=at + STATE["delay"])


B.launch = S.launch = launch                  # v1 · 원본 시나리오 함수는 모듈 전역 launch 를 부른다
_build0 = S.build


def build(name, seed, attempt, ident, n_frames, res, engine, frames_dir):
    d = int(np.random.default_rng([seed, attempt, 9]).integers(0, DELAY_MAX + 1))
    STATE.update(delay=d, launched=False)
    rng, sc, env, objs, occ, info = _build0(name, seed, attempt, ident, n_frames, res, engine, frames_dir)
    if STATE["launched"]:                     # 진자는 launch 가 없다 (처음부터 흔들린다)
        env["launch_delay"] = d
        if d > 0:                             # kinematic 이동이 클립 장 d − 3 부터 — 그 장부터 예측 못 한다
            info = dict(info, unpred=sorted(info.get("unpred", []) + [max(0, d - 3)]))
    return rng, sc, env, objs, occ, info


S.build = build


# --- 사건 행 ---------------------------------------------------------------------------
def event_rows(ev, lab, unpred, rng, T, min_ctx=2 * STRIDE, min_pred=2 * STRIDE):
    """예측 첫 장 = 사건 (충돌 · 이탈 · 가려짐). ctx_start 0 (문맥에 앞선 사건 허용), ctx_gate = 앞선 충돌 · 이탈 post.
    가려짐은 다시 보임이 end 전에 있어야 하고 pred_min = 다시 보임 장 − b + 1."""
    gate = sorted({e["frame"] for e in ev if e["kind"] == "leave" or (e["kind"] == "impact" and e["dv"] >= B.DV_ROW)})
    imp = [e for e in ev if e["kind"] == "impact" and e["dv"] >= B.DV_ROW]
    if len(imp) > MAX_IMPACT_ROWS:
        imp = [imp[j] for j in sorted(rng.choice(len(imp), MAX_IMPACT_ROWS, replace=False))]
    cand = imp + [e for e in ev if e["kind"] in ("leave", "occlude")]
    reap = sorted(e["frame"] for e in ev if e["kind"] == "reappear")
    pos = lab["pos"]
    rows, used = [], set()
    for e in sorted(cand, key=lambda d: d["frame"]):
        b = e["frame"] + int(rng.integers(-2, 2))
        end = min([u for u in unpred if u >= b] + [T])
        if b < min_ctx or end - b < min_pred or b in used:
            continue
        pred_min = 1
        if e["kind"] == "occlude":
            r = [t for t in reap if t > e["frame"]]
            if not r or r[0] >= end:
                continue
            pred_min = r[0] - b + 1
        else:
            q = min(end, b + 8) - 1
            if np.linalg.norm(pos[q] - pos[b], axis=-1).max() < 6:
                continue                       # 예측 앞 8 장에 움직임이 없다
        used.add(b)
        rows.append(dict(boundary=int(b), end=int(end), ctx_start=0, event=e["kind"], event_frame=int(e["frame"]),
                         obj=int(e["obj"]), dv=round(float(e["dv"]), 3), pred_min=int(pred_min),
                         ctx_gate=int(max([p for p in gate if p < b and p < e["frame"]] + [0]))))
    return rows


B.event_rows = event_rows


# --- 시나리오 ---------------------------------------------------------------------------
def s_moving_screen(rng, n_frames, ident):
    air = rng.random() < 0.4
    o = make_object(ident, (float(rng.uniform(-2, 2)), 0, float(rng.uniform(1, 3)) if air else 0.45), 0.5, rng,
                    friction=float(rng.uniform(0.01, 0.3)))
    launch(o, (rng.uniform(-2.5, 2.5), 0, rng.uniform(0, 3) if air else 0))
    n_pass = int(rng.integers(max(2, n_frames // 48), n_frames // 24 + 1))
    x0 = float(rng.choice([-1, 1]) * 3.0)
    keys, f = [], 1
    step = (n_frames + PRE) / n_pass
    for k in range(n_pass + 1):
        keys.append((int(f), x0 if k % 2 == 0 else -x0))
        f += step
    return [o], [occluder("moving_screen", x0, float(rng.uniform(1.0, 1.6)), float(rng.uniform(2.5, 4.0)), rng, anim=keys)], \
        {"no_trim": True, "skip_motion": True}


def s_drop_behind(rng, n_frames, ident):
    x = float(rng.uniform(-1.8, 1.8))
    o = make_object(ident, (x + float(rng.uniform(-0.3, 0.3)), 0, float(rng.uniform(3.0, 4.5))), 0.5, rng,
                    rot=tuple(rng.uniform(0, 3.14, 3)))
    launch(o, (rng.uniform(-0.4, 0.4), 0, rng.uniform(-1, 1)))
    w = float(rng.uniform(1.5, 2.2))
    t_move = STATE["delay"] + int(rng.integers(20, 44))
    dx = float(rng.choice([-1, 1]) * rng.uniform(2.5, 3.5))
    keys = [(1, x), (PRE + t_move, x), (PRE + t_move + int(rng.integers(8, 16)), x + dx)]
    occ = [occluder("screen", x, w, float(rng.uniform(2.0, 2.8)), rng, anim=keys)]
    return [o], occ, {"unpred": [t_move], "min_T": t_move + 16 + S.TAIL, "skip_motion": True}


# 표의 순서는 그대로 (build 의 rng 가 시나리오 번호를 쓴다) — 값만 바꾼다
S.SCENARIOS["moving_screen"] = ("permanence", "가림막이 오가며 물체를 여러 번 가림 — 물체는 정지 · 미끄러짐 · 튕김", s_moving_screen)
S.SCENARIOS["drop_behind"] = ("permanence", "물체가 가림막 뒤로 떨어져 멈춤, 그다음 가림막이 비켜나며 드러남", s_drop_behind)

if __name__ == "__main__":
    B.main()
