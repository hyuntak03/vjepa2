"""phys2d_sup 프레임별 상태 라벨 읽기 (2026-10-18). 생성기 `gen_phys2d_sup.py` 산출물 전용.

    from z_training.data.phys2d_sup_labels import load_clip, window_labels
    clip = load_clip("/local_datasets/.../clips/occ_pass/occ_pass_00012.mp4")
    y = window_labels(clip, start=10, n_frames=32, stride=1, tubelet=2, max_obj=8)
    y["pos"]          # (16, 8, 2)  튜블릿마다 물체 중심 (px, 256 화면)
    y["mask_state"]   # (16, 8)     위치 · 속도 손실을 걸어도 되는 칸

라벨 (npz, 장 T × 물체 n) — 좌표 px (원점 왼쪽 위, y 아래, 픽셀 j = [j, j+1)), 시간 1 장 = 1/12 초
  pos (T,n,2) · vel (T,n,2) px/장 (순간 속도) · acc (T,n,2) px/장² (중앙 차분, 충돌 장에서 크다)
  angle (T,n) rad · omega (T,n) rad/장 · bbox (T,n,4) amodal x0 y0 x1 y1
  present (T,n) 시뮬에 있음 (화면 밖에서 들어오는 물체는 들어오기 전 False, 그 칸 값은 NaN)
  vis_frac (T,n) 전체 면적 중 화면에 보이는 비율 · inframe_frac (T,n) 전체 면적 중 화면 안 비율 (가림 무시)
  hidden (T,n) 화면 안인데 완전히 가려짐 · seen (T,n) 지금까지 30 % 넘게 보인 적 있음 · impact (T,n) 충돌 속도 변화 px/장
  identity / shape_id / color_id (n,) · size · area (n,) · occluder_rect (T,m,4)
속도의 방향 · 크기는 (vx, vy) 에서 얻는다: 크기 = ‖v‖, 방향 = atan2(vy, vx). 방향 각을 직접 회귀하지 말 것
(±π 에서 끊기고, 정지 물체는 방향이 정의되지 않는다) — (vx, vy) 또는 (cos, sin) · ‖v‖ 로 건다.
"""
from __future__ import annotations

import json
import os

import numpy as np

SHAPES = ("circle", "square", "rectangle", "triangle", "pentagon", "hexagon")
COLORS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")
N_SHAPE, N_COLOR = len(SHAPES), len(COLORS)
W = H = 256

PER_FRAME = ("pos", "vel", "acc", "angle", "omega", "bbox", "present", "vis_frac", "inframe_frac", "hidden", "seen", "impact")
PER_OBJ = ("identity", "shape_id", "color_id", "size", "area")


def load_clip(path: str, with_meta: bool = False, with_seg: bool = False) -> dict:
    """path = mp4 · json · _states.npz 중 아무거나."""
    stem = path
    for suf in (".mp4", ".json", "_states.npz", "_seg.npz"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    z = np.load(stem + "_states.npz")
    out = {k: z[k] for k in z.files}
    if with_meta:
        with open(stem + ".json") as f:
            out["meta"] = json.load(f)
    if with_seg:
        out["seg"] = np.load(stem + "_seg.npz")["seg"]
    return out


def window_labels(clip: dict, start: int, n_frames: int, stride: int = 1, tubelet: int = 1, max_obj: int | None = None,
                  reduce: str = "last", normalize: bool = False) -> dict:
    """학습 창 (start, start+stride, …, n_frames 장) 의 라벨. tubelet > 1 이면 튜블릿마다 하나로 줄인다.
    reduce: last = 튜블릿 마지막 장 (그 토큰이 '본' 가장 최근 상태), mean = 평균 (연속 값만; 불리언은 last).
    normalize: 위치를 [0,1], 속도를 화면 폭 / 장으로.
    max_obj: 물체 축을 이 길이로 채운다 (빈 칸은 present=False). 반환 mask_* 는 손실 마스크:
      mask_state    = present & seen & 화면 안 (inframe_frac > 0.5) — 가려져 있어도 한 번 본 물체면 위치 · 속도 손실을 건다 (영속)
      mask_visible  = mask_state & 보임 (vis_frac > 0.3) — 보이는 물체만 (읽기 · 지속)
      mask_identity = present & seen — 정체 손실"""
    idx = start + stride * np.arange(n_frames)
    T = clip["present"].shape[0]
    if idx[-1] >= T:
        raise IndexError(f"창 끝 {idx[-1]} ≥ 영상 길이 {T}")
    y = {k: clip[k][idx] for k in PER_FRAME}
    if stride != 1:          # 장 간격이 stride 면 속도 · 가속도 단위를 창의 1 장에 맞춘다
        y["vel"] = y["vel"] * stride
        y["acc"] = y["acc"] * stride ** 2
        y["omega"] = y["omega"] * stride
    if tubelet > 1:
        assert n_frames % tubelet == 0
        nt = n_frames // tubelet
        for k, v in y.items():
            v = v.reshape(nt, tubelet, *v.shape[1:])
            if reduce == "mean" and v.dtype.kind == "f" and k not in ("impact",):
                y[k] = np.nanmean(v, axis=1) if np.isnan(v).any() else v.mean(1)
            elif k == "impact":
                y[k] = v.max(1)
            elif k == "present":
                y[k] = v.all(1)
            else:
                y[k] = v[:, -1]
    for k in PER_OBJ:
        y[k] = clip[k]
    if normalize:
        y["pos"] = y["pos"] / W
        y["vel"] = y["vel"] / W
        y["acc"] = y["acc"] / W
        y["bbox"] = y["bbox"] / W
    y["mask_identity"] = y["present"] & y["seen"]
    y["mask_state"] = y["mask_identity"] & (y["inframe_frac"] > 0.5) & np.isfinite(y["pos"]).all(-1)
    y["mask_visible"] = y["mask_state"] & (y["vis_frac"] > 0.3)
    if max_obj is not None:
        n = y["identity"].shape[0]
        if n > max_obj:
            raise ValueError(f"물체 {n} > max_obj {max_obj}")
        for k, v in y.items():
            ax = 0 if k in PER_OBJ else 1
            if v.shape[ax] == n and n < max_obj:
                pad = [(0, 0)] * v.ndim
                pad[ax] = (0, max_obj - n)
                fill = False if v.dtype == bool else (-1 if v.dtype.kind == "i" else 0.0)
                y[k] = np.pad(v, pad, constant_values=fill)
    return y
