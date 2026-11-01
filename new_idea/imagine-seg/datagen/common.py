"""가림 합성 데이터 생성 — 공통 설정과 입출력.

계획: imagine-seg/DATA_PIPELINE_2026-10-31.md
결정 (2026-10-31 사용자): 6 fps · 256×256 · SSv2 창 16 장 / K400 창 48 장 · 문맥 = 앞 절반.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = 256
FPS = 6.0

TRAIN_DATA_ROOT = Path("/workspace/dataset/training_data")
DATA_CSV = Path("/workspace/world_model/vjepa2/data_csv")
COCO_ROOT = Path("/workspace/dataset/assets/coco_val2017")
SAM2_ROOT = Path("/workspace/tools/sam2")
SAM2_CKPT = Path("/workspace/world_model/vjepa2/checkpoint/sam2/sam2.1_hiera_base_plus.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# 창과 가림 일정 제약 (프레임 단위, 6 fps 기준). 원 계획 §2.4.
#   k_vis : 창 첫 k_vis 장은 물체가 보여야 한다 (덮임 ≤ vis_max_cov)
#   d_min : 완전 가림 (덮임 ≥ full_cov) 이 이어지는 최소 장 수
#   r_vis : 재등장 = 완전 가림 뒤 처음으로 보이는 비율 ≥ r_vis 인 프레임. 예측 구간 안이어야 한다
SOURCES = {
    "ssv2": dict(M=16, k_vis=3, d_min=2),
    "k400": dict(M=48, k_vis=6, d_min=3),
}
SCHEDULE = dict(full_cov=0.95, vis_max_cov=0.05, r_vis=0.5, max_occ_area=0.35)
# 물체 거르기 (원 계획 §2.2). 면적은 창 첫 프레임 기준, 나머지는 창 전체.
# 움직임 = 카메라 궤적을 뺀 물체 중심이 창 안에서 처음 자리로부터 가장 멀어진 거리 (px, 256 화면)
OBJECT = dict(area_min=0.02, area_max=0.15, area_ratio_max=2.5, step_ratio_max=1.6,
              min_rel_net_px=20.0)


def window(source: str) -> tuple[int, int]:
    """(M, T_ctx). 문맥 = 앞 절반, 예측 = 뒤 절반."""
    M = SOURCES[source]["M"]
    return M, M // 2


def pack(m: np.ndarray) -> dict:
    m = np.asarray(m, bool)
    return {"bits": np.packbits(m.reshape(-1)), "shape": np.array(m.shape)}


def unpack(bits: np.ndarray, shape: np.ndarray) -> np.ndarray:
    n = int(np.prod(shape))
    return np.unpackbits(bits)[:n].reshape(tuple(shape)).astype(bool)


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_np_default))


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
