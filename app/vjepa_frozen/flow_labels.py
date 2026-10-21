"""미리 뽑아 둔 optical flow 캐시를 읽어 학습 창 · 증강에 맞춰 돌려준다 (2026-10-22).

캐시는 `z_training/data/build_flow_cache.py` 가 만든다:
    flow.f16  (P, 2, G, G)  원본 프레임 비율 단위 (u = du_px / W_orig)
    valid.u8  (P, G, G)     전후 일관성 통과 픽셀 비율 × 255
    index.tsv path, n4, off, npair   — 영상 v 의 fps 4 쌍 j (j → j+1) 는 전역 행 off(v) + j

쓰는 쪽이 해야 할 두 가지
  1. **창 시작을 fps 4 격자에 맞춘다** (`var_window.snap_start`). 캐시는 전역 fps 4 격자 위의 인접 쌍만 담는다.
     맞추지 않으면 영상마다 fstp 가지 위상이 생겨 라벨이 어긋난다.
  2. ClipTransform 이 쓴 **crop box 와 좌우 반전을 그대로 적용**한다 (`ClipTransform.__call__(..., return_params=True)`).
     자르기는 flow 의 **크기도 바꾼다** — 좁게 자를수록 같은 픽셀 이동이 화면에서 커진다.

단위 변환: u_patch = u_norm × (W_orig / w_crop) × (res / patch).  res 256 · patch 16 이면 폭 전체를 자를 때 ×16.
  = "블록 하나 (0.25 s) 동안 패치 몇 칸 움직이나". 1 초에 화면을 가로지르는 물체 ≈ 4.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class FlowCache:
    """데이터셋 하나의 flow 캐시. 프로세스 (DataLoader worker) 마다 하나 만든다 — memmap 은 fork 후 열어야 안전하다.

    root 는 디렉토리 하나 또는 여러 개 (shard 별 출력). 여러 개면 경로 → (shard, off, npair) 로 합친다.
    """

    def __init__(self, root):
        self.roots = [root] if isinstance(root, str) else list(root)
        self.flow, self.valid = [], []
        self.off: Dict[str, Tuple[int, int, int]] = {}
        for si, r in enumerate(self.roots):
            self.flow.append(np.load(os.path.join(r, "flow.f16"), mmap_mode="r"))     # (P, 2, G, G) f16
            self.valid.append(np.load(os.path.join(r, "valid.u8"), mmap_mode="r"))    # (P, G, G) u8
            for ln in open(os.path.join(r, "index.tsv"), encoding="utf-8"):
                p, n4, off, npair = ln.rstrip("\n").split("\t")
                self.off[p] = (si, int(off), int(npair))
        self.G = int(self.flow[0].shape[-1])

    def has(self, path: str) -> bool:
        return path in self.off

    def blocks(self, path: str, j0: int, nb: int, tubelet: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """블록 k (k = 0..nb-1) 의 라벨 = fps 4 쌍 (j0 + k·tubelet) → 그 다음 장.
        tubelet 2 면 블록 안 두 장 사이 flow — 타깃 encoder 의 블록 토큰이 실제로 담는 운동이다.
        return (nb, 2, G, G) float32, (nb, G, G) float32 [0, 1]. 범위를 벗어나면 None."""
        got = self.off.get(path)
        if got is None:
            return None
        si, off, npair = got
        idx = [j0 + k * tubelet for k in range(nb)]
        if idx[0] < 0 or idx[-1] >= npair:
            return None
        rows = [off + j for j in idx]
        f = np.asarray(self.flow[si][rows], dtype=np.float32)
        v = np.asarray(self.valid[si][rows], dtype=np.float32) / 255.0
        return f, v

    def span(self, path: str, j0: int, n: int):
        """fps 4 쌍 j0 … j0+n-1 을 **이어서** 돌려준다 (궤적 이어 붙이기용).
        영상 끝을 넘으면 있는 데까지만 — block_traj 가 모자란 지평선을 ok = 0 으로 지운다.
        return (f (m, 2, G, G), v (m, G, G)) 또는 None."""
        got = self.off.get(path)
        if got is None:
            return None
        si, off, npair = got
        if j0 < 0 or j0 >= npair:
            return None
        m = min(n, npair - j0)
        f = np.asarray(self.flow[si][off + j0: off + j0 + m], dtype=np.float32)
        v = np.asarray(self.valid[si][off + j0: off + j0 + m], dtype=np.float32) / 255.0
        return f, v


def apply_transform(f: np.ndarray, v: np.ndarray, box, HW, flip: bool, g_out: int, res: int, patch: int):
    """f (nb, 2, G, G) · v (nb, G, G) 를 ClipTransform 과 같은 자르기 · 반전으로.

    box  (top, left, h, w)  원본 픽셀 좌표. 자르기가 없었으면 (0, 0, H, W).
    HW   (H, W) 원본 크기.
    return flow (nb, 2, g_out, g_out) float32 **패치/블록 단위**, valid (nb, g_out, g_out) float32.
    """
    top, left, h, w = box
    H, W = HW
    ft = torch.from_numpy(f)                                   # (nb, 2, G, G)
    vt = torch.from_numpy(v).unsqueeze(1)                      # (nb, 1, G, G)

    # 자를 영역을 [-1, 1] 격자로. 출력 칸 중앙이 원본의 어디인지.
    c = (torch.arange(g_out, dtype=torch.float32) + 0.5) / g_out
    ys = (top + c * h) / H * 2 - 1
    xs = (left + c * w) / W * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], -1).unsqueeze(0).expand(ft.shape[0], -1, -1, -1)
    ft = F.grid_sample(ft, grid, mode="bilinear", padding_mode="border", align_corners=False)
    vt = F.grid_sample(vt, grid, mode="bilinear", padding_mode="border", align_corners=False)

    # 비율 단위 → 자른 창 안의 픽셀 → 패치 칸. 가로 · 세로가 다르게 늘어난다 (화면비 왜곡).
    ft[:, 0] *= (W / max(w, 1)) * (res / patch)
    ft[:, 1] *= (H / max(h, 1)) * (res / patch)
    if flip:
        ft = ft.flip(-1); ft[:, 0] *= -1
        vt = vt.flip(-1)
    return ft.contiguous(), vt[:, 0].clamp_(0, 1).contiguous()


# ---------------------------------------------------------------- 궤적 (2026-10-22, 사용자 결정 "traj 로")
# 왜 순간 flow 가 아니라 궤적인가 [측정 2026-10-22, k400 395 편 · ssv2 130 편]:
#   * 순간 flow 는 0.5 s 를 넘으면 사실상 예측 불가다 — 마지막 문맥 flow 와 미래 flow 의 cos 가
#     0.25 s 0.32 / 0.5 s 0.11 / 0.75 s 이후 0.04. 부호가 대칭이라 L1 의 최적 상수는 0 이고
#     (실측 0 예측 0.430 < 복사 0.581), 그 자리에 L1 을 걸면 head 는 "0 을 내라" 만 배운다.
#   * 누적 변위는 적분이라 **방향이 오래 남는다** — 등속 외삽과 진짜 궤적의 cos 가
#     0.5 s 0.78 / 1 s 0.56 / 2 s 0.39 / 4 s 0.26 (k400 · ssv2 거의 같다).
#   * 그런데 **크기는 등속이 아니다** — 등속 외삽 L1 / 0 예측 L1 이 0.5 s 0.53 → 4 s 2.49 로,
#     등속은 2.5 배 과한 거리를 예측한다 (진짜 운동은 느려지거나 되돌아온다).
#   => 배울 거리가 남아 있는 목표다: **방향은 대체로 이어지고, 크기는 꺾인다.** 그게 물리 부분이다.
#
# 앵커를 **그 블록 자신**으로 둔다 (마지막 문맥 장이 아니라). 두 가지 이유:
#   1. predictor 의 블록 b 토큰은 블록 b 시점의 격자 위에 있다. 앵커를 다른 시점에 두면 좌표계가 어긋난다.
#   2. 전방 flow 만 있으면 된다 (역방향 flow 를 저장하지 않아도 된다).
# 질문이 이렇게 된다: "이 토큰이 가리키는 내용은 h 장 뒤에 어디로 가 있나."
def block_traj(f, v, nb: int, tubelet: int, horizons, j_off: int = 0):
    """블록마다 여러 지평선의 누적 전방 변위.

    f (T, 2, G, G) · v (T, G, G)   fps 4 인접 쌍 (쌍 t = 장 t → t+1), 창 시작부터 이어진 것.
    nb · tubelet                    블록 수 · 블록당 장 수. 블록 b 의 앵커 장 = j_off + b·tubelet + tubelet - 1.
    horizons                        앞으로 몇 장 (예: [1, 2, 4, 8] = 0.25 · 0.5 · 1 · 2 s).
    return d (nb, H, 2, G, G) 프레임 비율 변위, ok (nb, H, G, G) 유효도.

    d_{t+1}(x) = d_t(x) + F_t(x + d_t(x)) — 칸을 따라가며 더한다. 따라간 자리의 유효도를 곱해 쌓는다
    (한 번이라도 일관성 검사를 떨어뜨렸거나 화면을 나간 칸은 0 으로 수렴 → 손실에서 빠진다).
    창 밖으로 나가는 지평선은 ok = 0 으로 돌려준다 (라벨 없음).
    """
    ft = torch.as_tensor(np.ascontiguousarray(f), dtype=torch.float32)
    vt = torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32)
    T, G = ft.shape[0], ft.shape[-1]
    H = list(int(h) for h in horizons)
    hmax = max(H)
    cy, cx = torch.meshgrid(torch.arange(G, dtype=torch.float32), torch.arange(G, dtype=torch.float32), indexing="ij")
    base = torch.stack([(cx + 0.5) / G, (cy + 0.5) / G]).unsqueeze(0).expand(nb, -1, -1, -1)   # (nb, 2, G, G)
    anchors = [j_off + b * tubelet + tubelet - 1 for b in range(nb)]

    d = torch.zeros(nb, 2, G, G)
    ok = torch.ones(nb, G, G)
    outd = torch.zeros(nb, len(H), 2, G, G)
    outo = torch.zeros(nb, len(H), G, G)
    for step in range(hmax):
        pos = base + d
        grid = (pos.permute(0, 2, 3, 1) * 2 - 1)                                 # (nb, G, G, 2)
        rows = [a + step for a in anchors]
        live = torch.tensor([1.0 if 0 <= r < T else 0.0 for r in rows]).view(nb, 1, 1)
        rows = [min(max(r, 0), T - 1) for r in rows]
        fs = F.grid_sample(ft[rows], grid, mode="bilinear", padding_mode="border", align_corners=False)
        vs = F.grid_sample(vt[rows].unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=False)[:, 0]
        inside = ((pos[:, 0] >= 0) & (pos[:, 0] <= 1) & (pos[:, 1] >= 0) & (pos[:, 1] <= 1)).float()
        d = d + fs * live.unsqueeze(1)
        ok = ok * vs * inside * live
        if (step + 1) in H:
            k = H.index(step + 1)
            outd[:, k], outo[:, k] = d, ok
    return outd, outo


def apply_transform_traj(d, ok, box, HW, flip: bool, g_out: int, res: int, patch: int):
    """block_traj 의 출력 (nb, H, 2, G, G) · (nb, H, G, G) 을 ClipTransform 과 같은 자르기 · 반전으로.
    축 하나만 더 있을 뿐 apply_transform 과 같은 규칙이다 — 변위도 flow 처럼 창 크기에 반비례해 커진다."""
    nb, Hn = d.shape[0], d.shape[1]
    f2 = d.reshape(nb * Hn, 2, d.shape[-2], d.shape[-1]).numpy()
    v2 = ok.reshape(nb * Hn, ok.shape[-2], ok.shape[-1]).numpy()
    ft, vt = apply_transform(f2, v2, box, HW, flip, g_out, res, patch)
    return ft.reshape(nb, Hn, 2, g_out, g_out), vt.reshape(nb, Hn, g_out, g_out)


def const_velocity(d, horizons):
    """파라미터 0 기준선 — 첫 지평선 (h = H[0]) 의 변위를 비례로 늘린 등속 외삽. d (nb, H, 2, g, g) → 같은 모양.
    실측 (2026-10-22): 0.5 s 에서는 0 예측보다 좋고 (0.53 배), 4 s 에서는 2.5 배 나쁘다 — 거리를 과하게 잡는다."""
    H = [int(h) for h in horizons]
    return torch.stack([d[:, 0] * (h / H[0]) for h in H], 1)
