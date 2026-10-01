"""frozen-encoder predictor 학습용 데이터.

두 소스를 받는다 (둘 다 [B, 3, T, H, W] 정규화 클립 + 마스크 인덱스를 낸다):

  frames_index   PNG 프레임 폴더 + index.csv (world_model_analysis 와 같은 스키마).
                 IntPhysGen v11 / RollOut_v2 / IntPhys1 train 이 전부 이 형식이다.
                 프레임 경로 = frames_root / frames_pattern.format(frame=f, **row)
                 f = start, start+stride, ... (n_frames 장). start 는 frames_start_choices 에서
                 샘플마다 무작위로 고를 수 있다 (시간 jitter).
  video_csv      "<mp4 경로> <라벨>" 공백 구분 csv (src/datasets/video_dataset.VideoDataset).
                 IntPhys2 / EK100 / Kinetics 류.

전처리는 채점(evals/world_model_analysis/data.py:clip) 과 같은 커널을 쓴다 —
bilinear, antialias=False, ImageNet mean/std. 증강은 기본 꺼져 있고
(hflip_p: 0, random_resized_crop: null) config 로 켠다. 물체 하나가 움직이는 합성
영상에서 scale 0.3 crop 은 물체를 잘라내므로 켤 때는 scale 을 넓게 두지 말 것.

⚠️ **비정방형 자연 영상(K400 등)에는 `aug.square_crop: center` 를 반드시 켠다** (2026-09-22).
   안 켜면 720x1280 이 (256,256) 으로 눌려 **가로 속도가 1.78배 압축**된다. `random_resized_crop`
   을 켜면 RRC 창이 이기고 square_crop 은 폴백으로만 쓰인다. 정사각 원본(v11/IntPhys)에는 no-op.

마스크 (MaskSampler)
  temporal_prefix   context = 앞 C 프레임 전부, target = 뒤 T−C 프레임 전부.
                    surprise_c16t32 채점과 같은 토폴로지. C 는 context_frames 에서 배치마다 고른다.
  prefix_window     context = 앞 C 블록, target = **그 뒤 K 블록만** (C·K 독립, 2026-09-22).
                    `gap_blocks` (2026-09-30) 를 주면 문맥과 예측 사이에 G 블록을 비운다 (문맥도 타깃도 아님).
                    `context_blocks` x `predict_blocks` 에서 배치마다 하나씩 고른다
                    (C+K <= T 인 조합만). prefix-마스크 predictor 와 짝이다.
  block3d           릴리즈 사전학습의 3D 블록 마스크 (src/masks/multiseq_multiblock3d._MaskGenerator).
  여러 스펙을 weight 로 섞으면 **배치마다 하나**를 고른다 (per-batch). 스펙마다 context encoder
  forward 가 따로 필요하므로 한 배치에 두 스펙을 같이 걸지 않는다.

토큰 순서는 PatchEmbed3D 와 같다: tubelet 이 바깥, 그 안에서 공간 패치. 그래서 앞 C 프레임의
토큰은 flat index [0, C/tubelet * S) 이다 (analysis/intphys2/surprise._context_target_indices 와 동일).
"""
from __future__ import annotations

import csv
import logging
import math
import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


# ----------------------------------------------------------------------------- transform
class ClipTransform:
    """(T, H, W, 3) uint8 -> (3, T, H, W) float, 정규화.

    resolution     리사이즈 목표 한 변 (모델 img_size). 원본이 같으면 안 건드린다.
    hflip_p        좌우 반전 확률 (클립 전체에 같은 결정)
    random_resized_crop  {scale: [lo, hi], ratio: [lo, hi]} 또는 None. 클립 전체에 같은 창.
    square_crop    비정방형 입력을 **화면비를 지켜** 정사각으로 자른다 (2026-09-22 추가).
                   none   기존 동작 — (r, r) 로 바로 리사이즈
                   center 짧은 변 기준 중앙 정사각
                   ⚠️ **자연 영상(K400 등)에는 반드시 center 를 켠다.** 720x1280 을 그냥
                      (256,256) 으로 누르면 **가로가 1.78배 압축**되어 가로 속도가 세로보다
                      그만큼 작아진다. 우리가 재는 것이 "스텝당 이동 거리" 라 치명적이다.
                      v11/IntPhys 류는 원본이 정사각이라 켜도 no-op 이다.
                   random_resized_crop 이 켜져 있으면 RRC 창이 이기고, **RRC 가 창을 못 찾을 때
                   폴백**으로 쓰인다 (예전 폴백은 전체 프레임 = 눌림이었다).
    """

    def __init__(self, resolution: int, hflip_p: float = 0.0, random_resized_crop: Optional[dict] = None,
                 square_crop: str = "none", out_uint8: bool = False):
        # out_uint8 (2026-09-22): crop·resize 만 하고 **uint8 (3,T,H,W)** 로 돌려준다. 정규화는 GPU 에서
        # (`normalize_clips`). float32 로 넘기면 배치 하나가 28x3x48x256²x4B = 1.06 GB 라 worker->main
        # shm + pinned 가 8 rank x 8 worker x prefetch 2 에서 130 GB 를 넘고, SLURM cgroup 상한(256 GB)
        # 을 epoch 경계에서 넘겨 OOM-killer 가 rank 를 죽였다 (job 429418, 21:12). uint8 은 4 배 작다.
        # 리사이즈 뒤 8-bit 양자화 오차(<=0.5/255)는 encoder bf16 잡음(6.5e-2)보다 두 자리 작다.
        self.out_uint8 = bool(out_uint8)
        self.resolution = int(resolution)
        self.hflip_p = float(hflip_p)
        if square_crop not in ("none", "center"):
            raise ValueError(f"square_crop 은 none | center: {square_crop!r}")
        self.square_crop = square_crop
        self.rrc = random_resized_crop or None
        if self.rrc:
            sc, ra = self.rrc.get("scale", [0.5, 1.0]), self.rrc.get("ratio", [0.75, 1.3333])
            if not (0 < sc[0] <= sc[1] <= 1.0):
                raise ValueError(f"random_resized_crop.scale 이 이상하다: {sc}")
            self.rrc = {"scale": (float(sc[0]), float(sc[1])), "ratio": (float(ra[0]), float(ra[1]))}

    def _rrc_box(self, H: int, W: int):
        area = H * W
        for _ in range(10):
            target_area = area * random.uniform(*self.rrc["scale"])
            log_r = (math.log(self.rrc["ratio"][0]), math.log(self.rrc["ratio"][1]))
            ar = math.exp(random.uniform(*log_r))
            w = int(round(math.sqrt(target_area * ar)))
            h = int(round(math.sqrt(target_area / ar)))
            if 0 < w <= W and 0 < h <= H:
                return random.randint(0, H - h), random.randint(0, W - w), h, w
        return self._square_box(H, W)      # 실패하면 중앙 정사각 (전체를 돌려주면 화면비가 깨진다)

    @staticmethod
    def _square_box(H: int, W: int):
        """짧은 변 기준 중앙 정사각 (top, left, h, w)."""
        side = min(H, W)
        return (H - side) // 2, (W - side) // 2, side, side

    def __call__(self, buffer) -> torch.Tensor:
        if not torch.is_tensor(buffer):
            buffer = torch.from_numpy(np.ascontiguousarray(buffer))
        # ⚠️ **crop 을 float 변환보다 먼저 한다** (2026-09-22). 예전 순서는 720p 48 장을
        #    통째로 float32 로 올린 뒤 잘라서, 쓰지도 않을 530 MB (48x3x720x1280x4B) 를
        #    매 샘플 만들었다. K400 이 들어오면서 이게 decode 다음으로 큰 CPU 비용이 된다.
        T, H, W, _ = buffer.shape
        r = self.resolution
        if self.rrc:
            top, left, h, w = self._rrc_box(H, W)
        elif self.square_crop == "center" and H != W:
            top, left, h, w = self._square_box(H, W)
        else:
            top, left, h, w = 0, 0, H, W
        if (top, left, h, w) != (0, 0, H, W):
            buffer = buffer[:, top: top + h, left: left + w, :]
        x = buffer.permute(3, 0, 1, 2).float().div_(255.0)          # (3, T, H, W) 0..1
        if tuple(x.shape[-2:]) != (r, r):
            # 채점과 같은 커널: bilinear / antialias=False (공식 Garrido 전처리와 동일)
            x = F.interpolate(x.transpose(0, 1), size=(r, r), mode="bilinear",
                              align_corners=False).transpose(0, 1)
        if self.hflip_p > 0 and random.random() < self.hflip_p:
            x = x.flip(-1)
        x = x.contiguous()
        if self.out_uint8:
            return x.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8)
        return (x - MEAN) / STD


def normalize_clips(clips: torch.Tensor) -> torch.Tensor:
    """uint8 (B,3,T,H,W) -> ImageNet 정규화 float (GPU). 이미 float 이면 그대로 (정규화된 것으로 본다)."""
    if clips.dtype == torch.uint8:
        return (clips.float().div_(255.0) - MEAN.to(clips.device)) / STD.to(clips.device)
    return clips


# ----------------------------------------------------------------------------- frames_index
def _passes(row: dict, include: Optional[dict], exclude: Optional[dict]) -> bool:
    for col, vals in (include or {}).items():
        if str(row.get(col)) not in {str(v) for v in vals}:
            return False
    for col, vals in (exclude or {}).items():
        if str(row.get(col)) in {str(v) for v in vals}:
            return False
    return True


class FramesIndexDataset(torch.utils.data.Dataset):
    """index.csv 한 행 = 클립 하나. PNG 프레임을 직접 읽는다.

    spec 키
      root, index_csv, frames_root, frames_pattern, frames_start, frames_stride
      frames_start_choices : [int, ...]  샘플마다 시작 프레임을 여기서 고른다 (없으면 frames_start 고정)
      include / exclude    : {column: [values]}  행 필터. 기본 include={plausible: ["1"]}
      limit                : 앞 N 행만 (디버그)
      columns_required     : 없으면 죽일 컬럼 (기본 video_id)
    """

    def __init__(self, spec: dict, n_frames: int, transform: ClipTransform):
        from PIL import Image  # noqa: F401  (import 실패를 여기서 잡는다)
        self.spec = spec
        self.n_frames = int(n_frames)
        self.transform = transform
        self.root = spec["root"]
        self.frames_root = spec["frames_root"]
        self.pattern = spec.get("frames_pattern", "{file_name}/{frame:06d}.png")
        self.stride = int(spec.get("frames_stride", 1))
        self.starts = [int(s) for s in (spec.get("frames_start_choices") or [spec.get("frames_start", 0)])]
        # raw_span 모드: 연속 raw 프레임 raw_span 장을 통째로 읽어 (T_raw, H, W, 3) uint8 로 돌려준다.
        # 창 자르기·리사이즈·마스크는 collate(WindowGridCollator) 가 배치마다 한다 (data.window_grid).
        self.raw_span = int(spec["raw_span"]) if spec.get("raw_span") else None
        self.raw_start = int(spec.get("frames_start", 0))
        if not os.path.isdir(self.frames_root):
            raise FileNotFoundError(f"frames_root 가 없다: {self.frames_root}")
        ipath = os.path.join(self.root, spec.get("index_csv", "index.csv"))
        if not os.path.isfile(ipath):
            raise FileNotFoundError(f"index 가 없다: {ipath}")
        include = spec.get("include", {"plausible": ["1"]})
        exclude = spec.get("exclude")
        with open(ipath, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if _passes(r, include, exclude)]
        if not rows:
            raise ValueError(f"{ipath}: 필터(include={include}, exclude={exclude}) 뒤에 행이 0개다")
        lim = spec.get("limit")
        if lim:
            rows = rows[: int(lim)]
        self.rows = rows
        # 첫 클립의 마지막 프레임이 실재하는지 확인 (예산 초과를 모델 로드 전에 잡는다)
        for st in ([self.raw_start] if self.raw_span else self.starts):
            last = st + self.raw_span - 1 if self.raw_span else st + (self.n_frames - 1) * self.stride
            p = self._path(rows[0], last)
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"프레임 예산 초과 또는 패턴 오류: {p} 가 없다 "
                    f"(start={st}, stride={self.stride}, n_frames={self.n_frames})")
        logger.info(f"[frames_index] {ipath}: {len(rows)} clips, starts={self.starts}, "
                    f"stride={self.stride}, n_frames={self.n_frames}, include={include}, exclude={exclude}")

    def _path(self, row: dict, frame: int) -> str:
        return os.path.join(self.frames_root, self.pattern.format(frame=frame, **row))

    def __len__(self):
        return len(self.rows)

    def read_uint8(self, i: int, start: Optional[int] = None) -> torch.Tensor:
        """(T, H, W, 3) uint8."""
        from PIL import Image
        row = self.rows[i]
        st = self.starts[0] if start is None else start
        fr = []
        for f in range(st, st + self.n_frames * self.stride, self.stride):
            fr.append(np.asarray(Image.open(self._path(row, f)).convert("RGB")))
        return torch.from_numpy(np.stack(fr))

    def read_raw(self, i: int) -> torch.Tensor:
        """raw_span 모드: (raw_span, H, W, 3) uint8, frames_start 부터 연속."""
        from PIL import Image
        row = self.rows[i]
        fr = [np.asarray(Image.open(self._path(row, f)).convert("RGB"))
              for f in range(self.raw_start, self.raw_start + self.raw_span)]
        return torch.from_numpy(np.stack(fr))

    def __getitem__(self, i: int):
        if self.raw_span:
            return self.read_raw(i), 0, [np.arange(self.raw_start, self.raw_start + self.raw_span, dtype=np.int64)]
        st = random.choice(self.starts)
        x = self.transform(self.read_uint8(i, st))
        # VideoDataset 과 같은 구조: ([clip], label, [frame indices])
        idx = np.arange(st, st + self.n_frames * self.stride, self.stride, dtype=np.int64)
        return [x], 0, [idx]


# ----------------------------------------------------------------------------- video_csv
def make_video_csv_dataset(spec: dict, n_frames: int, transform: ClipTransform):
    """mp4 csv. VideoDataset 은 fps / duration / frame_step 중 정확히 하나를 요구한다."""
    from src.datasets.video_dataset import VideoDataset
    kw = dict(frames_per_clip=n_frames, transform=transform, num_clips=1,
              random_clip_sampling=bool(spec.get("random_clip_sampling", True)),
              filter_short_videos=bool(spec.get("filter_short_videos", False)),
              fps=spec.get("fps"), duration=spec.get("duration"), frame_step=spec.get("frame_step"),
              uniform_sampling=bool(spec.get("uniform_sampling", False)),
              center_sampling=bool(spec.get("center_sampling", False)),
              # ⚠️ 리더당 decord 스레드. 기본 -1(전 코어)은 worker 가 여럿이면 과다구독이다
              decord_threads=int(spec.get("decord_threads", 1)))
    if sum(v is not None for v in (kw["fps"], kw["duration"], kw["frame_step"])) != 1:
        raise ValueError("video_csv 는 fps / duration / frame_step 중 정확히 하나를 줘야 한다")
    return VideoDataset(data_paths=[spec["csv"]], dataset_fpcs=[n_frames], **kw)


# ----------------------------------------------------------------------------- 합치기
class WeightedConcat(torch.utils.data.Dataset):
    """여러 데이터셋을 이어 붙이고 샘플 가중치를 낸다 (DistributedWeightedSampler 용)."""

    def __init__(self, datasets: Sequence[torch.utils.data.Dataset], weights: Optional[Sequence[float]]):
        self.datasets = list(datasets)
        self.sizes = [len(d) for d in self.datasets]
        self.cum = np.cumsum(self.sizes)
        self.sample_weights = None
        if weights is not None:
            if len(weights) != len(self.datasets):
                raise ValueError("datasets 와 weights 길이가 다르다")
            self.sample_weights = []
            for w, n in zip(weights, self.sizes):
                self.sample_weights += [float(w) / n] * n

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, i):
        d = int(np.searchsorted(self.cum, i, side="right"))
        j = i if d == 0 else i - int(self.cum[d - 1])
        return self.datasets[d][j]


def build_dataset(cfg_data: dict, n_frames: int, resolution: int) -> WeightedConcat:
    aug = cfg_data.get("aug") or {}
    tf = ClipTransform(resolution, hflip_p=aug.get("hflip_p", 0.0),
                       random_resized_crop=aug.get("random_resized_crop"),
                       square_crop=aug.get("square_crop", "none"),
                       out_uint8=bool(cfg_data.get("transfer_uint8", True)))   # 기본 uint8 전송 (위 docstring)
    specs = cfg_data["datasets"]
    if not specs:
        raise ValueError("data.datasets 가 비어 있다")
    grid = cfg_data.get("window_grid")
    dsets, weights = [], []
    for s in specs:
        if not isinstance(s, dict) or "type" not in s:
            raise ValueError(f"data.datasets 항목은 dict(type=...) 여야 한다 (resolve_train.py 가 이름을 풀어 준다): {s}")
        t = s["type"]
        if t == "frames_index":
            if grid:
                s = dict(s, raw_span=int(grid["raw_span"]))
            dsets.append(FramesIndexDataset(s, n_frames, tf))
        elif t == "video_csv":
            if grid:
                raise ValueError("data.window_grid 는 frames_index 데이터셋에서만 된다 (video_csv 는 raw_span 을 못 읽는다)")
            dsets.append(make_video_csv_dataset(s, n_frames, tf))
        else:
            raise ValueError(f"data.datasets[].type={t!r}; frames_index | video_csv")
        weights.append(s.get("weight"))
    use_w = any(w is not None for w in weights)
    if use_w:
        weights = [1.0 if w is None else float(w) for w in weights]
    out = WeightedConcat(dsets, weights if use_w else None)
    out.transform = tf                     # window_grid collate 가 쓴다
    return out


# ----------------------------------------------------------------------------- masks
class MaskSampler:
    """배치마다 (masks_enc [B, K], masks_pred [B, M], info) 를 낸다. long 인덱스."""

    def __init__(self, specs: List[dict], n_frames: int, crop_size: int, patch_size: int, tubelet_size: int,
                 seed: Optional[int] = None):
        """seed 를 주면 **전용 RNG** 를 쓴다 — 모든 rank 가 같은 seed 로 만들고 step 당 한 번씩만
        부르면 rank 끼리 같은 (C, K) 가 나온다. 이게 필요한 이유 (2026-09-22):

        마스크를 DataLoader **worker 안(collate)** 에서 뽑으면 rank 마다 다른 (C, K) 가 걸릴 수
        있다. 그러면 rank 별로 predictor 토큰 수가 달라져 **VRAM 과 계산량이 불균형**해지고
        (DDP 는 제일 느린 rank 를 기다린다), 한 step 의 손실이 rank 마다 다른 과제의 평균이 된다.
        그래서 학습 루프가 step 당 한 번 여기서 뽑아 쓴다 (train.py).
        ⚠️ `WindowGridCollator` 는 창을 worker 에서 잘라야 해서 아직 이 경로가 아니다.
        """
        if not specs:
            raise ValueError("mask 스펙이 비어 있다")
        self.rng = random.Random(seed) if seed is not None else random
        self.T = n_frames // tubelet_size
        self.S = (crop_size // patch_size) ** 2
        self.N = self.T * self.S
        self.tub = tubelet_size
        self.n_frames = n_frames
        self.specs, self.weights = [], []
        for m in specs:
            t = m.get("type", "temporal_prefix")
            if t == "temporal_prefix":
                cf = m.get("context_frames", [n_frames // 2])
                cf = [int(c) for c in (cf if isinstance(cf, (list, tuple)) else [cf])]
                for c in cf:
                    if c % tubelet_size or not (0 < c < n_frames):
                        raise ValueError(f"context_frames={c}: tubelet({tubelet_size}) 정렬이고 0<C<{n_frames} 여야 한다")
                self.specs.append({"type": t, "context_frames": cf})
            elif t == "prefix_window":
                # 문맥 C 블록 -> **그 뒤 K 블록만** 예측. C 와 K 가 서로 독립이다
                # (릴리즈식 temporal_prefix 는 K = T - C 로 묶여 있다).
                cb = [int(x) for x in m.get("context_blocks", [self.T // 2])]
                kb = [int(x) for x in m.get("predict_blocks", [self.T // 2])]
                # 문맥 끝과 예측 시작 사이의 빈 블록 수 (2026-09-30, 사용자 결정 — "복사로 풀리는 바로 다음 칸" 을 줄인다).
                # 기본 [0] = 기존 동작과 비트 단위로 같다. 빈 블록은 predictor 입력에도 손실에도 없다 (RoPE 절대 위치라 자리만 건너뛴다)
                gb = [int(x) for x in m.get("gap_blocks", [0])]
                for g in gb:
                    if g < 0:
                        raise ValueError(f"gap_blocks={g}: 0 이상")
                for c in cb:
                    if not (0 < c < self.T):
                        raise ValueError(f"context_blocks={c}: 0 < C < {self.T} 여야 한다")
                for k in kb:
                    if k <= 0:
                        raise ValueError(f"predict_blocks={k}: 1 이상")
                if not any(c + g + k <= self.T for c in cb for g in gb for k in kb):
                    raise ValueError(f"prefix_window: C+G+K <= {self.T} 인 조합이 하나도 없다 "
                                     f"(context_blocks={cb}, gap_blocks={gb}, predict_blocks={kb})")
                self.specs.append({"type": t, "context_blocks": cb, "predict_blocks": kb, "gap_blocks": gb})
            elif t == "block3d":
                # 릴리즈 사전학습 마스크 (multiblock3d, tube). `window_blocks` (2026-09-27) 를 주면 창 길이(블록 수)를
                # 그 목록에서 step 마다 하나 뽑아 **그 길이의 생성기**를 쓴다 — prefix_window 팔과 같은 창 분포로
                # 맞추기 위한 것 (목록의 중복 = 가중). 안 주면 창 = n_frames 전체.
                from src.masks.multiseq_multiblock3d import _MaskGenerator
                wb = [int(x) for x in (m.get("window_blocks") or [self.T])]
                for nb in wb:
                    if not (0 < nb <= self.T):
                        raise ValueError(f"block3d window_blocks={nb}: 0 < nb <= {self.T} 여야 한다")
                gens = {}
                for nb in sorted(set(wb)):
                    gens[nb] = _MaskGenerator(
                        crop_size=(crop_size, crop_size), num_frames=nb * tubelet_size,
                        spatial_patch_size=(patch_size, patch_size), temporal_patch_size=tubelet_size,
                        spatial_pred_mask_scale=tuple(m.get("spatial_scale", (0.15, 0.15))),
                        temporal_pred_mask_scale=tuple(m.get("temporal_scale", (1.0, 1.0))),
                        aspect_ratio=tuple(m.get("aspect_ratio", (0.75, 1.5))),
                        npred=int(m.get("num_blocks", 8)),
                        max_context_frames_ratio=float(m.get("max_temporal_keep", 1.0)),
                        max_keep=m.get("max_keep"), full_complement=bool(m.get("full_complement", False)),
                        pred_full_complement=bool(m.get("pred_full_complement", False)),
                        inv_block=bool(m.get("inv_block", False)))
                self.specs.append({"type": t, "gens": gens, "window_blocks": wb, "name": str(m.get("name", len(self.specs))),
                                   "mask_index": int(m.get("mask_index", len(self.specs)))})
            else:
                raise ValueError(f"mask.type={t!r}; temporal_prefix | prefix_window | block3d")
            self.weights.append(float(m.get("weight", 1.0)))
        z = sum(self.weights)
        self.weights = [w / z for w in self.weights]

    def prefix_window(self, B: int, n_ctx_blocks: int, n_pred_blocks: int, n_gap_blocks: int = 0):
        """문맥 = 앞 C 블록 전부, 예측 = **그 뒤 K 블록만** (그 뒤는 아예 안 만든다).

        `temporal_prefix` 와 달리 K 가 T-C 에 묶이지 않는다 — 채점 그리드처럼
        "8 보고 4", "8 보고 8", "16 보고 4" 를 같은 창 안에서 만들 수 있다.
        RoPE 가 절대 인덱스를 쓰므로 창을 잘라 줄 필요가 없다.
        """
        a = n_ctx_blocks * self.S
        a2 = (n_ctx_blocks + n_gap_blocks) * self.S          # 예측 시작 (gap 만큼 건너뛴다)
        b = a2 + n_pred_blocks * self.S
        enc = torch.arange(a, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
        pred = torch.arange(a2, b, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
        return enc, pred

    def temporal_prefix(self, B: int, n_frames: int, C: int):
        """앞 C 프레임 -> 뒤 (n_frames−C) 프레임. 토큰 = tubelet-major (surprise._context_target_indices 와 동일)."""
        N = n_frames // self.tub * self.S
        n_ctx = C // self.tub * self.S
        enc = torch.arange(n_ctx, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
        pred = torch.arange(n_ctx, N, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
        return enc, pred

    def all(self, B: int):
        """**모든 스펙을 한 step 에** — 릴리즈 사전학습(app/vjepa/train.py)과 같다 (2026-09-29, `mask_mode: all`).

        릴리즈는 배치마다 short·long 마스크를 **둘 다** 만들어 predictor 를 각각 돌리고 손실을 평균한다.
        block3d 전용 (prefix 류는 창을 자르므로 스펙끼리 창이 달라진다). 창 = n_frames 전체.
        돌려주는 것: [(enc, pred, info), ...] 스펙 순서대로. rank 동기 RNG 를 그대로 쓴다.
        """
        out = []
        for s in self.specs:
            if s["type"] != "block3d":
                raise ValueError(f"mask_mode=all 은 block3d 스펙만 된다 (받은 것: {s['type']})")
            if s["window_blocks"] != [self.T]:
                raise ValueError("mask_mode=all 은 window_blocks 없이 (창 = n_frames 전체) 쓴다")
            enc, pred = s["gens"][self.T](B)
            out.append((enc.long(), pred.long(), {"type": f"block3d_{s['name']}", "context_frames": -1,
                                                  "win_blocks": self.T, "mask_index": s["mask_index"]}))
        return out

    def __call__(self, B: int):
        k = self.rng.choices(range(len(self.specs)), weights=self.weights, k=1)[0]
        s = self.specs[k]
        if s["type"] == "temporal_prefix":
            C = self.rng.choice(s["context_frames"])
            enc, pred = self.temporal_prefix(B, self.n_frames, C)
            return enc, pred, {"type": "temporal_prefix", "context_frames": C}
        if s["type"] == "prefix_window":
            C = self.rng.choice(s["context_blocks"])
            gs = s.get("gap_blocks", [0])
            if gs == [0]:                               # 기존 경로 — RNG 호출 순서까지 그대로 (재현성)
                G = 0
            else:
                okg = [g for g in gs if any(C + g + k <= self.T for k in s["predict_blocks"])] or [0]
                G = self.rng.choice(okg)
            ok = [k for k in s["predict_blocks"] if C + G + k <= self.T]
            if not ok:                                  # 그 C 로는 어떤 K 도 안 들어간다 -> 남는 만큼
                ok = [self.T - C - G]
            K = self.rng.choice(ok)
            enc, pred = self.prefix_window(B, C, K, G)
            tag = f"prefix C{C}K{K}" if G == 0 else f"prefix C{C}G{G}K{K}"
            return enc, pred, {"type": tag, "context_frames": C * self.tub,
                               "predict_blocks": K, "context_blocks": C, "gap_blocks": G, "win_blocks": C + G + K}
        nb = self.rng.choice(s["window_blocks"])
        enc, pred = s["gens"][nb](B)
        # 릴리즈처럼 마스크 스펙마다 다른 mask token 을 쓴다 (mask_index = 스펙 순서). 창 길이는 win_blocks 로 알린다
        return enc.long(), pred.long(), {"type": f"block3d_{s['name']}", "context_frames": -1,
                                         "win_blocks": nb, "mask_index": s["mask_index"]}


class WindowGridSampler:
    """채점 그리드(intphys1_sliding 의 skip × window × 시작점 × C)를 학습 샘플 공간으로 쓴다. 배치마다 하나.

    config `data.window_grid`
      raw_span     : 데이터셋이 읽어 둘 연속 raw 프레임 수 (모든 셀의 stride×(n_frames−1)+1 이상)
      raw_frames   : 원본 영상 프레임 수 (frame_budget 계산용, 기본 100)
      start_step   : 시작점 간격, **샘플 프레임 단위** (채점 stride, 기본 2)
      frame_budget : official = (raw_frames−1)//stride 장만 쓴다 (공식 num_frames = 99//skip) | full = raw_span//stride
      cells        : [{stride, n_frames, context_frames: [..], starts: [raw 오프셋..](선택), weight(선택)}]
      weights      : auto = 셀마다 (시작점 수 × C 수) 비례 (= 채점 창 빈도) | [w, ...]
      jitter_raw   : 시작점에 더할 raw 프레임 무작위 오프셋 상한 (0 = 채점과 같은 자리)
    시작점: starts 를 안 주면 sampled index 0, start_step, 2·start_step, … 중 창이 budget 안에 드는 것 → raw 오프셋 = idx × stride.
    """

    def __init__(self, grid: dict, tubelet_size: int):
        self.raw_span = int(grid["raw_span"])
        raw_frames = int(grid.get("raw_frames", 100))
        step = int(grid.get("start_step", 2))
        budget_mode = str(grid.get("frame_budget", "official"))
        self.jitter = int(grid.get("jitter_raw", 0))
        self.cells = []
        for c in grid["cells"]:
            stride, n = int(c["stride"]), int(c["n_frames"])
            if n % tubelet_size:
                raise ValueError(f"window_grid cell n_frames={n} 가 tubelet({tubelet_size}) 배수가 아니다")
            cf = [int(x) for x in c["context_frames"]]
            for x in cf:
                if x % tubelet_size or not (0 < x < n):
                    raise ValueError(f"window_grid cell context_frames={x}: tubelet 정렬이고 0<C<{n} 여야 한다")
            if c.get("starts") is not None:
                starts = [int(s) for s in c["starts"]]
            else:
                budget = (raw_frames - 1) // stride if budget_mode == "official" else self.raw_span // stride
                starts = [i * stride for i in range(0, budget - n + 1, step)]
            if not starts:
                raise ValueError(f"window_grid cell stride={stride} n_frames={n}: 창이 하나도 안 들어간다")
            need = max(starts) + (n - 1) * stride + self.jitter
            if need >= self.raw_span:
                raise ValueError(f"window_grid cell stride={stride} n_frames={n}: 마지막 raw 오프셋 {need} >= raw_span {self.raw_span}")
            self.cells.append({"stride": stride, "n_frames": n, "context_frames": cf, "starts": starts,
                               "n_windows": len(starts) * len(cf), "weight": c.get("weight")})
        w = grid.get("weights", "auto")
        if w == "auto" or w is None:
            self.weights = [float(c["weight"]) if c["weight"] is not None else float(c["n_windows"]) for c in self.cells]
        else:
            if len(w) != len(self.cells):
                raise ValueError("window_grid.weights 길이가 cells 와 다르다")
            self.weights = [float(x) for x in w]
        z = sum(self.weights)
        if z <= 0:
            raise ValueError("window_grid.weights 합이 0 이다")
        self.weights = [x / z for x in self.weights]

    def describe(self) -> str:
        return " | ".join(f"s{c['stride']}_w{c['n_frames']}: {len(c['starts'])} starts x C{c['context_frames']} "
                          f"= {c['n_windows']} (p={p:.2f})" for c, p in zip(self.cells, self.weights))

    def sample(self):
        k = random.choices(range(len(self.cells)), weights=self.weights, k=1)[0]
        c = self.cells[k]
        o = random.choice(c["starts"]) + (random.randint(0, self.jitter) if self.jitter else 0)
        C = random.choice(c["context_frames"])
        return {"cell": k, "stride": c["stride"], "n_frames": c["n_frames"], "offset": o, "context_frames": C}


class WindowGridCollator:
    """raw 버퍼 배치 -> 배치마다 (stride, n_frames, 시작점, C) 하나로 잘라 transform 하고 temporal_prefix 마스크를 만든다.
    DataLoader worker 안에서 돈다 (num_workers>0 이면 collate_fn 은 worker 가 실행한다)."""

    def __init__(self, grid_sampler: WindowGridSampler, mask_sampler: MaskSampler, transform: ClipTransform):
        self.grid = grid_sampler
        self.masks = mask_sampler
        self.transform = transform

    def __call__(self, batch):
        w = self.grid.sample()
        o, s, n = w["offset"], w["stride"], w["n_frames"]
        clips = torch.stack([self.transform(b[0][o: o + n * s: s]) for b in batch])   # (B, 3, n, H, W)
        enc, pred = self.masks.temporal_prefix(len(batch), n, w["context_frames"])
        info = {"type": f"grid s{s}_w{n}", "context_frames": w["context_frames"], "n_frames": n, "stride": s, "offset": o}
        return clips, enc, pred, info


class TrainCollator:
    """DataLoader collate_fn. worker 안에서 마스크를 뽑는다 (block3d 의 step 카운터가 공유 Value 라 그래도 된다)."""

    def __init__(self, mask_sampler: Optional[MaskSampler] = None):
        # None 이면 마스크를 **학습 루프가** 만든다 (rank 동기화. MaskSampler docstring 참고).
        self.mask_sampler = mask_sampler

    def __call__(self, batch):
        clips = torch.stack([b[0][0] for b in batch])            # (B, 3, T, H, W)
        if self.mask_sampler is None:
            return clips, None, None, None
        enc, pred, info = self.mask_sampler(len(batch))
        return clips, enc, pred, info


def build_loader(dataset: WeightedConcat, collator, batch_size: int, rank: int, world_size: int,
                 num_workers: int, pin_mem: bool, persistent: bool, seed: int, drop_last: bool = True,
                 prefetch_factor: int = 2, timeout_s: float = 0.0):
    """⚠️ `prefetch_factor` 를 노출하는 이유 (2026-09-22).

    in-flight 클립 수 = world_size x num_workers x prefetch_factor x batch_size 다.
    (8 x 12 x 2 x 28 = **5,376 클립**) 코어가 그만큼으로 쪼개지면 **배치 하나가 완성되지 않아**
    rank 가 첫 배치를 못 받고 DDP 전체가 멈춘다 (실측: 워커 96 개가 CPU 8,564 % 를 쓰는데
    step 로그 0 회). 처리량이 아니라 **지연**의 문제라 워커를 늘리면 더 나빠진다.
    자연 영상 디코드 비용(K400 314 ms/clip)을 기준으로 in-flight 를 몇 배치 수준으로 묶을 것.
    """
    if dataset.sample_weights is not None:
        from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
        sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    loader = torch.utils.data.DataLoader(
        dataset, collate_fn=collator, sampler=sampler, batch_size=batch_size, drop_last=drop_last,
        pin_memory=pin_mem, num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent,
        # ⚠️ timeout: 워커 하나가 디코드에서 멈추면(K400 의 잘린 서브클립, 2026-09-22) DataLoader 는
        #    배치를 **순서대로만** 내놓기 때문에 그 rank 가 영원히 멈추고 DDP 전체가 all-reduce 에서
        #    공회전한다. 침묵 대신 RuntimeError 로 죽게 한다.
        timeout=float(timeout_s) if num_workers > 0 else 0,
        **({"prefetch_factor": int(prefetch_factor)} if num_workers > 0 else {}))
    return loader, sampler
