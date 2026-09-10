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

마스크 (MaskSampler)
  temporal_prefix   context = 앞 C 프레임 전부, target = 뒤 T−C 프레임 전부.
                    surprise_c16t32 채점과 같은 토폴로지. C 는 context_frames 에서 배치마다 고른다.
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
    """

    def __init__(self, resolution: int, hflip_p: float = 0.0, random_resized_crop: Optional[dict] = None):
        self.resolution = int(resolution)
        self.hflip_p = float(hflip_p)
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
        return 0, 0, H, W  # 실패하면 전체

    def __call__(self, buffer) -> torch.Tensor:
        if not torch.is_tensor(buffer):
            buffer = torch.from_numpy(np.ascontiguousarray(buffer))
        x = buffer.permute(3, 0, 1, 2).float().div_(255.0)          # (3, T, H, W) 0..1
        _, T, H, W = x.shape
        r = self.resolution
        if self.rrc:
            top, left, h, w = self._rrc_box(H, W)
            x = x[:, :, top: top + h, left: left + w]
        if tuple(x.shape[-2:]) != (r, r):
            # 채점과 같은 커널: bilinear / antialias=False (공식 Garrido 전처리와 동일)
            x = F.interpolate(x.transpose(0, 1), size=(r, r), mode="bilinear",
                              align_corners=False).transpose(0, 1)
        if self.hflip_p > 0 and random.random() < self.hflip_p:
            x = x.flip(-1)
        x = x.contiguous()
        return (x - MEAN) / STD


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
        for st in self.starts:
            p = self._path(rows[0], st + (self.n_frames - 1) * self.stride)
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

    def __getitem__(self, i: int):
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
              center_sampling=bool(spec.get("center_sampling", False)))
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
                       random_resized_crop=aug.get("random_resized_crop"))
    specs = cfg_data["datasets"]
    if not specs:
        raise ValueError("data.datasets 가 비어 있다")
    dsets, weights = [], []
    for s in specs:
        if not isinstance(s, dict) or "type" not in s:
            raise ValueError(f"data.datasets 항목은 dict(type=...) 여야 한다 (resolve_train.py 가 이름을 풀어 준다): {s}")
        t = s["type"]
        if t == "frames_index":
            dsets.append(FramesIndexDataset(s, n_frames, tf))
        elif t == "video_csv":
            dsets.append(make_video_csv_dataset(s, n_frames, tf))
        else:
            raise ValueError(f"data.datasets[].type={t!r}; frames_index | video_csv")
        weights.append(s.get("weight"))
    use_w = any(w is not None for w in weights)
    if use_w:
        weights = [1.0 if w is None else float(w) for w in weights]
    return WeightedConcat(dsets, weights if use_w else None)


# ----------------------------------------------------------------------------- masks
class MaskSampler:
    """배치마다 (masks_enc [B, K], masks_pred [B, M], info) 를 낸다. long 인덱스."""

    def __init__(self, specs: List[dict], n_frames: int, crop_size: int, patch_size: int, tubelet_size: int):
        if not specs:
            raise ValueError("mask 스펙이 비어 있다")
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
            elif t == "block3d":
                from src.masks.multiseq_multiblock3d import _MaskGenerator
                gen = _MaskGenerator(
                    crop_size=(crop_size, crop_size), num_frames=n_frames,
                    spatial_patch_size=(patch_size, patch_size), temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=tuple(m.get("spatial_scale", (0.15, 0.15))),
                    temporal_pred_mask_scale=tuple(m.get("temporal_scale", (1.0, 1.0))),
                    aspect_ratio=tuple(m.get("aspect_ratio", (0.75, 1.5))),
                    npred=int(m.get("num_blocks", 8)),
                    max_context_frames_ratio=float(m.get("max_temporal_keep", 1.0)),
                    max_keep=m.get("max_keep"), full_complement=bool(m.get("full_complement", False)),
                    pred_full_complement=bool(m.get("pred_full_complement", False)),
                    inv_block=bool(m.get("inv_block", False)))
                self.specs.append({"type": t, "gen": gen})
            else:
                raise ValueError(f"mask.type={t!r}; temporal_prefix | block3d")
            self.weights.append(float(m.get("weight", 1.0)))
        z = sum(self.weights)
        self.weights = [w / z for w in self.weights]

    def __call__(self, B: int):
        k = random.choices(range(len(self.specs)), weights=self.weights, k=1)[0]
        s = self.specs[k]
        if s["type"] == "temporal_prefix":
            C = random.choice(s["context_frames"])
            n_ctx = C // self.tub * self.S
            enc = torch.arange(n_ctx, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
            pred = torch.arange(n_ctx, self.N, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
            return enc, pred, {"type": "temporal_prefix", "context_frames": C}
        enc, pred = s["gen"](B)
        return enc.long(), pred.long(), {"type": "block3d", "context_frames": -1}


class TrainCollator:
    """DataLoader collate_fn. worker 안에서 마스크를 뽑는다 (block3d 의 step 카운터가 공유 Value 라 그래도 된다)."""

    def __init__(self, mask_sampler: MaskSampler):
        self.mask_sampler = mask_sampler

    def __call__(self, batch):
        clips = torch.stack([b[0][0] for b in batch])            # (B, 3, T, H, W)
        enc, pred, info = self.mask_sampler(len(batch))
        return clips, enc, pred, info


def build_loader(dataset: WeightedConcat, collator, batch_size: int, rank: int, world_size: int,
                 num_workers: int, pin_mem: bool, persistent: bool, seed: int, drop_last: bool = True):
    if dataset.sample_weights is not None:
        from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
        sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    loader = torch.utils.data.DataLoader(
        dataset, collate_fn=collator, sampler=sampler, batch_size=batch_size, drop_last=drop_last,
        pin_memory=pin_mem, num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent)
    return loader, sampler
