# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pathlib
import warnings
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torchvision
from decord import cpu, VideoReader

from src.datasets.utils.dataloader import (
    ConcatIndices,
    MonitoredDataset,
    NondeterministicDataLoader,
)
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    uniform_sampling=False,
    center_sampling=False,
    headtail_sampling=False,
    keystones_by_path=None,
):
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        uniform_sampling=uniform_sampling,
        center_sampling=center_sampling,
        headtail_sampling=headtail_sampling,
        keystones_by_path=keystones_by_path,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Worker ID will replace '%w'
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("VideoDataset dataset created")
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    if deterministic:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    else:
        data_loader = NondeterministicDataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    logger.info("VideoDataset unsupervised data loader created")

    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
        uniform_sampling=False,  # sample fpc frames evenly across the whole video (ignores frame_step)
        center_sampling=False,   # take the CENTER fpc frames (contiguous window around video midpoint).
                                 # Ignores frame_step and uniform_sampling when true.
        headtail_sampling=False, # take the FIRST ceil(fpc/2) frames + the LAST floor(fpc/2) frames
                                 # (a "front-half + back-half" clip; drops the middle). For a 48-frame
                                 # clip with fpc=16 -> indices [0..7]+[40..47]. Ignores frame_step /
                                 # num_clips. Precedence: keystones > center > headtail > uniform > default.
        keystones_by_path=None,  # dict[str, int]: per-video keystone (break-point) frame index.
                                 # When provided, sampling centers a contiguous fpc-frame window on
                                 # this frame for every path in the dict; other paths fall through
                                 # to center/uniform/default sampling. This is the paper (Joseph
                                 # 2026) 'keystone-centered' sampling: 16 frames symmetric around
                                 # the physics break-point. Takes precedence over center_sampling
                                 # and uniform_sampling.
        decord_threads=-1,       # 2026-09-27 Ariel 이식: 리더당 decord 스레드. 기본 -1 = 기존 동작 (리더마다 전 코어).
                                 # ⚠️ DataLoader worker 를 여러 개 쓰면 과다구독이다 — 학습은 1~2 를 준다 (app/vjepa_frozen/data.py)
        resample_on_error=False, # 2026-09-27: 디코드 **예외**를 다른 영상으로 재추첨할지. 기본 False = 기존 동작 (예외로 죽는다).
                                 # ⚠️ 평가 코드는 켜지 말 것 — 다른 영상 · 라벨이 조용히 들어간다. 학습 로더 (make_video_csv_dataset) 만 켠다
    ):
        self.data_paths = data_paths
        self.decord_threads = int(decord_threads)
        self.resample_on_error = bool(resample_on_error)
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.uniform_sampling = uniform_sampling
        self.center_sampling = center_sampling
        self.headtail_sampling = headtail_sampling
        self.keystones_by_path = keystones_by_path or {}
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps

        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(
                f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}."
            )
        # 2026-09-27: 예전 `assert fstp > 0` 이 잡던 설정 오류는 여기서 바로 죽인다 (영상별 건너뛰기는 파일 문제만)
        if frame_step is not None and int(frame_step) <= 0:
            raise ValueError(f"frame_step 은 1 이상이어야 한다: {frame_step}")

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError(
                    "Frames per clip not properly specified for NFS data paths"
                )
            self.dataset_fpcs = dataset_fpcs

        if VideoReader is None:
            raise ImportError(
                'Unable to import "decord" which is required to read videos.'
            )

        # Load video paths and labels
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:

            if data_path[-4:] == ".csv":
                try:
                    data = pd.read_csv(data_path, header=None, delimiter=" ")
                except pd.errors.ParserError:
                    # In image captioning datasets where we have space, we use :: as delimiter.
                    data = pd.read_csv(data_path, header=None, delimiter="::")
                if data.shape[1] == 2:
                    # 기존 경로 그대로 (경로에 공백이 없는 csv — 우리 csv 전부). 라벨 dtype 도 pandas 그대로.
                    samples += list(data.values[:, 0])
                    labels += list(data.values[:, 1])
                    num_samples = len(data)
                else:
                    # 2026-09-27 Ariel 이식 (0923 L210-226): 경로에 **공백**이 있으면 (K400 클래스 폴더
                    #   "bouncing on trampoline") pandas 가 열 수를 최대치로 맞춰 경로 "/x/bouncing", 라벨 "on" 으로
                    #   **조용히 오파싱**하거나 ParserError -> "::" 폴백에서 열이 1 개가 된다. 그때만 마지막 구분자 기준으로
                    #   한 번 자른다. 열이 정확히 2 개인 기존 csv 는 위 분기라 결과가 바뀌지 않는다.
                    samples_, labels_ = self._read_csv_rsplit(data_path)
                    samples += samples_
                    labels += labels_
                    num_samples = len(samples_)
                self.num_samples_per_dataset.append(num_samples)

            elif data_path[-4:] == ".npy":
                data = np.load(data_path, allow_pickle=True)
                data = list(map(lambda x: repr(x)[1:-1], data))
                samples += data
                labels += [0] * len(data)
                num_samples = len(data)
                self.num_samples_per_dataset.append(len(data))

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels

    @staticmethod
    def _read_csv_rsplit(data_path):
        """'<경로><sep><라벨>' 을 마지막 구분자 기준으로 자른다 (경로 안 공백 허용). 2026-09-27 Ariel 이식."""
        rows = [ln.rstrip() for ln in open(data_path, encoding="utf-8") if ln.strip()]   # \r · 끝 공백도 벗긴다
        if not rows:
            raise ValueError(f"{data_path}: 빈 csv")
        sep = "::" if "::" in rows[0] else " "
        pairs = [r.rsplit(None if sep == " " else sep, 1) for r in rows]   # 공백 구분이면 연속 공백도 하나로
        bad = [i for i, q in enumerate(pairs) if len(q) != 2]
        if bad:
            raise ValueError(f"{data_path}: '<경로>{sep}<라벨>' 형식이 아닌 줄 {len(bad)}개 "
                             f"(첫 줄 {bad[0]}): {rows[bad[0]][:120]!r}")

        def _label(s):                       # pandas 와 같은 꼴: int -> float -> str
            for cast in (int, float):
                try:
                    return cast(s)
                except ValueError:
                    pass
            return s

        return [q[0] for q in pairs], [_label(q[1]) for q in pairs]

    # 2026-09-27 Ariel 이식: 디코드 **예외**를 재추첨으로 흡수하되, 한 __getitem__ 안에서 예외가 이만큼 쌓이면
    #   (decord 자체가 깨진 경우 등) 무한 재추첨 대신 마지막 예외로 죽는다. 빈 반환 (짧은 영상 등) 은 기존처럼 세지 않는다.
    max_decode_exceptions = 50

    def __getitem__(self, index):
        sample = self.samples[index]
        loaded_sample = False
        n_exc = 0
        # Keep trying to load videos until you find a valid sample
        while not loaded_sample:
            if not isinstance(sample, str):
                logger.warning("Invalid sample.")
            else:
                # ⚠️ 2026-09-22 (Ariel, 2026-09-27 이식): 디코드 **예외**도 재시도로 흡수한다. 예전엔 빈 반환만 잡아서,
                #    헤더 fps 가 깨진 파일 하나(`fstp = fps//target = 0` -> AssertionError)가
                #    worker -> rank -> DDP 전체를 죽였다. 파일 경로를 한 번 경고하고 다른 인덱스로 간다.
                try:
                    if sample.split(".")[-1].lower() in ("jpg", "png", "jpeg"):
                        loaded_sample = self.get_item_image(index)
                    else:
                        loaded_sample = self.get_item_video(index)
                except Exception as e:                                   # noqa: BLE001
                    if not self.resample_on_error:                       # 기존 동작 (평가 · upstream 호출자)
                        raise
                    n_exc += 1
                    if n_exc >= self.max_decode_exceptions:
                        raise RuntimeError(f"VideoDataset: 한 샘플을 찾는 동안 디코드 예외 {n_exc} 회 — 데이터/디코더를 확인할 것 "
                                           f"(마지막 {sample})") from e
                    warnings.warn(f"decode failed, resampling: {sample} ({type(e).__name__}: {e})")
                    loaded_sample = False

            if not loaded_sample:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        return loaded_sample

    def get_item_video(self, index):
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_idx]

        buffer, clip_indices = self.loadvideo_decord(
            sample, frames_per_clip
        )  # [T H W 3]
        loaded_video = len(buffer) > 0
        if not loaded_video:
            return

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """Split video into a list of clips"""
            fpc = frames_per_clip
            nc = self.num_clips
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices

    def get_item_image(self, index):
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        fpc = self.dataset_fpcs[dataset_idx]

        try:
            image_tensor = torchvision.io.read_image(
                path=sample, mode=torchvision.io.ImageReadMode.RGB
            )
        except Exception:
            return
        label = self.labels[index]
        clip_indices = [np.arange(start=0, stop=fpc, dtype=np.int32)]

        # Expanding the input image [3, H, W] ==> [T, 3, H, W]
        buffer = image_tensor.unsqueeze(dim=0).repeat((fpc, 1, 1, 1))
        buffer = buffer.permute((0, 2, 3, 1))  # [T, 3, H, W] ==> [T H W 3]

        if self.shared_transform is not None:
            # Technically we can have only transform, doing this just for the sake of consistency with videos.
            buffer = self.shared_transform(buffer)

        if self.transform is not None:
            buffer = [self.transform(buffer)]

        return buffer, label, clip_indices

    def loadvideo_decord(self, sample, fpc):
        """Load video content using Decord"""

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f"video path not found {fname=}")
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize > self.filter_long_videos:
            warnings.warn(f"skipping long video of size {_fsize=} (bytes)")
            return [], None

        try:
            # 2026-09-27 Ariel 이식: 스레드 수를 ctor 인자로 (기본 -1 = 기존). -1 은 **리더마다 전 코어**라
            #   worker 가 여럿이면 (rank 8 x worker 8 = 64 리더) 컨텍스트 스위치로 첫 배치조차 안 나온다 (CLAUDE.md §7-1).
            vr = VideoReader(fname, num_threads=self.decord_threads, ctx=cpu(0))
        except Exception:
            return [], None

        fstp = self.frame_step
        video_fps = 0                            # 2026-09-27 Ariel 이식: get_avg_fps 가 던지면 예전엔 NameError 였다
        if self.duration is not None or self.fps is not None:
            try:
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)

            if self.duration is not None:
                assert self.fps is None
                fstp = int(self.duration * video_fps / fpc)
            else:
                assert self.duration is None
                fstp = video_fps // self.fps

        # 2026-09-27 Ariel 이식 (0923 L370-377): 예전엔 `assert fstp > 0` 이라 파일 하나가 worker 를 죽였다.
        if fstp is not None and fstp <= 0 and self.fps is not None and video_fps > 0:
            # 헤더 fps 가 목표 fps 보다 낮다 (K400 에 fps 10~11 이 0.6%). 원래 upstream 은 assert 로 죽고,
            # 2026-09-22 첫 학습은 건너뛰었다. 목표 fps 자체가 ceil(fps)//target 이라 25→12.5, 30→15 로
            # 이미 ±25% 흔들리므로, 원 fps 그대로(step 1) 쓰는 것이 같은 허용 범위 안이다. (Ariel 쪽 사용자 결정 2026-09-22)
            fstp = 1
        if fstp is None or fstp <= 0:            # fps 를 아예 못 읽은 파일 -> 이 영상은 못 쓴다
            warnings.warn(f"skipping video with fps {video_fps if self.fps is not None else '?'} (fstp={fstp}): {sample}")
            self._n_fstp_skip = getattr(self, "_n_fstp_skip", 0) + 1     # 모든 파일이 이러면 무한 재추첨이 된다 -> 한도에서 죽인다
            if self._n_fstp_skip >= 10 * self.max_decode_exceptions and self._n_fstp_skip >= len(self.samples):
                raise RuntimeError(f"VideoDataset: fps 를 못 읽거나 fstp<=0 인 파일이 {self._n_fstp_skip} 번 — 설정 (fps/duration) 을 확인할 것")
            return [], None
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"skipping video of length {len(vr)}")
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # keystone-centered sampling: take a contiguous fpc-frame window centered on the video's
        # physics break-point (per keystones_by_path lookup). tick is the frame index (0-indexed).
        # window = [tick - fpc//2 : tick + fpc//2] (clipped to video bounds). Highest precedence.
        _ks = getattr(self, "keystones_by_path", None)
        if _ks and sample in _ks:
            n = len(vr)
            tick = int(_ks[sample])
            half = fpc // 2
            start = max(0, tick - half)
            end = min(n, start + fpc)
            start = max(0, end - fpc)  # slide back if clipped at end
            indices = np.arange(start, end).astype(np.int64)
            if len(indices) < fpc:
                indices = np.concatenate([indices, np.full(fpc - len(indices), n - 1)]).astype(np.int64)
            buffer = vr.get_batch(list(indices)).asnumpy()
            return buffer, [indices]

        # center_sampling: take a contiguous fpc-frame window centered on the video midpoint.
        # For IntPhys (native 100 frames, fpc=16) this gives indices [42..57]. Ignores
        # frame_step / num_clips. Precedence: center_sampling > uniform_sampling > default.
        if getattr(self, "center_sampling", False):
            n = len(vr)
            if n >= fpc:
                start = (n - fpc) // 2
                indices = np.arange(start, start + fpc).astype(np.int64)
            else:
                # video shorter than fpc: pad with the last frame
                indices = np.concatenate(
                    [np.arange(n), np.full(fpc - n, n - 1)]
                ).astype(np.int64)
            buffer = vr.get_batch(list(indices)).asnumpy()
            return buffer, [indices]

        # headtail_sampling: take the FIRST ceil(fpc/2) frames + the LAST floor(fpc/2) frames
        # (a "front-half + back-half" clip; the middle is dropped). For a 48-frame clip with
        # fpc=16 this yields indices [0..7]+[40..47] -- the start and end of the motion. The
        # 8|8 split aligns with tubelet_size=2, so no tubelet straddles the head/tail gap.
        # Ignores frame_step / num_clips. Precedence: keystones > center > headtail > uniform > default.
        if getattr(self, "headtail_sampling", False):
            n = len(vr)
            n_head = (fpc + 1) // 2          # ceil -> front half
            n_tail = fpc - n_head            # floor -> back half
            if n >= fpc:
                head = np.arange(0, n_head)
                tail = np.arange(n - n_tail, n)
                indices = np.concatenate([head, tail]).astype(np.int64)
            else:
                # video shorter than fpc: take all frames, pad with the last frame
                indices = np.concatenate(
                    [np.arange(n), np.full(fpc - n, n - 1)]
                ).astype(np.int64)
            buffer = vr.get_batch(list(indices)).asnumpy()
            return buffer, [indices]

        # uniform_sampling: pick `fpc` frames evenly across the WHOLE video (length-agnostic;
        # ignores frame_step / num_clips). Avoids the contiguous-window default that, when
        # fpc*frame_step < len(video), only covers a sub-segment -> sub-patch motion per tubelet.
        if getattr(self, "uniform_sampling", False):
            n = len(vr)
            indices = np.clip(np.linspace(0, n - 1, num=fpc).round(), 0, n - 1).astype(np.int64)
            buffer = vr.get_batch(list(indices)).asnumpy()
            return buffer, [indices]

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - partition_len // fstp) * partition_len,
                        )
                    )
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - sample_len // fstp) * sample_len,
                        )
                    )
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)
