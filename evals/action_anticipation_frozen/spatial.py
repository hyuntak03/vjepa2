# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""EK100 anticipation 의 spatial 변환 — 종횡비를 절대 늘리지 않는 세 모드.

`dataloader.VideoTransform` 이 `spatial_mode` 로 골라 쓴다. **세 모드 다 cv2.resize 한 번 +
크롭/패딩**이고, 원본 (1920x1080) 을 바로 정사각형으로 눌러 담는 경로는 없다.

| mode | 무엇 | 1920x1080 -> 256x256 |
|---|---|---|
| **`short_side`** | **짧은변을 출력 높이에 맞추고, 긴변만 가운데로 자른다** (별칭 `cover`) | 1080 -> **256** (전부 보존), 1920 -> 455 -> 가운데 **256** |
| `center_crop` | 짧은변 -> `int(H*256/224)` = 292, 그 뒤 CenterCrop | 1080 -> 292 -> 가운데 256 (**1.14배 더 확대**) |
| `letterbox` | 긴변을 맞추고 남는 자리는 상수 패딩. 아무것도 안 자른다 | 1920 -> 256, 1080 -> 144, 위아래 패딩 |

`short_side` 가 요청받은 규칙이다 — **짧은변을 먼저 모델 입력(256)에 맞추고 나머지 변을 자른다.**
공식 `center_crop` 과의 차이는 **확대율 하나**다: 공식은 짧은변을 292 로 키우고 256 을 떼므로
세로 화각의 12.5% 를 더 버린다. `short_side` 는 **세로를 통째로 보존**한다.
가로는 둘 다 자른다 (정사각 출력이라 피할 수 없다). 어느 쪽도 비율을 늘리지 않는다.

구현은 `scale = max(out_h/h, out_w/w)` 후 CenterCrop 이다 (CSS `object-fit: cover`).
정사각 출력이면 짧은변이 정확히 `out_h` 가 된다 — 세로가 짧은 16:9 든 4:3 이든 같다.

⚠️ `out_w != out_h` 로 비정사각 출력을 만들면 encoder 는 되지만 **predictor 의 RoPE 가
정사각을 가정**한다. 그때는 `modelcustom.vit_encoder_predictor_concat_ar_nonsquare` 를
`module_name` 으로 지정할 것 (`resolve.py` 가 검사한다).
"""

import numbers
from logging import getLogger

import numpy as np

logger = getLogger()


def _as_thwc(clip):
    """decord 출력([T,H,W,C] ndarray) / 리스트 둘 다 [T,H,W,C] ndarray 로."""
    if isinstance(clip, np.ndarray):
        if clip.ndim != 4:
            raise ValueError(f"expected [T,H,W,C], got {clip.shape}")
        return clip
    return np.stack(clip, axis=0)


def _resize(clip, new_h, new_w, interpolation="bilinear"):
    import cv2

    inter = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
    return np.stack([cv2.resize(f, (new_w, new_h), interpolation=inter) for f in clip], axis=0)


def _center_crop(clip, out_h, out_w):
    _, h, w, _ = clip.shape
    if h < out_h or w < out_w:
        raise ValueError(f"center crop {out_h}x{out_w} > clip {h}x{w}")
    i = (h - out_h) // 2
    j = (w - out_w) // 2
    return clip[:, i : i + out_h, j : j + out_w, :]


class ResizeShortSideCenterCrop:
    """짧은변을 출력 크기에 맞추고 긴변만 가운데로 자른다. 종횡비 보존.

    `scale = max(out_h/h, out_w/w)` 후 CenterCrop(out_h, out_w).
    정사각 출력(out_h == out_w)이면 **짧은변이 정확히 out_h 가 되고 긴변만 잘린다**
    = 요청받은 규칙. 공식 `center_crop` 은 짧은변을 out_h*256/224 로 키워 더 확대한다.
    """

    def __init__(self, size, interpolation="bilinear"):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        self.out_h, self.out_w = int(size[0]), int(size[1])
        self.interpolation = interpolation

    def __call__(self, clip):
        clip = _as_thwc(clip)
        _, h, w, _ = clip.shape
        scale = max(self.out_h / h, self.out_w / w)
        new_h, new_w = max(self.out_h, int(round(h * scale))), max(self.out_w, int(round(w * scale)))
        if (new_h, new_w) != (h, w):
            clip = _resize(clip, new_h, new_w, self.interpolation)
        return _center_crop(clip, self.out_h, self.out_w)


class ResizeLetterbox:
    """scale = min(out_h/h, out_w/w) 로 줄인 뒤 남는 자리를 상수로 채운다. 아무것도 안 자른다.

    ⚠️ pretrain 은 letterbox 를 본 적이 없다. FOV 전체가 필요할 때의 대조군으로만 쓸 것.
    """

    def __init__(self, size, interpolation="bilinear", pad_value=None):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        self.out_h, self.out_w = int(size[0]), int(size[1])
        self.interpolation = interpolation
        self.pad_value = pad_value  # None -> 클립 평균색

    def __call__(self, clip):
        clip = _as_thwc(clip)
        t, h, w, c = clip.shape
        scale = min(self.out_h / h, self.out_w / w)
        new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        new_h, new_w = min(new_h, self.out_h), min(new_w, self.out_w)
        resized = _resize(clip, new_h, new_w, self.interpolation)
        fill = self.pad_value
        if fill is None:
            fill = resized.reshape(-1, c).mean(0)
        out = np.empty((t, self.out_h, self.out_w, c), dtype=resized.dtype)
        out[:] = np.asarray(fill, dtype=resized.dtype)
        i = (self.out_h - new_h) // 2
        j = (self.out_w - new_w) // 2
        out[:, i : i + new_h, j : j + new_w, :] = resized
        return out


def build_eval_spatial(spatial_mode, crop_height, crop_width, interpolation="bilinear"):
    """val/test 용 결정적 spatial 변환 리스트를 만든다 (ClipToTensor 이전 단계).

    :returns: video_transforms.Compose 에 넣을 callable 들의 list
    """
    import src.datasets.utils.video.transforms as video_transforms

    if spatial_mode == "center_crop":
        # 공식 구현 그대로: 짧은변을 crop*256/224 로 키우고 가운데를 정사각으로 자른다
        short_side = int(crop_height * 256 / 224)
        return [
            video_transforms.Resize(short_side, interpolation=interpolation),
            video_transforms.CenterCrop(size=(crop_height, crop_width)),
        ]
    if spatial_mode in ("short_side", "cover"):   # cover 는 옛 이름 (별칭)
        return [ResizeShortSideCenterCrop((crop_height, crop_width), interpolation=interpolation)]
    if spatial_mode == "letterbox":
        return [ResizeLetterbox((crop_height, crop_width), interpolation=interpolation)]
    raise ValueError(f"unknown spatial_mode={spatial_mode!r} (short_side | center_crop | letterbox)")


ResizeCoverCenterCrop = ResizeShortSideCenterCrop  # 옛 이름
