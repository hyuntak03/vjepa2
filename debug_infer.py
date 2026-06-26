# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# V-JEPA 2 인코더 디버깅용 최소 스크립트.
# sample_video/ 안의 mp4 들을 64프레임으로 샘플링해 인코더에 통과시키고
# patch feature 를 출력한다. (probe / 라벨 / 데이터셋 CSV 불필요)
#
# 실행:   python debug_infer.py
# 디버깅:  VSCode 의 "Encoder debug (debug_infer.py)" 구성으로 F5,
#         또는 src/models/vision_transformer.py:161 (forward) 에 breakpoint.

import glob

import numpy as np
import torch
from decord import VideoReader

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_large_rope

# ----------------------------------------------------------------------- #
#  설정
# ----------------------------------------------------------------------- #
CKPT = glob.glob(
    "checkpoint/models--facebook--vjepa2-vitl-fpc64-256/snapshots/*/original/model.pth"
)[0]
CHECKPOINT_KEY = "encoder"  # 'encoder' 또는 'target_encoder'
IMG_SIZE = 256
NUM_FRAMES = 64  # fpc64 모델이므로 64. tubelet_size=2 라 반드시 짝수.
TUBELET_SIZE = 2
VIDEO_GLOB = "sample_video/*.mp4"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(img_size):
    short_side = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def sample_frames(path, num_frames):
    """비디오 전체 길이에서 num_frames 장을 균등 샘플링 (짧은 영상도 안전)."""
    vr = VideoReader(path)
    n = len(vr)
    idx = np.linspace(0, n - 1, num_frames).round().astype(int)
    print(f"  {path}: total {n} frames -> sample {num_frames}")
    return vr.get_batch(idx).asnumpy()  # T x H x W x C


def load_encoder(ckpt, img_size, num_frames):
    model = vit_large_rope(
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=TUBELET_SIZE,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)[CHECKPOINT_KEY]
    # 'module.' / 'backbone.' prefix 제거
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"loaded {ckpt} [{CHECKPOINT_KEY}]\n  missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    transform = build_transform(IMG_SIZE)
    model = load_encoder(CKPT, IMG_SIZE, NUM_FRAMES).to(device).eval()

    videos = sorted(glob.glob(VIDEO_GLOB))
    assert videos, f"no videos matched {VIDEO_GLOB}"

    for vp in videos:
        frames = sample_frames(vp, NUM_FRAMES)  # T x H x W x C
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)  # T x C x H x W
        x = transform(video).unsqueeze(0).to(device)  # 1 x C x T x H x W
        print(f"  input tensor: {tuple(x.shape)}")
        with torch.inference_mode():
            feats = model(x)  # <-- F11 로 들어가면 vision_transformer.forward (RoPE)
        print(f"  -> features: {tuple(feats.shape)}\n")


if __name__ == "__main__":
    main()
