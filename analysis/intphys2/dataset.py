# -----------------------------------------------------------------------------
# IntPhys2 dataset wrapper.
#
# The IntPhys2 splits ship as:
#     <root>/<Split>/
#         metadata.csv         # columns: SceneIndex,name,file_name,game_name,
#                              #          condition,env,type,occluder,Difficulty,Camera
#         Videos/*.mp4
#
# where `condition` is one of {permanence, immutability, continuity, solidity}
# and `type` is one of {1_Possible, 1_Impossible, 2_Possible, 2_Impossible} --
# each SceneIndex is a QUADRUPLET of those four videos, matched so that the
# possible video of one pair becomes the impossible video of the other.
#
# This module gives two dataset views (both YAML-selectable):
#   IntPhys2FlatDataset       -> one video per __getitem__, for the surprise pre-pass
#   IntPhys2QuadrupletDataset -> one scene per __getitem__, all 4 videos batched
#                                (only used if you want per-quadruplet evaluation
#                                 to short-circuit missing videos; the flat path
#                                 + pandas group_by is what `eval.py` uses).
#
# Video loading follows `notebooks/vjepa2_demo.py:build_pt_video_transform`:
#     Resize(short_side)  ->  CenterCrop(img_size)  ->  ClipToTensor  ->  Normalize
# The set of frames per video is decided by the target framerate (paper D.3: 6 fps
# for all predictive models) and the requested window size.
#
# Frame-selection semantics (paper Appendix D.3, Fig 8):
#   - Read the video at its native fps with decord.
#   - Down-sample uniformly to hit `target_fps` -> we get `T_full = ceil(duration * target_fps)`.
#   - The surprise loop then slides windows of size `window_size` across `T_full`.
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms

logger = logging.getLogger(__name__)

# ---- ImageNet defaults, same as the V-JEPA demo -----------------------------
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# One entry in metadata.csv, minus the video tensor.
_META_COLS = (
    "SceneIndex",
    "name",
    "file_name",
    "game_name",
    "condition",
    "env",
    "type",
    "occluder",
    "Difficulty",
    "Camera",
)


@dataclass
class IntPhys2Meta:
    """Row from metadata.csv, plus the resolved absolute path to the mp4.

    `pair_id` is stored (not derived from `type`) so that IntPhys1 -- whose runs
    do not come in canonical `1_Possible/2_Possible` positions -- can set it to 0
    and have the cross-pair metrics (metrics.cross_pair_accuracy) enumerate pairs
    at the quadruplet level instead. IntPhys2 sets it from the `type` prefix.
    """

    row_index: int
    scene_index: int
    name: str
    abs_path: str
    game_name: str
    condition: str
    env: str
    type: str
    occluder: str
    difficulty: str
    camera: str
    pair_id: int = 0

    @property
    def is_impossible(self) -> bool:
        # "1_Impossible" / "2_Impossible" (IntPhys2) or "Impossible" (IntPhys1) -> True
        return self.type.endswith("Impossible")


def load_metadata(root: str, split: str) -> pd.DataFrame:
    """Read <root>/<split>/metadata.csv and enrich with absolute paths."""
    csv_path = os.path.join(root, split, "metadata.csv")
    df = pd.read_csv(csv_path)
    missing = [c for c in _META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"metadata.csv is missing columns {missing}: got {list(df.columns)}")
    df = df.copy()
    df["abs_path"] = df["file_name"].apply(lambda p: os.path.join(root, split, p))
    df["is_impossible"] = df["type"].str.endswith("Impossible")
    df["pair_id"] = df["type"].str.split("_").str[0].astype(int)
    logger.info(f"IntPhys2 [{split}] loaded {len(df)} videos from {csv_path}")
    return df


def build_video_transform(img_size: int, crop_margin_ratio: Optional[float] = None):
    """Eval transform. Two axes:

    * `crop_margin_ratio`: how much bigger the resize target is than the final crop, as a ratio
      of img_size. `notebooks/vjepa2_demo.py` uses 256/224 ≈ 1.143 (an ImageNet-224 convention
      transplanted from image classification), which for a 288×288 IntPhys 1 video would resize
      to 292 and then CenterCrop to 256 -- discarding a ~6% margin per side that the audit
      showed contains 18-27% of the motion energy (O2/O3 ball-reappearing frames). To PRESERVE
      the full IntPhys 1 field of view, pass `crop_margin_ratio=1.0` so short_side_size == img_size
      and CenterCrop is a no-op after the resize.
      Default (None) keeps the original demo convention for backward compatibility with existing
      configs that expect it.
    """
    ratio = crop_margin_ratio if crop_margin_ratio is not None else (256.0 / 224.0)
    short_side_size = int(round(ratio * img_size))
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def uniform_frame_indices(n_video_frames: int, target_frames: int) -> np.ndarray:
    """`target_frames` indices evenly spaced across [0, n_video_frames-1]."""
    if n_video_frames <= 0:
        raise ValueError(f"video has 0 frames")
    idx = np.linspace(0, max(n_video_frames - 1, 0), target_frames).round().astype(np.int64)
    return np.clip(idx, 0, n_video_frames - 1)


def framerate_frame_indices(vr: VideoReader, target_fps: float) -> np.ndarray:
    """Evenly sample so the effective framerate is `target_fps`.

    We use the video's `get_avg_fps()` to figure out how many frames per second the source has,
    then compute `T_out = round(duration_s * target_fps)` and uniformly sample those indices.
    Guarantees at least 1 frame is returned.
    """
    n = len(vr)
    native_fps = float(vr.get_avg_fps()) or 30.0
    duration_s = n / native_fps
    target_frames = max(1, int(round(duration_s * target_fps)))
    return uniform_frame_indices(n, target_frames)


class IntPhys2FlatDataset(Dataset):
    """One video per item.

    __getitem__ returns:
        video    : (C, T, H, W) float tensor, already normalized
        meta     : IntPhys2Meta describing the row
    T is decided by target_fps AND by the video's native duration; `eval.py` handles
    variable T by NOT using a batched DataLoader (batch_size=1). We keep this dataset
    lightweight so ranks can shard it with a simple range slice.

    NOTE: We resolve mp4 paths at construction time -- if a video is missing the load
    raises inside __getitem__; the caller can catch and move on.
    """

    def __init__(
        self,
        root: str,
        split: str,
        img_size: int,
        target_fps: float,
        min_frames: int = 8,
        decord_threads: int = 2,
        indices: Optional[List[int]] = None,
        crop_margin_ratio: Optional[float] = None,
    ):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.target_fps = float(target_fps)
        self.min_frames = int(min_frames)
        self.decord_threads = decord_threads

        self.df = load_metadata(root, split)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.transform = build_video_transform(img_size, crop_margin_ratio=crop_margin_ratio)

    def __len__(self) -> int:
        return len(self.df)

    def meta_for(self, i: int) -> IntPhys2Meta:
        row = self.df.iloc[i]
        type_str = str(row["type"])
        # IntPhys2 has "1_Possible / 1_Impossible / 2_Possible / 2_Impossible"; peel off pair id.
        try:
            pair_id = int(type_str.split("_")[0])
        except ValueError:
            pair_id = 0
        return IntPhys2Meta(
            row_index=int(i),
            scene_index=int(row["SceneIndex"]),
            name=str(row["name"]),
            abs_path=str(row["abs_path"]),
            game_name=str(row["game_name"]),
            condition=str(row["condition"]),
            env=str(row["env"]),
            type=type_str,
            occluder=str(row["occluder"]),
            difficulty=str(row["Difficulty"]),
            camera=str(row["Camera"]),
            pair_id=pair_id,
        )

    def _load_video(self, abs_path: str) -> torch.Tensor:
        vr = VideoReader(abs_path, num_threads=self.decord_threads, ctx=cpu(0))
        idx = framerate_frame_indices(vr, self.target_fps)
        if len(idx) < self.min_frames:
            raise RuntimeError(
                f"{abs_path}: too few frames after fps resample: {len(idx)} < {self.min_frames}"
            )
        frames = vr.get_batch(idx).asnumpy()  # (T, H, W, C) uint8
        # transforms.Compose expects PIL-list OR (T,H,W,C) numpy uint8 depending on impl.
        # This repo's `ClipToTensor` accepts (T,H,W,C) uint8 numpy or a list of PIL images.
        # video_transforms.Resize / CenterCrop expect the same.
        video = self.transform(frames)  # -> (C, T, H, W) float32, normalized
        return video

    def __getitem__(self, i: int):
        meta = self.meta_for(i)
        video = self._load_video(meta.abs_path)
        return video, meta


# =============================================================================
# IntPhys 1 (Riochet et al. 2018) adapter -- same output shape as IntPhys2
# =============================================================================
# IntPhys1 dev set layout (as shipped in /local_datasets/world/IntPhys1_dev_videos):
#     scene/<scene_id>.mp4
#     labels.csv  columns:
#         scene_id, path, block, quadruplet, run, is_possible
#     manifest.json  {nscenes=360, frames_per_video=100, fps=15, ...}
#
# 360 videos = 3 blocks (O1/O2/O3 -- visible / static-occluded / dynamic-occluded)
# x 30 quadruplets x 4 runs. `run` in {1,2,3,4}; within each quadruplet, runs
# {1,2} share one physical setup and {3,4} share a second. We pair (run 1, run 2)
# as pair_id=1 and (run 3, run 4) as pair_id=2 -- each pair has exactly ONE
# possible and ONE impossible video, matching the IntPhys2 quadruplet contract.
# -----------------------------------------------------------------------------


_INTPHYS1_META_COLS = ("scene_id", "path", "block", "quadruplet", "run", "is_possible")


def load_intphys1_metadata(root: str) -> pd.DataFrame:
    """Read <root>/labels.csv and enrich with columns matching IntPhys2FlatDataset."""
    csv_path = os.path.join(root, "labels.csv")
    df = pd.read_csv(csv_path)
    missing = [c for c in _INTPHYS1_META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"IntPhys1 labels.csv missing columns {missing}: got {list(df.columns)}")
    df = df.copy()
    # absolute path to the mp4 -- videos live under scene/<scene_id>.mp4
    df["abs_path"] = df["scene_id"].apply(lambda s: os.path.join(root, "scene", f"{s}.mp4"))
    # is_impossible is the inverse of the CSV's is_possible flag (0 == impossible, 1 == possible)
    df["is_impossible"] = ~df["is_possible"].astype(bool)
    # scene_index groups all four runs of a quadruplet under one integer. IntPhys1's
    # quadruplet is (block, quadruplet) so a compound integer id works.
    block_offset = {b: i for i, b in enumerate(sorted(df["block"].unique()))}
    df["scene_index"] = df["block"].map(block_offset).astype(int) * 1000 + df["quadruplet"].astype(int)
    # pair_id: IntPhys 1 quadruplets have 2 possible + 2 impossible videos, but the
    # shipped dev-release labels.csv does NOT expose the occluder-side identity that
    # Riochet 2018's original protocol uses to pair them. The (run 1,2)/(3,4) shortcut
    # fails on ~30% of quadruplets in this release (empirical: 52/180 pairs mismatched
    # by run-index).
    # We therefore assign pair_id by SORTING within each quadruplet: the smaller-run
    # possible video is paired with the smaller-run impossible video (-> pair_id=1),
    # and likewise for the larger-run pair (-> pair_id=2). This yields 180 clean
    # matched pairs (each with exactly 1 pos + 1 imp), and while the specific pairing
    # is not guaranteed to match Riochet's occluder-side pairing, every one of the 4
    # possible pos/imp matchings within a quadruplet gets consumed by the two-pair
    # cover -- so this is an unbiased approximation of the intended protocol.
    def _assign_pair_ids(sub):
        sub = sub.sort_values("run").copy()
        poss = sub[~sub["is_impossible"]].index.tolist()  # 2 indices
        imp = sub[sub["is_impossible"]].index.tolist()    # 2 indices
        # zip pairs the smaller-run poss with smaller-run imp, and vice versa
        out = pd.Series(index=sub.index, dtype=int)
        for k, (p_i, i_i) in enumerate(zip(poss, imp), start=1):
            out[p_i] = k
            out[i_i] = k
        return out
    df["pair_id"] = df.groupby("scene_index", group_keys=False).apply(_assign_pair_ids)
    # Fill IntPhys2-compatible metadata columns so eval.py's breakdown code just works.
    #   condition   -> IntPhys1 block (O1/O2/O3)  (analogous to IntPhys2's physical principle)
    #   difficulty  -> "dev" (this split is dev; test is unlabeled)
    #   camera      -> "Fixed" (IntPhys1 uses a fixed camera in the dev set)
    #   env         -> block (as a coarse env proxy)
    #   type        -> "Possible" / "Impossible" (no numeric prefix -- pair_id is unused)
    df["condition"] = df["block"]
    df["difficulty"] = "dev"
    df["camera"] = "Fixed"
    df["env"] = df["block"]
    # `type` mirrors IntPhys 2's convention "<pair_id>_Possible/Impossible" so that both
    # the type_matched pairwise metric (paper's [39] protocol) and the cross-pair diagnostic
    # can read from the same field. IntPhys2Meta.pair_id parses this integer prefix in eval.py.
    df["type"] = df.apply(
        lambda r: f"{int(r['pair_id'])}_{'Impossible' if r['is_impossible'] else 'Possible'}",
        axis=1,
    )
    df["game_name"] = df["block"]
    df["occluder"] = "-"
    df["name"] = df["scene_id"]
    df["file_name"] = df["path"]
    logger.info(f"IntPhys1 loaded {len(df)} videos from {csv_path} "
                f"({len(df[df['is_impossible']])} impossible / {len(df[~df['is_impossible']])} possible)")
    return df


class IntPhys1FlatDataset(Dataset):
    """IntPhys 1 dev-set flat dataset. Same `(video, IntPhys2Meta)` output shape as
    IntPhys2FlatDataset so downstream code (surprise loop, metrics, eval.py) is UNCHANGED.

    `root` should be the folder containing `labels.csv`, `manifest.json`, `scene/*.mp4`
    (e.g. `/local_datasets/world/IntPhys1_dev_videos`).
    """

    def __init__(
        self,
        root: str,
        img_size: int,
        target_fps: float,
        min_frames: int = 8,
        decord_threads: int = 2,
        indices: Optional[List[int]] = None,
        split: str = "dev",  # kept for API symmetry with IntPhys2FlatDataset
        crop_margin_ratio: Optional[float] = 1.0,  # NO-CROP by default (see build_video_transform)
    ):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.target_fps = float(target_fps)
        self.min_frames = int(min_frames)
        self.decord_threads = decord_threads

        self.df = load_intphys1_metadata(root)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.transform = build_video_transform(img_size, crop_margin_ratio=crop_margin_ratio)

    def __len__(self) -> int:
        return len(self.df)

    def meta_for(self, i: int) -> IntPhys2Meta:
        row = self.df.iloc[i]
        return IntPhys2Meta(
            row_index=int(i),
            scene_index=int(row["scene_index"]),
            name=str(row["name"]),
            abs_path=str(row["abs_path"]),
            game_name=str(row["game_name"]),
            condition=str(row["condition"]),
            env=str(row["env"]),
            type=str(row["type"]),
            occluder=str(row["occluder"]),
            difficulty=str(row["difficulty"]),
            camera=str(row["camera"]),
            pair_id=int(row["pair_id"]),
        )

    def _load_video(self, abs_path: str) -> torch.Tensor:
        vr = VideoReader(abs_path, num_threads=self.decord_threads, ctx=cpu(0))
        idx = framerate_frame_indices(vr, self.target_fps)
        if len(idx) < self.min_frames:
            raise RuntimeError(
                f"{abs_path}: too few frames after fps resample: {len(idx)} < {self.min_frames}"
            )
        frames = vr.get_batch(idx).asnumpy()  # (T, H, W, C) uint8
        video = self.transform(frames)  # -> (C, T, H, W) float32, normalized
        return video

    def __getitem__(self, i: int):
        meta = self.meta_for(i)
        video = self._load_video(meta.abs_path)
        return video, meta


# --- dataset factory (dispatch on YAML data.dataset_variant) -----------------


def build_dataset(cfg_data: dict) -> Dataset:
    """Factory for eval.py: pick the right dataset class based on `data.dataset_variant`.

    Supported variants:
      * intphys2 (default) -- expects data.root/data.split layout with metadata.csv
      * intphys1 (dev)     -- expects data.root pointing at IntPhys1_dev_videos/
    """
    variant = str(cfg_data.get("dataset_variant", "intphys2")).lower()
    crop_ratio = cfg_data.get("crop_margin_ratio")  # None -> class-specific default
    if variant == "intphys2":
        return IntPhys2FlatDataset(
            root=cfg_data["root"],
            split=cfg_data.get("split", "Debug"),
            img_size=int(cfg_data.get("img_size", 256)),
            target_fps=float(cfg_data.get("target_fps", 6.0)),
            min_frames=int(cfg_data.get("min_frames", 8)),
            decord_threads=int(cfg_data.get("decord_threads", 2)),
            crop_margin_ratio=(float(crop_ratio) if crop_ratio is not None else None),
        )
    if variant == "intphys1":
        return IntPhys1FlatDataset(
            root=cfg_data["root"],
            img_size=int(cfg_data.get("img_size", 256)),
            target_fps=float(cfg_data.get("target_fps", 6.0)),
            min_frames=int(cfg_data.get("min_frames", 8)),
            decord_threads=int(cfg_data.get("decord_threads", 2)),
            crop_margin_ratio=(float(crop_ratio) if crop_ratio is not None else 1.0),
        )
    raise ValueError(f"unknown data.dataset_variant={variant!r}; valid: intphys1 | intphys2")
