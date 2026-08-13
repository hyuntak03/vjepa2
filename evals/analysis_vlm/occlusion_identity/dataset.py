"""Blender occlusion dataset loader for the identity probe-transfer eval.

Layout (produced by data_gen/, mirrored at /local_datasets/world/blender_occlusion_dataset):
    <root>/index.csv    # one row per (scene, variant); `file` -> videos/scene_XXXXXX_<variant>.mp4
    <root>/frames.csv   # one row per (video, frame); carries per-frame `state`
    <root>/videos/*.mp4 # 40 frames @ 8fps, 256x256

Every variant of a scene shares the same context (frames 0..31): the variants diverge
only at `divergence_frame` (33..36), i.e. after the ball re-emerges from the occluder.
So `shape`/`color` describe the object BEFORE occlusion and `shape_after`/`color_after`
describe what is actually rendered in the target window (frames 32..39).

`imp_vanish` has `state == annihilated` for most of frames 32..39 (the ball never comes
back), so its post-occlusion identity label is undefined -- the caller filters it out of
the probe train/eval variant lists.
"""
from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:
    import decord
except ImportError:  # pragma: no cover
    decord = None

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

# per-frame `state` values in frames.csv that mean "the ball is not visible at all"
HIDDEN_STATES = ("occluded", "annihilated")


def _natkey(v):
    """정렬 키. scene_id / row_id 가 숫자 문자열이면 숫자로, 아니면 문자열로."""
    try:
        return (0, int(v))
    except (TypeError, ValueError):
        return (1, str(v))


def _decode(path: str, n_frames: int) -> torch.Tensor:
    """-> (C, T, H, W) float in [0, 1]."""
    vr = decord.VideoReader(path, num_threads=1)
    assert len(vr) >= n_frames, f"{path}: {len(vr)} frames < requested {n_frames}"
    arr = vr.get_batch(list(range(n_frames))).asnumpy()          # (T, H, W, C) uint8
    return torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0


@dataclass
class VideoRecord:
    video_id: str
    scene_id: str                    # split 을 묶는 그룹 키 (occlusion: scene_id, probe set: row_id)
    variant: str                     # 리포트 기본 그룹 (occlusion: variant, probe set: 설정된 컬럼)
    file: str
    split: str                       # 'train' | 'val'
    labels_after: Dict[str, int]     # target name -> class index (rendered in frames 32..39)
    labels_before: Dict[str, int]    # target name -> class index (pre-occlusion identity)
    hidden_frac: Dict[str, float] = field(default_factory=dict)  # "24-32" -> frac hidden
    strata: Dict[str, str] = field(default_factory=dict)   # 리포트용 층화 컬럼 값들
    meta: Dict[str, str] = field(default_factory=dict)


class OcclusionVideos:
    """Indexable list of VideoRecords; `clip(i)` decodes + normalizes video i.

    Parameters
    ----------
    targets : list of {name, column, before_column, classes}
    variants : which `variant` values to keep (order-insensitive)
    split_cfg : {mode: column|ratio, ...}  -- always grouped by `group_by` (scene_id)
    occlusion_windows : list of [f0, f1) frame ranges to measure hidden-fraction for
    limit_scenes : keep only the first N scene_ids (smoke runs)
    """

    def __init__(
        self,
        root: str,
        index_csv: str = "index.csv",
        *,
        targets: Sequence[dict],
        variants: Sequence[str],
        n_frames: int = 40,
        split_cfg: Optional[dict] = None,
        occlusion_windows: Sequence[Sequence[int]] = ((24, 32),),
        frames_csv: str = "frames.csv",
        limit_scenes: Optional[int] = None,
        group_column: str = "variant",     # 리포트 기본 그룹을 만들 index.csv 컬럼
        strata: Sequence[str] = (),        # 추가로 층화해서 리포트할 컬럼들
    ):
        assert decord is not None, "decord is required (pip install decord)"
        self.root = root
        self.n_frames = int(n_frames)
        self.targets = list(targets)
        self.variants = tuple(variants)
        self.group_column = group_column
        self.strata_columns = tuple(strata)
        self.occlusion_windows = [tuple(w) for w in occlusion_windows]

        self.class_index = {
            t["name"]: {c: i for i, c in enumerate(t["classes"])} for t in self.targets
        }

        rows = list(csv.DictReader(open(os.path.join(root, index_csv))))
        if rows and group_column not in rows[0]:
            raise KeyError(f"group_column {group_column!r} not in {index_csv}; "
                           f"available: {sorted(rows[0])}")
        if self.variants:
            rows = [r for r in rows if r[group_column] in self.variants]
        else:                                    # 빈 리스트 = 전부 사용
            self.variants = tuple(sorted({r[group_column] for r in rows}))
        gcol = (split_cfg or {}).get("group_by", "scene_id")
        if rows and gcol not in rows[0]:
            raise KeyError(f"split.group_by {gcol!r} not in {index_csv}; "
                           f"available: {sorted(rows[0])}")
        self.group_by = gcol
        if limit_scenes is not None:
            keep = set(sorted({r[gcol] for r in rows}, key=_natkey)[:limit_scenes])
            rows = [r for r in rows if r[gcol] in keep]

        hidden = self._hidden_fractions(
            os.path.join(root, frames_csv), {r["video_id"] for r in rows}
        )
        split_of = self._assign_splits(rows, split_cfg or {"mode": "column"})

        self.records: List[VideoRecord] = []
        for r in rows:
            la, lb = {}, {}
            for t in self.targets:
                name, idx = t["name"], self.class_index[t["name"]]
                # before_column 이 없는 데이터셋(변이가 없는 probe set)은 column 을 그대로 쓴다
                for col, dst in ((t["column"], la), (t.get("before_column", t["column"]), lb)):
                    if col not in r:
                        raise KeyError(f"{index_csv}: 컬럼 {col!r} 없음 (target {t['name']!r})")
                    v = r[col]
                    if v not in idx:
                        raise KeyError(
                            f"{r['video_id']}: {col}={v!r} not in configured classes for "
                            f"target {name!r} ({sorted(idx)})"
                        )
                    dst[name] = idx[v]
            self.records.append(
                VideoRecord(
                    video_id=r["video_id"],
                    scene_id=r[gcol],
                    variant=r[group_column],
                    file=r["file"],
                    split=split_of[r["video_id"]],
                    labels_after=la,
                    labels_before=lb,
                    hidden_frac=hidden.get(r["video_id"], {}),
                    strata={c: r[c] for c in self.strata_columns if c in r},
                    meta={
                        k: r[k]
                        for k in (
                            "entry_frame", "reappear_frame", "divergence_frame",
                            "n_hidden_frames", "plausible", "violation",
                            "obj_px", "is_static", "speed_px_f", "obj_tokens_16px",
                        )
                        if k in r
                    },
                )
            )
        self.records.sort(key=lambda x: (_natkey(x.scene_id), x.variant))

    # -- helpers ---------------------------------------------------------------

    def _hidden_fractions(self, frames_csv: str, want: set) -> Dict[str, Dict[str, float]]:
        """video_id -> {"24-32": frac of frames in [24,32) where the ball is invisible}."""
        if not os.path.exists(frames_csv):
            return {}
        states: Dict[str, Dict[int, str]] = {}
        with open(frames_csv) as f:
            rd = csv.DictReader(f)
            if "state" not in (rd.fieldnames or []):
                return {}                        # 가림이 없는 데이터셋 -> 은닉률 개념 없음
            for row in rd:
                vid = row["video_id"]
                if vid in want:
                    states.setdefault(vid, {})[int(row["frame"])] = row["state"]
        out: Dict[str, Dict[str, float]] = {}
        for vid, per_frame in states.items():
            d = {}
            for f0, f1 in self.occlusion_windows:
                st = [per_frame.get(f) for f in range(f0, f1)]
                st = [s for s in st if s is not None]
                d[f"{f0}-{f1}"] = (
                    sum(s in HIDDEN_STATES for s in st) / len(st) if st else float("nan")
                )
            out[vid] = d
        return out

    @staticmethod
    def _assign_splits(rows: List[dict], cfg: dict) -> Dict[str, str]:
        """-> video_id -> 'train'|'val'. Always grouped so a scene never straddles."""
        mode = cfg.get("mode", "column")
        group_by = cfg.get("group_by", "scene_id")
        if mode == "column":
            col = cfg.get("column", "split")
            tr, va = cfg.get("train_value", "train"), cfg.get("val_value", "test")
            # sanity: the split column must be constant within a group
            per_group = {}
            for r in rows:
                g = r[group_by]
                per_group.setdefault(g, set()).add(r[col])
            bad = {g: v for g, v in per_group.items() if len(v) > 1}
            if bad:
                raise ValueError(f"split column {col!r} straddles groups: {list(bad)[:3]}")
            out = {}
            for r in rows:
                v = r[col]
                if v not in (tr, va):
                    raise ValueError(f"unexpected {col}={v!r}; expected {tr!r}/{va!r}")
                out[r["video_id"]] = "train" if v == tr else "val"
            return out
        if mode == "ratio":
            frac = float(cfg.get("train_frac", 0.5))
            seed = int(cfg.get("seed", 0))
            strat = cfg.get("stratify_by")
            # stratify_by 를 주면 그 컬럼 값별로 각각 train_frac 을 나눈 뒤 합친다.
            # 안 주면 전체 그룹을 한 번에 섞는데, 작은 층은 비율이 크게 흔들린다:
            # probe set 의 정지 32 row 가 16/16 이 아니라 14/18 로 갈렸고, seed 를 바꾸면
            # 또 달라진다. 층별로 학습/평가하려면 반드시 층화해야 한다.
            group_stratum = {}
            for r in rows:
                g = r[group_by]
                k = r[strat] if strat else "_all"
                if group_stratum.setdefault(g, k) != k:
                    raise ValueError(
                        f"stratify_by={strat!r} is not constant within {group_by}={g!r}; "
                        f"그룹 하나가 여러 층에 걸쳐 있어 층화 분할이 불가능하다")
            train_groups = set()
            for k in sorted({v for v in group_stratum.values()}):
                gs = sorted([g for g, v in group_stratum.items() if v == k])
                random.Random(seed).shuffle(gs)          # 층마다 같은 seed, 독립 셔플
                train_groups |= set(gs[: int(round(len(gs) * frac))])
            return {
                r["video_id"]: ("train" if r[group_by] in train_groups else "val")
                for r in rows
            }
        raise ValueError(f"unknown split mode {mode!r}; valid: column | ratio")

    # -- access ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def clip(self, i: int) -> torch.Tensor:
        """-> (C, T, H, W), ImageNet-normalized."""
        path = os.path.join(self.root, self.records[i].file)
        return (_decode(path, self.n_frames) - IMAGENET_MEAN) / IMAGENET_STD

    def loader(self, batch_size: int, indices: Optional[Sequence[int]] = None,
               num_workers: int = 4):
        """DataLoader yielding (idx (B,), clips (B, C, T, H, W)).

        Decoding runs in worker processes so the GPU never waits on decord: with 3
        ViT-L forwards per clip the encoder is the only thing that should be busy.
        """
        from torch.utils.data import DataLoader, Dataset

        parent = self
        order = list(range(len(parent.records))) if indices is None else list(indices)

        class _Clips(Dataset):
            def __len__(self):
                return len(order)

            def __getitem__(self, j):
                i = order[j]
                return i, parent.clip(i)

        return DataLoader(
            _Clips(), batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, persistent_workers=False,
            prefetch_factor=(2 if num_workers > 0 else None),
        )

    def summary(self) -> str:
        from collections import Counter

        cv = Counter((r.variant, r.split) for r in self.records)
        out = (f"{len(self.records)} videos | {len({r.scene_id for r in self.records})} "
               f"groups({self.group_by}) | "
               + " ".join(f"{v}/{s}={n}" for (v, s), n in sorted(cv.items())))
        for c in self.strata_columns:
            vc = Counter(r.strata.get(c, "?") for r in self.records)
            out += f"\n           {c}: " + " ".join(f"{k}={n}" for k, n in sorted(vc.items()))
        return out
