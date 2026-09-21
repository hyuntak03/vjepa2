# -----------------------------------------------------------------------------
# IntPhys2 pairwise-surprise evaluation — main entry.
#
# Usage:
#     python -m analysis.intphys2.eval --config analysis/intphys2/configs/vjepa2_vitl_debug.yaml
#     python -m analysis.intphys2.eval --config <cfg> --device cuda:0
#     torchrun --nproc-per-node=<N> -m analysis.intphys2.eval --config <cfg>
#         (DDP shards videos over ranks; rank 0 writes aggregated results)
#
# Outputs (written under <folder>/<tag>/):
#     per_video.csv          -- one row per (video, context_length): surprise_{avg,max}, n_windows
#     per_window.parquet     -- (optional) per-window traces for later analysis
#     summary.json           -- machine-readable metrics: pairwise + AUROC + breakdowns
#     summary.txt            -- human-readable summary
#     config.resolved.yaml   -- the YAML actually used (with defaults filled in)
#     plots/*.png            -- (optional) breakdown bars + context sweep + per-scene traces
#
# Design notes:
#     * All settings live in the YAML config; no hard-coded model / data paths.
#     * The surprise loop is per-video (batch_size = 1 at the outer level), matching
#       the fact that IntPhys2 videos have varying frame counts. Inside a video we
#       DO already batch across sliding windows through the encoder if
#       `evaluation.window_batch > 1` -- but the default is 1, which is simplest.
#     * DDP is opt-in: `evaluation.ddp: true` shards the flat dataset over ranks,
#       reduces per-video results, and rank 0 writes outputs. Without torchrun we
#       silently run single-process.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

# repo-relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.intphys2.dataset import IntPhys2FlatDataset, IntPhys2Meta, build_dataset  # noqa: E402
from torch.utils.data import DataLoader, Sampler  # noqa: E402
from analysis.intphys2.metrics import (  # noqa: E402
    VideoRecord,
    cross_pair_accuracy,
    cross_pair_accuracy_breakdown,
    format_summary,
    pairwise_accuracy,
    pairwise_accuracy_breakdown,
    single_video_auroc,
    sweep_max_pairwise,
    to_dataframe,
)
from analysis.intphys2.model import VJEPA2Bundle, build_from_config  # noqa: E402
from analysis.intphys2.plotting import (  # noqa: E402
    plot_breakdown_bars,
    plot_context_sweep,
    plot_scene_surprise,
)
from analysis.intphys2.surprise import (  # noqa: E402
    VideoSurprise,
    score_video,
    score_video_batched,
    score_videos_batched,
)


# --------------------------- logging setup -----------------------------------


def _setup_logging(rank: int, verbose: bool = True):
    fmt = f"[rank{rank}] %(asctime)s %(levelname)s %(name)s :: %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO if verbose else logging.WARNING,
        format=fmt,
        force=True,
    )
    # Silence libraries we don't want to hear from.
    logging.getLogger("PIL").setLevel(logging.WARNING)


logger = logging.getLogger("analysis.intphys2.eval")

# model.autocast: 공식은 fp32 가중치 + fp16 autocast 다 (Garrido eval.py:437).
_AC_DTYPE = {"float16": torch.float16, "fp16": torch.float16,
             "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "none": None, "": None}


# --------------------------- DDP helpers -------------------------------------


def _init_ddp_if_requested(ddp: bool):
    """Return (rank, world_size, use_ddp). No-op if `torchrun` env is absent."""
    if not ddp:
        return 0, 1, False
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger.warning("evaluation.ddp: true but RANK/WORLD_SIZE not set; running single-process")
        return 0, 1, False
    import torch.distributed as dist
    if not dist.is_initialized():
        # NCCL 기본 collective timeout 은 600 s 다. 랭크마다 영상 난이도가 달라 끝나는 시각이
        # 벌어지면 마지막 all_gather 에서 그 벽에 걸린다 (2026-09-21: IntPhys2 x 2.1-g 이
        # 85 분 다 돌고 gather 에서 죽었다). run.sh 하네스의 EVAL_DDP_TIMEOUT_S 와 같은 취지.
        _to = int(os.environ.get("EVAL_DDP_TIMEOUT_S", "7200"))
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=_to))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size, True


def _all_gather_records(records: List[VideoRecord], world_size: int) -> List[VideoRecord]:
    """all_gather Python objects across ranks; return the concatenated list."""
    import torch.distributed as dist
    gathered: List[List[VideoRecord]] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(gathered, records)
    out: List[VideoRecord] = []
    for chunk in gathered:
        out.extend(chunk)
    return out


# --------------------------- config resolution -------------------------------


DEFAULT_CFG: Dict[str, Any] = {
    "tag": "intphys2_run",
    "folder": None,  # required
    "data": {
        "root": None,  # required
        "dataset_variant": "intphys2",  # intphys1 | intphys2
        "split": "Debug",               # ignored for intphys1 (only dev is public with labels)
        "target_fps": 6.0,
        "min_frames": 8,
        "img_size": 256,
        "decord_threads": 2,
    },
    "model": {
        # `checkpoint` is required; other model fields have defaults defined in model.py
    },
    "surprise": {
        "window_size": 48,
        "context_length": 12,
        "context_length_sweep": None,   # list -> sweep and pick MAX pairwise (paper D.3)
        "stride": 4,
        "distance": "l1",
        "loss_exp": 1.0,
        "target_layer_norm": True,
        "aggregation": "avg",           # avg | max -- per-video aggregation over windows
        "protocol": "fixed",            # fixed (Fig 8A) | growing (Fig 8B, TODO)
        "max_window_batch": None,       # None => encode all windows of a video in one shot
        # CRITICAL default (audit finding #1): only mask_tokens[0] is trained in the
        # released V-JEPA 2 checkpoint; indices 1..9 remain zero-init, so passing any
        # other value substitutes a zero vector for the target-position input.
        "mask_index": 0,
        # HIGH default (audit finding #4): call context_encoder with masks_enc so its self-
        # attention is restricted to context tokens, matching train.py:435-438. "sliced" is
        # the (faster, non-training-faithful) old behavior kept as an escape hatch.
        "context_forward_mode": "masked",   # masked | sliced
    },
    "evaluation": {
        "compute_pairwise": True,
        "compute_auroc": True,
        "breakdown_by": ["condition", "difficulty", "camera"],
        "ddp": False,
        "num_workers": 4,               # DataLoader workers per rank (CPU decord decode)
        "prefetch_factor": 2,           # DataLoader prefetch per worker
        "pin_memory": True,             # host memory pinning for CPU->GPU transfer
        "verbose": True,
        "log_every_videos": 25,
        "limit_videos": None,           # for smoke tests
    },
    "output": {
        "save_per_video": True,
        "save_per_window": False,
        "save_config": True,
        "plot_breakdowns": True,
        "plot_context_sweep": True,
        "plot_scene_curves": False,     # optional; N per-scene figures
        "scene_curve_indices": None,    # list of scene_index to plot; None => none
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge (override wins). Lists are replaced, not concatenated."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_config(cfg_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = _load_yaml(cfg_path)
    cfg = _deep_merge(DEFAULT_CFG, raw)
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    # sanity
    if cfg["folder"] is None:
        raise ValueError("config: `folder` is required (output root)")
    if cfg["data"]["root"] is None:
        raise ValueError("config: `data.root` is required (IntPhys2 dataset root)")
    if "checkpoint" not in cfg.get("model", {}):
        raise ValueError("config: `model.checkpoint` is required")

    # Sync surprise.window_size -> model.window_size so the ViT is built with the right num_frames.
    cfg["model"] = dict(cfg["model"])  # shallow copy so we don't mutate the loaded dict
    cfg["model"].setdefault("window_size", cfg["surprise"]["window_size"])
    if cfg["model"]["window_size"] != cfg["surprise"]["window_size"]:
        raise ValueError(
            f"model.window_size ({cfg['model']['window_size']}) must equal "
            f"surprise.window_size ({cfg['surprise']['window_size']})"
        )
    return cfg


def _output_dir(cfg: Dict[str, Any]) -> str:
    d = os.path.join(cfg["folder"], cfg["tag"])
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "plots"), exist_ok=True)
    return d


# --------------------------- video loop --------------------------------------


class _UnpaddedShardSampler(Sampler):
    """Contiguous strided shard (rank::world_size), no padding.

    The union across ranks is EXACTLY the dataset with each item seen once, so
    downstream reduce operations get exact totals (unlike torch.utils.data.
    DistributedSampler which pads with duplicates and would double-count some
    videos in a small dataset like Debug/60). Matches the pattern used elsewhere
    in this repo (evals/analysis_vlm/data.py:_UnpaddedShardSampler).
    """

    def __init__(self, n: int, world_size: int, rank: int, limit: Optional[int] = None):
        n_use = n if not limit else min(n, int(limit))
        self.indices = list(range(rank, n_use, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _single_video_collate(batch):
    """Batch size is 1 at the outer level; keep the (Tensor, IntPhys2Meta) tuple as-is."""
    assert len(batch) == 1
    return batch[0]


def _make_video_loader(
    dataset: IntPhys2FlatDataset,
    rank: int,
    world_size: int,
    e_cfg: Dict[str, Any],
) -> DataLoader:
    sampler = _UnpaddedShardSampler(
        n=len(dataset), world_size=world_size, rank=rank,
        limit=e_cfg.get("limit_videos"),
    )
    num_workers = int(e_cfg.get("num_workers", 4))
    prefetch = int(e_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_single_video_collate,
        pin_memory=bool(e_cfg.get("pin_memory", True)),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch,
    )


def _score_all_videos(
    dataset: IntPhys2FlatDataset,
    bundle: VJEPA2Bundle,
    cfg: Dict[str, Any],
    rank: int,
    world_size: int,
) -> List[VideoRecord]:
    """Per-rank surprise loop with DataLoader prefetching + batched-encoder fast path.

    Optimizations:
      * DataLoader with num_workers spawns CPU decoders that prefetch the next video(s)
        while the GPU works on the current one -> hides decord decode latency.
      * `score_video_batched` runs ONE encoder forward per video and iterates the
        predictor over context lengths -- avoiding the naive `for C in Cs: full pipeline`
        which was ~len(Cs)x wasteful of encoder time.
      * All per-window distances stay on device inside the batched call; a single
        `.cpu()` happens per video.
    """
    s_cfg = cfg["surprise"]
    e_cfg = cfg["evaluation"]
    o_cfg = cfg["output"]

    Cs: List[int] = (
        list(s_cfg["context_length_sweep"])
        if s_cfg.get("context_length_sweep")
        else [s_cfg["context_length"]]
    )
    Cs = [int(c) for c in Cs]

    loader = _make_video_loader(dataset, rank, world_size, e_cfg)
    n_local = len(loader.sampler)  # type: ignore[arg-type]
    logger.info(
        f"scoring {n_local}/{len(dataset)} videos (rank {rank}/{world_size}) "
        f"across {len(Cs)} context length(s): {Cs}  "
        f"(num_workers={e_cfg.get('num_workers', 4)})"
    )

    records: List[VideoRecord] = []
    per_window_traces: List[Dict[str, Any]] = []
    save_per_window = bool(o_cfg["save_per_window"])
    log_every = int(e_cfg.get("log_every_videos", 25))
    t_start = time.time()

    max_wb = s_cfg.get("max_window_batch")
    max_wb = int(max_wb) if max_wb is not None else None

    # video_batch: 한 번에 GPU 에 올리는 **영상 수** (Garrido 하네스 `video_batch` 이식, 2026-09-22).
    #   IntPhys 2 는 6fps 에서 영상이 64~72 프레임이라 영상당 창이 w16 ~27 / w32 ~19 / w48 ~11 개다.
    #   w48 은 max_window_batch 를 올려도 11 에서 막히므로 영상을 쌓아야 배치가 커진다.
    #   1 이면 예전과 배치 구성이 완전히 같다 (수치 동일).
    vbs = max(1, int(s_cfg.get("video_batch", 1)))
    nonlocal_fail = {"n": 0}          # 채점 실패 영상 수 (summary 에 남긴다)
    _buf: List[Tuple[torch.Tensor, Any]] = []

    def _flush(buf):
        """버퍼의 영상들을 한 배치로 채점해 [(meta, per_C), ...] 로 돌려준다."""
        if not buf:
            return []
        vids = [v for v, _ in buf]
        try:
            res = score_videos_batched(
                vids if len(vids) > 1 else vids[0], bundle,
                window_size=s_cfg["window_size"], context_lengths=Cs, stride=s_cfg["stride"],
                distance=s_cfg["distance"], loss_exp=float(s_cfg["loss_exp"]),
                target_layer_norm=bool(s_cfg["target_layer_norm"]),
                protocol=s_cfg.get("protocol", "fixed"), max_window_batch=max_wb,
                mask_index=int(s_cfg.get("mask_index", 0)),
                context_forward_mode=str(s_cfg.get("context_forward_mode", "masked")),
                autocast_dtype=_AC_DTYPE[str(cfg["model"].get("autocast", "none")).lower()],
            )
            res = [res] if len(vids) == 1 else res
        except torch.cuda.OutOfMemoryError as e:
            # ⚠️ OOM 을 NaN 으로 조용히 넘기면 **전 배치가 실패해도 summary.json 이 정상처럼 나온다**
            #    (2026-09-22 실제로 당했다: video_batch=8 이 전부 OOM 인데 행 수는 멀쩡해서
            #     "4 배 빨라졌다" 로 읽힐 뻔했다). 배치를 반으로 줄여 재시도하고, 1 에서도
            #     터지면 그때는 **죽는다** — 조용한 오염보다 낫다.
            torch.cuda.empty_cache()
            if len(vids) > 1:
                half = (len(vids) + 1) // 2
                logger.warning(f"batch of {len(vids)}: OOM -> {half} 로 줄여 재시도")
                return _flush(buf[:half]) + _flush(buf[half:])
            raise RuntimeError(
                f"video_batch=1 에서도 OOM 이다. surprise.max_window_batch 를 낮출 것 "
                f"(현재 {max_wb}). 원인: {e}") from e
        except Exception as e:  # noqa: BLE001
            nonlocal_fail["n"] += len(vids)
            logger.warning(f"batch of {len(vids)}: score failed ({e}); emitting NaN rows")
            empty = {C: VideoSurprise(window_starts=np.zeros(0, dtype=np.int64),
                                      window_context_lengths=np.zeros(0, dtype=np.int64),
                                      surprise=np.zeros(0, dtype=np.float32),
                                      context_length=C) for C in Cs}
            res = [empty] * len(vids)
        return [(m, r) for (_, m), r in zip(buf, res)]

    def _record(meta, per_C):
        """한 영상의 per-C 결과를 records / per_window_traces 에 넣는다."""
        for C in Cs:
            vs = per_C[C]
            avg = vs.aggregate("avg")
            max_ = vs.aggregate("max")
            records.append(VideoRecord(
                row_index=meta.row_index,
                scene_index=meta.scene_index,
                pair_id=meta.pair_id,
                is_impossible=meta.is_impossible,
                condition=meta.condition,
                difficulty=meta.difficulty,
                camera=meta.camera,
                env=meta.env,
                context_length=int(C),
                surprise_avg=float(avg),
                surprise_max=float(max_),
                n_windows=int(len(vs.surprise)),
            ))
            if save_per_window and len(vs.surprise) > 0:
                per_window_traces.append({
                    "row_index": meta.row_index,
                    "scene_index": meta.scene_index,
                    "type": meta.type,
                    "context_length": C,
                    "window_starts": vs.window_starts.tolist(),
                    "surprise": vs.surprise.tolist(),
                })


    for k, (video, meta) in enumerate(loader):
        _buf.append((video, meta))
        if len(_buf) >= vbs:
            for m, r in _flush(_buf):
                _record(m, r)
            _buf = []

        if (k + 1) % log_every == 0:
            elapsed = time.time() - t_start
            rate = (k + 1) / max(elapsed, 1e-6)
            logger.info(
                f"[{k+1}/{n_local}] rate={rate:.2f} v/s "
                f"elapsed={elapsed:.1f}s ETA={((n_local-k-1)/max(rate,1e-6)):.1f}s"
            )

    # 남은 버퍼를 비운다 (마지막 배치가 vbs 보다 작을 때)
    for m, r in _flush(_buf):
        _record(m, r)
    _buf = []

    if nonlocal_fail["n"]:
        frac = nonlocal_fail["n"] / max(n_local, 1)
        msg = f"채점 실패 {nonlocal_fail['n']}/{n_local} 영상 ({frac:.1%}) -- NaN 으로 남았다"
        if frac > 0.02:
            raise RuntimeError(msg + ". 2% 를 넘어 중단한다 (조용한 오염 방지).")
        logger.warning(msg)
    _score_all_videos._n_failed = nonlocal_fail["n"]  # type: ignore[attr-defined]

    # Stash per-window traces on the records list side-channel so caller can grab them.
    _score_all_videos._per_window = per_window_traces  # type: ignore[attr-defined]
    return records


# --------------------------- summary + write ---------------------------------


def _compute_summary(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    e_cfg = cfg["evaluation"]
    s_cfg = cfg["surprise"]
    surprise_col = "surprise_avg" if s_cfg["aggregation"] == "avg" else "surprise_max"
    break_by = list(e_cfg.get("breakdown_by") or [])

    # Pairing protocol: default to `type_matched` (paper protocol [39] for IntPhys 1 and
    # the natural IntPhys 2 layout), unless explicitly overridden in the config.
    # NOTE: audit finding #6 pointed out our earlier `cross_pair` default for IntPhys 1
    # was mixing matched + unmatched pairs and capping accuracy well below paper's number.
    # With `dataset.py` now filling pair_id from run index for IntPhys 1, both datasets
    # share the type_matched code path.
    variant = str(cfg["data"].get("dataset_variant", "intphys2")).lower()
    pairing = str(e_cfg.get("pairing", "type_matched")).lower()
    if pairing not in {"type_matched", "cross_pair"}:
        raise ValueError(f"evaluation.pairing must be 'type_matched' or 'cross_pair'; got {pairing!r}")
    pair_fn = cross_pair_accuracy if pairing == "cross_pair" else pairwise_accuracy
    break_fn = cross_pair_accuracy_breakdown if pairing == "cross_pair" else pairwise_accuracy_breakdown

    Cs = sorted(df["context_length"].unique().tolist())
    result: Dict[str, Any] = {
        "dataset_variant": variant,
        "pairing_protocol": pairing,
        "aggregation": s_cfg["aggregation"],
        "surprise_col": surprise_col,
        "context_lengths": [int(c) for c in Cs],
        "n_videos_total": int(len(df) // max(len(Cs), 1)),
    }

    # single-C or sweep-max
    if len(Cs) > 1:
        sweep = sweep_max_pairwise(df, surprise_col, break_by, pairing=pairing)
        result["sweep"] = sweep
        result["best_context_length"] = sweep["best_context_length"]
        result["overall"] = sweep["best"]["overall"] if sweep["best"] else None
        result["breakdown"] = sweep["best"]["breakdown"] if sweep["best"] else None
        result["auroc"] = sweep["best"]["auroc"] if sweep["best"] else None
    else:
        C = int(Cs[0])
        sub = df[df["context_length"] == C]
        overall = pair_fn(sub, surprise_col) if e_cfg["compute_pairwise"] else None
        breakdown = (
            break_fn(sub, surprise_col, break_by)
            if e_cfg["compute_pairwise"] else None
        )
        auroc = single_video_auroc(sub, surprise_col) if e_cfg["compute_auroc"] else None
        result["context_length"] = C
        result["overall"] = overall
        result["breakdown"] = breakdown
        result["auroc"] = auroc

    return result


def _summary_txt(summary: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    lines = ["===== IntPhys2 pairwise-surprise summary ====="]
    lines.append(f"tag: {cfg['tag']}")
    lines.append(f"split: {cfg['data']['split']}   root: {cfg['data']['root']}")
    lines.append(f"model.checkpoint: {cfg['model']['checkpoint']}")
    lines.append(
        f"surprise: window={cfg['surprise']['window_size']} stride={cfg['surprise']['stride']} "
        f"C={cfg['surprise'].get('context_length_sweep') or cfg['surprise']['context_length']} "
        f"dist={cfg['surprise']['distance']} agg={summary['aggregation']}"
    )
    lines.append(f"context_lengths evaluated: {summary['context_lengths']}")
    if "best_context_length" in summary and summary["best_context_length"] is not None:
        lines.append(f"best_context_length (max pairwise): C={summary['best_context_length']}")
    if summary.get("overall"):
        lines.append(format_summary("overall (pairwise)", summary["overall"], indent=2))
    if summary.get("auroc"):
        lines.append(format_summary("single-video (AUROC)", summary["auroc"], indent=2))
    if summary.get("breakdown"):
        lines.append("  breakdowns:")
        for k, v in summary["breakdown"].items():
            lines.append(format_summary(k, v, indent=4))
    return "\n".join(lines)


def _write_outputs(
    out_dir: str,
    cfg: Dict[str, Any],
    df: pd.DataFrame,
    summary: Dict[str, Any],
    per_window_traces: Optional[List[Dict[str, Any]]] = None,
) -> None:
    o_cfg = cfg["output"]

    # 1. per_video.csv
    if o_cfg["save_per_video"]:
        df.to_csv(os.path.join(out_dir, "per_video.csv"), index=False)

    # 2. per_window (only if collected)
    if o_cfg["save_per_window"] and per_window_traces:
        pw_df = pd.DataFrame(per_window_traces)
        try:
            pw_df.to_parquet(os.path.join(out_dir, "per_window.parquet"), index=False)
        except Exception:
            # Parquet not available -> fall back to JSONL
            with open(os.path.join(out_dir, "per_window.jsonl"), "w") as f:
                for row in per_window_traces:
                    f.write(json.dumps(row) + "\n")

    # 3. summary.json + summary.txt
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(_summary_txt(summary, cfg))

    # 4. resolved config
    if o_cfg["save_config"]:
        with open(os.path.join(out_dir, "config.resolved.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    # 5. plots
    plots_dir = os.path.join(out_dir, "plots")
    if o_cfg["plot_breakdowns"] and summary.get("breakdown"):
        plot_breakdown_bars(
            summary["breakdown"], os.path.join(plots_dir, "pairwise_breakdown.png"),
            title=f"IntPhys2 pairwise accuracy — {cfg['tag']}",
        )
    if o_cfg["plot_context_sweep"] and "sweep" in summary:
        plot_context_sweep(
            summary["sweep"]["per_context_length"],
            os.path.join(plots_dir, "context_sweep.png"),
        )
    scene_ids = o_cfg.get("scene_curve_indices")
    if o_cfg["plot_scene_curves"] and scene_ids and per_window_traces:
        pw = pd.DataFrame(per_window_traces)
        for sid in scene_ids:
            sub = pw[pw["scene_index"] == int(sid)]
            if sub.empty:
                continue
            traces = {
                str(row["type"]): {
                    "window_starts": row["window_starts"],
                    "surprise": row["surprise"],
                }
                for _, row in sub.iterrows()
            }
            plot_scene_surprise(
                int(sid), traces,
                os.path.join(plots_dir, f"scene_{sid}.png"),
                context_length=int(sub.iloc[0]["context_length"]),
            )


# --------------------------- entry point -------------------------------------


def _apply_sets(cfg: dict, sets) -> dict:
    """`--set a.b=1` 점 경로 덮어쓰기. `z_research/scripts/run.sh` 의 `SET=` 과 같은 규약이다.

    실험마다 yaml 을 뜨지 않기 위한 장치다 (configs/README.md 의 규칙).
    값은 YAML 로 파싱한다 -> `[4,6,8]`, `true`, `null` 전부 된다. `null` 은 그 키를 지운다.
    """
    import yaml as _y
    for kv in sets or []:
        if "=" not in kv:
            raise SystemExit(f"--set 은 KEY=VALUE 여야 한다 -> {kv}")
        key, raw = kv.split("=", 1)
        try:
            val = _y.safe_load(raw)
        except Exception:
            val = raw
        # ⚠️ 점 없는 최상위 키(`tag=...`)도 되어야 한다. 예전 판은 `[node] + rest[:-1]` 를
        #    돌아서 `cfg["tag"]` 를 dict 로 만들어 버렸다 (2026-09-22).
        parts = key.split(".")
        cur = cfg
        for k in parts[:-1]:
            if not isinstance(cur.get(k), dict):
                cur[k] = {}
            cur = cur[k]
        leaf = parts[-1]
        if val is None:
            cur.pop(leaf, None)
        else:
            cur[leaf] = val
    return cfg


def main(cfg_path: str, override_device: Optional[str] = None, verbose: bool = True,
         sets=None) -> Dict[str, Any]:
    cfg = _apply_sets(_resolve_config(cfg_path), sets)
    rank, world_size, use_ddp = _init_ddp_if_requested(bool(cfg["evaluation"]["ddp"]))
    _setup_logging(rank, verbose=verbose and cfg["evaluation"]["verbose"])
    logger.info(f"rank {rank}/{world_size} (ddp={use_ddp})")
    logger.info(f"config: {cfg_path}")
    if rank == 0:
        logger.info(json.dumps({k: v for k, v in cfg.items() if k != "model"}, indent=2, default=str))

    # device
    if override_device:
        device = torch.device(override_device)
    elif use_ddp:
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    bundle = build_from_config(cfg["model"], device=device)

    # dataset (whole thing on every rank; we shard by index at score time)
    dataset = build_dataset(cfg["data"])
    logger.info(f"{cfg['data'].get('dataset_variant', 'intphys2')} dataset: {len(dataset)} videos")

    # scoring
    records = _score_all_videos(dataset, bundle, cfg, rank, world_size)
    per_window_traces = getattr(_score_all_videos, "_per_window", [])

    # gather across ranks (records) and (optionally) traces.
    # IMPORTANT: both gathers must be UNCONDITIONAL across ranks -- gating on
    # per-rank list truthiness would deadlock all_gather_object when only some
    # ranks skip the collective (see review finding: happens when limit_videos <
    # world_size, or when a rank's videos all fail scoring, leaving traces=[]).
    if use_ddp and world_size > 1:
        records = _all_gather_records(records, world_size)
        if bool(cfg["output"].get("save_per_window", False)):
            per_window_traces = _all_gather_records(per_window_traces, world_size)

    if rank != 0:
        # Only rank 0 aggregates + writes.
        return {"rank": rank, "n_records": len(records)}

    df = to_dataframe(records)
    summary = _compute_summary(df, cfg)
    out_dir = _output_dir(cfg)
    _write_outputs(out_dir, cfg, df, summary, per_window_traces=per_window_traces)

    logger.info("\n" + _summary_txt(summary, cfg))
    logger.info(f"outputs written to {out_dir}")
    return summary


def _cli():
    p = argparse.ArgumentParser(description="IntPhys2 pairwise-surprise evaluation")
    p.add_argument("--config", "--fname", dest="config", required=True, help="YAML config path")
    p.add_argument("--device", default=None, help="override device (cuda:0, cpu, ...)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                   help="점 경로로 config 덮어쓰기 (여러 번 가능). 예:\n"
                        "  --set surprise.window_size=16\n"
                        "  --set 'surprise.context_length_sweep=[4,6,8,10,12,14]'\n"
                        "  --set tag=my_run   --set evaluation.limit_videos=8")
    args = p.parse_args()
    return main(args.config, override_device=args.device, verbose=not args.quiet, sets=args.sets)


if __name__ == "__main__":
    _cli()
