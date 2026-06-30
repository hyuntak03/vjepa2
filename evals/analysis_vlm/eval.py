# -----------------------------------------------------------------------------
# Unified frozen-encoder probing analysis across model families.
#
# Selectable encoder (config: experiment.analysis.model):
#   vjepa       -> V-JEPA2 ViT, per-LAYER features (clip-tensor data path; reuses
#                  the stock make_dataloader + evals.analysis.modelcustom backend)
#   llavavideo  -> LLaVA-Video SigLIP vision tower / projector (raw-frame path)
#   qwen3vl     -> Qwen3-VL vision ViT before/after merger + deepstack (raw-frame)
#
# For each (stage x probe-spec) it builds one probe head, trains them jointly on
# top of the FROZEN encoder, and reports a [stage x probe] accuracy matrix.
#
# Routed by `eval_name: analysis_vlm` (scaffold.py dynamic import) so main.py /
# scaffold.py / the existing evals/analysis/ are untouched.
#
# NOTE: each model family lives in its own conda env (vjepa2 / lmms_eval_llavavideo
# / lmms_eval_py311_2.7). Only the selected backend is imported (lazily, via
# importlib on module_name), so heavy/conflicting deps never co-load.
# -----------------------------------------------------------------------------

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import json
import logging
import pprint
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from evals.video_classification_frozen.models import init_module
from evals.analysis.probes import build_probe, probe_name

from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduceSum, init_distributed
from src.utils.logging import CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

pp = pprint.PrettyPrinter(indent=4)

# model -> (backend module, data mode). data mode picks the dataloader + how the
# encoder forward is called. Override either via model_kwargs.module_name /
# experiment.analysis.data_mode if needed.
_BACKENDS = {
    "vjepa": ("evals.analysis.modelcustom.vit_encoder_multilayer", "clip"),
    "llavavideo": ("evals.analysis_vlm.modelcustom.llava_video_encoder", "raw"),
    "qwen3vl": ("evals.analysis_vlm.modelcustom.qwen3vl_encoder", "raw"),
}

_VIT_DEPTH = {"vit_large": 24, "vit_huge": 32, "vit_giant": 40, "vit_giant_xformers": 40, "vit_gigantic": 48}


def _resolve_layers(layers_cfg, model_name):
    if isinstance(layers_cfg, str) and layers_cfg.lower() == "all":
        if model_name not in _VIT_DEPTH:
            raise ValueError(f"stages='all' but depth of {model_name} unknown; give an explicit list")
        return list(range(_VIT_DEPTH[model_name]))
    assert isinstance(layers_cfg, (list, tuple)) and len(layers_cfg) > 0, "stages must be a non-empty list or 'all'"
    return [int(x) for x in layers_cfg]


def _unwrap(m):
    """Underlying module of a (possibly) DDP-wrapped head -> topology-independent checkpoints."""
    return m.module if isinstance(m, DistributedDataParallel) else m


def main(args_eval, resume_preempt=False):

    # --------------------------------------------------------------------- #
    #  CONFIG
    # --------------------------------------------------------------------- #
    val_only = args_eval.get("val_only", False)
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")          # local .pth (vjepa) OR HF repo id (vlm)
    cache_dir = args_pretrain.get("cache_dir")            # HF cache root to resolve a repo-id checkpoint
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs") or {}
    args_wrapper = dict(args_pretrain.get("wrapper_kwargs") or {})
    enc_model_name = (args_model.get("encoder") or {}).get("model_name")

    args_exp = args_eval.get("experiment")
    args_analysis = args_exp.get("analysis")

    # -- model selection -> backend module + data mode
    model_sel = (args_analysis.get("model") or "").lower()
    data_mode = args_analysis.get("data_mode")
    if model_sel:
        if model_sel not in _BACKENDS:
            raise ValueError(f"unknown analysis.model={model_sel!r}; valid: {list(_BACKENDS)}")
        default_module, default_mode = _BACKENDS[model_sel]
        module_name = module_name or default_module
        data_mode = data_mode or default_mode
    if data_mode is None:
        data_mode = "raw" if (module_name and "analysis_vlm" in module_name) else "clip"
    assert module_name, "must set experiment.analysis.model or model_kwargs.module_name"
    logger.info(f"analysis: model={model_sel or '(from module_name)'} module={module_name} data_mode={data_mode}")

    # -- stages.  Forms: structured dict {vision_encoder: [..]|"all", <toggles>: true},
    #    or shorthand "all" / [int,...], or legacy [concrete-name,...].
    #    Only the `vision_encoder` stage carries a per-layer selection.
    stages_cfg = args_analysis.get("stages", args_analysis.get("layers"))
    if data_mode == "clip":
        # V-JEPA backbone == vision_encoder; it has no other stages.
        ve = stages_cfg
        if isinstance(stages_cfg, dict):
            if "vision_encoder" not in stages_cfg:
                raise ValueError(
                    "clip/vjepa: analysis.stages dict must contain 'vision_encoder' (list[int] or 'all'). "
                    f"got keys {list(stages_cfg)} — check for typos."
                )
            ve = stages_cfg["vision_encoder"]
            extras = [k for k in stages_cfg if k != "vision_encoder" and stages_cfg.get(k)]
            if extras:
                logger.warning(f"clip/vjepa encoder has only vision_encoder layers; ignoring {extras}")
        stages = _resolve_layers(ve, enc_model_name)
        args_wrapper["out_layers"] = stages
    else:
        # the VLM backend resolves the spec (incl. "all"/layer-list/toggles) -> encoder.stages
        args_wrapper["out_stages"] = stages_cfg
        if cache_dir and "cache_dir" not in args_wrapper:
            args_wrapper["cache_dir"] = cache_dir  # let resolve_model_dir find a repo-id checkpoint
        stages = None

    probe_specs = args_analysis.get("probes")
    assert isinstance(probe_specs, list) and len(probe_specs) > 0, "experiment.analysis.probes must be a non-empty list"
    make_plot = args_analysis.get("plot", False)
    plot_pez = args_analysis.get("plot_pez")  # [lo,hi] layer-fraction band to shade (Physics Emergence
    if plot_pez:                              # Zone, ~1/3 depth); None/false -> no shading
        assert len(plot_pez) == 2 and 0.0 <= plot_pez[0] < plot_pez[1] <= 1.0, \
            f"plot_pez must be [lo,hi] with 0<=lo<hi<=1, got {plot_pez}"

    # -- DATA
    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    num_classes = args_data.get("num_classes")
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)
    # clip path (V-JEPA) spatial handling: 'crop' = shorter-side resize + center-crop (stock,
    # default) | 'resize' = direct resize to resolution^2 (aspect squashed, like the VLM path).
    clip_resize_mode = args_data.get("resize_mode", "crop")
    if clip_resize_mode not in ("crop", "resize"):  # fail loud (don't silently fall back to crop)
        raise ValueError(f"data.resize_mode must be 'crop' or 'resize', got {clip_resize_mode!r}")

    # -- TASK: 'classification' (CrossEntropy->accuracy) or 'regression' (MSE->R^2 of probes
    # predicting CONTINUOUS variables). The CSV integer label INDEXES regression.targets_npy
    # (an (N,D) .npy), so the dataloaders (incl. the shared clip VideoDataset AND the VLM raw
    # path) stay unchanged — the harness maps label->target vector. `variables` lists one or
    # more named targets, each a column-slice of that array; EACH becomes its own R^2 curve on
    # the SAME plot (paper Fig.2c: speed / direction / accel together). e.g.
    #   regression:
    #     targets_npy: /.../toyball_targets.npy
    #     variables:
    #       - {name: speed,     cols: [0]}
    #       - {name: direction, cols: [1, 2]}   # sin,cos of angle (circular)
    #       - {name: accel_mag, cols: [3]}
    task = str(args_analysis.get("task", "classification")).lower()
    reg_cfg = args_analysis.get("regression") or {}
    targets_arr, reg_vars = None, [(None, None)]   # (var_name, cols); classification = single dummy
    if task == "regression":
        tpath = reg_cfg.get("targets_npy") or reg_cfg.get("targets")
        assert tpath, "task=regression needs experiment.analysis.regression.targets_npy ((N,D) .npy)"
        targets_arr = np.load(tpath).astype(np.float32)
        if targets_arr.ndim == 1:
            targets_arr = targets_arr[:, None]
        # standardize per-column (NaN-aware: a column may be defined on only a subset of videos,
        # NaN elsewhere). R^2 is invariant to this affine transform; it keeps MSE/lr well-scaled
        # regardless of units (pixels vs sin/cos in [-1,1]). NaNs stay NaN -> masked out per head.
        mu = np.nanmean(targets_arr, axis=0, keepdims=True)
        sd = np.nanstd(targets_arr, axis=0, keepdims=True)
        targets_arr = (targets_arr - mu) / np.clip(sd, 1e-6, None)
        var_cfg = reg_cfg.get("variables")
        if not var_cfg:  # default: one variable spanning all columns
            var_cfg = [{"name": reg_cfg.get("name", "target"), "cols": list(range(targets_arr.shape[1]))}]
        reg_vars = [(v["name"], [int(c) for c in v["cols"]]) for v in var_cfg]
        D = targets_arr.shape[1]
        for vn, cols in reg_vars:
            assert all(0 <= c < D for c in cols), f"variable {vn!r} cols {cols} out of range (D={D})"
        logger.info(f"task=regression: targets={tpath} shape={targets_arr.shape} "
                    f"variables={[(n, c) for n, c in reg_vars]}")
    elif task != "classification":
        raise ValueError(f"experiment.analysis.task must be 'classification' or 'regression', got {task!r}")

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    default_opt = args_opt.get("default_head", {})
    # attentive probes over high-dim VLM stages (2560/3584) are big; saving AdamW
    # states too can make latest.pt 10s of GB. Default: probe weights only.
    save_optimizer = args_opt.get("save_optimizer", False)
    # FEATURE CACHE: frozen encoder + deterministic preprocessing (no augmentation)
    # -> identical features every epoch, so encode ONCE and train probes over the cache.
    cache_features = args_opt.get("cache_features", False)
    cache_pooling = args_opt.get("cache_pooling", "tokens")  # 'pooled' (linear-only, tiny) | 'tokens' (all probes)
    cache_max_gb = args_opt.get("cache_max_gb", 64)          # abort if est. per-rank cache RAM exceeds this

    def _opt_kwargs(spec):
        o = dict(default_opt)
        o.update(spec.get("optimization", {}))
        return dict(
            ref_wd=o.get("weight_decay", 0.01),
            final_wd=o.get("final_weight_decay", 0.01),
            start_lr=o.get("start_lr", 0.0),
            ref_lr=o.get("lr", 0.001),
            final_lr=o.get("final_lr", 0.0),
            warmup=o.get("warmup", 1.0),
        )

    # --------------------------------------------------------------------- #
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # regression targets live on-device (N,D); per-batch we index targets_t[label] (label = CSV int)
    targets_t = torch.from_numpy(targets_arr).to(device) if targets_arr is not None else None

    folder = os.path.join(pretrain_folder, "analysis_vlm/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    # -- encoder (frozen), exposing the requested stages
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    if stages is None:  # raw mode: backend resolved the spec (incl. "all") -> read it back
        stages = list(encoder.stages)
    embed_dims = getattr(encoder, "embed_dims", None) or [encoder.embed_dim] * len(stages)
    assert len(embed_dims) == len(stages), f"encoder exposed {len(embed_dims)} dims for {len(stages)} stages"

    # -- heads: one probe per (stage x probe-spec) --------------------------
    use_ddp = dist.is_available() and dist.is_initialized()
    if not use_ddp:
        logger.info("No distributed process group -> running probes without DDP (single-process).")

    heads, head_opt_kwargs, _name_set = [], [], set()
    for stage_pos, stage in enumerate(stages):
        if data_mode == "clip":
            # block index is unique across clip stages -> use it directly as the plot x
            stage_tag, layer_val = f"L{int(stage):02d}", int(stage)
        else:
            # raw/VLM: use stage POSITION as the plot x (unique — avoids block_5 vs deepstack_5
            # colliding on x=5); the actual stage name labels the tick. summary.json keeps exact acc.
            stage_tag, layer_val = str(stage), stage_pos
        for spec in probe_specs:
            ptype = str(spec.get("type", "attentive")).lower()
            tpos = str(spec.get("temporal_pos", "none")).lower()  # none | learnable | rope (attentive only)
            pooling = str(spec.get("pooling", "mean")).lower()
            framewise = ptype == "linear" and pooling.startswith("framewise")  # spatial-pool per frame, keep T
            use_tpos = ptype == "attentive" and tpos in ("learnable", "rope")
            pname = probe_name(spec) + (f"-{tpos}" if use_tpos else "")
            if cache_features and cache_pooling == "pooled":
                # cache stores only [mean‖max] -> linear probes only (attentive needs tokens)
                if ptype != "linear":
                    raise ValueError(
                        "cache_pooling='pooled' caches only pooled vectors -> linear probes only. "
                        "Use cache_pooling='tokens' (or cache_features=false) for attentive probes."
                    )
                if framewise:
                    raise ValueError(
                        f"pooling='{pooling}' keeps the per-frame token structure, which the 'pooled' "
                        f"cache has already collapsed. Use cache_pooling='tokens' (or cache_features=false)."
                    )

            def _build(out_dim):  # build one probe head with the given output dim
                if cache_features and cache_pooling == "pooled":
                    from evals.analysis_vlm.cache import PooledLinearProbe

                    return PooledLinearProbe(embed_dim=embed_dims[stage_pos], num_classes=out_dim,
                                             pooling=pooling, pre_norm=spec.get("pre_norm", True))
                if framewise:
                    # temporal-preserving linear: spatial-pool within each frame, concat, Linear.
                    nt = getattr(encoder, "num_temporal", None)
                    if nt is None:
                        raise ValueError(f"pooling='{pooling}' needs encoder.num_temporal (VLM backends only).")
                    from evals.analysis_vlm.probes import TemporalLinearProbe

                    return TemporalLinearProbe(embed_dim=embed_dims[stage_pos], num_temporal=nt,
                                               num_classes=out_dim, spatial_pool=pooling.split("_", 1)[1],
                                               pre_norm=spec.get("pre_norm", True))
                if use_tpos:
                    nt = getattr(encoder, "num_temporal", None)
                    if nt is None:
                        raise ValueError(
                            f"temporal_pos='{tpos}' needs encoder.num_temporal (VLM backends only); "
                            f"this encoder does not expose it (V-JEPA already encodes time via RoPE)."
                        )
                    from evals.analysis_vlm.probes import TemporalAttentiveClassifier

                    return TemporalAttentiveClassifier(
                        embed_dim=embed_dims[stage_pos], num_temporal=nt, mode=tpos,
                        num_heads=spec.get("num_heads", 16),
                        depth=spec.get("num_probe_blocks", spec.get("depth", 1)),
                        num_classes=out_dim, use_activation_checkpointing=True)
                return build_probe(spec, embed_dim=embed_dims[stage_pos], num_classes=out_dim,
                                   use_activation_checkpointing=True)

            # one head per regressed VARIABLE (classification: a single dummy var) -> each
            # variable becomes its own R^2 curve on the plot (grouped by `series`).
            for var_name, var_cols in reg_vars:
                out_dim = len(var_cols) if var_cols is not None else num_classes
                module = _build(out_dim).to(device)
                if use_ddp:
                    module = DistributedDataParallel(module, static_graph=True)
                var_tag = f"__{var_name}" if var_name is not None else ""
                name = f"{stage_tag}_{pname}{var_tag}"
                if name in _name_set:  # de-collide duplicate specs
                    k = 2
                    while f"{name}#{k}" in _name_set:
                        k += 1
                    name = f"{name}#{k}"
                _name_set.add(name)
                # `series` = plot line grouping (variable for regression, probe for classification).
                # carry stage name explicitly so plotting doesn't re-parse `name`.
                # plot series: variable for regression (probe for classification); when MULTIPLE
                # probe specs exist (e.g. linear vs attentive), append the probe so each is its own curve.
                if var_name is None:
                    series = pname
                elif len(probe_specs) > 1:
                    series = f"{var_name}·{pname}"
                else:
                    series = var_name
                heads.append(dict(name=name, layer=layer_val, layer_pos=stage_pos,
                                  probe=pname, series=series,
                                  stage=stage_tag, module=module, tcols=var_cols))
                head_opt_kwargs.append(_opt_kwargs(spec))
    head_names = [h["name"] for h in heads]
    logger.info(f"Built {len(heads)} probe heads over {len(stages)} stages: {head_names}")

    # -- data (clip vs raw), with optional one-time FEATURE CACHE
    run_mode = data_mode

    def _split_loader(root, training, persistent=True, workers=None):
        """A loader over a single split. training=False => deterministic (no augmentation).
        workers overrides num_workers (the cache pre-pass uses 0: a 2nd DataLoader's worker
        respawn deadlocks under spawn multiprocessing, so decode single-threaded in-process)."""
        w = num_workers if workers is None else workers
        if data_mode == "clip":
            if clip_resize_mode == "resize":
                # DIRECT resize to resolution^2 (no shorter-side + center-crop; aspect squashed,
                # like the VLM SigLIP path). Mirrors make_dataloader's init_data call EXACTLY,
                # swapping ONLY the transform (deterministic; no train-time augmentation).
                from evals.video_classification_frozen.eval import DEFAULT_NORMALIZATION
                from src.datasets.data_manager import init_data

                transform = _DirectResizeClipTransform(resolution, normalization or DEFAULT_NORMALIZATION)
                ld, samp = init_data(
                    data=dataset_type, root_path=[root], transform=transform,
                    batch_size=batch_size, world_size=world_size, rank=rank,
                    clip_len=frames_per_clip, frame_sample_rate=frame_step, duration=duration,
                    num_clips=num_segments, allow_clip_overlap=True, num_workers=w, drop_last=False,
                )
            else:
                from evals.video_classification_frozen.eval import make_dataloader

                ld, samp = make_dataloader(
                    dataset_type=dataset_type, root_path=[root], img_size=resolution,
                    frames_per_clip=frames_per_clip, frame_step=frame_step, eval_duration=duration,
                    num_segments=num_segments, num_views_per_segment=1, allow_segment_overlap=True,
                    batch_size=batch_size, world_size=world_size, rank=rank, training=training,
                    num_workers=w, normalization=normalization,
                )
            # init_data/make_videodataset ignore the drop_last arg and default to drop_last=True,
            # which silently drops partial batches — fatal for small splits (e.g. 20 val / 8 ranks
            # ≈ 3/rank < batch_size -> 0 batches -> "saw 0 samples"). Force-keep them. Safe for DDP:
            # DistributedSampler pads to equal per-rank counts, so batch counts stay aligned.
            if getattr(ld, "batch_sampler", None) is not None:
                ld.batch_sampler.drop_last = False
            return ld, samp
        from evals.analysis_vlm.data import make_raw_dataloader

        return make_raw_dataloader(root, frames_per_clip, batch_size, world_size, rank,
                                   training=training, num_workers=w, persistent=persistent)

    if cache_features:
        # ONE deterministic pre-pass per split (training=False -> NO augmentation), cache features,
        # then train probes over the cache. Each rank caches its own shard. Build the two pre-pass
        # loaders SEQUENTIALLY (drop train before creating val) with persistent_workers=False, so
        # workers don't pile up / deadlock at the train->val transition under spawn multiprocessing.
        from evals.analysis_vlm.cache import build_feature_cache, make_cached_loader

        def encode_fn(d):
            return _encode(encoder, d, device, data_mode, use_bfloat16)

        enc_num_temporal = getattr(encoder, "num_temporal", None)  # for cache_pooling='framewise'
        logger.info(f"cache_features=true (pooling={cache_pooling}): one deterministic pre-pass per split...")

        # workers=0: the cache pre-pass iterates each loader once; using DataLoader subprocess
        # workers here deadlocks at the train->val loader transition under spawn multiprocessing.
        tr_loader, _ = _split_loader(train_data_path[0], training=False, persistent=False, workers=0)
        tr_feats, tr_labels = build_feature_cache(encode_fn, tr_loader, cache_pooling,
                                                  num_temporal=enc_num_temporal, max_gb=cache_max_gb,
                                                  label="train-cache", rank=rank)
        del tr_loader

        va_loader, _ = _split_loader(val_data_path[0], training=False, persistent=False, workers=0)
        va_feats, va_labels = build_feature_cache(encode_fn, va_loader, cache_pooling,
                                                  num_temporal=enc_num_temporal, max_gb=cache_max_gb,
                                                  label="val-cache", rank=rank)
        del va_loader
        train_loader = make_cached_loader(tr_feats, tr_labels, batch_size, training=True)
        val_loader = make_cached_loader(va_feats, va_labels, batch_size, training=False)
        train_sampler = None
        run_mode = "cached"
    else:
        train_loader, train_sampler = _split_loader(train_data_path[0], training=True)
        val_loader, _ = _split_loader(val_data_path[0], training=False)

    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe} (mode={run_mode})")

    # -- optimizer: ONE fused AdamW with one param-group per head (collapses N AdamW.step()/
    #    zero_grad()/scaler.step() launches into one; per-group LR/WD schedule stays identical).
    optimizer, scaler, scheduler, wd_scheduler = _init_opt_fused(
        classifiers=[h["module"] for h in heads], opt_kwargs=head_opt_kwargs,
        iterations_per_epoch=ipe, num_epochs=num_epochs, use_bfloat16=use_bfloat16,
    )

    if rank == 0:
        cols = [("%d", "epoch")]
        for n in head_names:
            cols += [("%.5f", f"{n}_train"), ("%.5f", f"{n}_val")]
        csv_logger = CSVLogger(log_file, *cols)

    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        heads, optimizer, scaler, start_epoch = load_checkpoint(
            device, latest_path, heads, optimizer, scaler, val_only=val_only
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch):
        if rank != 0:
            return
        ckpt = {
            "classifiers": [_unwrap(h["module"]).state_dict() for h in heads],
            "epoch": epoch, "head_names": head_names, "stages": stages,
            "batch_size": batch_size, "world_size": world_size,
        }
        if save_optimizer:  # optional: large for high-dim attentive probes
            ckpt["opt"] = [o.state_dict() for o in optimizer]
            ckpt["scaler"] = None if scaler[0] is None else [s.state_dict() for s in scaler]
        torch.save(ckpt, latest_path)

    # --------------------------------------------------------------------- #
    #  TRAIN / EVAL LOOP
    # --------------------------------------------------------------------- #
    best_val = {n: -float("inf") for n in head_names}   # R^2 can be < 0; -inf is a safe floor
    last_epoch = start_epoch
    for epoch in range(start_epoch, num_epochs):
        last_epoch = epoch + 1
        logger.info("Epoch %d" % (epoch + 1))
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if val_only:
            train_acc = {n: -1.0 for n in head_names}
        else:
            train_acc = run_one_epoch(
                device=device, training=True, encoder=encoder, heads=heads, scaler=scaler,
                optimizer=optimizer, scheduler=scheduler, wd_scheduler=wd_scheduler,
                data_loader=train_loader, use_bfloat16=use_bfloat16, data_mode=run_mode, rank=rank,
                task=task, targets=targets_t,
            )

        val_acc = run_one_epoch(
            device=device, training=False, encoder=encoder, heads=heads, scaler=scaler,
            optimizer=optimizer, scheduler=scheduler, wd_scheduler=wd_scheduler,
            data_loader=val_loader, use_bfloat16=use_bfloat16, data_mode=run_mode, rank=rank,
            task=task, targets=targets_t,
        )
        for n in head_names:
            best_val[n] = max(best_val[n], val_acc[n])

        logger.info("[epoch %d] stage x probe acc (train | val | best):" % (epoch + 1))
        for h in heads:
            n = h["name"]
            logger.info("    %-32s  train %6.2f%%  val %6.2f%%  best %6.2f%%"
                        % (n, train_acc[n], val_acc[n], best_val[n]))
        if rank == 0:
            row = [epoch + 1]
            for n in head_names:
                row += [train_acc[n], val_acc[n]]
            csv_logger.log(*row)
            with open(os.path.join(folder, "summary.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "num_epochs": num_epochs, "model": model_sel,
                           "data_mode": data_mode, "num_classes": num_classes,
                           "task": task, "metric": ("r2" if task == "regression" else "accuracy"),
                           "variables": ([{"name": n, "cols": c} for n, c in reg_vars]
                                         if task == "regression" else None),
                           "stages": [str(s) for s in stages],
                           "head_names": head_names, "val_acc": val_acc, "train_acc": train_acc,
                           "best_val_acc": best_val}, f, indent=2)

        save_checkpoint(epoch + 1)
        if val_only:
            break

    if rank == 0 and make_plot:
        from evals.analysis.plotting import plot_layer_val_acc

        metric = "r2" if task == "regression" else "accuracy"
        sub = f"{model_sel} | best {'R²' if metric == 'r2' else 'val'} over {last_epoch} epoch(s)"
        # multi-variable R²: each variable is its own curve (legend), so no single target_label
        plot_layer_val_acc(heads, best_val, os.path.join(folder, "stage_val_acc.png"),
                           subtitle=sub, num_classes=num_classes, metric=metric, pez=plot_pez)


class _DirectResizeClipTransform:
    """Eval-style V-JEPA clip transform that DIRECTLY resizes each frame to (crop, crop) —
    no shorter-side-resize + center-crop — so the FULL frame is kept (aspect ratio squashed,
    like the VLM SigLIP 384x384 path). Deterministic (no augmentation). Same call contract as
    VideoTransform in eval mode: __call__(buffer) -> [clip_tensor (C,T,H,W) normalized].

    Reuses the stock video-transform primitives (Compose/Resize/ClipToTensor/Normalize); used
    only for the clip path when experiment.data.resize_mode == 'resize'. The original
    VideoTransform / make_transforms are left untouched."""

    def __init__(self, crop_size, normalize):
        import src.datasets.utils.video.transforms as vt
        import src.datasets.utils.video.volume_transforms as vvt

        self.eval_transform = vt.Compose([
            vt.Resize((crop_size, crop_size), interpolation="bilinear"),  # (w,h) tuple -> direct resize
            vvt.ClipToTensor(),
            vt.Normalize(mean=normalize[0], std=normalize[1]),
        ])

    def __call__(self, buffer):
        return [self.eval_transform(buffer)]


def _init_opt_fused(classifiers, opt_kwargs, iterations_per_epoch, num_epochs, use_bfloat16=False):
    """ONE AdamW with one param-group per head (each carrying its own mc_* schedule keys),
    a SINGLE LR/WD schedule and a SINGLE GradScaler.

    Numerically identical to the one-optimizer-per-head path (the WarmupCosineLRSchedule /
    CosineWDSchedule already iterate self.optimizer.param_groups and set LR/WD per group), but
    collapses N AdamW.step() / zero_grad() / scaler.step() launches into ONE — ~25% off the
    cached probe step when there are many heads (e.g. 26-layer scan). Returned as length-1
    lists so the train loop and checkpoint code (which iterate the optimizer/scaler/scheduler/
    wd_scheduler lists) keep working unchanged."""
    from evals.video_classification_frozen.eval import CosineWDSchedule, WarmupCosineLRSchedule

    param_groups = []
    for c, kw in zip(classifiers, opt_kwargs):
        param_groups.append({
            "params": list(c.parameters()),  # materialize (a generator would exhaust)
            "mc_warmup_steps": int(kw.get("warmup") * iterations_per_epoch),
            "mc_start_lr": kw.get("start_lr"),
            "mc_ref_lr": kw.get("ref_lr"),
            "mc_final_lr": kw.get("final_lr"),
            "mc_ref_wd": kw.get("ref_wd"),
            "mc_final_wd": kw.get("final_wd"),
        })
    logger.info(f"Using ONE fused AdamW over {len(param_groups)} head param-group(s)")
    optimizer = torch.optim.AdamW(param_groups)
    T = int(num_epochs * iterations_per_epoch)
    scheduler = WarmupCosineLRSchedule(optimizer, T_max=T)
    wd_scheduler = CosineWDSchedule(optimizer, T_max=T)
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return [optimizer], [scaler], [scheduler], [wd_scheduler]


def _encode(encoder, data, device, data_mode, use_bfloat16):
    """Return (feats: list[(B,N,D) fp32 detached], labels, bsz)."""
    if data_mode == "cached":
        # data = (list[stage] of cached (B,...) tensors, labels) — no encoder call
        feats = [f.to(device, non_blocking=True).float() for f in data[0]]
        labels = data[1].to(device, non_blocking=True)
        return feats, labels, labels.size(0)
    if data_mode == "clip":
        clips = [[dij.to(device, non_blocking=True) for dij in di] for di in data[0]]
        clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
        labels = data[1].to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_bfloat16):
            feats = encoder(clips, clip_indices)
    else:
        frames_list, labels = data[0], data[1]
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():  # VLM encoder runs in its own (half) dtype, no autocast
            feats = encoder(frames_list)
    # keep the encoder dtype (fp16): the stages are views into the hidden states, so this is
    # ~free; upcasting all stages to fp32 here would double peak GPU memory (OOM on all-layer).
    # The cache stores fp16 (.half()) and the non-cached probe runs under autocast anyway.
    feats = [f.detach() for f in feats]
    return feats, labels, labels.size(0)


def run_one_epoch(device, training, encoder, heads, scaler, optimizer, scheduler,
                  wd_scheduler, data_loader, use_bfloat16, data_mode, rank=0,
                  task="classification", targets=None):
    """task='classification' -> CrossEntropy loss, returns per-head val ACCURACY (%).
    task='regression'        -> MSE loss on a continuous target (looked up as targets[label],
                                where the integer label indexes the (N,D) targets tensor),
                                returns per-head val R^2 (1 - SS_res/SS_tot, all-reduced)."""
    for h in heads:
        h["module"].train(mode=training)

    is_reg = task == "regression"
    criterion = torch.nn.MSELoss() if is_reg else torch.nn.CrossEntropyLoss()
    n_heads = len(heads)
    amp_scaler = scaler[0]
    total = torch.zeros((), device=device)
    if is_reg:
        head_cols = [h["tcols"] for h in heads]         # per-head column slice into the (N,D) targets
        Dmax = max(len(c) for c in head_cols)
        # PER-HEAD stats over each head's VALID (non-NaN target) samples — lets one combined
        # dataset hold multiple variables defined on different video subsets (NaN elsewhere),
        # e.g. speed (velocity clips) + accel_mag (acceleration clips) + direction (both).
        ss_res = torch.zeros(n_heads, device=device)         # Σ‖pred-y‖² per head
        sum_y = torch.zeros(n_heads, Dmax, device=device)    # Σy per head (padded to Dmax)
        sum_y2 = torch.zeros(n_heads, device=device)         # Σ‖y‖² per head
        cnt = torch.zeros(n_heads, device=device)            # #valid samples per head
    else:
        correct = torch.zeros(n_heads, device=device)

    iterator = data_loader
    if rank == 0:
        try:
            from tqdm import tqdm

            iterator = tqdm(data_loader, desc=("train" if training else "  val"),
                            dynamic_ncols=True, mininterval=2.0, leave=False)
        except Exception:
            pass

    for itr, data in enumerate(iterator):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        feats, labels, bsz = _encode(encoder, data, device, data_mode, use_bfloat16)
        yfull = targets[labels].float() if is_reg else None   # (B,D) all target columns

        # In validation, run the heads under no_grad: no autograd graph is built and the
        # DDP reducer is NOT armed (a grad-enabled DDP forward with no backward trips the
        # static_graph reducer on the next forward under multi-GPU).
        grad_ctx = nullcontext() if training else torch.no_grad()
        with grad_ctx, torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_bfloat16):
            preds = [h["module"](feats[h["layer_pos"]]) for h in heads]
            if not training:
                losses = None
            elif is_reg:
                # masked-mean MSE per head: NaN target rows contribute 0 (kept in the graph so the
                # DDP static-graph structure is identical across ranks regardless of which rows are valid).
                losses = []
                for hi in range(n_heads):
                    yh = yfull[:, head_cols[hi]]
                    m = (~torch.isnan(yh).any(dim=1)).float()                       # (B,)
                    err = ((preds[hi] - torch.nan_to_num(yh)) ** 2).sum(dim=1) * m  # (B,)
                    losses.append(err.sum() / m.sum().clamp(min=1.0))
            else:
                losses = [criterion(p, labels) for p in preds]

        if training:
            loss_total = sum(losses)
            if amp_scaler is not None:
                amp_scaler.scale(loss_total).backward()
                for o in optimizer:
                    amp_scaler.step(o)
                amp_scaler.update()
            else:
                loss_total.backward()
                for o in optimizer:
                    o.step()
            for o in optimizer:
                o.zero_grad(set_to_none=True)

        with torch.no_grad():
            total += bsz
            if is_reg:
                for hi in range(n_heads):
                    d = len(head_cols[hi])
                    yh = yfull[:, head_cols[hi]]
                    m = ~torch.isnan(yh).any(dim=1)            # valid rows for this variable
                    if m.any():
                        p, y = preds[hi][m].float(), yh[m]
                        ss_res[hi] += ((p - y) ** 2).sum()
                        sum_y[hi, :d] += y.sum(dim=0)
                        sum_y2[hi] += (y ** 2).sum()
                        cnt[hi] += m.sum()
            else:
                for hi, p in enumerate(preds):
                    correct[hi] += (p.argmax(dim=1) == labels).sum()

        if rank == 0 and hasattr(iterator, "set_postfix") and (itr % 20 == 0):
            if is_reg:
                best = -9.9
                for hi in range(n_heads):
                    d, nh = len(head_cols[hi]), cnt[hi].clamp(min=1)
                    sst = (sum_y2[hi] - (sum_y[hi, :d] ** 2).sum() / nh).clamp(min=1e-12)
                    best = max(best, (1.0 - ss_res[hi] / sst).item())
                iterator.set_postfix(R2=f"{best:.3f}")  # best head so far
            else:
                iterator.set_postfix(best=f"{(100.0 * correct.max() / total.clamp(min=1)).item():.1f}%")

    total = AllReduceSum.apply(total)
    if is_reg:
        ss_res = AllReduceSum.apply(ss_res)
        sum_y = AllReduceSum.apply(sum_y)
        sum_y2 = AllReduceSum.apply(sum_y2)
        cnt = AllReduceSum.apply(cnt)
        r2 = []
        for hi in range(n_heads):
            d, nh = len(head_cols[hi]), cnt[hi].clamp(min=1)
            sst = (sum_y2[hi] - (sum_y[hi, :d] ** 2).sum() / nh).clamp(min=1e-12)
            r2.append((1.0 - ss_res[hi] / sst).item())
        return {h["name"]: r2[hi] for hi, h in enumerate(heads)}
    correct = AllReduceSum.apply(correct)
    accs = (100.0 * correct / total.clamp(min=1)).tolist()
    return {h["name"]: accs[hi] for hi, h in enumerate(heads)}


def load_checkpoint(device, r_path, heads, opt, scaler, val_only=False):
    ckpt = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")
    for h, pd in zip(heads, ckpt["classifiers"]):
        _unwrap(h["module"]).load_state_dict(pd)
    if val_only:
        return heads, opt, scaler, 0
    epoch = ckpt["epoch"]
    if ckpt.get("opt") is not None:  # weights-only checkpoints (save_optimizer=false) have no opt
        [o.load_state_dict(pd) for o, pd in zip(opt, ckpt["opt"])]
        if scaler[0] is not None and ckpt.get("scaler") is not None:
            [s.load_state_dict(pd) for s, pd in zip(scaler, ckpt["scaler"])]
        logger.info(f"loaded probes + optimizers from epoch {epoch}")
    else:
        logger.info(f"loaded probe weights from epoch {epoch} (optimizer state absent; restarting optimizer)")
    return heads, opt, scaler, epoch
