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
            pname = probe_name(spec)
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
                from evals.analysis_vlm.cache import PooledLinearProbe

                module = PooledLinearProbe(
                    embed_dim=embed_dims[stage_pos], num_classes=num_classes,
                    pooling=pooling, pre_norm=spec.get("pre_norm", True),
                ).to(device)
            elif framewise:
                # temporal-preserving linear: spatial-pool within each frame, concat frames, Linear.
                # Better than global mean for direction (keeps 'up' vs 'down').
                num_temporal = getattr(encoder, "num_temporal", None)
                if num_temporal is None:
                    raise ValueError(
                        f"pooling='{pooling}' (temporal-preserving) needs encoder.num_temporal (VLM backends only)."
                    )
                from evals.analysis_vlm.probes import TemporalLinearProbe

                module = TemporalLinearProbe(
                    embed_dim=embed_dims[stage_pos], num_temporal=num_temporal, num_classes=num_classes,
                    spatial_pool=pooling.split("_", 1)[1], pre_norm=spec.get("pre_norm", True),
                ).to(device)
            elif ptype == "attentive" and tpos in ("learnable", "rope"):
                # temporal positional encoding for the attentive pooler (e.g. LLaVA per-frame SigLIP)
                num_temporal = getattr(encoder, "num_temporal", None)
                if num_temporal is None:
                    raise ValueError(
                        f"temporal_pos='{tpos}' needs encoder.num_temporal (VLM backends only); "
                        f"this encoder does not expose it (V-JEPA already encodes time via RoPE)."
                    )
                from evals.analysis_vlm.probes import TemporalAttentiveClassifier

                module = TemporalAttentiveClassifier(
                    embed_dim=embed_dims[stage_pos], num_temporal=num_temporal, mode=tpos,
                    num_heads=spec.get("num_heads", 16),
                    depth=spec.get("num_probe_blocks", spec.get("depth", 1)),
                    num_classes=num_classes, use_activation_checkpointing=True,
                ).to(device)
                pname = f"{pname}-{tpos}"
            else:
                module = build_probe(
                    spec, embed_dim=embed_dims[stage_pos], num_classes=num_classes,
                    use_activation_checkpointing=True,
                ).to(device)
            if use_ddp:
                module = DistributedDataParallel(module, static_graph=True)
            name = f"{stage_tag}_{pname}"
            if name in _name_set:  # probe_name ignores num_heads/pre_norm -> de-collide duplicate specs
                k = 2
                while f"{name}#{k}" in _name_set:
                    k += 1
                name = f"{name}#{k}"
            _name_set.add(name)
            # carry probe label + stage name explicitly so plotting doesn't re-parse `name`
            # (VLM stage tags contain underscores, which breaks name.split("_", 1)).
            heads.append(dict(name=name, layer=layer_val, layer_pos=stage_pos,
                              probe=pname, stage=stage_tag, module=module))
            head_opt_kwargs.append(_opt_kwargs(spec))
    head_names = [h["name"] for h in heads]
    logger.info(f"Built {len(heads)} probe heads over {len(stages)} stages: {head_names}")

    # -- data (clip vs raw), with optional one-time FEATURE CACHE
    run_mode = data_mode

    def _split_loader(root, training):
        """A loader over a single split. training=False => deterministic (no augmentation)."""
        if data_mode == "clip":
            from evals.video_classification_frozen.eval import make_dataloader

            ld, samp = make_dataloader(
                dataset_type=dataset_type, root_path=[root], img_size=resolution,
                frames_per_clip=frames_per_clip, frame_step=frame_step, eval_duration=duration,
                num_segments=num_segments, num_views_per_segment=1, allow_segment_overlap=True,
                batch_size=batch_size, world_size=world_size, rank=rank, training=training,
                num_workers=num_workers, normalization=normalization,
            )
            return ld, samp
        from evals.analysis_vlm.data import make_raw_dataloader

        return make_raw_dataloader(root, frames_per_clip, batch_size, world_size, rank,
                                   training=training, num_workers=num_workers)

    if cache_features:
        # ONE deterministic pre-pass per split (training=False -> NO augmentation), cache features,
        # then train probes over the cache. Each rank caches its own shard.
        from evals.analysis_vlm.cache import build_feature_cache, make_cached_loader

        def encode_fn(d):
            return _encode(encoder, d, device, data_mode, use_bfloat16)

        logger.info(f"cache_features=true (pooling={cache_pooling}): one deterministic pre-pass per split...")
        tr_loader, _ = _split_loader(train_data_path[0], training=False)
        va_loader, _ = _split_loader(val_data_path[0], training=False)
        enc_num_temporal = getattr(encoder, "num_temporal", None)  # for cache_pooling='framewise'
        tr_feats, tr_labels = build_feature_cache(encode_fn, tr_loader, cache_pooling,
                                                  num_temporal=enc_num_temporal, max_gb=cache_max_gb,
                                                  label="train-cache")
        va_feats, va_labels = build_feature_cache(encode_fn, va_loader, cache_pooling,
                                                  num_temporal=enc_num_temporal, max_gb=cache_max_gb,
                                                  label="val-cache")
        del tr_loader, va_loader
        train_loader = make_cached_loader(tr_feats, tr_labels, batch_size, training=True)
        val_loader = make_cached_loader(va_feats, va_labels, batch_size, training=False)
        train_sampler = None
        run_mode = "cached"
    else:
        train_loader, train_sampler = _split_loader(train_data_path[0], training=True)
        val_loader, _ = _split_loader(val_data_path[0], training=False)

    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe} (mode={run_mode})")

    # -- optimizers (one per head)
    from evals.video_classification_frozen.eval import init_opt

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
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
    best_val = {n: -1.0 for n in head_names}
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
            )

        val_acc = run_one_epoch(
            device=device, training=False, encoder=encoder, heads=heads, scaler=scaler,
            optimizer=optimizer, scheduler=scheduler, wd_scheduler=wd_scheduler,
            data_loader=val_loader, use_bfloat16=use_bfloat16, data_mode=run_mode, rank=rank,
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
                           "data_mode": data_mode, "stages": [str(s) for s in stages],
                           "head_names": head_names, "val_acc": val_acc, "train_acc": train_acc,
                           "best_val_acc": best_val}, f, indent=2)

        save_checkpoint(epoch + 1)
        if val_only:
            break

    if rank == 0 and make_plot:
        from evals.analysis.plotting import plot_layer_val_acc

        plot_layer_val_acc(heads, best_val, os.path.join(folder, "stage_val_acc.png"),
                           subtitle=f"{model_sel} | best val over {last_epoch} epoch(s)")


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
    feats = [f.detach().float() for f in feats]
    return feats, labels, labels.size(0)


def run_one_epoch(device, training, encoder, heads, scaler, optimizer, scheduler,
                  wd_scheduler, data_loader, use_bfloat16, data_mode, rank=0):
    for h in heads:
        h["module"].train(mode=training)

    criterion = torch.nn.CrossEntropyLoss()
    n_heads = len(heads)
    correct = torch.zeros(n_heads, device=device)
    total = torch.zeros((), device=device)
    amp_scaler = scaler[0]

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

        # In validation, run the heads under no_grad: no autograd graph is built and the
        # DDP reducer is NOT armed (a grad-enabled DDP forward with no backward trips the
        # static_graph reducer on the next forward under multi-GPU).
        grad_ctx = nullcontext() if training else torch.no_grad()
        with grad_ctx, torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_bfloat16):
            logits = [h["module"](feats[h["layer_pos"]]) for h in heads]
            losses = [criterion(o, labels) for o in logits] if training else None

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
            for hi, o in enumerate(logits):
                correct[hi] += (o.argmax(dim=1) == labels).sum()

        if rank == 0 and hasattr(iterator, "set_postfix") and (itr % 20 == 0):
            iterator.set_postfix(best=f"{(100.0 * correct.max() / total.clamp(min=1)).item():.1f}%")

    correct = AllReduceSum.apply(correct)
    total = AllReduceSum.apply(total)
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
