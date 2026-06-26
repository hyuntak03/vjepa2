# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
# Layer-wise probing analysis for a frozen V-JEPA encoder.
#
# Reuses the exact data / encoder / optimizer machinery of
# evals.video_classification_frozen, but:
#   * extracts features from MULTIPLE encoder layers (config: analysis.layers)
#   * attaches one probe per (layer x probe-spec); probe type is config-driven
#     (config: analysis.probes -> linear | attentive | ...)
#   * reports a [layer x probe] accuracy matrix.
#
# Routed automatically by evals/scaffold.py when a config sets
#   eval_name: analysis
# so neither main.py nor scaffold.py need any change.
# -----------------------------------------------------------------------------

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import json
import logging
import pprint

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

# -- reuse the stock machinery (import-only; existing code is not modified) --
from evals.video_classification_frozen.eval import (
    DEFAULT_NORMALIZATION,
    init_opt,
    make_dataloader,
)
from evals.video_classification_frozen.models import init_module

from evals.analysis.probes import build_probe, probe_name
from evals.analysis.plotting import plot_layer_val_acc
from tqdm import tqdm

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
# -- speed: allow TF32 on Ampere+ (RTX 3090/4090) for matmul/conv. negligible acc impact.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

pp = pprint.PrettyPrinter(indent=4)

# depth lookup so analysis.layers can be the literal string "all"
_VIT_DEPTH = {
    "vit_large": 24,
    "vit_huge": 32,
    "vit_giant": 40,
    "vit_giant_xformers": 40,
    "vit_gigantic": 48,
}


def _resolve_layers(layers_cfg, model_name):
    if isinstance(layers_cfg, str) and layers_cfg.lower() == "all":
        if model_name not in _VIT_DEPTH:
            raise ValueError(f"analysis.layers='all' but depth of {model_name} unknown; give an explicit list")
        return list(range(_VIT_DEPTH[model_name]))
    assert isinstance(layers_cfg, (list, tuple)) and len(layers_cfg) > 0, "analysis.layers must be a non-empty list or 'all'"
    return [int(x) for x in layers_cfg]


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  CONFIG
    # ----------------------------------------------------------------------- #
    val_only = args_eval.get("val_only", False)
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")
    enc_model_name = args_model["encoder"].get("model_name")

    args_exp = args_eval.get("experiment")

    # -- ANALYSIS (new) -- which layers, which probes
    args_analysis = args_exp.get("analysis")
    layers = _resolve_layers(args_analysis.get("layers"), enc_model_name)
    probe_specs = args_analysis.get("probes")
    assert isinstance(probe_specs, list) and len(probe_specs) > 0, "experiment.analysis.probes must be a non-empty list"
    make_plot = args_analysis.get("plot", False)  # save [layer x val-acc] png each epoch
    # encoder must expose exactly these layers
    args_wrapper = dict(args_wrapper or {})
    args_wrapper["out_layers"] = layers

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

    # -- OPTIMIZATION (a default; each probe spec may override via spec['optimization'])
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    default_opt = args_opt.get("default_head", {})

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

    # ----------------------------------------------------------------------- #
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    folder = os.path.join(pretrain_folder, "analysis/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    # -- encoder (frozen), exposing the requested layers
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    # -- heads: one probe per (layer x probe-spec) --------------------------- #
    # single-GPU local debugging (no process group) -> skip DDP gracefully
    use_ddp = dist.is_available() and dist.is_initialized()
    if not use_ddp:
        logger.info("No distributed process group -> running probes without DDP (single-process).")

    heads = []  # each: dict(name, layer, layer_pos, module)
    head_opt_kwargs = []
    for layer_pos, layer in enumerate(layers):
        for spec in probe_specs:
            module = build_probe(spec, embed_dim=encoder.embed_dim, num_classes=num_classes,
                                 use_activation_checkpointing=True).to(device)
            if use_ddp:
                module = DistributedDataParallel(module, static_graph=True)
            name = f"L{layer:02d}_{probe_name(spec)}"
            heads.append(dict(name=name, layer=layer, layer_pos=layer_pos, module=module))
            head_opt_kwargs.append(_opt_kwargs(spec))
    head_names = [h["name"] for h in heads]
    logger.info(f"Built {len(heads)} probe heads: {head_names}")

    # -- data
    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type, root_path=train_data_path, img_size=resolution,
        frames_per_clip=frames_per_clip, frame_step=frame_step, eval_duration=duration,
        num_segments=num_segments, num_views_per_segment=1, allow_segment_overlap=True,
        batch_size=batch_size, world_size=world_size, rank=rank, training=True,
        num_workers=num_workers, normalization=normalization,
    )
    val_loader, _ = make_dataloader(
        dataset_type=dataset_type, root_path=val_data_path, img_size=resolution,
        frames_per_clip=frames_per_clip, frame_step=frame_step, num_segments=num_segments,
        eval_duration=duration, num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True, batch_size=batch_size, world_size=world_size, rank=rank,
        training=False, num_workers=num_workers, normalization=normalization,
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # -- optimizers (one per head)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=[h["module"] for h in heads],
        opt_kwargs=head_opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # -- csv logger (epoch + train/val per head)
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
        torch.save({
            "classifiers": [h["module"].state_dict() for h in heads],
            "opt": [o.state_dict() for o in optimizer],
            "scaler": None if scaler[0] is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "head_names": head_names,
            "layers": layers,
            "batch_size": batch_size,
            "world_size": world_size,
        }, latest_path)

    # ----------------------------------------------------------------------- #
    #  TRAIN / EVAL LOOP
    # ----------------------------------------------------------------------- #
    best_val = {n: -1.0 for n in head_names}  # best val acc per head over all epochs
    last_epoch = start_epoch
    for epoch in range(start_epoch, num_epochs):
        last_epoch = epoch + 1
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)

        if val_only:
            train_acc = {n: -1.0 for n in head_names}
        else:
            train_acc = run_one_epoch(
                device=device, training=True, encoder=encoder, heads=heads,
                scaler=scaler, optimizer=optimizer, scheduler=scheduler,
                wd_scheduler=wd_scheduler, data_loader=train_loader, use_bfloat16=use_bfloat16,
                rank=rank,
            )

        val_acc = run_one_epoch(
            device=device, training=False, encoder=encoder, heads=heads,
            scaler=scaler, optimizer=optimizer, scheduler=scheduler,
            wd_scheduler=wd_scheduler, data_loader=val_loader, use_bfloat16=use_bfloat16,
            rank=rank,
        )
        for n in head_names:
            best_val[n] = max(best_val[n], val_acc[n])

        # report (current epoch + running best)
        logger.info("[epoch %d] layer x probe val acc (current | best):" % (epoch + 1))
        for h in heads:
            n = h["name"]
            logger.info("    %-28s  train %6.2f%%  val %6.2f%%  best %6.2f%%"
                        % (n, train_acc[n], val_acc[n], best_val[n]))
        if rank == 0:
            row = [epoch + 1]
            for n in head_names:
                row += [train_acc[n], val_acc[n]]
            csv_logger.log(*row)
            # dump a human-readable matrix (current epoch + best-so-far)
            with open(os.path.join(folder, "summary.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "num_epochs": num_epochs, "layers": layers,
                           "head_names": head_names, "val_acc": val_acc, "train_acc": train_acc,
                           "best_val_acc": best_val}, f, indent=2)

        save_checkpoint(epoch + 1)
        if val_only:
            break

    # ----- training finished: save the final [layer x val-acc] plot once ----- #
    if rank == 0 and make_plot:
        plot_layer_val_acc(
            heads, best_val, os.path.join(folder, "layer_val_acc.png"),
            subtitle=f"best val over {last_epoch} epoch(s)",
        )


def run_one_epoch(device, training, encoder, heads, scaler, optimizer, scheduler,
                  wd_scheduler, data_loader, use_bfloat16, rank=0):
    for h in heads:
        h["module"].train(mode=training)

    criterion = torch.nn.CrossEntropyLoss()
    n_heads = len(heads)
    # accumulate correct/total on-GPU -> only ONE host sync per epoch (no per-iter .item())
    correct = torch.zeros(n_heads, device=device)
    total = torch.zeros((), device=device)
    amp_scaler = scaler[0]  # single shared GradScaler (init_opt makes identical ones)

    iterator = data_loader
    if rank == 0:
        iterator = tqdm(data_loader, desc=("train" if training else "  val"),
                        dynamic_ncols=True, mininterval=2.0, leave=False)

    for itr, data in enumerate(iterator):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_bfloat16):
            clips = [[dij.to(device, non_blocking=True) for dij in di] for di in data[0]]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device, non_blocking=True)
            bsz = labels.size(0)

            with torch.no_grad():
                feats = encoder(clips, clip_indices)  # list over layers, each (B, N, D)
            feats = [f.detach() for f in feats]

            logits = [h["module"](feats[h["layer_pos"]]) for h in heads]
            losses = [criterion(o, labels) for o in logits]

        # one combined backward (feats are detached -> heads are independent), then step each opt
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

        # accuracy accumulation stays on GPU (no host sync here)
        with torch.no_grad():
            total += bsz
            for hi, o in enumerate(logits):
                correct[hi] += (o.argmax(dim=1) == labels).sum()

        if rank == 0 and (itr % 20 == 0):
            iterator.set_postfix(best=f"{(100.0 * correct.max() / total.clamp(min=1)).item():.1f}%")

    # single all-reduce + single host sync at the end of the epoch
    correct = AllReduceSum.apply(correct)
    total = AllReduceSum.apply(total)
    accs = (100.0 * correct / total.clamp(min=1)).tolist()
    return {h["name"]: accs[hi] for hi, h in enumerate(heads)}


def load_checkpoint(device, r_path, heads, opt, scaler, val_only=False):
    ckpt = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")
    for h, pd in zip(heads, ckpt["classifiers"]):
        h["module"].load_state_dict(pd)
    if val_only:
        return heads, opt, scaler, 0
    epoch = ckpt["epoch"]
    [o.load_state_dict(pd) for o, pd in zip(opt, ckpt["opt"])]
    if scaler[0] is not None and ckpt.get("scaler") is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, ckpt["scaler"])]
    logger.info(f"loaded probes + optimizers from epoch {epoch}")
    return heads, opt, scaler, epoch
