"""frozen encoder + predictor-only 학습 (scratch 또는 릴리즈 predictor post-fine-tuning).

    python z_training/harness/launch.py --fname <merged config>.yaml --devices cuda:0 ... cuda:7
    (사람은 z_training/train.sh 를 쓴다. 병합·검사·포트·스레드가 거기 있다. app/main.py 는 안 건드린다.)

app/vjepa/train.py 와 다른 점
  * encoder 두 개(context <- `encoder`, target <- `target_encoder`)를 체크포인트에서 읽어 **동결**한다.
    EMA 없음. 둘 다 no_grad 라 메모리는 predictor 것만 든다.
  * predictor 만 AdamW. `model.load_predictor: true` 면 릴리즈 predictor 에서 시작 (post-FT),
    false 면 새로 초기화 (scratch). 구조는 릴리즈와 같게 두어 채점기가 그대로 읽는다.
  * 마스크는 config `mask:` 리스트 (temporal_prefix / block3d, per-batch 혼합). 기본은
    surprise_c16t32 와 같은 "앞 16 프레임 -> 뒤 16 프레임".
  * 손실은 채점과 같은 식: mean |p − LN(h)|^exp / exp (fp32 에서 계산).
  * epoch 끝마다 latest.pt (predictor + opt) 저장, val 블록이 있으면 matched-pair 정확도.

체크포인트는 predictor 만 담는다 (~90 MB + Adam). encoder 는 `base_checkpoint` 경로로 가리킨다.
채점: bash z_training/eval.sh <run>   (= run.sh 에 model.predictor_checkpoint 를 넣는다)
"""
from __future__ import annotations

import gc
import json
import logging
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_frozen import utils as U
from app.vjepa_frozen.data import MaskSampler, TrainCollator, build_dataset, build_loader
from app.vjepa_frozen.val import ValSet, format_val, run_val
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

warnings.filterwarnings("ignore", message=r".*sdp_kernel.*", category=FutureWarning)
logger = get_logger(__name__, force=True)


def main(args, resume_preempt=False):
    cfg = args
    folder = cfg["folder"]
    M = cfg["meta"]
    seed = int(M.get("seed", 0))
    log_freq = int(M.get("log_freq", 10))
    save_every = int(M.get("save_every_freq", -1))
    auto_resume = bool(M.get("auto_resume", True))
    ac_dtype = U.parse_dtype(M.get("dtype", "bfloat16"))              # autocast dtype (None = fp32)
    enc_dtype = U.parse_dtype(M.get("encoder_dtype", "bfloat16")) or torch.float32
    use_scaler = ac_dtype == torch.float16
    use_sdpa = bool(M.get("use_sdpa", True))

    MD = cfg["model"]
    MD.setdefault("use_sdpa", use_sdpa)
    D = cfg["data"]
    n_frames, res = int(D["n_frames"]), int(D["resolution"])
    if res != int(MD.get("img_size", 256)):
        raise ValueError(f"data.resolution({res}) != model.img_size({MD.get('img_size')})")
    batch_size = int(D["batch_size"])
    L = cfg.get("loss") or {}
    loss_exp = float(L.get("loss_exp", 1.0))
    target_ln = bool(L.get("target_layer_norm", True))
    O = cfg["optimization"]
    V = cfg.get("val")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass
    nthreads = int(os.environ.get("OMP_NUM_THREADS", 0) or 0)
    if nthreads > 0:
        torch.set_num_threads(nthreads)

    world_size, rank = init_distributed()
    expect = os.environ.get("TRAIN_EXPECT_WS")
    if expect and int(expect) != world_size:
        raise RuntimeError(f"world_size {world_size} != TRAIN_EXPECT_WS {expect} — DDP 가 조용히 갈라졌다 (포트 충돌?)")
    logger.info(f"rank {rank}/{world_size}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # ---- 모델 ---------------------------------------------------------------------
    state = U.load_base_checkpoint(MD["checkpoint"])
    ctx_enc, tgt_enc = U.build_frozen_encoders(MD, state, device, enc_dtype, n_frames)
    predictor = U.build_predictor(MD, ctx_enc.embed_dim, state if MD.get("load_predictor") else None, device, n_frames)
    del state
    tubelet, patch = int(MD.get("tubelet_size", 2)), int(MD.get("patch_size", 16))
    spatial = (res // patch) ** 2
    mask_index = int(MD.get("mask_index", 0))
    arch = {"img_size": res, "patch_size": patch, "tubelet_size": tubelet, "num_frames": n_frames,
            "encoder_embed_dim": ctx_enc.embed_dim, **(MD.get("predictor") or {})}
    # mask token 은 매 step 하나만 쓰이므로(나머지 9개 미사용) find_unused_parameters 가 필요하다
    predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True) \
        if world_size > 1 else predictor
    pred_mod = predictor.module if hasattr(predictor, "module") else predictor

    # ---- 데이터 -------------------------------------------------------------------
    dataset = build_dataset(D, n_frames, res)
    sampler_masks = MaskSampler(cfg["mask"], n_frames, res, patch, tubelet)
    loader, sampler = build_loader(dataset, TrainCollator(sampler_masks), batch_size, rank, world_size,
                                   num_workers=int(D.get("num_workers", 8)), pin_mem=bool(D.get("pin_mem", True)),
                                   persistent=bool(D.get("persistent_workers", True)), seed=seed)
    ipe = int(O.get("ipe") or len(loader))
    if len(loader) == 0 or ipe <= 0:
        raise RuntimeError(f"loader 가 비었다 (ipe={ipe}, loader {len(loader)}): {len(dataset)} clips / ws {world_size} / "
                           f"batch {batch_size} (drop_last) — LIMIT 이나 batch_size 를 조정할 것")
    logger.info(f"dataset {len(dataset)} clips | rank batch {batch_size} x ws {world_size} = {batch_size*world_size} | "
                f"ipe {ipe} (loader {len(loader)}) | masks {[m.get('type','temporal_prefix') for m in cfg['mask']]}")

    # ---- 옵티마이저 ---------------------------------------------------------------
    num_epochs = int(O["epochs"])
    optimizer, scaler, scheduler, wd_scheduler = U.init_opt(
        pred_mod, ipe, start_lr=float(O.get("start_lr", 0.0)), ref_lr=float(O["lr"]),
        final_lr=float(O.get("final_lr", 0.0)), warmup_epochs=float(O.get("warmup", 0)), num_epochs=num_epochs,
        wd=float(O.get("weight_decay", 0.04)), final_wd=float(O.get("final_weight_decay", O.get("weight_decay", 0.04))),
        ipe_scale=float(O.get("ipe_scale", 1.0)), betas=tuple(O.get("betas", (0.9, 0.999))),
        eps=float(O.get("eps", 1e-8)), use_scaler=use_scaler)
    clip_grad = O.get("clip_grad")

    # ---- val ----------------------------------------------------------------------
    valset = None
    if V and V.get("data"):
        valset = ValSet(V, n_frames, res)
        if rank == 0:
            logger.info(f"val: {len(valset)} clips from {V['data']['root']} (n_blocks_per_type {V.get('n_blocks_per_type', 8)})")

    # ---- resume -------------------------------------------------------------------
    latest = os.path.join(folder, "latest.pt")
    start_epoch, step = 0, 0
    if (auto_resume or resume_preempt) and os.path.exists(latest):
        start_epoch, step = U.load_resume(latest, pred_mod, optimizer, scaler)
        for _ in range(start_epoch * ipe):
            scheduler.step(); wd_scheduler.step()
    elif os.path.exists(latest):
        raise RuntimeError(f"{latest} 가 있는데 meta.auto_resume 이 false 다. 새 NAME 을 쓰거나 폴더를 비울 것")

    csv_logger = CSVLogger(os.path.join(folder, f"log_r{rank}.csv"), ("%d", "epoch"), ("%d", "itr"),
                           ("%.5f", "loss"), ("%d", "ctx_frames"), ("%.2e", "lr"),
                           ("%d", "iter-time(ms)"), ("%d", "gpu-time(ms)"), ("%d", "data-time(ms)"))
    metrics_path = os.path.join(folder, "metrics.jsonl")

    def _ac():
        return torch.autocast("cuda", dtype=ac_dtype) if ac_dtype else torch.autocast("cuda", enabled=False)

    def _val(epoch):
        if valset is None:
            return {}
        res = run_val(valset, ctx_enc, tgt_enc, predictor, device, ac_dtype, rank, world_size, tubelet, spatial,
                      int(V.get("context_length", 16)), int(V.get("batch_size", 8)), mask_index, loss_exp, target_ln)
        if rank == 0:
            logger.info(f"[epoch {epoch}] {format_val(res)}")
        return res

    if rank == 0:
        logger.info(f"predictor trainable {U.count_parameters(pred_mod)/1e6:.2f}M | load_predictor={bool(MD.get('load_predictor'))} "
                    f"| autocast={ac_dtype} enc_dtype={enc_dtype} | lr {O['lr']} wd {O.get('weight_decay')} epochs {num_epochs}")
    # ★ 모든 rank 가 들어가야 한다 — run_val 안의 all_gather_object 는 collective 라 rank 0 만 부르면 나머지가 영원히 기다린다
    if start_epoch == 0 and V and V.get("at_start", True):
        _val(0)

    # ---- 학습 루프 ----------------------------------------------------------------
    sampler.set_epoch(start_epoch)
    loader_it = iter(loader)
    lr_now = wd_now = float("nan")
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        loss_meter, it_meter, gpu_meter, data_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        for itr in range(ipe):
            t_it = time.time()
            try:
                clips, m_enc, m_pred, info = next(loader_it)
            except StopIteration:
                sampler.set_epoch(epoch + 1)
                loader_it = iter(loader)
                clips, m_enc, m_pred, info = next(loader_it)
            clips = clips.to(device, non_blocking=True)
            m_enc = m_enc.to(device, non_blocking=True)
            m_pred = m_pred.to(device, non_blocking=True)
            data_ms = (time.time() - t_it) * 1000

            def train_step():
                lr_now = scheduler.step()
                wd_now = wd_scheduler.step()
                with torch.no_grad(), _ac():
                    xe = clips.to(enc_dtype)
                    h = tgt_enc(xe)                                                   # (B, N, D)
                    h = torch.gather(h, 1, m_pred.unsqueeze(-1).expand(-1, -1, h.size(-1)))
                    if target_ln:
                        h = torch.nn.functional.layer_norm(h, (h.size(-1),))       # affine-free, target 에만
                    z = ctx_enc(xe, masks=[m_enc])                                    # context 토큰만 (B, K, D)
                with _ac():
                    p = predictor(z, m_enc, m_pred, mask_index=mask_index)            # (B, M, D)
                loss = (p.float() - h.float()).abs().pow(loss_exp).mean() / loss_exp
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(pred_mod.parameters(), float(clip_grad))
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                return float(loss.detach()), lr_now, wd_now

            (loss, lr_now, wd_now), gpu_ms = gpu_timer(train_step)
            step += 1
            it_ms = (time.time() - t_it) * 1000
            loss_meter.update(loss); it_meter.update(it_ms); gpu_meter.update(gpu_ms); data_meter.update(data_ms)
            csv_logger.log(epoch + 1, itr, loss, info["context_frames"], lr_now, it_ms, gpu_ms, data_ms)
            if (itr % log_freq == 0) or (itr == ipe - 1) or not np.isfinite(loss):
                logger.info("[%d, %5d] loss %.4f (avg %.4f) | mask %s C=%s | lr %.2e wd %.2e | mem %.1fG | iter %.0f ms gpu %.0f ms data %.0f ms"
                            % (epoch + 1, itr, loss, loss_meter.avg, info["type"], info["context_frames"], lr_now, wd_now,
                               torch.cuda.max_memory_allocated() / 1024**3, it_meter.avg, gpu_meter.avg, data_meter.avg))
            if not np.isfinite(loss):
                raise RuntimeError("loss 가 nan/inf")
            if M.get("sync_gc") and (itr + 1) % 50 == 0:
                gc.collect()

        logger.info(f"epoch {epoch + 1} avg loss {loss_meter.avg:.4f}")
        if rank == 0:
            U.save_checkpoint(latest, predictor, optimizer, scaler, epoch + 1, step, loss_meter.avg, cfg, arch)
            if save_every > 0 and ((epoch + 1) % save_every == 0 or epoch + 1 == num_epochs):
                U.save_checkpoint(os.path.join(folder, f"e{epoch + 1}.pt"), predictor, optimizer, scaler,
                                  epoch + 1, step, loss_meter.avg, cfg, arch)
        res = {}
        if V and int(V.get("every_epochs", 1)) > 0 and ((epoch + 1) % int(V.get("every_epochs", 1)) == 0 or epoch + 1 == num_epochs):
            res = _val(epoch + 1)
        if rank == 0:
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"epoch": epoch + 1, "step": step, "train_loss": loss_meter.avg, "lr": lr_now,
                                    "val": res, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    logger.info("done")
