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
from app.vjepa_frozen.data import ClipTransform, MaskSampler, TrainCollator, WindowGridCollator, WindowGridSampler, build_dataset, build_loader, normalize_clips
from app.vjepa_frozen.pusht_data import ACTION_TYPES, PushTCollator, PushTDataset, make_traj_dataset
from app.vjepa_frozen.val import ValSet, format_val, run_val
from app.vjepa_frozen import ar as AR
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
    rollout_steps = int(L.get("rollout_steps", 8))                        # 팔 C: R = min(K, rollout_steps)
    tf_weight = float(L.get("tf_weight", 1.0))                            # ctx_state: 0 이면 TF 패스(+ 블록별 관측 인코딩) 생략
    kind = str((MD.get("predictor") or {}).get("kind", "oneshot"))
    is_ar = kind in ("ar", "ar_state")   # 팔 C (app/vjepa_frozen/ar.py): 문맥·미래 블록별 인코딩 (AC 판). ar_state = 상태 되먹임 (2026-10-10)
    is_ctx_ar = kind == "ctx_ar"    # 팔 C' (2026-09-27): 문맥은 창 단위 z, 미래 GT 만 블록별 LN(h)
    is_ctx_state = kind == "ctx_state"  # vast 이식 (2026-10-16): 문맥 = 창 단위 z, 타깃 = 창 전체 LN(h), rollout 되먹임 = predictor 상태
    is_ctx = is_ctx_ar or is_ctx_state  # "문맥을 video 로" 계열 — context encoder 를 쓴다
    is_seq = is_ar or is_ctx           # tf + rollout 두 손실을 쓰는 팔 전부
    # ctx_state 타깃 (vast 2026-10-16 ctxt_video 이식): window (기본, 창 전체 LN(h) = 릴리즈 채점 정의)
    #   | block (블록별 LN(h) = TF 관측과 같은 공간, 창 전체 forward 생략)
    ctx_target = str((MD.get("predictor") or {}).get("ctx_target", "window"))
    if ctx_target not in ("window", "block"):
        raise ValueError(f"model.predictor.ctx_target 는 window | block: {ctx_target!r}")
    if ctx_target != "window" and not is_ctx_state:
        raise ValueError("model.predictor.ctx_target 는 kind=ctx_state 전용")
    is_pusht = any((s.get("type") in ACTION_TYPES) for s in D.get("datasets") or [])   # action 조건부 궤적 (Push-T 2026-10-09, Wall 2026-10-11)
    if is_pusht and not is_ar:
        raise ValueError("pusht 데이터는 kind=ar (AC 판) 로만 돈다")
    VP = cfg.get("val_pusht") if is_pusht else None
    if is_seq and not target_ln:
        raise ValueError(f"kind={kind} 는 loss.target_layer_norm=true 가 전제다 (입력·타깃·출력이 모두 LN 공간)")
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
    # ── CPU affinity 를 부모(launch.py) 의 것으로 되돌린다 (2026-09-23) ─────────────────────
    #    SLURM job 430837 (6 GPU) 에서 rank 0·1 이 8 코어(22-25,86-89), rank 2~5 가 64 코어에 묶여
    #    rank 0·1 의 loader 가 1.2~3.4 s 로 병목이 됐다 (다른 rank 는 0.1 s). 레포 코드에는 affinity 를
    #    건드리는 곳이 없고 부모 프로세스는 job cpuset 전체를 갖고 있었다. DataLoader worker 는 여기서
    #    상속하므로 loader 를 만들기 전에 넓혀 둔다. 실패해도 학습엔 영향 없으니 조용히 넘어간다.
    #    ⚠️ 2026-09-24 정정: 좁아지는 시점은 프로세스 시작이 아니라 **CUDA/NCCL 초기화 뒤**다 (job 431419 에서
    #    시작 시점엔 부모와 같아 아무것도 안 했는데 loader 를 만들 때는 rank 0·1 이 8 코어였다 — GPU 의
    #    NUMA 노드 ∩ job cpuset 으로 묶인 모양). 그래서 init_distributed 뒤와 loader 생성 직전에 다시 건다.
    def _restore_affinity(where):
        try:
            _par = os.sched_getaffinity(os.getppid()); _me = os.sched_getaffinity(0)
            if _par and _par != _me:
                os.sched_setaffinity(0, _par)
                logger.info(f"cpu affinity {len(_me)} -> {len(_par)} cores (부모 mask 로 복원, {where})")
        except Exception:
            pass
    _restore_affinity("start")

    world_size, rank = init_distributed()
    _restore_affinity('after init_distributed')
    expect = os.environ.get("TRAIN_EXPECT_WS")
    if expect and int(expect) != world_size:
        raise RuntimeError(f"world_size {world_size} != TRAIN_EXPECT_WS {expect} — DDP 가 조용히 갈라졌다 (포트 충돌?)")
    logger.info(f"rank {rank}/{world_size}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # ---- 모델 ---------------------------------------------------------------------
    state = U.load_base_checkpoint(MD["checkpoint"])
    ctx_enc, tgt_enc = U.build_frozen_encoders(MD, state, device, enc_dtype, n_frames)
    # meta.compile_encoders (Ariel 이식, 기본 false): 동결 encoder torch.compile (실측 1.8x, bf16 잡음 수준 오차)
    if bool(M.get("compile_encoders", False)):
        _shared = ctx_enc is tgt_enc              # 2026-09-27: compile 전에 판정 (뒤에 하면 늘 False — 같은 encoder 를 두 번 compile)
        tgt_enc = torch.compile(tgt_enc, mode="default", dynamic=True)
        ctx_enc = tgt_enc if _shared else torch.compile(ctx_enc, mode="default", dynamic=True)
        logger.info("encoders: torch.compile(mode=default, dynamic=True)")
    # ── model.encoder_lora (2026-10-18): context encoder (online `encoder`) 에 LoRA. target encoder 는 얼린 채 (app/vjepa_frozen/lora.py)
    #    {rank: 8, alpha: 16, targets: [qkv, proj, fc1, fc2], lr_mult: 10, act_ckpt: true}. 기본 train_step (release predictor post-FT) 에서만.
    LORA = MD.get("encoder_lora") or None
    lora_params = []
    if LORA:
        from app.vjepa_frozen import lora as LR
        if ctx_enc is tgt_enc:
            raise ValueError("encoder_lora 는 context · target encoder 가 다른 키여야 한다 (target 을 얼린 채 두려고)")
        if is_ar or is_ctx_ar or is_ctx_state:
            raise ValueError("encoder_lora 는 기본 train_step (release predictor post-FT 계열) 에만 구현돼 있다")
        lora_params = LR.apply_lora(ctx_enc, int(LORA.get("rank", 8)), float(LORA.get("alpha", 16)), LORA.get("targets", LR.DEFAULT_TARGETS))
        ctx_enc.use_activation_checkpointing = bool(LORA.get("act_ckpt", True))
        logger.info(f"encoder LoRA on context encoder: {ctx_enc.lora_info} · act_ckpt {ctx_enc.use_activation_checkpointing} · lr_mult {LORA.get('lr_mult', 10)}")
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
    # ── state_head (2026-10-19, 사용자 결정): phys2d_sup 상태 라벨로 AR 출력 (TF · rollout, 1280) 에 보조 손실.
    #    정체 쿼리 48 개 attention pooling → 존재 · 위치 · 속도 head (app/vjepa_frozen/state_head.py). predictor state_dict 밖 —
    #    체크포인트에는 `state_head` 로 따로. DDP 는 predictor 만 감쌌으므로 head grad 는 직접 평균한다 (LoRA 와 같다).
    SH = cfg.get("state_head") or None
    state_head, head_params = None, []
    if SH:
        if not is_ar:
            raise ValueError("state_head 는 kind=ar | ar_state 에서만 구현돼 있다")
        from app.vjepa_frozen import state_head as SHM
        state_head = SHM.build_head(SH, ctx_enc.embed_dim).to(device)       # type: attn (기본, 755 만) | light (6.9 만, vast 36f4921)
        head_params = list(state_head.parameters())
        logger.info(f"state_head ({SH.get('type', 'attn')}): {sum(q.numel() for q in head_params) / 1e6:.2f}M params · "
                    f"weight {SH.get('weight', 1.0)} · w {SH.get('w', {})} · split {SH.get('split', {})} · lr_mult {SH.get('lr_mult', 10)}")
    # ── flow_head (2026-10-22, 사용자 요청): 자연 영상의 optical flow 로 AR 출력에 보조 손실.
    #    라벨은 **관측만으로** 얻는다 (phys2d 상태 라벨과 다른 점 — CLAUDE.md §12 Q6). app/vjepa_frozen/flow_head.py
    FH = cfg.get("flow_head") or None
    flow_head, FHM = None, None
    if FH:
        if not is_ar:
            raise ValueError("flow_head 는 kind=ar | ar_state 에서만 구현돼 있다")
        if not ((D.get("var_window") or {}).get("flow")):
            raise ValueError("flow_head 를 켜려면 data.var_window.flow (캐시 경로) 도 있어야 한다")
        from app.vjepa_frozen import flow_head as FHM
        # ⚠️ 지평선은 **데이터 쪽 한 곳** (data.var_window.flow.horizons) 에만 적는다 — 라벨을 만드는 쪽이 거기다.
        #    head 쪽에 또 적으면 둘이 어긋나 head 출력 차원과 라벨 축이 안 맞는다 (2026-10-22 실제로 그랬다).
        FHZ = [int(h) for h in (D.get("var_window") or {}).get("flow", {}).get("horizons", [1, 2, 4, 8])]
        FSUP = [int(h) for h in (FH.get("supervise") or FHZ[1:] or FHZ)]     # 첫 지평선은 등속 기준선 재료 (기본 비감독)
        if any(h not in FHZ for h in FSUP):
            raise ValueError(f"flow_head.supervise {FSUP} 는 data.var_window.flow.horizons {FHZ} 의 부분집합이어야 한다")
        flow_head = FHM.build_head({**FH, "supervise": FSUP}, ctx_enc.embed_dim).to(device)
        head_params = head_params + list(flow_head.parameters())
        logger.info(f"flow_head: {sum(q.numel() for q in flow_head.parameters()) / 1e3:.1f}k params · "
                    f"지평선 라벨 {FHZ} 장 · 감독 {FSUP} 장 (= {[h / 2 for h in FSUP]} 블록) · "
                    f"weight {FH.get('weight', 1.0)} · split {FH.get('split', {})} · lr_mult {FH.get('lr_mult', 10)}")
    HCFG = SH or FH or {}        # head 공통 노브 (lr_mult · clip_separate · clip_head · freeze_predictor)

    # ---- 데이터 -------------------------------------------------------------------
    # 샘플링 두 갈래 (2026-10-16)
    #   고정 창 (기본, data.datasets)  : build_dataset + MaskSampler. data.n_frames 장을 linspace 로 뽑고 (C,K) 를 목록에서 고른다.
    #                                    창 길이가 늘 같아 batch 가 고정이다. ctx 팔 (문맥 = video) 에 적합 — encoder 가 창을 통째로 돈다
    #   가변 창 (data.var_window)      : app/vjepa_frozen/var_window.py. step 마다 (C, K) 와 창 W 를 바꾸고 batch 를 토큰 예산으로 맞춘다.
    #                                    fps 4 정수 간격 · 패딩 없음. ar 팔 (블록별 인코딩, 창 길이에 선형) 에 적합
    VW = D.get("var_window")
    dataset = None if VW else build_dataset(D, n_frames, res)
    # meta.sync_masks (Ariel 이식, 기본 false): 마스크를 학습 루프가 step 당 한 번, seed 로 rank 동기로 뽑고
    # 창을 C+K 블록으로 자른다. false = 기존 경로 (worker 안 collate 에서 뽑음).
    sync_masks = bool(M.get("sync_masks", False)) and not D.get("window_grid")
    sampler_masks = MaskSampler(cfg["mask"], n_frames, res, patch, tubelet, seed=seed if sync_masks else None)
    pusht_ctx_choices = [int(c) for c in (cfg["mask"][0].get("context_blocks") or [n_frames // 4])] if is_pusht else None
    if VW and (D.get("window_grid") or is_pusht):
        raise ValueError("data.var_window 는 window_grid · pusht 와 같이 못 쓴다")
    if VW and not is_seq:
        raise ValueError(f"data.var_window 는 kind=ar | ar_state | ctx_ar | ctx_state 전용이다 (지금 {kind})")
    if VW and not O.get("ipe"):
        raise ValueError("data.var_window 는 optimization.ipe 가 필요하다 (epoch 길이를 데이터 크기로 정할 수 없다)")
    if D.get("window_grid"):
        # 채점 그리드(skip × window × 시작점 × C)에서 배치마다 하나를 뽑는다. data.n_frames 는 최대 창(RoPE 상한)
        grid = WindowGridSampler(D["window_grid"], tubelet)
        if max(c["n_frames"] for c in grid.cells) > n_frames:
            raise ValueError(f"window_grid 의 n_frames 최대가 data.n_frames({n_frames}) 보다 크다")
        collator = WindowGridCollator(grid, sampler_masks, dataset.transform)
        logger.info(f"window_grid: {grid.describe()} | raw_span {grid.raw_span} jitter {grid.jitter}")
    elif is_pusht:
        collator = PushTCollator()              # (clips, None, None, {actions, proprio, state, ...})
    else:
        collator = TrainCollator(None if sync_masks else sampler_masks)
    if is_ar and not sync_masks and not VW:
        # 팔 C 는 step 당 (C, K) 를 알아야 하고 (info['context_blocks']) 창을 C+K 블록으로 잘라야 한다 (ar_losses 가 nb == C+K 검사).
        # 우리 레포는 그 경로가 meta.sync_masks 뒤에 있다 (기본 false). window_grid 와는 같이 못 쓴다.
        raise ValueError("kind=ar 는 meta.sync_masks: true 가 필요하다 (data.window_grid 와 같이 못 쓴다)")
    _restore_affinity('before build_loader')   # DataLoader worker 가 여기서 mask 를 상속한다
    if VW:
        loader, sampler = None, None            # 로더는 resume 뒤에 만든다 (start_step 이 필요하다)
    else:
        loader, sampler = build_loader(dataset, collator, batch_size, rank, world_size,
                                       num_workers=int(D.get("num_workers", 8)), pin_mem=bool(D.get("pin_mem", True)),
                                       persistent=bool(D.get("persistent_workers", True)), seed=seed,
                                       prefetch_factor=int(D.get("prefetch_factor", 2)),
                                       timeout_s=float(D.get("loader_timeout_s", 0)))
    # ── gradient accumulation (2026-10-10): optimization.grad_accum = A. ipe 는 **optimizer step** 수 (scheduler 기준),
    #    loader 는 epoch 마다 ipe × A 개 micro-batch 를 낸다. global batch = rank batch × world_size × A.
    #    지금은 ar 계열 (train_step_ar) 만 지원한다 — 다른 kind 는 A = 1 이어야 한다.
    grad_accum = int(O.get("grad_accum", 1))
    if grad_accum < 1:
        raise ValueError("optimization.grad_accum 은 1 이상")
    ipe = int(O["ipe"]) if VW else int(O.get("ipe") or (len(loader) // grad_accum))
    if not VW:
        if len(loader) == 0 or ipe <= 0:
            raise RuntimeError(f"loader 가 비었다 (ipe={ipe}, loader {len(loader)}): {len(dataset)} clips / ws {world_size} / "
                               f"batch {batch_size} (drop_last) — LIMIT 이나 batch_size 를 조정할 것")
        logger.info(f"dataset {len(dataset)} clips | rank batch {batch_size} x ws {world_size} = {batch_size*world_size} | "
                    f"ipe {ipe} (loader {len(loader)}) | masks {'window_grid' if D.get('window_grid') else [m.get('type','temporal_prefix') for m in cfg['mask']]}")

    # ---- 옵티마이저 ---------------------------------------------------------------
    num_epochs = int(O["epochs"])
    optimizer, scaler, scheduler, wd_scheduler = U.init_opt(
        pred_mod, ipe, start_lr=float(O.get("start_lr", 0.0)), ref_lr=float(O["lr"]),
        final_lr=float(O.get("final_lr", 0.0)), warmup_epochs=float(O.get("warmup", 0)), num_epochs=num_epochs,
        wd=float(O.get("weight_decay", 0.04)), final_wd=float(O.get("final_weight_decay", O.get("weight_decay", 0.04))),
        ipe_scale=float(O.get("ipe_scale", 1.0)), betas=tuple(O.get("betas", (0.9, 0.999))),
        eps=float(O.get("eps", 1e-8)), use_scaler=use_scaler, is_anneal=bool(O.get("is_anneal", False)),
        anneal_epochs=float(O.get("anneal_epochs", 0)))
    clip_grad = O.get("clip_grad")
    if lora_params:
        optimizer.add_param_group({"params": lora_params, "WD_exclude": True, "weight_decay": 0.0, "lr": float(O["lr"]),
                                   "lora_lr_mult": float(LORA.get("lr_mult", 10))})
    if state_head is not None:        # ⚠️ head_params 로 분기하면 안 된다 — flow_head 만 켠 경우 SH 가 None 이다 (2026-10-22)
        _hm = float(SH.get("lr_mult", 10))
        optimizer.add_param_group({"params": [q for n, q in state_head.named_parameters() if q.ndim > 1],
                                   "lr": float(O["lr"]), "head_lr_mult": _hm})
        optimizer.add_param_group({"params": [q for n, q in state_head.named_parameters() if q.ndim <= 1],
                                   "WD_exclude": True, "weight_decay": 0.0, "lr": float(O["lr"]), "head_lr_mult": _hm})
    if flow_head is not None:
        _fm = float(FH.get("lr_mult", 10))
        optimizer.add_param_group({"params": [q for n, q in flow_head.named_parameters() if q.ndim > 1],
                                   "lr": float(O["lr"]), "head_lr_mult": _fm})
        optimizer.add_param_group({"params": [q for n, q in flow_head.named_parameters() if q.ndim <= 1],
                                   "WD_exclude": True, "weight_decay": 0.0, "lr": float(O["lr"]), "head_lr_mult": _fm})

    # ---- val ----------------------------------------------------------------------
    valset = None
    if VP:
        spec_p = next(s for s in D["datasets"] if s.get("type") in ACTION_TYPES)
        from app.vjepa_frozen.pusht_data import WallDataset
        tr_stats = (WallDataset if spec_p["type"] == "wall" else PushTDataset).compute_action_stats(spec_p["root"])
        valset = make_traj_dataset(spec_p, "val", n_blocks=n_frames // 2, frameskip=int(spec_p.get("frameskip", 5)),
                                   resolution=res, fixed_starts=[int(x) for x in VP.get("starts", [0, 20, 40, 60])], action_stats=tr_stats)
        if rank == 0:
            logger.info(f"val[pusht]: {len(valset)} windows (val split, starts {VP.get('starts', [0, 20, 40, 60])}), "
                        f"C={VP.get('context_blocks', 3)} K={n_frames // 2 - int(VP.get('context_blocks', 3))}")
    elif V and V.get("data"):
        # val 창은 data.n_frames 와 따로다 — 가변 창은 data.n_frames 가 **최대 창** (64) 이라 IntPhys1 (100 장) 에
        # stride 3 으로 안 들어간다 (1 + 63x3 = 190 > 100). `val.n_frames` 로 준다 (resolve_train 도 같은 값으로 검사한다)
        valset = ValSet(V, int(V.get("n_frames", n_frames)), res)
        if rank == 0:
            logger.info(f"val: {len(valset)} clips from {V['data']['root']} (n_blocks_per_type {V.get('n_blocks_per_type', 8)})")

    # ---- resume -------------------------------------------------------------------
    latest = os.path.join(folder, "latest.pt")
    start_epoch, step = 0, 0
    if (auto_resume or resume_preempt) and os.path.exists(latest):
        start_epoch, step = U.load_resume(latest, pred_mod, optimizer, scaler)
        if lora_params:
            _ck = torch.load(latest, map_location="cpu", weights_only=False)
            LR.load_lora(ctx_enc, _ck["encoder_lora"]); del _ck
            logger.info("resume: encoder LoRA <- latest.pt")
        if state_head is not None:
            _ck = torch.load(latest, map_location="cpu", weights_only=False)
            state_head.load_state_dict(_ck["state_head"]); del _ck
            logger.info("resume: state_head <- latest.pt")
        if flow_head is not None:
            _ck = torch.load(latest, map_location="cpu", weights_only=False)
            flow_head.load_state_dict(_ck["flow_head"]); del _ck
            logger.info("resume: flow_head <- latest.pt")
        for _ in range(start_epoch * ipe):
            scheduler.step(); wd_scheduler.step()
    elif os.path.exists(latest):
        raise RuntimeError(f"{latest} 가 있는데 meta.auto_resume 이 false 다. 새 NAME 을 쓰거나 폴더를 비울 것")
    elif O.get("is_anneal"):
        # vast 이식 (2026-10-01 규약): 릴리즈 cooldown 과 같다 — 새 cooldown run 은 앞 단계 run 의 predictor
        # 가중치에서 시작한다 (옵티마이저 · 스케줄은 새로). 이어 돌리기(latest.pt 있음)는 위 분기가 먼저 탄다
        ak = O.get("anneal_ckpt")
        if not ak or not os.path.isfile(ak):
            raise FileNotFoundError(f"optimization.is_anneal 인데 anneal_ckpt 가 없다: {ak!r}")
        pred_mod.load_state_dict(torch.load(ak, map_location="cpu", weights_only=False)["predictor"], strict=True)
        logger.info(f"is_anneal: predictor <- {ak} (strict). lr {O['lr']} -> {O.get('final_lr', 0.0)} 선형 감소")

    csv_logger = CSVLogger(os.path.join(folder, f"log_r{rank}.csv"), ("%d", "epoch"), ("%d", "itr"),
                           ("%.5f", "loss"), ("%d", "ctx_frames"), ("%.2e", "lr"),
                           ("%d", "iter-time(ms)"), ("%d", "gpu-time(ms)"), ("%d", "data-time(ms)"))
    metrics_path = os.path.join(folder, "metrics.jsonl")

    def _ac():
        return torch.autocast("cuda", dtype=ac_dtype) if ac_dtype else torch.autocast("cuda", enabled=False)

    def _val(epoch):
        if valset is None:
            return {}
        if VP:
            res = AR.run_val_pusht(valset, tgt_enc, pred_mod, device, ac_dtype, rank, world_size, tubelet, spatial,
                                   int(VP.get("context_blocks", 3)), int(VP.get("batch_size", 8)), loss_exp)
            if rank == 0:
                logger.info(f"[epoch {epoch}] {AR.format_val_pusht(res)}")
            return res
        if is_seq:
            if is_ar:
                res = AR.run_val_ar(valset, tgt_enc, pred_mod, device, ac_dtype, rank, world_size, tubelet, spatial,
                                    int(V.get("context_length", 16)), int(V.get("batch_size", 8)), loss_exp)
            elif is_ctx_state:
                res = AR.run_val_ctx_state(valset, ctx_enc, tgt_enc, pred_mod, device, ac_dtype, rank, world_size, tubelet,
                                           spatial, int(V.get("context_length", 16)), int(V.get("batch_size", 8)), loss_exp,
                                           ctx_target=ctx_target)
            else:
                res = AR.run_val_ctx_ar(valset, ctx_enc, tgt_enc, pred_mod, device, ac_dtype, rank, world_size, tubelet,
                                        spatial, int(V.get("context_length", 16)), int(V.get("batch_size", 8)), loss_exp)
            if rank == 0:
                logger.info(f"[epoch {epoch}] {AR.format_val_ar(res)}")
            return res
        res = run_val(valset, ctx_enc, tgt_enc, predictor, device, ac_dtype, rank, world_size, tubelet, spatial,
                      int(V.get("context_length", 16)), int(V.get("batch_size", 8)), mask_index, loss_exp, target_ln)
        if rank == 0:
            logger.info(f"[epoch {epoch}] {format_val(res)}")
        return res

    if rank == 0:
        logger.info(f"predictor trainable {U.count_parameters(pred_mod)/1e6:.2f}M | load_predictor={bool(MD.get('load_predictor'))} "
                    f"| autocast={ac_dtype} enc_dtype={enc_dtype} | lr {O['lr']} wd {O.get('weight_decay')} epochs {num_epochs}"
                    + (f" | AR: 블록별 인코딩, loss = tf + rollout(R<={rollout_steps})" if is_ar else "")
                    + (f" | CTX_AR: 문맥 창 단위 z + 미래 블록별 LN(h), loss = tf + rollout(R<={rollout_steps})" if is_ctx_ar else "")
                    + (f" | CTX_STATE: 문맥 창 z + TF 관측(블록별 LN(h)) / rollout 상태 되먹임, 타깃 = {ctx_target}, "
                       f"loss = {tf_weight}*tf + rollout(R<={rollout_steps})" if is_ctx_state else ""))
    # ★ 모든 rank 가 들어가야 한다 — run_val 안의 all_gather_object 는 collective 라 rank 0 만 부르면 나머지가 영원히 기다린다
    VV = VP or V
    if start_epoch == 0 and VV and VV.get("at_start", True):
        _val(0)

    # ---- 학습 루프 ----------------------------------------------------------------
    MIX = False
    if VW:
        # 가변 창: step 번호가 (C, K, 데이터셋, B) 를 정한다 → 이어 돌릴 때 micro-batch 번호를 그대로 이어야 재현된다
        from app.vjepa_frozen.var_window import build_var_window_loader
        aug = D.get("aug") or {}
        vw_tf = ClipTransform(res, hflip_p=aug.get("hflip_p", 0.0), random_resized_crop=aug.get("random_resized_crop"),
                              square_crop=aug.get("square_crop", "none"), out_uint8=True)
        _ld = dict(num_workers=int(D.get("num_workers", 8)), pin_mem=bool(D.get("pin_mem", True)),
                   prefetch_factor=int(D.get("prefetch_factor", 2)), timeout_s=float(D.get("loader_timeout_s", 0)))
        # 2026-10-20 (vast f6e7934 이식): boundary.mix = sum — 합성을 **step 교대가 아니라 매 step 보조 손실**로.
        #   교대는 합성 step 의 큰 gradient (자연의 5~6 배) 가 Adam 2 차 모멘트를 키워 자연 step 갱신을 약 1/2.4 로 줄였다.
        #   자연 로더 = boundary 없음 · 합성 로더 = boundary p 1.0 · token 예산 x budget_frac. 두 손실을 더해 step 한 번.
        #   키가 없으면 MIX False 라 종전과 완전히 같다.
        BX = (VW or {}).get("boundary") or {}
        MIX = str(BX.get("mix", "")) == "sum"
        if MIX:
            VW_nat = {**VW, "boundary": None}
            VW_syn = {**VW, "boundary": {**BX, "p": 1.0},
                      "token_budget": max(1, int(int(VW["token_budget"]) * float(BX.get("budget_frac", 0.25))))}
            vw_ds, loader = build_var_window_loader(VW_nat, vw_tf, tubelet, seed, rank, start_epoch * ipe * grad_accum, **_ld)
            syn_ds, syn_loader = build_var_window_loader(VW_syn, vw_tf, tubelet, seed + 7919, rank,
                                                        start_epoch * ipe * grad_accum, **_ld)
            syn_it = iter(syn_loader)
            if not is_ar:
                raise ValueError("boundary.mix=sum 은 kind=ar | ar_state 의 train_step_ar 에만 붙였다 "
                                 "(다른 팔은 합성 batch 를 안 읽는다 — 붙이려면 그 train_step 에도 같은 블록을 넣을 것)")
            if rank == 0:
                logger.info(f"boundary mix=sum: 매 step 자연 batch + 합성 batch (합성 token 예산 {syn_ds.budget}, "
                            f"합성 latent x{BX.get('latent_weight', 1.0)} · 상태 x{(SH or {}).get('weight', 1.0)})")
        else:
            vw_ds, loader = build_var_window_loader(VW, vw_tf, tubelet, seed, rank, start_epoch * ipe * grad_accum, **_ld)
        if rank == 0:
            logger.info(f"var_window: ipe {ipe} x grad_accum {grad_accum}, token_budget {vw_ds.budget}, "
                        f"조합 {len(vw_ds.combos)}개, 시작 step {start_epoch * ipe * grad_accum}\n" + vw_ds.describe())
    else:
        sampler.set_epoch(start_epoch)
    loader_it = iter(loader)
    lr_now = wd_now = float("nan")
    if grad_accum > 1 and not is_seq:
        raise ValueError("grad_accum > 1 은 kind=ar | ar_state | ctx_ar | ctx_state 에서만 구현돼 있다")
    if rank == 0 and grad_accum > 1:
        logger.info(f"grad_accum {grad_accum}: global batch {batch_size} x {world_size} x {grad_accum} = {batch_size * world_size * grad_accum} | "
                    f"optimizer step/epoch {ipe}, micro-batch/epoch {ipe * grad_accum}")
    from contextlib import nullcontext
    lr_hold = [float("nan"), float("nan")]
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        loss_meter, it_meter, gpu_meter, data_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        jl_meter, sl_meter = AverageMeter(), AverageMeter()                       # 팔 C 전용 (tf / rollout)
        head_meters = {}                                                          # state_head 로그 (epoch 평균)
        for itr in range(ipe * grad_accum):                  # itr = micro-batch. optimizer step 은 (itr+1) % grad_accum == 0 에서
            t_it = time.time()
            first_micro = (itr % grad_accum == 0)
            do_update = ((itr + 1) % grad_accum == 0)
            if VW:
                # 로더가 (C, K, 데이터셋, B) 를 정해 창 W 장을 정확히 준다 (자르지 않는다). 마스크는 그 (C, K) 로 만든다
                clips, info = next(loader_it)
                m_enc, m_pred = sampler_masks.prefix_window(clips.size(0), info["context_blocks"], info["predict_blocks"])
            else:
                try:
                    clips, m_enc, m_pred, info = next(loader_it)
                except StopIteration:
                    sampler.set_epoch(epoch + 1)
                    loader_it = iter(loader)
                    clips, m_enc, m_pred, info = next(loader_it)
            lab = info.get("labels") if (isinstance(info, dict) and m_enc is None) else None    # phys2d_sup (state_head)
            if state_head is not None and lab is None and not MIX:
                raise ValueError("state_head 인데 배치에 라벨이 없다 — data.datasets 가 type: phys2d_sup 이어야 한다")
            if MIX:                                           # 보조 합성 batch (라벨은 여기에 있다)
                clips_s, info_s = next(syn_it)
                if state_head is not None and info_s.get("labels") is None:
                    raise ValueError("mix=sum + state_head 인데 합성 batch 에 라벨이 없다 — boundary.labels: phys2d_sup 이어야 한다")
            if VW:
                pass
            elif is_pusht:
                # Push-T: 창 = nb 관측 블록 전부 (자르지 않음). 문맥 길이 C 만 rank 동기 RNG 로 뽑고 K = nb - C.
                nb_p = clips.size(2) // tubelet
                C_p = sampler_masks.rng.choice(pusht_ctx_choices)
                info = {"type": f"pusht C{C_p}K{nb_p - C_p}", "context_frames": C_p * tubelet, "context_blocks": C_p,
                        "predict_blocks": nb_p - C_p, "actions": info["actions"].to(device, non_blocking=True)}
                m_enc = m_pred = torch.zeros(0, device=device)        # 안 쓴다 (아래 .to 호출용)
            elif m_enc is None:                  # sync_masks 경로: rank 동기 마스크 + 창을 C+K 블록으로 자른다
                m_enc, m_pred, info = sampler_masks(clips.size(0))
                n_blk = (m_enc.size(1) + m_pred.size(1)) // sampler_masks.S
                # ⚠️ 2026-10-18: 자르기는 enc|pred 가 **앞쪽 C+K 블록을 빈틈없이 덮는** 마스크에서만 뜻이 있다
                #    (temporal_prefix · prefix_window). block3d 는 시공간 블록이라 enc + pred < N 이고
                #    (예: 16 프레임에서 288 + 1536 = 1824 < 2048) 인덱스는 창 끝까지 간다 —
                #    그대로 자르면 gather 가 범위를 넘어 **device-side assert** 로 죽는다 (실제로 죽었다).
                if info.get("type") == "block3d":
                    n_blk = sampler_masks.T
                if n_blk < sampler_masks.T:
                    o = sampler_masks._r().randint(0, sampler_masks.T - n_blk)
                    clips = clips[:, :, o * tubelet: (o + n_blk) * tubelet]
                    info = {**info, "win_blocks": n_blk, "win_offset": o}
                    if lab is not None:                                  # 라벨은 블록 축 — 같은 창으로 자른다
                        lab = {k: v[:, o: o + n_blk] for k, v in lab.items()}
            clips = normalize_clips(clips.to(device, non_blocking=True))   # uint8 전송이면 GPU 정규화, float 이면 그대로
            if MIX:                                                        # 합성도 같은 경로 — 안 옮기면 CPU 텐서가 encoder 에 들어가 죽는다 (2026-10-21)
                clips_s = normalize_clips(clips_s.to(device, non_blocking=True))
            m_enc = m_enc.to(device, non_blocking=True)
            m_pred = m_pred.to(device, non_blocking=True)
            data_ms = (time.time() - t_it) * 1000

            aux = {}

            def train_step_ar():
                # 팔 C (2026-09-26): 블록별 인코딩 -> LN -> predictor 한 forward 로 (p_tf, p_ar) -> jloss + sloss
                # grad_accum (2026-10-10): scheduler 는 묶음의 첫 micro 에서, optimizer 는 마지막 micro 에서. 앞 micro 는 DDP no_sync
                if first_micro:
                    lr_hold[0] = scheduler.step(); lr_hold[1] = wd_scheduler.step()
                lr_now, wd_now = lr_hold
                if head_params:
                    for g in optimizer.param_groups:                                  # head 그룹 lr = 스케줄 lr × lr_mult
                        if "head_lr_mult" in g:
                            g["lr"] = lr_now * g["head_lr_mult"]
                        elif HCFG.get("freeze_predictor"):                            # 대조군: predictor 를 얼리고 head 만 (읽어 내기)
                            g["lr"] = 0.0
                if "context_blocks" not in info:
                    raise ValueError("kind=ar 는 mask.type=prefix_window 만 지원한다 (C·K 블록 수가 필요)")
                with torch.no_grad(), _ac():
                    h = AR.encode_blocks(tgt_enc, clips.to(enc_dtype), tubelet)                # (B, nb*S, D) LN
                sync_ctx = predictor.no_sync() if (grad_accum > 1 and not do_update and hasattr(predictor, "no_sync")) else nullcontext()
                # mix=sum (2026-10-20): 자연 손실은 **늘** no_sync 로 gradient 만 쌓고, 뒤의 합성 backward 가 한 번에 동기화한다
                nat_ctx = (predictor.no_sync() if hasattr(predictor, "no_sync") else nullcontext()) if MIX else sync_ctx
                with nat_ctx:
                    outs = {} if head_params else None
                    with _ac():
                        loss, jl, sl, R = AR.ar_losses(predictor, h, spatial, int(info["context_blocks"]),
                                                       int(info["predict_blocks"]), loss_exp, rollout_steps, actions=info.get("actions"),
                                                       out=outs)
                        if state_head is not None and lab is not None:
                            lh, hlog = SHM.ar_state_loss(state_head, outs["p_tf"], outs["p_ar"], SHM.labels_to_device(lab, device),
                                                         spatial, int(info["context_blocks"]), R, SH.get("w") or {}, SH.get("split") or {})
                            aux["latent"] = float(loss.detach())
                            aux["head"] = float(lh.detach())
                            aux["hlog"] = hlog
                            loss = loss + float(SH.get("weight", 1.0)) * lh
                        if flow_head is not None:
                            fl_t, fl_v = info.get("flow"), info.get("flow_valid")
                            if fl_t is None:
                                raise ValueError("flow_head 인데 배치에 flow 라벨이 없다 — data.var_window.flow.roots 를 볼 것")
                            lf, flog = FHM.ar_flow_loss(flow_head, outs["p_tf"], outs["p_ar"],
                                                        fl_t.to(device, non_blocking=True), fl_v.to(device, non_blocking=True),
                                                        spatial, int(info["context_blocks"]), R, FH.get("split") or {},
                                                        horizons=FHZ, supervise=FSUP, norm=FH.get("norm", "auto"))
                            aux["latent"] = float(loss.detach()) if "latent" not in aux else aux["latent"]
                            aux["flow"] = float(lf.detach())
                            aux["flog"] = flog
                            loss = loss + float(FH.get("weight", 1.0)) * lf
                    lb = loss / grad_accum
                    if scaler is not None:
                        scaler.scale(lb).backward()
                    else:
                        lb.backward()
                if MIX:
                    # 보조 합성 batch: latent x latent_weight + 상태 x state_head.weight. 자연 gradient 위에 더해 step 한 번.
                    if scaler is not None:
                        raise ValueError("boundary.mix=sum 은 bf16 (scaler 없음) 에서만 — fp16 scaler 와 섞으면 두 backward 의 스케일이 어긋난다")
                    aux["nat_latent"] = float(loss.detach())
                    with torch.no_grad(), _ac():
                        hs = AR.encode_blocks(tgt_enc, clips_s.to(enc_dtype), tubelet)
                    with sync_ctx:
                        outs_s = {} if state_head is not None else None
                        with _ac():
                            ls, _jls, _sls, Rs = AR.ar_losses(predictor, hs, spatial, int(info_s["context_blocks"]),
                                                              int(info_s["predict_blocks"]), loss_exp, rollout_steps, out=outs_s)
                            loss_s = float(BX.get("latent_weight", 1.0)) * ls
                            aux["syn_latent"] = float(ls.detach())
                            if state_head is not None:
                                lh, hlog = SHM.ar_state_loss(state_head, outs_s["p_tf"], outs_s["p_ar"],
                                                             SHM.labels_to_device(info_s["labels"], device),
                                                             spatial, int(info_s["context_blocks"]), Rs,
                                                             SH.get("w") or {}, SH.get("split") or {})
                                aux["latent"] = float(ls.detach())
                                aux["head"] = float(lh.detach())
                                aux["hlog"] = hlog
                                loss_s = loss_s + float(SH.get("weight", 1.0)) * lh
                        (loss_s / grad_accum).backward()
                    loss = loss.detach() + loss_s.detach()
                if do_update:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    if head_params and world_size > 1:                               # DDP 는 predictor 만 — head grad 는 직접 평균
                        for q in head_params:
                            if q.grad is not None:
                                torch.distributed.all_reduce(q.grad); q.grad.div_(world_size)
                    if head_params and HCFG.get("clip_separate"):   # 2026-10-19 (vast 167db20): clip 은 predictor 에만 (gn = predictor norm).
                        #   head 는 state_head.clip_head (null = 안 자름) — 같이 자르면 head grad 가 predictor norm 에 눌려 거의 못 배운다
                        _gn = torch.nn.utils.clip_grad_norm_(pred_mod.parameters(), float(clip_grad) if clip_grad else float("inf"))
                        if HCFG.get("clip_head"):
                            torch.nn.utils.clip_grad_norm_(head_params, float(HCFG["clip_head"]))
                        aux["gn"] = float(_gn)
                    elif clip_grad or head_params:  # 2026-10-19 (vast 36f4921): head 가 있으면 grad norm 을 늘 잰다 (clip 전 값, 로그 gn).
                        #   왜: clip 없음 + head lr x10 으로 vast 에서 epoch 4 에 predictor 가 무너졌다
                        #   (합성 step grad norm 13~22 vs 자연 0.25~0.85). 재야 무너지기 전에 보인다. clip_grad 없으면 자르지 않는다
                        _gn = torch.nn.utils.clip_grad_norm_(list(pred_mod.parameters()) + head_params,
                                                             float(clip_grad) if clip_grad else float("inf"))
                        if head_params:
                            aux["gn"] = float(_gn)
                    if scaler is not None:
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                aux.update(jloss=float(jl.detach()), sloss=float(sl.detach()), R=R)
                return float(loss.detach()), lr_now, wd_now

            def _ctx_backward_step(loss):
                """ctx 팔 공통 뒷단 — grad_accum micro-batch 규약 (train_step_ar 과 같다)."""
                lb = loss / grad_accum
                if scaler is not None:
                    scaler.scale(lb).backward()
                else:
                    lb.backward()

            def _ctx_update():
                if not do_update:
                    return
                if scaler is not None:
                    scaler.unscale_(optimizer)
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(pred_mod.parameters(), float(clip_grad))
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            def train_step_ctx_ar():
                # 팔 C' (2026-09-27, vast 이식 2026-10-16): 문맥 = context encoder 가 C 블록 창을 **한 번에** (m_enc 로 미래 토큰 제거,
                # prefix 팔과 같다), 미래 GT = target encoder **블록별** -> LN. predictor 한 forward 로 (p_tf, p_ar) -> jloss + sloss.
                if first_micro:
                    lr_hold[0] = scheduler.step(); lr_hold[1] = wd_scheduler.step()
                lr_now, wd_now = lr_hold
                if "context_blocks" not in info:
                    raise ValueError("kind=ctx_ar 는 mask.type=prefix_window 만 지원한다 (C·K 블록 수가 필요)")
                C_, K_ = int(info["context_blocks"]), int(info["predict_blocks"])
                with torch.no_grad(), _ac():
                    xe = clips.to(enc_dtype)
                    z = ctx_enc(xe, masks=[m_enc])                                       # (B, C*S, D) 문맥 z
                    h_fut = AR.encode_blocks(tgt_enc, xe[:, :, C_ * tubelet:], tubelet)   # (B, K*S, D) 블록별 LN
                sync_ctx = predictor.no_sync() if (grad_accum > 1 and not do_update and hasattr(predictor, "no_sync")) else nullcontext()
                with sync_ctx:
                    with _ac():
                        loss, jl, sl, R = AR.ctx_ar_losses(predictor, z, h_fut, spatial, C_, K_, loss_exp, rollout_steps)
                    _ctx_backward_step(loss)
                _ctx_update()
                aux.update(jloss=float(jl.detach()), sloss=float(sl.detach()), R=R)
                return float(loss.detach()), lr_now, wd_now

            def train_step_ctx_state():
                # vast 이식 (2026-10-16, 사용자 요청 "context 를 video 로 주고 target 도 video 로"):
                #   문맥 = context encoder 가 C 블록 창을 한 번에 (video) -> z
                #   타깃 = target encoder 가 **창 (C+K) 전체**를 한 번에 -> LN(h) 의 미래 블록 (video, 표준 채점과 같은 타깃)
                #   TF 관측 입력만 블록별 LN(h) (encoder 가 양방향이라 미래를 한 번에 넣으면 누수). tf_weight 0 이면 아예 안 만든다
                if first_micro:
                    lr_hold[0] = scheduler.step(); lr_hold[1] = wd_scheduler.step()
                lr_now, wd_now = lr_hold
                if "context_blocks" not in info:
                    raise ValueError("kind=ctx_state 는 mask.type=prefix_window 만 지원한다 (C·K 블록 수가 필요)")
                C_, K_ = int(info["context_blocks"]), int(info["predict_blocks"])
                with torch.no_grad(), _ac():
                    xe = clips.to(enc_dtype)
                    z = ctx_enc(xe, masks=[m_enc])                                       # (B, C*S, D) 문맥 z
                    if ctx_target == "block":          # 타깃 = TF 관측과 같은 블록별 LN (창 전체 forward 생략)
                        tgt = AR.encode_blocks(tgt_enc, xe[:, :, C_ * tubelet:], tubelet)  # (B, K*S, D)
                        h_obs = tgt if tf_weight > 0 else None
                    else:
                        h_obs = (AR.encode_blocks(tgt_enc, xe[:, :, C_ * tubelet:], tubelet)  # (B, K*S, D) TF 관측 (블록별 LN)
                                 if tf_weight > 0 else None)
                        tgt = torch.nn.functional.layer_norm(tgt_enc(xe), (z.size(-1),))[:, C_ * spatial:]   # (B, K*S, D) 창 LN
                sync_ctx = predictor.no_sync() if (grad_accum > 1 and not do_update and hasattr(predictor, "no_sync")) else nullcontext()
                with sync_ctx:
                    with _ac():
                        loss, jl, sl, R = AR.ctx_state_losses(predictor, z, h_obs, tgt, spatial, C_, K_, loss_exp,
                                                              rollout_steps, tf_weight)
                    _ctx_backward_step(loss)
                _ctx_update()
                aux.update(jloss=float(jl.detach()), sloss=float(sl.detach()), R=R)
                return float(loss.detach()), lr_now, wd_now

            def train_step():
                lr_now = scheduler.step()
                wd_now = wd_scheduler.step()
                for g in optimizer.param_groups:                                      # LoRA 그룹 lr = 스케줄 lr × lr_mult
                    if "lora_lr_mult" in g:
                        g["lr"] = lr_now * g["lora_lr_mult"]
                with torch.no_grad(), _ac():
                    xe = clips.to(enc_dtype)
                    h = tgt_enc(xe)                                                   # (B, N, D)
                    h = torch.gather(h, 1, m_pred.unsqueeze(-1).expand(-1, -1, h.size(-1)))
                    if target_ln:
                        h = torch.nn.functional.layer_norm(h, (h.size(-1),))       # affine-free, target 에만
                    if not lora_params:
                        z = ctx_enc(xe, masks=[m_enc])                                # context 토큰만 (B, K, D)
                if lora_params:
                    with _ac():
                        z = ctx_enc(xe, masks=[m_enc])                                # LoRA: context encoder 에 gradient
                with _ac():
                    p = predictor(z, m_enc, m_pred, mask_index=mask_index)            # (B, M, D)
                loss = (p.float() - h.float()).abs().pow(loss_exp).mean() / loss_exp
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if lora_params and world_size > 1:                                    # DDP 는 predictor 만 감쌌다 → LoRA grad 는 직접 평균
                    for q in lora_params:
                        if q.grad is not None:
                            torch.distributed.all_reduce(q.grad); q.grad.div_(world_size)
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(list(pred_mod.parameters()) + lora_params, float(clip_grad))
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                return float(loss.detach()), lr_now, wd_now

            _step_fn = (train_step_ar if is_ar else train_step_ctx_ar if is_ctx_ar
                        else train_step_ctx_state if is_ctx_state else train_step)
            (loss, lr_now, wd_now), gpu_ms = gpu_timer(_step_fn)
            if do_update:
                step += 1                                        # optimizer step 수 (체크포인트·metrics 의 step)
            if is_seq:
                jl_meter.update(aux["jloss"]); sl_meter.update(aux["sloss"])
            if "flog" in aux:
                for k_, v_ in [("latent", aux["latent"]), ("flow", aux["flow"])] + [("f_" + a, b) for a, b in aux["flog"].items()]:
                    if isinstance(v_, (int, float)):
                        head_meters.setdefault(k_, AverageMeter()).update(v_)
            if "hlog" in aux:
                for k_, v_ in [("latent", aux["latent"]), ("head", aux["head"])] + list(aux["hlog"].items()):
                    if np.isfinite(v_):
                        head_meters.setdefault(k_, AverageMeter()).update(v_)
            if "syn_latent" in aux:      # mix=sum: 자연 · 합성 latent 를 따로 남긴다 (metrics.jsonl 의 train_head)
                for k_, v_ in (("nat_latent", aux["nat_latent"]), ("syn_latent", aux["syn_latent"])):
                    if np.isfinite(v_):
                        head_meters.setdefault(k_, AverageMeter()).update(v_)
            it_ms = (time.time() - t_it) * 1000
            loss_meter.update(loss); it_meter.update(it_ms); gpu_meter.update(gpu_ms); data_meter.update(data_ms)
            csv_logger.log(epoch + 1, itr, loss, info["context_frames"], lr_now, it_ms, gpu_ms, data_ms)
            if (itr % log_freq == 0) or (itr == ipe * grad_accum - 1) or not np.isfinite(loss):
                ar_s = (" [tf %.4f ar %.4f R=%d]" % (aux["jloss"], aux["sloss"], aux["R"])) if is_seq else ""
                if "flog" in aux:
                    fl = aux["flog"]
                    ar_s += (" [traj %.4f tf %.3f ar %.3f cv %.3f val %.2f"
                             % (aux["flow"], fl["tf_l"], fl["ar_l"], fl["ar_c"], fl["valid"]))
                    ar_s += " h " + " ".join("%s:%.2f/%.2f" % (k[1:], v, fl["c" + k[1:]])
                                             for k, v in fl.items() if k[0] == "l" and k[1:].isdigit()) + "]"
                if "hlog" in aux:
                    hl = aux["hlog"]
                    ar_s += (" [head %.4f | ex %.3f/%.3f pos %.4f/%.4f dir %.3f/%.3f spd %.3f/%.3f | ar px vis %.1f hid %.1f]"
                             % (aux["head"], hl["tf_ex"], hl["ar_ex"], hl["tf_pos"], hl["ar_pos"], hl["tf_dir"], hl["ar_dir"],
                                hl["tf_spd"], hl["ar_spd"], hl["ar_pos_px_vis"], hl["ar_pos_px_hid"]))
                    if "ar_id_acc" in hl:        # 2026-10-19 (vast 167db20) single head: 49 지선다 정체 정확도 + attention 집중도
                        ar_s += " id %.3f conc %.1f" % (hl["ar_id_acc"], hl.get("ar_conc", float("nan")))
                if "syn_latent" in aux:   # mix=sum: 자연 batch 손실 (상태 없음) vs 합성 batch latent
                    ar_s += " | nat %.4f syn %.4f" % (aux["nat_latent"], aux["syn_latent"])
                if "gn" in aux:
                    ar_s += " gn %.2f" % aux["gn"]
                logger.info("[%d, %5d] loss %.4f (avg %.4f)%s | mask %s C=%s | lr %.2e wd %.2e | mem %.1fG | iter %.0f ms gpu %.0f ms data %.0f ms"
                            % (epoch + 1, itr, loss, loss_meter.avg, ar_s, info["type"], info["context_frames"], lr_now, wd_now,
                               torch.cuda.max_memory_allocated() / 1024**3, it_meter.avg, gpu_meter.avg, data_meter.avg))
            if not np.isfinite(loss):
                raise RuntimeError("loss 가 nan/inf")
            if M.get("sync_gc") and (itr + 1) % 50 == 0:
                gc.collect()

        logger.info(f"epoch {epoch + 1} avg loss {loss_meter.avg:.4f}" + (f" (tf {jl_meter.avg:.4f} ar {sl_meter.avg:.4f})" if is_seq else ""))
        if rank == 0:
            _extra = {"encoder_lora": LR.lora_state_dict(ctx_enc), "encoder_lora_cfg": ctx_enc.lora_info} if lora_params else None
            if state_head is not None:
                _extra = {**(_extra or {}), "state_head": state_head.state_dict(), "state_head_cfg": SH}
            if flow_head is not None:
                _extra = {**(_extra or {}), "flow_head": flow_head.state_dict(), "flow_head_cfg": FH}
            U.save_checkpoint(latest, predictor, optimizer, scaler, epoch + 1, step, loss_meter.avg, cfg, arch, extra=_extra)
            if save_every > 0 and ((epoch + 1) % save_every == 0 or epoch + 1 == num_epochs):
                U.save_checkpoint(os.path.join(folder, f"e{epoch + 1}.pt"), predictor, optimizer, scaler,
                                  epoch + 1, step, loss_meter.avg, cfg, arch, extra=_extra)
        res = {}
        if VV and int(VV.get("every_epochs", 1)) > 0 and ((epoch + 1) % int(VV.get("every_epochs", 1)) == 0 or epoch + 1 == num_epochs):
            res = _val(epoch + 1)
        if rank == 0:
            with open(metrics_path, "a") as f:
                rec = {"epoch": epoch + 1, "step": step, "train_loss": loss_meter.avg, "lr": lr_now,
                       "val": res, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                if is_ar:
                    rec.update(train_jloss=jl_meter.avg, train_sloss=sl_meter.avg)
                if head_meters:
                    rec["train_head"] = {k_: m_.avg for k_, m_ in head_meters.items()}
                f.write(json.dumps(rec) + "\n")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    logger.info("done")
