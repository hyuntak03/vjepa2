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

import contextlib
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
from app.vjepa_frozen.val import ValSet, format_val, run_val
from app.vjepa_frozen import ar as AR
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

class _MultiMaskPredictor(torch.nn.Module):
    """`mask_mode: all` 용 — 한 step 의 마스크 여러 개를 **DDP forward 한 번**으로 돌린다.

    DDP 는 backward 한 번에 forward 한 번을 기대한다. predictor(DDP) 를 마스크마다 따로 부르면
    같은 파라미터가 두 번 ready 로 표시돼 죽는다. 그래서 루프를 모듈 안에 넣는다 (릴리즈
    PredictorMultiSeqWrapper 와 같은 발상). 체크포인트는 안쪽 predictor 만 저장하므로 형식이 그대로다.
    """

    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, zs, encs, preds, idxs):
        return [self.pred(z, e, p, mask_index=i) for z, e, p, i in zip(zs, encs, preds, idxs)]


warnings.filterwarnings("ignore", message=r".*sdp_kernel.*", category=FutureWarning)
logger = get_logger(__name__, force=True)



def _grad_probe(pred_mod, terms: dict):
    """2026-10-26 (사용자 요청): predictor 파라미터로 가는 gradient 를 손실 항별로 따로 잰다 (autograd.grad, 그래프 유지, 학습 grad 에 영향 없음).
    terms = {이름: 스칼라 손실 (가중 적용)}. return {이름: |∇|} + {cos_<이름>: cos(∇ 이름, ∇ lat)} (lat 이 있으면)."""
    ps = [q for q in pred_mod.parameters() if q.requires_grad]
    G = {}
    for n, l in terms.items():
        if not torch.is_tensor(l) or not l.requires_grad:
            continue
        g = torch.autograd.grad(l, ps, retain_graph=True, allow_unused=True)
        G[n] = torch.cat([(x if x is not None else torch.zeros_like(q)).reshape(-1).float() for x, q in zip(g, ps)])
    out = {n: float(v.norm()) for n, v in G.items()}
    if "lat" in G:
        for n, v in G.items():
            if n != "lat":
                out[f"cos_{n}"] = float(torch.nn.functional.cosine_similarity(v, G["lat"], dim=0))
    return out


_COMPS = ("pos", "dir", "spd", "ex")


def _matched_head(pred_mod, parts: dict, lat, GM: dict, st: dict, probe: bool, world_size: int):
    """2026-10-26 (사용자 결정): 고정 head 손실의 성분별 가중치를 predictor gradient 크기로 맞춘다.
    성분 c (위치 · 방향 · 속력 · 존재/정체, 각각 TF + 굴린 합) 의 가중치 w_c = GM[c] / EMA(‖∇ c‖ / ‖∇ lat‖) — 그러면 ‖∇ (w_c · c)‖ ≈ GM[c] · ‖∇ lat‖.
    비율은 probe step 마다 (autograd.grad) 재고 rank 평균 · EMA (GM.ema, 기본 0.9). 처음 잴 때까지는 head 손실 0.
    return (head 손실 (가중 합), {c: w_c}, gprobe 로그 또는 None)."""
    comps = {c: parts[f"tf_{c}"] + parts[f"ar_{c}"] for c in _COMPS if f"tf_{c}" in parts}
    gp = None
    if probe:
        gp = _grad_probe(pred_mod, {"lat": lat, **comps, **parts})
        r = torch.tensor([gp.get(c, 0.0) / max(gp["lat"], 1e-12) for c in _COMPS], device=lat.device, dtype=torch.float64)
        if world_size > 1:
            torch.distributed.all_reduce(r); r /= world_size
        a = float(GM.get("ema", 0.9))
        old = st.get("r")
        st["r"] = r if old is None else a * old + (1 - a) * r
    w = {c: (float(GM.get(c, 0.0)) / max(float(st["r"][i]), 1e-12) if st.get("r") is not None else 0.0) for i, c in enumerate(_COMPS)}
    lh = sum(w[c] * comps[c] for c in comps if w[c] > 0) if any(w[c] > 0 for c in comps) else lat * 0.0
    if probe:
        gp.update(_grad_probe(pred_mod, {"lat": lat, "head": lh}) if lh.requires_grad else {})
        gp.update({f"w_{c}": w[c] for c in _COMPS})
    return lh, w, gp


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
    tf_weight = float(L.get("tf_weight", 1.0))                           # ctx_state: 0 이면 TF(블록별 관측 인코딩 + TF 패스) 생략
    delta_weight = float(L.get("delta_weight", 0.0))                    # 차이 L1 (2026-10-13, kind=ar 전용)
    CTX_HOLE = L.get("ctx_hole")                                         # 문맥 구멍 (2026-10-16, kind=ctx_state 전용). 없으면 꺼짐
    dyn_weight = float(L.get("dyn_weight", 0.0))                        # 움직임 크기 손실 (2026-10-14, kind=ar 전용, predictor.dyn_head 필요)
    tf_span = str(L.get("tf_span", "all"))                              # TF 손실 범위 (2026-10-15): all | predict (kind=ar 전용)
    HINT = cfg.get("hint")                                               # 힌트 일정 (2026-10-11, kind=ar + hint_cross_dim 전용). 없으면 꺼짐
    kind = str((MD.get("predictor") or {}).get("kind", "oneshot"))
    is_ar = kind == "ar"            # 팔 C (app/vjepa_frozen/ar.py): 문맥·미래 모두 블록별 인코딩 (AC 판)
    is_ctx_ar = kind == "ctx_ar"    # 팔 C' (2026-09-27): 문맥은 창 단위 z, 미래 GT 만 블록별 LN(h)
    is_ctx_state = kind == "ctx_state"  # 2026-09-30: ctx_ar + 상태 되먹임 (predictor 공간), 타깃 = 창 LN(h)
    is_seq = is_ar or is_ctx_ar or is_ctx_state   # tf + rollout 손실을 쓰는 팔들
    # ctx_state 타깃 (2026-10-16): window (기본, 창 전체 LN(h)) | block (블록별 LN(h) = TF 관측과 같은 공간, 창 전체 forward 생략)
    ctx_target = str((MD.get("predictor") or {}).get("ctx_target", "window"))
    if ctx_target not in ("window", "block"):
        raise ValueError(f"model.predictor.ctx_target 는 window | block: {ctx_target!r}")
    if ctx_target != "window" and not is_ctx_state:
        raise ValueError("model.predictor.ctx_target 는 kind=ctx_state 전용")
    if CTX_HOLE and not (is_ctx_state and tf_weight > 0):   # 2026-10-17: 블록 타깃도 허용 (복원 정답 = 가린 문맥 블록의 블록별 LN(h))
        # kind=ar (블록 단위 인코딩) 에 encoding 뒤 구멍을 내면 같은 블록의 보이는 토큰이 가린 자리를 이미 본 뒤라 누수다 → 안 받는다
        raise ValueError("loss.ctx_hole 은 kind=ctx_state + tf_weight>0 전용 (구멍은 context encoder 입력 전에 낸다)")
    # gating_residual (2026-10-17): ŷ = g·A + proj(s), A = 마지막 문맥 블록 LN(target_encoder). 블록 타깃 공간이어야 A 가 의미 있다
    GATING = bool((MD.get("predictor") or {}).get("gating_residual", False))
    # ar_input (2026-10-18, 사용자 결정, kind=ar): block (기본, 블록별 LN(target_encoder)) | video = 입력 = target_encoder 문맥 pass
    #   (영상), 정답 = 같은 encoder 전체 pass (창, ar_target: window). TF 는 문맥 자리만, 예측 구간은 rollout 만 (ar_losses_video)
    ar_input = str((MD.get("predictor") or {}).get("ar_input", "block"))
    ar_target = str((MD.get("predictor") or {}).get("ar_target", "block"))
    if ar_input not in ("block", "video") or ar_target not in ("block", "window"):
        raise ValueError(f"model.predictor.ar_input 는 block | video, ar_target 은 block | window: {ar_input!r} {ar_target!r}")
    AR_VIDEO = is_ar and ar_input == "video"
    if is_ar and (ar_input == "video") != (ar_target == "window"):
        raise ValueError("kind=ar 학습은 (ar_input, ar_target) = (block, block) | (video, window) 만 받는다")
    if AR_VIDEO and (GATING or HINT or dyn_weight or delta_weight or tf_span != "all"):
        raise ValueError("ar_input=video 는 gating · hint · dyn · delta · tf_span 과 같이 쓰지 않는다 "
                         "(영상 공간 복사 anchor 는 0.81 — z_research/scripts/analysis/video_space_copy_gap.py)")
    if GATING and not ((is_ctx_state and ctx_target == "block") or is_ar):
        raise ValueError("model.predictor.gating_residual 는 kind=ar 또는 kind=ctx_state + ctx_target: block 전용 (anchor = 블록별 LN(target_encoder))")
    # ctx_input (2026-10-17, 사용자 결정): video (기본) = 문맥 창을 문맥 encoder 로 한 번 (z 는 LN 없이) / block = 문맥도 **블록마다 따로 LN(문맥 encoder)**.
    #   문맥 encoder 는 ctx_encoder (아래) 가 정한다. 샘플링 · 문맥 양방향 attention · TF + 상태 rollout 은 그대로.
    #   ctx_encoder: target_encoder + ctx_input: block 이면 문맥 · TF 관측 · 타깃이 같은 공간이고 gating anchor = 문맥 z 의 마지막 블록
    ctx_input = str((MD.get("predictor") or {}).get("ctx_input", "video"))
    if ctx_input not in ("video", "block"):
        raise ValueError(f"model.predictor.ctx_input 는 video | block: {ctx_input!r}")
    CTX_BLOCK = ctx_input == "block"
    # ctx_encoder (2026-10-17, 사용자: ablation 두 축 = 어느 encoder × 어떻게 뽑나): encoder (online) | target_encoder (EMA).
    #   주면 model.context_encoder_key 를 덮는다 (predictor 블록 안에 둬야 채점기 · 레지스트리까지 같이 간다 — 평가 프로토콜은 context_encoder_key 를 고정한다)
    ctx_encoder = (MD.get("predictor") or {}).get("ctx_encoder")
    if ctx_encoder is not None:
        if ctx_encoder not in ("encoder", "target_encoder"):
            raise ValueError(f"model.predictor.ctx_encoder 는 encoder | target_encoder: {ctx_encoder!r}")
        if not is_ctx_state:
            raise ValueError("model.predictor.ctx_encoder 는 kind=ctx_state 전용 (ar 은 늘 target_encoder)")
        MD["context_encoder_key"] = ctx_encoder
    if CTX_BLOCK and not is_ctx_state:
        raise ValueError("model.predictor.ctx_input: block 은 kind=ctx_state 전용")
    if CTX_BLOCK and CTX_HOLE:
        raise ValueError("model.predictor.ctx_input: block 과 loss.ctx_hole 은 같이 쓰지 않는다")
    if GATING and CTX_HOLE:
        raise ValueError("model.predictor.gating_residual 와 loss.ctx_hole 은 같이 쓰지 않는다 (anchor 블록이 가려질 수 있다)")
    if dyn_weight and not (is_ar and (MD.get("predictor") or {}).get("dyn_head")):
        raise ValueError("loss.dyn_weight 는 kind=ar + model.predictor.dyn_head: true 에서만 쓴다")
    if delta_weight and not is_ar:
        raise ValueError("loss.delta_weight 는 kind=ar 전용")
    if tf_span not in ("all", "predict"):
        raise ValueError(f"loss.tf_span 은 all | predict: {tf_span!r}")
    if tf_span != "all" and not (is_ar and not dyn_weight and not delta_weight and not HINT):
        raise ValueError("loss.tf_span=predict 는 kind=ar 기본 경로 전용 (hint · delta_weight · dyn_weight 와 같이 쓰지 않는다)")
    if HINT and not (is_ar and int((MD.get("predictor") or {}).get("hint_cross_dim", 0) or 0)):
        raise ValueError("hint: 는 kind=ar + model.predictor.hint_cross_dim > 0 에서만 쓴다")
    if is_seq and not target_ln:
        raise ValueError("kind=ar 는 loss.target_layer_norm=true 가 전제다 (입력·타깃·출력이 모두 LN 공간)")
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
    # ── model.encoder_lora (kind=ar, 2026-10-25, 사용자: "ar 에 intphys2 pft 할 때 encoder 에 lora 붙인 버전도"). main 의 app/vjepa_frozen/lora.py 이식.
    #    AR 은 입력 · 타깃이 같은 target_encoder 라서, **입력 경로 전용 복사본** 에만 LoRA 를 단다 (타깃 encoder 는 얼린 채 — 타깃이 안 움직인다).
    #    입력 = LN(in_enc(블록)) (LoRA, grad), 타깃 = LN(tgt_enc(블록)) (얼림). 채점기는 체크포인트의 encoder_lora 로 문맥 encoder 를 다시 만든다.
    LORA = MD.get("encoder_lora") or None
    lora_params, in_enc = [], None
    if LORA:
        if not (str((MD.get("predictor") or {}).get("kind", "oneshot")) == "ar" and str((MD.get("predictor") or {}).get("ar_input", "block")) == "block"):
            raise ValueError("이 판의 encoder_lora 는 kind=ar + ar_input: block 전용이다 (main 의 release 판은 아직 안 옮겼다)")
        import copy
        from app.vjepa_frozen import lora as LR
        in_enc = copy.deepcopy(tgt_enc)
        lora_params = LR.apply_lora(in_enc, int(LORA.get("rank", 8)), float(LORA.get("alpha", 16)), LORA.get("targets", LR.DEFAULT_TARGETS))
        in_enc.use_activation_checkpointing = bool(LORA.get("act_ckpt", True))
        logger.info(f"encoder LoRA (입력 경로 복사본, 타깃 encoder 는 얼림): {in_enc.lora_info} · act_ckpt {in_enc.use_activation_checkpointing} · lr_mult {LORA.get('lr_mult', 10)}")
    # ── 동결 encoder torch.compile (2026-09-22) ─────────────────────────────────
    # 실측 B=28, 48장: eager 5.8 s -> compiled(default, dynamic) 3.2 s (1.8x). 오차는 bf16 잡음 수준
    # (eager bf16 vs fp32 6.5e-2, compiled vs fp32 4.8e-2 — compiled 가 오히려 fp32 에 가깝다).
    # 첫 호출 ~17 s, (C+K) 조합마다 ~5 s 재컴파일 (16 조합). `meta.compile_encoders: false` 로 끈다.
    # ⚠️ mode=reduce-overhead(CUDA graph) 는 459 ms 로 나왔는데 A6000 peak 를 넘는 값이라 믿지 않는다.
    if bool(M.get("compile_encoders", True)):
        tgt_enc = torch.compile(tgt_enc, mode="default", dynamic=True)
        ctx_enc = tgt_enc if ctx_enc is tgt_enc else torch.compile(ctx_enc, mode="default", dynamic=True)
        logger.info("encoders: torch.compile(mode=default, dynamic=True)")
    predictor = U.build_predictor(MD, ctx_enc.embed_dim, state if MD.get("load_predictor") else None, device, n_frames)
    if MD.get("predictor_init_checkpoint"):   # 2026-10-22: 우리 학습 run 의 predictor 에서 시작 (AR post-FT). resume 이 있으면 그쪽이 덮는다
        _ck = torch.load(MD["predictor_init_checkpoint"], map_location="cpu", weights_only=False)
        _msg = predictor.load_state_dict(_ck["predictor"], strict=True)
        logger.info(f"predictor init <- {MD['predictor_init_checkpoint']} (epoch {_ck.get('epoch')}, {_msg})")
        del _ck
    # mask_mode: sample(기본) = step 마다 스펙 하나 / all = 릴리즈 사전학습처럼 스펙 전부, 손실 평균
    mask_all = str(cfg.get("mask_mode", "sample")) == "all"
    if mask_all and (kind in ("ar", "ctx_ar", "ctx_state") or D.get("window_grid")):
        raise ValueError("mask_mode=all 은 oneshot/prefix 계열 + block3d 전용이다 (ar·window_grid 불가)")
    pred_core = predictor
    del state
    tubelet, patch = int(MD.get("tubelet_size", 2)), int(MD.get("patch_size", 16))
    spatial = (res // patch) ** 2
    mask_index = int(MD.get("mask_index", 0))
    arch = {"img_size": res, "patch_size": patch, "tubelet_size": tubelet, "num_frames": n_frames,
            "encoder_embed_dim": ctx_enc.embed_dim, **(MD.get("predictor") or {})}
    # ── latent action (2026-10-24, 사용자 결정): Garrido et al. 2026 "Learning Latent Action World Models In The Wild" 이식
    #    (app/vjepa_frozen/latent_action.py). 역동역학 z_t = g(h_t, h_{t+1}) (잡음 방식, β·KL) 로 predictor 층을 변조. DDP 는 감싼 모듈 전체를.
    #    z = 0 이면 변조가 정확히 0 → 저장 · 학습 중 검증 · 채점은 안쪽 predictor 그대로 (= 추론 때 z = 0).
    LA = cfg.get("latent_action") or None
    lam_mod, lam_params = None, []
    if LA:
        if not (kind == "ar" and ar_input == "block") or mask_all:
            raise ValueError("latent_action 은 kind=ar + ar_input: block 전용이다")
        if cfg.get("state_head") or cfg.get("motion") or dyn_weight or delta_weight or HINT or tf_span != "all" or GATING:
            raise ValueError("latent_action 은 아직 state_head · motion · dyn · delta · hint · tf_span · gating 과 같이 쓰지 않는다")
        if str(((D.get("var_window") or {}).get("boundary") or {}).get("mix", "")) == "sum":
            raise ValueError("latent_action 은 아직 boundary.mix=sum 과 같이 쓰지 않는다")
        from app.vjepa_frozen import latent_action as LAM
        lam_mod = LAM.LatentActionAR(predictor, LA, ctx_enc.embed_dim).to(device)
        lam_params = list(lam_mod.idm.parameters()) + list(lam_mod.zmod.parameters())
        logger.info(f"latent_action: 역동역학 {sum(q.numel() for q in lam_mod.idm.parameters()) / 1e6:.2f}M · 변조 "
                    f"{sum(q.numel() for q in lam_mod.zmod.parameters()) / 1e6:.2f}M · z {lam_mod.z_dim} · β {LA.get('beta', 1e-4)}")
        predictor = lam_mod                     # DDP 는 이것을 감싼다 (forward 한 번에 역동역학 · TF · rollout)
    # mask token 은 매 step 하나만 쓰이므로(나머지 9개 미사용) find_unused_parameters 가 필요하다
    if mask_all:
        predictor = _MultiMaskPredictor(predictor)
    predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True) \
        if world_size > 1 else predictor
    pred_mod = pred_core            # 옵티마이저·저장·val 은 늘 안쪽 predictor (래퍼·DDP 없이)

    # ── state_head (2026-10-19, 사용자 결정, main 425d67d 이식): 합성 데이터 (phys2d_sup) 상태 라벨로 AR 출력 (TF · rollout) 에 보조 손실.
    #    정체 쿼리 48 개 attention pooling → 존재 · 위치 · 속도 head (app/vjepa_frozen/state_head.py). config `state_head:` 가 있으면 켠다.
    #    라벨은 var_window boundary 소스 (`labels: phys2d_sup`) 가 그 step 에만 싣는다 — 자연 영상 step 은 head 를 안 쓴다.
    #    predictor state_dict 밖 (채점기 strict) — 체크포인트에 `state_head` 로 따로. DDP 는 predictor 만이라 head grad 는 직접 평균.
    SH = cfg.get("state_head") or None
    state_head, head_params = None, []
    HEAD_ONLY = bool(SH and SH.get("head_only"))     # 2026-10-26: head 만 정답 토큰에서 먼저 학습하는 단계
    GM = (SH or {}).get("grad_match") or None        # 2026-10-26: head 성분 가중치를 predictor gradient 크기로 맞춤 (_matched_head)
    GMS = {}                                          # 그 비율의 EMA 상태
    if GM and not ((SH or {}).get("freeze") and (SH or {}).get("grad_probe_every")):
        raise ValueError("state_head.grad_match 는 freeze + grad_probe_every 와 같이 쓴다")
    if SH:
        if not is_ar:
            raise ValueError("state_head 는 kind=ar 전용이다")
        from app.vjepa_frozen import state_head as SHM
        state_head = SHM.build_head(SH, ctx_enc.embed_dim).to(device)        # type: attn (기본) | light (2026-10-19)
        head_params = list(state_head.parameters())
        if SH.get("init_checkpoint"):   # 2026-10-26 (사용자 결정): 정답 토큰으로 먼저 학습해 둔 head 를 불러온다
            state_head.load_state_dict(torch.load(SH["init_checkpoint"], map_location="cpu", weights_only=False)["state_head"])
            logger.info(f"state_head <- {SH['init_checkpoint']}")
        if SH.get("head_only") and SH.get("freeze"):
            raise ValueError("state_head.head_only 와 freeze 는 같이 못 쓴다")
        if SH.get("freeze"):            # 2026-10-26: head 고정 — optimizer · rank 평균 · 정답 토큰 head 손실에서 뺀다. predictor 만 head 손실을 받는다
            if SH.get("fit_on") != "gt" or not SH.get("init_checkpoint"):
                raise ValueError("state_head.freeze 는 fit_on: gt + init_checkpoint 와 같이 쓴다")
            state_head.requires_grad_(False).eval(); head_params = []
        logger.info(f"state_head ({SH.get('type', 'attn')}): {sum(q.numel() for q in head_params) / 1e6:.2f}M params · weight {SH.get('weight', 1.0)} · "
                    f"w {SH.get('w', {})} · split {SH.get('split', {})} · lr_mult {SH.get('lr_mult', 10)}")
    # ── motion (2026-10-24, 사용자 결정): MotionJEPA DISReg 이식 (app/vjepa_frozen/motion.py). 차분 영상 encoder + 차분 예측기 + SIGReg.
    #    차분 예측기는 정답 토큰 쌍에서만 학습, predictor 는 고정된 차분 예측기로 자기 예측 쌍 (TF · rollout) 에 손실. 라벨 없음 → 자연 영상 step 마다.
    #    predictor state_dict 밖 (채점기 strict) — 체크포인트에 `motion` 으로 따로. DDP 는 predictor 만이라 motion grad 는 직접 평균.
    MO = cfg.get("motion") or None
    motion_mod, motion_params = None, []
    if MO:
        if not is_ar or ar_input != "block":
            raise ValueError("motion 은 kind=ar + ar_input: block 전용이다 (블록별 정답 토큰 쌍이 필요)")
        if SH or dyn_weight or delta_weight or HINT or tf_span != "all":
            raise ValueError("motion 은 아직 state_head · dyn · delta · hint · tf_span 과 같이 쓰지 않는다")
        if str(((D.get("var_window") or {}).get("boundary") or {}).get("mix", "")) == "sum":
            raise ValueError("motion 은 아직 boundary.mix=sum 과 같이 쓰지 않는다")
        from app.vjepa_frozen import motion as MOM
        motion_mod = MOM.Motion(MO, ctx_enc.embed_dim, res, patch, tubelet).to(device)
        motion_params = list(motion_mod.parameters())
        logger.info(f"motion: 차분 encoder {sum(q.numel() for q in motion_mod.enc.parameters()) / 1e6:.2f}M · 차분 예측기 "
                    f"{sum(q.numel() for q in motion_mod.pred.parameters()) / 1e6:.2f}M · w_sigreg {MO.get('w_sigreg', 2.0)} · "
                    f"w_fit {MO.get('w_fit', 0.5)} · w_pred {MO.get('w_pred', 0.5)} · split {MO.get('split', {})} · lr_mult {MO.get('lr_mult', 1.0)}")

    # ---- 데이터 -------------------------------------------------------------------
    VW = D.get("var_window")           # 2026-10-02: 가변 창 batch 로더 (app/vjepa_frozen/var_window.py)
    dataset = None if VW else build_dataset(D, n_frames, res)
    # seed 를 주어 **rank 마다 같은 (C, K)** 가 나오게 한다 (VRAM·부하 균형. data.py MaskSampler docstring).
    sampler_masks = MaskSampler(cfg["mask"], n_frames, res, patch, tubelet, seed=seed)
    if D.get("window_grid"):
        # 채점 그리드(skip × window × 시작점 × C)에서 배치마다 하나를 뽑는다. data.n_frames 는 최대 창(RoPE 상한)
        grid = WindowGridSampler(D["window_grid"], tubelet)
        if max(c["n_frames"] for c in grid.cells) > n_frames:
            raise ValueError(f"window_grid 의 n_frames 최대가 data.n_frames({n_frames}) 보다 크다")
        collator = WindowGridCollator(grid, sampler_masks, dataset.transform)
        logger.info(f"window_grid: {grid.describe()} | raw_span {grid.raw_span} jitter {grid.jitter}")
    else:
        collator = TrainCollator(None)          # 마스크는 아래 루프가 step 당 한 번 만든다
    _restore_affinity('before build_loader')   # DataLoader worker 가 여기서 mask 를 상속한다
    if VW:
        # 로더는 resume 뒤에 만든다 (start_step 이 필요하다). ipe 는 config 로 고정해야 한다
        if not O.get("ipe"):
            raise ValueError("data.var_window 는 optimization.ipe 가 필요하다 (epoch 길이를 데이터 크기로 정할 수 없다)")
        if kind not in ("ctx_state", "ar"):
            raise ValueError("data.var_window 는 지금 kind=ctx_state / ar 만 지원한다")
        loader, sampler = None, None
        ipe = int(O["ipe"])
    else:
        loader, sampler = build_loader(dataset, collator, batch_size, rank, world_size,
                                       num_workers=int(D.get("num_workers", 8)), pin_mem=bool(D.get("pin_mem", True)),
                                       persistent=bool(D.get("persistent_workers", True)), seed=seed,
                                       prefetch_factor=int(D.get("prefetch_factor", 2)),
                                       timeout_s=float(D.get("loader_timeout_s", 0)))
        ipe = int(O.get("ipe") or len(loader))
    if VW:
        pass
    elif len(loader) == 0 or ipe <= 0:
        raise RuntimeError(f"loader 가 비었다 (ipe={ipe}, loader {len(loader)}): {len(dataset)} clips / ws {world_size} / "
                           f"batch {batch_size} (drop_last) — LIMIT 이나 batch_size 를 조정할 것")
    if not VW:
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
    if state_head is not None:      # head 그룹: lr = 스케줄 lr × lr_mult (WSDSchedule 의 lr_scale), 1-D 는 weight decay 없음
        _hm = float(SH.get("lr_mult", 10))
        optimizer.add_param_group({"params": [q for q in head_params if q.ndim > 1], "lr": float(O["lr"]), "lr_scale": _hm})
        optimizer.add_param_group({"params": [q for q in head_params if q.ndim <= 1], "lr": float(O["lr"]), "lr_scale": _hm,
                                   "WD_exclude": True, "weight_decay": 0.0})
    if lora_params:                 # encoder LoRA 그룹: lr = 스케줄 lr × lr_mult (main 과 같은 10 배), weight decay 없음
        optimizer.add_param_group({"params": lora_params, "lr": float(O["lr"]), "lr_scale": float(LORA.get("lr_mult", 10)),
                                   "WD_exclude": True, "weight_decay": 0.0})
    if lam_mod is not None:         # latent action 그룹 (역동역학 · 변조): predictor 와 같은 일정, 1-D 는 weight decay 없음
        optimizer.add_param_group({"params": [q for q in lam_params if q.ndim > 1], "lr": float(O["lr"]), "lr_scale": float(LA.get("lr_mult", 1.0))})
        optimizer.add_param_group({"params": [q for q in lam_params if q.ndim <= 1], "lr": float(O["lr"]), "lr_scale": float(LA.get("lr_mult", 1.0)),
                                   "WD_exclude": True, "weight_decay": 0.0})
    if motion_mod is not None:      # motion 그룹: lr = 스케줄 lr × lr_mult, 1-D 는 weight decay 없음
        _mm = float(MO.get("lr_mult", 1.0))
        optimizer.add_param_group({"params": [q for q in motion_params if q.ndim > 1], "lr": float(O["lr"]), "lr_scale": _mm})
        optimizer.add_param_group({"params": [q for q in motion_params if q.ndim <= 1], "lr": float(O["lr"]), "lr_scale": _mm,
                                   "WD_exclude": True, "weight_decay": 0.0})

    # ---- val ----------------------------------------------------------------------
    valset = None
    if V and V.get("data"):
        valset = ValSet(V, int(V.get("n_frames", n_frames)), res)   # var_window 는 data.n_frames 가 최대 창이라 val 은 따로 준다
        if rank == 0:
            logger.info(f"val: {len(valset)} clips from {V['data']['root']} (n_blocks_per_type {V.get('n_blocks_per_type', 8)})")

    # ---- resume -------------------------------------------------------------------
    latest = os.path.join(folder, "latest.pt")
    start_epoch, step = 0, 0
    if (auto_resume or resume_preempt) and os.path.exists(latest):
        start_epoch, step = U.load_resume(latest, pred_mod, optimizer, scaler)
        if state_head is not None:
            _ck = torch.load(latest, map_location="cpu", weights_only=False)
            if "state_head" not in _ck:
                raise KeyError(f"{latest} 에 state_head 가 없다 — config 와 run 이 맞는지 확인")
            state_head.load_state_dict(_ck["state_head"]); del _ck
            logger.info("resume: state_head <- latest.pt")
        if lora_params:
            _ck = torch.load(latest, map_location="cpu", weights_only=False)
            if "encoder_lora" not in _ck:
                raise KeyError(f"{latest} 에 encoder_lora 가 없다 — config 와 run 이 맞는지 확인")
            LR.load_lora(in_enc, _ck["encoder_lora"]); del _ck
            logger.info("resume: encoder LoRA <- latest.pt")
        if lam_mod is not None:
            _ck = torch.load(latest, map_location="cpu", weights_only=False)
            if "latent_action" not in _ck:
                raise KeyError(f"{latest} 에 latent_action 이 없다 — config 와 run 이 맞는지 확인")
            lam_mod.idm.load_state_dict(_ck["latent_action"]["idm"]); lam_mod.zmod.load_state_dict(_ck["latent_action"]["zmod"]); del _ck
            logger.info("resume: latent_action <- latest.pt")
        if motion_mod is not None:
            _ck = torch.load(latest, map_location="cpu", weights_only=False)
            if "motion" not in _ck:
                raise KeyError(f"{latest} 에 motion 이 없다 — config 와 run 이 맞는지 확인")
            motion_mod.load_state_dict(_ck["motion"]); del _ck
            logger.info("resume: motion <- latest.pt")
        for _ in range(start_epoch * ipe):
            scheduler.step(); wd_scheduler.step()
    elif os.path.exists(latest):
        raise RuntimeError(f"{latest} 가 있는데 meta.auto_resume 이 false 다. 새 NAME 을 쓰거나 폴더를 비울 것")
    elif O.get("is_anneal"):
        # 2026-10-01: 릴리즈 cooldown 과 같은 규약 (app/vjepa/train.py: is_anneal + anneal_ckpt) — 새 cooldown run 은
        # 앞 단계 run 의 predictor 가중치에서 시작한다 (옵티마이저·스케줄은 새로). 이어 돌리기(latest.pt 있음)는 위 분기가 먼저 탄다
        ak = O.get("anneal_ckpt")
        if not ak or not os.path.isfile(ak):
            raise FileNotFoundError(f"optimization.is_anneal 인데 anneal_ckpt 가 없다: {ak!r}")
        sd = torch.load(ak, map_location="cpu", weights_only=False)["predictor"]
        pred_mod.load_state_dict(sd, strict=True)
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
        if is_seq:
            if is_ar:
                res = AR.run_val_ar(valset, tgt_enc, pred_mod, device, ac_dtype, rank, world_size, tubelet, spatial,
                                    int(V.get("context_length", 16)), int(V.get("batch_size", 8)), loss_exp,
                                    ar_input=ar_input, in_enc=in_enc)
            elif is_ctx_state:
                res = AR.run_val_ctx_state(valset, ctx_enc, tgt_enc, pred_mod, device, ac_dtype, rank, world_size, tubelet,
                                           spatial, int(V.get("context_length", 16)), int(V.get("batch_size", 8)), loss_exp,
                                           ctx_target=ctx_target, gating=GATING, ctx_input=ctx_input)   # 문맥 encoder 는 ctx_enc (ctx_encoder 반영)
            else:
                res = AR.run_val_ctx_ar(valset, ctx_enc, tgt_enc, pred_mod, device, ac_dtype, rank, world_size, tubelet,
                                        spatial, int(V.get("context_length", 16)), int(V.get("batch_size", 8)), loss_exp)
            if rank == 0:
                logger.info(f"[epoch {epoch}] {AR.format_val_ar(res)}")
            return res
        res = run_val(valset, ctx_enc, tgt_enc, pred_mod if mask_all else predictor, device, ac_dtype, rank, world_size, tubelet, spatial,
                      int(V.get("context_length", 16)), int(V.get("batch_size", 8)), mask_index, loss_exp, target_ln)
        if rank == 0:
            logger.info(f"[epoch {epoch}] {format_val(res)}")
        return res

    if rank == 0:
        logger.info(f"predictor trainable {U.count_parameters(pred_mod)/1e6:.2f}M | load_predictor={bool(MD.get('load_predictor'))} "
                    f"| autocast={ac_dtype} enc_dtype={enc_dtype} | lr {O['lr']} wd {O.get('weight_decay')} epochs {num_epochs}"
                    + (f" | AR: 블록별 인코딩, loss = tf + rollout(R<={rollout_steps})" if is_ar else "")
                    + (f" | CTX_AR: 문맥 창 단위 z + 미래 블록별 LN(h), loss = tf + rollout(R<={rollout_steps})" if is_ctx_ar else "")
                    + (f" | CTX_STATE: 문맥 z + TF 관측(블록별 LN(h)) / rollout 상태 되먹임, 타깃 = 창 LN(h), loss = tf + rollout(R<={rollout_steps})" if is_ctx_state else ""))
    # ★ 모든 rank 가 들어가야 한다 — run_val 안의 all_gather_object 는 collective 라 rank 0 만 부르면 나머지가 영원히 기다린다
    if start_epoch == 0 and V and V.get("at_start", True):
        _val(0)

    # ---- 학습 루프 ----------------------------------------------------------------
    MIX = False
    if VW:
        from app.vjepa_frozen.var_window import build_var_window_loader
        aug = D.get("aug") or {}
        vw_tf = ClipTransform(res, hflip_p=aug.get("hflip_p", 0.0), random_resized_crop=aug.get("random_resized_crop"),
                              square_crop=aug.get("square_crop", "none"), out_uint8=True)
        # 2026-10-20 (사용자): boundary.mix = sum — 합성을 step 교대가 아니라 보조 손실처럼 **매 step** 자연 batch 와 함께.
        #   step 교대는 합성 step 의 큰 gradient 가 Adam 2 차 모멘트를 키워 자연 step 갱신을 줄인다 (합성 grad 5 배 · 20 % → 약 1/2.4).
        #   자연 로더 = boundary 없음, 합성 로더 = boundary p 1 · token 예산 × budget_frac. 둘의 손실을 더해 한 번에 step.
        BX = (VW or {}).get("boundary") or {}
        MIX = str(BX.get("mix", "")) == "sum"
        _ld = dict(num_workers=int(D.get("num_workers", 8)), pin_mem=bool(D.get("pin_mem", True)),
                   prefetch_factor=int(D.get("prefetch_factor", 2)), timeout_s=float(D.get("loader_timeout_s", 0)))
        if MIX:
            VW_nat = {**VW, "boundary": None}
            VW_syn = {**VW, "boundary": {**BX, "p": 1.0}, "token_budget": max(1, int(int(VW["token_budget"]) * float(BX.get("budget_frac", 0.25))))}
            vw_ds, loader = build_var_window_loader(VW_nat, vw_tf, tubelet, seed, rank, start_epoch * ipe, **_ld)
            syn_ds, syn_loader = build_var_window_loader(VW_syn, vw_tf, tubelet, seed + 7919, rank, start_epoch * ipe, **_ld)
            syn_it = iter(syn_loader)
            if rank == 0:
                logger.info(f"boundary mix=sum: 매 step 자연 batch + 합성 batch (token 예산 {syn_ds.budget}, 합성 latent ×{BX.get('latent_weight', 1.0)} · 상태 ×{(SH or {}).get('weight', 1.0)})")
        else:
            vw_ds, loader = build_var_window_loader(VW, vw_tf, tubelet, seed, rank, start_epoch * ipe, **_ld)
        if rank == 0:
            logger.info(f"var_window: ipe {ipe}, token_budget {vw_ds.budget}, 조합 {len(vw_ds.combos)}개, 시작 step {start_epoch * ipe}\n"
                        + vw_ds.describe())
    else:
        sampler.set_epoch(start_epoch)
    loader_it = iter(loader)
    lr_now = wd_now = float("nan")
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        loss_meter, it_meter, gpu_meter, data_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        jl_meter, sl_meter = AverageMeter(), AverageMeter()                       # 팔 C 전용 (tf / rollout)
        for itr in range(ipe):
            t_it = time.time()
            if VW:                               # 가변 창: 로더가 (C, K, 데이터셋, B) 를 정해 W 장 정확히 준다 (자르지 않는다)
                clips, info = next(loader_it)
                if info.get("starts") is None:   # multistart 는 마스크를 안 쓴다
                    m_enc, m_pred = sampler_masks.prefix_window(clips.size(0), info["context_blocks"], info["predict_blocks"])
            else:
                try:
                    clips, m_enc, m_pred, info = next(loader_it)
                except StopIteration:
                    sampler.set_epoch(epoch + 1)
                    loader_it = iter(loader)
                    clips, m_enc, m_pred, info = next(loader_it)
            if VW:
                pass
            elif m_enc is None and mask_all:       # 릴리즈 사전학습: 스펙 전부 (창 = n_frames 전체, 자르지 않는다)
                mlist = sampler_masks.all(clips.size(0))
                m_enc = [e.to(device, non_blocking=True) for e, _, _ in mlist]
                m_pred = [p_.to(device, non_blocking=True) for _, p_, _ in mlist]
                info = {"type": "+".join(i["type"] for _, _, i in mlist), "context_frames": -1,
                        "mask_index": [i["mask_index"] for _, _, i in mlist]}
            elif m_enc is None:                  # TrainCollator(None) 경로 — rank 동기 마스크
                m_enc, m_pred, info = sampler_masks(clips.size(0))
                # ── 창을 C+K 블록으로 자른다 (2026-09-22) ─────────────────────
                # 안 자르면 target encoder 가 **늘 n_frames 전부**(24 블록, 6144 토큰)를 돈다.
                # 쓰는 건 m_pred (K 블록, 평균 3.75) 뿐이라 대부분 버려진다 — E[C+K]=11.25
                # 이므로 평균 2.1 배 낭비다. 게다가 예측 구간 **밖** 프레임까지 타깃에 섞여
                # 예측 불가능한 성분이 늘어(손실 바닥이 올라간다), 채점과도 어긋난다 —
                # 채점은 창 = 문맥+미래 로 `LN(tgt_enc(window))` 를 쓴다. 자르는 쪽이 맞다.
                # 시작 블록을 무작위로 잡아 **창이 영상 위를 미끄러지는 효과**도 같이 얻는다.
                # ⚠️ 토큰 수로 역산하지 않는다 (2026-09-27): block3d 는 min_keep 절단으로 enc+pred < N 이라
                #    역산하면 창이 한 블록 짧아져 pred 인덱스가 밖으로 나간다. 샘플러가 win_blocks 로 알려 준다.
                n_blk = int(info.get("win_blocks") or (m_enc.size(1) + m_pred.size(1)) // sampler_masks.S)
                if n_blk < sampler_masks.T:
                    o = sampler_masks.rng.randint(0, sampler_masks.T - n_blk)
                    clips = clips[:, :, o * tubelet: (o + n_blk) * tubelet]
                    info = {**info, "win_blocks": n_blk, "win_offset": o}
            clips = normalize_clips(clips.to(device, non_blocking=True))   # uint8 전송 -> GPU 정규화
            if MIX:                                                          # 2026-10-20: 보조 합성 batch (mix=sum)
                clips_s, info_s = next(syn_it)
                clips_s = normalize_clips(clips_s.to(device, non_blocking=True))
            if not mask_all:
                m_enc = m_enc.to(device, non_blocking=True)
                m_pred = m_pred.to(device, non_blocking=True)
            data_ms = (time.time() - t_it) * 1000

            aux = {}

            def train_step_ar():
                # 팔 C (2026-09-26): 블록별 인코딩 -> LN -> predictor 한 forward 로 (p_tf, p_ar) -> jloss + sloss
                lr_now = scheduler.step()
                wd_now = wd_scheduler.step()
                if int(info.get("gap_blocks", 0)):
                    raise ValueError("kind=ar/ctx_ar 는 gap_blocks 를 아직 지원하지 않는다 (예측이 문맥 바로 뒤에서 시작한다고 가정)")
                if info.get("starts") is not None:              # 2026-10-14: multistart (C·K 없음, 시작점 여러 개)
                    if HINT or delta_weight or tf_span != "all":
                        raise ValueError("multistart 는 hint · delta_weight · tf_span 과 아직 같이 쓰지 않는다")
                    with torch.no_grad(), _ac():
                        h = AR.encode_blocks(tgt_enc, clips.to(enc_dtype), tubelet)
                    with _ac():
                        loss, jl, sl, R = AR.ar_losses_multistart(predictor, h, spatial, info["starts"], info["rollouts"], loss_exp)
                    if scaler is not None:
                        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    if clip_grad:
                        torch.nn.utils.clip_grad_norm_(pred_mod.parameters(), float(clip_grad))
                    if scaler is not None:
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    aux.update(jloss=float(jl.detach()), sloss=float(sl.detach()), R=R)
                    return float(loss.detach()), lr_now, wd_now
                if "context_blocks" not in info:
                    raise ValueError("kind=ar 는 mask.type=prefix_window 만 지원한다 (C·K 블록 수가 필요)")
                if dyn_weight:                                  # 2026-10-14: 움직임 크기 손실 (TF + rollout 이웃 쌍)
                    if HINT or delta_weight:
                        raise ValueError("dyn_weight 는 hint · delta_weight 와 아직 같이 쓰지 않는다")
                    with torch.no_grad(), _ac():
                        xe = clips.to(enc_dtype)
                        h = AR.encode_blocks(tgt_enc, xe, tubelet)
                        pe = AR.patch_embed_blocks(tgt_enc, xe, tubelet)
                    with _ac():
                        loss, jl, sl, R = AR.ar_losses_dyn(predictor, h, pe, spatial, int(info["context_blocks"]),
                                                           int(info["predict_blocks"]), loss_exp, rollout_steps, dyn_weight, aux=aux)
                    if scaler is not None:
                        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    if clip_grad:
                        torch.nn.utils.clip_grad_norm_(pred_mod.parameters(), float(clip_grad))
                    if scaler is not None:
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    aux.update(jloss=float(jl.detach()), sloss=float(sl.detach()), R=R)
                    return float(loss.detach()), lr_now, wd_now
                hint, n_hint = None, 0
                if AR_VIDEO:   # 2026-10-18: 문맥 pass · 전체 pass (target_encoder 두 번), TF 문맥 자리만 + rollout
                    C_, K_ = int(info["context_blocks"]), int(info["predict_blocks"])
                    with torch.no_grad(), _ac():
                        zc, zf = AR.encode_video_ctx_full(tgt_enc, clips.to(enc_dtype)[:, :, : (C_ + K_) * tubelet], C_, spatial)
                    with _ac():
                        loss, jl, sl, R = AR.ar_losses_video(predictor, zc, zf, spatial, C_, K_, loss_exp, rollout_steps, aux=aux)
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
                    aux.update(jloss=float(jl.detach()), sloss=float(sl.detach()), R=R, n_hint=0)
                    return float(loss.detach()), lr_now, wd_now
                with torch.no_grad(), _ac():
                    h = AR.encode_blocks(tgt_enc, clips.to(enc_dtype), tubelet)                # (B, nb*S, D) LN — 타깃 (LoRA 판에서도 얼린 encoder)
                    if HINT:                                                                   # 정해진 일정, 0 개면 cross-attn 생략
                        n_hint = AR.hint_count(step / max(1, num_epochs * ipe), HINT, spatial)
                        if n_hint:
                            # 힌트 위치 = (seed, rank, step) 전용 생성기 — rank 마다 다르고 이어 돌려도 재현된다 (2026-10-12)
                            g = torch.Generator(device=clips.device).manual_seed(int(seed) * 10**9 + rank * 10**7 + int(step))
                            hint = AR.encode_hints(tgt_enc, clips.to(enc_dtype), tubelet, spatial, n_hint, gen=g)
                if HEAD_ONLY:   # 2026-10-26 (사용자 결정): head 먼저 — 정답 토큰 h 에서 head 만 학습. predictor 는 forward 도 안 한다
                    lab = info.get("labels")
                    if lab is None or scaler is not None:
                        raise ValueError("state_head.head_only: 라벨 있는 합성 step 만 (boundary.p 1.0 · mix 없음) · bf16")
                    with _ac():
                        lh, hlog = SHM.gt_state_loss(state_head, h, SHM.labels_to_device(lab, device), spatial, SH.get("w") or {})
                    lh.backward()
                    if world_size > 1:
                        for q in head_params:
                            if q.grad is not None:
                                torch.distributed.all_reduce(q.grad); q.grad.div_(world_size)
                    aux["gn"] = float(torch.nn.utils.clip_grad_norm_(head_params, float(SH.get("clip_head") or float("inf"))))
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    aux.update(head=float(lh.detach()), hlog=hlog, jloss=float(lh.detach()), sloss=0.0, R=0, n_hint=0)
                    return float(lh.detach()), lr_now, wd_now
                with _ac():
                    lab = info.get("labels") if state_head is not None else None    # 2026-10-19: 합성 (phys2d_sup) step 에만 있다
                    outs = {} if (lab is not None or motion_mod is not None) else None
                    if lam_mod is not None:        # 2026-10-24: latent action — 정답 쌍 z 로 변조한 TF + rollout L1 + β·KL
                        loss, jl, sl, R, aux["lam"] = LAM.lam_losses(predictor, h, spatial, int(info["context_blocks"]),
                                                                     int(info["predict_blocks"]), loss_exp, rollout_steps, float(LA.get("beta", 1e-4)))
                    else:
                        h_in = AR.encode_blocks(in_enc, clips.to(enc_dtype), tubelet) if in_enc is not None else None   # 2026-10-25: LoRA 입력 (grad)
                        loss, jl, sl, R = AR.ar_losses(predictor, h, spatial, int(info["context_blocks"]),
                                                       int(info["predict_blocks"]), loss_exp, rollout_steps, hint=hint,
                                                       delta_weight=delta_weight, aux=aux, tf_span=tf_span, out=outs, h_in=h_in)
                    if motion_mod is not None:     # 2026-10-24: DISReg 이식 — ① 차분 encoder · 예측기 맞추기 ② 고정 예측기로 predictor 손실
                        lm, mlog = MOM.motion_losses(motion_mod, clips, h, outs["p_tf"], outs["p_ar"], spatial,
                                                     int(info["context_blocks"]), R, MO)
                        aux["latent"] = float(loss.detach()); aux["mlog"] = mlog
                        loss = loss + lm
                    if lab is not None:
                        if SH.get("fit_on") == "gt":     # 2026-10-24: head 는 정답 토큰에서만, predictor 는 고정 head 로 (SHM.ar_state_loss_gt)
                            _parts = {} if (GM or (SH.get("grad_probe_every") and step % int(SH["grad_probe_every"]) == 0)) else None
                            lh, hlog = SHM.ar_state_loss_gt(state_head, h, outs["p_tf"], outs["p_ar"], SHM.labels_to_device(lab, device),
                                                            spatial, int(info["context_blocks"]), R, SH.get("w") or {}, SH.get("split") or {},
                                                            fit_gt=not SH.get("freeze"), parts=_parts)
                        else:
                            lh, hlog = SHM.ar_state_loss(state_head, outs["p_tf"], outs["p_ar"], SHM.labels_to_device(lab, device),
                                                         spatial, int(info["context_blocks"]), R, SH.get("w") or {}, SH.get("split") or {})
                        aux["latent"] = float(loss.detach()); aux["head"] = float(lh.detach()); aux["hlog"] = hlog
                        if GM:      # 2026-10-26: 성분 가중치를 gradient 크기로 맞춘 head 손실
                            _lh, aux["hw"], _gp = _matched_head(pred_mod, _parts, loss, GM, GMS, bool(SH.get("grad_probe_every") and step % int(SH["grad_probe_every"]) == 0), world_size)
                            if _gp:
                                aux["gprobe"] = _gp
                            aux["head_m"] = float(_lh.detach())
                            loss = loss + _lh
                        else:
                            if SH.get("grad_probe_every") and step % int(SH["grad_probe_every"]) == 0:
                                _W = float(SH.get("weight", 1.0))
                                aux["gprobe"] = _grad_probe(pred_mod, {"lat": loss, "head": _W * lh, **{k: _W * v for k, v in (locals().get("_parts") or {}).items()}})
                            loss = loss + float(SH.get("weight", 1.0)) * lh
                loss_nat = None
                if MIX:     # 2026-10-20: 자연 손실은 DDP 동기화 없이 gradient 만 쌓고, 합성 손실 backward 때 합쳐서 한 번 동기화
                    if scaler is not None:
                        raise ValueError("boundary.mix=sum 은 bf16 (scaler 없음) 에서만")
                    if SH and SH.get("grad_probe_every") and step % int(SH["grad_probe_every"]) == 0:   # 2026-10-26: 자연 영상 latent 손실의 predictor grad
                        aux["gprobe_nat"] = {"nat": _grad_probe(pred_mod, {"lat": loss})["lat"]}
                    with (predictor.no_sync() if hasattr(predictor, "no_sync") else contextlib.nullcontext()):
                        loss.backward()
                    loss_nat = loss.detach()
                    with torch.no_grad(), _ac():
                        hs = AR.encode_blocks(tgt_enc, clips_s.to(enc_dtype), tubelet)
                    with _ac():
                        outs_s = {}
                        ls, jls, sls, Rs = AR.ar_losses(predictor, hs, spatial, int(info_s["context_blocks"]), int(info_s["predict_blocks"]),
                                                        loss_exp, rollout_steps, tf_span=tf_span, out=outs_s)
                        lab = info_s.get("labels") if state_head is not None else None
                        loss = float(BX.get("latent_weight", 1.0)) * ls
                        aux["syn_latent"] = float(ls.detach())
                        if lab is not None:
                            if SH.get("fit_on") == "gt":     # 2026-10-24
                                _parts = {} if (GM or (SH.get("grad_probe_every") and step % int(SH["grad_probe_every"]) == 0)) else None
                                lh, hlog = SHM.ar_state_loss_gt(state_head, hs, outs_s["p_tf"], outs_s["p_ar"], SHM.labels_to_device(lab, device),
                                                                spatial, int(info_s["context_blocks"]), Rs, SH.get("w") or {}, SH.get("split") or {},
                                                                fit_gt=not SH.get("freeze"), parts=_parts)
                            else:
                                lh, hlog = SHM.ar_state_loss(state_head, outs_s["p_tf"], outs_s["p_ar"], SHM.labels_to_device(lab, device),
                                                             spatial, int(info_s["context_blocks"]), Rs, SH.get("w") or {}, SH.get("split") or {})
                            aux["latent"] = float(ls.detach()); aux["head"] = float(lh.detach()); aux["hlog"] = hlog
                            if GM:      # 2026-10-26: 성분 가중치를 gradient 크기로 맞춘 head 손실 (기준 = 합성 latent)
                                _lh, aux["hw"], _gp = _matched_head(pred_mod, _parts, float(BX.get("latent_weight", 1.0)) * ls, GM, GMS,
                                                                    bool(SH.get("grad_probe_every") and step % int(SH["grad_probe_every"]) == 0), world_size)
                                if _gp:
                                    aux["gprobe"] = {**aux.pop("gprobe_nat", {}), **_gp}
                                aux["head_m"] = float(_lh.detach())
                                loss = loss + _lh
                            else:
                                if SH.get("grad_probe_every") and step % int(SH["grad_probe_every"]) == 0:
                                    _W = float(SH.get("weight", 1.0))
                                    aux["gprobe"] = {**aux.pop("gprobe_nat", {}), **_grad_probe(pred_mod, {"lat": float(BX.get("latent_weight", 1.0)) * ls, "head": _W * lh,
                                                                                                           **{k: _W * v for k, v in (_parts or {}).items()}})}
                                loss = loss + float(SH.get("weight", 1.0)) * lh
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if head_params and world_size > 1 and lab is not None:       # 모든 rank 가 같은 step 종류 ((seed, step) 로 정해진다)
                    for q in head_params:
                        if q.grad is not None:
                            torch.distributed.all_reduce(q.grad); q.grad.div_(world_size)
                if lora_params:      # 2026-10-25: encoder LoRA — DDP 는 predictor 만이라 rank 평균은 직접 (한 번에 묶어서). clip 은 predictor 와 같이
                    _g = [q.grad if q.grad is not None else torch.zeros_like(q) for q in lora_params]
                    if world_size > 1:
                        _flat = torch.cat([g.reshape(-1) for g in _g])
                        torch.distributed.all_reduce(_flat); _flat.div_(world_size)
                        _o = 0
                        for q, g in zip(lora_params, _g):
                            q.grad = _flat[_o: _o + g.numel()].view_as(q).clone(); _o += g.numel()
                    aux["lgn"] = float(torch.nn.utils.clip_grad_norm_(lora_params, float("inf")))
                    if clip_grad:
                        torch.nn.utils.clip_grad_norm_(list(pred_mod.parameters()) + lora_params, float(clip_grad))
                if lam_params:       # 2026-10-24: latent action — grad 동기화는 DDP 가 한다. grad norm 은 로그만 (predictor · 역동역학/변조)
                    aux["gn"] = float(torch.nn.utils.clip_grad_norm_(pred_mod.parameters(), float(clip_grad) if clip_grad else float("inf")))
                    aux["lgn"] = float(torch.nn.utils.clip_grad_norm_(lam_params, float(LA.get("clip", 0)) or float("inf")))
                if motion_params:    # 2026-10-24: motion grad — rank 평균 (한 번에 묶어서), 따로 clip, predictor grad norm 은 로그만
                    _g = [q.grad for q in motion_params]
                    if any(g is None for g in _g):
                        raise RuntimeError("motion 파라미터 중 grad 가 없는 것이 있다 — rank 간 all_reduce 크기가 어긋난다")
                    if world_size > 1:
                        _flat = torch.cat([g.reshape(-1) for g in _g])
                        torch.distributed.all_reduce(_flat); _flat.div_(world_size)
                        _o = 0
                        for g in _g:
                            g.copy_(_flat[_o: _o + g.numel()].view_as(g)); _o += g.numel()
                    aux["mgn"] = float(torch.nn.utils.clip_grad_norm_(motion_params, float(MO.get("clip", 1.0))))
                    aux["gn"] = float(torch.nn.utils.clip_grad_norm_(pred_mod.parameters(), float(clip_grad) if clip_grad else float("inf")))
                elif head_params and SH.get("clip_separate"):   # 2026-10-19: clip 은 predictor 에만 (gn = predictor norm). head 는 state_head.clip_head (null = 안 자름)
                    _gn = torch.nn.utils.clip_grad_norm_(pred_mod.parameters(), float(clip_grad) if clip_grad else float("inf"))
                    if SH.get("clip_head"):
                        torch.nn.utils.clip_grad_norm_(head_params, float(SH["clip_head"]))
                    aux["gn"] = float(_gn)
                elif clip_grad or head_params:   # 2026-10-19: head 가 있으면 grad norm 을 늘 잰다 (clip 전 값, 로그 gn). 없으면 기존과 같다
                    _gn = torch.nn.utils.clip_grad_norm_(list(pred_mod.parameters()) + head_params, float(clip_grad) if clip_grad else float("inf"))
                    if head_params:
                        aux["gn"] = float(_gn)
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                aux.update(jloss=float(jl.detach()), sloss=float(sl.detach()), R=R, n_hint=n_hint)
                return float(loss.detach() + (loss_nat if loss_nat is not None else 0.0)), lr_now, wd_now

            def train_step_ctx_ar():
                # 팔 C' (2026-09-27): 문맥 = ctx encoder 가 C 블록 창을 한 번에 (m_enc 로 미래 토큰 제거, prefix 팔과 같다),
                # 미래 GT = target encoder 블록별 -> LN. predictor 한 forward 로 (p_tf, p_ar) -> jloss + sloss
                lr_now = scheduler.step()
                wd_now = wd_scheduler.step()
                if int(info.get("gap_blocks", 0)):
                    raise ValueError("kind=ar/ctx_ar 는 gap_blocks 를 아직 지원하지 않는다 (예측이 문맥 바로 뒤에서 시작한다고 가정)")
                if "context_blocks" not in info:
                    raise ValueError("kind=ctx_ar 는 mask.type=prefix_window 만 지원한다")
                C_, K_ = int(info["context_blocks"]), int(info["predict_blocks"])
                with torch.no_grad(), _ac():
                    xe = clips.to(enc_dtype)
                    z = ctx_enc(xe, masks=[m_enc])                                              # (B, C*S, D) 문맥 z
                    h_fut = AR.encode_blocks(tgt_enc, xe[:, :, C_ * tubelet:], tubelet)          # (B, K*S, D) LN
                with _ac():
                    loss, jl, sl, R = AR.ctx_ar_losses(predictor, z, h_fut, spatial, C_, K_, loss_exp, rollout_steps)
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
                aux.update(jloss=float(jl.detach()), sloss=float(sl.detach()), R=R)
                return float(loss.detach()), lr_now, wd_now

            def train_step_ctx_state():
                # 2026-09-30: ctx_ar 와 같은 입력 + 상태 되먹임. 타깃만 창 전체 LN(h) (TF 관측 입력은 블록별 — 누수 방지)
                lr_now = scheduler.step()
                wd_now = wd_scheduler.step()
                if int(info.get("gap_blocks", 0)):
                    raise ValueError("kind=ar/ctx_ar 는 gap_blocks 를 아직 지원하지 않는다 (예측이 문맥 바로 뒤에서 시작한다고 가정)")
                if "context_blocks" not in info:
                    raise ValueError("kind=ctx_state 는 mask.type=prefix_window 만 지원한다")
                C_, K_ = int(info["context_blocks"]), int(info["predict_blocks"])
                hole = keep = None
                if CTX_HOLE:   # 적용 여부 · k = (seed, step) 전용 생성기 (rank 공통 → step 시간 균형), 직사각형 = (seed, step, rank) 생성기 (rank 마다 다른 구멍)
                    gh = torch.Generator().manual_seed(int(seed) * 10**9 + int(step) * 1000 + 17)
                    gs = torch.Generator().manual_seed(int(seed) * 10**9 + int(step) * 1000 + 17 + 100 * (int(rank) + 1))
                    hole, keep = AR.make_ctx_hole(clips.size(0), C_, spatial, CTX_HOLE, gh, gs)
                with torch.no_grad(), _ac():
                    xe = clips.to(enc_dtype)
                    SHARED = CTX_BLOCK and ctx_enc is tgt_enc and ctx_target == "block"
                    if SHARED:         # 2026-10-17: 문맥 · 타깃 모두 블록별 LN(target_encoder) — 한 번에 (블록마다 따로라 값이 같다)
                        hb = AR.encode_blocks(tgt_enc, xe[:, :, : (C_ + K_) * tubelet], tubelet)   # (B, (C+K)*S, D)
                        z = hb[:, : C_ * spatial]
                    elif CTX_BLOCK:    # 문맥만 블록별 LN(ctx_enc) (예: online encoder 를 블록 단위로)
                        z = AR.encode_blocks(ctx_enc, xe[:, :, : C_ * tubelet], tubelet)
                    elif hole is None:
                        z = ctx_enc(xe, masks=[m_enc])                                          # (B, C*S, D) 문맥 z
                    else:      # 2026-10-16: encoder 는 보이는 토큰만 본다 (V-JEPA 2 사전학습과 같다). 가린 자리 z = 0 (predictor 가 mask token 으로 바꾼다)
                        hole, keep = hole.to(xe.device), keep.to(xe.device)
                        zk = ctx_enc(xe, masks=[keep])                                          # (B, n_keep, D)
                        z = zk.new_zeros(zk.size(0), C_ * spatial, zk.size(-1))
                        z.scatter_(1, keep.unsqueeze(-1).expand(-1, -1, zk.size(-1)), zk)
                    tgt_ctx = anchor = None
                    if SHARED:
                        tgt = hb[:, C_ * spatial:]
                        h_obs = tgt if tf_weight > 0 else None
                        anchor = z[:, -spatial:] if GATING else None
                    elif ctx_target == "block" and GATING:                                         # 2026-10-17: anchor (마지막 문맥 블록) 를 미래 블록과 한 번에 (블록마다 따로 인코딩이라 값이 같다)
                        hb = AR.encode_blocks(tgt_enc, xe[:, :, (C_ - 1) * tubelet:], tubelet)   # (B, (1+K)*S, D)
                        anchor, tgt = hb[:, :spatial], hb[:, spatial:]
                        h_obs = tgt if tf_weight > 0 else None
                    elif ctx_target == "block":                                                  # 2026-10-16: 타깃 = TF 관측과 같은 블록별 LN
                        tgt = AR.encode_blocks(tgt_enc, xe[:, :, C_ * tubelet:], tubelet)        # (B, K*S, D)
                        h_obs = tgt if tf_weight > 0 else None
                        if hole is not None and float((CTX_HOLE or {}).get("recon_weight", 1.0)) > 0:   # 2026-10-17: 복원 정답 = 가린 문맥 블록들만 블록별 LN(target_encoder). 복원 손실이 없으면 안 만든다
                            kb = int(hole.view(hole.size(0), C_, spatial)[0].any(-1).sum())        # 구멍 블록 수 (step 공통, 문맥 끝)
                            tgt_ctx = z.new_zeros(z.size(0), C_ * spatial, tgt.size(-1))
                            tgt_ctx[:, (C_ - kb) * spatial:] = AR.encode_blocks(tgt_enc, xe[:, :, (C_ - kb) * tubelet: C_ * tubelet], tubelet)
                    else:
                        h_obs = (AR.encode_blocks(tgt_enc, xe[:, :, C_ * tubelet:], tubelet)     # (B, K*S, D) 블록별 LN (TF 관측)
                                 if tf_weight > 0 else None)
                        tgt_all = torch.nn.functional.layer_norm(tgt_enc(xe), (z.size(-1),))                    # 창 LN
                        tgt, tgt_ctx = tgt_all[:, C_ * spatial:], tgt_all[:, : C_ * spatial]                     # (B, K*S, D) 타깃 · 문맥 자리 (구멍 복원)
                with _ac():
                    loss, jl, sl, R = AR.ctx_state_losses(predictor, z, h_obs, tgt, spatial, C_, K_, loss_exp, rollout_steps,
                                                          tf_weight, hole=hole, tgt_ctx=tgt_ctx,
                                                          recon_weight=float((CTX_HOLE or {}).get("recon_weight", 1.0)), aux=aux,
                                                          anchor=anchor)
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
                aux.update(jloss=float(jl.detach()), sloss=float(sl.detach()), R=R)
                return float(loss.detach()), lr_now, wd_now

            def train_step_all():
                # 릴리즈 사전학습 (app/vjepa/train.py): target 은 창 전체 1 회, 마스크마다 context·predictor, 손실 평균
                lr_now = scheduler.step()
                wd_now = wd_scheduler.step()
                with torch.no_grad(), _ac():
                    xe = clips.to(enc_dtype)
                    h_full = tgt_enc(xe)                                              # (B, N, D)
                    hs = []
                    for mp_ in m_pred:
                        h = torch.gather(h_full, 1, mp_.unsqueeze(-1).expand(-1, -1, h_full.size(-1)))
                        hs.append(torch.nn.functional.layer_norm(h, (h.size(-1),)) if target_ln else h)
                    del h_full
                    zs = [ctx_enc(xe, masks=[me]) for me in m_enc]                   # context 토큰만
                with _ac():
                    ps = predictor(zs, m_enc, m_pred, info["mask_index"])
                loss = sum((p.float() - h.float()).abs().pow(loss_exp).mean() / loss_exp
                           for p, h in zip(ps, hs)) / len(ps)
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
                    p = predictor(z, m_enc, m_pred, mask_index=int(info.get("mask_index", mask_index)))   # (B, M, D); block3d 는 스펙별 mask token
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

            (loss, lr_now, wd_now), gpu_ms = gpu_timer(train_step_ar if is_ar else (train_step_ctx_ar if is_ctx_ar else (
                train_step_ctx_state if is_ctx_state else (train_step_all if mask_all else train_step))))
            step += 1
            if is_seq:
                jl_meter.update(aux["jloss"]); sl_meter.update(aux["sloss"])
            it_ms = (time.time() - t_it) * 1000
            loss_meter.update(loss); it_meter.update(it_ms); gpu_meter.update(gpu_ms); data_meter.update(data_ms)
            csv_logger.log(epoch + 1, itr, loss, info["context_frames"], lr_now, it_ms, gpu_ms, data_ms)
            if rank == 0 and (SH or {}).get("monitor_json") and ((itr % log_freq == 0) or aux.get("gprobe")):   # 2026-10-26: 모니터 값 전부 <run>/monitor.jsonl
                _rec = {"epoch": epoch + 1, "itr": itr, "step": step, "loss": float(loss), "lr": float(lr_now), "it_ms": float(it_ms),
                        "C": info.get("context_frames"), "K": info.get("predict_frames"), "src": str(info.get("source", info.get("dataset", "")))}
                for _k, _v in aux.items():
                    if isinstance(_v, dict):
                        _rec[_k] = {a: float(b) for a, b in _v.items() if isinstance(b, (int, float))}
                    elif isinstance(_v, (int, float)):
                        _rec[_k] = float(_v)
                with open(os.path.join(folder, "monitor.jsonl"), "a") as _f:
                    _f.write(json.dumps(_rec) + "\n")
            if (itr % log_freq == 0) or (itr == ipe - 1) or not np.isfinite(loss):
                ar_s = (" [tf %.4f ar %.4f R=%d]" % (aux["jloss"], aux["sloss"], aux["R"])) if is_seq else ""
                if HINT:
                    ar_s += " hint %d" % aux.get("n_hint", 0)
                if state_head is not None and aux.get("hlog"):   # 2026-10-19: 마지막 합성 step 의 상태 head (rollout 기준) — 위치 px 보임/가림 · 존재 정확도
                    g_ = aux["hlog"]
                    ar_s += " st %.4f pos %.1f/%.1fpx ex %.3f" % (aux.get("head", 0.0), g_.get("ar_pos_px_vis", float("nan")),
                                                                g_.get("ar_pos_px_hid", float("nan")), g_.get("ar_ex_acc", float("nan")))
                if state_head is not None and aux.get("hlog") and "gt_pos_px_vis" in aux["hlog"]:   # 2026-10-24 fit_on gt: 같은 head 의 위치 오차 — 정답 토큰 / TF 예측 / 굴린 예측 (보이는 블록)
                    g_ = aux["hlog"]
                    ar_s += " posx %.1f/%.1f/%.1f" % tuple(g_.get(f"{a}_pos_px_vis", float("nan")) for a in ("gt", "tf", "ar"))
                if HEAD_ONLY and aux.get("hlog"):                 # 2026-10-26: head 사전학습 — 출력별 (물체 존재 · 정체 / 위치 px / 방향 L1 · 속력 L1), 정답 토큰
                    g_ = aux["hlog"]
                    ar_s += " head ex %.3f id %.3f pos %.1fpx dir %.3f spd %.3f" % tuple(g_.get(f"gt_{k}", float("nan")) for k in ("ex_acc", "id_acc", "pos_px_vis", "dir", "spd"))
                if state_head is not None and aux.get("hlog") and "ar_id_acc" in aux["hlog"]:
                    ar_s += " id %.3f" % aux["hlog"]["ar_id_acc"]
                if state_head is not None and aux.get("hlog"):   # 2026-10-19: 성분 (물체 CE · 위치 · 방향 · 속력, tf/ar) · 집중도 (rollout)
                    g_ = aux["hlog"]
                    ar_s += " cmp %.3f/%.3f %.4f/%.4f %.3f/%.3f %.3f/%.3f" % tuple(g_.get(f"{a}_{b}", float("nan")) for b in ("ex", "pos", "dir", "spd") for a in ("tf", "ar"))
                    if "ar_conc" in g_:
                        ar_s += " conc %.1f" % g_["ar_conc"]
                if state_head is not None and "gn" in aux:
                    ar_s += " gn %.2f" % aux["gn"]
                if aux.get("gprobe"):                             # 2026-10-26: predictor 로 가는 grad norm — 자연 latent / 합성 latent / head (가중 적용) / 성분 · cos(·, 합성 latent)
                    g_ = aux["gprobe"]
                    ar_s += " gprobe nat %.3f lat %.3f head %.3f (cos %.3f)" % tuple(g_.get(k, float("nan")) for k in ("nat", "lat", "head", "cos_head"))
                    ar_s += " " + " ".join("%s %.3f" % (k, g_[k]) for k in ("ar_pos", "ar_dir", "ar_spd", "ar_ex", "tf_pos", "tf_dir", "tf_spd", "tf_ex") if k in g_)
                if aux.get("hw"):                                 # 2026-10-26: grad_match 성분 가중치
                    ar_s += " hw " + " ".join("%s %.2e" % (k, v) for k, v in aux["hw"].items())
                if lora_params and "lgn" in aux:                  # 2026-10-25: encoder LoRA grad norm (clip 전)
                    ar_s += " lora gn %.3f" % aux["lgn"]
                if lam_mod is not None and aux.get("lam"):        # 2026-10-24: KL · z 표준편차 · μ² · σ 평균 · grad norm (predictor / 역동역학+변조)
                    g_ = aux["lam"]
                    ar_s += " lam kl %.2f zstd %.3f mu2 %.3f sig %.3f gn %.2f lgn %.2f" % (
                        g_["kl"], g_["zstd"], g_["mu2"], g_["sig"], aux.get("gn", float("nan")), aux.get("lgn", float("nan")))
                if motion_mod is not None and aux.get("mlog"):   # 2026-10-24: 잠재 L1 · SIGReg · 맞추기 · 예측 쌍 (TF/rollout) · 멈춤 기준 · d 분산 · grad norm (predictor / motion)
                    g_ = aux["mlog"]
                    ar_s += " lat %.4f mo sig %.3f fit %.4f tf %.4f ar %.4f copy %.4f dvar %.3f gn %.2f mgn %.2f" % (
                        aux.get("latent", float("nan")), g_["sig"], g_["fit"], g_["tf"], g_["ar"], g_["copy"], g_["dvar"],
                        aux.get("gn", float("nan")), aux.get("mgn", float("nan")))
                if delta_weight:
                    ar_s += " dl %.4f" % aux.get("dloss", 0.0)
                if dyn_weight:
                    ar_s += " dyn %.4f" % aux.get("dyn", 0.0)
                if GATING:     # 2026-10-17: rollout gate — 평균 · 움직이는 칸 (|정답−A| 상위 25 %) · 정지 칸 (하위 25 %) · 첫/마지막 걸음 · >0.95 비율
                    ar_s += " gate %.3f mov %.3f sta %.3f k1 %.3f kR %.3f >.95 %.2f" % tuple(
                        aux.get(k, float("nan")) for k in ("gate", "gate_mov", "gate_sta", "gate_k1", "gate_kR", "gate_hi"))
                if CTX_HOLE:   # 2026-10-16: 이 step 의 구멍 비율 (문맥 토큰 중) · 가려진 문맥 토큰 복원 손실 (구멍 없는 step 은 0)
                    ar_s += " hole %.2f rec %.4f" % (aux.pop("hole_frac", 0.0), aux.pop("rloss", 0.0))
                logger.info("[%d, %5d] loss %.4f (avg %.4f)%s | mask %s C=%s | lr %.2e wd %.2e | mem %.1fG | iter %.0f ms gpu %.0f ms data %.0f ms"
                            % (epoch + 1, itr, loss, loss_meter.avg, ar_s, info["type"], info["context_frames"], lr_now, wd_now,
                               torch.cuda.max_memory_allocated() / 1024**3, it_meter.avg, gpu_meter.avg, data_meter.avg))
            if not np.isfinite(loss):
                raise RuntimeError("loss 가 nan/inf")
            if M.get("sync_gc") and (itr + 1) % 50 == 0:
                gc.collect()

        logger.info(f"epoch {epoch + 1} avg loss {loss_meter.avg:.4f}" + (f" (tf {jl_meter.avg:.4f} ar {sl_meter.avg:.4f})" if is_seq else ""))
        if rank == 0:
            _extra = {"state_head": state_head.state_dict(), "state_head_cfg": SH} if state_head is not None else None
            if lora_params:             # 2026-10-25: 채점기가 문맥 (입력) encoder 를 다시 만든다 (analysis/intphys2/model.py, kind=ar)
                _extra = {**(_extra or {}), "encoder_lora": LR.lora_state_dict(in_enc), "encoder_lora_cfg": in_enc.lora_info}
            if motion_mod is not None:
                _extra = {**(_extra or {}), "motion": motion_mod.state_dict(), "motion_cfg": MO}
            if lam_mod is not None:     # predictor 키는 그대로 (= z = 0 추론), 역동역학 · 변조는 따로
                _extra = {**(_extra or {}), "latent_action": {"idm": lam_mod.idm.state_dict(), "zmod": lam_mod.zmod.state_dict()}, "latent_action_cfg": LA}
            U.save_checkpoint(latest, pred_mod, optimizer, scaler, epoch + 1, step, loss_meter.avg, cfg, arch, extra=_extra)
            if save_every > 0 and ((epoch + 1) % save_every == 0 or epoch + 1 == num_epochs):
                U.save_checkpoint(os.path.join(folder, f"e{epoch + 1}.pt"), pred_mod, optimizer, scaler,
                                  epoch + 1, step, loss_meter.avg, cfg, arch, extra=_extra)
        res = {}
        if V and int(V.get("every_epochs", 1)) > 0 and ((epoch + 1) % int(V.get("every_epochs", 1)) == 0 or epoch + 1 == num_epochs):
            res = _val(epoch + 1)
        if rank == 0:
            with open(metrics_path, "a") as f:
                rec = {"epoch": epoch + 1, "step": step, "train_loss": loss_meter.avg, "lr": lr_now,
                       "val": res, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                if is_seq:
                    rec.update(train_jloss=jl_meter.avg, train_sloss=sl_meter.avg)
                f.write(json.dumps(rec) + "\n")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    logger.info("done")
