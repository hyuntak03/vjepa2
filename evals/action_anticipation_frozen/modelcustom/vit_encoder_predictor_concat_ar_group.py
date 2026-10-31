"""
vast 판 (2026-10-21): main 의 `vit_encoder_predictor_concat_ar_nonsquare` (2026-10-18, predictor_group · MultiAnticipative) 를 가져와
predictor_checkpoint 빌드만 vast 학습 코드 (vit_prefix_predictor, kind=ar) 로 바꿨다. 나머지 (encoder 공유 · AR 블록 rollout 정의 ·
release 경로) 는 main 과 같다. 시트 "EK100 subset" (cap16metric) 을 vast 에서 재는 모듈. 사용법: cvpr/eval/ek100/run_group.sh
"""
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
------------------------------------------------------------------------------

`vit_encoder_predictor_concat_ar` 의 비정사각 + mask_index 설정 가능 판.

**원본 모듈은 건드리지 않는다.** config 의 `model_kwargs.module_name` 을 이 파일로 바꿔서 고른다.
정사각 입력 + `mask_index: 1` 이면 원본과 수치적으로 같아야 한다 (§README 검증 1).

원본과 다른 점은 셋뿐이다.

1. **비정사각 입력.** 원본은 `grid_size = crop_size // patch_size` 하나로 `grid_size**2` 를
   frame 당 토큰 수로 쓴다. 여기서는 `(grid_h, grid_w)` 를 따로 들고 `grid_h * grid_w` 를 쓴다.
   `resolution` 에 `[height, width]` 를 주면 된다.
2. **predictor 의 RoPE 에 H/W 주입.** `src/models/predictor.py` 의 forward 는 block 을
   `blk(x, mask=masks)` 로만 불러서 `RoPEAttention` 이 `H_patches=None` 을 받고
   `tokens_per_frame = grid_size**2` 로 폴백한다 (= 정사각 가정). 여기서는 predictor 의 각
   block.forward 를 감싸 `H_patches`/`W_patches` 를 넣는다. **src/ 를 수정하지 않는다.**
   정사각이면 폴백 값과 같으므로 주입해도 결과가 바뀌지 않는다.
3. **`mask_index` 를 설정할 수 있다** (기본 1 = 원본 동작).
   ⚠️ 릴리즈 체크포인트에서 학습된 mask token 은 `mask_tokens.0` 하나뿐이고 1~9 는 전부
   정확히 0 이다 (실측: norm 0.0000). 원본은 `predictor.forward` 의 기본값 `mask_index=1` 을
   그대로 써서 **0 벡터 mask token** 으로 미래를 예측한다. 자세한 것은 README 차이 1.
"""

import logging
import types

import torch

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _as_hw(resolution):
    if isinstance(resolution, (list, tuple)):
        assert len(resolution) == 2, resolution
        return int(resolution[0]), int(resolution[1])
    return int(resolution), int(resolution)


def _get_model_modules(pretrain_kwargs):
    """Import encoder/predictor modules based on config."""
    use_v2_1 = pretrain_kwargs.get("use_v2_1", False)
    if use_v2_1:
        import app.vjepa_2_1.models.predictor as vit_pred
        import app.vjepa_2_1.models.vision_transformer as vit
    else:
        import src.models.predictor as vit_pred
        import src.models.vision_transformer as vit
    return vit, vit_pred


def _inject_grid_into_predictor(predictor, grid_h, grid_w):
    """predictor 의 각 block.forward 에 H_patches/W_patches 를 고정 주입한다.

    `RoPEAttention.forward(..., H_patches, W_patches)` 는 mask 가 주어지면
    `separate_positions(mask, H_patches, W_patches)` 로만 위치를 쪼갠다 (T 는 안 쓴다).
    해상도가 런마다 고정이므로 상수 주입으로 충분하다.
    """
    n = 0
    for blk in predictor.predictor_blocks:
        if not hasattr(blk.attn, "separate_positions"):  # RoPE 가 아니면 할 일 없음
            continue
        orig_forward = blk.__class__.forward

        def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, _f=orig_forward):
            return _f(
                self,
                x,
                mask=mask,
                attn_mask=attn_mask,
                T=T,
                H_patches=grid_h if H_patches is None else H_patches,
                W_patches=grid_w if W_patches is None else W_patches,
            )

        blk.forward = types.MethodType(forward, blk)
        n += 1
    logger.info(f"injected (H_patches, W_patches)=({grid_h}, {grid_w}) into {n} predictor blocks")
    return predictor


def init_module(
    frames_per_clip: int,
    frames_per_second: int,
    resolution,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
    shared: dict = None,          # 2026-10-18: 여러 모델이 **같은 frozen encoder** 를 쓸 때 건네는 캐시. MultiAnticipative 참고
    **kwargs,
):
    img_h, img_w = _as_hw(resolution)

    # ----------------------------------------------------------------------- #
    # Initialize Encoder
    # ----------------------------------------------------------------------- #
    vit, vit_pred = _get_model_modules(model_kwargs)

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

    if shared is not None and "encoder" in shared:
        # 같은 체크포인트 · 같은 arch 라는 것은 부르는 쪽 (eval.py 의 predictor_group) 이 보장한다
        encoder, checkpoint = shared["encoder"], shared["ckpt"]
        logger.info("encoder 재사용 (predictor_group 공유)")
    else:
        logger.info(f"Loading pretrained model from {checkpoint} (input {img_h}x{img_w})")
        checkpoint = torch.load(checkpoint, map_location="cpu")
        encoder = vit.__dict__[enc_model_name](img_size=(img_h, img_w), num_frames=frames_per_clip, **enc_kwargs)
        if not getattr(encoder, "handle_nonsquare_inputs", False) and img_h != img_w:
            raise RuntimeError("encoder was built with handle_nonsquare_inputs=False but input is non-square")
        pretrained_dict = checkpoint[enc_ckp_key]
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        for k, v in encoder.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                pretrained_dict[k] = v
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained encoder with msg: {msg}")
        if shared is not None:
            shared["encoder"], shared["ckpt"] = encoder, checkpoint

    def _load_encoder(ckp_key):
        """같은 체크포인트의 다른 encoder 키를 같은 arch 로 올린다 (ctx_state 의 문맥 encoder = online `encoder`)."""
        enc = vit.__dict__[enc_model_name](img_size=(img_h, img_w), num_frames=frames_per_clip, **enc_kwargs)
        pd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in checkpoint[ckp_key].items()}
        missing = [k for k in enc.state_dict() if k not in pd]
        if missing:
            logger.info(f"[{ckp_key}] {len(missing)} 키가 체크포인트에 없다 (init 유지): {missing[:3]}")
        enc.load_state_dict({k: v for k, v in pd.items() if k in enc.state_dict()}, strict=False)
        return enc

    # ----------------------------------------------------------------------- #
    # Initialize Predictor
    # ----------------------------------------------------------------------- #
    prd_kwargs = model_kwargs["predictor"]
    prd_ckp_key = prd_kwargs.get("checkpoint_key")
    prd_model_name = prd_kwargs.get("model_name")

    teacher_embed_dim = prd_kwargs.get("teacher_embed_dim")
    n_output_distillation = prd_kwargs.get("n_output_distillation", 4)
    prd_out_embed_dim = teacher_embed_dim // n_output_distillation if teacher_embed_dim is not None else None

    pred_ckpt_path = model_kwargs.get("predictor_checkpoint")          # 2026-09-30: 학습한 predictor (z_training run · Ariel 팔). kind 는 ckpt arch 가 선언
    if pred_ckpt_path:
        # vast 2026-10-21: main 은 analysis/predictors (kind_from_arch · ar_state) 로 짓는다. vast 에는 그 패키지가 없으므로
        #   학습 코드와 같은 vit_prefix_predictor (src/models/rollout_predictor.py, mask_mode=ar) 로 체크포인트의 arch 대로 짓는다.
        #   render_predictor.py · analysis/intphys2/model.py (_ArScorer) 와 같은 빌드다.
        from src.models.rollout_predictor import vit_prefix_predictor
        _pst = torch.load(str(pred_ckpt_path), map_location="cpu", weights_only=False)
        _arch = _pst.get("arch") or {}
    if pred_ckpt_path and _arch.get("kind") in (None, "oneshot"):
        # 2026-10-29: 릴리즈형 predictor (z_training kind=oneshot · arch 가 없는 옛 post-FT) — 릴리즈와 같은 클래스라
        #   아래 기본 경로처럼 짓고 가중치만 이 체크포인트의 predictor 로 (strict). EK100 동작은 기본 (mask token) 경로와 같다.
        if _pst.get("encoder_lora"):
            raise RuntimeError(f"릴리즈형 predictor + encoder LoRA 는 아직 안 된다 ({pred_ckpt_path})")
        predictor = vit_pred.__dict__[prd_model_name](
            img_size=(img_h, img_w), embed_dim=encoder.embed_dim, patch_size=encoder.patch_size,
            tubelet_size=encoder.tubelet_size, out_embed_dim=prd_out_embed_dim, **prd_kwargs)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in _pst["predictor"].items()}
        msg = predictor.load_state_dict(_sd, strict=True)
        predictor.ek_kind = "default"
        logger.info(f"predictor <- {pred_ckpt_path} (릴리즈형 kind={_arch.get('kind')}, epoch {_pst.get('epoch')}): {msg}")
        del _pst, _sd
    elif pred_ckpt_path:
        if _arch.get("kind") != "ar":
            raise RuntimeError(f"vast group 모듈은 kind=ar · oneshot 체크포인트만 ({pred_ckpt_path}: kind={_arch.get('kind')})")
        _kw = dict(img_size=img_h, patch_size=encoder.patch_size, tubelet_size=encoder.tubelet_size,
                   num_frames=int(_arch.get("num_frames", 64)), embed_dim=encoder.embed_dim,
                   predictor_embed_dim=int(_arch.get("embed_dim", 384)), depth=int(_arch.get("depth", 12)),
                   num_heads=int(_arch.get("num_heads", 12)), num_mask_tokens=int(_arch.get("num_mask_tokens", 10)),
                   n_registers=int(_arch.get("n_registers", 0) or 0), mask_mode="ar",
                   ar_feedback=_arch.get("ar_feedback", "output"))
        for _k in ("gating_residual", "gating_anchor", "gating_out_ln"):
            if _arch.get(_k) is not None:
                _kw[_k] = _arch[_k]
        if img_h != img_w:
            raise RuntimeError("predictor_checkpoint 경로는 정사각 입력만 (rollout predictor 의 grid 는 img_size 하나로 짓는다)")
        predictor = vit_prefix_predictor(**_kw)
        _sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in _pst["predictor"].items()}
        _sd = {k: v for k, v in _sd.items() if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}   # 학습 보조 가지 (채점과 같다)
        _unexp = set(_sd) - set(predictor.state_dict())
        if _unexp:      # main 과 같은 이유로 먼저 죽인다 — 구조 옵션 (예: gating) 이 빠지면 키가 조용히 버려진다
            raise RuntimeError(f"predictor 체크포인트에 모델이 모르는 키 {len(_unexp)} 개: {sorted(_unexp)[:6]} (arch {_arch})")
        msg = predictor.load_state_dict(_sd, strict=True)
        predictor.ek_kind = "ar"
        logger.info(f"predictor <- {pred_ckpt_path} (kind ar, ar_feedback {predictor.ar_feedback}, "
                    f"gating {getattr(predictor, 'gating_residual', False)}, epoch {_pst.get('epoch')}): {msg}")
        if _pst.get("encoder_lora"):
            # 2026-10-29: 학습 (model.encoder_lora, kind=ar) 은 **입력 경로** (블록별 문맥 인코딩) 복사본에만 LoRA 를 달았다.
            #   채점기 (analysis/intphys2/model.py) 와 같게 target encoder 를 복사해 같은 자리에 다시 붙인다. probe 의 encoder 몫 (x_full) 은
            #   group 이 나눠 쓰는 원래 encoder 그대로 — 이 팔의 블록 인코딩 (hb) 만 따로 계산한다 (forward 의 ar 분기).
            import copy as _copy
            from app.vjepa_frozen import lora as _LR
            _lc = _pst.get("encoder_lora_cfg") or {}
            _le = _copy.deepcopy(encoder)
            _LR.apply_lora(_le, int(_lc["rank"]), float(_lc["alpha"]), _lc["targets"])
            _LR.load_lora(_le, _pst["encoder_lora"])
            for _p in _le.parameters():
                _p.requires_grad_(False)
            predictor.ek_lora_encoder = [_le.eval()]          # 리스트로 감싼다 — predictor 의 하위 모듈로 등록되면 state_dict 와 옵티마이저에 섞인다
            logger.info(f"입력 경로 encoder = LoRA 복사본 ({_lc})")
        del _pst, _sd
    else:
        predictor = vit_pred.__dict__[prd_model_name](
            img_size=(img_h, img_w),
            embed_dim=encoder.embed_dim,
            patch_size=encoder.patch_size,
            tubelet_size=encoder.tubelet_size,
            out_embed_dim=prd_out_embed_dim,
            **prd_kwargs,
        )
        predictor.ek_kind = "default"
        pretrained_dict = checkpoint[prd_ckp_key]
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        for k, v in predictor.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(
                    f'key "{k}" is of different shape in model and loaded state dict: '
                    f"{pretrained_dict[k].shape} vs {v.shape}"
                )
                pretrained_dict[k] = v
        msg = predictor.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained predictor with msg: {msg}")

    # ctx_state (2026-10-18): 문맥이 **블록별 LN(h)** 가 아니라 **online encoder 로 창 한 번** 인코딩한 z 다.
    #   래퍼의 self.encoder 는 target_encoder 라 그것으로 대신하면 학습 때와 다른 입력이 된다
    #   (vast 가 ctx_target 을 안 넘겨 IntPhys1 43.89 를 낸 사고와 같은 종류). 같은 체크포인트의 `encoder` 키를 하나 더 올린다.
    ctx_encoder = None
    if getattr(predictor, "ek_kind", "default") == "ctx_state":
        if shared is not None and "ctx_encoder" in shared:
            ctx_encoder = shared["ctx_encoder"]
            logger.info("ctx_encoder 재사용 (predictor_group 공유)")
        else:
            ctx_encoder = _load_encoder("encoder")
            logger.info("ctx_encoder <- 체크포인트의 'encoder' 키 (online, ctx_state 문맥용)")
            if shared is not None:
                shared["ctx_encoder"] = ctx_encoder

    grid_h = img_h // encoder.patch_size
    grid_w = img_w // encoder.patch_size
    if getattr(predictor, "ek_kind", "default") in ("default", "oneshot"):
        _inject_grid_into_predictor(predictor, grid_h, grid_w)

    # -- mask token 진단: 어떤 것이 실제로 학습됐는지 로그로 남긴다
    try:
        norms = [float(predictor.mask_tokens[i].float().norm()) for i in range(len(predictor.mask_tokens))]
        logger.info("predictor mask_token norms: " + ", ".join(f"[{i}]={n:.4f}" for i, n in enumerate(norms)))
    except Exception:
        pass

    # ----------------------------------------------------------------------- #
    # Build Wrapper
    # ----------------------------------------------------------------------- #
    model = AnticipativeWrapper(
        encoder=encoder,
        predictor=predictor,
        ctx_encoder=ctx_encoder,
        frames_per_second=frames_per_second,
        crop_size=(img_h, img_w),
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        **wrapper_kwargs,
    )
    model.embed_dim = encoder.embed_dim

    if hasattr(predictor, "hierarchical_layers") and len(predictor.hierarchical_layers) > 1:
        encoder.return_hierarchical = True

    return model


class AnticipativeWrapper(torch.nn.Module):
    """Use predictor for inference. 비정사각 grid 와 mask_index 를 지원한다."""

    def __init__(
        self,
        encoder,
        predictor,
        ctx_encoder=None,
        frames_per_second=4,
        crop_size=224,
        patch_size=16,
        tubelet_size=2,
        # -- wrapper kwargs
        no_predictor=False,
        num_output_frames=2,
        num_steps=1,
        no_encoder=False,
        mask_index=1,
        pred_replace=None,
    ):
        super().__init__()
        self.encoder = encoder                 # = target_encoder (probe 입력 · ar 계열 블록 인코딩)
        self.ctx_encoder = ctx_encoder         # = online encoder. ctx_state 전용, 아니면 None
        self.predictor = predictor
        img_h, img_w = _as_hw(crop_size)
        self.grid_h = img_h // patch_size
        self.grid_w = img_w // patch_size
        self.tokens_per_tubelet = self.grid_h * self.grid_w
        self.tubelet_size = tubelet_size
        self.no_predictor = no_predictor
        self.num_output_frames = max(num_output_frames, tubelet_size)
        self.frames_per_second = frames_per_second
        self.num_steps = num_steps
        self.no_encoder = no_encoder
        self.mask_index = mask_index
        # 2026-09-25 (auto_research SC3): predictor 출력을 대조 토큰으로 바꿔 끼워 probe 가 predictor 의 '미래' 를 쓰는지 본다.
        #   None (기본, 원래 동작) | "copy_last" = encoder 마지막 N_pred 토큰의 LayerNorm (예측 없이 마지막 관측 복사) | "mean" = encoder 토큰 평균의 LN
        self.pred_replace = pred_replace
        # 2026-10-18: predictor_group 일 때 MultiAnticipative 가 넣어 주는 batch 단위 memo (dict).
        #   None 이면 원래 동작 그대로. encoder 가 group 안에서 **같은 객체 · 같은 가중치**라 값이 같다.
        self.share = None

        assert not (self.no_predictor and self.no_encoder), "Anticipative wrapper must use predictor or encoder"
        logger.info(
            f"AnticipativeWrapper: grid=({self.grid_h}, {self.grid_w}) "
            f"tokens/tubelet={self.tokens_per_tubelet} mask_index={self.mask_index} "
            f"num_output_frames={self.num_output_frames} num_steps={self.num_steps}"
        )

    def _shared(self, key, fn):
        """encoder 결과를 group 안에서 한 번만 계산한다. share 가 없으면 그냥 fn() (원래 동작)."""
        if self.share is None:
            return fn()
        if key not in self.share:
            self.share[key] = fn()
        return self.share[key]

    def forward(self, x, anticipation_times):
        """
        :param x: (Tensor) video of shape [B, C, T, H, W]
        :param anticipation_times: (Tensor) [B] seconds into the future to predict
        """
        x_raw = x
        x_full = self._shared("x_full", lambda: self.encoder(x))

        if self.no_predictor:
            return x_full

        B, N, D_full = x_full.size()
        embed_dim = self.encoder.embed_dim
        use_hierarchical = D_full > embed_dim

        if use_hierarchical:
            x = x_full[:, :, -embed_dim:]
        else:
            x = x_full

        if self.no_encoder:
            x_accumulate = torch.rand(B, 0, embed_dim).to(x.device)
        else:
            x_accumulate = x.clone()

        # Position IDs of the encoder patch tokens [B, N]
        ctxt_positions = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)

        # Position IDs of tokens to skip for each sample in batch [B]
        anticipation_steps = (anticipation_times * self.frames_per_second / self.tubelet_size).to(torch.int64)
        skip_positions = N + self.tokens_per_tubelet * anticipation_steps

        # Position IDs of tokens to predict [B, N_pred]
        N_pred = int(self.tokens_per_tubelet * (self.num_output_frames // self.tubelet_size))
        tgt_positions = torch.arange(N_pred, device=x.device).unsqueeze(0).repeat(B, 1)
        tgt_positions = tgt_positions + skip_positions.unsqueeze(1).repeat(1, N_pred)

        if getattr(self.predictor, "ek_kind", "default") == "ctx_state":
            # 2026-10-18 kind=ctx_state (vast ar_video_ctxt_block_target): 문맥 = online encoder 로 **창 한 번** (양방향) 인코딩한 z,
            #   rollout 되먹임 = predictor 상태 (384) + type[2], 출력 = render(상태) = LN(proj(.)). 학습이 ctx_target=block 이라
            #   타깃 공간이 블록별 LN(h) 로 ar 계열과 같다. 정의는 analysis/predictors/ar_scoring.ctx_state_from_window.
            from app.vjepa_frozen.ar import block_indices
            m = getattr(self.predictor, "module", self.predictor)
            n_pred_blk = int(self.num_output_frames // self.tubelet_size)
            S = self.tokens_per_tubelet
            nb = N // S                                                                    # 입력 창 전부가 문맥이다
            idx = block_indices(B, nb, S, x.device)
            def _zc():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return self.ctx_encoder(x_raw, masks=[idx])                            # (B, nb*S, D)
            z_ctx = self._shared("z_ctx", _zc)
            steps = anticipation_steps.to(torch.int64) + n_pred_blk                        # (B,)
            x_pred = torch.empty(B, n_pred_blk * S, z_ctx.size(-1), device=x.device, dtype=z_ctx.dtype)
            roll = m.ctx_rollout_state_kv if getattr(m, "kv_rollout", True) else m.ctx_rollout_state
            for k_ in torch.unique(steps).tolist():                                        # 걸음 수가 같은 샘플끼리 묶는다
                sel = (steps == k_).nonzero(as_tuple=True)[0]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr = m.render(roll(z_ctx[sel], idx[sel], int(k_)))                     # (n, k_*S, D)
                x_pred[sel] = pr[:, -n_pred_blk * S:].to(z_ctx.dtype)
            x_pred = x_pred.to(x.dtype)
            if self.pred_replace == "copy_last":
                x_pred = torch.nn.functional.layer_norm(x[:, -N_pred:, :], (embed_dim,)).to(x_pred.dtype)
            return torch.cat([x_accumulate, x_pred], dim=1)

        if getattr(self.predictor, "ek_kind", "default") in ("ar", "ar_state"):
            # 2026-09-30 kind=ar (Ariel 팔 C): 문맥 = target encoder 를 **블록마다 따로** 돌린 LN(h) (encoder 는 wrapper 의 self.encoder = target_encoder),
            #   미래 = anticipation_steps 블록 건너뛴 뒤 num_output_frames/tubelet 블록 → rollout (skip + n_pred) 걸음의 마지막 n_pred 블록.
            #   probe 입력의 encoder 부분 (x_accumulate) 은 원래대로 (창 전체 인코딩) 두고 predictor 부분만 바꾼다. 출력 공간은 블록별 LN(h).
            from app.vjepa_frozen.ar import encode_blocks, block_indices
            n_pred_blk = int(self.num_output_frames // self.tubelet_size)
            _le = getattr(self.predictor, "ek_lora_encoder", None)                              # 2026-10-29: LoRA 입력 경로 (이 팔만)
            if _le is not None:
                _p0 = next(self.encoder.parameters())
                if next(_le[0].parameters()).device != _p0.device:
                    _le[0].to(device=_p0.device, dtype=_p0.dtype)
            _enc_in = self.encoder if _le is None else _le[0]
            def _hb():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return encode_blocks(_enc_in, x_raw, self.tubelet_size)                    # (B, nb*S, D) LN
            hb = self._shared("hb" if _le is None else f"hb_lora_{id(_le[0])}", _hb)
            S = self.tokens_per_tubelet; nb = hb.size(1) // S
            steps = anticipation_steps.to(torch.int64) + n_pred_blk                            # (B,)
            x_pred = torch.empty(B, n_pred_blk * S, hb.size(-1), device=x.device, dtype=hb.dtype)
            for k_ in torch.unique(steps).tolist():                                            # 걸음 수가 같은 샘플끼리 묶어 rollout (값은 샘플별 루프와 같다)
                sel = (steps == k_).nonzero(as_tuple=True)[0]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr = self.predictor.rollout_kv(hb[sel], block_indices(len(sel), nb, S, x.device), int(k_))   # (n, k_*S, D) vast: KV 판 (state 되먹임 · gating 지원)
                x_pred[sel] = pr[:, -n_pred_blk * S:].to(hb.dtype)
            x_pred = x_pred.to(x.dtype)
            if self.pred_replace == "copy_last":
                x_pred = hb[:, -N_pred:, :].to(x_pred.dtype)
            return torch.cat([x_accumulate, x_pred], dim=1)

        x_pred_input = x_full
        for _ in range(self.num_steps):
            pred_out = self.predictor(
                x_pred_input, masks_x=ctxt_positions, masks_y=tgt_positions, mask_index=self.mask_index
            )
            x_pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out

            if x_pred_full.size(-1) != embed_dim:
                x_pred = x_pred_full[:, :, -embed_dim:]
            else:
                x_pred = x_pred_full
            if self.pred_replace == "copy_last":
                x_pred = torch.nn.functional.layer_norm(x[:, -N_pred:, :], (embed_dim,)).to(x_pred.dtype)
            elif self.pred_replace == "mean":
                x_pred = torch.nn.functional.layer_norm(x.mean(1, keepdim=True), (embed_dim,)).expand(-1, N_pred, -1).to(x_pred.dtype)

            x_accumulate = torch.cat([x_accumulate, x_pred], dim=1)
            x_pred_for_input = x_pred_full if x_pred_full.size(-1) == x_pred_input.size(-1) else x_pred
            x_pred_input = torch.cat([x_pred_input[:, N_pred:, :], x_pred_for_input], dim=1)

        return x_accumulate


class MultiAnticipative(torch.nn.Module):
    """frozen encoder **하나** 위에 predictor 여러 개 (2026-10-18).

    EK100 에서 비교하는 모델들은 같은 릴리즈 ViT-H (632M) 를 쓰고 predictor (22M) 만 다르다. 따로 돌리면
    encoder forward 를 모델 수만큼 반복한다. 여기서는 batch 마다 `x_full` (창 전체 인코딩) 과 `hb`
    (AR kind 의 블록별 인코딩) 를 **한 번** 계산해 group 전체가 나눠 쓴다.

    forward 는 모델 수만큼의 feature 를 **리스트로** 돌려준다 (단일 모델 경로는 텐서 그대로). 값은
    모델을 따로 돌린 것과 같다 — 공유하는 두 값은 encoder 와 입력에만 의존하고, 둘 다 group 안에서 같다.
    """

    def __init__(self, models, names):
        super().__init__()
        assert len(models) == len(names) and len(models) > 0
        enc = models[0].encoder
        for m in models[1:]:
            if m.encoder is not enc:
                raise RuntimeError("MultiAnticipative: group 의 encoder 객체가 다르다 (공유가 안 됐다)")
        if len({id(m.ctx_encoder) for m in models if m.ctx_encoder is not None}) > 1:
            raise RuntimeError("MultiAnticipative: group 의 ctx_encoder 객체가 다르다 (공유가 안 됐다)")
        for a in ("tokens_per_tubelet", "tubelet_size", "num_output_frames", "frames_per_second", "num_steps"):
            vals = {getattr(m, a) for m in models}
            if len(vals) != 1:
                raise RuntimeError(f"MultiAnticipative: group 안에서 {a} 가 다르다 ({vals}) — 비교가 성립하지 않는다")
        self.models = torch.nn.ModuleList(models)
        self.names = list(names)
        self.encoder = enc
        self.embed_dim = models[0].embed_dim
        self.tokens_per_tubelet = models[0].tokens_per_tubelet
        self.tubelet_size = models[0].tubelet_size
        self._share = {}
        for m in self.models:
            m.share = self._share
        logger.info("MultiAnticipative: encoder 1 개 + predictor %d 개 (%s)" % (len(models), ", ".join(self.names)))

    def forward(self, x, anticipation_times):
        self._share.clear()
        out = [m(x, anticipation_times) for m in self.models]
        self._share.clear()          # 다음 batch 로 안 새게. 텐서 참조도 여기서 놓는다
        return out
