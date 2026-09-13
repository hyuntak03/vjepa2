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
    **kwargs,
):
    img_h, img_w = _as_hw(resolution)
    logger.info(f"Loading pretrained model from {checkpoint} (input {img_h}x{img_w})")
    checkpoint = torch.load(checkpoint, map_location="cpu")

    # ----------------------------------------------------------------------- #
    # Initialize Encoder
    # ----------------------------------------------------------------------- #
    vit, vit_pred = _get_model_modules(model_kwargs)

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

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

    # ----------------------------------------------------------------------- #
    # Initialize Predictor
    # ----------------------------------------------------------------------- #
    prd_kwargs = model_kwargs["predictor"]
    prd_ckp_key = prd_kwargs.get("checkpoint_key")
    prd_model_name = prd_kwargs.get("model_name")

    teacher_embed_dim = prd_kwargs.get("teacher_embed_dim")
    n_output_distillation = prd_kwargs.get("n_output_distillation", 4)
    prd_out_embed_dim = teacher_embed_dim // n_output_distillation if teacher_embed_dim is not None else None

    predictor = vit_pred.__dict__[prd_model_name](
        img_size=(img_h, img_w),
        embed_dim=encoder.embed_dim,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        out_embed_dim=prd_out_embed_dim,
        **prd_kwargs,
    )
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

    grid_h = img_h // encoder.patch_size
    grid_w = img_w // encoder.patch_size
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
    ):
        super().__init__()
        self.encoder = encoder
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

        assert not (self.no_predictor and self.no_encoder), "Anticipative wrapper must use predictor or encoder"
        logger.info(
            f"AnticipativeWrapper: grid=({self.grid_h}, {self.grid_w}) "
            f"tokens/tubelet={self.tokens_per_tubelet} mask_index={self.mask_index} "
            f"num_output_frames={self.num_output_frames} num_steps={self.num_steps}"
        )

    def forward(self, x, anticipation_times):
        """
        :param x: (Tensor) video of shape [B, C, T, H, W]
        :param anticipation_times: (Tensor) [B] seconds into the future to predict
        """
        x_full = self.encoder(x)

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

            x_accumulate = torch.cat([x_accumulate, x_pred], dim=1)
            x_pred_for_input = x_pred_full if x_pred_full.size(-1) == x_pred_input.size(-1) else x_pred
            x_pred_input = torch.cat([x_pred_input[:, N_pred:, :], x_pred_for_input], dim=1)

        return x_accumulate
