"""모델 로더 레지스트리 — 채점기는 그대로 두고 **모델만 갈아 끼운다**.

기존 채점기(`analysis/intphys2/surprise.py`, `evals/world_model_analysis/eval.py`)는
번들의 세 모듈을 아래 계약으로 부른다. 새 모델은 그 계약에 맞는 어댑터만 쓰면 된다.

    h_full = bundle.target_encoder(clip)                       -> (B, N, D_t)
    z_ctx  = bundle.context_encoder(clip, masks=[ctx_idx])     -> (B, N_ctx, D_c)
    z_pred = bundle.predictor(z_ctx, ctx_idx, tgt_idx, mask_index=0)  -> (B, N_tgt, D_t)
    surprise = mean |z_pred - LN(h_full[tgt_idx])|             # LN 은 target_layer_norm 스위치

    ⚠️ D_c 와 D_t 는 달라도 된다 (VideoMAEv2 는 문맥 1408 / 타깃 1176 픽셀).
       맞아야 하는 것은 **predictor 출력과 target 차원**뿐이다.

config:
    model:
      family: vjepa2 | vjepa2_1 | videomae2      # 없으면 vjepa2 (기존 동작)
      checkpoint: ...

새 모델을 붙일 때 고칠 곳은 **이 파일의 BUILDERS 한 줄**이다. 채점기·데이터셋·런처는 건드리지 않는다.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

REPO = "/data/hyuntak/project/2026/2027_cvpr/vjepa2"
VMAE_REPO = "/data/hyuntak/project/2026/2027_cvpr/VideoMAEv2"

# 샤드가 같은 체크포인트를 NFS 에서 각각 읽으면 그게 병목이다 (2.1-g 는 16.9 GB).
CKPT_LOCAL = os.environ.get("BENCH_CKPT_LOCAL", "/data2/local_datasets/world/Benchmark/ckpt")


def local_first(path: str) -> str:
    cand = os.path.join(CKPT_LOCAL, os.path.basename(path))
    try:
        if os.path.isfile(cand) and os.path.getsize(cand) == os.path.getsize(path):
            return cand
    except OSError:
        pass
    return path


# =========================================================================== #
#  V-JEPA 2.1 — app/vjepa_2_1.  latent 5632 = 4 x 1408 (Deep Self-Supervision)
# =========================================================================== #
class _Jepa21Encoder(nn.Module):
    """training=True 로 불러 4 단계 concat(5632)을 낸다. predictor 가 그 차원을 받는다."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.embed_dim = 4 * m.embed_dim

    def forward(self, x, masks=None):
        return self.m(x, masks=masks, training=True)


class _Jepa21Target(_Jepa21Encoder):
    """타깃은 **1408 씩 네 토막 각각 LN** 이다 (train.py forward_target).
    채점기의 전체 LN 과 다르므로 여기서 걸고 config 는 `target_layer_norm: false` 로 둔다."""

    def forward(self, x, masks=None):
        h = self.m(x, masks=masks, training=True).float()
        E = self.m.embed_dim
        return torch.cat([F.layer_norm(h[:, :, i * E:(i + 1) * E], (E,)) for i in range(4)], dim=2)


class _Jepa21Predictor(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, z, masks_x, masks_y, mask_index: int = 0):
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]
        # ⚠️ forward 기본값이 mask_index=1 이다. 학습된 mask token 은 0 하나뿐이라 반드시 넘긴다.
        out, _ctx = self.m(z, masks_x, masks_y, mod="video", mask_index=mask_index)
        return out


def build_vjepa2_1(cfg: dict, device: torch.device, num_frames: int, img_size: int):
    import app.vjepa_2_1.models.vision_transformer as vit21
    import app.vjepa_2_1.models.predictor as pred21

    ck = local_first(cfg["checkpoint"])
    logger.info(f"[loader vjepa2_1] {ck}")
    sd = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
    strip = lambda d: {k.replace("module.backbone.", ""): v for k, v in d.items()}

    # ⚠️ 아래 값들은 **모델의 성질**이라 config 에서 읽지 않는다.
    #    프로토콜 yaml 의 model 블록이 V-JEPA 2 기준값(uniform_power false, predictor depth 12 …)을
    #    들고 있고 병합 규칙상 프로토콜이 이기기 때문에, 읽으면 조용히 틀린 모델이 만들어진다.
    #    config 에서 받는 것은 창 기하(img_size / num_frames / patch / tubelet)뿐이다.
    arch = cfg.get("arch_name", "vit_giant")
    enc_kw = dict(img_size=img_size, num_frames=num_frames,
                  patch_size=int(cfg.get("patch_size", 16)),
                  tubelet_size=int(cfg.get("tubelet_size", 2)),
                  uniform_power=True, use_rope=True,
                  img_temporal_dim_size=1, interpolate_rope=True, modality_embedding=True)
    enc = vit21.__dict__[arch](**enc_kw)
    tgt = vit21.__dict__[arch](**enc_kw)
    enc.load_state_dict(strip(sd[cfg.get("context_encoder_key", "encoder")]), strict=True)
    tgt.load_state_dict(strip(sd[cfg.get("target_encoder_key", "target_encoder")]), strict=True)

    # predictor 도 모델 성질이다 (2.1-g: depth 24 / mask token 8). 체크포인트가 그렇게 생겼다.
    pred = pred21.__dict__["vit_predictor"](
        img_size=img_size, num_frames=num_frames,
        patch_size=int(cfg.get("patch_size", 16)), tubelet_size=int(cfg.get("tubelet_size", 2)),
        embed_dim=enc.embed_dim, predictor_embed_dim=384,
        depth=24, num_heads=12,
        uniform_power=True, use_rope=True,
        num_mask_tokens=8, use_mask_tokens=True,
        return_all_tokens=True, teacher_embed_dim=4 * enc.embed_dim, n_output_distillation=4,
        img_temporal_dim_size=1, interpolate_rope=True, modality_embedding=True)
    pred.load_state_dict(strip(sd["predictor"]), strict=True)

    ctx_m, tgt_m, pred_m = _Jepa21Encoder(enc), _Jepa21Target(tgt), _Jepa21Predictor(pred)
    for m in (ctx_m, tgt_m, pred_m):
        m.to(device).eval().requires_grad_(False)
    return ctx_m, tgt_m, pred_m, 4 * enc.embed_dim


# =========================================================================== #
#  VideoMAEv2-g — encoder + MAE decoder. 타깃이 **정규화 픽셀** 1176 = 3 x 2 x 14^2
#    RoPE 가 없다 (고정 sinusoid 표). patch 14 라 해상도는 224 고정.
# =========================================================================== #
class _VMaeContext(nn.Module):
    """masks=[ctx_idx] 를 bool 마스크로 바꿔 encoder 에 넣는다 (VideoMAE 는 보이는 토큰만 받는다)."""

    def __init__(self, m, n_tokens):
        super().__init__()
        self.m, self.n_tokens = m, n_tokens
        self.embed_dim = m.encoder.patch_embed.proj.out_channels

    def forward(self, x, masks=None):
        B = x.shape[0]
        if masks is None:
            bm = torch.zeros(B, self.n_tokens, dtype=torch.bool, device=x.device)
        else:
            idx = masks[0] if isinstance(masks, list) else masks
            if idx.dim() == 1:
                idx = idx.unsqueeze(0).expand(B, -1)
            bm = torch.ones(B, self.n_tokens, dtype=torch.bool, device=x.device)
            bm.scatter_(1, idx, False)                 # True = 가림 = encoder 가 안 봄
        self._last_mask = bm
        return self.m.encoder(x, bm)                    # (B, N_vis, C_e)


class _VMaeTarget(nn.Module):
    """타깃 = 패치 안에서 정규화한 원본 픽셀 (engine_for_pretraining.py 와 같은 식)."""

    def __init__(self, patch, tubelet, mean, std):
        super().__init__()
        self.patch, self.tubelet = patch, tubelet
        self.register_buffer("mean", mean); self.register_buffer("std", std)
        self.embed_dim = 3 * tubelet * patch * patch

    def forward(self, x, masks=None):
        from einops import rearrange
        raw = x * self.std.to(x.device) + self.mean.to(x.device)          # 정규화 해제 -> [0,1]
        sq = rearrange(raw, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                       p0=self.tubelet, p1=self.patch, p2=self.patch)
        z = (sq - sq.mean(-2, keepdim=True)) / (sq.var(-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        return rearrange(z, 'b n p c -> b n (p c)')


class _VMaePredictor(nn.Module):
    """encoder_to_decoder -> mask token + pos -> decoder. PretrainVisionTransformer.forward 와 같다."""

    def __init__(self, m, ctx_module):
        super().__init__()
        self.m, self.ctx = m, ctx_module

    def forward(self, z_vis, masks_x, masks_y, mask_index: int = 0):
        m = self.m
        B = z_vis.shape[0]
        x_vis = m.encoder_to_decoder(z_vis)
        pos = m.pos_embed.expand(B, -1, -1).type_as(x_vis).to(x_vis.device).clone().detach()
        bm = self.ctx._last_mask                                    # True = 가림
        pos_vis = pos[~bm].reshape(B, -1, x_vis.shape[-1])
        pos_msk = pos[bm].reshape(B, -1, x_vis.shape[-1])
        x_full = torch.cat([x_vis + pos_vis, m.mask_token + pos_msk], dim=1)
        return m.decoder(x_full, pos_msk.shape[1])                  # (B, N_mask, 1176)


def build_videomae2(cfg: dict, device: torch.device, num_frames: int, img_size: int):
    import sys
    if VMAE_REPO not in sys.path:
        sys.path.insert(0, VMAE_REPO)
    from models.modeling_pretrain import pretrain_videomae_giant_patch14_224

    assert img_size == 224, f"VideoMAEv2-g 는 patch 14 라 224 고정이다 (받은 값 {img_size})"
    ck = local_first(cfg["checkpoint"])
    logger.info(f"[loader videomae2] {ck}")
    # ★ decoder_depth=4 는 반드시 넘긴다 (scripts/pretrain/vit_g_hybrid_pt.sh).
    #   기본값이면 디코더 뒷블록이 랜덤인 채 조용히 돈다 (2026-09-20 에 실제로 당했다).
    m = pretrain_videomae_giant_patch14_224(decoder_depth=int(cfg.get("decoder_depth", 4)),
                                            all_frames=num_frames)
    r = m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"], strict=False)
    assert not r.missing_keys and not r.unexpected_keys, (r.missing_keys[:5], r.unexpected_keys[:5])
    m.to(device).eval().requires_grad_(False)

    patch, tub = 14, 2
    n_tokens = (num_frames // tub) * (img_size // patch) ** 2
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    ctx_m = _VMaeContext(m, n_tokens)
    tgt_m = _VMaeTarget(patch, tub, mean, std)
    pred_m = _VMaePredictor(m, ctx_m)
    for mod in (ctx_m, tgt_m, pred_m):
        mod.to(device).eval().requires_grad_(False)
    return ctx_m, tgt_m, pred_m, 3 * tub * patch * patch


BUILDERS = {
    "vjepa2_1": build_vjepa2_1,
    "videomae2": build_videomae2,
    # "vjepa2" 는 analysis/intphys2/model.py 의 기존 경로가 그대로 처리한다.
}
