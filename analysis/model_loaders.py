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



# ─────────────────────────────────────────────────────────────────────────────
# DINO-Foresight (Karypidis et al., NeurIPS 2025) — frozen DINOv2 ViT-B/14-reg 특징 (4 층 concat 3072 → PCA 1152)
# 을 MaskTransformer (12 층, 1152, 분리 attention) 가 **프레임 단위** 로 예측한다. 문맥 4 프레임 + 마스크 1 프레임 = sequence 5.
#   2026-10-02 추가. 레포 /data/hyuntak/project/2026/2027_cvpr/DINO-Foresight (z_smoke/README.md 가 환경 · 스모크).
#   계약에 맞추기:
#     tubelet_size = 1 (프레임 = 토큰 시간 단위), S = (img/14)^2, D_t = D_pred = 1152 (PCA 공간; 이미 표준화돼 있으니 surprise.target_layer_norm: false)
#     target_encoder(clip)          = 프레임마다 DINOv2 → PCA                                  (B, T·S, 1152)
#     context_encoder(clip, masks)  = 앞 C 프레임만 같은 것                                      (B, C·S, 1152)   C 는 **4 여야 한다**
#     predictor(z, ci, ti)          = 4 프레임 문맥 + 마스크 프레임 → 다음 프레임, M 걸음 자기회귀 unroll (sample_unroll 과 같음) (B, M·S, 1152)
#   정방형 입력: 학습은 448×896 (h 32 × w 64 패치). 정방형 448 이면 w 위치 임베딩 64 → 32 를 bilinear 로 줄인다
#   (레포 on_load_checkpoint 의 high_res_adapt 와 같은 연산). 위치 임베딩 밖의 가중치는 그대로다.
#   lightning · scikit-learn 없이 로드한다: PCA 평균/성분/표준화는 ckpt state_dict 에 Parameter 로 들어 있고, DINOv2 가중치도
#   ckpt 의 `dino_v2.*` 가 공식 hub 가중치와 동일 (10-02 검증) 이라 hub 코드만 third_party/dinov2 (NFS 복사본) 에서 source=local 로 짓는다.
# ─────────────────────────────────────────────────────────────────────────────
DINOF_REPO = "/data/hyuntak/project/2026/2027_cvpr/DINO-Foresight"


class _DinofFeat(nn.Module):
    """프레임 (N, 3, H, W) → DINOv2 중간 4 층 concat → PCA → (N, S, 1152). 32 장씩 끊어 돈다."""

    def __init__(self, dino, layers, pca_mean, pca_comp, mean, std, chunk=32, stretch_to=None):
        super().__init__()
        self.dino, self.layers, self.chunk, self.stretch_to = dino, list(layers), chunk, stretch_to   # stretch_to=(H, W): 입력을 그 크기로 bilinear (448x896 원래 비율)
        self.register_buffer("pca_mean", pca_mean); self.register_buffer("pca_comp", pca_comp)
        self.register_buffer("mean", mean); self.register_buffer("std", std)

    def forward(self, imgs):
        outs = []
        if self.stretch_to is not None and tuple(imgs.shape[-2:]) != tuple(self.stretch_to):
            imgs = F.interpolate(imgs.float(), size=tuple(self.stretch_to), mode="bilinear", align_corners=False).to(imgs.dtype)
        for k in range(0, imgs.shape[0], self.chunk):
            f = self.dino.get_intermediate_layers(imgs[k:k + self.chunk], n=self.layers, reshape=False)
            f = torch.cat(list(f), dim=-1).float()                                  # (n, S, 3072)
            f = (f - self.mean) / self.std - self.pca_mean
            outs.append(f @ self.pca_comp.T)                                        # (n, S, 1152)
        return torch.cat(outs, 0)


class _DinofTarget(nn.Module):
    def __init__(self, feat, S):
        super().__init__(); self.feat, self.S, self.embed_dim = feat, S, 1152

    def forward(self, x, masks=None):                                               # x (B, 3, T, H, W)
        B, _, T = x.shape[:3]
        f = self.feat(x.transpose(1, 2).flatten(0, 1))                              # (B·T, S, D)
        return f.reshape(B, T * self.S, -1)


class _DinofContext(nn.Module):
    def __init__(self, feat, S):
        super().__init__(); self.feat, self.S, self.embed_dim = feat, S, 1152

    def forward(self, x, masks=None):
        B, _, T = x.shape[:3]
        C = T if masks is None else int((masks[0] if isinstance(masks, list) else masks).shape[-1]) // self.S
        f = self.feat(x[:, :, :C].transpose(1, 2).flatten(0, 1))
        return f.reshape(B, C * self.S, -1)


class _DinofPredictor(nn.Module):
    def __init__(self, maskvit, embed, replace_vector, hw, seq_len=5, copy_last=False):
        super().__init__(); self.maskvit, self.embed = maskvit, embed
        self.register_buffer("replace_vector", replace_vector.reshape(-1)); self.hw, self.seq_len, self.copy_last = hw, seq_len, copy_last

    def forward(self, z_ctx, masks_x, masks_y, mask_index: int = 0):
        h, w = self.hw; S = h * w; B = z_ctx.shape[0]; C = z_ctx.shape[1] // S; Cm = self.seq_len - 1
        if C < Cm:
            raise ValueError(f"DINO-Foresight 는 문맥 {Cm} 프레임이 필요하다 (sequence_length {self.seq_len}). 받은 C={C}.")
        if C > Cm and not getattr(self, "_warned", False):      # 2026-10-06 v11 (C16): 모델은 4 프레임만 받으므로 **문맥의 마지막 4 프레임** 만 쓴다 (앞 12 장은 버림)
            logger.info(f"[dinof predictor] 문맥 C={C} > {Cm}: 마지막 {Cm} 프레임만 predictor 에 넣는다 (앞 {C - Cm} 장 무시)"); self._warned = True
        M = int((masks_y if not isinstance(masks_y, list) else masks_y[0]).shape[-1]) // S
        seq = z_ctx.reshape(B, C, h, w, -1)[:, C - Cm:].float(); preds = []
        if self.copy_last:                                                          # 복사 기준선: 마지막 문맥 프레임 특징을 M 번 (model.dinof_copy=true)
            return seq[:, -1:].expand(B, M, h, w, seq.shape[-1]).reshape(B, M * S, -1)
        for _ in range(M):
            x = torch.cat([seq, torch.zeros_like(seq[:, :1])], dim=1)             # (B, 5, h, w, 1152)
            emb = self.embed(x)
            emb = torch.cat([emb[:, :-1], self.replace_vector.to(emb.dtype).expand(B, 1, h, w, -1)], dim=1)
            out, = self.maskvit(emb)                                                # (B, 5·h·w, 1152)
            nxt = out[:, -S:].reshape(B, 1, h, w, -1).float()
            preds.append(nxt); seq = torch.cat([seq[:, 1:], nxt], dim=1)
        return torch.cat(preds, dim=1).reshape(B, M * S, -1)                       # PCA 공간 (= target 공간)


def build_dinof(cfg: dict, device: torch.device, num_frames: int, img_size: int):
    import sys
    if DINOF_REPO not in sys.path:
        sys.path.insert(0, DINOF_REPO)
    from src.attention_masked import MaskTransformer
    ck = local_first(cfg["checkpoint"]); logger.info(f"[loader dinof] {ck}")
    st = torch.load(ck, map_location="cpu", weights_only=False); sd = st["state_dict"]
    hp = st.get("hyper_parameters", {}).get("args"); hp = vars(hp) if hasattr(hp, "__dict__") else (hp or {})
    patch = 14; assert img_size % patch == 0, f"dinof 는 14 의 배수 해상도 (받은 값 {img_size})"
    stretch = bool(cfg.get("dinof_stretch", False))                      # SET model.dinof_stretch=true: 정방형 입력을 학습 비율 (H x 2H = 448x896) 로 늘려 넣는다 (위치 임베딩 보간 없음)
    h = img_size // patch; w = 2 * h if stretch else h; S = h * w; seq_len = int(hp.get("sequence_length", 5)); layers = list(hp.get("d_layers", [2, 5, 8, 11]))
    hidden, depth, heads = int(hp.get("hidden_dim", 1152)), int(hp.get("layers", 12)), int(hp.get("heads", 8))
    assert hp.get("masking", "simple_replace") == "simple_replace" and not hp.get("use_first_last", False), hp
    # DINOv2 (hub 코드 로컬, 가중치는 ckpt)
    dino = torch.hub.load(f"{DINOF_REPO}/third_party/dinov2", "dinov2_" + str(hp.get("dinov2_variant", "vitb14_reg")), source="local", pretrained=False)
    r = dino.load_state_dict({k[len("dino_v2."):]: v for k, v in sd.items() if k.startswith("dino_v2.")}, strict=True); assert not r.missing_keys
    feat = _DinofFeat(dino, layers, sd["pca_mean"], sd["pca_components"], sd["mean"], sd["std"], stretch_to=((img_size, 2 * img_size) if stretch else None))
    if stretch:
        logger.info(f"[loader dinof] ★ stretch — 입력 {img_size}x{img_size} → {img_size}x{2*img_size} (학습 비율), 격자 {h}x{w}, S={S}")
    # MaskTransformer (정방형 grid 로 짓고 위치 임베딩만 보간)
    mv = MaskTransformer(shape=(seq_len, h, w), embedding_dim=int(sd["pca_components"].shape[0]), hidden_dim=hidden, depth=depth, heads=heads,
                         mlp_dim=4 * hidden, dropout=0.0, use_fc_bias=bool(hp.get("use_fc_bias", True)),
                         seperable_attention=bool(hp.get("seperable_attention", True)), seperable_window_size=int(hp.get("seperable_window_size", 1)), use_first_last=False)
    mv.fc_in = nn.Identity()
    msd = {k[len("maskvit."):]: v for k, v in sd.items() if k.startswith("maskvit.")}
    for i, n in ((1, h), (2, w)):
        key = f"pos_embd.emb.d_{i}"; e = msd[key]
        if e.shape[0] != n:
            logger.info(f"[loader dinof] pos emb {key} {tuple(e.shape)} -> ({n}, {e.shape[1]}) bilinear (정방형 입력)")
            msd[key] = F.interpolate(e[None, None].float(), size=(n, e.shape[1]), mode="bilinear", align_corners=False)[0, 0].to(e.dtype)
    r = mv.load_state_dict(msd, strict=True); assert not r.missing_keys and not r.unexpected_keys, r
    embed = nn.Linear(hidden, hidden, bias=True); embed.load_state_dict({"weight": sd["embed.weight"], "bias": sd["embed.bias"]})
    ctx_m, tgt_m = _DinofContext(feat, S), _DinofTarget(feat, S)
    copy_last = bool(cfg.get("dinof_copy", False))
    if copy_last:
        logger.info("[loader dinof] ★ 복사 기준선 — predictor 대신 마지막 문맥 프레임 특징을 낸다 (model.dinof_copy=true)")
    pred_m = _DinofPredictor(mv, embed, sd["replace_vector"], (h, w), seq_len, copy_last=copy_last)
    for mod in (ctx_m, tgt_m, pred_m):
        mod.to(device).eval().requires_grad_(False)
    ctx_m.spatial_tokens = S                                               # 비정방형 격자 → bundle.num_spatial_tokens 가 이 값을 쓴다 (model.py _BundleSpatialOverride)
    logger.info(f"[loader dinof] grid {h}x{w} S={S} seq_len={seq_len} layers={layers} hidden={hidden} depth={depth}")
    return ctx_m, tgt_m, pred_m, int(sd["pca_components"].shape[0])


BUILDERS = {
    "vjepa2_1": build_vjepa2_1,
    "videomae2": build_videomae2,
    "dinof": build_dinof,
    # "vjepa2" 는 analysis/intphys2/model.py 의 기존 경로가 그대로 처리한다.
}
