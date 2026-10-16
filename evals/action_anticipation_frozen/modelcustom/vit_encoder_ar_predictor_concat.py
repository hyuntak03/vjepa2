"""
EK100 action anticipation — **kind=ar (자기회귀) predictor 판** (2026-10-14, vast).

기존 `vit_encoder_predictor_concat_ar_nonsquare` (릴리즈 predictor: mask token 질의로 미래 tubelet 하나) 는 건드리지 않는다.
config 의 `model_kwargs.module_name` 을 이 파일로 바꿔 고른다.

probe 입력 = [encoder 출력 (창 전체, 릴리즈 모듈과 같다) ; AR predictor 가 그린 미래 tubelet 하나]
  * encoder 쪽은 릴리즈 모듈과 같은 계산이라, 두 모듈의 차이는 predictor 가 낸 미래 토큰 뿐이다
  * AR 쪽은 학습과 같은 경로: 문맥 32 장 → tubelet 별 LN(target_encoder) 16 블록 → `rollout_kv` 로 앞으로 굴려
    anticipation 시점의 블록 (문맥 끝 + steps) 을 꺼낸다. steps = anticipation_time × fps / tubelet (1 s @ 8 fps → 4)
  * 힌트 가중치 (`hint_attn.*`) 는 버린다 (힌트 없이 rollout)

pretrain_kwargs (= init_module 의 model_kwargs) 에 필요한 키:
  encoder: 릴리즈 모듈과 같다
  predictor_ar: {checkpoint: <run>/latest.pt, ar_feedback: output|state, embed_dim 384, depth 12, num_heads 12, num_frames 64}
⚠️ 시간 척도: AR 은 fps 4 (블록 0.5 s) 로 학습했고 EK100 프로토콜은 8 fps (블록 0.25 s) 다. 그대로 쓴다 (단서로 적는다).
"""

import logging

import torch
import torch.nn.functional as F

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _clean(sd):
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def init_module(frames_per_clip: int, frames_per_second: int, resolution, checkpoint: str,
                model_kwargs: dict, wrapper_kwargs: dict, **kwargs):
    import src.models.vision_transformer as vit
    from src.models.rollout_predictor import vit_prefix_predictor

    if isinstance(resolution, (list, tuple)):
        if int(resolution[0]) != int(resolution[1]):
            raise ValueError("AR predictor 는 정사각 입력 (16x16 격자) 만 지원한다")
        resolution = int(resolution[0])
    ck = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = dict(model_kwargs["encoder"])
    enc_key = enc_kwargs.pop("checkpoint_key")
    enc_name = enc_kwargs.pop("model_name")
    encoder = vit.__dict__[enc_name](img_size=resolution, num_frames=frames_per_clip, **enc_kwargs)
    msg = encoder.load_state_dict(_clean(ck[enc_key]), strict=False)
    logger.info(f"[ar] encoder <- {enc_key}: {msg}")

    pk = dict(model_kwargs["predictor_ar"])
    pred = vit_prefix_predictor(
        img_size=resolution, patch_size=encoder.patch_size, tubelet_size=encoder.tubelet_size,
        num_frames=int(pk.get("num_frames", 64)), embed_dim=encoder.embed_dim,
        predictor_embed_dim=int(pk.get("embed_dim", 384)), depth=int(pk.get("depth", 12)),
        num_heads=int(pk.get("num_heads", 12)), num_mask_tokens=int(pk.get("num_mask_tokens", 10)),
        mask_mode="ar", ar_feedback=str(pk.get("ar_feedback", "output")))
    psd = _clean(torch.load(pk["checkpoint"], map_location="cpu", weights_only=False)["predictor"])
    psd = {k: v for k, v in psd.items() if not k.startswith(("hint_attn.", "dyn_head."))}
    msg = pred.load_state_dict(psd, strict=True)
    logger.info(f"[ar] predictor <- {pk['checkpoint']} (ar_feedback={pred.ar_feedback}): {msg}")

    model = ARAnticipativeWrapper(encoder=encoder, predictor=pred, frames_per_second=frames_per_second,
                                  **wrapper_kwargs)
    model.embed_dim = encoder.embed_dim
    return model


class ARAnticipativeWrapper(torch.nn.Module):
    def __init__(self, encoder, predictor, frames_per_second=8, no_predictor=False, **unused):
        super().__init__()
        self.encoder, self.predictor = encoder, predictor
        self.fps = frames_per_second
        self.tubelet = encoder.tubelet_size
        self.S = predictor.tokens_per_block
        self.no_predictor = bool(no_predictor)
        if unused:
            logger.info(f"[ar] 이 래퍼가 안 쓰는 wrapper_kwargs: {sorted(unused)}")

    def _encode_blocks(self, x):
        B, C, T, H, W = x.shape
        nb = T // self.tubelet
        xb = x[:, :, : nb * self.tubelet].reshape(B, C, nb, self.tubelet, H, W).permute(0, 2, 1, 3, 4, 5)
        h = self.encoder(xb.reshape(B * nb, C, self.tubelet, H, W))
        h = h.reshape(B, nb * h.size(1), h.size(-1))
        return F.layer_norm(h, (h.size(-1),)), nb

    def forward(self, x, anticipation_times):
        x_full = self.encoder(x)                                                        # 릴리즈 모듈과 같은 encoder 출력
        if self.no_predictor:
            return x_full
        h, nb = self._encode_blocks(x)                                                  # (B, nb*S, D) 학습과 같은 블록별 LN(h)
        B, S = x.size(0), self.S
        steps = (anticipation_times * self.fps / self.tubelet).to(torch.int64).clamp(min=0)   # 문맥 끝 + steps 블록
        R = int(steps.max().item()) + 1
        idx = torch.arange(nb * S, device=x.device).unsqueeze(0).expand(B, -1)
        p = self.predictor.rollout_kv(h, idx, R)                                        # (B, R*S, D) 블록 nb .. nb+R-1
        sel = (steps.view(B, 1) * S + torch.arange(S, device=x.device).view(1, S))      # (B, S)
        p_t = torch.gather(p, 1, sel.unsqueeze(-1).expand(-1, -1, p.size(-1)))
        return torch.cat([x_full, p_t.to(x_full.dtype)], dim=1)
