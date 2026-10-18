#!/usr/bin/env python
"""EK100 래퍼의 ctx_state 분기 검사 (CPU, 몇 초) — 2026-10-18.

`kind=ctx_state` (vast `ar_video_ctxt_block_target`) 는 문맥이 **블록별 LN(h)** 가 아니라
**online encoder 로 창 한 번** 인코딩한 z 다. 래퍼에 그 분기를 새로 넣었으므로 (2026-10-18)
정본 정의 `analysis/predictors/ar_scoring.ctx_state_from_window` 와 같은 값을 내는지 본다.

    python z_vll7_training/tests/ek100_ctx_state_cpu_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from app.vjepa_frozen.ar import block_indices
from evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar_nonsquare import AnticipativeWrapper
from src.models.rollout_predictor import vit_prefix_predictor

G, PATCH, TUB, FPS = 4, 16, 2, 4          # 4x4 grid -> S = 16 토큰/블록
S, D_ENC, D_PRD = G * G, 64, 48
NB, N_OUT = 6, 4                           # 입력 6 블록, 미래 N_OUT//TUB = 2 블록
ok, fail = [], []


def check(name, cond, extra=""):
    (ok if cond else fail).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")


class FakeEnc(torch.nn.Module):
    """창 한 번 인코딩. masks 를 받으면 그 자리만 돌려준다 (실제 ViT 의 masks 규약과 같은 모양)."""
    def __init__(self, tag):
        super().__init__()
        self.embed_dim, self.patch_size, self.tubelet_size = D_ENC, PATCH, TUB
        torch.manual_seed(hash(tag) % 10000)
        self.proj = torch.nn.Linear(3 * PATCH * PATCH * TUB, D_ENC)

    def forward(self, x, masks=None):
        B, C, T, H, W = x.shape
        t = x.view(B, C, T // TUB, TUB, H // PATCH, PATCH, W // PATCH, PATCH)
        t = t.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, (T // TUB) * (H // PATCH) * (W // PATCH), -1)
        out = self.proj(t)
        if masks is not None:
            out = torch.gather(out, 1, masks[0].unsqueeze(-1).expand(-1, -1, D_ENC))
        return out


def build():
    torch.manual_seed(0)
    pred = vit_prefix_predictor(
        img_size=G * PATCH, patch_size=PATCH, num_frames=NB * TUB * 4, tubelet_size=TUB,
        embed_dim=D_ENC, predictor_embed_dim=D_PRD, depth=2, num_heads=2,
        use_rope=True, use_sdpa=False, num_mask_tokens=1, mask_mode="ctx_state").eval()
    pred.ek_kind = "ctx_state"
    w = AnticipativeWrapper(encoder=FakeEnc("tgt"), predictor=pred, ctx_encoder=FakeEnc("ctx"),
                            frames_per_second=FPS, crop_size=G * PATCH, patch_size=PATCH,
                            tubelet_size=TUB, num_output_frames=N_OUT, mask_index=0).eval()
    return w, pred


w, pred = build()
B = 3
torch.manual_seed(1)
x = torch.randn(B, 3, NB * TUB, G * PATCH, G * PATCH)
n_pred_blk = N_OUT // TUB

print("== 1. 단일 anticipation_time: 정본 정의와 일치 ==")
at = torch.full((B,), 2.0 / FPS)                       # 2 블록 건너뜀
with torch.no_grad():
    got = w(x, at)
    idx = block_indices(B, NB, S, x.device)
    z = w.ctx_encoder(x, masks=[idx])
    k = int(at[0] * FPS / TUB) + n_pred_blk
    want_pred = pred.render(pred.ctx_rollout_state_kv(z, idx, k))[:, -n_pred_blk * S:]
    want = torch.cat([w.encoder(x), want_pred], dim=1)
check("출력 shape = (B, nb*S + n_pred*S, D)", tuple(got.shape) == (B, NB * S + n_pred_blk * S, D_ENC),
      str(tuple(got.shape)))
check("ar_scoring 정의 (z=ctx_encoder(창), render(ctx_rollout_state_kv)) 와 일치",
      torch.allclose(got, want, atol=1e-5), f"max|Δ| {float((got - want).abs().max()):.2e}")

print("\n== 2. 문맥은 **online encoder** 다 (target_encoder 로 하면 값이 달라야 한다) ==")
with torch.no_grad():
    z_tgt = w.encoder(x, masks=[idx])
    wrong = pred.render(pred.ctx_rollout_state_kv(z_tgt, idx, k))[:, -n_pred_blk * S:]
check("ctx_encoder != encoder 로 계산한 값", not torch.allclose(want_pred, wrong, atol=1e-3),
      f"max|Δ| {float((want_pred - wrong).abs().max()):.2e}")

print("\n== 3. anticipation_time 이 섞여 있어도 샘플별 루프와 같다 ==")
at_mix = torch.tensor([1.0, 3.0, 1.0]) * (TUB / FPS)   # 블록 1, 3, 1 건너뜀
with torch.no_grad():
    got_mix = w(x, at_mix)
    rows = []
    for b in range(B):                                  # 샘플 하나씩 따로
        kb = int(at_mix[b] * FPS / TUB) + n_pred_blk
        zb = w.ctx_encoder(x[b:b + 1], masks=[idx[b:b + 1]])
        rows.append(pred.render(pred.ctx_rollout_state_kv(zb, idx[b:b + 1], kb))[:, -n_pred_blk * S:])
    want_mix = torch.cat([w.encoder(x), torch.cat(rows, dim=0)], dim=1)
check("걸음 수로 묶은 결과 == 샘플별 루프", torch.allclose(got_mix, want_mix, atol=1e-5),
      f"max|Δ| {float((got_mix - want_mix).abs().max()):.2e}")
check("걸음 수가 다르면 예측도 다르다", not torch.allclose(got_mix[0, -n_pred_blk * S:],
                                                      got_mix[1, -n_pred_blk * S:], atol=1e-4))

print("\n== 4. KV 판 == 전체 재계산판 ==")
with torch.no_grad():
    a = pred.render(pred.ctx_rollout_state_kv(z, idx, k))
    b_ = pred.render(pred.ctx_rollout_state(z, idx, k))
check("ctx_rollout_state_kv == ctx_rollout_state", torch.allclose(a, b_, atol=1e-5),
      f"max|Δ| {float((a - b_).abs().max()):.2e}")

print("\n== 5. 다른 kind 는 건드리지 않았다 ==")
torch.manual_seed(0)
p2 = vit_prefix_predictor(img_size=G * PATCH, patch_size=PATCH, num_frames=NB * TUB * 4, tubelet_size=TUB,
                          embed_dim=D_ENC, predictor_embed_dim=D_PRD, depth=2, num_heads=2,
                          use_rope=True, use_sdpa=False, num_mask_tokens=1, mask_mode="ar_state").eval()
p2.ek_kind = "ar_state"
w2 = AnticipativeWrapper(encoder=FakeEnc("tgt"), predictor=p2, frames_per_second=FPS,
                         crop_size=G * PATCH, patch_size=PATCH, tubelet_size=TUB,
                         num_output_frames=N_OUT, mask_index=0).eval()
check("ctx_encoder=None 이 기본 (ar_state 는 안 만든다)", w2.ctx_encoder is None)

print(f"\n{'='*60}\n  PASS {len(ok)} / FAIL {len(fail)}" + (f"  -> {fail}" if fail else ""))
sys.exit(1 if fail else 0)
