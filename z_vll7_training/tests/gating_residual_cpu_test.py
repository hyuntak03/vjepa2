#!/usr/bin/env python
"""gating_residual 이식 검사 (CPU, 모델 체크포인트 없이 몇 초) — 2026-10-18.

origin/vast 의 `gating_residual` (416f284) · `gating_anchor` (2910622) 를 main 의
`src/models/rollout_predictor.py` 로 옮긴 것이 맞는지 본다. 가장 중요한 항목은 **1번 (회귀)** 이다 —
gating 을 끄면 기존 수치가 비트 단위로 같아야 한다. 아니면 이미 낸 모든 ar 결과가 무효가 된다.

    python z_vll7_training/tests/gating_residual_cpu_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from src.models.rollout_predictor import vit_prefix_predictor

S_GRID = 4                      # 4x4 = 16 토큰/블록 (작게)
D_ENC, D_PRD = 64, 48      # head_dim 24 = RoPE 3 축 x 8 (각 축 짝수여야 한다)
ok, fail = [], []


def check(name, cond, extra=""):
    (ok if cond else fail).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")


def raises(fn, msg=""):
    try:
        fn()
        return False
    except Exception as e:
        return (msg in str(e)) if msg else True


def build(seed=0, **kw):
    torch.manual_seed(seed)
    return vit_prefix_predictor(
        img_size=S_GRID * 16, patch_size=16, num_frames=16, tubelet_size=2,
        embed_dim=D_ENC, predictor_embed_dim=D_PRD, depth=2, num_heads=2,
        use_rope=True, use_sdpa=False, num_mask_tokens=1, **kw).eval()


def inputs(B=2, C=3, seed=1):
    torch.manual_seed(seed)
    S = S_GRID * S_GRID
    x = torch.randn(B, C * S, D_ENC)
    idx = torch.arange(C * S).unsqueeze(0).expand(B, -1).contiguous()
    return x, idx, S


print("== 1. 회귀: gating 을 끄면 기존과 비트 동일 ==")
# 기준값을 '이식 전' 식으로 직접 계산해 맞춘다 (LN(proj(predictor_norm(h)))).
m = build(mask_mode="ar_state")
x, idx, S = inputs()
with torch.no_grad():
    out_seq = m.forward_seq(x, idx)
    out_roll = m.rollout(x, idx, 3)
    out_kv = m.rollout_kv(x, idx, 3)
check("gating_residual 기본값 False", m.gating_residual is False)
check("keep_gate 를 만들지 않는다", not hasattr(m, "keep_gate"))
check("state_dict 에 gating 키 없음", not [k for k in m.state_dict() if "keep_gate" in k or "sig_b0" in k])
with torch.no_grad():
    # _ar_out 을 거치지 않는 손계산
    h = m.predictor_embed(x) + m.type_embed[0]
    from src.models.rollout_predictor import PrefixSpec
    nb = x.size(1) // S
    h = m._run_blocks(h, idx, PrefixSpec(1, nb - 1, S, future_causal=True))
    man = torch.nn.functional.layer_norm(m.predictor_proj(m.predictor_norm(h)), (D_ENC,))
check("forward_seq == 손계산 LN(proj(norm(h)))", torch.equal(out_seq, man))
check("rollout_state == rollout_kv (gating off)", torch.allclose(out_roll, out_kv, atol=1e-5),
      f"max|Δ| {float((out_roll - out_kv).abs().max()):.2e}")

print("\n== 2. gating on: 구조 ==")
g = build(mask_mode="ar_state", gating_residual=True)
check("keep_gate 생김 (Linear d->1)", hasattr(g, "keep_gate") and g.keep_gate.out_features == 1)
sd = set(g.state_dict())
base = set(build(mask_mode="ar_state").state_dict())
check("state_dict 추가 키가 정확히 keep_gate.{weight,bias}",
      sd - base == {"keep_gate.weight", "keep_gate.bias"}, f"{sorted(sd - base)}")
check("sig_b0 는 persistent=False (ckpt 에 없다)", "sig_b0" not in sd)
check("gating_anchor 기본 last_obs", g.gating_anchor == "last_obs")

print("\n== 3. gating 수식: g = σ(w·st+b)/σ(2) ==")
with torch.no_grad():                       # w=0, b=2 -> g == 1 이므로 out = anchor + proj(st)
    g.keep_gate.weight.zero_(); g.keep_gate.bias.fill_(2.0)
    st = torch.randn(2, S, D_PRD); anc = torch.randn(2, S, D_ENC)
    out = g._ar_out(st, anc)
check("w=0,b=2 면 out == anchor + proj(st)",
      torch.allclose(out, anc + g.predictor_proj(st), atol=1e-6),
      f"max|Δ| {float((out - anc - g.predictor_proj(st)).abs().max()):.2e}")
with torch.no_grad():
    g.keep_gate.bias.fill_(-1e9)            # g -> 0 이면 out = proj(st) (LN 없음)
    out0 = g._ar_out(st, anc)
check("g->0 이면 out == proj(st) (LN 없음)", torch.allclose(out0, g.predictor_proj(st), atol=1e-6))
check("anchor 가 없으면 죽는다", raises(lambda: g._ar_out(st, None), "anchor"))
check("anchor 토큰 수가 다르면 죽는다", raises(lambda: g._ar_out(st, anc[:, :-1]), "anchor"))

print("\n== 4. rollout anchor 규칙 ==")
g2 = build(mask_mode="ar_state", gating_residual=True)
g3 = build(mask_mode="ar_state", gating_residual=True, gating_anchor="prev")
g3.load_state_dict(g2.state_dict())
with torch.no_grad():
    r_last = g2.rollout(x, idx, 3)
    r_prev = g3.rollout(x, idx, 3)
    kv_last = g2.rollout_kv(x, idx, 3)
    kv_prev = g3.rollout_kv(x, idx, 3)
check("prev 는 last_obs 와 첫 걸음이 같다", torch.allclose(r_last[:, :S], r_prev[:, :S], atol=1e-6))
check("prev 는 둘째 걸음부터 다르다", not torch.allclose(r_last[:, S:2 * S], r_prev[:, S:2 * S], atol=1e-4))
check("rollout_state == rollout_kv (last_obs)", torch.allclose(r_last, kv_last, atol=1e-5),
      f"max|Δ| {float((r_last - kv_last).abs().max()):.2e}")
check("rollout_state == rollout_kv (prev)", torch.allclose(r_prev, kv_prev, atol=1e-5),
      f"max|Δ| {float((r_prev - kv_prev).abs().max()):.2e}")

print("\n== 5. 가드 ==")
check("ctx_state + gating 은 아직 NotImplementedError",
      raises(lambda: build(mask_mode="ctx_state", gating_residual=True), "ctx_state"))
check("gating_anchor=prev 는 gating_residual 전용",
      raises(lambda: build(mask_mode="ar_state", gating_anchor="prev")))
check("ar (출력 되먹임) 비 KV rollout + gating 은 막혀 있다",
      raises(lambda: build(mask_mode="ar", gating_residual=True).rollout(x, idx, 2), "rollout_kv"))
check("ar + gating 은 rollout_kv 로는 된다",
      build(mask_mode="ar", gating_residual=True).rollout_kv(x, idx, 2).shape == (2, 2 * S, D_ENC))

print(f"\n{'='*60}\n  PASS {len(ok)} / FAIL {len(fail)}" + (f"  -> {fail}" if fail else ""))
sys.exit(1 if fail else 0)
