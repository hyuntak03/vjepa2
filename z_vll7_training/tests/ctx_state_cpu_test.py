#!/usr/bin/env python3
"""ctx_state (문맥 = video, 타깃 = video) 이식 검증 — CPU, 모델 로드 없음.

    python z_vll7_training/tests/ctx_state_cpu_test.py

무엇을 재나
  1. forward(ar=...) 가 (p_tf, p_ar) 를 맞는 모양으로 낸다 (p_tf = K 블록, p_ar = R 블록, 둘 다 encoder 공간)
  2. **KV 캐시판 `ctx_rollout_state_kv` 와 전체 재계산판 `ctx_rollout_state` 의 값이 같다**
     (문맥끼리 양방향 · 미래 block-causal 이라 이미 계산한 토큰의 층별 (k, v) 가 안 변한다는 전제의 실측 검사)
  3. 두 경로의 **gradient** 도 같다
  4. tf_weight=0 이면 TF 패스를 건너뛰고 p_tf 가 None 이다 (관측 인코딩 비용이 안 든다)
  5. ctx_state 는 mask token 규약 forward 를 거부한다 (조용히 다른 규칙으로 돌지 않게)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
from src.models.rollout_predictor import vit_prefix_predictor

torch.manual_seed(0)
B, C, K, R = 2, 3, 4, 4
EMB, PD, GRID, PATCH, TUB = 64, 48, 2, 16, 2            # grid 2x2 -> S = 4.
# ⚠️ predictor_embed_dim / num_heads >= 12 이어야 한다 — 레포 RoPE 의 `emb_sin.squeeze(-1).repeat(1,1,1,2)` 는
#    d_dim//2 == 1 이면 차원이 어긋난다 (릴리즈 구성 384/12 = 32 -> d_dim 10 에서는 안 걸린다)
S = GRID * GRID
net = vit_prefix_predictor(mask_mode="ctx_state", img_size=GRID * PATCH, patch_size=PATCH, tubelet_size=TUB,
                           num_frames=(C + K) * TUB, embed_dim=EMB, predictor_embed_dim=PD, depth=2, num_heads=4,
                           num_mask_tokens=2).double().eval()
z = torch.randn(B, C * S, EMB, dtype=torch.float64)
h_obs = torch.randn(B, K * S, EMB, dtype=torch.float64)
idx = torch.arange((C + K) * S).unsqueeze(0).expand(B, -1).contiguous()

# 1. forward 모양
p_tf, p_ar = net(z, ar={"h_obs": h_obs, "idx": idx, "n_ctx_blocks": C, "rollout_steps": R})
assert p_tf.shape == (B, K * S, EMB), p_tf.shape
assert p_ar.shape == (B, R * S, EMB), p_ar.shape
print(f"1. forward 모양 OK  p_tf {tuple(p_tf.shape)}  p_ar {tuple(p_ar.shape)}")

# 2. KV 판 vs 전체 재계산판 (값)
a = net.render(net.ctx_rollout_state_kv(z, idx[:, : C * S], R))
b = net.render(net.ctx_rollout_state(z, idx[:, : C * S], R))
d = (a - b).abs().max().item()
rel = d / b.abs().max().item()
assert rel < 1e-10, f"KV 판과 재계산판이 다르다: 절대 {d:.3e} 상대 {rel:.3e}"
print(f"2. rollout KV vs 재계산  절대 {d:.3e}  상대 {rel:.3e}  OK")

# 3. gradient
def grads(fn):
    net.zero_grad(set_to_none=True)
    net.render(fn(z, idx[:, : C * S], R)).sum().backward()
    return torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])
ga, gb = grads(net.ctx_rollout_state_kv), grads(net.ctx_rollout_state)
gd = (ga - gb).abs().max().item()
assert gd / gb.abs().max().item() < 1e-9, f"grad 가 다르다: {gd:.3e}"
print(f"3. grad KV vs 재계산  절대 {gd:.3e}  OK")

# 4. tf_weight = 0 (h_obs None)
p_tf0, p_ar0 = net(z, ar={"h_obs": None, "idx": idx, "n_ctx_blocks": C, "rollout_steps": R})
assert p_tf0 is None and p_ar0.shape == (B, R * S, EMB)
print("4. tf_weight=0 (h_obs None) -> p_tf None  OK")

# 5. mask token 규약 거부
try:
    net(z, idx[:, : C * S], idx[:, C * S:])
    raise SystemExit("5. FAIL: mask token 규약 forward 가 통과했다")
except RuntimeError as e:
    assert "ctx_state" in str(e)
print("5. mask token 규약 forward 거부  OK")

# 6. 손실 함수 경로 (app/vjepa_frozen/ar.py)
from app.vjepa_frozen import ar as AR
tgt = torch.randn(B, K * S, EMB, dtype=torch.float64)
loss, jl, sl, R_ = AR.ctx_state_losses(net, z, h_obs, tgt, S, C, K, 1.0, R, tf_weight=1.0)
assert R_ == R and loss.ndim == 0 and float(jl) > 0 and float(sl) > 0
loss0, jl0, sl0, _ = AR.ctx_state_losses(net, z, None, tgt, S, C, K, 1.0, R, tf_weight=0.0)
assert float(jl0) == 0.0 and torch.allclose(loss0, sl0)
print(f"6. ctx_state_losses  loss {float(loss):.4f} (tf {float(jl):.4f} ar {float(sl):.4f}) | tf_weight=0 -> loss == sloss  OK")
print("\n전부 통과")
