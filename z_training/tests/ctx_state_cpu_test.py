"""ctx_state (문맥 창 z + 상태 되먹임, vast 판 이식 2026-10-17) 정합성 (CPU, 작은 모델). python z_training/tests/ctx_state_cpu_test.py
검사: KV 판 == 전체 재계산판 (값·grad) · rollout 첫 걸음 == TF 위치 C-1 · TF 인과 (미래 관측이 앞 예측에 안 샌다) ·
      문맥끼리 양방향 (문맥 앞 블록을 바꾸면 첫 예측이 바뀐다) · forward(ar=) 의 TF 켬/끔 · ctx_state_losses 가 window/block 타깃 둘 다 돈다"""
import torch, sys; sys.path.insert(0, '.')
from src.models.rollout_predictor import vit_prefix_predictor
from app.vjepa_frozen import ar as AR
torch.manual_seed(0)
kw = dict(img_size=64, patch_size=16, tubelet_size=2, num_frames=48, embed_dim=32, predictor_embed_dim=96, depth=3, num_heads=2, use_sdpa=True)
P = vit_prefix_predictor(mask_mode="ctx_state", **kw).train()
S = P.tokens_per_block; B, C, K = 2, 3, 5; nb = C + K
z = torch.randn(B, C * S, 32)
h_obs = torch.nn.functional.layer_norm(torch.randn(B, K * S, 32), (32,))
idx = torch.arange(nb * S).unsqueeze(0).expand(B, -1).contiguous()
with torch.no_grad():
    full = P.rollout_ctx_state(z, idx[:, :C * S], K)
    kv = P.rollout_state_kv(z, idx[:, :C * S], K)
    tf = P.forward_state(z, P.embed_obs(h_obs[:, :-S]), idx[:, :-S], C)
    h2 = h_obs.clone(); h2[:, 3 * S:] = torch.randn_like(h2[:, 3 * S:])
    tf2 = P.forward_state(z, P.embed_obs(h2[:, :-S]), idx[:, :-S], C)
    z2 = z.clone(); z2[:, :S] = torch.randn_like(z2[:, :S])
    kv_z2 = P.rollout_state_kv(z2, idx[:, :C * S], K)
print(f"KV vs full value {(full - kv).abs().max():.1e} | rollout[0] vs TF[C-1] {(kv[:, :S] - tf[:, :S]).abs().max():.1e} | "
      f"TF 인과 Δ(블록 C..C+3 예측) {(tf[:, :4 * S] - tf2[:, :4 * S]).abs().max():.1e} (0 이어야) | "
      f"문맥 양방향: 첫 문맥 블록 바꾸면 첫 예측 Δ {(kv_z2[:, :S] - kv[:, :S]).abs().max():.2e} (> 0 이어야)")
g = []
for f in (lambda: P.rollout_ctx_state(z, idx[:, :C * S], K), lambda: P.rollout_state_kv(z, idx[:, :C * S], K)):
    P.zero_grad(); f().pow(2).mean().backward(); g.append([p.grad.clone() for p in P.parameters() if p.grad is not None])
print(f"KV vs full grad {max((a - b).abs().max().item() for a, b in zip(*g)):.1e} (n {len(g[0])}/{len(g[1])}); type_embed grad {P.type_embed.grad.abs().sum().item() > 0}")
tgt_win = torch.nn.functional.layer_norm(torch.randn(B, K * S, 32), (32,))
for name, tgt, tfw in (("window, tf 0", tgt_win, 0.0), ("window, tf 1", tgt_win, 1.0), ("block, tf 1", h_obs, 1.0)):
    P.zero_grad()
    loss, jl, sl, R = AR.ctx_state_losses(P, z, h_obs if tfw > 0 else None, tgt, S, C, K, 1.0, 8, tfw)
    loss.backward()
    print(f"  ctx_state_losses [{name}] loss {loss.item():.4f} = {tfw}·{jl.item():.4f} + {sl.item():.4f} (R={R}) · grad {sum(p.grad.abs().sum().item() for p in P.parameters() if p.grad is not None) > 0}")
print("ok")
