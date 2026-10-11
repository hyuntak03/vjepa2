"""rollout_kv == rollout (값·gradient), action 있음/없음 (CPU 작은 모델). python z_training/tests/ar_kv_cpu_test.py"""
import torch, sys; sys.path.insert(0, '.')
from src.models.rollout_predictor import vit_prefix_predictor
torch.manual_seed(0)
for ad in (0, 10):
    P = vit_prefix_predictor(img_size=64, patch_size=16, tubelet_size=2, num_frames=48, embed_dim=32, predictor_embed_dim=96,
                             depth=3, num_heads=2, mask_mode="ar", use_sdpa=True, action_dim=ad).train()
    S = P.tokens_per_block; B, C, K = 2, 3, 5; nb = C + K
    h = torch.nn.functional.layer_norm(torch.randn(B, nb*S, 32), (32,)); idx = torch.arange(nb*S).unsqueeze(0).expand(B, -1)
    A = torch.randn(B, nb, ad) if ad else None
    for C_ in (1, 3):
        r1 = P.rollout(h[:, :C_*S], idx[:, :C_*S], K, actions=A); (r1.pow(2).mean()).backward(); g1 = [p.grad.clone() for p in P.parameters() if p.grad is not None]; P.zero_grad()
        r2 = P.rollout_kv(h[:, :C_*S], idx[:, :C_*S], K, actions=A); (r2.pow(2).mean()).backward(); g2 = [p.grad.clone() for p in P.parameters() if p.grad is not None]; P.zero_grad()
        print(f"action_dim={ad:2d} C={C_}: Δ value {(r1-r2).abs().max().item():.2e}  Δ grad {max((a-b).abs().max().item() for a,b in zip(g1,g2)):.2e}  (n grads {len(g1)}/{len(g2)})")
    # forward(ar=) 가 kv 판을 쓰는지 + 결과 동일
    P.kv_rollout = False; tf1, ar1 = P(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": K, "actions": A})
    P.kv_rollout = True;  tf2, ar2 = P(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": K, "actions": A})
    print(f"action_dim={ad:2d} forward(ar) kv vs full: Δ tf {(tf1-tf2).abs().max().item():.1e} Δ ar {(ar1-ar2).abs().max().item():.2e}")
