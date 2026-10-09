"""ctx_ar predictor 정합성 (CPU, 작은 모델). 레포 루트에서: python tmp/.../tests/ctx_ar_cpu_test.py
2026-09-27 결과: causal Δ 0.0 / rollout[0] vs tf Δ 8e-7 / forward(ar) Δ 0.0, 3e-6 / K=1 Δ 0.0 / ckpt Δ 0.0 / type_embed grad True."""
import torch, sys
sys.path.insert(0, '.')
from src.models.rollout_predictor import vit_prefix_predictor
torch.manual_seed(0)
P = vit_prefix_predictor(img_size=64, patch_size=16, tubelet_size=2, num_frames=48, embed_dim=32,
                         predictor_embed_dim=96, depth=2, num_heads=2, mask_mode="ctx_ar", use_sdpa=True).eval()
S = P.tokens_per_block; B, C, K = 2, 3, 4; nb = C + K
z = torch.randn(B, C*S, 32)
h = torch.nn.functional.layer_norm(torch.randn(B, K*S, 32), (32,))
idx = torch.arange(nb*S).unsqueeze(0).expand(B, -1)
with torch.no_grad():
    p = P.forward_mixed(z, h[:, :-S], idx[:, :-S], C)          # (B, K*S): 블록 C..C+K-1 예측
    print("shape", tuple(p.shape), "expect", (B, K*S, 32))
    h2 = h.clone(); h2[:, 2*S:] = torch.randn_like(h2[:, 2*S:])   # 블록 C+2, C+3 의 GT 를 바꿈
    p2 = P.forward_mixed(z, h2[:, :-S], idx[:, :-S], C)
    print("causal Δ(blk C..C+2 preds) =", (p[:, :3*S]-p2[:, :3*S]).abs().max().item(), " Δ(blk C+3 pred) =", (p[:, 3*S:]-p2[:, 3*S:]).abs().max().item())
    z2 = z.clone(); z2[:, :S] = torch.randn_like(z2[:, :S])
    p3 = P.forward_mixed(z2, h[:, :-S], idx[:, :-S], C)
    print("ctx blk0 change -> Δ(blk C pred) =", (p[:, :S]-p3[:, :S]).abs().max().item(), "(> 0 이어야 한다: 문맥 전체를 본다)")
    r = P.rollout_ctx(z, idx[:, :C*S], K)
    print("rollout[0] vs tf[C] Δ =", (r[:, :S]-p[:, :S]).abs().max().item(), " rollout shape", tuple(r.shape))
    ptf, par = P(z, ar={"h_fut": h, "idx": idx, "n_ctx_blocks": C, "rollout_steps": K})
    print("forward(ar) Δ tf =", (ptf-p).abs().max().item(), " Δ ar =", (par-r).abs().max().item())
    p1, r1 = P(z, ar={"h_fut": h[:, :S], "idx": idx[:, :(C+1)*S], "n_ctx_blocks": C, "rollout_steps": 1})
    print("K=1 shapes", tuple(p1.shape), tuple(r1.shape), "Δ", (p1-r1).abs().max().item())
P.train(); P.use_activation_checkpointing = True
ptf2, par2 = P(z, ar={"h_fut": h, "idx": idx, "n_ctx_blocks": C, "rollout_steps": K})
print("ckpt path Δ tf =", (ptf2-ptf).abs().max().item(), " Δ ar =", (par2-par).abs().max().item())
(ptf2.abs().mean()+par2.abs().mean()).backward()
print("type_embed grad:", P.type_embed.grad.abs().sum().item() > 0, "| params with grad", sum(p.grad is not None for p in P.parameters()), "/", sum(1 for _ in P.parameters()))
sd = vit_prefix_predictor(img_size=64, patch_size=16, tubelet_size=2, num_frames=48, embed_dim=32, predictor_embed_dim=96, depth=2, num_heads=2, mask_mode="prefix").state_dict()
print("prefix state_dict unchanged (no type_embed):", "type_embed" not in sd)
