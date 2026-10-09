"""AR predictor 정합성 (CPU, 작은 모델). 레포 루트에서: python tmp/.../tests/ar_cpu_test.py
검사: 블록 인과성 / split(flash) == dense 마스크 / rollout 첫 스텝 == TF 위치 C-1 / rollout 이 문맥 뒤 GT 와 무관 /
forward(ar=) == 개별 호출 / 출력 LN / activation checkpointing 경로 동일 / grad 흐름 / release 규약 차단."""
import torch, sys
sys.path.insert(0, '.')
from src.models.rollout_predictor import vit_prefix_predictor, build_prefix_mask
torch.manual_seed(0)
P = vit_prefix_predictor(img_size=64, patch_size=16, tubelet_size=2, num_frames=48, embed_dim=32,
                         predictor_embed_dim=96, depth=2, num_heads=2, mask_mode="ar", use_sdpa=True).eval()
S = P.tokens_per_block; B, nb, C = 2, 6, 3
h = torch.nn.functional.layer_norm(torch.randn(B, nb*S, 32), (32,))
idx = torch.arange(nb*S).unsqueeze(0).expand(B, -1)
with torch.no_grad():
    p = P.forward_seq(h, idx)
    h2 = h.clone(); h2[:, 4*S:] = torch.randn_like(h2[:, 4*S:])
    p2 = P.forward_seq(h2, idx)
    print("causal Δ(pos<=3) =", (p[:, :4*S]-p2[:, :4*S]).abs().max().item(), " Δ(pos>=4) =", (p[:, 4*S:]-p2[:, 4*S:]).abs().max().item())
    hh = P.predictor_embed(h); m = build_prefix_mask(idx[0]//S, 1, True)
    for blk in P.predictor_blocks: hh = blk(hh, mask=idx, attn_mask=m, T=None, H=P.grid_height, W=P.grid_width, action_tokens=0)
    pd = torch.nn.functional.layer_norm(P.predictor_proj(P.predictor_norm(hh)), (32,))
    print("split vs dense Δ =", (p-pd).abs().max().item())
    r = P.rollout(h[:, :C*S], idx[:, :C*S], nb-C)
    print("rollout[0] vs tf[C-1] Δ =", (r[:, :S]-p[:, (C-1)*S:C*S]).abs().max().item())
    r2 = P.rollout(h2[:, :C*S], idx[:, :C*S], nb-C)
    print("rollout indep of future GT Δ =", (r-r2).abs().max().item())
    ptf, par = P(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": nb-C})
    print("forward(ar) Δ tf =", (ptf-p[:, :-S]).abs().max().item(), " Δ ar =", (par-r).abs().max().item())
    print("out LN mean/std =", ptf.mean(-1).abs().max().item(), ptf.std(-1, unbiased=False).mean().item())
P.train(); P.use_activation_checkpointing = True
ptf2, par2 = P(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": nb-C})
print("ckpt path Δ tf =", (ptf2-ptf).abs().max().item(), " Δ ar =", (par2-par).abs().max().item())
loss = ptf2.abs().mean() + par2.abs().mean(); loss.backward()
print("grad ok:", sum(p.grad is not None for p in P.parameters()), "/", sum(1 for _ in P.parameters()), "params with grad (mask_tokens 10개 제외 기대)")
try:
    P(h[:, :C*S], idx[:, :C*S], idx[:, C*S:])
except RuntimeError as e: print("release 규약 차단 OK:", str(e)[:60])
