"""AR predictor + action 토큰 정합성 (CPU, 작은 모델). 레포 루트에서: python z_training/tests/ar_action_cpu_test.py"""
import torch, sys
sys.path.insert(0, '.')
from src.models.rollout_predictor import vit_prefix_predictor
torch.manual_seed(0)
kw = dict(img_size=64, patch_size=16, tubelet_size=2, num_frames=48, embed_dim=32, predictor_embed_dim=96, depth=2, num_heads=2, mask_mode="ar", use_sdpa=True)
P0 = vit_prefix_predictor(**kw).eval()
print("action_dim=0 state_dict keys unchanged (no action_embed):", not any(k.startswith("action_embed") for k in P0.state_dict()))
P = vit_prefix_predictor(action_dim=10, **kw).eval()
S = P.tokens_per_block; B, C, K = 2, 3, 4; nb = C + K
h = torch.nn.functional.layer_norm(torch.randn(B, nb*S, 32), (32,))
idx = torch.arange(nb*S).unsqueeze(0).expand(B, -1)
A = torch.randn(B, nb, 10)
with torch.no_grad():
    p = P.forward_seq(h[:, :-S], idx[:, :-S], A)
    print("shape", tuple(p.shape), "expect", (B, (nb-1)*S, 32))
    # 1) 인과: 블록 4 뒤의 h 와 action 을 바꿔도 위치 <=3 출력 불변
    h2 = h.clone(); h2[:, 4*S:] = torch.randn_like(h2[:, 4*S:]); A2 = A.clone(); A2[:, 4:] = torch.randn_like(A2[:, 4:])
    p2 = P.forward_seq(h2[:, :-S], idx[:, :-S], A2)
    print("causal Δ(pos<=3) =", (p[:, :4*S]-p2[:, :4*S]).abs().max().item(), " Δ(pos>=4) =", (p[:, 4*S:]-p2[:, 4*S:]).abs().max().item())
    # 2) action 이 출력을 바꾸나 (블록 2 의 action 만 바꾸면 위치 2 부터 변함, 위치 <2 불변)
    A3 = A.clone(); A3[:, 2] += 1.0
    p3 = P.forward_seq(h[:, :-S], idx[:, :-S], A3)
    print("action blk2 change -> Δ(pos<2) =", (p[:, :2*S]-p3[:, :2*S]).abs().max().item(), " Δ(pos 2) =", (p[:, 2*S:3*S]-p3[:, 2*S:3*S]).abs().max().item())
    # 3) rollout 첫 스텝 == TF 위치 C-1 ; forward(ar=) == 개별
    r = P.rollout(h[:, :C*S], idx[:, :C*S], K, actions=A)
    print("rollout[0] vs tf[C-1] Δ =", (r[:, :S]-p[:, (C-1)*S:C*S]).abs().max().item())
    ptf, par = P(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": K, "actions": A})
    print("forward(ar) Δ tf =", (ptf-p).abs().max().item(), " Δ ar =", (par-r).abs().max().item())
    # 4) PrefixSpec(split) == dense 블록 인과 마스크 (action 토큰 포함, 블록 = 1+S)
    from src.models.rollout_predictor import build_prefix_mask
    hh = P._interleave_actions(P.predictor_embed(h[:, :-S]), A, nb-1)
    blk_of = torch.arange((nb-1)*(S+1)) // (S+1); m = build_prefix_mask(blk_of, 1, True)
    hh = P._run_blocks(hh, idx[:, :-S], m, T=nb-1, n_act=1)
    pd = torch.nn.functional.layer_norm(P.predictor_proj(P.predictor_norm(P._strip_actions(hh, nb-1))), (32,))
    print("split vs dense Δ =", (p-pd).abs().max().item())
P.train(); P.use_activation_checkpointing = True
ptf2, par2 = P(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": K, "actions": A})
print("ckpt path Δ tf =", (ptf2-ptf).abs().max().item(), " Δ ar =", (par2-par).abs().max().item())
(ptf2.abs().mean()+par2.abs().mean()).backward()
print("action_embed grad:", P.action_embed.weight.grad.abs().sum().item() > 0)
# 5) 자연 영상 AR ckpt (action 없음) 를 action_dim>0 모델에 strict=False 로 넣으면 빠지는 키가 action_embed 뿐인가
sd0 = P0.state_dict(); r = P.load_state_dict(sd0, strict=False); print("post-FT load: missing", r.missing_keys, "unexpected", r.unexpected_keys)
