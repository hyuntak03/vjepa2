"""ar_state (상태 되먹임) 정합성 (CPU, 작은 모델). python z_training/tests/ar_state_cpu_test.py
검사: TF 는 ar 와 같은 열 (관측 + type[0]) · 인과성 · rollout 첫 걸음 == TF 위치 C-1 · KV 판 == 전체 재계산판 (값·grad) ·
      되먹임이 render 가 아니라 상태임 (ar 와 다른 값) · action 유무 · post-FT 로드 시 새 키 = type_embed (+ action_embed)"""
import torch, sys; sys.path.insert(0, '.')
from src.models.rollout_predictor import vit_prefix_predictor
torch.manual_seed(0)
kw = dict(img_size=64, patch_size=16, tubelet_size=2, num_frames=48, embed_dim=32, predictor_embed_dim=96, depth=3, num_heads=2, use_sdpa=True)
for ad in (0, 10):
    P = vit_prefix_predictor(mask_mode="ar_state", action_dim=ad, **kw).train()
    S = P.tokens_per_block; B, C, K = 2, 3, 5; nb = C + K
    h = torch.nn.functional.layer_norm(torch.randn(B, nb*S, 32), (32,)); idx = torch.arange(nb*S).unsqueeze(0).expand(B, -1)
    A = torch.randn(B, nb, ad) if ad else None
    with torch.no_grad():
        tf = P.forward_seq(h[:, :-S], idx[:, :-S], A)
        h2 = h.clone(); h2[:, 5*S:] = torch.randn_like(h2[:, 5*S:])
        tf2 = P.forward_seq(h2[:, :-S], idx[:, :-S], A)
        r_full = P.rollout_state(h[:, :C*S], idx[:, :C*S], K, actions=A)
        r_kv = P.rollout_kv(h[:, :C*S], idx[:, :C*S], K, actions=A)
    print(f"[ad={ad:2d}] causal Δ(pos<=4) {(tf[:, :5*S]-tf2[:, :5*S]).abs().max():.1e}  | rollout[0] vs TF[C-1] {(r_full[:, :S]-tf[:, (C-1)*S:C*S]).abs().max():.1e}"
          f"  | KV vs full value {(r_full-r_kv).abs().max():.1e}")
    g = []
    for f in (lambda: P.rollout_state(h[:, :C*S], idx[:, :C*S], K, actions=A), lambda: P.rollout_kv(h[:, :C*S], idx[:, :C*S], K, actions=A)):
        P.zero_grad(); f().pow(2).mean().backward(); g.append([p.grad.clone() for p in P.parameters() if p.grad is not None])
    print(f"        KV vs full grad {max((a-b).abs().max().item() for a, b in zip(*g)):.1e}  (n {len(g[0])}/{len(g[1])}); type_embed grad {P.type_embed.grad.abs().sum().item() > 0}")
    # 되먹임이 상태인지: 같은 가중치로 ar 모드를 흉내 (type 0 으로) 하면 값이 달라야 한다
    Q = vit_prefix_predictor(mask_mode="ar", action_dim=ad, **kw); Q.load_state_dict({k: v for k, v in P.state_dict().items() if k != "type_embed"}, strict=True)
    with torch.no_grad():
        rq = Q.rollout_kv(h[:, :C*S], idx[:, :C*S], K, actions=A)
    print(f"        ar (render 되먹임) 와 걸음 2 이후 다름: {(rq[:, S:]-r_kv[:, S:]).abs().max():.2e} (> 0 이어야)")
    # forward(ar=) 경로
    P.kv_rollout = True; a1 = P(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": K, "actions": A})[1]
    P.kv_rollout = False; a2 = P(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": K, "actions": A})[1]
    print(f"        forward(ar=) kv vs full {(a1-a2).abs().max():.1e}")
    r = P.load_state_dict(vit_prefix_predictor(mask_mode="ar", **kw).state_dict(), strict=False)
    print(f"        ar ckpt → ar_state 로드: missing {sorted(r.missing_keys)} unexpected {r.unexpected_keys}")
