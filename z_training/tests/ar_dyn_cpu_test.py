"""시간 변화량 보조 손실 (dyn_head, 2026-10-15) 정합성 (CPU, 작은 모델). python z_training/tests/ar_dyn_cpu_test.py
검사: dyn_dim 0 이면 state_dict · 출력이 기존과 같다 · dyn 경로의 p_tf / p_ar 가 기존 경로와 같다 · rollout_kv 상태 == rollout_state 상태 ·
      a_t 가 |s_{t+1} − s_t| 의 Linear 다 · ar_losses 의 쌍 정렬 (TF: 예측 블록 1..nb-1, rollout: C..C+R-1) · grad 가 predictor 몸통까지 간다"""
import torch, sys; sys.path.insert(0, '.')
from src.models.rollout_predictor import vit_prefix_predictor
from app.vjepa_frozen import ar as AR
torch.manual_seed(0)
kw = dict(img_size=64, patch_size=16, tubelet_size=2, num_frames=48, embed_dim=32, predictor_embed_dim=96, depth=3, num_heads=2, use_sdpa=True)
for ad in (0, 10):
    P0 = vit_prefix_predictor(mask_mode="ar_state", action_dim=ad, **kw).train()
    P = vit_prefix_predictor(mask_mode="ar_state", action_dim=ad, dyn_dim=40, **kw).train()
    r = P.load_state_dict(P0.state_dict(), strict=False)
    assert sorted(r.missing_keys) == ["dyn_head.bias", "dyn_head.weight"] and not r.unexpected_keys, r
    S = P.tokens_per_block; B, C, K = 2, 3, 5; nb = C + K
    h = torch.nn.functional.layer_norm(torch.randn(B, nb*S, 32), (32,)); idx = torch.arange(nb*S).unsqueeze(0).expand(B, -1)
    A = torch.randn(B, nb, ad) if ad else None
    ar = {"idx": idx, "n_ctx_blocks": C, "rollout_steps": K, "actions": A}
    with torch.no_grad():
        q_tf, q_ar = P0(h, ar=ar)
        p_tf, p_ar, a_tf, a_ar = P(h, ar=dict(ar, dyn=True))
        _, st_kv = P.rollout_kv(h[:, :C*S], idx[:, :C*S], K, actions=A, return_states=True)
        _, st_tf = P.forward_seq(h[:, :-S], idx[:, :-S], A, return_states=True)
        # rollout_state 의 상태 (전체 재계산판) 를 직접 재현: 마지막 블록 상태 = predictor_norm 출력
        r_full = P.rollout_state(h[:, :C*S], idx[:, :C*S], K, actions=A)
        r_kv = P.rollout_kv(h[:, :C*S], idx[:, :C*S], K, actions=A)
        a_ref = P.dyn_head((st_kv.view(B, K, S, -1)[:, 1:] - st_kv.view(B, K, S, -1)[:, :-1]).abs()).flatten(1, 2)
    print(f"[ad={ad:2d}] dyn 경로 p_tf/p_ar == 기존 {max((p_tf-q_tf).abs().max(), (p_ar-q_ar).abs().max()):.1e} | "
          f"KV vs full {(r_full-r_kv).abs().max():.1e} | a_ar == Linear(|Δs|) {(a_ar-a_ref).abs().max():.1e} | "
          f"shape a_tf {tuple(a_tf.shape)} (기대 {(B, (nb-2)*S, 40)}) a_ar {tuple(a_ar.shape)} (기대 {(B, (K-1)*S, 40)})")
    assert a_tf.shape == (B, (nb - 2) * S, 40) and a_ar.shape == (B, (K - 1) * S, 40)
    # ar_losses 쌍 정렬: pe 를 블록 번호가 박힌 값으로 두면 b_k = |PE_{k+1} − PE_k| 가 정확히 1 → a 가 무엇이든 정렬이 맞는지 본다
    pe = torch.arange(nb).repeat_interleave(S).view(1, nb*S, 1).float().expand(B, -1, 40) * 1.0
    pe = pe * torch.linspace(1, 2, nb).repeat_interleave(S).view(1, nb*S, 1) ** 0   # 블록마다 상수
    ex = {}
    P.zero_grad()
    loss, jl, sl, Rr = AR.ar_losses(P, h, S, C, K, 1.0, 8, actions=A, pe=pe, dyn_weight=0.5, extra=ex)
    loss.backward()
    with torch.no_grad():
        p_tf, p_ar, a_tf, a_ar = P(h, ar=dict(ar, dyn=True))
        exp_tf = (a_tf - 1.0).abs().mean(); exp_ar = (a_ar - 1.0).abs().mean()
    print(f"        ar_losses dyn_tf {ex['dyn_tf']:.5f} == {exp_tf:.5f} · dyn_ar {ex['dyn_ar']:.5f} == {exp_ar:.5f} · "
          f"loss == jl+sl+0.5(dyn) {abs(loss.item() - (jl + sl + 0.5*(ex['dyn_tf']+ex['dyn_ar'])).item()):.1e}")
    g_body = P.predictor_blocks[0].attn.qkv.weight.grad.abs().sum().item(); g_head = P.dyn_head.weight.grad.abs().sum().item()
    print(f"        grad: dyn_head {g_head > 0} · predictor_blocks[0] {g_body > 0}")
print("ok")
