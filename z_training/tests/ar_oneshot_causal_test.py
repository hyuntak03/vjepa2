"""ar_oneshot_causal 검사 (2026-11-04, CPU, 작은 predictor). GPU 불필요.
  python z_training/tests/ar_oneshot_causal_test.py
1. 블록 인과판 = 명시적 dense 블록 인과 마스크 (블록 b query 는 블록 <= b key) 와 값이 같다.
2. 블록 인과판: 미래 블록 t 의 출력은 문맥 마지막 블록을 바꾸면 바뀌고, 문맥 블록 j 의 은닉은 j 뒤 블록을 바꿔도 안 바뀐다.
3. 양방향판 (ar_oneshot 만) 은 1 의 dense 마스크와 다르다 (마스크가 실제로 걸렸다는 대조).
"""
import os, sys, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.rollout_predictor import vit_prefix_predictor

torch.manual_seed(0)
kw = dict(img_size=64, patch_size=16, tubelet_size=2, num_frames=16, embed_dim=32, predictor_embed_dim=48,
          depth=2, num_heads=4, num_mask_tokens=1, use_rope=True, use_sdpa=True, mask_mode="ar",
          ar_feedback="state", ar_oneshot=True)
pc = vit_prefix_predictor(ar_oneshot_causal=True, **kw).eval()
pb = vit_prefix_predictor(**kw).eval()
pb.load_state_dict(pc.state_dict())
for p in pc.parameters():                       # 0 초기화된 mask token 을 무작위로 (검사 민감도)
    pass
with torch.no_grad():
    pc.mask_tokens[0].normal_(); pb.mask_tokens[0].copy_(pc.mask_tokens[0])
S = pc.tokens_per_block                          # 16
C, K, B = 3, 2, 2
x = torch.nn.functional.layer_norm(torch.randn(B, C * S, 32), (32,))
idx = torch.arange(C * S)[None].expand(B, -1)

def dense_ref(model, x_ctx, idx_ctx, K):
    h_ctx = model._ar_embed_obs(x_ctx)
    q = model.mask_tokens[0].expand(B, K * S, -1)
    idx_q = idx_ctx[:, -1:] + 1 + torch.arange(K * S)[None]
    ii = torch.cat([idx_ctx, idx_q], 1)
    blk = ii[0] // S
    m = blk.view(1, -1) <= blk.view(-1, 1)        # (N, N) True = attend
    h = model._run_blocks(torch.cat([h_ctx, q], 1), ii, m)
    return model.predictor_norm(h), model._ar_out(model.predictor_norm(h[:, x_ctx.size(1):]))

with torch.no_grad():
    out_c = pc._ar_oneshot(x, idx, K)
    st_ref, out_ref = dense_ref(pc, x, idx, K)
    d1 = (out_c - out_ref).abs().max().item()
    out_b = pb._ar_oneshot(x, idx, K)
    d3 = (out_b - out_ref).abs().max().item()
    x2 = x.clone(); x2[:, (C - 1) * S:] += 1.0      # 문맥 마지막 블록만 바꾼다
    st2, out2 = dense_ref(pc, x2, idx, K)
    d_fut = (pc._ar_oneshot(x2, idx, K) - out_c).abs().max().item()
    d_ctx0 = (st2[:, : (C - 1) * S] - st_ref[:, : (C - 1) * S]).abs().max().item()
print(f"1 causal vs dense block-causal mask  max|d| = {d1:.2e}  (기대 ~1e-6)")
print(f"2 last ctx block changed -> future out max|d| = {d_fut:.2e} (기대 > 0) · earlier ctx hidden max|d| = {d_ctx0:.2e} (기대 0)")
print(f"3 bidirectional vs dense causal       max|d| = {d3:.2e}  (기대 > 0)")
ok = d1 < 1e-5 and d_fut > 1e-3 and d_ctx0 < 1e-6 and d3 > 1e-3
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
