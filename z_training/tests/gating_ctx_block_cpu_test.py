"""gating_residual + ctx_input=block CPU 검사 (2026-10-17). 모델·데이터 없이 몇 초.

  /venv/vjepa/bin/python z_training/tests/gating_ctx_block_cpu_test.py

1. gating 끔: 파라미터 이름 · 값 · 출력이 기존 (kwarg 없음) 과 같다
2. gating 켬, 초기화 직후: TF · rollout 출력 = anchor 정확히 (ŷ = A)
3. 한 step 뒤 gate · proj 에 gradient 가 흐른다
4. 채점 래퍼 (_CtxStateScorer, ctx_block) 가 학습 경로 rollout 과 같은 값을 낸다
5. 문맥 encoder ≠ target 일 때 anchor 래퍼 (_AnchorCtxEncoder) 가 타깃 블록 LN 을 anchor 로 넘긴다
"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.rollout_predictor import vit_prefix_predictor
from app.vjepa_frozen import ar as AR
from analysis.intphys2.model import _CtxStateScorer, _AnchorCtxEncoder

D, S, C, K = 64, 16, 3, 4                  # img 64 / patch 16 → 4×4 = 16 토큰 / 블록
kw = dict(img_size=64, patch_size=16, tubelet_size=2, num_frames=16, embed_dim=D, predictor_embed_dim=96,
          depth=2, num_heads=3, num_mask_tokens=10, mask_mode="ctx_state", use_sdpa=True)

def build(**extra):
    torch.manual_seed(0)
    return vit_prefix_predictor(**kw, **extra).eval()

B = 2
torch.manual_seed(1)
hb = torch.nn.functional.layer_norm(torch.randn(B, (C + K) * S, D), (D,))   # 블록별 LN(h) 흉내
z, tgt = hb[:, : C * S], hb[:, C * S:]
A = z[:, -S:]
idx = AR.block_indices(B, C + K, S, "cpu")
arg = {"h_obs": tgt, "idx": idx, "n_ctx_blocks": C, "rollout_steps": K}

# 1
p0, p1 = build(), build(gating_residual=False)
assert list(p0.state_dict()) == list(p1.state_dict()) and all(torch.equal(p0.state_dict()[k], p1.state_dict()[k]) for k in p0.state_dict())
with torch.no_grad():
    a0, a1 = p0(z, ar=dict(arg)), p1(z, ar=dict(arg, anchor=A))      # 끈 모델은 anchor 를 무시
assert torch.equal(a0[0], a1[0]) and torch.equal(a0[1], a1[1]), "gating 끔인데 출력이 다르다"
print("1 OK  gating 끔 = 기존과 같다 (파라미터", len(p0.state_dict()), "개)")

# 2
pg = build(gating_residual=True)
extra = sorted(set(pg.state_dict()) - set(p0.state_dict()))
assert extra == ["keep_gate.bias", "keep_gate.weight"], extra
with torch.no_grad():
    t, r = pg(z, ar=dict(arg, anchor=A))
Ak = A.unsqueeze(1).expand(B, K, S, D).reshape(B, K * S, D)
assert torch.allclose(t, Ak, atol=1e-6) and torch.allclose(r, Ak, atol=1e-6), (float((t - Ak).abs().max()), float((r - Ak).abs().max()))
print("2 OK  초기화 직후 TF · rollout = anchor (max |Δ|", float((r - Ak).abs().max()), ")")
try:
    pg(z, ar=dict(arg))
    raise AssertionError("anchor 없이 통과했다")
except ValueError:
    print("   OK  anchor 없으면 죽는다")

# 3
pg.train()
aux = {}
loss, jl, sl, R = AR.ctx_state_losses(pg, z, tgt, tgt, S, C, K, 1.0, K, 1.0, aux=aux, anchor=A)
loss.backward()
gw, pw = pg.keep_gate.weight.grad, pg.predictor_proj.weight.grad
assert gw is not None and pw is not None and float(pw.abs().sum()) > 0, "gradient 가 안 흐른다"
print("3 OK  loss %.4f | grad |proj| %.3e |gate w| %.3e |gate b| %.3e" % (float(loss), float(pw.abs().sum()), float(gw.abs().sum()),
      float(pg.keep_gate.bias.grad.abs().sum())), "| gate 로그", {k: round(v, 3) for k, v in aux.items()})

# 4  (gate 를 흔들어 초기값이 아닌 상태로)
with torch.no_grad():
    pg.keep_gate.weight.normal_(0, 0.5); pg.predictor_proj.weight.normal_(0, 0.05)
pg.eval()
with torch.no_grad():
    want = pg.render(pg.rollout_state_kv(z, idx[:, : C * S], K), A)
    raw = z * 3.0 + 0.5                       # 채점기의 _ArBlockEncoder 는 LN 전 값을 낸다 — scorer 가 LN 한다
    sc = _CtxStateScorer(pg, ctx_block=True)
    got = sc(raw, idx[:, : C * S], idx[:, C * S: (C + K) * S])
assert torch.allclose(got, want, atol=1e-5), float((got - want).abs().max())
print("4 OK  채점 래퍼 = 학습 경로 rollout (max |Δ|", float((got - want).abs().max()), ")")

# 5  문맥 = 다른 encoder (블록 LN 은 scorer), anchor = 타깃 encoder 블록 — 가짜 encoder 둘
class FakeBlockEnc(torch.nn.Module):          # (B, 3, T, H, W) -> 블록마다 (B, nb*S, D), 블록 내용의 결정적 함수
    def __init__(self, seed):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(3 * 2 * 16 * 16, D, generator=g) * 0.05; self.embed_dim = D
    def forward(self, x, masks=None):
        B, Cc, T, H, W = x.shape; nb = T // 2
        p = x.reshape(B, Cc, nb, 2, 4, 16, 4, 16).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, nb * S, -1)
        h = p @ self.W
        return h if masks is None else torch.gather(h, 1, masks[0].unsqueeze(-1).expand(-1, -1, D))
ctxE, tgtE = FakeBlockEnc(3), FakeBlockEnc(4)
x = torch.randn(B, 3, 2 * (C + K), 64, 64)
wrap = _AnchorCtxEncoder(ctxE, tgtE, 2, S)
sc2 = _CtxStateScorer(pg, anchor_src=wrap, ctx_block=True); sc2.anchor_from_z = False
with torch.no_grad():
    zc = wrap(x, masks=[idx[:, : C * S]])
    got = sc2(zc, idx[:, : C * S], idx[:, C * S: (C + K) * S])
    zl = torch.nn.functional.layer_norm(ctxE(x)[:, : C * S], (D,))
    Aw = torch.nn.functional.layer_norm(tgtE(x)[:, (C - 1) * S: C * S], (D,))
    want = pg.render(pg.rollout_state_kv(zl, idx[:, : C * S], K), Aw)
assert torch.allclose(got, want, atol=1e-5), float((got - want).abs().max())
try:
    sc2(zc, idx[:, : C * S], idx[:, C * S: (C + K) * S]); raise AssertionError("옛 anchor 를 다시 썼다")
except RuntimeError:
    pass
print("5 OK  다른 문맥 encoder + 타깃 anchor 래퍼 = 기대값 (max |Δ|", float((got - want).abs().max()), "), 두 번 꺼내면 죽는다")
print("전부 통과")
