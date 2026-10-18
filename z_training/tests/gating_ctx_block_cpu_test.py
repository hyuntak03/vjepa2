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

# ---------------------------------------------------------------- kind=ar + gating (2026-10-17)
from analysis.intphys2.model import _ArScorer
kwa = dict(kw, mask_mode="ar", ar_feedback="state")
def build_ar(**extra):
    torch.manual_seed(0)
    return vit_prefix_predictor(**kwa, **extra).eval()
nb = C + K
idx_all = AR.block_indices(B, nb, S, "cpu")
arg_ar = {"idx": idx_all, "n_ctx_blocks": C, "rollout_steps": K}
# 6 끔 = 기존
a0, a1 = build_ar(), build_ar(gating_residual=False)
with torch.no_grad():
    o0, o1 = a0(hb, ar=dict(arg_ar)), a1(hb, ar=dict(arg_ar))
assert torch.equal(o0[0], o1[0]) and torch.equal(o0[1], o1[1])
print("6 OK  ar gating 끔 = 기존과 같다")
# 7 초기화 직후: TF 위치 b = 입력 블록 b, rollout 전 걸음 = 마지막 문맥 블록
ag = build_ar(gating_residual=True)
with torch.no_grad():
    t, r = ag(hb, ar=dict(arg_ar))
assert torch.allclose(t, hb[:, :-S], atol=1e-6), float((t - hb[:, :-S]).abs().max())
Alast = hb[:, (C - 1) * S: C * S].unsqueeze(1).expand(B, K, S, D).reshape(B, K * S, D)
assert torch.allclose(r, Alast, atol=1e-6), float((r - Alast).abs().max())
print("7 OK  ar 초기화 직후 TF = 입력 블록, rollout = 마지막 문맥 블록")
# 8 학습 한 step: gradient · gate 로그
ag.train(); aux = {}
loss, jl, sl, R = AR.ar_losses(ag, hb, S, C, K, 1.0, K, aux=aux)
loss.backward()
assert ag.keep_gate.weight.grad is not None and float(ag.predictor_proj.weight.grad.abs().sum()) > 0
assert {"gate", "gate_mov", "gate_sta", "gate_k1", "gate_kR"} <= set(aux), aux
print("8 OK  ar loss %.4f, gate 로그" % float(loss), {k: round(v, 3) for k, v in aux.items()})
# 9 흔든 뒤: TF 첫 예측 자리 (위치 C-1) = rollout 첫 걸음 (둘 다 anchor = 블록 C-1), 채점 래퍼 = rollout_kv
with torch.no_grad():
    ag.keep_gate.weight.normal_(0, 0.5); ag.predictor_proj.weight.normal_(0, 0.05)
ag.eval()
with torch.no_grad():
    t, r = ag(hb, ar=dict(arg_ar))
    assert torch.allclose(t[:, (C - 1) * S: C * S], r[:, :S], atol=1e-5), float((t[:, (C - 1) * S: C * S] - r[:, :S]).abs().max())
    sc = _ArScorer(ag)
    raw = hb[:, : C * S] * 2.0 + 0.3          # 채점기 블록 encoder 출력 (LN 전) — scorer 가 LN
    got = sc(raw, idx_all[:, : C * S], idx_all[:, C * S:])
    want = ag.rollout_kv(hb[:, : C * S], idx_all[:, : C * S], K)
assert torch.allclose(got, want, atol=1e-4), float((got - want).abs().max())
print("9 OK  TF 위치 C-1 = rollout 첫 걸음, 채점 래퍼 = rollout_kv (max |Δ|", float((got - want).abs().max()), ")")
try:
    ag(hb, ar=dict(arg_ar, dyn=True)); raise AssertionError("dyn 과 같이 통과했다")
except ValueError:
    print("   OK  multistart · dyn 과 같이 쓰면 죽는다")
print("ar 전부 통과")

# ---------------------------------------------------------------- ar gating_anchor=prev (2026-10-18)
# 10 기본 (last_obs) 명시 = 생략과 같다
ap0, ap1 = build_ar(gating_residual=True), build_ar(gating_residual=True, gating_anchor="last_obs")
with torch.no_grad():
    for m in (ap0, ap1):
        m.keep_gate.weight.normal_(0, 0.5, generator=torch.Generator().manual_seed(7)); m.predictor_proj.weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(8))
    assert torch.equal(ap0(hb, ar=dict(arg_ar))[1], ap1(hb, ar=dict(arg_ar))[1])
print("10 OK  gating_anchor=last_obs = 기본")
# 11 prev: 초기화 직후 전 걸음 = 마지막 문맥 블록 (LN(A) = A), TF 는 last_obs 와 같다
pp = build_ar(gating_residual=True, gating_anchor="prev")
with torch.no_grad():
    t, r = pp(hb, ar=dict(arg_ar))
assert torch.allclose(r, Alast, atol=1e-5) and torch.allclose(t, hb[:, :-S], atol=1e-6)
print("11 OK  prev 초기화 직후 rollout = 마지막 문맥 블록, TF = 입력 블록")
# 12 흔든 뒤: 걸음 1 은 last_obs 와 같고, 걸음 2.. 는 다르며, 걸음 r = g·LN(ŷ_{r-1}) + proj(s) 를 직접 재구성해 맞춘다
with torch.no_grad():
    pp.load_state_dict(ap0.state_dict())
    t2, r2 = pp(hb, ar=dict(arg_ar)); _, r0 = ap0(hb, ar=dict(arg_ar))
assert torch.allclose(r2[:, :S], r0[:, :S], atol=1e-6) and not torch.allclose(r2[:, S:], r0[:, S:], atol=1e-4)
assert torch.allclose(t2, ap0(hb, ar=dict(arg_ar))[0], atol=1e-6)
_, st = pp.rollout_kv(hb[:, : C * S], idx_all[:, : C * S], K, return_states=True)
with torch.no_grad():
    for k in range(1, K):
        s_k = st[:, k * S:(k + 1) * S]
        g = torch.sigmoid(pp.keep_gate(s_k).float()) / pp.sig_b0
        prevA = torch.nn.functional.layer_norm(r2[:, (k - 1) * S: k * S], (D,))
        want = g * prevA + pp.predictor_proj(s_k).float()
        assert torch.allclose(r2[:, k * S:(k + 1) * S], want, atol=1e-5), (k, float((r2[:, k * S:(k + 1) * S] - want).abs().max()))
print("12 OK  prev: 걸음 1 = last_obs, 걸음 r = g·LN(ŷ_{r-1}) + proj(s) (직접 재구성 일치), TF 는 그대로")
with torch.no_grad():
    got = _ArScorer(pp)(hb[:, : C * S] * 2.0 + 0.3, idx_all[:, : C * S], idx_all[:, C * S:])
assert torch.allclose(got, r2, atol=1e-4), float((got - r2).abs().max())
print("13 OK  prev 채점 래퍼 = 학습 경로 rollout")
try:
    build(gating_residual=True, gating_anchor="prev"); raise AssertionError("ctx_state 에 prev 가 통과했다")
except ValueError:
    print("   OK  ctx_state 에 prev 는 죽는다")
print("prev 전부 통과")
