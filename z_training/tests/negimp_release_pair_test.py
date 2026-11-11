#!/usr/bin/env python3
"""negimp 릴리즈 짝 손실 (release_pair_loss, 2026-11-06) CPU 검사.
  /venv/vjepa/bin/python z_training/tests/negimp_release_pair_test.py
(1) λ 0 이면 L1 평균과 같다  (2) 짝 없는 샘플 (code 0) 은 불가능 항이 0  (3) 손실 식 = dp + λ (dp − di)
(4) 정답이 하나인 미래에서는 λ 3 으로 경사 하강해도 가능 정답으로 간다 (넘어가지 않는다)  (5) 배경 (가능 = 불가능) 칸은 λ 와 무관하게 당긴다."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from app.vjepa_frozen.negimp import release_pair_loss

torch.manual_seed(0)
B, M, D = 4, 6, 8
h = torch.randn(B, M, D); hi = h.clone(); hi[:, :2] += 3.0          # 불가능 = 미래 앞 2 토큰만 다름 (물체 칸), 나머지 = 배경 같음
p = torch.randn(B, M, D)
imp = torch.tensor([1, 2, 0, 3])
l0, _ = release_pair_loss(p, h, hi, imp, 1.0, weight=0.0)
assert torch.allclose(l0, (p - h).abs().mean()), "λ 0 ≠ L1"
l3, st = release_pair_loss(p, h, hi, imp, 1.0, weight=3.0)
dp = (p - h).abs().mean(dim=(1, 2)); di = (p - hi).abs().mean(dim=(1, 2))
ref = (dp + 3.0 * torch.where(imp > 0, dp - di, torch.zeros_like(dp))).mean()
assert torch.allclose(l3, ref), "손실 식"
assert int(st[3]) == 3 and int(st[0]) == 4, "stats 짝 수"
# (4) 경사 하강: λ 3 에서도 p → h
q = torch.zeros(B, M, D, requires_grad=True)
opt = torch.optim.SGD([q], lr=1.0)          # 손실이 원소 평균 (B·M·D = 192) 이라 원소당 걸음이 lr / 192 — 크게 시작해 줄인다
n_it = 6000
for it in range(n_it):
    for g_ in opt.param_groups:
        g_["lr"] = 20.0 * (1 - it / n_it) + 0.01
    opt.zero_grad(); l, _ = release_pair_loss(q, h, hi, torch.ones(B, dtype=torch.long), 1.0, weight=3.0); l.backward(); opt.step()
err = (q.detach() - h).abs().max().item()
assert err < 0.1, f"정답으로 안 감 (최대 차 {err:.3f})"
# (5) 배경 칸 기울기는 λ 와 무관 (가능 = 불가능 이면 차 항이 상쇄)
g = {}
for lam in (0.0, 3.0):
    q = p.clone().requires_grad_(True); l, _ = release_pair_loss(q, h, hi, torch.ones(B, dtype=torch.long), 1.0, weight=lam); l.backward(); g[lam] = q.grad[:, 2:].clone()
assert torch.allclose(g[0.0], g[3.0]), "배경 칸 기울기가 λ 에 따라 바뀐다"
print(f"OK release_pair_loss: λ0 = L1 · 식 · 짝 수 · 정답 수렴 (최대 차 {err:.4f}) · 배경 칸 기울기 λ 무관")
