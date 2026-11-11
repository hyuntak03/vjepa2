"""곡률 항 검사 (CPU): 직선 0 · 독립 잡음 cos −0.5 · 평균 제거가 공통 궤적을 지운다.  /venv/vjepa/bin/python z_training/tests/curv_test.py"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from app.vjepa_frozen.curv import curv_term
g = torch.Generator().manual_seed(0)
t = torch.arange(15.).view(1, -1, 1)
u = torch.randn(64, 1, 384, generator=g)
l, c = curv_term(t * u)
assert float(l) < 1e-4, l
l, c = curv_term(torch.randn(256, 15, 384, generator=g))
assert abs(float(c) + 0.5) < 0.02, c
common = torch.randn(1, 15, 384, generator=g).expand(32, -1, -1)
l, c = curv_term(common + t * u[:32])
assert float(l) < 1e-3, l
l2, _ = curv_term(torch.randn(64, 15, 8, generator=g), 0.3)
assert torch.isfinite(l2)
p = torch.randn(6, 15, 384, requires_grad=True)
curv_term(p)[0].backward()
assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
print("curv_test 통과")
