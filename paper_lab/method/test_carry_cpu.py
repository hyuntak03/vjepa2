"""M2-t 운반 가지 CPU 검사 (진행자 2, 2026-10-22). python paper_lab/method/test_carry_cpu.py"""
import sys, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "paper_lab"))
from method import carry as CY
torch.manual_seed(0); torch.set_num_threads(4)
B, G, D, nb = 2, 16, 32, 6; S = G * G
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond); print(("ok  " if cond else "FAIL"), name)


h = torch.randn(B, nb * S, D)
p_tf = torch.randn(B, (nb - 1) * S, D)
head = CY.CarryHead(D, n_src=3)
# 1 시작값: 속도 0 · 문 σ(−2) · 출발 균등
q, aux = CY.apply_carry(head, p_tf, h, list(range(1, nb)), S)
Hb = h.view(B, nb, G, G, D); P = p_tf.view(B, nb - 1, G, G, D)
t = 3; srcs = [max(t - 1 - l, 0) for l in range(3)]
exp = torch.sigmoid(torch.tensor(-2.0)) * torch.stack([Hb[:, s_] for s_ in srcs]).mean(0) + (1 - torch.sigmoid(torch.tensor(-2.0))) * P[:, t - 1]
check("시작값 = 0.12 · 출발 평균 + 0.88 · p", torch.allclose(q.view(B, nb - 1, G, G, D)[:, t - 1], exp, atol=1e-5))
# 2 정확한 옮기기: 속도 (+1, 0) 칸/블록, 문 열림, 출발 = 바로 앞 블록 → 목표 t 의 칸 (y, x) = h_{t−1}[y, x−1]
with torch.no_grad():
    head.out.bias.zero_(); head.out.bias[0] = 1.0; head.out.bias[2] = 30.0; head.out.bias[3] = 30.0
q, _ = CY.apply_carry(head, p_tf, h, list(range(1, nb)), S)
Q = q.view(B, nb - 1, G, G, D)
check("오른쪽 한 칸 옮기기 (안쪽 칸)", torch.allclose(Q[:, t - 1, 4:12, 5:12], Hb[:, t - 1, 4:12, 4:11], atol=1e-4))
# 3 rollout: 출발 = 문맥 끝, 목표 C+k 는 (k+1) 칸 옮김
C = 4; p_ar = torch.randn(B, 2 * S, D)
q, _ = CY.apply_carry(head, p_ar, h[:, :C * S], [C, C + 1], S, ctx_blocks=C)
Q = q.view(B, 2, G, G, D)
check("rollout 두 걸음 = 두 칸 옮김", torch.allclose(Q[:, 1, 4:12, 6:12], Hb[:, C - 1, 4:12, 4:10], atol=1e-4))
# 4 기울기가 속도 · 문 · 출발 가중으로 흐른다
head2 = CY.CarryHead(D, n_src=3)
tgt = torch.roll(Hb, shifts=1, dims=3)[:, 1:].reshape(B, -1, D)        # 내용이 오른쪽으로 한 칸 (정답)
q, aux = CY.apply_carry(head2, p_tf, h, list(range(1, nb)), S)
(q - tgt).abs().mean().backward()
gb = head2.out.bias.grad
check("기울기: 속도 x · 문 · 출발 가중", gb[0].abs() > 0 and gb[2].abs() > 0 and gb[3:].abs().sum() > 0)
# 4b 부호: 무작위 토큰은 칸 사이가 매끄럽지 않아 bilinear 기울기 방향이 잡음이다 → 매끄러운 토큰 · 속도 0.3 에서 본다
import torch.nn.functional as F
hs = F.avg_pool2d(torch.randn(B * nb, D, G + 4, G + 4), 5, 1).view(B, nb, D, G, G).permute(0, 1, 3, 4, 2).contiguous()
head4 = CY.CarryHead(D, n_src=3)
with torch.no_grad():
    head4.out.bias[0] = 0.3; head4.out.bias[2] = 30.0; head4.out.bias[3] = 30.0
tgt4 = torch.roll(hs, shifts=1, dims=3)[:, 1:].reshape(B, -1, D)
q4, _ = CY.apply_carry(head4, p_tf, hs.reshape(B, -1, D), list(range(1, nb)), S)
(q4.view(B, nb - 1, G, G, D)[:, :, 3:13, 3:13] - tgt4.view(B, nb - 1, G, G, D)[:, :, 3:13, 3:13]).abs().mean().backward()
check("속도 x 기울기 부호 = 오른쪽으로 더 (−, 매끄러운 토큰)", head4.out.bias.grad[0] < 0)
# 5 속도 손실: 라벨 (앞으로 1 칸 오른쪽) 이면 v = (+1, 0) 일 때 0
H = (1, 2, 4, 8); flow = torch.zeros(B, nb, 4, 2, G, G); flow[:, :, 1, 0] = 1.0; valid = torch.ones(B, nb, 4, G, G, dtype=torch.bool)
v = torch.zeros(B, nb - 1, G, G, 2); v[..., 0] = 1.0
check("속도 손실 0 (맞는 속도)", float(CY.velocity_loss(v, flow, valid, H)) < 1e-6)
check("속도 손실 > 0 (속도 0)", float(CY.velocity_loss(torch.zeros_like(v), flow, valid, H)) > 0.9)
# 6 학습 끼우기: 가짜 predictor 로 ar_losses 가 운반 손실을 내고 head 로 기울기가 간다
from app.vjepa_frozen import ar as AR
from app.vjepa_frozen import flow_head as FHM


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.lin = torch.nn.Linear(D, D)

    def forward(self, hh, ar):
        C_, R_ = ar["n_ctx_blocks"], ar["rollout_steps"]
        return self.lin(hh[:, :-S]), self.lin(hh[:, (C_ - 1) * S:(C_ - 1 + R_) * S])


head3 = CY.CarryHead(D, n_src=3)
orig_l, orig_f = CY.install_train(head3, w_vel=1.0)
try:
    loss, jl, sl, R = AR.ar_losses(Dummy(), h, S, 4, 2, 1.0, 2, out={})
    loss.backward()
    check("끼운 ar_losses: 손실 · head 기울기", torch.isfinite(loss) and head3.out.weight.grad is not None and CY.HOLD["v_tf"] is not None)
finally:
    AR.ar_losses, FHM.ar_flow_loss = orig_l, orig_f
print("ALL OK" if ok else "SOME FAILED")
# 7 predictor 에 붙이기 · 평가에서 불러 걸기 (가짜 build_predictor · 가짜 rollout)
from app.vjepa_frozen import utils as U
import tempfile, os


class Dummy2(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.lin = torch.nn.Linear(D, D)

    def rollout(self, h_ctx, idx, K):
        return self.lin(h_ctx[:, -S:]).repeat(1, K, 1)


orig_b = U.build_predictor
U.build_predictor = lambda MD, ed, st, dev, *a, **k: Dummy2()
try:
    CY.install_attach(n_src=3)
    pr = U.build_predictor({}, D, None, "cpu")
    check("붙이기: carry_head 가 predictor 매개변수 안에", any(n.startswith("carry_head.") for n, _ in pr.named_parameters()))
    with torch.no_grad():
        pr.carry_head.out.bias[2] = 30.0                         # 문 열림 → rollout 출력이 바뀌어야 한다
    f = os.path.join(tempfile.mkdtemp(), "ck.pt"); torch.save({"predictor": pr.state_dict()}, f)
    pr2 = Dummy2(); pr2.load_state_dict({k: v for k, v in pr.state_dict().items() if not k.startswith("carry_head.")})
    base = pr2.rollout(h[:, :4 * S], None, 2)
    okw = CY.wrap_for_eval(pr2, f)
    out = pr2.rollout(h[:, :4 * S], None, 2)
    check("평가: 체크포인트의 head 를 불러 rollout 에 건다", okw and not torch.allclose(out, base))
finally:
    U.build_predictor = orig_b
print("ALL OK (7)" if ok else "SOME FAILED (7)")
