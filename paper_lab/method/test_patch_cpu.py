#!/usr/bin/env python3
"""patch.py CPU 검사 (모델 없이 몇 초). 가짜 encoder · predictor 로:
  1. 설정 없음 → 원래 ar_losses 와 손실 값이 같다 (비트 단위).
  2. input=raw, target=smooth → predictor 입력은 원래 판, 손실 정답은 평활 판 (정답만 바뀜을 확인).
  3. occ 팔 → 입력의 가린 칸만 0, 정답은 그대로.
  4. mnf → k = D 이면 원래 토큰을 (LN 까지) 되돌린다.
  python paper_lab/method/test_patch_cpu.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "paper_lab"))
from app.vjepa_frozen import ar as AR  # noqa: E402
from method import patch as P  # noqa: E402

torch.manual_seed(0)
S, D, C, K = 16, 8, 3, 2
NB = C + K
enc_lin = torch.nn.Linear(3 * 2 * 4 * 4, S * D)


def enc(x):                                                     # (B·nb, 3, 2, 4, 4) → (B·nb, S, D)
    return enc_lin(x.flatten(1)).view(x.size(0), S, D)


class Pred(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Linear(D, D)

    def forward(self, h, ar):
        R = ar["rollout_steps"]; C_ = ar["n_ctx_blocks"]
        p_tf = self.w(h[:, :-S]); p_ar = self.w(h[:, (C_ - 1) * S:(C_ - 1 + R) * S])
        return p_tf, p_ar


pred = Pred(); clips = torch.randn(2, 3, 2 * NB, 4, 4)
orig_enc, orig_loss = P._ORIG_ENC, P._ORIG_LOSS
h0 = orig_enc(enc, clips, 2); l0 = orig_loss(pred, h0, S, C, K, 1.0, 8)[0]

P.install({})
h = AR.encode_blocks(enc, clips, 2); l1 = AR.ar_losses(pred, h, S, C, K, 1.0, 8)[0]
assert torch.equal(h, h0) and torch.equal(l0, l1), "1. 설정 없음인데 다르다"
print("1 ok  설정 없음 = 원래 (손실", float(l1), ")")

P.install({"input": "raw", "target": "smooth", "k_noise": 3})
h = AR.encode_blocks(enc, clips, 2)
assert torch.equal(h, h0), "2. 입력이 원래 판이 아니다"
assert not torch.equal(P.HOLD["h_tgt"], h0), "2. 정답이 평활되지 않았다"
l2 = AR.ar_losses(pred, h, S, C, K, 1.0, 8)[0]
p_tf, p_ar = pred(h0, {"rollout_steps": 2, "n_ctx_blocks": C})
ref = AR.l1(p_tf, P.HOLD["h_tgt"][:, S:], 1.0) + AR.l1(p_ar, P.HOLD["h_tgt"][:, C * S:(C + 2) * S], 1.0)
assert torch.allclose(l2, ref), "2. 손실이 (원래 입력, 평활 정답) 이 아니다"
print("2 ok  input=raw · target=smooth (손실", float(l2), "≠", float(l0), ")")

P.install({"occ": {"p": 1.0, "area": [0.2, 0.3]}, "seed": 1})
h = AR.encode_blocks(enc, clips, 2); hz = P._occlude(h, S, C)
z = (hz == 0).all(-1).view(2, NB, S)
assert z.any() and not (h == 0).all(-1).any(), "3. 가림이 없다"
first = [int(z[b].any(-1).nonzero()[0]) for b in range(2)]
assert all(1 <= f <= C - 1 for f in first) and all(z[b, f:].any(-1).all() for b, f in zip(range(2), first)), "3. 가림이 t0 부터 끝까지가 아니다"
assert torch.equal(P.HOLD["h_tgt"], h0), "3. 정답이 바뀌었다"
print("3 ok  occ: 가린 블록", first, "부터 끝까지, 정답 그대로")

with tempfile.TemporaryDirectory() as d:
    hh = h0.detach().reshape(-1, D).double().numpy(); dd = (hh + 0.1 * np.random.randn(*hh.shape)) - hh
    np.savez(f"{d}/c.npz", SN=2 * dd.T @ dd, nN=len(dd), SS=hh.T @ hh, mS=hh.sum(0), nS=len(hh))
    P._MNF.clear(); P.install({"input": "mnf", "target": "mnf", "mnf": {"calib": f"{d}/c.npz", "k": D}})
    h = AR.encode_blocks(enc, clips, 2)
    assert torch.allclose(h, torch.nn.functional.layer_norm(h0.detach(), (D,)), atol=1e-3), "4. k = D 인데 원래로 안 돌아간다"
print("4 ok  mnf (k = D) = 원래 + LN")
P.install({"cw": {"ref": 2, "gamma": 1.0}})
h = AR.encode_blocks(enc, clips, 2); lc = AR.ar_losses(pred, h, S, C, K, 1.0, 8)[0]
P.install({"cw": {"ref": 2, "gamma": 0.0}})                       # gamma 0 → 가중 1 → 원래 손실과 같아야
lc0 = AR.ar_losses(pred, AR.encode_blocks(enc, clips, 2), S, C, K, 1.0, 8)[0]
assert torch.allclose(lc0, l0, atol=1e-6), "5. gamma 0 인데 원래 손실과 다르다"
print("5 ok  운반 가중 손실 (gamma 1:", float(lc), ", gamma 0 = 원래", float(lc0), ")")
P.install({"occ_px": {"p": 1.0, "area": [0.2, 0.3]}, "seed": 3})
h = AR.encode_blocks(enc, clips, 2)
assert torch.equal(P.HOLD["h_tgt"], h0) and not torch.equal(h, h0), "6. 픽셀 가림: 입력만 바뀌어야"
xo = P._occlude_px(clips, 2); dif = (xo != clips).any(1)                 # (B, T, H, W)
first = [int(dif[b].flatten(1).any(1).nonzero()[0]) for b in range(2)]
assert all(f % 2 == 0 and f >= 2 for f in first) and all(dif[b, f:].flatten(1).any(1).all() for b, f in zip(range(2), first)), "6. 가림막이 블록 경계부터 끝까지가 아니다"
lp = AR.ar_losses(pred, h, S, C, K, 1.0, 8)[0]
print("6 ok  픽셀 가림막: 입력만 바뀜, 시작 프레임", first, "손실", float(lp))
P.install_eval({"occ_px": {"p": 1.0}, "cw": {"ref": 2}, "input": "raw"})
assert "occ_px" not in P.CFG and "cw" not in P.CFG, "7. 평가 설치에 학습 전용 성분이 남았다"
h = AR.encode_blocks(enc, clips, 2); assert torch.equal(h, h0), "7. 평가 입력이 원래와 다르다"
sc = P.split_cfg({"occ_px": {"p": 1.0}, "cw": {}, "input": "mnf", "mnf": {"k": 64}})
assert set(sc["infer"]) == {"input", "mnf"} and set(sc["train_only"]) == {"occ_px", "cw"}, "7. 나누기 틀림"
print("7 ok  평가 설치는 입력 가림막 · 운반 가중을 뺀다:", sc["infer"])
P.install({"occ_px": {"p": 1.0}})
P._EVAL["on"] = True; hv = AR.encode_blocks(enc, clips, 2); P._EVAL["on"] = False
assert torch.equal(hv, h0), "8. in-loop 검증 중인데 가림막이 걸렸다"
assert AR.run_val_ar is P._run_val_ar, "8. 검증 함수가 감싸지지 않았다"
print("8 ok  in-loop 검증에는 입력 가림막이 안 걸린다")
P.install({"cw_ctx": {"ref": 2, "gamma": 1.0}})
h = AR.encode_blocks(enc, clips, 2); lx = AR.ar_losses(pred, h, S, C, K, 1.0, 8)[0]
w1, w2 = P._cw_ctx_weights(h, S, C, 2)
tgt2 = P.HOLD["h_tgt"].clone(); tgt2[:, S:] += 5.0                              # 정답을 바꿔도 문맥 판 가중은 그대로여야
P.HOLD["h_tgt"] = tgt2; w1b, w2b = P._cw_ctx_weights(h, S, C, 2)
assert torch.equal(w1, w1b) and torch.equal(w2, w2b), "9. 문맥 판 가중이 정답에 의존한다"
assert w1.shape == (2, (NB - 1) * S) and w2.shape == (2, 2 * S), "9. 가중 shape"
P.install({"cw_ctx": {"ref": 2, "gamma": 0.0}})
l0c = AR.ar_losses(pred, AR.encode_blocks(enc, clips, 2), S, C, K, 1.0, 8)[0]
assert torch.allclose(l0c, l0, atol=1e-6), "9. gamma 0 인데 원래 손실과 다르다"
print("9 ok  문맥 판 운반 가중: 정답과 독립, gamma 0 = 원래 (손실", float(lx), ")")
torch.manual_seed(5); cl = torch.zeros(1, 3, 2 * NB, 32, 32); cl[0, :, 3, 16:24, 0:8] = 3.0      # 프레임 3 에만 움직인 칸 (오른쪽 아래 칸 묶음)
P.install({"occ_px": {"p": 1.0, "area": [0.05, 0.06], "where": "motion", "topk": 1}, "seed": 2})
xo = P._occlude_px(cl, 2); d = (xo != cl).any(1)[0]                      # (T, H, W)
f = int(d.flatten(1).any(1).nonzero()[0]); ys, xs = d[f].nonzero(as_tuple=True)
print("10 info  가림막 시작 프레임", f, "· 덮은 영역 y", int(ys.min()), "–", int(ys.max()), "x", int(xs.min()), "–", int(xs.max()))
if f == 4: assert ys.min() <= 23 and ys.max() >= 16 and xs.min() <= 7, "10. 움직인 칸을 안 덮었다"
print("10 ok  움직이는 칸 가림막")
with tempfile.TemporaryDirectory() as d:
    hh = h0.detach().reshape(-1, D).double().numpy(); dd = 0.1 * np.random.randn(*hh.shape)
    np.savez(f"{d}/c.npz", SN=2 * dd.T @ dd, nN=len(dd), SS=hh.T @ hh, mS=hh.sum(0), nS=len(hh))
    P._MNF.clear(); P.install({"cw_ctx": {"ref": 2, "gamma": 3.0, "space": "mnf"}, "mnf": {"calib": f"{d}/c.npz", "k": 4}})
    h = AR.encode_blocks(enc, clips, 2)
    assert torch.equal(h, h0) and torch.equal(P.HOLD["h_tgt"], h0), "11. 가중 공간만 바꿨는데 입력 · 정답이 바뀌었다"
    lm = AR.ar_losses(pred, h, S, C, K, 1.0, 8)[0]
print("11 ok  가중 지도만 떨림 억제 공간 (입력 · 정답 그대로), 손실", float(lm))
print("ALL OK")
