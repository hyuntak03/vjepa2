"""M2-t 운반 가지 (paper_lab 진행자 2, 2026-10-22, PLAN §11-4). 기존 학습 코드는 고치지 않고 런타임에 끼운다. ⚠️ 아직 띄우지 않음 (D10 · 파일럿 승인 전).

무엇을: 블록 AR predictor 의 출력 토큰 p (목표 블록 t 의 칸 q) 마다 작은 head 가
  속도 v̂(q) (블록당 칸 단위, 2) · 문 g(q) (1) · 출발 블록 가중 w_l(q) (L) 을 낸다.
  운반 토큰 c(q) = Σ_l softmax(w)_l · h_{s_l}[q − v̂(q) · (t − s_l)]    (h = 블록별 LN 표적 토큰, bilinear 로 읽음)
  출력 p'(q) = σ(g) · c(q) + (1 − σ(g)) · p(q)
  출발 블록 s_l: 정답 입력 학습 (TF) 은 목표 블록 바로 앞 L 개 입력 블록, rollout 은 문맥 끝 L 개 블록 (rollout 은 미래 정답을 못 본다).
  → 가려진 물체는 '마지막으로 보인 블록' (앞 블록) 에서 더 멀리 옮겨 가져올 수 있다 (출발 블록 가중이 고른다).
변위 학습 신호 (선택): 사용자 궤적 라벨 (흐름 기반, 관측만; 앵커 = 블록 b 의 마지막 장, 지평선 2 장 = 1 블록 앞) 로
  v̂(q) ≈ d_{t−1}(q, 1 블록)   (도착 칸 q 로 온 내용의 속도를 같은 자리 q 에서 출발하는 내용의 앞으로 가는 변위로 근사 — 흐름이 매끄러운 곳에서만 맞다.
  읽는 자리 = q − v̂ · Δt 이므로 v̂ 는 앞으로 가는 속도다)
학습 · 평가 같은 꼴: 운반은 predictor 의 rollout 출력 뒤에 붙는다 (되먹임 안쪽은 바꾸지 않는다) → 평가의 rollout 에도 같은 함수를 건다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CarryHead(nn.Module):
    def __init__(self, dim: int = 1280, n_src: int = 4, gate_bias: float = -2.0):
        super().__init__()
        self.n_src = int(n_src)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 3 + self.n_src)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():                                   # 시작: 속도 0 · 문 거의 닫힘 (σ(−2) = 0.12) · 출발 블록 균등
            self.out.bias.zero_()
            self.out.bias[2] = gate_bias

    def forward(self, p: torch.Tensor):
        """p (B, T, G, G, D) → v (B, T, G, G, 2) · g (B, T, G, G, 1) · w (B, T, G, G, L)."""
        y = self.out(self.norm(p.float()))
        return y[..., :2], y[..., 2:3], y[..., 3:]


def sample_shift(h_src: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """h_src (B, G, G, D), disp (B, G, G, 2) 칸 단위 (x, y), 도착 칸 q 에서 출발점 q + disp 를 bilinear 로 읽는다 → (B, G, G, D)."""
    B, G, _, D = h_src.shape
    yy, xx = torch.meshgrid(torch.arange(G, device=h_src.device), torch.arange(G, device=h_src.device), indexing="ij")
    cx = (xx.float() + 0.5).unsqueeze(0) + disp[..., 0]
    cy = (yy.float() + 0.5).unsqueeze(0) + disp[..., 1]
    grid = torch.stack([cx / G * 2 - 1, cy / G * 2 - 1], -1)
    s = F.grid_sample(h_src.permute(0, 3, 1, 2).float(), grid.float(), mode="bilinear", padding_mode="border", align_corners=False)
    return s.permute(0, 2, 3, 1)


def apply_carry(head: CarryHead, p: torch.Tensor, h: torch.Tensor, targets, S: int, ctx_blocks=None):
    """p (B, T*S, D) predictor 출력 (목표 블록 targets[i] 의 예측), h (B, nb*S, D) 입력 블록 토큰 (LN).
    targets: 목표 블록 번호 목록 (길이 T). ctx_blocks = None 이면 TF (목표 t 의 출발 = t−1 … t−L), 정수 C 면 rollout (출발 = C−1 … C−L).
    → (p' 같은 shape, aux = {v, g, w})."""
    B, N, D = p.shape
    T = N // S; G = int(round(S ** 0.5)); L = head.n_src
    nb = h.size(1) // S
    P = p.view(B, T, G, G, D)
    Hb = h.view(B, nb, G, G, D)
    v, g, w = head(P)
    w = torch.softmax(w, -1)
    outs = []
    for i, t in enumerate(targets):
        last = (t - 1) if ctx_blocks is None else (ctx_blocks - 1)
        acc = 0.0
        for l in range(L):
            s = max(last - l, 0)
            dt = float(t - s)                                    # 블록 수 (출발 → 목표)
            acc = acc + w[:, i, ..., l:l + 1] * sample_shift(Hb[:, s], -v[:, i] * dt)
        gi = torch.sigmoid(g[:, i])
        outs.append(gi * acc + (1 - gi) * P[:, i].float())
    out = torch.stack(outs, 1).to(p.dtype).view(B, N, D)
    return out, {"v": v, "g": g, "w": w}


def velocity_loss(v_tf: torch.Tensor, flow: torch.Tensor, valid: torch.Tensor, horizons, h_one_block: int = 2):
    """정답 입력 학습 출력 (목표 블록 1..T) 의 속도 v_tf (B, T, G, G, 2) 를 궤적 라벨로: v(q) ≈ d_{t−1}(q, 1 블록) (앞으로 가는 속도).
    flow (B, nb, H, 2, g, g) · valid (B, nb, H, g, g) — 사용자 라벨 꼴 (패치 = 칸 단위, 앵커 = 블록의 마지막 장).
    → 유효 칸의 L1 평균 (스칼라)."""
    hi = [int(x) for x in horizons].index(int(h_one_block))
    T = v_tf.size(1)
    d = flow[:, 0:T, hi]                                          # 출발 블록 0..T−1 (목표 1..T) 의 1 블록 앞 변위 (B, T, 2, g, g)
    ok = valid[:, 0:T, hi].float()                                # (B, T, g, g)
    tgt = d.permute(0, 1, 3, 4, 2)                               # (B, T, g, g, 2) — 읽는 자리 q − v · Δt 와 맞는 부호
    err = (v_tf.float() - tgt.float()).abs().sum(-1)              # (B, T, g, g)
    return (err * ok).sum() / ok.sum().clamp_min(1.0)


# ---------------------------------------------------------------- 학습 · 평가에 끼우기 (아직 실행 경로에 연결 안 함 — D10 · 파일럿 승인 뒤)
HOLD = {"head": None, "v_tf": None, "w_vel": 0.0, "horizons": (1, 2, 4, 8)}


def install_train(head: CarryHead, w_vel: float = 1.0, horizons=(1, 2, 4, 8)):
    """AR.ar_losses 를 감싸 p_tf · p_ar 에 운반을 건 뒤 L1 을 계산하고, 사용자 궤적 손실 함수 (flow_head.ar_flow_loss) 를 감싸
    같은 배치의 궤적 라벨로 속도 손실을 더한다 (그쪽 함수가 라벨을 받는 유일한 자리라서). 둘 다 런타임 교체 — 기존 파일은 그대로."""
    from app.vjepa_frozen import ar as AR
    from app.vjepa_frozen import flow_head as FHM
    HOLD.update(head=head, w_vel=float(w_vel), horizons=tuple(horizons))
    orig_loss, orig_flow = AR.ar_losses, FHM.ar_flow_loss

    def ar_losses(predictor, h, S, C, K, loss_exp, rollout_steps, actions=None, out=None):
        B, N = h.shape[:2]; nb = N // S
        R = max(1, min(int(K), int(rollout_steps)))
        idx = AR.block_indices(B, nb, S, h.device)
        p_tf, p_ar = predictor(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": R, "actions": actions})
        if out is not None:
            out["p_tf"], out["p_ar"] = p_tf, p_ar                 # 사용자 보조 head 는 운반 전 토큰을 읽는다 (그쪽 정의 그대로)
        q_tf, a_tf = apply_carry(head, p_tf, h, list(range(1, nb)), S)
        q_ar, _ = apply_carry(head, p_ar, h, list(range(C, C + R)), S, ctx_blocks=C)
        HOLD["v_tf"] = a_tf["v"]
        jloss = AR.l1(q_tf, h[:, S:], loss_exp)
        sloss = AR.l1(q_ar, h[:, C * S:(C + R) * S], loss_exp)
        return jloss + sloss, jloss, sloss, R

    def ar_flow_loss(fh, p_tf, p_ar, flow, valid, S, C, R, split, horizons=(1, 2, 4, 8), supervise=(2, 4, 8), **kw):
        lf, log = orig_flow(fh, p_tf, p_ar, flow, valid, S, C, R, split, horizons=horizons, supervise=supervise, **kw)
        if HOLD["v_tf"] is not None and HOLD["w_vel"] > 0:
            lv = velocity_loss(HOLD["v_tf"], flow, valid, horizons)
            log = dict(log, carry_vel=float(lv.detach()))
            lf = lf + HOLD["w_vel"] * lv
        return lf, log

    AR.ar_losses = ar_losses
    FHM.ar_flow_loss = ar_flow_loss
    return orig_loss, orig_flow


def wrap_rollout(predictor, head: CarryHead):
    """평가: predictor.rollout(h_ctx, idx, K) 출력에 같은 운반을 건다 (문맥 블록 = 입력 블록 수)."""
    orig = predictor.rollout

    def rollout(h_ctx, idx, K, *a, **k):
        out = orig(h_ctx, idx, K, *a, **k)
        S = 256
        C = h_ctx.size(1) // S
        q, _ = apply_carry(head, out, h_ctx, list(range(C, C + K)), S, ctx_blocks=C)
        return q
    predictor.rollout = rollout
    return orig


# ---------------------------------------------------------------- predictor 에 붙이기 (optimizer · 체크포인트) · 평가에서 불러 걸기
def install_attach(n_src: int = 4, gate_bias: float = -2.0):
    """train.py 가 부르는 U.build_predictor 를 감싸 만든 predictor 에 운반 head 를 하위 모듈 'carry_head' 로 붙인다
    → optimizer (predictor 매개변수) · 체크포인트 (predictor state_dict) 에 자동으로 들어간다. 기존 파일은 그대로."""
    from app.vjepa_frozen import utils as U
    orig = U.build_predictor

    def build_predictor(MD, embed_dim, state, device, *a, **k):
        pred = orig(MD, embed_dim, state, device, *a, **k)
        mod = pred.module if hasattr(pred, "module") else pred
        if not hasattr(mod, "carry_head"):
            head = CarryHead(int(embed_dim), n_src=n_src, gate_bias=gate_bias).to(device)
            if isinstance(state, dict):                          # 이어 학습: 체크포인트에 운반 head 가 있으면 불러온다
                sd = state.get("predictor", state)
                hs = {kk[len("carry_head."):]: v for kk, v in sd.items() if kk.startswith("carry_head.")}
                if hs:
                    head.load_state_dict(hs)
            mod.add_module("carry_head", head)
        HOLD["head"] = mod.carry_head
        return pred
    U.build_predictor = build_predictor
    return orig


def wrap_for_eval(predictor, ckpt_path: str, n_src: int = 4):
    """평가: 체크포인트의 'carry_head.*' 를 불러 predictor.rollout 에 같은 운반을 건다. head 가 없으면 아무것도 안 한다 (False)."""
    import torch as _t
    sd = _t.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("predictor", sd)
    hs = {k[len("carry_head."):]: v for k, v in sd.items() if k.startswith("carry_head.")}
    if not hs:
        return False
    dim = hs["norm.weight"].numel()
    head = CarryHead(dim, n_src=hs["out.weight"].shape[0] - 3).to(next(predictor.parameters()).device)
    head.load_state_dict(hs)
    head.eval()
    wrap_rollout(predictor, head)
    return True
