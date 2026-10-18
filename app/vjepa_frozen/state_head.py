"""상태 supervision head — phys2d_sup 라벨로 AR predictor 출력에 보조 손실 (2026-10-19, 사용자 결정).

predictor 출력 p (TF · rollout, encoder 공간 1280) 를 블록마다 (S=256 토큰, D) 로 보고
**정체 쿼리 48 개** (모양 6 × 색 8, 쿼리 k = 정체 k 전용) 로 attention pooling → (T, 48, D) → 공유 head 3 개:
  존재  1 logit          BCE  — 그 정체를 그 블록까지 본 적 있나 (seen). 48 쿼리 전부
  위치  (x, y) / 256      L1   — 있고 · 본 적 있고 · 화면 안인 칸 (가려져 있어도 건다 = 영속)
  속도  (cos, sin, ‖v‖/8) L1   — 같은 칸. ‖v‖ < 0.5 px/장이면 cos · sin 은 끈다 (방향이 정의되지 않음)
한 영상 안 정체는 모두 다르게 만들었으므로 (gen_phys2d_sup.py) 쿼리 ↔ 물체 짝이 정해져 있다 — 매칭이 없다.

head 는 predictor 의 state_dict 밖에 둔다 (채점기 · export 가 predictor 를 strict 로 읽는다). 체크포인트에는 `state_head` 로 따로 저장.
라벨 시간축: 블록 b 의 라벨 = 그 블록 마지막 장 상태 (phys2d_sup_labels.window_labels reduce=last).
TF 출력 위치 b 는 블록 b+1 을 예측하므로 라벨 블록 b+1 과, rollout 걸음 r 은 블록 C+r 과 맞춘다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ID = 48
SPEED_SCALE = 8.0       # px/장 → 목표 ‖v‖ / 8 (중앙 ~0.4)
DIR_MIN_SPEED = 0.5     # px/장


class StateHead(nn.Module):
    def __init__(self, dim: int = 1280, n_query: int = N_ID, num_heads: int = 16, hidden: int = 512):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.post = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.exist = nn.Linear(hidden, 1)
        self.pos = nn.Linear(hidden, 2)
        self.vel = nn.Linear(hidden, 3)

    def forward(self, p: torch.Tensor, S: int):
        """p (B, T*S, D) → dict exist (B,T,48) · pos (B,T,48,2) · vel (B,T,48,3) · attn (B,T,48,S) 평균 head attention"""
        B, N, D = p.shape
        T = N // S
        x = self.norm(p.reshape(B * T, S, D).float().to(self.query.dtype))
        q = self.query.unsqueeze(0).expand(B * T, -1, -1)
        o, w = self.attn(q, x, x, need_weights=True, average_attn_weights=True)    # (B*T, 48, D), (B*T, 48, S)
        z = self.mlp(self.post(o))
        sh = (B, T, q.size(1))
        return {"exist": self.exist(z).reshape(sh), "pos": self.pos(z).reshape(*sh, 2), "vel": self.vel(z).reshape(*sh, 3),
                "attn": w.reshape(*sh, S)}


# ----------------------------------------------------------------------------- 라벨 (데이터 쪽, numpy)
def pack_labels(y: dict, flip: bool = False) -> dict:
    """window_labels (tubelet 줄인 것, 물체 축 n) → 정체 48 칸 배열 (float32 / bool, torch 로 collate).
    flip: 영상을 좌우 반전했으면 라벨도 (x → 1 − x, vx → −vx)."""
    T = y["present"].shape[0]
    ident = y["identity"].astype(np.int64)
    ok = ident >= 0
    out = {"seen": np.zeros((T, N_ID), bool), "m_state": np.zeros((T, N_ID), bool), "m_dir": np.zeros((T, N_ID), bool),
           "hidden": np.zeros((T, N_ID), bool), "pos": np.zeros((T, N_ID, 2), np.float32), "vel": np.zeros((T, N_ID, 3), np.float32)}
    pos = np.nan_to_num(y["pos"][:, ok] / 256.0)
    v = np.nan_to_num(y["vel"][:, ok])
    if flip:
        pos[..., 0] = 1.0 - pos[..., 0]
        v[..., 0] = -v[..., 0]
    sp = np.linalg.norm(v, axis=-1)
    cs = v / np.maximum(sp, 1e-6)[..., None]
    k = ident[ok]
    out["seen"][:, k] = y["mask_identity"][:, ok]
    out["m_state"][:, k] = y["mask_state"][:, ok]
    out["m_dir"][:, k] = y["mask_state"][:, ok] & (sp > DIR_MIN_SPEED)
    out["hidden"][:, k] = y["hidden"][:, ok] & y["mask_state"][:, ok]
    out["pos"][:, k] = pos
    out["vel"][:, k] = np.concatenate([cs, (sp / SPEED_SCALE)[..., None]], -1)
    return out


# ----------------------------------------------------------------------------- 손실
def _masked_l1(pred, tgt, m):
    """pred · tgt (B,T,48,c), m (B,T,48) → 성분 평균 L1 의 마스크 평균"""
    m = m.float()
    d = (pred.float() - tgt.float()).abs().mean(-1)
    n = m.sum()
    return (d * m).sum() / n.clamp_min(1.0), n


def state_losses(out: dict, lab: dict, w: dict) -> tuple:
    """out = StateHead 출력, lab = 같은 블록에 맞춘 라벨 (B,T,48,...) torch. return (loss, 로그 dict)."""
    l_ex = F.binary_cross_entropy_with_logits(out["exist"].float(), lab["seen"].float())
    l_pos, n_pos = _masked_l1(out["pos"], lab["pos"], lab["m_state"])
    l_dir, _ = _masked_l1(out["vel"][..., :2], lab["vel"][..., :2], lab["m_dir"])
    l_spd, _ = _masked_l1(out["vel"][..., 2:], lab["vel"][..., 2:], lab["m_state"])
    loss = w.get("exist", 1.0) * l_ex + w.get("pos", 1.0) * l_pos + w.get("vel", 1.0) * (l_dir + l_spd)
    with torch.no_grad():
        err = ((out["pos"].float() - lab["pos"].float()).norm(dim=-1) * 256.0)          # px
        mh = lab["hidden"]
        mv = lab["m_state"] & ~mh
        log = {"ex": float(l_ex), "pos": float(l_pos), "dir": float(l_dir), "spd": float(l_spd),
               "pos_px_vis": float((err * mv).sum() / mv.sum().clamp_min(1)),
               "pos_px_hid": float((err * mh).sum() / mh.sum().clamp_min(1)) if mh.any() else float("nan"),
               "ex_acc": float(((out["exist"] > 0) == lab["seen"]).float().mean()), "n_state": float(n_pos)}
    return loss, log


def labels_to_device(lab: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in lab.items()}


def slice_blocks(lab: dict, a: int, b: int) -> dict:
    return {k: v[:, a:b] for k, v in lab.items()}


def ar_state_loss(head, p_tf, p_ar, lab: dict, S: int, C: int, R: int, w: dict, split: dict):
    """TF 출력 (블록 1..nb-1) 과 rollout 출력 (블록 C..C+R-1) 둘 다에 같은 head. split = {tf: 가중, ar: 가중}.
    return (loss, 로그)."""
    o_tf = head(p_tf, S)
    l_tf, g_tf = state_losses(o_tf, slice_blocks(lab, 1, 1 + p_tf.size(1) // S), w)
    o_ar = head(p_ar, S)
    l_ar, g_ar = state_losses(o_ar, slice_blocks(lab, C, C + R), w)
    loss = split.get("tf", 1.0) * l_tf + split.get("ar", 1.0) * l_ar
    log = {f"tf_{k}": v for k, v in g_tf.items()}
    log.update({f"ar_{k}": v for k, v in g_ar.items()})
    return loss, log
