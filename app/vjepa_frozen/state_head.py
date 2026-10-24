"""상태 supervision head — phys2d_sup 라벨로 AR predictor 출력에 보조 손실 (2026-10-19, 사용자 결정).
(main 425d67d 에서 그대로 가져왔다 — vast 는 var_window boundary 소스가 라벨을 함께 싣는다, 2026-10-19)

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


class LightStateHead(nn.Module):
    """약한 head (2026-10-19, 사용자 결정, `state_head.type: light`) — StateHead 와 같은 입출력, 투영 · MLP 없음.
    x_i = LN(토큰 i) (affine 없음), a_ki = softmax_i(q_k · x_i / √D), o_k = Σ_i a_ki x_i, 출력 = Linear(D → 6)(o_k) (48 쿼리 공용).
    학습 파라미터 = 쿼리 48 × D + Linear (hidden 0 이면 D·6+6). 큰 투영이 대신 정리해 줄 수 없으므로
    predictor 출력 토큰 자체가 정체 쿼리와 닮고 · 상태를 선형으로 담아야 맞는다. hidden > 0 이면 D → hidden → 6 작은 MLP."""

    def __init__(self, dim: int = 1280, n_query: int = N_ID, hidden: int = 0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, dim) * 0.02)
        self.scale = dim ** -0.5
        self.out = nn.Linear(dim, 6) if hidden <= 0 else nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 6))

    def forward(self, p: torch.Tensor, S: int):
        B, N, D = p.shape
        T = N // S
        x = F.layer_norm(p.reshape(B * T, S, D).float().to(self.query.dtype), (D,))
        w = torch.softmax(torch.einsum("kd,nsd->nks", self.query, x) * self.scale, dim=-1)     # (B*T, 48, S)
        o = torch.einsum("nks,nsd->nkd", w, x)                                                  # (B*T, 48, D)
        y = self.out(o)
        sh = (B, T, self.query.size(0))
        return {"exist": y[..., 0].reshape(sh), "pos": y[..., 1:3].reshape(*sh, 2), "vel": y[..., 3:6].reshape(*sh, 3),
                "attn": w.reshape(*sh, S)}


class Query3StateHead(nn.Module):
    """정체 쿼리 48 + 공용 linear head 3 개 (2026-10-19, 사용자 설계, `state_head.type: query3`).
    x_i = LN(토큰 i) (affine 없음), a_ki = softmax_i(τ · q_k · x_i / √D), z_k = Σ_i a_ki x_i,
    위치 Linear(D → 2) · 속도 Linear(D → 3) · 물체 Linear(D → 49) (48 정체 + 없음) — 48 쿼리 공용.
    물체 head 는 쿼리 번호를 모른 채 z_k 내용만으로 정체를 맞힌다 (모양 · 색 압력). τ 학습 (log 로), q init std = q_std (기본 1.0 —
    0.02 면 τ 가 혼자 50 배를 감당해야 해 attention 이 균등에서 못 벗어난다).
    출력 exist = max(정체 로짓) − 없음 로짓 (기존 로그 · 평가 호환), obj (B,T,48,49) 는 손실이 쓴다."""

    def __init__(self, dim: int = 1280, n_query: int = N_ID, q_std: float = 1.0, tau: float = 1.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, dim) * q_std)
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau))))
        self.scale = dim ** -0.5
        self.pos = nn.Linear(dim, 2)
        self.vel = nn.Linear(dim, 3)
        self.obj = nn.Linear(dim, n_query + 1)

    def forward(self, p: torch.Tensor, S: int):
        B, N, D = p.shape
        T = N // S
        x = F.layer_norm(p.reshape(B * T, S, D).float().to(self.query.dtype), (D,))
        lg = torch.einsum("kd,nsd->nks", self.query, x) * (self.scale * self.log_tau.exp())
        w = torch.softmax(lg, dim=-1)                                                            # (B*T, 48, S)
        z = torch.einsum("nks,nsd->nkd", w, x)
        sh = (B, T, self.query.size(0))
        obj = self.obj(z).reshape(*sh, -1)
        return {"obj": obj, "exist": obj[..., :-1].max(-1).values - obj[..., -1],
                "pos": self.pos(z).reshape(*sh, 2), "vel": self.vel(z).reshape(*sh, 3), "attn": w.reshape(*sh, S)}


class SingleStateHead(nn.Module):
    """물체 1 개 장면 (phys2d_single) 용 decoder (2026-10-19, 사용자 설계, `state_head.type: single`).
    블록마다 x_i = LN(토큰 i), a_i = softmax_i(τ · q · x_i / √D) (query 1 개, 투영 없음), z = Σ_i a_i x_i → (T, D),
    공용 linear head 3 개: 물체 Linear(D → 49) (48 정체 + 없음) · 위치 Linear(D → 2) · 속도 Linear(D → 3).
    출력은 (B, T, 1, …) — 손실은 single_losses (48 칸 라벨에서 그 영상의 정체 하나를 뽑아 쓴다)."""
    single = True

    def __init__(self, dim: int = 1280, n_cls: int = N_ID + 1, q_std: float = 1.0, tau: float = 1.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, dim) * q_std)
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau))))
        self.scale = dim ** -0.5
        self.obj = nn.Linear(dim, n_cls)
        self.pos = nn.Linear(dim, 2)
        self.vel = nn.Linear(dim, 3)

    def forward(self, p: torch.Tensor, S: int):
        B, N, D = p.shape
        T = N // S
        x = F.layer_norm(p.reshape(B * T, S, D).float().to(self.query.dtype), (D,))
        w = torch.softmax(torch.einsum("kd,nsd->nks", self.query, x) * (self.scale * self.log_tau.exp()), dim=-1)   # (B*T, 1, S)
        z = torch.einsum("nks,nsd->nkd", w, x)                                                                      # (B*T, 1, D)
        sh = (B, T, 1)
        return {"obj": self.obj(z).reshape(*sh, -1), "pos": self.pos(z).reshape(*sh, 2), "vel": self.vel(z).reshape(*sh, 3),
                "attn": w.reshape(*sh, S)}


class MeanStateHead(SingleStateHead):
    """SingleStateHead 와 같은 출력 · 같은 손실 (single_losses), 모으기만 **평균 풀링** (2026-10-23, 사용자 결정).
    블록마다 z = mean_i LN(토큰 i) → 공용 linear 3 개 (물체 49 · 위치 2 · 속도 3). query · τ 없음 (attention 없음).
    뜻: 상태가 물체 칸 몇 개가 아니라 블록 토큰 평균 (화면 전체) 에서 선형으로 읽히도록 predictor 출력에 요구한다.
    (측정: 문맥 블록 토큰 평균 → 속도 R² 0.59, 굴린 예측 평균 → 0.00 — velocity_transfer.py)"""

    def __init__(self, dim: int = 1280, n_cls: int = N_ID + 1):
        nn.Module.__init__(self)
        self.obj = nn.Linear(dim, n_cls)
        self.pos = nn.Linear(dim, 2)
        self.vel = nn.Linear(dim, 3)

    def forward(self, p: torch.Tensor, S: int):
        B, N, D = p.shape
        T = N // S
        x = F.layer_norm(p.reshape(B * T, S, D).float().to(self.obj.weight.dtype), (D,))
        z = x.mean(1, keepdim=True)                                                                                 # (B*T, 1, D)
        sh = (B, T, 1)
        return {"obj": self.obj(z).reshape(*sh, -1), "pos": self.pos(z).reshape(*sh, 2), "vel": self.vel(z).reshape(*sh, 3),
                "attn": torch.full((B, T, 1, S), 1.0 / S, device=p.device)}


def single_losses(out: dict, lab: dict, w: dict) -> tuple:
    """SingleStateHead 손실. lab 은 48 칸 라벨 (B,T,48,…) — 장면 물체가 하나라 그 정체 칸 k 를 영상마다 뽑는다.
    물체: CE 49 지선다, 정답 = k (보였으면) / 없음 (48, 아직 안 보였거나 화면 밖). w.exist_neg = 없음 정답 가중치.
    위치 · 속도: L1, 그 블록에 상태가 있으면 (m_state — 가려져도 포함). 방향은 거의 정지면 뺀다 (m_dir)."""
    obj = out["obj"][:, :, 0].float()                                                   # (B, T, 49)
    any_ = (lab["seen"] | lab["m_state"]).any(1)                                        # (B, 48)
    k = any_.float().argmax(-1)                                                         # (B,) 영상의 정체
    has = any_.any(-1)
    g = lambda t: t.gather(2, k.view(-1, 1, 1).expand(-1, t.size(1), 1)).squeeze(2) if t.dim() == 3 else \
        t.gather(2, k.view(-1, 1, 1, 1).expand(-1, t.size(1), 1, t.size(-1))).squeeze(2)
    seen, ms, md, hid = g(lab["seen"]) & has[:, None], g(lab["m_state"]) & has[:, None], g(lab["m_dir"]) & has[:, None], g(lab["hidden"]) & has[:, None]
    pos_t, vel_t = g(lab["pos"]), g(lab["vel"])                                         # (B, T, 2) · (B, T, 3)
    NONE = obj.size(-1) - 1
    tgt = torch.where(seen, k[:, None].expand_as(seen), torch.full_like(seen, NONE, dtype=torch.long))
    ce = F.cross_entropy(obj.reshape(-1, NONE + 1), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
    wt = torch.where(seen, 1.0, float(w.get("exist_neg", 1.0)))
    l_ex = (ce * wt).sum() / wt.sum()
    pos, vel = out["pos"][:, :, 0], out["vel"][:, :, 0]
    l_pos, n_pos = _masked_l1(pos, pos_t, ms)
    l_dir, _ = _masked_l1(vel[..., :2], vel_t[..., :2], md)
    l_spd, _ = _masked_l1(vel[..., 2:], vel_t[..., 2:], ms)
    loss = w.get("exist", 1.0) * l_ex + w.get("pos", 1.0) * l_pos + w.get("vel", 1.0) * (l_dir + l_spd)
    with torch.no_grad():
        err = (pos.float() - pos_t.float()).norm(dim=-1) * 256.0
        mv = ms & ~hid
        pred = obj.argmax(-1)
        log = {"ex": float(l_ex), "pos": float(l_pos), "dir": float(l_dir), "spd": float(l_spd),
               "pos_px_vis": float((err * mv).sum() / mv.sum().clamp_min(1)),
               "pos_px_hid": float((err * hid).sum() / hid.sum().clamp_min(1)) if hid.any() else float("nan"),
               "ex_acc": float(((pred != NONE) == seen).float().mean()), "n_state": float(n_pos),
               "ex_rec": float((pred != NONE)[seen].float().mean()) if seen.any() else float("nan"),
               "id_acc": float((pred == k[:, None])[seen].float().mean()) if seen.any() else float("nan"),
               "conc": float((out["attn"][:, :, 0].amax(-1) * out["attn"].size(-1))[seen].float().mean()) if seen.any() else float("nan")}
    return loss, log


def build_head(SH: dict, dim: int) -> nn.Module:
    """config `state_head` → head. type: attn (기본, StateHead) | light (LightStateHead) | query3 (Query3StateHead)."""
    t = str(SH.get("type", "attn"))
    if t == "attn":
        return StateHead(dim, num_heads=int(SH.get("num_heads", 16)), hidden=int(SH.get("hidden", 512)))
    if t == "light":
        return LightStateHead(dim, hidden=int(SH.get("hidden", 0)))
    if t == "single":
        return SingleStateHead(dim, q_std=float(SH.get("q_std", 1.0)), tau=float(SH.get("tau", 1.0)))
    if t == "mean":                 # 2026-10-23: 평균 풀링 (MeanStateHead)
        return MeanStateHead(dim)
    if t == "query3":
        return Query3StateHead(dim, q_std=float(SH.get("q_std", 1.0)), tau=float(SH.get("tau", 1.0)))
    raise ValueError(f"state_head.type 은 attn | light | query3 | single | mean: {t!r}")


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
    if "obj" in out:                        # 2026-10-19 query3: 쿼리 k 의 물체 head 49 지선다 — 정답 = k (보였으면) / 없음 (48)
        obj = out["obj"].float()
        K_ = obj.size(-2)
        tgt = torch.where(lab["seen"], torch.arange(K_, device=obj.device).expand_as(lab["seen"]), torch.full_like(lab["seen"], K_, dtype=torch.long))
        ce = F.cross_entropy(obj.reshape(-1, K_ + 1), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
        wt = torch.where(lab["seen"], 1.0, float(w.get("exist_neg", 1.0)))
        l_ex = (ce * wt).sum() / wt.sum()
    elif w.get("exist_neg", 1.0) != 1.0:    # 2026-10-19: "없음" 칸 가중치 (DETR 의 none 가중치). 기본 1 = 기존과 같다
        y = lab["seen"].float()
        wt = y + (1.0 - y) * float(w["exist_neg"])
        l_ex = (F.binary_cross_entropy_with_logits(out["exist"].float(), y, reduction="none") * wt).sum() / wt.sum()
    else:
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
               "ex_acc": float(((out["exist"] > 0) == lab["seen"]).float().mean()), "n_state": float(n_pos),
               "ex_rec": float((out["exist"] > 0)[lab["seen"]].float().mean()) if lab["seen"].any() else float("nan")}
        if "obj" in out and lab["seen"].any():   # 정체 정확도 (보인 정체 중 49 지선다 최고가 자기 정체) · 집중도 (w_max × S, 균등 = 1)
            sn = lab["seen"]
            log["id_acc"] = float((out["obj"].argmax(-1) == torch.arange(out["obj"].size(-2), device=sn.device).expand_as(sn))[sn].float().mean())
        if "attn" in out and lab["seen"].any():
            log["conc"] = float((out["attn"].amax(-1) * out["attn"].size(-1))[lab["seen"]].float().mean())
    return loss, log


def labels_to_device(lab: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in lab.items()}


def slice_blocks(lab: dict, a: int, b: int) -> dict:
    return {k: v[:, a:b] for k, v in lab.items()}


def visible_only(lab: dict) -> dict:
    """2026-10-24 (fit_on: gt): 그 블록에 **보이는** 물체만 남긴다. 정답 토큰 (블록 2 장만 encoder 에 넣은 것) 은 가려지면 물체 정보가 없다
    (측정: 정답 토큰 head 위치 오차 보임 20 px · 가림 40 px, gt_head_check.py) — 그래서 head 학습 · predictor 손실 둘 다 보이는 블록만.
    seen (지금까지 본 적 있나) 도 '지금 보이나' 로 바꾼다 — 블록 하나의 정답 토큰은 과거를 모른다."""
    v = lab["m_state"] & ~lab["hidden"]
    out = dict(lab)
    out["seen"], out["m_state"], out["m_dir"] = v, v, lab["m_dir"] & v
    return out


def ar_state_loss_gt(head, h, p_tf, p_ar, lab: dict, S: int, C: int, R: int, w: dict, split: dict):
    """2026-10-24 (사용자 결정, `state_head.fit_on: gt`): head 는 **정답 토큰** h (= predictor 입력 · 채점 타깃, LN(target_encoder(블록)),
    문맥 + 미래 전 블록) 에서만 학습하고, predictor 는 **같은 head (가중치 고정, gradient 끊음)** 로 자기 예측 (TF · rollout) 에 손실을 받는다.
    → predictor 가 정답 토큰과 같은 방식으로 상태를 담아야 손실이 내려간다 (head 가 예측 쪽 코드를 따로 배우는 우회를 막는다).
    보이는 블록만 (visible_only). return (loss, 로그) — loss = head 손실 (head 만 grad) + predictor 손실 (predictor 만 grad)."""
    fn = single_losses if getattr(head, "single", False) else state_losses
    lab = visible_only(lab)
    nb = h.size(1) // S
    l_gt, g_gt = fn(head(h.detach(), S), slice_blocks(lab, 0, nb), w)                       # head 만
    frozen = {n: q.detach() for n, q in head.named_parameters()}
    fc = lambda x: torch.func.functional_call(head, frozen, (x, S))                           # predictor 만
    l_tf, g_tf = fn(fc(p_tf), slice_blocks(lab, 1, 1 + p_tf.size(1) // S), w)
    l_ar, g_ar = fn(fc(p_ar), slice_blocks(lab, C, C + R), w)
    loss = l_gt + split.get("tf", 1.0) * l_tf + split.get("ar", 1.0) * l_ar
    log = {f"tf_{k}": v for k, v in g_tf.items()}
    log.update({f"ar_{k}": v for k, v in g_ar.items()})
    log.update({f"gt_{k}": v for k, v in g_gt.items()})
    return loss, log


def ar_state_loss(head, p_tf, p_ar, lab: dict, S: int, C: int, R: int, w: dict, split: dict):
    """TF 출력 (블록 1..nb-1) 과 rollout 출력 (블록 C..C+R-1) 둘 다에 같은 head. split = {tf: 가중, ar: 가중}.
    return (loss, 로그)."""
    fn = single_losses if getattr(head, "single", False) else state_losses
    o_tf = head(p_tf, S)
    l_tf, g_tf = fn(o_tf, slice_blocks(lab, 1, 1 + p_tf.size(1) // S), w)
    o_ar = head(p_ar, S)
    l_ar, g_ar = fn(o_ar, slice_blocks(lab, C, C + R), w)
    loss = split.get("tf", 1.0) * l_tf + split.get("ar", 1.0) * l_ar
    log = {f"tf_{k}": v for k, v in g_tf.items()}
    log.update({f"ar_{k}": v for k, v in g_ar.items()})
    return loss, log
