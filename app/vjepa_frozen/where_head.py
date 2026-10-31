"""운동 head + binding 항 — 릴리즈 predictor post-FT 에 붙이는 보조 손실 (2026-10-31, 사용자 결정, z_analysis/release).

왜: late (문맥 끝 가림) 에서 릴리즈 · PFT 의 병목은 "과거 물체를 다시 나타날 미래 자리로 옮겨 그 자리 토큰에 쓰는 일" 이다
    (z_analysis/release/SUMMARY_visible_late_pft_transfer.md). PFT 는 자리를 학습셋 시간표로 외웠다.
무엇:
  (1) 운동 head — predictor 마지막 층의 **문맥 토큰** 상태 (B, 8·256, Dp) 를 읽어 물체의
      c₀ (블록 8 의 자리) · v · a (블록 단위) · 있음 logit 을 낸다. 블록 b 의 자리 = c₀ + v·τ + ½·a·τ², τ = b − 8 (b = 1..16).
      자리를 직접 내지 않고 속도 · 가속만 내게 해 "학습셋 끝 자리 외우기" 의 틈을 줄인다. 손실 L_pos = 16 블록 SmoothL1 (칸 단위)
      + 있음 BCE. head 가 predictor 내부 상태를 읽으므로 기울기가 predictor 로 흐른다. 시험 때는 predictor 만 쓴다.
  (2) binding 항 — L1 에 더하는 항. 참 미래 물체 자리 (블록 9–16 마다 중심 칸의 3×3) 토큰만의 |p − LN(h)| 평균.
      물체 3×3 은 전체 칸의 ~4 % 라 표준 L1 은 정체를 화면에 퍼뜨리는 것과 자리에 묶는 것을 거의 가르지 않는다 (측정).
  손실 = L1 + λ_bind · L_bind + λ_pos · (L_pos + L_pres).
라벨: VOBench (UE TestBed) label.json 의 프레임별 물체 중심 (u, v px@512) · 가려진 비율. 특권 정보 (시뮬레이터) — "관측만으로" 판이 아니다.
  좌표는 같은 자르기 · 리사이즈 · 좌우 반전을 거쳐 토큰 칸 단위 (칸 c 의 중심 = c) 로, 블록 = 튜블릿 2 장의 평균.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.vjepa_frozen.data import FramesIndexDataset

PATCH = 16


# ----------------------------------------------------------------------------- 데이터
class TrackFramesDataset(FramesIndexDataset):
    """frames_index + VOBench 물체 궤적 라벨. 반환 ([clip], 0, [idx], {pos (T_b, 2) [행, 열] 칸 · present () · hid (T_b,)})."""

    def __init__(self, spec: dict, n_frames: int, transform, tubelet: int = 2):
        super().__init__(spec, n_frames, transform)
        if transform.window:
            raise ValueError("frames_index_track: window crop 은 프레임마다 창이 달라 라벨을 못 옮긴다")
        self.tubelet = int(tubelet)
        self.tracks = {}
        miss = 0
        for r in self.rows:
            d = os.path.join(self.frames_root, os.path.dirname(r["file_name"]))
            key = d
            if key not in self.tracks:
                with open(os.path.join(d, "label.json")) as f:
                    self.tracks[key] = json.load(f)["labels"]
        for r in self.rows:
            arm = "A" if r["variant"] in ("pos_a", "imp_ab") else "B"
            lab = self.tracks[os.path.join(self.frames_root, os.path.dirname(r["file_name"]))][arm]
            if lab and len(lab) < 96:
                miss += 1
        if miss:
            raise ValueError(f"frames_index_track: 96 장보다 짧은 라벨 {miss} 개")

    def _labels(self, row, idx, params):
        arm = "A" if row["variant"] in ("pos_a", "imp_ab") else "B"
        lab = self.tracks[os.path.join(self.frames_root, os.path.dirname(row["file_name"]))][arm]
        nb = len(idx) // self.tubelet
        if not lab:                                                   # 빈 장면 (permanence B)
            return {"pos": torch.zeros(nb, 2), "present": torch.tensor(0.0), "hid": torch.ones(nb)}
        top, left, h, w = params["box"]; r = self.transform.resolution
        u = np.array([(lab[f]["u"] - left) * r / w for f in idx], np.float32)
        v = np.array([(lab[f]["v"] - top) * r / h for f in idx], np.float32)
        hid = np.array([lab[f]["hidden"] for f in idx], np.float32)
        if params["flip"]:
            u = r - u
        col, rw = u / PATCH - 0.5, v / PATCH - 0.5                      # 칸 단위 (칸 c 의 중심 = c)
        pos = np.stack([rw, col], -1).reshape(nb, self.tubelet, 2).mean(1)
        return {"pos": torch.from_numpy(pos.astype(np.float32)), "present": torch.tensor(1.0),
                "hid": torch.from_numpy(hid.reshape(nb, self.tubelet).mean(1))}

    def __getitem__(self, i: int):
        st = random.choice(self.starts)
        x, params = self.transform(self.read_uint8(i, st), return_params=True)
        idx = np.arange(st, st + self.n_frames * self.stride, self.stride, dtype=np.int64)
        return [x], 0, [idx], self._labels(self.rows[i], idx, params)


# ----------------------------------------------------------------------------- head
class MotionHead(nn.Module):
    """문맥 토큰 상태 (B, Tc·S, Dp) → c₀ (2) · v (2) · a (2) · 있음 logit. 위치 임베딩 (블록 · 행 · 열) 을 더하고
    작은 transformer 2 층 → 쿼리 1 개 attention pooling → MLP."""

    def __init__(self, dim_in: int = 384, d: int = 256, n_ctx_blocks: int = 8, grid: int = 16, layers: int = 2, heads: int = 4):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, d))
        self.pt = nn.Parameter(torch.zeros(n_ctx_blocks, d)); self.pr = nn.Parameter(torch.zeros(grid, d)); self.pc = nn.Parameter(torch.zeros(grid, d))
        for p_ in (self.pt, self.pr, self.pc):
            nn.init.normal_(p_, std=0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True), layers)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 7))
        self.Tc, self.G = n_ctx_blocks, grid

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, _ = x.shape
        pe = (self.pt[:, None, None] + self.pr[None, :, None] + self.pc[None, None, :]).reshape(self.Tc * self.G * self.G, -1)
        h = self.enc(self.inp(x.float()) + pe[None, :N])
        o, _ = self.pool(self.q.expand(B, -1, -1), h, h)
        y = self.out(o[:, 0])
        return {"c0": y[:, 0:2] + (self.G - 1) / 2.0, "v": y[:, 2:4], "a": y[:, 4:6], "pres": y[:, 6]}


class OccHead(nn.Module):
    """공간 점유 head (2026-10-31, 06_blender_track): predictor 마지막 층 (predictor_norm 뒤, 384) 토큰마다 target 이 차지하는
    비율의 logit 하나. 선형이라 "그 토큰 안에 물체가 있다" 가 predictor 상태에 직접 실려 있어야 맞힌다. 기울기는 predictor 로 흐른다."""

    def __init__(self, dim_in: int = 384):
        super().__init__()
        self.lin = nn.Linear(dim_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x.float()).squeeze(-1)                                       # (B, N)


class WhereHeads(nn.Module):
    """운동 head (선택) + 점유 head 를 한 모듈로 (optimizer · DDP 평균 · 저장 경로를 train.py 가 하나로 다룬다)."""

    def __init__(self, motion: Optional[nn.Module], occ: Optional[nn.Module]):
        super().__init__()
        self.motion, self.occ = motion, occ


def build_heads(cfg: dict, dim_in: int) -> WhereHeads:
    mot = MotionHead(dim_in=dim_in, n_ctx_blocks=int(cfg.get("n_ctx_blocks", 8))) if float(cfg.get("pos_weight", 0.0)) > 0 else None
    return WhereHeads(mot, OccHead(dim_in))


def occ_loss(head: OccHead, states: torch.Tensor, occ: torch.Tensor, n_ctx_blocks: int, hid: torch.Tensor, cfg: dict):
    """states (B, nb·S, D) 전 토큰 (정렬 순서 = 블록 · 행 · 열) · occ (B, nb, G, G) 목표 비율 → BCE (양성 가중) + 로그 (미래 IoU)."""
    B, nb = occ.shape[:2]
    logit = head(states).reshape(B, nb, occ.shape[2], occ.shape[3])
    pw = torch.tensor(float(cfg.get("occ_pos_weight", 5.0)), device=logit.device)
    l = F.binary_cross_entropy_with_logits(logit, occ.float(), pos_weight=pw)
    log = {"occ": float(l.detach())}
    with torch.no_grad():
        pr, gt = torch.sigmoid(logit[:, n_ctx_blocks:]) > 0.5, occ[:, n_ctx_blocks:] > 0.25
        inter, uni = (pr & gt).sum((-1, -2)).float(), (pr | gt).sum((-1, -2)).float()
        ok = gt.any(-1).any(-1)
        if ok.any():
            log["occ_fut_iou"] = float((inter[ok] / uni[ok].clamp(min=1)).mean())
            hm = ok & (hid[:, n_ctx_blocks:] > 0.9)                                  # 미래 블록 중 target 이 가려진 블록
            if hm.any():
                log["occ_fut_hid_iou"] = float((inter[hm] / uni[hm].clamp(min=1)).mean())
    return l, log


def trajectory(o: Dict[str, torch.Tensor], n_blocks: int = 16, ref: int = 8) -> torch.Tensor:
    """블록 1..n_blocks 의 자리 (B, n_blocks, 2) = c₀ + v·τ + ½·a·τ², τ = b − ref (블록 ref = 마지막 문맥 블록)."""
    tau = torch.arange(1, n_blocks + 1, device=o["c0"].device, dtype=o["c0"].dtype) - ref
    return o["c0"][:, None] + o["v"][:, None] * tau[None, :, None] + 0.5 * o["a"][:, None] * (tau ** 2)[None, :, None]


def binding_mask(pos_fut: torch.Tensor, present: torch.Tensor, grid: int = 16, radius: int = 1) -> torch.Tensor:
    """참 미래 자리 (B, K, 2) [행, 열] → (B, K·S) bool: 블록마다 중심 칸 ± radius (3×3). 빈 장면 · 화면 밖은 False."""
    B, K, _ = pos_fut.shape
    rc = torch.round(pos_fut).long()
    rr = torch.arange(grid, device=pos_fut.device)
    in_r = (rr[None, None, :] - rc[..., 0:1]).abs() <= radius                     # (B, K, G)
    in_c = (rr[None, None, :] - rc[..., 1:2]).abs() <= radius
    m = in_r[..., :, None] & in_c[..., None, :]                                    # (B, K, G, G)
    onscreen = ((pos_fut >= -0.5) & (pos_fut <= grid - 0.5)).all(-1)              # (B, K)
    m = m & onscreen[..., None, None] & (present > 0.5)[:, None, None, None]
    return m.reshape(B, K * grid * grid)


def where_losses(head, ctx_states: torch.Tensor, p: torch.Tensor, h: torch.Tensor, lab: Dict[str, torch.Tensor],
                 n_ctx_blocks: int, loss_exp: float, cfg: dict, all_states: Optional[torch.Tensor] = None):
    """p · h: (B, K·S, D) 미래 (h 는 LN). lab: pos (B, T, 2) · present (B,) · hid (B, T) [· occ (B, T, G, G)].
    head: MotionHead (운동만) 또는 WhereHeads (운동 선택 + 점유). → (손실 합, 로그 dict)."""
    T = lab["pos"].shape[1]; K = T - n_ctx_blocks
    present = lab["present"]
    log, total = {}, p.new_zeros(())
    if isinstance(head, WhereHeads):                                                 # 2026-10-31 점유 head
        lam_o = float(cfg.get("occ_weight", 0.0))
        if lam_o > 0:
            if all_states is None or "occ" not in lab:
                raise ValueError("occ_weight > 0 인데 전 토큰 상태 (hook) 나 occ 라벨이 없다")
            lo, olog = occ_loss(head.occ, all_states, lab["occ"], n_ctx_blocks, lab["hid"], cfg)
            total = total + lam_o * lo
            log.update(olog)
        head = head.motion
    lam_b = float(cfg.get("bind_weight", 0.0))
    if lam_b > 0:
        m = binding_mask(lab["pos"][:, n_ctx_blocks:], present, radius=int(cfg.get("bind_radius", 1)))   # (B, K·S)
        err = (p.float() - h.float()).abs().pow(loss_exp).mean(-1) / loss_exp                             # (B, K·S)
        nm = m.sum()
        lb = (err * m).sum() / nm.clamp(min=1)
        total = total + lam_b * lb
        log["bind"] = float(lb.detach()); log["bind_frac"] = float(nm) / m.numel()
    lam_p = float(cfg.get("pos_weight", 0.0))
    if lam_p > 0 and head is not None:
        o = head(ctx_states)
        traj = trajectory(o, n_blocks=T, ref=n_ctx_blocks)                                               # (B, T, 2)
        pm = present > 0.5
        lpres = F.binary_cross_entropy_with_logits(o["pres"].float(), present.float())
        if pm.any():
            lpos = F.smooth_l1_loss(traj[pm].float(), lab["pos"][pm].float())
            with torch.no_grad():
                d = (traj[pm] - lab["pos"][pm]).abs().amax(-1)                                         # 체비셰프 (칸)
                log["pos_ctx_w1"] = float((d[:, :n_ctx_blocks] <= 1).float().mean())
                log["pos_fut_w1"] = float((d[:, n_ctx_blocks:] <= 1).float().mean())
                log["pos_fut_err"] = float(d[:, n_ctx_blocks:].mean())
        else:
            lpos = traj.sum() * 0.0
        total = total + lam_p * (lpos + lpres)
        log["pos"] = float(lpos.detach()); log["pres"] = float(lpres.detach())
    return total, log


class CtxStateHook:
    """predictor_norm 출력 (정렬된 순서의 전체 토큰) 을 잡아 둔다. temporal_prefix 마스크에서 문맥 = 앞 n_ctx 토큰."""

    def __init__(self, pred_mod: nn.Module):
        self.out: Optional[torch.Tensor] = None
        self.h = pred_mod.predictor_norm.register_forward_hook(self._f)

    def _f(self, mod, inp, out):
        self.out = out
