"""paper_lab method 패치 (진행자 2, 2026-10-21) — 블록 AR 학습에서 predictor **입력** (문맥 · 정답 입력 블록 토큰) 과
**정답** (표적 토큰) 을 따로 만든다. 기존 학습 코드 (app/vjepa_frozen/ar.py · train.py) 는 고치지 않고 모듈 속성만 바꿔 끼운다
(train.py 는 `AR.encode_blocks` · `AR.ar_losses` 를 모듈 속성으로 부른다).

설정 (dict, launch.py 가 환경변수 PL_METHOD 의 JSON 으로 넘긴다):
  input   : "raw" | "smooth" | "mnf" | "pca"     predictor 입력 블록 토큰 ("pca" = 같은 보정 클립의 토큰 공분산 상위 k 주성분 사영, 10-23 추가)
  target  : "raw" | "smooth" | "mnf" | "pca"     손실 정답
  k_noise : 잡음 평균 횟수 (smooth), 기본 4
  sigma   : 화소 잡음 표준편차 (0–1 화소 척도), 기본 2/255 (E1 · 다른 팀과 같은 정의)
  mnf     : {"calib": <E1 보정 npz>, "k": 256}  — 떨림 억제 부분공간 (MNF) 으로 사영 뒤 1280 차원으로 되돌림 + LN
  occ     : {"p": 0.5, "area": [0.15, 0.4]}  — Xie & Alanwar 꼴 대조 팔: 입력 토큰에서 한 직사각형 칸 묶음을
            임의의 문맥 블록부터 창 끝까지 0 벡터로 가린다 (정답은 가리지 않음)
  occ_px  : {"p": 0.5, "area": [0.15, 0.4], "where": "random"|"motion", "topk": 8}  — 같은 생각을 **픽셀 가림막**으로: 입력 영상에만 단색 직사각형을 임의 블록부터
            창 끝까지 그린 뒤 인코딩한다 (정답은 가리지 않은 영상). 블록 AR 에는 mask token 이 없어 0 토큰은 분포 밖이라,
            평가 (VOBench 끝 가림 = 보이는 가림막) 와 같은 꼴의 입력을 쓰는 판 (작성자 반박 대응). 인코딩이 한 번 더 든다
  cw_ctx  : {"ref": 4, "gamma": 1.0, "floor": 0.0, "space": "raw"|"mnf"}  — 운반 가중의 **문맥 판** ("space": "mnf" 이면
            가중 지도만 떨림 억제 공간에서 계산 — "mnf" 설정의 사영을 쓰되 입력 · 정답은 바꾸지 않는다): 가중을 정답이 아니라 입력 (그 블록까지의
            문맥) 의 변화로만 정한다 (마지막 입력 블록 − 그 앞 ref 블록 평균, 칸 이웃으로 넓힘). 가중이 정답과 독립이면
            칸별 L1 최적 (조건부 중앙값) 이 안 바뀐다 → '늘 바뀐다고 찍는' 편향 없이 움직이는 칸에 학습을 더 쓴다
            (다른 팀 열 번째 회신: 정답 의존 가중은 최적을 분위수 1/(1+λ) 로 옮긴다)
  cw      : {"ref": 4, "gamma": 1.0, "floor": 0.0}  — 운반 가중 예측 손실: 칸마다 정답이 '앞 ref 블록 정답 평균' 과 다른 만큼
            (블록 안 평균 1 로 정규화, gamma 제곱, floor 를 더해) 손실에 가중. 문맥 평균 · 복사로 맞는 칸의 몫을 줄여
            움직이는 내용 (운반) 의 몫을 키운다 (E1 · E2: 전체 토큰 L1 에서 운반 보상 3.5 %, 물체 자리만 보면 35 %)
아무 키도 없으면 원래 학습과 비트 단위로 같다 (raw / raw / occ 없음).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from app.vjepa_frozen import ar as AR

CFG: dict = {}
HOLD = {"h_in": None, "h_tgt": None}
_ORIG_ENC = AR.encode_blocks
_ORIG_LOSS = AR.ar_losses
_MNF = {}
_GEN = {}


TRAIN_ONLY = ("occ_px", "occ", "cw", "cw_ctx")          # 학습 전용 성분 — 평가 · planning 입력에는 절대 걸지 않는다
INFER_KEYS = ("input", "target", "k_noise", "sigma", "mnf", "seed")   # 추론에도 같은 변환이 필요한 성분 (관측 인코딩)


_EVAL = {"on": False}
_ORIG_VAL = AR.run_val_ar


def _run_val_ar(*a, **k):
    """학습 중 in-loop 검증: 학습 전용 성분 (입력 가림막) 을 끄고 돈다 — 검증 입력이 가려지지 않고, 가림막 난수도 안 쓴다."""
    _EVAL["on"] = True
    try:
        return _ORIG_VAL(*a, **k)
    finally:
        _EVAL["on"] = False


def install(cfg: dict, train: bool = True):
    """train=False (평가 · planning) 이면 학습 전용 성분을 뺀다. 평가 하네스는 install_eval 을 부른다."""
    c = dict(cfg or {})
    if not train:
        c = {k: v for k, v in c.items() if k not in TRAIN_ONLY}
    CFG.clear(); CFG.update(c)
    AR.encode_blocks = encode_blocks
    AR.ar_losses = ar_losses
    AR.run_val_ar = _run_val_ar
    return CFG


def install_eval(cfg: dict):
    return install(cfg, train=False)


def split_cfg(cfg: dict):
    """pl_method.json 에 남길 꼴: {"method": 전체, "infer": 평가에 거는 것, "train_only": 학습 전용}."""
    cfg = dict(cfg or {})
    return {"method": cfg, "infer": {k: v for k, v in cfg.items() if k not in TRAIN_ONLY},
            "train_only": {k: v for k, v in cfg.items() if k in TRAIN_ONLY}}


def _std(dev):
    return torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)


def _gen(dev):
    if dev not in _GEN:
        g = torch.Generator(device=dev); g.manual_seed(int(CFG.get("seed", 0)) + 7919 * (dev.index or 0)); _GEN[dev] = g
    return _GEN[dev]


def _mnf_mats(dev, D):
    key = (str(dev), D)
    if key not in _MNF:
        import scipy.linalg as sl
        z = np.load(CFG["mnf"]["calib"])
        SN = z["SN"] / float(z["nN"]) / 2.0                                   # cov(h1 − h2) = 2 Σ_n
        nS = float(z["nS"]); mu = z["mS"] / nS; SS = z["SS"] / nS - np.outer(mu, mu)
        reg = 1e-4 * np.trace(SN) / D
        lam, V = sl.eigh(SS, SN + reg * np.eye(D)); V = V[:, ::-1][:, : int(CFG["mnf"].get("k", 256))]
        M = V @ V.T @ (SN + reg * np.eye(D))                                   # 행 벡터: x̂ = μ + (x − μ) M   (MNF 역변환에서 잡음 성분만 0)
        _MNF[key] = (torch.tensor(mu, dtype=torch.float32, device=dev), torch.tensor(M, dtype=torch.float32, device=dev))
    return _MNF[key]


def _pca_mats(dev, D):
    """같은 보정 파일 (CFG["mnf"]["calib"]) 의 토큰 공분산 상위 k 주성분으로 사영 (떨림 억제 64 와 같은 차원의 대조). 행 벡터 x̂ = μ + (x − μ) U Uᵀ."""
    key = ("pca", str(dev), D)
    if key not in _MNF:
        z = np.load(CFG["mnf"]["calib"])
        nS = float(z["nS"]); mu = z["mS"] / nS; SS = z["SS"] / nS - np.outer(mu, mu)
        ev, U = np.linalg.eigh(SS); U = U[:, ::-1][:, : int(CFG["mnf"].get("k", 256))]
        _MNF[key] = (torch.tensor(mu, dtype=torch.float32, device=dev), torch.tensor(U @ U.T, dtype=torch.float32, device=dev))
    return _MNF[key]


def _view(enc, clips, tubelet, mode, cache):
    if mode in cache: return cache[mode]
    if mode == "raw":
        h = _ORIG_ENC(enc, clips, tubelet)
    elif mode == "smooth":
        x = clips.float(); s = float(CFG.get("sigma", 2.0 / 255.0)) / _std(x.device); g = _gen(x.device)
        hs = [_ORIG_ENC(enc, (x + torch.randn(x.shape, generator=g, device=x.device) * s).to(clips.dtype), tubelet).float()
              for _ in range(int(CFG.get("k_noise", 4)))]
        h = torch.stack(hs).mean(0).to(hs[0].dtype)                           # 블록별 LN 뒤 평균 (E1 정의)
    elif mode == "mnf":
        h0 = _view(enc, clips, tubelet, "raw", cache).float(); mu, M = _mnf_mats(h0.device, h0.size(-1))
        h = F.layer_norm(mu + (h0 - mu) @ M, (h0.size(-1),))
    elif mode == "pca":
        h0 = _view(enc, clips, tubelet, "raw", cache).float(); mu, M = _pca_mats(h0.device, h0.size(-1))
        h = F.layer_norm(mu + (h0 - mu) @ M, (h0.size(-1),))
    else:
        raise ValueError(f"paper_lab patch: 모르는 mode {mode}")
    cache[mode] = h
    return h


def _occlude_px(clips, tubelet):
    """clips (B, 3, T, H, W) 정규화. 표본마다 확률 p 로 단색 직사각형을 블록 b0 (1..nb−1) 의 첫 프레임부터 끝까지 그린다."""
    o = CFG["occ_px"]; B, _, T, H, W = clips.shape; nb = T // tubelet; dev = clips.device; g = _gen(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); std = _std(dev)[0]
    x = clips.clone()
    for b in range(B):
        if torch.rand(1, generator=g, device=dev).item() >= float(o.get("p", 0.5)): continue
        lo, hi = o.get("area", [0.15, 0.4]); a = lo + (hi - lo) * torch.rand(1, generator=g, device=dev).item()
        ar_ = 0.75 + 0.75 * torch.rand(1, generator=g, device=dev).item()
        hh = min(H, max(min(8, H), int(round(math.sqrt(a * H * W / ar_))))); ww = min(W, max(min(8, W), int(round(a * H * W / hh))))
        y0 = int(torch.randint(0, H - hh + 1, (1,), generator=g, device=dev)); x0 = int(torch.randint(0, W - ww + 1, (1,), generator=g, device=dev))
        b0 = int(torch.randint(1, max(2, nb), (1,), generator=g, device=dev))
        if o.get("where") == "motion":
            # 움직이는 칸 가림막: 가림막이 시작하는 블록 b0 직전 두 프레임의 화소 차가 큰 칸 (상위 k 중 하나) 을 덮는다.
            # 입력만 보고 정한다 (정답 무관) → 손실 최적은 그대로, '가려진 내용이 움직이는' 학습 칸을 늘린다
            f1, f0 = b0 * tubelet - 1, max(0, b0 * tubelet - 1 - tubelet)
            dmap = (clips[b, :, f1].float() - clips[b, :, f0].float()).abs().mean(0)               # (H, W)
            cell = 16 if H % 16 == 0 and W % 16 == 0 and H >= 16 else 1
            cm = F.avg_pool2d(dmap[None, None], cell, cell)[0, 0].flatten()
            top = torch.topk(cm, k=min(int(o.get("topk", 8)), cm.numel())).indices
            c = int(top[int(torch.randint(0, len(top), (1,), generator=g, device=dev))])
            gw = W // cell; cy, cx = (c // gw) * cell + cell // 2, (c % gw) * cell + cell // 2
            y0 = int(min(max(cy - hh // 2, 0), H - hh)); x0 = int(min(max(cx - ww // 2, 0), W - ww))
        col = torch.rand(3, 1, 1, 1, generator=g, device=dev)
        x[b, :, b0 * tubelet:, y0:y0 + hh, x0:x0 + ww] = ((col - mean) / std).to(x.dtype)
    return x


def encode_blocks(enc, clips, tubelet):
    cache = {}
    if CFG.get("occ_px") and not _EVAL["on"]:
        h_in = _view(enc, _occlude_px(clips, tubelet), tubelet, CFG.get("input", "raw"), {})
    else:
        h_in = _view(enc, clips, tubelet, CFG.get("input", "raw"), cache)
    h_tgt = _view(enc, clips, tubelet, CFG.get("target", "raw"), cache)
    HOLD["h_in"], HOLD["h_tgt"] = h_in, h_tgt
    return h_in


def _occlude(h, S, C):
    """Xie & Alanwar 꼴: 직사각형 칸 묶음을 임의 문맥 블록 t0 (1..C−1) 부터 창 끝까지 입력에서 0 으로."""
    o = CFG["occ"]; B, N, D = h.shape; nb = N // S; G = int(round(math.sqrt(S)))
    g = _gen(h.device); h = h.clone()
    for b in range(B):
        if torch.rand(1, generator=g, device=h.device).item() >= float(o.get("p", 0.5)): continue
        lo, hi = o.get("area", [0.15, 0.4]); a = lo + (hi - lo) * torch.rand(1, generator=g, device=h.device).item()
        ar_ = 0.75 + 0.75 * torch.rand(1, generator=g, device=h.device).item()
        hh = max(1, min(G, int(round(math.sqrt(a * S / ar_))))); ww = max(1, min(G, int(round(a * S / hh))))
        y0 = int(torch.randint(0, G - hh + 1, (1,), generator=g, device=h.device)); x0 = int(torch.randint(0, G - ww + 1, (1,), generator=g, device=h.device))
        t0 = int(torch.randint(1, max(2, C), (1,), generator=g, device=h.device))
        m = torch.zeros(G, G, dtype=torch.bool, device=h.device); m[y0:y0 + hh, x0:x0 + ww] = True
        hv = h[b].view(nb, S, D); hv[t0:, m.view(-1)] = 0
    return h


def ar_losses(predictor, h, S, C, K, loss_exp, rollout_steps, actions=None, out=None):
    """원래 ar_losses 와 같은 식. 다른 점: predictor 입력 = h (입력 판, 가림 팔이면 가린 판), 손실 정답 = 같은 영상의 정답 판."""
    tgt = HOLD["h_tgt"] if HOLD["h_in"] is h else h
    if CFG.get("occ"): h = _occlude(h, S, C)
    if tgt is h and not CFG.get("occ") and not CFG.get("cw") and not CFG.get("cw_ctx"):          # (occ_px 는 encode 에서 입력 ≠ 정답이라 여기 안 온다)
        return _ORIG_LOSS(predictor, h, S, C, K, loss_exp, rollout_steps, actions=actions, out=out)
    B, N = h.shape[:2]; nb = N // S
    if nb != C + K or nb < 2:
        raise ValueError(f"ar_losses: nb={nb} 가 C+K={C + K} 와 다르거나 2 미만")
    R = max(1, min(int(K), int(rollout_steps)))
    idx = AR.block_indices(B, nb, S, h.device)
    p_tf, p_ar = predictor(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": R, "actions": actions})
    if out is not None:
        out["p_tf"], out["p_ar"] = p_tf, p_ar
    if CFG.get("cw_ctx"):
        hw = h
        if CFG["cw_ctx"].get("space") == "mnf":                               # 가중 지도만 떨림 억제 공간에서 (학습 · 채점은 그대로)
            mu, M = _mnf_mats(h.device, h.size(-1))
            hw = F.layer_norm(mu + (h.detach().float() - mu) @ M, (h.size(-1),))
        w_tf, w_ar = _cw_ctx_weights(hw, S, C, R)
        _log_wstat(w_tf, w_ar, h, S, C, R)
        jloss = _wl1(p_tf, tgt[:, S:], w_tf, loss_exp)
        sloss = _wl1(p_ar, tgt[:, C * S: (C + R) * S], w_ar, loss_exp)
    elif CFG.get("cw"):
        w_tf, w_ar = _cw_weights(tgt, S, C, R)
        jloss = _wl1(p_tf, tgt[:, S:], w_tf, loss_exp)
        sloss = _wl1(p_ar, tgt[:, C * S: (C + R) * S], w_ar, loss_exp)
    else:
        jloss = AR.l1(p_tf, tgt[:, S:], loss_exp)
        sloss = AR.l1(p_ar, tgt[:, C * S: (C + R) * S], loss_exp)
    return jloss + sloss, jloss, sloss, R


_WSTAT = {"n": 0, "acc": []}


def _log_wstat(w_tf, w_ar, h, S, C, R, every=50):
    """가중이 실제로 걸렸는지 (학습 배치에서): 걸린 가중의 변동 계수와, 같은 배치에서 A2c 판 (원래 공간 · gamma 1) 가중의 변동 계수.
    PL_WSTAT_FILE 이 있을 때만 every 번마다 평균을 jsonl 한 줄로 쓴다. 학습에는 영향 없음 (no_grad, 별도 계산)."""
    path = os.environ.get("PL_WSTAT_FILE")
    if not path or _EVAL["on"]:
        return
    with torch.no_grad():
        cv = lambda w: (w.float().view(w.size(0), -1, S).std(-1) / w.float().view(w.size(0), -1, S).mean(-1).clamp(min=1e-6)).mean().item()
        o = CFG["cw_ctx"]; saved = dict(o)
        try:
            CFG["cw_ctx"] = {"ref": o.get("ref", 4), "gamma": 1.0}
            r_tf, r_ar = _cw_ctx_weights(h, S, C, R)
        finally:
            CFG["cw_ctx"] = saved
        _WSTAT["acc"].append((cv(w_tf), cv(w_ar), cv(r_tf), cv(r_ar)))
    _WSTAT["n"] += 1
    if _WSTAT["n"] % every == 0:
        a = np.asarray(_WSTAT["acc"]); _WSTAT["acc"] = []
        with open(path, "a") as f:
            f.write(json.dumps({"call": _WSTAT["n"], "cv_tf": round(float(a[:, 0].mean()), 4), "cv_ar": round(float(a[:, 1].mean()), 4),
                                "a2c_cv_tf": round(float(a[:, 2].mean()), 4), "a2c_cv_ar": round(float(a[:, 3].mean()), 4)}) + "\n")


def _cw_weights(tgt, S, C, R):
    """칸 가중 (B, n, S): 정답 블록 b 가 앞 ref 개 정답 블록 평균과 다른 정도 (특징 평균 절댓값), 블록 안 평균 1.
    정답 입력 학습 (TF) 의 정답 = 블록 1..nb−1, rollout 의 정답 = 블록 C..C+R−1 — 둘 다 그 블록 바로 앞까지만 본다 (인과)."""
    o = CFG["cw"]; ref = int(o.get("ref", 4)); gam = float(o.get("gamma", 1.0)); flo = float(o.get("floor", 0.0))
    B, N, D = tgt.shape; nb = N // S
    with torch.no_grad():
        t = tgt.detach().float().view(B, nb, S, D)
        cs = torch.cumsum(t, 1); cs = torch.cat([torch.zeros_like(cs[:, :1]), cs], 1)        # cs[:, j] = Σ_{i<j}
        idx = torch.arange(nb, device=t.device); lo = (idx - ref).clamp(min=0)
        cnt = (idx - lo).clamp(min=1).view(1, nb, 1, 1).float()
        mean_prev = (cs[:, idx] - cs[:, lo]) / cnt                                                 # 블록 j 앞 ref 개 평균 (j=0 은 0)
        d = (t - mean_prev).abs().mean(-1)                                                         # (B, nb, S)
        d = d.pow(gam); d = d / d.mean(-1, keepdim=True).clamp(min=1e-6); d = (d + flo) / (1.0 + flo)
        # rollout 정답 블록 C..C+R−1 의 기준 = 문맥 끝 ref 블록 평균 (rollout 은 정답 블록을 못 본다)
        m_ctx = t[:, max(0, C - ref):C].mean(1, keepdim=True)
        d_ar = (t[:, C:C + R] - m_ctx).abs().mean(-1).pow(gam)
        d_ar = d_ar / d_ar.mean(-1, keepdim=True).clamp(min=1e-6); d_ar = (d_ar + flo) / (1.0 + flo)
    return d[:, 1:].reshape(B, -1), d_ar.reshape(B, -1)


def _cw_ctx_weights(x, S, C, R):
    """문맥 판 가중 (B, n, S): 정답 블록 j 의 가중은 입력 블록 j−1 과 그 앞 ref 블록 평균의 차 (특징 평균 절댓값) 를
    칸 이웃으로 넓힌 것. 정답을 보지 않는다. rollout 걸음 k 의 가중 = 문맥 끝 블록의 변화를 반경 min(k, 3) 으로 넓힌 것."""
    o = CFG["cw_ctx"]; ref = int(o.get("ref", 4)); gam = float(o.get("gamma", 1.0)); flo = float(o.get("floor", 0.0))
    B, N, D = x.shape; nb = N // S; G = int(round(math.sqrt(S)))
    with torch.no_grad():
        t = x.detach().float().view(B, nb, S, D)
        cs = torch.cumsum(t, 1); cs = torch.cat([torch.zeros_like(cs[:, :1]), cs], 1)
        idx = torch.arange(nb, device=t.device); lo = (idx - ref).clamp(min=0)
        cnt = (idx - lo).clamp(min=1).view(1, nb, 1, 1).float()
        mean_prev = (cs[:, idx] - cs[:, lo]) / cnt
        ch = (t - mean_prev).abs().mean(-1)                                    # (B, nb, S) 블록 j 의 '앞 블록 평균 대비 변화' (입력만)
        ch[:, 0] = ch[:, 0].mean(-1, keepdim=True)                              # 첫 블록은 앞이 없다 → 균등

        def dil(m, r):                                                          # (B, n, S) → 칸 이웃 최댓값으로 넓힘
            g = m.reshape(-1, 1, G, G)
            return F.max_pool2d(g, 2 * r + 1, 1, r).reshape(m.shape)

        def norm(d):
            d = d.pow(gam); d = d / d.mean(-1, keepdim=True).clamp(min=1e-6); return (d + flo) / (1.0 + flo)
        w_tf = norm(dil(ch[:, :-1], 1))                                         # 정답 블록 1..nb−1 ← 입력 블록 0..nb−2
        last = ch[:, C - 1:C]                                                   # 문맥 끝 블록의 변화
        w_ar = torch.cat([norm(dil(last, min(k, 3))) for k in range(1, R + 1)], 1)
    return w_tf.reshape(B, -1), w_ar.reshape(B, -1)


def _wl1(p, h, w, loss_exp):
    e = (p.float() - h.float()).abs().pow(loss_exp).mean(-1)                                   # (B, n·S)
    return (w * e).mean() / loss_exp
