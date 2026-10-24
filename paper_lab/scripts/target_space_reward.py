#!/usr/bin/env python3
"""정답 공간과 '블록 평균 vs 운반' 보상 — 학습 없는 측정 (paper_lab 진행자 2, 2026-10-21, 실험 E1 · E2).

질문: 블록별 target encoder 표적 (블록 AR 의 학습 정답) 에서 L1 은 '같은 칸의 문맥 블록 평균' 을 크게 보상하고
'물체를 참 변위만큼 옮기기 (운반)' 는 거의 보상하지 않는다 (slow_feature_analysis 회신, 3D 합성).
표적을 바꾸면 (픽셀 잡음 평균 = 떨림 제거 / 잡음 대비 신호 큰 부분공간) 이 보상의 균형이 바뀌는가,
그리고 '참 변위로 옮긴 평균' (운반 oracle) 이 VoE 쌍을 얼마나 푸는가.

공간 (전부 블록별 LN(target_encoder), 블록 = 2 장, 32 장 = 16 블록, 문맥 8 · 미래 8):
  raw     LN(h(x)), bf16                           지금 블록 AR 의 정답
  fp32    LN(h(x)), fp32 가중치 · fp32 계산 (TF32 끔)          (slow_feature_analysis 의 잡음 줄인 정답 (i))
  smooth  mean_k LN(h(x + e_k)), e_k ~ N(0, (2/255)²) 화소마다, k = 1..8 (블록별 정규화 뒤 평균 = slow_feature_analysis 정의 (ii), Cohen 2019 꼴)
  smooth_pm1 (SSv2 만) mean_k LN(h(x ± 1/255 무작위 부호))  (첫 정의 — 압축 · 센서 섭동을 못 덮는다는 지적 검정용)
  snrK    raw @ V_K, V = Σ_s v = λ Σ_n v 의 상위 K 고유벡터 (Σ_n = 픽셀 잡음 차의 공분산, Σ_s = 토큰 공분산; SSv2 보정 clip 에서)
  pcaK    raw @ U_K, U = Σ_s 의 상위 K 주성분 (대조: 잡음을 보지 않고 차원만 줄이면)
render (문맥만으로 만든 미래 8 블록. matched 쌍의 두 미래에 같게 쓰인다):
  copy · mean2 · mean4 · mean8 (같은 칸 문맥 n 블록 평균) · bg (물체가 지나간 칸은 물체 없는 블록만 평균)
  paste_or (마지막 블록 위에 마지막으로 보인 블록의 물체 3×3 을 참 자리에) · mc_or / mc_cv / mc_ca (bg 위에 보인 블록들의 물체 3×3 평균을
  참 자리 / 등속 외삽 자리 / 2 차 외삽 자리에) — 위치는 라벨 (지각이 완벽하다는 가정, design_headroom.py 와 같은 규칙)

  python paper_lab/scripts/target_space_reward.py calib   --n 200 --gpus 1      # SSv2 0..199: Σ_n, Σ_s, SSv2 지표 (raw · smooth)
  python paper_lab/scripts/target_space_reward.py basis                         # → basis.npz (snr64 · snr256 · pca256)
  python paper_lab/scripts/target_space_reward.py extract --set ssv2 --n 200 --gpus 1   # SSv2 200..399, 다섯 공간
  python paper_lab/scripts/target_space_reward.py extract --set vobench --gpus 1        # VOBench 720 block × 4 clip
  python paper_lab/scripts/target_space_reward.py analyze
산출: $TSR_OUT (기본 paper_lab/_scratch/tspace/) 의 npz · result.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
OUT = Path(os.environ.get("TSR_OUT", REPO / "paper_lab/_scratch/tspace"))
S, NB, C, K, D, G = 256, 16, 8, 8, 1280, 16
# (2026-10-22 PLAN §12-1) TSR_ENC=dinov2 → 블록 b = DINOv2 ViT-B/14-reg 한 장 (f_{2b+1}, 224 px → 16 × 16) 토큰 LN, D = 768.
#   잡음 (가우시안 · ±1/255) 은 그 encoder 의 입력 해상도 (224) 에서 넣는다. 기본 (vjepa2) 은 예전과 같다.
# TSR_SPACES=raw,pca64 … → extract 에서 그 공간만 (fp32 · 잡음 평균 계산을 건너뛴다 — 주성분 64 만 덧붙여 잴 때).
ENC_NAME = os.environ.get("TSR_ENC", "vjepa2")
assert ENC_NAME in ("vjepa2", "dinov2"), ENC_NAME
if ENC_NAME == "dinov2":
    D = 768
SPACES_ONLY = [x for x in os.environ.get("TSR_SPACES", "").split(",") if x]
KN = int(os.environ.get("TSR_KNOISE", "8"))            # 잡음 평균 횟수
VAL = REPO / "data_csv/ssv2/val_min48.csv"
RENDERS = ("copy", "mean2", "mean4", "mean8", "shr50", "shr70", "shr85", "z50", "z70", "z85", "sp70", "nb70", "bg", "paste_or", "mc_or", "mc_cv", "mc_ca")
SHR = (("shr50", 0.5), ("shr70", 0.7), ("shr85", 0.85))   # 마지막 블록을 문맥 평균 토큰 쪽으로 (⚠️ 당기는 표적에 역사가 들어 있다 — 진행자 1 반박)
# 역사 없는 수축 (진행자 1 제안): z = 원점 쪽 축소, sp = 마지막 블록의 칸 평균 쪽, nb = 마지막 블록의 3×3 이웃 평균 쪽


def _histfree(torch, last):
    """last (G, G, D') 마지막 블록만으로 만든 수축 예측들."""
    import torch.nn.functional as F_
    sp = last.mean((0, 1))
    nb = F_.avg_pool2d(last.permute(2, 0, 1).unsqueeze(0), 3, 1, 1, count_include_pad=False)[0].permute(1, 2, 0)
    out = {f"z{int(l * 100)}": l * last for l in (0.5, 0.7, 0.85)}
    out["sp70"] = 0.7 * last + 0.3 * sp
    out["nb70"] = 0.7 * last + 0.3 * nb
    return out
SPACES_ALL = ("raw", "fp32", "smooth", "smooth_pm1", "snr64", "snr256", "pca64", "pca256")   # pca64 = 떨림 억제 64 와 같은 차원의 주성분 대조 (10-22 추가)
BASIS_KEYS = ("snr64", "snr256", "pca64", "pca256")


class _Noisy:
    """DINOv2 판: 잡음을 encoder 입력 해상도 (224) 에서 넣으려고 '잡음 요청' 을 enc 까지 미룬다."""
    def __init__(self, x, gen, kind):
        self.x, self.gen, self.kind = x, gen, kind
LIMIT = int(os.environ.get("TSR_LIMIT", "0"))          # 스모크: 앞 N 개만


def _setup(rank):
    import torch
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config
    torch.cuda.set_device(rank); torch.set_num_threads(6); dev = torch.device("cuda", rank)
    os.environ.setdefault("PRESENCE_SET", "vob_readout_primitive")
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    model = dict(ir.rp.MODEL)                           # 릴리즈 ViT-H fpc64-256 (target_encoder = EMA 만 쓴다)
    if ENC_NAME == "dinov2":                            # V-JEPA 2 는 안 올린다 (WMADataset 용 model dict 만)
        return torch, dev, (None, None), model
    bundle = build_from_config(model, dev)
    bundle32 = build_from_config(dict(model, dtype="float32"), dev)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return torch, dev, (bundle, bundle32), model


def _encoders(torch, dev, bundles):
    import torch.nn.functional as F
    if ENC_NAME == "dinov2":
        return _encoders_dino(torch, dev)
    from app.vjepa_frozen.ar import encode_blocks
    bundle, bundle32 = bundles
    STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    step = (1.0 / 255.0) / STD

    def enc(x):                                         # x (B, 3, 32, 256, 256) fp32 정규화 → (B, 16, G, G, D) LN, bf16 (학습 · 채점 관례)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = encode_blocks(bundle.target_encoder, x.to(bundle.dtype), 2).float()
        return h.view(x.size(0), -1, G, G, D)

    def enc32(x):                                       # fp32 가중치 · autocast 끔 · TF32 끔
        with torch.autocast("cuda", enabled=False):
            h = encode_blocks(bundle32.target_encoder, x.float(), 2).float()
        return h.view(x.size(0), -1, G, G, D)

    def gauss(x, gen):
        return x + torch.randn(x.shape, generator=gen, device=dev) * (2.0 * step)

    def pm1(x, gen):
        s_ = torch.randint(0, 2, x.shape, generator=gen, device=dev, dtype=torch.int8).float().mul_(2).sub_(1)
        return x + s_ * step

    def smooth_from(hs):                                # 블록별 정규화 (enc 의 LN) 뒤 평균
        return torch.stack(hs).mean(0)
    return enc, enc32, gauss, pm1, smooth_from


def _ssv2_loader(lo, hi):
    import decord
    from analysis.intphys2.dataset import build_video_transform
    rows = [r for r in csv.reader(open(VAL), delimiter=" ")]
    rng = np.random.default_rng(0); rng.shuffle(rows); files = [r[0] for r in rows[lo:hi]]
    tf_ = build_video_transform(256, 1.0)

    def load(f):
        try:
            vr = decord.VideoReader(f, num_threads=1)
            return tf_(vr.get_batch(list(range(32))).asnumpy()) if len(vr) >= 32 else None
        except Exception:
            return None
    return files, load


def _encoders_dino(torch, dev):
    """DINOv2 ViT-B/14-reg 한 장 판 (진행자 1 의 carry_ext 로더 · 같은 가중치). x (B, 3, 32, 256, 256) 정규화 → (B, 16, G, G, 768) LN."""
    import torch.nn.functional as F
    sys.path.insert(0, str(REPO / "paper_lab/scripts"))
    import carry_ext as CE
    STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def enc(xin):
        noisy = isinstance(xin, _Noisy)
        x = xin.x if noisy else xin
        B, _, T, H, W = x.shape
        fr = x[:, :, 1:T:2].transpose(1, 2).reshape(-1, 3, H, W).float()
        fr = F.interpolate(fr, size=(224, 224), mode="bilinear", align_corners=False, antialias=False)
        if noisy:
            if xin.kind == "gauss":
                fr = fr + torch.randn(fr.shape, generator=xin.gen, device=dev) * (2.0 / 255.0) / STD
            else:
                s_ = torch.randint(0, 2, fr.shape, generator=xin.gen, device=dev, dtype=torch.int8).float().mul_(2).sub_(1)
                fr = fr + s_ * (1.0 / 255.0) / STD
        m = CE._dino(dev)
        outs = []
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            for k in range(0, fr.shape[0], 64):
                outs.append(m.forward_features(fr[k:k + 64])["x_norm_patchtokens"].float())
        tok = torch.cat(outs).view(B, T // 2, G, G, -1)
        return F.layer_norm(tok, (tok.shape[-1],))

    def gauss(x, gen):
        return _Noisy(x, gen, "gauss")

    def pm1(x, gen):
        return _Noisy(x, gen, "pm1")

    def smooth_from(hs):
        return torch.stack(hs).mean(0)
    return enc, enc, gauss, pm1, smooth_from      # enc32 = enc (DINOv2 는 이미 fp32 로 돈다)


# ------------------------------------------------------------------------------------------------ calib
def calib(rank, world, n, bs):
    torch, dev, bundles, _ = _setup(rank)
    enc, enc32, gauss, pm1, smooth_from = _encoders(torch, dev, bundles)
    files, load = _ssv2_loader(0, n)
    mine = files[rank::world]
    if LIMIT: mine = mine[:LIMIT]
    gen = torch.Generator(device=dev); gen.manual_seed(1000 + rank)
    SN = torch.zeros(D, D, dtype=torch.float64, device=dev); nN = 0
    SS = torch.zeros(D, D, dtype=torch.float64, device=dev); mS = torch.zeros(D, dtype=torch.float64, device=dev); nS = 0
    rec = defaultdict(list); t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, len(mine), bs):
            xs = [v for v in (load(f) for f in mine[b0:b0 + bs]) if v is not None]
            if len(xs) < 2: continue
            x = torch.stack(xs).to(dev).float(); B = x.size(0)
            h0 = enc(x); h32 = enc32(x)
            hg = [enc(gauss(x, gen)) for _ in range(KN)]
            hp = [enc(pm1(x, gen)) for _ in range(KN)]
            d = (hg[0] - hg[1]).reshape(-1, D).double()                     # Σ_n: 가우시안 2/255 두 판의 차
            SN += d.T @ d; nN += d.size(0)
            t = h0.reshape(-1, D).double(); SS += t.T @ t; mS += t.sum(0); nS += t.size(0)
            cross = (h0 - h0.roll(1, 0)).abs().mean().item()                  # 다른 clip 같은 칸 거리 (척도)
            rec["jit_cross"].append(cross)
            rec["jit_g2"].append((hg[0] - hg[1]).abs().mean().item() / cross)
            rec["jit_pm1"].append((hp[0] - hp[1]).abs().mean().item() / cross)
            rec["jit_fp32"].append((h0 - h32).abs().mean().item() / cross)
            sm = smooth_from(hg); sm2 = smooth_from([enc(gauss(x, gen)) for _ in range(KN)])
            rec["jit_smooth"].append((sm - sm2).abs().mean().item() / cross)  # 두 평활 판끼리 (남은 떨림)
            for sp, X in (("raw", h0), ("fp32", h32), ("smooth", sm), ("smooth_pm1", smooth_from(hp))):
                for b in range(B):
                    _ssv2_metrics(torch, X[b], rec, sp)
            if rank == 0 and (b0 // bs) % 5 == 0: print(f"[calib r0] {b0 + B}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"calib_r{rank}.npz", SN=SN.cpu().numpy(), nN=nN, SS=SS.cpu().numpy(), mS=mS.cpu().numpy(), nS=nS,
             **{k: np.array(v) for k, v in rec.items()})
    print(f"[calib r{rank}] done ({time.time() - t0:.0f}s)", flush=True)


def _ssv2_metrics(torch, X, rec, sp):
    """X (16, G, G, D') 한 clip. copy · mean_n 의 미래 8 블록 L1 (블록별)."""
    F_ = X[C:]
    mu = X[:C].mean((0, 1, 2))
    rs = [("copy", X[C - 1]), ("mean2", X[C - 2:C].mean(0)), ("mean4", X[C - 4:C].mean(0)), ("mean8", X[:C].mean(0))]
    rs += [(n_, lam * X[C - 1] + (1 - lam) * mu) for n_, lam in SHR]
    rs += list(_histfree(torch, X[C - 1]).items())
    for name, R in rs:
        e = (R.unsqueeze(0) - F_).abs().mean(-1).mean((-1, -2))           # (8,)
        rec[f"{sp}/{name}"].append(e.cpu().numpy())
    # 이웃 블록 변화 (떨림 + 진짜 변화) 와 같은 블록 안 칸끼리 거리 (그 공간의 척도)
    rec[f"{sp}/tchg"].append((X[1:C] - X[:C - 1]).abs().mean().item())
    rec[f"{sp}/xpos"].append((X[:C] - X[:C].roll(5, 1).roll(7, 2)).abs().mean().item())


# ------------------------------------------------------------------------------------------------ basis
def basis():
    import scipy.linalg as sl
    Z = [np.load(f) for f in sorted(OUT.glob("calib_r*.npz"))]
    SN = sum(z["SN"] for z in Z) / sum(int(z["nN"]) for z in Z) / 2.0      # cov(h1 - h2) = 2 Σ_n
    nS = sum(int(z["nS"]) for z in Z); mS = sum(z["mS"] for z in Z) / nS
    SS = sum(z["SS"] for z in Z) / nS - np.outer(mS, mS)
    reg = 1e-4 * np.trace(SN) / D
    lam, V = sl.eigh(SS, SN + reg * np.eye(D))                             # 오름차순, V.T SN V = I
    lam, V = lam[::-1], V[:, ::-1]
    ev, U = np.linalg.eigh(SS); ev, U = ev[::-1], U[:, ::-1]
    snr_raw = np.diag(U.T @ SS @ U) / np.maximum(np.diag(U.T @ SN @ U), 1e-12)
    np.savez(OUT / "basis.npz", snr64=V[:, :64].astype(np.float32), snr256=V[:, :256].astype(np.float32),
             pca64=U[:, :64].astype(np.float32), pca256=U[:, :256].astype(np.float32), gen_eig=lam, pca_eig=ev, pca_snr=snr_raw,
             noise_frac=np.trace(SN) / np.trace(SS))
    print(f"[basis] tr Σ_n / tr Σ_s = {np.trace(SN) / np.trace(SS):.3f} · 일반화 고윳값 상위 1/64/256: {lam[0]:.1f} / {lam[63]:.2f} / {lam[255]:.2f} · "
          f"PCA 상위 256 의 SNR 중앙값 {np.median(snr_raw[:256]):.2f}")


# ------------------------------------------------------------------------------------------------ extract
def extract(rank, world, setname, n, bs):
    torch, dev, bundles, model = _setup(rank)
    enc, enc32, gauss, pm1, smooth_from = _encoders(torch, dev, bundles)
    Bz = np.load(OUT / "basis.npz") if (OUT / "basis.npz").exists() else None
    P = {k: torch.from_numpy(Bz[k]).to(dev) for k in BASIS_KEYS if k in Bz.files} if Bz is not None else {}
    spaces = [s for s in SPACES_ALL if s in ("raw", "fp32", "smooth") or s in P or (s == "smooth_pm1" and setname == "ssv2")]
    if SPACES_ONLY:
        spaces = [s for s in spaces if s in SPACES_ONLY]
        P = {k: M for k, M in P.items() if k in spaces}
    need = set(spaces)
    print(f"[extract] encoder {ENC_NAME} · spaces {spaces}", flush=True)
    gen = torch.Generator(device=dev); gen.manual_seed(2000 + rank)
    rec = defaultdict(list); t0 = time.time()

    def views(h0, h32, hsm, hpm=None):
        out = {"raw": h0}
        if h32 is not None: out["fp32"] = h32
        if hsm is not None: out["smooth"] = hsm
        if hpm is not None: out["smooth_pm1"] = hpm
        for k, M in P.items(): out[k] = h0 @ M
        return out

    with torch.no_grad():
        if setname == "ssv2":
            files, load = _ssv2_loader(n, 2 * n)
            mine = files[rank::world]
            if LIMIT: mine = mine[:LIMIT]
            for b0 in range(0, len(mine), bs):
                xs = [v for v in (load(f) for f in mine[b0:b0 + bs]) if v is not None]
                if not xs: continue
                x = torch.stack(xs).to(dev).float()
                h0 = enc(x); h32 = enc32(x) if "fp32" in need else None
                hsm = smooth_from([enc(gauss(x, gen)) for _ in range(KN)]) if "smooth" in need else None
                hpm = smooth_from([enc(pm1(x, gen)) for _ in range(KN)]) if "smooth_pm1" in need else None
                V = views(h0, h32, hsm, hpm)
                for sp in spaces:
                    for b in range(x.size(0)): _ssv2_metrics(torch, V[sp][b].float(), rec, sp)
                if rank == 0 and (b0 // bs) % 5 == 0: print(f"[ssv2 r0] {b0 + len(xs)}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
        else:
            import yaml
            import design_headroom as H
            from evals.world_model_analysis.data import WMADataset
            cfg = yaml.safe_load(open(H.SETS["vobench"]["cfg"]))
            ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=model, features={"cache_dir": "/tmp"}, surprise={}))
            rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
            idx = {rec_.video_id: i for i, rec_ in enumerate(ds.records)}
            blocks = defaultdict(dict)
            for v, r in rows.items(): blocks[r["source_block"]][r["variant"]] = v
            blist = sorted(sb for sb, a in blocks.items() if all(k in a and a[k] in idx for k in ("pos_a", "pos_b", "imp_ab", "imp_ba")))
            mine = blist[rank::world]
            if LIMIT: mine = mine[:LIMIT]
            for j, sb in enumerate(mine):
                arms = blocks[sb]; order = ("pos_a", "imp_ab", "pos_b", "imp_ba")
                x = torch.stack([ds.clip(idx[arms[v]]) for v in order]).to(dev).float()
                h0 = enc(x); h32 = enc32(x) if "fp32" in need else None
                hsm = smooth_from([enc(gauss(x, gen)) for _ in range(KN)]) if "smooth" in need else None
                Vw = views(h0, h32, hsm)
                trk = {}
                for v in order:
                    t = H.track_vobench(rows[arms[v]])
                    trk[v] = None if t is None else (t, *H.plan_positions(*t))
                if True:
                    for sp in spaces:
                        X = Vw[sp].float()
                        for ci, v in enumerate(order):
                            _vob_clip(torch, X[ci], trk[v], rec, sp)
                        for (pv, iv) in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
                            dmap = (X[order.index(pv), C:] - X[order.index(iv), C:]).abs().mean(-1)      # (8, G, G)
                            om = _obj_mask(torch, trk[pv], dev)
                            rec[f"{sp}/pairshare"].append(float((dmap * om).sum() / dmap.sum()) if om.sum() > 0 else float("nan"))
                            rec[f"{sp}/pairarea"].append(float(om.mean()))
                for v in order: rec["video_id"].append(arms[v])
                for _ in ("pos_a", "pos_b"): rec["pair_block"].append(sb)
                if rank == 0 and j % 20 == 0: print(f"[vob r0] {j + 1}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{setname}_r{rank}.npz", **{k: np.array(v) for k, v in rec.items()})
    print(f"[{setname} r{rank}] done ({time.time() - t0:.0f}s)", flush=True)


def _tok(xy):
    return int(np.clip(xy[1] // 32.0, 0, G - 1)), int(np.clip(xy[0] // 32.0, 0, G - 1))


def _inb(xy):
    return 0 <= xy[0] < 512 and 0 <= xy[1] < 512


def _obj_mask(torch, tr, dev):
    m = torch.zeros(K, G, G, device=dev)
    if tr is None: return m
    (Pp, hid), pos, _, _ = tr
    for k in range(K):
        if not _inb(pos["move_or"][k]): continue
        cy, cx = _tok(pos["move_or"][k])
        m[k, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = 1
    return m


def _paste(frame, app, cy, cx):
    """app (3, 3, D) 를 frame (G, G, D) 의 (cy, cx) 중심 3×3 에 (화면 밖은 자른다)."""
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y, x_ = cy + dy, cx + dx
            if 0 <= y < G and 0 <= x_ < G: frame[y, x_] = app[dy + 1, dx + 1]


def _vob_clip(torch, X, tr, rec, sp):
    """X (16, G, G, D') 한 clip. render 별 미래 8 블록 L1: 전체 / 물체 자리 (참 미래 3×3) / 그 밖."""
    Fu = X[C:]
    R = {"copy": X[C - 1].expand(K, -1, -1, -1), "mean2": X[C - 2:C].mean(0).expand(K, -1, -1, -1),
         "mean4": X[C - 4:C].mean(0).expand(K, -1, -1, -1), "mean8": X[:C].mean(0).expand(K, -1, -1, -1)}
    mu = X[:C].mean((0, 1, 2))
    for n_, lam in SHR: R[n_] = (lam * X[C - 1] + (1 - lam) * mu).expand(K, -1, -1, -1)
    for n_, v in _histfree(torch, X[C - 1]).items(): R[n_] = v.expand(K, -1, -1, -1)
    if tr is None:
        for m in ("bg", "paste_or", "mc_or", "mc_cv", "mc_ca"): R[m] = R["mean8"]
    else:
        (Pp, hid), pos, last_vis, ctx_c = tr
        anyvis = [b for b in range(C) if hid[2 * b] < 0.5 or hid[2 * b + 1] < 0.5]
        fullvis = [b for b in range(C) if hid[2 * b] < 0.5 and hid[2 * b + 1] < 0.5]
        M = torch.ones(C, G, G, device=X.device)
        for b in anyvis:
            cy, cx = _tok(ctx_c[b]); M[b, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = 0
        cnt = M.sum(0)
        bg = (M.unsqueeze(-1) * X[:C]).sum(0) / cnt.clamp(min=1).unsqueeze(-1)
        bg = torch.where((cnt > 0).unsqueeze(-1), bg, X[:C].mean(0))
        R["bg"] = bg.expand(K, -1, -1, -1)
        apps = []
        for b in fullvis:
            cy, cx = _tok(ctx_c[b])
            if 1 <= cy <= G - 2 and 1 <= cx <= G - 2: apps.append(X[b, cy - 1:cy + 2, cx - 1:cx + 2])
        app = torch.stack(apps).mean(0) if apps else None
        app_last = None
        if last_vis is not None:
            cy, cx = _tok(ctx_c[last_vis])
            if 1 <= cy <= G - 2 and 1 <= cx <= G - 2: app_last = X[last_vis, cy - 1:cy + 2, cx - 1:cx + 2]
        canvas = torch.where((M[C - 1] > 0).unsqueeze(-1), X[C - 1], bg)
        fr = canvas.unsqueeze(0).repeat(K, 1, 1, 1)
        if app_last is not None:
            for k in range(K):
                if _inb(pos["move_or"][k]): _paste(fr[k], app_last, *_tok(pos["move_or"][k]))
        R["paste_or"] = fr
        for m in ("or", "cv", "ca"):
            fr = bg.unsqueeze(0).repeat(K, 1, 1, 1)
            if app is not None:
                for k in range(K):
                    p = pos[f"move_{m}"][k]
                    if _inb(p): _paste(fr[k], app, *_tok(p))
            R[f"mc_{m}"] = fr
    om = _obj_mask(torch, tr, X.device)
    for m in RENDERS:
        e = (R[m] - Fu).abs().mean(-1)                                       # (K, G, G)
        rec[f"{sp}/{m}/all"].append(e.mean((-1, -2)).cpu().numpy())
        if om.sum() > 0:
            rec[f"{sp}/{m}/obj"].append(float((e * om).sum() / om.sum())); rec[f"{sp}/{m}/rest"].append(float((e * (1 - om)).sum() / (1 - om).sum()))
        else:
            rec[f"{sp}/{m}/obj"].append(float("nan")); rec[f"{sp}/{m}/rest"].append(float(e.mean()))
    rec[f"{sp}/tchg"].append((X[1:C] - X[:C - 1]).abs().mean().item())
    rec[f"{sp}/xpos"].append((X[:C] - X[:C].roll(5, 1).roll(7, 2)).abs().mean().item())


# ------------------------------------------------------------------------------------------------ analyze
def _load(prefix):
    Z = defaultdict(list)
    for f in sorted(OUT.glob(f"{prefix}_r*.npz")):
        z = np.load(f, allow_pickle=True)
        for k in z.files: Z[k].append(z[k])
    return {k: np.concatenate([np.atleast_1d(a) if a.ndim == 0 else a for a in v]) for k, v in Z.items()}


def _boot(fn, groups, n=1000, seed=0):
    """묶음 (source_block) 단위 bootstrap 95 % 구간."""
    g = np.array(groups)
    if len(g) == 0: return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed); u = np.unique(g); where = {p: np.where(g == p)[0] for p in u}
    vals = []
    for _ in range(n):
        pick = rng.choice(u, len(u)); vals.append(fn(np.concatenate([where[p] for p in pick])))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def analyze():
    res = {}
    # ---- SSv2 (평가 clip 200..399, 다섯 공간) + 보정 clip (raw · smooth)
    for tag in ("calib", "ssv2"):
        Z = _load(tag)
        if not Z: continue
        out = {}
        for sp in SPACES_ALL:
            if f"{sp}/copy" not in Z: continue
            L = {m: Z[f"{sp}/{m}"].mean() for m in ("copy", "mean2", "mean4", "mean8", "shr50", "shr70", "shr85", "z50", "z70", "z85", "sp70", "nb70") if f"{sp}/{m}" in Z}
            Lk = {m: Z[f"{sp}/{m}"].mean(0).tolist() for m in ("copy", "mean8")}
            out[sp] = dict(L=L, adv2=1 - L["mean2"] / L["copy"], adv8=1 - L["mean8"] / L["copy"], L_by_k=Lk,
                           adv_shrink={m: 1 - L[m] / L["copy"] for m in ("shr50", "shr70", "shr85", "z50", "z70", "z85", "sp70", "nb70") if m in L},
                           tchg_over_xpos=float(Z[f"{sp}/tchg"].mean() / Z[f"{sp}/xpos"].mean()), n=int(len(Z[f"{sp}/copy"])))
        if tag == "calib":
            out["jitter_ratio"] = {k: float(np.mean(Z[f"jit_{k}"])) for k in ("g2", "pm1", "fp32", "smooth") if f"jit_{k}" in Z}
        res[f"ssv2_{tag}"] = out
    # ---- VOBench
    Z = _load("vobench")
    if Z:
        rows = {r["video_id"]: r for r in csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv"))}
        vids = list(Z["video_id"]); R = [rows[v] for v in vids]
        mot = np.array(["static" if r["motion"] == "flat_static" else "moving" for r in R])
        occ = np.array([r["occlusion"] for r in R]); var = np.array([r["variant"] for r in R])
        vio = np.array([r["violation"] for r in R]); blk = np.array([r["source_block"] for r in R])
        pos = np.isin(var, ("pos_a", "pos_b"))
        has_obj = np.array([r["obj_a" if r["variant"] in ("pos_a", "imp_ab") else "obj_b"] != "empty" for r in R])
        vob = {}
        for sp in SPACES_ALL:
            if f"{sp}/copy/all" not in Z: continue
            RS = [m for m in RENDERS if f"{sp}/{m}/all" in Z]
            A = {m: Z[f"{sp}/{m}/all"].mean(1) for m in RS}
            O = {m: Z[f"{sp}/{m}/obj"] for m in RS}
            d = {}
            # (a) 보상 균형: 가능 clip · 이동 · 물체 있음 · 가림 없음 (visible) / 전 가림
            for cell, sel in (("visible_moving", pos & has_obj & (mot == "moving") & (occ == "visible")),
                              ("all_moving", pos & has_obj & (mot == "moving")),
                              ("visible_static", pos & has_obj & (mot == "static") & (occ == "visible"))):
                if sel.sum() == 0: continue
                L = {m: float(A[m][sel].mean()) for m in RS}
                Lo = {m: float(np.nanmean(O[m][sel])) for m in RS}
                adv8 = 1 - L["mean8"] / L["copy"]; tg = 1 - L["mc_or"] / L["mean8"]; tgb = 1 - L["mc_or"] / L["bg"]
                d[cell] = dict(n=int(sel.sum()), L=L, L_obj=Lo, adv8=adv8, transport_gain=tg, transport_gain_vs_bg=tgb,
                               adv_shrink={m: 1 - L[m] / L["copy"] for m in ("shr50", "shr70", "shr85", "z50", "z70", "z85", "sp70", "nb70") if m in L},
                               transport_by_k=(1 - Z[f"{sp}/mc_or/all"][sel].mean(0) / Z[f"{sp}/mean8/all"][sel].mean(0)).tolist(),
                               adv8_by_k=(1 - Z[f"{sp}/mean8/all"][sel].mean(0) / Z[f"{sp}/copy/all"][sel].mean(0)).tolist(),
                               transport_over_avg=tg / adv8 if adv8 > 0 else float("nan"),
                               obj_transport_gain=1 - Lo["mc_or"] / Lo["mean8"],
                               adv8_ci=_boot(lambda ix: 1 - A["mean8"][sel][ix].mean() / A["copy"][sel][ix].mean(), blk[sel]),
                               tg_ci=_boot(lambda ix: 1 - A["mc_or"][sel][ix].mean() / A["mean8"][sel][ix].mean(), blk[sel]))
            # (b) matched 쌍 정확도 (render 가 문맥만으로 정해지므로 두 미래 중 가능 쪽에 더 가까운가)
            bv = {(rows[v]["source_block"], rows[v]["variant"]): i for i, v in enumerate(vids)}
            acc = defaultdict(lambda: defaultdict(list))
            for i, r in enumerate(R):
                if r["variant"] not in ("pos_a", "pos_b"): continue
                imp = {"pos_a": "imp_ab", "pos_b": "imp_ba"}[r["variant"]]
                j = bv.get((r["source_block"], imp))
                if j is None: continue
                dirn = "A" if r["variant"] == "pos_a" else "B"
                for m in RS:
                    hit = 1.0 if A[m][j] > A[m][i] else (0.5 if A[m][j] == A[m][i] else 0.0)
                    for key in ("all", f"{r['occlusion']}", f"{r['occlusion']}/{mot[i]}", f"{r['violation']}/{r['occlusion']}/{mot[i]}",
                                f"{r['violation']}/{r['occlusion']}/{mot[i]}/{dirn}"):
                        acc[key][m].append(hit)
            d["matched_acc"] = {k: {m: [float(np.mean(v)), len(v)] for m, v in dd.items()} for k, dd in acc.items()}
            # (c) 표적 국소성: 쌍 차이의 물체 3×3 몫 vs 면적
            sh = Z[f"{sp}/pairshare"]; ar = Z[f"{sp}/pairarea"]
            mot_of = {rows[v]["source_block"]: rows[v]["motion"] for v in vids}
            pb = Z["pair_block"]; pmot = np.array(["static" if mot_of[b] == "flat_static" else "moving" for b in pb])
            d["pair_obj_share"] = dict(moving=float(np.nanmean(sh[pmot == "moving"])), static=float(np.nanmean(sh[pmot == "static"])),
                                       area=float(np.nanmean(ar[ar > 0])))
            d["tchg_over_xpos"] = float(Z[f"{sp}/tchg"].mean() / Z[f"{sp}/xpos"].mean())
            vob[sp] = d
        res["vobench"] = vob
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        g = analyze_gravity()
        if g is not None: res["gravity"] = g
    except Exception as e:                                          # 중력 판이 없거나 깨져도 주 결과는 남긴다
        print(f"[gravity analyze skipped] {e}")
    json.dump(res, open(OUT / "result.json", "w"), indent=1, default=float)
    # ---- 출력
    for tag in ("ssv2_calib", "ssv2_ssv2"):
        if tag in res:
            print(f"\n[{tag}] 공간: adv2 / adv8 (= 1 − L(평균)/L(복사)) · 이웃 블록 변화 / 칸끼리 거리 · n")
            for sp, v in res[tag].items():
                if isinstance(v, dict) and "adv2" in v: print(f"  {sp:7s} {100 * v['adv2']:6.1f} / {100 * v['adv8']:6.1f} %  · 수축 " + " ".join(f"{k}:{100 * x:5.1f}" for k, x in v['adv_shrink'].items()) + f" · {v['tchg_over_xpos']:.3f} · {v['n']}")
            if "jitter_ratio" in res[tag]: print(f"  떨림 비 (같은 clip 두 판 차 / 다른 clip 같은 칸 차): {res[tag]['jitter_ratio']}")
    if "vobench" in res:
        for cell in ("visible_moving", "all_moving", "visible_static"):
            print(f"\n[VOBench {cell}] 공간: adv8 · 운반 이득 (1 − L(mc_or)/L(mean8)) · 비 · 물체 자리 운반 이득 · n")
            for sp, d in res["vobench"].items():
                if cell not in d: continue
                c = d[cell]
                print(f"  {sp:7s} adv8 {100 * c['adv8']:6.1f} % {c['adv8_ci']} · 운반 {100 * c['transport_gain']:6.2f} % {c['tg_ci']} · 비 {c['transport_over_avg']:.3f} · "
                      f"물체 자리 {100 * c['obj_transport_gain']:6.1f} % · 수축 " + " ".join(f"{k}:{100 * x:5.1f}" for k, x in c['adv_shrink'].items()) + f" · n {c['n']}")
                print("          운반 보상 걸음별 " + " ".join(f"{100 * x:5.2f}" for x in c['transport_by_k']) + " | 평균 보상 걸음별 " + " ".join(f"{100 * x:5.1f}" for x in c['adv8_by_k']))
        print("\n[VOBench matched 정확도] 공간 × render (copy · mean8 · shr70 · bg · paste_or · mc_or · mc_cv · mc_ca)")
        for key in ("all", "visible/moving", "late/moving", "permanence/late/moving", "shape/late/moving", "color/late/moving",
                    "permanence/late/moving/A", "permanence/visible/moving", "shape/visible/moving"):
            for sp, d in res["vobench"].items():
                if key in d["matched_acc"]:
                    a = d["matched_acc"][key]
                    print(f"  {key:28s} {sp:7s} " + " ".join(f"{100 * a[m][0]:5.1f}" for m in ("copy", "mean8", "shr70", "bg", "paste_or", "mc_or", "mc_cv", "mc_ca") if m in a) + f"  [{a['copy'][1]}]")
        print("\n[표적 국소성] 쌍 차이 중 물체 3×3 몫 (moving / static) · 면적")
        for sp, d in res["vobench"].items():
            p = d["pair_obj_share"]; print(f"  {sp:7s} {100 * p['moving']:5.1f} / {100 * p['static']:5.1f} % · 면적 {100 * p['area']:4.1f} % · 이웃 블록 변화/칸 거리 {d['tchg_over_xpos']:.3f}")


# ------------------------------------------------------------------------------------------------ E2b gravity (중력 세트)
GRAV_RENDERS = ("copy", "mean8", "shr70", "bg", "paste_or", "mc_or", "mc_cv", "mc_ca")


def _ctx_renders(torch, Xc, tr):
    """Xc (8, G, G, D') 문맥 블록. 미래 8 블록 render (K, G, G, D') — _vob_clip 과 같은 규칙."""
    R = {"copy": Xc[C - 1].expand(K, -1, -1, -1), "mean8": Xc.mean(0).expand(K, -1, -1, -1)}
    mu = Xc.mean((0, 1, 2)); R["shr70"] = (0.7 * Xc[C - 1] + 0.3 * mu).expand(K, -1, -1, -1)
    (Pp, hid), pos, last_vis, ctx_c = tr
    anyvis = [b for b in range(C) if hid[2 * b] < 0.5 or hid[2 * b + 1] < 0.5]
    fullvis = [b for b in range(C) if hid[2 * b] < 0.5 and hid[2 * b + 1] < 0.5]
    M = torch.ones(C, G, G, device=Xc.device)
    for b in anyvis:
        cy, cx = _tok(ctx_c[b]); M[b, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = 0
    cnt = M.sum(0)
    bg = (M.unsqueeze(-1) * Xc).sum(0) / cnt.clamp(min=1).unsqueeze(-1)
    bg = torch.where((cnt > 0).unsqueeze(-1), bg, Xc.mean(0))
    R["bg"] = bg.expand(K, -1, -1, -1)
    apps = []
    for b in fullvis:
        cy, cx = _tok(ctx_c[b])
        if 1 <= cy <= G - 2 and 1 <= cx <= G - 2: apps.append(Xc[b, cy - 1:cy + 2, cx - 1:cx + 2])
    app = torch.stack(apps).mean(0) if apps else None
    app_last = None
    if last_vis is not None:
        cy, cx = _tok(ctx_c[last_vis])
        if 1 <= cy <= G - 2 and 1 <= cx <= G - 2: app_last = Xc[last_vis, cy - 1:cy + 2, cx - 1:cx + 2]
    fr = torch.where((M[C - 1] > 0).unsqueeze(-1), Xc[C - 1], bg).unsqueeze(0).repeat(K, 1, 1, 1)
    if app_last is not None:
        for k in range(K):
            if _inb(pos["move_or"][k]): _paste(fr[k], app_last, *_tok(pos["move_or"][k]))
    R["paste_or"] = fr
    for m in ("or", "cv", "ca"):
        fr = bg.unsqueeze(0).repeat(K, 1, 1, 1)
        if app is not None:
            for k in range(K):
                q = pos[f"move_{m}"][k]
                if _inb(q): _paste(fr[k], app, *_tok(q))
        R[f"mc_{m}"] = fr
    return R


def extract_gravity(rank, world, setname, n, bs):
    """중력 세트 216 묶음 (선반 144 · 포물선 72). 문맥은 모든 후보 미래에 같다 → 문맥은 한 번, 미래 8 블록만 후보마다 인코딩."""
    import yaml
    import design_headroom as H
    from evals.world_model_analysis.data import WMADataset
    torch, dev, bundles, model = _setup(rank)
    enc, enc32, gauss, pm1, smooth_from = _encoders(torch, dev, bundles)
    Bz = np.load(OUT / "basis.npz")
    P = {k: torch.from_numpy(Bz[k]).to(dev) for k in ("snr64", "snr256", "pca256")}
    spaces = ("raw", "smooth", "snr64", "snr256", "pca256")
    gen = torch.Generator(device=dev); gen.manual_seed(3000 + rank)
    cfg = yaml.safe_load(open(H.SETS["gravity"]["cfg"]))
    ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=model, features={"cache_dir": "/tmp"}, surprise={}))
    rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["gravity"]["index"]))}
    idx = {rec_.video_id: i for i, rec_ in enumerate(ds.records)}
    blocks = defaultdict(dict)
    for v, r in rows.items(): blocks[r["source_block"]][r["variant"]] = v
    blist = sorted(sb for sb, a in blocks.items() if "pos" in a and all(v in idx for v in a.values()))
    mine = blist[rank::world]
    if LIMIT: mine = mine[:LIMIT]
    out = []; t0 = time.time()
    with torch.no_grad():
        for j, sb in enumerate(mine):
            arms = sorted(blocks[sb]); x = torch.stack([ds.clip(idx[blocks[sb][a]]) for a in arms]).to(dev).float()
            ctx0 = enc(x[:1, :, :2 * C]); futs = enc(x[:, :, 2 * C:])
            ctxs = smooth_from([enc(gauss(x[:1, :, :2 * C], gen)) for _ in range(KN)])
            futs_s = smooth_from([enc(gauss(x[:, :, 2 * C:], gen)) for _ in range(KN)])
            tr0 = H.track_gravity(rows[blocks[sb]["pos"]]); tr = (tr0, *H.plan_positions(*tr0))
            rec_b = {"sb": sb, "arms": arms, "law": rows[blocks[sb]["pos"]]["law"], "cell": rows[blocks[sb]["pos"]]["cell"]}
            for sp in spaces:
                if sp == "raw": Xc, Fa = ctx0[0], futs
                elif sp == "smooth": Xc, Fa = ctxs[0], futs_s
                else: Xc, Fa = ctx0[0] @ P[sp], futs @ P[sp]
                R = _ctx_renders(torch, Xc, tr)
                for m in GRAV_RENDERS:
                    rec_b[f"{sp}/{m}"] = (R[m].unsqueeze(0) - Fa).abs().mean(-1).mean((-1, -2, -3)).cpu().numpy()   # (n_arms,)
            out.append(rec_b)
            if rank == 0 and j % 20 == 0: print(f"[grav r0] {j + 1}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"gravity_r{rank}.npy", np.array(out, dtype=object), allow_pickle=True)
    print(f"[gravity r{rank}] done ({time.time() - t0:.0f}s)", flush=True)


def analyze_gravity():
    recs = [r for f in sorted(OUT.glob("gravity_r*.npy")) for r in np.load(f, allow_pickle=True)]
    if not recs: return None
    res = {}
    for sp in ("raw", "smooth", "snr64", "snr256", "pca256"):
        d = {}
        for cell in sorted({("ledge" if r["law"] == "ledge" else f"arc/{r['cell']}") for r in recs}):
            sel = [r for r in recs if ("ledge" if r["law"] == "ledge" else f"arc/{r['cell']}") == cell]
            dd = {}
            for m in GRAV_RENDERS:
                pick = [r["arms"][int(np.argmin(r[f"{sp}/{m}"]))] for r in sel]
                acc = float(np.mean([p_ == "pos" for p_ in pick]))
                dd[m] = dict(choice_acc=acc, n=len(sel), picks=dict(Counter(pick)))
                if cell == "ledge":
                    dd[m]["pair_acc"] = float(np.mean([r[f"{sp}/{m}"][r["arms"].index("imp_float")] > r[f"{sp}/{m}"][r["arms"].index("pos")] for r in sel]))
            d[cell] = dd
        res[sp] = d
    print("\n[중력 세트] 고른 미래 = 진짜 (정답 수 / 묶음) — copy · mean8 · shr70 · bg · paste_or · mc_or · mc_cv · mc_ca")
    for sp, d in res.items():
        for cell, dd in d.items():
            print(f"  {sp:7s} {cell:10s} " + " ".join(f"{round(dd[m]['choice_acc'] * dd[m]['n']):3d}" for m in GRAV_RENDERS) + f"  / {dd['copy']['n']}"
                  + ("  | 선반 쌍 " + " ".join(f"{100 * dd[m]['pair_acc']:5.1f}" for m in GRAV_RENDERS) if cell == "ledge" else ""))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["calib", "basis", "extract", "analyze"])
    ap.add_argument("--set", default="vobench", choices=["vobench", "ssv2", "gravity"])
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--gpus", type=int, default=1); ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    if a.phase in ("calib", "extract"):
        import torch
        fn = calib if a.phase == "calib" else (extract_gravity if a.set == "gravity" else extract)
        args = (a.gpus, a.n, a.bs) if a.phase == "calib" else (a.gpus, a.set, a.n, a.bs)
        if a.gpus == 1: fn(0, *args)
        else: torch.multiprocessing.spawn(fn, args=args, nprocs=a.gpus, join=True)
    elif a.phase == "basis":
        basis()
    else:
        analyze()
