#!/usr/bin/env python3
"""[물리] 같은 칸 예측 가능성 행렬판 (사전 등록 KST 04:57, 학습 없는 ridge).

  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/samecell_ridge.py [--limit N]

표현 셋 (samecell_predictability.py 와 같다): blk · win · onl. 창마다 칸별 지도 3 장 (마지막 문맥 자리 · 문맥 7 자리 평균 · 정답) 을 표현마다 보관.
판 (1) 같은 칸: [PCA(마지막)(c), PCA(평균)(c)] → 정답 (c) 1280.
판 (2) 3×3 이웃: [PCA(마지막)(c 둘레 9 칸, 화면 밖 0), PCA(평균)(c)] → 정답 (c).
PCA = 표현마다 주성분 384 (맞춤 창 64 개의 마지막 · 평균 토큰으로). ridge 세기 α · tr(XᵀX)/d, α ∈ {1e-3 … 10} 를 맞춤 창 안 반 나누기로 고르고 맞춤 창 전체로 다시 맞춤.
맞춤 = 짝수 창, 평가 = 홀수 창. R² = 1 − SSE / Σ 정답². 정지 칸은 창 쪽당 최대 20,000 칸."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--pca", type=int, default=0); a = ap.parse_args()
torch.set_num_threads(8); dev = torch.device("cuda"); rng = np.random.default_rng(0)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
sd = _load_checkpoint(CK)
def enc(key, nf):
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=nf, use_rope=True, uniform_power=False)
    _load_state_dict(e, sd[key], tag=key[:1]); return e.to(dev, torch.bfloat16).eval()
T2, TW, ON = enc("target_encoder", 2), enc("target_encoder", 16), enc("encoder", 16)
del sd
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
S, G, NS = 256, 16, 7
REPS = ("blk", "win", "onl")
cy, cx = np.divmod(np.arange(S), G)
NB9 = np.array([[(r + dr) * G + (c + dc) if 0 <= r + dr < G and 0 <= c + dc < G else -1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)] for r, c in zip(cy, cx)])   # (256, 9)


@torch.no_grad()
def tokens(x):
    xb = x.reshape(3, 8, 2, 256, 256).permute(1, 0, 2, 3, 4).to(torch.bfloat16)
    hb = F.layer_norm(T2(xb).float(), (1280,))
    w0 = F.layer_norm(TW(x[None, :, :14].to(torch.bfloat16)).float()[0], (1280,)).view(NS, S, -1)
    w1 = F.layer_norm(TW(x[None, :, 2:16].to(torch.bfloat16)).float()[0], (1280,)).view(NS, S, -1)
    o0 = F.layer_norm(ON(x[None, :, :14].to(torch.bfloat16)).float()[0], (1280,)).view(NS, S, -1)
    out = {}
    for r, (X, y) in {"blk": (hb[:7], hb[7]), "win": (w0, w1[6]), "onl": (o0, hb[7])}.items():
        out[r] = torch.stack([X[-1], X.mean(0), y]).half().cpu()          # (3, 256, D): 마지막 · 평균 · 정답
    return out


def decode(path, start, step, n=16):
    vr = decord.VideoReader(path, num_threads=1)
    idx = [start + step * k for k in range(n)]
    if idx[-1] >= len(vr):
        return None
    fr = vr.get_batch(idx).asnumpy()
    Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
    x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(3, 0, 1, 2).float()
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1).clamp(0, 255).round() / 255.0
    return (x - MEAN) / STD


W = {"natural": [], "phys3d": []}                                    # 창마다 dict(maps, enter, static)
t0 = time.time()
nat = []
mk = json.load(open("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_meta.json"))
Uk = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy").astype(np.float32)
nat += [(m["path"], m["start"], m["fstp"], Uk[i]) for i, m in enumerate(mk)]
mh = json.load(open("/workspace/auto_research_cache/enc_noise/ht100m_target_encoder_s0_meta.json")) + json.load(open("/workspace/auto_research_cache/enc_noise/ht100m_target_encoder_s1_meta.json"))
Uh = np.load(os.path.join(REPO, "auto_research/results/natural_transport_ctx5/rows_ht100m.npz"))["Uwin"].astype(np.float32)
nat += [(m["path"], m["start"], m["stride"], Uh[i]) for i, m in enumerate(mh)]
if a.limit:
    nat = nat[: a.limit] + nat[len(mk): len(mk) + a.limit]
for i, (path, st, step, U) in enumerate(nat):
    x = decode(path, st, step)
    if x is None:
        continue
    u7, u6 = np.linalg.norm(U[7].reshape(S, 2), axis=1), np.linalg.norm(U[6].reshape(S, 2), axis=1)
    W["natural"].append(dict(maps=tokens(x), enter=np.where(u7 >= 0.5)[0], static=np.where((u7 < 0.1) & (u6 < 0.1))[0]))
print(f"자연 창 {len(W['natural'])} ({time.time() - t0:.0f}s)", flush=True)
ccx, ccy = (cx + 0.5) * 16, (cy + 0.5) * 16
for sh in range(4):
    PX = np.load(f"/workspace/auto_research_cache/phys3d/pix_val_s{sh}.npy", mmap_mode="r"); LB = np.load(f"/workspace/auto_research_cache/phys3d/lab_val_s{sh}.npz")
    n = PX.shape[0] if not a.limit else min(PX.shape[0], a.limit // 4 + 1)
    for i in range(n):
        for tp1 in (7, 9, 11, 13, 15):
            b0 = tp1 - 7; f1, f0 = 2 * tp1 + 1, 2 * (tp1 - 1) + 1
            if LB["hidden"][i, f1]:
                continue
            def inbox(f, pad=0.0):
                x0, y0, x1, y1 = LB["bbox"][i, f]
                return (ccx >= x0 - pad) & (ccx <= x1 + pad) & (ccy >= y0 - pad) & (ccy <= y1 + pad)
            enter = np.where(inbox(f1) & ~inbox(f0))[0]; static = np.where(~inbox(f1, 32) & ~inbox(f0, 32))[0]
            if len(enter) == 0:
                continue
            fr = torch.from_numpy(np.asarray(PX[i, 2 * b0: 2 * b0 + 16])).to(dev).permute(3, 0, 1, 2).float() / 255.0
            W["phys3d"].append(dict(maps=tokens((fr - MEAN) / STD), enter=enter, static=static))
    print(f"3D shard {sh} · 창 {len(W['phys3d'])} ({time.time() - t0:.0f}s)", flush=True)


def pca_basis(wins, r, k):
    if k <= 0:                                                                   # 줄이지 않음 (결과 전 변경: 주성분 절단이 같은 기저 표현을 깎는 것을 스모크에서 봄)
        return torch.zeros(1280, device=dev), torch.eye(1280, device=dev)
    idx = rng.choice(len(wins), size=min(64, len(wins)), replace=False)
    Z = torch.cat([wins[j]["maps"][r][:2].reshape(-1, 1280).float() for j in idx]).to(dev)
    mu = Z.mean(0); C = (Z - mu).T @ (Z - mu) / len(Z)
    ev, V = torch.linalg.eigh(C)
    return mu, V[:, -k:].flip(1)


def feats(wins, r, group, mode, mu, V, cap=None):
    Xs, Ys = [], []
    for w in wins:
        cells = w[group]
        if cap is not None and len(cells) > cap:
            cells = rng.choice(cells, cap, replace=False)
        if len(cells) == 0:
            continue
        m = w["maps"][r].float().to(dev)                                       # (3,256,D)
        pl, pm = (m[0] - mu) @ V, (m[1] - mu) @ V                              # (256,k)
        if mode == 1:
            X = torch.cat([pl[cells], pm[cells]], 1)
        else:
            nb = torch.from_numpy(NB9[cells]).to(dev)
            pl0 = torch.cat([pl, torch.zeros(1, pl.shape[1], device=dev)])        # 화면 밖 = 0 (색인 −1 → 마지막 줄)
            X = torch.cat([pl0[nb].reshape(len(cells), -1), pm[cells]], 1)
        Xs.append(X); Ys.append(m[2][cells])
    return torch.cat(Xs), torch.cat(Ys)


def base_cols(X, mode):
    """스칼라 판 (같은 칸 마지막 · 평균, 계수 2 개) 에 쓰는 열: 판 1 = [0:1280] · [1280:2560], 판 2 = 가운데 칸 [4·1280:5·1280] · 마지막 1280"""
    return (X[:, :1280], X[:, 1280:2560]) if mode == 1 else (X[:, 4 * 1280:5 * 1280], X[:, -1280:])


def ridge(X, Y, lam, mode):
    """스칼라 판을 먼저 맞추고 (계수 2 개, 차원 공유), 그 잔차를 ridge 로 — ridge 가 0 으로 줄면 스칼라 판과 같다 (스칼라 판을 품는 꼴)."""
    a_, b_ = base_cols(X, mode)
    Gm = torch.stack([torch.stack([(a_ * a_).sum(), (a_ * b_).sum()]), torch.stack([(a_ * b_).sum(), (b_ * b_).sum()])]).double()
    w = torch.linalg.solve(Gm, torch.stack([(a_ * Y).sum(), (b_ * Y).sum()]).double()).float()
    R = Y - w[0] * a_ - w[1] * b_
    A = X.T @ X; A.diagonal().add_(lam)
    return w, torch.linalg.solve(A, X.T @ R)


def r2(X, Y, Wt, mode):
    w, M = Wt; a_, b_ = base_cols(X, mode)
    return float(1 - ((Y - w[0] * a_ - w[1] * b_ - X @ M) ** 2).sum() / (Y ** 2).sum())


out = {}
for ds, wins in W.items():
    tr = [w for k, w in enumerate(wins) if k % 2 == 0]; te = [w for k, w in enumerate(wins) if k % 2 == 1]
    tr_a, tr_b = tr[0::2], tr[1::2]
    for r in REPS:
        mu, V = pca_basis(tr, r, a.pca)
        for mode in (1, 2):
            for group in ("enter", "static"):
                cap = 20000 // max(1, len(tr) // 2) if group == "static" else None
                Xa, Ya = feats(tr_a, r, group, mode, mu, V, cap); Xb, Yb = feats(tr_b, r, group, mode, mu, V, cap)
                sc = float(torch.trace(Xa.T @ Xa) / Xa.shape[1])
                best = max((r2(Xb, Yb, ridge(Xa, Ya, al * sc, mode), mode), al) for al in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4))
                Xt, Yt = torch.cat([Xa, Xb]), torch.cat([Ya, Yb])
                Xe, Ye = feats(te, r, group, mode, mu, V, cap)
                Wt = ridge(Xt, Yt, best[1] * float(torch.trace(Xt.T @ Xt) / Xt.shape[1]), mode)
                Ws = (Wt[0], torch.zeros_like(Wt[1]))                                    # 스칼라 판만 (같은 칸, 계수 2 개)
                out[f"{ds}|{group}|{r}|m{mode}"] = dict(R2=round(r2(Xe, Ye, Wt, mode), 4), R2_scalar2=round(r2(Xe, Ye, Ws, mode), 4),
                                                       R2_train=round(r2(Xt, Yt, Wt, mode), 4), alpha=best[1],
                                                       n_train=int(len(Yt)), n_test=int(len(Ye)), d=int(Xt.shape[1]))
                print(ds, group, r, f"판{mode}", out[f"{ds}|{group}|{r}|m{mode}"], flush=True)
json.dump(out, open(os.path.join(REPO, "auto_research/results/samecell_ridge.json"), "w"), indent=1)
print(f"끝 {time.time() - t0:.0f}s")
