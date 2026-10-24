#!/usr/bin/env python3
"""[물리] 같은 칸 예측 가능성 (사전 등록 KST 04:44, 학습 없음): 물체가 들어오는 칸에서 다음 정답을 같은 칸 문맥 토큰 7 개로 선형 회귀 (옮기기 없음).

  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/samecell_predictability.py [--limit N]

창 = 블록 8 개 (16 장). 표현 셋:
  blk : 문맥 = 블록 0–6 블록 단위 목표 encoder 토큰 (LN), 정답 = 블록 7 블록 토큰
  win : 문맥 = 장 0–13 목표 encoder 한 번 (LN) 7 자리, 정답 = 장 2–15 창의 마지막 자리 (리더 명세)
  onl : 문맥 = 장 0–13 online encoder 한 번 (LN) 7 자리, 정답 = 블록 7 블록 토큰 (영상 문맥 모델의 실제 입력 꼴)
칸 무리: enter (들어오는 칸) · static (정지 칸). 자연 = 흐름 (블록 7→6 ≥ 0.5 칸 · 정지 = 두 걸음 < 0.1 칸), 3D = 라벨 bbox.
회귀: y ≈ Σ_k w_k x_k (자리마다 스칼라, 차원 공유), 충분통계를 창마다 모아 짝수 창으로 맞추고 홀수 창으로 R² = 1 − SSE / Σy².
  last-only: y ≈ w x_6 (마지막 자리 하나) 도 같이."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
torch.set_num_threads(8); dev = torch.device("cuda")
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


@torch.no_grad()
def tokens(x):
    """x (3, 16, 256, 256) 정규화된 장 → dict rep → (X (7,256,D), y (256,D))"""
    xb = x.reshape(3, 8, 2, 256, 256).permute(1, 0, 2, 3, 4).to(torch.bfloat16)      # (8,3,2,H,W)
    hb = F.layer_norm(T2(xb).float(), (1280,))                                       # (8,256,D)
    w0 = F.layer_norm(TW(x[None, :, :14].to(torch.bfloat16)).float()[0], (1280,)).view(NS, S, -1)
    w1 = F.layer_norm(TW(x[None, :, 2:16].to(torch.bfloat16)).float()[0], (1280,)).view(NS, S, -1)
    o0 = F.layer_norm(ON(x[None, :, :14].to(torch.bfloat16)).float()[0], (1280,)).view(NS, S, -1)
    return {"blk": (hb[:7], hb[7]), "win": (w0, w1[6]), "onl": (o0, hb[7])}


class Acc:
    def __init__(self):
        self.s = {}
    def add(self, key, split, X, y):                                                  # X (7, n, D), y (n, D)
        if y.shape[0] == 0:
            return
        Gm = torch.einsum("knd,lnd->kl", X, X).double().cpu().numpy(); b = torch.einsum("knd,nd->k", X, y).double().cpu().numpy()
        yy = float((y.double() ** 2).sum()); n = int(y.shape[0])
        d = self.s.setdefault((key, split), dict(G=np.zeros((NS, NS)), b=np.zeros(NS), yy=0.0, n=0))
        d["G"] += Gm; d["b"] += b; d["yy"] += yy; d["n"] += n
    def report(self, key):
        tr, te = self.s.get((key, 0)), self.s.get((key, 1))
        if tr is None or te is None:
            return None
        w = np.linalg.solve(tr["G"] + 1e-6 * np.eye(NS), tr["b"])
        sse = te["yy"] - 2 * w @ te["b"] + w @ te["G"] @ w
        wl = tr["b"][-1] / tr["G"][-1, -1]
        sse_l = te["yy"] - 2 * wl * te["b"][-1] + wl * wl * te["G"][-1, -1]
        return dict(R2=round(float(1 - sse / te["yy"]), 4), R2_last=round(float(1 - sse_l / te["yy"]), 4), w=[round(float(v), 3) for v in w],
                    w_last=round(float(wl), 3), n_cells_test=te["n"], n_cells_train=tr["n"])


def decode(path, start, step, n=16):
    vr = decord.VideoReader(path, num_threads=1)
    idx = [start + step * k for k in range(n)]
    if idx[-1] >= len(vr):
        return None
    fr = vr.get_batch(idx).asnumpy()
    Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
    x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(3, 0, 1, 2).float()       # (3,T,H,W)
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1).clamp(0, 255).round() / 255.0
    return (x - MEAN) / STD


acc = Acc(); t0 = time.time()
# ---- 자연 영상: K710 보류 (흐름 캐시) · HT100M ([E] 캐시 메타 + natural_transport_ctx5 흐름)
nat = []
mk = json.load(open("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_meta.json"))
Uk = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy").astype(np.float32)
nat += [("k710", m["path"], m["start"], m["fstp"], Uk[i]) for i, m in enumerate(mk)]
mh = json.load(open("/workspace/auto_research_cache/enc_noise/ht100m_target_encoder_s0_meta.json")) + json.load(open("/workspace/auto_research_cache/enc_noise/ht100m_target_encoder_s1_meta.json"))
Uh = np.load(os.path.join(REPO, "auto_research/results/natural_transport_ctx5/rows_ht100m.npz"))["Uwin"].astype(np.float32)
nat += [("ht100m", m["path"], m["start"], m["stride"], Uh[i]) for i, m in enumerate(mh)]
if a.limit:
    nat = nat[: a.limit] + nat[len(mk): len(mk) + a.limit]
for wi, (src, path, st, step, U) in enumerate(nat):
    x = decode(path, st, step)
    if x is None:
        continue
    toks = tokens(x)
    u7, u6 = np.linalg.norm(U[7].reshape(S, 2), axis=1), np.linalg.norm(U[6].reshape(S, 2), axis=1)
    groups = {"enter": np.where(u7 >= 0.5)[0], "static": np.where((u7 < 0.1) & (u6 < 0.1))[0]}
    for gname, cells in groups.items():
        ci = torch.from_numpy(cells).to(dev)
        for r in REPS:
            X, y = toks[r]
            for sname in (src, "natural"):
                acc.add((sname, gname, r), wi % 2, X[:, ci], y[ci])
    if wi % 50 == 0:
        print(f"자연 {wi}/{len(nat)} ({time.time() - t0:.0f}s)", flush=True)
# ---- 3D 합성: phys3d 검증 캐시 (장 32, 라벨 장마다), 정답 블록 t+1 ∈ {7, 11, 15}
cy, cx = np.divmod(np.arange(S), G); ccx, ccy = (cx + 0.5) * 16, (cy + 0.5) * 16
wi = 0
for sh in range(4):
    PX = np.load(f"/workspace/auto_research_cache/phys3d/pix_val_s{sh}.npy", mmap_mode="r"); LB = np.load(f"/workspace/auto_research_cache/phys3d/lab_val_s{sh}.npz")
    n = PX.shape[0] if not a.limit else min(PX.shape[0], a.limit // 4 + 1)
    for i in range(n):
        for tp1 in (7, 11, 15):
            b0 = tp1 - 7
            fr = torch.from_numpy(np.asarray(PX[i, 2 * b0: 2 * b0 + 16])).to(dev).permute(3, 0, 1, 2).float() / 255.0
            x = (fr - MEAN) / STD
            f1, f0 = 2 * tp1 + 1, 2 * (tp1 - 1) + 1                                  # 블록 대표 장 (둘째 장)
            if LB["hidden"][i, f1]:
                continue
            def inbox(f, pad=0.0):
                x0, y0, x1, y1 = LB["bbox"][i, f]
                return (ccx >= x0 - pad) & (ccx <= x1 + pad) & (ccy >= y0 - pad) & (ccy <= y1 + pad)
            enter = np.where(inbox(f1) & ~inbox(f0))[0]
            static = np.where(~inbox(f1, 32) & ~inbox(f0, 32))[0]
            toks = tokens(x)
            for gname, cells in (("enter", enter), ("static", static)):
                ci = torch.from_numpy(cells).to(dev)
                for r in REPS:
                    X, y = toks[r]
                    acc.add(("phys3d", gname, r), wi % 2, X[:, ci], y[ci])
            wi += 1
    print(f"3D 합성 shard {sh} · 창 {wi} ({time.time() - t0:.0f}s)", flush=True)
out = {}
for src in ("natural", "k710", "ht100m", "phys3d"):
    for g in ("enter", "static"):
        for r in REPS:
            rep = acc.report((src, g, r))
            if rep:
                out[f"{src}|{g}|{r}"] = rep
                print(f"{src:8s} {g:7s} {r:4s} R² {rep['R2']:.3f} (마지막 하나 {rep['R2_last']:.3f}) · 계수 {rep['w']} · 칸 {rep['n_cells_test']}", flush=True)
json.dump(out, open(os.path.join(REPO, "auto_research/results/samecell_predictability.json"), "w"), indent=1)
print(f"끝 {time.time() - t0:.0f}s")
