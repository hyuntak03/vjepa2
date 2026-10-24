#!/usr/bin/env python3
"""[S] 곧음 측정 보충 (학습 없음): 영상 모드 창 토큰이 이웃 자리 사이에 블록 토큰보다 크게 바뀌는 이유 — "자리마다 공통 성분" 인가.

  CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/straightening_posmean.py

창 = 16 장 (straightening.py 와 같은 읽기). 공간: blk (블록마다 목표 encoder) · twin · owin (16 장 한 번) · tpre (접두 창 마지막 자리, 목표 encoder).
잴 것:
  (1) 자리 공통 성분 μ_t = 모든 창 · 칸 평균 (자리 t 별, 1280). 이웃 자리 변화 에너지 중 μ_t − μ_{t−1} 의 몫 = |Δμ|² / E|v|²
  (2) 창마다 화면 평균을 자리별로 뺀 토큰 (z_t(c) − mean_c z_t(c)) 과 전역 μ_t 를 뺀 토큰의 곡률 · 이웃 변화 크기 (정지 · 움직이는 칸)
  (3) 같은 칸 이웃 자리 cos (정지 칸) — 원래 · 화면 평균 뺀 것
자연 = K710 보류 40 + HT100M 40 (흐름 캐시로 칸 무리), 3D 합성 = 검증 40 clip (먼 배경 칸)."""
import json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord

REPO = "/workspace/world_model/vjepa2"; CACHE = "/workspace/auto_research_cache"
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
torch.set_num_threads(8); dev = torch.device("cuda")
S, G, D = 256, 16, 1280
SP = ("blk", "twin", "owin", "tpre")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
sd = _load_checkpoint(CK)


def enc(key, nf):
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=nf, use_rope=True, uniform_power=False)
    _load_state_dict(e, sd[key], tag=key[:1]); return e.to(dev, torch.bfloat16).eval()
T2, T16, O16 = enc("target_encoder", 2), enc("target_encoder", 16), enc("encoder", 16)
del sd
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
ln = lambda t: F.layer_norm(t.float(), (D,))


@torch.no_grad()
def spaces(x):
    xb = x.reshape(3, 8, 2, 256, 256).permute(1, 0, 2, 3, 4).to(torch.bfloat16)
    hb = ln(T2(xb)); tp = [hb[0]]
    for t in range(1, 8):
        tp.append(ln(T16(x[None, :, : 2 * t + 2].to(torch.bfloat16)))[0].view(t + 1, S, D)[t])
    xx = x[None].to(torch.bfloat16)
    return dict(blk=hb, twin=ln(T16(xx))[0].view(8, S, D), owin=ln(O16(xx))[0].view(8, S, D), tpre=torch.stack(tp))


def decode(path, start, step):
    vr = decord.VideoReader(path, num_threads=1)
    idx = [start + step * k for k in range(16)]
    if idx[-1] >= len(vr):
        return None
    fr = vr.get_batch(idx).asnumpy()
    Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
    xx = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
    xx = F.interpolate(xx, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round() / 255.0
    return (xx.permute(1, 0, 2, 3) - MEAN) / STD


items = []                                                                            # (dom, x-getter, masks (8,256) dict)
mk = json.load(open(f"{CACHE}/natural_holdout/k710_320_holdout_meta.json")); Uk = np.load(f"{CACHE}/natural_holdout/k710_320_holdout_U.npy").astype(np.float32)
mh = json.load(open(f"{CACHE}/natural_train/ht100m_1pct_meta.json")); Uh = np.load(f"{CACHE}/natural_train/ht100m_1pct_Uflow.npy").astype(np.float32)
nat = [(m, Uk[i]) for i, m in enumerate(mk[:40])] + [(m, Uh[i]) for i, m in enumerate(mh[:40])]
TOK = {d: {s: [] for s in SP} for d in ("natural", "phys3d")}; MSK = {"natural": [], "phys3d": []}
t0 = time.time()
for m, U in nat:
    x = decode(m["path"], m["start"], m["fstp"])
    if x is None:
        continue
    mg = np.linalg.norm(U.reshape(8, S, 2), axis=-1)
    sta = np.zeros((8, S), bool); mov = np.zeros((8, S), bool)
    for t in range(1, 7):
        sta[t] = (mg[t] < 0.1) & (mg[t + 1] < 0.1); mov[t] = (mg[t] >= 0.5) & (mg[t + 1] >= 0.5)
    sp = spaces(x)
    for s in SP:
        TOK["natural"][s].append(sp[s].half().cpu())
    MSK["natural"].append((sta, mov))
print(f"자연 {len(MSK['natural'])} ({time.time() - t0:.0f}s)", flush=True)
PX = np.load(f"{CACHE}/phys3d/pix_val_s2.npy", mmap_mode="r"); LB = np.load(f"{CACHE}/phys3d/lab_val_s2.npz")
cy, cx = np.divmod(np.arange(S), G); ccx, ccy = (cx + 0.5) * 16, (cy + 0.5) * 16
for i in range(40):
    fr = torch.from_numpy(np.asarray(PX[i, 0:16])).to(dev).permute(3, 0, 1, 2).float() / 255.0
    x = (fr - MEAN) / STD
    bb = LB["bbox"][i, [2 * b + 1 for b in range(8)]]
    far = np.zeros((8, S), bool)
    for t in range(1, 7):
        o = np.zeros(S, bool)
        for b in (t - 1, t, t + 1):
            o |= (ccx >= bb[b, 0] - 32) & (ccx <= bb[b, 2] + 32) & (ccy >= bb[b, 1] - 32) & (ccy <= bb[b, 3] + 32)
        far[t] = ~o
    sp = spaces(x)
    for s in SP:
        TOK["phys3d"][s].append(sp[s].half().cpu())
    MSK["phys3d"].append((far, np.zeros((8, S), bool)))
print(f"3D 합성 40 ({time.time() - t0:.0f}s)", flush=True)

out = {}
for dom in ("natural", "phys3d"):
    for s in SP:
        Z = torch.stack(TOK[dom][s]).float()                                          # (W, 8, 256, D)
        mu = Z.mean((0, 2))                                                           # (8, D)
        v = Z[:, 1:] - Z[:, :-1]
        dmu = mu[1:] - mu[:-1]
        frac = float((dmu ** 2).sum() / (v ** 2).sum((-1)).mean((0, 2)).sum())     # 전역 공통 성분의 변화 에너지 몫 (자리 1..7 합)
        scr = Z.mean(2, keepdim=True)                                                 # 창 · 자리별 화면 평균
        fr_scr = float(((scr[:, 1:] - scr[:, :-1]) ** 2).sum() / ((v ** 2).sum(-1).mean(2)).sum())
        res = dict(frac_global_pos=round(frac, 4), frac_screen=round(fr_scr, 4), mu_norm_rel=round(float(mu.norm(dim=-1).mean() / Z.norm(dim=-1).mean()), 4))
        for vn, ZZ in (("raw", Z), ("minus_global_mu", Z - mu[None, :, None]), ("minus_screen", Z - scr)):
            for gi, gname in enumerate(("sta", "mov")):
                th = cs = st = n = 0.0
                adj = 0.0
                for w in range(ZZ.shape[0]):
                    msk = torch.from_numpy(MSK[dom][w][gi])
                    for t in range(2, 6):
                        m = msk[t]
                        if m.sum() == 0:
                            continue
                        v1, v2 = ZZ[w, t, m] - ZZ[w, t - 1, m], ZZ[w, t + 1, m] - ZZ[w, t, m]
                        c = F.cosine_similarity(v1, v2, dim=-1)
                        th += float(torch.rad2deg(torch.arccos(c.clamp(-1, 1))).sum()); cs += float(c.sum())
                        st += float((v1.norm(dim=-1) / ZZ[w, t, m].norm(dim=-1)).sum())
                        adj += float(F.cosine_similarity(ZZ[w, t, m], ZZ[w, t - 1, m], dim=-1).sum()); n += float(m.sum())
                if n:
                    res[f"{vn}|{gname}"] = dict(theta=round(th / n, 2), cos=round(cs / n, 4), step=round(st / n, 4), adj_cos=round(adj / n, 4), n=int(n))
        out[f"{dom}|{s}"] = res
        print(dom, s, json.dumps(res, ensure_ascii=False), flush=True)
json.dump(out, open(os.path.join(REPO, "auto_research/results/straightening_posmean.json"), "w"), indent=1)
print("끝", flush=True)
