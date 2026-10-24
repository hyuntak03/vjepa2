#!/usr/bin/env python3
"""[E] 잡음 가설 — 영상 모드 창 토큰은 블록 토큰보다 블록마다 새 nuisance 가 작은가 (auto_research, 2026-10-22). 학습 없음.

  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/vid_noise.py --n 48

소스: K710 · HT100M (enc_noise_collect 의 창 메타로 원본을 다시 읽음, 학습 간격 · 같은 전처리) · 3D 합성 (캐시 픽셀 32 장).
표본마다 24 장 F. 밀기 s = 0..4: 창 W_s = F[2s : 2s+16] (16 장) 을 한 번에 인코딩 → 마지막 tubelet 자리 토큰 z_s (장 2s+14, 2s+15).
같은 두 장을 블록으로 인코딩 → b_s. 모드: 블록 (목표 encoder) · 영상-목표 · 영상-online (영상 문맥 모델의 입력 encoder). 모두 bf16, LN.
(1) 정지 칸 (z_s 와 z_{s+1} 의 두 장 사이 칸 평균 |Δ픽셀| < 1) 의 |z_{s+1} − z_s| / 다른 표본 같은 자리 거리 · 창 내내 정지 칸 변이도 (밀기 1–4) → nugget 몫
(2) W_0 (블록은 b_0 의 두 장) 에 ±1/255 섭동 → 마지막 자리 토큰 변화 / 다른 표본 거리."""
import argparse, glob, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--nat", default="/workspace/auto_research_cache/enc_noise")
ap.add_argument("--p3", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/vid_noise.json"))
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda"); G, R, NS, NF = 16, 256, 5, 24
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
SD = _load_checkpoint(CK)
ENC = {}
for key in ("target_encoder", "encoder"):
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=16, use_rope=True, uniform_power=False)
    _load_state_dict(e, SD[key], tag=key); ENC[key] = e.to(dev, torch.bfloat16).eval()
del SD
gen = torch.Generator(device=dev).manual_seed(0)


@torch.no_grad()
def enc(key, fr):                     # fr (T,R,R,3) float 0..255 → 마지막 tubelet 자리 (256, D) LN
    x = ((fr.div(255).permute(3, 0, 1, 2)[None]) - MEAN) / STD
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = ENC[key](x.to(torch.bfloat16)).float()[0]
    h = F.layer_norm(h, (h.size(-1),))
    return h.view(-1, 256, h.size(-1))[-1]


def square(fr):
    t = torch.from_numpy(fr).float()
    H_, W_ = t.shape[1:3]; s = min(H_, W_); y0, x0 = (H_ - s) // 2, (W_ - s) // 2
    x = t[:, y0:y0 + s, x0:x0 + s].permute(0, 3, 1, 2).to(dev)
    return F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False).round().clamp(0, 255).permute(0, 2, 3, 1)


def samples(src):
    out = []
    if src == "phys3d":
        P = np.load(sorted(glob.glob(os.path.join(a.p3, "pix_val_s*.npy")))[0], mmap_mode="r")
        for w in np.random.default_rng(0).choice(P.shape[0], min(a.n, P.shape[0]), replace=False):
            out.append(torch.from_numpy(np.asarray(P[w][:NF])).float().to(dev))
        return out
    metas = sum([json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.nat, f"{src}_target_encoder_s*_meta.json")))], [])
    random.Random(0).shuffle(metas)
    for m in metas:
        if len(out) >= a.n:
            break
        try:
            vr = decord.VideoReader(m["path"], num_threads=2)
            idx = [m["start"] + m["stride"] * k for k in range(NF)]
            if idx[-1] >= len(vr):
                continue
            out.append(square(vr.get_batch(idx).asnumpy()))
        except Exception:
            continue
    return out


def cellpix(x):
    return x.reshape(x.shape[0], G, 16, G, 16, 3).mean(dim=(2, 4, 5)).reshape(x.shape[0], -1)


def nug(v):
    k = np.arange(1, 4); sl, ic = np.polyfit(k, np.asarray(v[:3]), 1)
    return float(ic / v[0])


RES = {}
for src in ("k710", "ht100m", "phys3d"):
    S_ = samples(src)
    Z = {m: [] for m in ("block", "vid_target", "vid_online")}
    NZ = {m: [] for m in Z}
    PD = []
    for fr in S_:
        zs = {m: [] for m in Z}
        for s in range(NS):
            W = fr[2 * s:2 * s + 16]
            zs["vid_target"].append(enc("target_encoder", W)); zs["vid_online"].append(enc("encoder", W))
            zs["block"].append(enc("target_encoder", fr[2 * s + 14:2 * s + 16]))
        n = torch.randint(-1, 2, fr[:16].shape, device=dev, generator=gen).float()
        Wn = (fr[:16] + n).clamp(0, 255)
        NZ["vid_target"].append(enc("target_encoder", Wn)); NZ["vid_online"].append(enc("encoder", Wn))
        NZ["block"].append(enc("target_encoder", Wn[14:16]))
        for m in Z:
            Z[m].append(torch.stack(zs[m]))                                      # (NS, 256, D)
        cp = cellpix(fr).view(NF // 2, 2, -1)                                    # (12, 2, 256)
        last = torch.stack([cp[s + 7] for s in range(NS)])                       # 블록 s+7 = 장 2s+14, 2s+15
        PD.append((last[1:] - last[:-1]).abs().mean(1).cpu())                    # (NS−1, 256)
    PD = torch.stack(PD)                                                         # (n, NS−1, 256)
    st = PD < 1; allst = (PD < 2).all(1)                                         # (n, NS−1, 256), (n, 256)
    rng = np.random.default_rng(0)
    R_ = {"n": len(S_), "static_frac": float(st.float().mean()), "n_cells_static_all": int(allst.sum())}
    for m in Z:
        z = torch.stack(Z[m]).cpu()                                              # (n, NS, 256, D)
        cr = [float((z[i, 0] - z[j, 0]).abs().mean()) for i, j in (rng.choice(len(z), 2, replace=False) for _ in range(200))]
        cross = float(np.median(cr))
        dz = (z[:, 1:] - z[:, :-1]).abs().mean(-1)                               # (n, NS−1, 256)
        vg = [float((z[:, k:] - z[:, :-k]).abs().mean(-1)[allst[:, None].expand(-1, NS - k, -1)].mean()) for k in range(1, NS)]
        sens = float(torch.stack([(NZ[m][i].cpu() - z[i, 0]).abs().mean() for i in range(len(z))]).mean())
        R_[m] = {"cross": cross, "static_change": float(dz[st].mean()), "static_change_over_cross": float(dz[st].mean()) / cross,
                 "all_change_over_cross": float(dz.mean()) / cross, "variogram_static_all": vg, "nugget_frac": nug(vg) if allst.sum() > 20 else None,
                 "sens_pm1": sens, "sens_pm1_over_cross": sens / cross}
    RES[src] = R_
    print(f"\n== {src} (표본 {len(S_)}, 정지 칸 {R_['static_frac']:.0%}, 창 내내 정지 칸 {R_['n_cells_static_all']})", flush=True)
    for m in Z:
        r = R_[m]
        print(f"  {m:10s} 다른 표본 {r['cross']:.3f} | 정지 칸 이웃 창 변화 {r['static_change']:.3f} = {r['static_change_over_cross']:.0%} (모든 칸 {r['all_change_over_cross']:.0%}) | "
              f"변이도 {' '.join(f'{x:.3f}' for x in r['variogram_static_all'])} → 독립 몫 {r['nugget_frac'] if r['nugget_frac'] is None else round(r['nugget_frac'], 2)} | ±1/255 {r['sens_pm1_over_cross']:.0%}", flush=True)
json.dump(RES, open(a.out, "w"), indent=1)
print("→", a.out)
