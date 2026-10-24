#!/usr/bin/env python3
"""[T] method 고르기 진단 (학습 없음): 고정 encoder 에서 얻는 표적 공간마다 κ (옮긴 내용이 정답과 맞는 정도) · 도메인 비 · 내용 보존 · 정리 3′ 두 성분 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/target_space_diag.py [--limit N]

데이터: 자연 = [E] 수집 픽셀 (K710 · HT100M, 16 장 = 블록 8 개), 3D 합성 = 검증 캐시 픽셀의 앞 16 장 (블록 0..7) + 라벨.
표적 공간 (모두 목표 encoder, 학습 없음; 토큰마다 계수 없는 정규화):
  final  지금 표적 = 블록별 마지막 층 (encoder 출력)          L8 · L16 · L24  블록별 8 · 16 · 24 번째 transformer 블록 출력
  video  16 장을 한 번에 넣은 같은 encoder 출력 (같은 시점 tubelet)  pixel  블록 두 장 평균의 16×16 패치 (768)
  final_blur3  final 의 칸 3×3 평균
흐름: RAFT-large, 블록 대표 장 (둘째 장) 사이 역방향 U · 순방향 V, 칸 평균. 앞-뒤 일관 = |U_{t+1}(c) + V_t(c + U_{t+1}(c))| < 0.5 칸.
κ: 움직이는 칸 (|U_{t+1}| ≥ 0.5 칸), t = 4, 5, 6: 정답 ≈ a·(같은 칸 5 블록 평균) + κ·(h_t 를 U_{t+1} 로 옮긴 것). 전체 / 일관 칸.
내용 보존 (3D 합성): 블록마다 물체 안쪽 칸 (두 장 모두 덮임 ≥ 0.5, 보임) 평균 토큰 → 모양 (6) · 색 (8) 능선 선형 판독, clip 짝수 / 홀수 창으로 맞춤 / 평가.
정리 3′ 두 성분 (3D 합성): 정지 물체 안쪽 블록 변화 (증거 오염) · 칸 보정 뒤 움직인 물체 변화 (δ) — 둘 다 그 공간의 다른 창 같은 칸 거리로 나눔."""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights                # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda"); G, S = 16, 256
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
SD = _load_checkpoint(CK)["target_encoder"]


def build(nf):
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=nf, use_rope=True, uniform_power=False)
    _load_state_dict(e, SD, tag="t")
    return e.to(dev, torch.bfloat16).eval()


enc2, enc16 = build(2), build(16)
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
LAYERS = (7, 15, 23)
SPACES = ["final", "L8", "L16", "L24", "video", "pixel", "final_blur3"]
ln = lambda t: F.layer_norm(t.float(), (t.size(-1),))


@torch.no_grad()
def spaces_of(fr):                      # fr (16,256,256,3) uint8 numpy → dict name → (8,256,D) float32 cpu
    x = torch.from_numpy(np.ascontiguousarray(fr)).to(dev).float().div(255)          # (16,H,W,3)
    xb = x.view(8, 2, 256, 256, 3).permute(0, 4, 1, 2, 3)                            # (8,3,2,H,W)
    xn = ((xb - MEAN) / STD).to(torch.bfloat16)
    H = {}
    hooks = [enc2.blocks[i].register_forward_hook(lambda m, inp, out, i=i: H.__setitem__(i, out)) for i in LAYERS]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fin = enc2(xn)
        vid = enc16(((x.permute(3, 0, 1, 2)[None] - MEAN) / STD).to(torch.bfloat16))   # (1, 8*256, D)
    for h in hooks:
        h.remove()
    out = {"final": ln(fin), "L8": ln(H[7]), "L16": ln(H[15]), "L24": ln(H[23]), "video": ln(vid.view(8, S, -1))}
    pix = xb.mean(2)                                                                  # (8,3,H,W)
    pix = pix.unfold(2, 16, 16).unfold(3, 16, 16).permute(0, 2, 3, 1, 4, 5).reshape(8, S, -1)
    out["pixel"] = ln(pix)
    g = out["final"].view(8, G, G, -1).permute(0, 3, 1, 2)
    out["final_blur3"] = F.avg_pool2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), 3, 1).permute(0, 2, 3, 1).reshape(8, S, -1)
    out = {k: v.cpu() for k, v in out.items()}
    rep = x[1::2].permute(0, 3, 1, 2) * 2 - 1                                          # (8,3,H,W)
    U = torch.zeros(8, S, 2); V = torch.zeros(8, S, 2)
    fb = raft(rep[1:], rep[:-1], num_flow_updates=12)[-1]; ff = raft(rep[:-1], rep[1:], num_flow_updates=12)[-1]
    U[1:] = (F.avg_pool2d(fb, 16) / 16).permute(0, 2, 3, 1).reshape(7, S, 2).cpu()
    V[:-1] = (F.avg_pool2d(ff, 16) / 16).permute(0, 2, 3, 1).reshape(7, S, 2).cpu()
    return out, U.numpy(), V.numpy()


cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.where(ok, np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), np.arange(S)), ok


KAP = {}                                 # (dom, space, subset) → [XtX, Xty]
def kappa_add(dom, sp, U, V):
    for t in (4, 5, 6):
        u1 = U[t + 1]
        mov = np.linalg.norm(u1, axis=1) >= 0.5
        if mov.sum() < 4:
            continue
        j, okp = cell_of(P0 + u1)
        cons = mov & okp & (np.linalg.norm(u1 + V[t][j], axis=1) < 0.5)
        jt = torch.from_numpy(j)
        for name, h in sp.items():
            m5 = h[t - 4:t + 1].mean(0); wc = h[t][jt]; y = h[t + 1]
            for sub, msk in (("all", mov), ("cons", cons)):
                if msk.sum() < 2:
                    continue
                mt = torch.from_numpy(msk)
                X = torch.stack([m5[mt], wc[mt]], -1).reshape(-1, 2).double(); yy = y[mt].reshape(-1).double()
                A = KAP.setdefault((dom, name, sub), [torch.zeros(2, 2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)])
                A[0] += X.T @ X; A[1] += X.T @ yy


EN = "/workspace/auto_research_cache/enc_noise"
nat_n = 0
for src in ("k710", "ht100m"):
    for sh in (0, 1):
        P = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_pix.npy"), mmap_mode="r")
        for w in range(P.shape[0] if not a.limit else min(a.limit, P.shape[0])):
            sp, U, V = spaces_of(np.asarray(P[w]))
            kappa_add("natural", sp, U, V); nat_n += 1
    print(f"자연 {src}: 창 {nat_n}", flush=True)


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


C3 = "/workspace/auto_research_cache/phys3d"
FEAT = {n: [] for n in SPACES}; LBL = []; HALF = []
DEC = {n: {"static": [], "moving": []} for n in SPACES}; SCALE = {n: [] for n in SPACES}
prev = None
syn_n = 0
for f in sorted(glob.glob(os.path.join(C3, "pix_val_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    P = np.load(f, mmap_mode="r"); lab = np.load(os.path.join(C3, f"lab_val_s{sh}.npz")); meta = json.load(open(os.path.join(C3, f"meta_val_s{sh}.json")))
    for w in range(P.shape[0] if not a.limit else min(a.limit, P.shape[0])):
        sp, U, V = spaces_of(np.asarray(P[w, :16]))
        kappa_add("phys3d", sp, U, V)
        cv = cover(np.nan_to_num(lab["bbox"][w][:16], nan=-100.0))                       # (16,G,G)
        cmin, cmax = np.minimum(cv[0::2], cv[1::2]), np.maximum(cv[0::2], cv[1::2])
        vis = (lab["vis_frac"][w][:16] > 0.7) & (lab["inframe_frac"][w][:16] > 0.9)
        vb = vis[0::2] & vis[1::2]; pos = lab["pos"][w][:16]
        for b in range(8):
            ins = (cmin[b] >= 0.5).reshape(-1)
            if vb[b] and ins.sum() >= 1:
                for n in SPACES:
                    FEAT[n].append(sp[n][b][torch.from_numpy(ins)].mean(0).numpy())
                LBL.append((meta[w]["shape"], meta[w]["color"])); HALF.append(syn_n % 2)
        for t in range(7):
            if not (vb[t] and vb[t + 1]):
                continue
            d = pos[2 * t + 3] - pos[2 * t + 1]; dn = np.linalg.norm(d)
            if not np.isfinite(dn):
                continue
            if dn < 1:
                ins = (cmin[t] >= 0.8) & (cmin[t + 1] >= 0.8)
                if ins.sum():
                    m = torch.from_numpy(ins.reshape(-1))
                    for n in SPACES:
                        DEC[n]["static"].append(float((sp[n][t + 1][m] - sp[n][t][m]).abs().mean()))
            elif dn > 8:
                dx, dy = int(round(d[0] / 16)), int(round(d[1] / 16))
                dst = (cmin[t + 1] >= 0.8)
                src_ok = np.roll(np.roll(cmin[t] >= 0.8, dy, 0), dx, 1)
                m_ = dst & src_ok
                if m_.sum():
                    m = torch.from_numpy(m_.reshape(-1))
                    for n in SPACES:
                        mv = torch.roll(torch.roll(sp[n][t].view(G, G, -1), dy, 0), dx, 1).reshape(S, -1)
                        DEC[n]["moving"].append(float((sp[n][t + 1][m] - mv[m]).abs().mean()))
        if prev is not None:
            for n in SPACES:
                SCALE[n].append(float((sp[n][4] - prev[n][4]).abs().mean()))
        prev = {n: sp[n] for n in SPACES}
        syn_n += 1
    print(f"3D 합성 s{sh}: 창 {syn_n}", flush=True)

res = {}
LBL = np.array(LBL); HALF = np.array(HALF)
for n in SPACES:
    r = {}
    for dom in ("natural", "phys3d"):
        for sub in ("all", "cons"):
            k = (dom, n, sub)
            if k in KAP:
                r[f"kappa_{dom}_{sub}"] = float(torch.linalg.solve(*KAP[k])[1])
    for sub in ("all", "cons"):
        if f"kappa_natural_{sub}" in r and f"kappa_phys3d_{sub}" in r:
            r[f"ratio_{sub}"] = r[f"kappa_natural_{sub}"] / r[f"kappa_phys3d_{sub}"]
    X = np.stack(FEAT[n]).astype(np.float64); X = (X - X[HALF == 0].mean(0)) / (X[HALF == 0].std(0) + 1e-6)
    for li, lname in ((0, "shape"), (1, "color")):
        y = LBL[:, li]; cls = np.unique(y); Y = (y[:, None] == cls[None]).astype(np.float64)
        tr, te = HALF == 0, HALF == 1
        Wt = np.linalg.solve(X[tr].T @ X[tr] + 10.0 * np.eye(X.shape[1]), X[tr].T @ Y[tr])
        r[f"acc_{lname}"] = float((np.argmax(X[te] @ Wt, 1) == np.searchsorted(cls, y[te])).mean())
    sc = float(np.mean(SCALE[n]))
    r["static_change_rel"] = float(np.mean(DEC[n]["static"]) / sc); r["moving_change_rel"] = float(np.mean(DEC[n]["moving"]) / sc)
    r["n_static"] = len(DEC[n]["static"]); r["n_moving"] = len(DEC[n]["moving"])
    res[n] = r
base = res["final"]
print("\n표적 공간 | κ 자연 · 3D 합성 · 비 (전체 칸) | 일관 칸 κ 자연 · 합성 · 비 | 모양 · 색 판독 (지금 표적 대비) | 정지 물체 변화 · 움직인 물체 변화 (다른 창 거리 대비)")
for n, r in res.items():
    print(f"  {n:12s} | {r.get('kappa_natural_all', float('nan')):.3f} · {r.get('kappa_phys3d_all', float('nan')):.3f} · {r.get('ratio_all', float('nan')):.2f} | "
          f"{r.get('kappa_natural_cons', float('nan')):.3f} · {r.get('kappa_phys3d_cons', float('nan')):.3f} · {r.get('ratio_cons', float('nan')):.2f} | "
          f"{r['acc_shape']:.2f} ({r['acc_shape'] / max(base['acc_shape'], 1e-6):.2f}) · {r['acc_color']:.2f} ({r['acc_color'] / max(base['acc_color'], 1e-6):.2f}) | "
          f"{r['static_change_rel']:.2f} · {r['moving_change_rel']:.2f}", flush=True)
res["n"] = {"natural_windows": nat_n, "phys3d_windows": syn_n, "readout_samples": int(len(LBL))}
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/target_space_diag.json"), "w"), indent=1)
