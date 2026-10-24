#!/usr/bin/env python3
"""문맥 토큰에 속도가 선형으로 있는가 — 블록 토큰 vs 영상 문맥 토큰 (2026-10-20, P1 원인 가르기).

  # GPU 4 장에 나눠 특징 뽑기 → 합쳐 probe
  for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k python ... --out F.npz --shard $k --nshard 4 & done; wait
  python ... --out F.npz --nshard 4 --probe_only

질문: 블록 AR 이 미래에서 물체를 안 움직이는 것 (P1) 이 **입력** (블록 문맥 토큰에 속도가 없다) 탓인가, **predictor · 손실** 탓인가.
데이터: phys3d_single_v1 **val** (물체 1 개, 라벨 속도). 채점과 같은 3 장 간격 (clip 길이 48~96 장).
문맥 = 16 장 (블록 8 개, raw 시작 r0 = 0, hop, 2·hop … 창이 들어가는 곳 전부), probe 하는 자리 = **문맥 마지막 블록 (8 번째)**.
라벨 = 그 블록 마지막 장의 속도 (px / 샘플 장 = raw ×3). 물체가 보이는 (vis_frac > 0.3 · 화면 안) 자리만.

표현 — **모두 문맥 16 장 안만 본다** (미래 누출 없음). affine 없는 LN.
  block       블록 AR 입력. target_encoder 가 그 블록 2 장만 (= planning 의 "이미지 한 장" 조건)
  block_diff  block_8 − block_7 (같은 칸) — 블록 사이 차이
  block_pair  [block_8, block_7] 이어 붙임 (2560) — 블록 AR predictor 가 가진 두 블록
  tgt_ctx16   target_encoder 가 문맥 16 장을 한 번에 — block 과 **같은 가중치**, 시간 문맥만 다르다
  ctx_video   online encoder 가 문맥 16 장을 한 번에 — AR_context_video 입력
pooling: obj = 물체 중심 칸 3×3 평균 · all = 256 칸 전체 평균 (속도가 물체 칸 밖에 퍼져 있으면 all 에서만 읽힌다).
probe: ridge (closed form), clip 단위 50/50 split, λ 는 train 안 5-fold. R² (전체 · 움직이는 |v|>2 표본만) · MAE · 방향 cos (|v|>2).

덤: **블록 사이 변화가 물체에 몰리나** (움직이는 칸 가중 L1 설계 전 확인). 문맥 블록 b=1..7 마다
  변화 c = mean_D |h_b − h_{b−1}| (256 칸) 의 상위 10 % 칸 중 물체 칸 (블록 b−1 · b 의 bbox 합집합과 겹치는 칸) 비율 = precision,
  물체 칸 중 상위 10 % 에 든 비율 = recall, 화면 중 물체 칸 비율 = 기준선. 같은 것을 픽셀 변화로도 (참고: 진짜 움직임이 얼마나 몰렸나).
근거: linear probe 로 표현에 정보가 선형으로 있는지 본다 (Alain & Bengio 2016)."""
import argparse, glob, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from app.vjepa_frozen import ar as AR                                                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_single_v1/val/clips")
ap.add_argument("--n_clips", type=int, default=1040)
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--hop", type=int, default=3, help="문맥 창 시작 간격 (raw 장)")
ap.add_argument("--out", required=True, help="특징 npz (shard 마다 <out>.<k>.npz)")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--probe_only", action="store_true")
a = ap.parse_args()
dev, S, G, TUB = "cuda", 256, 16, 2
CKE = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
LN = lambda t: F.layer_norm(t, (t.size(-1),))
CB = 8                                                                       # 문맥 블록 수
REPS = ("block", "block_diff", "block_pair", "tgt_ctx16", "ctx_video")
TOPQ = 0.10


def cells_of_box(bx):
    x0, y0, x1, y1 = [float(v) for v in bx]
    if not np.isfinite([x0, y0, x1, y1]).all():
        return np.zeros(S, bool)
    m = np.zeros((G, G), bool)
    m[max(0, int(y0 // 16)): min(G, int(y1 // 16) + 1), max(0, int(x0 // 16)): min(G, int(x1 // 16) + 1)] = True
    return m.reshape(-1)


def extract():
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    sd = _load_checkpoint(CKE)
    tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=64, use_rope=True, uniform_power=False)
    _load_state_dict(tgt, sd["target_encoder"], tag="t"); tgt = tgt.to(dev).eval()
    onl = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=64, use_rope=True, uniform_power=False)
    _load_state_dict(onl, sd["encoder"], tag="o"); onl = onl.to(dev).eval()
    del sd
    paths = sorted(glob.glob(os.path.join(a.root, "*", "*.mp4")))
    random.Random(0).shuffle(paths)
    paths = list(enumerate(paths[: a.n_clips]))[a.shard::a.nshard]
    out = {f"{r}_{p}": [] for r in REPS for p in ("obj", "all")}
    out.update(vel=[], clip=[], scen=[], chg=[])                             # chg: (7, 8) = 블록 b 마다 [latent prec, rec, prec@|물체|, pixel 같은 셋, 물체 칸 비율, 움직임]
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for n_done, (ci, path) in enumerate(paths):
            st = np.load(path[:-4] + "_states.npz")
            vr = decord.VideoReader(path, num_threads=2)
            T = min(len(vr), st["pos"].shape[0])
            span = a.stride * (2 * CB - 1)                                                     # 문맥 16 장이 덮는 raw 길이 − 1
            for r0 in range(0, T - span, a.hop):
                f = r0 + span                                                                  # 문맥 마지막 블록의 마지막 장
                pos, vel = st["pos"][f, 0], st["vel"][f, 0] * a.stride
                ok = st["vis_frac"][f, 0] > 0.3 and st["inframe_frac"][f, 0] > 0.5 and np.isfinite(pos).all() and np.isfinite(vel).all()
                if not ok:
                    continue
                idx = [r0 + a.stride * k for k in range(2 * CB)]
                raw = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)   # (1,3,16,H,W)
                if raw.shape[-1] != 256:
                    raw = F.interpolate(raw[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
                x = (raw - MEAN) / STD
                hb = AR.encode_blocks(tgt, x, TUB).float().view(CB, S, -1)                     # 블록 8 개, 각자 2 장만
                ht = LN(tgt(x)).float().view(CB, S, -1)                                         # target_encoder 문맥 16 장
                hc = LN(onl(x)).float().view(CB, S, -1)                                         # online encoder 문맥 16 장
                cx, cy = int(np.clip(pos[0] // 16, 0, G - 1)), int(np.clip(pos[1] // 16, 0, G - 1))
                cells = [yy * G + xx for yy in range(max(0, cy - 1), min(G, cy + 2)) for xx in range(max(0, cx - 1), min(G, cx + 2))]
                po = lambda t: t[cells].mean(0).cpu().numpy()
                pa = lambda t: t.mean(0).cpu().numpy()
                for p, fn in (("obj", po), ("all", pa)):
                    b8, b7 = fn(hb[CB - 1]), fn(hb[CB - 2])
                    out[f"block_{p}"].append(b8)
                    out[f"block_diff_{p}"].append(b8 - b7)
                    out[f"block_pair_{p}"].append(np.concatenate([b8, b7]))
                    out[f"tgt_ctx16_{p}"].append(fn(ht[CB - 1]))
                    out[f"ctx_video_{p}"].append(fn(hc[CB - 1]))
                # 블록 사이 변화가 물체 칸에 몰리나
                lat = (hb[1:] - hb[:-1]).abs().mean(-1).cpu().numpy()                            # (7, 256)
                pb = raw[0].view(3, CB, 2, G, 16, G, 16).mean(2)                                 # 블록당 2 장 평균 (3, 8, 16, 16, 16, 16)
                pix = (pb[:, 1:] - pb[:, :-1]).abs().mean((0, 3, 5)).reshape(CB - 1, S).cpu().numpy()   # (7, 256)
                k = int(round(TOPQ * S))
                rows = []
                for b in range(1, CB):
                    fa, fb_ = idx[2 * b - 1], idx[2 * b + 1]                                    # 블록 b−1 · b 의 마지막 장
                    om = cells_of_box(st["bbox"][fa, 0]) | cells_of_box(st["bbox"][fb_, 0])
                    mov = np.linalg.norm(st["vel"][fb_, 0] * a.stride) > 2 and st["vis_frac"][fb_, 0] > 0.3
                    if om.sum() == 0:
                        rows.append([np.nan] * 8); continue
                    r_ = []
                    no = int(om.sum())
                    for c in (lat[b - 1], pix[b - 1]):
                        o = np.argsort(-c)
                        top = np.zeros(S, bool); top[o[:k]] = True
                        topo = np.zeros(S, bool); topo[o[:no]] = True
                        r_ += [(top & om).sum() / k, (top & om).sum() / no, (topo & om).sum() / no]
                    rows.append([r_[0], r_[1], r_[2], r_[3], r_[4], r_[5], om.mean(), float(mov)])
                out["chg"].append(np.asarray(rows, np.float32))
                out["vel"].append(vel); out["clip"].append(ci); out["scen"].append(os.path.basename(os.path.dirname(path)))
            if n_done % 25 == 0:
                print(f"[shard {a.shard}] {n_done}/{len(paths)} clip · 표본 {len(out['vel'])} · {time.time() - t0:.0f}s", flush=True)
    out = {k: np.asarray(v, dtype=np.float32 if k not in ("clip", "scen") else None) for k, v in out.items()}
    np.savez(f"{a.out}.{a.shard}.npz", **out)
    print(f"[shard {a.shard}] 끝 · 표본 {len(out['vel'])}")


def load_all():
    parts = [np.load(f"{a.out}.{k}.npz", allow_pickle=True) for k in range(a.nshard)]
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0].files}


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    Xt = torch.from_numpy(Xs).double().cuda(); Yt = torch.from_numpy(Y - ym).double().cuda()
    n, d = Xt.shape
    if d > n:                                                                   # dual
        K = Xt @ Xt.T
        W = Xt.T @ torch.linalg.solve(K + lam * torch.eye(n, dtype=K.dtype, device=K.device), Yt)
    else:
        W = torch.linalg.solve(Xt.T @ Xt + lam * torch.eye(d, dtype=Xt.dtype, device=Xt.device), Xt.T @ Yt)
    W = W.cpu()
    return lambda Z: ((torch.from_numpy((Z - mu) / sd).double() @ W).numpy() + ym)


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def probe(X, Y, clip):
    uc = np.unique(clip); rng = np.random.RandomState(0); rng.shuffle(uc)
    tr_c = set(uc[: len(uc) // 2])
    tr = np.array([c in tr_c for c in clip]); te = ~tr
    Xtr, Ytr, ctr = X[tr], Y[tr], clip[tr]
    folds = np.array([int(c) % 5 for c in ctr])
    best = None
    for lam in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5):
        sc = np.mean([r2(Ytr[folds == k], ridge_fit(Xtr[folds != k], Ytr[folds != k], lam)(Xtr[folds == k])) for k in range(5)])
        if best is None or sc > best[0]:
            best = (sc, lam)
    P = ridge_fit(Xtr, Ytr, best[1])(X[te])
    Yte = Y[te]
    mv = np.linalg.norm(Yte, axis=1) > 2
    cos = (P[mv] * Yte[mv]).sum(1) / (np.linalg.norm(P[mv], axis=1) * np.linalg.norm(Yte[mv], axis=1) + 1e-9)
    return dict(r2=r2(Yte, P), r2m=r2(Yte[mv], P[mv]), mae=np.abs(P - Yte).mean(), mae0=np.abs(Yte - Ytr.mean(0)).mean(),
                cos=cos.mean(), lam=best[1], n_tr=int(tr.sum()), n_te=int(te.sum()))


if not a.probe_only:
    extract()
    sys.exit()

D = load_all()
Y, clip = D["vel"], D["clip"]
sp = np.linalg.norm(Y, axis=1)
print(f"표본 {len(Y)} (clip {len(np.unique(clip))}) · |v| 평균 {sp.mean():.1f} px/장 (raw ×{a.stride}) · 움직이는 (>2) 비율 {np.mean(sp > 2):.2f}")
print(f"  {'표현':11s} {'pool':4s} {'R²':>6s} {'R²(움직임)':>10s} {'MAE':>5s} {'(평균)':>6s} {'방향cos':>7s} {'λ':>6s}")
for r in REPS:
    for p in ("obj", "all"):
        m = probe(D[f"{r}_{p}"], Y, clip)
        print(f"  {r:11s} {p:4s} {m['r2']:6.3f} {m['r2m']:10.3f} {m['mae']:5.2f} {m['mae0']:6.2f} {m['cos']:7.3f} {m['lam']:6.0e}")
print(f"  (train {m['n_tr']} / test {m['n_te']})")

C = D["chg"].reshape(-1, 8)
C = C[np.isfinite(C).all(1)]
print(f"\n블록 사이 변화가 물체 칸 (블록 b−1 · b bbox 합집합) 에 몰리나 — 블록 쌍 {len(C)}")
print(f"  상위 {int(TOPQ * 100)} % ({int(round(TOPQ * S))} 칸) precision · recall, 그리고 상위 |물체 칸 수| 개 중 물체 비율 (= 완벽하면 1)")
print(f"  {'':18s} {'prec10%':>7s} {'rec10%':>7s} {'prec@|obj|':>10s}   {'물체 칸 비율 (무작위 기준)':s}")
for name, sel in (("전체", np.ones(len(C), bool)), ("물체 움직임 >2", C[:, 7] > 0.5), ("물체 정지", C[:, 7] < 0.5)):
    c = C[sel]
    print(f"  latent {name:10s} {c[:, 0].mean():7.3f} {c[:, 1].mean():7.3f} {c[:, 2].mean():10.3f}   {c[:, 6].mean():.3f}   (n {len(c)})")
    print(f"  pixel  {name:10s} {c[:, 3].mean():7.3f} {c[:, 4].mean():7.3f} {c[:, 5].mean():10.3f}")
