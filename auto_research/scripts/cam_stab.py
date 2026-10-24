#!/usr/bin/env python3
"""[물리] (5) 정답 안정화 판정 — 카메라 움직임 창의 다음 블록을 문맥 마지막 장 시점에 맞춰 (픽셀) 다시 인코딩하면 물체 칸 운반 신뢰 b 가 오르나 (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/cam_stab.py

사전 등록: 진행판 [물리] 06:36 UTC (KST 15:36), 리더 KST 15:33.
자연 네 무리 (K710 보류 · HT100M · SSv2 수동 · 행위자), 블록 8 개 창, t = 4 · 5 · 6. 프레임은 캐시 규약대로 다시 읽고 모든 블록을 같은 target encoder 호출로 다시 인코딩.
카메라 움직임 = 문맥 카메라 (캐시 U[t] 에 맞춘 사상) 의 칸 변위 중앙값 ≥ 0.3 칸.
안정화: 블록 t+1 두 장 → 블록 t 대표 장 RAFT 칸 흐름에 강건 호모그래피 → 픽셀 bilinear 로 대표 장 시점에 맞춤 → 다시 인코딩.
빈 칸 = 옮긴 두 장 중 하나라도 화면 밖 화소가 있는 칸 (제외). 물체 칸: 안정화 전 = 사상 잔차 ≥ 0.5 칸, 안정화 뒤 = 안정화 장 → 대표 장 흐름 ≥ 0.5 칸.
회귀 y ≈ a·m + b·W (class_transport). W: 문맥 외삽 (U[t]) / 카메라 뺀 물체 외삽 (U[t] − 문맥 카메라).
재샘플링 대조: 카메라 정지 창의 블록 t+1 에 움직임 창의 사상을 걸었다 되돌림.
토큰은 옮기지 않는다 — 옮기는 것은 픽셀 프레임이고 정답 토큰은 encoder 가 다시 만든다. W 의 옮긴 토큰은 데이터 최적 b 를 재는 회귀 도구."""
import argparse, collections, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--nb", type=int, default=1000)
ap.add_argument("--out", default="/workspace/world_model/vjepa2/auto_research/results/cam_state_stab.json")
a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--parts", "", "--limit", "1"]
import cam_state as CS                                                         # 호모그래피 · splat 등 (부분 실행 없음)
sys.argv = _argv
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
from app.vjepa_frozen import ar as AR
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
torch.set_num_threads(8)
dev = torch.device("cuda")
S, G = 256, 16
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
sd = _load_checkpoint(CK)
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, sd["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval(); del sd
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
EN = "/workspace/auto_research_cache/enc_noise"


def sq256(fr):                                                                  # (T,H,W,3) uint8 → (T,3,256,256) float 0..255 (캐시 규약)
    x = torch.from_numpy(fr).to(dev).permute(0, 3, 1, 2).float()
    Hh, Ww = x.shape[-2:]; s = min(Hh, Ww); y0, x0 = (Hh - s) // 2, (Ww - s) // 2
    x = x[:, :, y0:y0 + s, x0:x0 + s]
    return F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()


def frames_iter():
    nh = "/workspace/auto_research_cache/natural_holdout"
    meta = json.load(open(f"{nh}/k710_320_holdout_meta.json")); U = np.load(f"{nh}/k710_320_holdout_U.npy").astype(np.float32).reshape(-1, 8, S, 2)
    for w, m in enumerate(meta[: a.limit or None]):
        vr = decord.VideoReader(m["path"], num_threads=2)
        idx = [m["start"] + m["fstp"] * k for k in range(16)]
        if idx[-1] >= len(vr):
            continue
        yield "k710_holdout", w, sq256(vr.get_batch(idx).asnumpy()), U[w]
    U = np.load(os.path.join(REPO, "auto_research/results/natural_transport_ctx5/rows_ht100m.npz"))["Uwin"].astype(np.float32).reshape(-1, 8, S, 2)
    w0 = 0
    for sh in (0, 1):
        PX = np.load(os.path.join(EN, f"ht100m_target_encoder_s{sh}_pix.npy"), mmap_mode="r")
        for w in range(PX.shape[0] if not a.limit else min(a.limit, PX.shape[0])):
            yield "ht100m", w0 + w, torch.from_numpy(np.asarray(PX[w])).to(dev).permute(0, 3, 1, 2).float(), U[w0 + w]
        w0 += PX.shape[0]
    sc = "/workspace/auto_research_cache/ssv2_classes"
    for grp in ("passive", "actor"):
        meta = json.load(open(f"{sc}/{grp}_meta.json")); U = np.load(f"{sc}/{grp}_U.npy").astype(np.float32).reshape(-1, 8, S, 2)
        for w, m in enumerate(meta[: a.limit or None]):
            vr = decord.VideoReader(m["path"], num_threads=2)
            yield f"ssv2_{grp}", w, sq256(vr.get_batch(list(range(m["start"], m["start"] + 15 * m["fstp"] + 1, m["fstp"]))).asnumpy()), U[w]


@torch.no_grad()
def enc(x):                                                                    # (T,3,256,256) 0..255, T 짝수 → (T/2, 256, D) LN 블록 토큰
    xx = (x / 255).permute(1, 0, 2, 3)[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = AR.encode_blocks(tgt, ((xx - MEAN) / STD).to(torch.bfloat16), 2)
    return h.float()[0].view(x.shape[0] // 2, S, -1).cpu().numpy()


@torch.no_grad()
def flow(i0, i1):                                                              # (n,3,256,256) 0..255 → i0 → i1 칸 흐름 (n, 256, 2)
    f = raft(i0 / 127.5 - 1, i1 / 127.5 - 1, num_flow_updates=12)[-1]
    return (F.avg_pool2d(f, 16) / 16).permute(0, 2, 3, 1).reshape(-1, S, 2).cpu().numpy()


def sample(img, Hc):
    """img (n,3,256,256), Hc (n,3,3) 칸 사상 (출력 장 → img 장). out(x) = img(Hc(x)). 유효 마스크 (n,256,256)."""
    yy, xx = torch.meshgrid(torch.arange(256, device=dev) + 0.5, torch.arange(256, device=dev) + 0.5, indexing="ij")
    p = torch.stack([(xx / 16 - 8) / 8, (yy / 16 - 8) / 8, torch.ones_like(xx)], -1).view(-1, 3).double()
    out, val = [], []
    for k in range(img.shape[0]):
        q = p @ torch.from_numpy(Hc[k]).to(dev).T
        q = q[:, :2] / q[:, 2:3]                                               # 정규화 칸 좌표 (−1..1 = 화면)
        grid = q.float().view(1, 256, 256, 2)
        out.append(F.grid_sample(img[k:k + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=False))
        val.append((grid.abs() <= 1).all(-1)[0])
    return torch.cat(out), torch.stack(val)


KEYS = ("mov:unstab_lag", "mov:unstab_obj", "mov:stab_obj", "mov:stab_lag", "mov:unstab_lag_valid", "still:lag", "still:rt", "still:lag_rtcells")
ACC = {}                                                                      # (src, key) → list of (win, XtX, Xty, n)
INFO = collections.defaultdict(list)
# 무작위 사상 풀 (재샘플링 대조): 캐시 U[t+1] 에서 맞춘 카메라 움직임 걸음의 사상
POOLH = []
rng = np.random.RandomState(0)


def add(src, win, key, h_t_W, m, y, sel):
    if sel.sum() < 2:
        return
    X = np.stack([m[sel].ravel(), h_t_W[sel].ravel()], 1).astype(np.float64); yy = y[sel].ravel().astype(np.float64)
    ACC.setdefault((src, key), []).append((win, X.T @ X, X.T @ yy, int(sel.sum())))


t0 = time.time()
for n_done, (src, w, x, U) in enumerate(frames_iter()):
    win = f"{src}:{w}"
    h = enc(x)                                                                 # (8, 256, D) 원래 정답 (같은 호출)
    steps = []
    for t in (4, 5, 6):
        Ht, _, _ = CS.fit_cam(U[t])
        steps.append((t, Ht, CS.cam_mag(Ht) >= 0.3))
        if CS.cam_mag(Ht) >= 0.3:
            Hn, _, _ = CS.fit_cam(U[t + 1]); POOLH.append(Hn)
    # 블록 t+1 두 장 → 블록 t 대표 장 흐름 (세 걸음 한 번에)
    i0 = torch.stack([x[2 * t + 2 + j] for t, _, _ in steps for j in (0, 1)]); i1 = torch.stack([x[2 * t + 1] for t, _, _ in steps for j in (0, 1)])
    fl = flow(i0, i1)                                                          # (6, 256, 2)
    stab_imgs, stab_meta, rt_imgs, rt_meta = [], [], [], []
    for si, (t, Ht, mov) in enumerate(steps):
        Ha, _, _ = CS.fit_cam(fl[2 * si]); Hb, _, _ = CS.fit_cam(fl[2 * si + 1])
        resb = np.linalg.norm(fl[2 * si + 1] - CS.cam_flow(Hb), axis=1)
        obj_un = resb >= 0.5
        m = h[t - 4:t + 1].mean(0)
        W_lag = CS.splat(h[t], U[t]); W_obj = CS.splat(h[t], U[t] - CS.cam_flow(Ht))
        if mov:
            add(src, win, "mov:unstab_lag", W_lag, m, h[t + 1], obj_un)
            add(src, win, "mov:unstab_obj", W_obj, m, h[t + 1], obj_un)
            # 안정화: 출력 (블록 t 대표 장 시점) 의 칸 x → 원래 장의 자리 = 사상의 역 (원래 장 → 대표 장 사상 H 의 역)
            img, val = sample(torch.stack([x[2 * t + 2], x[2 * t + 3]]), np.stack([CS.hinv(Ha), CS.hinv(Hb)]))
            stab_imgs.append(img); stab_meta.append((t, si, obj_un, val, W_lag, W_obj, m, CS.cam_mag(Hb)))
        else:
            add(src, win, "still:lag", W_lag, m, h[t + 1], obj_un)
            if POOLH:
                Hr = POOLH[rng.randint(len(POOLH))]
                img1, v1 = sample(torch.stack([x[2 * t + 2], x[2 * t + 3]]), np.stack([CS.hinv(Hr)] * 2))      # 걸기
                img2, v2 = sample(img1, np.stack([Hr] * 2))                                                     # 되돌리기
                rt_imgs.append(img2); rt_meta.append((t, obj_un, v2, v1, Hr, W_lag, m))
    if stab_imgs:
        hs = enc(torch.cat(stab_imgs).clamp(0, 255).round())                   # (k, 256, D)
        fs = flow(torch.stack([im[1] for im in stab_imgs]).clamp(0, 255).round(), torch.stack([x[2 * mt[0] + 1] for mt in stab_meta]))
        for k, (t, si, obj_un, val, W_lag, W_obj, m, mag_b) in enumerate(stab_meta):
            cval = (F.avg_pool2d(val.float()[:, None], 16).view(2, S) > 0.999).all(0).cpu().numpy()   # 두 장 모두 칸 전체 유효
            obj_st = (np.linalg.norm(fs[k], axis=1) >= 0.5) & cval
            add(src, win, "mov:stab_obj", W_obj, m, hs[k], obj_st)
            add(src, win, "mov:stab_lag", W_lag, m, hs[k], obj_st)
            add(src, win, "mov:unstab_lag_valid", W_lag, m, h[t + 1], obj_un & cval)
            Hs_, _, _ = CS.fit_cam(np.where(cval[:, None], fs[k], 0.0))
            INFO[src].append(dict(t=t, invalid_frac=float(1 - cval.mean()), obj_un=int(obj_un.sum()), obj_un_invalid=int((obj_un & ~cval).sum()), obj_st=int(obj_st.sum()),
                                  cam_before=float(mag_b), cam_after=float(np.median(np.linalg.norm(fs[k][cval], axis=1))) if cval.any() else float('nan')))
    if rt_imgs:
        hr = enc(torch.cat(rt_imgs).clamp(0, 255).round())
        for k, (t, obj_un, v2, v1, Hr, W_lag, m) in enumerate(rt_meta):
            # 되돌린 장에서 유효 = 되돌림 샘플이 화면 안 (v2) 이고 그 자리가 첫 걸기에서 유효 — 첫 걸기 유효 마스크를 같은 사상으로 끌어온다
            yyg, xxg = torch.meshgrid(torch.arange(256, device=dev) + 0.5, torch.arange(256, device=dev) + 0.5, indexing="ij")
            p = torch.stack([(xxg / 16 - 8) / 8, (yyg / 16 - 8) / 8, torch.ones_like(xxg)], -1).view(-1, 3).double()
            q = p @ torch.from_numpy(Hr).to(dev).T; q = (q[:, :2] / q[:, 2:3]).float().view(1, 256, 256, 2)
            vb = F.grid_sample(v1.float()[:, None], q.expand(2, -1, -1, -1), mode="nearest", align_corners=False)[:, 0] > 0.5
            vv = v2 & vb
            cval = (F.avg_pool2d(vv.float()[:, None], 16).view(2, S) > 0.999).all(0).cpu().numpy()
            add(src, win, "still:rt", W_lag, m, hr[k], obj_un & cval)
            add(src, win, "still:lag_rtcells", W_lag, m, h[t + 1], obj_un & cval)
            INFO[src + ":rt"].append(dict(t=t, invalid_frac=float(1 - cval.mean())))
    if n_done % 50 == 0:
        print(f"{src} {w} · {n_done} 창 · {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ 요약
def summarize(srcs_):
    wins = sorted({r[0] for s in srcs_ for k in KEYS for r in ACC.get((s, k), [])})
    wi = {u: i for i, u in enumerate(wins)}; n = len(wins)
    XtX = np.zeros((len(KEYS), n, 2, 2)); Xty = np.zeros((len(KEYS), n, 2)); NC = np.zeros((len(KEYS), n))
    for s in srcs_:
        for ki, k in enumerate(KEYS):
            for (u, A, v, c) in ACC.get((s, k), []):
                XtX[ki, wi[u]] += A; Xty[ki, wi[u]] += v; NC[ki, wi[u]] += c

    def bfit(cnt):
        A = np.einsum("w,kwij->kij", cnt, XtX); v = np.einsum("w,kwi->ki", cnt, Xty)
        out = np.full(len(KEYS), np.nan)
        for ki in range(len(KEYS)):
            if np.linalg.det(A[ki]) > 1e-9:
                out[ki] = np.linalg.solve(A[ki], v[ki])[1]
        return out
    b0 = bfit(np.ones(n))
    B = np.array([bfit(np.bincount(rng.randint(0, n, n), minlength=n).astype(float)) for _ in range(a.nb)])
    K = {k: i for i, k in enumerate(KEYS)}
    r = {"n_windows": n}
    for k in KEYS:
        r[k] = dict(b=float(b0[K[k]]), ci=np.nanpercentile(B[:, K[k]], [2.5, 97.5]).tolist(), n_cells=int(NC[K[k]].sum()), n_win=int((NC[K[k]] > 0).sum()))

    def diff(k1, k2):
        d = B[:, K[k1]] - B[:, K[k2]]
        return dict(d=float(b0[K[k1]] - b0[K[k2]]), ci=np.nanpercentile(d, [2.5, 97.5]).tolist())
    r["gain_stab"] = diff("mov:stab_obj", "mov:unstab_lag")
    r["gain_stab_vs_validcells"] = diff("mov:stab_obj", "mov:unstab_lag_valid")
    r["gap_still"] = diff("still:lag", "mov:unstab_lag")
    r["resample_ctrl"] = diff("still:rt", "still:lag_rtcells")
    g = (B[:, K["mov:stab_obj"]] - B[:, K["mov:unstab_lag"]]) / (B[:, K["still:lag"]] - B[:, K["mov:unstab_lag"]])
    r["frac_gap_closed"] = dict(v=float((b0[K["mov:stab_obj"]] - b0[K["mov:unstab_lag"]]) / (b0[K["still:lag"]] - b0[K["mov:unstab_lag"]])), ci=np.nanpercentile(g, [2.5, 97.5]).tolist())
    return r


res = {"definition": "사전 등록 [물리] 06:36 UTC (KST 15:36)"}
NAT = ("k710_holdout", "ht100m", "ssv2_passive", "ssv2_actor")
for nm, ss in [(s, (s,)) for s in NAT] + [("natural_pooled", NAT)]:
    r = summarize(ss)
    inf = [d for s in ss for d in INFO[s]]
    if inf:
        r["stab_check"] = dict(cam_before_median=float(np.median([d["cam_before"] for d in inf])), flow_after_median=float(np.nanmedian([d["cam_after"] for d in inf])))
        r["border"] = dict(invalid_cell_frac_mean=float(np.mean([d["invalid_frac"] for d in inf])),
                           obj_cells_lost_to_border=float(sum(d["obj_un_invalid"] for d in inf) / max(sum(d["obj_un"] for d in inf), 1)))
    rt = [d for s in ss for d in INFO[s + ":rt"]]
    if rt:
        r["border_rt"] = float(np.mean([d["invalid_frac"] for d in rt]))
    res[nm] = r
    print(f"\n== {nm}: 창 {r['n_windows']}")
    for k in KEYS:
        print(f"   {k:22s} b {r[k]['b']:.3f} [{r[k]['ci'][0]:.3f}, {r[k]['ci'][1]:.3f}] 칸 {r[k]['n_cells']} 창 {r[k]['n_win']}")
    for k in ("gain_stab", "gain_stab_vs_validcells", "gap_still", "resample_ctrl", "frac_gap_closed"):
        v = r[k]; print(f"   {k:22s} {v.get('d', v.get('v')):+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
    if "border" in r:
        print(f"   안정화 확인: 카메라 크기 중앙 (전) {r['stab_check']['cam_before_median']:.2f} 칸 → 안정화 장 흐름 크기 중앙 (뒤, 유효 칸) {r['stab_check']['flow_after_median']:.2f} 칸")
        print(f"   빈 칸 몫 (움직임 걸음 평균) {r['border']['invalid_cell_frac_mean']:.3f} · 빈 칸으로 빠진 물체 칸 몫 {r['border']['obj_cells_lost_to_border']:.3f} · 대조 빈 칸 {r.get('border_rt', float('nan')):.3f}")
json.dump(res, open(a.out, "w"), indent=1)
print("저장", a.out, f"{time.time() - t0:.0f}s")
