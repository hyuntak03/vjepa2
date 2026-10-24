#!/usr/bin/env python3
"""[S] 시간 곧음 (Hénaff, Goris & Simoncelli 2019 의 곡률) — 블록 토큰 vs 영상 모드 창 토큰 (사전 등록 KST 04:51, 학습 없음).

  CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/straightening.py [--limit N]
  /venv/vjepa/bin/python auto_research/scripts/straightening.py --report

창 = 16 장 = 블록 8 개 (4 fps). 공간 (모두 칸별 affine 없는 LN, 픽셀만 정규화 화소):
  pix   칸의 블록 2 장 픽셀 (3·2·16·16 = 1536)
  blk   블록마다 따로 목표 encoder (학습 정답과 같다)
  blks  blk 를 시간 [1,2,1]/4 평활 (자리 1..6 만 유효) — 평균 대조군
  twin  16 장을 목표 encoder 한 번 → 8 자리 (양방향, 창 안 미래 장이 섞임)
  owin  16 장을 online encoder 한 번 → 8 자리 (영상 문맥 모델이 받는 문맥 꼴)
  tpre  자리 t = 장 0..2t+1 만 목표 encoder 에 넣은 창의 마지막 자리 (미래 안 봄). t=0 은 blk 와 같다
  opre  같은 것을 online encoder 로
측정:
  곡률  θ_t = arccos cos(v_t, v_{t+1}), v_t = z_t − z_{t−1}, t = 2..5 (보폭 1) · t = 3, 4 (보폭 2: z_{t−2}, z_t, z_{t+2})
        칸 · 3×3 묶음 (가운데 칸 무리) · 화면 평균
  (가)  같은 공간 다음 걸음 z_{t+1} (t = 2..6): 복사 · 직선 외삽 · 같은 칸 평균 · z_t + α v_t (α 격자)
  (나)  실제 정답 h_{t+1} (다음 블록 토큰): λ·h_t + (1−λ)·평균 + α·Δ, Δ ∈ {블록 변화, 접두 목표 창 마지막 두 자리 차, 접두 online 창 차}
  (다)  창 토큰 자리 t ≈ Σ_s w_s · blk_s (같은 칸, 스칼라, 차원 공유) R², 짝수 창 맞춤 · 홀수 창 평가
칸 무리: 자연 = RAFT 칸 흐름 |U| (걸음 t · t+1 모두 ≥ 0.5 칸 = mov, 모두 < 0.1 = sta). 3D = bbox (움직이는 물체 · 멈춘 물체 · 먼 배경)."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"
CACHE = "/workspace/auto_research_cache"
OUT = os.path.join(REPO, "auto_research/results/straightening")
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--report", action="store_true")
ap.add_argument("--n_k710", type=int, default=150); ap.add_argument("--n_ht", type=int, default=150); ap.add_argument("--n_ssv2", type=int, default=100)
ap.add_argument("--n_p3d_shards", type=int, default=2)
a = ap.parse_args()

SPACES = ("pix", "blk", "blks", "twin", "owin", "tpre", "opre")
POOLS = ("cell", "box3", "screen")
ALPHAS = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
LAMS = (1.0, 0.75, 0.5)
NEXT_SP = ("pix", "blk", "tpre", "opre", "twin", "owin")       # (가)
DELTAS = ("dblk", "dtw", "dow")                                  # (나)
M4_T = ("twin", "owin", "tpre", "opre")                          # (다)
S, G, D = 256, 16, 1280


# ============================================================================ 측정
def run():
    import decord
    sys.path.insert(0, REPO)
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
    torch.set_num_threads(8); dev = torch.device("cuda")
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    sd = _load_checkpoint(CK)

    def enc(key, nf):
        e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=nf, use_rope=True, uniform_power=False)
        _load_state_dict(e, sd[key], tag=key[:1]); return e.to(dev, torch.bfloat16).eval()
    T2, T16, O2, O16 = enc("target_encoder", 2), enc("target_encoder", 16), enc("encoder", 2), enc("encoder", 16)
    del sd
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
    ln = lambda t: F.layer_norm(t.float(), (D,))

    @torch.no_grad()
    def spaces(x):
        """x (3,16,256,256) 정규화 → dict 공간 → (8, 256, d) fp32, + Δ 들 (8,256,D) (자리 t 의 접두 창 마지막 두 자리 차, t ≥ 1)"""
        xb = x.reshape(3, 8, 2, 256, 256).permute(1, 0, 2, 3, 4).to(torch.bfloat16)
        hb = ln(T2(xb)); ob = ln(O2(xb))                                              # (8,256,D)
        tpre, opre = [hb[0]], [ob[0]]
        dtw, dow = [torch.zeros_like(hb[0])], [torch.zeros_like(hb[0])]
        for t in range(1, 8):
            xt = x[None, :, : 2 * t + 2].to(torch.bfloat16)
            wt = ln(T16(xt))[0].view(t + 1, S, D); ot = ln(O16(xt))[0].view(t + 1, S, D)
            tpre.append(wt[t]); opre.append(ot[t]); dtw.append(wt[t] - wt[t - 1]); dow.append(ot[t] - ot[t - 1])
            if t == 7:
                twin, owin = wt, ot
        pix = x.reshape(3, 8, 2, G, 16, G, 16).permute(1, 3, 5, 0, 2, 4, 6).reshape(8, S, 3 * 2 * 256).float()
        blks = hb.clone(); blks[1:7] = (hb[:6] + 2 * hb[1:7] + hb[2:]) / 4
        sp = dict(pix=pix, blk=hb, blks=blks, twin=twin, owin=owin, tpre=torch.stack(tpre), opre=torch.stack(opre))
        return sp, dict(dblk=torch.cat([torch.zeros_like(hb[:1]), hb[1:] - hb[:-1]]), dtw=torch.stack(dtw), dow=torch.stack(dow))

    def pool(z, kind):                                                                # (8,256,d) → (8, n, d)
        if kind == "cell":
            return z
        if kind == "box3":
            zz = z.view(8, G, G, -1).permute(0, 3, 1, 2)
            return F.avg_pool2d(zz, 3, 1, 1, count_include_pad=False).permute(0, 2, 3, 1).reshape(8, S, -1)
        return z.mean(1, keepdim=True)

    def curv(z, t, k):                                                                # z (8,n,d) → cos, ‖v‖/‖z‖ (n,)
        v1, v2 = z[t] - z[t - k], z[t + k] - z[t]
        c = F.cosine_similarity(v1, v2, dim=-1, eps=1e-8)
        return c, v1.norm(dim=-1) / z[t].norm(dim=-1).clamp_min(1e-8)

    rec = {}                                                                          # (src, key) → list per window of np.array

    def put(src, key, val):
        rec.setdefault((src, key), []).append(np.asarray(val, dtype=np.float64))

    def window(src, x, masks1, masks2, scr1, scr2, m4acc, widx):
        """masks1[g] (8, 256) bool: 칸 무리 at 자리 t (보폭 1, 다음 걸음 포함). masks2 (보폭 2). scr1/scr2[g] (8,) bool (화면)"""
        sp, dl = spaces(x)
        # ---- 곡률
        for s in SPACES:
            for pk in POOLS:
                z = pool(sp[s], pk)
                for k, ts, mk, sk in ((1, (2, 3, 4, 5), masks1, scr1), (2, (3, 4), masks2, scr2)):
                    acc = {}
                    for t in ts:
                        c, st = curv(z, t, k)
                        th = torch.rad2deg(torch.arccos(c.clamp(-1, 1)))
                        for g in (mk if pk != "screen" else sk):
                            m = mk[g][t] if pk != "screen" else torch.tensor([bool(sk[g][t])], device=dev)
                            if m.sum() == 0:
                                continue
                            o = acc.setdefault(g, np.zeros(4))
                            o += [float(th[m].sum()), float(c[m].sum()), float(st[m].sum()), float(m.sum())]
                    for g in (mk if pk != "screen" else sk):
                        put(src, f"curv|{s}|{pk}|k{k}|{g}", acc.get(g, np.zeros(4)))
        # ---- (가) 같은 공간 다음 걸음 (칸 단위)
        for s in NEXT_SP:
            z = sp[s]
            acc = {g: np.zeros(3 + len(ALPHAS) + 1) for g in masks1}
            for t in range(2, 7):
                y = z[t + 1]; v = z[t] - z[t - 1]
                preds = [z[t], 2 * z[t] - z[t - 1], z[: t + 1].mean(0)] + [z[t] + al * v for al in ALPHAS]
                errs = torch.stack([(p - y).abs().mean(-1) for p in preds])               # (P, 256)
                for g in masks1:
                    m = masks1[g][t]
                    if m.sum():
                        acc[g][:-1] += errs[:, m].sum(1).double().cpu().numpy(); acc[g][-1] += float(m.sum())
            for g in masks1:
                put(src, f"next|{s}|{g}", acc[g])
        # ---- (나) 실제 정답 h_{t+1}
        hb = sp["blk"]
        acc = {g: np.zeros(1 + len(DELTAS) * len(LAMS) * len(ALPHAS) + 1) for g in masks1}
        for t in range(2, 7):
            y = hb[t + 1]; mu = hb[: t + 1].mean(0)
            preds = [mu]
            for dk in DELTAS:
                for lam in LAMS:
                    base = lam * hb[t] + (1 - lam) * mu
                    preds += [base + al * dl[dk][t] for al in ALPHAS]
            errs = torch.stack([(p - y).abs().mean(-1) for p in preds])
            for g in masks1:
                m = masks1[g][t]
                if m.sum():
                    acc[g][:-1] += errs[:, m].sum(1).double().cpu().numpy(); acc[g][-1] += float(m.sum())
        for g in masks1:
            put(src, f"true|{g}", acc[g])
        # ---- (다) 창 토큰 ≈ 같은 칸 블록 토큰 가중합 (Gram 누적)
        for tgt in M4_T:
            for t in range(2, 6):
                y = sp[tgt][t]
                X = hb if tgt in ("twin", "owin") else hb[: t + 1]
                for g in masks1:
                    m = masks1[g][t]
                    if m.sum() == 0:
                        continue
                    Xm, ym = X[:, m], y[m]
                    key = (src, tgt, t, g, widx % 2)
                    d = m4acc.setdefault(key, dict(G=np.zeros((X.shape[0],) * 2), b=np.zeros(X.shape[0]), yy=0.0, n=0))
                    d["G"] += torch.einsum("knd,lnd->kl", Xm, Xm).double().cpu().numpy()
                    d["b"] += torch.einsum("knd,nd->k", Xm, ym).double().cpu().numpy()
                    d["yy"] += float((ym.double() ** 2).sum()); d["n"] += int(m.sum())

    def flow_masks(U):
        """U (8,16,16,2) 칸 단위 (U[0] = 0) → masks1/masks2 (mov · sta) (8,256) · 화면"""
        mg = torch.from_numpy(np.linalg.norm(U.reshape(8, S, 2), axis=-1)).to(dev)
        z = torch.zeros(8, S, dtype=torch.bool, device=dev)
        m1 = {"mov": z.clone(), "sta": z.clone()}; m2 = {"mov": z.clone(), "sta": z.clone()}
        sc1 = {"mov": np.zeros(8, bool), "sta": np.zeros(8, bool)}; sc2 = {"mov": np.zeros(8, bool), "sta": np.zeros(8, bool)}
        gm = mg.mean(1).cpu().numpy()
        for t in range(1, 7):
            m1["mov"][t] = (mg[t] >= 0.5) & (mg[t + 1] >= 0.5); m1["sta"][t] = (mg[t] < 0.1) & (mg[t + 1] < 0.1)
            sc1["mov"][t] = gm[t] >= 0.5 and gm[t + 1] >= 0.5; sc1["sta"][t] = gm[t] < 0.1 and gm[t + 1] < 0.1
        for t in range(2, 6):
            r = mg[t - 1: t + 3]
            m2["mov"][t] = (r >= 0.5).all(0); m2["sta"][t] = (r < 0.1).all(0)
            sc2["mov"][t] = bool((gm[t - 1: t + 3] >= 0.5).all()); sc2["sta"][t] = bool((gm[t - 1: t + 3] < 0.1).all())
        return m1, m2, sc1, sc2

    def decode(path, start, step, n=16):
        vr = decord.VideoReader(path, num_threads=1)
        idx = [start + step * k for k in range(n)]
        if idx[-1] >= len(vr):
            return None
        fr = vr.get_batch(idx).asnumpy()
        Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
        xx = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
        xx = F.interpolate(xx, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round() / 255.0   # (T,3,H,W)
        return (xx.permute(1, 0, 2, 3) - MEAN) / STD

    m4acc = {}; t0 = time.time()
    # ---- 자연 영상
    nat = []
    mk = json.load(open(f"{CACHE}/natural_holdout/k710_320_holdout_meta.json"))
    Uk = np.load(f"{CACHE}/natural_holdout/k710_320_holdout_U.npy").astype(np.float32)
    nat += [("k710", m["path"], m["start"], m["fstp"], Uk[i]) for i, m in enumerate(mk[: a.n_k710])]
    for src, n in (("ht100m_1pct", a.n_ht), ("ssv2", a.n_ssv2)):
        mm = json.load(open(f"{CACHE}/natural_train/{src}_meta.json")); Uu = np.load(f"{CACHE}/natural_train/{src}_Uflow.npy").astype(np.float32)
        nat += [(src.split("_")[0], m["path"], m["start"], m["fstp"], Uu[i]) for i, m in enumerate(mm[:n])]
    if a.limit:
        keep, cnt = [], {}
        for r in nat:
            cnt[r[0]] = cnt.get(r[0], 0) + 1
            if cnt[r[0]] <= a.limit:
                keep.append(r)
        nat = keep
    for wi, (src, path, st, step, U) in enumerate(nat):
        try:
            x = decode(path, st, step)
        except Exception as e:                                                        # noqa: BLE001
            print("decode 실패", path, e, flush=True); x = None
        if x is None:
            continue
        m1, m2, s1, s2 = flow_masks(U)
        window(src, x, m1, m2, s1, s2, m4acc, wi)
        if wi % 25 == 0:
            print(f"자연 {wi}/{len(nat)} ({time.time() - t0:.0f}s)", flush=True)
    # ---- 3D 합성 (검증 세트, 카메라 고정, 물체 1 개)
    cy, cx = np.divmod(np.arange(S), G); ccx, ccy = (cx + 0.5) * 16, (cy + 0.5) * 16
    wi = 0
    for sh in range(a.n_p3d_shards):
        PX = np.load(f"{CACHE}/phys3d/pix_val_s{sh}.npy", mmap_mode="r"); LB = np.load(f"{CACHE}/phys3d/lab_val_s{sh}.npz")
        n = PX.shape[0] if not a.limit else min(PX.shape[0], a.limit)
        for i in range(n):
            for w0 in (0, 16):
                fr = torch.from_numpy(np.asarray(PX[i, w0: w0 + 16])).to(dev).permute(3, 0, 1, 2).float() / 255.0
                x = (fr - MEAN) / STD
                f = [w0 + 2 * b + 1 for b in range(8)]
                bb = LB["bbox"][i, f]; hid = LB["hidden"][i, f].astype(bool) | ~LB["present"][i, f].astype(bool)
                ctr = np.stack([(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2], 1)
                inb = lambda b, pad=0.0: torch.from_numpy((ccx >= bb[b, 0] - pad) & (ccx <= bb[b, 2] + pad) & (ccy >= bb[b, 1] - pad) & (ccy <= bb[b, 3] + pad)).to(dev)
                zz = torch.zeros(8, S, dtype=torch.bool, device=dev)
                m1 = {"mov": zz.clone(), "still": zz.clone(), "bg": zz.clone()}; m2 = {"mov": zz.clone(), "still": zz.clone(), "bg": zz.clone()}
                for t in range(1, 7):
                    vis = not hid[t - 1: t + 2].any()
                    d1, d2 = np.linalg.norm(ctr[t] - ctr[t - 1]), np.linalg.norm(ctr[t + 1] - ctr[t])
                    if vis and d1 >= 8 and d2 >= 8:
                        m1["mov"][t] = inb(t)
                    if vis and d1 < 2 and d2 < 2:
                        m1["still"][t] = inb(t)
                    m1["bg"][t] = ~(inb(t - 1, 32) | inb(t, 32) | inb(t + 1, 32))
                for t in range(2, 6):
                    vis = not hid[t - 2: t + 3].any()
                    ds = np.linalg.norm(np.diff(ctr[t - 2: t + 3], axis=0), axis=1)
                    if vis and (ds >= 8).all():
                        m2["mov"][t] = inb(t)
                    if vis and (ds < 2).all():
                        m2["still"][t] = inb(t)
                    m2["bg"][t] = ~(inb(t - 2, 32) | inb(t - 1, 32) | inb(t, 32) | inb(t + 1, 32) | inb(t + 2, 32))
                sc = {"all": np.ones(8, bool)}
                window("phys3d", x, m1, m2, sc, sc, m4acc, wi); wi += 1
        print(f"3D shard {sh} · 창 {wi} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "per_window.npz"), **{f"{k[0]}||{k[1]}": np.stack(v) for k, v in rec.items()})
    m4 = {"|".join(map(str, k)): {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in d.items()} for k, d in m4acc.items()}
    json.dump(m4, open(os.path.join(OUT, "m4_gram.json"), "w"))
    print(f"끝 {time.time() - t0:.0f}s", flush=True)


# ============================================================================ 보고
def boot(per, fn, nb=1000, seed=0):
    """per (W, k) 창별 합 → fn(합) 의 값과 창 bootstrap 95 %"""
    rng = np.random.default_rng(seed); W = per.shape[0]
    v = fn(per.sum(0))
    bs = [fn(per[rng.integers(0, W, W)].sum(0)) for _ in range(nb)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return v, lo, hi


def report():
    P = np.load(os.path.join(OUT, "per_window.npz"))
    keys = list(P.keys())
    srcs = sorted({k.split("||")[0] for k in keys})
    groups = {"natural": [s for s in srcs if s != "phys3d"], "k710": ["k710"], "ht100m": ["ht100m"], "ssv2": ["ssv2"], "phys3d": ["phys3d"]}

    def get(gname, key):
        arrs = [P[f"{s}||{key}"] for s in groups[gname] if f"{s}||{key}" in P]
        return np.concatenate(arrs) if arrs else None
    out = {"curv": {}, "next": {}, "true": {}, "m4": {}}
    lines = []
    # 곡률
    for gname in groups:
        cg = ("mov", "sta") if gname != "phys3d" else ("mov", "still", "bg")
        for pk in POOLS:
            for k in (1, 2):
                for g in (cg if not (pk == "screen" and gname == "phys3d") else ("all",)):
                    for s in SPACES:
                        per = get(gname, f"curv|{s}|{pk}|k{k}|{g}")
                        if per is None or per[:, 3].sum() == 0:
                            continue
                        th, tl, thh = boot(per, lambda q: q[0] / q[3])
                        c = per[:, 1].sum() / per[:, 3].sum(); st = per[:, 2].sum() / per[:, 3].sum()
                        r = (1 + 2 * c) / (1 - c) if c < 1 else np.inf
                        out["curv"][f"{gname}|{pk}|k{k}|{g}|{s}"] = dict(theta=round(th, 2), ci=[round(tl, 2), round(thh, 2)], cos=round(c, 4), r=round(r, 3), step=round(st, 4), n=int(per[:, 3].sum()), W=int((per[:, 3] > 0).sum()))
    # (가) 같은 공간 다음 걸음
    for gname in groups:
        cg = ("mov", "sta") if gname != "phys3d" else ("mov", "still", "bg")
        for g in cg:
            for s in NEXT_SP:
                per = get(gname, f"next|{s}|{g}")
                if per is None or per[:, -1].sum() == 0:
                    continue
                ev, od = per[0::2], per[1::2]
                ba = int(np.argmin(ev[:, 3:-1].sum(0)))
                rel = lambda q, j: q[j] / q[0]
                o = dict(lin=boot(od, lambda q: rel(q, 1)), mean=boot(od, lambda q: rel(q, 2)), best=boot(od, lambda q: rel(q, 3 + ba)), alpha=ALPHAS[ba],
                         l1_copy=float(od[:, 0].sum() / od[:, -1].sum()), n=int(od[:, -1].sum()))
                out["next"][f"{gname}|{g}|{s}"] = {kk: ([round(float(x), 4) for x in vv] if isinstance(vv, tuple) else vv) for kk, vv in o.items()}
    # (나) 실제 정답
    na = len(ALPHAS)
    for gname in groups:
        cg = ("mov", "sta") if gname != "phys3d" else ("mov", "still", "bg")
        for g in cg:
            per = get(gname, f"true|{g}")
            if per is None or per[:, -1].sum() == 0:
                continue
            ev, od = per[0::2], per[1::2]
            ic = lambda di, li, ai: 1 + (di * len(LAMS) + li) * na + ai
            copy_i = ic(0, 0, ALPHAS.index(0.0))
            o = {"mean": boot(od, lambda q: q[0] / q[copy_i]), "l1_copy": float(od[:, copy_i].sum() / od[:, -1].sum()), "n": int(od[:, -1].sum())}
            # 블록만 최선 (λ, α on Δblk) · 각 Δ 의 최선
            for di, dk in enumerate(DELTAS):
                best = min(((li, ai) for li in range(len(LAMS)) for ai in range(na)), key=lambda p: ev[:, ic(di, *p)].sum())
                j = ic(di, *best)
                o[dk] = boot(od, lambda q, j=j: q[j] / q[copy_i]); o[dk + "_lam_alpha"] = [LAMS[best[0]], ALPHAS[best[1]]]
                bestc = min(range(na), key=lambda ai: ev[:, ic(di, 0, ai)].sum())          # λ = 1 (복사 + α Δ)
                j = ic(di, 0, bestc)
                o[dk + "_copy"] = boot(od, lambda q, j=j: q[j] / q[copy_i]); o[dk + "_copy_alpha"] = ALPHAS[bestc]
            bestm = min(range(len(LAMS)), key=lambda li: ev[:, ic(0, li, ALPHAS.index(0.0))].sum())
            j = ic(0, bestm, ALPHAS.index(0.0))
            o["mix"] = boot(od, lambda q, j=j: q[j] / q[copy_i]); o["mix_lam"] = LAMS[bestm]
            out["true"][f"{gname}|{g}"] = {kk: ([round(float(x), 4) for x in vv] if isinstance(vv, tuple) else vv) for kk, vv in o.items()}
    # (다)
    m4 = json.load(open(os.path.join(OUT, "m4_gram.json")))
    agg = {}
    for k, d in m4.items():
        src, tgt, t, g, sp = k.split("|")
        gn = "phys3d" if src == "phys3d" else "natural"
        for gg in (gn,):
            key = (gg, tgt, int(t), g, int(sp))
            e = agg.setdefault(key, dict(G=0, b=0, yy=0.0, n=0))
            e["G"] = e["G"] + np.array(d["G"]); e["b"] = e["b"] + np.array(d["b"]); e["yy"] += d["yy"]; e["n"] += d["n"]
    for gn in ("natural", "phys3d"):
        for tgt in M4_T:
            for g in ("mov", "sta", "still", "bg"):
                sse = sse1 = yy = 0.0; ws = []
                for t in range(2, 6):
                    tr, te = agg.get((gn, tgt, t, g, 0)), agg.get((gn, tgt, t, g, 1))
                    if tr is None or te is None:
                        continue
                    Gm, b = tr["G"], tr["b"]
                    w = np.linalg.solve(Gm + 1e-6 * np.eye(len(b)), b)
                    sse += te["yy"] - 2 * w @ te["b"] + w @ te["G"] @ w
                    j = t                                                               # 같은 자리 블록 하나
                    w1 = tr["b"][j] / tr["G"][j, j]
                    sse1 += te["yy"] - 2 * w1 * te["b"][j] + w1 * w1 * te["G"][j, j]
                    yy += te["yy"]; ws.append([round(float(x), 3) for x in w])
                if yy:
                    out["m4"][f"{gn}|{tgt}|{g}"] = dict(R2=round(1 - sse / yy, 4), R2_same=round(1 - sse1 / yy, 4), w=ws)
    json.dump(out, open(os.path.join(REPO, "auto_research/results/straightening.json"), "w"), indent=1)
    # 화면 출력
    for k, v in out["curv"].items():
        if "|cell|k1|" in k or "|screen|" in k or "|box3|k1|" in k or "|cell|k2|" in k:
            print(f"curv {k:40s} θ {v['theta']:6.1f} [{v['ci'][0]:.1f}, {v['ci'][1]:.1f}] cos {v['cos']:+.3f} r {v['r']:.2f} step {v['step']:.3f} n {v['n']} W {v['W']}")
    for k, v in out["next"].items():
        print(f"next {k:28s} lin {v['lin'][0]:.3f} mean {v['mean'][0]:.3f} best {v['best'][0]:.3f} (α {v['alpha']}) L1copy {v['l1_copy']:.3f} n {v['n']}")
    for k, v in out["true"].items():
        print(f"true {k:16s} " + " ".join(f"{dk} {v[dk][0]:.4f}[{v[dk][1]:.4f},{v[dk][2]:.4f}] λα{v[dk + '_lam_alpha']} | copy+αΔ {v[dk + '_copy'][0]:.4f} α{v[dk + '_copy_alpha']}" for dk in DELTAS) + f" | mix {v['mix'][0]:.4f} λ{v['mix_lam']} mean {v['mean'][0]:.4f} L1copy {v['l1_copy']:.3f} n {v['n']}")
    for k, v in out["m4"].items():
        print(f"m4 {k:24s} R² {v['R2']:.3f} (같은 자리 하나 {v['R2_same']:.3f}) w(t=3) {v['w'][1] if len(v['w']) > 1 else v['w']}")


if __name__ == "__main__":
    report() if a.report else run()
