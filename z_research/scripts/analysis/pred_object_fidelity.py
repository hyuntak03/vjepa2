"""예측이 만든 물체는 encoder 표현과 얼마나 다른가 — 릴리즈 predictor vs 블록 AR, 같은 데이터 · 같은 창 · 같은 지표 (2026-10-25, 학습 없음).
사용자: "release, ar predictor 가 만들어낸 물체 정보 자체는 encoder 표현이랑 얼마나 달라? pixel decoder 에선 왜 물체 표현 못하는거지.
둘 다 예측 초반에만 유지하다가 사라지더라" → "다시 실험하라".

데이터: phys3d_single_v1 (물체 1 개 · 고정 카메라 · 장마다 위치 · 상자 · 보임 라벨), 96 장 영상만. 원본 0, 3, …, 93 장 (4 fps) 32 장 = 블록 16.
  판독기 학습 = train 1,600 편 (시나리오마다 고르게), 시험 = val 전부 (두 모델 모두 이 세트로 학습한 적 없다).
창: 문맥 블록 0..7 (16 장) → 미래 블록 8..15 (걸음 1..8). 두 모델 같은 창.
  릴리즈  : context_encoder(문맥 16 장) → predictor (마스크 토큰, 한 번에) → 미래 토큰. 정답 = LN(target_encoder(32 장 한 번에)) 의 미래 블록 (채점 surprise_c16t32 와 같다)
  블록 AR : 블록마다 LN(target_encoder(2 장)) → natural_ar_vw_state 상태 되먹임 rollout 8 걸음. 정답 = 미래 블록의 블록 토큰
  각 모델은 **자기 정답 공간** 과 비교한다.
물체 칸 O_j = 블록 j 두 장의 물체 상자를 합친 상자 안에 중심이 든 칸 (없으면 중심 칸). 미래 블록은 물체가 보일 때만 (vis_frac ≥ 0.5 · 화면 안 ≥ 0.9).
배경 기준 bg_c = 문맥 블록 가운데 물체가 칸 c 를 안 덮은 블록들의 정답 토큰 평균. O_j 의 모든 칸에 bg 가 있어야 진하기 · decoder 측정에 쓴다 (= 물체가 문맥에서 안 있던 새 자리).
  ⚠️ 릴리즈 정답 공간은 32 장을 한 번에 넣어 문맥 블록 토큰에도 미래 물체가 섞일 수 있다 → 릴리즈 bg 는 덜 깨끗하다 (진하기 분모가 작아질 수 있음).

측정 (걸음 1..8, 그리고 마지막 문맥 위치에서 떨어진 거리별):
  A1 진하기 (물체 칸) = Σ⟨P − bg, Y − bg⟩ / Σ‖Y − bg‖²   1 = 정답만큼, 0 = 배경 그대로
  A2 잔상 (옛 자리 = 블록 7 물체 칸 중 지금 물체가 없는 칸) = Σ⟨P − bg, Y₇ − bg⟩ / Σ‖Y₇ − bg‖²   1 = 물체가 그대로 남음, 0 = 지움
  A3 코사인 (물체 칸 · 배경 칸 따로) cos(P_c, Y_c)
  A4 진하기 ~ 걸음 + 거리 회귀 (편 단위 bootstrap) — 무엇이 사라짐을 정하나
  B  정체 판독 (물체 칸 평균 → 표준화 → ridge, 모양 6 · 색 8): 정답→정답 (상한) · 정답→예측 (encoder 와 같은 형식인가) · 예측→예측 (정보가 있나)
  C  decoder (z_pixel_decoder vob_grav_i1_i2_vitb_both, 블록 · 창 latent 둘 다로 학습): 물체 상자 안 픽셀 세기
       s(X) = mean_R |dec(X) − dec(B)| / mean_R |dec(Y) − dec(B)|,  B = Y 에서 O_j 칸만 bg 로 바꾼 것 (물체 지운 그림)
       X = B + α (Y − B) (α 0.25 · 0.5 · 0.75) → decoder 응답 곡선 / X = B 에 O_j 칸만 예측으로 (물체 칸만) / X = 예측 전체

  CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/pred_object_fidelity.py
출력: /workspace/auto_research_cache/pred_fidelity/{train,val}.npz · auto_research/results/pred_object_fidelity.json
"""
import argparse, glob, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "z_pixel_decoder")); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=1600); ap.add_argument("--n_val", type=int, default=0)
ap.add_argument("--ar", default="natural_ar_vw_state")
ap.add_argument("--decoder", default=os.path.join(REPO, "z_pixel_decoder/runs/vob_grav_i1_i2_vitb_both/latest.pt"))
ap.add_argument("--probe_only", action="store_true")
a = ap.parse_args()
P3 = "/workspace/dataset/training_data/phys3d_single_v1"
CACHE = "/workspace/auto_research_cache/pred_fidelity"
OUT = os.path.join(REPO, "auto_research/results/pred_object_fidelity.json")
T, C, S, G, D = 32, 16, 256, 16, 1280
NBC, NBF = C // 2, (T - C) // 2                                   # 문맥 블록 8 · 미래 블록 8
IDX = [3 * k for k in range(T)]
ALPHAS = (0.25, 0.5, 0.75)
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)


def clips(split, n):
    fs = sorted(glob.glob(f"{P3}/{split}/clips/*/*.mp4"))
    by = {}
    for f in fs:
        by.setdefault(f.split("/")[-2], []).append(f)
    rng = random.Random(0); out = []
    for k in sorted(by):
        rng.shuffle(by[k])
    cap = None if n == 0 else max(1, n // len(by)) * 3
    while (n == 0 or len(out) < n) and any(by.values()):
        for k in sorted(by):
            while by[k]:
                f = by[k].pop()
                if np.load(f[:-4] + "_states.npz")["pos"].shape[0] >= 96:
                    out.append(f); break
            if n and len(out) >= n:
                break
    return out


def block_labels(st):
    """블록마다 (보임, 중심 x, y, 물체 칸 목록)."""
    pos, bb, vis, inf = st["pos"][:, 0], st["bbox"][:, 0], st["vis_frac"][:, 0], st["inframe_frac"][:, 0]
    cc = (np.arange(G) * 16 + 8.0)
    out = []
    for b in range(T // 2):
        f0, f1 = IDX[2 * b], IDX[2 * b + 1]
        ok = bool(min(vis[f0], vis[f1]) >= 0.5 and min(inf[f0], inf[f1]) >= 0.9)
        box = np.array([min(bb[f0, 0], bb[f1, 0]), min(bb[f0, 1], bb[f1, 1]), max(bb[f0, 2], bb[f1, 2]), max(bb[f0, 3], bb[f1, 3])])
        cells = []
        if np.isfinite(box).all():
            cols = np.nonzero((cc >= box[0]) & (cc <= box[2]))[0]; rows = np.nonzero((cc >= box[1]) & (cc <= box[3]))[0]
            cells = [int(r * G + c) for r in rows for c in cols]
        ctr = 0.5 * (pos[f0] + pos[f1])
        if not cells and np.isfinite(ctr).all():
            cells = [int(np.clip(ctr[1] // 16, 0, G - 1) * G + np.clip(ctr[0] // 16, 0, G - 1))]
        out.append((ok, ctr, cells, box))
    return out


def extract(split, files, with_decoder):
    import decord, yaml
    from analysis.intphys2.model import build_from_config, _build_encoder, _load_state_dict, _load_checkpoint
    from evals.world_model_analysis.eval import _context_target_indices
    _argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
    import ar_crps_ft as FT
    sys.argv = _argv
    from model import PixelDecoder
    dev = torch.device("cuda")
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    MCFG = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True,
            "uniform_power": False, "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
            "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32", "autocast": "float16", "window_size": 32}
    rel = build_from_config(MCFG, dev)
    enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.to(dev).eval()
    ar = FT.load_pred(a.ar, "latest.pt").to(dev).eval()
    if with_decoder:
        st_ = torch.load(a.decoder, map_location="cpu", weights_only=False); mc = st_["cfg"]["model"]
        dec = PixelDecoder(in_dim=D, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval(); dec.load_state_dict(st_["decoder"]); del st_
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
    ci, ti = _context_target_indices(ctx_frames=C, tgt_frames=T - C, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev)
    LN = lambda t: F.layer_norm(t, (t.size(-1),))
    idx = torch.arange(NBC * S, device=dev).unsqueeze(0)
    R = {k: [] for k in ("clip", "shape", "color", "scen")}
    rec = []                                                        # 블록 단위 기록
    pooled = {k: [] for k in ("gt_blk", "gt_win", "p_ar", "p_rel")}   # (편, 블록 16, D) 물체 칸 평균 (없으면 NaN)
    for i, f in enumerate(files):
        st = np.load(f[:-4] + "_states.npz"); lab = block_labels(st)
        fr = decord.VideoReader(f, num_threads=4).get_batch(IDX).asnumpy()
        x01 = torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0
        xn = ((x01 - MEAN) / STD).unsqueeze(0)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                z = rel.context_encoder(xn, masks=[ci]); p_rel = rel.predictor(z, ci, ti, mask_index=0); hw = rel.target_encoder(xn)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                blk = enc2(xn[0].view(3, T // 2, 2, 256, 256).permute(1, 0, 2, 3, 4))
                hb = LN(blk.float()).reshape(-1, D)
                p_ar = ar.rollout_kv(hb[: NBC * S].unsqueeze(0), idx, NBF).float()[0]
        SP = {"ar": (hb.view(16, S, D), torch.cat([hb[: NBC * S], p_ar]).view(16, S, D)),
              "rel": (LN(hw.float())[0].view(16, S, D), torch.cat([LN(hw.float())[0][: NBC * S], p_rel.float()[0]]).view(16, S, D))}
        # 물체 칸 평균 (판독기용)
        for nm, key_gt, key_p in (("ar", "gt_blk", "p_ar"), ("rel", "gt_win", "p_rel")):
            Y, P = SP[nm]
            pg = np.full((16, D), np.nan, np.float32); pp = np.full((16, D), np.nan, np.float32)
            for b in range(16):
                if lab[b][0] and lab[b][2]:
                    pg[b] = Y[b, lab[b][2]].mean(0).cpu().numpy()
                    if b >= NBC:
                        pp[b] = P[b, lab[b][2]].mean(0).cpu().numpy()
            pooled[key_gt].append(pg); pooled[key_p].append(pp)
        # 블록별 진하기 · 잔상 · 코사인 · decoder
        cov = [set(l[2]) if np.isfinite(l[3]).all() else set() for l in lab]
        last = lab[NBC - 1]
        all_obj = set().union(*cov)
        bgcells = [c for c in range(S) if c not in all_obj]
        for b in range(NBC, 16):
            ok, ctr, cells, box = lab[b]
            if not ok or not cells:
                continue
            k = b - NBC + 1
            dist = float(np.linalg.norm(ctr - last[1])) if (last[0] and np.isfinite(last[1]).all()) else float("nan")
            r = {"clip": i, "block": b, "step": k, "dist": dist}
            for nm in ("ar", "rel"):
                Y, P = SP[nm]
                bgok = all(any(c not in cov[q] for q in range(NBC)) for c in cells)
                cosO = float(F.cosine_similarity(P[b, cells], Y[b, cells], dim=-1).mean())
                cosB = float(F.cosine_similarity(P[b, bgcells], Y[b, bgcells], dim=-1).mean()) if bgcells else float("nan")
                r[f"{nm}_cos_obj"], r[f"{nm}_cos_bg"] = cosO, cosB
                if not bgok:
                    continue
                bg = torch.stack([Y[[q for q in range(NBC) if c not in cov[q]], c].mean(0) for c in cells])
                y, p = Y[b, cells] - bg, P[b, cells] - bg
                r[f"{nm}_presence"] = float((p * y).sum() / (y * y).sum())
                old = [c for c in (last[2] if last[0] else []) if c not in cells and any(c not in cov[q] for q in range(NBC))]
                if old:
                    bgo = torch.stack([Y[[q for q in range(NBC) if c not in cov[q]], c].mean(0) for c in old])
                    yo, po = Y[NBC - 1, old] - bgo, P[b, old] - bgo
                    r[f"{nm}_residue"] = float((po * yo).sum() / (yo * yo).sum())
                if with_decoder:
                    Bt = Y[b].clone(); Bt[cells] = bg
                    Xs = [Y[b], Bt] + [Bt + al * (Y[b] - Bt) for al in ALPHAS]
                    Po = Bt.clone(); Po[cells] = P[b, cells]
                    Xs += [Po, P[b]]
                    with torch.no_grad():
                        y_ = dec(torch.stack(Xs).float())                     # (n, 3, 2, H, W)
                    x0, y0, x1, y1 = [int(np.clip(v, 0, 255)) for v in box]
                    if x1 <= x0 or y1 <= y0:
                        continue
                    reg = lambda t: t[:, :, :, y0:y1 + 1, x0:x1 + 1]
                    dY = (reg(y_[0:1]) - reg(y_[1:2])).abs().mean()
                    if float(dY) < 1e-4:
                        continue
                    sc = [(reg(y_[j:j + 1]) - reg(y_[1:2])).abs().mean() / dY for j in range(len(Xs))]
                    for j, al in enumerate(ALPHAS):
                        r[f"{nm}_dec_alpha{al}"] = float(sc[2 + j])
                    r[f"{nm}_dec_pred_objcells"] = float(sc[2 + len(ALPHAS)]); r[f"{nm}_dec_pred_full"] = float(sc[3 + len(ALPHAS)])
            rec.append(r)
        R["clip"].append(f); R["shape"].append(int(st["shape_id"][0])); R["color"].append(int(st["color_id"][0])); R["scen"].append(f.split("/")[-2])
        if i % 100 == 0:
            log(split, f"{i}/{len(files)}")
    os.makedirs(CACHE, exist_ok=True)
    np.savez(f"{CACHE}/{split}.npz", **{k: np.stack(v) for k, v in pooled.items()}, shape=np.array(R["shape"]), color=np.array(R["color"]),
             scen=np.array(R["scen"]), clip=np.array(R["clip"]))
    json.dump(rec, open(f"{CACHE}/{split}_blocks.json", "w"))


# ------------------------------------------------------------------ 분석
def ridge(X, Y, lams=(1e0, 1e1, 1e2, 1e3, 1e4)):
    n = X.shape[0]; fold = np.arange(n) % 5; best = None
    for lam in lams:
        e = 0.0
        for f in range(5):
            tr = fold != f; u, s, vt = np.linalg.svd(X[tr], full_matrices=False)
            W = vt.T @ ((s / (s ** 2 + lam))[:, None] * (u.T @ Y[tr])); e += ((X[~tr] @ W - Y[~tr]) ** 2).sum()
        if best is None or e < best[0]:
            best = (e, lam)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    return vt.T @ ((s / (s ** 2 + best[1]))[:, None] * (u.T @ Y))


def probes(TR, VA):
    out = {}
    for nm, kg, kp in (("ar", "gt_blk", "p_ar"), ("rel", "gt_win", "p_rel")):
        res = {}
        for t, K in (("shape", 6), ("color", 8)):
            oh = lambda y: np.eye(K, dtype=np.float32)[y]
            # 정답 판독기: train 의 보이는 모든 블록 (0..15) 정답
            Xg = TR[kg].reshape(-1, D); yg = np.repeat(TR[t], 16); m = ~np.isnan(Xg).any(1)
            mu, sd = Xg[m].mean(0), Xg[m].std(0) + 1e-6
            Wg = ridge((Xg[m] - mu) / sd, oh(yg[m]) - oh(yg[m]).mean(0)); bg_ = oh(yg[m]).mean(0)
            # 예측 판독기: train 의 미래 블록 예측
            Xp = TR[kp].reshape(-1, D); mp = ~np.isnan(Xp).any(1)
            mup, sdp = Xp[mp].mean(0), Xp[mp].std(0) + 1e-6
            Wp = ridge((Xp[mp] - mup) / sdp, oh(yg[mp]) - oh(yg[mp]).mean(0)); bp_ = oh(yg[mp]).mean(0)
            per = {}
            for k in range(1, NBF + 1):
                b = NBC + k - 1
                Gv, Pv, yv = VA[kg][:, b], VA[kp][:, b], VA[t]
                mm = ~np.isnan(Pv).any(1) & ~np.isnan(Gv).any(1)
                acc = lambda X, W, mu_, sd_, b0: round(100 * float((((X[mm] - mu_) / sd_ @ W + b0).argmax(1) == yv[mm]).mean()), 1)
                per[k] = {"n": int(mm.sum()), "gt->gt": acc(Gv, Wg, mu, sd, bg_), "gt->pred": acc(Pv, Wg, mu, sd, bg_), "pred->pred": acc(Pv, Wp, mup, sdp, bp_)}
            res[t] = per
        out[nm] = res
    return out


def summarize(rec):
    import collections
    out = {}
    for nm in ("ar", "rel"):
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rec:
            for key in ("presence", "residue", "cos_obj", "cos_bg", "dec_pred_objcells", "dec_pred_full") + tuple(f"dec_alpha{al}" for al in ALPHAS):
                v = r.get(f"{nm}_{key}")
                if v is not None and np.isfinite(v):
                    by[key][r["step"]].append(v)
        out[nm] = {key: {k: [round(float(np.median(v)), 3), round(float(np.mean(v)), 3), len(v)] for k, v in sorted(d.items())} for key, d in by.items()}
        # 거리별 진하기
        dd = collections.defaultdict(list)
        for r in rec:
            v = r.get(f"{nm}_presence")
            if v is not None and np.isfinite(r["dist"]):
                dd[min(int(r["dist"] // 16), 5)].append(v)
        out[nm]["presence_by_dist_cells"] = {f"{k}–{k + 1} 칸" if k < 5 else "5 칸+": [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(dd.items())}
        # 진하기 ~ 걸음 + 거리 (100 px), 편 단위 bootstrap
        rows = [(r["clip"], r["step"], r["dist"] / 100, r[f"{nm}_presence"]) for r in rec if r.get(f"{nm}_presence") is not None and np.isfinite(r["dist"])]
        if len(rows) > 20:
            A = np.array(rows, dtype=np.float64); cl = A[:, 0].astype(int); u = np.unique(cl)
            fit = lambda M: np.linalg.lstsq(np.c_[np.ones(len(M)), M[:, 1], M[:, 2]], M[:, 3], rcond=None)[0]
            b0 = fit(A); rng = np.random.default_rng(0); bs = []
            for _ in range(300):
                pick = rng.choice(u, len(u)); M = np.concatenate([A[cl == c] for c in pick]); bs.append(fit(M))
            bs = np.array(bs)
            out[nm]["presence_regression"] = {"per_step": [round(b0[1], 3), *np.round(np.percentile(bs[:, 1], [2.5, 97.5]), 3).tolist()],
                                              "per_100px": [round(b0[2], 3), *np.round(np.percentile(bs[:, 2], [2.5, 97.5]), 3).tolist()],
                                              "corr_step_dist": round(float(np.corrcoef(A[:, 1], A[:, 2])[0, 1]), 3), "n": len(rows)}
    return out


if __name__ == "__main__":
    if not a.probe_only:
        tr, va = clips("train", a.n_train), clips("val", a.n_val)
        log("train", len(tr), "val", len(va))
        extract("val", va, True)
        extract("train", tr, False)
    TR, VA = dict(np.load(f"{CACHE}/train.npz")), dict(np.load(f"{CACHE}/val.npz"))
    rec = json.load(open(f"{CACHE}/val_blocks.json"))
    res = {"blocks": summarize(rec), "identity": probes(TR, VA), "n_val_clips": int(len(VA["shape"])), "n_train_clips": int(len(TR["shape"]))}
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    log("saved", OUT)
