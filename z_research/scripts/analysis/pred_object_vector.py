"""예측이 만든 물체 벡터는 encoder 의 물체 벡터와 어떻게 다른가, 그 차이 중 무엇이 decoder 를 망치나 (2026-10-25, 학습 없음).
사용자: "예측에서 어떤 표현들이 달라져서 decoder 가 잘 못 그리는지" · "물체를 표현하는 벡터가 어떻게 달라졌길래 encoder 표현과 비교해서".

데이터 · 창 · 모델 · 물체 칸 · 배경 기준은 pred_object_fidelity.py 와 같다 (phys3d_single_v1 96 장, 문맥 블록 0..7 → 미래 8..15,
블록 AR natural_ar_vw_state (블록 공간) · 릴리즈 predictor (32 장 창 공간), 각자 자기 정답 공간과 비교).
대상 = 미래 블록에서 물체가 보이고, 물체 칸 전부에 배경 기준 bg 가 있는 블록 (= 문맥에서 물체가 없던 새 자리).

  1 분해   물체 칸 c 마다 (모두 데이터 평균 토큰 μ 를 뺀 뒤) P_c ≈ a·Y_c + b·bg_c + k·Ō + r, 블록마다 칸 공통 계수 최소제곱.
           Y = 정답 물체 토큰, bg = 그 칸 배경, Ō = 마지막 문맥 블록 (7) 의 물체 칸 평균 (마지막에 본 물체). R² = 1 − ‖r‖²/‖P − μ‖².
  2 차이   Δ = P_c − Y_c 의 에너지가 어디에 있나: 물체–배경 방향 (Y_c − bg_c, 칸마다 1 차원) · 정체 부분공간 (모양 · 색 probe 활성 패턴, 12 차원)
           · 물체 토큰 주성분 대역 (train 정답 물체 토큰 PCA: 1–8 · 9–32 · 33–128 · 129–512 · 513–1280). 차원 비율 대비 몇 배인지 같이.
           대역별 에너지 비 E_k(P − μ) / E_k(Y − μ) (1 보다 작으면 그 대역이 줄었다).
  3 decoder 정답 블록의 물체 칸만 예측으로 바꾼 X₀ 를 기준으로, 물체 칸 벡터의 한 성분만 정답으로 고친 판을 decoder 로 그린다.
           물체 세기 s(X) = mean_상자 |dec(X) − dec(B)| / mean_상자 |dec(Y) − dec(B)|, B = 물체 칸을 bg 로 바꾼 정답 (물체 지운 그림).
           회복 = (s(고친 판) − s(X₀)) / (1 − s(X₀)).  고친 판: 물체–배경 계수 · 정체 부분공간 · 대역 5 개 · "섞임만" (aY + b bg + kŌ, 나머지 없음) · "정답 + 나머지만" (Y + r).
           참고: 예측 블록 전체 (배경 칸도 예측) 의 s.

  CUDA_VISIBLE_DEVICES=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/pred_object_vector.py --stage extract
  CUDA_VISIBLE_DEVICES=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/pred_object_vector.py --stage analyze
출력: /workspace/auto_research_cache/pred_objvec/ · auto_research/results/pred_object_vector.json
"""
import argparse, glob, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "z_pixel_decoder")); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["extract", "analyze", "all"], default="all")
ap.add_argument("--n_train", type=int, default=400); ap.add_argument("--n_val", type=int, default=0)
ap.add_argument("--ar", default="natural_ar_vw_state")
ap.add_argument("--decoder", default=os.path.join(REPO, "z_pixel_decoder/runs/vob_grav_i1_i2_vitb_both/latest.pt"))
a = ap.parse_args()
P3 = "/workspace/dataset/training_data/phys3d_single_v1"
CACHE = "/workspace/auto_research_cache/pred_objvec"
OUT = os.path.join(REPO, "auto_research/results/pred_object_vector.json")
T, C, S, G, D = 32, 16, 256, 16, 1280
NBC, NBF = C // 2, (T - C) // 2
IDX = [3 * k for k in range(T)]
BANDS = [(0, 8), (8, 32), (32, 128), (128, 512), (512, 1280)]
SPACES = ("ar", "rel")
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)


def clips(split, n):
    fs = sorted(glob.glob(f"{P3}/{split}/clips/*/*.mp4")); by = {}
    for f in fs:
        by.setdefault(f.split("/")[-2], []).append(f)
    rng = random.Random(1)
    for k in sorted(by):
        rng.shuffle(by[k])
    out = []
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
    pos, bb, vis, inf = st["pos"][:, 0], st["bbox"][:, 0], st["vis_frac"][:, 0], st["inframe_frac"][:, 0]
    cc = np.arange(G) * 16 + 8.0; out = []
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
        out.append({"ok": ok, "ctr": ctr.tolist(), "cells": cells, "box": box.tolist()})
    return out


def extract():
    import decord
    from analysis.intphys2.model import build_from_config, _build_encoder, _load_state_dict, _load_checkpoint
    from evals.world_model_analysis.eval import _context_target_indices
    _argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
    import ar_crps_ft as FT
    sys.argv = _argv
    dev = torch.device("cuda")
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    MCFG = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True,
            "uniform_power": False, "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
            "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32", "autocast": "float16", "window_size": 32}
    rel = build_from_config(MCFG, dev)
    enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.to(dev).eval()
    ar = FT.load_pred(a.ar, "latest.pt").to(dev).eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
    ci, ti = _context_target_indices(ctx_frames=C, tgt_frames=T - C, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev)
    LN = lambda t: F.layer_norm(t, (t.size(-1),))
    idx = torch.arange(NBC * S, device=dev).unsqueeze(0)
    for split, files in (("val", clips("val", a.n_val)), ("train", clips("train", a.n_train))):
        os.makedirs(f"{CACHE}/{split}", exist_ok=True)
        meta = []
        for i, f in enumerate(files):
            st = np.load(f[:-4] + "_states.npz"); lab = block_labels(st)
            fr = decord.VideoReader(f, num_threads=4).get_batch(IDX).asnumpy()
            x01 = torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0
            xn = ((x01 - MEAN) / STD).unsqueeze(0)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16):
                    z = rel.context_encoder(xn, masks=[ci]); p_rel = rel.predictor(z, ci, ti, mask_index=0); hw = rel.target_encoder(xn)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hb = LN(enc2(xn[0].view(3, T // 2, 2, 256, 256).permute(1, 0, 2, 3, 4)).float())          # (16, S, D)
                    p_ar = ar.rollout_kv(hb[:NBC].reshape(1, -1, D), idx, NBF).float()[0].view(NBF, S, D)
            Ys = {"ar": hb, "rel": LN(hw.float())[0].view(16, S, D)}
            Ps = {"ar": p_ar, "rel": p_rel.float()[0].view(NBF, S, D)}
            cov = [set(l["cells"]) if np.isfinite(l["box"]).all() else set() for l in lab]
            sv = {}
            for sp in SPACES:
                Y = Ys[sp]
                # 칸별 배경: 물체가 그 칸을 안 덮은 문맥 블록 평균 (없으면 NaN)
                bgm = torch.full((S, D), float("nan"), device=dev)
                for c in range(S):
                    qs = [q for q in range(NBC) if c not in cov[q]]
                    if qs:
                        bgm[c] = Y[qs, c].mean(0)
                last = lab[NBC - 1]
                obar = Y[NBC - 1, last["cells"]].mean(0) if (last["ok"] and last["cells"]) else torch.full((D,), float("nan"), device=dev)
                if split == "val":
                    sv[f"Y_{sp}"] = Y[NBC:].half().cpu().numpy(); sv[f"P_{sp}"] = Ps[sp].half().cpu().numpy()
                    sv[f"bg_{sp}"] = bgm.half().cpu().numpy(); sv[f"obar_{sp}"] = obar.float().cpu().numpy()
                else:   # PCA · 정체 probe 용: 보이는 블록의 물체 칸 정답 토큰 (칸 단위) + 물체 칸 평균 (블록 단위)
                    toks, pooled = [], []
                    for b in range(16):
                        if lab[b]["ok"] and lab[b]["cells"]:
                            toks.append(Y[b, lab[b]["cells"]].half().cpu().numpy()); pooled.append(Y[b, lab[b]["cells"]].mean(0).cpu().numpy())
                    sv[f"objtok_{sp}"] = np.concatenate(toks) if toks else np.zeros((0, D), np.float16)
                    sv[f"objpool_{sp}"] = np.stack(pooled) if pooled else np.zeros((0, D), np.float32)
            if split == "val":
                sv["frames"] = (x01[:, C:].permute(1, 0, 2, 3).cpu().numpy() * 255).astype(np.uint8)    # 미래 16 장
            np.savez(f"{CACHE}/{split}/{i:05d}.npz", **sv)
            meta.append({"clip": f, "shape": int(st["shape_id"][0]), "color": int(st["color_id"][0]), "labels": lab})
            if i % 50 == 0:
                log(split, f"{i}/{len(files)}")
        json.dump(meta, open(f"{CACHE}/{split}_meta.json", "w"))


# ------------------------------------------------------------------ 분석
def ridge(X, Y, lam):
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    return vt.T @ ((s / (s ** 2 + lam))[:, None] * (u.T @ Y))


def analyze():
    from model import PixelDecoder
    dev = torch.device("cuda")
    st_ = torch.load(a.decoder, map_location="cpu", weights_only=False); mc = st_["cfg"]["model"]
    dec = PixelDecoder(in_dim=D, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval(); dec.load_state_dict(st_["decoder"]); del st_
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    tmeta = json.load(open(f"{CACHE}/train_meta.json")); vmeta = json.load(open(f"{CACHE}/val_meta.json"))
    basis = {}
    for sp in SPACES:   # 물체 토큰 PCA + 정체 부분공간 (train)
        tr = [np.load(f"{CACHE}/train/{i:05d}.npz") for i in range(len(tmeta))]
        tok = np.concatenate([z[f"objtok_{sp}"] for z in tr]).astype(np.float32)
        mu = tok.mean(0); Xc = torch.from_numpy(tok - mu).to(dev)
        ev, U = torch.linalg.eigh(Xc.T @ Xc / len(Xc)); U = U[:, ev.argsort(descending=True)]
        pooled = np.concatenate([z[f"objpool_{sp}"] for z in tr]); lab_s = np.concatenate([[m["shape"]] * len(z[f"objpool_{sp}"]) for m, z in zip(tmeta, tr)])
        lab_c = np.concatenate([[m["color"]] * len(z[f"objpool_{sp}"]) for m, z in zip(tmeta, tr)])
        lab_s, lab_c = lab_s.astype(int), lab_c.astype(int)                 # 빈 목록이 섞이면 float 가 된다
        pm, ps = pooled.mean(0), pooled.std(0) + 1e-6; Xs = (pooled - pm) / ps; Sig = np.cov(Xs, rowvar=False)
        pats = []
        for y, K in ((lab_s, 6), (lab_c, 8)):
            Yo = np.eye(K, dtype=np.float32)[y]; W = ridge(Xs, Yo - Yo.mean(0), 1e2)
            pats.append((Sig @ W) / ps[:, None])                         # 표준화 공간 패턴 → 원 좌표 방향
        Qid, _ = torch.linalg.qr(torch.from_numpy(np.concatenate(pats, 1)).float().to(dev))
        Qid = Qid[:, :11]                                                # 모양 5 + 색 7 − 1 (중심화로 각 1 줄어든 열 공간; 12 열 중 독립 11)
        basis[sp] = {"mu": torch.from_numpy(mu).to(dev), "U": U, "Qid": Qid}
        log(sp, "PCA 토큰", len(tok), "물체 칸 평균", len(pooled))
    recs = []
    for i, m in enumerate(vmeta):
        z = np.load(f"{CACHE}/val/{i:05d}.npz"); lab = m["labels"]
        real = torch.from_numpy(z["frames"]).to(dev).float() / 255.0                  # (16, 3, H, W)
        for sp in SPACES:
            mu, U, Qid = basis[sp]["mu"], basis[sp]["U"], basis[sp]["Qid"]
            Yf = torch.from_numpy(z[f"Y_{sp}"]).to(dev).float(); Pf = torch.from_numpy(z[f"P_{sp}"]).to(dev).float()
            bgm = torch.from_numpy(z[f"bg_{sp}"]).to(dev).float(); obar = torch.from_numpy(z[f"obar_{sp}"]).to(dev).float()
            for j in range(NBF):
                b = NBC + j; L = lab[b]; cells = L["cells"]
                if not L["ok"] or not cells or torch.isnan(bgm[cells]).any() or torch.isnan(obar).any():
                    continue
                Y, P, B = Yf[j, cells], Pf[j, cells], bgm[cells]
                Yc, Pc, Bc, Oc = Y - mu, P - mu, B - mu, (obar - mu).expand_as(Y)
                # 1 분해 (칸 공통 계수)
                A = torch.stack([Yc.reshape(-1), Bc.reshape(-1), Oc.reshape(-1)], 1)
                coef = torch.linalg.lstsq(A, Pc.reshape(-1, 1)).solution[:, 0]
                fit = (A @ coef).view_as(Pc); res = Pc - fit
                # 같은 식을 바꿔 쓰면 P − μ ≈ (a+b)(bg − μ) + a (Y − bg) + k (Ō − μ): bg_keep = 배경 성분을 얼마나 남겼나, obj_part = 물체 몫 (Y − bg) 을 얼마나 그렸나
                r = {"clip": i, "space": sp, "step": j + 1, "a_gt": float(coef[0]), "b_bg": float(coef[1]), "k_last": float(coef[2]),
                     "obj_part": float(coef[0]), "bg_keep": float(coef[0] + coef[1]),
                     "R2": float(1 - (res ** 2).sum() / (Pc ** 2).sum()),
                     "dist": float(np.linalg.norm(np.array(L["ctr"]) - np.array(lab[NBC - 1]["ctr"]))) if lab[NBC - 1]["ok"] else float("nan")}
                # 2 차이 Δ 의 에너지 위치
                Dl = P - Y; E = (Dl ** 2).sum()
                ob = (Y - B) / (Y - B).norm(dim=1, keepdim=True).clamp_min(1e-6)
                r["dfrac_objdir"] = float(((Dl * ob).sum(1) ** 2).sum() / E)
                r["dfrac_id"] = float(((Dl @ Qid) ** 2).sum() / E)
                cP, cY = (P - mu) @ U, (Y - mu) @ U; cD = Dl @ U
                for lo, hi in BANDS:
                    r[f"dfrac_pc{lo + 1}_{hi}"] = float((cD[:, lo:hi] ** 2).sum() / E)
                    r[f"eratio_pc{lo + 1}_{hi}"] = float((cP[:, lo:hi] ** 2).sum() / (cY[:, lo:hi] ** 2).sum().clamp_min(1e-9))
                    r[f"cos_pc{lo + 1}_{hi}"] = float(F.cosine_similarity(cP[:, lo:hi].reshape(-1), cY[:, lo:hi].reshape(-1), dim=0))
                # 3 decoder
                def blk(obj):
                    X = Yf[j].clone(); X[cells] = obj; return X
                fixes = {"obj_dir": P + ((Y - P) * ob).sum(1, keepdim=True) * ob,
                         "identity": P + ((Y - P) @ Qid) @ Qid.T}
                for lo, hi in BANDS:
                    Ub = U[:, lo:hi]; fixes[f"pc{lo + 1}_{hi}"] = P + ((Y - P) @ Ub) @ Ub.T
                fixes["blend_only"] = fit + mu; fixes["gt_plus_resid"] = Y + res
                Xs = [Yf[j], blk(B), blk(P)] + [blk(v) for v in fixes.values()] + [Pf[j]]
                with torch.no_grad():
                    y_ = (dec(torch.stack(Xs)) * STD + MEAN).clamp(0, 1)                     # (n, 3, 2, H, W)
                x0, y0, x1, y1 = [int(np.clip(v, 0, 255)) for v in L["box"]]
                if x1 <= x0 or y1 <= y0:
                    continue
                reg = lambda t: t[..., y0:y1 + 1, x0:x1 + 1]
                d = lambda k: float((reg(y_[k]) - reg(y_[1])).abs().mean())
                dY = d(0)
                if dY < 1e-4:
                    continue
                s0 = d(2) / dY; r["s_pred_objcells"] = s0; r["s_pred_full"] = d(len(Xs) - 1) / dY
                for n_, k in zip(fixes, range(3, 3 + len(fixes))):
                    sk = d(k) / dY; r[f"s_{n_}"] = sk; r[f"rec_{n_}"] = (sk - s0) / max(1e-6, 1 - s0)
                rr = real[2 * j: 2 * j + 2].permute(1, 0, 2, 3)                                # (3, 2, H, W)
                r["pix_err_gt"] = float((reg(y_[0]) - reg(rr)).abs().mean()); r["pix_err_pred_objcells"] = float((reg(y_[2]) - reg(rr)).abs().mean())
                recs.append(r)
        if i % 50 == 0:
            log("analyze", f"{i}/{len(vmeta)}", len(recs))
    json.dump(recs, open(f"{CACHE}/records.json", "w"))
    summ = {}
    keys = [k for k in recs[0] if k not in ("clip", "space", "step", "dist")]
    for sp in SPACES:
        R = [r for r in recs if r["space"] == sp]
        summ[sp] = {"n": len(R), "by_step": {}}
        for k in range(1, NBF + 1):
            Rk = [r for r in R if r["step"] == k]
            if Rk:
                summ[sp]["by_step"][k] = {"n": len(Rk), **{key: round(float(np.median([r[key] for r in Rk if key in r])), 3) for key in keys if any(key in r for r in Rk)}}
        summ[sp]["all"] = {key: round(float(np.median([r[key] for r in R if key in r])), 3) for key in keys}
        summ[sp]["dim_share"] = {"obj_dir (칸마다 1/1280)": round(1 / 1280, 4), "identity 11/1280": round(11 / 1280, 4),
                                 **{f"pc{lo + 1}_{hi}": round((hi - lo) / 1280, 4) for lo, hi in BANDS}}
    json.dump(summ, open(OUT, "w"), indent=1, ensure_ascii=False)
    log("saved", OUT)


if __name__ == "__main__":
    if a.stage in ("extract", "all"):
        extract()
    if a.stage in ("analyze", "all"):
        analyze()
