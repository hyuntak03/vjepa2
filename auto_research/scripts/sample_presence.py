#!/usr/bin/env python3
"""[P] 분포 예측 predictor 의 표본별 측정 (auto_research, 2026-10-21) — [T] 의 "공정 CRPS + 잡음 입력" 체크포인트용으로 미리 준비.

  # 결정적 모델 (L1 이어 학습 비교군): 표본 K 개가 모두 같다
  CUDA_VISIBLE_DEVICES=4 /venv/vjepa/bin/python -W ignore auto_research/scripts/sample_presence.py --run natural_ar_vw_state latest.pt --K 1
  # 분포 모델: 표본 함수 sample(model, x, idx, K, gen) -> (K, B, N_out, D) 를 모듈:함수 로 넘긴다 (x, idx = forward_seq 와 같은 입력)
  ... --sampler auto_research.scripts.<모듈>:sample --K 16

3D 합성 검증 세트, TF 한 걸음 (t = 7..14, 문맥 전체 0..t), 움직인 쌍 (이동 > 8 px, 두 블록 보임).
표본마다 세 지표 (같은 문맥 · 잡음만 다름):
  존재감_선형 (새 칸) · 잔상_선형 (떠난 칸) — pred_mean_solution.py 와 같은 물체 점수 비 (블록 t 템플릿, 먼 칸 대비)
  사영 존재감 새 · 옛 — proj_presence.py ([E] 정의, 기준 = 물체가 안 닿은 문맥 블록 평균)
  운반 계수 β (새 칸, 옮긴 복사 회귀, 표본마다 따로 적합 / 정답 β 로 나눈 비)
출력: 표본별 값의 분포 (히스토그램 · 두 봉우리 계수 b = (왜도² + 1) / (첨도 + 3(n−1)²/((n−2)(n−3))) · 0.2 미만 / 0.8 초과 비율),
      표본 평균 예측 (K 개 평균) 의 같은 지표, 쌍별 (새, 옛) 산점의 상관 (두 갈래면 음의 상관 · 합 ≈ 1)."""
import argparse, glob, importlib, json, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _clean_backbone_key                       # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, required=True)
ap.add_argument("--sampler", default=None, help="모듈:함수. 없으면 결정적 forward_seq 를 K 번")
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--strict", type=int, default=1, help="state_dict strict 로드")
ap.add_argument("--ft", default=None, help="[T] ar_crps_ft.py 체크포인트 (/workspace/auto_research_cache/ar_ft/<name>.pt). 주면 --run 은 구조 (config) 만 쓴다")
ap.add_argument("--xi", type=int, default=0, help="ft 가 crps 면 표본에 블록 고유 잡음 s ⊙ ξ 도 더한다 (1) / g(H, z) 만 (0)")
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/sample_presence"))
a = ap.parse_args()
dev = torch.device("cuda")
torch.set_num_threads(8)                                                     # CPU 제한 (사용자 학습과 같이 돈다)
S, G, C, NB, L = 256, 16, 8, 16, 8
tag = (f"ft_{os.path.basename(a.ft)[:-3]}__xi{a.xi}" if a.ft else f"{a.run[0]}__{a.run[1].replace('.pt', '')}") + f"__K{a.K}"
os.makedirs(a.out, exist_ok=True)
D_ = os.path.join(REPO, "z_training/runs", a.run[0])
pc = yaml.safe_load(open(os.path.join(D_, "config.yaml")))["model"]["predictor"]
kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
          num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
for k_, v_ in pc.items():                                                    # 새 키 (잡음 입력 등) 는 config 에서 그대로 넘긴다
    if k_ not in ("kind",) and k_ not in kw and not isinstance(v_, (dict, list)):
        kw[k_] = v_
model = vit_prefix_predictor(**kw)
FT = torch.load(a.ft, map_location="cpu", weights_only=False) if a.ft else None
src = FT["predictor"] if FT else torch.load(os.path.join(D_, a.run[1]), map_location="cpu", weights_only=False)["predictor"]
sd = {k: v for k, v in _clean_backbone_key(src).items() if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
print(model.load_state_dict(sd, strict=bool(a.strict)))
model = model.to(dev).eval()
gen = torch.Generator(device=dev); gen.manual_seed(0)
if FT is not None:
    fa = FT["args"]; nz = int(fa.get("nz", 32))
    proj_z = torch.nn.Linear(nz, 384)
    if "noise_proj" in FT:
        proj_z.load_state_dict(FT["noise_proj"])
    proj_z = proj_z.to(dev).eval()
    s_xi = FT["log_s"].to(dev).exp() if "log_s" in FT else torch.zeros(1280, device=dev)
    fa.setdefault("loss", "l1"); fa.setdefault("xi", "none")
    stoch = fa["loss"] == "crps"
    CODES = FT.get("codes").to(dev) if fa["loss"] == "wta" else None
    if CODES is not None and a.K != CODES.shape[0]:
        print(f"[ft] 다중 가설: K 를 가설 수 {CODES.shape[0]} 로 맞춘다"); a.K = CODES.shape[0]
    _orig = model._ar_embed_obs
    ZV = {"v": None}
    def _emb(x):
        e = _orig(x)
        return e if ZV["v"] is None else e + ZV["v"][:, None, :].to(e.dtype)
    model._ar_embed_obs = _emb
    print(f"[ft] {a.ft} loss {fa['loss']} nz {nz} ξ {fa.get('xi')} 척도 평균 {float(s_xi.mean()):.3f} · 표본 {'확률' if stoch else '결정 (L1 대조군)'}")

    def SAMPLE(model, x, idx, K, gen):
        outs = []
        for k in range(K):
            ZV["v"] = proj_z(torch.randn(1, nz, device=dev, generator=gen)) if stoch else (CODES[k][None] if CODES is not None else None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model.forward_seq(x, idx).float()
            ZV["v"] = None
            if stoch and a.xi and fa.get("xi", "none") != "none":
                if fa["xi"] == "gauss":
                    p = p + s_xi * torch.randn(p.shape, device=dev, generator=gen)
                else:
                    u = torch.rand(p.shape, device=dev, generator=gen) - 0.5
                    p = p + s_xi * (-torch.sign(u) * torch.log1p(-2 * u.abs())) / np.sqrt(2)
            outs.append(p)
        return torch.stack(outs)
elif a.sampler:
    mod, fn = a.sampler.split(":"); SAMPLE = getattr(importlib.import_module(mod), fn)
else:
    def SAMPLE(model, x, idx, K, gen):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model.forward_seq(x, idx).float()
        return p[None].expand(K, *p.shape)


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


def dil(m, k=1):
    m = m.reshape(G, G).copy()
    for _ in range(k):
        o = m.copy()
        o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
        o[1:, 1:] |= m[:-1, :-1]; o[:-1, :-1] |= m[1:, 1:]; o[1:, :-1] |= m[:-1, 1:]; o[:-1, 1:] |= m[1:, :-1]
        m = o
    return m.reshape(-1)


def proj(T, B_, R_, cells):
    num = ((T - B_) * (R_ - B_)).sum(-1); den = ((R_ - B_) ** 2).sum(-1).clamp_min(1e-6)
    return (num / den)[..., cells].mean(-1)


REC = []
WIN = []
nwin = 0
for f in sorted(glob.glob(os.path.join(a.cache, "h_val_target_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    X = np.load(f, mmap_mode="r"); lab = np.load(os.path.join(a.cache, f"lab_val_s{sh}.npz"))
    for w in range(X.shape[0]):
        if a.limit and nwin >= a.limit:
            break
        cv = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0))
        cmax = np.maximum(cv[0::2], cv[1::2]).reshape(NB, S)
        objb = cmax >= 0.25; touch = cmax > 0
        ok = lab["inframe_frac"][w] > 0.9
        visb = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & ok[0::2] & ok[1::2]
        far = ~dil(touch.any(0), 2)
        pos = lab["pos"][w]
        ts = [t for t in range(C - 1, NB - 1) if visb[t] and visb[t + 1] and np.isfinite(pos[[2 * t + 1, 2 * t + 3]]).all()
              and np.linalg.norm(pos[2 * t + 3] - pos[2 * t + 1]) > 8 and far.sum() >= 10]
        if not ts:
            continue
        nwin += 1
        h = torch.from_numpy(np.asarray(X[w], dtype=np.float32)).to(dev)
        with torch.no_grad():
            P = SAMPLE(model, h[:NB - 1].reshape(1, -1, h.size(-1)), torch.arange((NB - 1) * S, device=dev).unsqueeze(0), a.K, gen)
        P = P[:, 0].view(a.K, NB - 1, S, -1).float()                              # (K, 15, S, D)
        farm = torch.from_numpy(far).to(dev)
        eK = (P[:, C - 1:] - h[C:][None]).abs().mean((1, 2, 3))                     # (K,) 창 전체 TF 걸음 (블록 8..15) L1 → 승자
        WIN.append(int(eK.argmin()))
        for t in ts:
            new_l = objb[t + 1] & ~objb[t]; old_l = objb[t] & ~objb[t + 1]
            new_p = (cmax[t + 1] > 0.3) & (cmax[t] < 0.05); old_p = (cmax[t] > 0.3) & (cmax[t + 1] < 0.05)
            free = ~touch[t - L + 1:t + 1]; cnt = free.sum(0)
            new_p &= cnt > 0; old_p &= cnt > 0
            if not (new_l.any() and old_l.any()):
                continue
            Pt = P[:, t]                                                           # (K, S, D)
            Pm = Pt.mean(0, keepdim=True)
            T_hat = F.normalize(h[t][torch.from_numpy(objb[t]).to(dev)].mean(0), dim=-1)
            sc = lambda x: (x @ T_hat) - (x @ T_hat)[..., farm].mean(-1, keepdim=True)
            s_p, s_m, s_true, s_copy = sc(Pt), sc(Pm), sc(h[t + 1]), sc(h[t])
            nl, ol = torch.from_numpy(new_l).to(dev), torch.from_numpy(old_l).to(dev)
            rec = dict(t=t, win=f"{sh}:{w}")
            dn_, do_ = float(s_true[nl].mean() - s_copy[nl].mean()), float(s_copy[ol].mean() - s_true[ol].mean())
            if dn_ > 2.0:                                                          # 분모가 작은 걸음은 비가 불안정 → 뺀다 (대비 단위, 물체 대비 ≈ 13)
                rec.update(lin_new=((s_p[:, nl].mean(-1) - s_copy[nl].mean()) / dn_).cpu().numpy(), lin_new_mean=float((s_m[0, nl].mean() - s_copy[nl].mean()) / dn_))
            if do_ > 2.0:
                rec.update(lin_old=((s_p[:, ol].mean(-1) - s_true[ol].mean()) / do_).cpu().numpy(), lin_old_mean=float((s_m[0, ol].mean() - s_true[ol].mean()) / do_))
            if new_p.any() and old_p.any():
                fm = torch.from_numpy(free).to(dev).float()
                B_ = (h[t - L + 1:t + 1] * fm[..., None]).sum(0) / torch.from_numpy(np.maximum(cnt, 1)).to(dev).float()[:, None]
                npm, opm = torch.from_numpy(new_p).to(dev), torch.from_numpy(old_p).to(dev)
                rec.update(proj_new=proj(Pt, B_, h[t + 1], npm).cpu().numpy(), proj_old=proj(Pt, B_, h[t], opm).cpu().numpy(),
                           proj_new_mean=float(proj(Pm[0], B_, h[t + 1], npm)), proj_old_mean=float(proj(Pm[0], B_, h[t], opm)))
            # β (표본마다): 새 칸에서 [8 지연 + 옮긴 복사] 최소제곱 — 충분통계만 저장
            d = pos[2 * t + 3] - pos[2 * t + 1]
            mc = torch.roll(h[t].view(G, G, -1), shifts=(int(round(d[1] / 16)), int(round(d[0] / 16))), dims=(0, 1)).reshape(S, -1)
            Xr = torch.stack([h[t - k][nl] for k in range(L)] + [mc[nl]])          # (9, n, D)
            rec["XtX"] = torch.einsum("knd,lnd->kl", Xr, Xr).cpu().numpy()
            rec["Xty"] = torch.einsum("knd,Knd->Kk", Xr, Pt[:, nl]).cpu().numpy()    # (K, 9)
            rec["Xty_true"] = torch.einsum("knd,nd->k", Xr, h[t + 1][nl]).cpu().numpy()
            REC.append(rec)
    print(f, len(REC), flush=True)


def bimod(x):
    x = np.asarray(x); n = len(x)
    if n < 8:
        return float("nan")
    m = x.mean(); s = x.std()
    g = ((x - m) ** 3).mean() / s ** 3; k = ((x - m) ** 4).mean() / s ** 4 - 3
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


out = {"run": a.run, "K": a.K, "sampler": a.sampler, "n": len(REC)}
out["win_share"] = (np.bincount(WIN, minlength=a.K) / max(1, len(WIN))).tolist()
print(f"\n표본 (가설) 별 승리 몫 (창 {len(WIN)} 개, 블록 8..15 TF L1 최소): {np.round(out['win_share'], 2).tolist()}")
XtX = sum(r["XtX"] for r in REC).astype(np.float64); lam = 1e-6 * np.trace(XtX) / 9
bt = np.linalg.solve(XtX + lam * np.eye(9), sum(r["Xty_true"] for r in REC))[L]
bk = [np.linalg.solve(XtX + lam * np.eye(9), sum(r["Xty"][k] for r in REC))[L] / bt for k in range(a.K)]
out["beta_ratio_per_sample"] = bk
out["beta_ratio_mean_pred"] = float(np.linalg.solve(XtX + lam * np.eye(9), sum(r["Xty"].mean(0) for r in REC))[L] / bt)
print(f"\n== {tag}: 표본 {a.K} 개, 걸음 {len(REC)}")
print(f"  β 비 — 표본별 (풀링 적합) {np.round(bk, 2).tolist()} · 표본 평균 예측 {out['beta_ratio_mean_pred']:.2f}")
for key in ("lin_new", "lin_old", "proj_new", "proj_old"):
    R_ = [r for r in REC if key in r]
    allv = np.concatenate([r[key] for r in R_])                                 # 표본 × 걸음
    perstep_sd = np.mean([r[key].std() for r in R_])                             # 같은 문맥 안 표본 산포
    meanv = np.array([r[key + "_mean"] for r in R_])
    hist = np.histogram(np.clip(allv, -0.5, 1.5), bins=10, range=(-0.5, 1.5))[0].tolist()
    cen = np.concatenate([r[key] - r[key].mean() for r in R_])                    # 같은 문맥 안 편차 (표본 사이 모양)
    out[key] = dict(n=len(R_), mean_of_samples=float(allv.mean()), mean_pred=float(meanv.mean()), within_ctx_sd=float(perstep_sd),
                    frac_lt02=float((allv < 0.2).mean()), frac_gt08=float((allv > 0.8).mean()), bimodality=bimod(allv),
                    bimodality_within=bimod(cen) if a.K > 3 else None, hist=hist)
    per_k = [float(np.mean([r[key][k] for r in R_])) for k in range(a.K)]
    out[key]["per_sample_mean"] = per_k
    print(f"  {key:9s} 표본별 평균 {np.round(per_k, 3).tolist()}")
    print(f"  {key:9s} 표본 평균 {allv.mean():.3f} · 평균 예측의 값 {meanv.mean():.3f} · 같은 문맥 표본 SD {perstep_sd:.3f} · <0.2 {100 * (allv < 0.2).mean():.0f} % · >0.8 {100 * (allv > 0.8).mean():.0f} % · 두 봉우리 b 전체 {bimod(allv):.2f} · 같은 문맥 편차 {out[key]['bimodality_within']} (> 0.555 = 두 봉우리 쪽) · 히스토그램 (−0.5→1.5) {hist}")
R_ = [r for r in REC if "proj_new" in r]
if R_:
    nv = np.concatenate([r["proj_new"] for r in R_]); ov = np.concatenate([r["proj_old"] for r in R_])
    wn = np.concatenate([r["proj_new"] - r["proj_new"].mean() for r in R_]); wo = np.concatenate([r["proj_old"] - r["proj_old"].mean() for r in R_])
    out["new_old_corr_within_ctx"] = float(np.corrcoef(wn, wo)[0, 1]) if a.K > 1 else None
    out["new_plus_old"] = float((nv + ov).mean())
    print(f"  사영 (새 + 옛) 표본 평균 {out['new_plus_old']:.3f} · 같은 문맥 안 (새, 옛) 상관 {out['new_old_corr_within_ctx']} (두 갈래 표본이면 음수)")
json.dump(out, open(os.path.join(a.out, tag + ".json"), "w"), indent=1, default=float)
