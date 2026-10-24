#!/usr/bin/env python3
"""[P] 외부 팀 (vll5 auto_paper) "문맥 자르기" 검사를 기준 블록 AR 에 정의 그대로 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=4 /venv/vjepa/bin/python -W ignore auto_research/scripts/ctx_trunc_external.py --set ssv2_native --n 800
  CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/ctx_trunc_external.py --set k710_4fps --n 800
  CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/ctx_trunc_external.py --set phys3d

정의 (외부 팀):
  데이터  ssv2_native: SSv2 validation 중 48 장 이상 영상에서 무작위 n 개 (seed 0), 앞 32 장을 원래 프레임 간격 그대로 (12 fps).
          k710_4fps  : (우리 학습 간격) K710 val 중 4 fps 간격 (fstp = ceil(fps) // 4) 으로 32 장이 나오는 영상에서 무작위 n 개, 앞 32 장.
                       SSv2 는 4 fps 로 32 장이 나오는 val 영상이 7 개뿐이라 K710 으로 대신한다.
          k710_native: K710 val, 앞 32 장 원래 간격 (데이터셋 효과와 간격 효과를 가르는 칸).
          phys3d     : 3D 합성 검증 세트 캐시 (raw 0 부터 간격 3, 32 장).
          화면: 짧은 변 256 으로 줄이고 가운데 256 × 256.
  표적    블록 (연속 2 장) 마다 따로 목표 encoder → 블록별 LN (affine 없음), bf16 autocast.
  예측    굴린 예측 8 걸음 (모델 자기 되먹임 — 우리 기준 블록 AR 은 상태 되먹임). 문맥 = 마지막 c 블록 (c = 8 · 4 · 2 · 1), 위치 번호는 원래 자리.
          + TF 한 걸음 (정답 블록 t−c+1..t → 블록 t+1, t = 7..14) — 이건 16 블록 창에서만.
  지표    걸음 k 마다 L = 예측 블록 vs 표적 블록 토큰 평균 L1. 복사 = 마지막 문맥 블록 (블록 7) 을 모든 걸음에.
          역사 값 = (L_c − L_8) / (L_copy − L_8). 전체 토큰 / 변한 토큰 (걸음마다 복사 오차 상위 10 % 토큰).
  추가    (2) 학습 없는 같은 칸 n 블록 평균 (n = 2 · 4 · 8) 의 같은 지표.
          (3) 정지 칸 (자연 영상: 창 내내 칸 평균 |Δ화소| < 2 단계 / 3D: 물체가 한 번도 안 닿은 칸 2 칸 팽창의 여집합) 에서
              c = 1 TF 한 걸음 예측 ≈ a · h_t + b (b = 차원별 상수 = (1 − a) · μ) 의 a, 칸별 상수판 a_cell, 정답에 대한 같은 적합 (최적 수축),
              그리고 L1 (예측 / 복사 / n 블록 평균 / c = 8 예측 → 정답)."""
import argparse, glob, json, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F, yaml
import decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402
from app.vjepa_frozen import ar as AR                                                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["ssv2_native", "ssv2_native_lenval", "k710_4fps", "k710_native", "phys3d"])
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--run", nargs=2, default=["natural_ar_vw_state", "latest.pt"])
ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/ctx_trunc"))
a = ap.parse_args()
dev = torch.device("cuda")
S, G, C, K = 256, 16, 8, 8
CS = (8, 4, 2, 1)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
os.makedirs(a.out, exist_ok=True)
tag = f"{a.set}__{a.run[0]}__{a.run[1].replace('.pt', '')}"

D_ = os.path.join(REPO, "z_training/runs", a.run[0])
pc = yaml.safe_load(open(os.path.join(D_, "config.yaml")))["model"]["predictor"]
kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
          num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
for k_ in ("gating_residual", "gating_anchor", "gating_out_ln"):
    if pc.get(k_) is not None:
        kw[k_] = pc[k_]
model = vit_prefix_predictor(**kw)
model.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D_, a.run[1]), map_location="cpu", weights_only=False)["predictor"]).items()
                       if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
model = model.to(dev).eval()

# ------------------------------------------------------------------ 데이터 목록
items = []
if a.set == "ssv2_native":
    # 외부 코드와 같은 목록 파일 (공백 구분, 첫 열 = 경로). 이 머신의 파일은 952 행 (공식 label 로 val id 128 · train id 696) — 내용은 그대로 쓴다
    rows = [l.split(" ")[0].strip() for l in open(os.path.join(REPO, "data_csv/ssv2/val_min48.csv")) if l.strip()]
    np.random.default_rng(0).shuffle(rows)
    items = [(p, 1) for p in rows]
elif a.set == "ssv2_native_lenval":
    # lengths.tsv 에서 공식 validation id 이고 길이 ≥ 48 장인 영상 → 같은 형식 목록 (auto_research/results/ssv2_val_min48_from_lengths.csv)
    val = json.load(open("/workspace/dataset/training_data/SSv2/labels/validation.json")); ids = {v["id"] for v in val}
    rows = [r for r in (l.rstrip("\n").split("\t") for l in open(os.path.join(REPO, "data_csv/ssv2/lengths.tsv"))) if os.path.basename(r[0])[:-5] in ids and int(r[1]) >= 48]
    lp = os.path.join(REPO, "auto_research/results/ssv2_val_min48_from_lengths.csv")
    with open(lp, "w") as f_:
        f_.writelines(f"{r[0]} 0\n" for r in rows)
    rows = [r[0] for r in rows]
    np.random.default_rng(0).shuffle(rows)
    items = [(p, 1) for p in rows]
elif a.set.startswith("k710"):
    val = [l.split(" ")[0] for l in open(os.path.join(REPO, "data_csv/k710_320/val_min224.csv"))]
    np.random.default_rng(0).shuffle(val)
    items = [(p, None) for p in val]
if a.set != "phys3d":
    from analysis.intphys2.dataset import build_video_transform
    TRANSFORM = build_video_transform(256, 1.0)                                    # 짧은 변 256 → 가운데 256, ImageNet 정규화 (외부와 같다)
    enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev).eval()   # fp32 가중치 + bf16 autocast


def view256(fr):
    x = torch.from_numpy(fr).to(dev).permute(0, 3, 1, 2).float().div_(255)
    H, W = x.shape[-2:]; s = 256 / min(H, W)
    x = F.interpolate(x, size=(max(256, round(H * s)), max(256, round(W * s))), mode="bilinear", align_corners=False, antialias=True)
    H, W = x.shape[-2:]; y0, x0 = (H - 256) // 2, (W - 256) // 2
    return x[:, :, y0:y0 + 256, x0:x0 + 256]                                       # (T, 3, 256, 256)


def static_mask_pix(x):
    """x (32, 3, 256, 256) [0,1] → 칸별 창 내내 블록 사이 평균 |Δ화소| (회색, 0–255) 의 최대 < 2 단계."""
    g = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1) * 255   # (32, 256, 256)
    gb = g[0::2]                                                                    # 블록마다 첫 장
    d = (gb[1:] - gb[:-1]).abs().view(15, G, 16, G, 16).mean((2, 4)).view(15, S)
    return (d.max(0).values < 2.0)


def phys3d_iter():
    from glob import glob as gl
    for f in sorted(gl("/workspace/auto_research_cache/phys3d/h_val_target_s*.npy")):
        sh = f.split("_s")[-1].split(".")[0]
        X = np.load(f, mmap_mode="r"); lab = np.load(f"/workspace/auto_research_cache/phys3d/lab_val_s{sh}.npz")
        for w in range(X.shape[0]):
            bb = np.nan_to_num(lab["bbox"][w], nan=-100.0)
            x0, y0, x1, y1 = [bb[..., i, None] for i in range(4)]
            c = np.arange(G) * 16.0
            ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
            oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
            touch = ((oy[..., :, None] * ox[..., None, :]) > 0).any(0)             # (G, G)
            m = touch.copy()
            for _ in range(2):
                o = m.copy(); o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
                o[1:, 1:] |= m[:-1, :-1]; o[:-1, :-1] |= m[1:, 1:]; o[1:, :-1] |= m[:-1, 1:]; o[:-1, 1:] |= m[1:, :-1]; m = o
            yield f"{os.path.basename(f)}:{w}", torch.from_numpy(np.asarray(X[w], dtype=np.float32)).to(dev), torch.from_numpy(~m.reshape(-1)).to(dev)


def natural_iter():
    n_ok = 0
    for p, stride in items:
        if n_ok >= a.n:
            break
        try:
            vr = decord.VideoReader(p, num_threads=1)                         # webm 은 여러 스레드 디코딩이 깨진다
            st = stride if stride is not None else (1 if a.set.endswith("native") else max(1, int(math.ceil(vr.get_avg_fps())) // 4))
            if len(vr) < 31 * st + 1:
                continue
            fr = vr.get_batch([st * j for j in range(32)]).asnumpy()
        except Exception:
            continue
        xv = TRANSFORM(fr).to(dev)                                                    # (3, 32, 256, 256) 정규화
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = AR.encode_blocks(enc, xv[None], 2)[0].float().view(16, S, -1)
        x = (xv * STD[0] + MEAN[0]).permute(1, 0, 2, 3).clamp(0, 1)                    # 정지 칸 판정용 화소 (32, 3, 256, 256)
        n_ok += 1
        yield p, h, static_mask_pix(x)


def l1tok(p, y):
    return (p - y).abs().mean(-1)                                                   # (..., S)


@torch.no_grad()
def roll(h, c):
    idx = torch.arange((C - c) * S, C * S, device=dev).unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.rollout_kv(h[C - c:C].reshape(1, c * S, -1), idx, K).float()[0].view(K, S, -1)


@torch.no_grad()
def tf(h, c):
    ts = list(range(C - 1, 15))
    x = torch.stack([h[t - c + 1:t + 1].reshape(c * S, -1) for t in ts])
    idx = torch.stack([torch.arange((t - c + 1) * S, (t + 1) * S, device=dev) for t in ts])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.forward_seq(x, idx).float()[:, -S:]                            # (8, S, D): t = 7..14 → 블록 8..15


NAMES = [f"c{c}" for c in CS] + ["copy", "mean_2", "mean_4", "mean_8"]
RO = {n: {"all": [], "chg": []} for n in NAMES}; TF = {n: {"all": [], "chg": []} for n in NAMES}
ST = {k_: [] for k_ in ("tf_c1", "tf_c8", "copy", "mean_2", "mean_4", "mean_8")}   # 정지 칸 L1
acc = {nm: dict(n=0, sp=torch.zeros(1280, device=dev, dtype=torch.float64), sh=torch.zeros(1280, device=dev, dtype=torch.float64), sph=0.0, shh=0.0,
                cp=torch.zeros(S, 1280, device=dev, dtype=torch.float64), ch=torch.zeros(S, 1280, device=dev, dtype=torch.float64),
                cn=torch.zeros(S, device=dev, dtype=torch.float64)) for nm in ("pred_c1", "true", "pred_c8")}
n_static = []
t0 = time.time(); n = 0
for key, h, stat in (phys3d_iter() if a.set == "phys3d" else natural_iter()):
    if n >= a.n:
        break
    fut = h[C:]
    ctx = h[:C]
    P = {f"c{c}": roll(h, c) for c in CS}
    P.update({"copy": ctx[-1:].expand(K, S, -1), **{f"mean_{m}": ctx[-m:].mean(0, keepdim=True).expand(K, S, -1) for m in (2, 4, 8)}})
    ecopy = l1tok(P["copy"], fut)                                                   # (K, S)
    chg = ecopy >= ecopy.quantile(0.9, dim=1, keepdim=True)
    for nm, p in P.items():
        e = l1tok(p, fut)
        RO[nm]["all"].append(e.mean(1).cpu().numpy()); RO[nm]["chg"].append(((e * chg).sum(1) / chg.sum(1)).cpu().numpy())
    # TF 한 걸음 (t = 7..14)
    Ptf = {f"c{c}": tf(h, c) for c in CS}
    hist = lambda m: torch.stack([h[t - m + 1:t + 1].mean(0) for t in range(C - 1, 15)])
    Ptf.update({"copy": h[C - 1:15], **{f"mean_{m}": hist(m) for m in (2, 4, 8)}})
    ecopy = l1tok(Ptf["copy"], fut)
    chg = ecopy >= ecopy.quantile(0.9, dim=1, keepdim=True)
    for nm, p in Ptf.items():
        e = l1tok(p, fut)
        TF[nm]["all"].append(e.mean(1).cpu().numpy()); TF[nm]["chg"].append(((e * chg).sum(1) / chg.sum(1)).cpu().numpy())
    # 정지 칸: TF 한 걸음 (t = 7..14) 의 수축 적합
    n_static.append(float(stat.float().mean()))
    if stat.sum() >= 4:
        hs = h[C - 1:15][:, stat]                                                  # (8, n, D) = h_t
        ys = {"pred_c1": Ptf["c1"][:, stat], "true": fut[:, stat], "pred_c8": Ptf["c8"][:, stat]}
        for nm_, y in ys.items():
            A_ = acc[nm_]
            A_["n"] += hs.shape[0] * hs.shape[1]
            A_["sp"] += y.sum((0, 1)).double(); A_["sh"] += hs.sum((0, 1)).double()
            A_["sph"] += float((y * hs).sum()); A_["shh"] += float((hs * hs).sum())
            A_["cp"][stat] += y.sum(0).double(); A_["ch"][stat] += hs.sum(0).double(); A_["cn"][stat] += hs.shape[0]
        for k_, p in (("tf_c1", Ptf["c1"]), ("tf_c8", Ptf["c8"]), ("copy", Ptf["copy"]), ("mean_2", Ptf["mean_2"]), ("mean_4", Ptf["mean_4"]), ("mean_8", Ptf["mean_8"])):
            ST[k_].append(float((p[:, stat] - fut[:, stat]).abs().mean()))
    n += 1
    if n % 50 == 0:
        print(f"{n} 창 ({time.time() - t0:.0f}s)", flush=True)

rng = np.random.RandomState(0)
out = {"set": a.set, "n": n, "run": a.run, "static_frac": float(np.mean(n_static)), "ro": {}, "tf": {}, "static": {}}


def summarize(D, label):
    L = {nm: {g: np.stack(D[nm][g]) for g in ("all", "chg")} for nm in NAMES}          # (n, K)
    res = {}
    for g in ("all", "chg"):
        mean = {nm: L[nm][g].mean(0) for nm in NAMES}
        hv = {nm: (mean[nm] - mean["c8"]) / (mean["copy"] - mean["c8"]) for nm in NAMES}
        # 학습 없는 평균의 역사 값: (L_mean_n − L_mean_8) / (L_copy − L_mean_8)
        hv_mean = {nm: (mean[nm] - mean["mean_8"]) / (mean["copy"] - mean["mean_8"]) for nm in ("mean_2", "mean_4")}
        bs = []
        for _ in range(300):
            ix = rng.randint(0, n, n)
            m_ = {nm: L[nm][g][ix].mean(0) for nm in ("c8", "c2", "c1", "copy")}
            bs.append([(m_["c2"] - m_["c8"]) / (m_["copy"] - m_["c8"]), (m_["c1"] - m_["c8"]) / (m_["copy"] - m_["c8"])])
        bs = np.array(bs)                                                           # (300, 2, K)
        res[g] = dict(L={nm: mean[nm].tolist() for nm in NAMES}, hist={nm: hv[nm].tolist() for nm in NAMES},
                      hist_mean={nm: v.tolist() for nm, v in hv_mean.items()},
                      ci_c2=np.percentile(bs[:, 0], [2.5, 97.5], axis=0).tolist(), ci_c1=np.percentile(bs[:, 1], [2.5, 97.5], axis=0).tolist())
        print(f"\n== {label} · {'전체 토큰' if g == 'all' else '변한 토큰 (복사 오차 상위 10 %)'} · n {n}")
        print("   걸음            " + " ".join(f"{k + 1:>7d}" for k in range(K)))
        for nm in NAMES:
            print(f"   L {nm:7s}      " + " ".join(f"{v:7.3f}" for v in mean[nm]))
        for nm in ("c4", "c2", "c1"):
            print(f"   역사 8→{nm[1:]} 블록  " + " ".join(f"{v:7.3f}" for v in hv[nm]))
        print(f"   [8→2 CI k1 {res[g]['ci_c2'][0][0]:.3f}–{res[g]['ci_c2'][1][0]:.3f} · 8→1 CI k1 {res[g]['ci_c1'][0][0]:.3f}–{res[g]['ci_c1'][1][0]:.3f}]")
        for nm, v in hv_mean.items():
            print(f"   학습 없는 평균 역사 (mean_8 기준) {nm}  " + " ".join(f"{x:7.3f}" for x in v))
        print("   (copy − c8) / (copy − mean_8) = 모델 이득 / 학습 없는 8 블록 평균 이득  " + " ".join(f"{x:7.2f}" for x in (mean['copy'] - mean['c8']) / (mean['copy'] - mean['mean_8'])))
    return res


out["ro"] = summarize(RO, f"{a.set} 굴린 예측 (문맥 블록 0..7 → 미래 8..15)")
out["tf"] = summarize(TF, f"{a.set} TF 한 걸음 (t = 7..14, 걸음 = t − 6)")
print(f"\n== 정지 칸 (창 평균 비율 {out['static_frac']:.3f}) — TF 한 걸음, t = 7..14")
for nm_, A_ in acc.items():
    if A_["n"] == 0:
        continue
    sp, sh = A_["sp"], A_["sh"]
    a_g = (A_["sph"] - float((sp * sh).sum()) / A_["n"]) / (A_["shh"] - float((sh * sh).sum()) / A_["n"])
    m = A_["cn"] > 0
    cp, ch = A_["cp"][m] / A_["cn"][m, None], A_["ch"][m] / A_["cn"][m, None]
    # 칸별 상수: a_cell = [Σph − Σ_cell n_c p̄_c·h̄_c] / [Σhh − Σ_cell n_c |h̄_c|²]
    a_c = (A_["sph"] - float((A_["cn"][m, None] * cp * ch).sum())) / (A_["shh"] - float((A_["cn"][m, None] * ch * ch).sum()))
    b = (sp / A_["n"] - a_g * sh / A_["n"]); mu = b / max(1e-6, 1 - a_g)
    out["static"][nm_] = dict(a_global=float(a_g), a_cell=float(a_c), mu_norm=float(mu.norm()), hbar_norm=float((sh / A_["n"]).norm()))
    print(f"   {nm_:8s} ≈ a · h_t + b : a (전역 상수) {a_g:.3f} · a (칸별 상수) {a_c:.3f} · |μ| {float(mu.norm()):.1f} (|h̄| {float((sh / A_['n']).norm()):.1f}, |h_t| ≈ 35.8)")
for k_, v in ST.items():
    out["static"][f"L1_{k_}"] = float(np.mean(v)) if v else None
print("   L1 → 정답 (정지 칸): " + " · ".join(f"{k_} {np.mean(v):.4f}" for k_, v in ST.items() if v))
json.dump(out, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
print("저장:", os.path.join(a.out, f"{tag}.json"))
