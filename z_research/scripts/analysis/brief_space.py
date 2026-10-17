#!/usr/bin/env python3
"""속도를 "어느 공간에서" 담나 — probe 이식 시험 (2026-10-17, 사용자 질문, 100 샘플).

  python z_research/scripts/analysis/brief_space.py --device cuda

샘플 = brief_diag.py 와 같은 규칙 (VOBench primitive pos_a 움직이는 clip, late 70 + visible 30, seed 0).
라벨: 블록 b 의 물체 속도 (du, dv) = 위치 (b) − 위치 (b−1), 토큰 단위. 입력: 그 블록의 물체 자리 3×3 토큰 평균. 물체가 보이는 블록만.
공간:
  ctx  : 문맥 블록 C−5 … C−1 (AR = 블록 LN(target_encoder) / ctxt = 영상 z)
  tgt  : 미래 블록 C … C+7 의 정답 (블록 LN(target_encoder)) — 기준 공간
  out  : 같은 블록의 예측 출력 (render 된 rollout)
  st   : 같은 걸음의 predictor 상태 (384, rollout 중 되먹이는 상태)
시험: 같은 공간 안 R² (clip 단위 5-fold) · 이식 R² (A 에서 학습 → B 에 그대로 적용, 같은 차원끼리만)
  tgt → out : 예측이 정답과 같은 좌표로 속도를 담나 · ctx → tgt : 문맥이 정답과 같은 좌표인가
출력: <out>/brief_space.json + 표."""
import argparse, csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
from src.models.rollout_predictor import vit_prefix_predictor
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["natural_ar_vw_state", "natural_ar_vw_state_ctxt_video"])
ap.add_argument("--vobench", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--n_late", type=int, default=70)
ap.add_argument("--n_vis", type=int, default=30)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default=os.path.join(REPO, "z_research/VOBench_v1/figures/brief_diag"))
a = ap.parse_args(); dev = a.device
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
S, C, K, G = 256, 8, 8, 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
state = _load_checkpoint(CK)
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, state["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()
rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv")))
        if r["variant"] == "pos_a" and r["objset"] == "primitive" and r["motion"] != "flat_static"]
rng = np.random.default_rng(0)
late = [r for r in rows if r["occlusion"] == "late"]; vis = [r for r in rows if r["occlusion"] == "visible"]
pick = list(rng.permutation(late))[: a.n_late] + list(rng.permutation(vis))[: a.n_vis]

def load(r):
    d = os.path.join(a.vobench, r["file_name"])
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
    W0 = x.shape[-1]
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(d), "label.json")))["labels"]["A"]}
    posf, hid = [], []
    for b in range(16):
        q = [lab[IDX[2 * b]], lab[IDX[2 * b + 1]]]
        posf.append((np.mean([t["v"] for t in q]) / W0 * G, np.mean([t["u"] for t in q]) / W0 * G)); hid.append(np.mean([t["hidden"] for t in q]))
    return (x - MEAN) / STD, np.array(posf), np.array(hid)

def region(p):
    r0, c0 = int(np.clip(p[0], 0, G - 1)), int(np.clip(p[1], 0, G - 1))
    return torch.tensor([rr * G + cc for rr in range(max(0, r0 - 1), min(G, r0 + 2)) for cc in range(max(0, c0 - 1), min(G, c0 + 2))])

@torch.no_grad()
def blocks(x):
    xb = x.reshape(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(xb[i:i + 8]) for i in range(0, 16, 8)], 0).float()
    return F.layer_norm(h, (1280,)).reshape(1, 16 * S, -1)

class Model:
    def __init__(self, run):
        cfg = yaml.safe_load(open(os.path.join(REPO, "z_training/runs", run, "config.yaml")))
        pc = cfg["model"]["predictor"]; self.kind = pc.get("kind", "ar"); self.name = run
        kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384,
                  depth=12, num_heads=12, num_mask_tokens=10, mask_mode=self.kind)
        if self.kind == "ar":
            kw["ar_feedback"] = pc.get("ar_feedback", "output")
        self.pred = vit_prefix_predictor(**kw)
        sd = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(REPO, "z_training/runs", run, "latest.pt"),
              map_location="cpu", weights_only=False)["predictor"]).items() if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
        self.pred.load_state_dict(sd, strict=True); self.pred = self.pred.to(dev).eval()
        if self.kind != "ar":
            key = cfg["model"].get("context_encoder_key", "encoder")
            self.enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2 * C, use_rope=True, uniform_power=False)
            _load_state_dict(self.enc, state[key], tag=key); self.enc = self.enc.to(dev).eval()

    @torch.no_grad()
    def go(self, x, hb):
        idx = torch.arange(C * S, device=dev).unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = hb[:, : C * S] if self.kind == "ar" else self.enc(x[:, : 2 * C].unsqueeze(0).to(dev)).float()
            if self.kind == "ar":
                p, st = self.pred.rollout_kv(ctx, idx, K, return_states=True)
            else:
                st = self.pred.rollout_state_kv(ctx, idx, K); p = self.pred.render(st)
        return ctx.float()[0], p.float()[0], st.float()[0]

models = [Model(r) for r in a.runs]
D = {m.name: {"ctx": [], "out": [], "st": []} for m in models}; T = {"tgt": [], "ctxblk": []}
for i, r in enumerate(pick):
    x, posf, hid = load(r); hb = blocks(x)[0]
    for m in models:
        ctx, p, st = m.go(x, hb.unsqueeze(0))
        for b in range(C - 5, C):
            if hid[b] < 0.5 and hid[b - 1] < 0.5:
                D[m.name]["ctx"].append((i, ctx[b * S + region(posf[b])].mean(0).cpu().numpy(), posf[b] - posf[b - 1]))
        for s_ in range(K):
            b = C + s_
            if hid[b] < 0.5 and hid[b - 1] < 0.5:
                R = region(posf[b]); y = posf[b] - posf[b - 1]
                D[m.name]["out"].append((i, p[s_ * S + R].mean(0).cpu().numpy(), y))
                D[m.name]["st"].append((i, st[s_ * S + R].mean(0).cpu().numpy(), y))
    for b in range(C - 5, C + K):
        if hid[b] < 0.5 and hid[b - 1] < 0.5:
            (T["tgt"] if b >= C else T["ctxblk"]).append((i, hb[b * S + region(posf[b])].mean(0).cpu().numpy(), posf[b] - posf[b - 1]))
    if i % 25 == 0:
        print(f"{i}/{len(pick)}", flush=True)

def arr(L):
    return np.array([t[0] for t in L]), np.array([t[1] for t in L], np.float64), np.array([t[2] for t in L], np.float64)

def fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6; A = (X - mu) / sd; ym = Y.mean(0)
    W = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(len(A)), Y - ym)
    return lambda Z: ((Z - mu) / sd) @ W + ym

def r2(P, Y):
    return float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())

FOLD = np.random.default_rng(0).permutation(len(pick)) % 5          # clip 단위 fold
def within(L):
    g, X, Y = arr(L); best = -9
    for lam in (10.0, 100.0, 1000.0, 10000.0):
        P = np.zeros_like(Y)
        for f in range(5):
            tr, te = FOLD[g] != f, FOLD[g] == f
            P[te] = fit(X[tr], Y[tr], lam)(X[te])
        best = max(best, r2(P, Y))
    return best, len(Y)

def transfer(Ltr, Lte):
    """A 에서 학습 (clip fold 의 학습 쪽) → B 의 시험 쪽 clip 에 그대로. λ 는 A 안에서 고른다."""
    g1, X1, Y1 = arr(Ltr); g2, X2, Y2 = arr(Lte)
    if X1.shape[1] != X2.shape[1]:
        return None
    lam_best, r_best = None, -9
    for lam in (10.0, 100.0, 1000.0, 10000.0):
        P = np.zeros_like(Y1)
        for f in range(5):
            P[FOLD[g1] == f] = fit(X1[FOLD[g1] != f], Y1[FOLD[g1] != f], lam)(X1[FOLD[g1] == f])
        rr = r2(P, Y1)
        if rr > r_best:
            r_best, lam_best = rr, lam
    P = np.zeros_like(Y2)
    for f in range(5):
        P[FOLD[g2] == f] = fit(X1[FOLD[g1] != f], Y1[FOLD[g1] != f], lam_best)(X2[FOLD[g2] == f])
    corr = [float(np.corrcoef(P[:, j], Y2[:, j])[0, 1]) for j in range(Y2.shape[1])]          # 크기 · 치우침 무시, 방향만
    scale = [float(np.std(P[:, j]) / (np.std(Y2[:, j]) + 1e-9)) for j in range(Y2.shape[1])]    # 예측 속도 크기 / 실제 (1 이면 같음)
    return {"r2": r2(P, Y2), "corr_dv_du": corr, "scale_dv_du": scale}

res = {"within": {}, "transfer": {}}
w, n = within(T["tgt"]); res["within"]["tgt (정답 미래 블록)"] = (w, n)
w, n = within(T["ctxblk"]); res["within"]["tgt 공간의 문맥 블록"] = (w, n)
print("\n[같은 공간 안] 속도 (du, dv) R² · 표본 수")
print(f"   정답 블록 LN(h) (미래)            {res['within']['tgt (정답 미래 블록)'][0]:.2f}  n{res['within']['tgt (정답 미래 블록)'][1]}")
print(f"   정답 블록 LN(h) (문맥 구간)       {res['within']['tgt 공간의 문맥 블록'][0]:.2f}  n{res['within']['tgt 공간의 문맥 블록'][1]}")
for m in models:
    for sp in ("ctx", "out", "st"):
        w, n = within(D[m.name][sp]); res["within"][f"{m.name}/{sp}"] = (w, n)
        print(f"   {m.name[-22:]:>22s} {sp:4s}  {w:.2f}  n{n}")
print("\n[이식] A 에서 학습한 probe 를 B 에 그대로 적용한 R² (같은 차원끼리만, 음수 = 평균보다 못함)")
for m in models:
    pairs = [("정답 미래 → 예측 출력", T["tgt"], D[m.name]["out"]), ("예측 출력 → 정답 미래", D[m.name]["out"], T["tgt"]),
             ("문맥 표현 → 정답 미래", D[m.name]["ctx"], T["tgt"]), ("정답 문맥 블록 → 문맥 표현", T["ctxblk"], D[m.name]["ctx"]),
             ("문맥 표현 → 예측 출력", D[m.name]["ctx"], D[m.name]["out"])]
    for nm, A_, B_ in pairs:
        t = transfer(A_, B_); res["transfer"][f"{m.name}/{nm}"] = t
        print(f"   {m.name[-22:]:>22s} {nm:22s} " + ("차원 다름" if t is None else f"R² {t['r2']:.2f} | 상관 dv {t['corr_dv_du'][0]:.2f} du {t['corr_dv_du'][1]:.2f} | 크기비 dv {t['scale_dv_du'][0]:.2f} du {t['scale_dv_du'][1]:.2f}"))
json.dump(res, open(f"{a.out}/brief_space.json", "w"), indent=1, ensure_ascii=False)
