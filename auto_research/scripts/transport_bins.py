#!/usr/bin/env python3
"""[P] 가닥 2 (운반 절반) — 옮긴 복사 계수 β 와 새 자리 존재감을 물체 속도 · 칸 아래 위상 구간별로 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=4 /venv/vjepa/bin/python -W ignore auto_research/scripts/transport_bins.py --run natural_ar_vw_state latest.pt

3D 합성 검증 세트 캐시, TF 한 걸음 (t = 7..14), 움직인 쌍 (블록 t → t+1 이동 > 8 px, 두 블록 보임).
  블록 안 움직임 v = |pos[2t+1] − pos[2t]| (px / 장, 블록 t 의 두 장 사이) — 구간 < 2 · 2–5 · > 5.
  칸 아래 위상 φ = 블록 t+1 둘째 장 물체 중심에서 가장 가까운 칸 중심까지 거리 (px, 0 = 칸 중심, 최대 11.3) — 구간 < 3 · 3–5 · > 5.
  β: 새 칸 (obj_{t+1} − obj_t) 에서 y ≈ Σ_k w_k h_{t−k} (같은 칸 8 지연) + β · (h_t 를 정답 변위 정수 칸만큼 옮긴 것), 구간 안 표본 전체 최소제곱. 모델 출력과 정답 둘 다 → 비 β_pred / β_true.
  존재감: 새 칸 선형 물체 점수 (블록 t 물체 템플릿) 의 (예측 − 복사) / (정답 − 복사). 구간마다 clip bootstrap 95 %."""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _clean_backbone_key                       # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, required=True)
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/transport_bins"))
ap.add_argument("--ctx", type=int, nargs="+", default=[0], help="TF 문맥 블록 수 (0 = 전체 0..t). 위치 번호는 원래 자리")
a = ap.parse_args()
dev = torch.device("cuda")
S, G, C, NB, L = 256, 16, 8, 16, 8
tag = f"{a.run[0]}__{a.run[1].replace('.pt', '')}" + ("" if a.ctx == [0] else "__ctx" + "-".join(map(str, a.ctx)))
os.makedirs(a.out, exist_ok=True)
D_ = os.path.join(REPO, "z_training/runs", a.run[0])
pc = yaml.safe_load(open(os.path.join(D_, "config.yaml")))["model"]["predictor"]
model = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                             num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
model.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D_, a.run[1]), map_location="cpu", weights_only=False)["predictor"]).items()
                       if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
model = model.to(dev).eval()


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


REC = []
for f in sorted(glob.glob(os.path.join(a.cache, "h_val_target_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    X = np.load(f, mmap_mode="r"); lab = np.load(os.path.join(a.cache, f"lab_val_s{sh}.npz"))
    for w in range(X.shape[0]):
        cv = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0))
        objb = (np.maximum(cv[0::2], cv[1::2]) >= 0.25).reshape(NB, S)
        touch = ((cv[0::2] > 0) | (cv[1::2] > 0)).reshape(NB, S)
        ok = lab["inframe_frac"][w] > 0.9
        visb = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & ok[0::2] & ok[1::2]
        far = ~dil(touch.any(0), 2)
        pos = lab["pos"][w]
        ts = [t for t in range(C - 1, NB - 1) if visb[t] and visb[t + 1] and np.isfinite(pos[[2 * t, 2 * t + 1, 2 * t + 3]]).all()
              and np.linalg.norm(pos[2 * t + 3] - pos[2 * t + 1]) > 8 and (objb[t + 1] & ~objb[t]).any() and far.sum() >= 10]
        if not ts:
            continue
        h = torch.from_numpy(np.asarray(X[w], dtype=np.float32)).to(dev)
        PT = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for c in a.ctx:
                if c == 0:
                    PT[c] = model.forward_seq(h[:NB - 1].reshape(1, -1, h.size(-1)), torch.arange((NB - 1) * S, device=dev).unsqueeze(0)).float()[0].view(NB - 1, S, -1)
                else:                                                      # 위치 t 의 예측을 블록 t−c+1..t 만으로 (t = 7..14)
                    xx = torch.stack([h[t - c + 1:t + 1].reshape(c * S, -1) for t in range(C - 1, NB - 1)])
                    ii = torch.stack([torch.arange((t - c + 1) * S, (t + 1) * S, device=dev) for t in range(C - 1, NB - 1)])
                    q = model.forward_seq(xx, ii).float()[:, -S:]
                    PT[c] = torch.cat([torch.zeros(C - 1, S, q.size(-1), device=dev), q])          # 인덱스 t 로 맞춘다
        farm = torch.from_numpy(far).to(dev)
        for t, c in [(t, c) for t in ts for c in a.ctx]:
            p_tf = PT[c]
            d = pos[2 * t + 3] - pos[2 * t + 1]
            dx, dy = int(round(d[0] / 16)), int(round(d[1] / 16))
            mc = torch.roll(h[t].view(G, G, -1), shifts=(dy, dx), dims=(0, 1)).reshape(S, -1)
            new = torch.from_numpy(objb[t + 1] & ~objb[t]).to(dev)
            X_ = torch.stack([h[t - k][new] for k in range(L)] + [mc[new]])            # (9, n, D)
            r = dict(ctx=c, clip=f"{sh}:{w}", v=float(np.linalg.norm(pos[2 * t + 1] - pos[2 * t])), dn=float(np.linalg.norm(d)),
                     phase=float(np.linalg.norm((pos[2 * t + 3] % 16) - 8)), XtX=torch.einsum("knd,lnd->kl", X_, X_).cpu().numpy())
            for nm, y in (("pred", p_tf[t][new]), ("true", h[t + 1][new])):
                r[f"Xty_{nm}"] = torch.einsum("knd,nd->k", X_, y).cpu().numpy(); r[f"yty_{nm}"] = float((y * y).sum())
            T_hat = F.normalize(h[t][torch.from_numpy(objb[t]).to(dev)].mean(0), dim=-1)
            sc = torch.stack([p_tf[t], h[t + 1], h[t]]) @ T_hat
            sc = sc - sc[:, farm].mean(1, keepdim=True)
            r["s_new"] = sc[:, new].mean(1).cpu().numpy()                              # pred · true · copy
            REC.append(r)
    print(f, len(REC), flush=True)


def beta(recs, nm):
    XtX = sum(r["XtX"] for r in recs).astype(np.float64); Xty = sum(r[f"Xty_{nm}"] for r in recs).astype(np.float64)
    return np.linalg.solve(XtX + 1e-6 * np.trace(XtX) / 9 * np.eye(9), Xty)[L]


def pres(recs):
    s = np.stack([r["s_new"] for r in recs]).mean(0)
    return (s[0] - s[2]) / (s[1] - s[2])


rng = np.random.RandomState(0)


def ci(recs, fn):
    cl = np.array([r["clip"] for r in recs]); uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}
    v = [fn([recs[i] for i in np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])]) for _ in range(300)]
    return np.nanpercentile(v, [2.5, 97.5]).tolist()


out = {"run": a.run, "n": len(REC), "bins": {}, "ctx": {}}
if a.ctx != [0]:
    print(f"\n== {tag}: TF 문맥 길이별 운반 비 · 새 칸 존재감 (움직인 쌍 전체)")
    for c in a.ctx:
        R = [r for r in REC if r["ctx"] == c]
        bp, bt = beta(R, "pred"), beta(R, "true")
        e = dict(n=len(R), beta_pred=bp, beta_true=bt, ratio=bp / bt, ci_ratio=ci(R, lambda rr: beta(rr, "pred") / beta(rr, "true")), pres=pres(R), ci_pres=ci(R, pres))
        out["ctx"][str(c)] = e
        print(f"  문맥 {c if c else '전체'} 블록 n {len(R):4d}  β 모델 {bp:.3f} · 정답 {bt:.3f} · 비 {bp / bt:.2f} [{e['ci_ratio'][0]:.2f}, {e['ci_ratio'][1]:.2f}] | 존재감 {e['pres']:.2f} [{e['ci_pres'][0]:.2f}, {e['ci_pres'][1]:.2f}]")
    REC = [r for r in REC if r["ctx"] == max(a.ctx)]
print(f"\n== {tag}: 운반 계수 β (모델 / 정답 / 비) · 새 칸 존재감 — 구간별 (n = 걸음 수)")
for axis, edges in (("v", (0, 2, 5, 99)), ("phase", (0, 3, 5, 99)), ("dn", (8, 16, 32, 999))):
    for lo, hi in zip(edges[:-1], edges[1:]):
        R = [r for r in REC if lo <= r[axis] < hi]
        if len(R) < 10:
            continue
        bp, bt = beta(R, "pred"), beta(R, "true")
        e = dict(n=len(R), beta_pred=bp, beta_true=bt, ratio=bp / bt, ci_ratio=ci(R, lambda rr: beta(rr, "pred") / beta(rr, "true")),
                 pres=pres(R), ci_pres=ci(R, pres))
        out["bins"][f"{axis}:{lo}-{hi}"] = e
        print(f"  {axis:5s} {lo:>3}–{hi:<3} n {len(R):4d}  β 모델 {bp:.3f} · 정답 {bt:.3f} · 비 {bp / bt:.2f} [{e['ci_ratio'][0]:.2f}, {e['ci_ratio'][1]:.2f}]"
              f" | 존재감 {e['pres']:.2f} [{e['ci_pres'][0]:.2f}, {e['ci_pres'][1]:.2f}]")
json.dump(out, open(os.path.join(a.out, tag + ".json"), "w"), indent=1, default=float)
