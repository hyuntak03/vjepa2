#!/usr/bin/env python3
"""[L] 문맥만으로 할 수 있는 운반 (등속 외삽) 의 위약 뺀 b — 데이터 최적 vs 기준 AR (auto_research, 2026-10-22 KST 17 시).

물음: 지금까지 "데이터 0.07–0.09 vs 모델 0.009–0.021" 의 W 는 실제 다음 흐름 U[t+1] (정답을 봐야 아는 변위) 로 옮긴 블록이다.
문맥만 보는 predictor 가 닿을 수 있는 최적은 문맥 흐름 U[t] 의 등속 외삽으로 옮긴 W_lag 쪽이다. 그 위약 뺀 데이터 최적과 모델을 비교한다.

  OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=<g> /venv/vjepa/bin/python -W ignore auto_research/scripts/transport_lag_placebo.py

W_lag = 블록 t 의 칸 s 내용을 s − U[t](s) 로 보냄 (class_transport 의 splat, 문맥만), W_lag_rot = U[t] 를 90° 돌려 같은 일.
W_true · W_rot (정답 흐름, 앞 판과 같음) 도 같이. 회귀 y ≈ a·m + b·W, 최소제곱 (L2) 과 데이터는 정확한 L1 최적 (lad_profile 과 같은 황금분할) 도."""
import json, os, sys
import numpy as np, torch, yaml

torch.set_num_threads(16)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
CACHE = "/workspace/auto_research_cache/ssv2_classes"
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1)
R90 = np.array([[0.0, -1.0], [1.0, 0.0]], np.float32)


def warp(ht, u):
    s = np.floor(P0 + u).astype(int)
    ok = (s >= 0).all(1) & (s < G).all(1)
    j = np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


def splat(ht, u0):
    d = np.floor(P0 - u0).astype(int)
    ok = (d >= 0).all(1) & (d < G).all(1)
    j = d[:, 1] * G + d[:, 0]
    acc = np.zeros_like(ht); n = np.zeros(S)
    np.add.at(acc, j[ok], ht[ok]); np.add.at(n, j[ok], 1)
    out = ht.copy(); m = n > 0
    out[m] = acc[m] / n[m][:, None]
    return out


def wmedian(v, w):
    o = torch.argsort(v); cw = torch.cumsum(w[o], 0)
    return v[o][torch.searchsorted(cw, cw[-1] / 2)]


def prof(y, m, W, b):
    r = y - b * W; ok = m.abs() > 1e-6
    a = wmedian(r[ok] / m[ok], m[ok].abs())
    return float((r - a * m).abs().mean())


def golden(y, m, W, lo=-0.2, hi=0.8, it=26):
    g = (5 ** 0.5 - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo); f1, f2 = prof(y, m, W, x1), prof(y, m, W, x2)
    for _ in range(it):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1; x1 = hi - g * (hi - lo); f1 = prof(y, m, W, x1)
        else:
            lo, x1, f1 = x1, x2, f2; x2 = lo + g * (hi - lo); f2 = prof(y, m, W, x2)
    b = (lo + hi) / 2
    return b, prof(y, m, W, b)


def ls_b(y, m, W):
    X = torch.stack([m, W], 1)
    return float(torch.linalg.solve(X.T @ X, X.T @ y)[1])


from analysis.intphys2.model import _clean_backbone_key
from src.models.rollout_predictor import vit_prefix_predictor
dev = torch.device("cuda")
D_ = os.path.join(REPO, "z_training/runs/natural_ar_vw_state")
pc = yaml.safe_load(open(os.path.join(D_, "config.yaml")))["model"]["predictor"]
model = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                             num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
model.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D_, "latest.pt"), map_location="cpu", weights_only=False)["predictor"]).items()
                       if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
model = model.to(dev).eval()

res = {}
rs = np.random.RandomState(0)
for grp in ("passive", "actor"):
    Hs = np.load(os.path.join(CACHE, f"{grp}_h.npy"), mmap_mode="r"); Us = np.load(os.path.join(CACHE, f"{grp}_U.npy")).astype(np.float32)
    D = {k: [] for k in ("y", "p", "cp", "m", "wt", "wr", "wl", "wlr")}
    for w in range(Hs.shape[0]):
        h = np.asarray(Hs[w]).astype(np.float32); U = Us[w].reshape(-1, S, 2)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hx = torch.from_numpy(h).to(dev)
            p_all = model.forward_seq(hx[:7].reshape(1, -1, 1280), torch.arange(7 * S, device=dev).unsqueeze(0)).float()[0].view(7, S, -1).cpu().numpy()
        for t in (4, 5, 6):
            sel = np.linalg.norm(U[t + 1], axis=1) >= 0.5
            if sel.sum() < 2:
                continue
            D["y"].append(h[t + 1][sel]); D["p"].append(p_all[t][sel]); D["cp"].append(h[t][sel]); D["m"].append(h[t - 4:t + 1].mean(0)[sel])
            D["wt"].append(warp(h[t], U[t + 1])[sel]); D["wr"].append(warp(h[t], U[t + 1] @ R90.T)[sel])
            D["wl"].append(splat(h[t], U[t])[sel]); D["wlr"].append(splat(h[t], U[t] @ R90.T)[sel])
    A = {k: torch.from_numpy(np.concatenate(v).ravel()) for k, v in D.items()}
    idx = torch.from_numpy(rs.choice(A["y"].numel(), min(30_000_000, A["y"].numel()), replace=False))
    A = {k: v[idx] for k, v in A.items()}
    e = {}
    for wname, (Wk, Wp) in (("true", ("wt", "wr")), ("lag", ("wl", "wlr"))):
        for nm in ("y", "p", "cp"):
            bt, br = ls_b(A[nm], A["m"], A[Wk]), ls_b(A[nm], A["m"], A[Wp])
            e[f"LS_{wname}_{nm}"] = dict(b=bt, b_rot=br, net=bt - br)
        f0 = prof(A["y"], A["m"], A[Wk], 0.0)
        bt, ft = golden(A["y"], A["m"], A[Wk]); br, fr = golden(A["y"], A["m"], A[Wp])
        e[f"L1_{wname}_data"] = dict(b=bt, b_rot=br, net=bt - br, reward_net=(fr - ft) / f0)
        print(f"{grp} W={wname}: LS 데이터 위약 뺀 {e[f'LS_{wname}_y']['net']:.4f} · 모델 {e[f'LS_{wname}_p']['net']:.4f} "
              f"(b {e[f'LS_{wname}_p']['b']:.3f} / rot {e[f'LS_{wname}_p']['b_rot']:.3f}) · 복사 {e[f'LS_{wname}_cp']['net']:.4f} · "
              f"L1 최적 데이터 위약 뺀 {bt - br:.4f} · L1 보상 {100 * (fr - ft) / f0:.3f} %", flush=True)
    res[grp] = e
out = os.path.join(REPO, "auto_research/results/transport_lag_placebo.json")
json.dump(res, open(out, "w"), indent=1)
print("저장:", out)
