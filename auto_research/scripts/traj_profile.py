#!/usr/bin/env python3
"""[P] 실패 모양 판정 — "평균 (A)" 인가 "중앙값 소실 (B)" 인가 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=4 /venv/vjepa/bin/python -W ignore auto_research/scripts/traj_profile.py --run natural_ar_vw_state latest.pt

A (시간 평균 필터): 미래 블록의 물체 점수 지도 = 문맥 블록 점수 지도의 가중합 → 물체가 옛 자리 · 궤적 위에 흐린 잔상으로 남고,
   지연 k 자리의 존재감 ≈ 그 블록의 가중치 w_k (누적). 물체 총량 (궤적 근처 양의 대비 합) 은 보존된다.
B (칸별 중앙값 소실): 물체가 어느 칸에도 안 남는다 (궤적 위 존재감 ≈ 정답 수준), 총량이 0 으로, 이동 거리 / 물체 크기에 따라 문턱처럼 꺼진다.
측정 (3D 합성 검증 세트 캐시, 굴린 예측 문맥 0..7 → 블록 8..15 / TF 한 걸음 t = 7..14):
  s(x) = x · T̂ (T̂ = 블록 7 (TF 는 블록 t) 물체 칸 평균 토큰, 정규화), 대비 = s − (먼 칸 평균 s).
  1 궤적 프로필: 지연 k (블록 7−k) 의 "전용 칸" E_k = 그 블록 물체 칸 − 더 최근 문맥 블록 물체 칸 − 지금 정답 물체 칸 (1 칸 팽창).
     r_k = E_k 의 예측 대비 / E_k 의 그 블록 (h_{7−k}) 대비. 같은 값을 정답 (바닥) · 같은 칸 재귀 선형 AR (A 모형, 이 모델의 먼 칸 적합 가중치) 에.
  2 물체 총량 M(x) = Σ_{궤적 근처 칸} max(대비, 0), 정답 대비 비. 3 TF 한 걸음 존재감 (새 칸) · 잔상 (떠난 칸) 을 이동 거리 / 물체 폭 구간별로."""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _clean_backbone_key                       # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, required=True)
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/traj_profile"))
a = ap.parse_args()
dev = torch.device("cuda")
S, G, C, K, NB, L = 256, 16, 8, 8, 16, 8
tag = f"{a.run[0]}__{a.run[1].replace('.pt', '')}"
os.makedirs(a.out, exist_ok=True)
W = json.load(open(os.path.join(REPO, "auto_research/results/pred_mean", tag + ".json")))["tf"]["mov/far"]["pred"]["M0_w"]

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


PROF = {"ro": [], "tf": []}       # (j 또는 t, k, 예측 r · 정답 r · A r · copy r, 이동 클래스)
MASS = []
BIN = []
for f in sorted(glob.glob(os.path.join(a.cache, "h_val_target_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    X = np.load(f, mmap_mode="r"); lab = np.load(os.path.join(a.cache, f"lab_val_s{sh}.npz"))
    for w in range(X.shape[0]):
        h = torch.from_numpy(np.asarray(X[w], dtype=np.float32)).to(dev)
        cv = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0))
        objb = (np.maximum(cv[0::2], cv[1::2]) >= 0.25).reshape(NB, S)
        touch = ((cv[0::2] > 0) | (cv[1::2] > 0)).reshape(NB, S)
        ok = lab["inframe_frac"][w] > 0.9
        visb = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & ok[0::2] & ok[1::2]
        far = ~dil(touch.any(0), 2)
        if far.sum() < 10:
            continue
        farm = torch.from_numpy(far).to(dev)
        pos = lab["pos"][w]; bw = np.nanmedian(lab["bbox"][w][:, 2] - lab["bbox"][w][:, 0])
        nbhd = torch.from_numpy(dil(touch.any(0), 2)).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p_ro = model.rollout_kv(h[:C].reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, -1)
            p_tf = model.forward_seq(h[:NB - 1].reshape(1, -1, h.size(-1)), torch.arange((NB - 1) * S, device=dev).unsqueeze(0)).float()[0].view(NB - 1, S, -1)
        seq = [h[b] for b in range(C)]; la = []
        for j in range(K):
            nxt = sum(float(W[k]) * seq[-1 - k] for k in range(L)); la.append(nxt); seq.append(nxt)

        def con(x, T_hat):
            s_ = x @ T_hat
            return s_ - s_[..., farm].mean(-1, keepdim=True)

        # ---- 굴린 예측
        if visb[C - 1] and objb[C - 1].any():
            T7 = F.normalize(h[C - 1][torch.from_numpy(objb[C - 1]).to(dev)].mean(0), dim=-1)
            cctx = con(h[:C], T7)                                                     # (8, S) 문맥 블록 대비
            cpred, ctrue, cla = con(p_ro, T7), con(h[C:], T7), con(torch.stack(la), T7)
            ccopy = cctx[C - 1:C].expand(K, S)
            for j in range(K):
                b = C + j
                if not visb[b]:
                    continue
                D = float(np.linalg.norm(pos[2 * b + 1] - pos[2 * C - 1]))
                if not np.isfinite(D):
                    continue
                cls = "static" if D < 4 else ("moving" if D > 16 else None)
                cur = dil(objb[b], 1)
                later = np.zeros(S, bool)
                for k in range(L):
                    E = objb[C - 1 - k] & ~later & ~cur
                    later |= objb[C - 1 - k]
                    if E.sum() == 0 or not visb[C - 1 - k]:
                        continue
                    Em = torch.from_numpy(E).to(dev)
                    den = float(cctx[C - 1 - k][Em].mean())
                    if den < 2.0:
                        continue
                    PROF["ro"].append((j, k, float(cpred[j][Em].mean()) / den, float(ctrue[j][Em].mean()) / den, float(cla[j][Em].mean()) / den,
                                       float(ccopy[j][Em].mean()) / den, cls, D / bw))
                if cls is not None:
                    mt = float(ctrue[j][nbhd].clamp(min=0).sum())
                    MASS.append((j, cls, float(cpred[j][nbhd].clamp(min=0).sum()) / mt, float(cla[j][nbhd].clamp(min=0).sum()) / mt,
                                 float(ccopy[j][nbhd].clamp(min=0).sum()) / mt, D / bw))
        # ---- TF 한 걸음: 궤적 프로필 + 이동 거리 구간별 존재감 · 잔상
        for t in range(C - 1, NB - 1):
            if not (visb[t] and visb[t + 1] and objb[t].any()):
                continue
            d = pos[2 * t + 3] - pos[2 * t + 1]; dn = float(np.linalg.norm(d))
            if not np.isfinite(dn):
                continue
            Tt = F.normalize(h[t][torch.from_numpy(objb[t]).to(dev)].mean(0), dim=-1)
            cc = con(h[t - L + 1:t + 2], Tt)                                          # 블록 t−7..t+1 (9, S)
            cp = con(p_tf[t], Tt)
            cl = sum(float(W[k]) * cc[L - 1 - k] for k in range(L))                    # A 모형 (한 걸음)
            cur = dil(objb[t + 1], 1)
            later = np.zeros(S, bool)
            for k in range(L):
                E = objb[t - k] & ~later & ~cur
                later |= objb[t - k]
                if E.sum() == 0 or not visb[t - k]:
                    continue
                Em = torch.from_numpy(E).to(dev)
                den = float(cc[L - 1 - k][Em].mean())
                if den < 2.0:
                    continue
                PROF["tf"].append((t, k, float(cp[Em].mean()) / den, float(cc[L][Em].mean()) / den, float(cl[Em].mean()) / den,
                                   float(cc[L - 1][Em].mean()) / den, "moving" if dn > 8 else ("static" if dn < 1 else None), dn / bw))
            if dn > 4:
                new = objb[t + 1] & ~objb[t]; old = objb[t] & ~objb[t + 1]
                rec = dict(r=dn / bw)
                for g, msk in (("new", new), ("old", old)):
                    if msk.any():
                        mm = torch.from_numpy(msk).to(dev)
                        rec[g] = (float(cp[mm].mean()), float(cc[L][mm].mean()), float(cc[L - 1][mm].mean()), float(cl[mm].mean()))
                mt = float(cc[L][nbhd].clamp(min=0).sum())
                rec["mass"] = (float(cp[nbhd].clamp(min=0).sum()) / mt, float(cl[nbhd].clamp(min=0).sum()) / mt, float(cc[L - 1][nbhd].clamp(min=0).sum()) / mt)
                BIN.append(rec)
    print(f, flush=True)

out = {"run": a.run, "W": W, "profile": {}, "mass": {}, "bins": {}}
print(f"\n== {tag}: 궤적 프로필 r_k (전용 칸 대비 / 그 블록 대비) — 예측 · 정답 (바닥) · A 모형 · 복사, 그리고 A 모형 가중치 누적 참고 w = {np.round(W, 2).tolist()}")
for mode in ("tf", "ro"):
    P = PROF[mode]
    for cls in ("moving",):
        print(f"  [{mode}] {cls}")
        out["profile"][f"{mode}/{cls}"] = {}
        for k in range(L):
            R = np.array([r[2:6] for r in P if r[1] == k and r[6] == cls])
            if len(R) < 10:
                continue
            m = R.mean(0)
            out["profile"][f"{mode}/{cls}"][k] = dict(n=len(R), pred=m[0], true=m[1], A=m[2], copy=m[3])
            print(f"    지연 {k}: n {len(R):4d}  예측 {m[0]:.2f} · 정답 {m[1]:.2f} · A {m[2]:.2f} · 복사 {m[3]:.2f}")
print("\n== 굴린 예측 물체 총량 (궤적 근처 양의 대비 합 / 정답) — 예측 · A 모형 · 복사")
for cls in ("static", "moving"):
    for j in range(K):
        R = np.array([r[2:5] for r in MASS if r[0] == j and r[1] == cls])
        if len(R) < 5:
            continue
        m = np.median(R, 0)
        out["mass"][f"{cls}/{j}"] = dict(n=len(R), pred=m[0], A=m[1], copy=m[2])
        print(f"  {cls:7s} 걸음 {j + 1}: n {len(R):3d} 예측 {m[0]:.2f} · A {m[1]:.2f} · 복사 {m[2]:.2f}")
print("\n== TF 한 걸음 — 이동 거리 / 물체 폭 구간별: 새 칸 존재감 (예측−복사)/(정답−복사) · 떠난 칸 잔상 (예측−정답)/(복사−정답) · 총량 (예측 / A / 복사, 중앙값)")
for lo, hi in ((0.15, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 9)):
    R = [r for r in BIN if lo <= r["r"] < hi]
    if len(R) < 8:
        continue
    nw = np.array([r["new"] for r in R if "new" in r]); od = np.array([r["old"] for r in R if "old" in r]); ms = np.median([r["mass"] for r in R], 0)
    pres = (nw[:, 0].mean() - nw[:, 2].mean()) / (nw[:, 1].mean() - nw[:, 2].mean()) if len(nw) > 4 else np.nan
    presA = (nw[:, 3].mean() - nw[:, 2].mean()) / (nw[:, 1].mean() - nw[:, 2].mean()) if len(nw) > 4 else np.nan
    trail = (od[:, 0].mean() - od[:, 1].mean()) / (od[:, 2].mean() - od[:, 1].mean()) if len(od) > 4 else np.nan
    trailA = (od[:, 3].mean() - od[:, 1].mean()) / (od[:, 2].mean() - od[:, 1].mean()) if len(od) > 4 else np.nan
    out["bins"][f"{lo}-{hi}"] = dict(n=len(R), pres=pres, presA=presA, trail=trail, trailA=trailA, mass=ms.tolist())
    print(f"  {lo:.2f}–{hi:.2f} n {len(R):4d}: 존재감 {pres:.2f} (A {presA:.2f}) · 잔상 {trail:.2f} (A {trailA:.2f}) · 총량 {ms[0]:.2f} / {ms[1]:.2f} / {ms[2]:.2f}")
json.dump(out, open(os.path.join(a.out, tag + ".json"), "w"), indent=1, default=float)
