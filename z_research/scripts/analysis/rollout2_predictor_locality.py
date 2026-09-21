#!/usr/bin/env python3
"""RollOut_v2 — predictor 의 미래 토큰이 문맥의 '어디' 에서 물체를 가져오는가 (공간 조회 범위). 2026-09-19.

가설 H1: 미래 토큰 (t,y,x) 은 물체 상태를 굴리지 않고 자기 공간 위치 근처의 문맥을 다시 읽는다 →
마지막 관측 자리에서 멀어진 미래 칸은 물체를 못 가져온다. 경쟁 H2: 전역 조회는 되지만 불확실성 평균으로 흐려진다.
H3: 문맥 encoder (z) 가 속도를 약하게 담는다.

frozen, 재학습 없음. 캐시를 만든 config (exp_results/attn_probe__rollout_v2_vith/_resolved.yaml) 그대로 WMADataset + bundle 을 만들고
문맥 encoder (online, 문맥 16 장) 를 다시 돌려 raw z 를 얻는다 — 캐시 ctx_masked 는 LN(z) 라 predictor 입력으로 못 쓴다
(evals/analysis_vlm/occlusion_identity/forward.py:163). 그 위에 hook 을 단 릴리즈 predictor. 먼저 캐시 p (rollout_v2_vith/predictor) 와 재현 대조.

  exp1 attention : 진실 물체 칸의 미래 토큰 (슬롯마다 1 개) 을 query 로, 층별 (head 평균) attention 이
                   (a) 물체의 문맥 궤적 3×3 칸 (b) query 자기 자리의 문맥 3×3 기둥 (c) 문맥 전체 에 얼마나 실리는지 + 문맥 key 까지의
                   attention 가중 공간 거리 (칸). 마지막 관측 자리에서 query 까지 거리로 나눠 본다.
  exp2 knockout  : 미래 query → 문맥 key 를 공간 Chebyshev 반경 r 칸 밖이면 막는다 (전 층). 미래↔미래, 문맥↔문맥 은 그대로.
                   p 가 얼마나 바뀌는지 + 자 없는 물체 적중 (argmax cos(p 토큰, LN(h) 진실 물체 토큰) 이 진실 1 칸 안) 을 거리별로.
  exp3 velocity  : z (문맥 encoder) / h (target encoder) 의 마지막 두 문맥 튜블릿 물체 3×3 토큰 평균 → 문맥 끝 속도 (vx, vy) ridge.
                   block 단위 반반 split. 대조: 같은 자리 토큰이 위치만 주고 속도는 없을 때의 기준 = 평균 예측 (R² 0).

  CUDA_VISIBLE_DEVICES=0 python z_research/scripts/analysis/rollout2_predictor_locality.py [--n 200] [--exp 1,2,3]
출력: z_research/RollOutV2/exp_results/v5/locality/{exp1_attention,exp2_knockout,exp3_velocity}.json + report.md
"""
from __future__ import annotations
import argparse, csv, json, math, re, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import yaml                                                                  # noqa: E402
from analysis.intphys2.model import build_from_config                      # noqa: E402
from evals.world_model_analysis.data import WMADataset                       # noqa: E402
from evals.analysis_vlm.occlusion_identity.forward import _context_target_indices   # noqa: E402
from src.models.utils.modules import rotate_queries_or_keys                  # noqa: E402

CACHE = Path("/local_datasets/world/world_analysis/cache")
Z_NPY, P_NPY, H_NPY = CACHE / "rollout_v2_z16_vith/ctx_masked.npy", CACHE / "rollout_v2_vith/predictor.npy", CACHE / "rollout_v2_vith/target.npy"
INDEX = ROOT / "data_csv/rollout_v2/index_probe.csv"
RESOLVED = ROOT / "z_research/RollOutV2/exp_results/attn_probe__rollout_v2_vith/_resolved.yaml"
OUT = ROOT / "z_research/RollOutV2/exp_results/v5/locality"
G, TC, TF, CELL = 16, 8, 8, 18.0            # 16×16 토큰, 문맥/미래 8 튜블릿, 1 칸 = 18 px (288 px 좌표계)
NCTX = TC * G * G
BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 200)]
RADII = [1, 2, 3, 5, 8]                      # + inf (= baseline)


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", s.strip()) if v])


def load_meta():
    ids = json.load(open(CACHE / "rollout_v2_vith/meta.json"))["video_ids"]
    assert ids == json.load(open(CACHE / "rollout_v2_z16_vith/meta.json"))["video_ids"]
    rows = {r["video_id"]: r for r in csv.DictReader(INDEX.open())}
    return ids, rows


def track(r):
    """튜블릿 16 개의 물체 중심 (px, 288 좌표) — 프레임 2 장 평균."""
    return np.stack([arr(r["px_x_by_sample"]).reshape(16, 2).mean(1), arr(r["px_y_by_sample"]).reshape(16, 2).mean(1)], 1)


def cell(xy):
    return np.clip((np.asarray(xy) // CELL).astype(int), 0, G - 1)          # (x, y) 칸


# ---------------------------------------------------------------- predictor + hooks
class Hook:
    """RoPEAttention.forward 대체. rec_rows 가 있으면 그 query 행의 softmax 를 층마다 모은다; kmask 가 있으면 sdpa attn_mask."""
    def __init__(self):
        self.rec_rows = None; self.kmask = None; self.store = []


def patch_attention(pred, hook):
    for blk in pred.predictor_blocks:
        att = blk.attn

        def fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, _a=att):
            B, N, C = x.size()
            qkv = _a.qkv(x).unflatten(-1, (3, _a.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            m = mask.unsqueeze(1).repeat(1, _a.num_heads, 1)
            d, h, w = _a.separate_positions(m, H_patches, W_patches)
            s = 0; qs, ks = [], []
            for pos, dim in ((d, _a.d_dim), (h, _a.h_dim), (w, _a.w_dim)):
                qs.append(rotate_queries_or_keys(q[..., s:s + dim], pos=pos)); ks.append(rotate_queries_or_keys(k[..., s:s + dim], pos=pos)); s += dim
            if s < _a.head_dim:
                qs.append(q[..., s:]); ks.append(k[..., s:])
            q = torch.cat(qs, -1); k = torch.cat(ks, -1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=hook.kmask)
            if hook.rec_rows is not None:                                  # (B, R) query 행
                qi = torch.gather(q, 2, hook.rec_rows[:, None, :, None].expand(-1, q.shape[1], -1, q.shape[-1]))
                logit = (qi.float() @ k.float().transpose(-2, -1)) * _a.scale
                if hook.kmask is not None:
                    km = hook.kmask[hook.rec_rows[0]]                     # 모든 clip 같은 반경일 때만 (exp1 은 kmask 없음)
                    logit = logit.masked_fill(~km[None, None], float("-inf"))
                hook.store.append(logit.softmax(-1).mean(1))             # (B, R, N) head 평균
            x = out.transpose(1, 2).reshape(B, N, C)
            return _a.proj_drop(_a.proj(x))
        att.forward = fwd


def build(device):
    cfg = yaml.safe_load(open(RESOLVED)); cfg.pop("probing", None)
    ds = WMADataset(cfg); bundle = build_from_config(cfg["model"], device)
    return ds, bundle


def ctx_z(ds, bundle, recidx, device):
    clips = torch.stack([ds.clip(i) for i in recidx]).to(device, bundle.dtype)
    ci, _ = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=G * G, batch_size=len(recidx), device=device)
    with torch.no_grad():
        z = bundle.context_encoder(clips, masks=[ci])
    return z[-1] if isinstance(z, list) else z


def run_pred(pred, z, device):
    B = z.shape[0]
    ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=G * G, batch_size=B, device=device)
    with torch.no_grad():
        return pred(z, ci, ti, mask_index=0)


def knock_mask(r, device):
    """(N,N) bool, True = 허용. 미래 query → 문맥 key 만 Chebyshev 반경 r 칸으로 제한."""
    i = torch.arange(2 * NCTX, device=device); hh, ww = (i % (G * G)) // G, i % G
    fut = i >= NCTX
    cheb = torch.maximum((hh[:, None] - hh[None]).abs(), (ww[:, None] - ww[None]).abs())
    allow = torch.ones(2 * NCTX, 2 * NCTX, dtype=torch.bool, device=device)
    blk = fut[:, None] & (~fut)[None] & (cheb > r)
    allow[blk] = False
    return allow


def ln(x):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-6)


def pf_hit(p, h, cells):
    """p (8,256,D), h=LN target 미래 (8,256,D), cells (8,2) 진실 칸 → 슬롯별 적중 (bool 8)."""
    pn = p / np.linalg.norm(p, axis=-1, keepdims=True); out = np.zeros(TF, bool)
    for t in range(TF):
        cx, cy = cells[t]; tpl = h[t, cy * G + cx]; tpl = tpl / np.linalg.norm(tpl)
        a = int((pn[t] @ tpl).argmax()); ay, ax = divmod(a, G); out[t] = max(abs(ax - cx), abs(ay - cy)) <= 1
    return out


# ---------------------------------------------------------------- 실험
def pick(ids, rows, scen, n, seed=0):
    rng = np.random.default_rng(seed); pos = {v: i for i, v in enumerate(ids)}; out = []
    for s in scen:
        vs = [v for v, r in rows.items() if r["scenario"] == s and r["plausible"] == "1" and v in pos]
        for v in rng.choice(vs, min(n, len(vs)), replace=False):
            out.append((s, v, pos[v]))
    return out


def exp12(args, ids, rows, device, do1, do2):
    Pm, Hm = (np.load(f, mmap_mode="r") for f in (P_NPY, H_NPY))
    ds, bundle = build(device); pred = bundle.predictor; hook = Hook(); patch_attention(pred, hook)
    ridx = {r.video_id: i for i, r in enumerate(ds.records)}
    clips = pick(ids, rows, ["flat_v", "flat_a", "flat_d", "arc"], args.n)
    kms = {r: knock_mask(r, device) for r in RADII} if do2 else {}
    # 문맥 key 좌표
    kt, kh, kw = np.arange(NCTX) // (G * G), (np.arange(NCTX) % (G * G)) // G, np.arange(NCTX) % G
    rec1, rec2, repro = [], [], []
    t0 = time.time()
    for bi in range(0, len(clips), args.bs):
        chunk = clips[bi:bi + args.bs]
        z = ctx_z(ds, bundle, [ridx[v] for _, v, _ in chunk], device)
        H = [ln(np.asarray(Hm[j], dtype=np.float32).reshape(16, G * G, -1)[TC:]) for _, _, j in chunk]
        XY = [track(rows[v]) for _, v, _ in chunk]
        fc = [np.stack([cell(xy[TC + t]) for t in range(TF)]) for xy in XY]           # 미래 진실 칸 (8,2) (x,y)
        dist = [np.linalg.norm(xy[TC:] - xy[TC - 1], axis=1) for xy in XY]
        # --- baseline (+ exp1 기록)
        hook.kmask = None; hook.store = []
        hook.rec_rows = torch.tensor([[NCTX + t * G * G + c[t, 1] * G + c[t, 0] for t in range(TF)] for c in fc], device=device) if do1 else None
        p0 = run_pred(pred, z, device).float().cpu().numpy()                         # (B, 2048, D)
        for b, (s, v, j) in enumerate(chunk):
            ref = np.asarray(Pm[j], dtype=np.float32)
            repro.append(float(np.abs(p0[b] - ref).mean() / np.abs(ref).mean()))
        if do1:
            A = torch.stack(hook.store, 1).cpu().numpy()                              # (B, L, R=8, N)
            for b, (s, v, j) in enumerate(chunk):
                xy = XY[b]; oc = np.stack([cell(xy[t]) for t in range(TC)])           # 문맥 물체 칸 (8,2)
                obj_key = np.zeros(NCTX, bool)
                for t in range(TC):
                    obj_key |= (kt == t) & (np.abs(kw - oc[t, 0]) <= 1) & (np.abs(kh - oc[t, 1]) <= 1)
                last_key = (kt == TC - 1) & (np.abs(kw - oc[-1, 0]) <= 1) & (np.abs(kh - oc[-1, 1]) <= 1)
                for t in range(TF):
                    qx, qy = fc[b][t]; own = (np.abs(kw - qx) <= 1) & (np.abs(kh - qy) <= 1)
                    dcell = np.sqrt((kw - qx) ** 2 + (kh - qy) ** 2)
                    a = A[b, :, t]                                                    # (L, N)
                    ac = a[:, :NCTX]; cm = ac.sum(1)
                    rec1.append({"scen": s, "slot": t, "dist": float(dist[b][t]), "ctx_mass": cm.tolist(),
                                 "obj_track": ac[:, obj_key & ~own].sum(1).tolist(), "obj_last": ac[:, last_key & ~own].sum(1).tolist(),
                                 "own_col": ac[:, own].sum(1).tolist(), "own_col_not_obj": ac[:, own & ~obj_key].sum(1).tolist(),
                                 "mean_dcell": ((ac * dcell).sum(1) / cm).tolist(),
                                 "n_obj_only": int((obj_key & ~own).sum()), "n_own": int(own.sum())})
        hook.rec_rows = None
        # --- exp2 knockout
        if do2:
            base_hit = [pf_hit(p0[b].reshape(TF, G * G, -1), H[b], fc[b]) for b in range(len(chunk))]
            res = {"inf": (base_hit, [np.zeros(TF)] * len(chunk))}
            for r, km in kms.items():
                hook.kmask = km
                pr = run_pred(pred, z, device).float().cpu().numpy()
                hits = [pf_hit(pr[b].reshape(TF, G * G, -1), H[b], fc[b]) for b in range(len(chunk))]
                dch = [np.abs(pr[b] - p0[b]).reshape(TF, -1).mean(1) / np.abs(p0[b]).reshape(TF, -1).mean(1) for b in range(len(chunk))]
                res[str(r)] = (hits, dch)
            hook.kmask = None
            for b, (s, v, j) in enumerate(chunk):
                for t in range(TF):
                    rec2.append({"scen": s, "slot": t, "dist": float(dist[b][t]),
                                 **{f"hit_{k}": bool(h_[b][t]) for k, (h_, _) in res.items()},
                                 **{f"dp_{k}": float(d_[b][t]) for k, (_, d_) in res.items() if k != "inf"}})
        if bi == 0 or (bi // args.bs) % 20 == 0:
            print(f"  {bi + len(chunk)}/{len(clips)} clips  {time.time() - t0:.0f}s  repro rel-L1 max {max(repro):.4f}", flush=True)
    return rec1, rec2, repro


def exp3(ids, rows):
    Zm = np.load(Z_NPY, mmap_mode="r"); Hm = np.load(H_NPY, mmap_mode="r")
    pos = {v: i for i, v in enumerate(ids)}
    X = {"z": [], "h": []}; Y, blocks, scen = [], [], []
    for v, r in rows.items():
        if r["plausible"] != "1" or v not in pos:
            continue
        xy = track(r); vel = xy[TC - 1] - xy[TC - 2]; j = pos[v]
        feats = {"z": [], "h": []}
        for t in (TC - 2, TC - 1):
            cx, cy = cell(xy[t]); keys = [t * G * G + yy * G + xx for yy in range(cy - 1, cy + 2) for xx in range(cx - 1, cx + 2) if 0 <= yy < G and 0 <= xx < G]
            feats["z"].append(np.asarray(Zm[j, keys], dtype=np.float32).mean(0))
            feats["h"].append(ln(np.asarray(Hm[j, keys], dtype=np.float32)).mean(0))
        for k in X:
            X[k].append(np.concatenate(feats[k]))
        Y.append(vel); blocks.append(r["block_id"]); scen.append(r["scenario"])
    Y = np.array(Y); blocks = np.array(blocks); ub = np.unique(blocks)
    rng = np.random.default_rng(0); tr_b = set(rng.choice(ub, len(ub) // 2, replace=False)); tr = np.array([b in tr_b for b in blocks])
    out = {"n_train": int(tr.sum()), "n_test": int((~tr).sum()), "target": "context-end velocity (px/tubelet), tubelet 6->7",
           "y_std": Y[~tr].std(0).tolist()}
    for k in X:
        A = np.array(X[k]); mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6; A = (A - mu) / sd
        best = None
        for lam in (1, 10, 100, 1000, 10000):                                        # train 안 5-fold 로 λ
            f = np.arange(tr.sum()) % 5; At, Yt = A[tr], Y[tr]; err = 0
            for q in range(5):
                a, b_ = At[f != q], Yt[f != q]; W = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ (b_ - b_.mean(0)))
                err += ((At[f == q] @ W + b_.mean(0) - Yt[f == q]) ** 2).sum()
            best = (err, lam) if best is None or err < best[0] else best
        lam = best[1]; W = np.linalg.solve(A[tr].T @ A[tr] + lam * np.eye(A.shape[1]), A[tr].T @ (Y[tr] - Y[tr].mean(0)))
        yp = A[~tr] @ W + Y[tr].mean(0); res = yp - Y[~tr]
        r2 = 1 - (res ** 2).sum(0) / ((Y[~tr] - Y[~tr].mean(0)) ** 2).sum(0)
        sp_true = np.linalg.norm(Y[~tr], axis=1); sp_pred = np.linalg.norm(yp, axis=1)
        out[k] = {"lambda": lam, "r2_vx": float(r2[0]), "r2_vy": float(r2[1]), "mae_vx": float(np.abs(res[:, 0]).mean()),
                  "mae_vy": float(np.abs(res[:, 1]).mean()), "r2_speed": float(1 - ((sp_pred - sp_true) ** 2).sum() / ((sp_true - sp_true.mean()) ** 2).sum()),
                  "per_scen_mae": {s: float(np.linalg.norm(res[np.array(scen)[~tr] == s], axis=1).mean()) for s in sorted(set(scen))}}
    return out


def summarize(rec1, rec2, repro, e3):
    L = []
    L.append("# predictor 공간 조회 범위 — exp1 attention · exp2 knockout · exp3 속도 probe (2026-09-19)\n")
    L.append("exp3 입력은 캐시 LN(z) (ctx_masked) / LN(h). exp1·2 는 문맥 encoder 재실행 raw z → predictor.\n")
    L.append(f"재현: 캐시 p 대비 재계산 p 상대 L1 — 평균 {np.mean(repro):.4f}, 최대 {np.max(repro):.4f} (n={len(repro)})\n")
    if rec1:
        nL = len(rec1[0]["ctx_mass"]); d = np.array([r["dist"] for r in rec1]); fl = np.array([r["scen"] != "arc" for r in rec1])
        L.append("## exp1 — 진실 물체 칸 미래 query 의 attention (head 평균, 층 평균 / 마지막 층)\n")
        L.append(f"문맥 key 2048 개. own = query 자기 자리 3×3 × 문맥 8 튜블릿 (72 key, 균등이면 문맥 질량의 {72/2048:.3f}). "
                 "obj_track = 물체 문맥 궤적 3×3 (own 과 겹치는 key 제외). obj_last = 마지막 문맥 튜블릿 물체 3×3 (own 제외). "
                 "mean_dcell = 문맥 key 까지 attention 가중 공간 거리 (칸, 균등이면 약 6.5).\n")
        for scope, msk in (("flat", fl), ("arc", ~fl)):
            L.append(f"\n### {scope}\n")
            L.append("| 거리 (px) | n | ctx 질량 | own 기둥 | obj_track (own 밖) | obj_last (own 밖) | mean_dcell | 마지막층 own | 마지막층 obj_track |")
            L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for a, b in BINS:
                k = msk & (d >= a) & (d < b)
                if k.sum() < 20:
                    continue
                g = lambda key, layer=None: np.mean([np.mean(r[key]) if layer is None else r[key][layer] for r, kk in zip(rec1, k) if kk])
                L.append(f"| {a}–{b} | {k.sum()} | {g('ctx_mass'):.3f} | {g('own_col'):.3f} | {g('obj_track'):.3f} | {g('obj_last'):.3f} | "
                         f"{g('mean_dcell'):.2f} | {g('own_col', nL - 1):.3f} | {g('obj_track', nL - 1):.3f} |")
        L.append("\n층별 (flat, 전 거리): own 기둥 / obj_track / mean_dcell\n")
        L.append("| 층 | " + " | ".join(str(i) for i in range(nL)) + " |"); L.append("|---|" + "---:|" * nL)
        for key, fmt in (("own_col", "{:.3f}"), ("obj_track", "{:.3f}"), ("mean_dcell", "{:.2f}")):
            L.append(f"| {key} | " + " | ".join(fmt.format(np.mean([r[key][i] for r, kk in zip(rec1, fl) if kk])) for i in range(nL)) + " |")
    if rec2:
        d = np.array([r["dist"] for r in rec2]); fl = np.array([r["scen"] != "arc" for r in rec2])
        L.append("\n## exp2 — 미래→문맥 attention 을 공간 반경 r 칸으로 제한 (전 층)\n")
        L.append("자 없는 적중 (argmax cos(p, LN(h) 진실 물체 토큰) 이 진실 1 칸 안). r=inf 가 원래 predictor. 괄호 = p 의 상대 변화 (평균 |Δp|/|p|).\n")
        for scope, msk in (("flat", fl), ("arc", ~fl)):
            L.append(f"\n### {scope}\n")
            L.append("| 거리 (px) | n | " + " | ".join(f"r={r}" for r in RADII) + " | r=inf |")
            L.append("|---|---:|" + "---:|" * (len(RADII) + 1))
            for a, b in BINS:
                k = msk & (d >= a) & (d < b)
                if k.sum() < 20:
                    continue
                cells = [f"{np.mean([rec2[i][f'hit_{r}'] for i in np.where(k)[0]]):.2f} ({np.mean([rec2[i][f'dp_{r}'] for i in np.where(k)[0]]):.2f})" for r in RADII]
                L.append(f"| {a}–{b} | {k.sum()} | " + " | ".join(cells) + f" | {np.mean([rec2[i]['hit_inf'] for i in np.where(k)[0]]):.2f} |")
    if e3:
        L.append("\n## exp3 — 문맥 끝 속도 ridge probe (마지막 두 문맥 튜블릿 물체 3×3 토큰 평균, block 반반)\n")
        L.append(f"n train {e3['n_train']} / test {e3['n_test']}. 진실 속도 표준편차 (vx, vy) = ({e3['y_std'][0]:.1f}, {e3['y_std'][1]:.1f}) px/튜블릿.\n")
        L.append("| 표현 | R² vx | R² vy | R² 속력 | MAE vx | MAE vy |"); L.append("|---|---:|---:|---:|---:|---:|")
        for k in ("z", "h"):
            e = e3[k]; L.append(f"| {k} | {e['r2_vx']:.3f} | {e['r2_vy']:.3f} | {e['r2_speed']:.3f} | {e['mae_vx']:.2f} | {e['mae_vy']:.2f} |")
        L.append("\n시나리오별 오차 (|Δv| px/튜블릿): " + "; ".join(f"{s} z {e3['z']['per_scen_mae'][s]:.1f} / h {e3['h']['per_scen_mae'][s]:.1f}" for s in e3["z"]["per_scen_mae"]))
    L.append("\n## 재현\n\n```\nCUDA_VISIBLE_DEVICES=0 /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/rollout2_predictor_locality.py --n 200\n```\n"
             "입력: 캐시 rollout_v2_z16_vith/ctx_masked (문맥), rollout_v2_vith/{predictor,target}, data_csv/rollout_v2/index_probe.csv, "
             "predictor z_training/runs/release_vith/latest.pt (bf16).")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--exp", default="1,2,3"); a = ap.parse_args(); ex = set(a.exp.split(","))
    OUT.mkdir(parents=True, exist_ok=True); ids, rows = load_meta(); device = torch.device("cuda")
    rec1 = rec2 = []; repro = [0.0]; e3 = None
    if "1" in ex or "2" in ex:
        rec1, rec2, repro = exp12(a, ids, rows, device, "1" in ex, "2" in ex)
        json.dump(rec1, open(OUT / "exp1_attention.json", "w")); json.dump(rec2, open(OUT / "exp2_knockout.json", "w"))
    if "3" in ex:
        e3 = exp3(ids, rows); json.dump(e3, open(OUT / "exp3_velocity.json", "w"), indent=1)
    rep = summarize(rec1, rec2, repro, e3); open(OUT / "report.md", "w").write(rep); print(rep)


if __name__ == "__main__":
    main()
