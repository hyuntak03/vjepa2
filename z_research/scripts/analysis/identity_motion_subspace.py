"""정체 (모양 · 색) 와 움직임 (속도 · 위치) 이 같은 부분공간을 쓰나 — 공간 평균 선형 probe 부분공간 겹침 (2026-10-25, 학습 없음).
사용자: "identity 구분하는 linear probe 랑 위치 구분하는 linear probe weight svd 하면 subspace 가 겹치는지 알 수 있지 않아? (T*N, D) → (T, D) 로 spatial pooling".

데이터: phys3d_single_v1 val 1,039 편 (13 시나리오 × 80, 12 fps 75 장, 물체 1 개, 카메라 고정). 3 장 간격 24 장 → 블록 12 개 (4 fps, 학습과 같다).
표현 (블록마다 칸 256 개 평균 → 1280):
  h       정답 블록 토큰 LN(target_encoder)
  tf_<m>  모델 m 의 teacher forcing 한 걸음 예측 (블록 1..11, 정답 과거를 다 봄)
  roll_<m> 모델 m 의 굴린 예측 (문맥 4 블록 → 블록 4..11)
  m = base (natural_ar_vw_state) · syn (natural_ar_vw_state_phys3d_only, 3D 합성만으로 학습 = 물체를 옮기는 모델)
라벨 (그 블록 두 장 기준, 보이는 블록만: vis_frac ≥ 0.5 · inframe ≥ 0.9): 모양 6 · 색 8 · 위치 (x, y) px · 블록 안 변위 (vx, vy) px/0.25 초.
probe: 표준화 → ridge (λ 는 train 안 5 겹 CV), 분류는 중심화 one-hot 회귀 → argmax. split = 편 단위 반반.
부분공간: probe 가중 W 그대로 (사용자 안) 와 활성 패턴 A = Σ_x W (Haufe 2014; 상관된 특징에서 가중은 Σ⁻¹ 로 돌아가 '어디에 담겼나' 를 직접 말하지 않는다) 둘 다.
  겹침 = ‖U₁ᵀU₂‖²_F / min(r₁, r₂) (주각 cos² 평균, 0 = 직교 · 1 = 포함). 귀무 = 같은 차원의 무작위 부분공간 (가중: 등방 · 패턴: Σ_x g) 200 회.
기능 검사: 정체 probe 를 느린 블록 (속도 하위 절반) 에서 학습 → 빠른 블록에 적용 (속도에 따라 정체 부호가 바뀌나), 표현별 속도 구간 정확도.

  CUDA_VISIBLE_DEVICES=3 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/identity_motion_subspace.py           # 특징 추출 (캐시) + probe
  /venv/vjepa/bin/python z_research/scripts/analysis/identity_motion_subspace.py --probe_only                                  # 캐시로 probe 만
출력: /workspace/auto_research_cache/idmotion/feats.npz · auto_research/results/identity_motion_subspace.json
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, REPO)
DATA = "/workspace/dataset/training_data/phys3d_single_v1/val"
CACHE = "/workspace/auto_research_cache/idmotion"
OUT = os.path.join(REPO, "auto_research/results/identity_motion_subspace.json")
S, NB, C, TUB = 256, 16, 4, 2
IDX = list(range(0, 3 * 2 * NB, 3))                                            # 최대 32 장 (12 fps 에서 3 장 간격 = 4 fps). 편 길이 48–96 장 → 블록 8–16 개, 모자란 자리는 NaN
MODELS = {"base": "natural_ar_vw_state", "syn": "natural_ar_vw_state_phys3d_only"}
ap = argparse.ArgumentParser(); ap.add_argument("--probe_only", action="store_true"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--models", default="", help="쉼표로 base,syn. 비우면 encoder (정답 블록 토큰 h) 만 — 사용자: \"encoder feature 에서 먼저 확인\"")
a = ap.parse_args()
MODELS = {k: MODELS[k] for k in a.models.split(",") if k}
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)


def labels(npz_path, nb):
    z = np.load(npz_path)
    pos, vis, inf = z["pos"][:, 0], z["vis_frac"][:, 0], z["inframe_frac"][:, 0]
    out = []
    for b in range(NB):
        if b >= nb:
            out.append([0.0] * 5); continue
        f0, f1 = IDX[2 * b], IDX[2 * b + 1]
        ok = min(vis[f0], vis[f1]) >= 0.5 and min(inf[f0], inf[f1]) >= 0.9
        out.append([float(ok), *(0.5 * (pos[f0] + pos[f1])), *(pos[f1] - pos[f0])])
    return np.array(out, np.float32), int(z["shape_id"][0]), int(z["color_id"][0])


def extract():
    import torch, yaml, decord
    sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
    _argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
    import ar_crps_ft as FT
    sys.argv = _argv
    from app.vjepa_frozen import utils as U, ar as AR
    from app.vjepa_frozen.data import normalize_clips
    dev = torch.device("cuda")
    cfg = yaml.safe_load(open(f"{REPO}/z_training/runs/natural_ar_vw_state/config.yaml"))
    state = U.load_base_checkpoint(cfg["model"]["checkpoint"])
    _, enc = U.build_frozen_encoders(cfg["model"], state, dev, torch.bfloat16, int(cfg["data"]["n_frames"])); del state
    preds = {k: FT.load_pred(r, "latest.pt").to(dev).eval() for k, r in MODELS.items()}
    vids = [r[0].split()[0] for r in csv.reader(open(f"{DATA}/index.csv"))]   # 줄 = "경로 라벨"
    if a.limit:
        vids = vids[:: max(1, len(vids) // a.limit)][: a.limit]
    F = {k: [] for k in ["h"] + [f"{p}_{m}" for m in MODELS for p in ("tf", "roll")]}
    L, SH, CO, SC = [], [], [], []
    idx = torch.arange(NB * S, device=dev).unsqueeze(0)
    pad = lambda x: np.concatenate([x, np.full((NB - x.shape[0], x.shape[1]), np.nan, np.float32)])
    for i, v in enumerate(vids):
        vr = decord.VideoReader(v, num_threads=4)
        nb = min(NB, len(vr) // 6)                                                   # 블록 b = 원본 6b · 6b+3 장
        fr = vr.get_batch(IDX[: 2 * nb]).asnumpy()                                   # (2nb, 256, 256, 3)
        x = normalize_clips(torch.from_numpy(fr).permute(3, 0, 1, 2)[None].to(dev))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = AR.encode_blocks(enc, x.to(torch.bfloat16), TUB)                       # (1, nb·S, D) LN
            F["h"].append(pad(h.float().view(nb, S, -1).mean(1).cpu().numpy()))
            for m, P in preds.items():
                tf = P.forward_seq(h[:, :-S], idx[:, :(nb - 1) * S]).float().view(nb - 1, S, -1).mean(1)          # 블록 1..nb−1 예측
                ro = P.rollout_kv(h[:, :C * S], idx[:, :C * S], nb - C).float().view(nb - C, S, -1).mean(1)      # 블록 C..nb−1
                F[f"tf_{m}"].append(pad(np.concatenate([np.full((1, tf.size(1)), np.nan, np.float32), tf.cpu().numpy()])))
                F[f"roll_{m}"].append(pad(np.concatenate([np.full((C, ro.size(1)), np.nan, np.float32), ro.cpu().numpy()])))
        lab, s_, c_ = labels(v.replace(".mp4", "_states.npz"), nb)
        L.append(lab); SH.append(s_); CO.append(c_); SC.append(v.split("/")[-2])
        if i % 100 == 0:
            log(f"{i}/{len(vids)}")
    os.makedirs(CACHE, exist_ok=True)
    np.savez(f"{CACHE}/feats.npz", **{k: np.stack(x).astype(np.float32) for k, x in F.items()}, lab=np.stack(L), shape=np.array(SH),
             color=np.array(CO), scen=np.array(SC), vids=np.array(vids))


# ------------------------------------------------------------------ probe
def ridge_fit(X, Y, lams=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)):
    """X 표준화 끝, Y 중심화 끝. 5 겹 CV 로 λ → (W, λ)."""
    n = X.shape[0]; fold = np.arange(n) % 5
    U_, s, Vt = np.linalg.svd(X, full_matrices=False)
    best = None
    for lam in lams:
        err = 0.0
        for f in range(5):
            tr = fold != f
            u, ss, vt = np.linalg.svd(X[tr], full_matrices=False)
            W = vt.T @ ((ss / (ss ** 2 + lam))[:, None] * (u.T @ Y[tr]))
            err += ((X[~tr] @ W - Y[~tr]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, lam)
    lam = best[1]
    return Vt.T @ ((s / (s ** 2 + lam))[:, None] * (U_.T @ Y)), lam


def onehot(y, k):
    o = np.zeros((len(y), k), np.float32); o[np.arange(len(y)), y] = 1
    return o


def orth(M, r):
    u, s, _ = np.linalg.svd(M, full_matrices=False)
    return u[:, :r]


def overlap(A, B):
    return float((np.linalg.norm(A.T @ B) ** 2) / min(A.shape[1], B.shape[1]))


def probe(Fz, lab, shape, color, clip_tr, key, rng, scen=None, half=None):
    """Fz (N, NB, D) 의 한 표현. 보이는 블록 · 그 표현이 있는 블록만.
    scen 이 있으면 위치 · 속도 라벨과 특징에서 시나리오 평균 (train 으로 잰 것) 을 뺀다 — 장면 배치가 위치 · 속도를 대신 알려 주는 교란 제거.
    half 가 있으면 (편 단위 0/1) train 을 반으로 나눠 같은 probe 의 부분공간 재현도 (반 A vs 반 B 겹침) 를 같이 잰다."""
    ok = (lab[..., 0] > 0) & ~np.isnan(Fz).any(-1)
    ci, bi = np.nonzero(ok)
    X = Fz[ci, bi].copy(); tr = clip_tr[ci]
    tg = {"shape": onehot(shape[ci], 6), "color": onehot(color[ci], 8), "pos": lab[ci, bi, 1:3].copy(), "vel": lab[ci, bi, 3:5].copy()}
    if scen is not None:
        sc = scen[ci]
        for u in np.unique(sc):
            m = sc == u; mt = m & tr
            X[m] -= X[mt].mean(0)
            for t in ("pos", "vel"):
                tg[t][m] -= tg[t][mt].mean(0)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    rank = {"shape": 5, "color": 7, "pos": 2, "vel": 2}
    res, W, Pat = {"n_train": int(tr.sum()), "n_test": int((~tr).sum())}, {}, {}
    Sig = np.cov(Xs[tr], rowvar=False)
    for t, Y in tg.items():
        ym = Y[tr].mean(0)
        w, lam = ridge_fit(Xs[tr], Y[tr] - ym)
        yh = Xs[~tr] @ w + ym
        if t in ("shape", "color"):
            res[t] = round(100 * float((yh.argmax(1) == Y[~tr].argmax(1)).mean()), 1)
        else:
            res[t] = round(float(1 - ((yh - Y[~tr]) ** 2).sum() / ((Y[~tr] - Y[~tr].mean(0)) ** 2).sum()), 3)
        W[t] = orth(w, rank[t]); Pat[t] = orth(Sig @ w, rank[t])
    # 겹침 + 귀무
    ev, evec = np.linalg.eigh(Sig); ev = np.clip(ev, 0, None)
    sq = evec * np.sqrt(ev)
    pairs = [("shape", "vel"), ("color", "vel"), ("shape", "pos"), ("color", "pos"), ("pos", "vel"), ("shape", "color")]
    ov = {}
    for p, q in pairs:
        null_w = [overlap(orth(rng.standard_normal((Xs.shape[1], rank[p])), rank[p]), orth(rng.standard_normal((Xs.shape[1], rank[q])), rank[q])) for _ in range(200)]
        null_p = [overlap(orth(sq @ rng.standard_normal((Xs.shape[1], rank[p])), rank[p]), orth(sq @ rng.standard_normal((Xs.shape[1], rank[q])), rank[q])) for _ in range(200)]
        ov[f"{p}-{q}"] = {"weight": round(overlap(W[p], W[q]), 4), "weight_null95": round(float(np.percentile(null_w, 95)), 4),
                          "pattern": round(overlap(Pat[p], Pat[q]), 4), "pattern_null95": round(float(np.percentile(null_p, 95)), 4)}
    res["overlap"] = ov
    if half is not None:   # 같은 라벨 probe 를 train 반 A · 반 B 에서 따로 → 부분공간 재현도. 교차 겹침이 이보다 훨씬 작아야 "안 겹친다" 를 말할 수 있다
        hb = half[ci]; rel = {}
        for t, Y in tg.items():
            sub = []
            for hv in (0, 1):
                m = tr & (hb == hv); ym = Y[m].mean(0)
                w, _ = ridge_fit(Xs[m], Y[m] - ym)
                sub.append((orth(w, rank[t]), orth(Sig @ w, rank[t])))
            rel[t] = {"weight": round(overlap(sub[0][0], sub[1][0]), 4), "pattern": round(overlap(sub[0][1], sub[1][1]), 4)}
        res["split_half_reliability"] = rel
    # 기능 검사: 속도 구간
    spd = np.linalg.norm(lab[ci, bi, 3:5], axis=1)
    med = np.median(spd[tr])
    for t in ("shape", "color"):
        Y = tg[t]; out = {}
        for fit_name, fit_m in (("slow", spd <= med), ("all", np.ones_like(spd, bool))):
            m = tr & fit_m
            ym = Y[m].mean(0); w, _ = ridge_fit(Xs[m], Y[m] - ym)
            for ev_name, ev_m in (("slow", spd <= med), ("fast", spd > med)):
                e = ~tr & ev_m
                out[f"fit_{fit_name}->{ev_name}"] = round(100 * float(((Xs[e] @ w + ym).argmax(1) == Y[e].argmax(1)).mean()), 1)
        res[f"{t}_by_speed"] = out
    res["speed_median_px"] = round(float(med), 2)
    return res, W, Pat


def run_probes():
    Z = np.load(f"{CACHE}/feats.npz", allow_pickle=True)
    lab, shape, color = Z["lab"], Z["shape"], Z["color"]
    rng = np.random.default_rng(0)
    n = lab.shape[0]; perm = rng.permutation(n); clip_tr = np.zeros(n, bool); clip_tr[perm[: n // 2]] = True
    keys = ["h"] + [f"{p}_{m}" for m in MODELS for p in ("tf", "roll") if p + "_" + m in Z.files]
    half = rng.integers(0, 2, n)
    out, PAT = {}, {}
    for k in keys:
        r, W, P = probe(Z[k], lab, shape, color, clip_tr, k, rng, half=half)
        out[k] = r; PAT[k] = P
        log(k, {t: r[t] for t in ("shape", "color", "pos", "vel")}, "n", r["n_test"])
        r2, _, _ = probe(Z[k], lab, shape, color, clip_tr, k, rng, scen=Z["scen"], half=half)
        out[k + "__scen_removed"] = r2
        log(k, "시나리오 평균 뺌", {t: r2[t] for t in ("shape", "color", "pos", "vel")})
    # 표현 사이: 같은 라벨의 패턴 부분공간이 정답 h 와 얼마나 같은 방향인가
    out["pattern_vs_h"] = {k: {t: round(overlap(PAT[k][t], PAT["h"][t]), 3) for t in PAT[k]} for k in keys if k != "h"}
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    log("saved", OUT)


if __name__ == "__main__":
    if not a.probe_only:
        extract()
    run_probes()
