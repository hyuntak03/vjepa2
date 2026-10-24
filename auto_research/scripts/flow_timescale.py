#!/usr/bin/env python3
"""[P] 시간 간격 Δt 별 흐름 지속 계수 k · 흐름 공간 조건화 보상 ([물리] flow_persistence.py · flow_cond_reward.py 정의 그대로) (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/flow_timescale.py

사전 등록: 진행판 [P] KST 05:28.
창: K710 보류 150 · HT100M 120 (같은 영상 · 시작 장, 3.5 초 구간) · 3D 합성 검증 clip ([물리] k 0.95 와 같은 phys3d_single_v1 val, 무작위 150, 12 fps, 시작 r0, 3.5 초) · VOBench 가림 없음 · 움직임 pos_a (30 fps, 0–3.2 초).
전처리: 가운데 정사각 → 256 bilinear (기존 캐시와 같음). Δt 마다 간격 = max(1, round(Δt·fps)) 장, U[i] = RAFT(장 i → 장 i−1) 의 16 px 칸 평균 / 16 (칸 단위).
k: t = 1 .. n−2, Y = U[t+1](c), Xa = U[t](s), s = c + U[t+1](c) 의 칸 (내용 따라가기), Xb = U[t](c) (같은 칸). 움직이는 칸 |Y| ≥ 문턱.
  문턱: 속도 기준 = 0.5 × (Δt_실제 / 0.5) 칸 (주) · 고정 0.5 칸 (부). 카메라 움직임 = 칸 흐름 크기 중앙값 > 같은 문턱.
조건화 보상: x = U[t](s), x′ = U[t−1](s′), s′ = s + x 의 칸, r = |x − x′| / (|x| + |x′|), cam = 중앙값 |U[t]| > 문턱. r 5 분위 (자연 통합 경계, Δt 마다) × cam 2.
출력: auto_research/results/flow_timescale.json · logs/flow_timescale.log."""
import csv, glob, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord

torch.set_num_threads(8)
dev = torch.device("cuda")
REPO = "/workspace/world_model/vjepa2"
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights  # noqa: E402
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
G, R = 16, 256
DTS = (1 / 16, 1 / 8, 1 / 4, 1 / 2)
SPAN = 3.5
cy, cx = np.divmod(np.arange(G * G), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
OUT = os.path.join(REPO, "auto_research/results/flow_timescale" + ("_smoke" if os.environ.get("LIMIT") else "") + ".json")


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), ok


def prep(fr):                                                                   # (T,H,W,3) uint8 → (T,3,256,256) −1..1
    Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
    x = torch.from_numpy(np.ascontiguousarray(fr[:, y0:y0 + side, x0:x0 + side])).to(dev).permute(0, 3, 1, 2).float()
    if side != R:
        x = F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False).clamp(0, 255).round()
    return x / 127.5 - 1


@torch.no_grad()
def cell_flow(x):                                                               # x (n,3,R,R) → U (n−1, 256, 2): 장 i → i−1 (i = 1..n−1)
    out = []
    for k in range(1, x.shape[0], 16):
        a = x[k:k + 16]; b = x[k - 1:k - 1 + a.shape[0]]
        out.append(raft(a, b, num_flow_updates=12)[-1])
    f = torch.cat(out)
    return (F.avg_pool2d(f, 16) / 16).permute(0, 2, 3, 1).reshape(f.shape[0], G * G, 2).cpu().numpy()


LIM = int(os.environ.get("LIMIT", "0")) or None


def windows():
    for m in json.load(open("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_meta.json"))[:150][:LIM]:
        yield "k710_holdout", m["path"], m["start"], None
    for m in json.load(open("/workspace/auto_research_cache/natural_train/ht100m_1pct_meta.json"))[:120][:LIM]:
        yield "ht100m", m["path"], m["start"], None
    ms = sum([json.load(open(f"/workspace/auto_research_cache/phys3d/meta_val_s{k}.json")) for k in range(4)], [])   # [물리] k 0.95 와 같은 3D 합성 검증 clip
    random.Random(0).shuffle(ms)
    for m in ms[:150][:LIM]:
        yield "phys3d", os.path.join("/workspace/dataset/training_data/phys3d_single_v1", m["path"]), int(m.get("r0", 0)), None
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "visible" and r["variant"] == "pos_a" and r["motion"] != "flat_static"]
    for r in rows[:LIM]:
        yield "vobench", os.path.join("/workspace/dataset/benchmarks/VOBench_v1", r["file_name"]), 0, "png"


def load(path, start, kind):
    if kind == "png":
        from PIL import Image
        fr = np.stack([np.asarray(Image.open(f"{path}/{i:06d}.png").convert("RGB")) for i in range(96)])
        return fr, 30.0
    vr = decord.VideoReader(path, num_threads=1)
    fps = vr.get_avg_fps(); n = int(round(SPAN * fps)) + 1
    if start + n > len(vr):
        start = max(0, len(vr) - n)
    idx = list(range(start, min(len(vr), start + n)))
    return vr.get_batch(idx).asnumpy(), fps


ACC = {}                                                                        # (src, Δt) → 목록
DT_REAL = {}
t0 = time.time()
for wi, (src, path, start, kind) in enumerate(windows()):
    try:
        fr, fps = load(path, start, kind)
    except Exception as e:
        print("skip", path, e, flush=True); continue
    x_all = prep(fr)
    for dt in DTS:
        step = max(1, int(round(dt * fps))); dtr = step / fps
        x = x_all[::step]
        if x.shape[0] < 4:
            continue
        U = cell_flow(x)                                                        # U[i-1] = 장 i → i−1
        DT_REAL.setdefault((src, dt), []).append(dtr)
        a = ACC.setdefault((src, dt), {th: dict(Y=[], Xa=[], Xb=[], cam=[], X=[], R=[], C=[], Yc=[]) for th in ("vel", "fix")})
        for th in ("vel", "fix"):
            thr = 0.5 * (dtr / 0.5) if th == "vel" else 0.5
            d = a[th]
            for t in range(1, U.shape[0] - 1):                                  # U 인덱스: Y = U[t+1], X = U[t], 앞 = U[t−1]
                u1, u0, um = U[t + 1], U[t], U[t - 1]
                m1 = np.linalg.norm(u1, axis=1)
                mov = m1 >= thr
                s, ok = cell_of(P0 + u1)
                sel = np.where(mov & ok)[0]
                if sel.size == 0:
                    continue
                d["Y"].append(u1[sel]); d["Xa"].append(u0[s[sel]]); d["Xb"].append(u0[sel]); d["cam"].append(np.full(sel.size, np.median(m1) > thr))
                s2, ok2 = cell_of(P0[s] + u0[s])
                sel2 = np.where(mov & ok & ok2)[0]
                if sel2.size:
                    xx = u0[s[sel2]]; xp = um[s2[sel2]]
                    r = np.linalg.norm(xx - xp, axis=1) / (np.linalg.norm(xx, axis=1) + np.linalg.norm(xp, axis=1) + 1e-6)
                    d["Yc"].append(u1[sel2]); d["X"].append(xx); d["R"].append(r); d["C"].append(np.full(sel2.size, np.median(np.linalg.norm(u0, axis=1)) > thr))
    if (wi + 1) % 30 == 0:
        print(f"{wi + 1} 창 {src} ({time.time() - t0:.0f}s)", flush=True)


def coef(Y, X):
    k2 = (Y * X).sum() / max((X * X).sum(), 1e-9)
    ks = np.linspace(0, 1.5, 151); l1 = [np.abs(Y - k * X).sum() for k in ks]
    r2 = 1 - ((Y - k2 * X) ** 2).sum() / max((Y ** 2).sum(), 1e-9)
    cosd = np.mean((Y * X).sum(1) / (np.linalg.norm(Y, axis=1) * np.linalg.norm(X, axis=1) + 1e-9) > 0.7071)
    return dict(n=int(len(Y)), k_L2=round(float(k2), 3), k_L1=round(float(ks[int(np.argmin(l1))]), 3), R2=round(float(r2), 3),
                relL1_constvel=round(float(np.abs(Y - X).sum() / max(np.abs(Y).sum(), 1e-9)), 3), dir_within45=round(float(cosd), 3),
                mean_disp_cells=round(float(np.linalg.norm(Y, axis=1).mean()), 3))


def cond(Y, X, Rr, C, edges):
    fitk = lambda Y_, X_: (Y_ * X_).sum() / max((X_ * X_).sum(), 1e-9)
    sse = lambda Y_, X_, k: ((Y_ - k * X_) ** 2).sum()
    kg = fitk(Y, X); Sg = sse(Y, X, kg); S0 = sse(Y, X, 0.0); b = np.digitize(Rr, edges); Sb = 0.0; used = np.zeros(len(Y), bool); bins = []
    for cam in (False, True):
        for q in range(5):
            m = (b == q) & (C == cam)
            if m.sum() < 30:
                continue
            kb = fitk(Y[m], X[m]); Sb += sse(Y[m], X[m], kb); used |= m
            bins.append(dict(cam=bool(cam), r_bin=q, n=int(m.sum()), k=round(float(kb), 3)))
    Sb += sse(Y[~used], X[~used], kg)
    return dict(n=int(len(Y)), k_global=round(float(kg), 3), transport_pct=round(float(100 * (S0 - Sg) / Sg), 2), cond_pct=round(float(100 * (Sg - Sb) / Sg), 2), bins=bins)


res = {}
for dt in DTS:
    for th in ("vel", "fix"):
        nat = [ACC[(s, dt)][th] for s in ("k710_holdout", "ht100m") if (s, dt) in ACC]
        Rn = np.concatenate([np.concatenate(d["R"]) for d in nat if d["R"]]) if nat else np.zeros(1)
        edges = np.quantile(Rn, [0.2, 0.4, 0.6, 0.8])
        groups = {"k710_holdout": ["k710_holdout"], "ht100m": ["ht100m"], "natural": ["k710_holdout", "ht100m"], "phys3d": ["phys3d"], "vobench": ["vobench"]}
        for gname, srcs in groups.items():
            ds = [ACC[(s, dt)][th] for s in srcs if (s, dt) in ACC]
            if not ds or not any(d["Y"] for d in ds):
                continue
            cat = lambda key: np.concatenate([np.concatenate(d[key]) for d in ds if d[key]])
            Y, Xa, Xb, cam = cat("Y"), cat("Xa"), cat("Xb"), cat("cam")
            r_ = dict(dt_real=round(float(np.mean(sum([DT_REAL[(s, dt)] for s in srcs if (s, dt) in DT_REAL], []))), 4),
                      follow_content=coef(Y, Xa), same_cell=coef(Y, Xb),
                      follow_cam_static=coef(Y[~cam], Xa[~cam]) if (~cam).sum() > 30 else None, follow_cam_moving=coef(Y[cam], Xa[cam]) if cam.sum() > 30 else None)
            if any(d["X"] for d in ds):
                r_["cond"] = cond(cat("Yc"), cat("X"), cat("R"), cat("C"), edges)
            res[f"{gname}|dt={dt:.4f}|{th}"] = r_
            print(gname, f"Δt {dt:.4f} (실제 {r_['dt_real']})", th, "k", r_["follow_content"]["k_L2"], "R2", r_["follow_content"]["R2"], "n", r_["follow_content"]["n"],
                  "| cam 정지/움직임 k", (r_["follow_cam_static"] or {}).get("k_L2"), (r_["follow_cam_moving"] or {}).get("k_L2"),
                  "| 운반 / 조건화 보상 %", r_.get("cond", {}).get("transport_pct"), r_.get("cond", {}).get("cond_pct"), flush=True)
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("저장", OUT, f"({time.time() - t0:.0f}s)")
