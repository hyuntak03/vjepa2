#!/usr/bin/env python3
"""[P] 얼마나 작은 행동이 얼마나 푸나 — 흐름 주성분 k 개 행동 대리의 위약 뺀 L1 운반 보상 (a) · 벤치마크 위반 흡수 (b) (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/action_size.py [--part a|b|ab]

사전 등록: 진행판 [P] 08:56 UTC (KST 17:56), 리더 08:54 배정.
(a) 세트: SSv2 수동 · 행위자 (ssv2_classes) · K710 보류. 창 반 / 반 (seed 0). 맞춤 쪽 U[t+1] (512 차원, t = 4 · 5 · 6) 으로 주성분.
    평가 쪽: U_k = 평균 + 상위 k 복원, W_k = warp(h_t, U_k) (리더 warp), 위약 = U_k 90° 돌림. 문맥만 = splat(h_t, U[t]) (리더 transport_lag_placebo).
    움직이는 칸 |U[t+1]| ≥ 0.5. L1 최적 = lad_profile 황금분할 (GPU), 평가 쪽 점 최대 1 천만. 정답 data = h_{t+1}, 복사 copy = h_t.
(b) 기저 = 세 세트 맞춤 쪽 합침. a_k = 상위 k 계수.
    IntPhys 2 Main: 블록 흐름 U (intphys2/Main/U.npy, U[:, b] = 블록 b → b−1). 사건 장 = 짝 픽셀 차 (pix 캐시) 가 처음 8 장 중앙값의 3 배를 3 장 연속 넘는 첫 장.
      바닥 = 사건 전 블록 (1 .. 사건 블록 − 1) 의 ‖Δa_k‖ 평균. 흡수 = 사건 뒤 블록 ‖Δa_k‖ 최대 > 3 × 바닥.
    VOBench 끝 가림 (216 block, 짝 둘): 대표 장 (32 장 중 홀수) RAFT 흐름. 사건 블록 = 가능 · 불가능 대표 장이 처음 다른 블록.
      바닥 = 가능 영상에 ±1/255 균등 잡음을 넣은 흐름과의 ‖Δa_k‖ (사건 뒤 블록 평균). 흡수 = 사건 뒤 ‖Δa_k(가능, 불가능)‖ 최대 > 3 × 바닥.
출력: auto_research/results/action_size.json."""
import argparse, collections, csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--part", default="ab")
ap.add_argument("--npts", type=int, default=10_000_000)
ap.add_argument("--out", default="/workspace/world_model/vjepa2/auto_research/results/action_size.json")
a = ap.parse_args()
torch.set_num_threads(8)
dev = torch.device("cuda")
REPO = "/workspace/world_model/vjepa2"
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
R90 = np.array([[0.0, -1.0], [1.0, 0.0]], np.float32)
KS = (2, 8, 32, 128, 512)
SETS = {"ssv2_passive": ("/workspace/auto_research_cache/ssv2_classes/passive_h.npy", "/workspace/auto_research_cache/ssv2_classes/passive_U.npy"),
        "ssv2_actor": ("/workspace/auto_research_cache/ssv2_classes/actor_h.npy", "/workspace/auto_research_cache/ssv2_classes/actor_U.npy"),
        "k710_holdout": ("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_h.npy", "/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy")}


def warp(ht, u):                                                               # 리더 transport_placebo / lad_profile 과 같다
    s = np.floor(P0 + u).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    j = np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


def splat(ht, u0):                                                             # 리더 transport_lag_placebo 와 같다
    d = np.floor(P0 - u0).astype(int); ok = (d >= 0).all(1) & (d < G).all(1)
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
    aa = wmedian(r[ok] / m[ok], m[ok].abs())
    return float((r - aa * m).abs().mean())


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


def split_windows(n):
    rs = np.random.RandomState(0); w = np.arange(n); rs.shuffle(w)
    return set(w[: n // 2].tolist())


res = {"definition": "사전 등록 [P] 08:56 UTC (KST 17:56)"}
BASIS = {}
t0 = time.time()
for name, (hp, up) in SETS.items():
    Hs = np.load(hp, mmap_mode="r"); Us = np.load(up, mmap_mode="r")
    n = Hs.shape[0]; fit = split_windows(n)
    Ufit = np.concatenate([np.asarray(Us[w], np.float32).reshape(8, S * 2)[5:8] for w in sorted(fit)])        # U[t+1], t = 4..6
    BASIS[name] = Ufit
    mu = Ufit.mean(0); _, _, Vt = np.linalg.svd(Ufit - mu, full_matrices=False)                               # (512, 512)
    ev = None
    if "a" not in a.part:
        continue
    D = collections.defaultdict(list)
    for w in range(n):
        if w in fit:
            continue
        h = np.asarray(Hs[w], np.float32); U = np.asarray(Us[w], np.float32).reshape(8, S, 2)
        for t in (4, 5, 6):
            sel = np.linalg.norm(U[t + 1], axis=1) >= 0.5
            if sel.sum() < 2:
                continue
            D["y"].append(h[t + 1][sel]); D["cp"].append(h[t][sel]); D["m"].append(h[t - 4:t + 1].mean(0)[sel])
            D["lag"].append(splat(h[t], U[t])[sel]); D["lag_r"].append(splat(h[t], U[t] @ R90.T)[sel])
            v = U[t + 1].reshape(-1) - mu
            for k in KS:
                uk = (mu + (v @ Vt[:k].T) @ Vt[:k]).reshape(S, 2)
                D[f"k{k}"].append(warp(h[t], uk)[sel]); D[f"k{k}_r"].append(warp(h[t], uk @ R90.T)[sel])
    N = sum(x.size for x in D["y"])
    rs = np.random.RandomState(1); idx = torch.from_numpy(rs.choice(N, min(a.npts, N), replace=False)).to(dev)
    A = {k: torch.from_numpy(np.concatenate(v).ravel()).to(dev)[idx] for k, v in D.items()}
    del D
    e = {"n_points_total": int(N), "n_points": int(idx.numel())}
    for wn in ["lag"] + [f"k{k}" for k in KS]:
        for tg in ("y", "cp"):
            f0 = prof(A[tg], A["m"], A[wn], 0.0)
            bt, ft = golden(A[tg], A["m"], A[wn]); br, fr = golden(A[tg], A["m"], A[wn + "_r"])
            e[f"{wn}|{'data' if tg == 'y' else 'copy'}"] = dict(b_true=round(bt, 4), b_rot=round(br, 4), net_b=round(bt - br, 4), net_gain_pct=round(100 * (fr - ft) / f0, 3))
    gl, gt = e["lag|data"]["net_gain_pct"], e["k512|data"]["net_gain_pct"]
    bl, bt_ = e["lag|data"]["net_b"], e["k512|data"]["net_b"]
    for k in KS:
        e[f"k{k}|gap_closed_gain"] = round((e[f"k{k}|data"]["net_gain_pct"] - gl) / max(gt - gl, 1e-9), 3)
        e[f"k{k}|gap_closed_b"] = round((e[f"k{k}|data"]["net_b"] - bl) / max(bt_ - bl, 1e-9), 3)
    res[name] = e
    print(f"\n== {name}: 점 {e['n_points']} / {N} ({time.time() - t0:.0f}s)", flush=True)
    for wn in ["lag"] + [f"k{k}" for k in KS]:
        d, c = e[f"{wn}|data"], e[f"{wn}|copy"]
        extra = f" · 간극 메움 (L1 이득) {e[f'{wn}|gap_closed_gain']:.2f} · (b) {e[f'{wn}|gap_closed_b']:.2f}" if wn != "lag" else ""
        print(f"   {wn:5s} 데이터 위약 뺀 b {d['net_b']:+.4f} · L1 이득 {d['net_gain_pct']:+.3f} % | 복사 {c['net_b']:+.4f} · {c['net_gain_pct']:+.3f} %{extra}", flush=True)
    del A
    torch.cuda.empty_cache()

json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
if "b" not in a.part:
    sys.exit(0)

# ------------------------------------------------------------------ (b) 흡수
Uall = np.concatenate(list(BASIS.values()))
MU = Uall.mean(0); _, _, VT = np.linalg.svd(Uall - MU, full_matrices=False)


def coeffs(U):                                                                 # U (..., 256, 2) → (..., 512) 계수 (전부)
    v = U.reshape(*U.shape[:-2], S * 2) - MU
    return v @ VT.T


def absorb(post_d, floor):                                                     # post_d: (n_pairs, n_post, 512) 계수 차 · floor (n_pairs,) per k 별로 따로
    out = {}
    for k in KS:
        dp = np.array([np.linalg.norm(p[:, :k], axis=1).max() for p in post_d])
        fl = np.array([f[k] for f in floor])
        out[f"k{k}"] = dict(frac=round(float(np.mean(dp > 3 * fl)), 3), n=int(len(dp)), median_ratio=round(float(np.median(dp / np.maximum(fl, 1e-9))), 2))
    return out


# IntPhys 2
C2 = "/workspace/auto_research_cache/intphys2/Main/"
meta = json.load(open(C2 + "meta.json")); U2 = np.load(C2 + "U.npy", mmap_mode="r"); PX = np.load(C2 + "pix.npy", mmap_mode="r")
pairs = collections.defaultdict(dict)
for m in meta:
    pairs[(m["scene_index"], str(m["pair_id"]))][bool(m["is_impossible"])] = m["row"]
post_d, floors, ev_blocks, skipped = [], [], [], 0
for key, d in pairs.items():
    if True not in d or False not in d:
        continue
    rp, ri = d[False], d[True]
    xp = torch.from_numpy(np.asarray(PX[rp])).to(dev).float(); xi = torch.from_numpy(np.asarray(PX[ri])).to(dev).float()
    dif = (xp - xi).abs().mean((1, 2, 3)).cpu().numpy()                          # 64 장
    base = np.median(dif[:8]) + 1e-6
    above = dif > 3 * base
    ef = next((f for f in range(8, 62) if above[f] and above[f + 1] and above[f + 2]), None)
    if ef is None:
        skipped += 1; continue
    eb = ef // 2
    if eb < 3:
        skipped += 1; continue
    cp_, ci_ = coeffs(np.asarray(U2[rp], np.float32).reshape(32, S, 2)), coeffs(np.asarray(U2[ri], np.float32).reshape(32, S, 2))
    dd = cp_ - ci_
    pre = dd[1:eb]; post = dd[eb:]
    floors.append({k: float(np.linalg.norm(pre[:, :k], axis=1).mean()) for k in KS}); post_d.append(post); ev_blocks.append(eb)
res["intphys2_absorption"] = dict(absorb(post_d, floors), n_pairs=len(post_d), skipped=skipped, event_block_median=float(np.median(ev_blocks)))
print("\n== IntPhys 2 흡수", json.dumps(res["intphys2_absorption"], ensure_ascii=False), flush=True)

# VOBench 끝 가림
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
VB = "/workspace/dataset/benchmarks/VOBench_v1"
IDX = list(range(0, 96, 3))[:32]
REP = IDX[1::2]                                                                # 블록 대표 장 16 개
rows = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))):
    if r["occlusion"] == "late":
        rows[r["block_id"]][r["variant"]] = r


def load_rep(fn, noise=False, g=None):
    fr = np.stack([np.asarray(Image.open(f"{VB}/{fn}/{i:06d}.png").convert("RGB")) for i in REP])
    x = torch.from_numpy(fr).to(dev).permute(0, 3, 1, 2).float()
    if noise:
        x = (x + torch.randint(-1, 2, x.shape, device=dev, generator=g).float()).clamp(0, 255)
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    return x / 127.5 - 1, fr


@torch.no_grad()
def bflow(x):                                                                  # (16,3,256,256) → U (16, 256, 2), U[0] = 0
    f = raft(x[1:], x[:-1], num_flow_updates=12)[-1]
    u = (F.avg_pool2d(f, 16) / 16).permute(0, 2, 3, 1).reshape(-1, S, 2).cpu().numpy()
    return np.concatenate([np.zeros((1, S, 2), np.float32), u])


gen = torch.Generator(device=dev); gen.manual_seed(0)
post_d, floors, by_viol = [], [], collections.defaultdict(list)
for bi, (b, arms) in enumerate(sorted(rows.items())):
    for pos, imp in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
        xp, frp = load_rep(arms[pos]["file_name"]); xi, fri = load_rep(arms[imp]["file_name"])
        diff = np.abs(frp.astype(np.int16) - fri.astype(np.int16)).reshape(16, -1).max(1)
        eb = int(np.argmax(diff > 0)) if (diff > 0).any() else None
        if eb is None or eb < 2:
            continue
        xn, _ = load_rep(arms[pos]["file_name"], noise=True, g=gen)
        cp_, ci_, cn_ = coeffs(bflow(xp)), coeffs(bflow(xi)), coeffs(bflow(xn))
        post = (cp_ - ci_)[eb:]; nf = (cp_ - cn_)[eb:]
        fl = {k: float(np.linalg.norm(nf[:, :k], axis=1).mean()) for k in KS}
        post_d.append(post); floors.append(fl); by_viol[(arms[pos]["motion"] == "flat_static", arms[pos]["violation"])].append(len(post_d) - 1)
    if bi % 40 == 0:
        print(f"VOBench {bi}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
res["vobench_absorption"] = dict(absorb(post_d, floors), n_pairs=len(post_d))
res["vobench_absorption_by"] = {f"{'static' if st else 'moving'}|{vi}": absorb([post_d[i] for i in ix], [floors[i] for i in ix]) for (st, vi), ix in by_viol.items()}
print("\n== VOBench 흡수", json.dumps(res["vobench_absorption"], ensure_ascii=False), flush=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print("저장", a.out, f"({time.time() - t0:.0f}s)")
