#!/usr/bin/env python3
"""[T] 이어 학습 · 손실 · 규제 팔의 위약 뺀 운반 (학습 없음) — [L] transport_placebo.py 의 도구를 여러 체크포인트 · 세 세트로 (auto_research, 2026-10-22 KST 15:5x).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/transport_placebo_arms.py [--limit N]

회귀 · 옮김 · 위약은 transport_placebo.py 와 같다: 움직이는 칸 (|U[t+1]| ≥ 0.5 칸), t = 4 · 5 · 6, y ≈ a · (블록 t−4..t 같은 칸 평균) + b · W,
  W_true = h_t 를 U[t+1] 로 옮긴 것, W_rot = 90° 돌린 흐름으로 옮긴 것 (위약). 위약 뺀 운반 = b_true − b_rot.
y = 데이터 (h_{t+1}) · 복사 정답 (h_t, 0 이어야 함) · 모델 TF 한 걸음 (블록 0..t → t+1).
세트: SSv2 수동 · 행위자 (ssv2_classes) · K710 보류 (natural_holdout). CI = 창 bootstrap (300). 팔 − 기준 차이는 같은 재표본 창으로 짝 bootstrap.
대가: K710 보류 TF 한 걸음 L1 (블록 0..6 → 1..7 전부, 모든 칸).
모델: 기준 AR (natural_ar_vw_state latest) + /workspace/auto_research_cache/ar_ft/*.pt (L1 · 다중 가설 (가설마다) · CRPS (잡음 0) · 가중 L1 · 조각 · 안정도 · 보조)
      + /workspace/auto_research_cache/cf_ft_smoke/*.pt. 출력 results/transport_placebo_arms.json."""
import argparse, json, os, sys, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--nb", type=int, default=300)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/transport_placebo_arms.json"))
ap.add_argument("--arms", nargs="*", default=None, help="--ftdir 의 체크포인트 이름 목록 (주면 기본 목록 대신)")
ap.add_argument("--ftdir", default="/workspace/auto_research_cache/ar_ft")
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(__file__))
_argv = sys.argv; sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
sys.argv = _argv
torch.set_num_threads(8)
dev = torch.device("cuda")
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1)
R90 = np.array([[0.0, -1.0], [1.0, 0.0]], np.float32)
SETS = {"ssv2_passive": ("/workspace/auto_research_cache/ssv2_classes/passive_h.npy", "/workspace/auto_research_cache/ssv2_classes/passive_U.npy"),
        "ssv2_actor": ("/workspace/auto_research_cache/ssv2_classes/actor_h.npy", "/workspace/auto_research_cache/ssv2_classes/actor_U.npy"),
        "k710_holdout": ("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_h.npy", "/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy")}
FTD = a.ftdir
ARMS = ["ft_l1", "ft_l1_lr1e4", "ft_wta3", "ft_wta3_posctrl", "ft_crpsxi", "ft_wl1", "ft_wl1_lr1e4", "ft_patch",
        "ft_stab_l01_lr1e4", "ft_stab_l05_lr1e4", "ft_stab_l05_lr1e5", "aux_w0", "aux_d005", "aux_d02"]
SMOKE = ["sm_b", "sm_a100", "sm_c100"]


def warp(ht, u):
    s = np.floor(P0 + u).astype(int)
    ok = (s >= 0).all(1) & (s < G).all(1)
    j = np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


def make(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck.get("args") or {}
    W = FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), int(args.get("nz", 32)), 0.0)
    W.pred.load_state_dict(ck["predictor"])
    if ck.get("codes") is not None and args.get("loss") == "wta":
        W.codes = torch.nn.Parameter(ck["codes"])
        return W.to(dev).eval(), [("h%d" % k, k) for k in range(ck["codes"].shape[0])]
    return W.to(dev).eval(), [("", None)]


MODELS = [("base", lambda: (FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), 32, 0.0).to(dev).eval(), [("", None)]))]
if a.arms is not None:
    ARMS, SMOKE = a.arms, []
MODELS += [(n, (lambda n_=n: make(os.path.join(FTD, f"{n_}.pt")))) for n in ARMS]
MODELS += [(n, (lambda n_=n: make(os.path.join("/workspace/auto_research_cache/cf_ft_smoke", f"{n_}.pt")))) for n in SMOKE]

# 1) 데이터 쪽 (모델 무관) 을 한 번: 창마다 X 를 다시 만들 수 있게 h 는 mmap, 회귀 누적만 저장
DATA = {}
for sn, (hp, up) in SETS.items():
    H = np.load(hp, mmap_mode="r"); U = np.load(up).astype(np.float32).reshape(-1, 8, S, 2)
    n = H.shape[0] if not a.limit else min(a.limit, H.shape[0])
    recs = []
    for w in range(n):
        h = np.asarray(H[w], np.float32)
        for t in (4, 5, 6):
            sel = np.linalg.norm(U[w, t + 1], axis=1) >= 0.5
            if sel.sum() < 2:
                continue
            m = h[t - 4:t + 1].mean(0)
            r = dict(win=w, t=t, n=int(sel.sum()))
            for kind in ("true", "rot"):
                Wk = warp(h[t], U[w, t + 1] if kind == "true" else U[w, t + 1] @ R90.T)
                X = np.stack([m[sel].ravel(), Wk[sel].ravel()], 1).astype(np.float64)
                r[f"XtX_{kind}"] = X.T @ X
                r[f"Xty_{kind}_data"] = X.T @ h[t + 1][sel].ravel().astype(np.float64)
                r[f"Xty_{kind}_copy"] = X.T @ h[t][sel].ravel().astype(np.float64)
            recs.append(r)
    DATA[sn] = dict(H=H, U=U, n=n, recs=recs)
    print(f"[data] {sn}: 창 {len(set(r['win'] for r in recs))} · 걸음 {len(recs)} · 칸 {sum(r['n'] for r in recs)}", flush=True)


def bfit(recs, kind, nm):
    A = sum(r[f"XtX_{kind}"] for r in recs); bv = sum(r[f"Xty_{kind}_{nm}"] for r in recs)
    return np.linalg.solve(A, bv)[1]


rng = np.random.RandomState(0)
BOOT = {}
for sn, D in DATA.items():
    wins = np.array([r["win"] for r in D["recs"]]); uw = np.unique(wins); ix = {u: np.where(wins == u)[0] for u in uw}
    BOOT[sn] = [np.concatenate([ix[u] for u in rng.choice(uw, len(uw))]) for _ in range(a.nb)]

RES = {sn: {} for sn in SETS}
for sn, D in DATA.items():
    for nm in ("data", "copy"):
        net = bfit(D["recs"], "true", nm) - bfit(D["recs"], "rot", nm)
        bs = [bfit([D["recs"][i] for i in ii], "true", nm) - bfit([D["recs"][i] for i in ii], "rot", nm) for ii in BOOT[sn]]
        RES[sn][nm] = dict(b_true=float(bfit(D["recs"], "true", nm)), b_rot=float(bfit(D["recs"], "rot", nm)), net=float(net), ci=np.percentile(bs, [2.5, 97.5]).tolist())
    print(f"[{sn}] 데이터 위약 뺀 {RES[sn]['data']['net']:.4f} {np.round(RES[sn]['data']['ci'], 4).tolist()} · 복사 {RES[sn]['copy']['net']:+.4f} {np.round(RES[sn]['copy']['ci'], 4).tolist()}", flush=True)

L1 = {}
t0 = time.time()
for name, mk in MODELS:
    W, heads = mk()
    for hn, code in heads:
        key = name + (f"_{hn}" if hn else "")
        for sn, D in DATA.items():
            need = {r["win"] for r in D["recs"]}
            P = {}
            l1s, l1n = 0.0, 0
            wl = range(D["n"]) if sn == "k710_holdout" else sorted(need)
            for w in wl:
                h = torch.from_numpy(np.asarray(D["H"][w], np.float32)).to(dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    p = W.tf(h.reshape(1, -1, 1280), code=code).float()[0].view(8, S, -1)[:7]
                if sn == "k710_holdout":
                    l1s += float((p - h[1:]).abs().mean()); l1n += 1
                if w in need:
                    P[w] = p[4:7].cpu().numpy()                         # 위치 4..6 = 블록 5..7 예측
            for r in D["recs"]:
                w, t = r["win"], r["t"]
                hh = np.asarray(D["H"][w], np.float32); u = D["U"][w, t + 1]
                sel = np.linalg.norm(u, axis=1) >= 0.5; m = hh[t - 4:t + 1].mean(0)
                for kind in ("true", "rot"):
                    Wk = warp(hh[t], u if kind == "true" else u @ R90.T)
                    X = np.stack([m[sel].ravel(), Wk[sel].ravel()], 1).astype(np.float64)
                    r[f"Xty_{kind}_m"] = X.T @ P[w][t - 4][sel].ravel().astype(np.float64)
            net = bfit(D["recs"], "true", "m") - bfit(D["recs"], "rot", "m")
            bs, bd = [], []
            for k, ii in enumerate(BOOT[sn]):
                rr = [D["recs"][i] for i in ii]
                v = bfit(rr, "true", "m") - bfit(rr, "rot", "m"); bs.append(v)
                if key != "base":
                    bd.append(v - RES[sn]["base"]["boot"][k])
            e = dict(b_true=float(bfit(D["recs"], "true", "m")), b_rot=float(bfit(D["recs"], "rot", "m")), net=float(net), ci=np.percentile(bs, [2.5, 97.5]).tolist(),
                     ratio_to_data=float(net / RES[sn]["data"]["net"]), boot=bs)
            if key != "base":
                e["diff_vs_base"] = float(net - RES[sn]["base"]["net"]); e["diff_ci"] = np.percentile(bd, [2.5, 97.5]).tolist()
            RES[sn][key] = e
            if sn == "k710_holdout":
                L1[key] = l1s / max(l1n, 1)
        line = " · ".join(f"{sn} {RES[sn][key]['net']:.4f}" + (f" (차 {RES[sn][key]['diff_vs_base']:+.4f} [{RES[sn][key]['diff_ci'][0]:+.4f}, {RES[sn][key]['diff_ci'][1]:+.4f}])" if key != "base" else "") for sn in SETS)
        print(f"[{key}] 위약 뺀 운반: {line} | K710 보류 L1 {L1[key]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    del W; torch.cuda.empty_cache()

for sn in SETS:
    for k, v in RES[sn].items():
        v.pop("boot", None)
json.dump({"sets": RES, "k710_holdout_tf_l1": L1}, open(a.out if not a.limit else a.out.replace(".json", "_smoke.json"), "w"), indent=1)
print("저장", flush=True)
