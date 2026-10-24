#!/usr/bin/env python3
"""[물리] 고친 꼴 (관측 가능한 구멍 건너 변위) 의 확인 판 — 안 쓴 창 (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/interp_confirm.py [--limit 0]

사전 등록: 진행판 [물리] 11:23 UTC (KST 20:23).
흐름: SSv2 수동 · 행위자 창 120–299 · HT100M 창 120–299 (앞 판 경로 뺌) · K710 = enc_noise k710 160 · 3D = 시드 1 (앞 판 120 뺌). 0.5 · 1/4 초, 카메라 정지 걸음.
  M = 구멍 장으로 들어오는 움직임, P1 = 그 앞, N1 = 구멍 장에서 나가는 움직임, D = M + N1 (구멍 앞 장 → 뒤 장 변위, 관측 가능).
  구멍 건너 ridge [P1, D] → M · D/2 · 과거만 ridge [P1, P2] · 귀무 (N1 을 같은 판 안에서 무작위로 섞은 D 로 같은 ridge) · 이론 귀무 0.5.
토큰: enc_noise k710 · ssv2 · ht100m (각 160, ssv2_classes 와 겹치는 영상 뺌). t = 4 · 5 · 6, 정답 h_t, L1 최적.
  외삽 = m 문맥 4 블록 평균 + W splat(h_(t−1), U[t−1]) · 내삽 = m 앞뒤 평균 + W 구멍 건너 RAFT F 로 앞뒤 내용을 p + F/2 로. 위약 90°, 복사 = 정답 h_(t−1).
  칸: 움직이는 |U[t]| ≥ 0.5 / 정지 |U[t]| < 0.25."""
import argparse, collections, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--npts", type=int, default=10_000_000)
ap.add_argument("--out", default="/workspace/world_model/vjepa2/auto_research/results/interp_confirm.json")
a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--parts", "", "--limit", "1"]
import cam_state as CS                                                          # noqa: E402
sys.argv = _argv
torch.set_num_threads(8)
dev = torch.device("cuda")
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights      # noqa: E402
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
G, R, S = 16, 256, 256
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
R90 = np.array([[0.0, -1.0], [1.0, 0.0]])
SPAN = 3.5
CA = "/workspace/auto_research_cache"
EN = f"{CA}/enc_noise"
LIM = a.limit or None
res = {"definition": "사전 등록 [물리] 11:23 UTC (KST 20:23)"}


def prep(fr):
    Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
    x = torch.from_numpy(np.ascontiguousarray(fr[:, y0:y0 + side, x0:x0 + side])).to(dev).permute(0, 3, 1, 2).float()
    if side != R:
        x = F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False).clamp(0, 255).round()
    return x / 127.5 - 1


@torch.no_grad()
def raft_cells(i0, i1):
    f = torch.cat([raft(i0[k:k + 16], i1[k:k + 16], num_flow_updates=12)[-1] for k in range(0, i0.shape[0], 16)])
    return (F.avg_pool2d(f, 16) / 16).permute(0, 2, 3, 1).reshape(f.shape[0], S, 2).cpu().numpy().astype(np.float64)


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), ok


def r2(Y, P):
    return float(1 - ((Y - P) ** 2).sum() / max(((Y - Y.mean(0)) ** 2).sum(), 1e-12))


def ridge_eval(X, Y, win):
    te = win % 2 == 1; tr = ~te
    if tr.sum() < 50 or te.sum() < 50:
        return float("nan")
    Xa = np.c_[X, np.ones(len(X))]; I_ = np.diag([1] * X.shape[1] + [0]); E_ = 1e-9 * np.eye(X.shape[1] + 1)
    f5 = (win[tr] // 2) % 5; best, bl = None, 1.0
    for lam in (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000):
        ss = 0.0
        for k in range(5):
            m = f5 != k
            if m.sum() < 20 or (~m).sum() < 20:
                continue
            W = np.linalg.solve(Xa[tr][m].T @ Xa[tr][m] + lam * m.sum() / 1000 * I_ + E_, Xa[tr][m].T @ Y[tr][m]); ss += ((Y[tr][~m] - Xa[tr][~m] @ W) ** 2).sum()
        if best is None or ss < best:
            best, bl = ss, lam
    W = np.linalg.solve(Xa[tr].T @ Xa[tr] + bl * tr.sum() / 1000 * I_ + E_, Xa[tr].T @ Y[tr])
    return r2(Y[te], Xa[te] @ W)


# ------------------------------------------------------------------ 흐름
used_ht = set(m["path"] for m in json.load(open(f"{CA}/natural_train/ht100m_1pct_meta.json"))[:120])


def flow_windows():
    for g in ("passive", "actor"):
        for m in json.load(open(f"{CA}/ssv2_classes/{g}_meta.json"))[120:300][:LIM]:
            yield f"ssv2_{g}", m["path"], m["start"]
    for m in [m for m in json.load(open(f"{CA}/natural_train/ht100m_1pct_meta.json"))[120:300] if m["path"] not in used_ht][:LIM]:
        yield "ht100m", m["path"], m["start"]
    km = json.load(open(f"{EN}/k710_target_encoder_s0_meta.json")) + json.load(open(f"{EN}/k710_target_encoder_s1_meta.json"))
    for m in km[:LIM]:
        yield "k710_train", m["path"], m["start"]
    ms = sum([json.load(open(f"{CA}/phys3d/meta_val_s{k}.json")) for k in range(4)], [])
    ms0 = list(ms); random.Random(0).shuffle(ms0); used = set(x["path"] for x in ms0[:120])
    ms1 = list(ms); random.Random(1).shuffle(ms1)
    for m in [x for x in ms1 if x["path"] not in used][:120][:LIM]:
        yield "phys3d", os.path.join("/workspace/dataset/training_data/phys3d_single_v1", m["path"]), int(m.get("r0", 0))


D = {}; DTR = collections.defaultdict(list); widx = collections.Counter(); t0 = time.time()
for src, path, start in flow_windows():
    wi = widx[src]; widx[src] += 1
    try:
        vr = decord.VideoReader(path, num_threads=2)
        fps = vr.get_avg_fps(); n = int(round(SPAN * fps)) + 1
        if start + n > len(vr):
            start = max(0, len(vr) - n)
        x_all = prep(vr.get_batch(list(range(start, min(len(vr), start + n)))).asnumpy())
    except Exception as e:
        print("skip", path, e, flush=True); continue
    for dt in (0.5, 0.25):
        step = max(1, int(round(dt * fps))); dtr = step / fps
        x = x_all[::step]
        if x.shape[0] < 5:
            continue
        U = raft_cells(x[1:], x[:-1]); DTR[(src, dt)].append(dtr)
        mag = np.array([CS.cam_mag(CS.fit_cam(U[i])[0]) for i in range(U.shape[0])])
        d = D.setdefault((src, dt), dict(M=[], P1=[], P2=[], N1=[], win=[]))
        for t in range(2, U.shape[0] - 1):
            if not (mag[t - 2:t + 2] < dtr).all():
                continue
            s1, o1 = cell_of(P0 + U[t + 1]); s2, o2 = cell_of(P0[s1] + U[t][s1]); s3, o3 = cell_of(P0[s2] + U[t - 1][s2])
            Mv = U[t][s1]
            sel = np.where((np.linalg.norm(Mv, axis=1) >= dtr) & o1 & o2 & o3)[0]
            if sel.size == 0:
                continue
            if sel.size > 64:
                sel = np.random.RandomState(wi * 100 + t).choice(sel, 64, replace=False)
            d["M"].append(Mv[sel]); d["P1"].append(U[t - 1][s2[sel]]); d["P2"].append(U[t - 2][s3[sel]]); d["N1"].append(U[t + 1][sel]); d["win"].append(np.full(sel.size, wi))
    if sum(widx.values()) % 50 == 0:
        print(f"흐름 {src} {wi} · {time.time() - t0:.0f}s", flush=True)

fr = {}
groups = {"ssv2_passive": ["ssv2_passive"], "ssv2_actor": ["ssv2_actor"], "ht100m": ["ht100m"], "k710_train": ["k710_train"],
          "natural": ["ssv2_passive", "ssv2_actor", "ht100m", "k710_train"], "phys3d": ["phys3d"]}
rng = np.random.RandomState(0)
for dt in (0.5, 0.25):
    for gname, srcs in groups.items():
        ss = [s for s in srcs if (s, dt) in D and D[(s, dt)]["M"]]
        if not ss:
            continue
        cat = {k: [] for k in ("M", "P1", "P2", "N1", "win")}
        for j, s_ in enumerate(ss):
            for k in ("M", "P1", "P2", "N1"):
                cat[k].append(np.concatenate(D[(s_, dt)][k]))
            cat["win"].append(np.concatenate(D[(s_, dt)]["win"]) + j * 100000)
        M, P1, P2, N1, win = (np.concatenate(cat[k]) for k in ("M", "P1", "P2", "N1", "win"))
        te = win % 2 == 1
        if te.sum() < 100:
            continue
        Dv = M + N1
        perm = rng.permutation(len(N1)); Dn = M + N1[perm]
        e = dict(n=int(len(M)), n_test=int(te.sum()), dt_real=float(np.mean(sum([DTR[(s, dt)] for s in ss], []))),
                 across_ridge=ridge_eval(np.c_[P1, Dv], M, win), across_half=r2(M[te], Dv[te] / 2), past_ridge=ridge_eval(np.c_[P1, P2], M, win),
                 null_shuffled=ridge_eval(np.c_[P1, Dn], M, win), null_theory=0.5,
                 corr_M_N1=float((np.c_[M - M.mean(0)] * np.c_[N1 - N1.mean(0)]).sum() / np.sqrt(((M - M.mean(0)) ** 2).sum() * ((N1 - N1.mean(0)) ** 2).sum())))
        fr[f"{gname}|{dt}"] = e
        print(f"흐름 {gname:13s} Δt {dt} (실제 {e['dt_real']:.3f}) n {e['n_test']:6d} | 구멍 건너 ridge {e['across_ridge']:+.3f} · D/2 {e['across_half']:+.3f} · 과거만 {e['past_ridge']:+.3f} "
              f"| 귀무 (섞음) {e['null_shuffled']:+.3f} · 이론 0.5 · M–N1 상관 {e['corr_M_N1']:+.3f}", flush=True)
res["flow"] = fr
json.dump(res, open(a.out, "w"), indent=1)


# ------------------------------------------------------------------ 토큰
def wmedian(v, w):
    o = torch.argsort(v); cw = torch.cumsum(w[o], 0)
    return v[o][torch.searchsorted(cw, cw[-1] / 2)]


def prof(y, m, W, b):
    r = y - b * W; ok = m.abs() > 1e-6
    aa = wmedian(r[ok] / m[ok], m[ok].abs())
    return float((r - aa * m).abs().mean())


def golden(y, m, W, lo=-0.2, hi=1.2, it=26):
    g = (5 ** 0.5 - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo); f1, f2 = prof(y, m, W, x1), prof(y, m, W, x2)
    for _ in range(it):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1; x1 = hi - g * (hi - lo); f1 = prof(y, m, W, x1)
        else:
            lo, x1, f1 = x1, x2, f2; x2 = lo + g * (hi - lo); f2 = prof(y, m, W, x2)
    b = (lo + hi) / 2
    return b, prof(y, m, W, b)


def interp_W(h0, h1, Fl):
    q, oq = cell_of(P0 + Fl / 2); r_, orr = cell_of(P0 + Fl)
    ok = oq & orr
    acc = np.zeros_like(h0); n = np.zeros(S)
    np.add.at(acc, q[ok], h1[ok]); np.add.at(acc, q[ok], h0[r_[ok]]); np.add.at(n, q[ok], 2)
    out = (h0 + h1) / 2; m = n > 0
    out[m] = acc[m] / n[m][:, None]
    return out


ssv2_used = set()
for g in ("passive", "actor"):
    ssv2_used |= set(x["path"] for x in json.load(open(f"{CA}/ssv2_classes/{g}_meta.json")))
DK = {"mov": collections.defaultdict(list), "sta": collections.defaultdict(list)}
SRC = {}
t0 = time.time()
for src in ("k710", "ssv2", "ht100m"):
    Uw = np.load(os.path.join(REPO, f"auto_research/results/natural_transport_ctx5/rows_{src}.npz"))["Uwin"].astype(np.float64).reshape(-1, 8, S, 2)
    w0 = 0
    for sh in (0, 1):
        T_ = np.load(f"{EN}/{src}_target_encoder_s{sh}_tok.npy", mmap_mode="r"); PX = np.load(f"{EN}/{src}_target_encoder_s{sh}_pix.npy", mmap_mode="r")
        meta = json.load(open(f"{EN}/{src}_target_encoder_s{sh}_meta.json"))
        for w in range(T_.shape[0] if not LIM else min(LIM, T_.shape[0])):
            gw = w0 + w
            if src == "ssv2" and meta[w]["path"] in ssv2_used:
                continue
            h = np.asarray(T_[w, 0], np.float32); U = Uw[gw]
            xs = torch.from_numpy(np.asarray(PX[w])[[7, 9, 11, 13, 15]]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            Fls = raft_cells(xs[2:5], xs[0:3])
            for i, t in enumerate((4, 5, 6)):
                mg = np.linalg.norm(U[t], axis=1)
                WE, WEr = CS.splat(h[t - 1], U[t - 1]), CS.splat(h[t - 1], U[t - 1] @ R90.T)
                WI, WIr = interp_W(h[t - 1], h[t + 1], Fls[i]), interp_W(h[t - 1], h[t + 1], Fls[i] @ R90.T)
                mE, mI = h[t - 4:t].mean(0), (h[t - 1] + h[t + 1]) / 2
                for grp, sel in (("mov", mg >= 0.5), ("sta", mg < 0.25)):
                    if sel.sum() < 2:
                        continue
                    d = DK[grp]
                    for key, arr in (("y", h[t]), ("cp", h[t - 1]), ("mE", mE), ("mI", mI), ("E", WE), ("E_r", WEr), ("I", WI), ("I_r", WIr)):
                        d[key].append(arr[sel])
                    d["src"].append(np.full(int(sel.sum()) * h.shape[-1], {"k710": 0, "ssv2": 1, "ht100m": 2}[src], np.int8))
            if w % 40 == 0:
                print(f"토큰 {src} s{sh} {w} · {time.time() - t0:.0f}s", flush=True)
        w0 += T_.shape[0]

tok = {}
for grp in ("mov", "sta"):
    d = DK[grp]
    srcv = np.concatenate(d.pop("src"))
    N = srcv.size
    idx = np.random.RandomState(1).choice(N, min(a.npts, N), replace=False)
    A = {k: torch.from_numpy(np.concatenate(v).ravel()[idx]).to(dev) for k, v in d.items()}
    sv = torch.from_numpy(srcv[idx]).to(dev)
    DK[grp] = None
    for sub, code in (("pooled", None), ("k710", 0), ("ssv2", 1), ("ht100m", 2)):
        B = A if code is None else {k: v[sv == code] for k, v in A.items()}
        if B["y"].numel() < 1000:
            continue
        e = {"n_points": int(B["y"].numel())}
        for tg, tname in (("y", "data"), ("cp", "copy")):
            for wn, mn in (("E", "mE"), ("I", "mI")):
                f0 = prof(B[tg], B[mn], B[wn], 0.0)
                bt, ft = golden(B[tg], B[mn], B[wn]); br, frr = golden(B[tg], B[mn], B[wn + "_r"])
                e[f"{wn}|{tname}"] = dict(b_true=round(bt, 4), b_rot=round(br, 4), net_gain_pct=round(100 * (frr - ft) / f0, 4))
        e["ratio_I_over_E"] = e["I|data"]["net_gain_pct"] / e["E|data"]["net_gain_pct"] if abs(e["E|data"]["net_gain_pct"]) > 1e-9 else float("nan")
        tok[f"{grp}|{sub}"] = e
        print(f"토큰 [{grp} · {sub}] 점 {e['n_points']} | 외삽 {e['E|data']['net_gain_pct']:+.3f} % (복사 {e['E|copy']['net_gain_pct']:+.3f}) · 내삽 {e['I|data']['net_gain_pct']:+.3f} % "
              f"(복사 {e['I|copy']['net_gain_pct']:+.4f}) · 내삽÷외삽 {e['ratio_I_over_E']:.2f}", flush=True)
    del A
    torch.cuda.empty_cache()
res["tokens"] = tok
f1 = fr.get("natural|0.5", {}).get("across_ridge", float("nan"))
tm = tok.get("mov|pooled", {})
pass1 = bool(f1 >= 0.6)
pass2 = bool(tm and tm["ratio_I_over_E"] >= 1.5 and abs(tm["I|copy"]["net_gain_pct"]) <= 0.05)
res["verdict"] = dict(flow_R2=f1, pass1=pass1, token_ratio=tm.get("ratio_I_over_E"), token_copy=tm.get("I|copy", {}).get("net_gain_pct"), pass2=pass2, confirmed=pass1 and pass2)
print("\n판정:", res["verdict"])
json.dump(res, open(a.out, "w"), indent=1)
print("저장", a.out)
