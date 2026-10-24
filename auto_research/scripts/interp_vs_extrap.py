#!/usr/bin/env python3
"""[물리] 내삽 대 외삽 예측 가능성 — 흐름 R² + 토큰 위약 뺀 L1 이득 (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/interp_vs_extrap.py [--part ft] [--limit 0]

사전 등록: 진행판 [물리] 11:11 UTC (KST 20:11), 리더 배정 KST 20:09 (B).
흐름 판 (motion_map 처리): 출처 K710 보류 150 · SSv2 수동/행위자 120+120 · HT100M 120 · 3D 합성 120, 3.5 초, 간격 0.5 · 1/4 초.
  U[i] = 장 i+1 → 장 i 칸 흐름. 다음 장 칸 c → s1 = c + U[t+1](c) → s2 → s3. M = U[t](s1) (가린 걸음), P1 = U[t−1](s2), P2 = U[t−2](s3), N1 = U[t+1](c).
  칸 |M| ≥ 실제 Δt, 카메라 정지 걸음. (i) 과거만 ridge [P1,P2,1] · (ii) 앞뒤 ridge [P1,N1,1] · 등속 P1 · 앞뒤 평균. 짝수 창 맞춤 · 홀수 창 R².
토큰 판 (캐시 ssv2_classes 두 부류 · natural_holdout K710, 블록 0.5 초, t = 4 · 5 · 6, 움직이는 칸 |U[t]| ≥ 0.5): y = h_t, y ≈ a·m + b·W (L1 최적).
  외삽 E: m = h_(t−4..t−1) 평균, W = splat(h_(t−1), U[t−1]). 내삽 I: m = (h_(t−1)+h_(t+1))/2, W = 블록 t+1 → t−1 RAFT 흐름 F 로 앞뒤 내용을 p + F/2 로.
  위약 = 흐름 90° 돌림. 위약 뺀 L1 이득 % = 100 (L1_위약 − L1_참) / L1_(b=0). 복사 검사 = 정답 h_(t−1)."""
import argparse, collections, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--part", default="ft")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--npts", type=int, default=10_000_000)
ap.add_argument("--out", default="/workspace/world_model/vjepa2/auto_research/results/interp_vs_extrap.json")
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
res = {"definition": "사전 등록 [물리] 11:11 UTC (KST 20:11)"}
LIM = a.limit or None


def prep(fr):
    Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
    x = torch.from_numpy(np.ascontiguousarray(fr[:, y0:y0 + side, x0:x0 + side])).to(dev).permute(0, 3, 1, 2).float()
    if side != R:
        x = F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False).clamp(0, 255).round()
    return x / 127.5 - 1


@torch.no_grad()
def raft_cells(i0, i1):                                                         # (n,3,R,R) i0 → i1 칸 흐름 (n, 256, 2)
    out = [raft(i0[k:k + 16], i1[k:k + 16], num_flow_updates=12)[-1] for k in range(0, i0.shape[0], 16)]
    f = torch.cat(out)
    return (F.avg_pool2d(f, 16) / 16).permute(0, 2, 3, 1).reshape(f.shape[0], S, 2).cpu().numpy().astype(np.float64)


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), ok


# ------------------------------------------------------------------ 흐름 판
if "f" in a.part:
    def flow_windows():
        for m in json.load(open("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_meta.json"))[:150][:LIM]:
            yield "k710_holdout", m["path"], m["start"]
        for g in ("passive", "actor"):
            for m in json.load(open(f"/workspace/auto_research_cache/ssv2_classes/{g}_meta.json"))[:120][:LIM]:
                yield f"ssv2_{g}", m["path"], m["start"]
        for m in json.load(open("/workspace/auto_research_cache/natural_train/ht100m_1pct_meta.json"))[:120][:LIM]:
            yield "ht100m", m["path"], m["start"]
        ms = sum([json.load(open(f"/workspace/auto_research_cache/phys3d/meta_val_s{k}.json")) for k in range(4)], [])
        random.Random(0).shuffle(ms)
        for m in ms[:120][:LIM]:
            yield "phys3d", os.path.join("/workspace/dataset/training_data/phys3d_single_v1", m["path"]), int(m.get("r0", 0))

    D = {}
    DTR = collections.defaultdict(list)
    t0 = time.time(); widx = collections.Counter()
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
            U = raft_cells(x[1:], x[:-1])
            DTR[(src, dt)].append(dtr)
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

    def r2(Y, P):
        return float(1 - ((Y - P) ** 2).sum() / max(((Y - Y.mean(0)) ** 2).sum(), 1e-12))

    def ridge_eval(X, Y, win):
        te = win % 2 == 1; tr = ~te
        Xa = np.c_[X, np.ones(len(X))]
        best, bl = None, None
        f5 = (win[tr] // 2) % 5
        for lam in (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000):
            ss = 0.0
            for k in range(5):
                m = f5 != k
                if m.sum() < 20 or (~m).sum() < 20:
                    continue
                A = Xa[tr][m].T @ Xa[tr][m] + lam * m.sum() / 1000 * np.diag([1] * X.shape[1] + [0]) + 1e-9 * np.eye(X.shape[1] + 1)
                W = np.linalg.solve(A, Xa[tr][m].T @ Y[tr][m]); ss += ((Y[tr][~m] - Xa[tr][~m] @ W) ** 2).sum()
            if best is None or ss < best:
                best, bl = ss, lam
        if tr.sum() < 50:
            return float("nan")
        bl = 1.0 if bl is None else bl
        A = Xa[tr].T @ Xa[tr] + bl * tr.sum() / 1000 * np.diag([1] * X.shape[1] + [0]) + 1e-9 * np.eye(X.shape[1] + 1)
        W = np.linalg.solve(A, Xa[tr].T @ Y[tr])
        return r2(Y[te], Xa[te] @ W)

    fr = {}
    groups = {"k710_holdout": ["k710_holdout"], "ssv2": ["ssv2_passive", "ssv2_actor"], "ssv2_passive": ["ssv2_passive"], "ssv2_actor": ["ssv2_actor"],
              "ht100m": ["ht100m"], "natural": ["k710_holdout", "ssv2_passive", "ssv2_actor", "ht100m"], "phys3d": ["phys3d"]}
    for dt in (0.5, 0.25):
        for gname, srcs in groups.items():
            ds = [D[(s, dt)] for s in srcs if (s, dt) in D and D[(s, dt)]["M"]]
            if not ds:
                continue
            off = 0; parts = collections.defaultdict(list)
            for s_, d in zip([s for s in srcs if (s, dt) in D and D[(s, dt)]["M"]], ds):
                for key in ("M", "P1", "P2", "N1"):
                    parts[key].append(np.concatenate(d[key]))
                parts["win"].append(np.concatenate(d["win"]) + off); off += 100000
            M, P1, P2, N1, win = (np.concatenate(parts[k]) for k in ("M", "P1", "P2", "N1", "win"))
            te = win % 2 == 1
            if te.sum() < 100:
                continue
            e = dict(n=int(len(M)), n_test=int(te.sum()), dt_real=float(np.mean(sum([DTR[(s, dt)] for s in srcs], []))),
                     past_ridge=ridge_eval(np.c_[P1, P2], M, win), both_ridge=ridge_eval(np.c_[P1, N1], M, win),
                     constvel=r2(M[te], P1[te]), both_avg=r2(M[te], (P1[te] + N1[te]) / 2), next_only=r2(M[te], N1[te]),
                     # 등록 뒤 보충: 구멍 장을 안 쓰는 관측 가능 판 — 구멍 건너 변위 D = M + N1 (구멍 앞 장 ↔ 뒤 장 흐름과 같다)
                     across_ridge=ridge_eval(np.c_[P1, M + N1], M, win), across_half=r2(M[te], (M[te] + N1[te]) / 2))
            fr[f"{gname}|{dt}"] = e
            print(f"흐름 {gname:13s} Δt {dt} (실제 {e['dt_real']:.3f}) n {e['n_test']:6d} | 과거만 ridge {e['past_ridge']:+.3f} · 앞뒤 ridge {e['both_ridge']:+.3f} | "
                  f"등속 {e['constvel']:+.3f} · 앞뒤 평균 {e['both_avg']:+.3f} · 다음만 {e['next_only']:+.3f} | 보충 구멍 건너 ridge {e['across_ridge']:+.3f} · D/2 {e['across_half']:+.3f}", flush=True)
    res["flow"] = fr
    json.dump(res, open(a.out, "w"), indent=1)


# ------------------------------------------------------------------ 토큰 판
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


def interp_W(h0, h1, Fl):                                                     # h0 = 블록 t−1, h1 = 블록 t+1, Fl = 블록 t+1 칸 → 블록 t−1 자리 변위
    q, oq = cell_of(P0 + Fl / 2); r, orr = cell_of(P0 + Fl)
    ok = oq & orr
    acc = np.zeros_like(h0); n = np.zeros(S)
    np.add.at(acc, q[ok], h1[ok]); np.add.at(acc, q[ok], h0[r[ok]]); np.add.at(n, q[ok], 2)
    out = (h0 + h1) / 2; m = n > 0
    out[m] = acc[m] / n[m][:, None]
    return out


if "t" in a.part:
    SETS = {"ssv2_passive": "/workspace/auto_research_cache/ssv2_classes/passive", "ssv2_actor": "/workspace/auto_research_cache/ssv2_classes/actor",
            "k710_holdout": "/workspace/auto_research_cache/natural_holdout/k710_320_holdout"}
    tr_ = {}
    t0 = time.time()
    for name, base in SETS.items():
        Hs = np.load(base + "_h.npy", mmap_mode="r"); Us = np.load(base + "_U.npy", mmap_mode="r"); meta = json.load(open(base + "_meta.json"))
        Dk = collections.defaultdict(list)
        for w, m in enumerate(meta[:LIM]):
            try:
                vr = decord.VideoReader(m["path"], num_threads=2)
                fidx = [m["start"] + m["fstp"] * j for j in (7, 9, 11, 13, 15)]                          # 블록 3..7 대표 장
                if fidx[-1] >= len(vr):
                    continue
                xs = prep(vr.get_batch(fidx).asnumpy())
            except Exception as e:
                print("skip", m["path"], e, flush=True); continue
            Fls = raft_cells(xs[2:5], xs[0:3])                                 # 블록 t+1 → t−1, t = 4, 5, 6
            h = np.asarray(Hs[w], np.float32); U = np.asarray(Us[w], np.float32).reshape(8, S, 2).astype(np.float64)
            for i, t in enumerate((4, 5, 6)):
                sel = np.linalg.norm(U[t], axis=1) >= 0.5
                if sel.sum() < 2:
                    continue
                Fl = Fls[i]
                Dk["half"].append(np.full(int(sel.sum()) * h.shape[-1], w % 2, np.int8))
                Dk["y"].append(h[t][sel]); Dk["cp"].append(h[t - 1][sel])
                Dk["mE"].append(h[t - 4:t].mean(0)[sel]); Dk["mI"].append(((h[t - 1] + h[t + 1]) / 2)[sel])
                Dk["E"].append(CS.splat(h[t - 1], U[t - 1])[sel]); Dk["E_r"].append(CS.splat(h[t - 1], U[t - 1] @ R90.T)[sel])
                Dk["I"].append(interp_W(h[t - 1], h[t + 1], Fl)[sel]); Dk["I_r"].append(interp_W(h[t - 1], h[t + 1], Fl @ R90.T)[sel])
            if w % 50 == 0:
                print(f"토큰 {name} {w} · {time.time() - t0:.0f}s", flush=True)
        half = np.concatenate(Dk.pop("half"))
        N = half.size
        rs = np.random.RandomState(1); idx = rs.choice(N, min(a.npts, N), replace=False)
        A = {k: torch.from_numpy(np.concatenate(v).ravel()[idx]).to(dev) for k, v in Dk.items()}
        hh = torch.from_numpy(half[idx]).to(dev)
        del Dk
        e = {"n_points_total": int(N), "n_points": int(idx.size)}
        for sub, msk in (("all", None), ("half0", hh == 0), ("half1", hh == 1)):
            B = A if msk is None else {k: v[msk] for k, v in A.items()}
            if B["y"].numel() < 1000:
                continue
            for tg, tname in (("y", "data"), ("cp", "copy")):
                for wn, mn in (("E", "mE"), ("I", "mI")):
                    f0 = prof(B[tg], B[mn], B[wn], 0.0)
                    bt, ft = golden(B[tg], B[mn], B[wn]); br, frr = golden(B[tg], B[mn], B[wn + "_r"])
                    e[f"{sub}|{wn}|{tname}"] = dict(b_true=round(bt, 4), b_rot=round(br, 4), net_b=round(bt - br, 4), net_gain_pct=round(100 * (frr - ft) / f0, 3), L1_base=round(f0, 4))
            if sub == "all":
                # 방향 없는 비교: 문맥 평균 · 앞 블록 복사 · 앞뒤 평균 각각의 L1 (정답과의 평균 절댓값)
                e["L1_ctxmean"] = float((B["y"] - B["mE"]).abs().mean()); e["L1_copyprev"] = float((B["y"] - B["cp"]).abs().mean())
                e["L1_avg_before_after"] = float((B["y"] - B["mI"]).abs().mean())
        tr_[name] = e
        print(f"\n== 토큰 {name}: 점 {e['n_points']} / {N}")
        for sub in [x for x in ("all", "half0", "half1") if f"{x}|E|data" in e]:
            print(f"   [{sub}] " + " | ".join(f"{wn} 데이터 위약 뺀 이득 {e[f'{sub}|{wn}|data']['net_gain_pct']:+.3f} % (b {e[f'{sub}|{wn}|data']['b_true']:.3f} / 위약 {e[f'{sub}|{wn}|data']['b_rot']:.3f})"
                                        f" · 복사 {e[f'{sub}|{wn}|copy']['net_gain_pct']:+.3f} %" for wn in ("E", "I")), flush=True)
        print(f"   방향 없는 L1: 문맥 평균 {e['L1_ctxmean']:.4f} · 앞 블록 복사 {e['L1_copyprev']:.4f} · 앞뒤 평균 {e['L1_avg_before_after']:.4f}", flush=True)
        del A
        torch.cuda.empty_cache()
    res["tokens"] = tr_
    prev = json.load(open(a.out)) if os.path.exists(a.out) and "f" not in a.part else {}
    if prev.get("flow") and "flow" not in res:
        res["flow"] = prev["flow"]
    json.dump(res, open(a.out, "w"), indent=1)
print("저장", a.out)
