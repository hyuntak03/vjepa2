#!/usr/bin/env python3
"""[P] 실제 영상에서 가려진 물체가 (어디서든) 다시 나타나는 비율 π_obj — CoTracker3 점 추적 (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/obj_return.py --sets posctrl
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/obj_return.py --sets natural

사전 등록: 진행판 [P] KST 04:00.
장치: CoTracker3 offline (torch hub facebookresearch/co-tracker, scaled_offline.pth). 프레임 ≈ 12–15 fps, 짧은 변 256.
질문 점: 가운데 정사각 (VOBench 는 화면 전체) 안 20×20 격자 × 질문 시각 {0, 0.5 초, 1 초}.
가림 사건: v[j−2], v[j−1] 보임 · v[j], v[j+1] 안 보임 (j ≥ 질문 시각 + 2). 추정 위치가 화면 안 (여백 3 px) 이면 가림, 아니면 화면 밖.
돌아옴: j 뒤 처음 (보임 · 화면 안 · 다음 장도 보임 (마지막 장 예외)) 인 j2. Δ 마다 j + H ≤ T−1 인 사건만 (H = round(Δ·fps)), j2 − j ≤ H 이면 다시 보임.
패치 맞음: 9×9 RGB 평균 절대차 |I_{j2}(p[j2]) − I_{j−1}(p[j−1])| < τ(간격), τ = 한 번도 안 가려진 점의 같은 간격 차이 90 분위 (데이터셋별).
  우연: I_{j−1} 쪽을 (+48, +48) px 옆 (화면 안으로 접음) 으로 바꾼 비교.
층: moving = 가림 직전 0.5 초 이동 − 같은 구간 보이는 점 이동 중앙값 > 4 px. occluder = 시작 장에서 16 px 안 다른 보이는 점의 2 장 이동이 이 점 추정 이동과 > 3 px 다름.
묶음: 같은 질문 시각 · 시작 장 ±1 · 격자 8-이웃 사건을 합집합. Δ 마다 자격 있는 점의 절반 이상이 돌아오면 (패치 맞음) 묶음 돌아옴.
양성 대조 (VOBench pos_a, 가운데 · 끝 가림): 물체 점 = 질문 시각에 라벨 물체 원 안. 참값 = 라벨 가림 < 0.5 로 돌아오는 시각.
출력: auto_research/results/obj_return_{sets}.json (요약 · 창 bootstrap) + _events.npz."""
import argparse, csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--sets", choices=["posctrl", "natural"], required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(8)
dev = torch.device("cuda")
REPO = "/workspace/world_model/vjepa2"
OUT = os.path.join(REPO, f"auto_research/results/obj_return_{a.sets}.json")
VB = "/workspace/dataset/benchmarks/VOBench_v1"
ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", verbose=False).to(dev).eval()
S0 = 256; GQ = 20; DELTAS = (0.5, 1.0, 2.0); PATCH = 9; MARG = 3


def resize_short(x):                                                            # x (T,3,H,W) float 0–255 → 짧은 변 256
    T, _, H, W = x.shape; s = S0 / min(H, W)
    return F.interpolate(x, size=(round(H * s), round(W * s)), mode="bilinear", align_corners=False, antialias=True).clamp(0, 255)


def load_natural(m):
    vr = decord.VideoReader(m["path"], num_threads=1)
    fps = vr.get_avg_fps(); step = max(1, round(fps / 12.5))
    span = m.get("span", 15 * m["fstp"])
    idx = list(range(m["start"], m["start"] + span + 1, step))[:60]
    if idx[-1] >= len(vr):
        return None
    x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(0, 3, 1, 2).float()
    return resize_short(x), fps / step, None


def load_vob(m):
    from PIL import Image
    idx = list(range(0, 96, 2))
    x = torch.from_numpy(np.stack([np.asarray(Image.open(f"{m['png_dir']}/{i:06d}.png").convert("RGB")) for i in idx])).to(dev).permute(0, 3, 1, 2).float()
    W0 = x.shape[-1]; x = resize_short(x); sc = x.shape[-1] / W0
    lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(m["png_dir"]), "label.json")))["labels"]["A"]}
    L = [lab.get(i) for i in idx]
    obj = [(q["u"] * sc, q["v"] * sc, q["r_px"] * sc, q["hidden"]) if q else None for q in L]
    return x, 15.0, obj


def patches(x, t_idx, pos):                                                    # x (T,3,H,W), t_idx (n,), pos (n,2) → (n, 3·P·P), 장마다 묶어 읽음
    T, C, H, W = x.shape
    r = torch.arange(PATCH, device=dev).float() - PATCH // 2
    gy, gx = torch.meshgrid(r, r, indexing="ij")
    px = pos[:, 0, None, None] + gx; py = pos[:, 1, None, None] + gy
    g = torch.stack([2 * px / (W - 1) - 1, 2 * py / (H - 1) - 1], -1)            # (n,P,P,2)
    out = torch.zeros(len(t_idx), C * PATCH * PATCH, device=dev)
    for t in torch.unique(t_idx).tolist():
        m = t_idx == t
        o = F.grid_sample(x[t:t + 1], g[m].reshape(1, -1, PATCH, 2), align_corners=True, padding_mode="border")   # (1,3,n_t·P,P)
        out[m] = o.reshape(C, -1, PATCH, PATCH).permute(1, 0, 2, 3).reshape(-1, C * PATCH * PATCH)
    return out


def analyse(x, fps, obj, wi):
    T, _, H, W = x.shape
    side = min(H, W); y0, x0 = (H - side) // 2, (W - side) // 2
    lin = torch.linspace(8, side - 8, GQ, device=dev)
    gy, gx = torch.meshgrid(lin + y0, lin + x0, indexing="ij")
    tqs = [0, round(0.5 * fps), round(1.0 * fps)]
    q = torch.cat([torch.stack([torch.full((GQ * GQ,), float(t), device=dev), gx.flatten(), gy.flatten()], -1) for t in tqs])[None]
    with torch.no_grad():
        tr, vis = ct(x[None], queries=q)
    P = tr[0].float(); V = vis[0]                                               # (T,N,2), (T,N)
    Pc, Vc = P.cpu().numpy(), V.cpu().numpy()
    N = P.shape[1]
    ins = (Pc[..., 0] >= MARG) & (Pc[..., 0] <= W - 1 - MARG) & (Pc[..., 1] >= MARG) & (Pc[..., 1] <= H - 1 - MARG)
    H05 = round(0.5 * fps)
    ev = []                                                                     # 사건
    ctrl = {}                                                                   # 간격 → 안 가려진 점 패치 차이 쌍 (t0, t1, n)
    for n in range(N):
        tq = tqs[n // (GQ * GQ)]; v = Vc[:, n]
        j1 = None
        for j in range(tq + 2, T - 1):
            if v[j - 2] and v[j - 1] and not v[j] and not v[j + 1]:
                j1 = j; break
        if j1 is None:
            if v[tq:].all():                                                    # 대조 후보: 끝까지 보임
                ctrl.setdefault("pts", []).append(n)
            continue
        kind = "occ" if (ins[j1, n] and ins[j1 + 1, n]) else "left"
        j2 = None
        for t in range(j1 + 2, T):
            if v[t] and ins[t, n] and (t == T - 1 or v[t + 1]):
                j2 = t; break
        k0 = max(tq, j1 - 1 - H05)
        both = Vc[k0] & Vc[j1 - 1]
        cam = np.median(Pc[j1 - 1, both] - Pc[k0, both], 0) if both.sum() >= 10 else np.zeros(2)
        rel = np.linalg.norm((Pc[j1 - 1, n] - Pc[k0, n]) - cam)
        near = Vc[j1] & (np.linalg.norm(Pc[j1] - Pc[j1, n], axis=1) < 16); near[n] = False
        if near.any():
            dm = (Pc[j1, near] - Pc[j1 - 2, near]) - (Pc[j1, n] - Pc[j1 - 2, n])
            occl = bool((np.linalg.norm(dm, axis=1) > 3).any())
        else:
            occl = False
        isobj = False; truth_ret = None
        if obj is not None and obj[tq] is not None:
            u, vv, r, _ = obj[tq]
            isobj = bool(np.hypot(gx.flatten()[n % (GQ * GQ)].item() - u, gy.flatten()[n % (GQ * GQ)].item() - vv) < 0.8 * r)
            hid = [o[3] if o is not None else 1.0 for o in obj]
            tr_ = [t for t in range(j1, T) if hid[t] < 0.5]
            truth_ret = (tr_[0] - j1) if tr_ else None
        ev.append(dict(n=n, tq=tq, gi=n % (GQ * GQ), j1=j1, j2=j2, kind=kind, rel=float(rel), occl=occl, isobj=isobj, truth=truth_ret))
    # 패치 비교: 드러난 패치가 "가림 직전 자기 패치" 와, 같은 장의 무작위 자리 32 곳의 패치보다 얼마나 가까운가 (식별 검사)
    e_occ = [e for e in ev if e["j2"] is not None]
    if e_occ:
        K = 32; rng = np.random.default_rng(1000 + wi)
        nn = torch.tensor([e["n"] for e in e_occ], device=dev)
        t0 = torch.tensor([e["j1"] - 1 for e in e_occ], device=dev); t1 = torch.tensor([e["j2"] for e in e_occ], device=dev)
        pre = patches(x, t0, P[t0, nn]); post = patches(x, t1, P[t1, nn])
        alt_pos = torch.from_numpy(np.stack([rng.uniform(PATCH, W - 1 - PATCH, (len(e_occ), K)), rng.uniform(PATCH, H - 1 - PATCH, (len(e_occ), K))], -1)).float().to(dev)
        alt = patches(x, t0.repeat_interleave(K), alt_pos.reshape(-1, 2)).reshape(len(e_occ), K, -1)
        d = (post - pre).abs().mean(1)                                          # (n,)
        da = (post[:, None] - alt).abs().mean(-1)                               # (n,K)
        q10 = torch.quantile(da[:, 1:], 0.1, dim=1)
        match = (d < q10).cpu().numpy(); match_r = (da[:, 0] < q10).cpu().numpy()
        g = pre.reshape(len(e_occ), 3, PATCH, PATCH).mean(1).flatten(1).std(1).cpu().numpy()
        for e, m1, m2, di, ti in zip(e_occ, match, match_r, d.cpu().numpy(), g):
            e["match"] = bool(m1); e["match_r"] = bool(m2); e["d"] = float(di); e["tex"] = float(ti)
    cd = []
    for e in ev:
        e["win"] = wi
    return ev, cd, dict(T=T, fps=fps, N=N, H=H, W=W)


def groups(evs):
    """같은 창 · 질문 시각 · 시작 장 ±1 · 격자 8-이웃 → 묶음 번호."""
    par = list(range(len(evs)))
    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    key = {}
    for i, e in enumerate(evs):
        key.setdefault((e["win"], e["tq"]), []).append(i)
    for ids in key.values():
        pos = {evs[i]["gi"]: i for i in ids}
        for i in ids:
            r, c = divmod(evs[i]["gi"], GQ)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    k = (r + dr) * GQ + (c + dc)
                    if (dr or dc) and 0 <= r + dr < GQ and 0 <= c + dc < GQ and k in pos:
                        j = pos[k]
                        if abs(evs[j]["j1"] - evs[i]["j1"]) <= 1:
                            par[f(i)] = f(j)
    return [f(i) for i in range(len(evs))]


def summarize(evs, cd, fps_by_win, nwin):
    occ = [e for e in evs if e["kind"] == "occ"]
    gid = groups(occ)
    out = dict(n_windows=nwin, n_events=len(evs), n_occ=len(occ), n_left=sum(e["kind"] == "left" for e in evs),
               frac_moving=float(np.mean([e["rel"] > 4 for e in occ])) if occ else None,
               frac_occluder=float(np.mean([e["occl"] for e in occ])) if occ else None, by_delta={})
    rng = np.random.default_rng(0)
    for D in DELTAS:
        res = {}
        for strat, fn in (("all", lambda e: True), ("moving", lambda e: e["rel"] > 4), ("static", lambda e: e["rel"] <= 4),
                          ("moving_occluder", lambda e: e["rel"] > 4 and e["occl"]), ("static_occluder", lambda e: e["rel"] <= 4 and e["occl"]),
                          ("object_label", lambda e: e["isobj"])):
            el = [(i, e) for i, e in enumerate(occ) if fn(e) and e["j1"] + round(D * fps_by_win[e["win"]]) <= e["Tw"] - 1]
            if not el:
                continue
            Hd = lambda e: round(D * fps_by_win[e["win"]])
            rv = np.array([e["j2"] is not None and e["j2"] - e["j1"] <= Hd(e) for _, e in el])
            mt = np.array([bool(r and e["match"]) for r, (_, e) in zip(rv, el)])
            mr = np.array([bool(r and e["match_r"]) for r, (_, e) in zip(rv, el)])
            wins = np.array([e["win"] for _, e in el])
            # 묶음
            gmap = {}
            for (i, e), m_ in zip(el, mt):
                gmap.setdefault(gid[i], []).append(m_)
            gret = np.array([np.mean(v) >= 0.5 for v in gmap.values()])
            gwin = np.array([occ[k]["win"] for k in gmap.keys()])
            def boot(vals, ws):
                uw = np.unique(ws); idx = {w: np.where(ws == w)[0] for w in uw}; bs = []
                for _ in range(1000):
                    pick = rng.choice(uw, len(uw)); ii = np.concatenate([idx[w] for w in pick]); bs.append(vals[ii].mean())
                return [round(float(np.quantile(bs, 0.025)), 3), round(float(np.quantile(bs, 0.975)), 3)]
            low = float(mr.mean()); m_ = float(mt.mean()); rvm = float(rv.mean())
            r_ = dict(n=len(el), n_win=int(len(np.unique(wins))), revisible=round(rvm, 3), returned_match=round(m_, 3), returned_match_ci=boot(mt.astype(float), wins),
                      chance_match=round(low, 3), pi_corr=round(float(np.clip((m_ - low) / max(0.9 * rvm - low, 1e-6), 0, 1)) * rvm if rvm > 0 else 0.0, 3),
                      n_groups=int(len(gret)), group_returned=round(float(gret.mean()), 3), group_returned_ci=boot(gret.astype(float), gwin))
            if strat == "object_label":
                tr = np.array([e["truth"] is not None and e["truth"] <= Hd(e) for _, e in el]); r_["truth_return"] = round(float(tr.mean()), 3)
                trw = np.array([e["truth"] is not None for _, e in el]); r_["truth_return_window"] = round(float(trw.mean()), 3)
            res[strat] = r_
        out["by_delta"][str(D)] = res
    # 창 끝까지 (Δ 제한 없음): 다시 보인 비율 · 패치 맞음
    for strat, fn in (("all", lambda e: True), ("moving", lambda e: e["rel"] > 4), ("object_label", lambda e: e["isobj"])):
        el = [e for e in occ if fn(e)]
        if el:
            rv = np.array([e["j2"] is not None for e in el]); mt = np.array([e["j2"] is not None and e["match"] for e in el])
            out.setdefault("window_end", {})[strat] = dict(n=len(el), revisible=round(float(rv.mean()), 3), returned_match=round(float(mt.mean()), 3),
                                                            truth=round(float(np.mean([e["truth"] is not None for e in el])), 3) if strat == "object_label" else None)
    return out


# ------------------------------------------------------------------ 세트
SETS = {}
if a.sets == "posctrl":
    rows = list(csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))))
    for occl in ("mid", "late"):
        rr = [r for r in rows if r["occlusion"] == occl and r["variant"] == "pos_a"]
        SETS[f"vob_{occl}_static"] = ("vob", [dict(png_dir=os.path.join(VB, r["file_name"])) for r in rr if r["motion"] == "flat_static"])
        SETS[f"vob_{occl}_moving"] = ("vob", [dict(png_dir=os.path.join(VB, r["file_name"])) for r in rr if r["motion"] != "flat_static"][:72])
else:
    SETS["k710_holdout"] = ("nat", json.load(open("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_meta.json"))[:150])
    SETS["ht100m"] = ("nat", json.load(open("/workspace/auto_research_cache/natural_train/ht100m_1pct_meta.json"))[:120])
    ek = sorted(os.listdir("/workspace/dataset/benchmarks/EK100/videos_256p/videos"))
    rng0 = np.random.default_rng(0); pick = rng0.choice(len(ek), 120, replace=False)
    ekw = []
    for i in sorted(pick):
        pth = os.path.join("/workspace/dataset/benchmarks/EK100/videos_256p/videos", ek[i])
        try:
            vr = decord.VideoReader(pth, num_threads=1); n = len(vr); fps = vr.get_avg_fps()
        except Exception:
            continue
        span = int(round(3.5 * fps))
        if n <= span + 1:
            continue
        ekw.append(dict(path=pth, start=int(rng0.integers(0, n - span - 1)), fstp=1, span=span))
    SETS["ek100"] = ("nat", ekw)

result = {}
t0 = time.time()
ALL = {}
for name, (kind, metas) in SETS.items():
    metas = metas[: a.limit or None]
    evs, cds, fpsw, nwin = [], [], {}, 0
    for wi, m in enumerate(metas):
        got = load_vob(m) if kind == "vob" else load_natural(m)
        if got is None:
            continue
        x, fps, obj = got
        ev, cd, info = analyse(x, fps, obj, wi)
        for e in ev:
            e["Tw"] = info["T"]
        evs += ev; cds += cd; fpsw[wi] = fps; nwin += 1
        if (wi + 1) % 20 == 0:
            print(f"{name} {wi + 1}/{len(metas)} 사건 {len(evs)} ({time.time() - t0:.0f}s)", flush=True)
    result[name] = summarize(evs, cds, fpsw, nwin)
    ALL[name] = (evs, cds, fpsw, nwin)
    print(name, json.dumps(result[name], ensure_ascii=False), flush=True)
    np.savez_compressed(OUT.replace(".json", f"_{name}_events.npz"), ev=np.array([json.dumps(e) for e in evs]))
# K710 + HT100M 합친 주 숫자
def pool(names, key):
    evs, cds, fpsw, nwin = [], [], {}, 0
    for k, nm in enumerate(names):
        if nm not in ALL:
            continue
        e_, c_, f_, n_ = ALL[nm]
        off = (k + 1) * 100000
        for e in e_:
            e2 = dict(e); e2["win"] = e["win"] + off; evs.append(e2)
        cds += c_; fpsw.update({w + off: v for w, v in f_.items()}); nwin += n_
    result[key] = summarize(evs, cds, fpsw, nwin)
    print(key, json.dumps(result[key], ensure_ascii=False), flush=True)
if a.sets == "natural":
    pool(["k710_holdout", "ht100m"], "k710_ht100m")
else:
    pool(["vob_mid_static", "vob_late_static"], "vob_static")
    pool(["vob_mid_moving", "vob_late_moving"], "vob_moving")
json.dump(result, open(OUT, "w"), indent=1, ensure_ascii=False)
print("저장", OUT, f"({time.time() - t0:.0f}s)")
