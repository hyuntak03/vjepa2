#!/usr/bin/env python3
"""predictor 은닉 곡률 진단 (학습 없음). 판정 기준은 z_analysis/docs/HIDDEN_CURVATURE_2026-11-07.md 의 사전 등록.

  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/hidden_curvature.py extract --out <dir>
  /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/hidden_curvature.py analyze --out <dir> --json z_analysis/results/hidden_curvature_20261107/summary.json

extract: 영상마다 블록별 predictor 은닉을 256 칸 평균해 저장한다 (teacher forcing 경로 · 문맥 8 블록 뒤 상태 되먹임 경로).
analyze: 곡률 · 물체 성분 · 위치 판독 · 정지 mask · 정지 화면 대조를 낸다.
"""
import argparse, csv, hashlib, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
S, C, NB, T = 256, 8, 16, 32
LAYERS = [0, 3, 6, 9, 11, 12]                      # 통과한 predictor 층 수. 마지막 칸 = 상태 (predictor_norm)
LNAMES = [f"L{l}" for l in LAYERS] + ["ST"]
MODELS = {"base": "natural_ar_vw_state", "v4bc4_l1": "v4bc4_ar_vw_state_postft", "v4bc4_neg": "v4bc4_ar_vw_state_negimp", "v4bc4_curv": "v4bc4_ar_vw_state_curv"}
REAL = "/workspace/dataset/benchmarks/real_world_permanence"
NAT = "/workspace/auto_research_cache/natural_holdout/k710_320_holdout_meta.json"


def h8(s):
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)


def synth_label(path):
    z = np.load(path.replace("/clips/", "/labels/").replace(".mp4", ".npz"))
    vis, amo, cen = z["visible_n"][:, 0].astype(np.float32), z["amodal_n"][:, 0].astype(np.float32), z["center_px"][:, 0]
    f = np.arange(NB) * 4
    r = 0.5 * (vis[f] / np.maximum(amo[f], 1) + vis[f + 2] / np.maximum(amo[f + 2], 1))
    return dict(r=r.astype(np.float32), amodal=(0.5 * (amo[f] + amo[f + 2])).astype(np.float32), cen=cen[f].astype(np.float32))


def build_items(n_fam, n_slide, n_nat, n_frozen, seed=0):
    rng = random.Random(seed)
    d = json.load(open(REPO / "data_csv/blender_track_v4bc4_negimp/families.json"))
    lst = [l.strip() for l in open(d["list"]) if l.strip()]
    groups = {}
    for f in d["families"]:
        if not any(f["hs"]) or f["occ"] == "object":
            continue
        groups.setdefault(f"{f['plane']}_{f['occ']}", []).append(f)
    items, seen = [], set()
    for g, fs in sorted(groups.items()):
        rng.shuffle(fs)
        k = 0
        for f in fs:
            cand = [a for a, h in zip(f["A"], f["hs"]) if h]
            a = rng.choice(cand)
            if lst[a] in seen:
                continue
            seen.add(lst[a])
            fid = f"{f['src']}_{f['plane']}_{f['family']}"
            idx = list(range(0, 64, 2))
            items.append(dict(id=f"A_{fid}", kind="A", group=g, fam=fid, path=lst[a], idx=idx, crop=True, label=synth_label(lst[a])))
            items.append(dict(id=f"B_{fid}", kind="B", group=g, fam=fid, path=lst[f["B"]], idx=idx, crop=True))
            k += 1
            if k >= (n_slide if f["occ"] == "slide" else n_fam):
                break
    rows = list(csv.DictReader(open(REPO / "data_csv/real_world_permanence_uniform/index.csv")))
    for r in rows:
        if r["file"].endswith(("possible_present.mp4", "possible_absent.mp4")):
            kind = "RA" if r["file"].endswith("present.mp4") else "RB"
            fr = [int(x) for x in r["frames"].split()]
            vdir = os.path.join(REAL, os.path.dirname(r["file"]))
            q = json.load(open(vdir + "/info.json"))["quad"]
            a, b = q["hidden_run"]
            area = np.load(vdir + "/masks.npz")["sam2_area"]
            fa = np.array(fr)
            hid = (fa >= a) & (fa <= b)
            seen_obj = (area[np.clip(fa, 0, len(area) - 1)] > 0) & ~hid
            items.append(dict(id=f"{kind}_{r['block_id']}", kind=kind, group="real", fam=f"real_{r['block_id']}", path=os.path.join(REAL, r["file"]), idx=fr, crop=False,
                              hid=hid.reshape(NB, 2).all(1).tolist(), vis=seen_obj.reshape(NB, 2).all(1).tolist(), src=r["game_name"]))
    for i, m in enumerate(json.load(open(NAT))[:n_nat]):
        items.append(dict(id=f"N_{i}", kind="N", group="natural", fam=f"nat_{i}", path=m["path"], idx=[m["start"] + m["fstp"] * k for k in range(T)], crop=True))
    syn = [x for x in items if x["kind"] == "A"]
    rng.shuffle(syn)
    for x in syn[:n_frozen]:
        items.append(dict(id=f"F_{x['fam']}", kind="F", group="frozen_synth", fam=x["fam"], path=x["path"], idx=[10] * T, crop=True))
    rl = [x for x in items if x["kind"] == "RA"]
    rng.shuffle(rl)
    for x in rl[: n_frozen // 2]:
        items.append(dict(id=f"F_{x['fam']}", kind="F", group="frozen_real", fam=x["fam"], path=x["path"], idx=[x["idx"][4]] * T, crop=False))
    for x in [x for x in items if x["kind"] == "N"][: n_frozen // 2]:
        items.append(dict(id=f"FN_{x['fam']}", kind="F", group="frozen_natural", fam=x["fam"], path=x["path"], idx=[x["idx"][4]] * T, crop=True))
    return items


def extract(a):
    import decord, torch, torch.nn.functional as F, yaml
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
    from app.vjepa_frozen import utils as U
    from app.vjepa_frozen.ar import encode_blocks
    torch.set_num_threads(8)
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    items = build_items(a.n_fam, a.n_slide, a.n_nat, a.n_frozen)
    print("영상", len(items), {k: sum(x["kind"] == k for x in items) for k in "A B RA RB N F".split()}, flush=True)
    cfg0 = yaml.safe_load(open(REPO / "z_training/runs" / MODELS["base"] / "config.yaml"))
    sd = _load_checkpoint(cfg0["model"]["checkpoint"])
    enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(enc, sd["target_encoder"], tag="t")
    enc = enc.to(dev, torch.bfloat16).eval()
    del sd
    preds = {}
    for k, run in MODELS.items():
        c = yaml.safe_load(open(REPO / "z_training/runs" / run / "config.yaml"))
        p = U.build_predictor(c["model"], 1280, None, dev, T)
        st = _load_checkpoint(str(REPO / "z_training/runs" / run / "latest.pt"))
        p.load_state_dict(_clean_backbone_key(st["predictor"]), strict=True)
        preds[k] = p.float().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def load(x):
        try:
            vr = decord.VideoReader(x["path"], num_threads=2)
            n = len(vr)
            ids = x["idx"]
            if x["kind"] == "N" and ids[-1] >= n:           # 창이 영상 끝을 넘으면 같은 간격으로 앞으로 민다
                sh = ids[-1] - (n - 1)
                if ids[0] - sh < 0:
                    return None
                ids = [i - sh for i in ids]
            fr = vr.get_batch([min(i, n - 1) for i in ids]).asnumpy()
        except Exception as e:
            print("읽기 실패", x["path"], e, flush=True)
            return None
        t = torch.from_numpy(fr).permute(0, 3, 1, 2).float() / 255
        if x["crop"] and t.shape[2] != t.shape[3]:
            m = min(t.shape[2:])
            y0, x0 = (t.shape[2] - m) // 2, (t.shape[3] - m) // 2
            t = t[:, :, y0:y0 + m, x0:x0 + m]
        if t.shape[2] != 256 or t.shape[3] != 256:
            t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
        return (t.permute(1, 0, 2, 3) - mean) / std

    @torch.no_grad()
    def step(p, xb, pos, caches, embedded):
        h = xb if embedded else p._ar_embed_obs(xb)
        pooled, new = [h.mean(1)], []
        for i, blk in enumerate(p.predictor_blocks):
            h, kv = blk.forward_kv(h, pos, p.grid_height, p.grid_width, None if caches is None else caches[i])
            new.append(kv)
            if (i + 1) in LAYERS:
                pooled.append(h.mean(1))
        st = p.predictor_norm(h)
        pooled.append(st.mean(1))
        return torch.stack(pooled, 1), new, st

    @torch.no_grad()
    def run(p, x):
        B = x.size(0)
        idx = torch.arange(NB * S, device=dev).unsqueeze(0).expand(B, -1)
        tf, ro, caches, cc, st_c = [], [], None, None, None
        for b in range(NB):
            sl = slice(b * S, (b + 1) * S)
            pl, caches, st = step(p, x[:, sl], idx[:, sl], caches, False)
            tf.append(pl)
            if b == C - 1:
                cc, st_c = caches, st
        ro = tf[:C]
        caches, st = cc, st_c
        for b in range(C, NB - 1):
            sl = slice(b * S, (b + 1) * S)
            pl, caches, st = step(p, st + p.type_embed[1], idx[:, sl], caches, True)
            ro.append(pl)
        return torch.stack(tf, 1), torch.stack(ro, 1), st

    out = {k: {"tf": [], "ro": []} for k in MODELS}
    inp, keep = [], []
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for s in range(0, len(items), a.bs):
            chunk = items[s:s + a.bs]
            vids = list(ex.map(load, chunk))
            ok = [i for i, v in enumerate(vids) if v is not None]
            if not ok:
                continue
            clips = torch.stack([vids[i] for i in ok]).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x = torch.cat([encode_blocks(enc, clips[j:j + 8], 2) for j in range(0, len(ok), 8)])
            x = x.float()
            inp.append(x.view(len(ok), NB, S, -1).mean(2).cpu().numpy().astype(np.float16))
            for k, p in preds.items():
                tf, ro, st = run(p, x)
                if s == 0:                                  # 측정 도구 검사: 손으로 쓴 경로 = 모델의 굴림
                    with torch.no_grad():
                        ref = p.rollout_kv(x[:, : C * S], torch.arange(C * S, device=dev).unsqueeze(0).expand(x.size(0), -1), NB - C, return_states=True)[1]
                    print(k, "상태 되먹임 경로 일치 max|d|", float((ref[:, -S:] - st).abs().max()), flush=True)
                out[k]["tf"].append(tf.cpu().numpy().astype(np.float16))
                out[k]["ro"].append(ro.cpu().numpy().astype(np.float16))
            keep += [chunk[i] for i in ok]
            if (s // a.bs) % 10 == 0:
                print(f"{s}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(a.out, "pooled.npz"), inp=np.concatenate(inp),
             **{f"{k}_{p}": np.concatenate(v[p]) for k, v in out.items() for p in ("tf", "ro")})
    json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in x.items() if k != "label"} | ({"label": {a_: b.tolist() for a_, b in x["label"].items()}} if "label" in x else {}) for x in keep],
              open(os.path.join(a.out, "items.json"), "w"))
    print("끝", time.time() - t0, flush=True)


# ---------------------------------------------------------------- 분석
def vel(X):
    return X[:, 1:] - X[:, :-1]


def cosv(X):
    v = vel(X)
    a, b = v[:, :-1], v[:, 1:]
    na, nb_ = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(na * nb_, 1e-9), np.minimum(na, nb_)


def theta(c):
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def stat(c, m):
    """c (n, K) cos, m (n, K) bool → 곡률 평균 · 영상 bootstrap 95 %."""
    if m.sum() == 0:
        return dict(theta=None, cos=None, n=0, clips=0)
    t = theta(c)
    rows = np.where(m.any(1))[0]
    per = np.array([t[i][m[i]].mean() for i in rows])
    g = np.random.default_rng(0)
    bs = per[g.integers(0, len(per), (1000, len(per)))].mean(1)
    return dict(theta=round(float(t[m].mean()), 1), ci=[round(float(np.percentile(bs, q)), 1) for q in (2.5, 97.5)], cos=round(float(c[m].mean()), 3), n=int(m.sum()), clips=len(rows))


def ridge_fit(Xa, Ya, Xv, Yv):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    A = (Xa - mu) / sd
    ym = Ya.mean(0)
    best = None
    for al in (1, 10, 100, 1000, 10000):
        W = np.linalg.solve(A.T @ A + al * np.eye(A.shape[1]), A.T @ (Ya - ym))
        e = np.linalg.norm(((Xv - mu) / sd) @ W + ym - Yv, axis=1).mean()
        if best is None or e < best[0]:
            best = (e, W)
    W = best[1]
    return lambda X: ((X - mu) / sd) @ W + ym


def analyze(a):
    Z = np.load(os.path.join(a.out, "pooled.npz"))
    it = json.load(open(os.path.join(a.out, "items.json")))
    kind = np.array([x["kind"] for x in it])
    grp = np.array([x["group"] for x in it])
    fam = np.array([x["fam"] for x in it])
    R = {"n": {k: int((kind == k).sum()) for k in "A B RA RB N F".split()}}
    A = np.where(kind == "A")[0]
    idB = {f: i for i, f in zip(np.where(kind == "B")[0], fam[kind == "B"])}
    A = np.array([i for i in A if fam[i] in idB])
    Bm = np.array([idB[fam[i]] for i in A])
    r = np.array([it[i]["label"]["r"] for i in A])
    amo = np.array([it[i]["label"]["amodal"] for i in A])
    cen = np.array([it[i]["label"]["cen"] for i in A])
    spd = np.concatenate([np.linalg.norm(cen[:, 1:] - cen[:, :-1], axis=-1), np.zeros((len(A), 1))], 1) / 4     # px / 원본 프레임
    infr = amo > 10
    hidb = (r <= 0.02) & infr
    visb = (r >= 0.8) & infr
    movb = spd > 0.5
    print("합성 블록 비율: 가림", round(hidb.mean(), 3), "보임", round(visb.mean(), 3), "움직임", round(movb.mean(), 3), "| 물체 칸 수 평균", round(float(amo.mean() / 4096 * 256), 1), "/ 256")

    def tri(b, off=0):
        K = NB - 2 - off
        return np.stack([b[:, k + off] & b[:, k + 1 + off] & b[:, k + 2 + off] for k in range(K)], 1)
    Rr = np.where(kind == "RA")[0]
    idRB = {f: i for i, f in zip(np.where(kind == "RB")[0], fam[kind == "RB"])}
    Rr = np.array([i for i in Rr if fam[i] in idRB])
    RBm = np.array([idRB[fam[i]] for i in Rr])
    rh = np.array([it[i]["hid"] for i in Rr])
    rv = np.array([it[i]["vis"] for i in Rr])
    N = np.where(kind == "N")[0]
    R["curv"], R["pair"], R["mask"], R["frozen"], R["decode"], R["batch6"] = {}, {}, {}, {}, {}, {}

    def spaces():
        yield "IN", "tf", Z["inp"].astype(np.float32)
        for m in MODELS:
            for p in ("tf", "ro"):
                arr = Z[f"{m}_{p}"].astype(np.float32)
                for li, ln in enumerate(LNAMES):
                    yield f"{m}|{ln}", p, arr[:, :, li]
    for name, path, X in spaces():
        key = f"{name}|{path}"
        nb = X.shape[1]
        XA, XB, XR, XRB = (X[ix] - X[ix].mean(0, keepdims=True) for ix in (A, Bm, Rr, RBm))
        if path == "tf":
            cm = lambda b: tri(b)[:, : nb - 2]
            lo = 0
        else:                                               # 상태 되먹임: 세 자리 모두 미래. 자리 b 의 정답은 블록 b+1
            def cm(b):
                t = tri(b, 1)[:, : nb - 2].copy()
                t[:, :C] = False
                return t
        cA, sA = cosv(XA)
        cB, _ = cosv(XB)
        cR, sR = cosv(XR)
        cRB, _ = cosv(XRB)
        hid, vis = cm(hidb), cm(visb & movb)
        slide = (grp[A] == "flat_slide")[:, None]
        res = dict(syn_hid=stat(cA, hid), syn_vis=stat(cA, vis), syn_hid_empty=stat(cB, hid), syn_vis_empty=stat(cB, vis),
                   hid_fixed=stat(cA, hid & ~slide), vis_fixed=stat(cA, vis & ~slide), hid_fixed_empty=stat(cB, hid & ~slide), vis_fixed_empty=stat(cB, vis & ~slide),
                   hid_slide=stat(cA, hid & slide), vis_slide=stat(cA, vis & slide),
                   real_hid=stat(cR, cm(rh)), real_vis=stat(cR, cm(rv)), real_hid_empty=stat(cRB, cm(rh)), real_vis_empty=stat(cRB, cm(rv)))
        cN, _ = cosv(X[N] - X[N].mean(0, keepdims=True))
        res["natural"] = stat(cN, np.ones_like(cN, bool)) if path == "tf" else stat(cN[:, C:], np.ones_like(cN[:, C:], bool))
        shuf = np.random.default_rng(0).permuted(XA, axis=1)
        res["null_shuffle"] = stat(cosv(shuf)[0], np.ones_like(cA, bool))
        R["curv"][key] = res
        D, DR = X[A] - X[Bm], X[Rr] - X[RBm]
        cD, _ = cosv(D)
        cDR, _ = cosv(DR)
        nd = np.linalg.norm(D, axis=-1)[:, :nb]
        ndr = np.linalg.norm(DR, axis=-1)[:, :nb]
        sB = cosv(XB)[1]
        o = 1 if path == "ro" else 0                        # 상태 되먹임: 자리 b 의 정답은 블록 b+1
        hb, vb = hidb[:, o:nb + o], (visb & movb)[:, o:nb + o]
        fut = np.zeros(nb, bool)
        fut[:] = True
        if path == "ro":
            fut[:C] = False
        R["pair"][key] = dict(syn_hid=stat(cD, hid), syn_vis=stat(cD, vis), real_hid=stat(cDR, cm(rh)), real_vis=stat(cDR, cm(rv)),
                              norm_hid=round(float(nd[hb & fut].mean()), 4) if (hb & fut).any() else None, norm_vis=round(float(nd[vb & fut].mean()), 4) if (vb & fut).any() else None,
                              rnorm_hid=round(float(ndr[rh[:, o:nb + o] & fut].mean()), 4) if (rh[:, o:nb + o] & fut).any() else None, rnorm_vis=round(float(ndr[rv[:, o:nb + o] & fut].mean()), 4) if (rv[:, o:nb + o] & fut).any() else None,
                              step_empty_over_obj_hid=round(float(sB[hid].mean() / max(sA[hid].mean(), 1e-9)), 3) if hid.any() else None,
                              step_empty_over_obj_vis=round(float(sB[vis].mean() / max(sA[vis].mean(), 1e-9)), 3) if vis.any() else None)
        if path == "tf":
            thr = np.percentile(sA, 30)
            low = sA < thr
            R["mask"][key] = dict(hid=round(float(low[hid].mean()), 3), vis=round(float(low[vis].mean()), 3), step_hid=round(float(sA[hid].mean()), 4), step_vis=round(float(sA[vis].mean()), 4))
            g = np.random.default_rng(1).permutation(len(A))
            c6 = np.zeros_like(cA)
            for s in range(0, len(g), 6):
                ix = g[s:s + 6]
                c6[ix] = cosv(X[A][ix] - X[A][ix].mean(0, keepdims=True))[0]
            R["batch6"][key] = dict(hid=stat(c6, hid), vis=stat(c6, vis))
        for fg in ("frozen_synth", "frozen_real", "frozen_natural"):
            F_ = np.where(grp == fg)[0]
            if len(F_) < 6:
                continue
            Xf = X[F_]
            raw, _ = cosv(Xf)
            rm, _ = cosv(Xf - Xf.mean(0, keepdims=True))
            Y = Xf - Xf.mean(0, keepdims=True)
            Yc = Y - Y.mean(1, keepdims=True)
            t = np.arange(nb, dtype=np.float32) - (nb - 1) / 2
            beta = (Yc * t[None, :, None]).sum(1, keepdims=True) / (t ** 2).sum()
            lin = 1 - ((Yc - beta * t[None, :, None]) ** 2).sum() / max((Yc ** 2).sum(), 1e-9)
            R["frozen"].setdefault(key, {})[fg] = dict(theta_raw=round(float(theta(raw).mean()), 1), theta_rm=round(float(theta(rm).mean()), 1),
                                                      step_rm=round(float(np.linalg.norm(vel(Y), axis=-1).mean()), 4), step_vis_ref=round(float(sA[vis].mean()), 4) if vis.any() else None, lin_r2=round(float(lin), 3))
    # 위치 판독 (합성). 자리 b 의 정답 = 블록 b+1 의 대상 중심
    tr = np.array([h8(fam[i]) % 10 < 7 for i in A])
    fitv = np.array([h8(fam[i] + "v") % 5 == 0 for i in A])
    Y = cen[:, 1:]
    hp = hidb[:, :-1] & hidb[:, 1:]
    vp = visb[:, :-1] & visb[:, 1:]
    last = np.full_like(cen, np.nan)
    prev = np.full_like(cen, np.nan)
    for i in range(len(A)):
        l, q = None, None
        for b in range(NB):
            last[i, b] = np.nan if l is None else cen[i, l]
            prev[i, b] = np.nan if (l is None or q is None) else (cen[i, l] + (cen[i, l] - cen[i, q]) * (b - l) / max(l - q, 1))
            if visb[i, b]:
                q, l = l, b
    base_c, base_v = last[:, :-1], prev[:, :-1]

    def err(P, Yt):
        return float(np.nanmean(np.linalg.norm(P - Yt, axis=-1)))
    for path in ("tf", "ro"):
        fut = np.zeros(NB - 1, bool)
        fut[C - 1 if path == "ro" else 0:] = True
        test_h = (hp & fut[None]) & (~tr)[:, None]
        train_h = (hp & fut[None]) & (tr & ~fitv)[:, None]
        val_h = (hp & fut[None]) & (tr & fitv)[:, None]
        train_v = (vp & fut[None]) & (tr & ~fitv)[:, None]
        val_v = (vp & fut[None]) & (tr & fitv)[:, None]
        test_v = (vp & fut[None]) & (~tr)[:, None]
        ok = ~np.isnan(base_c).any(-1) & ~np.isnan(base_v).any(-1)
        th = test_h & ok
        yt = Y[th]
        out = dict(n_train=int(train_h.sum()), n_test=int(th.sum()), mean_pos=err(np.broadcast_to(Y[train_h].mean(0), yt.shape), yt), copy_last=err(base_c[th], yt), const_vel=err(base_v[th], yt), rows={})
        for m in MODELS:
            arr = Z[f"{m}_{path}"].astype(np.float32)
            for li, ln in enumerate(LNAMES):
                XA_, XB_ = arr[A][:, : NB - 1, li], arr[Bm][:, : NB - 1, li]
                f_self = ridge_fit(XA_[train_h], Y[train_h], XA_[val_h], Y[val_h])
                f_vis = ridge_fit(XA_[train_v], Y[train_v], XA_[val_v], Y[val_v])
                f_emp = ridge_fit(XB_[train_h], Y[train_h], XB_[val_h], Y[val_h])
                out["rows"][f"{m}|{ln}"] = dict(self=round(err(f_self(XA_[th]), yt), 2), vis2hid=round(err(f_vis(XA_[th]), yt), 2), empty=round(err(f_emp(XB_[th]), yt), 2), vis_self=round(err(f_vis(XA_[test_v]), Y[test_v]), 2))
        R["decode"][path] = out
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump(R, open(a.json, "w"), ensure_ascii=False, indent=1)
    print("저장", a.json)


def drift(a):
    """지름길 측정: 물리와 무관한 영상별 표류 (시간에 비례, 영상마다 무작위 방향) 를 L9 은닉에 직접 더하면 곡률과 한 걸음 L1 이 어떻게 바뀌나.
    표류 세기 = α × 그 모델의 칸 평균 은닉 평균 걸음. 방향이 무작위라 학습이 고를 값싼 방향의 상한 비용이다."""
    import decord, torch, torch.nn.functional as F, yaml
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
    from app.vjepa_frozen import utils as U
    from app.vjepa_frozen.ar import encode_blocks
    torch.set_num_threads(8)
    dev = torch.device("cuda")
    it = json.load(open(os.path.join(a.out, "items.json")))
    rng = random.Random(3)
    pick = []
    for k, n in (("A", 64), ("RA", 48), ("N", 48)):
        c = [x for x in it if x["kind"] == k]
        rng.shuffle(c)
        pick += c[:n]
    cfg0 = yaml.safe_load(open(REPO / "z_training/runs" / MODELS["base"] / "config.yaml"))
    sd = _load_checkpoint(cfg0["model"]["checkpoint"])
    enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(enc, sd["target_encoder"], tag="t")
    enc = enc.to(dev, torch.bfloat16).eval()
    del sd
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def load(x):
        vr = decord.VideoReader(x["path"], num_threads=2)
        n = len(vr)
        ids = x["idx"]
        if x["kind"] == "N" and ids[-1] >= n:
            sh = ids[-1] - (n - 1)
            ids = [i - sh for i in ids]
        t = torch.from_numpy(vr.get_batch([min(i, n - 1) for i in ids]).asnumpy()).permute(0, 3, 1, 2).float() / 255
        if x["crop"] and t.shape[2] != t.shape[3]:
            m = min(t.shape[2:])
            y0, x0 = (t.shape[2] - m) // 2, (t.shape[3] - m) // 2
            t = t[:, :, y0:y0 + m, x0:x0 + m]
        if t.shape[2] != 256 or t.shape[3] != 256:
            t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
        return (t.permute(1, 0, 2, 3) - mean) / std
    toks, kinds = [], []
    for s in range(0, len(pick), 8):
        c = torch.stack([load(x) for x in pick[s:s + 8]]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            toks.append(encode_blocks(enc, c, 2).float().cpu())
        kinds += [x["kind"] for x in pick[s:s + 8]]
    X = torch.cat(toks)
    kinds = np.array(kinds)
    del enc
    res = {}
    INJ = 9
    tc = (torch.arange(NB, device=dev).float() - (NB - 1) / 2)
    for mk, run in MODELS.items():
        c = yaml.safe_load(open(REPO / "z_training/runs" / run / "config.yaml"))
        p = U.build_predictor(c["model"], 1280, None, dev, T)
        p.load_state_dict(_clean_backbone_key(_load_checkpoint(str(REPO / "z_training/runs" / run / "latest.pt"))["predictor"]), strict=True)
        p = p.float().eval()

        @torch.no_grad()
        def tf(x, u=None, amp=0.0):
            B = x.size(0)
            idx = torch.arange(NB * S, device=dev).unsqueeze(0).expand(B, -1)
            caches, pooled, outs = None, [], []
            for b in range(NB):
                sl = slice(b * S, (b + 1) * S)
                h = p._ar_embed_obs(x[:, sl])
                new = []
                for i, blk in enumerate(p.predictor_blocks):
                    h, kv = blk.forward_kv(h, idx[:, sl], p.grid_height, p.grid_width, None if caches is None else caches[i])
                    new.append(kv)
                    if i + 1 == INJ:
                        if u is not None:
                            h = h + amp * tc[b] * u[:, None, :]
                        pooled.append(h.mean(1))
                caches = new
                st = p.predictor_norm(h)
                outs.append(p._ar_out(st, x[:, sl]))
            return torch.stack(pooled, 1), torch.stack(outs, 1)
        base_p, l1_0, base_o = [], [], []
        for s in range(0, len(X), 16):
            x = X[s:s + 16].to(dev)
            pl, o = tf(x)
            base_p.append(pl.cpu().numpy())
            base_o.append(o.cpu())
        P0 = np.concatenate(base_p)
        step = float(np.linalg.norm(vel(P0 - P0.mean(0, keepdims=True)), axis=-1).mean())
        O0 = torch.cat(base_o)
        tgt = X.view(len(X), NB, S, -1)[:, 1:]
        res[mk] = dict(step=round(step, 3), pooled_norm=round(float(np.linalg.norm(P0, axis=-1).mean()), 2), rows=[])
        g = torch.Generator(device="cpu").manual_seed(0)
        U_ = F.normalize(torch.randn(len(X), P0.shape[-1], generator=g), dim=-1)
        for al in (0, 0.5, 1, 2, 4, 8, 16):
            Pp, L1, D = [], [], []
            for s in range(0, len(X), 16):
                x = X[s:s + 16].to(dev)
                u = U_[s:s + 16].to(dev)
                pl, o = tf(x, u, al * step)
                Pp.append(pl.cpu().numpy())
                L1.append((o[:, :-1].cpu() - tgt[s:s + 16]).abs().mean(dim=(1, 2, 3)))
                D.append((o.cpu() - O0[s:s + 16]).abs().mean(dim=(1, 2, 3)))
            Pp = np.concatenate(Pp)
            L1 = torch.cat(L1).numpy()
            D = torch.cat(D).numpy()
            row = dict(alpha=al)
            for k in ("A", "RA", "N"):
                m = kinds == k
                cc, _ = cosv(Pp[m] - Pp[m].mean(0, keepdims=True))
                row[k] = dict(theta=round(float(theta(cc).mean()), 1), cos=round(float(cc.mean()), 3), l1=round(float(L1[m].mean()), 4), out_change=round(float(D[m].mean()), 4))
            res[mk]["rows"].append(row)
            print(mk, row, flush=True)
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1)


def extras(a):
    """보충 표: 보임 구간 위치 기준선 · 고정 가림막 걸음비 · 굴린 걸음별 곡률 · 곡률-성능 표."""
    Z = np.load(os.path.join(a.out, "pooled.npz"))
    it = json.load(open(os.path.join(a.out, "items.json")))
    kind = np.array([x["kind"] for x in it]); grp = np.array([x["group"] for x in it]); fam = np.array([x["fam"] for x in it])
    idB = {f: i for i, f in zip(np.where(kind == "B")[0], fam[kind == "B"])}
    A = np.array([i for i in np.where(kind == "A")[0] if fam[i] in idB]); Bm = np.array([idB[fam[i]] for i in A])
    r = np.array([it[i]["label"]["r"] for i in A]); amo = np.array([it[i]["label"]["amodal"] for i in A]); cen = np.array([it[i]["label"]["cen"] for i in A])
    infr = amo > 10
    hidb = (r <= 0.02) & infr; visb = (r >= 0.8) & infr
    R = {}
    tr = np.array([h8(fam[i]) % 10 < 7 for i in A]); fv = np.array([h8(fam[i] + "v") % 5 == 0 for i in A])
    Y = cen[:, 1:]; vp = visb[:, :-1] & visb[:, 1:]
    a_tr, a_va, a_te = vp & (tr & ~fv)[:, None], vp & (tr & fv)[:, None], vp & (~tr)[:, None]
    yt = Y[a_te]
    dec = dict(n_train=int(a_tr.sum()), n_test=int(a_te.sum()), mean_pos=round(float(np.linalg.norm(Y[a_tr].mean(0) - yt, axis=1).mean()), 2), copy_prev=round(float(np.linalg.norm(cen[:, :-1][a_te] - yt, axis=1).mean()), 2))
    I = Z["inp"].astype(np.float32)[A][:, : NB - 1]
    f = ridge_fit(I[a_tr], Y[a_tr], I[a_va], Y[a_va])
    dec["input_pool"] = round(float(np.linalg.norm(f(I[a_te]) - yt, axis=1).mean()), 2)
    for m in MODELS:
        X = Z[f"{m}_tf"].astype(np.float32)[A][:, : NB - 1, 3]
        f = ridge_fit(X[a_tr], Y[a_tr], X[a_va], Y[a_va])
        dec[f"{m}_L9"] = round(float(np.linalg.norm(f(X[a_te]) - yt, axis=1).mean()), 2)
    R["decode_visible"] = dec
    fixed = np.isin(grp[A], ["flat_board", "ramp_board", "flat_wall", "ramp_wall"])
    R["fixed_step"] = {}
    for name, X in [("IN", Z["inp"].astype(np.float32))] + [(f"{m}|L9", Z[f"{m}_tf"].astype(np.float32)[:, :, 3]) for m in MODELS]:
        XA = X[A[fixed]] - X[A[fixed]].mean(0, keepdims=True)
        XB = X[Bm[fixed]] - X[Bm[fixed]].mean(0, keepdims=True)
        sA, sB = np.linalg.norm(vel(XA), axis=-1), np.linalg.norm(vel(XB), axis=-1)
        hv = (hidb[fixed][:, :-1] & hidb[fixed][:, 1:]); vv = (visb[fixed][:, :-1] & visb[fixed][:, 1:])
        F_ = np.where(grp == "frozen_synth")[0]
        sF = np.linalg.norm(vel(X[F_] - X[F_].mean(0, keepdims=True)), axis=-1).mean()
        R["fixed_step"][name] = dict(obj_vis=round(float(sA[vv].mean()), 3), empty_same=round(float(sB[vv].mean()), 3), obj_hid=round(float(sA[hv].mean()), 3), empty_hid=round(float(sB[hv].mean()), 3), frozen=round(float(sF), 3))
    R["rollout_by_step"] = {}
    N = np.where(kind == "N")[0]
    idRB = {f: i for i, f in zip(np.where(kind == "RB")[0], fam[kind == "RB"])}
    Rr = np.array([i for i in np.where(kind == "RA")[0] if fam[i] in idRB]); RBm = np.array([idRB[fam[i]] for i in Rr])
    for m in MODELS:
        X = Z[f"{m}_ro"].astype(np.float32)[:, :, 3]
        o = {}
        for nm, ix in (("syn_obj", A), ("syn_empty", Bm), ("real_obj", Rr), ("real_empty", RBm), ("natural", N)):
            c, s_ = cosv(X[ix] - X[ix].mean(0, keepdims=True))
            o[nm] = [round(float(theta(c[:, b]).mean()), 1) for b in range(C - 2, c.shape[1])]
            o[nm + "_step"] = [round(float(s_[:, b].mean()), 2) for b in range(C - 2, c.shape[1])]
        R["rollout_by_step"][m] = o
    Rj = json.load(open(a.json))
    R["frozen_all"] = {k: v for k, v in Rj["frozen"].items() if "|L9|" in k or "|ST|" in k}
    p = a.json.replace("summary.json", "extras.json")
    json.dump(R, open(p, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "frozen_all"}, ensure_ascii=False)[:5000])
    for k, v in R["frozen_all"].items():
        print(k, {g: (x["theta_rm"], x["step_rm"], x["lin_r2"]) for g, x in v.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "analyze", "drift", "extras"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_fam", type=int, default=64)
    ap.add_argument("--n_slide", type=int, default=64)
    ap.add_argument("--n_nat", type=int, default=96)
    ap.add_argument("--n_frozen", type=int, default=96)
    a = ap.parse_args()
    {"extract": extract, "analyze": analyze, "drift": drift, "extras": extras}[a.cmd](a)
